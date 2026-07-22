"""NSY 图片感知哈希、相似分组、质量比较与对比报告渲染。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..draw.plot import (
    Canvas,
    DEFAULT_BOLD_FONT,
    DEFAULT_FONT,
    FillBg,
    Grid,
    ImageBox,
    RoundRectBg,
    TextBox,
    TextStyle,
    VSplit,
)


PHASH_SIZE = 8
PHASH_IMAGE_SIZE = 32
PHASH_HEX_LENGTH = PHASH_SIZE * PHASH_SIZE // 4
REPORT_IMAGE_MAX_SIZE = (300, 190)
REJECTION_REPORT_IMAGE_MAX_SIZE = (460, 320)
REPORT_CANVAS_WIDTH = 1000
REPORT_CONTENT_WIDTH = 968
REPORT_GROUPS_PER_PAGE = 3
REPORT_IMAGES_PER_GROUP_SECTION = 6


@dataclass(frozen=True)
class ImageFeature:
    """参与相似比较和报告渲染的单张图片信息。"""

    identity: str
    gallery: str
    path: str
    phash: str
    width: int
    height: int
    pid: int | None = None
    pending: bool = False
    action: str = ''

    @property
    def pixels(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class SimilarityGroup:
    """所有成员都与质量最优项直接相似的去重分组。"""

    images: tuple[ImageFeature, ...]
    keep: ImageFeature


def normalize_phash(value: str | None) -> str | None:
    """校验并规范化 64 位十六进制 pHash。"""
    value = str(value or '').strip().lower()
    if len(value) != PHASH_HEX_LENGTH:
        return None
    try:
        int(value, 16)
    except ValueError:
        return None
    return value


def compute_phash(path: str | Path) -> str:
    """
    计算标准 64 位 pHash。

    图片先按 EXIF 方向旋转、透明区域合成到白底，再缩放为 32x32 灰度图；
    对二维 DCT 左上角 8x8 低频系数按中位数二值化，最终输出 16 位十六进制。
    动图使用首帧参与特征计算。
    """
    with Image.open(path) as source:
        try:
            source.seek(0)
        except EOFError:
            pass
        image = ImageOps.exif_transpose(source).convert('RGBA')
        background = Image.new('RGBA', image.size, (255, 255, 255, 255))
        background.alpha_composite(image)
        gray = background.convert('L').resize(
            (PHASH_IMAGE_SIZE, PHASH_IMAGE_SIZE),
            Image.Resampling.LANCZOS,
        )

    dct = cv2.dct(np.asarray(gray, dtype=np.float32))
    low_frequency = dct[:PHASH_SIZE, :PHASH_SIZE]
    # 排除直流分量计算阈值，避免整体亮度支配其余低频结构。
    median = float(np.median(low_frequency.flatten()[1:]))
    bits = low_frequency >= median
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f'{value:0{PHASH_HEX_LENGTH}x}'


def hamming_distance(first: str, second: str) -> int:
    """返回两个 64 位 pHash 的汉明距离。"""
    first = normalize_phash(first)
    second = normalize_phash(second)
    if first is None or second is None:
        raise ValueError('pHash 需是 16 位十六进制字符串')
    return (int(first, 16) ^ int(second, 16)).bit_count()


def select_best_image(images: Sequence[ImageFeature]) -> ImageFeature:
    """
    按分辨率选择保留图片。

    先比较总像素数，再比较宽、高；完全相同时优先保留图库已有图片，已有
    图片之间保留 pid 较小者，保证结果稳定且避免无意义替换。
    """
    if not images:
        raise ValueError('无法从空图片列表中选择保留项')
    return max(
        images,
        key=lambda image: (
            image.pixels,
            image.width,
            image.height,
            not image.pending,
            -(image.pid if image.pid is not None else 2**63),
        ),
    )


def find_similarity_groups(
    images: Sequence[ImageFeature],
    distance_threshold: int = 4,
) -> list[SimilarityGroup]:
    """
    在给定图片集合内计算 pHash 距离并返回安全的去重分组。

    ``distance_threshold`` 为严格上界；设为 4 时只有差异 0~3 位才建立
    相似关系。每轮先选择剩余图片中分辨率最高者，再只吸收与它直接相似
    的图片，避免 pHash 相似关系不满足传递性时误删与保留项并不相似的图片。
    """
    if distance_threshold <= 0:
        return []
    remaining = [image for image in images if normalize_phash(image.phash) is not None]
    distances: dict[tuple[str, str], int] = {}

    def pair_key(first: ImageFeature, second: ImageFeature) -> tuple[str, str]:
        return tuple(sorted((first.identity, second.identity)))

    # 先完成集合内的全量两两比较，后续分组只读取距离表。
    for first_idx, first in enumerate(remaining):
        for second in remaining[first_idx + 1:]:
            distances[pair_key(first, second)] = hamming_distance(first.phash, second.phash)

    groups = []
    while remaining:
        keep = select_best_image(remaining)
        similar = [
            image
            for image in remaining
            if image.identity != keep.identity
            and distances[pair_key(keep, image)] < distance_threshold
        ]
        grouped_identities = {keep.identity} | {image.identity for image in similar}
        remaining = [
            image for image in remaining
            if image.identity not in grouped_identities
        ]
        if similar:
            ordered = tuple(sorted([keep, *similar], key=lambda image: image.identity))
            groups.append(SimilarityGroup(ordered, keep))
    return sorted(groups, key=lambda group: (group.keep.gallery, group.keep.identity))


def _load_report_image(
    path: str,
    max_size: tuple[int, int] = REPORT_IMAGE_MAX_SIZE,
) -> Image.Image:
    """加载报告图片并按原始长宽比缩放到较大的可变尺寸。"""
    try:
        with Image.open(path) as source:
            try:
                source.seek(0)
            except EOFError:
                pass
            image = ImageOps.exif_transpose(source).convert('RGBA')
            image.load()
            scale = min(
                max_size[0] / max(1, image.width),
                max_size[1] / max(1, image.height),
            )
            size = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            return image.resize(size, Image.Resampling.LANCZOS)
    except Exception:
        return Image.new('RGBA', max_size, (232, 236, 241, 255))


def _split_report_sections(
    groups: Sequence[SimilarityGroup],
    images_per_section: int = REPORT_IMAGES_PER_GROUP_SECTION,
) -> list[tuple[int, int, SimilarityGroup, tuple[ImageFeature, ...]]]:
    """把大分组拆成固定大小的报告区块，防止单页画布超过尺寸上限。"""
    sections = []
    for group_idx, group in enumerate(groups, 1):
        images = group.images
        part_count = (len(images) + images_per_section - 1) // images_per_section
        for part_idx in range(part_count):
            start = part_idx * images_per_section
            chunk = images[start:start + images_per_section]
            sections.append((group_idx, part_idx + 1, group, chunk))
    return sections


async def render_similarity_report(
    groups: Sequence[SimilarityGroup],
    title: str,
    show_decisions: bool = True,
) -> list[Image.Image]:
    """用项目 draw 组件将全部相似分组渲染为一页或多页对比图。"""
    image_max_size = (
        REPORT_IMAGE_MAX_SIZE if show_decisions
        else REJECTION_REPORT_IMAGE_MAX_SIZE
    )
    column_count = 3 if show_decisions else 2
    images_per_section = REPORT_IMAGES_PER_GROUP_SECTION if show_decisions else 4
    groups_per_page = REPORT_GROUPS_PER_PAGE if show_decisions else 2
    canvas_padding = 16 if show_decisions else 12
    group_padding = 12 if show_decisions else 8
    tile_width = (
        REPORT_IMAGE_MAX_SIZE[0] if show_decisions
        else (REPORT_CONTENT_WIDTH - group_padding * 2 - 10) // 2
    )
    sections = _split_report_sections(groups, images_per_section)
    pages = []
    for page_start in range(0, len(sections), groups_per_page):
        page_sections = sections[page_start:page_start + groups_per_page]
        with Canvas(w=REPORT_CANVAS_WIDTH, bg=FillBg((235, 243, 250, 255))).set_padding(canvas_padding) as canvas:
            content_width = REPORT_CANVAS_WIDTH - canvas_padding * 2
            with VSplit().set_w(content_width).set_sep(8).set_content_and_item_align('l'):
                TextBox(
                    title,
                    TextStyle(DEFAULT_BOLD_FONT, 28, (35, 51, 67, 255)),
                ).set_w(content_width).set_padding(0)
                description = (
                    '绿色-根文件；蓝色-转为链接；'
                    '红色-删除。'
                    if show_decisions else
                    '红色-同图库相似图片，拒绝添加；'
                    '蓝色-跨图库相似图片，已自动创建链接。'
                )
                TextBox(
                    description,
                    TextStyle(DEFAULT_FONT, 14, (83, 101, 118, 255)),
                ).set_w(content_width).set_padding(0)

                for group_idx, part_idx, group, images in page_sections:
                    part_total = (
                        len(group.images) + images_per_section - 1
                    ) // images_per_section
                    part_text = f' · 第 {part_idx}/{part_total} 部分' if part_total > 1 else ''
                    galleries = ' / '.join(sorted({image.gallery for image in group.images}))
                    with (
                        VSplit()
                        .set_w(content_width)
                        .set_padding(group_padding)
                        .set_sep(6)
                        .set_bg(RoundRectBg(fill=(255, 255, 255, 235), radius=16))
                        .set_content_and_item_align('l')
                    ):
                        TextBox(
                            f'相似组 {group_idx} · 图库“{galleries}”{part_text}',
                            TextStyle(DEFAULT_BOLD_FONT, 18, (42, 63, 81, 255)),
                        ).set_padding(0)
                        with Grid(col_count=column_count).set_sep(10, 8).set_item_align('t'):
                            for image in images:
                                is_keep = image.identity == group.keep.identity
                                if show_decisions:
                                    label = '待上传' if image.pending else f'pid={image.pid}'
                                    if image.action:
                                        status_text = f'{image.action} · {label}'
                                        if image.action == '保留文件':
                                            status_color = (38, 135, 92, 255)
                                        elif '链接' in image.action:
                                            status_color = (58, 104, 145, 255)
                                        else:
                                            status_color = (176, 76, 76, 255)
                                    else:
                                        status = '保留' if is_keep else '删除'
                                        status_text = f'{status} · {label}'
                                        status_color = (
                                            (38, 135, 92, 255) if is_keep
                                            else (176, 76, 76, 255)
                                        )
                                else:
                                    if image.pending:
                                        status_text = image.action or '待上传 · 相似图片拒绝添加'
                                        status_color = (
                                            (58, 104, 145, 255)
                                            if '链接' in status_text else
                                            (176, 76, 76, 255)
                                        )
                                    else:
                                        status_text = f'已有 · pid={image.pid}'
                                        status_color = (58, 104, 145, 255)
                                distance = hamming_distance(image.phash, group.keep.phash)
                                preview = _load_report_image(image.path, image_max_size)
                                with VSplit().set_w(tile_width).set_sep(4).set_content_and_item_align('c'):
                                    ImageBox(
                                        preview,
                                        size=preview.size,
                                        use_alphablend=True,
                                    )
                                    TextBox(
                                        status_text,
                                        TextStyle(DEFAULT_BOLD_FONT, 15, status_color),
                                    ).set_w(tile_width).set_padding(0).set_content_align('c')
                                    TextBox(
                                        f'{image.width}×{image.height} · Hamming distance {distance}',
                                        TextStyle(DEFAULT_FONT, 13, (85, 99, 113, 255)),
                                    ).set_w(tile_width).set_padding(0).set_content_align('c')
        pages.append(await canvas.get_img())
    return pages


__all__ = [
    'ImageFeature',
    'SimilarityGroup',
    'compute_phash',
    'find_similarity_groups',
    'hamming_distance',
    'normalize_phash',
    'render_similarity_report',
    'select_best_image',
]
