"""NSY 图库统计、图库缩略预览与按 PID 单图查询。"""

from __future__ import annotations

import math
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageOps

from ..draw.painter import Painter, get_font, get_font_desc, get_text_size
from ..draw.plot import (
    DEFAULT_BOLD_FONT,
    DEFAULT_FONT,
    Frame,
    Grid,
    ImageBox,
    RoundRectBg,
    Spacer,
    TextBox,
    TextStyle,
    VSplit,
)
from ..utils.draw import (
    UNIFIED_OTHER_CHART_COLOR,
    UnifiedDrawTheme,
    create_report_canvas,
    create_report_column,
    report_card,
    report_description_panel,
    report_header,
    report_image_panel,
    report_section_title,
    resolve_configured_draw_background,
    resolve_draw_categorical_palette,
    resolve_draw_palette,
    resolve_draw_theme,
    subtle_panel_background,
    themed_text_style,
)
from ..utils.handler import (
    HandlerContext,
    ReplyException,
    check_superuser,
    extract_image_data,
    get_image_cq,
)
from ..utils.utils import run_in_pool
from . import (
    NsyImage,
    NsyManager,
    _get_image_temp_file,
    _inspect_image_file,
    _nsy_cmd,
    _sha256_file,
    config,
    logger,
)


STATS_CANVAS_WIDTH = 1440
STATS_SUMMARY_HEIGHT = 500
BAR_ROW_HEIGHT = 29

PREVIEW_CANVAS_WIDTH = 1120
PREVIEW_COLUMN_COUNT = 4
PREVIEW_GRID_GAP = 12

DETAIL_CANVAS_WIDTH = 1080

OTHER_CHART_COLOR = UNIFIED_OTHER_CHART_COLOR


@dataclass(frozen=True)
class NsyDrawStyle:
    """一次 NSY 报告使用的主题与调色板快照。"""

    theme: UnifiedDrawTheme
    category_colors: tuple[tuple[int, int, int, int], ...]
    smooth_colors: tuple[tuple[int, int, int, int], ...]


def _resolve_nsy_style(group_id: int | str | None = None) -> NsyDrawStyle:
    return NsyDrawStyle(
        theme=resolve_draw_theme(config, group_id=group_id),
        category_colors=resolve_draw_categorical_palette(
            config,
            group_id=group_id,
        ),
        smooth_colors=resolve_draw_palette(config, group_id=group_id),
    )


_DEFAULT_NSY_STYLE = _resolve_nsy_style()
_NSY_STYLE_CONTEXT: ContextVar[NsyDrawStyle] = ContextVar(
    'nsy_draw_style',
    default=_DEFAULT_NSY_STYLE,
)


def _current_nsy_style() -> NsyDrawStyle:
    return _NSY_STYLE_CONTEXT.get()


class _NsyThemeProxy:
    def __getattr__(self, name):
        return getattr(_current_nsy_style().theme, name)


class _NsyPaletteProxy:
    def __init__(self, attribute: str):
        self.attribute = attribute

    def _palette(self):
        return getattr(_current_nsy_style(), self.attribute)

    def __len__(self):
        return len(self._palette())

    def __getitem__(self, index):
        return self._palette()[index]

    def __iter__(self):
        return iter(self._palette())


NSY_DRAW_THEME = _NsyThemeProxy()
CATEGORY_COLORS = _NsyPaletteProxy('category_colors')
SMOOTH_COLORS = _NsyPaletteProxy('smooth_colors')

STATS_CONTENT_WIDTH = (
    STATS_CANVAS_WIDTH - _DEFAULT_NSY_STYLE.theme.page_padding * 2
)
PREVIEW_CONTENT_WIDTH = (
    PREVIEW_CANVAS_WIDTH - _DEFAULT_NSY_STYLE.theme.page_padding * 2
)
PREVIEW_TILE_WIDTH = (
    PREVIEW_CONTENT_WIDTH - PREVIEW_GRID_GAP * (PREVIEW_COLUMN_COUNT - 1)
) // PREVIEW_COLUMN_COUNT
PREVIEW_IMAGE_SIZE = (PREVIEW_TILE_WIDTH - 16, 164)
DETAIL_CONTENT_WIDTH = (
    DETAIL_CANVAS_WIDTH - _DEFAULT_NSY_STYLE.theme.page_padding * 2
)


@dataclass(frozen=True)
class GalleryCount:
    """一个规范图库及其索引图片数量。"""

    name: str
    count: int


@dataclass(frozen=True)
class GalleryPreviewItem:
    """图库预览所需的最小图片信息。"""

    pid: int
    path: str
    related_galleries: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreparedGalleryPreviewItem:
    """已完成磁盘读取和缩放的图库预览项。"""

    pid: int
    image: Image.Image
    related_galleries: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageDetail:
    """单图报告所需信息；权限无关字段与上传来源分开渲染。"""

    pid: int
    path: str
    gallery: str
    format: str
    width: int
    height: int
    size: int
    created_at: str
    uploader_id: int | None
    group_id: int | None
    related_galleries: tuple[str, ...] = ()
    link_root_pid: int | None = None
    linked_record_count: int = 1


def _draw_text(
    painter: Painter,
    text: str,
    x: int,
    y: int,
    *,
    font: str = DEFAULT_FONT,
    size: int = 16,
    color: tuple[int, int, int, int] = (32, 48, 68, 255),
    align: str = 'left',
) -> None:
    """在自绘图表中按左、中、右锚点放置单行文字。"""

    if align != 'left':
        width, _ = get_text_size(get_font(font, size), text)
        if align == 'center':
            x -= width // 2
        elif align == 'right':
            x -= width
        else:
            raise ValueError(f'不支持的文字对齐方式: {align}')
    painter.text(text, (x, y), get_font_desc(font, size), fill=color)


def _draw_emoji_textbox(
    painter: Painter,
    text: str,
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    font: str = DEFAULT_FONT,
    size: int = 16,
    color: tuple[int, int, int, int] = (32, 48, 68, 255),
    align: str = 'left',
) -> None:
    """
    用与 STA 昵称相同的 TextBox 路径绘制动态图库名。

    TextBox 会按可用宽度自动省略，并在底层交给 Painter/Pilmoji，因此
    Emoji 的测量、对齐和实际绘制保持一致，不再由图表代码手动切字符串。
    """

    align_map = {'left': 'l', 'center': 'c', 'right': 'r'}
    if align not in align_map:
        raise ValueError(f'不支持的文字对齐方式: {align}')
    textbox = (
        TextBox(
            text,
            TextStyle(font=font, size=size, color=color),
            line_count=1,
        )
        .set_size((width, height))
        .set_padding(0)
        .set_content_align(align_map[align])
    )
    painter.move_region((x, y), (width, height))
    textbox.draw(painter)
    painter.restore_region()


def _nice_axis(max_value: int) -> tuple[int, list[int]]:
    """生成适合整数图片数量的 1/2/5 刻度轴。"""

    if max_value <= 0:
        return 1, [0, 1]
    raw_step = max_value / 5
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step else 1
    normalized = raw_step / magnitude
    factor = next((value for value in (1, 2, 5, 10) if normalized <= value), 10)
    step = max(1, int(factor * magnitude))
    axis_max = max(step, math.ceil(max_value / step) * step)
    return axis_max, list(range(0, axis_max + 1, step))


def _load_contained_image(
    path: str,
    size: tuple[int, int],
    *,
    radius: int = 14,
) -> Image.Image:
    """
    读取图片首帧并等比放入固定预览框。

    预览不做中心裁切，避免竖图或多人照片的重要区域被截掉；透明图片先
    合成到浅色背景。读取失败时返回同尺寸占位图，让一张坏图不阻断整库。
    """

    background = Image.new('RGBA', size, (236, 242, 247, 255))
    try:
        with Image.open(path) as source:
            try:
                source.seek(0)
            except EOFError:
                pass
            image = ImageOps.exif_transpose(source).convert('RGBA')
            image.load()
        image.thumbnail(size, Image.Resampling.LANCZOS)
        position = (
            (size[0] - image.width) // 2,
            (size[1] - image.height) // 2,
        )
        background.alpha_composite(image, position)
    except Exception:
        pass

    mask = Image.new('L', size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1),
        radius=radius,
        fill=255,
    )
    background.putalpha(mask)
    return background


def _load_detail_image(path: str, max_size: tuple[int, int]) -> Image.Image:
    """读取单图首帧并等比缩放，保留完整画面与透明通道。"""

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
        2.0,
    )
    target_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    if target_size != image.size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    return image


def _prepare_gallery_preview_items(
    items: Sequence[GalleryPreviewItem],
) -> list[PreparedGalleryPreviewItem]:
    """在线程池中批量完成图库预览图片的磁盘读取。"""

    return [
        PreparedGalleryPreviewItem(
            pid=item.pid,
            image=_load_contained_image(item.path, PREVIEW_IMAGE_SIZE),
            related_galleries=item.related_galleries,
        )
        for item in items
    ]


def _pie_entries(counts: Sequence[GalleryCount]) -> list[tuple[str, int]]:
    """只保留数量前十的具体图库，其余正数图库统一合并为“其他”。"""

    positive = [item for item in counts if item.count > 0]
    entries = [(item.name, item.count) for item in positive[:10]]
    other_count = sum(item.count for item in positive[10:])
    if other_count:
        entries.append(('其他', other_count))
    return entries


def _gallery_chart_color(index: int, name: str) -> tuple[int, int, int, int]:
    """具体图库循环使用成员色，“其他”固定使用中性灰。"""

    if name == '其他':
        return OTHER_CHART_COLOR
    return CATEGORY_COLORS[index % len(CATEGORY_COLORS)]


def _draw_statistics_summary(
    painter: Painter,
    width: int,
    height: int,
    counts: tuple[GalleryCount, ...],
) -> None:
    """绘制统计摘要、Top 10 环形饼图及图例。"""

    total = sum(item.count for item in counts)
    gallery_count = len(counts)
    average = total / gallery_count if gallery_count else 0
    largest = counts[0] if counts else None

    metric_width = 305
    painter.roundrect(
        (0, 0),
        (metric_width, height),
        (232, 244, 249, 205),
        radius=20,
    )
    _draw_text(painter, '图库概览', 24, 23, font=DEFAULT_BOLD_FONT, size=19)

    metrics = (
        ('图片总数', f'{total:,}'),
        ('图库数量', f'{gallery_count:,}'),
        ('平均每库', f'{average:.1f}'),
    )
    for index, (label, value) in enumerate(metrics):
        top = 76 + index * 92
        _draw_text(
            painter,
            label,
            24,
            top,
            size=14,
            color=NSY_DRAW_THEME.text_muted,
        )
        _draw_text(
            painter,
            value,
            24,
            top + 25,
            font=DEFAULT_BOLD_FONT,
            size=29,
            color=NSY_DRAW_THEME.accent_dark,
        )

    largest_name = largest.name if largest else '暂无'
    largest_count = largest.count if largest else 0
    _draw_emoji_textbox(
        painter,
        largest_name,
        24,
        height - 78,
        metric_width - 48,
        font=DEFAULT_BOLD_FONT,
        size=18,
        height=28,
    )
    _draw_text(
        painter,
        '图片最多',
        24,
        height - 102,
        size=14,
        color=NSY_DRAW_THEME.text_muted,
    )
    _draw_text(
        painter,
        f'{largest_count:,} 张',
        24,
        height - 43,
        size=15,
        color=NSY_DRAW_THEME.accent,
    )

    entries = _pie_entries(counts)
    donut_size = 365
    donut_x = metric_width + 36
    donut_y = (height - donut_size) // 2
    current_angle = -90.0
    for index, (name, count) in enumerate(entries):
        color = _gallery_chart_color(index, name)
        angle = 360 * count / total if total else 0
        painter.pieslice(
            (donut_x, donut_y),
            (donut_size, donut_size),
            current_angle,
            current_angle + angle,
            color,
            stroke=(255, 255, 255, 255),
            stroke_width=2,
        )
        current_angle += angle

    inner_size = 205
    inner_offset = (donut_size - inner_size) // 2
    painter.pieslice(
        (donut_x + inner_offset, donut_y + inner_offset),
        (inner_size, inner_size),
        0,
        360,
        (255, 255, 255, 255),
    )
    _draw_text(
        painter,
        f'{total:,}',
        donut_x + donut_size // 2,
        donut_y + donut_size // 2 - 28,
        font=DEFAULT_BOLD_FONT,
        size=31,
        color=NSY_DRAW_THEME.text_primary,
        align='center',
    )
    _draw_text(
        painter,
        '张图片',
        donut_x + donut_size // 2,
        donut_y + donut_size // 2 + 12,
        size=15,
        color=NSY_DRAW_THEME.text_muted,
        align='center',
    )

    legend_x = donut_x + donut_size + 45
    legend_width = max(1, width - legend_x)
    _draw_text(
        painter,
        '数量占比 · 前 10 名',
        legend_x,
        15,
        font=DEFAULT_BOLD_FONT,
        size=18,
    )
    if not entries:
        _draw_text(
            painter,
            '暂无图片数据',
            legend_x,
            62,
            size=16,
            color=NSY_DRAW_THEME.text_muted,
        )
        return

    legend_top = 52
    legend_row_height = 39
    for index, (name, count) in enumerate(entries):
        y = legend_top + index * legend_row_height
        color = _gallery_chart_color(index, name)
        painter.roundrect((legend_x, y + 4), (14, 14), color, radius=5)
        _draw_emoji_textbox(
            painter,
            name,
            legend_x + 25,
            y - 2,
            max(1, legend_width - 155),
            font=DEFAULT_FONT,
            size=15,
            height=26,
        )
        percentage = count / total * 100 if total else 0
        _draw_text(
            painter,
            f'{count:,} · {percentage:.1f}%',
            width,
            y,
            size=14,
            color=NSY_DRAW_THEME.text_secondary,
            align='right',
        )


def _statistics_summary_widget(counts: Sequence[GalleryCount], width: int) -> Spacer:
    """把统计摘要封装成 draw 控件。"""

    frozen_counts = tuple(counts)
    widget = Spacer(width, STATS_SUMMARY_HEIGHT)
    widget.add_draw_func(
        lambda _, painter: _draw_statistics_summary(
            painter,
            width,
            STATS_SUMMARY_HEIGHT,
            frozen_counts,
        )
    )
    return widget


def _draw_bar_chart(
    painter: Painter,
    width: int,
    height: int,
    page_counts: tuple[GalleryCount, ...],
    axis_max: int,
    ticks: tuple[int, ...],
) -> None:
    """绘制一页横向柱状图，所有页面共用同一数量坐标轴。"""

    if not page_counts:
        _draw_text(
            painter,
            '暂无图库',
            width // 2,
            height // 2 - 10,
            font=DEFAULT_BOLD_FONT,
            size=22,
            color=NSY_DRAW_THEME.text_muted,
            align='center',
        )
        return

    top = 58
    label_width = 238
    plot_x = label_width + 18
    count_space = 58
    plot_width = width - plot_x - count_space
    bottom = top + len(page_counts) * BAR_ROW_HEIGHT

    _draw_text(
        painter,
        '图库',
        label_width - 10,
        4,
        font=DEFAULT_BOLD_FONT,
        size=15,
        color=NSY_DRAW_THEME.text_secondary,
        align='right',
    )
    _draw_text(
        painter,
        '图片数量',
        plot_x,
        4,
        font=DEFAULT_BOLD_FONT,
        size=15,
        color=NSY_DRAW_THEME.text_secondary,
    )

    for tick in ticks:
        x = plot_x + round(plot_width * tick / axis_max)
        painter.rect(
            (x, top - 9),
            (1, max(1, bottom - top + 10)),
            (207, 219, 229, 175),
        )
        _draw_text(
            painter,
            str(tick),
            x,
            30,
            size=13,
            color=NSY_DRAW_THEME.text_muted,
            align='center',
        )

    for index, item in enumerate(page_counts):
        y = top + index * BAR_ROW_HEIGHT
        if index % 2:
            painter.roundrect(
                (0, y),
                (width, BAR_ROW_HEIGHT - 1),
                (241, 246, 250, 125),
                radius=8,
            )
        _draw_emoji_textbox(
            painter,
            item.name,
            0,
            y + 1,
            label_width - 12,
            font=DEFAULT_FONT,
            size=15,
            height=27,
            align='right',
        )

        bar_width = round(plot_width * item.count / axis_max) if item.count else 0
        if bar_width:
            painter.roundrect(
                (plot_x, y + 5),
                (max(3, bar_width), BAR_ROW_HEIGHT - 10),
                SMOOTH_COLORS[index % len(SMOOTH_COLORS)],
                radius=8,
            )
        value_x = min(width, plot_x + bar_width + 10)
        _draw_text(
            painter,
            str(item.count),
            value_x,
            y + 5,
            font=DEFAULT_BOLD_FONT,
            size=14,
            color=NSY_DRAW_THEME.accent_dark,
        )


def _bar_chart_widget(
    page_counts: Sequence[GalleryCount],
    all_counts: Sequence[GalleryCount],
    width: int,
) -> Spacer:
    """构造固定宽度、随当前页图库数增长的柱状图控件。"""

    frozen_page_counts = tuple(page_counts)
    max_count = max((item.count for item in all_counts), default=0)
    axis_max, ticks = _nice_axis(max_count)
    height = max(150, 68 + len(frozen_page_counts) * BAR_ROW_HEIGHT)
    widget = Spacer(width, height)
    widget.add_draw_func(
        lambda _, painter: _draw_bar_chart(
            painter,
            width,
            height,
            frozen_page_counts,
            axis_max,
            tuple(ticks),
        )
    )
    return widget


async def _render_gallery_statistics(
    counts: Sequence[GalleryCount],
) -> Image.Image:
    """
    将完整 NSY 图库统计渲染为一张自适应高度的长图。

    图库按数量降序排列，每个图库对应一行柱状图，因此画布高度会随图库
    数量自然增长；饼图仅显示前十名具体图库，其余合并为“其他”。
    """

    ordered_counts = sorted(counts, key=lambda item: (-item.count, item.name))
    total = sum(item.count for item in ordered_counts)
    background = await resolve_configured_draw_background(
        config, theme=NSY_DRAW_THEME, logger=logger, label="图库统计"
    )
    with create_report_canvas(
        background=background,
        width=STATS_CANVAS_WIDTH,
        theme=NSY_DRAW_THEME,
    ) as canvas:
        with create_report_column(STATS_CONTENT_WIDTH, gap=17, theme=NSY_DRAW_THEME):
            report_header(
                'NSY 图库统计',
                width=STATS_CONTENT_WIDTH,
                eyebrow='GALLERY OVERVIEW',
                meta=f'{len(ordered_counts):,} 个图库 · {total:,} 张图片',
                theme=NSY_DRAW_THEME,
            )

            with report_card(STATS_CONTENT_WIDTH, padding=22, theme=NSY_DRAW_THEME):
                _statistics_summary_widget(
                    ordered_counts,
                    STATS_CONTENT_WIDTH - 44,
                )

            report_section_title(
                '各图库图片数量',
                width=STATS_CONTENT_WIDTH,
                theme=NSY_DRAW_THEME,
            )
            with report_card(STATS_CONTENT_WIDTH, padding=22, theme=NSY_DRAW_THEME):
                _bar_chart_widget(
                    ordered_counts,
                    ordered_counts,
                    STATS_CONTENT_WIDTH - 44,
                )
    return await canvas.get_img()


def _preview_tile(item: PreparedGalleryPreviewItem) -> VSplit:
    """构造含 PID 与跨图库提示的单个缩略图卡片。"""

    with (
        VSplit()
        .set_w(PREVIEW_TILE_WIDTH)
        .set_padding(8)
        .set_sep(6)
        .set_bg(
            RoundRectBg(
                fill=(255, 255, 255, 226),
                radius=18,
                stroke=(255, 255, 255, 245),
                stroke_width=1,
            )
        )
        .set_content_and_item_align('c')
    ) as tile:
        ImageBox(item.image, use_alphablend=True)
        TextBox(
            f'PID {item.pid}',
            themed_text_style('heading', size=16, theme=NSY_DRAW_THEME),
        ).set_w(PREVIEW_IMAGE_SIZE[0]).set_padding(0).set_content_align('c')
        related_text = (
            '多人图 · ' + ' / '.join(item.related_galleries)
            if len(item.related_galleries) > 1 else ' '
        )
        TextBox(
            related_text,
            themed_text_style('muted', size=12, theme=NSY_DRAW_THEME),
            line_count=2,
            line_sep=3,
        ).set_w(PREVIEW_IMAGE_SIZE[0]).set_padding(0).set_content_align('c')
    return tile


async def _render_gallery_preview(
    gallery: str,
    aliases: Sequence[str],
    items: Sequence[GalleryPreviewItem],
) -> Image.Image:
    """将指定图库的全部缩略图渲染为一张自适应高度的长图。"""

    prepared = await run_in_pool(
        _prepare_gallery_preview_items,
        tuple(items),
    )
    alias_text = ' / '.join(aliases)
    background = await resolve_configured_draw_background(
        config, theme=NSY_DRAW_THEME, logger=logger, label="图库预览"
    )
    with create_report_canvas(
        background=background,
        width=PREVIEW_CANVAS_WIDTH,
        theme=NSY_DRAW_THEME,
    ) as canvas:
        with create_report_column(PREVIEW_CONTENT_WIDTH, gap=14, theme=NSY_DRAW_THEME):
            report_header(
                f'图库 · {gallery}',
                width=PREVIEW_CONTENT_WIDTH,
                eyebrow='NSY GALLERY',
                meta=f'{len(items):,} 张',
                theme=NSY_DRAW_THEME,
            )
            if alias_text:
                TextBox(
                    f'别名：{alias_text}',
                    themed_text_style('secondary', size=14, theme=NSY_DRAW_THEME),
                    use_real_line_count=True,
                ).set_w(PREVIEW_CONTENT_WIDTH).set_padding(0)
            with (
                Grid(col_count=PREVIEW_COLUMN_COUNT)
                .set_sep(PREVIEW_GRID_GAP, PREVIEW_GRID_GAP)
                .set_item_align('t')
            ):
                for item in prepared:
                    _preview_tile(item)
    return await canvas.get_img()


def _format_file_size(size: int) -> str:
    """以简短单位显示文件体积。"""

    if size < 1024:
        return f'{size} B'
    if size < 1024 * 1024:
        return f'{size / 1024:.1f} KiB'
    return f'{size / 1024 / 1024:.2f} MiB'


def _detail_info_cell(label: str, value: str, width: int) -> Frame:
    """构造单图报告中的紧凑元数据卡。"""

    with (
        Frame()
        .set_w(width)
        .set_padding((15, 12))
        .set_bg(subtle_panel_background(radius=15, theme=NSY_DRAW_THEME))
        .set_content_align('lt')
    ) as cell:
        with VSplit().set_w(width - 30).set_sep(5).set_content_and_item_align('l'):
            TextBox(
                label,
                themed_text_style('muted', size=12, theme=NSY_DRAW_THEME),
            ).set_w(width - 30).set_padding(0)
            TextBox(
                value,
                themed_text_style('body', size=15, theme=NSY_DRAW_THEME),
                line_count=2,
            ).set_w(width - 30).set_padding(0)
    return cell


def _format_upload_group(group_id: int | None) -> str:
    if group_id is None:
        return '未记录'
    if group_id == 0:
        return '私聊'
    return str(group_id)


async def _render_image_detail(
    detail: ImageDetail,
    *,
    show_source: bool,
) -> Image.Image:
    """
    用 draw 渲染 PID 单图与简洁元数据。

    ``show_source`` 为唯一的上传来源显示开关；普通用户的渲染树中完全
    不创建上传者与上传群控件，避免仅靠遮盖文字造成信息泄漏。
    """

    try:
        preview = await run_in_pool(
            _load_detail_image,
            detail.path,
            (DETAIL_CONTENT_WIDTH - 84, 660),
        )
    except Exception:
        raise ReplyException(
            f'图片 pid={detail.pid} 无法读取，请联系管理员重载图库'
        )
    inner_width = DETAIL_CONTENT_WIDTH - 44
    info_gap = 12
    info_width = (inner_width - info_gap) // 2
    is_multi_gallery = len(detail.related_galleries) > 1

    background = await resolve_configured_draw_background(
        config, theme=NSY_DRAW_THEME, logger=logger, label="图库单图"
    )
    with create_report_canvas(
        background=background,
        width=DETAIL_CANVAS_WIDTH,
        theme=NSY_DRAW_THEME,
    ) as canvas:
        with create_report_column(DETAIL_CONTENT_WIDTH, gap=17, theme=NSY_DRAW_THEME):
            report_header(
                f'PID {detail.pid}',
                width=DETAIL_CONTENT_WIDTH,
                eyebrow='NSY 单图查询',
                meta=f'图库 · {detail.gallery}',
                theme=NSY_DRAW_THEME,
            )
            with report_card(DETAIL_CONTENT_WIDTH, padding=22, theme=NSY_DRAW_THEME):
                with VSplit().set_w(inner_width).set_sep(15).set_content_and_item_align('l'):
                    report_image_panel(
                        preview,
                        width=inner_width,
                        height=preview.height + 32,
                        padding=16,
                        compact=True,
                        theme=NSY_DRAW_THEME,
                    )
                    with (
                        Grid(col_count=2)
                        .set_sep(info_gap, info_gap)
                        .set_item_size_mode('fixed')
                        .set_item_align('t')
                    ):
                        _detail_info_cell('图库', detail.gallery, info_width)
                        _detail_info_cell('格式', detail.format or '未知', info_width)
                        dimensions = (
                            f'{detail.width} × {detail.height}'
                            if detail.width and detail.height else '未记录'
                        )
                        _detail_info_cell('尺寸', dimensions, info_width)
                        _detail_info_cell('文件大小', _format_file_size(detail.size), info_width)
                        _detail_info_cell('添加时间', detail.created_at or '未记录', info_width)
                        link_state = (
                            f'多人图 · {len(detail.related_galleries)} 个图库'
                            if is_multi_gallery else '独立图片'
                        )
                        _detail_info_cell('链接状态', link_state, info_width)

                    if is_multi_gallery:
                        link_text = '关联图库：' + ' / '.join(detail.related_galleries)
                        if detail.link_root_pid is not None:
                            link_text += (
                                f'\n根 PID：{detail.link_root_pid} · '
                                f'索引 {detail.linked_record_count} 条'
                            )
                        report_description_panel(
                            link_text,
                            width=inner_width,
                            label='多人图',
                            theme=NSY_DRAW_THEME,
                        )

                    if show_source:
                        uploader = (
                            str(detail.uploader_id)
                            if detail.uploader_id is not None else '未记录'
                        )
                        report_description_panel(
                            f'上传者：{uploader}\n上传群：{_format_upload_group(detail.group_id)}',
                            width=inner_width,
                            label='上传来源 · 仅超级管理员可见',
                            theme=NSY_DRAW_THEME,
                        )
    return await canvas.get_img()


async def render_gallery_statistics(
    counts: Sequence[GalleryCount],
    *,
    group_id: int | str | None = None,
) -> Image.Image:
    """按群级配置的主题绘制图库统计。"""

    token = _NSY_STYLE_CONTEXT.set(_resolve_nsy_style(group_id))
    try:
        return await _render_gallery_statistics(counts)
    finally:
        _NSY_STYLE_CONTEXT.reset(token)


async def render_gallery_preview(
    gallery: str,
    aliases: Sequence[str],
    items: Sequence[GalleryPreviewItem],
    *,
    group_id: int | str | None = None,
) -> Image.Image:
    """按群级配置的主题绘制图库缩略预览。"""

    token = _NSY_STYLE_CONTEXT.set(_resolve_nsy_style(group_id))
    try:
        return await _render_gallery_preview(gallery, aliases, items)
    finally:
        _NSY_STYLE_CONTEXT.reset(token)


async def render_image_detail(
    detail: ImageDetail,
    *,
    show_source: bool,
    group_id: int | str | None = None,
) -> Image.Image:
    """按群级配置的主题绘制 PID 单图报告。"""

    token = _NSY_STYLE_CONTEXT.set(_resolve_nsy_style(group_id))
    try:
        return await _render_image_detail(detail, show_source=show_source)
    finally:
        _NSY_STYLE_CONTEXT.reset(token)


def _linked_family(manager: NsyManager, image: NsyImage) -> list[NsyImage]:
    """
    返回图片所在的完整硬链接族。

    子记录先沿 ``linked_to_pid`` 找到根，再由根的 ``linked_pids`` 展开；
    同时补扫指向根的记录，以便旧索引的根反向列表不完整时仍能正确提示。
    """

    root = image
    visited = set()
    while root.linked_to_pid is not None and root.pid not in visited:
        visited.add(root.pid)
        indexed_root = manager.find_image(root.linked_to_pid)
        if indexed_root is None:
            break
        root = indexed_root

    pids = {root.pid, *root.linked_pids}

    def points_to_root(indexed: NsyImage) -> bool:
        """兼容旧索引中的链式指向，并用 visited 防止畸形环。"""

        current = indexed
        seen = set()
        while current.linked_to_pid is not None and current.pid not in seen:
            seen.add(current.pid)
            parent = manager.find_image(current.linked_to_pid)
            if parent is None:
                return False
            current = parent
        return current.pid == root.pid

    pids.update(
        indexed.pid
        for indexed in manager.images_by_pid.values()
        if points_to_root(indexed)
    )
    family = [manager.find_image(pid) for pid in sorted(pids)]
    return [indexed for indexed in family if indexed is not None]


def _gallery_counts(manager: NsyManager) -> list[GalleryCount]:
    """统计全部规范图库，包含索引为空的图库。"""

    names = set(manager.aliases_by_gallery) | set(manager.images_by_gallery)
    return [
        GalleryCount(name, len(manager.images_by_gallery.get(name, {})))
        for name in names
    ]


def _gallery_preview_items(
    manager: NsyManager,
    gallery: str,
) -> list[GalleryPreviewItem]:
    """按 PID 排序构造图库全量预览数据。"""

    images = sorted(
        manager.images_by_gallery.get(gallery, {}).values(),
        key=lambda image: image.pid,
    )
    forward_link_targets = {
        indexed.linked_to_pid
        for indexed in manager.images_by_pid.values()
        if indexed.linked_to_pid is not None
    }
    family_by_pid: dict[int, list[NsyImage]] = {}
    items = []
    for image in images:
        family = family_by_pid.get(image.pid)
        if family is None:
            has_link_relation = (
                image.linked_to_pid is not None
                or bool(image.linked_pids)
                or image.pid in forward_link_targets
            )
            family = (
                _linked_family(manager, image)
                if has_link_relation else [image]
            )
            for member in family:
                family_by_pid[member.pid] = family
        related_galleries = tuple(sorted({member.gallery for member in family}))
        items.append(GalleryPreviewItem(
            pid=image.pid,
            path=manager.get_image_path(image),
            related_galleries=(
                related_galleries if len(related_galleries) > 1 else ()
            ),
        ))
    return items


def _image_detail(manager: NsyManager, image: NsyImage) -> ImageDetail:
    """将索引模型转换为不会被渲染层修改的单图 DTO。"""

    family = _linked_family(manager, image)
    related_galleries = tuple(sorted({item.gallery for item in family}))
    if len(related_galleries) <= 1:
        related_galleries = ()
        root_pid = None
    else:
        root = next(
            (member for member in family if member.linked_to_pid is None),
            None,
        )
        root_pid = root.pid if root is not None else image.pid
    return ImageDetail(
        pid=image.pid,
        path=manager.get_image_path(image),
        gallery=image.gallery,
        format=image.format,
        width=image.width,
        height=image.height,
        size=image.size,
        created_at=image.created_at,
        uploader_id=image.uploader_id,
        group_id=image.group_id,
        related_galleries=related_galleries,
        link_root_pid=root_pid,
        linked_record_count=len(family),
    )


# ======================= 指令处理 ======================= #

nsy_gallery_query = _nsy_cmd(
    ['/查图库', '查图库'],
    force_whitespace=True,
)


@nsy_gallery_query.handle()
async def _(ctx: HandlerContext):
    args = ctx.get_args().strip().split()
    if len(args) > 1:
        raise ReplyException('使用方式: /查图库 [图库名称/别名]')

    manager = NsyManager.get()
    if args:
        gallery = manager.resolve_gallery(args[0], raise_if_missing=True)
        await ctx.block(f'nsy:gallery-query:{gallery}')
        items = _gallery_preview_items(manager, gallery)
        if not items:
            return await ctx.asend_reply_msg(f'图库“{gallery}”暂无图片')
        report = await render_gallery_preview(
            gallery,
            manager.aliases_by_gallery.get(gallery, []),
            items,
            group_id=ctx.group_id,
        )
        return await ctx.asend_reply_msg(
            await get_image_cq(report, low_quality=True)
        )

    await ctx.block('nsy:gallery-query:all')
    report = await render_gallery_statistics(
        _gallery_counts(manager),
        group_id=ctx.group_id,
    )
    await ctx.asend_reply_msg(
        await get_image_cq(report, low_quality=False)
    )


nsy_image_info = _nsy_cmd(['/查图信息'])


@nsy_image_info.handle()
async def _(ctx: HandlerContext):
    manager = NsyManager.get()
    reply_msg = ctx.get_reply_msg()
    if reply_msg:
        image_datas = extract_image_data(reply_msg)
        if len(image_datas) != 1:
            raise ReplyException('回复消息中须包含一张图片')

        await ctx.block('nsy:image-info:reply')
        temp = _get_image_temp_file(ctx, image_datas[0])
        async with temp as path:
            _inspect_image_file(path, check_size=False)
            image_hash = _sha256_file(path)

        images = manager.find_images_by_hash(image_hash)
        if not images:
            return await ctx.asend_reply_msg('该图片不在 NSY 索引中')
        # 同一文件可能链接到多个图库；优先选择仍可读取的根记录，
        # 再按 PID 稳定选择，避免失效根遮住可用的链接记录。
        image = min(
            images,
            key=lambda item: (
                not Path(manager.get_image_path(item)).is_file(),
                item.linked_to_pid is not None,
                item.pid,
            ),
        )
    else:
        args = ctx.get_args().strip().split()
        if len(args) != 1:
            raise ReplyException(
                '使用方式: /查图信息 pid，或回复一张图片发送 /查图信息'
            )
        try:
            pid = int(args[0])
        except ValueError:
            raise ReplyException('pid 必须是整数')
        if pid <= 0:
            raise ReplyException('pid 必须是正整数')

        await ctx.block(f'nsy:image-info:{pid}')
        image = manager.find_image(pid, raise_if_missing=True)

    detail = _image_detail(manager, image)
    if not Path(detail.path).is_file():
        raise ReplyException(
            f'图片 pid={image.pid} 的文件不存在，请联系管理员重载图库'
        )

    report = await render_image_detail(
        detail,
        show_source=check_superuser(ctx.event),
        group_id=ctx.group_id,
    )
    await ctx.asend_reply_msg(
        await get_image_cq(report, low_quality=True)
    )


__all__ = [
    'GalleryCount',
    'GalleryPreviewItem',
    'ImageDetail',
    'render_gallery_preview',
    'render_gallery_statistics',
    'render_image_detail',
]
