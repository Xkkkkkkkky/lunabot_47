from __future__ import annotations

import colorsys
import hashlib
import math
import random
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageFilter


NormalizedRegion = tuple[float, float, float, float]


def _validate_params(
    output_size: tuple[int, int],
    color_count: int,
    saturation: float,
    distortion: float,
    blur: float,
    excluded_regions: Sequence[NormalizedRegion],
) -> tuple[int, int]:
    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("output_size 的宽高必须大于 0")
    if color_count < 2:
        raise ValueError("color_count 必须至少为 2")
    if saturation < 0:
        raise ValueError("saturation 不能小于 0")
    if distortion < 0:
        raise ValueError("distortion 不能小于 0")
    if blur < 0:
        raise ValueError("blur 不能小于 0")
    for region in excluded_regions:
        x1, y1, x2, y2 = region
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("excluded_regions 必须使用 0 到 1 范围内的有效归一化坐标")
    return int(width), int(height)


def _extract_representative_colors(
    source_image: Image.Image,
    color_count: int,
    excluded_regions: Sequence[NormalizedRegion],
) -> list[tuple[int, int, int]]:
    sample = source_image.convert("RGBA")
    sample.thumbnail((240, 120), Image.Resampling.LANCZOS)
    pixels = np.asarray(sample)
    rgb = pixels[..., :3]
    alpha = pixels[..., 3]
    luminance = (
        rgb[..., 0] * 0.2126
        + rgb[..., 1] * 0.7152
        + rgb[..., 2] * 0.0722
    ) / 255

    excluded_mask = np.zeros(alpha.shape, dtype=bool)
    height, width = alpha.shape
    for x1, y1, x2, y2 in excluded_regions:
        excluded_mask[
            round(y1 * height):round(y2 * height),
            round(x1 * width):round(x2 * width),
        ] = True

    # 排除透明区域和近黑、近白像素，避免边框与文字主导取色结果。
    visible_mask = alpha >= 48
    theme_mask = (
        visible_mask
        & (luminance >= 0.08)
        & (luminance <= 0.96)
        & ~excluded_mask
    )
    theme_pixels = rgb[theme_mask]
    if not len(theme_pixels):
        theme_pixels = rgb[visible_mask & ~excluded_mask]
    if not len(theme_pixels):
        theme_pixels = rgb[visible_mask]

    fallback_colors = [
        (150, 185, 220),
        (190, 175, 215),
        (225, 190, 205),
        (175, 210, 195),
    ]
    if not len(theme_pixels):
        return [fallback_colors[i % len(fallback_colors)] for i in range(color_count)]

    palette_size = min(max(24, color_count * 6), 64, len(theme_pixels))
    palette_source = Image.fromarray(theme_pixels.reshape(1, -1, 3).astype(np.uint8))
    quantized = palette_source.quantize(
        colors=palette_size,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    palette = quantized.getpalette()
    candidates = []
    for count, color_index in quantized.getcolors(maxcolors=palette_size):
        color = tuple(palette[color_index * 3:color_index * 3 + 3])
        hue, lightness, color_saturation = colorsys.rgb_to_hls(
            *(channel / 255 for channel in color)
        )
        midtone_weight = max(0.4, 1 - abs(lightness - 0.55))
        score = count * (0.18 + color_saturation) ** 2 * midtone_weight
        candidates.append({
            "color": color,
            "hue": hue,
            "saturation": color_saturation,
            "score": score,
        })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    saturated_candidates = [item for item in candidates if item["saturation"] >= 0.18]
    if len(saturated_candidates) >= color_count:
        candidates = saturated_candidates

    selected = []
    for candidate in candidates:
        if not selected:
            selected.append(candidate)
        else:
            hue_distances = [
                min(
                    abs(candidate["hue"] - item["hue"]),
                    1 - abs(candidate["hue"] - item["hue"]),
                )
                for item in selected
            ]
            if min(hue_distances) >= 0.12:
                selected.append(candidate)
        if len(selected) == color_count:
            break

    # 色相不足时用差异明显的同色系明暗补足颜色数量。
    for candidate in candidates:
        if candidate in selected:
            continue
        color = np.asarray(candidate["color"])
        if all(
            np.linalg.norm(color - np.asarray(item["color"])) >= 36
            for item in selected
        ):
            selected.append(candidate)
        if len(selected) == color_count:
            break

    while len(selected) < color_count:
        selected.append(selected[-1])
    return [item["color"] for item in selected]


def _soften_colors(
    colors: Sequence[tuple[int, int, int]],
    saturation: float,
) -> list[tuple[int, int, int, int]]:
    softened_colors = []
    for red, green, blue in colors:
        hue, lightness, source_saturation = colorsys.rgb_to_hls(
            red / 255, green / 255, blue / 255
        )
        soft_saturation = min(1.0, source_saturation * saturation)
        soft_lightness = min(0.90, max(0.72, lightness + 0.12))
        color = colorsys.hls_to_rgb(hue, soft_lightness, soft_saturation)
        softened_colors.append((*[round(channel * 255) for channel in color], 255))
    return softened_colors


def _render_organic_gradient(
    colors: Sequence[tuple[int, int, int, int]],
    output_size: tuple[int, int],
    seed: int,
    distortion: float,
    blur: float,
) -> Image.Image:
    width, height = output_size
    scale = min(1.0, 320 / max(width, height))
    work_size = (
        max(64, round(width * scale)),
        max(64, round(height * scale)),
    )
    work_width, work_height = work_size
    rng = random.Random(seed)
    masks = []
    y_coords, x_coords = np.meshgrid(
        np.linspace(0, 1, work_height),
        np.linspace(0, 1, work_width),
        indexing="ij",
    )

    for _ in colors:
        mask = np.full((work_height, work_width), 0.015, dtype=np.float32)

        # 主径向场叠加多个大色块，再用低频坐标扰动制造柔和的不规则边缘。
        for blob_index in range(6):
            if blob_index == 0:
                radius_x = rng.uniform(0.34, 0.58)
                radius_y = rng.uniform(0.24, 0.46)
                amplitude = rng.uniform(1.25, 1.60)
            else:
                radius_x = rng.uniform(0.16, 0.38)
                radius_y = rng.uniform(0.11, 0.30)
                amplitude = rng.uniform(0.65, 1.20)

            center_x = rng.uniform(-0.08, 1.08)
            center_y = rng.uniform(-0.06, 1.06)
            warp_x = rng.uniform(0.045, 0.13) * distortion
            warp_y = rng.uniform(0.045, 0.13) * distortion
            frequency_x = rng.uniform(1.0, 2.6)
            frequency_y = rng.uniform(1.0, 2.6)
            phase_x = rng.uniform(0, 2 * math.pi)
            phase_y = rng.uniform(0, 2 * math.pi)

            distorted_x = x_coords + warp_x * np.sin(
                2 * math.pi * y_coords * frequency_y + phase_x
            )
            distorted_y = y_coords + warp_y * np.sin(
                2 * math.pi * x_coords * frequency_x + phase_y
            )
            distance = (
                ((distorted_x - center_x) / radius_x) ** 2
                + ((distorted_y - center_y) / radius_y) ** 2
            )
            mask += amplitude * np.exp(-1.7 * distance)

        mask = mask / np.max(mask) * 255
        mask_image = Image.fromarray(mask.astype(np.uint8))
        blur_radius = round(min(work_size) * blur)
        if blur_radius > 0:
            mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        masks.append(np.asarray(mask_image, dtype=np.float32) / 255 + 0.015)

    weights = np.stack(masks, axis=0)
    weights /= np.sum(weights, axis=0, keepdims=True)
    result = np.einsum("chw,cd->hwd", weights, np.asarray(colors, dtype=np.float32))
    result_image = Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))
    if result_image.size != output_size:
        result_image = result_image.resize(output_size, Image.Resampling.BICUBIC)
    return result_image


def generate_organic_gradient_background(
    source_image: Image.Image,
    output_size: tuple[int, int],
    *,
    color_count: int = 3,
    saturation: float = 0.55,
    distortion: float = 1.0,
    blur: float = 0.045,
    excluded_regions: Sequence[NormalizedRegion] = (),
    seed: int | None = None,
) -> Image.Image:
    """根据源图生成指定大小的低饱和度有机渐变背景。

    ``saturation`` 是源图代表色饱和度的倍率；``distortion`` 控制色块边缘扭曲；
    ``blur`` 是相对短边的高斯模糊半径。排除区域使用归一化坐标。
    """
    output_size = _validate_params(
        output_size,
        color_count,
        saturation,
        distortion,
        blur,
        excluded_regions,
    )
    colors = _extract_representative_colors(source_image, color_count, excluded_regions)
    colors = _soften_colors(colors, saturation)
    if seed is None:
        seed_sample = source_image.convert("RGBA")
        seed_sample.thumbnail((96, 48), Image.Resampling.LANCZOS)
        seed = int(hashlib.md5(seed_sample.tobytes()).hexdigest()[:16], 16)
    return _render_organic_gradient(colors, output_size, seed, distortion, blur)
