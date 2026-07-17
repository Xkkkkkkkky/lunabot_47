"""统一的图片报告主题与通用控件模板。

该模块仅封装绘图和布局能力，不包含 Bird、STA 等业务数据。
业务模块只需选择页面、卡片、标题、信息行和图片控件进行组合，
即可保持一致的视觉风格。
"""

from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from PIL import Image, ImageDraw, ImageOps

from ..draw.img_utils import open_image
from ..draw.plot import (
    Canvas,
    DEFAULT_BOLD_FONT,
    DEFAULT_FONT,
    FillBg,
    Frame,
    HSplit,
    ImageBg,
    ImageBox,
    LinearGradient,
    RoundRectBg,
    TextBox,
    TextStyle,
    VSplit,
    Widget,
    WidgetBg,
)
from .utils import b64_to_image, download_image, global_config


RGBA = tuple[int, int, int, int]

# 色号完整取自 gameCharacterUnits.json 的 26 个角色唯一 colorCode。
# 先按色相归档，便于审阅并确保没有遗漏角色色；组内大致按色相递进。
PROJECT_SEKAI_CHARACTER_COLOR_FAMILIES: tuple[tuple[RGBA, ...], ...] = (
    (  # 红 / 橙：MEIKO、穗波、彰人
        (221, 68, 68, 255),       # #dd4444
        (238, 102, 102, 255),     # #ee6666
        (255, 119, 34, 255),      # #ff7722
    ),
    (  # 黄 / 大地：实乃理、绘名、司、铃、咲希、连
        (255, 204, 170, 255),     # #ffccaa
        (204, 170, 136, 255),     # #ccaa88
        (255, 187, 0, 255),       # #ffbb00
        (255, 204, 17, 255),      # #ffcc11
        (255, 221, 68, 255),      # #ffdd44
        (255, 238, 17, 255),      # #ffee11
    ),
    (  # 绿 / 青绿：志步、宁宁、雫、初音未来
        (187, 221, 34, 255),      # #bbdd22
        (51, 221, 153, 255),      # #33dd99
        (153, 238, 221, 255),     # #99eedd
        (51, 204, 187, 255),      # #33ccbb
    ),
    (  # 青 / 蓝：杏、一歌、冬弥、遥、KAITO
        (0, 187, 221, 255),       # #00bbdd
        (51, 170, 238, 255),      # #33aaee
        (0, 119, 221, 255),       # #0077dd
        (153, 204, 255, 255),     # #99ccff
        (51, 102, 204, 255),      # #3366cc
    ),
    (  # 紫：真冬、类
        (136, 136, 204, 255),     # #8888cc
        (187, 136, 238, 255),     # #bb88ee
    ),
    (  # 粉：瑞希、笑梦、奏、心羽、爱莉、巡音流歌
        (221, 170, 204, 255),     # #ddaacc
        (255, 102, 187, 255),     # #ff66bb
        (187, 102, 136, 255),     # #bb6688
        (255, 102, 153, 255),     # #ff6699
        (255, 170, 204, 255),     # #ffaacc
        (255, 187, 204, 255),     # #ffbbcc
    ),
)
PROJECT_SEKAI_CHARACTER_COLORS: tuple[RGBA, ...] = tuple(
    color
    for family in PROJECT_SEKAI_CHARACTER_COLOR_FAMILIES
    for color in family
)

# 可配置主题的角色原色。Project SEKAI 色号来自本地 masterdata；
# BanG Dream! 色号来自角色代表色资料。实际绘图不会直接使用这些高饱和原色。
DRAW_THEME_BASE_PALETTES: dict[str, tuple[RGBA, ...]] = {
    "niigo": (
        (187, 102, 136, 255), (136, 136, 204, 255),
        (204, 170, 136, 255), (221, 170, 204, 255),
    ),
    "leoneed": (
        (51, 170, 238, 255), (255, 221, 68, 255),
        (238, 102, 102, 255), (187, 221, 34, 255),
    ),
    "mmj": (
        (255, 204, 170, 255), (153, 204, 255, 255),
        (255, 170, 204, 255), (153, 238, 221, 255),
    ),
    "vbs": (
        (255, 102, 153, 255), (0, 187, 221, 255),
        (255, 119, 34, 255), (0, 119, 221, 255),
    ),
    "wxs": (
        (255, 187, 0, 255), (255, 102, 187, 255),
        (51, 221, 153, 255), (187, 136, 238, 255),
    ),
    "virtual_singer": (
        (51, 204, 187, 255), (255, 204, 17, 255),
        (255, 238, 17, 255), (255, 187, 204, 255),
        (221, 68, 68, 255), (51, 102, 204, 255),
    ),
    "poppin_party": (
        (255, 84, 34, 255), (0, 119, 221, 255),
        (255, 84, 187, 255), (252, 190, 3, 255),
        (169, 102, 222, 255),
    ),
    "afterglow": (
        (238, 1, 33, 255), (0, 203, 170, 255),
        (255, 153, 152, 255), (187, 0, 51, 255),
        (254, 228, 80, 255),
    ),
    "pastel_palettes": (
        (255, 136, 187, 255), (84, 221, 238, 255),
        (251, 216, 122, 255), (152, 221, 135, 255),
        (222, 187, 255, 255),
    ),
    "roselia": (
        (135, 16, 136, 255), (0, 170, 187, 255),
        (221, 34, 0, 255), (220, 0, 135, 255),
        (187, 187, 187, 255),
    ),
    "hello_happy_world": (
        (255, 221, 0, 255), (170, 51, 204, 255),
        (255, 153, 34, 255), (67, 221, 255, 255),
        (0, 102, 153, 255), (221, 51, 204, 255),
    ),
    "morfonica": (
        (103, 118, 204, 255), (238, 102, 102, 255),
        (238, 118, 68, 255), (238, 118, 135, 255),
        (103, 152, 136, 255),
    ),
    "raise_a_suilen": (
        (204, 9, 15, 255), (187, 255, 100, 255),
        (227, 186, 58, 255), (255, 153, 190, 255),
        (63, 184, 255, 255),
    ),
    "mygo": (
        (118, 187, 219, 255), (253, 136, 152, 255),
        (119, 220, 119, 255), (250, 216, 132, 255),
        (119, 119, 169, 255),
    ),
    "ave_mujica": (
        (187, 153, 85, 255), (119, 153, 119, 255),
        (51, 85, 102, 255), (170, 68, 119, 255),
        (119, 153, 204, 255),
    ),
    "mugendai_mewtype": (
        (255, 238, 85, 255), (255, 187, 204, 255),
        (68, 119, 204, 255), (153, 119, 204, 255),
        (238, 85, 119, 255),
    ),
}
DRAW_THEME_ALIASES = {
    "25ji": "niigo", "25时": "niigo", "nightcord": "niigo",
    "l/n": "leoneed", "leo/need": "leoneed", "ln": "leoneed",
    "more_more_jump": "mmj", "vivid_bad_squad": "vbs",
    "wonderlands_showtime": "wxs", "virtualsinger": "virtual_singer",
    "popipa": "poppin_party", "poppinparty": "poppin_party",
    "poppin'party": "poppin_party",
    "pasupare": "pastel_palettes", "pastelpalettes": "pastel_palettes",
    "pastel*palettes": "pastel_palettes",
    "harohapi": "hello_happy_world", "hellohappyworld": "hello_happy_world",
    "monica": "morfonica", "ras": "raise_a_suilen",
    "mygo!!!!!": "mygo", "avemujica": "ave_mujica",
    "mugendai": "mugendai_mewtype", "mugen": "mugendai_mewtype",
}

# BanG Dream 团体主色用于限定插值色带的中心；成员色仍决定色带内部
# 的相对层次。这样既保留团体辨识度，又不会把互补色直接放在一起。
DRAW_THEME_PRIMARY_COLORS: dict[str, RGBA] = {
    "poppin_party": (255, 51, 119, 255),
    "afterglow": (238, 51, 68, 255),
    "pastel_palettes": (51, 221, 170, 255),
    "roselia": (51, 68, 170, 255),
    "hello_happy_world": (255, 192, 42, 255),
    "morfonica": (51, 170, 255, 255),
    "raise_a_suilen": (34, 204, 204, 255),
    "mygo": (50, 136, 187, 255),
    "ave_mujica": (136, 17, 68, 255),
    "mugendai_mewtype": (236, 115, 132, 255),
}
DEFAULT_DRAW_THEME_NAME = "niigo"
UNIFIED_OTHER_CHART_COLOR: RGBA = (112, 119, 132, 255)


def normalize_draw_theme_name(name: str | None) -> str:
    """规范化主题名；未知配置安全回退到 25时。"""

    normalized = str(name or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = DRAW_THEME_ALIASES.get(normalized, normalized)
    return normalized if normalized in DRAW_THEME_BASE_PALETTES else DEFAULT_DRAW_THEME_NAME


def _group_theme_settings(group_id: int | str | None) -> dict[str, Any]:
    """读取 ``global.draw.group_themes`` 中指定群的主题覆盖。"""

    if group_id is None:
        return {}
    settings = global_config.get(
        "draw.group_themes", {}, raise_exc=False
    ) or {}
    if not isinstance(settings, dict):
        return {}
    group_settings = settings.get(str(group_id))
    if group_settings is None:
        group_settings = settings.get(group_id)
    return group_settings if isinstance(group_settings, dict) else {}


def _theme_setting(
    module_config: Any,
    key: str,
    default: Any,
    *,
    group_id: int | str | None = None,
) -> Any:
    """按“群级覆盖、插件覆盖、全局默认”的顺序读取主题配置。"""

    group_value = _group_theme_settings(group_id).get(key)
    if group_value not in (None, ""):
        return group_value

    local_value = None
    if module_config is not None:
        local_value = module_config.get(
            f"draw.theme.{key}", None, raise_exc=False
        )
    if local_value not in (None, ""):
        return local_value
    return global_config.get(f"draw.theme.{key}", default, raise_exc=False)


def get_draw_theme_name(
    module_config: Any = None,
    *,
    group_id: int | str | None = None,
) -> str:
    return normalize_draw_theme_name(
        _theme_setting(
            module_config,
            "palette",
            DEFAULT_DRAW_THEME_NAME,
            group_id=group_id,
        )
    )


def get_draw_theme_base_palette(
    module_config: Any = None,
    *,
    group_id: int | str | None = None,
) -> tuple[RGBA, ...]:
    return DRAW_THEME_BASE_PALETTES[
        get_draw_theme_name(module_config, group_id=group_id)
    ]


def build_member_draw_palette(
    theme_name: str | None = None,
) -> tuple[RGBA, ...]:
    """返回成员原色，并让前几个颜色尽量保持高区分度。

    首色优先选取最接近团体主色的成员色，后续优先拉开色相，再参考
    明度与饱和度差。少量数据系列可直接使用该色板，不插值。
    """

    name = normalize_draw_theme_name(theme_name)
    remaining = list(DRAW_THEME_BASE_PALETTES[name])
    if not remaining:
        return ()

    primary = DRAW_THEME_PRIMARY_COLORS.get(name)
    if primary is None:
        current = remaining.pop(0)
    else:
        current = min(
            remaining,
            key=lambda color: sum(
                (color[channel] - primary[channel]) ** 2
                for channel in range(3)
            ),
        )
        remaining.remove(current)

    ordered = [current]

    def perceptual_distance(left: RGBA, right: RGBA) -> float:
        left_hue, left_lightness, left_saturation = colorsys.rgb_to_hls(
            *(channel / 255 for channel in left[:3])
        )
        right_hue, right_lightness, right_saturation = colorsys.rgb_to_hls(
            *(channel / 255 for channel in right[:3])
        )
        hue_distance = abs(left_hue - right_hue)
        hue_distance = min(hue_distance, 1.0 - hue_distance)
        if min(left_saturation, right_saturation) < 0.08:
            hue_distance = 0.0
        return (
            hue_distance * 4.0
            + abs(left_lightness - right_lightness)
            + abs(left_saturation - right_saturation) * 0.5
        )

    while remaining:
        current = max(
            remaining,
            key=lambda color: perceptual_distance(color, current),
        )
        remaining.remove(current)
        ordered.append(current)
    return tuple(ordered)


def resolve_draw_member_palette(
    module_config: Any = None,
    *,
    group_id: int | str | None = None,
) -> tuple[RGBA, ...]:
    """读取当前插件团体配置并返回不插值的成员原色色板。"""

    return build_member_draw_palette(
        get_draw_theme_name(module_config, group_id=group_id)
    )


def _ordered_theme_anchors(base_colors: tuple[RGBA, ...]):
    """沿避开最大色相断层的方向排列锚点，避免生成互补色跳变。"""

    hls_values = []
    chromatic_hues = []
    for color in base_colors:
        red, green, blue = (channel / 255 for channel in color[:3])
        hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)
        hls_values.append([hue, lightness, saturation])
        if saturation >= 0.08:
            chromatic_hues.append(hue)

    if chromatic_hues:
        x = sum(math.cos(hue * math.tau) for hue in chromatic_hues)
        y = sum(math.sin(hue * math.tau) for hue in chromatic_hues)
        neutral_hue = (math.atan2(y, x) / math.tau) % 1.0
    else:
        neutral_hue = 0.0
    for value in hls_values:
        if value[2] < 0.08:
            value[0] = neutral_hue

    ordered = sorted(hls_values, key=lambda value: value[0])
    if len(ordered) <= 1:
        return ordered
    gaps = [
        ((ordered[(index + 1) % len(ordered)][0] - ordered[index][0]) % 1.0)
        for index in range(len(ordered))
    ]
    start = (max(range(len(gaps)), key=gaps.__getitem__) + 1) % len(ordered)
    ordered = ordered[start:] + ordered[:start]
    previous_hue = ordered[0][0]
    for index in range(1, len(ordered)):
        while ordered[index][0] < previous_hue:
            ordered[index][0] += 1.0
        previous_hue = ordered[index][0]
    return ordered


def build_smooth_draw_palette(
    theme_name: str | None = None,
    *,
    color_count: int = 24,
    saturation_scale: float = 0.72,
    lightness: float = 0.56,
    hue_span: float = 0.30,
) -> tuple[RGBA, ...]:
    """从团体角色色生成低饱和、相邻连续且无对比色的图表色轮。"""

    name = normalize_draw_theme_name(theme_name)
    anchors = _ordered_theme_anchors(DRAW_THEME_BASE_PALETTES[name])
    color_count = max(1, int(color_count))
    saturation_scale = max(0.15, min(1.0, float(saturation_scale)))
    lightness = max(0.38, min(0.72, float(lightness)))
    hue_span = max(0.08, min(0.42, float(hue_span)))

    # 成员原色可能横跨互补色。保留各角色的相对色相顺序，但把整组
    # 压缩到有限色相带内，避免同一张统计图出现生硬的冷暖对撞。
    source_span = anchors[-1][0] - anchors[0][0]
    if source_span > hue_span:
        source_center_hue = (anchors[0][0] + anchors[-1][0]) / 2
        center_hue = source_center_hue
        primary_color = DRAW_THEME_PRIMARY_COLORS.get(name)
        if primary_color is not None:
            red, green, blue = (
                channel / 255 for channel in primary_color[:3]
            )
            center_hue = colorsys.rgb_to_hls(red, green, blue)[0]
            center_hue += round(source_center_hue - center_hue)
        scale = hue_span / source_span
        for anchor in anchors:
            anchor[0] = center_hue + (
                anchor[0] - source_center_hue
            ) * scale

    softened = []
    for hue, source_lightness, source_saturation in anchors:
        softened.append((
            hue,
            max(0.42, min(0.68, lightness + (source_lightness - 0.5) * 0.16)),
            max(0.18, min(0.78, source_saturation * saturation_scale)),
        ))
    if len(softened) == 1:
        softened *= 2

    colors = []
    for index in range(color_count):
        position = (
            0
            if color_count == 1
            else index / (color_count - 1) * (len(softened) - 1)
        )
        left_index = min(len(softened) - 2, int(position))
        ratio = position - left_index
        left, right = softened[left_index], softened[left_index + 1]
        hue = left[0] + (right[0] - left[0]) * ratio
        value_lightness = left[1] + (right[1] - left[1]) * ratio
        saturation = left[2] + (right[2] - left[2]) * ratio
        red, green, blue = colorsys.hls_to_rgb(hue % 1.0, value_lightness, saturation)
        colors.append((
            round(red * 255), round(green * 255), round(blue * 255), 255
        ))
    return tuple(colors)


def _separate_categorical_colors(colors: tuple[RGBA, ...]) -> tuple[RGBA, ...]:
    """重排色带，让饼图中相邻扇区尽量不使用相邻插值色。"""

    color_count = len(colors)
    if color_count < 3:
        return colors
    step = next(
        value
        for value in range(color_count // 2, 0, -1)
        if math.gcd(value, color_count) == 1
    )
    return tuple(
        colors[(index * step) % color_count]
        for index in range(color_count)
    )


def build_categorical_draw_palette(
    theme_name: str | None = None,
    *,
    color_count: int = 10,
    saturation_scale: float = 0.92,
    lightness: float = 0.56,
    hue_span: float = 0.42,
) -> tuple[RGBA, ...]:
    """生成插值数量更少、相邻差异更大的分类图表色板。"""

    colors = build_smooth_draw_palette(
        theme_name,
        color_count=color_count,
        saturation_scale=saturation_scale,
        lightness=lightness,
        hue_span=hue_span,
    )
    return _separate_categorical_colors(colors)


def build_coverage_draw_palette(
    theme_name: str | None = None,
    *,
    color_count: int = 10,
    saturation_scale: float = 0.92,
    lightness: float = 0.56,
) -> tuple[RGBA, ...]:
    """生成保留全部成员色相的分类色板。

    前一轮颜色与团体中每个成员色一一对应，只统一收敛
    饱和度和明度。分类数更多时，再生成同色相的深浅变体，
    使饼图既覆盖完整团体配色，又不会回到对比色插值。
    """

    name = normalize_draw_theme_name(theme_name)
    member_colors = build_member_draw_palette(name)
    if not member_colors:
        return ()

    color_count = max(len(member_colors), int(color_count))
    saturation_scale = max(0.15, min(1.0, float(saturation_scale)))
    lightness = max(0.38, min(0.72, float(lightness)))
    colors = []
    for index in range(color_count):
        source = member_colors[index % len(member_colors)]
        layer = index // len(member_colors)
        red, green, blue = (channel / 255 for channel in source[:3])
        hue, source_lightness, source_saturation = colorsys.rgb_to_hls(
            red,
            green,
            blue,
        )

        # 无彩色成员色（如 Roselia 的灰色）必须保持中性，
        # 否则 HLS 的默认色相会把它错误变成红色。
        if source_saturation < 0.08:
            adjusted_saturation = 0.0
        else:
            adjusted_saturation = max(
                0.36,
                min(0.82, source_saturation * saturation_scale),
            )
            adjusted_saturation *= max(0.62, 1.0 - layer * 0.12)

        adjusted_lightness = source_lightness * 0.55 + lightness * 0.45
        if layer:
            direction = -1 if layer % 2 else 1
            adjusted_lightness += direction * 0.045 * ((layer + 1) // 2)
        adjusted_lightness = max(0.42, min(0.70, adjusted_lightness))
        adjusted = colorsys.hls_to_rgb(
            hue,
            adjusted_lightness,
            adjusted_saturation,
        )
        colors.append((*tuple(round(channel * 255) for channel in adjusted), 255))
    return tuple(colors)


def resolve_draw_palette(
    module_config: Any = None,
    *,
    color_count: int | None = None,
    group_id: int | str | None = None,
) -> tuple[RGBA, ...]:
    """按全局配置和插件覆盖配置生成当前图表色轮。"""

    configured_count = _theme_setting(
        module_config, "color_count", 24, group_id=group_id
    )
    return build_smooth_draw_palette(
        get_draw_theme_name(module_config, group_id=group_id),
        color_count=color_count or int(configured_count),
        saturation_scale=float(
            _theme_setting(
                module_config, "saturation", 0.72, group_id=group_id
            )
        ),
        lightness=float(
            _theme_setting(
                module_config, "lightness", 0.56, group_id=group_id
            )
        ),
        hue_span=float(
            _theme_setting(
                module_config, "hue_span", 0.30, group_id=group_id
            )
        ),
    )


def resolve_draw_categorical_palette(
    module_config: Any = None,
    *,
    color_count: int | None = None,
    group_id: int | str | None = None,
) -> tuple[RGBA, ...]:
    """按插件配置生成适合饼图等高区分场景的分类色板。"""

    configured_count = _theme_setting(
        module_config,
        "categorical_color_count",
        10,
        group_id=group_id,
    )
    return build_categorical_draw_palette(
        get_draw_theme_name(module_config, group_id=group_id),
        color_count=color_count or int(configured_count),
        saturation_scale=float(
            _theme_setting(
                module_config,
                "categorical_saturation",
                0.92,
                group_id=group_id,
            )
        ),
        lightness=float(
            _theme_setting(
                module_config, "lightness", 0.56, group_id=group_id
            )
        ),
        hue_span=float(
            _theme_setting(
                module_config,
                "categorical_hue_span",
                0.42,
                group_id=group_id,
            )
        ),
    )


def resolve_draw_coverage_palette(
    module_config: Any = None,
    *,
    color_count: int | None = None,
    group_id: int | str | None = None,
) -> tuple[RGBA, ...]:
    """按当前团体配置生成成员色全覆盖的分类色板。"""

    configured_count = _theme_setting(
        module_config,
        "categorical_color_count",
        10,
        group_id=group_id,
    )
    return build_coverage_draw_palette(
        get_draw_theme_name(module_config, group_id=group_id),
        color_count=color_count or int(configured_count),
        saturation_scale=float(
            _theme_setting(
                module_config,
                "categorical_saturation",
                0.92,
                group_id=group_id,
            )
        ),
        lightness=float(
            _theme_setting(
                module_config,
                "lightness",
                0.56,
                group_id=group_id,
            )
        ),
    )


PROJECT_SEKAI_CHART_COLORS = build_smooth_draw_palette(DEFAULT_DRAW_THEME_NAME)
UNIFIED_CHART_COLORS = PROJECT_SEKAI_CHART_COLORS


def _mix_rgba(color: RGBA, target: RGBA, ratio: float) -> RGBA:
    ratio = max(0.0, min(1.0, ratio))
    return tuple(
        round(source + (destination - source) * ratio)
        for source, destination in zip(color, target)
    )


_DEFAULT_THEME_PALETTE = PROJECT_SEKAI_CHART_COLORS


@dataclass(frozen=True)
class UnifiedDrawTheme:
    """通用报告主题。

    默认参数沿用蓝灰文字与半透明白色玻璃卡片，并用 25 时的
    柔和主题色作为强调色，保留足够的中性空间供其他模块使用。
    """

    page_start: RGBA = (231, 244, 255, 255)
    page_end: RGBA = (255, 247, 251, 255)
    text_primary: RGBA = (32, 48, 68, 255)
    text_secondary: RGBA = (92, 109, 127, 255)
    text_muted: RGBA = (117, 134, 153, 255)
    accent: RGBA = _DEFAULT_THEME_PALETTE[12]
    accent_dark: RGBA = _mix_rgba(
        _DEFAULT_THEME_PALETTE[12], (35, 40, 52, 255), 0.32
    )
    card_fill: RGBA = (255, 255, 255, 219)
    compact_card_fill: RGBA = (255, 255, 255, 199)
    card_stroke: RGBA = (255, 255, 255, 224)
    subtle_fill: RGBA = _mix_rgba(
        _DEFAULT_THEME_PALETTE[14], (255, 255, 255, 184), 0.86
    )
    rank_fill: RGBA = _mix_rgba(
        _DEFAULT_THEME_PALETTE[14], (255, 255, 255, 255), 0.82
    )
    accent_start: RGBA = _DEFAULT_THEME_PALETTE[9]
    accent_end: RGBA = _DEFAULT_THEME_PALETTE[14]
    content_width: int = 940
    page_padding: int = 34
    section_gap: int = 18
    card_radius: int = 22
    compact_card_radius: int = 16


DEFAULT_DRAW_THEME = UnifiedDrawTheme()


def resolve_draw_theme(
    module_config: Any = None,
    *,
    group_id: int | str | None = None,
) -> UnifiedDrawTheme:
    """让标题、徽章和次级底色跟随当前插件配置的团体主题。"""

    palette = resolve_draw_palette(module_config, group_id=group_id)
    accent_index = round((len(palette) - 1) * 0.50)
    accent_start_index = round((len(palette) - 1) * 0.38)
    accent_end_index = round((len(palette) - 1) * 0.62)
    accent = palette[accent_index]
    return replace(
        DEFAULT_DRAW_THEME,
        accent=accent,
        accent_dark=_mix_rgba(accent, (35, 40, 52, 255), 0.32),
        subtle_fill=_mix_rgba(palette[accent_end_index], (255, 255, 255, 184), 0.86),
        rank_fill=_mix_rgba(palette[accent_end_index], (255, 255, 255, 255), 0.82),
        accent_start=palette[accent_start_index],
        accent_end=palette[accent_end_index],
    )


def default_page_background(
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> WidgetBg:
    """返回背景图片不可用时使用的低对比浅色兜底背景。"""

    gradient = LinearGradient(
        c1=theme.page_start,
        c2=theme.page_end,
        p1=(0, 1),
        p2=(1, 0),
    )
    return FillBg(gradient)


def report_card_background(
    *,
    compact: bool = False,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> WidgetBg:
    """创建带轻微阴影的半透明玻璃卡片背景。"""

    return RoundRectBg(
        fill=theme.compact_card_fill if compact else theme.card_fill,
        radius=theme.compact_card_radius if compact else theme.card_radius,
        stroke=theme.card_stroke,
        stroke_width=1,
        blurglass=True,
        blurglass_kwargs={
            "blur": 6,
            "shadow_width": 10 if compact else 14,
            "shadow_alpha": 0.10 if compact else 0.17,
        },
    )


def subtle_panel_background(
    *,
    radius: int = 14,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> WidgetBg:
    """创建用于简介、备注或次要信息的主题浅色底板。"""

    return RoundRectBg(fill=theme.subtle_fill, radius=radius)


def accent_badge_background(
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> WidgetBg:
    """创建用于概率、状态等强调信息的主题渐变背景。"""

    gradient = LinearGradient(
        c1=theme.accent_start,
        c2=theme.accent_end,
        p1=(0, 0),
        p2=(1, 1),
    )
    return RoundRectBg(fill=gradient, radius=999)


def themed_text_style(
    role: str = "body",
    *,
    size: int | None = None,
    color: RGBA | None = None,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> TextStyle:
    """
    按语义角色创建文字样式。

    支持 ``title``、``heading``、``body``、``secondary``、``muted``、
    ``accent`` 和 ``badge``，调用方无需重复硬编码字体与颜色。
    """

    presets = {
        "title": (DEFAULT_BOLD_FONT, 30, theme.text_primary),
        "heading": (DEFAULT_BOLD_FONT, 18, theme.text_primary),
        "body": (DEFAULT_FONT, 14, theme.text_primary),
        "secondary": (DEFAULT_FONT, 14, theme.text_secondary),
        "muted": (DEFAULT_FONT, 13, theme.text_muted),
        "accent": (DEFAULT_BOLD_FONT, 14, theme.accent),
        "badge": (DEFAULT_BOLD_FONT, 15, (255, 255, 255, 255)),
    }
    if role not in presets:
        raise ValueError(f"不支持的文字样式角色: {role}")
    font, default_size, default_color = presets[role]
    return TextStyle(
        font=font,
        size=size if size is not None else default_size,
        color=color if color is not None else default_color,
    )


async def load_draw_image(source: str) -> Image.Image:
    """
    将通用图片地址加载为已解码的 RGBA 图像。

    支持 HTTP(S)、data URI、file URI 和普通本地路径。函数会在
    返回前完成像素解码，避免后续绘制时依赖已关闭的文件对象。
    """

    source = str(source or "").strip()
    if not source:
        raise ValueError("图片地址为空")

    if source.startswith(("http://", "https://")):
        image = await download_image(source)
    elif source.startswith("data:"):
        image = b64_to_image(source)
    else:
        if source.startswith("file://"):
            source = unquote(urlparse(source).path)
        image = open_image(Path(source).expanduser().resolve())

    image.load()
    return image.convert("RGBA")


async def resolve_draw_background(
    source: str | None,
    *,
    inherited_source: str | None = None,
    fallback: WidgetBg | None = None,
    logger: Any = None,
    label: str = "报告",
) -> WidgetBg:
    """
    解析模块背景配置并返回可直接交给 ``Canvas`` 的背景。

    ``source`` 非空时优先使用模块自身配置；留空时使用
    ``inherited_source``。加载失败只记录警告并返回兜底背景，
    不让非关键的样式配置阻断业务图片发送。
    """

    fallback = fallback or default_page_background()
    selected = str(source or "").strip() or str(inherited_source or "").strip()
    if not selected:
        return fallback
    try:
        image = await load_draw_image(selected)
        # ImageBg 的 fit 模式为等比覆盖，不会拉伸背景图。
        return ImageBg(image, mode="fit", fade=0)
    except Exception as exc:
        if logger is not None:
            logger.warning(f"{label}背景图片加载失败，已使用默认背景: {exc}")
        return fallback


def get_configured_draw_background_sources(
    module_config: Any = None,
) -> tuple[str, str]:
    """返回插件背景覆盖和 global.yaml 的共享默认背景。"""

    configured = ""
    if module_config is not None:
        configured = str(
            module_config.get("draw.background_image", "", raise_exc=False) or ""
        ).strip()
        # 兼容迁移前的顶层 background_image，便于旧部署平滑升级。
        if not configured:
            configured = str(
                module_config.get("background_image", "", raise_exc=False) or ""
            ).strip()
    inherited = str(
        global_config.get("draw.background_image", "", raise_exc=False) or ""
    ).strip()
    return configured, inherited


async def resolve_configured_draw_background(
    module_config: Any = None,
    *,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
    logger: Any = None,
    label: str = "报告",
) -> WidgetBg:
    """解析插件级背景；留空时继承 global.yaml 中的 Markdown 背景。"""

    configured, inherited = get_configured_draw_background_sources(module_config)
    return await resolve_draw_background(
        configured,
        inherited_source=inherited,
        fallback=default_page_background(theme),
        logger=logger,
        label=label,
    )


def create_report_canvas(
    *,
    background: WidgetBg | None = None,
    width: int | None = None,
    padding: int | None = None,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> Canvas:
    """创建顶部居中的通用报告画布。"""

    return (
        Canvas(w=width, bg=background or default_page_background(theme))
        .set_padding(theme.page_padding if padding is None else padding)
        .set_content_align("t")
    )


def create_report_column(
    width: int,
    *,
    gap: int | None = None,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> VSplit:
    """创建固定内容宽度的报告主列，所有子项默认左对齐。"""

    return (
        VSplit()
        .set_w(width)
        .set_sep(theme.section_gap if gap is None else gap)
        .set_content_and_item_align("l")
    )


def report_card(
    width: int,
    *,
    height: int | None = None,
    padding: int | tuple[int, int] = 22,
    compact: bool = False,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> Frame:
    """创建可作为上下文管理器使用的通用玻璃卡片。"""

    card = (
        Frame()
        .set_w(width)
        .set_padding(padding)
        .set_bg(report_card_background(compact=compact, theme=theme))
        .set_content_align("lt")
    )
    if height is not None:
        card.set_h(height)
    return card


def style_report_card(
    widget: Widget,
    *,
    width: int | None = None,
    height: int | None = None,
    padding: int | tuple[int, int] | None = None,
    compact: bool = False,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> Widget:
    """将已有业务控件原地转换为统一卡片，不改变其内容结构。"""

    widget.set_bg(report_card_background(compact=compact, theme=theme))
    if width is not None:
        widget.set_w(width)
    if height is not None:
        widget.set_h(height)
    if padding is not None:
        widget.set_padding(padding)
    return widget


def report_header(
    title: str,
    *,
    width: int,
    eyebrow: str = "",
    meta: str = "",
    meta_width: int = 280,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> HSplit:
    """
    创建“小标签 + 主标题 + 右侧元信息”页头。

    该结构来自原识鸟报告页头，元信息为空时会自动让主标题
    占用全部宽度，适合统计、查询和概览类图片复用。
    """

    right_width = min(meta_width, width // 2) if meta else 0
    left_width = width - right_width
    with HSplit().set_w(width).set_sep(0).set_item_align("b") as header:
        with (
            VSplit()
            .set_w(left_width)
            .set_sep(4)
            .set_content_and_item_align("l")
        ):
            if eyebrow:
                TextBox(
                    eyebrow,
                    themed_text_style("accent", size=14, theme=theme),
                ).set_w(left_width).set_padding(0)
            TextBox(
                title,
                themed_text_style("title", theme=theme),
                line_count=2,
            ).set_w(left_width).set_padding(0)
        if meta:
            TextBox(
                meta,
                themed_text_style("muted", theme=theme),
                line_count=2,
            ).set_w(right_width).set_padding(0).set_content_align("rb")
    return header


def report_section_title(
    text: str,
    *,
    width: int,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> TextBox:
    """创建报告内的二级分区标题。"""

    return (
        TextBox(text, themed_text_style("heading", theme=theme))
        .set_w(width)
        .set_padding((4, 0))
    )


def report_badge(
    text: str,
    *,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> TextBox:
    """创建紧凑的主题渐变强调标签。"""

    return (
        TextBox(text, themed_text_style("badge", theme=theme))
        .set_padding((11, 6))
        .set_bg(accent_badge_background(theme))
        .set_content_align("c")
    )


def report_info_row(
    label: str,
    value: str,
    *,
    width: int,
    label_width: int = 48,
    gap: int = 8,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> HSplit:
    """创建左侧弱化标签、右侧主信息的通用键值行。"""

    value_width = max(1, width - label_width - gap)
    with HSplit().set_w(width).set_sep(gap).set_item_align("t") as row:
        TextBox(
            label,
            themed_text_style("muted", size=13, theme=theme),
        ).set_w(label_width).set_padding(0)
        TextBox(
            value,
            themed_text_style("body", size=13, theme=theme),
            use_real_line_count=True,
        ).set_w(value_width).set_padding(0)
    return row


def report_description_panel(
    text: str,
    *,
    width: int,
    label: str = "简介",
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> Frame:
    """创建带小标题的浅蓝多行文字面板。"""

    inner_width = max(1, width - 30)
    with (
        Frame()
        .set_w(width)
        .set_padding((15, 13))
        .set_bg(subtle_panel_background(theme=theme))
        .set_content_align("lt")
    ) as panel:
        with (
            VSplit()
            .set_w(inner_width)
            .set_sep(5)
            .set_content_and_item_align("l")
        ):
            TextBox(
                label,
                themed_text_style("accent", size=13, theme=theme),
            ).set_w(inner_width).set_padding(0)
            TextBox(
                text,
                themed_text_style("secondary", size=14, theme=theme),
                line_sep=7,
                use_real_line_count=True,
            ).set_w(inner_width).set_padding(0)
    return panel


def report_text_section(
    title: str,
    text: str,
    *,
    width: int,
    padding: int | tuple[int, int] = (22, 18),
    compact: bool = False,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> Frame:
    """创建高度随正文自动变化的通用报告章节。

    该控件适用于百科、说明书和分析报告等长短不固定的段落；正文保留
    显式换行，并由 ``TextBox`` 按可用宽度继续自动折行。
    """

    if isinstance(padding, int):
        horizontal_padding = padding
    else:
        horizontal_padding = padding[0]
    inner_width = max(1, width - horizontal_padding * 2)
    with report_card(
        width,
        padding=padding,
        compact=compact,
        theme=theme,
    ) as section:
        with (
            VSplit()
            .set_w(inner_width)
            .set_sep(9)
            .set_content_and_item_align("l")
        ):
            TextBox(
                title,
                themed_text_style("heading", size=18, theme=theme),
                use_real_line_count=True,
            ).set_w(inner_width).set_padding(0)
            TextBox(
                text,
                themed_text_style("body", size=15, theme=theme),
                line_sep=8,
                use_real_line_count=True,
            ).set_w(inner_width).set_padding(0)
    return section


def rounded_cover_image(
    image: Image.Image,
    *,
    size: tuple[int, int],
    radius: int = 17,
) -> ImageBox:
    """
    创建保持原始宽高比的圆角覆盖图。

    图像会先等比缩放到完全覆盖目标区域，再从中心裁切，不会因
    固定宽高而发生拉伸。
    """

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    fitted = ImageOps.fit(image.convert("RGBA"), size, method=resampling)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1),
        radius=radius,
        fill=255,
    )
    fitted.putalpha(mask)
    return ImageBox(fitted, use_alphablend=True)


def report_image_placeholder(
    *,
    size: tuple[int, int],
    text: str = "暂无图片",
    radius: int = 17,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> Frame:
    """创建与封面图同尺寸的统一占位控件。"""

    with (
        Frame()
        .set_size(size)
        .set_bg(RoundRectBg(fill=(233, 240, 245, 255), radius=radius))
        .set_content_align("c")
    ) as placeholder:
        TextBox(
            text,
            themed_text_style("muted", size=15, theme=theme),
        ).set_padding(0)
    return placeholder


def report_image_panel(
    image: Image.Image,
    *,
    width: int,
    height: int | None = None,
    padding: int | tuple[int, int] = 20,
    compact: bool = False,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> ImageBox:
    """
    将业务图像放入统一卡片，始终以等比 ``fit`` 模式绘制。

    默认只限制宽度，高度根据原图比例自动计算；即使调用方指定
    了固定高度，图片本身也不会被拉伸。
    """

    panel = (
        ImageBox(image, image_size_mode="fit", use_alphablend=True)
        .set_w(width)
        .set_padding(padding)
        .set_bg(report_card_background(compact=compact, theme=theme))
        .set_content_align("c")
    )
    if height is not None:
        panel.set_h(height)
    return panel


def ranked_info_card(
    rank: int | str,
    title: str,
    *,
    trailing: str,
    subtitle: str,
    detail: str,
    width: int,
    height: int = 118,
    theme: UnifiedDrawTheme = DEFAULT_DRAW_THEME,
) -> Frame:
    """
    创建可复用的排名信息卡。

    控件对应原识鸟报告的“其他可能”卡片，也适用于排行榜、
    搜索候选和统计摘要。
    """

    padding = 15
    gap = 13
    rank_size = 28
    inner_width = width - padding * 2
    text_width = inner_width - rank_size - gap
    trailing_width = min(100, max(70, text_width // 3))
    title_width = text_width - trailing_width - 8

    with report_card(
        width,
        height=height,
        padding=padding,
        compact=True,
        theme=theme,
    ) as card:
        with HSplit().set_w(inner_width).set_sep(gap).set_item_align("t"):
            TextBox(
                str(rank),
                themed_text_style("accent", size=15, theme=theme),
            ).set_size((rank_size, rank_size)).set_padding(0).set_bg(
                RoundRectBg(fill=theme.rank_fill, radius=9)
            ).set_content_align("c")

            with (
                VSplit()
                .set_w(text_width)
                .set_sep(4)
                .set_content_and_item_align("l")
            ):
                with HSplit().set_w(text_width).set_sep(8).set_item_align("t"):
                    TextBox(
                        title,
                        themed_text_style("heading", size=17, theme=theme),
                    ).set_w(title_width).set_padding(0)
                    TextBox(
                        trailing,
                        themed_text_style(
                            "accent",
                            size=13,
                            color=theme.accent_dark,
                            theme=theme,
                        ),
                    ).set_w(trailing_width).set_padding(0).set_content_align("r")
                TextBox(
                    subtitle,
                    themed_text_style("secondary", size=13, theme=theme),
                ).set_w(text_width).set_padding(0)
                TextBox(
                    detail,
                    themed_text_style("muted", size=12, theme=theme),
                    line_count=2,
                ).set_w(text_width).set_padding(0)
    return card


__all__ = [
    "PROJECT_SEKAI_CHARACTER_COLOR_FAMILIES",
    "PROJECT_SEKAI_CHARACTER_COLORS",
    "PROJECT_SEKAI_CHART_COLORS",
    "DRAW_THEME_BASE_PALETTES",
    "DRAW_THEME_PRIMARY_COLORS",
    "DEFAULT_DRAW_THEME_NAME",
    "UNIFIED_CHART_COLORS",
    "UNIFIED_OTHER_CHART_COLOR",
    "normalize_draw_theme_name",
    "get_draw_theme_name",
    "get_draw_theme_base_palette",
    "build_member_draw_palette",
    "resolve_draw_member_palette",
    "build_smooth_draw_palette",
    "build_categorical_draw_palette",
    "build_coverage_draw_palette",
    "resolve_draw_palette",
    "resolve_draw_categorical_palette",
    "resolve_draw_coverage_palette",
    "UnifiedDrawTheme",
    "DEFAULT_DRAW_THEME",
    "resolve_draw_theme",
    "default_page_background",
    "report_card_background",
    "subtle_panel_background",
    "accent_badge_background",
    "themed_text_style",
    "load_draw_image",
    "resolve_draw_background",
    "get_configured_draw_background_sources",
    "resolve_configured_draw_background",
    "create_report_canvas",
    "create_report_column",
    "report_card",
    "style_report_card",
    "report_header",
    "report_section_title",
    "report_badge",
    "report_info_row",
    "report_description_panel",
    "report_text_section",
    "rounded_cover_image",
    "report_image_placeholder",
    "report_image_panel",
    "ranked_info_card",
]
