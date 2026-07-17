import asyncio
import colorsys
import io
import math
from collections import Counter as StdCounter
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Sequence

import jieba
import jieba.posseg as pseg
import wordcloud
from PIL import Image, ImageDraw

from ..utils import *
from ..record.sql import query_msg_by_range
from .tokenizer import (
    GenericWordBaseline,
    KeywordAnalysis,
    MessageSample,
    analyze_messages,
    apply_keyword_decisions,
    build_generic_baselines,
    reset_smart_tokenizer,
)
from .word_dictionary import get_llm_dictionary_words

config = Config("sta")
logger = get_logger("Sta")
file_db = get_file_db("data/sta/db.json", logger)

FONT_PATH = global_config.get("font.path")

STA_CANVAS_WIDTH = 1180
STA_LAYOUT_GAP = 17

STA_OTHER_COLOR = UNIFIED_OTHER_CHART_COLOR
STA_GRID_COLOR = (207, 219, 229, 175)
STA_ROW_COLOR = (241, 246, 250, 150)
STA_WORDCLOUD_MEMBER_HUE_RATIO = 0.24
STA_CHART_THEME_HUE_RATIO = 0.22


def _shift_color_hue_toward(
    source: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
    ratio: float,
    *,
    neutral_source_tint: float = 0.0,
) -> tuple[int, int, int, int]:
    """沿最短色相路径把颜色推向目标，保留原明度与主体饱和度。"""

    ratio = max(0.0, min(1.0, ratio))
    source_rgb = tuple(channel / 255 for channel in source[:3])
    target_rgb = tuple(channel / 255 for channel in target[:3])
    source_hue, source_lightness, source_saturation = colorsys.rgb_to_hls(
        *source_rgb
    )
    target_hue, _, target_saturation = colorsys.rgb_to_hls(*target_rgb)

    # 无彩色没有可用色相：目标无彩时不改变原色相；
    # 原色无彩时可按需注入少量目标主题色。
    if target_saturation < 0.08:
        target_hue = source_hue
    if source_saturation < 0.08:
        source_hue = target_hue
        source_saturation = target_saturation * neutral_source_tint

    hue_delta = ((target_hue - source_hue + 0.5) % 1.0) - 0.5
    adjusted_hue = (source_hue + hue_delta * ratio) % 1.0
    adjusted = colorsys.hls_to_rgb(
        adjusted_hue,
        source_lightness,
        source_saturation,
    )
    return (*tuple(round(channel * 255) for channel in adjusted), source[3])


def _wordcloud_theme_colors(
    group_id: int | str | None = None,
    *,
    accent: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    """以页面强调色为主，只向各成员色相偏移少量。"""

    base_colors = get_draw_theme_base_palette(config, group_id=group_id)
    if not base_colors:
        return (accent,)
    return tuple(
        _shift_color_hue_toward(
            accent,
            member_color,
            STA_WORDCLOUD_MEMBER_HUE_RATIO,
        )
        for member_color in base_colors
    )


@dataclass(frozen=True)
class StaDrawStyle:
    """一次 STA 渲染所需的完整主题快照。"""

    theme: UnifiedDrawTheme
    chart_colors: tuple[tuple[int, int, int, int], ...]
    member_colors: tuple[tuple[int, int, int, int], ...]
    position_colors: tuple[tuple[int, int, int, int], ...]
    wordcloud_colors: tuple[tuple[int, int, int, int], ...]


def _resolve_sta_style(group_id: int | str | None = None) -> StaDrawStyle:
    """按群号解析主题，避免多群并发时串色。"""

    theme = resolve_draw_theme(config, group_id=group_id)
    coverage_colors = resolve_draw_coverage_palette(
        config,
        group_id=group_id,
    )
    return StaDrawStyle(
        theme=theme,
        chart_colors=tuple(
            _shift_color_hue_toward(
                color,
                theme.accent,
                STA_CHART_THEME_HUE_RATIO,
                neutral_source_tint=0.16,
            )
            for color in coverage_colors
        ),
        member_colors=resolve_draw_member_palette(config, group_id=group_id),
        position_colors=get_draw_theme_base_palette(config, group_id=group_id),
        wordcloud_colors=_wordcloud_theme_colors(
            group_id,
            accent=theme.accent,
        ),
    )


_DEFAULT_STA_STYLE = _resolve_sta_style()
_STA_STYLE_CONTEXT: ContextVar[StaDrawStyle] = ContextVar(
    "sta_draw_style",
    default=_DEFAULT_STA_STYLE,
)


def _current_sta_style() -> StaDrawStyle:
    return _STA_STYLE_CONTEXT.get()


class _StaThemeProxy:
    """使既有绘图函数始终读取当前请求的主题。"""

    def __getattr__(self, name):
        return getattr(_current_sta_style().theme, name)


class _StaPaletteProxy:
    """为既有调色板访问提供并发安全的动态视图。"""

    def __init__(self, attribute: str):
        self.attribute = attribute

    def _palette(self):
        return getattr(_current_sta_style(), self.attribute)

    def __len__(self):
        return len(self._palette())

    def __getitem__(self, index):
        return self._palette()[index]

    def __iter__(self):
        return iter(self._palette())


STA_DRAW_THEME = _StaThemeProxy()
STA_CHART_COLORS = _StaPaletteProxy("chart_colors")
STA_MEMBER_COLORS = _StaPaletteProxy("member_colors")
# 双系列消息图按成员原始站位取色，不受高区分成员色重排影响。
STA_POSITION_COLORS = _StaPaletteProxy("position_colors")
STA_WORDCLOUD_MEMBER_COLORS = _StaPaletteProxy("wordcloud_colors")
STA_CONTENT_WIDTH = STA_CANVAS_WIDTH - _DEFAULT_STA_STYLE.theme.page_padding * 2


@dataclass
class StaPieEntry:
    """饼图及图例需要的用户统计。"""

    user_id: int | str | None
    name: str
    count: int
    avatar: Image.Image | None = None
    is_other: bool = False


@dataclass(frozen=True)
class StaBarSeries:
    """一组柱状图序列。"""

    name: str
    values: tuple[int, ...]
    color: tuple[int, int, int, int]


@dataclass
class StaWordContributor:
    """关键词贡献者及其可选 QQ 头像。"""

    user_id: int | str
    name: str
    rate: float
    avatar: Image.Image | None = None


@dataclass(frozen=True)
class StaWordRank:
    """词云下方图形化排行所需的关键词与贡献者信息。"""

    word: str
    count: int
    contributors: tuple[StaWordContributor, ...]


def _chart_color(index: int, *, is_other: bool = False):
    if is_other:
        return STA_OTHER_COLOR
    return STA_CHART_COLORS[index % len(STA_CHART_COLORS)]


def _member_color(index: int):
    return STA_MEMBER_COLORS[index % len(STA_MEMBER_COLORS)]


def _position_color(index: int):
    return STA_POSITION_COLORS[index % len(STA_POSITION_COLORS)]


def _draw_chart_text(
    painter: Painter,
    text: str,
    x: int,
    y: int,
    *,
    font: str = DEFAULT_FONT,
    size: int = 14,
    color: tuple[int, int, int, int] | None = None,
    align: str = "left",
) -> None:
    """绘制单行图表文字；Painter 会自动用 Pilmoji 渲染 Emoji。"""

    if color is None:
        color = STA_DRAW_THEME.text_primary
    if align != "left":
        text_width, _ = get_text_size(get_font(font, size), text)
        x -= text_width // 2 if align == "center" else text_width
    painter.text(text, (x, y), get_font_desc(font, size), fill=color)


def _truncate_chart_text(text: str, width: int, *, font: str, size: int) -> str:
    """按包含 Emoji 的真实像素宽度截断图表标签。"""

    pil_font = get_font(font, size)
    if get_text_size(pil_font, text)[0] <= width:
        return text
    suffix = "…"
    while text and get_text_size(pil_font, text + suffix)[0] > width:
        text = text[:-1]
    return text + suffix


def _nice_axis(max_value: int) -> tuple[int, list[int]]:
    """为整数计数生成接近五等分的 1/2/5 刻度轴。"""

    if max_value <= 0:
        return 1, [0, 1]
    raw_step = max_value / 5
    magnitude = 10 ** math.floor(math.log10(raw_step)) if raw_step else 1
    normalized = raw_step / magnitude
    factor = next((value for value in (1, 2, 5, 10) if normalized <= value), 10)
    step = max(1, int(factor * magnitude))
    axis_max = max(step, math.ceil(max_value / step) * step)
    return axis_max, list(range(0, axis_max + 1, step))


def _build_pie_entries(
    recs,
    topk_user: Sequence[int | str],
    topk_name: Sequence[str],
) -> list[StaPieEntry]:
    """保留原 STA 的 Top K 与低于 3% 合并“其他”规则。"""

    counts = StdCounter(rec["user_id"] for rec in recs)
    selected_users = list(topk_user)
    names = {
        user_id: str(topk_name[index]) if index < len(topk_name) else str(user_id)
        for index, user_id in enumerate(selected_users)
    }
    selected_set = set(selected_users)
    other_users = {rec["user_id"] for rec in recs if rec["user_id"] not in selected_set}
    other_count = sum(counts[user_id] for user_id in other_users)
    total_count = max(1, len(recs))

    while selected_users and counts[selected_users[-1]] / total_count < 0.03:
        user_id = selected_users.pop()
        other_users.add(user_id)
        other_count += counts[user_id]

    entries = [
        StaPieEntry(user_id=user_id, name=names[user_id], count=counts[user_id])
        for user_id in selected_users
    ]
    if other_count:
        entries.append(
            StaPieEntry(
                user_id=None,
                name=f"其他（{len(other_users)}人）",
                count=other_count,
                is_other=True,
            )
        )
    return entries


async def _download_pie_avatars(gid, entries: Sequence[StaPieEntry]) -> None:
    """并发获取图例头像；单个头像失败不会阻断统计图。"""

    bot = await aget_group_bot(gid, raise_exc=False)

    async def download(entry: StaPieEntry):
        if entry.user_id is None:
            return None
        try:
            return await download_avatar(bot, entry.user_id, circle=True)
        except Exception:
            logger.print_exc(f"获取{entry.user_id}头像失败")
            return None

    avatars = await asyncio.gather(*(download(entry) for entry in entries))
    for entry, avatar in zip(entries, avatars):
        entry.avatar = avatar


def _spread_callout_positions(
    items: Sequence[tuple[int, float]],
    *,
    min_y: float,
    max_y: float,
    gap: float,
) -> dict[int, float]:
    """在保持扇区纵向顺序的同时，避免直连标签互相遮挡。"""

    if not items:
        return {}
    ordered = sorted(items, key=lambda item: item[1])
    if len(ordered) > 1:
        gap = min(gap, (max_y - min_y) / (len(ordered) - 1))
    positions = [max(min_y, min(max_y, natural_y)) for _, natural_y in ordered]
    for index in range(1, len(positions)):
        positions[index] = max(positions[index], positions[index - 1] + gap)
    if positions[-1] > max_y:
        positions = [value - (positions[-1] - max_y) for value in positions]
    for index in range(len(positions) - 2, -1, -1):
        positions[index] = min(positions[index], positions[index + 1] - gap)
    if positions[0] < min_y:
        positions = [value + (min_y - positions[0]) for value in positions]
    return {
        item_index: positions[index]
        for index, (item_index, _) in enumerate(ordered)
    }


def _connected_donut_widget(
    entries: Sequence[StaPieEntry],
    *,
    width: int,
    height: int = 500,
) -> Spacer:
    """绘制头像和昵称直接连向扇区的环形图。"""

    frozen_entries = tuple(entries)
    total = sum(entry.count for entry in frozen_entries)

    def draw(_, painter: Painter):
        donut_size = 290
        radius = donut_size / 2
        center_x, center_y = width / 2, height / 2
        donut_x, donut_y = center_x - radius, center_y - radius
        current_angle = -90.0
        callouts: list[tuple[int, float, float, float, float]] = []

        if total <= 0:
            painter.pieslice(
                (int(donut_x), int(donut_y)),
                (donut_size, donut_size),
                0,
                360,
                STA_OTHER_COLOR,
            )
        else:
            for index, entry in enumerate(frozen_entries):
                angle = 360 * entry.count / total
                color = _chart_color(index, is_other=entry.is_other)
                painter.pieslice(
                    (int(donut_x), int(donut_y)),
                    (donut_size, donut_size),
                    current_angle,
                    current_angle + angle,
                    color,
                    stroke=(255, 255, 255, 255),
                    stroke_width=2,
                )
                mid_angle = math.radians(current_angle + angle / 2)
                anchor_x = center_x + (radius - 2) * math.cos(mid_angle)
                anchor_y = center_y + (radius - 2) * math.sin(mid_angle)
                elbow_x = center_x + (radius + 16) * math.cos(mid_angle)
                elbow_y = center_y + (radius + 16) * math.sin(mid_angle)
                callouts.append((index, anchor_x, anchor_y, elbow_x, elbow_y))
                current_angle += angle

        inner_size = int(donut_size * 0.55)
        inner_offset = (donut_size - inner_size) // 2
        painter.pieslice(
            (int(donut_x + inner_offset), int(donut_y + inner_offset)),
            (inner_size, inner_size),
            0,
            360,
            (255, 255, 255, 255),
        )
        _draw_chart_text(
            painter,
            f"{total:,}",
            int(center_x),
            int(center_y - 27),
            font=DEFAULT_BOLD_FONT,
            size=31,
            align="center",
        )
        _draw_chart_text(
            painter,
            "条消息",
            int(center_x),
            int(center_y + 13),
            size=15,
            color=STA_DRAW_THEME.text_muted,
            align="center",
        )

        left_items = [
            (index, elbow_y)
            for index, _, _, elbow_x, elbow_y in callouts
            if elbow_x < center_x
        ]
        right_items = [
            (index, elbow_y)
            for index, _, _, elbow_x, elbow_y in callouts
            if elbow_x >= center_x
        ]
        positions = {
            **_spread_callout_positions(
                left_items, min_y=30, max_y=height - 30, gap=43
            ),
            **_spread_callout_positions(
                right_items, min_y=30, max_y=height - 30, gap=43
            ),
        }

        # Pillow 线层可绘制斜向折线；文字仍交给 Painter/Pilmoji 以保留 Emoji。
        leader_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        leader_draw = ImageDraw.Draw(leader_layer)
        for index, anchor_x, anchor_y, elbow_x, elbow_y in callouts:
            entry = frozen_entries[index]
            color = _chart_color(index, is_other=entry.is_other)
            is_left = elbow_x < center_x
            target_x = 134 if is_left else width - 134
            target_y = positions[index]
            end_x = target_x + 21 if is_left else target_x - 21
            leader_draw.line(
                (
                    (round(anchor_x), round(anchor_y)),
                    (round(elbow_x), round(elbow_y)),
                    (round(end_x), round(target_y)),
                ),
                fill=color,
                width=2,
                joint="curve",
            )
            leader_draw.ellipse(
                (anchor_x - 3, anchor_y - 3, anchor_x + 3, anchor_y + 3),
                fill=color,
            )
        painter.paste_with_alphablend(leader_layer, (0, 0))

        for index, _, _, elbow_x, _ in callouts:
            entry = frozen_entries[index]
            color = _chart_color(index, is_other=entry.is_other)
            is_left = elbow_x < center_x
            target_x = 134 if is_left else width - 134
            target_y = positions[index]
            painter.pieslice(
                (round(target_x - 21), round(target_y - 21)),
                (42, 42),
                0,
                360,
                color,
            )
            if entry.avatar is not None:
                painter.paste_with_alphablend(
                    entry.avatar,
                    (round(target_x - 18), round(target_y - 18)),
                    (36, 36),
                )
            else:
                painter.pieslice(
                    (round(target_x - 18), round(target_y - 18)),
                    (36, 36),
                    0,
                    360,
                    (*color[:3], 120),
                )

            align = "right" if is_left else "left"
            label_x = target_x - 29 if is_left else target_x + 29
            name = _truncate_chart_text(
                entry.name,
                105,
                font=DEFAULT_BOLD_FONT,
                size=14,
            )
            _draw_chart_text(
                painter,
                name,
                round(label_x),
                round(target_y - 18),
                font=DEFAULT_BOLD_FONT,
                size=14,
                align=align,
            )
            _draw_chart_text(
                painter,
                f"{entry.count:,} · {entry.count / max(1, total) * 100:.1f}%",
                round(label_x),
                round(target_y + 4),
                size=11,
                color=STA_DRAW_THEME.text_muted,
                align=align,
            )

    return Spacer(width, height).add_draw_func(draw)


def _speaker_bar_widget(
    entries: Sequence[StaPieEntry],
    *,
    width: int,
    height: int = 500,
) -> Spacer:
    """在饼图侧边以横向柱形同步呈现相同发言分布。"""

    frozen_entries = tuple(entries)
    total = sum(entry.count for entry in frozen_entries)
    max_count = max((entry.count for entry in frozen_entries), default=1)

    def draw(_, painter: Painter):
        _draw_chart_text(
            painter,
            "发言排行",
            0,
            2,
            font=DEFAULT_BOLD_FONT,
            size=18,
        )
        _draw_chart_text(
            painter,
            "人数占比与消息量",
            width,
            5,
            size=12,
            color=STA_DRAW_THEME.text_muted,
            align="right",
        )
        if total <= 0:
            _draw_chart_text(
                painter,
                "暂无匹配消息",
                width // 2,
                height // 2,
                size=15,
                color=STA_DRAW_THEME.text_muted,
                align="center",
            )
            return

        top = 50
        row_height = min(46, max(34, (height - top) // len(frozen_entries)))
        for index, entry in enumerate(frozen_entries):
            y = top + index * row_height
            color = _chart_color(index, is_other=entry.is_other)
            name = _truncate_chart_text(
                entry.name,
                max(90, width - 142),
                font=DEFAULT_BOLD_FONT,
                size=13,
            )
            _draw_chart_text(
                painter,
                name,
                0,
                y,
                font=DEFAULT_BOLD_FONT,
                size=13,
            )
            _draw_chart_text(
                painter,
                f"{entry.count:,} · {entry.count / total * 100:.1f}%",
                width,
                y,
                size=12,
                color=STA_DRAW_THEME.text_secondary,
                align="right",
            )
            bar_y = y + 23
            painter.roundrect(
                (0, bar_y),
                (width, 9),
                (226, 234, 240, 210),
                radius=999,
            )
            bar_width = max(5, round(width * entry.count / max_count))
            painter.roundrect(
                (0, bar_y),
                (bar_width, 9),
                color,
                radius=999,
            )

    return Spacer(width, height).add_draw_func(draw)


async def get_pie_frame(
    gid,
    date_str,
    recs,
    topk_user: list[int],
    topk_name: list[str],
    *,
    width: int,
) -> HSplit:
    """构造扇区直连人物标注与侧边横向排行。"""

    del date_str  # 固定 UI 色板不再按日期变色。
    logger.info("开始绘制饼图")
    entries = _build_pie_entries(recs, topk_user, topk_name)
    await _download_pie_avatars(gid, entries)

    return _pie_entries_frame(entries, width)


def _pie_entries_frame(entries: Sequence[StaPieEntry], width: int) -> HSplit:
    """把饼图直连标注和同数据横向柱形排成一个统一面板。"""

    display_entries = tuple(entries) or (
        StaPieEntry(
            user_id=None,
            name="暂无匹配消息",
            count=0,
            is_other=True,
        ),
    )
    gap = 24
    donut_width = min(650, round(width * 0.61))
    bar_width = width - donut_width - gap
    with HSplit().set_w(width).set_sep(gap).set_item_align("c") as frame:
        _connected_donut_widget(display_entries, width=donut_width)
        _speaker_bar_widget(display_entries, width=bar_width)
    return frame


def _bar_chart_widget(
    labels: Sequence[str],
    series: Sequence[StaBarSeries],
    *,
    width: int,
    height: int = 360,
    stacked: bool = False,
    overlay: bool = False,
    max_x_labels: int = 10,
) -> Spacer:
    """使用 Painter 绘制支持叠加或堆叠的统一柱状图。"""

    frozen_labels = tuple(labels)
    frozen_series = tuple(series)

    def draw(_, painter: Painter):
        left, top, right, bottom_margin = 62, 54, 20, 48
        plot_width = width - left - right
        plot_height = height - top - bottom_margin
        plot_bottom = top + plot_height
        count = len(frozen_labels)

        if stacked:
            maxima = [
                sum(item.values[index] for item in frozen_series)
                for index in range(count)
            ] if count else [0]
        else:
            maxima = [max(item.values, default=0) for item in frozen_series] or [0]
        axis_max, ticks = _nice_axis(max(maxima, default=0))

        for tick in ticks:
            y = plot_bottom - round(plot_height * tick / axis_max)
            painter.rect((left, y), (plot_width, 1), STA_GRID_COLOR)
            _draw_chart_text(
                painter,
                str(tick),
                left - 10,
                y - 7,
                size=12,
                color=STA_DRAW_THEME.text_muted,
                align="right",
            )

        legend_x = left
        for item in frozen_series:
            painter.roundrect((legend_x, 8), (12, 12), item.color, radius=4)
            legend_name = _truncate_chart_text(
                item.name,
                128,
                font=DEFAULT_FONT,
                size=13,
            )
            _draw_chart_text(painter, legend_name, legend_x + 19, 4, size=13)
            legend_x += 19 + get_text_size(get_font(DEFAULT_FONT, 13), legend_name)[0] + 24

        if count == 0:
            _draw_chart_text(
                painter,
                "暂无统计数据",
                left + plot_width // 2,
                top + plot_height // 2,
                font=DEFAULT_BOLD_FONT,
                size=20,
                color=STA_DRAW_THEME.text_muted,
                align="center",
            )
            return

        slot_width = plot_width / count
        bar_width = max(2, int(slot_width * 0.68))
        for index in range(count):
            center_x = left + (index + 0.5) * slot_width
            x = int(center_x - bar_width / 2)
            if stacked:
                current_bottom = plot_bottom
                for item in frozen_series:
                    value = item.values[index]
                    segment_height = round(plot_height * value / axis_max)
                    if segment_height <= 0:
                        continue
                    current_bottom -= segment_height
                    painter.rect(
                        (x, current_bottom),
                        (bar_width, segment_height),
                        item.color,
                    )
            else:
                for series_index, item in enumerate(frozen_series):
                    value = item.values[index]
                    bar_height = round(plot_height * value / axis_max)
                    if bar_height <= 0:
                        continue
                    current_width = (
                        max(2, int(bar_width * 0.54))
                        if overlay and series_index > 0 else bar_width
                    )
                    current_x = int(center_x - current_width / 2)
                    painter.roundrect(
                        (current_x, plot_bottom - bar_height),
                        (current_width, bar_height),
                        item.color,
                        radius=min(5, current_width // 2, max(1, bar_height // 2)),
                    )

        label_step = max(1, math.ceil(count / max_x_labels))
        shown_indices = set(range(0, count, label_step)) | {count - 1}
        for index in sorted(shown_indices):
            center_x = int(left + (index + 0.5) * slot_width)
            label = _truncate_chart_text(
                frozen_labels[index],
                max(46, int(slot_width * label_step) - 4),
                font=DEFAULT_FONT,
                size=12,
            )
            _draw_chart_text(
                painter,
                label,
                center_x,
                plot_bottom + 13,
                size=12,
                color=STA_DRAW_THEME.text_muted,
                align="center",
            )

    return Spacer(width, height).add_draw_func(draw)


def _message_interval_series(recs, interval: int):
    """按一天内的固定分钟区间统计总消息与图片消息。"""

    interval = max(1, int(interval))
    bucket_count = math.ceil(24 * 60 / interval)
    totals = [0] * bucket_count
    images = [0] * bucket_count
    for rec in recs:
        minute = rec["time"].hour * 60 + rec["time"].minute
        index = min(bucket_count - 1, minute // interval)
        totals[index] += 1
        if has_image(rec["msg"]):
            images[index] += 1
    labels = [
        f"{(index * interval // 60) % 24:02d}:{index * interval % 60:02d}"
        for index in range(bucket_count)
    ]
    return labels, totals, images


def _daily_message_series(recs):
    """构造按自然日升序排列的消息数量序列。"""

    if not recs:
        return [], []
    counts = StdCounter(rec["time"].date() for rec in recs)
    start_date = min(counts)
    end_date = max(counts)
    day_count = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=index) for index in range(day_count)]
    return [date.strftime("%m-%d") for date in dates], [counts[date] for date in dates]

last_userwords = []
last_llm_dictionary_words = []
jieba_inited = False

# jieba重置（用户词典）
def reset_jieba():
    global last_userwords, last_llm_dictionary_words, jieba_inited
    jieba.initialize()
    # 清空上次添加的用户词
    for word in last_userwords: jieba.del_word(word)
    # 读取用户词和停用词并且规范处理
    userwords = file_db.get("userwords", [])
    stopwords = file_db.get("stopwords", [])
    userwords = [word.strip() for word in userwords if word not in stopwords and word.strip() != ""]
    stopwords = [word.strip() for word in stopwords if word.strip() != ""]
    userwords = list(set(userwords))
    stopwords = list(set(stopwords))
    llm_dictionary_words = [
        word
        for word in get_llm_dictionary_words(stopwords=stopwords)
        if word not in userwords
    ]
    loaded_words = list(set(userwords + llm_dictionary_words))
    userwords_str = "\n".join([f'{word} n' for word in loaded_words])
    file_db.set("stopwords", stopwords)
    file_db.set("userwords", userwords)
    # 用户词设置给jieba
    userwords_file = io.StringIO(userwords_str)
    jieba.load_userdict(userwords_file)
    last_userwords = loaded_words
    last_llm_dictionary_words = llm_dictionary_words
    jieba_inited = True
    reset_smart_tokenizer()
    logger.info(
        f'jieba已重置 用户词数:{len(userwords)} '
        f'LLM辅助词数:{len(llm_dictionary_words)} 停用词数:{len(stopwords)}'
    )

# jieba初始化
def init_jieba():
    global jieba_inited, last_llm_dictionary_words
    stopwords = file_db.get("stopwords", [])
    llm_dictionary_words = get_llm_dictionary_words(stopwords=stopwords)
    if not jieba_inited or llm_dictionary_words != last_llm_dictionary_words:
        reset_jieba()


def _get_word_tokenizer_mode() -> str:
    return str(
        config.get("word_tokenizer.mode", "legacy", raise_exc=False) or "legacy"
    ).strip().lower()


def _build_message_samples(recs) -> list[MessageSample]:
    return [
        MessageSample(
            text=extract_text(rec['msg']),
            user_id=rec['user_id'],
        )
        for rec in recs
    ]


def _get_generic_burst_options() -> dict:
    return {
        "generic_burst_min_history_days": int(
            config.get(
                "word_tokenizer.generic_burst.min_history_days",
                5,
                raise_exc=False,
            )
        ),
        "generic_burst_min_documents": int(
            config.get(
                "word_tokenizer.generic_burst.min_documents",
                5,
                raise_exc=False,
            )
        ),
        "generic_burst_ratio_threshold": float(
            config.get(
                "word_tokenizer.generic_burst.ratio_threshold",
                2.5,
                raise_exc=False,
            )
        ),
        "generic_burst_min_rate_increase": float(
            config.get(
                "word_tokenizer.generic_burst.min_rate_increase",
                0.02,
                raise_exc=False,
            )
        ),
        "generic_burst_mad_multiplier": float(
            config.get(
                "word_tokenizer.generic_burst.mad_multiplier",
                3.0,
                raise_exc=False,
            )
        ),
    }


async def _load_generic_baselines(
    gid,
    date_str: str,
    userwords,
    stopwords,
    dictionary_words,
) -> dict[str, GenericWordBaseline]:
    """读取目标日期之前的消息，构造群级泛用词历史基线。"""

    if not bool(
        config.get(
            "word_tokenizer.generic_burst.enabled",
            True,
            raise_exc=False,
        )
    ):
        return {}

    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
    except (TypeError, ValueError):
        # /sta_sum 的日期是范围字符串，不做“单日异常升温”判断。
        return {}

    lookback_days = max(
        1,
        int(
            config.get(
                "word_tokenizer.generic_burst.lookback_days",
                14,
                raise_exc=False,
            )
        ),
    )
    start_time = target_date - timedelta(days=lookback_days)
    end_time = target_date - timedelta(microseconds=1)
    try:
        history_recs = await query_msg_by_range(gid, start_time, end_time)
    except Exception as exc:
        logger.warning(
            f"STA泛用词历史基线读取失败，使用保守过滤: "
            f"{type(exc).__name__}: {exc}"
        )
        return {}

    samples_by_date: dict[object, list[MessageSample]] = {}
    for rec in history_recs:
        samples_by_date.setdefault(rec["time"].date(), []).append(
            MessageSample(
                text=extract_text(rec["msg"]),
                user_id=rec["user_id"],
            )
        )
    baselines = build_generic_baselines(
        samples_by_date.values(),
        userwords=userwords,
        stopwords=stopwords,
        dictionary_words=dictionary_words,
    )
    logger.info(
        f"STA泛用词历史基线完成 group={gid} date={date_str} "
        f"历史消息={len(history_recs)} 有消息天数={len(samples_by_date)}"
    )
    return baselines


def _build_smart_word_analysis(
    recs,
    userwords,
    stopwords,
    statistical_filter: bool,
    generic_baselines: dict[str, GenericWordBaseline] | None = None,
    generic_burst_options: dict | None = None,
    dictionary_words=(),
) -> tuple[list[MessageSample], KeywordAnalysis]:
    samples = _build_message_samples(recs)
    analysis = analyze_messages(
        samples,
        userwords=userwords,
        stopwords=stopwords,
        dictionary_words=dictionary_words,
        statistical_filter=statistical_filter,
        generic_baselines=generic_baselines,
        **(generic_burst_options or {}),
    )
    return samples, analysis


async def prepare_word_analysis(gid, date_str, recs) -> KeywordAnalysis | None:
    """在异步绘图入口准备智能词频，并按配置执行可选 LLM 复核。"""

    if _get_word_tokenizer_mode() not in {"smart", "new"}:
        return None

    init_jieba()
    userwords = file_db.get("userwords", [])
    stopwords = file_db.get("stopwords", [])
    dictionary_words = get_llm_dictionary_words(stopwords=stopwords)
    statistical_filter = bool(
        config.get("word_tokenizer.statistical_filter", True, raise_exc=False)
    )
    generic_baselines = {}
    generic_burst_options = {}
    if statistical_filter:
        generic_baselines = await _load_generic_baselines(
            gid,
            date_str,
            userwords,
            stopwords,
            dictionary_words,
        )
        generic_burst_options = _get_generic_burst_options()
    samples, analysis = _build_smart_word_analysis(
        recs,
        userwords,
        stopwords,
        statistical_filter,
        generic_baselines,
        generic_burst_options,
        dictionary_words,
    )
    min_keywords = config.get(
        "word_tokenizer.min_keywords", 30, raise_exc=False
    )

    if not bool(config.get("word_tokenizer.llm.enabled", False, raise_exc=False)):
        return apply_keyword_decisions(
            analysis,
            [],
            protected_words=userwords,
            min_keywords=min_keywords,
        )

    model_name = config.get(
        "word_tokenizer.llm.model",
        "",
        raise_exc=False,
    )
    if not model_name:
        logger.warning("STA LLM关键词复核已启用但未配置模型，使用本地统计结果")
        return apply_keyword_decisions(
            analysis,
            [],
            protected_words=userwords,
            min_keywords=min_keywords,
        )

    # 延迟导入，legacy 模式和关闭 LLM 时不加载模型供应方。
    from .llm_filter import refine_keywords_with_llm

    return await refine_keywords_with_llm(
        group_id=gid,
        date_str=date_str,
        messages=samples,
        analysis=analysis,
        protected_words=userwords,
        stopwords=stopwords,
        model_name=model_name,
        candidate_limit=config.get(
            "word_tokenizer.llm.candidate_limit", 80, raise_exc=False
        ),
        contexts_per_word=config.get(
            "word_tokenizer.llm.contexts_per_word", 1, raise_exc=False
        ),
        min_keywords=min_keywords,
        supplemental_ratio=config.get(
            "word_tokenizer.llm.supplemental_ratio", 0.35, raise_exc=False
        ),
        timeout=config.get("word_tokenizer.llm.timeout", 30, raise_exc=False),
        max_tokens=config.get(
            "word_tokenizer.llm.max_tokens", 1600, raise_exc=False
        ),
    )


# 绘制词云图，并返回图形化关键词排行需要的数据。
def draw_wordcloud(
    gid,
    date_str,
    recs,
    users,
    names,
    word_analysis: KeywordAnalysis | None = None,
) -> tuple[Image.Image, list[StaWordRank]]:
    logger.info(f"开始绘制词云图")
    init_jieba()

    userwords = file_db.get("userwords", [])
    stopwords = file_db.get("stopwords", [])
    dictionary_words = get_llm_dictionary_words(stopwords=stopwords)
    tokenizer_mode = _get_word_tokenizer_mode()
    statistical_filter = bool(
        config.get("word_tokenizer.statistical_filter", True, raise_exc=False)
    )

    if tokenizer_mode in {"smart", "new"}:
        if word_analysis is None:
            _, word_analysis = _build_smart_word_analysis(
                recs,
                userwords,
                stopwords,
                statistical_filter,
                dictionary_words=dictionary_words,
            )
        analysis = word_analysis
        all_words = analysis.frequencies
        raw_word_counts = analysis.raw_counts
        word_user_count = analysis.user_counts
        logger.info(
            f"智能分词完成 统计过滤:{statistical_filter} 候选词数:{len(all_words)}"
        )
    else:
        if tokenizer_mode not in {"legacy", "old"}:
            logger.warning(f"未知分词模式 {tokenizer_mode}，回退到 legacy")
        legacy_userwords = set(userwords)
        legacy_stopwords = set(stopwords)
        all_words = {}
        raw_word_counts = {}
        word_user_count = {} # word_user_count[word][user] = count

        for rec in recs:
            msg = extract_text(rec['msg'])
            words = pseg.cut(msg)
            nouns = []
            for word, flag in words:
                if word in legacy_userwords:
                    nouns.append(word)
                elif flag.startswith('n') and word not in legacy_stopwords and len(word) > 1:
                    nouns.append(word)
            for noun in nouns:
                if noun not in all_words:
                    all_words[noun] = 0
                    raw_word_counts[noun] = 0
                    word_user_count[noun] = {}
                all_words[noun] += 1
                raw_word_counts[noun] += 1
                user = rec['user_id']
                if user not in word_user_count[noun]:
                    word_user_count[noun][user] = 0
                word_user_count[noun][user] += 1

    WORD_TOPK = 3
    # 小饼图直接使用成员原色，默认 25 时正好对应四位成员。
    WORD_USER_TOPK = min(5, len(STA_MEMBER_COLORS))

    # 统计前WORD_TOPK个词的前WORD_USER_TOPK个用户以及他们的比例(结果为topk_word_user=[[(user, rate), ...], ...])
    topk_words = [
        word
        for word, _ in sorted(
            all_words.items(), key=lambda item: item[1], reverse=True
        )[:WORD_TOPK]
    ]
    topk_word_user = {}
    for word in topk_words:
        contributors = sorted(
            word_user_count[word].items(),
            key=lambda item: item[1],
            reverse=True,
        )[:WORD_USER_TOPK]
        topk_word_user[word] = [
            (user, count / raw_word_counts[word])
            for user, count in contributors
        ]

    FONT_SIZE_MAX = 64
    FONT_SIZE_MIN = 14
    WC_W = 1000
    WC_H = 320

    positive_frequencies = [
        float(value)
        for value in all_words.values()
        if float(value) > 0
    ]
    min_frequency = min(positive_frequencies, default=1.0)
    max_frequency = max(positive_frequencies, default=1.0)
    log_min = math.log1p(min_frequency)
    log_range = max(1e-9, math.log1p(max_frequency) - log_min)
    ranked_color_words = sorted(
        all_words,
        key=lambda word: (-float(all_words[word]), word),
    )
    word_color_indices = {
        word: index % len(STA_WORDCLOUD_MEMBER_COLORS)
        for index, word in enumerate(ranked_color_words)
    }
    logger.info(
        f"词云组合主题: {get_draw_theme_name(config, group_id=gid)}"
    )

    # 高频词更深、更饱和，低频词更浅、更柔和。对数归一化可避免
    # 极高频词压扁其余词的视觉层次。
    def frequency_word_color(word, **kwargs):
        del kwargs
        frequency = max(0.0, float(all_words.get(word, min_frequency)))
        frequency_level = (
            (math.log1p(frequency) - log_min) / log_range
            if max_frequency > min_frequency
            else 1.0
        )
        # 按词频排名循环分配以页面强调色为中心的成员偏色；
        # 整体仍保持单一主色，只通过小幅色相差表达成员色影响。
        color_index = word_color_indices.get(
            word,
            sum(ord(character) for character in word),
        )
        base_color = STA_WORDCLOUD_MEMBER_COLORS[
            color_index % len(STA_WORDCLOUD_MEMBER_COLORS)
        ]
        red, green, blue = (channel / 255 for channel in base_color[:3])
        hue, _, source_saturation = colorsys.rgb_to_hls(red, green, blue)
        lightness = 0.74 - 0.24 * frequency_level
        # 所有团体统一使用 25 时的强层次曲线：高频高饱和，
        # 低频接近无彩色且更透明。
        low_saturation = 0.0 if source_saturation < 0.08 else 0.015
        high_saturation = 0.0 if source_saturation < 0.08 else 0.86
        saturation = low_saturation + (high_saturation - low_saturation) * (
            frequency_level ** 1.35
        )
        adjusted = colorsys.hls_to_rgb(hue, lightness, saturation)
        color = tuple(round(channel * 255) for channel in adjusted)
        alpha = round(70 + 185 * (frequency_level ** 1.15))
        return (*color, alpha)

    wc = wordcloud.WordCloud(
        font_path=FONT_PATH,
        background_color=None,
        width=WC_W,
        height=WC_H,
        max_words=100,
        max_font_size=FONT_SIZE_MAX,
        min_font_size=FONT_SIZE_MIN,
        # 保留明显字号差，同时避免绝对词频比过早把后续
        # 成员色代表词压到最小字号以下而直接丢弃。
        relative_scaling=0.65,
        random_state=42,
        color_func=frequency_word_color,
        mode='RGBA',
    )

    # WordCloud 不接受空词频；占位词只参与绘图，不进入排行榜和用户统计。
    cloud_frequencies = all_words or {"暂无可统计词汇": 1.0}
    wc.generate_from_frequencies(cloud_frequencies)
    img = wc.to_image()

    name_by_user = {
        user: str(names[index])
        for index, user in enumerate(users)
        if index < len(names)
    }
    ranks = [
        StaWordRank(
            word=word,
            count=int(raw_word_counts[word]),
            contributors=tuple(
                StaWordContributor(
                    user_id=user,
                    name=name_by_user.get(user, str(user)),
                    rate=rate,
                )
                for user, rate in topk_word_user[word]
            ),
        )
        for word in topk_words
    ]
    return img, ranks


async def _download_word_rank_avatars(
    gid,
    ranks: Sequence[StaWordRank],
) -> None:
    """为关键词贡献饼图并发下载唯一用户头像。"""

    contributors: dict[int | str, list[StaWordContributor]] = {}
    for rank in ranks:
        for contributor in rank.contributors:
            contributors.setdefault(contributor.user_id, []).append(contributor)
    if not contributors:
        return

    bot = await aget_group_bot(gid, raise_exc=False)

    async def download(user_id):
        try:
            return await download_avatar(bot, user_id, circle=True)
        except Exception:
            logger.print_exc(f"获取{user_id}关键词统计头像失败")
            return None

    user_ids = tuple(contributors)
    avatars = await asyncio.gather(*(download(user_id) for user_id in user_ids))
    for user_id, avatar in zip(user_ids, avatars):
        for contributor in contributors[user_id]:
            contributor.avatar = avatar


def _word_rank_widget(
    ranks: Sequence[StaWordRank],
    *,
    width: int,
) -> Spacer:
    """以带头像直连的小饼同时表达关键词热度和用户贡献。"""

    frozen_ranks = tuple(ranks)
    height = 166
    max_count = max((rank.count for rank in frozen_ranks), default=1)

    def draw(_, painter: Painter):
        if not frozen_ranks:
            _draw_chart_text(
                painter,
                "暂无关键词排行",
                width // 2,
                height // 2 - 8,
                size=15,
                color=STA_DRAW_THEME.text_muted,
                align="center",
            )
            return

        column_width = width / len(frozen_ranks)
        max_diameter = min(70, round(column_width - 210))
        leader_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        leader_draw = ImageDraw.Draw(leader_layer)
        labels = []

        for index, rank in enumerate(frozen_ranks):
            center_x = round(column_width * (index + 0.5))
            # 让圆面积与词频成正比，因此直径按词频平方根缩放。
            diameter = max(
                38,
                round(max_diameter * math.sqrt(rank.count / max_count)),
            )
            pie_x = round(center_x - diameter / 2)
            pie_y = round(61 - diameter / 2)
            current_angle = -90.0
            remaining_rate = 1.0
            callouts = []

            if rank.contributors:
                for contributor_index, contributor in enumerate(rank.contributors):
                    visible_rate = min(
                        remaining_rate,
                        max(0.0, contributor.rate),
                    )
                    if visible_rate <= 0:
                        continue
                    angle = 360 * visible_rate
                    # 每个关键词小饼图都从当前主题的第一个成员色开始。
                    color = _member_color(contributor_index)
                    painter.pieslice(
                        (pie_x, pie_y),
                        (diameter, diameter),
                        current_angle,
                        current_angle + angle,
                        color,
                        stroke=(255, 255, 255, 255),
                        stroke_width=2,
                    )
                    mid_angle = math.radians(current_angle + angle / 2)
                    radius = diameter / 2
                    anchor_x = center_x + radius * math.cos(mid_angle)
                    anchor_y = 61 + radius * math.sin(mid_angle)
                    elbow_x = center_x + (radius + 9) * math.cos(mid_angle)
                    elbow_y = 61 + (radius + 9) * math.sin(mid_angle)
                    callouts.append((
                        contributor,
                        color,
                        anchor_x,
                        anchor_y,
                        elbow_x,
                        elbow_y,
                    ))
                    current_angle += angle
                    remaining_rate -= visible_rate
                if remaining_rate > 1e-6:
                    painter.pieslice(
                        (pie_x, pie_y),
                        (diameter, diameter),
                        current_angle,
                        270,
                        STA_OTHER_COLOR,
                        stroke=(255, 255, 255, 255),
                        stroke_width=2,
                    )
            else:
                painter.pieslice(
                    (pie_x, pie_y),
                    (diameter, diameter),
                    0,
                    360,
                    STA_OTHER_COLOR,
                )

            left_items = [
                (item_index, elbow_y)
                for item_index, (_, _, _, _, elbow_x, elbow_y) in enumerate(callouts)
                if elbow_x < center_x
            ]
            right_items = [
                (item_index, elbow_y)
                for item_index, (_, _, _, _, elbow_x, elbow_y) in enumerate(callouts)
                if elbow_x >= center_x
            ]
            positions = {
                **_spread_callout_positions(
                    left_items, min_y=28, max_y=94, gap=25
                ),
                **_spread_callout_positions(
                    right_items, min_y=28, max_y=94, gap=25
                ),
            }
            for item_index, callout in enumerate(callouts):
                contributor, color, anchor_x, anchor_y, elbow_x, elbow_y = callout
                is_left = elbow_x < center_x
                target_x = center_x - 100 if is_left else center_x + 100
                target_y = positions[item_index]
                end_x = target_x + 14 if is_left else target_x - 14
                leader_draw.line(
                    (
                        (round(anchor_x), round(anchor_y)),
                        (round(elbow_x), round(elbow_y)),
                        (round(end_x), round(target_y)),
                    ),
                    fill=color,
                    width=2,
                    joint="curve",
                )
                labels.append((
                    contributor,
                    color,
                    round(target_x),
                    round(target_y),
                    is_left,
                ))

            word = _truncate_chart_text(
                rank.word,
                round(column_width - 26),
                font=DEFAULT_BOLD_FONT,
                size=18,
            )
            _draw_chart_text(
                painter,
                word,
                center_x,
                115,
                font=DEFAULT_BOLD_FONT,
                size=18,
                align="center",
            )
            _draw_chart_text(
                painter,
                f"{rank.count:,} 次",
                center_x,
                140,
                size=12,
                color=STA_DRAW_THEME.text_muted,
                align="center",
            )
            if index < len(frozen_ranks) - 1:
                divider_x = round(column_width * (index + 1))
                painter.rect(
                    (divider_x, 12),
                    (1, height - 24),
                    (226, 234, 240, 185),
                )

        painter.paste_with_alphablend(leader_layer, (0, 0))
        for contributor, color, target_x, target_y, is_left in labels:
            painter.pieslice(
                (target_x - 14, target_y - 14),
                (28, 28),
                0,
                360,
                color,
            )
            if contributor.avatar is not None:
                painter.paste_with_alphablend(
                    contributor.avatar,
                    (target_x - 11, target_y - 11),
                    (22, 22),
                )
            else:
                painter.pieslice(
                    (target_x - 11, target_y - 11),
                    (22, 22),
                    0,
                    360,
                    (*color[:3], 125),
                )

            align = "right" if is_left else "left"
            label_x = target_x - 18 if is_left else target_x + 18
            name = _truncate_chart_text(
                contributor.name,
                62,
                font=DEFAULT_BOLD_FONT,
                size=10,
            )
            _draw_chart_text(
                painter,
                name,
                label_x,
                target_y - 13,
                font=DEFAULT_BOLD_FONT,
                size=10,
                align=align,
            )
            _draw_chart_text(
                painter,
                f"{contributor.rate * 100:.0f}%",
                label_x,
                target_y + 2,
                size=9,
                color=STA_DRAW_THEME.text_muted,
                align=align,
            )

    return Spacer(width, height).add_draw_func(draw)


def _sta_report_canvas(
    background: WidgetBg,
    width: int = STA_CANVAS_WIDTH,
):
    """创建继承 STA 插件主题和可配置背景的报告画布。"""
    return create_report_canvas(
        width=width,
        background=background,
        theme=STA_DRAW_THEME,
    )


async def _render_sta_report(
    gid,
    recs,
    interval,
    topk1,
    topk2,
    users,
    names,
    date_str,
    *,
    include_daily_chart: bool,
):
    """绘制每日或长时间 STA 报告，图表全部由 Pillow/Painter 完成。"""

    del topk2  # 原 Matplotlib 用户折线早已停用，保留参数兼容配置和调用方。
    logger.info("开始绘制所有 STA 统计图")
    word_analysis = await prepare_word_analysis(gid, date_str, recs)
    wordcloud_image, word_ranks = draw_wordcloud(
        gid,
        date_str,
        recs,
        users,
        names,
        word_analysis,
    )
    await _download_word_rank_avatars(gid, word_ranks)
    time_labels, message_counts, image_counts = _message_interval_series(recs, interval)
    inner_width = STA_CONTENT_WIDTH - 44
    background = await resolve_configured_draw_background(
        config,
        theme=STA_DRAW_THEME,
        logger=logger,
        label="STA统计",
    )

    with _sta_report_canvas(background) as canvas:
        with create_report_column(
            STA_CONTENT_WIDTH,
            gap=STA_LAYOUT_GAP,
            theme=STA_DRAW_THEME,
        ):
            report_header(
                f"{date_str} 群聊消息统计",
                width=STA_CONTENT_WIDTH,
                eyebrow="CHAT STATISTICS",
                meta=f"{len(recs):,} 条消息",
                theme=STA_DRAW_THEME,
            )

            report_section_title(
                "发言分布", width=STA_CONTENT_WIDTH, theme=STA_DRAW_THEME
            )
            with report_card(
                STA_CONTENT_WIDTH, padding=22, theme=STA_DRAW_THEME
            ):
                await get_pie_frame(
                    gid,
                    date_str,
                    recs,
                    list(users[:topk1]),
                    list(names[:topk1]),
                    width=inner_width,
                )

            report_section_title(
                "词云", width=STA_CONTENT_WIDTH, theme=STA_DRAW_THEME
            )
            with report_card(
                STA_CONTENT_WIDTH,
                padding=22,
                theme=STA_DRAW_THEME,
            ):
                with (
                    VSplit()
                    .set_w(inner_width)
                    .set_sep(10)
                    .set_content_and_item_align("l")
                ):
                    ImageBox(
                        wordcloud_image,
                        image_size_mode="fit",
                        use_alphablend=True,
                    ).set_w(inner_width).set_padding(0).set_content_align("c")
                    TextBox(
                        "",
                        themed_text_style(
                            "muted",
                            size=13,
                            theme=STA_DRAW_THEME,
                        ),
                    ).set_w(inner_width).set_padding((4, 0))
                    _word_rank_widget(word_ranks, width=inner_width)

            report_section_title(
                "消息时段", width=STA_CONTENT_WIDTH, theme=STA_DRAW_THEME
            )
            with report_card(
                STA_CONTENT_WIDTH, padding=22, theme=STA_DRAW_THEME
            ):
                _bar_chart_widget(
                    time_labels,
                    (
                        StaBarSeries(
                            "全部消息",
                            tuple(message_counts),
                            _position_color(0),
                        ),
                        StaBarSeries(
                            "图片消息",
                            tuple(image_counts),
                            _position_color(3),
                        ),
                    ),
                    width=inner_width,
                    height=370,
                    overlay=True,
                    max_x_labels=9,
                )

            if include_daily_chart:
                daily_labels, daily_counts = _daily_message_series(recs)
                report_section_title(
                    "每日消息量", width=STA_CONTENT_WIDTH, theme=STA_DRAW_THEME
                )
                with report_card(
                    STA_CONTENT_WIDTH, padding=22, theme=STA_DRAW_THEME
                ):
                    _bar_chart_widget(
                        daily_labels,
                        (
                            StaBarSeries(
                                "日消息数",
                                tuple(daily_counts),
                                _member_color(0),
                            ),
                        ),
                        width=inner_width,
                        height=370,
                        max_x_labels=10,
                    )

    logger.info("STA 绘制完成")
    return await canvas.get_img()


async def draw_sta(gid, recs, interval, topk1, topk2, user, name, date_str):
    """绘制单日群聊统计。"""

    token = _STA_STYLE_CONTEXT.set(_resolve_sta_style(gid))
    try:
        return await _render_sta_report(
            gid,
            recs,
            interval,
            topk1,
            topk2,
            user,
            name,
            date_str,
            include_daily_chart=False,
        )
    finally:
        _STA_STYLE_CONTEXT.reset(token)


async def draw_sta_sum(gid, recs, interval, topk1, topk2, user, name, date_str):
    """绘制日期范围群聊统计，并附加每日消息柱状图。"""

    token = _STA_STYLE_CONTEXT.set(_resolve_sta_style(gid))
    try:
        return await _render_sta_report(
            gid,
            recs,
            interval,
            topk1,
            topk2,
            user,
            name,
            date_str,
            include_daily_chart=True,
        )
    finally:
        _STA_STYLE_CONTEXT.reset(token)


async def _render_date_count_report(dates, counts, user_counts=None) -> Image.Image:
    """绘制 `/sta_time` 使用的纯 Pillow 日期消息统计报告。"""

    combined = sorted(
        zip(dates, counts, user_counts or [None] * len(dates)),
        key=lambda item: item[0],
    )
    labels = [date.strftime("%m-%d") for date, _, _ in combined]
    all_counts = [count for _, count, _ in combined]
    series = [
        StaBarSeries("全部消息", tuple(all_counts), _position_color(0))
    ]
    if user_counts is not None:
        series.append(
            StaBarSeries(
                "指定用户",
                tuple(int(user_count or 0) for _, _, user_count in combined),
                _position_color(3),
            )
        )

    inner_width = STA_CONTENT_WIDTH - 44
    background = await resolve_configured_draw_background(
        config,
        theme=STA_DRAW_THEME,
        logger=logger,
        label="STA消息趋势",
    )
    with _sta_report_canvas(background) as canvas:
        with create_report_column(
            STA_CONTENT_WIDTH,
            gap=STA_LAYOUT_GAP,
            theme=STA_DRAW_THEME,
        ):
            report_header(
                "群聊消息趋势",
                width=STA_CONTENT_WIDTH,
                eyebrow="CHAT TIMELINE",
                meta=f"{len(labels):,} 天 · {sum(all_counts):,} 条消息",
                theme=STA_DRAW_THEME,
            )
            report_section_title(
                "每日消息量", width=STA_CONTENT_WIDTH, theme=STA_DRAW_THEME
            )
            with report_card(
                STA_CONTENT_WIDTH, padding=22, theme=STA_DRAW_THEME
            ):
                _bar_chart_widget(
                    labels,
                    series,
                    width=inner_width,
                    height=410,
                    overlay=user_counts is not None,
                    max_x_labels=10,
                )
    return await canvas.get_img()


async def _render_word_count_report(
    dates,
    topk_user,
    topk_name,
    user_counts,
    user_date_counts,
    word,
) -> Image.Image:
    """绘制 `/sta_word` 使用的用户占比与每日堆叠柱状图。"""

    other_users = [user for user in user_counts.keys() if user not in topk_user]
    entries = [
        StaPieEntry(
            user_id=user_id,
            name=topk_name[index] if index < len(topk_name) else str(user_id),
            count=user_counts[user_id],
        )
        for index, user_id in enumerate(topk_user)
    ]
    other_count = sum(user_counts[user_id] for user_id in other_users)
    if other_count:
        entries.append(
            StaPieEntry(
                user_id=None,
                name=f"其他（{len(other_users)}人）",
                count=other_count,
                is_other=True,
            )
        )

    ordered_indices = sorted(range(len(dates)), key=lambda index: dates[index])
    labels = [dates[index].strftime("%m-%d") for index in ordered_indices]
    series = []
    for series_index, user_id in enumerate(topk_user):
        name = topk_name[series_index] if series_index < len(topk_name) else str(user_id)
        values = tuple(user_date_counts[index][user_id] for index in ordered_indices)
        series.append(StaBarSeries(name, values, _chart_color(series_index)))
    if other_count:
        values = tuple(
            sum(user_date_counts[index][user_id] for user_id in other_users)
            for index in ordered_indices
        )
        series.append(StaBarSeries("其他", values, STA_OTHER_COLOR))

    inner_width = STA_CONTENT_WIDTH - 44
    total = sum(user_counts.values())
    background = await resolve_configured_draw_background(
        config,
        theme=STA_DRAW_THEME,
        logger=logger,
        label="STA词汇统计",
    )
    with _sta_report_canvas(background) as canvas:
        with create_report_column(
            STA_CONTENT_WIDTH,
            gap=STA_LAYOUT_GAP,
            theme=STA_DRAW_THEME,
        ):
            report_header(
                f"“{word}”词汇统计",
                width=STA_CONTENT_WIDTH,
                eyebrow="WORD STATISTICS",
                meta=f"{len(labels):,} 天 · {total:,} 次",
                theme=STA_DRAW_THEME,
            )
            report_section_title(
                "用户占比", width=STA_CONTENT_WIDTH, theme=STA_DRAW_THEME
            )
            with report_card(
                STA_CONTENT_WIDTH, padding=22, theme=STA_DRAW_THEME
            ):
                _pie_entries_frame(entries, inner_width)
            report_section_title(
                "每日出现次数", width=STA_CONTENT_WIDTH, theme=STA_DRAW_THEME
            )
            with report_card(
                STA_CONTENT_WIDTH, padding=22, theme=STA_DRAW_THEME
            ):
                _bar_chart_widget(
                    labels,
                    series,
                    width=inner_width,
                    height=410,
                    stacked=True,
                    max_x_labels=10,
                )
    return await canvas.get_img()


async def render_date_count_report(
    dates,
    counts,
    user_counts=None,
    *,
    group_id: int | str | None = None,
) -> Image.Image:
    """按请求群主题绘制 `/sta_time` 报告。"""

    token = _STA_STYLE_CONTEXT.set(_resolve_sta_style(group_id))
    try:
        return await _render_date_count_report(dates, counts, user_counts)
    finally:
        _STA_STYLE_CONTEXT.reset(token)


async def render_word_count_report(
    dates,
    topk_user,
    topk_name,
    user_counts,
    user_date_counts,
    word,
    *,
    group_id: int | str | None = None,
) -> Image.Image:
    """按请求群主题绘制 `/sta_word` 报告。"""

    token = _STA_STYLE_CONTEXT.set(_resolve_sta_style(group_id))
    try:
        return await _render_word_count_report(
            dates,
            topk_user,
            topk_name,
            user_counts,
            user_date_counts,
            word,
        )
    finally:
        _STA_STYLE_CONTEXT.reset(token)
