from __future__ import annotations

import math
from datetime import date, timedelta

from PIL import Image, ImageDraw

from ..draw.painter import get_font
from ..draw.plot import (
    DEFAULT_BOLD_FONT,
    DEFAULT_FONT,
    ImageBox,
    TextBox,
)
from ..utils.draw import (
    UnifiedDrawTheme,
    create_report_canvas,
    create_report_column,
    report_card,
    report_header,
    resolve_configured_draw_background,
    resolve_draw_theme,
    themed_text_style,
)
from ..utils.utils import run_in_pool
from .storage import AttendanceStats


RGBA = tuple[int, int, int, int]


def _mix_color(source: RGBA, target: RGBA, ratio: float) -> RGBA:
    ratio = max(0.0, min(1.0, ratio))
    return tuple(
        round(left + (right - left) * ratio)
        for left, right in zip(source, target)
    )


def _shade(color: RGBA, ratio: float) -> RGBA:
    return _mix_color(color, (24, 28, 36, color[3]), ratio)


def _heat_color(theme: UnifiedDrawTheme, ratio: float) -> RGBA:
    """把人数比例映射到同一主题色相下清晰可辨的明暗色阶。"""

    light = _mix_color(
        theme.accent_start,
        (255, 255, 255, 255),
        0.52,
    )
    dark = _shade(theme.accent_dark, 0.18)
    return _mix_color(light, dark, math.pow(max(0.0, min(1.0, ratio)), 0.78))


def _chart_date_bounds(start_date: date, end_date: date) -> tuple[date, date]:
    # GitHub 日历以周日为每列第一行，周六为最后一行。
    days_since_sunday = (start_date.weekday() + 1) % 7
    days_until_saturday = (5 - end_date.weekday()) % 7
    return (
        start_date - timedelta(days=days_since_sunday),
        end_date + timedelta(days=days_until_saturday),
    )


def _month_markers(
    start_date: date,
    end_date: date,
    chart_start: date,
) -> tuple[tuple[int, str], ...]:
    markers: list[tuple[int, str]] = []
    marker_day = start_date
    while marker_day <= end_date:
        week_index = (marker_day - chart_start).days // 7
        marker = (week_index, f"{marker_day.month}月")
        if markers and markers[-1][0] == week_index:
            # 同一周跨月时优先标出较新的月份，避免短范围只显示上月。
            markers[-1] = marker
        else:
            markers.append(marker)
        next_month = marker_day.month % 12 + 1
        next_year = marker_day.year + (marker_day.month == 12)
        marker_day = date(next_year, next_month, 1)
    return tuple(markers)


def _draw_isometric_heatmap(
    cells: tuple[tuple[int, int, int, bool], ...],
    weeks: int,
    month_markers: tuple[tuple[int, str], ...],
    tile: int,
    step: int,
    half_step: int,
    max_bar_height: int,
    max_count: int,
    theme: UnifiedDrawTheme,
    width: int,
    height: int,
    show_axes: bool,
    show_footer: bool,
) -> Image.Image:
    """预渲染抗锯齿的等距投影图层，供 draw 的 ImageBox 合成。"""

    scale = 2
    overlay = Image.new(
        "RGBA",
        (width * scale, height * scale),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(overlay)

    grid_width = (weeks + 5) * step + tile * 2
    grid_left = max(0, (width - grid_width) // 2)
    origin_x = grid_left + 6 * step + tile
    origin_y = max_bar_height + 34

    def scaled_points(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(x * scale, y * scale) for x, y in points]

    def polygon(
        points: list[tuple[int, int]],
        fill: RGBA,
        outline: RGBA | None = None,
    ) -> None:
        points_scaled = scaled_points(points)
        draw.polygon(points_scaled, fill=fill)
        if outline is not None:
            draw.line(
                points_scaled + [points_scaled[0]],
                fill=outline,
                width=max(1, scale),
                joint="curve",
            )

    def draw_text(
        position: tuple[int, int],
        text: str,
        *,
        size: int,
        fill: RGBA,
        anchor: str,
        bold: bool = False,
    ) -> None:
        draw.text(
            (position[0] * scale, position[1] * scale),
            text,
            font=get_font(
                DEFAULT_BOLD_FONT if bold else DEFAULT_FONT,
                size * scale,
            ),
            fill=fill,
            anchor=anchor,
        )

    zero_color = _mix_color(
        theme.subtle_fill,
        (226, 232, 238, theme.subtle_fill[3]),
        0.46,
    )
    zero_outline = _mix_color(zero_color, theme.text_muted, 0.17)
    bar_outline = _mix_color(theme.accent_dark, theme.text_primary, 0.30)

    # 远处格子先画，靠近观察者的格子后画，保证高柱遮挡关系正确。
    for week, weekday, count, in_range in sorted(
        cells,
        key=lambda item: (item[0] + item[1], item[0]),
    ):
        if not in_range:
            continue
        x = origin_x + (week - weekday) * step
        y = origin_y + (week + weekday) * half_step
        a = (tile, tile // 2)
        b = (-tile, tile // 2)
        ground = [
            (x, y),
            (x + a[0], y + a[1]),
            (x + a[0] + b[0], y + a[1] + b[1]),
            (x + b[0], y + b[1]),
        ]

        if count <= 0 or max_count <= 0:
            polygon(ground, zero_color, zero_outline)
            continue

        ratio = count / max_count
        min_bar_height = max(8, tile // 2)
        bar_height = round(
            min_bar_height
            + (max_bar_height - min_bar_height) * math.pow(ratio, 0.72)
        )
        top_color = _heat_color(theme, ratio)
        right_color = _shade(top_color, 0.23)
        left_color = _shade(top_color, 0.38)
        top = [
            (point_x, point_y - bar_height)
            for point_x, point_y in ground
        ]

        polygon(
            [top[1], ground[1], ground[2], top[2]],
            right_color,
            bar_outline,
        )
        polygon(
            [top[2], ground[2], ground[3], top[3]],
            left_color,
            bar_outline,
        )
        polygon(top, top_color, bar_outline)

    if show_axes:
        # 月份沿远端边缘排列，保持与等距投影周轴一致。
        for week, label in month_markers:
            x = origin_x + week * step
            y = origin_y + week * half_step - max_bar_height - 13
            draw_text(
                (x, y),
                label,
                size=13,
                fill=theme.text_secondary,
                anchor="ms",
                bold=True,
            )

        weekday_labels = {1: "周一", 3: "周三", 5: "周五"}
        for weekday, label in weekday_labels.items():
            x = origin_x - weekday * step - tile - 10
            y = origin_y + weekday * half_step + tile // 2
            draw_text(
                (x, y),
                label,
                size=12,
                fill=theme.text_muted,
                anchor="rm",
            )

    if show_footer:
        footer_y = height - 25

        legend_tile = 9
        legend_start_x = width - 220
        draw_text(
            (legend_start_x - 14, footer_y),
            "少",
            size=12,
            fill=theme.text_muted,
            anchor="rm",
        )
        for index in range(5):
            ratio = index / 4
            color = (
                zero_color
                if index == 0
                else _heat_color(theme, ratio)
            )
            x = legend_start_x + index * 24
            y = footer_y - legend_tile // 2
            polygon(
                [
                    (x, y),
                    (x + legend_tile, y + legend_tile // 2),
                    (x, y + legend_tile),
                    (x - legend_tile, y + legend_tile // 2),
                ],
                color,
            )
        draw_text(
            (legend_start_x + 5 * 24, footer_y),
            "多",
            size=12,
            fill=theme.text_muted,
            anchor="lm",
        )

    return overlay.resize((width, height), Image.Resampling.LANCZOS)


def _calendar_cells(
    stats: AttendanceStats,
) -> tuple[
    date,
    int,
    tuple[tuple[int, int, int, bool], ...],
    tuple[tuple[int, str], ...],
]:
    chart_start, chart_end = _chart_date_bounds(
        stats.start_date,
        stats.end_date,
    )
    weeks = (chart_end - chart_start).days // 7 + 1
    cells: list[tuple[int, int, int, bool]] = []
    current = chart_start
    while current <= chart_end:
        day_offset = (current - chart_start).days
        cells.append(
            (
                day_offset // 7,
                day_offset % 7,
                stats.daily_counts.get(current, 0),
                stats.start_date <= current <= stats.end_date,
            )
        )
        current += timedelta(days=1)
    return (
        chart_start,
        weeks,
        tuple(cells),
        _month_markers(
            stats.start_date,
            stats.end_date,
            chart_start,
        ),
    )


def _render_isometric_layer(
    stats: AttendanceStats,
    width: int,
    theme: UnifiedDrawTheme,
) -> Image.Image:
    _, weeks, cells, month_markers = _calendar_cells(stats)
    step = max(18, min(42, (width - 70) // (weeks + 7)))
    tile = max(12, step - 3)
    half_step = max(7, step // 2)
    max_bar_height = max(54, min(108, tile * 3))
    origin_y = max_bar_height + 34
    height = (
        origin_y
        + (weeks + 5) * half_step
        + tile
        + 24
    )

    return _draw_isometric_heatmap(
        cells,
        weeks,
        month_markers,
        tile,
        step,
        half_step,
        max_bar_height,
        stats.peak_count,
        theme,
        width,
        height,
        False,
        False,
    )


def _render_flat_heatmap(
    stats: AttendanceStats,
    width: int,
    theme: UnifiedDrawTheme,
) -> Image.Image:
    """绘制与 GitHub contribution graph 对应的传统平面日历。"""

    _, weeks, cells, month_markers = _calendar_cells(stats)
    scale = 2
    left_margin = 64
    right_margin = 12
    display_weeks = weeks
    step = max(
        12,
        min(
            64,
            (width - left_margin - right_margin) // display_weeks,
        ),
    )
    cell_gap = max(3, round(step * 0.12))
    cell_size = max(9, step - cell_gap)
    grid_left = width - right_margin - display_weeks * step
    top_margin = 34
    grid_height = 6 * step + cell_size
    footer_y = top_margin + grid_height + 35
    height = footer_y + 22
    image = Image.new(
        "RGBA",
        (width * scale, height * scale),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(image)

    def text(
        position: tuple[int, int],
        value: str,
        *,
        size: int,
        fill: RGBA,
        anchor: str = "lt",
        bold: bool = False,
    ) -> None:
        draw.text(
            (position[0] * scale, position[1] * scale),
            value,
            font=get_font(
                DEFAULT_BOLD_FONT if bold else DEFAULT_FONT,
                size * scale,
            ),
            fill=fill,
            anchor=anchor,
        )

    zero_color = _mix_color(
        theme.subtle_fill,
        (226, 232, 238, theme.subtle_fill[3]),
        0.46,
    )

    for week, weekday, count, in_range in cells:
        if not in_range:
            continue
        ratio = count / stats.peak_count if stats.peak_count else 0.0
        color = zero_color if count <= 0 else _heat_color(theme, ratio)
        x = grid_left + week * step
        y = top_margin + weekday * step
        draw.rounded_rectangle(
            (
                x * scale,
                y * scale,
                (x + cell_size) * scale,
                (y + cell_size) * scale,
            ),
            radius=max(1, 2 * scale),
            fill=color,
        )

    for week, label in month_markers:
        x = grid_left + week * step
        text(
            (x, 8),
            label,
            size=12,
            fill=theme.text_secondary,
            bold=True,
        )

    for weekday, label in {1: "周一", 3: "周三", 5: "周五"}.items():
        y = top_margin + weekday * step + cell_size // 2
        text(
            (grid_left - 10, y),
            label,
            size=12,
            fill=theme.text_muted,
            anchor="rm",
        )

    legend_x = width - 208
    text(
        (legend_x - 10, footer_y),
        "少",
        size=12,
        fill=theme.text_muted,
        anchor="rm",
    )
    for index in range(5):
        color = (
            zero_color
            if index == 0
            else _heat_color(theme, index / 4)
        )
        x = legend_x + index * 23
        y = footer_y - cell_size // 2
        draw.rounded_rectangle(
            (
                x * scale,
                y * scale,
                (x + cell_size) * scale,
                (y + cell_size) * scale,
            ),
            radius=max(1, 2 * scale),
            fill=color,
        )
    text(
        (legend_x + 5 * 23, footer_y),
        "多",
        size=12,
        fill=theme.text_muted,
        anchor="lm",
    )
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _draw_overview_statistics(
    image: Image.Image,
    stats: AttendanceStats,
    theme: UnifiedDrawTheme,
) -> None:
    """在 3D 主图两侧放置四项主题色统计。"""

    draw = ImageDraw.Draw(image)
    width, height = image.size

    def text(
        position: tuple[int, int],
        value: str,
        *,
        size: int,
        fill: RGBA,
        anchor: str = "lt",
        bold: bool = False,
    ) -> None:
        draw.text(
            position,
            value,
            font=get_font(
                DEFAULT_BOLD_FONT if bold else DEFAULT_FONT,
                size,
            ),
            fill=fill,
            anchor=anchor,
        )

    def statistic_value(
        position: tuple[int, int],
        value: str,
        unit: str,
        *,
        value_size: int,
        unit_size: int,
        gap: int,
    ) -> None:
        """让不同字号的统计数字与单位共用文字基线。"""

        value_font = get_font(DEFAULT_BOLD_FONT, value_size)
        draw.text(
            position,
            value,
            font=value_font,
            fill=theme.accent_dark,
            anchor="ls",
        )
        text(
            (
                position[0] + round(value_font.getlength(value)) + gap,
                position[1],
            ),
            unit,
            size=unit_size,
            fill=theme.text_primary,
            anchor="ls",
            bold=True,
        )

    panel_fill = _mix_color(
        theme.subtle_fill,
        (255, 255, 255, 236),
        0.62,
    )
    panel_outline = _mix_color(theme.accent, theme.accent_dark, 0.22)
    right_x = width - 326
    draw.rounded_rectangle(
        (right_x - 18, 10, width - 6, 260),
        radius=18,
        fill=panel_fill,
        outline=panel_outline,
        width=2,
    )
    text(
        (right_x, 28),
        "累计打卡",
        size=17,
        fill=theme.accent_dark,
        bold=True,
    )
    total_text = f"{stats.total_checkins:,}"
    statistic_value(
        (right_x, 91),
        total_text,
        "人次",
        value_size=48,
        unit_size=18,
        gap=10,
    )
    text(
        (right_x, 108),
        f"{stats.start_date:%Y-%m-%d} — {stats.end_date:%Y-%m-%d}",
        size=12,
        fill=theme.text_muted,
    )

    text(
        (right_x, 166),
        "单日峰值",
        size=17,
        fill=theme.accent_dark,
        bold=True,
    )
    peak_text = f"{stats.peak_count:,}"
    statistic_value(
        (right_x, 223),
        peak_text,
        "人",
        value_size=42,
        unit_size=17,
        gap=10,
    )
    peak_date_text = (
        stats.peak_date.strftime("%Y-%m-%d")
        if stats.peak_date is not None
        else "暂无打卡"
    )
    text(
        (right_x + 105, 212),
        peak_date_text,
        size=12,
        fill=theme.text_muted,
    )

    left_y = max(190, height - 212)
    draw.rounded_rectangle(
        (6, left_y - 10, 290, height - 6),
        radius=18,
        fill=panel_fill,
        outline=panel_outline,
        width=2,
    )
    for index, (label, value, unit) in enumerate(
        (
            (
                "旅行人数",
                stats.unique_people,
                "人",
            ),
            (
                "记录天数",
                stats.active_days,
                "天",
            ),
        )
    ):
        y = left_y + index * 96
        text(
            (24, y),
            label,
            size=16,
            fill=theme.accent_dark,
            bold=True,
        )
        value_text = f"{value:,}"
        statistic_value(
            (24, y + 55),
            value_text,
            unit,
            value_size=38,
            unit_size=16,
            gap=9,
        )


def _render_overview_layer(
    stats: AttendanceStats,
    flat_stats: AttendanceStats,
    width: int,
    theme: UnifiedDrawTheme,
) -> Image.Image:
    isometric = _render_isometric_layer(stats, width, theme)
    _draw_overview_statistics(isometric, stats, theme)
    flat = _render_flat_heatmap(flat_stats, width, theme)
    flat_top = isometric.height + 42
    image = Image.new(
        "RGBA",
        (width, flat_top + flat.height),
        (0, 0, 0, 0),
    )
    image.alpha_composite(isometric, (0, 0))
    divider_draw = ImageDraw.Draw(image)
    divider_draw.line(
        (4, isometric.height + 10, width - 4, isometric.height + 10),
        fill=_mix_color(theme.text_muted, (255, 255, 255, 255), 0.72),
        width=1,
    )
    image.alpha_composite(flat, (0, flat_top))
    return image


async def render_attendance_heatmap(
    stats: AttendanceStats,
    *,
    flat_stats: AttendanceStats | None = None,
    months: int,
    group_id: int,
    config,
    logger,
):
    theme = resolve_draw_theme(config, group_id=group_id)
    canvas_width = max(
        900,
        min(1800, int(config.get("draw.canvas_width", 1260))),
    )
    content_width = canvas_width - theme.page_padding * 2
    card_padding = 22
    inner_width = content_width - card_padding * 2
    background = await resolve_configured_draw_background(
        config,
        theme=theme,
        logger=logger,
        label="旅行出勤",
    )

    date_meta = (
        f"{stats.start_date:%Y-%m-%d} — {stats.end_date:%Y-%m-%d}"
    )
    overview_layer = await run_in_pool(
        _render_overview_layer,
        stats,
        flat_stats or stats,
        inner_width,
        theme,
    )

    with create_report_canvas(
        background=background,
        width=canvas_width,
        theme=theme,
    ) as canvas:
        with create_report_column(
            content_width,
            gap=16,
            theme=theme,
        ):
            report_header(
                f"{stats.country} 旅行打卡",
                width=content_width,
                eyebrow="TRAVEL ATTENDANCE",
                meta=date_meta,
                theme=theme,
            )

            with report_card(
                content_width,
                padding=card_padding,
                theme=theme,
            ):
                ImageBox(
                    overview_layer,
                    use_alphablend=True,
                )

    return await canvas.get_img()
