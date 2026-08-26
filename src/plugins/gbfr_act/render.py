from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from PIL import Image, ImageDraw

from ..draw.painter import Painter, get_font, get_font_desc, get_text_size
from ..draw.plot import (
    DEFAULT_BOLD_FONT,
    DEFAULT_FONT,
    Frame,
    Grid,
    HSplit,
    Spacer,
    TextBox,
    VSplit,
)
from ..utils.draw import (
    UnifiedDrawTheme,
    create_report_canvas,
    create_report_column,
    default_page_background,
    report_card,
    report_header,
    report_section_title,
    resolve_configured_draw_background,
    resolve_draw_background,
    resolve_draw_categorical_palette,
    resolve_draw_theme,
    subtle_panel_background,
    themed_text_style,
)
from ..utils.utils import get_shared_draw_background_image
from .models import ActorStats, BattleRecord, format_duration, format_number


@dataclass
class RenderConfig:
    width: int = 1280
    background: str = ""
    chart_bucket_seconds: int = 2
    top_actor_count: int = 4
    top_action_count: int = 5
    show_username: bool = True

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "RenderConfig":
        cfg = cfg or {}
        return cls(
            width=max(900, int(cfg.get("width", 1280) or 1280)),
            # 兼容旧配置；新配置使用 draw.background_image。
            background=str(cfg.get("background", "") or ""),
            chart_bucket_seconds=max(
                1, int(cfg.get("chart_bucket_seconds", 2) or 2)
            ),
            top_actor_count=max(
                1, int(cfg.get("top_actor_count", 4) or 4)
            ),
            top_action_count=max(
                1, int(cfg.get("top_action_count", 5) or 5)
            ),
            show_username=bool(cfg.get("show_username", True)),
        )


@dataclass(frozen=True)
class ActionPieEntry:
    name: str
    damage: int
    hit: int
    is_other: bool = False


async def render_battle_report(
    record: BattleRecord,
    cfg: RenderConfig | None = None,
    *,
    module_config: Any = None,
    group_id: int | str | None = None,
    logger: Any = None,
) -> Image.Image:
    """使用项目统一 draw 组件生成 GBFR 战斗报告。"""

    cfg = cfg or RenderConfig()
    theme = resolve_draw_theme(module_config, group_id=group_id)
    palette = resolve_draw_categorical_palette(
        module_config,
        color_count=max(10, cfg.top_actor_count, cfg.top_action_count + 1),
        group_id=group_id,
    )
    if cfg.background:
        background = await resolve_draw_background(
            cfg.background,
            inherited_source=get_shared_draw_background_image(),
            fallback=default_page_background(theme),
            logger=logger,
            label="GBFR 战斗报告",
        )
    else:
        background = await resolve_configured_draw_background(
            module_config,
            theme=theme,
            logger=logger,
            label="GBFR 战斗报告",
        )

    actors = record.actor_rank(cfg.top_actor_count)
    content_width = cfg.width - theme.page_padding * 2
    inner_width = content_width - 44
    status = (
        "进行中"
        if not record.archived
        else f"已结束 · {record.finish_reason or 'recorded'}"
    )

    with create_report_canvas(
        background=background,
        width=cfg.width,
        theme=theme,
    ) as canvas:
        with create_report_column(content_width, gap=17, theme=theme):
            report_header(
                "GBFR 输出报告",
                width=content_width,
                eyebrow="GRANBLUE FANTASY: RELINK",
                meta=(
                    f"{record.start_time:%Y-%m-%d %H:%M:%S}\n"
                    f"{status} · {format_duration(record.duration_seconds)}"
                ),
                theme=theme,
            )

            _summary_cards(record, content_width, theme)

            report_section_title("队伍输出", width=content_width, theme=theme)
            with report_card(content_width, padding=22, theme=theme):
                _actor_table_widget(
                    record,
                    actors,
                    width=inner_width,
                    cfg=cfg,
                    palette=palette,
                    theme=theme,
                )

            report_section_title("伤害统计", width=content_width, theme=theme)
            with report_card(content_width, padding=22, theme=theme):
                _trend_widget(
                    record,
                    actors,
                    width=inner_width,
                    bucket_seconds=cfg.chart_bucket_seconds,
                    show_username=cfg.show_username,
                    palette=palette,
                    theme=theme,
                )

            report_section_title("伤害构成", width=content_width, theme=theme)
            _action_pie_grid(
                actors,
                width=content_width,
                cfg=cfg,
                palette=palette,
                theme=theme,
            )

            TextBox(
                "数据来自 GBFR-ACT ",
                themed_text_style("muted", size=12, theme=theme),
            ).set_w(content_width).set_padding((4, 2)).set_content_align("c")

    return (await canvas.get_img()).convert("RGB")


def _summary_cards(
    record: BattleRecord,
    width: int,
    theme: UnifiedDrawTheme,
) -> HSplit:
    items = (
        ("总伤害", format_number(record.total_damage)),
        ("全队 DPS", format_number(record.dps)),
        ("命中数", format_number(record.hit)),
        ("记录事件", format_number(len(record.events))),
    )
    gap = 14
    card_width = (width - gap * (len(items) - 1)) // len(items)
    with HSplit().set_w(width).set_sep(gap).set_item_align("t") as row:
        for label, value in items:
            with (
                Frame()
                .set_w(card_width)
                .set_padding((18, 14))
                .set_bg(subtle_panel_background(radius=17, theme=theme))
                .set_content_align("lt")
            ):
                with (
                    VSplit()
                    .set_w(card_width - 36)
                    .set_sep(4)
                    .set_content_and_item_align("l")
                ):
                    TextBox(
                        label,
                        themed_text_style("muted", size=13, theme=theme),
                    ).set_w(card_width - 36).set_padding(0)
                    TextBox(
                        value,
                        themed_text_style("heading", size=23, theme=theme),
                    ).set_w(card_width - 36).set_padding(0)
    return row


def _draw_text(
    painter: Painter,
    text: str,
    x: int,
    y: int,
    *,
    font: str = DEFAULT_FONT,
    size: int = 14,
    color: tuple[int, int, int, int],
    align: str = "left",
) -> None:
    if align != "left":
        width = get_text_size(get_font(font, size), text)[0]
        x -= width // 2 if align == "center" else width
    painter.text(text, (x, y), get_font_desc(font, size), fill=color)


def _truncate_text(text: str, width: int, *, font: str, size: int) -> str:
    if get_text_size(get_font(font, size), text)[0] <= width:
        return text
    suffix = "…"
    while text and get_text_size(get_font(font, size), text + suffix)[0] > width:
        text = text[:-1]
    return text + suffix


def _actor_table_widget(
    record: BattleRecord,
    actors: Sequence[ActorStats],
    *,
    width: int,
    cfg: RenderConfig,
    palette: Sequence[tuple[int, int, int, int]],
    theme: UnifiedDrawTheme,
) -> Spacer:
    frozen_actors = tuple(actors)
    row_height = 51
    height = 39 + max(1, len(frozen_actors)) * row_height

    def draw(_, painter: Painter):
        columns = (
            ("#", 42, "left"),
            ("角色", 282, "left"),
            ("伤害", 150, "right"),
            ("占比", 82, "right"),
            ("DPS", 122, "right"),
            ("命中", 74, "right"),
            ("倒地", 66, "right"),
        )
        fixed_width = sum(item[1] for item in columns)
        action_width = max(120, width - fixed_width)
        all_columns = columns + (("最高伤害类型", action_width, "left"),)

        x = 0
        for label, column_width, align in all_columns:
            label_x = x if align == "left" else x + column_width
            if label == "最高伤害类型":
                label_x += 12
            _draw_text(
                painter,
                label,
                label_x,
                4,
                font=DEFAULT_BOLD_FONT,
                size=13,
                color=theme.text_secondary,
                align=align,
            )
            x += column_width

        if not frozen_actors:
            _draw_text(
                painter,
                "暂无伤害记录",
                width // 2,
                57,
                font=DEFAULT_BOLD_FONT,
                size=18,
                color=theme.text_muted,
                align="center",
            )
            return

        for index, actor in enumerate(frozen_actors):
            y = 35 + index * row_height
            if index % 2 == 0:
                painter.roundrect(
                    (0, y),
                    (width, row_height - 5),
                    theme.rank_fill,
                    radius=12,
                )
            color = palette[index % len(palette)]
            painter.roundrect((8, y + 12), (19, 19), color, radius=6)
            share = actor.damage / record.total_damage * 100 if record.total_damage else 0
            top_action = actor.top_actions(1)
            top_action_text = (
                f"{top_action[0].name} · {format_number(top_action[0].damage)}"
                if top_action
                else "—"
            )
            values = (
                str(index + 1),
                actor.display_name(cfg.show_username),
                format_number(actor.damage),
                f"{share:.1f}%",
                format_number(actor.damage / record.duration_seconds),
                format_number(actor.hit),
                format_number(actor.death_cnt),
                top_action_text,
            )
            x = 0
            for value, (_, column_width, align) in zip(values, all_columns):
                value_width = column_width - (34 if x == 0 else 12)
                display = _truncate_text(
                    value,
                    max(20, value_width),
                    font=DEFAULT_BOLD_FONT if x == 0 else DEFAULT_FONT,
                    size=14,
                )
                value_x = x + (32 if x == 0 else 0)
                if x == fixed_width:
                    value_x += 12
                if align == "right":
                    value_x = x + column_width
                _draw_text(
                    painter,
                    display,
                    value_x,
                    y + 13,
                    font=DEFAULT_BOLD_FONT if x == 0 else DEFAULT_FONT,
                    size=14,
                    color=theme.text_primary,
                    align=align,
                )
                x += column_width

    return Spacer(width, height).add_draw_func(draw)


def _nice_axis(max_value: float) -> tuple[float, tuple[float, ...]]:
    if max_value <= 0:
        return 1.0, (0.0, 1.0)
    raw_step = max_value / 4
    magnitude = 10 ** math.floor(math.log10(raw_step))
    normalized = raw_step / magnitude
    factor = next(value for value in (1, 2, 5, 10) if normalized <= value)
    step = factor * magnitude
    axis_max = math.ceil(max_value / step) * step
    tick_count = max(1, round(axis_max / step))
    return axis_max, tuple(step * index for index in range(tick_count + 1))


def _compact_number(value: float) -> str:
    absolute = abs(value)
    units = ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K"))
    for divisor, suffix in units:
        if absolute >= divisor:
            scaled = value / divisor
            return (
                f"{scaled:.1f}{suffix}"
                if abs(scaled) < 10
                else f"{scaled:.0f}{suffix}"
            )
    return f"{value:.0f}"


def _trend_widget(
    record: BattleRecord,
    actors: Sequence[ActorStats],
    *,
    width: int,
    bucket_seconds: int,
    show_username: bool,
    palette: Sequence[tuple[int, int, int, int]],
    theme: UnifiedDrawTheme,
) -> Spacer:
    frozen_actors = tuple(actors)
    height = 548 if frozen_actors and record.damage_points else 190

    if frozen_actors and record.damage_points:
        times, cumulative, rolling = _build_series(
            record,
            list(frozen_actors),
            bucket_seconds,
        )
    else:
        times, cumulative, rolling = [], {}, {}

    def draw(_, painter: Painter):
        if not times:
            _draw_text(
                painter,
                "暂无曲线数据",
                width // 2,
                height // 2 - 12,
                font=DEFAULT_BOLD_FONT,
                size=20,
                color=theme.text_muted,
                align="center",
            )
            return

        legend_x = 68
        legend_y = 2
        for index, actor in enumerate(frozen_actors):
            name = _truncate_text(
                actor.display_name(show_username),
                180,
                font=DEFAULT_FONT,
                size=13,
            )
            item_width = 26 + get_text_size(get_font(DEFAULT_FONT, 13), name)[0] + 26
            if legend_x + item_width > width:
                legend_x = 68
                legend_y += 24
            color = palette[index % len(palette)]
            painter.roundrect((legend_x, legend_y + 5), (15, 9), color, radius=4)
            _draw_text(
                painter,
                name,
                legend_x + 23,
                legend_y,
                size=13,
                color=theme.text_primary,
            )
            legend_x += item_width

        first_top = 48 + legend_y
        plot_height = 197
        gap = 55
        _draw_line_plot(
            painter,
            width=width,
            top=first_top,
            height=plot_height,
            title="累计伤害",
            times=times,
            series=tuple(cumulative[actor.key] for actor in frozen_actors),
            colors=palette,
            theme=theme,
            show_x_labels=False,
        )
        _draw_line_plot(
            painter,
            width=width,
            top=first_top + plot_height + gap,
            height=plot_height,
            title="DPS",
            times=times,
            series=tuple(rolling[actor.key] for actor in frozen_actors),
            colors=palette,
            theme=theme,
            show_x_labels=True,
        )

    return Spacer(width, height).add_draw_func(draw)


def _draw_line_plot(
    painter: Painter,
    *,
    width: int,
    top: int,
    height: int,
    title: str,
    times: Sequence[int],
    series: Sequence[Sequence[float]],
    colors: Sequence[tuple[int, int, int, int]],
    theme: UnifiedDrawTheme,
    show_x_labels: bool,
) -> None:
    left, right, title_height = 68, 18, 28
    bottom_margin = 31 if show_x_labels else 8
    plot_top = top + title_height
    plot_bottom = top + height - bottom_margin
    plot_width = width - left - right
    plot_height = plot_bottom - plot_top
    max_value = max((max(values, default=0) for values in series), default=0)
    axis_max, ticks = _nice_axis(max_value)

    _draw_text(
        painter,
        title,
        0,
        top,
        font=DEFAULT_BOLD_FONT,
        size=15,
        color=theme.text_primary,
    )
    for tick in ticks:
        y = plot_bottom - round(plot_height * tick / axis_max)
        painter.rect((left, y), (plot_width, 1), (201, 213, 224, 165))
        _draw_text(
            painter,
            _compact_number(tick),
            left - 10,
            y - 7,
            size=11,
            color=theme.text_muted,
            align="right",
        )

    max_time = max(times[-1], 1)
    if show_x_labels:
        for index in range(5):
            value = max_time * index / 4
            x = left + round(plot_width * index / 4)
            _draw_text(
                painter,
                format_duration(value),
                x,
                plot_bottom + 10,
                size=11,
                color=theme.text_muted,
                align="center",
            )

    layer = Image.new("RGBA", (width, top + height), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    for index, values in enumerate(series):
        points = [
            (
                left + round(plot_width * time / max_time),
                plot_bottom - round(plot_height * value / axis_max),
            )
            for time, value in zip(times, values)
        ]
        if len(points) >= 2:
            layer_draw.line(
                points,
                fill=colors[index % len(colors)],
                width=3,
                joint="curve",
            )
        elif points:
            x, y = points[0]
            layer_draw.ellipse(
                (x - 2, y - 2, x + 2, y + 2),
                fill=colors[index % len(colors)],
            )
    painter.paste_with_alphablend(layer, (0, 0))


def _action_pie_grid(
    actors: Sequence[ActorStats],
    *,
    width: int,
    cfg: RenderConfig,
    palette: Sequence[tuple[int, int, int, int]],
    theme: UnifiedDrawTheme,
) -> Grid | Frame:
    if not actors:
        with report_card(width, padding=22, theme=theme) as empty_card:
            TextBox(
                "暂无动作记录",
                themed_text_style("muted", size=16, theme=theme),
            ).set_w(width - 44).set_padding((0, 26)).set_content_align("c")
        return empty_card

    gap = 16
    card_width = (width - gap) // 2
    inner_width = card_width - 36
    with Grid(col_count=2).set_sep(gap, gap).set_item_align("t") as grid:
        for actor in actors:
            with report_card(
                card_width,
                padding=18,
                compact=True,
                theme=theme,
            ):
                with (
                    VSplit()
                    .set_w(inner_width)
                    .set_sep(7)
                    .set_content_and_item_align("l")
                ):
                    TextBox(
                        actor.display_name(cfg.show_username),
                        themed_text_style("heading", size=17, theme=theme),
                        line_count=1,
                    ).set_w(inner_width).set_padding(0)
                    _action_pie_widget(
                        actor,
                        width=inner_width,
                        limit=cfg.top_action_count,
                        palette=palette,
                        theme=theme,
                    )
    return grid


def _action_pie_entries(actor: ActorStats, limit: int) -> tuple[ActionPieEntry, ...]:
    actions = sorted(
        actor.actions.values(),
        key=lambda item: (-item.damage, item.action_id),
    )
    selected = [
        ActionPieEntry(action.name, action.damage, action.hit)
        for action in actions[:limit]
    ]
    remaining = actions[limit:]
    if remaining:
        selected.append(
            ActionPieEntry(
                "其他动作",
                sum(action.damage for action in remaining),
                sum(action.hit for action in remaining),
                is_other=True,
            )
        )
    return tuple(selected)


def _action_pie_widget(
    actor: ActorStats,
    *,
    width: int,
    limit: int,
    palette: Sequence[tuple[int, int, int, int]],
    theme: UnifiedDrawTheme,
) -> Spacer:
    entries = _action_pie_entries(actor, limit)
    height = max(238, len(entries) * 37 + 20)

    def draw(_, painter: Painter):
        donut_size = min(174, height - 36)
        donut_x = 4
        donut_y = (height - donut_size) // 2
        center_x = donut_x + donut_size // 2
        center_y = donut_y + donut_size // 2
        total = sum(entry.damage for entry in entries)

        if total <= 0:
            painter.pieslice(
                (donut_x, donut_y),
                (donut_size, donut_size),
                0,
                360,
                (112, 119, 132, 255),
            )
        else:
            angle = -90.0
            for index, entry in enumerate(entries):
                sweep = 360 * entry.damage / total
                painter.pieslice(
                    (donut_x, donut_y),
                    (donut_size, donut_size),
                    angle,
                    angle + sweep,
                    palette[index % len(palette)],
                    stroke=(255, 255, 255, 255),
                    stroke_width=2,
                )
                angle += sweep

        inner_size = round(donut_size * 0.56)
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
            _compact_number(total),
            center_x,
            center_y - 19,
            font=DEFAULT_BOLD_FONT,
            size=20,
            color=theme.text_primary,
            align="center",
        )
        _draw_text(
            painter,
            "总伤害",
            center_x,
            center_y + 11,
            size=12,
            color=theme.text_muted,
            align="center",
        )

        legend_x = donut_x + donut_size + 25
        legend_width = max(1, width - legend_x)
        content_height = len(entries) * 37
        legend_top = max(3, (height - content_height) // 2)
        if not entries:
            _draw_text(
                painter,
                "暂无动作记录",
                legend_x,
                height // 2 - 8,
                size=14,
                color=theme.text_muted,
            )
            return

        for index, entry in enumerate(entries):
            y = legend_top + index * 37
            color = palette[index % len(palette)]
            painter.roundrect((legend_x, y + 4), (13, 13), color, radius=5)
            name = _truncate_text(
                entry.name,
                max(20, legend_width - 23),
                font=DEFAULT_BOLD_FONT,
                size=13,
            )
            _draw_text(
                painter,
                name,
                legend_x + 21,
                y,
                font=DEFAULT_BOLD_FONT,
                size=13,
                color=theme.text_primary,
            )
            percentage = entry.damage / total * 100 if total else 0
            _draw_text(
                painter,
                f"{format_number(entry.damage)} · {percentage:.1f}% · {entry.hit} hit",
                legend_x + 21,
                y + 18,
                size=11,
                color=theme.text_muted,
            )

    return Spacer(width, height).add_draw_func(draw)


def _build_series(
    record: BattleRecord,
    actors: list[ActorStats],
    bucket_seconds: int,
):
    max_time = max(bucket_seconds, int(math.ceil(record.duration_seconds)))
    times = list(range(0, max_time + bucket_seconds, bucket_seconds))
    keys = [actor.key for actor in actors]
    cumulative = {key: [] for key in keys}
    rolling = {key: [] for key in keys}
    points_by_actor = {
        key: sorted(
            (
                ((point.time_ms - record.start_time_ms) / 1000, point.damage)
                for point in record.damage_points
                if point.actor_key == key
            ),
            key=lambda item: item[0],
        )
        for key in keys
    }

    for key in keys:
        points = points_by_actor[key]
        total = 0
        window_damage = 0
        right = 0
        left = 0
        for time in times:
            while right < len(points) and points[right][0] <= time:
                total += points[right][1]
                window_damage += points[right][1]
                right += 1
            while left < right and points[left][0] < time - 60:
                window_damage -= points[left][1]
                left += 1
            elapsed = max(1, min(60, time))
            cumulative[key].append(total)
            rolling[key].append(window_damage / elapsed)
    return times, cumulative, rolling
