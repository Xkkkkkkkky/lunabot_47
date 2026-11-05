"""用公共 draw 组件排版紧凑的雷达成绩看板。"""

from __future__ import annotations

from datetime import timedelta, timezone

from PIL import Image

from ..draw.painter import DEFAULT_BOLD_FONT, get_font, get_text_width
from ..draw.plot import Frame, HSplit, LinearGradient, RoundRectBg, Spacer, TextBox, VSplit
from ..utils.draw import (
    create_report_canvas,
    create_report_column,
    resolve_configured_draw_background,
    resolve_draw_categorical_palette,
    resolve_draw_theme,
    themed_text_style,
)
from .data import DIMENSIONS, RadarSnapshot


WHITE = (255, 255, 255, 255)


def _mix(color, target, ratio):
    return tuple(round(a + (b - a) * ratio) for a, b in zip(color, target))


def _model_name(model: str) -> str:
    """仅美化名称，不依赖固定模型清单。"""
    if model.startswith("gpt-"):
        parts = model[4:].split("-")
        return "GPT-" + parts[0] + "".join(" " + p.capitalize() for p in parts[1:])
    return " ".join(part.capitalize() for part in model.split("-"))


def _text(value, width, theme, *, role="body", size=14, lines=1, color=None, align="l"):
    return TextBox(
        value,
        themed_text_style(role, size=size, color=color, theme=theme),
        line_count=lines,
        use_real_line_count=lines is None,
    ).set_w(width).set_padding(0).set_content_align(align)


def _radar(widget, painter):
    """用绘图内核的圆环、扫描扇面和亮点装饰页头，不占用数据区。"""
    side = min(widget.w, widget.h)
    center = side // 2
    for radius in (side // 2 - 3, side // 3, side // 6):
        painter.pieslice(
            (center - radius, center - radius), (radius * 2, radius * 2),
            0, 360, (0, 0, 0, 0), stroke=(255, 255, 255, 64),
        )
    painter.rect((center, 5), (1, side - 10), (255, 255, 255, 32))
    painter.rect((5, center), (side - 10, 1), (255, 255, 255, 32))
    radius = side // 2 - 4
    painter.pieslice(
        (center - radius, center - radius), (radius * 2, radius * 2),
        285, 332, (255, 255, 255, 30),
    )
    for x, y, diameter in ((0.66, 0.28, 5), (0.31, 0.63, 4), (0.55, 0.57, 3)):
        painter.pieslice((round(side * x), round(side * y)), (diameter, diameter), 0, 360, WHITE)


def _header(snapshot, title, width, models, theme):
    ink = _mix(theme.accent_dark, theme.text_primary, 0.65)
    gradient = LinearGradient(ink, _mix(theme.accent_dark, theme.text_primary, 0.18), (0, 0), (1, 1))
    fetched = snapshot.fetched_at.astimezone(timezone(timedelta(hours=8)))
    wide = width >= 850
    padding, icon_width, gap = 18, 64 if wide else 48, 16
    right_width = 250 if wide else 0
    text_width = width - padding * 2 - icon_width - gap - (right_width + gap if wide else 0)
    with Frame().set_w(width).set_padding(padding).set_bg(RoundRectBg(gradient, 20)):
        with HSplit().set_sep(gap).set_item_align("c"):
            Spacer(icon_width, icon_width).add_draw_func(_radar)
            with VSplit().set_w(text_width).set_sep(7).set_content_and_item_align("l"):
                _text(f"Codex Radar · {title}", text_width, theme, role="title", size=28 if wide else 23, color=WHITE, lines=None)
                description = (
                    f"{len(models)} 模型 · {len(snapshot.points)} 档"
                    if wide else f"满分 {snapshot.score_max:g} 分 · 读取于 {fetched:%m-%d %H:%M}"
                )
                _text(description, text_width, theme, size=13, color=(233, 235, 255, 255), lines=None)
            if wide:
                with VSplit().set_w(right_width).set_sep(7).set_content_and_item_align("r"):
                    _text("数据来自 Codex 雷达站", right_width, theme, size=14, color=WHITE, align="r")
                    _text(f"读取于 {fetched:%Y-%m-%d %H:%M}（北京时间）", right_width, theme, size=12, color=(222, 225, 245, 255), align="r")


def _model_badge(model, best, width, height, color, theme, *, overview):
    gradient = LinearGradient(
        _mix(color, theme.text_primary, 0.35), _mix(color, theme.text_primary, 0.14),
        (0, 0), (1, 1),
    )
    with Frame().set_w(width).set_padding((14, 10)).set_bg(RoundRectBg(gradient, 13)).set_content_align("l"):
        # 用占位撑出最小高度；新模型名称换行时让整行自然增高。
        Spacer(1, height - 20)
        with VSplit().set_w(width - 28).set_sep(6 if overview else 4).set_content_and_item_align("l"):
            _text(_model_name(model), width - 28, theme, role="heading", size=21, lines=None, color=WHITE)
            label = "暂无有效成绩" if best is None else f"最高 {best[0]:.1f} · {best[1]}"
            _text(label, width - 28, theme, size=12 if overview else 11, color=(245, 243, 255, 255), lines=None)


def _meter(score, width, score_max, color):
    with Frame().set_size((width, 5)).set_content_align("l"):
        Spacer(width, 5).set_bg(RoundRectBg(_mix(color, WHITE, 0.88), 2))
        # 缺失值不进入这里；真实零分只画底轨，保留原始数值的含义。
        filled = round(width * min(1.0, max(0.0, score / score_max)))
        if filled:
            Spacer(filled, 5).set_bg(RoundRectBg(color, 2))


def _cell(point, source, width, height, score_max, color, theme, *, overview, best):
    score = point.scores.get(source) if point is not None else None
    is_best = best is not None and score is not None and score.score == best[0]
    cell = Frame().set_size((width, height)).set_padding((10, 8)).set_content_align("l")
    if is_best:
        cell.set_bg(RoundRectBg(_mix(color, WHITE, 0.93), 11, stroke=_mix(color, WHITE, 0.70)))
    inner = width - 20
    with cell:
        with VSplit().set_w(inner).set_sep(6 if overview else 5).set_content_and_item_align("l"):
            if point is None:
                status = "暂无数据"
            elif point.error:
                status = "读取失败"
            elif score is None:
                status = "暂未提供"
            elif score.coverage == 0:
                status = "暂无判分"
            elif score.score is None:
                status = "暂无数据"
            else:
                status = None
            if status:
                _text("—", inner, theme, role="heading", size=26 if overview else 22, color=_mix(color, WHITE, 0.65))
                _text(status, inner, theme, role="muted", size=12)
                return

            badge_width, badge_gap = (58, 8) if overview else (46, 4)
            score_width = inner - badge_width - badge_gap
            score_text = f"{score.score:.1f}"
            score_size = 34 if overview else 28
            # TextBox 的 shrink 会省略尾部；数值必须完整，改为按字宽缩小字号。
            while get_text_width(get_font(DEFAULT_BOLD_FONT, score_size), score_text) > score_width:
                score_size -= 1
            with HSplit().set_w(inner).set_sep(badge_gap).set_item_align("c"):
                _text(
                    score_text, score_width, theme,
                    role="heading", size=score_size, color=color,
                )
                count = "题次未知" if score.graded_runs_count is None else f"{score.graded_runs_count:,}题次"
                _text(count, badge_width, theme, size=11 if overview else 10, color=color, align="c").set_padding((4, 4)).set_bg(RoundRectBg(_mix(color, WHITE, 0.89), 6))
            _meter(score.score, inner, score_max, color)
            if overview:
                price = "费用未知" if point.average_price_usd is None else f"${point.average_price_usd:.2f}" + ("*" if point.price_partial else "")
                minutes = "耗时未知" if point.average_minutes is None else f"{point.average_minutes:.1f} 分钟" + ("*" if point.duration_partial else "")
                half = (inner - 6) // 2
                with HSplit().set_w(inner).set_sep(6):
                    _text(price, half, theme, role="secondary", size=12)
                    _text(minutes, inner - half - 6, theme, role="secondary", size=12, align="r")
            if score.coverage is not None and score.coverage < score.required_tasks:
                _text(f"已覆盖 {score.coverage}/{score.required_tasks} 题", inner, theme, role="muted", size=11)


async def render_radar(
    snapshot: RadarSnapshot,
    dimension: str,
    *,
    config,
    logger,
    group_id: int | None = None,
) -> Image.Image:
    title, source = DIMENSIONS[dimension]
    overview = dimension == "total"
    theme = resolve_draw_theme(config, group_id=group_id)
    models = list(dict.fromkeys(point.model for point in snapshot.points))
    present_efforts = list(dict.fromkeys(point.effort for point in snapshot.points))
    known = ("ultra", "max", "xhigh", "high", "medium", "low")
    efforts = [effort for effort in known if effort in present_efforts]
    efforts.extend(effort for effort in present_efforts if effort not in known)
    points = {(point.model, point.effort): point for point in snapshot.points}
    palette = resolve_draw_categorical_palette(config, color_count=max(1, len(models)), group_id=group_id)
    # 加深主题色，让白色模型名和浅色底上的数值都保持足够的对比度。
    colors = [_mix(color, theme.text_primary, 0.16) for color in palette]
    cell_width, cell_height = (180, 86) if overview else (150, 56)
    name_width, gap, inset, padding = 220, 8, 8, 24
    content_width = inset * 2 + name_width + len(efforts) * (cell_width + gap)
    background = await resolve_configured_draw_background(config, theme=theme, logger=logger, label="Codex 雷达")

    with create_report_canvas(background=background, width=content_width + padding * 2, padding=padding, theme=theme) as canvas:
        with create_report_column(content_width, gap=10, theme=theme):
            _header(snapshot, title, content_width, models, theme)
            if overview and content_width >= 850:
                with HSplit().set_w(content_width).set_sep(8).set_item_align("c"):
                    for key, (label, _) in DIMENSIONS.items():
                        TextBox(label, themed_text_style("heading" if key == dimension else "secondary", size=13, color=WHITE if key == dimension else theme.text_secondary, theme=theme)).set_padding((13, 7)).set_bg(RoundRectBg(theme.accent_dark if key == dimension else (255, 255, 255, 150), 9))
            # 表头与行内单元格共用相同宽度和内边距，消除原来独立卡片之间的空洞。
            with HSplit().set_w(content_width).set_padding((inset, 2)).set_sep(gap).set_item_align("c"):
                _text("模型", name_width, theme, role="muted", size=12).set_padding((14, 0))
                for effort in efforts:
                    _text(effort.upper(), cell_width, theme, role="heading", size=13, align="c", lines=None)
            with VSplit().set_w(content_width).set_sep(8).set_content_and_item_align("l"):
                for index, model in enumerate(models):
                    color = colors[index % len(colors)]
                    row_points = [point for point in snapshot.points if point.model == model]
                    valid = [(score.score, point.effort) for point in row_points if (score := point.scores.get(source)) is not None and score.score is not None and not point.error]
                    best = max(valid, key=lambda value: value[0]) if valid else None
                    height = cell_height
                    if any((score := point.scores.get(source)) is not None and score.coverage is not None and 0 < score.coverage < score.required_tasks for point in row_points):
                        height += 18
                    with HSplit().set_w(content_width).set_padding(inset).set_sep(gap).set_item_align("c").set_bg(RoundRectBg((255, 255, 255, 232), 18, stroke=WHITE)):
                        _model_badge(model, best, name_width, height, color, theme, overview=overview)
                        for effort in efforts:
                            _cell(points.get((model, effort)), source, cell_width, height, snapshot.score_max, color, theme, overview=overview, best=best)
            notes = [f"满分 {snapshot.score_max:g} 分。"]
            if overview:
                notes.append(f"共 {snapshot.total_tasks} 道题。费用为 API 等价费用（美元）；带 * 的均值仅涵盖部分样本。")
            _text("\n".join(notes), content_width, theme, role="muted", size=12, lines=None)
    return await canvas.get_img()
