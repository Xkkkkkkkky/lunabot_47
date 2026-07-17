from __future__ import annotations

import io
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_matplotlib_cache_dir = Path(tempfile.gettempdir()) / "gbfr_act_cache" / "matplotlib"
_matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)

# Keep the cache override scoped to Matplotlib. Setting XDG_CACHE_HOME here also
# changes Playwright/htmlrender browser lookup paths for the whole bot process.
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache_dir))

import matplotlib

matplotlib.use("Agg", force=True)
from matplotlib.font_manager import FontProperties
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from .models import (
    ActorStats,
    BattleRecord,
    action_display_name,
    format_duration,
    format_number,
)


PALETTE = [
    "#55C667",
    "#4D8CFF",
    "#FFB347",
    "#F26D85",
    "#9B7BFF",
    "#41C7C7",
]


@dataclass
class RenderConfig:
    width: int = 1280
    background: str = ""
    background_color: str = "#182029"
    chart_bucket_seconds: int = 2
    top_actor_count: int = 4
    top_action_count: int = 5
    show_username: bool = True
    font_path: str = ""

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "RenderConfig":
        cfg = cfg or {}
        return cls(
            width=max(900, int(cfg.get("width", 1280) or 1280)),
            background=str(cfg.get("background", "") or ""),
            background_color=str(cfg.get("background_color", "#182029") or "#182029"),
            chart_bucket_seconds=max(1, int(cfg.get("chart_bucket_seconds", 2) or 2)),
            top_actor_count=max(1, int(cfg.get("top_actor_count", 4) or 4)),
            top_action_count=max(1, int(cfg.get("top_action_count", 5) or 5)),
            show_username=bool(cfg.get("show_username", True)),
            font_path=str(cfg.get("font_path", "") or ""),
        )


def render_battle_report(record: BattleRecord, cfg: RenderConfig | None = None) -> Image.Image:
    cfg = cfg or RenderConfig()
    global _font_path_override
    _font_path_override = cfg.font_path
    chart = _draw_chart(record, cfg)
    actors = record.actor_rank(cfg.top_actor_count)
    action_block_h = _action_block_height(actors, cfg.top_action_count)
    height = 210 + 58 + max(220, 58 + len(actors) * 44) + 36 + chart.height + 36 + action_block_h + 50
    width = cfg.width
    img = _make_background(width, height, cfg)
    draw = ImageDraw.Draw(img)

    margin = 42
    y = 36
    title_font = _font(40, bold=True)
    normal_font = _font(22)
    small_font = _font(18)
    table_font = _font(20)
    muted = (205, 215, 225, 255)
    text = (247, 250, 252, 255)

    draw.text((margin, y), "GBFR 战斗输出报告", fill=text, font=title_font)
    status = "进行中" if not record.archived else f"已结束/{record.finish_reason or 'recorded'}"
    draw.text((margin, y + 54), status, fill=(124, 210, 255, 255), font=normal_font)
    right_lines = [
        record.start_time.strftime("%Y-%m-%d %H:%M:%S"),
        f"持续 {format_duration(record.duration_seconds)}",
    ]
    for i, line in enumerate(right_lines):
        tw = _text_width(draw, line, normal_font)
        draw.text((width - margin - tw, y + i * 34), line, fill=muted, font=normal_font)

    y += 118
    summary = [
        ("总伤害", format_number(record.total_damage)),
        ("全队 DPS", format_number(record.dps)),
        ("命中数", format_number(record.hit)),
        ("记录事件", format_number(len(record.events))),
    ]
    card_w = (width - margin * 2 - 18 * (len(summary) - 1)) // len(summary)
    for i, (label, value) in enumerate(summary):
        x = margin + i * (card_w + 18)
        _rounded_rect(draw, (x, y, x + card_w, y + 86), fill=(19, 31, 42, 210), outline=(255, 255, 255, 32))
        draw.text((x + 18, y + 14), label, fill=muted, font=small_font)
        draw.text((x + 18, y + 42), value, fill=text, font=_font(28, bold=True))

    y += 116
    y = _draw_actor_table(draw, record, actors, margin, y, width - margin * 2, cfg, table_font)
    y += 24
    img.alpha_composite(chart, (margin, y))
    y += chart.height + 34
    _draw_action_blocks(draw, actors, margin, y, width - margin * 2, cfg)

    footer = "数据来自 GBFR-ACT WebSocket，本图由 lunabot 读取本地 log 生成"
    draw.text((margin, height - 36), footer, fill=(166, 178, 190, 255), font=small_font)
    return img.convert("RGB")


def _draw_actor_table(
    draw: ImageDraw.ImageDraw,
    record: BattleRecord,
    actors: list[ActorStats],
    x: int,
    y: int,
    w: int,
    cfg: RenderConfig,
    font: ImageFont.ImageFont,
) -> int:
    header_font = _font(22, bold=True)
    small_font = _font(18)
    row_h = 44
    h = 58 + max(1, len(actors)) * row_h
    _rounded_rect(draw, (x, y, x + w, y + h), fill=(16, 24, 34, 220), outline=(255, 255, 255, 36))
    draw.text((x + 20, y + 16), "队伍输出", fill=(248, 250, 252, 255), font=header_font)
    y0 = y + 56
    cols = [
        ("#", 36),
        ("角色", 320),
        ("伤害", 170),
        ("占比", 90),
        ("DPS", 120),
        ("命中", 82),
        ("倒地", 70),
        ("最高动作", w - 36 - 320 - 170 - 90 - 120 - 82 - 70 - 42),
    ]
    cx = x + 18
    for label, cw in cols:
        draw.text((cx, y0 - 30), label, fill=(160, 172, 186, 255), font=small_font)
        cx += cw

    if not actors:
        draw.text((x + 20, y0 + 8), "暂无伤害记录", fill=(220, 226, 232, 255), font=font)
        return y + h

    for i, actor in enumerate(actors):
        ry = y0 + i * row_h
        if i % 2 == 0:
            _rounded_rect(draw, (x + 12, ry - 2, x + w - 12, ry + row_h - 4), fill=(255, 255, 255, 14))
        top_action = actor.top_actions(1)
        top_action_text = "-"
        if top_action:
            action = top_action[0]
            top_action_text = f"{action.name} {format_number(action.damage)}"
        share = actor.damage / record.total_damage * 100 if record.total_damage else 0
        row = [
            str(i + 1),
            actor.display_name(cfg.show_username),
            format_number(actor.damage),
            f"{share:.1f}%",
            format_number(actor.damage / record.duration_seconds),
            format_number(actor.hit),
            format_number(actor.death_cnt),
            top_action_text,
        ]
        cx = x + 18
        for value, (_, cw) in zip(row, cols):
            fill = (248, 250, 252, 255) if i < 3 else (220, 226, 232, 255)
            draw.text((cx, ry + 8), _clip_text(draw, value, font, cw - 10), fill=fill, font=font)
            cx += cw
    return y + h


def _draw_action_blocks(
    draw: ImageDraw.ImageDraw,
    actors: list[ActorStats],
    x: int,
    y: int,
    w: int,
    cfg: RenderConfig,
) -> None:
    title_font = _font(22, bold=True)
    font = _font(18)
    small_font = _font(16)
    draw.text((x, y), "动作构成", fill=(248, 250, 252, 255), font=title_font)
    y += 36
    cols = 2 if w >= 980 else 1
    gap = 18
    card_w = (w - gap * (cols - 1)) // cols
    card_h = 48 + cfg.top_action_count * 30
    for idx, actor in enumerate(actors):
        cx = x + (idx % cols) * (card_w + gap)
        cy = y + (idx // cols) * (card_h + gap)
        _rounded_rect(draw, (cx, cy, cx + card_w, cy + card_h), fill=(16, 24, 34, 220), outline=(255, 255, 255, 32))
        draw.text((cx + 16, cy + 14), _clip_text(draw, actor.display_name(cfg.show_username), title_font, card_w - 32), fill=(245, 248, 251, 255), font=title_font)
        actions = actor.top_actions(cfg.top_action_count)
        if not actions:
            draw.text((cx + 16, cy + 50), "无动作记录", fill=(190, 200, 210, 255), font=font)
            continue
        for i, action in enumerate(actions):
            ry = cy + 50 + i * 30
            share = action.damage / actor.damage * 100 if actor.damage else 0
            name = _clip_text(draw, action_display_name(action.action_id), font, card_w - 250)
            draw.text((cx + 16, ry), name, fill=(225, 232, 238, 255), font=font)
            right = f"{format_number(action.damage)} / {action.hit}hit / {share:.1f}%"
            tw = _text_width(draw, right, small_font)
            draw.text((cx + card_w - 16 - tw, ry + 2), right, fill=(166, 178, 190, 255), font=small_font)


def _action_block_height(actors: list[ActorStats], top_action_count: int) -> int:
    if not actors:
        return 70
    cols = 2
    card_h = 48 + top_action_count * 30
    rows = math.ceil(len(actors) / cols)
    return 36 + rows * card_h + max(0, rows - 1) * 18


def _draw_chart(record: BattleRecord, cfg: RenderConfig) -> Image.Image:
    actors = record.actor_rank(cfg.top_actor_count)
    width = cfg.width - 84
    height = 520
    if not actors or not record.damage_points:
        img = Image.new("RGBA", (width, 220), (16, 24, 34, 220))
        draw = ImageDraw.Draw(img)
        _rounded_rect(draw, (0, 0, width, 220), fill=(16, 24, 34, 220), outline=(255, 255, 255, 32))
        draw.text((24, 88), "暂无曲线数据", fill=(230, 236, 242, 255), font=_font(28, bold=True))
        return img

    times, cumulative, rolling = _build_series(record, actors, cfg.chart_bucket_seconds)
    fig, axes = plt.subplots(2, 1, figsize=(width / 120, height / 120), dpi=120, sharex=True)
    fig.patch.set_alpha(0)
    for ax in axes:
        ax.set_facecolor((0.05, 0.08, 0.11, 0.82))
        ax.grid(color="#FFFFFF", alpha=0.10, linewidth=0.8)
        ax.tick_params(colors="#D7DEE7", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#536170")
            spine.set_alpha(0.4)

    for idx, actor in enumerate(actors):
        color = PALETTE[idx % len(PALETTE)]
        label = actor.display_name(cfg.show_username)
        axes[0].plot(times, cumulative[actor.key], label=label, color=color, linewidth=2.2)
        axes[1].plot(times, rolling[actor.key], label=label, color=color, linewidth=1.8)

    axes[0].set_ylabel("Total Damage", color="#D7DEE7")
    axes[1].set_ylabel("60s DPS", color="#D7DEE7")
    axes[1].set_xlabel("Time (s)", color="#D7DEE7")
    legend_kwargs = {"loc": "upper left", "fontsize": 8, "framealpha": 0.25}
    if cfg.font_path and Path(cfg.font_path).is_file():
        legend_kwargs["prop"] = FontProperties(fname=cfg.font_path, size=8)
    axes[0].legend(**legend_kwargs)
    fig.tight_layout(pad=1.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    plt.close(fig)
    buf.seek(0)
    chart = Image.open(buf).convert("RGBA")
    card = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)
    _rounded_rect(draw, (0, 0, width, height), fill=(16, 24, 34, 220), outline=(255, 255, 255, 32))
    card.alpha_composite(chart, (0, 0))
    return card


def _build_series(record: BattleRecord, actors: list[ActorStats], bucket_seconds: int):
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
        for t in times:
            while right < len(points) and points[right][0] <= t:
                total += points[right][1]
                window_damage += points[right][1]
                right += 1
            while left < right and points[left][0] < t - 60:
                window_damage -= points[left][1]
                left += 1
            elapsed = max(1, min(60, t))
            cumulative[key].append(total)
            rolling[key].append(window_damage / elapsed)
    return times, cumulative, rolling


def _make_background(width: int, height: int, cfg: RenderConfig) -> Image.Image:
    bg_path = Path(cfg.background).expanduser() if cfg.background else None
    if bg_path and bg_path.is_file():
        bg = Image.open(bg_path).convert("RGB")
        scale = max(width / bg.width, height / bg.height)
        resized = bg.resize((int(bg.width * scale), int(bg.height * scale)), Image.Resampling.LANCZOS)
        left = (resized.width - width) // 2
        top = (resized.height - height) // 2
        bg = resized.crop((left, top, left + width, top + height)).convert("RGBA")
    else:
        bg = Image.new("RGBA", (width, height), _hex_to_rgba(cfg.background_color, 255))
        px = bg.load()
        for yy in range(height):
            ratio = yy / max(1, height - 1)
            for xx in range(width):
                tint = int(28 + 26 * ratio + 12 * (xx / max(1, width - 1)))
                px[xx, yy] = (18 + tint // 4, 28 + tint // 3, 38 + tint // 2, 255)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 82))
    bg.alpha_composite(overlay)
    return bg


def _rounded_rect(draw: ImageDraw.ImageDraw, box, fill, outline=None, radius: int = 8) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=1)


def _hex_to_rgba(value: str, alpha: int) -> tuple[int, int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) != 6:
        return (24, 32, 41, alpha)
    try:
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16), alpha)
    except ValueError:
        return (24, 32, 41, alpha)


_font_path_override = ""


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        _font_path_override,
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _clip_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> str:
    if _text_width(draw, text, font) <= max_width:
        return text
    suffix = "..."
    while text and _text_width(draw, text + suffix, font) > max_width:
        text = text[:-1]
    return text + suffix if text else suffix
