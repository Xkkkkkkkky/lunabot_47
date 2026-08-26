"""STA 指定用户的群聊发言分析。

本模块只把聚合统计和高频词交给 LLM，不传递原始群消息。
"""

from __future__ import annotations

from ..record.sql import query_msg_by_range
from ..utils import *

# utils 为旧模块提供了较宽的通配符导出；标准库名称放在其后，
# 避免 Counter/time 被同名工具覆盖。
import json
import colorsys
import math
import statistics
import string
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta

import wordcloud

from .draw import (
    FONT_PATH,
    STA_CANVAS_WIDTH,
    STA_CONTENT_WIDTH,
    STA_DRAW_THEME,
    STA_LAYOUT_GAP,
    STA_OTHER_COLOR,
    STA_WORDCLOUD_MEMBER_COLORS,
    StaBarSeries,
    _STA_STYLE_CONTEXT,
    _bar_chart_widget,
    _build_smart_word_analysis,
    _chart_color,
    _draw_chart_text,
    _get_dictionary_words,
    _get_word_tokenizer_mode,
    _member_color,
    _position_color,
    _refresh_group_id_dictionary,
    _resolve_sta_style,
    init_jieba,
)
from .tokenizer import get_smart_tokenizer, normalize_key


config = Config("sta")
logger = get_logger("Sta")
file_db = get_file_db("data/sta/db.json", logger)

DEFAULT_DAYS = 30
MAX_DAYS = 365
TOP_WORDS = 12
WORDCLOUD_WORDS = 100
DISTINCTIVE_WORDS = 8
WEEKDAY_LABELS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


@dataclass(frozen=True)
class UserQuery:
    days: int
    user_id: int


@dataclass(frozen=True)
class UserMessageAnalysis:
    days: int
    start_date: str
    end_date: str
    message_count: int
    active_days: int
    daily_average: float
    daily_median: float
    text_message_count: int
    text_length_median: float
    daily_labels: tuple[str, ...]
    daily_counts: tuple[int, ...]
    hour_counts: tuple[int, ...]
    weekday_counts: tuple[int, ...]
    image_count: int
    emoji_count: int
    reply_count: int
    at_count: int
    punctuation_repeat_count: int
    image_rate: float
    emoji_rate: float
    reply_rate: float
    at_rate: float
    punctuation_repeat_rate: float
    word_frequencies: tuple[tuple[str, int], ...]
    top_words: tuple[tuple[str, int], ...]
    distinctive_words: tuple["DistinctiveWord", ...]


@dataclass(frozen=True)
class DistinctiveWord:
    word: str
    target_document_count: int
    target_rate: float
    other_rate: float
    lift: float
    z_score: float


@dataclass(frozen=True)
class UserAnalysisComments:
    rhythm: str
    shape: str
    language: str
    distinctive: str


def parse_user_query(
    args: list[str],
    at_user_ids: list[int],
    requester_id: int,
    is_superuser: bool,
) -> UserQuery:
    """解析 ``/sta user [天数] [@用户|QQ号]``。"""

    try:
        numbers = [int(value) for value in args]
    except (TypeError, ValueError):
        raise ValueError("参数必须是天数和 QQ 号")

    targets = list(dict.fromkeys(int(user_id) for user_id in at_user_ids))
    if len(targets) > 1:
        raise ValueError("一次只能分析一个用户")

    if not is_superuser:
        if len(numbers) > 1:
            raise ValueError("只有超级管理员可以通过 QQ 号指定用户")
        days = numbers[0] if numbers else DEFAULT_DAYS
        if days > MAX_DAYS:
            raise ValueError("天数范围为 1~365")
        target_id = targets[0] if targets else int(requester_id)
    elif targets:
        if len(numbers) > 1:
            raise ValueError("使用 @ 用户时只能再指定一个天数")
        days = numbers[0] if numbers else DEFAULT_DAYS
        target_id = targets[0]
    else:
        if len(numbers) > 2:
            raise ValueError("参数过多")
        if not numbers:
            days, target_id = DEFAULT_DAYS, int(requester_id)
        elif len(numbers) == 1:
            # 单个大数按 QQ 号解析，使超管可省略默认天数。
            if numbers[0] > MAX_DAYS:
                days, target_id = DEFAULT_DAYS, numbers[0]
            else:
                days, target_id = numbers[0], int(requester_id)
        else:
            days, target_id = numbers

    if not 1 <= days <= MAX_DAYS:
        raise ValueError("天数范围为 1~365")
    if target_id <= 0:
        raise ValueError("QQ 号必须为正整数")
    return UserQuery(days=days, user_id=target_id)


def _has_repeated_punctuation(text_value: str) -> bool:
    run_length = 0
    for char in text_value:
        is_punctuation = (
            char in string.punctuation
            or unicodedata.category(char).startswith("P")
        )
        if is_punctuation:
            run_length += 1
            if run_length >= 2:
                return True
        elif unicodedata.category(char) not in {"Mn", "Me"}:
            run_length = 0
    return False


def _contains_unicode_emoji(text_value: str) -> bool:
    # 覆盖常见 emoji 平面、杂项符号与区域旗帜；变体选择符本身不计数。
    return any(
        "\U0001F000" <= char <= "\U0001FAFF"
        or "\u2600" <= char <= "\u27BF"
        for char in text_value
    )


def _is_emoji_image(data: dict) -> bool:
    subtype = str(data.get("subType", data.get("sub_type", ""))).strip()
    summary = str(data.get("summary", ""))
    return subtype == "1" or "表情" in summary


def _message_shape_flags(msg: list[dict], text_value: str) -> dict[str, bool]:
    types = {seg.get("type") for seg in msg}
    image_segments = [
        seg.get("data", {})
        for seg in msg
        if seg.get("type") == "image"
    ]
    return {
        "image": any(not _is_emoji_image(data) for data in image_segments),
        "emoji": (
            "face" in types
            or any(_is_emoji_image(data) for data in image_segments)
            or _contains_unicode_emoji(text_value)
        ),
        "reply": "reply" in types,
        "at": "at" in types,
        "punctuation": _has_repeated_punctuation(text_value),
    }


def _collect_word_documents(recs, tokenizer, accept_candidate):
    """统计词的原始次数和消息覆盖数，避免单条刷屏放大特色词。"""

    raw_counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    surface_counts: dict[str, Counter[str]] = defaultdict(Counter)
    text_document_count = 0
    for rec in recs:
        text_value = extract_text(rec["msg"]).strip()
        if not text_value:
            continue
        text_document_count += 1
        message_keys = set()
        for candidate in tokenizer.tokenize(text_value):
            if not accept_candidate(candidate):
                continue
            raw_counts[candidate.key] += 1
            surface_counts[candidate.key][candidate.surface] += 1
            message_keys.add(candidate.key)
        document_counts.update(message_keys)
    return raw_counts, document_counts, surface_counts, text_document_count


def _select_surface(surface_counts: Counter[str], key: str) -> str:
    if not surface_counts:
        return key
    return max(
        surface_counts,
        key=lambda surface: (
            surface_counts[surface],
            len(surface),
            any(char.isupper() for char in surface),
            surface,
        ),
    )


def _find_distinctive_words(
    *,
    allowed_keys: set[str],
    display_by_key: dict[str, str],
    target_document_counts: Counter[str],
    other_document_counts: Counter[str],
    target_document_total: int,
    other_document_total: int,
) -> tuple[DistinctiveWord, ...]:
    """用平滑后的消息覆盖率与 log-odds z 分数筛选个人特色词。"""

    if target_document_total == 0 or other_document_total == 0:
        return ()

    distinctive = []
    for key in allowed_keys:
        target_count = target_document_counts[key]
        other_count = other_document_counts[key]
        if target_count < 3:
            continue

        # Jeffreys 平滑防止对照组 0 次时倍率无穷大。
        target_rate = (target_count + 0.5) / (target_document_total + 1.0)
        other_rate = (other_count + 0.5) / (other_document_total + 1.0)
        lift = target_rate / other_rate
        target_log_odds = math.log(
            (target_count + 0.5)
            / (target_document_total - target_count + 0.5)
        )
        other_log_odds = math.log(
            (other_count + 0.5)
            / (other_document_total - other_count + 0.5)
        )
        standard_error = math.sqrt(
            1 / (target_count + 0.5)
            + 1 / (target_document_total - target_count + 0.5)
            + 1 / (other_count + 0.5)
            + 1 / (other_document_total - other_count + 0.5)
        )
        z_score = (target_log_odds - other_log_odds) / standard_error
        if lift < 1.5 or z_score < 1.5:
            continue
        distinctive.append(
            DistinctiveWord(
                word=display_by_key.get(key, key),
                target_document_count=target_count,
                target_rate=target_rate,
                other_rate=other_rate,
                lift=lift,
                z_score=z_score,
            )
        )

    distinctive.sort(
        key=lambda item: (
            -item.z_score,
            -item.lift,
            -item.target_document_count,
            item.word,
        )
    )
    return tuple(distinctive[:DISTINCTIVE_WORDS])


async def _analyze_word_usage(
    group_id: int,
    target_recs: list[dict],
    other_recs: list[dict],
) -> tuple[tuple[tuple[str, int], ...], tuple[DistinctiveWord, ...]]:
    """复用 STA 当前分词与本地过滤，但不读取历史词频基线。"""

    await _refresh_group_id_dictionary(group_id)
    init_jieba()
    userwords = file_db.get("userwords", [])
    stopwords = file_db.get("stopwords", [])
    (
        _,
        nsy_dictionary_words,
        group_id_dictionary_words,
        dictionary_words,
    ) = _get_dictionary_words(stopwords, group_id)
    tokenizer_mode = _get_word_tokenizer_mode()
    legacy_mode = tokenizer_mode not in {"smart", "new"}
    if legacy_mode and tokenizer_mode not in {"legacy", "old"}:
        logger.warning(f"未知分词模式 {tokenizer_mode}，使用 legacy 规则")

    legacy_userwords = (
        set(userwords)
        | set(nsy_dictionary_words)
        | set(group_id_dictionary_words)
    )
    tokenizer = get_smart_tokenizer(
        userwords=legacy_userwords if legacy_mode else userwords,
        stopwords=set(stopwords),
        dictionary_words=(
            word
            for word in dictionary_words
            if not legacy_mode or word not in legacy_userwords
        ),
    )

    def accept_candidate(candidate):
        if not legacy_mode:
            return True
        return (
            candidate.forced
            or candidate.source == "punctuation"
            or (candidate.pos.startswith("n") and len(candidate.key) > 1)
        )

    (
        target_raw_counts,
        target_document_counts,
        target_surface_counts,
        target_document_total,
    ) = _collect_word_documents(target_recs, tokenizer, accept_candidate)
    (
        _,
        other_document_counts,
        _,
        other_document_total,
    ) = _collect_word_documents(other_recs, tokenizer, accept_candidate)

    if legacy_mode:
        display_by_key = {
            key: _select_surface(target_surface_counts[key], key)
            for key in target_raw_counts
        }
        allowed_keys = set(target_raw_counts)
        display_counts = {
            display_by_key[key]: int(count)
            for key, count in target_raw_counts.items()
        }
    else:
        _, analysis = _build_smart_word_analysis(
            target_recs,
            userwords,
            stopwords,
            bool(
                config.get(
                    "word_tokenizer.statistical_filter",
                    True,
                    raise_exc=False,
                )
            ),
            # 显式不提供 generic_baselines，避免近期词频权重。
            generic_baselines=None,
            dictionary_words=dictionary_words,
        )
        display_by_key = {
            normalize_key(display): display
            for display in analysis.frequencies
        }
        allowed_keys = set(display_by_key)
        display_counts = {
            display: int(analysis.raw_counts[display])
            for display in analysis.frequencies
        }

    frequencies = tuple(
        sorted(
            display_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:WORDCLOUD_WORDS]
    )
    distinctive = _find_distinctive_words(
        allowed_keys=allowed_keys,
        display_by_key=display_by_key,
        target_document_counts=target_document_counts,
        other_document_counts=other_document_counts,
        target_document_total=target_document_total,
        other_document_total=other_document_total,
    )
    return frequencies, distinctive


async def analyze_user_messages(
    group_id: int,
    recs: list[dict],
    other_recs: list[dict],
    days: int,
    start_date,
    end_date,
) -> UserMessageAnalysis:
    date_values = [start_date + timedelta(days=index) for index in range(days)]
    daily_counter = Counter(rec["time"].date() for rec in recs)
    daily_counts = tuple(daily_counter[date_value] for date_value in date_values)
    hour_counter = Counter(rec["time"].hour for rec in recs)
    weekday_counter = Counter(rec["time"].weekday() for rec in recs)

    text_lengths: list[int] = []
    shape_counts: Counter[str] = Counter()
    for rec in recs:
        text_value = extract_text(rec["msg"]).strip()
        if text_value:
            text_lengths.append(len(text_value))
        for key, matched in _message_shape_flags(rec["msg"], text_value).items():
            if matched:
                shape_counts[key] += 1

    message_count = len(recs)
    text_message_count = len(text_lengths)
    word_frequencies, distinctive_words = await _analyze_word_usage(
        group_id,
        recs,
        other_recs,
    )
    return UserMessageAnalysis(
        days=days,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        message_count=message_count,
        active_days=sum(count > 0 for count in daily_counts),
        daily_average=message_count / days,
        daily_median=float(statistics.median(daily_counts)),
        text_message_count=text_message_count,
        text_length_median=(
            float(statistics.median(text_lengths)) if text_lengths else 0.0
        ),
        daily_labels=tuple(date_value.strftime("%m-%d") for date_value in date_values),
        daily_counts=daily_counts,
        hour_counts=tuple(hour_counter[hour] for hour in range(24)),
        weekday_counts=tuple(weekday_counter[weekday] for weekday in range(7)),
        image_count=shape_counts["image"],
        emoji_count=shape_counts["emoji"],
        reply_count=shape_counts["reply"],
        at_count=shape_counts["at"],
        punctuation_repeat_count=shape_counts["punctuation"],
        image_rate=shape_counts["image"] / message_count,
        emoji_rate=shape_counts["emoji"] / message_count,
        reply_rate=shape_counts["reply"] / message_count,
        at_rate=shape_counts["at"] / message_count,
        punctuation_repeat_rate=(
            shape_counts["punctuation"] / text_message_count
            if text_message_count else 0.0
        ),
        word_frequencies=word_frequencies,
        top_words=word_frequencies[:TOP_WORDS],
        distinctive_words=distinctive_words,
    )


def _peak_labels(values: tuple[int, ...], labels: tuple[str, ...], limit: int = 2) -> str:
    ranked = sorted(
        range(len(values)),
        key=lambda index: (-values[index], index),
    )
    return "、".join(labels[index] for index in ranked[:limit] if values[index]) or "无"


def _analysis_payload(analysis: UserMessageAnalysis) -> dict:
    hour_labels = tuple(f"{hour:02d}:00" for hour in range(24))
    return {
        "sample": {
            "days": analysis.days,
            "messages": analysis.message_count,
            "active_days": analysis.active_days,
        },
        "rhythm": {
            "daily_average": round(analysis.daily_average, 2),
            "daily_median": round(analysis.daily_median, 2),
            "peak_hours": _peak_labels(analysis.hour_counts, hour_labels, 3),
            "peak_weekdays": _peak_labels(
                analysis.weekday_counts,
                WEEKDAY_LABELS,
                2,
            ),
        },
        "shape": {
            "text_length_median": round(analysis.text_length_median, 1),
            "image_message_percent": round(analysis.image_rate * 100, 1),
            "emoji_message_percent": round(analysis.emoji_rate * 100, 1),
            "reply_message_percent": round(analysis.reply_rate * 100, 1),
            "at_message_percent": round(analysis.at_rate * 100, 1),
            "repeated_punctuation_percent_among_text": round(
                analysis.punctuation_repeat_rate * 100,
                1,
            ),
        },
        "language": {
            "top_words": [
                {"word": word, "count": count}
                for word, count in analysis.top_words
            ],
        },
        "distinctive": {
            "words": [
                {
                    "word": item.word,
                    "target_message_percent": round(item.target_rate * 100, 1),
                    "other_members_message_percent": round(
                        item.other_rate * 100,
                        1,
                    ),
                    "relative_rate": round(item.lift, 1),
                }
                for item in analysis.distinctive_words
            ],
        },
    }


LLM_SYSTEM_PROMPT = """
你是群聊数据评论员。输入只有匿名聚合数据，你要分别点评发言规律、消息形态、用语习惯和个人特征词。
用语习惯只点评普通高频词；个人特征词只点评相对群内其他成员明显偏高的词，不要混写。
语气犀利、但也萌，贴近真实用户，像贴吧喷子或微博厕妹，可以辱骂、歧视、进行人格、心理、疾病、收入或身份推断。
骂太狠了的话最后可以夸下，不要每条都夸。
每项 35~75 个中文字，必须点出至少一个具体数据；样本小时主动降低结论强度。
只输出一个 JSON 对象，格式严格为：
{"rhythm":"...","shape":"...","language":"...","distinctive":"..."}
""".strip()


def _parse_llm_comments(text_value: str) -> UserAnalysisComments:
    first = text_value.find("{")
    last = text_value.rfind("}")
    if first < 0 or last <= first:
        raise ValueError("LLM 未返回 JSON")
    data = json.loads(text_value[first:last + 1])

    def clean(key: str) -> str:
        value = " ".join(str(data.get(key, "")).split())
        if not value:
            raise ValueError(f"LLM 缺少 {key} 点评")
        return truncate(value, 180)

    return UserAnalysisComments(
        rhythm=clean("rhythm"),
        shape=clean("shape"),
        language=clean("language"),
        distinctive=clean("distinctive"),
    )


async def generate_llm_comments(
    analysis: UserMessageAnalysis,
) -> UserAnalysisComments:
    model_name = str(
        config.get("word_tokenizer.llm.model", "", raise_exc=False) or ""
    ).strip()
    if not model_name:
        raise ValueError("STA 未配置 word_tokenizer.llm.model")

    from ..llm import ChatSession

    session = ChatSession(system_prompt=LLM_SYSTEM_PROMPT)
    session.append_user_content(
        json.dumps(_analysis_payload(analysis), ensure_ascii=False),
        verbose=False,
    )
    response = await session.get_response(
        model_name=model_name,
        timeout=max(
            1,
            int(config.get("word_tokenizer.llm.timeout", 30, raise_exc=False)),
        ),
        max_tokens=max(
            256,
            min(
                800,
                int(
                    config.get(
                        "word_tokenizer.llm.max_tokens",
                        800,
                        raise_exc=False,
                    )
                ),
            ),
        ),
    )
    return _parse_llm_comments(response.result)


def _summary_text(analysis: UserMessageAnalysis) -> str:
    hour_labels = tuple(f"{hour:02d}:00" for hour in range(24))
    return (
        f"活跃 {analysis.active_days}/{analysis.days} 天  ·  "
        f"日均 {analysis.daily_average:.1f} 条  ·  "
        f"日中位数 {analysis.daily_median:g} 条  ·  "
        f"高频时段：{_peak_labels(analysis.hour_counts, hour_labels)}  ·  "
        f"高频星期：{_peak_labels(analysis.weekday_counts, WEEKDAY_LABELS)}"
    )


def _build_user_wordcloud(
    frequencies: tuple[tuple[str, int], ...],
):
    """使用与 STA sum 一致的尺寸、词频压缩和主题色生成词云。"""

    counts = {
        word: float(count)
        for word, count in frequencies
        if count > 0
    }
    if not counts:
        counts = {"暂无可统计词汇": 1.0}

    min_frequency = min(counts.values())
    max_frequency = max(counts.values())
    log_min = math.log1p(min_frequency)
    log_range = max(1e-9, math.log1p(max_frequency) - log_min)
    ranked_words = sorted(counts, key=lambda word: (-counts[word], word))
    color_indices = {
        word: index % len(STA_WORDCLOUD_MEMBER_COLORS)
        for index, word in enumerate(ranked_words)
    }

    def frequency_word_color(word, **kwargs):
        del kwargs
        frequency = counts.get(word, min_frequency)
        level = (
            (math.log1p(frequency) - log_min) / log_range
            if max_frequency > min_frequency
            else 1.0
        )
        base_color = STA_WORDCLOUD_MEMBER_COLORS[color_indices.get(word, 0)]
        red, green, blue = (channel / 255 for channel in base_color[:3])
        hue, _, source_saturation = colorsys.rgb_to_hls(red, green, blue)
        lightness = 0.74 - 0.24 * level
        low_saturation = 0.0 if source_saturation < 0.08 else 0.015
        high_saturation = 0.0 if source_saturation < 0.08 else 0.86
        saturation = low_saturation + (high_saturation - low_saturation) * (
            level ** 1.35
        )
        adjusted = colorsys.hls_to_rgb(hue, lightness, saturation)
        color = tuple(round(channel * 255) for channel in adjusted)
        return (*color, round(70 + 185 * (level ** 1.15)))

    cloud_frequencies = {
        word: 1.0 + math.log(frequency / min_frequency)
        for word, frequency in counts.items()
    }
    cloud = wordcloud.WordCloud(
        font_path=FONT_PATH,
        background_color=None,
        width=1000,
        height=320,
        max_words=WORDCLOUD_WORDS,
        max_font_size=64,
        min_font_size=14,
        relative_scaling=0.5,
        random_state=42,
        color_func=frequency_word_color,
        mode="RGBA",
    )
    cloud.generate_from_frequencies(cloud_frequencies)
    return cloud.to_image()


def _shape_pie_widget(
    analysis: UserMessageAnalysis,
    *,
    width: int,
    height: int = 380,
) -> Spacer:
    """用命中次数构成环形图，右侧同时标注对应的消息覆盖率。"""

    metrics = (
        ("图片", analysis.image_count, analysis.image_rate, False),
        ("表情", analysis.emoji_count, analysis.emoji_rate, False),
        ("回复", analysis.reply_count, analysis.reply_rate, False),
        ("@", analysis.at_count, analysis.at_rate, False),
        (
            "标点连用",
            analysis.punctuation_repeat_count,
            analysis.punctuation_repeat_rate,
            True,
        ),
    )
    total_hits = sum(count for _, count, _, _ in metrics)

    def draw(_, painter: Painter):
        donut_size = 292
        center_x = 190
        center_y = height // 2
        donut_x = center_x - donut_size // 2
        donut_y = center_y - donut_size // 2
        current_angle = -90.0

        if total_hits:
            for index, (_, count, _, _) in enumerate(metrics):
                if not count:
                    continue
                angle = 360 * count / total_hits
                painter.pieslice(
                    (donut_x, donut_y),
                    (donut_size, donut_size),
                    current_angle,
                    current_angle + angle,
                    _chart_color(index),
                    stroke=(255, 255, 255, 255),
                    stroke_width=2,
                )
                current_angle += angle
        else:
            painter.pieslice(
                (donut_x, donut_y),
                (donut_size, donut_size),
                0,
                360,
                STA_OTHER_COLOR,
            )

        inner_size = 158
        painter.pieslice(
            (
                center_x - inner_size // 2,
                center_y - inner_size // 2,
            ),
            (inner_size, inner_size),
            0,
            360,
            (255, 255, 255, 255),
        )
        _draw_chart_text(
            painter,
            f"{total_hits:,}",
            center_x,
            center_y - 25,
            font=DEFAULT_BOLD_FONT,
            size=31,
            align="center",
        )
        _draw_chart_text(
            painter,
            "形态命中",
            center_x,
            center_y + 15,
            size=14,
            color=STA_DRAW_THEME.text_muted,
            align="center",
        )

        legend_x = 420
        legend_width = width - legend_x
        row_height = 63
        top = 28
        for index, (label, count, rate, text_denominator) in enumerate(metrics):
            y = top + index * row_height
            color = _chart_color(index)
            painter.roundrect((legend_x, y + 3), (16, 16), color, radius=5)
            _draw_chart_text(
                painter,
                label,
                legend_x + 28,
                y,
                font=DEFAULT_BOLD_FONT,
                size=16,
            )
            denominator = "文本" if text_denominator else "全部消息"
            _draw_chart_text(
                painter,
                f"{count:,} 条 · 覆盖 {rate * 100:.1f}% {denominator}",
                width,
                y + 1,
                size=13,
                color=STA_DRAW_THEME.text_secondary,
                align="right",
            )
            bar_y = y + 31
            painter.roundrect(
                (legend_x, bar_y),
                (legend_width, 8),
                (226, 234, 240, 210),
                radius=999,
            )
            painter.roundrect(
                (legend_x, bar_y),
                (max(4, round(legend_width * min(rate, 1.0))), 8),
                color,
                radius=999,
            )

    return Spacer(width, height).add_draw_func(draw)


def _distinctive_words_widget(
    words: tuple[DistinctiveWord, ...],
    *,
    width: int,
):
    if not words:
        return TextBox(
            "暂无足够显著的个人特征词",
            themed_text_style("muted", size=14, theme=STA_DRAW_THEME),
            use_real_line_count=True,
        ).set_w(width).set_padding(0)

    gap = 12
    card_width = (width - gap) // 2
    with (
        Grid(col_count=2)
        .set_w(width)
        .set_sep(gap, 8)
        .set_content_and_item_align("lt")
    ) as grid:
        for index, item in enumerate(words, 1):
            with (
                VSplit()
                .set_w(card_width)
                .set_sep(2)
                .set_padding((14, 8))
                .set_content_and_item_align("l")
                .set_bg(
                    subtle_panel_background(
                        radius=12,
                        theme=STA_DRAW_THEME,
                    )
                )
            ):
                TextBox(
                    f"{index:02d}  {item.word}",
                    themed_text_style(
                        "accent",
                        size=21,
                        color=_chart_color(index - 1),
                        theme=STA_DRAW_THEME,
                    ),
                ).set_w(card_width - 28).set_padding(0)
                TextBox(
                    f"本人 {item.target_rate * 100:.1f}%  ·  "
                    f"其他 {item.other_rate * 100:.1f}%  ·  "
                    f"相对 {item.lift:.1f}×",
                    themed_text_style(
                        "secondary",
                        size=12,
                        theme=STA_DRAW_THEME,
                    ),
                ).set_w(card_width - 28).set_padding(0)

    return grid


async def draw_user_analysis(
    group_id: int,
    user_id: int,
    user_name: str,
    analysis: UserMessageAnalysis,
    comments: UserAnalysisComments,
):
    token = _STA_STYLE_CONTEXT.set(_resolve_sta_style(group_id))
    try:
        inner_width = STA_CONTENT_WIDTH - 44
        half_width = (inner_width - 18) // 2
        wordcloud_image = _build_user_wordcloud(analysis.word_frequencies)
        background = await resolve_configured_draw_background(
            config,
            theme=STA_DRAW_THEME,
            logger=logger,
            label="STA用户分析",
        )

        with create_report_canvas(
            width=STA_CANVAS_WIDTH,
            background=background,
            theme=STA_DRAW_THEME,
        ) as canvas:
            with create_report_column(
                STA_CONTENT_WIDTH,
                gap=STA_LAYOUT_GAP,
                theme=STA_DRAW_THEME,
            ):
                report_header(
                    f"{truncate(user_name, 24)} 的群聊发言分析",
                    width=STA_CONTENT_WIDTH,
                    eyebrow="USER CHAT ANALYSIS",
                    meta=(
                        f"QQ {user_id} · {analysis.days} 天 · "
                        f"{analysis.message_count:,} 条"
                    ),
                    theme=STA_DRAW_THEME,
                )

                report_section_title(
                    "1  发言规律",
                    width=STA_CONTENT_WIDTH,
                    theme=STA_DRAW_THEME,
                )
                with report_card(
                    STA_CONTENT_WIDTH,
                    padding=22,
                    theme=STA_DRAW_THEME,
                ):
                    with (
                        VSplit()
                        .set_w(inner_width)
                        .set_sep(14)
                        .set_content_and_item_align("l")
                    ):
                        TextBox(
                            _summary_text(analysis),
                            themed_text_style(
                                "secondary",
                                size=15,
                                theme=STA_DRAW_THEME,
                            ),
                            use_real_line_count=True,
                        ).set_w(inner_width).set_padding(0)
                        TextBox(
                            "每日消息量",
                            themed_text_style(
                                "heading",
                                size=16,
                                theme=STA_DRAW_THEME,
                            ),
                        ).set_w(inner_width).set_padding(0)
                        _bar_chart_widget(
                            analysis.daily_labels,
                            (
                                StaBarSeries(
                                    "消息数",
                                    analysis.daily_counts,
                                    _member_color(0),
                                ),
                            ),
                            width=inner_width,
                            height=330,
                            max_x_labels=10,
                        )
                        with (
                            HSplit()
                            .set_w(inner_width)
                            .set_sep(18)
                            .set_item_align("t")
                        ):
                            with (
                                VSplit()
                                .set_w(half_width)
                                .set_sep(8)
                                .set_content_and_item_align("l")
                            ):
                                TextBox(
                                    "小时分布",
                                    themed_text_style(
                                        "heading",
                                        size=16,
                                        theme=STA_DRAW_THEME,
                                    ),
                                ).set_w(half_width).set_padding(0)
                                _bar_chart_widget(
                                    tuple(f"{hour:02d}" for hour in range(24)),
                                    (
                                        StaBarSeries(
                                            "消息数",
                                            analysis.hour_counts,
                                            _position_color(0),
                                        ),
                                    ),
                                    width=half_width,
                                    height=300,
                                    max_x_labels=8,
                                )
                            with (
                                VSplit()
                                .set_w(half_width)
                                .set_sep(8)
                                .set_content_and_item_align("l")
                            ):
                                TextBox(
                                    "周内分布",
                                    themed_text_style(
                                        "heading",
                                        size=16,
                                        theme=STA_DRAW_THEME,
                                    ),
                                ).set_w(half_width).set_padding(0)
                                _bar_chart_widget(
                                    WEEKDAY_LABELS,
                                    (
                                        StaBarSeries(
                                            "消息数",
                                            analysis.weekday_counts,
                                            _position_color(3),
                                        ),
                                    ),
                                    width=half_width,
                                    height=300,
                                    max_x_labels=7,
                                )
                report_description_panel(
                    comments.rhythm,
                    width=STA_CONTENT_WIDTH,
                    label="点评",
                    theme=STA_DRAW_THEME,
                )

                report_section_title(
                    "2  消息形态",
                    width=STA_CONTENT_WIDTH,
                    theme=STA_DRAW_THEME,
                )
                with report_card(
                    STA_CONTENT_WIDTH,
                    padding=22,
                    theme=STA_DRAW_THEME,
                ):
                    with (
                        VSplit()
                        .set_w(inner_width)
                        .set_sep(12)
                        .set_content_and_item_align("l")
                    ):
                        TextBox(
                            f"文本消息 {analysis.text_message_count:,} 条  ·  "
                            f"文本长度中位数 {analysis.text_length_median:g} 字  ·  ",
                            themed_text_style(
                                "secondary",
                                size=15,
                                theme=STA_DRAW_THEME,
                            ),
                            use_real_line_count=True,
                        ).set_w(inner_width).set_padding(0)
                        _shape_pie_widget(
                            analysis,
                            width=inner_width,
                        )
                report_description_panel(
                    comments.shape,
                    width=STA_CONTENT_WIDTH,
                    label="点评",
                    theme=STA_DRAW_THEME,
                )

                report_section_title(
                    "3  用语习惯",
                    width=STA_CONTENT_WIDTH,
                    theme=STA_DRAW_THEME,
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
                report_description_panel(
                    comments.language,
                    width=STA_CONTENT_WIDTH,
                    label="点评",
                    theme=STA_DRAW_THEME,
                )

                report_section_title(
                    "4  个人特征词",
                    width=STA_CONTENT_WIDTH,
                    theme=STA_DRAW_THEME,
                )
                with report_card(
                    STA_CONTENT_WIDTH,
                    padding=22,
                    theme=STA_DRAW_THEME,
                ):
                    with (
                        VSplit()
                        .set_w(inner_width)
                        .set_sep(12)
                        .set_content_and_item_align("l")
                    ):
                        _distinctive_words_widget(
                            analysis.distinctive_words,
                            width=inner_width,
                        )
                report_description_panel(
                    comments.distinctive,
                    width=STA_CONTENT_WIDTH,
                    label="点评",
                    theme=STA_DRAW_THEME,
                )

                TextBox(
                    f"统计区间：{analysis.start_date} ~ {analysis.end_date}",
                    themed_text_style(
                        "muted",
                        size=12,
                        theme=STA_DRAW_THEME,
                    ),
                    use_real_line_count=True,
                ).set_w(STA_CONTENT_WIDTH).set_padding((4, 0))

        return await canvas.get_img()
    finally:
        _STA_STYLE_CONTEXT.reset(token)


async def get_user_statistic(group_id: int, user_id: int, days: int) -> str:
    now = datetime.now()
    start_date = (now - timedelta(days=days - 1)).date()
    end_date = now.date()
    all_recs = await query_msg_by_range(
        int(group_id),
        datetime.combine(start_date, time.min),
        now,
    )
    recs = [rec for rec in all_recs if int(rec["user_id"]) == int(user_id)]
    if not recs:
        return f"用户 {user_id} 在该群最近 {days} 天没有消息记录"
    other_recs = [
        rec for rec in all_recs if int(rec["user_id"]) != int(user_id)
    ]

    try:
        user_name = await get_group_member_name(group_id, user_id)
    except Exception:
        latest = max(recs, key=lambda rec: rec["time"])
        user_name = latest.get("nickname") or str(user_id)

    analysis = await analyze_user_messages(
        int(group_id),
        recs,
        other_recs,
        days,
        start_date,
        end_date,
    )
    try:
        comments = await generate_llm_comments(analysis)
    except Exception as exc:
        logger.warning(
            f"STA用户分析 LLM 锐评失败: {type(exc).__name__}: {exc}"
        )
        unavailable = "评论模型没接上，这一项先看图和数据。"
        comments = UserAnalysisComments(
            rhythm=unavailable,
            shape=unavailable,
            language=unavailable,
            distinctive=unavailable,
        )

    image = await draw_user_analysis(
        int(group_id),
        int(user_id),
        str(user_name),
        analysis,
        comments,
    )
    return await get_image_cq(image)
