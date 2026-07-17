"""STA 词云使用的中英文混合关键词提取器。

这个模块只负责从消息文本中产生稳定、可计数的关键词，不负责绘图。英文、
拼音缩写和技术标识符由规则提取，中文继续使用独立的 jieba 分词器。可选的
统计过滤不会把“在群里经常出现”等同于“泛用词”，而是结合消息覆盖、用户
分散度和一份保守的泛用词先验进行降权。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
import re
import statistics
import unicodedata

import jieba
import jieba.posseg as pseg


# URL 和邮箱必须先移除，否则开启英文词提取后，域名片段会污染词云。
URL_RE = re.compile(r"(?i)(?:https?://|www\.)\S+")
EMAIL_RE = re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+")

# 支持普通英文、拼音缩写、snake_case、Node.js、v2.1.0、C++ 和 C#。
# 边界只检查 ASCII 标识符字符，确保“用ChatGPT写代码”也能提取 ChatGPT。
LATIN_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:"
    r"[A-Za-z][A-Za-z0-9]*(?:\+\+|#)"
    r"|[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)+"
    r"|[A-Za-z][A-Za-z0-9_]*"
    r")(?![A-Za-z0-9_])"
)

CJK_RE = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


# 这里只放几乎没有主题信息的英文功能词。管理员用户词可以显式覆盖该集合。
DEFAULT_ENGLISH_STOPWORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "if", "then", "else",
        "to", "of", "in", "on", "at", "for", "from", "with", "without",
        "by", "as", "into", "about", "than", "is", "am", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does",
        "did", "can", "could", "would", "should", "will", "may", "might",
        "must", "this", "that", "these", "those", "it", "its", "i", "me",
        "my", "you", "your", "he", "his", "she", "her", "we", "our",
        "they", "their", "not", "no", "yes", "very", "just", "so", "too",
        "also", "get", "got", "http", "https", "www", "com",
    }
)


# 泛用词只在统计过滤开启时降权或过滤，不会被当成硬停用词。集合刻意保持
# 保守：“妈咪”等群聊领域词不在这里，也不会仅因为高频而被过滤。
GENERIC_WORDS = frozenset(
    {
        "感觉", "觉得", "认为", "知道", "看看", "看起来", "好像", "可能",
        "应该", "需要", "可以", "不能", "不会", "还有", "有点", "比较",
        "非常", "特别", "真的", "其实", "确实", "基本", "一般", "现在",
        "今天", "昨天", "明天", "时候", "东西", "事情", "问题", "情况",
        "这样", "那样", "这个", "那个", "一个", "什么", "怎么", "为什么",
        "没有", "不是", "就是", "还是", "已经", "然后", "但是", "因为",
        "所以", "而且", "不过", "出来", "起来", "进行", "使用", "大家",
        "别人", "一次", "一点", "的话", "意思", "喜欢", "人", "太",
    }
)

ALLOWED_POS_PREFIXES = ("n", "v", "a")
# j=简称、z=状态词、x=未知词。后三类只接收多字中文，补足网络词和新造词。
ALLOWED_POS = frozenset({"i", "l", "j", "z", "x"})


def normalize_text(text: str) -> str:
    """统一全半角等 Unicode 变体，但保留用于展示的英文字母大小写。"""

    return unicodedata.normalize("NFKC", str(text or ""))


def normalize_key(word: str) -> str:
    """生成计数键；只要含 ASCII 字母就进行大小写折叠。"""

    word = normalize_text(word).strip()
    if any("A" <= ch <= "Z" or "a" <= ch <= "z" for ch in word):
        return word.casefold()
    return word


@dataclass(frozen=True)
class MessageSample:
    """一条参与关键词统计的纯文本消息。"""

    text: str
    user_id: Hashable


@dataclass(frozen=True)
class TokenCandidate:
    """分词器产生的候选词及其来源信息。"""

    key: str
    surface: str
    pos: str
    source: str
    forced: bool = False
    acronym: bool = False


@dataclass
class KeywordCandidateStats:
    """LLM 可复核的完整候选；软过滤词也保留在这里。"""

    weight: float
    raw_count: int
    user_counts: dict[Hashable, int]
    locally_kept: bool
    filter_reason: str | None
    generic: bool
    forced: bool
    daily_burst: bool
    latin: bool


@dataclass
class KeywordAnalysis:
    """绘图所需的加权词频、原始词频和用户贡献计数。"""

    frequencies: dict[str, float]
    raw_counts: dict[str, int]
    user_counts: dict[str, dict[Hashable, int]]
    # 相对历史基线异常升温的泛用词。它们本日应当展示，LLM 也不能删除。
    burst_words: frozenset[str] = field(default_factory=frozenset)
    # 包含本地保留和软过滤的完整候选，供 LLM 双向筛选和补词。
    candidate_pool: dict[str, KeywordCandidateStats] = field(default_factory=dict)


@dataclass(frozen=True)
class GenericWordBaseline:
    """一个泛用词在历史自然日中的消息覆盖率基线。"""

    median_rate: float
    mad_rate: float
    history_days: int


class SmartTokenizer:
    """独立于 jieba 全局实例的 STA 中英文混合分词器。"""

    def __init__(
        self,
        userwords: Sequence[str] = (),
        stopwords: Sequence[str] = (),
        dictionary_words: Sequence[str] = (),
    ):
        normalized_stopwords = {
            normalize_key(word) for word in stopwords if normalize_key(word)
        }
        self.stopwords = normalized_stopwords

        self.userword_surfaces: dict[str, str] = {}
        for word in userwords:
            surface = normalize_text(word).strip()
            key = normalize_key(surface)
            if key and key not in self.stopwords:
                self.userword_surfaces[key] = surface

        self._jieba = jieba.Tokenizer()
        for surface in self.userword_surfaces.values():
            self._jieba.add_word(surface, tag="n")
        for word in dictionary_words:
            surface = normalize_text(word).strip()
            key = normalize_key(surface)
            if key and key not in self.stopwords and key not in self.userword_surfaces:
                self._jieba.add_word(surface, tag="n")
        self._posseg = pseg.POSTokenizer(self._jieba)

    def _is_stopped(self, key: str, forced: bool) -> bool:
        if key in self.stopwords:
            return True
        return not forced and key in DEFAULT_ENGLISH_STOPWORDS

    def tokenize(self, text: str) -> list[TokenCandidate]:
        """切分单条消息，返回中文内容词和英文/拼音/技术词候选。"""

        text = normalize_text(text)
        text = URL_RE.sub(" ", text)
        text = EMAIL_RE.sub(" ", text)
        candidates: list[TokenCandidate] = []

        for match in LATIN_TOKEN_RE.finditer(text):
            surface = match.group(0)
            key = normalize_key(surface)
            forced = key in self.userword_surfaces
            if self._is_stopped(key, forced):
                continue

            alnum_length = sum(ch.isalnum() for ch in surface)
            special_identifier = surface.endswith(("++", "#"))
            if not forced and alnum_length < 2 and not special_identifier:
                continue

            letters = "".join(ch for ch in surface if ch.isalpha())
            candidates.append(
                TokenCandidate(
                    key=key,
                    surface=self.userword_surfaces.get(key, surface),
                    pos="eng",
                    source="latin",
                    forced=forced,
                    acronym=len(letters) >= 2 and letters.isupper(),
                )
            )

        # 英文已单独提取，从中文输入中挖掉，防止 jieba 重复计数。
        chinese_text = LATIN_TOKEN_RE.sub(" ", text)
        for pair in self._posseg.cut(chinese_text):
            surface = normalize_text(pair.word).strip()
            key = normalize_key(surface)
            if not key:
                continue

            forced = key in self.userword_surfaces
            if key in self.stopwords:
                continue
            if not forced:
                allowed_pos = pair.flag.startswith(ALLOWED_POS_PREFIXES) or pair.flag in ALLOWED_POS
                if not allowed_pos:
                    continue
                if pair.flag in {"j", "z", "x"} and not CJK_RE.fullmatch(key):
                    continue
                # 单字动词/形容词噪声很大；单字名词则交给统计层判断是否稳定出现。
                if len(key) == 1 and not pair.flag.startswith("n"):
                    continue

            candidates.append(
                TokenCandidate(
                    key=key,
                    surface=self.userword_surfaces.get(key, surface),
                    pos=pair.flag,
                    source="chinese",
                    forced=forced,
                )
            )

        return candidates


_cached_tokenizer: SmartTokenizer | None = None
_cached_signature: tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
] | None = None


def reset_smart_tokenizer() -> None:
    """用户词或停用词改变后，使智能分词器在下次使用时重新构建。"""

    global _cached_tokenizer, _cached_signature
    _cached_tokenizer = None
    _cached_signature = None


def get_smart_tokenizer(
    userwords: Sequence[str] = (),
    stopwords: Sequence[str] = (),
    dictionary_words: Sequence[str] = (),
) -> SmartTokenizer:
    """按规范化后的词典内容复用分词器，避免每次绘图重建 jieba。"""

    global _cached_tokenizer, _cached_signature
    signature = (
        tuple(sorted({normalize_text(word).strip() for word in userwords if normalize_text(word).strip()})),
        tuple(sorted({normalize_text(word).strip() for word in stopwords if normalize_text(word).strip()})),
        tuple(sorted({normalize_text(word).strip() for word in dictionary_words if normalize_text(word).strip()})),
    )
    if _cached_tokenizer is None or signature != _cached_signature:
        _cached_tokenizer = SmartTokenizer(*signature)
        _cached_signature = signature
    return _cached_tokenizer


def build_generic_baselines(
    daily_messages: Iterable[Sequence[MessageSample]],
    *,
    userwords: Sequence[str] = (),
    stopwords: Sequence[str] = (),
    dictionary_words: Sequence[str] = (),
) -> dict[str, GenericWordBaseline]:
    """从多个历史自然日计算泛用词的消息覆盖率中位数和 MAD。

    每条消息对同一个词最多贡献一次。没有消息的日期不参与基线；有消息但没有
    出现该词的日期以零覆盖率参与，防止偶发词形成虚高基线。
    """

    tokenizer = get_smart_tokenizer(userwords, stopwords, dictionary_words)
    forced_keys = set(tokenizer.userword_surfaces)
    tracked_keys = GENERIC_WORDS.difference(forced_keys, tokenizer.stopwords)
    rates_by_key: dict[str, list[float]] = {
        key: [] for key in tracked_keys
    }

    for messages in daily_messages:
        samples = list(messages)
        if not samples:
            continue

        document_counts: Counter[str] = Counter()
        for sample in samples:
            keys = {
                candidate.key
                for candidate in tokenizer.tokenize(sample.text)
                if candidate.key in tracked_keys
            }
            document_counts.update(keys)

        message_count = len(samples)
        for key in tracked_keys:
            rates_by_key[key].append(document_counts[key] / message_count)

    baselines: dict[str, GenericWordBaseline] = {}
    for key, rates in rates_by_key.items():
        if not rates:
            continue
        median_rate = statistics.median(rates)
        mad_rate = statistics.median(
            abs(rate - median_rate) for rate in rates
        )
        baselines[key] = GenericWordBaseline(
            median_rate=median_rate,
            mad_rate=mad_rate,
            history_days=len(rates),
        )
    return baselines


def _select_surface(surface_counts: Counter[str], forced_surface: str | None) -> str:
    if forced_surface:
        return forced_surface
    # 同频时优先较长、含大写的原始写法，再按字典序保证结果稳定。
    return max(
        surface_counts,
        key=lambda surface: (
            surface_counts[surface],
            len(surface),
            any(ch.isupper() for ch in surface),
            surface,
        ),
    )


def _is_generic_burst(
    *,
    document_frequency: int,
    message_count: int,
    baseline: GenericWordBaseline | None,
    min_history_days: int,
    min_documents: int,
    ratio_threshold: float,
    min_rate_increase: float,
    mad_multiplier: float,
) -> bool:
    """判断泛用词当天是否相对自身历史基线异常升温。"""

    if (
        baseline is None
        or baseline.history_days < max(1, min_history_days)
        or document_frequency < max(1, min_documents)
        or message_count <= 0
    ):
        return False

    current_rate = document_frequency / message_count
    threshold = max(
        baseline.median_rate * max(1.0, ratio_threshold),
        baseline.median_rate + max(0.0, min_rate_increase),
        baseline.median_rate
        + baseline.mad_rate * max(0.0, mad_multiplier),
    )
    return current_rate >= threshold


@dataclass(frozen=True)
class StatisticalScore:
    """本地候选的权重和软过滤结果。"""

    weight: float
    kept: bool
    filter_reason: str | None = None


def _calculate_statistical_score(
    *,
    key: str,
    message_count: int,
    capped_count: int,
    document_frequency: int,
    user_document_counts: Counter[Hashable],
    forced: bool,
    latin: bool,
    acronym: bool,
    generic_burst: bool,
) -> StatisticalScore:
    """计算词云权重；不合格候选只做软过滤，仍留给 LLM 复核。"""

    min_document_frequency = 2 if message_count >= 20 else 1
    single_cjk = len(key) == 1 and bool(CJK_RE.fullmatch(key))
    technical_latin = acronym or any(ch.isdigit() or ch in ".+#_-" for ch in key)

    idf = math.log((message_count + 1.0) / (document_frequency + 0.5)) + 1.0
    frequency_signal = math.log1p(capped_count)

    dominant_user_documents = max(user_document_counts.values(), default=0)
    dominance = dominant_user_documents / max(document_frequency, 1)
    dispersion = 0.65 + 0.35 * (1.0 - dominance)
    if len(user_document_counts) >= 2:
        dispersion *= 1.08

    score = frequency_signal * idf * dispersion
    filter_reason: str | None = None

    if not forced and single_cjk and document_frequency < 2:
        filter_reason = "single_cjk"
    # 小写拉丁串可能是拼音缩写，也可能是随手输入。单次出现先交给 LLM 复核；
    # 大写缩写和带版本/技术符号的标识符按普通文档频率规则处理。
    elif latin and not forced and not technical_latin and document_frequency < 2:
        filter_reason = "singleton_latin"
    elif not forced and document_frequency < min_document_frequency:
        filter_reason = "low_document_frequency"

    # 泛用词平时显著降权；相对自己的历史基线异常升温时恢复为正常内容词。
    if key in GENERIC_WORDS and not forced:
        if generic_burst:
            score *= 0.9
        else:
            if document_frequency < min(3, message_count):
                filter_reason = filter_reason or "generic_low_frequency"
            score *= 0.12

    if single_cjk and not forced:
        score *= 0.8
    if forced:
        score *= 1.25

    return StatisticalScore(
        weight=max(score, 0.01),
        kept=filter_reason is None,
        filter_reason=filter_reason,
    )


def analyze_messages(
    messages: Iterable[MessageSample],
    *,
    userwords: Sequence[str] = (),
    stopwords: Sequence[str] = (),
    dictionary_words: Sequence[str] = (),
    statistical_filter: bool = True,
    per_message_cap: int = 3,
    generic_baselines: Mapping[str, GenericWordBaseline] | None = None,
    generic_burst_min_history_days: int = 5,
    generic_burst_min_documents: int = 5,
    generic_burst_ratio_threshold: float = 2.5,
    generic_burst_min_rate_increase: float = 0.02,
    generic_burst_mad_multiplier: float = 3.0,
) -> KeywordAnalysis:
    """提取消息关键词，并生成词云权重与用户贡献统计。

    ``statistical_filter=False`` 时返回原始出现次数；开启时，同一消息内的重复
    词最多贡献 ``per_message_cap`` 次，并结合文档频率、用户分散度和保守的
    泛用词先验计算权重。提供历史基线后，泛用词仅在当日消息覆盖率相对自身
    历史显著升高时恢复展示。原始用户贡献计数始终保留，排行榜百分比不会被
    权重扭曲。
    """

    samples = list(messages)
    tokenizer = get_smart_tokenizer(userwords, stopwords, dictionary_words)
    per_message_cap = max(int(per_message_cap), 1)

    raw_counts: Counter[str] = Counter()
    capped_counts: Counter[str] = Counter()
    document_frequency: Counter[str] = Counter()
    user_counts: dict[str, Counter[Hashable]] = defaultdict(Counter)
    user_document_counts: dict[str, Counter[Hashable]] = defaultdict(Counter)
    surface_counts: dict[str, Counter[str]] = defaultdict(Counter)
    forced_keys: set[str] = set()
    latin_keys: set[str] = set()
    acronym_keys: set[str] = set()

    for sample in samples:
        candidates = tokenizer.tokenize(sample.text)
        message_occurrences: Counter[str] = Counter()
        for candidate in candidates:
            message_occurrences[candidate.key] += 1
            raw_counts[candidate.key] += 1
            user_counts[candidate.key][sample.user_id] += 1
            surface_counts[candidate.key][candidate.surface] += 1
            if candidate.forced:
                forced_keys.add(candidate.key)
            if candidate.source == "latin":
                latin_keys.add(candidate.key)
            if candidate.acronym:
                acronym_keys.add(candidate.key)

        for key, count in message_occurrences.items():
            document_frequency[key] += 1
            capped_counts[key] += min(count, per_message_cap)
            user_document_counts[key][sample.user_id] += 1

    normalized_baselines = {
        normalize_key(key): baseline
        for key, baseline in (generic_baselines or {}).items()
    }
    burst_keys: set[str] = set()
    if statistical_filter and normalized_baselines:
        for key in raw_counts:
            if key not in GENERIC_WORDS or key in forced_keys:
                continue
            if _is_generic_burst(
                document_frequency=document_frequency[key],
                message_count=len(samples),
                baseline=normalized_baselines.get(key),
                min_history_days=generic_burst_min_history_days,
                min_documents=generic_burst_min_documents,
                ratio_threshold=generic_burst_ratio_threshold,
                min_rate_increase=generic_burst_min_rate_increase,
                mad_multiplier=generic_burst_mad_multiplier,
            ):
                burst_keys.add(key)

    score_by_key: dict[str, StatisticalScore] = {}
    weighted_by_key: dict[str, float] = {}
    for key in raw_counts:
        if statistical_filter:
            score_result = _calculate_statistical_score(
                key=key,
                message_count=len(samples),
                capped_count=capped_counts[key],
                document_frequency=document_frequency[key],
                user_document_counts=user_document_counts[key],
                forced=key in forced_keys,
                latin=key in latin_keys,
                acronym=key in acronym_keys,
                generic_burst=key in burst_keys,
            )
            score_by_key[key] = score_result
            if score_result.kept:
                weighted_by_key[key] = score_result.weight
        else:
            score_result = StatisticalScore(float(raw_counts[key]), True)
            score_by_key[key] = score_result
            weighted_by_key[key] = score_result.weight

    if statistical_filter:
        # 个体得分先对泛用词降权，再与当天真正的内容词作相对比较。这样不会把
        # “群里很常见”直接判为泛用，但能避免在候选较少时“感觉、事情”等词
        # 仅靠高频重新挤回 Top 榜。管理员用户词始终不参与该删除规则。
        content_scores = [
            score
            for key, score in weighted_by_key.items()
            if key not in GENERIC_WORDS
            or key in forced_keys
            or key in burst_keys
        ]
        reference_score = max(content_scores, default=0.0)
        for key in list(weighted_by_key):
            if (
                key not in GENERIC_WORDS
                or key in forced_keys
                or key in burst_keys
            ):
                continue
            if reference_score == 0.0 or weighted_by_key[key] < reference_score * 0.35:
                weighted_by_key.pop(key)
                previous = score_by_key[key]
                score_by_key[key] = StatisticalScore(
                    weight=previous.weight,
                    kept=False,
                    filter_reason="generic_low_information",
                )

    frequencies: dict[str, float] = {}
    display_raw_counts: dict[str, int] = {}
    display_user_counts: dict[str, dict[Hashable, int]] = {}
    display_burst_words: set[str] = set()
    candidate_pool: dict[str, KeywordCandidateStats] = {}
    for key, score_result in score_by_key.items():
        display = _select_surface(
            surface_counts[key],
            tokenizer.userword_surfaces.get(key),
        )
        display_counts = dict(user_counts[key])
        locally_kept = key in weighted_by_key
        candidate_pool[display] = KeywordCandidateStats(
            weight=score_result.weight,
            raw_count=raw_counts[key],
            user_counts=display_counts,
            locally_kept=locally_kept,
            filter_reason=None if locally_kept else score_result.filter_reason,
            generic=key in GENERIC_WORDS,
            forced=key in forced_keys,
            daily_burst=key in burst_keys,
            latin=key in latin_keys,
        )
        if locally_kept:
            frequencies[display] = score_result.weight
            display_raw_counts[display] = raw_counts[key]
            display_user_counts[display] = display_counts
        if key in burst_keys:
            display_burst_words.add(display)

    return KeywordAnalysis(
        frequencies=frequencies,
        raw_counts=display_raw_counts,
        user_counts=display_user_counts,
        burst_words=frozenset(display_burst_words),
        candidate_pool=candidate_pool,
    )


def apply_keyword_decisions(
    analysis: KeywordAnalysis,
    decisions: Iterable[dict],
    *,
    protected_words: Sequence[str] = (),
    min_keywords: int = 0,
) -> KeywordAnalysis:
    """应用经过校验的 LLM 保留/删除/别名归并结果。

    ``protected_words`` 通常来自 ``data/sta/db.json`` 的 ``userwords``。
    受保护词不能被 LLM 删除或改名；停用词在候选产生阶段已经被硬过滤，LLM
    也无法把它们重新加入结果。
    """

    frequencies = dict(analysis.frequencies)
    raw_counts = dict(analysis.raw_counts)
    user_counts = {
        word: dict(counts) for word, counts in analysis.user_counts.items()
    }
    candidate_pool = dict(analysis.candidate_pool)
    if not candidate_pool:
        # 兼容手工构造或旧缓存中的 KeywordAnalysis。
        candidate_pool = {
            word: KeywordCandidateStats(
                weight=weight,
                raw_count=raw_counts[word],
                user_counts=dict(user_counts[word]),
                locally_kept=True,
                filter_reason=None,
                generic=normalize_key(word) in GENERIC_WORDS,
                forced=False,
                daily_burst=word in analysis.burst_words,
                latin=any(ch.isascii() and ch.isalpha() for ch in word),
            )
            for word, weight in frequencies.items()
        }

    protected_keys = {normalize_key(word) for word in protected_words}
    protected_keys.update(normalize_key(word) for word in analysis.burst_words)
    pool_by_key = {
        normalize_key(display): display for display in candidate_pool
    }
    decision_list = list(decisions)
    merged_keys: set[str] = set()

    def resolve_candidate(term: object) -> str | None:
        return pool_by_key.get(normalize_key(str(term or "")))

    def activate(display: str) -> None:
        if display in frequencies:
            return
        stats = candidate_pool[display]
        frequencies[display] = stats.weight
        raw_counts[display] = stats.raw_count
        user_counts[display] = dict(stats.user_counts)

    explicit_drop_keys = {
        normalize_key(decision.get("term", ""))
        for decision in decision_list
        if str(decision.get("action", "")).lower() == "drop"
    }

    # 先删除，后归并，避免删除决定被同批别名结果重新引入。
    for decision in decision_list:
        if str(decision.get("action", "")).lower() != "drop":
            continue
        display = resolve_candidate(decision.get("term"))
        if display is None or normalize_key(display) in protected_keys:
            continue
        frequencies.pop(display, None)
        raw_counts.pop(display, None)
        user_counts.pop(display, None)

    # keep 可以把本地软过滤的词重新加入结果，再按 canonical 做候选内归并。
    for decision in decision_list:
        if str(decision.get("action", "keep")).lower() == "drop":
            continue
        display = resolve_candidate(decision.get("term"))
        if display is None:
            continue
        activate(display)

        canonical = resolve_candidate(decision.get("canonical")) or display
        if (
            normalize_key(display) in protected_keys
            or normalize_key(canonical) in explicit_drop_keys
        ):
            canonical = display
        if display == canonical:
            continue
        activate(canonical)

        frequencies[canonical] += frequencies.pop(display)
        raw_counts[canonical] += raw_counts.pop(display)
        merged_user_counts = Counter(user_counts.get(canonical, {}))
        merged_user_counts.update(user_counts.pop(display, {}))
        user_counts[canonical] = dict(merged_user_counts)
        merged_keys.add(normalize_key(display))

    # 数量安全线：优先从未被 LLM 明确删除的非泛用候选中补齐。候选仍然必须
    # 来自本地分词，停用词、URL 等硬过滤内容不可能被重新加入。
    target_count = max(0, int(min_keywords))
    if len(frequencies) < target_count:
        refill_candidates = [
            (display, stats)
            for display, stats in candidate_pool.items()
            if display not in frequencies
            and normalize_key(display) not in explicit_drop_keys
            and normalize_key(display) not in merged_keys
            and (not stats.generic or stats.forced or stats.daily_burst)
        ]
        refill_candidates.sort(
            key=lambda item: (
                item[1].forced or item[1].daily_burst,
                item[1].locally_kept,
                item[1].raw_count >= 2,
                len(item[1].user_counts),
                item[1].raw_count,
                item[1].weight,
                not item[1].latin,
            ),
            reverse=True,
        )
        for display, _ in refill_candidates:
            activate(display)
            if len(frequencies) >= target_count:
                break

    return KeywordAnalysis(
        frequencies=frequencies,
        raw_counts=raw_counts,
        user_counts=user_counts,
        burst_words=frozenset(
            word for word in analysis.burst_words if word in frequencies
        ),
        candidate_pool=candidate_pool,
    )
