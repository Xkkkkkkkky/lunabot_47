"""持久化 LLM 明确确认的 STA 辅助分词词典。"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from ..utils import get_file_db, get_logger
from .tokenizer import GENERIC_WORDS, normalize_key, normalize_text


logger = get_logger("Sta")
dictionary_db = get_file_db("data/sta/llm_words.json", logger)

DICTIONARY_VERSION = 1
MAX_DICTIONARY_WORDS = 2000
MAX_EVIDENCE_KEYS = 20


def _normalize_surface(word: object) -> tuple[str, str] | None:
    surface = normalize_text(str(word or "")).strip()
    key = normalize_key(surface)
    if not key or len(surface) < 2 or len(surface) > 64:
        return None
    if any(ch.isspace() for ch in surface):
        return None
    if not any(ch.isalnum() for ch in surface):
        return None
    return key, surface


def get_llm_dictionary_words(
    *,
    stopwords: Sequence[str] = (),
) -> list[str]:
    """读取可用于 jieba 的学习词；管理员停用词始终优先。"""

    stopped_keys = {normalize_key(word) for word in stopwords}
    entries = dictionary_db.get_copy("words", {})
    if isinstance(entries, list):
        raw_words = entries
    elif isinstance(entries, dict):
        raw_words = [
            value.get("surface", key) if isinstance(value, dict) else value
            for key, value in entries.items()
        ]
    else:
        raw_words = []

    words_by_key: dict[str, str] = {}
    for word in raw_words:
        normalized = _normalize_surface(word)
        if normalized is None:
            continue
        key, surface = normalized
        if key in stopped_keys or key in GENERIC_WORDS:
            continue
        words_by_key[key] = surface
    return sorted(words_by_key.values(), key=normalize_key)


def record_llm_dictionary_words(
    words: Iterable[str],
    *,
    evidence_key: str,
    observed_at: str | None = None,
    stopwords: Sequence[str] = (),
) -> int:
    """记录 LLM 明确救回的词，返回本次新增或新增证据的词数。"""

    stopped_keys = {normalize_key(word) for word in stopwords}
    timestamp = observed_at or datetime.now().isoformat(timespec="seconds")
    entries = dictionary_db.get_copy("words", {})
    if isinstance(entries, list):
        migrated_entries = {}
        for word in entries:
            normalized = _normalize_surface(word)
            if normalized is None:
                continue
            key, surface = normalized
            migrated_entries[key] = {
                "surface": surface,
                "confirmations": 0,
                "first_seen": timestamp,
                "last_seen": timestamp,
                "evidence_keys": [],
            }
        entries = migrated_entries
    elif not isinstance(entries, dict):
        entries = {}

    changed = 0
    for word in words:
        normalized = _normalize_surface(word)
        if normalized is None:
            continue
        key, surface = normalized
        if key in stopped_keys or key in GENERIC_WORDS:
            continue

        previous = entries.get(key, {})
        if not isinstance(previous, dict):
            previous = {}
        evidence_keys = previous.get("evidence_keys", [])
        if not isinstance(evidence_keys, list):
            evidence_keys = []
        evidence_keys = [str(item) for item in evidence_keys]
        is_new_evidence = evidence_key not in evidence_keys
        if is_new_evidence:
            evidence_keys.append(evidence_key)
            evidence_keys = evidence_keys[-MAX_EVIDENCE_KEYS:]
            changed += 1

        try:
            previous_confirmations = int(previous.get("confirmations", 0))
        except (TypeError, ValueError):
            previous_confirmations = 0
        entries[key] = {
            "surface": surface,
            "confirmations": previous_confirmations + int(is_new_evidence),
            "first_seen": previous.get("first_seen", timestamp),
            "last_seen": timestamp if is_new_evidence else previous.get(
                "last_seen", timestamp
            ),
            "evidence_keys": evidence_keys,
        }

    if not changed:
        return 0

    if len(entries) > MAX_DICTIONARY_WORDS:
        def rank_entry(item):
            value = item[1] if isinstance(item[1], dict) else {}
            try:
                confirmations = int(value.get("confirmations", 0))
            except (TypeError, ValueError):
                confirmations = 0
            return confirmations, str(value.get("last_seen", ""))

        ranked = sorted(
            entries.items(),
            key=rank_entry,
            reverse=True,
        )[:MAX_DICTIONARY_WORDS]
        entries = dict(ranked)

    dictionary_db.set("version", DICTIONARY_VERSION)
    dictionary_db.set("words", entries)
    logger.info(f"STA LLM辅助词典更新 新增证据词数:{changed} 总词数:{len(entries)}")
    return changed
