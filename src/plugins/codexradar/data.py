"""读取雷达站公开聚合成绩，保留上游分数与缺失状态。"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiohttp

from ..utils import get_client_session


DIMENSIONS = {
    "total": ("总分", "total"),
    "se": ("软件工程", "deepswe"),
    "vis": ("视觉空间推理", "pompeii"),
    "term": ("终端与系统操作", "tb4"),
    "sci": ("科学研究", "science"),
}


class RadarDataError(ValueError):
    """可以直接向用户说明的数据读取问题。"""


@dataclass(frozen=True)
class RadarScore:
    score: float | None
    coverage: int | None
    required_tasks: int
    graded_runs_count: int | None


@dataclass(frozen=True)
class RadarPoint:
    model: str
    effort: str
    scores: dict[str, RadarScore]
    average_price_usd: float | None = None
    average_minutes: float | None = None
    price_partial: bool = False
    duration_partial: bool = False
    error: str | None = None


@dataclass(frozen=True)
class RadarSnapshot:
    points: tuple[RadarPoint, ...]
    score_max: float
    total_tasks: int
    source_counts: dict[str, int]
    fetched_at: datetime


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _parse_score(row: dict, required: int, score_max: float) -> RadarScore:
    if not isinstance(row, dict):
        raise RadarDataError("雷达站返回的成绩字段异常")
    coverage = row.get("coverage")
    score = row.get("score")
    grades = row.get("graded_runs_count")
    if (
        row.get("required_tasks") != required
        or not _count(coverage)
        or coverage > required
        or (coverage == 0 and score is not None)
        or (coverage > 0 and (not _number(score) or score > score_max))
        or (grades is not None and (not _count(grades) or grades < coverage))
    ):
        raise RadarDataError("雷达站返回的成绩字段异常")
    # 真实零分必须保留，缺失成绩不能转换成零分。
    return RadarScore(score, coverage, required, grades)


def _metric_mean(payload: dict, key: str) -> tuple[float | None, bool]:
    if payload.get("statistics_state") != "available":
        return None, False
    statistics = payload.get("statistics") or {}
    if not isinstance(statistics, dict):
        return None, False
    if not isinstance(statistics.get("metrics"), dict):
        return None, False
    metric = (statistics.get("metrics") or {}).get(key) or {}
    if not isinstance(metric, dict):
        return None, False
    known = metric.get("known_samples")
    missing = metric.get("missing_samples")
    total = metric.get("known_total")
    if (
        not _count(known)
        or not _count(missing)
        or known + missing != statistics.get("sample_count")
        or not known
        or not _number(total)
    ):
        return None, False
    # 与站点一致，使用有证据的样本均值；不能用判分题次作分母。
    return total / known, missing > 0


def parse_point(payload: dict, binding: dict, selection: dict) -> RadarPoint:
    if not isinstance(payload, dict) or any(
        payload.get(key) != binding.get(key)
        for key in ("benchmark", "task_set_sha256", "score_version", "score_max")
    ):
        raise RadarDataError("雷达站的题集或计分版本发生变化，请稍后重试")
    if any(payload.get(key) != selection[key] for key in ("model", "effort")):
        raise RadarDataError("雷达站返回了其他模型档位的成绩")
    scores = {
        "total": _parse_score(payload, binding["total_tasks"], binding["score_max"])
    }
    sources = payload.get("source_scores") or {}
    if not isinstance(sources, dict):
        raise RadarDataError("雷达站返回的分项成绩异常")
    for name, required in binding["source_counts"].items():
        if name in sources:
            scores[name] = _parse_score(sources[name], required, binding["score_max"])
    price, price_partial = _metric_mean(payload, "api_equivalent_usd")
    elapsed_ms, duration_partial = _metric_mean(payload, "elapsed_ms")
    return RadarPoint(
        model=selection["model"],
        effort=selection["effort"],
        scores=scores,
        average_price_usd=price,
        average_minutes=None if elapsed_ms is None else elapsed_ms / 60000,
        price_partial=price_partial,
        duration_partial=duration_partial,
    )


def _validate_binding(binding: Any) -> list[dict[str, str]]:
    if not isinstance(binding, dict) or not (
        binding.get("enabled") is True and binding.get("production_read_verified") is True
    ):
        raise RadarDataError("雷达站暂未开放正式成绩查询")
    if (
        not _number(binding.get("score_max"))
        or not binding["score_max"]
        or not _count(binding.get("total_tasks"))
        or not binding["total_tasks"]
        or not isinstance(binding.get("source_counts"), dict)
        or not all(_count(n) and n > 0 for n in binding["source_counts"].values())
        or sum(binding["source_counts"].values()) != binding["total_tasks"]
        or not all(binding.get(k) for k in ("benchmark", "task_set_sha256", "score_version"))
    ):
        raise RadarDataError("雷达站返回的题集信息异常")
    selections = binding.get("model_efforts")
    if not isinstance(selections, list) or not selections:
        raise RadarDataError("雷达站暂未提供可查询的模型")
    seen = set()
    result = []
    for row in selections:
        if not isinstance(row, dict) or not all(
            isinstance(row.get(k), str) and row[k].strip() for k in ("model", "effort")
        ):
            raise RadarDataError("雷达站返回的模型清单异常")
        key = (row["model"], row["effort"])
        if key not in seen:
            seen.add(key)
            result.append({"model": key[0], "effort": key[1]})
    return result


class RadarClient:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self._lock = asyncio.Lock()
        self._snapshot: RadarSnapshot | None = None
        self._cache_key: tuple | None = None
        self._expires_at = 0.0

    async def get_snapshot(self) -> RadarSnapshot:
        binding_url = str(self.config.get("binding_url", "")).strip()
        score_url = str(self.config.get("score_url", "")).strip()
        if not binding_url or not score_url:
            raise RadarDataError("请先在 config/codexradar.yaml 中填写数据源地址")
        timeout_seconds = float(self.config.get("request_timeout_seconds", 40))
        cache_seconds = float(self.config.get("cache_seconds", 300))
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise RadarDataError("查询超时时间必须大于零")
        if not math.isfinite(cache_seconds) or cache_seconds < 0:
            raise RadarDataError("数据缓存时间不能为负数")
        cache_key = (binding_url, score_url, timeout_seconds, cache_seconds)
        async with self._lock:
            if self._cache_key == cache_key and time.monotonic() < self._expires_at:
                return self._snapshot
            timeout = aiohttp.ClientTimeout(total=timeout_seconds)
            session = get_client_session()
            async with session.get(binding_url, timeout=timeout) as response:
                response.raise_for_status()
                binding = await response.json(content_type=None)
            selections = _validate_binding(binding)
            semaphore = asyncio.Semaphore(2)

            async def fetch(selection: dict) -> RadarPoint:
                async with semaphore:
                    try:
                        async with session.get(
                            score_url,
                            params={**selection, "view": "summary"},
                            headers={"X-Radar-Bench-Selection": "user-selected-v1"},
                            timeout=timeout,
                        ) as response:
                            response.raise_for_status()
                            payload = await response.json(content_type=None)
                        return parse_point(payload, binding, selection)
                    except (aiohttp.ClientError, TimeoutError, ValueError, TypeError) as exc:
                        self.logger.warning(
                            f'雷达成绩读取失败：{selection["model"]} / '
                            f'{selection["effort"]}（{type(exc).__name__}）'
                        )
                        return RadarPoint(**selection, scores={}, error="读取失败")

            points = tuple(await asyncio.gather(*(fetch(s) for s in selections)))
            failures = sum(point.error is not None for point in points)
            if failures == len(points):
                raise RadarDataError("暂时无法读取雷达成绩，请稍后重试")
            snapshot = RadarSnapshot(
                points=points,
                score_max=binding["score_max"],
                total_tasks=binding["total_tasks"],
                source_counts=dict(binding["source_counts"]),
                fetched_at=datetime.now(timezone.utc),
            )
            self._snapshot = snapshot
            self._cache_key = cache_key
            # 不用旧成绩掩盖失败；部分读取失败时缩短本轮结果的缓存时间。
            self._expires_at = time.monotonic() + (
                min(cache_seconds, 60) if failures else cache_seconds
            )
            return snapshot
