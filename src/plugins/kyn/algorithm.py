"""947 表达式算法的统一入口；切换算法时只需修改此处。"""

import json
import logging
import threading
from pathlib import Path
from time import perf_counter

from .dp import (
    SearchLimits,
    normalize_result,
    proof_metadata,
    solve as _solve,
    validate_result,
)


RESULTS_PATH = Path("data/kyn/results.json")
_cache_lock = threading.Lock()
logger = logging.getLogger(__name__)


def _load_results() -> dict:
    try:
        with RESULTS_PATH.open("r", encoding="utf-8") as file:
            results = json.load(file)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("读取整数表达式缓存失败，将重新求解：%s", exc)
        return {}
    if not isinstance(results, dict):
        logger.warning("整数表达式缓存顶层必须为对象，将重新求解")
        return {}
    return results


def _save_results(results: dict) -> None:
    temporary_path = RESULTS_PATH.with_name(RESULTS_PATH.name + ".tmp")
    try:
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(results, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
        temporary_path.replace(RESULTS_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def generate_polynomial(target: int, limits: SearchLimits | None = None) -> dict:
    if type(target) is not int:
        raise TypeError("目标必须为整数")
    start = perf_counter()
    # 串行执行读取、搜索和写入，避免重复求解及缓存更新丢失。
    with _cache_lock:
        key = str(target)
        records = _load_results()
        cached = records.get(key)
        if cached is not None:
            try:
                cached = normalize_result(cached, target)
            except ValueError as exc:
                logger.warning("目标 %s 的整数表达式缓存无效，将重新求解：%s", target, exc)
                cached = None
        if (
            cached is not None
            and cached["optimal"] is True
            and cached["proof_status"] == "optimal"
        ):
            result = dict(cached)
            result["cache_hit"] = True
            result["elapsed_ms"] = round((perf_counter() - start) * 1000, 3)
            return result

        result = normalize_result(_solve(target, limits, initial_result=cached), target)
        validate_result(result, target)
        if cached is not None:
            selected = min(
                (cached, result),
                key=lambda item: (item["cycles"], len(item["expression"])),
            )
            selected = dict(selected)
            # 同一目标共享已确认的排除前缀，但不能沿用与所选解矛盾的证明。
            proofs = [
                item["exhaustive_through_cycles"]
                for item in (cached, result)
                if item["proof_status"] != "unproven"
                and item["exhaustive_through_cycles"] < selected["cycles"]
            ]
            selected.update(
                proof_metadata(
                    selected["cycles"], max(proofs, default=0), confirmed=bool(proofs)
                )
            )
            result = selected
        else:
            result = dict(result)
        validate_result(result, target)
        result["cache_hit"] = False
        result["elapsed_ms"] = round((perf_counter() - start) * 1000, 3)
        records[key] = dict(result)
        try:
            _save_results(records)
        except OSError as exc:
            logger.warning("保存整数表达式缓存失败：%s", exc)
        result["elapsed_ms"] = round((perf_counter() - start) * 1000, 3)
        return result
