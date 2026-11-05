"""独立测量 947 求解器的冷启动、缓存查询和进程峰值内存。"""

import importlib.util
import json
import math
import platform
import random
import resource
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from time import perf_counter


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/plugins/kyn/dp.py"
SPEC = importlib.util.spec_from_file_location("kyn_dp_standalone_benchmark", MODULE_PATH)
dp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dp
SPEC.loader.exec_module(dp)


def measure_query(solver, target, limits):
    """只对求解计时，随后重新解析并独立校验实际输出。"""
    started = perf_counter()
    result = solver.solve(target, limits)
    elapsed_ms = (perf_counter() - started) * 1000
    if not dp.validate(dp.parse_expression(result["expression"]), target):
        raise ValueError("基准输出未通过独立验证")
    return result, elapsed_ms


def main():
    """运行固定样本基准，将详细冷启动结果与总体统计写入数据目录。"""
    limits = dp.SearchLimits()
    cold_cases = []
    for target in (947, 2026):
        solver = dp.Solver()
        result, elapsed_ms = measure_query(solver, target, limits)
        cold_cases.append({
            "result": result,
            "actual_elapsed_ms": elapsed_ms,
            "statistics": solver.statistics(),
        })

    warm_solver = dp.Solver()
    started = perf_counter()
    warm_solver.precompute()
    precompute_ms = (perf_counter() - started) * 1000
    targets = random.Random(947).sample(range(-1000, 1001), 64)
    targets = list(dict.fromkeys(targets + [-3, 2, 0, 1, 47, 94, 947, 2026]))
    before = warm_solver.statistics()
    elapsed_times = []
    methods = Counter()
    optimal_count = 0
    for target in targets:
        result, elapsed_ms = measure_query(warm_solver, target, limits)
        elapsed_times.append(elapsed_ms)
        methods[result["method"]] += 1
        optimal_count += bool(result["optimal"])
    after = warm_solver.statistics()
    hits = after["cache_hits"] - before["cache_hits"]
    misses = after["cache_misses"] - before["cache_misses"]
    query_count = len(targets)
    warm_summary = {
        "query_count": query_count,
        "precompute_ms": precompute_ms,
        "mean_ms": mean(elapsed_times),
        "p95_ms": sorted(elapsed_times)[math.ceil(0.95 * query_count) - 1],
        "cache_hit_rate": hits / (hits + misses) if hits + misses else 0,
        "max_states": after["max_states"],
        "pair_expansions": after["pair_expansions"] - before["pair_expansions"],
        "optimal_rate": optimal_count / query_count,
        "method_counts": dict(methods),
    }
    # 单独测量四组近似搜索和预算中止，避免混入小整数样本的总体统计。
    bounded_cases = []
    for target, case_limits in (
        (525876624, dp.SearchLimits()),
        (10**30 + 947, dp.SearchLimits(max_pair_expansions=6000)),
    ):
        before_stats = warm_solver.statistics()
        result, elapsed_ms = measure_query(warm_solver, target, case_limits)
        after_stats = warm_solver.statistics()
        bounded_cases.append({
            "limits": asdict(case_limits),
            "result": result,
            "actual_elapsed_ms": elapsed_ms,
            "pair_expansions": after_stats["pair_expansions"] - before_stats["pair_expansions"],
            "truncated_intervals": after_stats["truncated_intervals"] - before_stats["truncated_intervals"],
            "complete": after_stats["complete"],
        })
    # Linux 的 ru_maxrss 单位为 KiB；这是整个进程的峰值，而非算法独占内存。
    summary = {
        "python_version": platform.python_version(),
        "limits": asdict(limits),
        "peak_process_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "memory_note": "峰值内存包含 Python 进程、缓存和基准本身，不是算法精确独占内存。",
        "warm_summary": warm_summary,
    }
    output_path = Path("data/kyn/dp_benchmark.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        json.dump({**summary, "cold_cases": cold_cases, "bounded_cases": bounded_cases},
                  output, ensure_ascii=False, indent=2)
        output.write("\n")
    bounded_summary = [
        {key: case["result"][key] for key in ("target", "cycles", "method", "optimal", "elapsed_ms")}
        for case in bounded_cases
    ]
    print(json.dumps({**summary, "bounded_cases": bounded_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
