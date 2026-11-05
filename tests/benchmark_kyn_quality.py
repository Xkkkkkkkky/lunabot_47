"""以固定目标和独立预算测量构造质量；支持逐项保存和中断续跑。"""

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import platform
import random
import resource
import sys
import types
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/plugins/kyn/dp.py"
SPEC = importlib.util.spec_from_file_location("kyn_quality_dp", MODULE_PATH)
dp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dp
SPEC.loader.exec_module(dp)
SOLVER = None


def load_algorithm():
    """为相对导入建立隔离包，绕过插件初始化和 NoneBot。"""
    package_name = "kyn_quality_package"
    package = types.ModuleType(package_name)
    package.__path__ = [str(MODULE_PATH.parent)]
    sys.modules[package_name] = package
    sys.modules[f"{package_name}.dp"] = dp
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.algorithm", MODULE_PATH.with_name("algorithm.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def initialize_worker():
    """每个进程独立预热，不共享搜索结果。"""
    global SOLVER
    SOLVER = dp.Solver()
    SOLVER.prepare_constructions()


def constructions(solver, target):
    costs = {}
    for name, expression in solver.construction_candidates(target).items():
        if not dp.validate(dp.parse_expression(dp.render(expression)), target):
            raise ValueError(f"构造验证失败：{name}，目标 {target}")
        costs[name] = expression.digits_used // 3
    if not costs:
        raise ValueError("构造候选不能为空")
    return costs


def run_job(job):
    limits = replace(dp.SearchLimits(), timeout_seconds=job["budget"], max_cycles_search="a")
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    started = perf_counter()
    result = SOLVER.solve(job["target"], limits)
    actual_ms = (perf_counter() - started) * 1000
    expression = result["expression"]
    if not dp.validate(dp.parse_expression(expression), job["target"]):
        raise ValueError(f"实际输出验证失败：{job['job_id']}")
    costs = constructions(SOLVER, job["target"])
    if result["initial_cycles"] != min(costs.values()):
        raise ValueError(f"初始上界不等于最优确定性构造：{job['job_id']}")
    fields = ("initial_cycles", "initial_method", "initial_construction_ms", "first_improvement_ms",
              "optimal", "timed_out", "termination_reason", "elapsed_ms", "method", "rules_version",
              "proof_version", "exhaustive_through_cycles", "proof_status", "effective_max_cycles_search",
              "search_upper_cycles")
    row = {**job, **{key: result[key] for key in fields}}
    row.update(construction_cycles=costs, final_cycles=result["cycles"],
               improvement_cycles=result["initial_cycles"] - result["cycles"],
               actual_elapsed_ms=actual_ms, peak_worker_rss_before_mb=rss_before,
               peak_worker_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               expression=expression)
    return row


def write_json(path, value):
    """用同目录替换保证状态文件完整。"""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {"count": len(values), "mean": mean(values),
            "p50": values[math.ceil(len(values) * .50) - 1],
            "p95": values[math.ceil(len(values) * .95) - 1]}


def summarize(rows):
    groups = {}
    for row in rows:
        name = f"random/{row['digits']}/{row['budget']}" if row["cohort"] == "random" else f"edge/{row['budget']}"
        groups.setdefault(name, []).append(row)
    output = {}
    for name, group in groups.items():
        count = len(group)
        output[name] = {
            "count": count,
            "initial_cycles": distribution([r["initial_cycles"] for r in group]),
            "final_cycles": distribution([r["final_cycles"] for r in group]),
            "mean_improvement": mean(r["improvement_cycles"] for r in group),
            "first_improvement_ms": distribution([r["first_improvement_ms"] for r in group if r["first_improvement_ms"] is not None]),
            "improvement_rate": sum(r["improvement_cycles"] > 0 for r in group) / count,
            "optimal_rate": sum(r["optimal"] for r in group) / count,
            "timeout_rate": sum(r["timed_out"] for r in group) / count,
            "termination_reasons": dict(Counter(r["termination_reason"] for r in group)),
            "latency_ms": distribution([r["actual_elapsed_ms"] for r in group]),
            "max_peak_worker_rss_mb": max(r["peak_worker_rss_mb"] for r in group),
            "construction_cycles": {method: distribution([r["construction_cycles"][method] for r in group])
                                    for method in group[0]["construction_cycles"]},
        }
    return {"percentile_definition": "最近秩：排序后取第 ceil(n*p) 项；空集合为 null。首次改进统计排除 None。",
            "rss_definition": "Linux ru_maxrss/1024，单位 MiB；为进程峰值，跨任务保留高水位，不代表独占内存。",
            "groups": output}


def targets_for(args, solver):
    rng = random.Random(args.seed)
    targets = []
    edges = [0, -1, -947, -2026, -10**18 + 1, 10**20 - 1, 10**20 - 2,
             10**20 - 252, -(10**20 - 1), 525876624, 9999999967]
    for digits in args.digits:
        low, high = 10**(digits - 1), 10**digits - 1
        if args.samples > high - low + 1:
            raise ValueError("样本数超过该位数的目标总数")
        chosen = set()
        while len(chosen) < args.samples:
            chosen.add(rng.randint(low, high))
        group = sorted(chosen)
        targets.extend({"cohort": "random", "digits": digits, "target": target} for target in group)
        costs = {target: constructions(solver, target) for target in group}
        edges.append(max(group, key=lambda target: (costs[target]["radix252"], target)))
        edges.append(max(group, key=lambda target: (min(costs[target].values()), target)))
    targets.extend({"cohort": "edge", "digits": len(str(abs(target))), "target": target}
                   for target in dict.fromkeys(edges))
    return targets


def progress(rows, jobs, status, error=None):
    counts = Counter((r["cohort"], r["digits"] if r["cohort"] == "random" else "all", r["budget"]) for r in rows)
    expected = Counter((j["cohort"], j["digits"] if j["cohort"] == "random" else "all", j["budget"]) for j in jobs)
    result = {"status": status, "completed": len(rows), "expected": len(jobs),
              "groups": {"/".join(map(str, key)): {"completed": counts[key], "expected": value}
                         for key, value in expected.items()}}
    if error is not None:
        result["error"] = error
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digits", nargs="+", type=int, default=[6, 12, 18])
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--budgets", nargs="+", type=float, default=[10, 60, 300])
    parser.add_argument("--seed", type=int, default=947)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("data/kyn/quality_benchmark"))
    args = parser.parse_args()
    if args.samples < 1 or any(d < 1 for d in args.digits) or any(b <= 0 for b in args.budgets):
        parser.error("位数、样本数和预算必须为正数")
    if len(set(args.digits)) != len(args.digits) or len(set(args.budgets)) != len(args.budgets):
        parser.error("位数和预算不能重复")
    if not 1 <= args.workers <= (os.cpu_count() or 1):
        parser.error("workers 必须为正整数且不能超过 CPU 数量")
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to((ROOT / "data").resolve()):
        parser.error("结果目录必须位于项目 data 目录内")
    core = {"dp_sha256": hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest(),
            "algorithm_sha256": hashlib.sha256(MODULE_PATH.with_name("algorithm.py").read_bytes()).hexdigest(),
            "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "seed": args.seed, "digits": args.digits, "samples": args.samples,
            "budgets": args.budgets, "workers": args.workers,
            "search_limits_defaults": asdict(dp.SearchLimits()), "python_version": platform.python_version()}
    existing = None
    if output.exists():
        manifest_path = output / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("已有目录缺少 manifest.json，不能续跑")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in core.items()):
            raise ValueError("已有基准参数或求解器源码不一致，请使用新的输出目录")
    solver = dp.Solver()
    cold_start = perf_counter()
    solver.prepare_constructions()
    cold_costs = constructions(solver, 525876624)
    cold_ms = (perf_counter() - cold_start) * 1000
    warm_start = perf_counter()
    constructions(solver, 525876624)
    warm_ms = (perf_counter() - warm_start) * 1000
    targets = targets_for(args, solver)
    jobs = [{**target, "budget": budget, "job_id": f"{target['cohort']}:{target['target']}:{budget}"}
            for target in targets for budget in args.budgets]
    manifest = {**core, "expected_jobs": len(jobs), "expected_random_jobs": args.samples * len(args.digits) * len(args.budgets),
                "expected_edge_jobs": sum(j["cohort"] == "edge" for j in jobs),
                "created_utc": datetime.now(timezone.utc).isoformat()}
    if existing is not None:
        if any(existing.get(k) != manifest[k] for k in ("expected_jobs", "expected_random_jobs", "expected_edge_jobs")):
            raise ValueError("任务数量与已有 manifest 不一致")
        if json.loads((output / "targets.json").read_text(encoding="utf-8")) != targets:
            raise ValueError("目标样本与已有 targets.json 不一致")
    else:
        output.mkdir(parents=True)
        write_json(output / "manifest.json", manifest)
        write_json(output / "targets.json", targets)
    rows = []
    checkpoint = output / "results.jsonl"
    valid_jobs = {job["job_id"]: job for job in jobs}
    completed = set()
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["job_id"] in completed or row["job_id"] not in valid_jobs:
                raise ValueError("checkpoint 包含重复或未知任务")
            if any(row[key] != value for key, value in valid_jobs[row["job_id"]].items()):
                raise ValueError("checkpoint 任务参数不一致")
            completed.add(row["job_id"])
            rows.append(row)
    try:
        report = solver.remainder_report()
        if report["coverage"] != 252:
            raise ValueError("余数表未覆盖全部 252 个余数")
        write_json(output / "remainder_report.json", report)
        metadata_path = output / "metadata.json"
        if not metadata_path.exists():
            probe_limits = replace(dp.SearchLimits(), timeout_seconds=min(args.budgets), max_cycles_search="a")
            cold_solver = dp.Solver()
            cold_start = perf_counter()
            cold_result = cold_solver.solve(525876624, probe_limits)
            cold_solve_ms = (perf_counter() - cold_start) * 1000
            if not dp.validate(dp.parse_expression(cold_result["expression"]), 525876624):
                raise ValueError("冷启动求解验证失败")
            exact_start = perf_counter()
            exact_result = cold_solver.solve(525876624, probe_limits)
            warm_exact_ms = (perf_counter() - exact_start) * 1000
            if not dp.validate(dp.parse_expression(exact_result["expression"]), 525876624):
                raise ValueError("精确缓存预热求解验证失败")
            algorithm = load_algorithm()
            algorithm.RESULTS_PATH = output / "cache_probe/results.json"
            # 元数据写入前异常退出时，重测必须仍从空结果缓存开始。
            algorithm.RESULTS_PATH.unlink(missing_ok=True)
            cache_start = perf_counter()
            cold_cached = algorithm.generate_polynomial(947, probe_limits)
            cold_cache_ms = (perf_counter() - cache_start) * 1000
            algorithm.validate_result(cold_cached, 947)
            cache_start = perf_counter()
            warm_cached = algorithm.generate_polynomial(947, probe_limits)
            warm_cache_ms = (perf_counter() - cache_start) * 1000
            if warm_cached["cache_hit"] is not True:
                raise ValueError("第二次结果缓存探测未命中")
            algorithm.validate_result(warm_cached, 947)
            warm_start = perf_counter()
            result = solver.solve(525876624, replace(dp.SearchLimits(), timeout_seconds=min(args.budgets), max_cycles_search="a"))
            solve_ms = (perf_counter() - warm_start) * 1000
            if not dp.validate(dp.parse_expression(result["expression"]), 525876624):
                raise ValueError("预热求解验证失败")
            write_json(metadata_path, {"target": 525876624, "cold_prepare_and_construction_ms": cold_ms,
                                      "cold_solve_ms": cold_solve_ms,
                                      "warm_exact_cache_solve_ms": warm_exact_ms,
                                      "result_cache_target": 947,
                                      "cold_result_cache_ms": cold_cache_ms,
                                      "warm_result_cache_ms": warm_cache_ms,
                                      "warm_construction_ms": warm_ms, "warm_solve_ms": solve_ms,
                                      "warm_solve_budget": min(args.budgets), "construction_cycles": cold_costs})
        write_json(output / "progress.json", progress(rows, jobs, "running"))
        last_print = perf_counter()
        with checkpoint.open("a", encoding="utf-8") as stream:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                     initializer=initialize_worker) as pool:
                futures = [pool.submit(run_job, job) for job in jobs if job["job_id"] not in completed]
                pending = set(futures)
                while pending:
                    finished, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    if not finished:
                        print(f"已完成 {len(rows)}/{len(jobs)}", flush=True)
                        last_print = perf_counter()
                    for future in finished:
                        row = future.result()
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        rows.append(row)
                        write_json(output / "progress.json", progress(rows, jobs, "running"))
                        if len(rows) % 10 == 0:
                            write_json(output / "partial_summary.json", summarize(rows))
                        if len(rows) % 10 == 0 or perf_counter() - last_print >= 30:
                            print(f"已完成 {len(rows)}/{len(jobs)}", flush=True)
                            last_print = perf_counter()
        write_json(output / "summary.json", summarize(rows))
        write_json(output / "progress.json", progress(rows, jobs, "complete"))
    except BaseException as exc:
        write_json(output / "progress.json", progress(rows, jobs, "failed", f"{type(exc).__name__}: {exc}"))
        raise
    print(f"{output.relative_to(ROOT)}：{len(rows)}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
