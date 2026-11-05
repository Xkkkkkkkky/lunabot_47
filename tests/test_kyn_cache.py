"""通过独立虚拟包验证持久缓存，避免导入 NoneBot 插件。"""

import importlib
import json
import sys
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


PLUGIN_PATH = Path(__file__).resolve().parents[1] / "src/plugins/kyn"


def load_algorithm(name):
    package = types.ModuleType(name)
    package.__path__ = [str(PLUGIN_PATH)]
    sys.modules[name] = package
    return importlib.import_module(name + ".algorithm")


algorithm = load_algorithm("kyn_cache_standalone_tests")
dp = importlib.import_module("kyn_cache_standalone_tests.dp")


def record(target, expression, cycles, through=0, optimal=False):
    return {
        "target": target,
        "expression": expression,
        "cycles": cycles,
        "digits_used": cycles * 3,
        "verified": True,
        **dp.proof_metadata(cycles, through),
        "optimal": optimal,
        "method": "beam" if cycles == 4 else "exact_dp",
        "elapsed_ms": 1,
        "exhaustive_through_cycles": through,
    }


class KynCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "results.json"
        patcher = mock.patch.object(algorithm, "RESULTS_PATH", self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.zero_limits = dp.SearchLimits(timeout_seconds=0)

    def write_records(self, records):
        self.path.write_text(json.dumps(records), encoding="utf-8")

    def read_records(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def four_cycle_record(self):
        return record(525876624, "(9+4)*7*(9+4)*7*9*4*7*9*4*7", 4, 2)

    def test_restart_reuses_optimal_result(self):
        first = algorithm.generate_polynomial(2)
        self.assertTrue(first["optimal"])
        self.assertTrue(first["verified"])
        self.assertIn("2", self.read_records())
        dp.validate_result(first, 2)

        # 新虚拟包拥有独立模块状态，只通过磁盘复用之前的结果。
        restarted = load_algorithm("kyn_cache_restart_tests")
        with mock.patch.object(restarted, "RESULTS_PATH", self.path), mock.patch.object(
            restarted, "_solve", side_effect=AssertionError("最优缓存不应搜索")
        ):
            second = restarted.generate_polynomial(2)
        for key in ("expression", "cycles", "exhaustive_through_cycles"):
            self.assertEqual(first[key], second[key])
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertEqual(set(first), set(second))
        self.assertGreaterEqual(second["elapsed_ms"], 0)
        dp.validate_result(second, 2)

    def test_unproven_result_is_used_as_initial_bound(self):
        cached = self.four_cycle_record()
        target = cached["target"]
        self.write_records({str(target): cached})
        with mock.patch.object(algorithm, "_solve", wraps=dp.solve) as solve:
            result = algorithm.generate_polynomial(target, self.zero_limits)
        solve.assert_called_once()
        self.assertEqual(solve.call_args.kwargs["initial_result"], cached)
        direct = dp.Solver().solve(target, self.zero_limits, initial_result=cached)
        for candidate in (result, direct, self.read_records()[str(target)]):
            dp.validate_result(candidate, target)
            self.assertLessEqual(
                (candidate["cycles"], len(candidate["expression"])),
                (cached["cycles"], len(cached["expression"])),
            )
            self.assertEqual(candidate["cycles"], 4)
            self.assertEqual(candidate["exhaustive_through_cycles"], 2)
            self.assertFalse(candidate["optimal"])

    def test_old_proof_is_only_upper_bound(self):
        # 历史证明不可复用，但独立校验通过的表达式仍应作为搜索上界。
        old = {
            "target": 1, "expression": "9+4+7+9-4*7", "cycles": 2,
            "digits_used": 6, "verified": True, "optimal": True,
            "method": "exact_dp", "elapsed_ms": 1,
            "search_exhaustive_through_cycles": 1,
        }
        current = record(1, old["expression"], 2, 1, optimal=True)
        missing_status = dict(current)
        del missing_status["proof_status"]
        variants = (
            old,
            dict(current, proof_version="unknown"),
            dict(current, rules_version="unknown"),
            missing_status,
            dict(current, proof_status="complete?"),
        )
        for cached in variants:
            with self.subTest(cached=cached):
                self.write_records({"1": cached})
                with mock.patch.object(algorithm, "_solve", wraps=dp.solve) as solve:
                    result = algorithm.generate_polynomial(1, self.zero_limits)
                solve.assert_called_once()
                initial = solve.call_args.kwargs["initial_result"]
                self.assertEqual(initial["expression"], cached["expression"])
                self.assertFalse(initial["optimal"])
                self.assertEqual(initial["proof_status"], "unproven")
                self.assertEqual(initial["exhaustive_through_cycles"], 0)
                dp.validate_result(result, 1)

    def test_invalid_cache_is_recomputed_without_losing_other_targets(self):
        preserved = record(-3, "9/(4-7)", 1, optimal=True)
        for expression in ("9+4+7", "94+7"):
            with self.subTest(expression=expression):
                self.write_records({"2": record(2, expression, 1, optimal=True),
                                    "-3": preserved})
                with mock.patch.object(algorithm, "_solve", wraps=dp.solve) as solve:
                    result = algorithm.generate_polynomial(2)
                solve.assert_called_once()
                dp.validate_result(result, 2)
                saved = self.read_records()
                dp.validate_result(saved["2"], 2)
                self.assertEqual(saved["-3"], preserved)
                self.assertFalse(self.path.with_name("results.json.tmp").exists())

    def test_merge_preserves_better_expression_and_combines_proof(self):
        cached = self.four_cycle_record()
        target = cached["target"]
        self.write_records({str(target): cached})
        longer_bound = dict(cached, expression="(" + cached["expression"] + ")")
        with mock.patch.object(algorithm, "_solve", return_value=longer_bound):
            result = algorithm.generate_polynomial(target, self.zero_limits)
        self.assertEqual(result["expression"], cached["expression"])
        dp.validate_result(self.read_records()[str(target)], target)

        # 使用真实求解得到的两组最优解及一组穷尽证明，避免伪造证明进度。
        proven = dp.solve(1)
        self.assertEqual(proven["cycles"], 2)
        self.assertTrue(proven["optimal"])
        shorter = dict(proven, **dp.proof_metadata(proven["cycles"], 0))
        longer = dict(proven, expression="(" + proven["expression"] + ")")
        self.write_records({"1": shorter})
        with mock.patch.object(algorithm, "_solve", return_value=longer):
            merged = algorithm.generate_polynomial(1, self.zero_limits)
        self.assertEqual(merged["expression"], shorter["expression"])
        self.assertEqual(merged["exhaustive_through_cycles"],
                         proven["exhaustive_through_cycles"])
        self.assertTrue(merged["optimal"])
        dp.validate_result(self.read_records()["1"], 1)

    def test_concurrent_requests_search_only_once(self):
        with mock.patch.object(algorithm, "_solve", wraps=dp.solve) as solve:
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(algorithm.generate_polynomial, (2, 2)))
        solve.assert_called_once()
        for result in results:
            dp.validate_result(result, 2)
            self.assertTrue(result["optimal"])
        dp.validate_result(self.read_records()["2"], 2)
        self.assertFalse(self.path.with_name("results.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
