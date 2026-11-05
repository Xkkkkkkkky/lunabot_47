"""验证确定性构造的余数覆盖、整数范围和自适应搜索上限。"""

import importlib.util
import random
import sys
import unittest
from fractions import Fraction
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/plugins/kyn/dp.py"
SPEC = importlib.util.spec_from_file_location("kyn_construct_standalone_tests", MODULE_PATH)
dp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dp
SPEC.loader.exec_module(dp)


class KynConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = dp.Solver()
        cls.solver.prepare_constructions()

    def assert_roundtrip(self, node, target):
        self.assertTrue(dp.validate(node, target))
        self.assertTrue(dp.validate(dp.parse_expression(dp.render(node)), target))

    def test_remainders(self):
        report = self.solver.remainder_report()
        self.assertEqual(report["coverage"], 252)
        self.assertEqual(report["max_cycles"], 3)
        one_cycle = self.solver.exact_values(0, 3)
        two_cycles = self.solver.exact_values(0, 6)
        for target in range(252):
            with self.subTest(target=target):
                candidates = self.solver.construction_candidates(target)
                self.assertEqual(set(candidates), {"radix252", "balanced252", "negabinary"})
                for method, node in candidates.items():
                    with self.subTest(method=method):
                        self.assert_roundtrip(node, target)
                # 比较已穷尽的一、二组集合，只主张组数最优，不主张字符最短。
                expected = (1 if Fraction(target) in one_cycle else
                            2 if Fraction(target) in two_cycles else 3)
                best = self.solver.construct(target)
                self.assertEqual(best.digits_used // 3, expected)
                self.assert_roundtrip(best, target)

    def test_large_constructions(self):
        rng = random.Random(947)
        targets = [0, -1, 10**20 - 1, -(10**20 - 1)]
        for offset in (-1, 0, 1):
            targets.extend((252**8 + offset, -(252**8 + offset)))
        for digits in (6, 12, 18, 20):
            for _ in range(4):
                value = rng.randrange(10**(digits - 1), 10**digits)
                targets.extend((value, -value))
        for target in targets:
            with self.subTest(target=target):
                candidates = self.solver.construction_candidates(target)
                for method, node in candidates.items():
                    with self.subTest(method=method):
                        self.assert_roundtrip(node, target)
                best = self.solver.construct(target)
                self.assert_roundtrip(best, target)
                self.assertLessEqual(best.digits_used,
                                     candidates["negabinary"].digits_used)
                if len(str(abs(target))) == 20:
                    self.assertLessEqual(candidates["radix252"].digits_used // 3, 35)

    def test_adaptive_cap(self):
        for target in (0, 1, -1, 947, 10**5, -(10**12), 10**19, 10**20 - 1):
            with self.subTest(target=target):
                # 同一十进制位数统一采用该位数可表示的最大绝对值。
                maximum = 10**len(str(abs(target))) - 1
                length = 1
                while maximum >= 252:
                    maximum //= 252
                    length += 1
                expected = 4 * length - 1
                self.assertEqual(dp.adaptive_cycle_limit(target, max_remainder_cycles=3),
                                 expected)
        self.assertGreaterEqual(dp.adaptive_cycle_limit(0, max_remainder_cycles=3), 2)
        for target in (10**19, 10**20 - 1):
            self.assertEqual(dp.adaptive_cycle_limit(target, max_remainder_cycles=3), 35)
        limits = dp.SearchLimits(timeout_seconds=0, max_cycles_search="a")
        with self.assertRaises(ValueError):
            dp.SearchLimits(max_cycles_search="x")
        result = self.solver.solve(10**20 - 1, limits)
        self.assertEqual(result["effective_max_cycles_search"], 35)
        self.assertLessEqual(result["search_upper_cycles"],
                             min(35, result["initial_cycles"] - 1))
        self.assertTrue(result["verified"])
        self.assertFalse(result["optimal"])
        self.assertEqual(result["proof_status"], "unproven")
        self.assertEqual(result["exhaustive_through_cycles"], 0)
        dp.validate_result(result, 10**20 - 1)


if __name__ == "__main__":
    unittest.main()
