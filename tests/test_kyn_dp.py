"""947 表达式求解器的独立精确校验。"""

import importlib.util
import random
import sys
import unittest
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from time import perf_counter


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/plugins/kyn/dp.py"
SPEC = importlib.util.spec_from_file_location("kyn_dp_standalone_tests", MODULE_PATH)
dp = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dp
SPEC.loader.exec_module(dp)


SAMPLES = (
    (-3, "9/(4-7)", 1),
    (2, "-(9-4-7)", 1),
    (0, "9+4+7-(9+4+7)", 2),
    (1, "9+4+7+9-4*7", 2),
    (47, "9+4+7-9*(4-7)", 2),
    (94, "(9+4)*7-9/(4-7)", 2),
    (947, "9*4*(7+9)+4*7*(9+4)+7", 3),
    (2026, "9*4*(7*9-4)-(7+(9+4)*7)", 3),
)


@lru_cache(maxsize=None)
def independent_values(digits):
    """直接枚举所有切分与正负值，不使用被测模块的运算。"""
    if len(digits) == 1:
        return frozenset((Fraction(digits[0]), Fraction(-digits[0])))
    values = set()
    for split in range(1, len(digits)):
        for left in independent_values(digits[:split]):
            for right in independent_values(digits[split:]):
                results = [left + right, left - right, left * right]
                if right:
                    results.append(left / right)
                for value in results:
                    values.add(value)
                    values.add(-value)
    return frozenset(values)


def inspect_tree(root):
    """迭代计算值和叶顺序，避免深表达式受 Python 递归深度限制。"""
    pending = [(root, False)]
    values = []
    digits = []
    while pending:
        node, visited = pending.pop()
        if isinstance(node, dp.Digit):
            digit = int(node.value)
            digits.append(digit)
            values.append(Fraction(digit))
        elif not visited:
            pending.append((node, True))
            if isinstance(node, dp.Neg):
                pending.append((node.child, False))
            else:
                pending.append((node.right, False))
                pending.append((node.left, False))
        elif isinstance(node, dp.Neg):
            values.append(-values.pop())
        else:
            right = values.pop()
            left = values.pop()
            if node.op == "+":
                values.append(left + right)
            elif node.op == "-":
                values.append(left - right)
            elif node.op == "*":
                values.append(left * right)
            elif node.op == "/":
                values.append(left / right)
            else:
                raise AssertionError("出现未知运算符")
    if len(values) != 1:
        raise AssertionError("表达式树结构不完整")
    return values[0], digits


class KynDPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = dp.Solver()
        cls.solver.precompute(dp.SearchLimits(
            timeout_seconds=60,
            max_states_per_interval=20000,
            max_pair_expansions=5000000,
            max_memory_mb=256,
            max_cycles_search=4,
        ))

    def assert_roundtrip(self, node, target):
        self.assertTrue(dp.validate(node, target))
        reparsed = dp.parse_expression(dp.render(node))
        self.assertTrue(dp.validate(reparsed, target))
        value, digits = inspect_tree(reparsed)
        self.assertEqual(value, Fraction(target))
        self.assertEqual(digits, [9, 4, 7] * (len(digits) // 3))
        self.assertEqual(len(digits), reparsed.digits_used)

    def test_specification_and_optimal_solutions(self):
        limits = dp.SearchLimits(
            timeout_seconds=5,
            max_states_per_interval=20000,
            max_pair_expansions=5000000,
            max_memory_mb=256,
            max_cycles_search=3,
        )
        for target, expression, cycles in SAMPLES:
            with self.subTest(target=target):
                node = dp.parse_expression(expression)
                self.assertEqual(node.digits_used, 3 * cycles)
                self.assert_roundtrip(node, target)
                result = self.solver.solve(target, limits)
                self.assertEqual(result["target"], target)
                self.assertTrue(result["verified"])
                self.assertTrue(result["optimal"])
                self.assertEqual(result["cycles"], cycles)
                self.assertEqual(result["digits_used"], cycles * 3)
                self.assertTrue(result["method"])
                self.assertGreaterEqual(result["elapsed_ms"], 0)
                self.assert_roundtrip(dp.parse_expression(result["expression"]), target)
                if target in (947, 2026):
                    self.assertGreaterEqual(result["exhaustive_through_cycles"], 2)

    def test_exact_value_sets(self):
        for phase in range(3):
            for length in range(1, 6):
                with self.subTest(phase=phase, length=length):
                    digits = tuple((9, 4, 7)[(phase + i) % 3] for i in range(length))
                    self.assertEqual(set(self.solver.exact_values(phase, length)),
                                     independent_values(digits))
        one_cycle = independent_values((9, 4, 7))
        two_cycles = independent_values((9, 4, 7, 9, 4, 7))
        self.assertEqual(len(one_cycle), 48)
        self.assertEqual(len(two_cycles), 11041)
        self.assertEqual(set(self.solver.exact_values(0, 6)), two_cycles)
        for target in (947, 2026):
            self.assertNotIn(Fraction(target), one_cycle)
            self.assertNotIn(Fraction(target), two_cycles)

    def test_exact_cache_rule_version(self):
        limits = dp.SearchLimits(
            timeout_seconds=15,
            max_states_per_interval=20000,
            max_pair_expansions=5000000,
            max_memory_mb=256,
            max_cycles_search=3,
        )
        interval = self.solver._cache[(0, 6)]
        for target in (947, 2026):
            with self.subTest(target=target):
                self.solver.solve(target, limits)
                self.assertIs(self.solver._cache[(0, 6)], interval)
                self.assertTrue(interval.complete)
                self.assertEqual(interval.rules_version, dp.RULES_VERSION)

        # 规则版本过期或枚举未完成时，必须重建区间，不能沿用旧证明。
        solver = dp.Solver()
        leaf_limits = dp.SearchLimits(timeout_seconds=5, max_cycles_search=2)
        budget = dp._Budget(leaf_limits, perf_counter(), memory=0)
        outdated = solver._exact(0, 1, budget)
        outdated.rules_version = "unknown"
        rebuilt = solver._exact(0, 1, budget)
        self.assertIsNot(rebuilt, outdated)
        self.assertEqual(rebuilt.rules_version, dp.RULES_VERSION)
        self.assertEqual(set(rebuilt.states), {Fraction(9), Fraction(-9)})
        rebuilt.complete = False
        complete = solver._exact(0, 1, budget)
        self.assertIsNot(complete, rebuilt)
        self.assertTrue(complete.complete)
        self.assertEqual(complete.rules_version, dp.RULES_VERSION)
        self.assertEqual(set(complete.states), {Fraction(9), Fraction(-9)})

    def test_invalid_expressions(self):
        expressions = (
            "94+7", "9+47", "4+9+7", "9+4", "9+4+7+9",
            "9**4+7", "9//4+7", "0x9+4+7", "9/0+4+7",
            "9/(4-4)+7", "9+4+7x",
        )
        for expression in expressions:
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                dp.validate(dp.parse_expression(expression), 0)

    def test_zero_budget_fallback(self):
        solver = dp.Solver()
        for budget in ("timeout_seconds", "max_pair_expansions",
                       "max_states_per_interval", "max_memory_mb"):
            for target in (947, 2026):
                with self.subTest(budget=budget, target=target):
                    result = solver.solve(target, dp.SearchLimits(**{budget: 0}))
                    self.assertTrue(result["verified"])
                    self.assertFalse(result["optimal"])
                    self.assertTrue(result["method"])
                    self.assertEqual(result["exhaustive_through_cycles"], 0)
                    self.assert_roundtrip(dp.parse_expression(result["expression"]), target)

    def test_search_limit_validation(self):
        for field in ("beam_width", "beam_max_pair_expansions_per_interval",
                      "budget_check_interval"):
            for value in (0, -1, True, 1.5):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    dp.SearchLimits(**{field: value})
        for field in ("timeout_seconds", "max_memory_mb"):
            for value in (True, float("nan"), float("inf"), -1):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    dp.SearchLimits(**{field: value})

    def test_custom_beam_limits(self):
        limits = dp.SearchLimits(
            timeout_seconds=15,
            max_states_per_interval=20000,
            max_pair_expansions=1000000,
            max_memory_mb=256,
            max_cycles_search=4,
            beam_width=3,
            beam_max_pair_expansions_per_interval=2,
            budget_check_interval=1,
        )
        budget = dp._Budget(
            limits, perf_counter(), self.solver._memory(),
            target=947, best=dp.fallback(947),
        )
        interval = self.solver._beam(0, 7, budget, {})
        self.assertFalse(interval.complete)
        self.assertGreater(len(interval.states), 0)
        self.assertLessEqual(len(interval.states), 3)
        self.assertGreater(budget.expansions, 0)
        self.assertLessEqual(budget.expansions, 2)

    def test_bounded_search(self):
        limits = dp.SearchLimits(
            timeout_seconds=15,
            max_states_per_interval=20000,
            max_pair_expansions=1000000,
            max_memory_mb=256,
            max_cycles_search=4,
        )
        result = self.solver.solve(525876624, limits)
        self.assertTrue(result["verified"])
        self.assertLessEqual(result["cycles"], 4)
        if result["cycles"] == 4:
            self.assertFalse(result["optimal"])
            self.assertEqual(result["exhaustive_through_cycles"], 2)
        self.assert_roundtrip(dp.parse_expression(result["expression"]), 525876624)

        target = 10**30 + 947
        limited = dp.SearchLimits(
            timeout_seconds=15,
            max_states_per_interval=20000,
            max_pair_expansions=6000,
            max_memory_mb=256,
            max_cycles_search=4,
        )
        before = self.solver.statistics()
        result = self.solver.solve(target, limited)
        after = self.solver.statistics()
        self.assertTrue(result["verified"])
        self.assertFalse(result["optimal"])
        self.assertTrue(result["method"])
        self.assertEqual(result["exhaustive_through_cycles"], 2)
        self.assertFalse(after["complete"])
        expansions = after["pair_expansions"] - before["pair_expansions"]
        self.assertGreater(expansions, 0)
        self.assertLessEqual(expansions, 6000)
        self.assert_roundtrip(dp.parse_expression(result["expression"]), target)

    def test_partial_budget_fallback(self):
        limits_to_check = (
            dp.SearchLimits(max_pair_expansions=1),
            dp.SearchLimits(max_states_per_interval=4),
            dp.SearchLimits(max_memory_mb=0.01),
        )
        for limits in limits_to_check:
            with self.subTest(limits=limits):
                result = dp.Solver().solve(947, limits)
                self.assertTrue(result["verified"])
                if result["optimal"]:
                    self.assertGreaterEqual(result["exhaustive_through_cycles"],
                                            result["cycles"] - 1)
                    self.assertEqual(result["proof_status"], "optimal")
                else:
                    self.assertNotEqual(result["proof_status"], "optimal")
                self.assertTrue(result["method"])
                self.assert_roundtrip(dp.parse_expression(result["expression"]), 947)

    def test_solver_large_integers(self):
        rng = random.Random(947)
        limits = dp.SearchLimits(timeout_seconds=0)
        for bits in (64, 256, 2048):
            for sign in (-1, 1):
                target = sign * (rng.getrandbits(bits) | (1 << (bits - 1)))
                with self.subTest(bits=bits, sign=sign):
                    result = self.solver.solve(target, limits)
                    self.assertTrue(result["verified"])
                    self.assertFalse(result["optimal"])
                    self.assert_roundtrip(dp.parse_expression(result["expression"]), target)

    def test_fallback_integer_range_and_large_integers(self):
        for target in range(-1000, 1001):
            with self.subTest(target=target):
                self.assert_roundtrip(dp.fallback(target), target)
        rng = random.Random(947)
        for bits in (64, 256, 2048):
            for sign in (-1, 1):
                target = sign * (rng.getrandbits(bits) | (1 << (bits - 1)))
                with self.subTest(bits=bits, sign=sign):
                    self.assert_roundtrip(dp.fallback(target), target)
        self.assert_roundtrip(dp.fallback(1 << 2048), 1 << 2048)

    def test_printer_parentheses(self):
        nine, four, seven = dp.Digit(9), dp.Digit(4), dp.Digit(7)
        nodes = (
            dp.Binary("-", nine, dp.Binary("-", four, seven)),
            dp.Binary("/", nine, dp.Binary("*", four, seven)),
            dp.Binary("/", nine, dp.Binary("/", four, seven)),
            dp.Neg(dp.Binary("+", nine, dp.Binary("-", four, seven))),
            dp.Binary("*", dp.Neg(nine), dp.Neg(dp.Binary("+", four, seven))),
        )
        for node in nodes:
            with self.subTest(expression=dp.render(node)):
                self.assertEqual(inspect_tree(dp.parse_expression(dp.render(node))),
                                 inspect_tree(node))


if __name__ == "__main__":
    unittest.main()
