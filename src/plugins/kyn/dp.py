"""947 整数表达式求解器：精确 DP、逆向匹配、有界搜索与负二进制构造。

所有运算使用 Fraction。搜索预算只约束搜索；构造、打印和独立验证始终执行。
每个数值保留一个代表，因此 optimal 仅表示循环数最少，不表示字符串最短。
max_memory_mb 按缓存及新增保留节点保守估算，不代表整个机器人进程的内存上限。
"""

from __future__ import annotations

import heapq
import math
import sys
import threading
from dataclasses import dataclass, field, replace
from fractions import Fraction
from time import perf_counter
from types import MappingProxyType
from typing import Iterable, Mapping


PERIOD = (9, 4, 7)
RULES_VERSION = "947-fraction-v1"
PROOF_VERSION = "exact-prefix-v2"
_MASK = (1 << 64) - 1
_OP_ID = {"+": 1, "-": 2, "*": 3, "/": 4}
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}


def _calculate(op: str, left: Fraction, right: Fraction) -> Fraction:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        if not right:
            raise ValueError("表达式除数不能为零")
        return left / right
    raise ValueError("表达式包含非法运算符")


@dataclass(frozen=True, slots=True, eq=False)
class Digit:
    digit: int
    value: Fraction = field(init=False)
    digits_used: int = field(init=False, default=1)
    print_cost: int = field(init=False, default=1)
    stable_hash: int = field(init=False)

    def __post_init__(self):
        if type(self.digit) is not int or self.digit not in PERIOD:
            raise ValueError("数字叶节点只能是单个 9、4、7")
        object.__setattr__(self, "value", Fraction(self.digit))
        object.__setattr__(self, "stable_hash", self.digit)


@dataclass(frozen=True, slots=True, eq=False)
class Neg:
    child: Expr
    value: Fraction = field(init=False)
    digits_used: int = field(init=False)
    print_cost: int = field(init=False)
    stable_hash: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "value", -self.child.value)
        object.__setattr__(self, "digits_used", self.child.digits_used)
        object.__setattr__(self, "print_cost", 1 + self.child.print_cost + 2 * isinstance(self.child, Binary))
        object.__setattr__(self, "stable_hash", (self.child.stable_hash * 1000003 + 5) & _MASK)


def _precedence(node: Expr) -> int:
    if isinstance(node, Binary):
        return _PRECEDENCE[node.op]
    return 3 if isinstance(node, Neg) else 4


def _parentheses(op: str, node: Expr, right: bool) -> bool:
    level = _precedence(node)
    parent_level = _PRECEDENCE[op]
    return level < parent_level or (right and level == parent_level and op in ("-", "/"))


def _binary_cost(op: str, left: Expr, right: Expr) -> int:
    return (left.print_cost + right.print_cost + 1
            + 2 * _parentheses(op, left, False) + 2 * _parentheses(op, right, True))


@dataclass(frozen=True, slots=True, eq=False)
class Binary:
    op: str
    left: Expr
    right: Expr
    value: Fraction = field(init=False)
    digits_used: int = field(init=False)
    print_cost: int = field(init=False)
    stable_hash: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "value", _calculate(self.op, self.left.value, self.right.value))
        object.__setattr__(self, "digits_used", self.left.digits_used + self.right.digits_used)
        object.__setattr__(self, "print_cost", _binary_cost(self.op, self.left, self.right))
        object.__setattr__(self, "stable_hash", (
            self.left.stable_hash * 1000003 + self.right.stable_hash * 1000033 + _OP_ID[self.op]
        ) & _MASK)


Expr = Digit | Neg | Binary


def _negative(node: Expr) -> Expr:
    # 消除双重负号不会改变任何数字叶节点及其顺序。
    return node.child if isinstance(node, Neg) else Neg(node)


def render(node: Expr) -> str:
    """按优先级打印；使用显式栈，支持很深的霍纳表达式。"""
    output: list[str] = []
    stack: list[Expr | str] = [node]
    while stack:
        part = stack.pop()
        if isinstance(part, str):
            output.append(part)
        elif isinstance(part, Digit):
            output.append(str(part.digit))
        elif isinstance(part, Neg):
            if isinstance(part.child, Binary):
                stack.extend((")", part.child, "-("))
            else:
                stack.extend((part.child, "-"))
        elif isinstance(part, Binary):
            right = (")", part.right, "(") if _parentheses(part.op, part.right, True) else (part.right,)
            left = (")", part.left, "(") if _parentheses(part.op, part.left, False) else (part.left,)
            stack.extend((*right, part.op, *left))
        else:
            raise ValueError("表达式包含未知节点")
    return "".join(output)


def parse_expression(text: str) -> Expr:
    """严格解析允许的语法，不使用 eval，也不受 Python 括号深度限制。"""
    nodes: list[Expr] = []
    operators: list[str] = []
    need_operand = True

    def apply():
        op = operators.pop()
        if op == "~":
            if not nodes:
                raise ValueError("一元负号缺少操作数")
            nodes.append(Neg(nodes.pop()))
        else:
            if len(nodes) < 2:
                raise ValueError("二元运算缺少操作数")
            right, left = nodes.pop(), nodes.pop()
            nodes.append(Binary(op, left, right))

    for char in text:
        if char.isspace():
            continue
        if char in "947":
            if not need_operand:
                raise ValueError("不允许拼接多位数字或省略运算符")
            nodes.append(Digit(int(char)))
            need_operand = False
            while operators and operators[-1] == "~":
                apply()
        elif char == "(":
            if not need_operand:
                raise ValueError("括号前缺少运算符")
            operators.append(char)
        elif char == ")":
            if need_operand:
                raise ValueError("括号内缺少操作数")
            while operators and operators[-1] != "(":
                apply()
            if not operators:
                raise ValueError("括号不匹配")
            operators.pop()
            while operators and operators[-1] == "~":
                apply()
        elif char in _PRECEDENCE:
            if need_operand:
                if char != "-":
                    raise ValueError("操作数位置只允许数字、括号或一元负号")
                operators.append("~")
            else:
                while operators and operators[-1] != "(" and (
                    operators[-1] == "~" or _PRECEDENCE[operators[-1]] >= _PRECEDENCE[char]
                ):
                    apply()
                operators.append(char)
                need_operand = True
        else:
            raise ValueError("表达式包含非法字符")
    if need_operand:
        raise ValueError("表达式为空或缺少操作数")
    while operators:
        if operators[-1] == "(":
            raise ValueError("括号不匹配")
        apply()
    if len(nodes) != 1:
        raise ValueError("表达式结构不完整")
    return nodes[0]


def validate(node: Expr, target: int) -> bool:
    """独立遍历并重新计算，不能用搜索器缓存的值代替验证。"""
    values: dict[int, Fraction] = {}
    digits = 0
    stack = [(node, False)]
    while stack:
        current, visited = stack.pop()
        if isinstance(current, Digit):
            if type(current.digit) is not int or current.digit != PERIOD[digits % 3]:
                raise ValueError("数字顺序必须严格为完整重复的 947")
            digits += 1
            value = Fraction(current.digit)
        elif isinstance(current, (Neg, Binary)):
            if not visited:
                stack.append((current, True))
                if isinstance(current, Neg):
                    stack.append((current.child, False))
                else:
                    stack.extend(((current.right, False), (current.left, False)))
                continue
            if isinstance(current, Neg):
                value = -values[id(current.child)]
            else:
                # 此处显式计算，独立于构造器和搜索使用的 _calculate。
                a, b = values[id(current.left)], values[id(current.right)]
                if current.op == "+":
                    value = a + b
                elif current.op == "-":
                    value = a - b
                elif current.op == "*":
                    value = a * b
                elif current.op == "/" and b:
                    value = a / b
                else:
                    raise ValueError("表达式含非法运算或零除")
        else:
            raise ValueError("表达式包含未知节点")
        if current.value != value:
            raise ValueError("节点缓存与独立求值不一致")
        values[id(current)] = value
    if not digits or digits % 3 or digits != node.digits_used:
        raise ValueError("数字必须组成完整的 947 循环")
    if values[id(node)] != target:
        raise ValueError("表达式不等于目标整数")
    return True


def proof_metadata(cycles: int, exhaustive: int, confirmed: bool = True) -> dict:
    """只有当前版本确认过的完整排除前缀，才能用于组数最优性证明。"""
    if type(exhaustive) is not int or not 0 <= exhaustive < cycles:
        raise ValueError("最优性证明范围无效")
    if not confirmed:
        exhaustive = 0
    optimal = confirmed and exhaustive >= cycles - 1
    status = "optimal" if optimal else ("partial" if exhaustive else "unproven")
    return dict(rules_version=RULES_VERSION, proof_version=PROOF_VERSION,
                exhaustive_through_cycles=exhaustive, proof_status=status, optimal=optimal)


def normalize_result(result: dict, target: int) -> dict:
    """历史表达式独立验证后可用；不明或过期证明仅降为合法上界。"""
    if not isinstance(result, dict) or type(result.get("target")) is not int or result["target"] != target:
        raise ValueError("缓存目标与查询整数不一致")
    if not isinstance(result.get("expression"), str):
        raise ValueError("缓存缺少表达式")
    node = parse_expression(result["expression"])
    validate(node, target)
    cycles = node.digits_used // 3
    prefix = result.get("exhaustive_through_cycles")
    status = result.get("proof_status")
    current = (result.get("rules_version") == RULES_VERSION
               and result.get("proof_version") == PROOF_VERSION
               and type(prefix) is int and 0 <= prefix < cycles)
    confirmed = current and (
        (status == "optimal" and result.get("optimal") is True and prefix >= cycles - 1)
        or (status == "partial" and result.get("optimal") is False and 0 < prefix < cycles - 1))
    elapsed = result.get("elapsed_ms", 0)
    if (type(elapsed) not in (int, float) or elapsed < 0
            or (type(elapsed) is float and not math.isfinite(elapsed))):
        elapsed = 0
    normalized = dict(target=target, expression=render(node), cycles=cycles,
                      digits_used=node.digits_used, verified=True,
                      method=result.get("method") if isinstance(result.get("method"), str) else "historical",
                      elapsed_ms=elapsed)
    normalized.update(proof_metadata(cycles, prefix if confirmed else 0, confirmed))
    return normalized


def validate_result(result: dict, target: int) -> Expr:
    """验证当前格式；历史记录需先 normalize_result，不能把旧 optimal 当作证明。"""
    normalized = normalize_result(result, target)
    for key in ("cycles", "digits_used", "verified", "rules_version", "proof_version",
                "exhaustive_through_cycles", "proof_status", "optimal"):
        if type(result.get(key)) is not type(normalized[key]) or result.get(key) != normalized[key]:
            raise ValueError(f"结果字段 {key} 与表达式或当前证明规则不一致")
    node = parse_expression(result["expression"])
    return node


def fallback(target: int) -> Expr:
    """按负二进制霍纳法构造任意整数，始终保留完整的数字循环。"""
    if type(target) is not int:
        raise TypeError("目标必须是整数")
    a = Binary("-", Binary("-", Digit(9), Digit(4)), Digit(7))
    b = Binary("-", Binary("+", Binary("+", Binary("+", Digit(9), Digit(4)), Digit(7)), Digit(9)),
               Binary("*", Digit(4), Digit(7)))
    if target == 0:
        return Binary("-", a, a)
    bits = []
    while target:
        bit = target % 2
        bits.append(bit)
        target = (target - bit) // -2
    expression = b
    for bit in reversed(bits[:-1]):
        expression = Binary("*", expression, a)
        if bit:
            expression = Binary("+", expression, b)
    return expression


def adaptive_cycle_limit(target: int, max_remainder_cycles: int = 3) -> int:
    """按十进制位数的最大值给出普通 252 进制的保守构造上界。"""
    if type(target) is not int or type(max_remainder_cycles) is not int or max_remainder_cycles <= 0:
        raise ValueError("自适应上限需要整数目标和正的余数组数上界")
    largest = 10 ** len(str(abs(target))) - 1
    places = 0
    while largest:
        places += 1
        largest //= 252
    return max(2, max(1, places) * max_remainder_cycles + max(0, places - 1))


@dataclass(frozen=True, slots=True)
class SearchLimits:
    timeout_seconds: float = 3.0
    max_states_per_interval: int = 16000
    max_pair_expansions: int = 1000000
    max_memory_mb: float = 128.0
    max_cycles_search: int | str = "a"
    beam_width: int = 128
    beam_max_pair_expansions_per_interval: int = 1024
    budget_check_interval: int = 64
    quotient_candidates: int = 24
    quotient_nodes: int = 256

    def __post_init__(self):
        for name in ("timeout_seconds", "max_memory_mb"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限的非负数")
        for name in ("max_states_per_interval", "max_pair_expansions"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.max_cycles_search != "a" and (type(self.max_cycles_search) is not int or self.max_cycles_search < 0):
            raise ValueError("max_cycles_search 必须是非负整数或 a")
        for name in ("beam_width", "beam_max_pair_expansions_per_interval", "budget_check_interval",
                     "quotient_candidates", "quotient_nodes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} 必须是正整数")


class _BudgetExceeded(Exception):
    pass


@dataclass(slots=True)
class _Interval:
    states: dict[Fraction, Expr]
    complete: bool
    rules_version: str = RULES_VERSION


def _node_bytes(node: Expr) -> int:
    return (sys.getsizeof(node) + sys.getsizeof(node.value)
            + sys.getsizeof(node.value.numerator) + sys.getsizeof(node.value.denominator))


class _Budget:
    def __init__(self, limits: SearchLimits, start: float, memory: int,
                 target: int | None = None, best: Expr | None = None):
        self.limits = limits
        self.deadline = start + limits.timeout_seconds
        self.start = start
        self.memory_bytes = memory
        self.peak_memory = memory
        self.target = target
        self.best = best
        self.method = "fallback"
        self.expansions = 0
        self.max_states = 0
        self.truncated = 0
        self.complete = True
        self.initial_cycles = best.digits_used // 3 if best is not None else 0
        self.first_improvement_ms = None
        self.stop_reason = "search_exhausted"

    def check(self):
        reason = None
        if perf_counter() >= self.deadline:
            reason = "timeout"
        elif not self.limits.max_pair_expansions:
            reason = "pair_limit"
        elif not self.limits.max_states_per_interval:
            reason = "state_limit"
        elif not self.limits.max_cycles_search:
            reason = "cycle_limit"
        elif self.memory_bytes > self.limits.max_memory_mb * 1024 * 1024 or not self.limits.max_memory_mb:
            reason = "memory_limit"
        if reason:
            self.stop_reason = reason
            self.complete = False
            raise _BudgetExceeded

    def pair(self):
        if self.expansions >= self.limits.max_pair_expansions:
            self.stop_reason = "pair_limit"
            self.complete = False
            raise _BudgetExceeded
        self.expansions += 1
        if self.expansions % self.limits.budget_check_interval == 0:
            self.check()

    def store(self, states: dict[Fraction, Expr], value: Fraction, node: Expr):
        if value not in states and len(states) >= self.limits.max_states_per_interval:
            self.stop_reason = "state_limit"
            self.complete = False
            raise _BudgetExceeded
        # 保守累计新保留节点和哈希项；替换、剪枝暂不扣除，避免低估搜索内存。
        self.memory_bytes += 512 + _node_bytes(node)
        if isinstance(node, Neg):
            self.memory_bytes += _node_bytes(node.child)
        self.peak_memory = max(self.peak_memory, self.memory_bytes)
        self.check()
        states[value] = node
        self.max_states = max(self.max_states, len(states))

    def offer(self, node: Expr, method: str):
        if self.target is None or node.value != self.target or node.digits_used % 3:
            return
        if type(self.limits.max_cycles_search) is int and node.digits_used > 3 * self.limits.max_cycles_search:
            return
        if self.best is None or (node.digits_used, node.print_cost, node.stable_hash) < (
            self.best.digits_used, self.best.print_cost, self.best.stable_hash
        ):
            validate(node, self.target)
            self.best, self.method = node, method
            if node.digits_used // 3 < self.initial_cycles and self.first_improvement_ms is None:
                self.first_improvement_ms = round((perf_counter() - self.start) * 1000, 3)


class Solver:
    """复用已完成的精确区间缓存；目标相关的 Beam 状态仅存活于本次查询。"""

    def __init__(self):
        self._cache: dict[tuple[int, int], _Interval] = {}
        for phase, digit in enumerate(PERIOD):
            node = Digit(digit)
            self._cache[phase, 1] = _Interval({node.value: node, -node.value: Neg(node)}, True)
        self._lock = threading.Lock()
        self._memory_dirty = True
        self._retained_bytes = 0
        self._remainders: dict[int, Expr] = {}
        self._remainder_rules_version = RULES_VERSION
        self._whole_nodes: dict[int, Expr] = {}
        self._stats = dict(cache_hits=0, cache_misses=0, max_states=2, pair_expansions=0,
                           truncated_intervals=0, estimated_memory_bytes=0, complete=True)

    def _prepare_constructions(self):
        if len(self._remainders) == 252 and self._remainder_rules_version == RULES_VERSION:
            return
        # 这是跨查询复用的构造字典初始化，单独测量冷启动，不吞掉目标搜索的时间预算。
        limits = SearchLimits(timeout_seconds=60, max_states_per_interval=20000,
                              max_pair_expansions=5000000, max_memory_mb=256, max_cycles_search=3)
        budget = _Budget(limits, perf_counter(), self._memory())
        for length in range(1, 7):
            for phase in range(3):
                self._exact(phase, length, budget)
        self._record_stats(budget)
        table: dict[int, Expr] = {}
        for remainder in range(252):
            node = self._cache[0, 3].states.get(Fraction(remainder))
            if node is None:
                node = self._cache[0, 6].states.get(Fraction(remainder))
            if node is None:
                match_budget = _Budget(limits, perf_counter(), self._memory(), remainder, fallback(remainder))
                for cut in (4, 5, 3, 6):
                    self._match(self._cache[0, cut].states,
                                self._cache[cut % 3, 9 - cut].states, match_budget, "mitm")
                self._record_stats(match_budget)
                node = match_budget.best
            validate(node, remainder)
            validate(parse_expression(render(node)), remainder)
            if node.digits_used > 9:
                raise RuntimeError(f"余数 {remainder} 未被三组以内的精确 DP/匹配覆盖")
            table[remainder] = node
        self._remainders = table
        whole: dict[int, Expr] = {}
        for length in (3, 6):
            for value, node in self._cache[0, length].states.items():
                if value.denominator == 1:
                    old = whole.get(value.numerator)
                    if old is None or (node.digits_used, node.print_cost) < (old.digits_used, old.print_cost):
                        whole[value.numerator] = node
        self._whole_nodes = whole
        self._remainder_rules_version = RULES_VERSION
        self._memory_dirty = True

    def prepare_constructions(self) -> None:
        with self._lock:
            self._prepare_constructions()

    def remainder_report(self) -> dict:
        self.prepare_constructions()
        entries = []
        for value, node in self._remainders.items():
            validate(node, value)
            cycles = node.digits_used // 3
            entries.append(dict(target=value, cycles=cycles, expression=render(node), verified=True,
                                **proof_metadata(cycles, cycles - 1)))
        return dict(rules_version=RULES_VERSION, proof_version=PROOF_VERSION,
                    coverage=len(entries), max_cycles=max(item["cycles"] for item in entries),
                    entries=entries)

    def _radix252(self, target: int, balanced: bool = False) -> Expr:
        if not target:
            return self._remainders[0]
        negative = target < 0 and not balanced
        value = abs(target) if negative else target
        digits = []
        while value:
            digit = (value + 126) % 252 - 126 if balanced else value % 252
            digits.append(digit)
            value = (value - digit) // 252
        base = Binary("*", Binary("*", Digit(9), Digit(4)), Digit(7))
        def coefficient(value: int) -> Expr:
            node = self._remainders[abs(value)]
            return _negative(node) if value < 0 else node
        expression = coefficient(digits[-1])
        for digit in reversed(digits[:-1]):
            expression = Binary("*", expression, base)
            if digit:
                expression = Binary("+", expression, coefficient(digit))
        return _negative(expression) if negative else expression

    def construction_candidates(self, target: int) -> dict[str, Expr]:
        if type(target) is not int:
            raise TypeError("目标必须是整数")
        self.prepare_constructions()
        candidates = dict(radix252=self._radix252(target), balanced252=self._radix252(target, True),
                          negabinary=fallback(target))
        for node in candidates.values():
            validate(node, target)
            validate(parse_expression(render(node)), target)
        return candidates

    def construct(self, target: int) -> Expr:
        return min(self.construction_candidates(target).values(),
                   key=lambda node: (node.digits_used, node.print_cost, node.stable_hash))

    def _quotient_search(self, budget: _Budget):
        """按 q*d+r 逆向缩短目标，候选按构造成本排序，所有剪枝都不充当穷尽证明。"""
        factors = [(value, node) for value, node in self._whole_nodes.items() if value > 1]
        factors.sort(key=lambda item: (-item[0], item[1].digits_used))
        memo: dict[int, Expr] = {}
        estimates: dict[int, int] = {}
        expanded = 0
        limited = False

        def known(value: int) -> Expr | None:
            node = self._whole_nodes.get(value)
            if node is not None:
                return node
            if abs(value) <= 251:
                node = self._remainders[abs(value)]
                return _negative(node) if value < 0 else node
            return None

        def estimate(value: int) -> int:
            value = abs(value)
            if value in estimates:
                return estimates[value]
            node = known(value)
            if node is not None:
                cost = node.digits_used // 3
            else:
                costs = []
                for balanced in (False, True):
                    current, cost, places = value, 0, 0
                    while current:
                        digit = (current + 126) % 252 - 126 if balanced else current % 252
                        if digit:
                            cost += self._remainders[abs(digit)].digits_used // 3
                        places += 1
                        current = (current - digit) // 252
                    costs.append(cost + max(0, places - 1))
                cost = min(costs)
            estimates[value] = cost
            budget.memory_bytes += 96 + sys.getsizeof(value)
            budget.peak_memory = max(budget.peak_memory, budget.memory_bytes)
            return cost

        def lower(value: int) -> int:
            node = known(value)
            if node is not None:
                return node.digits_used // 3
            # 对任意允许的有理表达式，分子和分母的绝对值均不超过
            # 2^(叶数-1)*各数字之积；这里只用作安全剪枝，不提高 exhaustive 前缀。
            bound, cycles = 2016 // 2, 1
            while abs(value) > bound:
                bound *= 2016
                cycles += 1
            return max(3, cycles)

        def initial(value: int) -> Expr:
            node = known(value)
            if node is None:
                node = min((self._radix252(value), self._radix252(value, True)),
                           key=lambda item: (item.digits_used, item.print_cost, item.stable_hash))
            validate(node, value)
            return node

        def improve(value: int) -> Expr:
            nonlocal expanded, limited
            if value < 0:
                return _negative(improve(-value))
            if value in memo:
                return memo[value]
            node = initial(value)
            memo[value] = node
            if known(value) is not None:
                return node
            budget.check()
            if expanded >= budget.limits.quotient_nodes:
                limited = True
                return node
            expanded += 1
            budget.memory_bytes += node.digits_used * 96
            budget.peak_memory = max(budget.peak_memory, budget.memory_bytes)
            proposals = []
            for divisor, factor in factors:
                quotient, remainder = divmod(value, divisor)
                for q, r in ((quotient, remainder), (quotient + 1, remainder - divisor)):
                    budget.pair()
                    if q <= 0 or q >= value or abs(r) >= value:
                        continue
                    cost_factor = factor.digits_used // 3
                    branch_lower = lower(q) + cost_factor + (lower(r) if r else 0)
                    ceiling = node.digits_used // 3 - 1
                    if value == abs(budget.target):
                        ceiling = min(ceiling, budget.limits.max_cycles_search,
                                      budget.best.digits_used // 3 - 1)
                    if branch_lower > ceiling:
                        continue
                    score = estimate(q) + cost_factor + (estimate(r) if r else 0)
                    proposals.append((score, abs(r), -divisor, q, r, factor))
            proposals.sort(key=lambda item: item[:3])
            for _, _, _, q, r, factor in proposals[:budget.limits.quotient_candidates]:
                budget.check()
                candidate = Binary("*", initial(q), factor)
                if r:
                    candidate = Binary("+", candidate, initial(r))
                if (candidate.digits_used, candidate.print_cost) < (node.digits_used, node.print_cost):
                    node = candidate
                    memo[value] = node
                    if value == abs(budget.target):
                        budget.offer(_negative(node) if budget.target < 0 else node, "quotient")
                # 只在该分支的乐观下界允许改善时，继续搜索两个更小的子目标。
                if lower(q) + factor.digits_used // 3 + (lower(r) if r else 0) >= node.digits_used // 3:
                    continue
                lhs = improve(q)
                rhs = improve(r) if r else None
                candidate = Binary("*", lhs, factor)
                if rhs is not None:
                    candidate = Binary("+", candidate, rhs)
                if (candidate.digits_used, candidate.print_cost) < (node.digits_used, node.print_cost):
                    validate(candidate, value)
                    node = candidate
                    memo[value] = node
                    if value == abs(budget.target):
                        budget.offer(_negative(node) if budget.target < 0 else node, "quotient")
            return node

        node = improve(abs(budget.target))
        budget.offer(_negative(node) if budget.target < 0 else node, "quotient")
        if limited:
            budget.stop_reason = "quotient_node_limit"
        budget.complete = False

    def _memory(self) -> int:
        if not self._memory_dirty:
            return self._retained_bytes
        seen: set[int] = set()
        total = sys.getsizeof(self._cache)
        stack: list[Expr] = []
        for key, interval in self._cache.items():
            total += sys.getsizeof(key) + sys.getsizeof(interval) + sys.getsizeof(interval.states)
            stack.extend(interval.states.values())
        stack.extend(self._remainders.values())
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            total += _node_bytes(node)
            if isinstance(node, Neg):
                stack.append(node.child)
            elif isinstance(node, Binary):
                stack.extend((node.left, node.right))
        self._retained_bytes, self._memory_dirty = total, False
        return total

    @staticmethod
    def _better(node: Expr, old: Expr | None) -> bool:
        return old is None or (node.print_cost, node.stable_hash) < (old.print_cost, old.stable_hash)

    def _insert(self, result: dict[Fraction, Expr], op: str, left: Expr, right: Expr,
                value: Fraction, budget: _Budget, phase: int, length: int, method: str):
        cost = _binary_cost(op, left, right)
        old, old_negative = result.get(value), result.get(-value)
        keep = old is None or cost <= old.print_cost
        keep_negative = old_negative is None or cost + 3 <= old_negative.print_cost
        if not keep and not keep_negative:
            return
        node = Binary(op, left, right)
        candidates = ((value, node), (-value, _negative(node)))
        for number, candidate in candidates:
            if self._better(candidate, result.get(number)):
                if phase == 0 and length % 3 == 0:
                    budget.offer(candidate, method)
                budget.store(result, number, candidate)

    def _exact(self, phase: int, length: int, budget: _Budget) -> _Interval:
        budget.check()
        key = (phase, length)
        if key in self._cache and self._cache[key].complete and self._cache[key].rules_version == RULES_VERSION:
            self._stats["cache_hits"] += 1
            return self._cache[key]
        self._stats["cache_misses"] += 1
        if length == 1:
            node = Digit(PERIOD[phase])
            interval = _Interval({node.value: node, -node.value: Neg(node)}, True, RULES_VERSION)
            self._cache[key] = interval
            self._memory_dirty = True
            return interval
        result: dict[Fraction, Expr] = {}
        for cut in range(1, length):
            left = self._exact(phase, cut, budget).states
            right = self._exact((phase + cut) % 3, length - cut, budget).states
            for a, lhs in left.items():
                for b, rhs in right.items():
                    budget.pair()
                    for op, value in (("+", a + b), ("-", a - b), ("*", a * b)):
                        self._insert(result, op, lhs, rhs, value, budget, phase, length, "exact_dp")
                    if b:
                        self._insert(result, "/", lhs, rhs, a / b, budget, phase, length, "exact_dp")
        interval = _Interval(result, True, RULES_VERSION)
        self._cache[key] = interval
        self._memory_dirty = True
        return interval

    def _match(self, left: Mapping[Fraction, Expr], right: Mapping[Fraction, Expr],
               budget: _Budget, method: str):
        """扫描较小一侧，通过目标反推另一侧的值，包含零值退化情形。"""
        for target, negate in ((Fraction(budget.target), False), (-Fraction(budget.target), True)):
            if len(left) <= len(right):
                any_right = next(iter(right), None)
                nonzero_right = next((b for b in right if b), None)
                for a, lhs in left.items():
                    budget.pair()
                    needs = [("+", target - a), ("-", a - target)]
                    if a:
                        needs.append(("*", target / a))
                    elif not target and any_right is not None:
                        needs.append(("*", any_right))
                    if target:
                        divisor = a / target
                        if divisor:
                            needs.append(("/", divisor))
                    elif not a and nonzero_right is not None:
                        needs.append(("/", nonzero_right))
                    for op, b in needs:
                        rhs = right.get(b)
                        if rhs is not None:
                            node = Binary(op, lhs, rhs)
                            budget.offer(_negative(node) if negate else node, method)
            else:
                any_left = next(iter(left), None)
                for b, rhs in right.items():
                    budget.pair()
                    needs = [("+", target - b), ("-", target + b)]
                    if b:
                        needs.extend((("*", target / b), ("/", target * b)))
                    elif not target and any_left is not None:
                        needs.append(("*", any_left))
                    for op, a in needs:
                        lhs = left.get(a)
                        if lhs is not None:
                            node = Binary(op, lhs, rhs)
                            budget.offer(_negative(node) if negate else node, method)

    @staticmethod
    def _select(states: Mapping[Fraction, Expr], target: int, count: int) -> dict[Fraction, Expr]:
        if len(states) <= count:
            return dict(states)
        items = list(states.items())
        quota = max(1, count // 8)
        # 小整数、正负、低分母、大数、目标与因子附近、稳定散列共同保留多样性。
        keys = (
            lambda item: (item[0].denominator != 1, abs(item[0]), item[1].print_cost),
            lambda item: (item[0] <= 0, abs(item[0]), item[1].print_cost),
            lambda item: (item[0] >= 0, abs(item[0]), item[1].print_cost),
            lambda item: (item[0].denominator, item[1].print_cost),
            lambda item: (-abs(item[0].numerator).bit_length(), item[0].denominator),
            lambda item: (abs(item[0] - target), item[1].print_cost),
            lambda item: (abs(target % item[0].numerator) if item[0] and item[0].denominator == 1
                          else abs(target) + 1, item[1].print_cost),
            lambda item: item[1].stable_hash,
        )
        selected: dict[Fraction, Expr] = {}
        for key in keys:
            for number, node in heapq.nsmallest(quota, items, key=key):
                selected[number] = node
                if len(selected) >= count:
                    return selected
        for number, node in heapq.nsmallest(count, items, key=lambda item: item[1].print_cost):
            if len(selected) >= count:
                break
            selected.setdefault(number, node)
        return selected

    def _beam(self, phase: int, length: int, budget: _Budget,
              memo: dict[tuple[int, int], _Interval]) -> _Interval:
        budget.check()
        key = (phase, length)
        if length <= 6:
            return self._exact(phase, length, budget)
        if key in memo:
            return memo[key]
        width = min(budget.limits.beam_width, max(1, budget.limits.max_states_per_interval // 2))
        growth_limit = min(budget.limits.max_states_per_interval, 2 * width)
        result: dict[Fraction, Expr] = {}
        local_pairs = 0
        # 每个近似区间只分配部分运算预算，给后续根切分和更多循环留下机会。
        quota = min(budget.limits.beam_max_pair_expansions_per_interval,
                    max(32, (budget.limits.max_pair_expansions - budget.expansions) // max(1, length)))
        cuts = sorted(range(1, length), key=lambda cut: abs(length - 2 * cut))
        for cut in cuts:
            left = self._select(self._beam(phase, cut, budget, memo).states, budget.target, width)
            right = self._select(self._beam((phase + cut) % 3, length - cut, budget, memo).states,
                                 budget.target, width)
            for a, lhs in left.items():
                for b, rhs in right.items():
                    if local_pairs >= quota:
                        break
                    budget.pair()
                    local_pairs += 1
                    for op, value in (("+", a + b), ("-", a - b), ("*", a * b)):
                        if len(result) >= growth_limit:
                            result = self._select(result, budget.target, max(1, width // 2))
                        self._insert(result, op, lhs, rhs, value, budget, phase, length, "beam")
                    if b:
                        if len(result) >= growth_limit:
                            result = self._select(result, budget.target, max(1, width // 2))
                        self._insert(result, "/", lhs, rhs, a / b, budget, phase, length, "beam")
                if local_pairs >= quota:
                    break
            if local_pairs >= quota:
                break
        budget.complete = False
        budget.truncated += 1
        interval = _Interval(self._select(result, budget.target, width), False)
        memo[key] = interval
        return interval

    def _record_stats(self, budget: _Budget):
        self._stats["pair_expansions"] += budget.expansions
        self._stats["max_states"] = max(self._stats["max_states"], budget.max_states)
        self._stats["truncated_intervals"] += budget.truncated
        self._stats["estimated_memory_bytes"] = max(self._stats["estimated_memory_bytes"], budget.peak_memory)
        self._stats["complete"] = budget.complete

    def statistics(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def precompute(self, limits: SearchLimits | None = None) -> None:
        """完整建立长度 1..6、全部三个相位的缓存；未完成时明确报错。"""
        limits = limits or SearchLimits(timeout_seconds=60, max_pair_expansions=5000000, max_memory_mb=256)
        start = perf_counter()
        if not self._lock.acquire(timeout=min(limits.timeout_seconds, threading.TIMEOUT_MAX)):
            raise TimeoutError("等待精确 DP 缓存超时")
        budget = _Budget(limits, start, self._memory())
        try:
            for length in range(1, 7):
                for phase in range(3):
                    self._exact(phase, length, budget)
        except _BudgetExceeded as exc:
            raise TimeoutError("精确 DP 预计算未在预算内完成") from exc
        finally:
            self._record_stats(budget)
            self._lock.release()

    def exact_values(self, phase: int, length: int) -> Mapping[Fraction, Expr]:
        with self._lock:
            if (phase, length) not in self._cache:
                raise ValueError("该精确区间尚未完成预计算")
            return MappingProxyType(self._cache[phase, length].states)

    @staticmethod
    def _result(target: int, best: Expr, method: str, exhaustive: int, start: float,
                budget: _Budget | None = None, initial: dict | None = None,
                effective_cap: int = 0, upper: int = 0) -> dict:
        validate(best, target)
        expression = render(best)
        # 重新解析实际输出，防止打印时括号、符号或数字顺序出错。
        validate(parse_expression(expression), target)
        if len(expression) != best.print_cost:
            raise ValueError("打印长度与 AST 缓存不一致")
        cycles = best.digits_used // 3
        result = dict(target=target, expression=expression, cycles=cycles, digits_used=best.digits_used,
                      verified=True, method=method, elapsed_ms=round((perf_counter() - start) * 1000, 3))
        result.update(proof_metadata(cycles, min(exhaustive, cycles - 1)))
        initial = initial or dict(cycles=cycles, method=method, construction_ms=0)
        reason = budget.stop_reason if budget else "busy"
        if result["optimal"] and reason == "search_exhausted":
            reason = "optimal"
        result.update(initial_cycles=initial["cycles"], initial_method=initial["method"],
                      initial_construction_ms=initial["construction_ms"],
                      first_improvement_ms=budget.first_improvement_ms if budget else None,
                      termination_reason=reason, timed_out=reason == "timeout",
                      effective_max_cycles_search=effective_cap, search_upper_cycles=upper,
                      search_pair_expansions=budget.expansions if budget else 0,
                      search_peak_memory_bytes=budget.peak_memory if budget else 0)
        return result

    def solve(self, target: int, limits: SearchLimits | None = None, *, initial_result: dict | None = None) -> dict:
        if type(target) is not int:
            raise TypeError("目标必须是整数")
        limits = limits or SearchLimits()
        start = perf_counter()
        candidates = self.construction_candidates(target)
        method, best = min(candidates.items(), key=lambda item: (
            item[1].digits_used, item[1].print_cost, item[1].stable_hash))
        exhaustive = 0
        if initial_result is not None:
            initial_result = normalize_result(initial_result, target)
            initial_node = validate_result(initial_result, target)
            # 历史证明独立于代表表达式；预算耗尽时也不能丢失已有上界和证明。
            exhaustive = initial_result["exhaustive_through_cycles"]
            if (initial_node.digits_used, initial_node.print_cost) <= (best.digits_used, best.print_cost):
                best, method = initial_node, initial_result["method"]
        if limits.max_cycles_search == "a":
            worst = max(node.digits_used // 3 for node in self._remainders.values())
            limits = replace(limits, max_cycles_search=adaptive_cycle_limit(target, worst))
        effective_cap = limits.max_cycles_search
        upper = min(effective_cap, max(0, best.digits_used // 3 - 1))
        initial_info = dict(cycles=best.digits_used // 3, method=method,
                            construction_ms=round((perf_counter() - start) * 1000, 3))
        search_start = perf_counter()
        remaining = max(0.0, search_start + limits.timeout_seconds - perf_counter())
        if not self._lock.acquire(timeout=min(remaining, threading.TIMEOUT_MAX)):
            return self._result(target, best, method, exhaustive, start, initial=initial_info,
                                effective_cap=effective_cap, upper=upper)
        budget = _Budget(limits, search_start, self._memory(), target, best)
        budget.method = method
        def finish():
            return self._result(target, budget.best, budget.method, exhaustive, start,
                                budget, initial_info, effective_cap, upper)
        try:
            budget.check()
            for cycles in range(1, min(2, upper) + 1):
                interval = self._exact(0, 3 * cycles, budget)
                found = interval.states.get(Fraction(target))
                if found is not None:
                    budget.offer(found, "exact_dp")
                    return finish()
                exhaustive = max(exhaustive, cycles)
            if exhaustive >= budget.best.digits_used // 3 - 1:
                return finish()
            self._quotient_search(budget)
            if exhaustive >= budget.best.digits_used // 3 - 1:
                return finish()
            if upper >= 3 and budget.best.digits_used > 9:
                if abs(target) <= 2016**3 // 2:
                    for cut in (4, 5, 3, 6):
                        left = self._exact(0, cut, budget).states
                        right = self._exact(cut % 3, 9 - cut, budget).states
                        self._match(left, right, budget, "mitm")
                        if budget.best.digits_used == 9:
                            return finish()
                memo: dict[tuple[int, int], _Interval] = {}
                # 根区间只尝试小规模和一个目标相关的尺度，不机械逐组枚举到构造上界。
                cycle_candidates = (3, 4)
                for cycles in cycle_candidates:
                    if cycles > min(upper, budget.best.digits_used // 3 - 1):
                        continue
                    if abs(target) > 2016**cycles // 2:
                        continue
                    length = 3 * cycles
                    cuts = (1, 2, 7, 8) if cycles == 3 else sorted(
                        range(1, length), key=lambda cut: abs(length - 2 * cut))
                    for cut in cuts:
                        left = self._beam(0, cut, budget, memo).states
                        right = self._beam(cut % 3, length - cut, budget, memo).states
                        self._match(left, right, budget, "beam")
                        if budget.best.digits_used == length:
                            return finish()
        except _BudgetExceeded:
            pass
        finally:
            self._record_stats(budget)
            self._lock.release()
        return finish()


_default_solver = Solver()


def solve(target: int, limits: SearchLimits | None = None, *, initial_result: dict | None = None) -> dict:
    """统一 JSON 数据接口；跨查询共享精确 DP 缓存。"""
    return _default_solver.solve(target, limits, initial_result=initial_result)


def solve_many(targets: Iterable[int], limits: SearchLimits | None = None) -> list[dict]:
    """依次批量查询，每个目标单独享有搜索预算。"""
    return [solve(target, limits) for target in targets]
