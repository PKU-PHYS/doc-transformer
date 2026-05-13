"""
数学函数注册表 — 基类、注册机制与工厂函数。

设计思路：
  每个"函数"注册为一个 generator callable，调用后返回一条 MathRelation，
  其中所有变量值已随机采样并通过精确计算填充。

  MathRelation 同时携带函数名同义词池和变量键名同义词池，
  供下游 TemplateEngine 渲染为多样化的 JSON 文档。
"""

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class MathRelation:
    """一条已填充所有变量值的数学关系实例。"""
    func_name: str                          # 规范名称，如 "sin"
    func_synonyms: List[str]                # 函数名同义词池
    category: str                           # 类别，如 "trigonometric"
    variables: Dict[str, Any]               # role -> value，如 {"input": 1.5, "output": 0.997}
    var_synonyms: Dict[str, List[str]]      # role -> 键名同义词列表
    include_func_name: bool = True          # 是否在文档中包含函数名字段
    invertible: bool = True                 # 函数是否可逆（可逆时允许 mask 输入变量）
    _generator: Optional[Callable] = None   # 生成此关系的 generator，用于重采样


class FunctionRegistry:
    """全局函数注册表。"""
    _generators: List[Callable[[], MathRelation]] = []
    
    _current_max_tier: Optional[int] = None
    _current_exact_tier: Optional[int] = None

    @classmethod
    def set_filter(cls, max_tier: Optional[int] = None, exact_tier: Optional[int] = None):
        cls._current_max_tier = max_tier
        cls._current_exact_tier = exact_tier

    @classmethod
    def register(cls, gen_fn: Callable[[], MathRelation]):
        cls._generators.append(gen_fn)
        return gen_fn

    @classmethod
    def sample(cls) -> MathRelation:
        gen = cls.sample_generator()
        return gen()

    @classmethod
    def sample_generator(cls) -> Callable[[], MathRelation]:
        """返回一个 generator 函数的引用，可多次调用获取同类函数的不同采样。"""
        gens = cls._generators
        if cls._current_exact_tier is not None:
            gens = [g for g in gens if getattr(g, "tier", 1) == cls._current_exact_tier]
        elif cls._current_max_tier is not None:
            gens = [g for g in gens if getattr(g, "tier", 1) <= cls._current_max_tier]
            
        if not gens:
            gens = cls._generators # fallback
            
        return random.choice(gens)

    @classmethod
    def count(cls) -> int:
        return len(cls._generators)

    @classmethod
    def clear(cls):
        cls._generators.clear()


# ─────────────────────────────────────────────
# 公共同义词池（各函数模块可直接复用）
# ─────────────────────────────────────────────

FUNC_NAME_KEYS = [
    "function", "func", "fn", "operation", "op", "transform",
    "method", "mapping", "computation", "operator", "type",
    "kind", "formula_type", "math_op", "procedure", "rule",
]

UNARY_INPUT_KEYS = [
    "x", "input", "arg", "argument", "param", "parameter",
    "value", "operand", "source", "independent_var", "domain_value",
    "in_val", "given", "input_value", "x_val",
]

UNARY_OUTPUT_KEYS = [
    "y", "output", "result", "answer", "return_val", "response",
    "computed", "derived", "target", "dependent_var", "range_value",
    "out_val", "outcome", "output_value", "y_val",
]

BINARY_INPUT_A_KEYS = [
    "a", "x", "left", "first", "operand_1", "lhs",
    "input_a", "val_a", "param_a", "x1", "arg_1",
    "first_value", "primary", "alpha",
]

BINARY_INPUT_B_KEYS = [
    "b", "y", "right", "second", "operand_2", "rhs",
    "input_b", "val_b", "param_b", "x2", "arg_2",
    "second_value", "secondary", "beta",
]

BINARY_OUTPUT_KEYS = [
    "result", "output", "answer", "computed", "return_val",
    "outcome", "derived", "value", "out", "response",
    "product", "sum_val", "total", "final",
]

CATEGORY_KEYS = [
    "category", "family", "group", "class", "domain",
    "branch", "field", "area", "subject",
]


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

def _sample_log_uniform(lo: float, hi: float) -> float:
    """
    在 [lo, hi] 范围内进行对数均匀采样，等概率覆盖每个量级。
    仅适用于 lo > 0 的正数域。
    """
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return math.exp(random.uniform(log_lo, log_hi))


def _sample_domain(lo: float, hi: float, log_uniform: bool = False) -> float:
    """
    统一的域采样入口。
    log_uniform=True 时：
      - 若 lo > 0: 对数均匀采样
      - 若 hi < 0: 对数均匀采样后取负
      - 若跨零: 先随机选符号，再对数均匀采样绝对值
    """
    if not log_uniform:
        return random.uniform(lo, hi)
    
    if lo > 0:
        return _sample_log_uniform(lo, hi)
    elif hi < 0:
        return -_sample_log_uniform(-hi, -lo)
    else:
        # 跨零域：先选符号，再采样绝对值
        abs_lo = max(abs(lo), 1e-10)
        abs_hi = max(abs(hi), 1e-10)
        sign = random.choice([-1, 1])
        return sign * _sample_log_uniform(min(abs_lo, abs_hi), max(abs_lo, abs_hi))


def _safe_compute(compute_fn, x, max_abs=2**32):
    """安全计算，捕获 domain error 并拒绝过大的值。"""
    try:
        y = compute_fn(x)
        if not math.isfinite(y):
            return None
        if abs(y) > max_abs:
            return None
        return y
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def make_unary_generator(
    name: str,
    synonyms: List[str],
    category: str,
    compute: Callable[[float], float],
    domain: Tuple[float, float] = (-10.0, 10.0),
    input_keys: Optional[List[str]] = None,
    output_keys: Optional[List[str]] = None,
    max_retries: int = 20,
    tier: int = 1,
    log_uniform: bool = False,
    invertible: bool = True,
) -> Callable[[], MathRelation]:
    """为单变量函数创建并注册数据生成器。"""
    _in_keys = input_keys or UNARY_INPUT_KEYS
    _out_keys = output_keys or UNARY_OUTPUT_KEYS

    def generator() -> MathRelation:
        for _ in range(max_retries):
            x = _sample_domain(*domain, log_uniform=log_uniform)
            y = _safe_compute(compute, x)
            if y is not None:
                rel = MathRelation(
                    func_name=name,
                    func_synonyms=synonyms,
                    category=category,
                    variables={"input": round(x, 6), "output": round(y, 6)},
                    var_synonyms={"input": _in_keys, "output": _out_keys},
                    invertible=invertible,
                    _generator=generator,
                )
                return rel
        # fallback: 使用域中心
        x = (domain[0] + domain[1]) / 2
        y = compute(x)
        return MathRelation(
            func_name=name,
            func_synonyms=synonyms,
            category=category,
            variables={"input": round(x, 6), "output": round(y, 6)},
            var_synonyms={"input": _in_keys, "output": _out_keys},
            invertible=invertible,
            _generator=generator,
        )

    generator.tier = tier
    FunctionRegistry.register(generator)
    return generator


def make_binary_generator(
    name: str,
    synonyms: List[str],
    category: str,
    compute: Callable[[float, float], float],
    domain_a: Tuple[float, float] = (-10.0, 10.0),
    domain_b: Tuple[float, float] = (-10.0, 10.0),
    input_a_keys: Optional[List[str]] = None,
    input_b_keys: Optional[List[str]] = None,
    output_keys: Optional[List[str]] = None,
    max_retries: int = 20,
    tier: int = 1,
    log_uniform: bool = False,
    invertible: bool = True,
) -> Callable[[], MathRelation]:
    """为双变量函数创建并注册数据生成器。"""
    _a_keys = input_a_keys or BINARY_INPUT_A_KEYS
    _b_keys = input_b_keys or BINARY_INPUT_B_KEYS
    _out_keys = output_keys or BINARY_OUTPUT_KEYS

    def generator() -> MathRelation:
        for _ in range(max_retries):
            a = _sample_domain(*domain_a, log_uniform=log_uniform)
            b = _sample_domain(*domain_b, log_uniform=log_uniform)
            y = _safe_compute(lambda _: compute(a, b), None)
            if y is not None:
                return MathRelation(
                    func_name=name,
                    func_synonyms=synonyms,
                    category=category,
                    variables={"input_a": round(a, 6), "input_b": round(b, 6), "output": round(y, 6)},
                    var_synonyms={"input_a": _a_keys, "input_b": _b_keys, "output": _out_keys},
                    invertible=invertible,
                    _generator=generator,
                )
        a = (domain_a[0] + domain_a[1]) / 2
        b = (domain_b[0] + domain_b[1]) / 2
        y = compute(a, b)
        return MathRelation(
            func_name=name,
            func_synonyms=synonyms,
            category=category,
            variables={"input_a": round(a, 6), "input_b": round(b, 6), "output": round(y, 6)},
            var_synonyms={"input_a": _a_keys, "input_b": _b_keys, "output": _out_keys},
            invertible=invertible,
            _generator=generator,
        )

    generator.tier = tier
    FunctionRegistry.register(generator)
    return generator


def make_multivar_generator(
    name: str,
    synonyms: List[str],
    category: str,
    var_defs: List[Tuple[str, List[str], Tuple[float, float]]],
    output_role: str,
    output_keys: List[str],
    compute: Callable[[Dict[str, float]], float],
    max_retries: int = 20,
    tier: int = 1,
    invertible: bool = True,
) -> Callable[[], MathRelation]:
    """
    为多变量函数创建并注册数据生成器。

    var_defs: [(role_name, key_synonyms, (lo, hi)), ...] — 输入变量定义
    output_role: 输出变量的角色名
    compute: 接收 {role: value} 返回输出值
    tier: 难度层级 (0=最简单, 1=标准复杂度)
    """
    _out_keys = output_keys

    def generator() -> MathRelation:
        for _ in range(max_retries):
            inputs = {}
            for role, _, (lo, hi) in var_defs:
                inputs[role] = random.uniform(lo, hi)
            y = _safe_compute(lambda _: compute(inputs), None)
            if y is not None:
                variables = {role: round(v, 6) for role, v in inputs.items()}
                variables[output_role] = round(y, 6)
                var_syns = {role: keys for role, keys, _ in var_defs}
                var_syns[output_role] = _out_keys
                return MathRelation(
                    func_name=name,
                    func_synonyms=synonyms,
                    category=category,
                    variables=variables,
                    var_synonyms=var_syns,
                    invertible=invertible,
                    _generator=generator,
                )
        # fallback with midpoints
        inputs = {role: (lo + hi) / 2 for role, _, (lo, hi) in var_defs}
        y = compute(inputs)
        variables = {role: round(v, 6) for role, v in inputs.items()}
        variables[output_role] = round(y, 6)
        var_syns = {role: keys for role, keys, _ in var_defs}
        var_syns[output_role] = _out_keys
        return MathRelation(
            func_name=name,
            func_synonyms=synonyms,
            category=category,
            variables=variables,
            var_synonyms=var_syns,
            invertible=invertible,
            _generator=generator,
        )

    generator.tier = tier
    FunctionRegistry.register(generator)
    return generator

