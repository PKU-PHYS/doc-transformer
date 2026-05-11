"""
函数复合生成器 — 从已注册的单变量函数中随机选 2 个进行嵌套。
产出的 MathRelation 包含中间变量，结构更复杂。
"""

import math
import random
from typing import List
from .registry import FunctionRegistry, MathRelation, _safe_compute

# 可安全复合的单变量函数定义（手动列出，确保 domain/range 兼容性）
_COMPOSABLE = [
    {"name": "sin",       "synonyms": ["sin", "sine"],           "compute": math.sin,    "domain": (-3, 3),   "range_approx": (-1, 1)},
    {"name": "cos",       "synonyms": ["cos", "cosine"],         "compute": math.cos,    "domain": (-3, 3),   "range_approx": (-1, 1)},
    {"name": "tanh",      "synonyms": ["tanh", "hyp_tan"],       "compute": math.tanh,   "domain": (-3, 3),   "range_approx": (-1, 1)},
    {"name": "sigmoid",   "synonyms": ["sigmoid", "logistic"],   "compute": lambda x: 1/(1+math.exp(-x)), "domain": (-5, 5), "range_approx": (0, 1)},
    {"name": "exp",       "synonyms": ["exp", "exponential"],    "compute": math.exp,    "domain": (-2, 2),   "range_approx": (0.14, 7.39)},
    {"name": "square",    "synonyms": ["square", "x_squared"],   "compute": lambda x: x*x, "domain": (-3, 3), "range_approx": (0, 9)},
    {"name": "sqrt",      "synonyms": ["sqrt", "square_root"],   "compute": math.sqrt,   "domain": (0.01, 10), "range_approx": (0.1, 3.16)},
    {"name": "abs",       "synonyms": ["abs", "absolute"],       "compute": abs,          "domain": (-5, 5),   "range_approx": (0, 5)},
    {"name": "log1p",     "synonyms": ["log1p", "ln_1_plus"],    "compute": math.log1p,  "domain": (0, 10),   "range_approx": (0, 2.4)},
    {"name": "relu",      "synonyms": ["relu", "rectified"],     "compute": lambda x: max(0.0, x), "domain": (-3, 3), "range_approx": (0, 3)},
    {"name": "erf",       "synonyms": ["erf", "error_function"], "compute": math.erf,    "domain": (-3, 3),   "range_approx": (-1, 1)},
    {"name": "atan",      "synonyms": ["atan", "arctan"],        "compute": math.atan,   "domain": (-10, 10), "range_approx": (-1.57, 1.57)},
]

# 中间变量和最终输出的键名池
_MID_KEYS = ["mid", "intermediate", "stage_1_out", "inner_result", "temp", "z", "h"]
_FINAL_KEYS = ["final", "result", "output", "composed", "f_g_x", "answer", "y"]
_INPUT_KEYS = ["x", "input", "arg", "source", "raw_input", "initial"]

_INNER_FUNC_KEYS = ["inner_func", "g", "first_op", "stage_1", "inner", "g_func"]
_OUTER_FUNC_KEYS = ["outer_func", "f", "second_op", "stage_2", "outer", "f_func"]


def _composite_generator():
    """生成一条函数复合关系: y = f(g(x))"""
    for _ in range(30):
        inner = random.choice(_COMPOSABLE)
        outer = random.choice(_COMPOSABLE)
        # 避免 f(f(x)) 同名复合
        if inner["name"] == outer["name"]:
            continue

        x = random.uniform(*inner["domain"])
        mid = _safe_compute(inner["compute"], x)
        if mid is None:
            continue
        y = _safe_compute(outer["compute"], mid)
        if y is None:
            continue

        # 随机决定是否包含中间值
        include_mid = random.random() < 0.5

        variables = {
            "input": round(x, 6),
            "output": round(y, 6),
        }
        var_syns = {
            "input": _INPUT_KEYS,
            "output": _FINAL_KEYS,
        }

        if include_mid:
            variables["mid"] = round(mid, 6)
            var_syns["mid"] = _MID_KEYS

        # 函数名作为文本变量
        variables["inner_func"] = random.choice(inner["synonyms"])
        variables["outer_func"] = random.choice(outer["synonyms"])
        var_syns["inner_func"] = _INNER_FUNC_KEYS
        var_syns["outer_func"] = _OUTER_FUNC_KEYS

        return MathRelation(
            func_name=f"{outer['name']}_of_{inner['name']}",
            func_synonyms=[
                f"{outer['name']}({inner['name']}(x))",
                f"{random.choice(outer['synonyms'])} of {random.choice(inner['synonyms'])}",
                "composed_function", "nested_function", "function_chain",
            ],
            category="composite",
            variables=variables,
            var_synonyms=var_syns,
            include_func_name=False,  # 复合函数名太长，不作为单独字段
        )

    # fallback: sin(cos(x))
    x = 1.0
    mid = math.cos(x)
    y = math.sin(mid)
    return MathRelation(
        func_name="sin_of_cos",
        func_synonyms=["sin(cos(x))", "composed_function"],
        category="composite",
        variables={"input": x, "mid": round(mid, 6), "output": round(y, 6),
                   "inner_func": "cos", "outer_func": "sin"},
        var_synonyms={"input": _INPUT_KEYS, "mid": _MID_KEYS, "output": _FINAL_KEYS,
                      "inner_func": _INNER_FUNC_KEYS, "outer_func": _OUTER_FUNC_KEYS},
        include_func_name=False,
    )


# 注册多个复合生成器实例以增加被选中的权重
for _ in range(5):
    FunctionRegistry.register(_composite_generator)
