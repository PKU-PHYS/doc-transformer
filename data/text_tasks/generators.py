"""
纯文本推理任务生成器。
生成需要 FrozenLM 语义理解能力的任务样本。

三类任务：
  A. 数值 → 文本分类 (根据数值属性推断文本标签)
  B. 文本 → 文本推理 (根据一个文本推断相关文本)
  C. 文本 → 数值 (根据文本描述推断数值常量)
"""

import math
import random
from typing import Any, Dict


class TextTaskGenerator:
    """纯文本推理任务生成器。"""

    # ─── A. 数值 → 文本分类 ───

    _SIGN_TASK_TEMPLATES = [
        lambda v, s: {"value": v, "sign": s},
        lambda v, s: {"number": v, "polarity": s},
        lambda v, s: {"x": v, "sign_label": s},
        lambda v, s: {"input": v, "classification": s},
        lambda v, s: {"observed": v, "category": s},
    ]

    _MAGNITUDE_LABELS = [
        (-float("inf"), -100, "large_negative"),
        (-100, -10, "moderate_negative"),
        (-10, -1, "small_negative"),
        (-1, 0, "tiny_negative"),
        (0, 1, "tiny_positive"),
        (1, 10, "small_positive"),
        (10, 100, "moderate_positive"),
        (100, float("inf"), "large_positive"),
    ]

    @classmethod
    def _gen_sign_task(cls) -> Dict[str, Any]:
        v = random.uniform(-100, 100)
        s = "positive" if v > 0 else ("negative" if v < 0 else "zero")
        v = round(v, 4)
        tmpl = random.choice(cls._SIGN_TASK_TEMPLATES)
        return tmpl(v, s)

    @classmethod
    def _gen_magnitude_task(cls) -> Dict[str, Any]:
        v = random.uniform(-200, 200)
        label = "unknown"
        for lo, hi, lbl in cls._MAGNITUDE_LABELS:
            if lo <= v < hi:
                label = lbl
                break
        v = round(v, 4)
        keys = random.choice([
            ("value", "magnitude"),
            ("number", "size_class"),
            ("x", "range_label"),
            ("input", "magnitude_category"),
        ])
        return {keys[0]: v, keys[1]: label}

    @classmethod
    def _gen_parity_task(cls) -> Dict[str, Any]:
        v = random.randint(-100, 100)
        parity = "even" if v % 2 == 0 else "odd"
        keys = random.choice([
            ("integer", "parity"),
            ("number", "even_or_odd"),
            ("n", "parity_label"),
            ("value", "divisibility_by_2"),
        ])
        return {keys[0]: v, keys[1]: parity}

    @classmethod
    def _gen_quadrant_task(cls) -> Dict[str, Any]:
        x = round(random.uniform(-10, 10), 4)
        y = round(random.uniform(-10, 10), 4)
        if x > 0 and y > 0:
            q = "first"
        elif x < 0 and y > 0:
            q = "second"
        elif x < 0 and y < 0:
            q = "third"
        else:
            q = "fourth"
        keys = random.choice([
            ("x", "y", "quadrant"),
            ("coord_x", "coord_y", "position"),
            ("horizontal", "vertical", "region"),
        ])
        return {keys[0]: x, keys[1]: y, keys[2]: q}

    @classmethod
    def _gen_comparison_task(cls) -> Dict[str, Any]:
        a = round(random.uniform(-50, 50), 4)
        b = round(random.uniform(-50, 50), 4)
        if a > b:
            rel = "greater"
        elif a < b:
            rel = "less"
        else:
            rel = "equal"
        keys = random.choice([
            ("a", "b", "relation"),
            ("first", "second", "comparison"),
            ("x", "y", "ordering"),
            ("left", "right", "relative_size"),
        ])
        return {keys[0]: a, keys[1]: b, keys[2]: rel}

    @classmethod
    def _gen_special_point_task(cls) -> Dict[str, Any]:
        """函数在特殊点的行为"""
        func_choices = [
            ("sin", math.sin, [0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi]),
            ("cos", math.cos, [0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi]),
            ("exp", math.exp, [0, 1, -1]),
            ("log", math.log, [1, math.e]),
        ]
        func_name, func, points = random.choice(func_choices)
        x = random.choice(points)
        y = round(func(x), 6)

        if abs(y) < 1e-10:
            label = "zero_crossing"
        elif abs(y - 1.0) < 1e-10:
            label = "unit_value"
        elif abs(y + 1.0) < 1e-10:
            label = "neg_unit_value"
        elif abs(y - 0.5) < 1e-10:
            label = "half_value"
        else:
            label = "general_point"

        keys = random.choice([
            ("function", "x", "y", "point_type"),
            ("func", "input", "output", "classification"),
            ("operation", "argument", "result", "label"),
        ])
        return {keys[0]: func_name, keys[1]: round(x, 6), keys[2]: y, keys[3]: label}

    # ─── B. 文本 → 文本推理 ───

    _FUNCTION_FAMILIES = {
        "sin": "trigonometric", "cos": "trigonometric", "tan": "trigonometric",
        "asin": "trigonometric", "acos": "trigonometric", "atan": "trigonometric",
        "sinh": "hyperbolic", "cosh": "hyperbolic", "tanh": "hyperbolic",
        "exp": "exponential", "log": "logarithmic", "log2": "logarithmic",
        "sqrt": "power", "square": "power", "cube": "power",
        "abs": "activation", "relu": "activation", "sigmoid": "activation",
        "floor": "rounding", "ceil": "rounding",
        "erf": "special",
    }

    _INVERSE_PAIRS = [
        ("sin", "arcsin"), ("cos", "arccos"), ("tan", "arctan"),
        ("exp", "log"), ("square", "sqrt"), ("log2", "exp2"),
        ("sinh", "arcsinh"), ("cosh", "arccosh"), ("tanh", "arctanh"),
    ]

    _DERIVATIVE_PAIRS = [
        ("sin(x)", "cos(x)"), ("cos(x)", "-sin(x)"),
        ("exp(x)", "exp(x)"), ("log(x)", "1/x"),
        ("x^2", "2x"), ("x^3", "3x^2"),
        ("sqrt(x)", "1/(2*sqrt(x))"), ("1/x", "-1/x^2"),
        ("tan(x)", "sec^2(x)"), ("sinh(x)", "cosh(x)"),
    ]

    _SYMMETRY = {
        "sin": "odd", "cos": "even", "tan": "odd",
        "abs": "even", "square": "even", "cube": "odd",
        "sinh": "odd", "cosh": "even", "tanh": "odd",
        "exp": "neither", "log": "neither",
        "sigmoid": "neither", "relu": "neither",
    }

    _PERIODICITY = {
        "sin": "periodic", "cos": "periodic", "tan": "periodic",
        "exp": "non_periodic", "log": "non_periodic",
        "square": "non_periodic", "abs": "non_periodic",
        "sinh": "non_periodic", "cosh": "non_periodic",
    }

    @classmethod
    def _gen_family_task(cls) -> Dict[str, Any]:
        name = random.choice(list(cls._FUNCTION_FAMILIES.keys()))
        family = cls._FUNCTION_FAMILIES[name]
        keys = random.choice([
            ("name", "family"),
            ("function", "category"),
            ("func", "group"),
            ("operation", "branch"),
        ])
        return {keys[0]: name, keys[1]: family}

    @classmethod
    def _gen_inverse_task(cls) -> Dict[str, Any]:
        pair = random.choice(cls._INVERSE_PAIRS)
        if random.random() < 0.5:
            pair = (pair[1], pair[0])
        keys = random.choice([
            ("function", "inverse"),
            ("original", "inverse_function"),
            ("f", "f_inverse"),
            ("forward", "backward"),
        ])
        return {keys[0]: pair[0], keys[1]: pair[1]}

    @classmethod
    def _gen_derivative_task(cls) -> Dict[str, Any]:
        pair = random.choice(cls._DERIVATIVE_PAIRS)
        keys = random.choice([
            ("function", "derivative"),
            ("f_x", "f_prime_x"),
            ("original", "differentiated"),
            ("expression", "d_dx"),
        ])
        return {keys[0]: pair[0], keys[1]: pair[1]}

    @classmethod
    def _gen_symmetry_task(cls) -> Dict[str, Any]:
        name = random.choice(list(cls._SYMMETRY.keys()))
        sym = cls._SYMMETRY[name]
        keys = random.choice([
            ("function", "symmetry"),
            ("func", "parity"),
            ("name", "symmetry_type"),
            ("operation", "even_odd"),
        ])
        return {keys[0]: name, keys[1]: sym}

    @classmethod
    def _gen_periodicity_task(cls) -> Dict[str, Any]:
        name = random.choice(list(cls._PERIODICITY.keys()))
        per = cls._PERIODICITY[name]
        keys = random.choice([
            ("function", "periodicity"),
            ("func", "is_periodic"),
            ("name", "periodic_label"),
        ])
        return {keys[0]: name, keys[1]: per}

    # ─── C. 文本 → 数值 (常数识别) ───

    _CONSTANTS = [
        (["pi", "pi_constant", "archimedes_constant"], round(math.pi, 6)),
        (["e", "euler_number", "napier_constant"], round(math.e, 6)),
        (["sqrt2", "root_2", "pythagoras_constant"], round(math.sqrt(2), 6)),
        (["sqrt3", "root_3"], round(math.sqrt(3), 6)),
        (["golden_ratio", "phi", "golden_number"], round((1 + math.sqrt(5)) / 2, 6)),
        (["ln2", "log_2", "natural_log_of_2"], round(math.log(2), 6)),
        (["ln10", "log_10", "natural_log_of_10"], round(math.log(10), 6)),
        (["pi_over_2", "half_pi"], round(math.pi / 2, 6)),
        (["pi_over_4", "quarter_pi"], round(math.pi / 4, 6)),
        (["two_pi", "tau", "full_circle"], round(2 * math.pi, 6)),
    ]

    @classmethod
    def _gen_constant_task(cls) -> Dict[str, Any]:
        names, value = random.choice(cls._CONSTANTS)
        name = random.choice(names)
        keys = random.choice([
            ("name", "value"),
            ("constant", "numerical_value"),
            ("symbol", "decimal"),
            ("identifier", "approx"),
        ])
        return {keys[0]: name, keys[1]: value}

    _SPECIAL_VALUES = [
        ("sin", "0", 0.0), ("sin", "pi/2", 1.0), ("sin", "pi", 0.0),
        ("cos", "0", 1.0), ("cos", "pi/2", 0.0), ("cos", "pi", -1.0),
        ("exp", "0", 1.0), ("exp", "1", round(math.e, 6)),
        ("log", "1", 0.0), ("log", "e", 1.0),
        ("sqrt", "4", 2.0), ("sqrt", "9", 3.0), ("sqrt", "16", 4.0),
        ("abs", "-1", 1.0), ("abs", "5", 5.0),
        ("sigmoid", "0", 0.5),
    ]

    @classmethod
    def _gen_special_value_task(cls) -> Dict[str, Any]:
        func, input_desc, output = random.choice(cls._SPECIAL_VALUES)
        keys = random.choice([
            ("function", "special_input", "output"),
            ("func", "at_point", "value"),
            ("operation", "argument", "result"),
        ])
        return {keys[0]: func, keys[1]: input_desc, keys[2]: output}

    # ─── 主接口 ───

    _ALL_GENERATORS = [
        # A 类: 数值 → 文本
        "_gen_sign_task",
        "_gen_magnitude_task",
        "_gen_parity_task",
        "_gen_quadrant_task",
        "_gen_comparison_task",
        "_gen_special_point_task",
        # B 类: 文本 → 文本
        "_gen_family_task",
        "_gen_inverse_task",
        "_gen_derivative_task",
        "_gen_symmetry_task",
        "_gen_periodicity_task",
        # C 类: 文本 → 数值
        "_gen_constant_task",
        "_gen_special_value_task",
    ]

    @classmethod
    def generate(cls) -> Dict[str, Any]:
        """随机生成一条文本推理任务。返回 dict。"""
        gen_name = random.choice(cls._ALL_GENERATORS)
        gen_fn = getattr(cls, gen_name)
        return gen_fn()
