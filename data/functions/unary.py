"""
单变量函数注册 — 35 个函数。
涵盖：三角、双曲、指数/对数、幂次、激活、取整、特殊函数。
"""

import math
from .registry import make_unary_generator

# ─────────────── 三角函数 ───────────────

make_unary_generator("sin", ["sin", "sine", "sine function", "sinusoidal", "trig_sin"], "trigonometric", math.sin, (-10, 10), tier=0)
make_unary_generator("cos", ["cos", "cosine", "cosine function", "trig_cos"], "trigonometric", math.cos, (-10, 10), tier=0)
make_unary_generator("tan", ["tan", "tangent", "tangent function", "trig_tan"], "trigonometric", math.tan, (-1.5, 1.5), tier=0)
make_unary_generator("asin", ["asin", "arcsin", "arcsine", "inverse_sine", "sin_inv"], "trigonometric", math.asin, (-0.99, 0.99))
make_unary_generator("acos", ["acos", "arccos", "arccosine", "inverse_cosine", "cos_inv"], "trigonometric", math.acos, (-0.99, 0.99))
make_unary_generator("atan", ["atan", "arctan", "arctangent", "inverse_tangent", "tan_inv"], "trigonometric", math.atan, (-10, 10))

# ─────────────── 双曲函数 ───────────────

make_unary_generator("sinh", ["sinh", "hyperbolic_sine", "hyp_sin"], "hyperbolic", math.sinh, (-5, 5))
make_unary_generator("cosh", ["cosh", "hyperbolic_cosine", "hyp_cos"], "hyperbolic", math.cosh, (-5, 5))
make_unary_generator("tanh", ["tanh", "hyperbolic_tangent", "hyp_tan"], "hyperbolic", math.tanh, (-5, 5))
make_unary_generator("asinh", ["asinh", "arcsinh", "inverse_hyperbolic_sine"], "hyperbolic", math.asinh, (-10, 10))
make_unary_generator("acosh", ["acosh", "arccosh", "inverse_hyperbolic_cosine"], "hyperbolic", math.acosh, (1.01, 10))
make_unary_generator("atanh", ["atanh", "arctanh", "inverse_hyperbolic_tangent"], "hyperbolic", math.atanh, (-0.99, 0.99))

# ─────────────── 指数 / 对数 ───────────────

make_unary_generator("exp", ["exp", "exponential", "e_to_the", "natural_exp"], "exponential", math.exp, (-5, 5), tier=0)
make_unary_generator("exp2", ["exp2", "two_to_the", "power_of_2", "2_exp"], "exponential", lambda x: 2.0 ** x, (-10, 10), tier=0)
make_unary_generator("expm1", ["expm1", "exp_minus_1", "e_x_minus_1"], "exponential", math.expm1, (-5, 5))
make_unary_generator("log", ["log", "ln", "natural_log", "logarithm", "nat_log"], "logarithmic", math.log, (0.01, 100), tier=0)
make_unary_generator("log2", ["log2", "log_base_2", "binary_log", "lb"], "logarithmic", math.log2, (0.01, 100), tier=0)
make_unary_generator("log10", ["log10", "log_base_10", "common_log", "lg", "decadic_log"], "logarithmic", math.log10, (0.01, 100), tier=0)
make_unary_generator("log1p", ["log1p", "log_1_plus_x", "ln_1_plus"], "logarithmic", math.log1p, (-0.99, 100))

# ─────────────── 幂次 ───────────────

make_unary_generator("sqrt", ["sqrt", "square_root", "root", "radical"], "power", math.sqrt, (0, 100), tier=0)
make_unary_generator("cbrt", ["cbrt", "cube_root", "cubic_root", "third_root"], "power", lambda x: math.copysign(abs(x) ** (1/3), x), (-50, 50))
make_unary_generator("square", ["square", "x_squared", "pow2", "quadratic_mono"], "power", lambda x: x * x, (-10, 10), tier=0)
make_unary_generator("cube", ["cube", "x_cubed", "pow3", "cubic_mono"], "power", lambda x: x * x * x, (-5, 5), tier=0)
make_unary_generator("reciprocal", ["reciprocal", "inverse", "one_over_x", "multiplicative_inverse"], "power", lambda x: 1.0 / x, (0.1, 10))

# ─────────────── 激活 / 符号 ───────────────

make_unary_generator("abs", ["abs", "absolute", "absolute_value", "magnitude", "modulus"], "activation", abs, (-10, 10), tier=0)
make_unary_generator("sign", ["sign", "signum", "sgn", "sign_function"], "activation", lambda x: float((x > 0) - (x < 0)), (-10, 10), tier=0)
make_unary_generator("sigmoid", ["sigmoid", "logistic", "logistic_function", "sigma"], "activation", lambda x: 1.0 / (1.0 + math.exp(-x)), (-10, 10))
make_unary_generator("relu", ["relu", "rectified_linear", "ramp", "positive_part"], "activation", lambda x: max(0.0, x), (-10, 10))
make_unary_generator("softplus", ["softplus", "smooth_relu", "log_exp_sum"], "activation", lambda x: math.log1p(math.exp(x)) if x < 20 else x, (-10, 10))
make_unary_generator("heaviside", ["heaviside", "step_function", "unit_step", "theta"], "activation", lambda x: 1.0 if x > 0 else (0.5 if x == 0 else 0.0), (-10, 10))

# ─────────────── 取整 ───────────────

make_unary_generator("floor", ["floor", "floor_function", "round_down", "greatest_integer"], "rounding", math.floor, (-10, 10))
make_unary_generator("ceil", ["ceil", "ceiling", "round_up", "least_integer"], "rounding", math.ceil, (-10, 10))
make_unary_generator("round", ["round", "round_nearest", "nint", "nearest_integer"], "rounding", lambda x: round(x), (-10, 10))
make_unary_generator("frac", ["frac", "fractional_part", "decimal_part", "mantissa"], "rounding", lambda x: x - math.floor(x), (-10, 10))

# ─────────────── 特殊函数 & 转换 ───────────────

make_unary_generator("erf", ["erf", "error_function", "gauss_error", "probability_integral"], "special", math.erf, (-3, 3))
make_unary_generator("degrees", ["degrees", "rad_to_deg", "to_degrees", "radians_to_degrees"], "conversion", math.degrees, (-6.28, 6.28))
make_unary_generator("radians", ["radians", "deg_to_rad", "to_radians", "degrees_to_radians"], "conversion", math.radians, (-180, 180))
