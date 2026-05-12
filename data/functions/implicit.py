"""
隐函数 / 约束方程注册 — 20 个。
特点：方程中任意一个变量都可以被 mask，其余变量足以推导出它。
每个方程都是一个完全对称的 generator，随机选择 mask 哪个变量不影响可解性。

实现方式：先采样所有变量使方程成立，文档中包含全部变量值。
Mask 策略由下游 SyntheticDataset 控制。
"""

import math
import random
from .registry import FunctionRegistry, MathRelation

def _register_implicit(name, synonyms, category, var_defs, satisfy):
    """
    注册一个隐函数。

    var_defs: [(role, synonyms_list, (lo, hi)), ...]
    satisfy: callable(partial_vars) -> full_vars
             给定 N-1 个变量的采样值，计算第 N 个使方程成立。
             接收 dict，返回 dict（补全缺失变量）。
    """
    def generator():
        vars_dict = {}
        for role, _, (lo, hi) in var_defs:
            vars_dict[role] = random.uniform(lo, hi)
        # satisfy 用采样值计算依赖变量，使方程精确成立
        vars_dict = satisfy(vars_dict)
        # 过滤无穷/NaN/极端值
        retry = False
        for v in vars_dict.values():
            if not math.isfinite(v) or abs(v) > 10000.0:
                retry = True
                break
        if retry:
            # retry with safe values
            vars_dict = {role: (lo+hi)/2 for role, _, (lo, hi) in var_defs}
            vars_dict = satisfy(vars_dict)
        variables = {role: round(v, 6) for role, v in vars_dict.items()}
        var_syns = {role: keys for role, keys, _ in var_defs}
        return MathRelation(
            func_name=name,
            func_synonyms=synonyms,
            category=category,
            variables=variables,
            var_synonyms=var_syns,
            include_func_name=True,
            _generator=generator,
        )
    FunctionRegistry.register(generator)
    return generator


# ─────────────── 勾股定理类 ───────────────

_register_implicit(
    "pythagorean", ["pythagorean_theorem", "a2_b2_c2", "right_triangle"],
    "geometry",
    [("a", ["a", "side_a", "leg_a", "cathetus_1"], (0.1, 20)),
     ("b", ["b", "side_b", "leg_b", "cathetus_2"], (0.1, 20)),
     ("c", ["c", "hypotenuse", "side_c", "longest_side"], (0.1, 30))],
    satisfy=lambda v: {**v, "c": math.sqrt(v["a"]**2 + v["b"]**2)},
)

# ─────────────── 圆方程 ───────────────

_register_implicit(
    "circle_eq", ["circle_equation", "x2_y2_r2", "unit_circle"],
    "geometry",
    [("x", ["x", "coord_x", "horizontal", "abscissa"], (-10, 10)),
     ("y", ["y", "coord_y", "vertical", "ordinate"], (-10, 10)),
     ("r", ["r", "radius", "circle_radius", "R"], (0.1, 15))],
    satisfy=lambda v: {**v, "r": math.sqrt(v["x"]**2 + v["y"]**2)},
)

# ─────────────── 线性方程 ───────────────

_register_implicit(
    "linear_eq", ["linear_equation", "ax_plus_by_eq_c", "line_eq"],
    "algebra",
    [("a", ["a", "coeff_a", "slope_coeff"], (0.1, 10)),
     ("x", ["x", "var_x", "x_val"], (-10, 10)),
     ("b", ["b", "coeff_b", "y_coeff"], (0.1, 10)),
     ("y", ["y", "var_y", "y_val"], (-10, 10)),
     ("c", ["c", "constant", "rhs", "sum_val"], (-50, 50))],
    satisfy=lambda v: {**v, "c": v["a"]*v["x"] + v["b"]*v["y"]},
)

# ─────────────── 三角恒等式 ───────────────

_register_implicit(
    "sin2_cos2", ["pythagorean_identity", "sin_sq_cos_sq", "trig_identity_1"],
    "trigonometric",
    [("theta", ["theta", "angle", "t", "rad"], (-6.28, 6.28)),
     ("sin_val", ["sin_val", "sine", "sin_theta", "s"], (-1, 1)),
     ("cos_val", ["cos_val", "cosine", "cos_theta", "c"], (-1, 1))],
    satisfy=lambda v: {**v, "sin_val": math.sin(v["theta"]),
                             "cos_val": math.cos(v["theta"])},
)

_register_implicit(
    "tan_identity", ["tan_eq_sin_over_cos", "tangent_identity"],
    "trigonometric",
    [("theta", ["theta", "angle", "t", "rad"], (-1.5, 1.5)),
     ("sin_val", ["sin_val", "sine", "numerator"], (-1, 1)),
     ("cos_val", ["cos_val", "cosine", "denominator"], (0.1, 1)),
     ("tan_val", ["tan_val", "tangent", "ratio"], (-10, 10))],
    satisfy=lambda v: {**v, "sin_val": math.sin(v["theta"]),
                             "cos_val": math.cos(v["theta"]),
                             "tan_val": math.tan(v["theta"])},
)

# ─────────────── 求和约束 ───────────────

_register_implicit(
    "sum_constraint", ["sum_equals", "a_plus_b_plus_c", "triple_sum"],
    "algebra",
    [("a", ["a", "first", "x1", "val_1"], (-10, 10)),
     ("b", ["b", "second", "x2", "val_2"], (-10, 10)),
     ("c", ["c", "third", "x3", "val_3"], (-10, 10)),
     ("S", ["S", "total", "sum", "sigma"], (-30, 30))],
    satisfy=lambda v: {**v, "S": v["a"] + v["b"] + v["c"]},
)

_register_implicit(
    "product_constraint", ["product_equals", "a_times_b_times_c", "triple_product"],
    "algebra",
    [("a", ["a", "factor_1", "x1", "val_1"], (0.1, 5)),
     ("b", ["b", "factor_2", "x2", "val_2"], (0.1, 5)),
     ("c", ["c", "factor_3", "x3", "val_3"], (0.1, 5)),
     ("P", ["P", "product", "total", "pi_val"], (0, 125))],
    satisfy=lambda v: {**v, "P": v["a"] * v["b"] * v["c"]},
)

# ─────────────── 均值约束 ───────────────

_register_implicit(
    "mean_constraint", ["mean_of_three", "average_eq", "arithmetic_mean_3"],
    "statistics",
    [("a", ["a", "x1", "val_1", "first"], (-10, 10)),
     ("b", ["b", "x2", "val_2", "second"], (-10, 10)),
     ("c", ["c", "x3", "val_3", "third"], (-10, 10)),
     ("mu", ["mu", "mean", "average", "avg"], (-10, 10))],
    satisfy=lambda v: {**v, "mu": (v["a"] + v["b"] + v["c"]) / 3},
)

# ─────────────── 能量守恒 ───────────────

_register_implicit(
    "energy_conservation", ["energy_conservation", "KE_PE_total", "mechanical_energy"],
    "physics",
    [("KE", ["KE", "kinetic_energy", "kinetic", "T"], (0, 100)),
     ("PE", ["PE", "potential_energy", "potential", "V"], (0, 100)),
     ("E", ["E", "total_energy", "mechanical", "H"], (0, 200))],
    satisfy=lambda v: {**v, "E": v["KE"] + v["PE"]},
)

# ─────────────── 动量守恒 ───────────────

_register_implicit(
    "momentum_conservation", ["momentum_conservation", "p1_plus_p2", "total_momentum"],
    "physics",
    [("m1", ["m1", "mass_1", "object_1_mass"], (0.1, 20)),
     ("v1", ["v1", "velocity_1", "speed_1"], (-10, 10)),
     ("m2", ["m2", "mass_2", "object_2_mass"], (0.1, 20)),
     ("v2", ["v2", "velocity_2", "speed_2"], (-10, 10)),
     ("P", ["P", "total_momentum", "p_total", "momentum_sum"], (-200, 200))],
    satisfy=lambda v: {**v, "P": v["m1"]*v["v1"] + v["m2"]*v["v2"]},
)

# ─────────────── 椭圆方程 ───────────────

_register_implicit(
    "ellipse_eq", ["ellipse_equation", "ellipse_standard_form"],
    "geometry",
    [("x", ["x", "coord_x", "horizontal"], (-10, 10)),
     ("a", ["a", "semi_major", "half_width"], (1, 15)),
     ("y", ["y", "coord_y", "vertical"], (-10, 10)),
     ("b", ["b", "semi_minor", "half_height"], (1, 15)),
     ("val", ["val", "equation_value", "lhs", "norm_sq"], (0, 2))],
    satisfy=lambda v: {**v, "val": (v["x"]/v["a"])**2 + (v["y"]/v["b"])**2},
)

# ─────────────── 理想气体 ───────────────

_register_implicit(
    "ideal_gas", ["ideal_gas_law", "PV_nRT", "gas_equation"],
    "physics",
    [("P", ["P", "pressure", "gas_pressure", "p_val"], (0.1, 100)),
     ("V", ["V", "volume", "gas_volume", "v_val"], (0.1, 100)),
     ("n", ["n", "moles", "amount", "mol"], (0.1, 10)),
     ("T", ["T", "temperature", "temp", "kelvin"], (100, 500))],
    # R = 8.314, but use R=1 for cleaner numbers
    satisfy=lambda v: {**v, "P": v["n"] * v["T"] / v["V"]},
)

# ─────────────── 欧姆定律三变量 ───────────────

_register_implicit(
    "ohms_law_implicit", ["ohms_law_3var", "V_I_R_relation"],
    "physics",
    [("V", ["V", "voltage", "potential", "emf"], (0.1, 100)),
     ("I", ["I", "current", "amperage", "amps"], (0.01, 10)),
     ("R", ["R", "resistance", "ohms", "impedance"], (0.1, 100))],
    satisfy=lambda v: {**v, "V": v["I"] * v["R"]},
)

# ─────────────── 角度互补 / 互余 ───────────────

_register_implicit(
    "complementary_angles", ["complementary", "sum_90", "right_angle_pair"],
    "geometry",
    [("alpha", ["alpha", "angle_1", "first_angle", "a_deg"], (1, 89)),
     ("beta", ["beta", "angle_2", "second_angle", "b_deg"], (1, 89))],
    satisfy=lambda v: {**v, "beta": 90.0 - v["alpha"]},
)

_register_implicit(
    "supplementary_angles", ["supplementary", "sum_180", "straight_angle_pair"],
    "geometry",
    [("alpha", ["alpha", "angle_1", "first_angle", "a_deg"], (1, 179)),
     ("beta", ["beta", "angle_2", "second_angle", "b_deg"], (1, 179))],
    satisfy=lambda v: {**v, "beta": 180.0 - v["alpha"]},
)

# ─────────────── 正弦定理 ───────────────

_register_implicit(
    "sine_rule", ["law_of_sines", "sine_rule", "sin_law"],
    "geometry",
    [("a", ["a", "side_a", "opp_A"], (1, 20)),
     ("A", ["A", "angle_A", "alpha"], (0.2, 2.9)),
     ("b", ["b", "side_b", "opp_B"], (1, 20)),
     ("B", ["B", "angle_B", "beta"], (0.2, 2.9))],
    # a/sin(A) = b/sin(B) => b = a * sin(B) / sin(A)
    satisfy=lambda v: {**v, "b": v["a"] * math.sin(v["B"]) / math.sin(v["A"])},
)

# ─────────────── 复数模 ───────────────

_register_implicit(
    "complex_modulus", ["complex_abs", "modulus", "absolute_value_complex"],
    "algebra",
    [("real", ["real", "re", "real_part", "x"], (-10, 10)),
     ("imag", ["imag", "im", "imaginary_part", "y"], (-10, 10)),
     ("mod", ["mod", "modulus", "absolute_value", "r"], (0, 15))],
    satisfy=lambda v: {**v, "mod": math.sqrt(v["real"]**2 + v["imag"]**2)},
)

# ─────────────── 指数 / 对数等式 ───────────────

_register_implicit(
    "exp_log_identity", ["exp_log", "exponential_logarithm", "inverse_pair"],
    "algebra",
    [("x", ["x", "input", "argument", "val"], (0.01, 10)),
     ("y", ["y", "log_x", "natural_log", "ln_val"], (-5, 3))],
    satisfy=lambda v: {**v, "y": math.log(v["x"])},
)
