"""
多变量函数注册 — 20 个函数（3-5 个变量）。
涵盖：多项式、统计、几何、向量、物理。
"""

import math
from .registry import make_multivar_generator

# ─────────────── 多项式 ───────────────

make_multivar_generator(
    "quadratic_eval", ["quadratic", "ax2_bx_c", "second_degree_polynomial", "parabola_eval"],
    "polynomial",
    var_defs=[
        ("a", ["a", "coeff_a", "leading_coeff", "quadratic_coeff"], (-5, 5)),
        ("b", ["b", "coeff_b", "linear_coeff", "first_coeff"], (-5, 5)),
        ("c", ["c", "coeff_c", "constant_term", "intercept"], (-5, 5)),
        ("x", ["x", "input", "variable", "eval_point"], (-5, 5)),
    ],
    output_role="y", output_keys=["y", "output", "result", "value", "f_x"],
    compute=lambda v: v["a"] * v["x"]**2 + v["b"] * v["x"] + v["c"],
)

make_multivar_generator(
    "discriminant", ["discriminant", "delta", "b2_minus_4ac"],
    "polynomial",
    var_defs=[
        ("a", ["a", "coeff_a", "leading"], (-5, 5)),
        ("b", ["b", "coeff_b", "linear"], (-10, 10)),
        ("c", ["c", "coeff_c", "constant"], (-5, 5)),
    ],
    output_role="D", output_keys=["D", "discriminant", "delta", "det", "disc"],
    compute=lambda v: v["b"]**2 - 4 * v["a"] * v["c"],
)

make_multivar_generator(
    "linear_eval", ["linear", "ax_plus_b", "affine", "first_degree"],
    "polynomial",
    var_defs=[
        ("a", ["a", "slope", "gradient", "m", "coeff"], (-10, 10)),
        ("x", ["x", "input", "variable", "point"], (-10, 10)),
        ("b", ["b", "intercept", "offset", "bias", "constant"], (-10, 10)),
    ],
    output_role="y", output_keys=["y", "output", "result", "f_x", "value"],
    compute=lambda v: v["a"] * v["x"] + v["b"],
)

# ─────────────── 统计 ───────────────

make_multivar_generator(
    "z_score", ["z_score", "standard_score", "z_value", "standardize"],
    "statistics",
    var_defs=[
        ("x", ["x", "value", "observation", "data_point", "sample"], (-50, 50)),
        ("mu", ["mu", "mean", "average", "expected_value", "center"], (-20, 20)),
        ("sigma", ["sigma", "std", "standard_deviation", "spread", "sd"], (0.1, 10)),
    ],
    output_role="z", output_keys=["z", "z_score", "standard_score", "z_val"],
    compute=lambda v: (v["x"] - v["mu"]) / v["sigma"],
)

make_multivar_generator(
    "normalize", ["normalize", "min_max_scale", "rescale", "feature_scale"],
    "statistics",
    var_defs=[
        ("x", ["x", "value", "raw", "original"], (-50, 50)),
        ("lo", ["lo", "min", "minimum", "lower_bound", "x_min"], (-60, -1)),
        ("hi", ["hi", "max", "maximum", "upper_bound", "x_max"], (1, 60)),
    ],
    output_role="scaled", output_keys=["scaled", "normalized", "result", "output", "n_val"],
    compute=lambda v: (v["x"] - v["lo"]) / (v["hi"] - v["lo"]),
)

make_multivar_generator(
    "weighted_average", ["weighted_average", "weighted_mean", "w_avg"],
    "statistics",
    var_defs=[
        ("w1", ["w1", "weight_1", "w_a", "importance_1"], (0.1, 10)),
        ("v1", ["v1", "value_1", "x_a", "data_1"], (-10, 10)),
        ("w2", ["w2", "weight_2", "w_b", "importance_2"], (0.1, 10)),
        ("v2", ["v2", "value_2", "x_b", "data_2"], (-10, 10)),
    ],
    output_role="avg", output_keys=["avg", "weighted_avg", "result", "mean", "w_mean"],
    compute=lambda v: (v["w1"]*v["v1"] + v["w2"]*v["v2"]) / (v["w1"] + v["w2"]),
)

# ─────────────── 向量运算 ───────────────

make_multivar_generator(
    "dot_product_2d", ["dot_product", "inner_product", "scalar_product", "dot_2d"],
    "vector",
    var_defs=[
        ("x1", ["x1", "a_x", "vec_a_x", "u1", "component_ax"], (-10, 10)),
        ("y1", ["y1", "a_y", "vec_a_y", "u2", "component_ay"], (-10, 10)),
        ("x2", ["x2", "b_x", "vec_b_x", "v1", "component_bx"], (-10, 10)),
        ("y2", ["y2", "b_y", "vec_b_y", "v2", "component_by"], (-10, 10)),
    ],
    output_role="dot", output_keys=["dot", "dot_product", "inner", "scalar_result", "d_val"],
    compute=lambda v: v["x1"]*v["x2"] + v["y1"]*v["y2"],
)

make_multivar_generator(
    "cross_product_2d", ["cross_product_2d", "cross_2d", "signed_area", "wedge_product"],
    "vector",
    var_defs=[
        ("x1", ["x1", "a_x", "u1", "vec_a_x"], (-10, 10)),
        ("y1", ["y1", "a_y", "u2", "vec_a_y"], (-10, 10)),
        ("x2", ["x2", "b_x", "v1", "vec_b_x"], (-10, 10)),
        ("y2", ["y2", "b_y", "v2", "vec_b_y"], (-10, 10)),
    ],
    output_role="cross", output_keys=["cross", "cross_z", "signed_area_val", "wedge"],
    compute=lambda v: v["x1"]*v["y2"] - v["y1"]*v["x2"],
)

make_multivar_generator(
    "vector_magnitude_3d", ["magnitude_3d", "vector_length_3d", "norm_3d", "euclidean_norm"],
    "vector",
    var_defs=[
        ("x", ["x", "vx", "comp_x", "i_comp"], (-10, 10)),
        ("y", ["y", "vy", "comp_y", "j_comp"], (-10, 10)),
        ("z", ["z", "vz", "comp_z", "k_comp"], (-10, 10)),
    ],
    output_role="mag", output_keys=["mag", "magnitude", "length", "norm", "abs_val"],
    compute=lambda v: math.sqrt(v["x"]**2 + v["y"]**2 + v["z"]**2),
)

# ─────────────── 几何 ───────────────

make_multivar_generator(
    "point_distance_2d", ["distance_2d", "point_distance", "euclidean_dist_2d"],
    "geometry",
    var_defs=[
        ("x1", ["x1", "p1_x", "start_x", "from_x"], (-20, 20)),
        ("y1", ["y1", "p1_y", "start_y", "from_y"], (-20, 20)),
        ("x2", ["x2", "p2_x", "end_x", "to_x"], (-20, 20)),
        ("y2", ["y2", "p2_y", "end_y", "to_y"], (-20, 20)),
    ],
    output_role="dist", output_keys=["dist", "distance", "d", "length", "separation"],
    compute=lambda v: math.sqrt((v["x2"]-v["x1"])**2 + (v["y2"]-v["y1"])**2),
)

make_multivar_generator(
    "trapezoid_area", ["trapezoid_area", "trapezium_area", "trap_area"],
    "geometry",
    var_defs=[
        ("a", ["a", "top", "parallel_side_1", "base_1", "upper"], (0.1, 20)),
        ("b", ["b", "bottom", "parallel_side_2", "base_2", "lower"], (0.1, 20)),
        ("h", ["h", "height", "altitude", "perpendicular", "dist"], (0.1, 20)),
    ],
    output_role="area", output_keys=["area", "A", "surface", "trap_area_val"],
    compute=lambda v: (v["a"] + v["b"]) * v["h"] / 2,
)

make_multivar_generator(
    "cylinder_volume", ["cylinder_volume", "vol_cylinder", "cyl_vol"],
    "geometry",
    var_defs=[
        ("r", ["r", "radius", "cyl_radius", "base_radius"], (0.1, 10)),
        ("h", ["h", "height", "length", "cyl_height"], (0.1, 20)),
    ],
    output_role="V", output_keys=["V", "volume", "capacity", "vol_val"],
    compute=lambda v: math.pi * v["r"]**2 * v["h"],
)

make_multivar_generator(
    "sphere_volume", ["sphere_volume", "vol_sphere", "ball_volume"],
    "geometry",
    var_defs=[
        ("r", ["r", "radius", "sphere_radius", "R"], (0.1, 10)),
    ],
    output_role="V", output_keys=["V", "volume", "capacity", "vol_val"],
    compute=lambda v: (4/3) * math.pi * v["r"]**3,
)

# ─────────────── 插值 / 变换 ───────────────

make_multivar_generator(
    "linear_interp", ["lerp", "linear_interpolation", "mix", "blend"],
    "interpolation",
    var_defs=[
        ("a", ["a", "start", "from_val", "begin", "v0"], (-10, 10)),
        ("b", ["b", "end", "to_val", "finish", "v1"], (-10, 10)),
        ("t", ["t", "factor", "alpha", "ratio", "blend_factor"], (0, 1)),
    ],
    output_role="result", output_keys=["result", "output", "interpolated", "mixed", "blended"],
    compute=lambda v: v["a"] + v["t"] * (v["b"] - v["a"]),
)

make_multivar_generator(
    "clamp", ["clamp", "clip", "constrain", "bound", "limit"],
    "utility",
    var_defs=[
        ("x", ["x", "value", "input", "raw"], (-20, 20)),
        ("lo", ["lo", "min", "lower", "floor", "low"], (-15, 0)),
        ("hi", ["hi", "max", "upper", "ceiling", "high"], (0, 15)),
    ],
    output_role="clamped", output_keys=["clamped", "clipped", "bounded", "result", "output"],
    compute=lambda v: max(v["lo"], min(v["hi"], v["x"])),
)

# ─────────────── 物理 ───────────────

make_multivar_generator(
    "kinetic_energy", ["kinetic_energy", "KE", "half_mv2", "energy_kinetic"],
    "physics",
    var_defs=[
        ("m", ["m", "mass", "object_mass", "mass_kg"], (0.1, 100)),
        ("v", ["v", "velocity", "speed", "v_val"], (0.1, 30)),
    ],
    output_role="E", output_keys=["E", "energy", "KE", "kinetic", "e_val"],
    compute=lambda v: 0.5 * v["m"] * v["v"]**2,
)

make_multivar_generator(
    "gravitational_force", ["gravitational_force", "newton_gravity", "F_gravity"],
    "physics",
    var_defs=[
        ("m1", ["m1", "mass_1", "object_a_mass", "M"], (1, 100)),
        ("m2", ["m2", "mass_2", "object_b_mass", "m"], (1, 100)),
        ("r", ["r", "distance", "separation", "radius", "d"], (1, 50)),
    ],
    output_role="F",
    output_keys=["F", "force", "gravitational_force_val", "attraction"],
    # G = 6.674e-11, but we use G=1 for cleaner numbers
    compute=lambda v: v["m1"] * v["m2"] / v["r"]**2,
)

make_multivar_generator(
    "law_of_cosines", ["law_of_cosines", "cosine_rule", "generalized_pythagorean"],
    "geometry",
    var_defs=[
        ("a", ["a", "side_a", "first_side", "s1"], (1, 20)),
        ("b", ["b", "side_b", "second_side", "s2"], (1, 20)),
        ("C", ["C", "angle_C", "included_angle", "gamma", "theta"], (0.1, 3.0)),
    ],
    output_role="c",
    output_keys=["c", "side_c", "opposite_side", "third_side", "result"],
    compute=lambda v: math.sqrt(v["a"]**2 + v["b"]**2 - 2*v["a"]*v["b"]*math.cos(v["C"])),
)
