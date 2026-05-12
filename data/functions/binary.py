"""
双变量函数注册 — 28 个函数。
涵盖：算术、比较、三角、统计、距离、物理公式。
"""

import math
from .registry import make_binary_generator

# ─────────────── 基础算术 ───────────────

make_binary_generator("add", ["add", "addition", "sum", "plus", "total"], "arithmetic", lambda a, b: a + b, domain_a=(-1e4, 1e4), domain_b=(-1e4, 1e4), tier=0, log_uniform=True)
make_binary_generator("subtract", ["subtract", "subtraction", "minus", "difference", "diff"], "arithmetic", lambda a, b: a - b, domain_a=(-1e4, 1e4), domain_b=(-1e4, 1e4), tier=0, log_uniform=True)
make_binary_generator("multiply", ["multiply", "multiplication", "product", "times", "mul"], "arithmetic", lambda a, b: a * b, domain_a=(-1e4, 1e4), domain_b=(-1e4, 1e4), tier=0, log_uniform=True)
make_binary_generator("divide", ["divide", "division", "quotient", "ratio", "div"], "arithmetic", lambda a, b: a / b, domain_a=(-1e4, 1e4), domain_b=(0.1, 1e4), tier=0, log_uniform=True)
make_binary_generator("power", ["power", "exponentiation", "pow", "raised_to"], "arithmetic", lambda a, b: a ** b, domain_a=(0.1, 100), domain_b=(0.1, 4), tier=0, log_uniform=True)
make_binary_generator("modulo", ["modulo", "mod", "remainder", "modular"], "arithmetic", lambda a, b: math.fmod(a, b), domain_a=(0, 1e4), domain_b=(1, 1e3), tier=0, log_uniform=True)

# ─────────────── 对数 ───────────────

make_binary_generator("log_base", ["log_base", "logarithm_base", "log_b", "change_of_base"],
                      "logarithmic", lambda a, b: math.log(b) / math.log(a),
                      domain_a=(1.1, 10), domain_b=(0.1, 100))

# ─────────────── 比较 / 选择 ───────────────

make_binary_generator("max", ["max", "maximum", "larger", "greater_of", "max_of"], "comparison", max, domain_a=(-1e4, 1e4), domain_b=(-1e4, 1e4), tier=0, log_uniform=True)
make_binary_generator("min", ["min", "minimum", "smaller", "lesser_of", "min_of"], "comparison", min, domain_a=(-1e4, 1e4), domain_b=(-1e4, 1e4), tier=0, log_uniform=True)

# ─────────────── 整数运算 ───────────────

make_binary_generator("gcd", ["gcd", "greatest_common_divisor", "hcf", "highest_common_factor"],
                      "integer", lambda a, b: float(math.gcd(int(a), int(b))),
                      domain_a=(1, 100), domain_b=(1, 100))
make_binary_generator("lcm", ["lcm", "least_common_multiple", "lowest_common_multiple"],
                      "integer", lambda a, b: float(abs(int(a) * int(b)) // math.gcd(int(a), int(b))),
                      domain_a=(1, 50), domain_b=(1, 50))

# ─────────────── 三角 ───────────────

make_binary_generator("hypot", ["hypot", "hypotenuse", "pythagorean_sum", "euclidean_norm_2d"],
                      "trigonometric", math.hypot, domain_a=(-10, 10), domain_b=(-10, 10))
make_binary_generator("atan2", ["atan2", "arctan2", "two_arg_arctan", "angle_of_vector"],
                      "trigonometric", math.atan2, domain_a=(-10, 10), domain_b=(-10, 10))

# ─────────────── 统计 ───────────────

make_binary_generator("mean", ["mean", "average", "arithmetic_mean", "avg", "midpoint"],
                      "statistics", lambda a, b: (a + b) / 2, tier=0)
make_binary_generator("geometric_mean", ["geometric_mean", "geo_mean", "geomean"],
                      "statistics", lambda a, b: math.sqrt(a * b),
                      domain_a=(0.01, 50), domain_b=(0.01, 50))
make_binary_generator("harmonic_mean", ["harmonic_mean", "harm_mean", "reciprocal_average"],
                      "statistics", lambda a, b: 2 * a * b / (a + b),
                      domain_a=(0.1, 50), domain_b=(0.1, 50))
make_binary_generator("rms", ["rms", "root_mean_square", "quadratic_mean"],
                      "statistics", lambda a, b: math.sqrt((a*a + b*b) / 2))

# ─────────────── 距离 ───────────────

make_binary_generator("abs_diff", ["abs_diff", "absolute_difference", "distance_1d", "gap"],
                      "distance", lambda a, b: abs(a - b))

# ─────────────── 百分比 ───────────────

make_binary_generator("percent_of", ["percent_of", "percentage", "pct", "fraction_of"],
                      "percentage", lambda a, b: a * b / 100,
                      domain_a=(0, 100), domain_b=(0, 1000))

# ─────────────── 物理 / 几何公式 ───────────────

_PHYS_A_KEYS = ["current", "I", "amperage", "amps", "electric_current", "i_val", "current_a"]
_PHYS_B_KEYS = ["resistance", "R", "ohms", "impedance", "r_val", "resistance_ohm"]
_PHYS_OUT_KEYS = ["voltage", "V", "potential", "emf", "v_val", "voltage_v"]
make_binary_generator("ohms_law", ["ohms_law", "V_equals_IR", "voltage_law"], "physics",
                      lambda a, b: a * b, domain_a=(0.1, 10), domain_b=(1, 100),
                      input_a_keys=_PHYS_A_KEYS, input_b_keys=_PHYS_B_KEYS, output_keys=_PHYS_OUT_KEYS)

_VEL_A_KEYS = ["distance", "d", "displacement", "path_length", "dist", "s"]
_VEL_B_KEYS = ["time", "t", "duration", "elapsed", "dt", "time_s"]
_VEL_OUT_KEYS = ["velocity", "v", "speed", "rate", "v_val"]
make_binary_generator("velocity", ["velocity", "speed_formula", "v_equals_d_over_t"], "physics",
                      lambda a, b: a / b, domain_a=(1, 100), domain_b=(0.1, 20),
                      input_a_keys=_VEL_A_KEYS, input_b_keys=_VEL_B_KEYS, output_keys=_VEL_OUT_KEYS)

_DENS_A_KEYS = ["mass", "m", "weight", "mass_kg", "m_val"]
_DENS_B_KEYS = ["volume", "V", "vol", "capacity", "v_vol"]
_DENS_OUT_KEYS = ["density", "rho", "mass_density", "specific_mass", "d_val"]
make_binary_generator("density", ["density", "mass_over_volume", "rho_formula"], "physics",
                      lambda a, b: a / b, domain_a=(0.1, 100), domain_b=(0.1, 50),
                      input_a_keys=_DENS_A_KEYS, input_b_keys=_DENS_B_KEYS, output_keys=_DENS_OUT_KEYS)

_PRES_A_KEYS = ["force", "F", "load", "force_n", "f_val"]
_PRES_B_KEYS = ["area", "A", "surface_area", "cross_section", "a_area"]
_PRES_OUT_KEYS = ["pressure", "P", "stress", "pascal", "p_val"]
make_binary_generator("pressure", ["pressure", "force_over_area", "P_formula"], "physics",
                      lambda a, b: a / b, domain_a=(1, 1000), domain_b=(0.1, 100),
                      input_a_keys=_PRES_A_KEYS, input_b_keys=_PRES_B_KEYS, output_keys=_PRES_OUT_KEYS)

make_binary_generator("rect_area", ["rect_area", "rectangle_area", "area_lw", "length_times_width"],
                      "geometry", lambda a, b: a * b, domain_a=(0.1, 50), domain_b=(0.1, 50),
                      input_a_keys=["length", "l", "width_a", "side_a", "dim_1", "len"],
                      input_b_keys=["width", "w", "breadth", "side_b", "dim_2", "wid"],
                      output_keys=["area", "A", "surface", "sq_units", "a_val"])

make_binary_generator("triangle_area", ["triangle_area", "half_base_height", "area_bh"],
                      "geometry", lambda a, b: 0.5 * a * b, domain_a=(0.1, 50), domain_b=(0.1, 50),
                      input_a_keys=["base", "b", "base_length", "triangle_base"],
                      input_b_keys=["height", "h", "altitude", "triangle_height"],
                      output_keys=["area", "A", "triangle_area_val", "surface"])

make_binary_generator("spring_force", ["spring_force", "hookes_law", "F_equals_kx"], "physics",
                      lambda a, b: a * b, domain_a=(0.1, 100), domain_b=(0.01, 5),
                      input_a_keys=["spring_constant", "k", "stiffness", "k_val"],
                      input_b_keys=["displacement", "x", "extension", "stretch", "dx"],
                      output_keys=["force", "F", "spring_force_val", "restoring_force"])

make_binary_generator("rect_perimeter", ["rect_perimeter", "perimeter_lw", "circumference_rect"],
                      "geometry", lambda a, b: 2 * (a + b), domain_a=(0.1, 50), domain_b=(0.1, 50),
                      input_a_keys=["length", "l", "side_a", "dim_1"],
                      input_b_keys=["width", "w", "side_b", "dim_2"],
                      output_keys=["perimeter", "P", "circumference", "boundary"])
