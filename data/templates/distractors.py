"""
干扰字段生成器。
生成与数学关系完全无关的字段，模型必须学会忽略它们。
"""

import random
import string
import uuid


def _rand_id():
    return f"entry_{random.randint(1000, 9999)}"

def _rand_uuid():
    return str(uuid.uuid4())[:8]

def _rand_author():
    return random.choice(["system", "generator_v2", "auto", "pipeline",
                          "compute_engine", "solver", "mathlib", "evaluator"])

def _rand_timestamp():
    y = random.randint(2020, 2026)
    m = random.randint(1, 12)
    d = random.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"

def _rand_version():
    return f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,9)}"

def _rand_notes():
    return random.choice(["verified", "auto-generated", "needs review",
                          "preliminary", "validated", "draft", "final",
                          "cross-checked", "approximate", "exact"])

def _rand_source():
    return random.choice(["computed", "measured", "estimated", "derived",
                          "simulated", "analytical", "numerical", "interpolated"])

def _rand_unit():
    return random.choice(["SI", "CGS", "natural", "dimensionless",
                          "arbitrary", "normalized", "reduced", "standard"])

def _rand_precision():
    return random.choice(["float32", "float64", "exact", "double",
                          "half", "extended", "arbitrary_precision"])

def _rand_status():
    return random.choice(["active", "archived", "draft", "published",
                          "deprecated", "pending", "reviewed"])

def _rand_tag():
    tags = ["math", "science", "physics", "algebra", "geometry",
            "calculus", "trigonometry", "statistics", "analysis",
            "computation", "numerical", "symbolic", "applied"]
    return random.choice(tags)

def _rand_confidence():
    return round(random.uniform(0.8, 1.0), 3)

def _rand_error():
    return round(random.uniform(0.0001, 0.05), 4)

def _rand_sample_size():
    return random.randint(10, 1000)

def _rand_iteration():
    return random.randint(1, 100)

def _rand_priority():
    return random.randint(1, 10)

def _rand_verified():
    return random.choice([True, False])

def _rand_exact():
    return random.choice([True, False])

def _rand_deprecated():
    return False

def _rand_converged():
    return True

def _rand_label():
    letters = random.choices(string.ascii_uppercase, k=2)
    return "".join(letters) + str(random.randint(1, 99))


# ═══════════════════════════════════════════════════
# 标量干扰字段池
# ═══════════════════════════════════════════════════

DISTRACTOR_POOL = {
    # 文本类
    "id": _rand_id,
    "uuid": _rand_uuid,
    "author": _rand_author,
    "timestamp": _rand_timestamp,
    "date": _rand_timestamp,
    "version": _rand_version,
    "notes": _rand_notes,
    "source": _rand_source,
    "unit": _rand_unit,
    "units": _rand_unit,
    "precision": _rand_precision,
    "status": _rand_status,
    "tag": _rand_tag,
    "label": _rand_label,
    "description": lambda: f"auto generated record {random.randint(1,9999)}",
    "comment": lambda: random.choice(["ok", "check", "reviewed", "n/a", "see notes"]),
    "format": lambda: random.choice(["json", "csv", "binary", "text"]),
    "encoding": lambda: random.choice(["utf-8", "ascii", "latin-1"]),
    "origin": lambda: random.choice(["local", "remote", "cached", "fresh"]),
    # 数值类
    "confidence": _rand_confidence,
    "error_bar": _rand_error,
    "uncertainty": _rand_error,
    "sample_size": _rand_sample_size,
    "iteration": _rand_iteration,
    "priority": _rand_priority,
    "index": lambda: random.randint(0, 999),
    "batch_id": lambda: random.randint(1, 50),
    "seed": lambda: random.randint(0, 2**16),
    # 布尔类
    "verified": _rand_verified,
    "is_exact": _rand_exact,
    "deprecated": _rand_deprecated,
    "converged": _rand_converged,
    "cached": lambda: random.choice([True, False]),
    "is_valid": lambda: True,
}


# ═══════════════════════════════════════════════════
# 嵌套干扰生成器 (对象 / 数组)
# ═══════════════════════════════════════════════════

def _gen_nested_metadata():
    """生成一个 metadata 嵌套对象 (3-5 叶子)"""
    fields = {}
    pool = [
        ("created", _rand_timestamp),
        ("modified", _rand_timestamp),
        ("version", _rand_version),
        ("author", _rand_author),
        ("status", _rand_status),
        ("source", _rand_source),
        ("encoding", lambda: random.choice(["utf-8", "ascii"])),
    ]
    n = random.randint(3, min(5, len(pool)))
    for key, fn in random.sample(pool, n):
        fields[key] = fn()
    return fields


def _gen_nested_provenance():
    """生成一个 provenance / 来源追踪嵌套对象 (3-6 叶子)"""
    fields = {
        "origin": random.choice(["local", "remote", "cloud", "archive"]),
        "pipeline_version": _rand_version(),
        "run_id": _rand_uuid(),
    }
    if random.random() < 0.5:
        fields["parent_id"] = _rand_uuid()
    if random.random() < 0.5:
        fields["elapsed_ms"] = random.randint(10, 5000)
    if random.random() < 0.5:
        fields["retries"] = random.randint(0, 3)
    return fields


def _gen_nested_quality():
    """生成一个 quality / 质量评估嵌套对象 (3-5 叶子)"""
    fields = {
        "confidence": _rand_confidence(),
        "validated": _rand_verified(),
    }
    if random.random() < 0.7:
        fields["error_bound"] = _rand_error()
    if random.random() < 0.5:
        fields["method"] = random.choice(["cross_validation", "bootstrap", "analytic", "monte_carlo"])
    if random.random() < 0.5:
        fields["sample_count"] = _rand_sample_size()
    return fields


def _gen_nested_config():
    """生成一个 config 嵌套对象 (3-6 叶子)"""
    fields = {
        "tolerance": round(random.uniform(1e-6, 1e-2), 8),
        "max_iterations": random.randint(100, 10000),
    }
    if random.random() < 0.6:
        fields["algorithm"] = random.choice(["newton", "bisection", "gradient_descent", "simplex", "bfgs"])
    if random.random() < 0.5:
        fields["convergence"] = _rand_converged()
    if random.random() < 0.5:
        fields["seed"] = random.randint(0, 2**16)
    if random.random() < 0.4:
        fields["precision"] = _rand_precision()
    return fields


def _gen_array_tags():
    """生成一个 tags 数组 (2-5 叶子)"""
    all_tags = ["math", "science", "physics", "algebra", "geometry",
                "calculus", "trigonometry", "statistics", "analysis",
                "validated", "draft", "v2", "production", "experimental",
                "numerical", "symbolic", "applied", "theoretical"]
    n = random.randint(2, 5)
    return random.sample(all_tags, n)


def _gen_array_references():
    """生成一个 references 数组 (2-4 叶子)"""
    refs = [f"ref_{random.randint(100,999)}" for _ in range(random.randint(2, 4))]
    return refs


def _gen_array_log_entries():
    """生成一个嵌套数组: log 条目 (每条 2-3 叶子, 共 2-4 条 = 4-12 叶子)"""
    entries = []
    n = random.randint(2, 4)
    for _ in range(n):
        entry = {
            "timestamp": _rand_timestamp(),
            "event": random.choice(["start", "checkpoint", "complete", "retry", "error"]),
        }
        if random.random() < 0.5:
            entry["duration_ms"] = random.randint(1, 500)
        entries.append(entry)
    return entries


# 嵌套干扰池: key -> generator
NESTED_DISTRACTOR_POOL = {
    # 嵌套对象 (每个产出 3-6 叶子)
    "metadata": _gen_nested_metadata,
    "provenance": _gen_nested_provenance,
    "quality": _gen_nested_quality,
    "solver_config": _gen_nested_config,
    # 数组 (每个产出 2-12 叶子)
    "tags": _gen_array_tags,
    "references": _gen_array_references,
    "log": _gen_array_log_entries,
}


# ═══════════════════════════════════════════════════
# 注入接口
# ═══════════════════════════════════════════════════

def inject_distractors(doc: dict, n_min: int = 0, n_max: int = 5,
                       nested_prob: float = 0.0) -> dict:
    """
    向文档中注入随机干扰字段。

    Args:
        doc: 原始文档 dict
        n_min: 最少注入标量字段数
        n_max: 最多注入标量字段数
        nested_prob: 注入嵌套干扰的概率 (0.0-1.0)。
                     设为 >0 时，在标量干扰之外，额外注入 1-3 个嵌套对象/数组。

    Returns:
        注入干扰字段后的文档（原地修改 + 返回）
    """
    # ── 标量干扰 ──
    n = random.randint(n_min, n_max)
    if n > 0:
        available = [k for k in DISTRACTOR_POOL if k not in doc]
        if available:
            chosen = random.sample(available, min(n, len(available)))
            for key in chosen:
                doc[key] = DISTRACTOR_POOL[key]()

    # ── 嵌套干扰 ──
    if nested_prob > 0 and random.random() < nested_prob:
        n_nested = random.randint(1, 3)
        available_nested = [k for k in NESTED_DISTRACTOR_POOL if k not in doc]
        if available_nested:
            chosen_nested = random.sample(available_nested, min(n_nested, len(available_nested)))
            for key in chosen_nested:
                doc[key] = NESTED_DISTRACTOR_POOL[key]()

    return doc
