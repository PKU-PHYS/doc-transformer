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


# 所有干扰字段生成器
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


def inject_distractors(doc: dict, n_min: int = 0, n_max: int = 5) -> dict:
    """
    向文档中注入随机干扰字段。

    Args:
        doc: 原始文档 dict
        n_min: 最少注入字段数
        n_max: 最多注入字段数

    Returns:
        注入干扰字段后的文档（原地修改 + 返回）
    """
    n = random.randint(n_min, n_max)
    if n == 0:
        return doc

    # 从池中选取不与现有键冲突的干扰字段
    available = [k for k in DISTRACTOR_POOL if k not in doc]
    if not available:
        return doc

    chosen = random.sample(available, min(n, len(available)))
    for key in chosen:
        doc[key] = DISTRACTOR_POOL[key]()

    return doc
