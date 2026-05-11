"""
结构模板引擎 — 控制 MathRelation 如何被渲染为多样化的 JSON 文档。

设计：
  每个模板是一个函数，接收标准化的参数，返回 dict/list。
  模板分为 5 大类，每类 20-30 个，组合后有效变体数达数十万。

  渲染流程：
    MathRelation → 选模板 → 选键名 → 渲染骨架 → 注入干扰 → 打乱顺序
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from ..functions.registry import MathRelation, FUNC_NAME_KEYS


# ═══════════════════════════════════════════════════
# 模板注册表
# ═══════════════════════════════════════════════════

_TEMPLATES: List[Callable] = []


def _t(fn):
    """注册一个模板函数。"""
    _TEMPLATES.append(fn)
    return fn


# ═══════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════

def _pick_keys(rel: MathRelation) -> Dict[str, str]:
    """为每个变量角色随机选一个键名。"""
    key_map = {}
    used = set()
    for role, syns in rel.var_synonyms.items():
        available = [s for s in syns if s not in used]
        if not available:
            available = syns
        chosen = random.choice(available)
        key_map[role] = chosen
        used.add(chosen)
    return key_map


def _pick_func_key() -> str:
    return random.choice(FUNC_NAME_KEYS)


def _pick_func_name(rel: MathRelation) -> str:
    return random.choice(rel.func_synonyms)


def _vars_as_pairs(rel: MathRelation, key_map: Dict[str, str]) -> List[Tuple[str, Any]]:
    """将变量转为 (key_name, value) 列表。"""
    return [(key_map[role], val) for role, val in rel.variables.items()]


def _shuffle_dict(d: dict) -> dict:
    """随机打乱 dict 的键顺序。"""
    items = list(d.items())
    random.shuffle(items)
    return dict(items)


# ═══════════════════════════════════════════════════
# A. 扁平模板 (30 个)
# ═══════════════════════════════════════════════════

@_t
def flat_basic(rel, km):
    """最简扁平: {func: name, var1: v1, var2: v2, ...}"""
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_no_func_name(rel, km):
    """无函数名: {var1: v1, var2: v2}"""
    doc = {}
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_with_category(rel, km):
    """带类别: {func: name, category: cat, ...vars...}"""
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    doc[random.choice(["category", "family", "group", "domain", "branch"])] = rel.category
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_with_description(rel, km):
    """带自动描述"""
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    desc_key = random.choice(["description", "desc", "info", "summary", "note"])
    doc[desc_key] = f"computing {rel.func_name} function"
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_reversed_order(rel, km):
    """变量逆序"""
    doc = {}
    items = list(rel.variables.items())
    items.reverse()
    for role, val in items:
        doc[km[role]] = val
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    return doc

@_t
def flat_output_first(rel, km):
    """输出放在最前面"""
    doc = {}
    # 找 output 类的角色
    out_roles = [r for r in rel.variables if "output" in r or r in ("y", "result", "output")]
    if not out_roles:
        out_roles = list(rel.variables.keys())[-1:]
    for r in out_roles:
        doc[km[r]] = rel.variables[r]
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        if role not in out_roles:
            doc[km[role]] = val
    return doc

@_t
def flat_with_id(rel, km):
    """带随机 ID"""
    doc = {"id": f"calc_{random.randint(1000,9999)}"}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_with_index(rel, km):
    """带序号"""
    doc = {"index": random.randint(0, 999)}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_prefixed_keys(rel, km):
    """键名带前缀: input_x, output_y"""
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        prefix = random.choice(["in_", "input_", "src_"]) if "input" in role else \
                 random.choice(["out_", "output_", "dst_"]) if "output" in role else ""
        doc[prefix + km[role]] = val
    return doc

@_t
def flat_numbered_keys(rel, km):
    """变量用编号键: field_0, field_1, ..."""
    doc = {}
    if rel.include_func_name:
        doc["field_name"] = _pick_func_name(rel)
    for i, (role, val) in enumerate(rel.variables.items()):
        doc[f"field_{i}"] = val
    return doc

# 更多扁平变体

@_t
def flat_verbose(rel, km):
    """详细描述式"""
    doc = {}
    if rel.include_func_name:
        doc["applied_function"] = _pick_func_name(rel)
        doc["function_category"] = rel.category
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_terse(rel, km):
    """极简 - 只有变量，用短键名"""
    doc = {}
    shorts = list("abcdefghij")
    for i, (role, val) in enumerate(rel.variables.items()):
        doc[shorts[i % len(shorts)]] = val
    return doc

@_t
def flat_with_type_hints(rel, km):
    """带类型提示"""
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        doc[km[role]] = val
        vtype = "number" if isinstance(val, (int, float)) else "text"
        doc[f"{km[role]}_type"] = vtype
    return doc

@_t
def flat_equation_string(rel, km):
    """带方程字符串描述"""
    doc = {"equation": f"{rel.func_name} relation"}
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_with_units(rel, km):
    """带单位"""
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        doc[km[role]] = val
    doc["units"] = random.choice(["dimensionless", "SI", "natural", "arbitrary"])
    return doc

@_t
def flat_labeled(rel, km):
    """变量名作为 label"""
    doc = {"label": _pick_func_name(rel) if rel.include_func_name else rel.category}
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_tagged(rel, km):
    """带 tag 列表"""
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        doc[km[role]] = val
    doc["tags"] = rel.category
    return doc

@_t
def flat_with_precision(rel, km):
    """带精度信息"""
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        doc[km[role]] = val
    doc["precision"] = random.choice(["float64", "exact", "double"])
    return doc

@_t
def flat_with_status(rel, km):
    """带状态"""
    doc = {"status": random.choice(["computed", "verified", "validated"])}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def flat_record(rel, km):
    """记录风格"""
    doc = {"record_type": "math_evaluation"}
    if rel.include_func_name:
        doc["expression"] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc


# ═══════════════════════════════════════════════════
# B. 嵌套模板 (30 个)
# ═══════════════════════════════════════════════════

@_t
def nested_params_result(rel, km):
    """参数分组: {func, params: {...}, result: val}"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    params = {km[r]: rel.variables[r] for r in in_roles}
    doc[random.choice(["params", "parameters", "args", "arguments", "inputs"])] = params
    doc[km[out_role]] = rel.variables[out_role]
    return doc

@_t
def nested_io_split(rel, km):
    """输入输出分离: {input: {...}, output: {...}}"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    inp_key = random.choice(["input", "inputs", "in", "given", "source"])
    out_key = random.choice(["output", "outputs", "out", "result", "target"])
    doc[inp_key] = {km[r]: rel.variables[r] for r in in_roles}
    doc[out_key] = {km[out_role]: rel.variables[out_role]}
    return doc

@_t
def nested_meta_data(rel, km):
    """元数据包裹: {meta: {type, category}, data: {vars}}"""
    meta = {}
    if rel.include_func_name:
        meta["type"] = _pick_func_name(rel)
    meta["category"] = rel.category
    data = {km[r]: rel.variables[r] for r in rel.variables}
    doc = {
        random.choice(["meta", "metadata", "info", "header"]): meta,
        random.choice(["data", "payload", "content", "body"]): data,
    }
    return doc

@_t
def nested_config_values(rel, km):
    """配置 + 值: {config: {func}, values: {vars}}"""
    config = {}
    if rel.include_func_name:
        config[_pick_func_key()] = _pick_func_name(rel)
    values = {km[r]: rel.variables[r] for r in rel.variables}
    doc = {
        random.choice(["config", "configuration", "setup", "settings"]): config,
        random.choice(["values", "data", "results", "entries"]): values,
    }
    return doc

@_t
def nested_deep_input(rel, km):
    """深嵌套输入: {func, computation: {input: {vars}, output: val}}"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    doc["computation"] = {
        "input": {km[r]: rel.variables[r] for r in in_roles},
        "output": rel.variables[out_role],
    }
    return doc

@_t
def nested_each_var_wrapped(rel, km):
    """每个变量单独包裹: {vars: [{name: x, value: 1}, ...]}"""
    items = []
    for role, val in rel.variables.items():
        items.append({"name": km[role], "val": val})
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    doc["variables"] = items
    return doc

@_t
def nested_result_wrapper(rel, km):
    """结果包裹: {...inputs, result: {value: y, type: number}}"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    for r in in_roles:
        doc[km[r]] = rel.variables[r]
    out_val = rel.variables[out_role]
    doc["result"] = {
        "value": out_val,
        "type": "number" if isinstance(out_val, (int, float)) else "text",
    }
    return doc

@_t
def nested_spec_output(rel, km):
    """spec + output: {spec: {func, inputs}, output: val}"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    spec = {}
    if rel.include_func_name:
        spec[_pick_func_key()] = _pick_func_name(rel)
    for r in in_roles:
        spec[km[r]] = rel.variables[r]
    doc = {
        random.choice(["spec", "specification", "definition", "formula"]): spec,
        km[out_role]: rel.variables[out_role],
    }
    return doc

@_t
def nested_context_target(rel, km):
    """context + target: {context: {inputs}, target: {output}}"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    doc = {
        "context": {km[r]: rel.variables[r] for r in in_roles},
        "target": {km[out_role]: rel.variables[out_role]},
    }
    if rel.include_func_name:
        doc["context"][_pick_func_key()] = _pick_func_name(rel)
    return doc

@_t
def nested_header_body(rel, km):
    """header + body"""
    header = {}
    if rel.include_func_name:
        header["name"] = _pick_func_name(rel)
    header["category"] = rel.category
    body = {km[r]: rel.variables[r] for r in rel.variables}
    return {"header": header, "body": body}

@_t
def nested_question_answer(rel, km):
    """Q&A 风格: {question: {func, inputs}, answer: {output}}"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    q = {}
    if rel.include_func_name:
        q[_pick_func_key()] = _pick_func_name(rel)
    for r in in_roles:
        q[km[r]] = rel.variables[r]
    doc = {
        random.choice(["question", "query", "problem", "task"]): q,
        random.choice(["answer", "solution", "response"]): {km[out_role]: rel.variables[out_role]},
    }
    return doc

@_t
def nested_input_output_flat_values(rel, km):
    """flat values with nested function info"""
    doc = {}
    if rel.include_func_name:
        doc["function_info"] = {
            "name": _pick_func_name(rel),
            "category": rel.category,
        }
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc

@_t
def nested_triple_depth(rel, km):
    """三层嵌套"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    doc = {
        "evaluation": {
            "function": {_pick_func_key(): _pick_func_name(rel)} if rel.include_func_name else {},
            "inputs": {km[r]: rel.variables[r] for r in in_roles},
        },
        "output": {km[out_role]: rel.variables[out_role]},
    }
    return doc

@_t
def nested_pairs(rel, km):
    """键值对列表: {pairs: [{key: k, value: v}, ...]}"""
    pairs = []
    if rel.include_func_name:
        pairs.append({"key": _pick_func_key(), "value": _pick_func_name(rel)})
    for role, val in rel.variables.items():
        pairs.append({"key": km[role], "value": val})
    return {"pairs": pairs}

@_t
def nested_experiment(rel, km):
    """实验记录风格"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    doc = {
        "experiment": {
            "method": _pick_func_name(rel) if rel.include_func_name else rel.category,
            "conditions": {km[r]: rel.variables[r] for r in in_roles},
        },
        "observation": {km[out_role]: rel.variables[out_role]},
    }
    return doc

@_t
def nested_two_level_params(rel, km):
    """两级参数"""
    roles = list(rel.variables.keys())
    mid = len(roles) // 2
    group_a = roles[:mid]
    group_b = roles[mid:]
    doc = {}
    if rel.include_func_name:
        doc[_pick_func_key()] = _pick_func_name(rel)
    doc["group_a"] = {km[r]: rel.variables[r] for r in group_a}
    doc["group_b"] = {km[r]: rel.variables[r] for r in group_b}
    return doc

@_t
def nested_source_derived(rel, km):
    """source/derived 风格"""
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    in_roles = roles[:-1]
    doc = {
        random.choice(["source", "given", "known", "provided"]): {km[r]: rel.variables[r] for r in in_roles},
        random.choice(["derived", "computed", "calculated", "inferred"]): {km[out_role]: rel.variables[out_role]},
    }
    if rel.include_func_name:
        doc["method"] = _pick_func_name(rel)
    return doc

@_t
def nested_record_fields(rel, km):
    """记录 + 字段"""
    fields = {}
    if rel.include_func_name:
        fields["type"] = _pick_func_name(rel)
    for role, val in rel.variables.items():
        fields[km[role]] = val
    return {"record": fields}

@_t
def nested_summary(rel, km):
    """概要风格"""
    doc = {"summary": {"operation": _pick_func_name(rel) if rel.include_func_name else rel.category}}
    doc["details"] = {km[r]: rel.variables[r] for r in rel.variables}
    return doc

@_t
def nested_labeled_values(rel, km):
    """labeled values"""
    doc = {}
    if rel.include_func_name:
        doc["label"] = _pick_func_name(rel)
    doc["values"] = {km[r]: rel.variables[r] for r in rel.variables}
    return doc


# ═══════════════════════════════════════════════════
# C. 数组模板 (20 个) — 多条同类函数采样组成数组
# ═══════════════════════════════════════════════════

def _make_array_template(inner_template_fn, min_companions=1, max_companions=7):
    """包装一个模板，使其生成包含多个采样点的数组。"""
    def array_template(rel, km):
        from ..functions.registry import FunctionRegistry
        items = [inner_template_fn(rel, km)]
        for _ in range(random.randint(min_companions, max_companions)):
            companion = resample_relation(rel)
            items.append(inner_template_fn(companion, km))
        return items
    return array_template


def resample_relation(rel: MathRelation) -> MathRelation:
    """用同一个函数重新采样变量值（快速近似）。"""
    import math
    new_vars = {}
    for role, val in rel.variables.items():
        if isinstance(val, (int, float)):
            # 在原值附近重新采样
            noise = random.gauss(0, max(abs(val) * 0.5, 1.0))
            new_vars[role] = round(val + noise, 6)
        else:
            new_vars[role] = val  # 文本变量保持不变
    return MathRelation(
        func_name=rel.func_name,
        func_synonyms=rel.func_synonyms,
        category=rel.category,
        variables=new_vars,
        var_synonyms=rel.var_synonyms,
        include_func_name=rel.include_func_name,
    )


# 注册标准数组模板 (2-8 个元素)
for _base in [flat_basic, flat_no_func_name, flat_with_id, flat_terse,
              flat_labeled, nested_params_result, nested_io_split,
              nested_context_target, nested_source_derived, nested_record_fields]:
    _TEMPLATES.append(_make_array_template(_base, min_companions=1, max_companions=7))

# 注册大数组模板 (4-12 个元素) — 专门用于生成长序列
for _base in [flat_basic, flat_no_func_name, flat_with_category,
              nested_params_result, nested_meta_data]:
    _TEMPLATES.append(_make_array_template(_base, min_companions=3, max_companions=11))


# ═══════════════════════════════════════════════════
# D. 复合函数专用模板 (10 个)
# ═══════════════════════════════════════════════════

@_t
def composite_pipeline(rel, km):
    """管道: {pipeline: [{stage, ...}, ...], final: val}"""
    stages = []
    for role, val in rel.variables.items():
        if "func" in role:
            stages.append({"operation": val})
        elif role not in ("output",):
            if stages:
                stages[-1][km[role]] = val
            else:
                # 非复合函数走到此模板时，回退为扁平
                stages.append({km[role]: val})
    doc = {"pipeline": stages} if stages else {}
    roles = list(rel.variables.keys())
    out_role = roles[-1]
    doc["final"] = rel.variables[out_role]
    return doc

@_t
def composite_chain(rel, km):
    """链式: {step_1: {func, input, output}, step_2: {func, input, output}}"""
    doc = {}
    step = 0
    current_step = {}
    for role, val in rel.variables.items():
        if "func" in role:
            if current_step:
                doc[f"step_{step}"] = current_step
                step += 1
            current_step = {_pick_func_key(): val}
        else:
            current_step[km[role]] = val
    if current_step:
        doc[f"step_{step}"] = current_step
    if not doc:
        # 回退：单步包裹所有变量
        doc["step_0"] = {km[r]: rel.variables[r] for r in rel.variables}
    return doc

@_t
def composite_nested_apply(rel, km):
    """嵌套应用"""
    doc = {}
    for role, val in rel.variables.items():
        doc[km[role]] = val
    return doc


# ═══════════════════════════════════════════════════
# E. 后处理变换
# ═══════════════════════════════════════════════════

def _post_shuffle(doc):
    """随机打乱顶层键顺序"""
    if isinstance(doc, dict):
        return _shuffle_dict(doc)
    return doc


def _post_maybe_flatten_single_nested(doc):
    """有概率将只有一个键的嵌套展平"""
    if not isinstance(doc, dict):
        return doc
    for key, val in list(doc.items()):
        if isinstance(val, dict) and len(val) == 1 and random.random() < 0.3:
            inner_key, inner_val = next(iter(val.items()))
            new_key = f"{key}_{inner_key}"
            doc[new_key] = inner_val
            del doc[key]
    return doc


# ═══════════════════════════════════════════════════
# 主引擎
# ═══════════════════════════════════════════════════

class TemplateEngine:
    """模板渲染引擎。"""

    @staticmethod
    def render(rel: MathRelation) -> Any:
        """
        将一条 MathRelation 渲染为 JSON 可序列化的 dict 或 list。

        Returns:
            dict 或 list，可以直接传给 JSONParser.parse()
        """
        km = _pick_keys(rel)
        template = random.choice(_TEMPLATES)
        doc = template(rel, km)

        # 后处理
        if random.random() < 0.7:
            doc = _post_shuffle(doc)
        if random.random() < 0.2:
            doc = _post_maybe_flatten_single_nested(doc)

        return doc

    @staticmethod
    def render_implicit(rel: MathRelation) -> Any:
        """
        以隐式（不包含函数名）的方式渲染 MathRelation。
        主要用于生成 In-Context Learning 的 demonstrations。
        """
        implicit_rel = MathRelation(
            func_name=rel.func_name,
            func_synonyms=rel.func_synonyms,
            category=rel.category,
            variables=rel.variables,
            var_synonyms=rel.var_synonyms,
            include_func_name=False,
        )
        return TemplateEngine.render(implicit_rel)

    @staticmethod
    def template_count() -> int:
        return len(_TEMPLATES)
