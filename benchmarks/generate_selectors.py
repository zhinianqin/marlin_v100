#!/usr/bin/env python3
r"""
从 best_params_computed.json 生成 SM70 Marlin C++ auto-params selector 函数。

================================================================================
设计原则
================================================================================

1. 每个模型最多 2 个 selector (Dense + MoE)，函数名纯 model_id，无后缀。

   - 函数名: sm70_marlin_{dense,moe}_try_select_<model_id>_params
   - <model_id> 仅从 HuggingFace 模型 ID 派生，例:
     bjk110/Qwen3.5-122B-A10B-abliterated-AWQ
     -> bjk110_qwen3_5_122b_a10b_abliterated_awq

2. 顶层 guard 仅包含所有条目共同拥有的条件 (共同条件分析)。

   同一模型的各层可能有不同的 has_bias、group_size、甚至 quant_format。
   生成器自动分析哪些字段在所有条目间是统一的，只将统一的字段放入顶层 guard。
   不统一的字段下沉到每个 (size_n, size_k) 分发条件中。

3. effective group_size 规则: gs == size_k -> -1

   当 group_size 等于 size_k (per-channel 量化，整个 K 维度只有一个 group)，
   上游 marlin.cu 运行时会将其转换为 -1。因此 selector 中必须检查
   ctx.group_size == -1 而非 ctx.group_size == <原始值>。

4. 四种 selector 模式 (按共同条件分类):

   - 模式 A (全部 uniform): qf, has_bias, eff_gs 三者统一。
     顶层 guard 检查三者全匹配。分发仅按 (size_n, size_k) -> size_m。
     覆盖约 17 个模型。

   - 模式 B (eff_gs 变化): qf 和 has_bias 统一，eff_gs 不统一。
     顶层 guard 检查 (qf, has_bias)。分发条件加入 ctx.group_size == -1 或 == <gs>。
     覆盖约 6 个模型 (FP8/AWQ 模型中有 gs==k 的 MoE w2 层)。

   - 模式 C (has_bias 变化): qf 和 eff_gs 统一，has_bias 不统一。
     顶层 guard 检查 (qf, eff_gs)。分发条件加入 ctx.has_bias == true/false。
     覆盖 2 个 GLM-4.7 模型 (dense 层有 shared_expert 带 bias)。

   - 模式 D (仅 has_bias 统一): qf 和 eff_gs 都不统一。
     顶层 guard 仅检查 has_bias。分发条件加入 qf 和 group_size。
     仅 nvidia/Qwen3.6-35B-A3B-NVFP4 Dense (fp8_e4m3 和 nvfp4 混合)。

5. size_m 外推泛化。

   对已知的 size_m (来自 benchmark CSV) 使用精确 == 或 <= 分支。
   最后一个分支始终是 else，承担对未知 size_m 的外推。

6. 硬编码数值，不抽象。

   生成的 C++ selector 直接在 if/else 中硬编码具体数值，
   无间接查表或计算。

================================================================================
输入/输出
================================================================================

输入: benchmarks/results/best_params_computed.json
  - 由 compute_best_params.py 从 benchmark CSV 中提取
  - 结构: {model_id: {has_dense, has_moe, dense: {label: {size_m: params}}, moe: {...}}}

输出: benchmarks/results/generated_selectors/
  - dense/<model_id>.cuh         每个有 Dense 层的模型一个 selector 函数
  - moe/<model_id>.cuh           每个有 MoE 层的模型一个 selector 函数
  - dispatch_dense.txt           Dense dispatch chain (用于插入 sm70_marlin_common.cuh)
  - dispatch_moe.txt             MoE dispatch chain (用于插入 sm70_marlin_gemm.cuh)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def hf_id_to_selector_suffix(hf_id: str) -> str:
    """将 HuggingFace 模型 ID 转为 C++ selector 函数名后缀。

    org/Model.Name-Variant -> org_model_name_variant

    例: bjk110/Qwen3.5-122B-A10B-abliterated-AWQ
      -> bjk110_qwen3_5_122b_a10b_abliterated_awq
    """
    s = hf_id.lower()
    s = s.replace("/", "_")
    s = s.replace(".", "_")
    s = s.replace("-", "_")
    return s


def parse_geometry(geo_str: str) -> tuple:
    """将 CTA geometry 字符串解析为 7 元组。

    "32x128x32x4x32x32x32" -> (32, 128, 32, 4, 32, 32, 32)
    对应: (CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK)
    """
    return tuple(int(x) for x in geo_str.split("x"))


def metadata_to_bool(meta_str: str) -> str:
    """将 metadata_cache 字符串转为 C++ bool 字面量。

    "vector_words" -> "true" (use_metadata_vector_words = true)
    "lane_vectors" -> "false"
    """
    return "true" if meta_str == "vector_words" else "false"


def effective_gs(gs: int, size_k: int) -> int:
    """计算运行时 effective group_size。

    当 group_size == size_k 时，上游 marlin.cu 将其转换为 -1
    (per-channel 量化，整个 K 维度只有一个 group)。
    因此 selector 中对应条件应为 ctx.group_size == -1。
    """
    return -1 if gs == size_k else gs


def simplify_size_m_branches(size_m_params: dict) -> list:
    """将 {size_m: best_params} 简化为分支规格列表。

    合并相邻的共享同一 best params 的 size_m 值。
    输出分支规格:
      ('exact', size_m, params)  — ctx.size_m == size_m
      ('range_le', max_m, params) — ctx.size_m <= max_m (多个连续 size_m 共享)
      ('else', None, params)      — 兜底分支 (最后一个)

    例: {1: {geo:A}, 32: {geo:A}, 64: {geo:B}, 2048: {geo:C}}
      -> [('range_le', 32, {geo:A}),   # m=1,32 共享同一 best
          ('exact', 64, {geo:B}),
          ('else', None, {geo:C})]      # m=2048 及任何未知值
    """
    items = [(int(m), p) for m, p in size_m_params.items()]
    items.sort(key=lambda x: x[0])
    if not items:
        return []

    def params_key(p):
        return (p["geometry"], p["split_k"], p["metadata"])

    groups = []
    current_group = [items[0]]
    current_key = params_key(items[0][1])
    for m, p in items[1:]:
        pk = params_key(p)
        if pk == current_key:
            current_group.append((m, p))
        else:
            groups.append(current_group)
            current_group = [(m, p)]
            current_key = pk
    groups.append(current_group)

    branches = []
    for i, group in enumerate(groups):
        is_last = (i == len(groups) - 1)
        p = group[0][1]
        if is_last:
            branches.append(("else", None, p))
        elif len(group) == 1:
            branches.append(("exact", group[0][0], p))
        else:
            branches.append(("range_le", group[-1][0], p))
    return branches


# ============================================================
# Dense selector 生成
# ============================================================

def generate_dense_selector(suffix: str, dense_data: dict) -> str:
    """为一个模型生成一个 Dense selector C++ 函数。

    参数:
      suffix: 函数名后缀 (例: "quanttrio_qwen3_6_27b_awq")
      dense_data: {label: {size_m: best_params}} 从 best_params_computed.json

    返回:
      完整的 C++ 函数定义字符串，结构如下:

        inline bool sm70_marlin_dense_try_select_<suffix>_params(
            Sm70MarlinDenseAutoParamsContext const& ctx,
            Sm70MarlinAutoParams& params) {
          // 顶层 guard — 仅检查所有条目共同的条件
          if (ctx.quant_format == nullptr || <common_checks> || ctx.size_m <= 0)
            return false;

          auto const set_params = [&](...) { ... };

          // 分发 — (qf?, has_bias?, eff_gs?, size_n, size_k) -> size_m
          if (<condition_1>) {
            if (ctx.size_m == <M>) { return set_params(...); }
            else if ...
            else { return set_params(...); }
          } else if (<condition_2>) { ... }
          return false;
        }

    分发条件构建逻辑:
      - qf:        仅在 common_qf is None 时加入 (模式 D)
      - has_bias:  仅在 common_has_bias is None 时加入 (模式 C)
      - group_size: 仅在 common_eff_gs is None 时加入 (模式 B)
      - size_n, size_k: 始终加入 (严格命中)
    """
    # Step 1: collect all entries with effective_gs
    entries = []  # list of (has_bias, qf, eff_gs, size_n, size_k, size_m_branches)
    for label, best_per_m in sorted(dense_data.items()):
        parts = {}
        for part in label.split(", "):
            k, v = part.split("=", 1)
            parts[k.strip()] = v.strip()

        has_bias = parts["has_bias"]
        qf = parts["qf"]
        gs = int(parts["gs"])
        n = int(parts["n"])
        k_val = int(parts["k"])
        eff_gs = effective_gs(gs, k_val)

        branches = simplify_size_m_branches(best_per_m)
        entries.append((has_bias, qf, eff_gs, n, k_val, branches))

    if not entries:
        return ""

    # Step 2: find common conditions
    all_has_bias = set(e[0] for e in entries)
    all_qf = set(e[1] for e in entries)
    all_eff_gs = set(e[2] for e in entries)

    common_has_bias = all_has_bias.pop() if len(all_has_bias) == 1 else None
    common_qf = all_qf.pop() if len(all_qf) == 1 else None
    common_eff_gs = all_eff_gs.pop() if len(all_eff_gs) == 1 else None

    # Step 3: generate guard (only common conditions)
    guard_conditions = []
    if common_qf is not None:
        guard_conditions.append(
            f'      std::strcmp(ctx.quant_format, "{common_qf}") != 0 ||'
        )
    if common_has_bias is not None:
        guard_conditions.append(
            f"      ctx.has_bias != {common_has_bias.lower()} ||"
        )
    if common_eff_gs is not None:
        guard_conditions.append(
            f"      ctx.group_size != {common_eff_gs} ||"
        )

    lines = []
    lines.append(f"inline bool")
    lines.append(f"sm70_marlin_dense_try_select_{suffix}_params(")
    lines.append(f"    Sm70MarlinDenseAutoParamsContext const& ctx,")
    lines.append(f"    Sm70MarlinAutoParams& params) {{")
    lines.append(f"  if (ctx.quant_format == nullptr ||")
    for cond in guard_conditions:
        lines.append(cond)
    lines.append(f"      ctx.size_m <= 0) {{")
    lines.append(f"    return false;")
    lines.append(f"  }}")
    lines.append(f"")
    lines.append(f"  auto const set_params = [&](Sm70CtaGeometry geometry,")
    lines.append(f"                              int requested_split_k,")
    lines.append(f"                              bool use_metadata_vector_words) {{")
    lines.append(f"    params = {{geometry, requested_split_k, use_metadata_vector_words,")
    lines.append(f"              ctx.packed_macro_n}};")
    lines.append(f"    return true;")
    lines.append(f"  }};")
    lines.append(f"")

    # Step 4: sort entries and generate dispatch
    # Sort by (has_bias, eff_gs, size_n, size_k)
    entries.sort(key=lambda e: (e[0], e[2], e[3], e[4]))

    for i, (has_bias, qf, eff_gs, n, k_val, branches) in enumerate(entries):
        # Build condition
        cond_parts = []
        if common_qf is None:
            cond_parts.append(f'std::strcmp(ctx.quant_format, "{qf}") == 0')
        if common_has_bias is None:
            cond_parts.append(f"ctx.has_bias == {has_bias.lower()}")
        if common_eff_gs is None:
            cond_parts.append(f"ctx.group_size == {eff_gs}")
        cond_parts.append(f"ctx.size_n == {n}")
        cond_parts.append(f"ctx.size_k == {k_val}")
        cond = "\n          && ".join(cond_parts)

        prefix = "  if" if i == 0 else "  } else if"
        lines.append(f"{prefix} ({cond}) {{")

        # size_m branches
        for j, branch in enumerate(branches):
            btype, bval, bp = branch
            geo = parse_geometry(bp["geometry"])
            sk = bp["split_k"]
            meta = metadata_to_bool(bp["metadata"])

            if btype == "exact":
                inner = "if" if j == 0 else "} else if"
                lines.append(f"    {inner} (ctx.size_m == {bval}) {{")
            elif btype == "range_le":
                inner = "if" if j == 0 else "} else if"
                lines.append(f"    {inner} (ctx.size_m <= {bval}) {{")
            elif btype == "else":
                if j > 0:
                    lines.append(f"    }} else {{")
            lines.append(f"      return set_params({{{geo[0]}, {geo[1]}, {geo[2]}, {geo[3]}, {geo[4]}, {geo[5]}, {geo[6]}}}, {sk}, {meta});")

        if len(branches) > 1:
            lines.append(f"    }}")

    if entries:
        lines.append(f"  }}")
    lines.append(f"")
    lines.append(f"  return false;")
    lines.append(f"}}")
    return "\n".join(lines)


# ============================================================
# MoE selector 生成
# ============================================================

def generate_moe_selector(suffix: str, moe_data: dict) -> str:
    """为一个模型生成一个 MoE selector C++ 函数。

    参数:
      suffix: 函数名后缀
      moe_data: {label: {size_m: best_params}} 从 best_params_computed.json

    返回:
      完整的 C++ 函数定义字符串。

    MoE 分发树比 Dense 多一层:
      (qf?, has_bias?, eff_gs?, top_k, size_k, size_n) -> moe_block_size -> size_m

    MoE 的 CTA_M 约束 (仅 32/64，禁用 128/256) 已内嵌在 benchmark best params 中。
    """
    # Step 1: collect all entries
    # entries: list of (has_bias, qf, eff_gs, top_k, size_k, size_n, moe_block_size, branches)
    entries = []
    for label, best_per_m in sorted(moe_data.items()):
        parts = {}
        for part in label.split(", "):
            k, v = part.split("=", 1)
            parts[k.strip()] = v.strip()

        has_bias = parts["has_bias"]
        qf = parts["qf"]
        gs = int(parts["gs"])
        bs = int(parts["bs"])
        tk = int(parts["top_k"])
        n = int(parts["n"])
        k_val = int(parts["k"])
        eff_gs = effective_gs(gs, k_val)

        branches = simplify_size_m_branches(best_per_m)
        entries.append((has_bias, qf, eff_gs, tk, k_val, n, bs, branches))

    if not entries:
        return ""

    # Step 2: find common conditions
    all_has_bias = set(e[0] for e in entries)
    all_qf = set(e[1] for e in entries)
    all_eff_gs = set(e[2] for e in entries)

    common_has_bias = all_has_bias.pop() if len(all_has_bias) == 1 else None
    common_qf = all_qf.pop() if len(all_qf) == 1 else None
    common_eff_gs = all_eff_gs.pop() if len(all_eff_gs) == 1 else None

    # Step 3: guard
    guard_conditions = []
    if common_qf is not None:
        guard_conditions.append(
            f'      std::strcmp(ctx.quant_format, "{common_qf}") != 0 ||'
        )
    if common_has_bias is not None:
        guard_conditions.append(
            f"      ctx.has_bias != {common_has_bias.lower()} ||"
        )
    if common_eff_gs is not None:
        guard_conditions.append(
            f"      ctx.group_size != {common_eff_gs} ||"
        )

    lines = []
    lines.append(f"inline bool")
    lines.append(f"sm70_marlin_moe_try_select_{suffix}_params(")
    lines.append(f"    Sm70MarlinMoeAutoParamsContext const& ctx,")
    lines.append(f"    Sm70MarlinAutoParams& params) {{")
    lines.append(f"  if (ctx.quant_format == nullptr ||")
    for cond in guard_conditions:
        lines.append(cond)
    lines.append(f"      ctx.size_m <= 0) {{")
    lines.append(f"    return false;")
    lines.append(f"  }}")
    lines.append(f"")
    lines.append(f"  auto const set_params = [&](Sm70CtaGeometry geometry,")
    lines.append(f"                              int requested_split_k,")
    lines.append(f"                              bool use_metadata_vector_words) {{")
    lines.append(f"    params = {{geometry, requested_split_k, use_metadata_vector_words,")
    lines.append(f"              ctx.packed_macro_n}};")
    lines.append(f"    return true;")
    lines.append(f"  }};")
    lines.append(f"")

    # Step 4: group entries by outer dispatch key (has_bias?, eff_gs?, top_k, size_k, size_n)
    # -> {outer_key: {moe_block_size: branches}}
    outer_groups = defaultdict(lambda: defaultdict(list))
    for has_bias, qf, eff_gs, tk, k_val, n, bs, branches in entries:
        outer_key = (has_bias, qf, eff_gs, tk, k_val, n)
        outer_groups[outer_key][bs] = branches

    outer_items = sorted(outer_groups.items(), key=lambda x: x[0])

    for i, (outer_key, bs_map) in enumerate(outer_items):
        has_bias, qf, eff_gs, tk, k_val, n = outer_key
        bs_items = sorted(bs_map.items())

        # Build outer condition
        cond_parts = []
        if common_qf is None:
            cond_parts.append(f'std::strcmp(ctx.quant_format, "{qf}") == 0')
        if common_has_bias is None:
            cond_parts.append(f"ctx.has_bias == {has_bias.lower()}")
        if common_eff_gs is None:
            cond_parts.append(f"ctx.group_size == {eff_gs}")
        cond_parts.append(f"ctx.top_k == {tk}")
        cond_parts.append(f"ctx.size_k == {k_val}")
        cond_parts.append(f"ctx.size_n == {n}")
        cond = "\n          && ".join(cond_parts)

        prefix = "  if" if i == 0 else "  } else if"
        lines.append(f"{prefix} ({cond}) {{")

        # moe_block_size sub-branches
        for bi, (bs, branches) in enumerate(bs_items):
            if len(bs_items) == 1:
                indent = "    "
            else:
                indent = "      "
                bprefix = "if" if bi == 0 else "} else if"
                lines.append(f"    {bprefix} (ctx.moe_block_size == {bs}) {{")

            for j, branch in enumerate(branches):
                btype, bval, bp = branch
                geo = parse_geometry(bp["geometry"])
                sk = bp["split_k"]
                meta = metadata_to_bool(bp["metadata"])

                if btype == "exact":
                    inner = "if" if j == 0 else "} else if"
                    lines.append(f"{indent}{inner} (ctx.size_m == {bval}) {{")
                elif btype == "range_le":
                    inner = "if" if j == 0 else "} else if"
                    lines.append(f"{indent}{inner} (ctx.size_m <= {bval}) {{")
                elif btype == "else":
                    if j > 0:
                        lines.append(f"{indent}}} else {{")

                lines.append(f"{indent}  return set_params({{{geo[0]}, {geo[1]}, {geo[2]}, {geo[3]}, {geo[4]}, {geo[5]}, {geo[6]}}}, {sk}, {meta});")

            if len(branches) > 1:
                lines.append(f"{indent}}}")

        if len(bs_items) > 1:
            lines.append(f"    }}")

    if outer_items:
        lines.append(f"  }}")
    lines.append(f"")
    lines.append(f"  return false;")
    lines.append(f"}}")
    return "\n".join(lines)


# ============================================================
# 主入口
# ============================================================

def generate_all_selectors(best_json_path: str) -> dict:
    """从 best_params_computed.json 生成全部 selector。

    参数:
      best_json_path: best_params_computed.json 路径

    返回:
      {
        "dense_selectors": {suffix: cpp_code, ...},
        "moe_selectors": {suffix: cpp_code, ...},
        "dense_dispatch_order": [suffix, ...],   # 按字母序的 dispatch 顺序
        "moe_dispatch_order": [suffix, ...],
      }
    """
    with open(best_json_path) as f:
        data = json.load(f)

    dense_selectors = {}
    moe_selectors = {}
    dense_dispatch = []
    moe_dispatch = []

    for hf_id in sorted(data.keys()):
        info = data[hf_id]
        suffix = hf_id_to_selector_suffix(hf_id)

        if info["has_dense"] and info["dense"]:
            code = generate_dense_selector(suffix, info["dense"])
            if code:
                dense_selectors[suffix] = code
                dense_dispatch.append(suffix)
                print(f"  Dense: {suffix}")

        if info["has_moe"] and info["moe"]:
            code = generate_moe_selector(suffix, info["moe"])
            if code:
                moe_selectors[suffix] = code
                moe_dispatch.append(suffix)
                print(f"  MoE:   {suffix}")

    return {
        "dense_selectors": dense_selectors,
        "moe_selectors": moe_selectors,
        "dense_dispatch_order": dense_dispatch,
        "moe_dispatch_order": moe_dispatch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-json", default="benchmarks/results/best_params_computed.json")
    parser.add_argument("--output-dir", default="benchmarks/results/generated_selectors")
    args = parser.parse_args()

    print(f"Loading best params from: {args.best_json}")
    result = generate_all_selectors(args.best_json)

    output_dir = Path(args.output_dir)
    # Clean and recreate
    import shutil
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    dense_dir = output_dir / "dense"
    moe_dir = output_dir / "moe"
    dense_dir.mkdir()
    moe_dir.mkdir()

    for suffix, code in result["dense_selectors"].items():
        (dense_dir / f"{suffix}.cuh").write_text(code)
    for suffix, code in result["moe_selectors"].items():
        (moe_dir / f"{suffix}.cuh").write_text(code)

    # Dispatch chains
    dispatch_dense = []
    for suffix in result["dense_dispatch_order"]:
        dispatch_dense.append(
            f"  if (sm70_marlin_dense_try_select_{suffix}_params(ctx, params)) {{\n"
            f'    validate_sm70_marlin_auto_params("Dense", params);\n'
            f"    return params;\n"
            f"  }}"
        )
    (output_dir / "dispatch_dense.txt").write_text("\n".join(dispatch_dense))

    dispatch_moe = []
    for suffix in result["moe_dispatch_order"]:
        dispatch_moe.append(
            f"  if (sm70_marlin_moe_try_select_{suffix}_params(ctx, params)) {{\n"
            f'    validate_sm70_marlin_auto_params("MoE", params);\n'
            f"    return params;\n"
            f"  }}"
        )
    (output_dir / "dispatch_moe.txt").write_text("\n".join(dispatch_moe))

    print(f"\nSummary:")
    print(f"  Dense selectors: {len(result['dense_selectors'])}")
    print(f"  MoE   selectors: {len(result['moe_selectors'])}")
    print(f"  Output: {output_dir}")


if __name__ == "__main__":
    main()
