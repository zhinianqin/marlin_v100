#!/usr/bin/env python3
r"""
验证生成的 SM70 Marlin C++ selector 对 benchmark CSV 的 best 命中率。

================================================================================
验证流程
================================================================================

1. 加载 selector 规则 (load_selector_rules)
   - 从 best_params_computed.json 解析每个模型的 selector 结构
   - 计算 effective_gs (= -1 when gs == size_k)
   - 分析共同条件 (哪些字段在所有条目间统一)
   - 构建 dispatch entry 列表 (与 generate_selectors.py 逻辑完全一致)

2. 读取 benchmark CSV，按严格 key 分组
   - Dense:  (qf, has_bias, eff_gs, size_n, size_k, size_m)
   - MoE:    (qf, has_bias, eff_gs, top_k, size_k, size_n, moe_block_size, size_m)
   - 每组内取 marlin_us 最小的作为 "actual best"

3. 模拟 selector 执行 (simulate_dispatch)
   - 先检查顶层 guard (仅共同条件)
   - 再在 dispatch entry 列表中查找匹配 (包括非共同条件)
   - 命中 size_m 分支，返回预测的 (geometry, split_k, metadata)

4. 比较预测与实际 best
   - geometry, split_k, metadata 三者全同 -> hit
   - 任一不同 -> miss
   - selector 不存在或 dispatch 未命中 -> unmatched

================================================================================
用法
================================================================================

    .venv/bin/python benchmarks/verify_selector_hit_rate.py \
        --csv-dir benchmarks/results \
        --best-json benchmarks/results/best_params_computed.json

输出:
  - 每个模型的命中率 (命中数/可比分组数)
  - 总体命中率
  - 未命中明细 (前 30 条)
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def model_path_to_hf_id(model_path: str) -> str:
    """从本地 snapshot 路径提取 HuggingFace 模型 ID。

    /mnt/.../models--bjk110--Qwen3.5-.../snapshots/hash -> bjk110/Qwen3.5-...
    """
    parts = model_path.split("/")
    for p in parts:
        if p.startswith("models--"):
            return p[len("models--"):].replace("--", "/", 1)
    return model_path


def hf_id_to_selector_suffix(hf_id: str) -> str:
    """将 HF 模型 ID 转为 selector 后缀 (与 generate_selectors.py 一致)。"""
    s = hf_id.lower()
    s = s.replace("/", "_")
    s = s.replace(".", "_")
    s = s.replace("-", "_")
    return s


def effective_gs(gs: int, size_k: int) -> int:
    """Effective group_size: -1 when gs == size_k (per-channel 量化)。"""
    return -1 if gs == size_k else gs


# ============================================================
# 加载 selector 规则（复用 generate_selectors 的分析逻辑）
# ============================================================

def load_selector_rules(best_json_path: str) -> dict:
    """从 best_params_computed.json 加载 selector 规则。

    Returns: {suffix: {"kind": "dense"|"moe",
                        "guard": {"qf": str|None, "has_bias": str|None, "eff_gs": int|None},
                        "entries": [dispatch_entry, ...]}}
    """
    with open(best_json_path) as f:
        data = json.load(f)

    rules = {}

    for hf_id in sorted(data.keys()):
        info = data[hf_id]
        suffix = hf_id_to_selector_suffix(hf_id)

        for kind in ("dense", "moe"):
            if not info.get(f"has_{kind}") or not info.get(kind):
                continue

            # Parse entries
            entries = []
            for label, best_per_m in sorted(info[kind].items()):
                parts = {}
                for part in label.split(", "):
                    k, v = part.split("=", 1)
                    parts[k.strip()] = v.strip()

                has_bias = parts["has_bias"]
                qf = parts["qf"]
                gs = int(parts["gs"])
                n = int(parts["n"])
                k_val = int(parts["k"])
                eff_gs_val = effective_gs(gs, k_val)

                # size_m branches
                from generate_selectors import simplify_size_m_branches
                branches = simplify_size_m_branches(best_per_m)

                entry = {
                    "has_bias": has_bias,
                    "qf": qf,
                    "eff_gs": eff_gs_val,
                    "size_n": n,
                    "size_k": k_val,
                    "branches": branches,
                }

                if kind == "moe":
                    entry["moe_block_size"] = int(parts["bs"])
                    entry["top_k"] = int(parts["top_k"])

                entries.append(entry)

            # Find common conditions
            all_has_bias = set(e["has_bias"] for e in entries)
            all_qf = set(e["qf"] for e in entries)
            all_eff_gs = set(e["eff_gs"] for e in entries)

            guard = {
                "qf": all_qf.pop() if len(all_qf) == 1 else None,
                "has_bias": all_has_bias.pop() if len(all_has_bias) == 1 else None,
                "eff_gs": all_eff_gs.pop() if len(all_eff_gs) == 1 else None,
            }

            sort_key = (
                "has_bias" if guard["has_bias"] is None else "",
                "eff_gs" if guard["eff_gs"] is None else "",
            )
            if kind == "dense":
                entries.sort(key=lambda e: (e["has_bias"], e["eff_gs"], e["size_n"], e["size_k"]))
            else:
                entries.sort(key=lambda e: (e["has_bias"], e["eff_gs"], e["top_k"], e["size_k"], e["size_n"], e["moe_block_size"]))

            key = f"{suffix}_{kind}"
            rules[key] = {
                "suffix": suffix,
                "kind": kind,
                "guard": guard,
                "entries": entries,
            }

    return rules


# ============================================================
# Selector 模拟
# ============================================================

def simulate_dispatch(rule: dict, size_m: int, size_n: int, size_k: int,
                      has_bias: str, group_size: int, qf: str = None,
                      moe_block_size: int = None, top_k: int = None) -> dict:
    """模拟 C++ selector 的分发逻辑，返回预测的 params 或 None。

    此函数完全镜像 generate_selectors.py 生成的 C++ selector 的
    if/else 分发逻辑:

    1. 先检查顶层 guard (仅 common 条件)
    2. 遍历 dispatch entry 列表
       - 对每个 entry，检查非 common 条件 (qf?, has_bias?, eff_gs?, size_n, size_k)
       - 对 MoE 额外检查 top_k, moe_block_size
    3. 命中 entry 后，按 size_m 分支匹配
       - 'exact': ctx.size_m == val
       - 'range_le': ctx.size_m <= val
       - 'else': 兜底

    参数:
      rule: load_selector_rules 返回的一条规则
      size_m, size_n, size_k: 形状参数
      has_bias: "true" / "false" (CSV 中的字符串)
      group_size: effective group_size (-1 when gs == size_k)
      qf: quant_format 字符串 (如 "fp8_e4m3", "nvfp4")
      moe_block_size, top_k: MoE 特有参数 (仅 moe kind)

    返回:
      {"geometry": <str>, "split_k": <int>, "metadata": <str>} 或 None
    """
    guard = rule["guard"]

    # Check guard
    if guard["qf"] is not None:
        if qf and qf != guard["qf"]:
            return None
    if guard["has_bias"] is not None:
        if has_bias.lower() != guard["has_bias"]:
            return None
    if guard["eff_gs"] is not None:
        if group_size != guard["eff_gs"]:
            return None

    # Search entries
    for entry in rule["entries"]:
        match = True
        if guard["qf"] is None:
            if qf and qf != entry["qf"]:
                match = False
        if guard["has_bias"] is None:
            if has_bias.lower() != entry["has_bias"]:
                match = False
        if guard["eff_gs"] is None:
            if group_size != entry["eff_gs"]:
                match = False
        if size_n != entry["size_n"] or size_k != entry["size_k"]:
            match = False
        if rule["kind"] == "moe":
            if moe_block_size != entry["moe_block_size"] or top_k != entry["top_k"]:
                match = False

        if match:
            # Found matching entry — apply size_m branches
            for btype, bval, bp in entry["branches"]:
                if btype == "exact" and size_m == bval:
                    return {"geometry": bp["geometry"], "split_k": bp["split_k"], "metadata": bp["metadata"]}
                elif btype == "range_le" and size_m <= bval:
                    return {"geometry": bp["geometry"], "split_k": bp["split_k"], "metadata": bp["metadata"]}
                elif btype == "else":
                    return {"geometry": bp["geometry"], "split_k": bp["split_k"], "metadata": bp["metadata"]}
            # Fallback to last branch
            if entry["branches"]:
                bp = entry["branches"][-1][2]
                return {"geometry": bp["geometry"], "split_k": bp["split_k"], "metadata": bp["metadata"]}

    return None


# ============================================================
# 主验证
# ============================================================

# ============================================================
# 主验证逻辑
# ============================================================

def verify_hit_rate(csv_dir: str, best_json_path: str):
    """主验证函数：读取 CSV + selector 规则，计算命中率。

    验证流程 (对每个模型):
    1. 读取 benchmark CSV，按严格 key 分组
       - Dense:  (qf, has_bias, eff_gs, size_n, size_k, size_m)
       - MoE:    (qf, has_bias, eff_gs, top_k, size_k, size_n, moe_block_size, size_m)
    2. 每组找 actual best (最小 marlin_us)
    3. 用 simulate_dispatch 模拟 selector 选择
    4. 比较 predicted vs actual best

    汇报:
      - 每模型命中率
      - 总体命中率 (hits / (hits + misses))
      - unmatched 计数 (selector 不存在或 dispatch 未命中)
      - 未命中明细
    """
    rules = load_selector_rules(best_json_path)
    print(f"Loaded {len(rules)} selector rules")
    for key, rule in sorted(rules.items()):
        g = rule["guard"]
        print(f"  {rule['suffix']} ({rule['kind']}): guard=(qf={g['qf']}, has_bias={g['has_bias']}, eff_gs={g['eff_gs']}), entries={len(rule['entries'])}")

    # Map CSV files to models
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    model_to_csv = {}
    for csv_path in csv_files:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            try:
                first_row = next(reader)
            except StopIteration:
                continue
        model_path = first_row.get("model", "")
        if not model_path:
            continue
        hf_id = model_path_to_hf_id(model_path)
        if hf_id not in model_to_csv or str(csv_path) > model_to_csv[hf_id]["csv"]:
            model_to_csv[hf_id] = {"csv": str(csv_path), "model_path": model_path}

    print(f"\nFound {len(model_to_csv)} models with CSVs\n")

    total_hits = 0
    total_misses = 0
    total_no_sel = 0
    model_results = {}
    all_misses = []

    for hf_id in sorted(model_to_csv.keys()):
        info = model_to_csv[hf_id]
        csv_path = info["csv"]
        suffix = hf_id_to_selector_suffix(hf_id)

        # Find matching selectors
        dense_sel = rules.get(f"{suffix}_dense")
        moe_sel = rules.get(f"{suffix}_moe")

        print(f"Model: {hf_id}")
        print(f"  Dense selector: {'YES' if dense_sel else 'NO'}, MoE selector: {'YES' if moe_sel else 'NO'}")

        # Read CSV, group by strict key + size_m
        dense_groups = defaultdict(list)
        moe_groups = defaultdict(list)

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["status"].strip() != "OK":
                    continue

                kind = row["kind"].strip()
                has_bias_str = row["has_bias"].strip()
                qf_str = row["quant_format"].strip()
                gs_val = int(row["group_size"])
                n_val = int(row["size_n"])
                k_val = int(row["size_k"])
                m_val = int(row["size_m"])
                eff_gs_val = effective_gs(gs_val, k_val)

                best_params = {
                    "geometry": row["env_cta_geometry"].strip(),
                    "split_k": int(row["env_split_k"]),
                    "metadata": row["env_metadata_cache"].strip(),
                    "marlin_us": float(row["marlin_us"]),
                }

                if kind == "dense":
                    key = (qf_str, has_bias_str, eff_gs_val, n_val, k_val, m_val)
                    dense_groups[key].append(best_params)
                elif kind == "moe":
                    bs_val = int(row["moe_block_size"])
                    tk_val = int(row["top_k"])
                    key = (qf_str, has_bias_str, eff_gs_val, tk_val, k_val, n_val, bs_val, m_val)
                    moe_groups[key].append(best_params)

        model_hits = 0
        model_misses = 0
        model_no_sel = 0

        # Verify Dense
        for key, params_list in sorted(dense_groups.items()):
            qf_str, has_bias_str, eff_gs_val, n_val, k_val, m_val = key
            best = min(params_list, key=lambda p: p["marlin_us"])

            if dense_sel is None:
                model_no_sel += 1
                continue

            predicted = simulate_dispatch(dense_sel, m_val, n_val, k_val,
                                          has_bias_str, eff_gs_val, qf=qf_str)
            if predicted is None:
                model_no_sel += 1
            elif (predicted["geometry"] == best["geometry"] and
                  predicted["split_k"] == best["split_k"] and
                  predicted["metadata"] == best["metadata"]):
                model_hits += 1
            else:
                model_misses += 1
                all_misses.append({
                    "model": hf_id, "kind": "dense",
                    "size_m": m_val, "size_n": n_val, "size_k": k_val,
                    "has_bias": has_bias_str, "eff_gs": eff_gs_val,
                    "best": best, "predicted": predicted,
                })

        # Verify MoE
        for key, params_list in sorted(moe_groups.items()):
            qf_str, has_bias_str, eff_gs_val, tk_val, k_val, n_val, bs_val, m_val = key
            best = min(params_list, key=lambda p: p["marlin_us"])

            if moe_sel is None:
                model_no_sel += 1
                continue

            predicted = simulate_dispatch(moe_sel, m_val, n_val, k_val,
                                          has_bias_str, eff_gs_val, qf=qf_str,
                                          moe_block_size=bs_val, top_k=tk_val)
            if predicted is None:
                model_no_sel += 1
            elif (predicted["geometry"] == best["geometry"] and
                  predicted["split_k"] == best["split_k"] and
                  predicted["metadata"] == best["metadata"]):
                model_hits += 1
            else:
                model_misses += 1
                all_misses.append({
                    "model": hf_id, "kind": "moe",
                    "size_m": m_val, "size_n": n_val, "size_k": k_val,
                    "has_bias": has_bias_str, "eff_gs": eff_gs_val,
                    "moe_block_size": bs_val, "top_k": tk_val,
                    "best": best, "predicted": predicted,
                })

        total = model_hits + model_misses
        hit_rate = (model_hits / total * 100) if total > 0 else 0
        no_sel_str = f" (+{model_no_sel} unmatched)" if model_no_sel > 0 else ""
        print(f"  Hits: {model_hits}, Misses: {model_misses}, No sel: {model_no_sel}")
        print(f"  Hit rate: {hit_rate:.2f}%{no_sel_str}")

        model_results[hf_id] = {
            "hits": model_hits, "misses": model_misses,
            "no_selector": model_no_sel, "hit_rate": hit_rate,
        }
        total_hits += model_hits
        total_misses += model_misses
        total_no_sel += model_no_sel

    # ============================================================
    # 总体汇报
    # ============================================================
    print(f"\n{'='*60}")
    print(f"OVERALL RESULTS")
    print(f"{'='*60}")
    total_comparable = total_hits + total_misses
    overall_hit_rate = (total_hits / total_comparable * 100) if total_comparable > 0 else 0
    print(f"Total hits:        {total_hits}")
    print(f"Total misses:      {total_misses}")
    print(f"Total no_selector: {total_no_sel}")
    print(f"Overall hit rate:  {overall_hit_rate:.2f}% ({total_hits}/{total_comparable})")

    print(f"\n{'='*60}")
    print(f"PER-MODEL HIT RATES")
    print(f"{'='*60}")
    for hf_id, mr in sorted(model_results.items()):
        total = mr["hits"] + mr["misses"]
        if total > 0:
            extra = f" + {mr['no_selector']} unmatched" if mr["no_selector"] > 0 else ""
            print(f"  {hf_id}: {mr['hit_rate']:.1f}% ({mr['hits']}/{total}){extra}")
        else:
            print(f"  {hf_id}: N/A (no comparable groups)")

    if all_misses:
        print(f"\n{'='*60}")
        print(f"MISS DETAILS (showing up to 30)")
        print(f"{'='*60}")
        for m in all_misses[:30]:
            extra = ""
            if m["kind"] == "moe":
                extra = f" bs={m.get('moe_block_size','?')} top_k={m.get('top_k','?')}"
            print(f"  {m['model']} {m['kind']} has_bias={m['has_bias']} eff_gs={m['eff_gs']} "
                  f"m={m['size_m']} n={m['size_n']} k={m['size_k']}{extra}")
            print(f"    Best:      geo={m['best']['geometry']} sk={m['best']['split_k']} "
                  f"meta={m['best']['metadata']} us={m['best']['marlin_us']}")
            print(f"    Predicted: geo={m['predicted']['geometry']} sk={m['predicted']['split_k']} "
                  f"meta={m['predicted']['metadata']}")
        if len(all_misses) > 30:
            print(f"  ... and {len(all_misses) - 30} more misses")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default="benchmarks/results")
    parser.add_argument("--best-json", default="benchmarks/results/best_params_computed.json")
    args = parser.parse_args()
    verify_hit_rate(args.csv_dir, args.best_json)


if __name__ == "__main__":
    main()
