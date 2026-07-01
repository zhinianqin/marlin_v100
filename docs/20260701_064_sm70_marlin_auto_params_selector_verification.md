# SM70 Marlin Auto Params Selector 命中率验证

## 概述

`benchmarks/verify_selector_hit_rate.py` 验证生成的 C++ selector 对 benchmark CSV 的 best 命中率。

## 验证流程

### 1. 加载 selector 规则

从 `best_params_computed.json` 解析每个模型的 selector 结构，与 `generate_selectors.py` 逻辑完全一致:
- 计算 effective_gs (= -1 when gs == size_k)
- 分析共同条件 (哪些字段在所有条目间统一)
- 构建 dispatch entry 列表

```python
rule = {
    "kind": "dense" | "moe",
    "guard": {"qf": str|None, "has_bias": str|None, "eff_gs": int|None},
    "entries": [
        {
            "qf": str, "has_bias": str, "eff_gs": int,
            "size_n": int, "size_k": int,
            "branches": [('exact', m, params) | ('range_le', m, params) | ('else', None, params)],
            # MoE only:
            "moe_block_size": int, "top_k": int,
        },
        ...
    ],
}
```

### 2. 按严格 key 分组 CSV

对每个 benchmark CSV，过滤 `status == "OK"` 的行，按严格 key + size_m 分组:

**Dense**: `(quant_format, has_bias, eff_gs, size_n, size_k, size_m)`
**MoE**: `(quant_format, has_bias, eff_gs, top_k, size_k, size_n, moe_block_size, size_m)`

每组内取 `marlin_us` 最小的 env 组合作为 **actual best**。

### 3. 模拟 selector 执行

`simulate_dispatch()` 完全镜像 C++ selector 的 if/else 分发逻辑:

1. 先检查顶层 guard (仅 common 条件)
2. 遍历 dispatch entry 列表，对每个 entry 检查非 common 条件
3. 命中 entry 后，按 size_m 分支匹配 (exact / range_le / else)
4. 返回预测的 `(geometry, split_k, metadata)`

### 4. 比较

| 结果 | 条件 |
|------|------|
| **hit** | geometry、split_k、metadata 三者全同 |
| **miss** | 任一不同 |
| **unmatched** | selector 不存在，或 dispatch 未命中 |

## 严格 key 对比

| 维度 | Dense | MoE |
|------|-------|-----|
| quant_format | ✓ | ✓ |
| has_bias | ✓ | ✓ |
| group_size | ✓ (effective) | ✓ (effective) |
| moe_block_size | — | ✓ |
| top_k | — | ✓ |
| size_n | ✓ | ✓ |
| size_k | ✓ | ✓ |
| size_m | ✓ (最末级) | ✓ (最末级) |

## effective group_size 规则

当 `group_size == size_k` 时，`eff_gs = -1`。验证脚本和 selector 都使用 effective_gs 进行匹配。

```python
eff_gs = -1 if group_size == size_k else group_size
```

这意味着: 两个不同的 CSV 条目如果只是原始 `group_size` 不同但 `size_k` 也不同从而使 `eff_gs` 相同，它们会匹配到同一个 selector entry。

例: `gs=128, k=128` → eff_gs=-1；`gs=3072, k=3072` → eff_gs=-1。两者在 selector 中均通过 `ctx.group_size == -1` 匹配。

## 输出

```
============================================================
OVERALL RESULTS
============================================================
Total hits:        1400
Total misses:      0
Total no_selector: 0
Overall hit rate:  100.00% (1400/1400)

============================================================
PER-MODEL HIT RATES
============================================================
  QuantTrio/GLM-4.7-AWQ: 100.0% (59/59)
  QuantTrio/MiniMax-M2.7-AWQ: 100.0% (29/29)
  ...
```

如有 miss，会输出明细:

```
============================================================
MISS DETAILS (showing up to 30)
============================================================
  <model> <kind> has_bias=... eff_gs=... m=... n=... k=... [bs=... top_k=...]
    Best:      geo=... sk=... meta=... us=...
    Predicted: geo=... sk=... meta=...
```

## 用法

```bash
.venv/bin/python benchmarks/verify_selector_hit_rate.py \
    --csv-dir benchmarks/results \
    --best-json benchmarks/results/best_params_computed.json
```

## 与 generator 的一致性

验证脚本的 `load_selector_rules` 和 `simulate_dispatch` 必须与 `generate_selectors.py` 的生成逻辑保持一致。验证过程中发现的任何 miss 都意味着生成器和验证器之间存在偏差，或者是 benchmark 数据本身的问题。
