# SM70 Marlin Auto Params Selector 生成器

## 概述

`benchmarks/generate_selectors.py` 从 benchmark CSV 的 best params JSON 生成 SM70 Marlin C++ auto-params selector 函数。

输入: `benchmarks/results/best_params_computed.json`
输出: `benchmarks/results/generated_selectors/` (C++ selector 函数 + dispatch chain)

## 设计原则

### 1. 函数命名：纯 model_id，无后缀

每个模型最多生成 **2** 个 selector：一个 Dense + 一个 MoE（如果模型有对应层）。

```
sm70_marlin_dense_try_select_<model_id>_params
sm70_marlin_moe_try_select_<model_id>_params
```

`<model_id>` 仅从 HuggingFace 模型 ID 派生，不包含 `_has_bias_false`、`_gs_128`、`_gs_m1` 等后缀。

例:
```
bjk110/Qwen3.5-122B-A10B-abliterated-AWQ
  -> bjk110_qwen3_5_122b_a10b_abliterated_awq
```

### 2. 顶层 guard：共同条件分析

同一模型的各层可能有不同的 `has_bias`、`group_size`、甚至 `quant_format`。生成器自动分析哪些字段在所有条目间统一，**仅将统一的字段放入顶层 guard**，不统一的字段下沉到每个 `(size_n, size_k)` 分发条件中。

### 3. effective group_size 规则

当 `group_size == size_k` (per-channel 量化，整个 K 维度只有一个 group)，上游 `marlin.cu` 运行时将其转换为 `-1`。因此 selector 中必须检查 `ctx.group_size == -1` 而非 `ctx.group_size == <原始值>`。

```python
eff_gs = -1 if group_size == size_k else group_size
```

### 4. 四种 selector 模式

按共同条件分类，共有四种模式：

#### 模式 A：全部 uniform (约 17 个模型)

`qf`、`has_bias`、`eff_gs` 三者统一。顶层 guard 检查三者全匹配。分发仅按 `(size_n, size_k) → size_m`。

```cpp
// qf, has_bias, eff_gs 均在 guard 中
if (qf != "uint4" || gs != 128 || has_bias != false || size_m <= 0) return false;

// 分发条件仅含 size_n, size_k
if (ctx.size_n == 4352 && ctx.size_k == 5120) { ... }
```

#### 模式 B：eff_gs 变化 (约 6 个模型)

`qf` 和 `has_bias` 统一，`eff_gs` 不统一。顶层 guard 检查 `(qf, has_bias)`。分发条件加入 `ctx.group_size == -1` 或 `== <gs>`。

典型场景：FP8/AWQ 模型中有 `gs==k` 的 MoE w2 层 (k=128, gs=128，eff_gs=-1)，同时有其他层 k≠gs (eff_gs=128)。

```cpp
if (qf != "fp8_e4m3" || has_bias != false || size_m <= 0) return false;

if (ctx.group_size == -1 && ctx.size_n == 3072 && ctx.size_k == 128) { ... }
else if (ctx.group_size == 128 && ctx.size_n == 3072 && ctx.size_k == 3072) { ... }
```

#### 模式 C：has_bias 变化 (2 个模型)

`qf` 和 `eff_gs` 统一，`has_bias` 不统一。仅 GLM-4.7 模型的 dense 层有此情况（shared_expert 层带 bias）。

```cpp
if (qf != "uint4" || gs != 128 || size_m <= 0) return false;

if (ctx.has_bias == false && ctx.size_n == 3072 && ctx.size_k == 5120) { ... }
else if (ctx.has_bias == true && ctx.size_n == 1792 && ctx.size_k == 5120) { ... }
```

#### 模式 D：仅 has_bias 统一 (1 个模型)

`qf` 和 `eff_gs` 都不统一。仅 `nvidia/Qwen3.6-35B-A3B-NVFP4` Dense (fp8_e4m3 和 nvfp4 混合)。

```cpp
if (has_bias != false || size_m <= 0) return false;

if (std::strcmp(ctx.quant_format, "fp8_e4m3") == 0 && ctx.group_size == -1 && ...) { ... }
else if (std::strcmp(ctx.quant_format, "nvfp4") == 0 && ctx.group_size == 16 && ...) { ... }
```

### 5. size_m 外推泛化

已知 size_m (来自 benchmark CSV): 使用 `==` 或 `<=` 精确分支。
最后一个分支始终是 `else`，承担对未知 size_m 的外推。

### 6. 硬编码数值

生成的 C++ selector 直接在 if/else 中硬编码具体数值，无间接查表。

## 核心函数

### `effective_gs(gs, size_k) -> int`

```python
def effective_gs(gs, size_k):
    return -1 if gs == size_k else gs
```

### `simplify_size_m_branches(size_m_params) -> list`

将 `{size_m: best_params}` 简化为分支规格列表，合并相邻的共享同一 best params 的 size_m 值。

输出:
- `('exact', size_m, params)` — `ctx.size_m == size_m`
- `('range_le', max_m, params)` — `ctx.size_m <= max_m` (连续 size_m 共享同一 best)
- `('else', None, params)` — 兜底 (最后一个分支)

例:
```
{1: {geo:A}, 32: {geo:A}, 64: {geo:B}, 2048: {geo:C}}
-> [('range_le', 32, {geo:A}), ('exact', 64, {geo:B}), ('else', None, {geo:C})]
```

### `generate_dense_selector(suffix, dense_data) -> str`

为一个模型生成一个 Dense selector C++ 函数。

生成流程:
1. 解析所有 dense label，计算 effective_gs
2. 分析共同条件 (`common_qf`, `common_has_bias`, `common_eff_gs`)
3. 生成顶层 guard (仅 common 条件 + `size_m <= 0`)
4. 按 `(has_bias, eff_gs, size_n, size_k)` 排序，生成 if/else 分发链
5. 每个分发条件下生成 size_m 分支

### `generate_moe_selector(suffix, moe_data) -> str`

为一个模型生成一个 MoE selector C++ 函数。

分发树比 Dense 多一层: `(qf?, has_bias?, eff_gs?, top_k, size_k, size_n) → moe_block_size → size_m`

## 输入格式

`best_params_computed.json` 结构:

```json
{
  "QuantTrio/Qwen3.6-27B-AWQ": {
    "has_dense": true,
    "has_moe": false,
    "dense": {
      "has_bias=false, qf=uint4, gs=128, n=4352, k=5120": {
        "1": {"geometry": "32x128x32x4x32x32x32", "split_k": 8, "metadata": "vector_words", ...},
        "32": {...},
        "64": {...},
        "2048": {...},
        "4096": {...}
      },
      ...
    }
  }
}
```

## 输出格式

```
generated_selectors/
├── dense/
│   ├── model_1.cuh      # 每个 Dense selector 一个文件
│   └── ...
├── moe/
│   ├── model_2.cuh      # 每个 MoE selector 一个文件
│   └── ...
├── dispatch_dense.txt   # Dense dispatch chain (插入 sm70_marlin_common.cuh)
└── dispatch_moe.txt     # MoE dispatch chain (插入 sm70_marlin_gemm.cuh)
```

## 用法

```bash
# 1. 先从 CSV 计算 best params
.venv/bin/python benchmarks/compute_best_params.py

# 2. 生成 C++ selector
.venv/bin/python benchmarks/generate_selectors.py \
    --best-json benchmarks/results/best_params_computed.json \
    --output-dir benchmarks/results/generated_selectors

# 3. 插入 .cuh 文件 (手动或脚本)
# 4. 构建: ./build.sh
```
