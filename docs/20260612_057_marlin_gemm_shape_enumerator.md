# Marlin GEMM Shape Enumerator 交付说明

`marlin_gemm_shapes.py` 是一个面向 Marlin kernel benchmark 设计的静态
shape 枚举工具。它读取本地模型目录中的配置文件和可选的
`model.safetensors.index.json`，反推出 dense Marlin GEMM 和 MoE Marlin GEMM
在不同 TP/EP、prefill/decode 场景下会用到的参数组合。

这个工具的目标不是启动 vLLM engine，也不是测真实性能；它只负责回答一个更靠前的
问题：对某个具体本地 checkpoint，要为 Marlin kernel 准备哪些
`M/N/K` 和 MoE block size benchmark 组合。

## 背景

典型启动命令里会包含模型路径、量化方式、TP、MoE backend、max batched
tokens、decode 并发等信息，例如：

```bash
SAFETENSORS_FAST_GPU=1 \
.venv/bin/python -m vllm.entrypoints.cli.main serve \
  --model /root/models/QuantTrio/Qwen3.5-122B-A10B-AWQ \
  --quantization awq_marlin \
  --moe-backend marlin \
  --dtype float16 \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 4096 \
  --attention-backend FLASH_ATTN \
  --max-num-seqs 1
```

真正做 kernel 优化时，需要把这类 serve 参数进一步转成底层 GEMM 的输入形状：

- dense 路径：`ops.marlin_gemm(size_m, size_n, size_k)`
- MoE 路径：`ops.moe_wna16_marlin_gemm(moe_block_size, top_k, size_m,
  size_n, size_k)`

困难点在于量化状态不是简单的模型级属性。ModelOpt 和 compressed-tensors
checkpoint 可能在同一个模型里混合 FP8、NVFP4、MXFP4、BF16/FP16 未量化模块。
因此这个脚本按候选模块逐个判断：只有当前模块有明确量化配置和 index 证据时，
才输出 `actual_marlin`；BF16/FP16/未量化模块只输出
`hypothetical_bf16`，用于 benchmark shape 参考。

## 开发目标

工具设计目标：

- 只支持本地叶子模型目录，目录内必须包含 `config.json`。
- 只读配置和 safetensors index 元数据，不读取 tensor payload。
- 不构建 engine，不初始化 CUDA，不启动 server。
- 以模块为粒度判断 Marlin 路径，而不是只看模型级 `quant_method`。
- 同时覆盖 dense linear 和 routed MoE expert GEMM。
- 默认枚举 `TP=4/8` 和 `TP+EP=4/8` 四类场景。
- 默认枚举 prefill `M=2048/4096` 和 decode 并发 `M=1/32/64`。
- 输出 pretty table 和 JSON，便于人工检查和后续脚本消费。

不属于本工具目标：

- 不预测 router 真实负载不均。
- 不判断某个 shape 的真实性能好坏。
- 不替代最终 kernel benchmark。
- 不保证所有未来模型结构都自动识别，需要按模型族持续补充映射。

## 运行依赖

运行位置：

```bash
cd /root/source/repos/marlin_v100
```

推荐解释器：

```bash
.venv/bin/python benchmarks/marlin_gemm_shapes.py --help
```

依赖说明：

- 脚本本身只使用 Python 标准库。
- 需要在 vLLM repo 内运行，便于使用固定相对路径和本地开发环境。
- `--model` 必须指向本地叶子模型目录，不能只指向模型集合父目录。
- 模型目录必须包含 `config.json`。
- 可选读取 `quantize_config.json`。
- 默认只读 `model.safetensors.index.json`；如果不存在，会退化为配置推断并标记
  `config_derived_quant`。

本地常见模型目录示例：

```text
/root/models/cyankiwi/...
/root/models/QuantTrio/...
/mnt/huggingface/hub/models--.../snapshots/<revision>
/mnt/modelscope/models/stepfun-ai/...
```

## 快速使用

### AWQ MoE 模型

```bash
.venv/bin/python benchmarks/marlin_gemm_shapes.py \
  --model /root/models/QuantTrio/Qwen3.5-122B-A10B-AWQ \
  --moe-backend marlin \
  --format both
```

常见现象：

- 如果 attention/shared expert 被 `modules_to_not_convert` 排除，Dense table
  可能为空。
- routed MoE expert 若存在 AWQ `qweight/qzeros/scales` 证据，会输出
  `actual_marlin`。

### ModelOpt / NVFP4 模型

```bash
.venv/bin/python benchmarks/marlin_gemm_shapes.py \
  --model /mnt/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/6c7f09d4036e97393f82e9f9ecd1a5c35ca5ee92 \
  --moe-backend marlin \
  --tp-sizes 4,8 \
  --ep-modes tp,tp_ep \
  --max-num-batched-tokens 2048,4096 \
  --decode-concurrency 1,32,64 \
  --format both
```

该类模型可能同时出现：

- attention dense 为 `fp8_e4m3`
- shared expert 或 routed MoE 为 `nvfp4`
- 未量化层为 `hypothetical_bf16`

### 只输出 JSON

```bash
.venv/bin/python benchmarks/marlin_gemm_shapes.py \
  --model /path/to/local/model \
  --format json > /tmp/marlin_shapes.json
```

### 写入 JSON 文件

```bash
.venv/bin/python benchmarks/marlin_gemm_shapes.py \
  --model /path/to/local/model \
  --format pretty \
  --json-out /tmp/marlin_shapes.json
```

### 展示 skipped 模块

```bash
.venv/bin/python benchmarks/marlin_gemm_shapes.py \
  --model /path/to/local/model \
  --include-skipped \
  --format both
```

`--include-skipped` 会显示被 ignore/exclude/modules_to_not_convert 排除的模块，
也会显示 router/gate 的 `skipped_router` 行。router/gate 不作为 Marlin GEMM
benchmark 目标。

## CLI 参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model PATH` | 必填 | 本地叶子模型目录，必须包含 `config.json`。 |
| `--max-num-batched-tokens LIST` | `2048,4096` | prefill 的 `size_m` 候选，支持逗号列表或重复传参。 |
| `--decode-concurrency LIST` | `1,32,64` | decode 单步 active sequence 数，每条 sequence 生成 1 token。 |
| `--tp-sizes LIST` | `4,8` | 要枚举的 tensor parallel size。 |
| `--ep-modes LIST` | `tp,tp_ep` | `tp` 表示不开 EP，`tp_ep` 表示 `enable_expert_parallel=True`。 |
| `--moe-backend NAME` | `marlin` | 工具目标是 Marlin shape；非 `marlin` 只打印 warning。 |
| `--dtype DTYPE` | `float16` | 用于描述 BF16/FP16 推测场景，不改变静态 shape。 |
| `--format {pretty,json,both}` | `both` | 输出 human-readable 表格、JSON 或两者都输出。 |
| `--json-out PATH` | 空 | 额外写出 JSON 文件。 |
| `--include-skipped` | 关闭 | 展示 skipped/excluded/router 行。 |
| `--verify-safetensors-index` | 开启 | 读取 `model.safetensors.index.json` 作为模块量化证据。 |
| `--no-verify-safetensors-index` | 关闭 | 不读取 index，仅按 config 推断并标记 `config_derived_quant`。 |

## 输出表解释

输出 JSON 顶层包含：

- `model`：解析后的模型目录绝对路径。
- `model_config`：脚本归一化后的关键模型结构字段。
- `quantization`：模型级量化配置摘要，只用于参考。
- `shape_inputs`：本次枚举的 prefill/decode 输入列表。
- `scenarios`：本次枚举的 TP/EP 场景。
- `dense`：Dense table 行。
- `moe`：MoE table 行。
- `warnings`：所有行 warning 的去重汇总。

Dense table 描述 dense linear GEMM shape。只有 `actual_marlin` 行会对应
`ops.marlin_gemm(size_m, size_n, size_k)`；BF16/FP16 推测行和 skipped 行只保留
shape 参考，`target_op` 为 `none`。关键字段：

| 字段 | 含义 |
| --- | --- |
| `scenario` | 例如 `tp4`、`tp4+ep`、`tp8`、`tp8+ep`。 |
| `phase` | `prefill` 或 `decode`。 |
| `layer_key` | 聚合后的模块 key 摘要。 |
| `layer_keys` | JSON 中保留的完整模块 key 列表。 |
| `op` | `qkv_proj`、`o_proj`、`gate_up_proj`、`down_proj` 等。 |
| `target_op` | 实际目标 torch op；只有 `actual_marlin` Dense 行为 `ops.marlin_gemm`，非 Marlin 行为 `none`。 |
| `size_m` | GEMM M；prefill 来自 `--max-num-batched-tokens`，decode 来自并发数。 |
| `size_n` | GEMM N；通常是输出通道或分片后的输出通道。 |
| `size_k` | GEMM K；通常是输入通道或分片后的输入通道。 |
| `group_size` | 有效 group size；配置为 `-1` 时会展开为当前 `size_k`。 |
| `quant_format` | `uint4`、`uint4b8`、`fp8_e4m3`、`nvfp4`、`mxfp4`、`bf16_or_fp16` 等。 |
| `marlin_path` | 推断的量化/内核家族，例如 `awq_marlin_wna16`、`fp4_marlin`。 |
| `call_status` | 当前模块是否真实走目标 Marlin op。 |
| `warning` | 配置/index 不一致、配置推断、BF16 推测等提示。 |
| `call_count` | 聚合到该 shape 行的模块数量。 |

MoE table 描述 routed expert GEMM shape。只有 `actual_marlin` 行会对应
`ops.moe_wna16_marlin_gemm(moe_block_size, top_k, size_m, size_n, size_k)`；
BF16/FP16 推测行和 skipped 行只保留 shape 参考，`target_op` 为 `none`。除
Dense 字段外还包含：

| 字段 | 含义 |
| --- | --- |
| `moe_block_size` | fused Marlin MoE kernel 的 block size M。 |
| `top_k` | w13 使用模型 experts-per-token；w2 固定为 `1`。 |
| `target_op` | 实际目标 torch op；只有 `actual_marlin` MoE 行为 `ops.moe_wna16_marlin_gemm`，非 Marlin 行为 `none`。 |
| `local_num_experts` | 当前 rank 本地 expert 数。 |
| `global_num_experts` | 模型全局 routed expert 数。 |
| `intermediate_size_per_partition` | TP/EP 后当前 GEMM 使用的 expert intermediate。 |

脚本会聚合 shape 完全相同且量化判定相同的行，因此同一行可能代表多个层。
需要逐层查看时，以 JSON 的 `layer_keys` 和 `call_count` 为准。

## call_status 判定语义

| `call_status` | 含义 |
| --- | --- |
| `actual_marlin` | 当前模块有量化配置和足够证据，真实可作为目标 Marlin op benchmark。 |
| `hypothetical_bf16` | 当前模块未量化、BF16/FP16，或 index 显示只有普通 `.weight`；仅作为 MNK 参考。 |
| `skipped` | 当前模块被 ignore/exclude/modules_to_not_convert 排除，或 fused shard 量化冲突。 |

常见 warning：

| warning | 含义 |
| --- | --- |
| `hypothetical_bf16` | BF16/FP16/unquantized 不会调用目标 Marlin op。 |
| `excluded_quant_module` | 命中排除规则，默认不展示，除非使用 `--include-skipped`。 |
| `index_disagrees_with_config` | config 声称量化，但 index 只有普通 weight 或证据不完整。 |
| `config_derived_quant` | index 不存在或粒度不足，只能按 config 推断。 |
| `modelopt_mixed_bf16_module` | ModelOpt mixed precision 中该模块未出现在量化映射里。 |
| `skipped_conflicting_quant:*` | fused shard 出现 FP8/NVFP4 等混合冲突。 |
| `skipped_router` | router/gate 不是 Marlin GEMM benchmark 目标。 |

## shape 推导规则

Dense `marlin_gemm(size_m, size_n, size_k)`：

```text
prefill size_m = each --max-num-batched-tokens
decode  size_m = each --decode-concurrency
```

QKV：

```text
size_k = hidden_size
q_heads_local = num_attention_heads / tp_size
kv_heads_local = max(1, num_kv_heads / tp_size)
q_size = q_heads_local * head_dim
kv_size = kv_heads_local * head_dim
size_n = q_multiplier * q_size + 2 * kv_size
```

`q_multiplier`：

- Qwen3.5/Qwen3Next 且 `attn_output_gate=true` 时为 `2`。
- MiniMax/GLM/Step 普通 QKV 为 `1`。

其他 dense linear：

```text
o_proj:
  size_k = q_heads_local * head_dim
  size_n = hidden_size

gate_up_proj:
  size_k = hidden_size
  size_n = 2 * intermediate_size / tp_size

down_proj:
  size_k = intermediate_size / tp_size
  size_n = hidden_size
```

MoE `moe_wna16_marlin_gemm(...)`：

```text
w13:
  top_k = experts_per_token
  size_m = M
  size_n = 2 * intermediate_size_per_partition
  size_k = hidden_size

w2:
  top_k = 1
  size_m = M * experts_per_token
  size_n = hidden_size
  size_k = intermediate_size_per_partition
```

TP/EP：

```text
非 EP:
  local_num_experts = global_num_experts
  intermediate_size_per_partition = moe_intermediate_size / tp_size

EP:
  local_num_experts = global_num_experts / tp_size
  intermediate_size_per_partition = moe_intermediate_size
```

`moe_block_size`：

```python
for block_size_m in [8, 16, 32, 48, 64]:
    if M * topk / local_num_experts / block_size_m < 0.9:
        break
```

decode 并发语义：

```text
decode_concurrency=C 表示一个 decode step 中 C 条 active sequence 各生成 1 token。
MoE w13.size_m = C
MoE w2.size_m = C * top_k
```

## 量化格式映射

| 来源 | 条件 | `quant_format` | `has_zp` | `marlin_path` |
| --- | --- | --- | --- | --- |
| AWQ | bits=4, zp=true | `uint4` | true | `awq_marlin_wna16` |
| AWQ | bits=8, zp=true | `uint8` | true | `awq_marlin_wna16` |
| GPTQ/WNA16 | int4 symmetric | `uint4b8` | false | `wna16_marlin` |
| GPTQ/WNA16 | int8 symmetric | `uint8b128` | false | `wna16_marlin` |
| FP8 | e4m3 weight-only | `fp8_e4m3` | false | `fp8_marlin` |
| NVFP4 | group size 通常为 16 | `nvfp4` | false | `fp4_marlin` |
| MXFP4 | group size 通常为 32 | `mxfp4` | false | `fp4_marlin` |
| BF16/FP16/unquantized | 无目标量化证据 | `bf16_or_fp16` | false | `none` |

AWQ 和 GPTQ/WNA16 都可能是 4-bit 或 8-bit，但含义不同：

- AWQ 的 `uint4/uint8` 表示 AWQ Marlin WNA16 路径，通常带 zero point。
- GPTQ/WNA16 的 `uint4b8/uint8b128` 表示 symmetric packed WNA16 格式，
  不带 AWQ zero point。
- `marlin_path` 描述量化/内核家族，不等价于 MoE。是否是 MoE 调用应看表名
  和 `target_op`。

## 支持的模型结构

当前实现重点覆盖以下模型族：

- `qwen3_5_text`
- `qwen3_5_moe_text`
- `qwen3_next`
- `minimax_m2`
- `glm4_moe`
- `step3p5`
- `step3p7`

归一化读取的关键字段包括：

- `hidden_size`
- `intermediate_size`
- `moe_intermediate_size`
- `shared_expert_intermediate_size`
- `share_expert_dim`
- `n_shared_experts`
- `num_hidden_layers`
- `moe_layer_indices`
- `moe_layers_enum`
- `first_k_dense_replace`
- `num_experts`
- `num_local_experts`
- `moe_num_experts`
- `num_experts_per_tok`
- `moe_top_k`
- `num_attention_heads`
- `num_key_value_heads`
- `num_attention_groups`
- `head_dim`
- `layer_types`

候选模块包括：

- attention `qkv_proj` 或等价 `q_proj/k_proj/v_proj`
- attention `o_proj`
- dense MLP `gate_up_proj` 或等价 `gate_proj/up_proj`
- dense MLP `down_proj`
- shared expert `gate_up/down`
- routed MoE experts `w13/w2`

router/gate 默认不输出；`--include-skipped` 时以 `skipped_router` 标记。

## safetensors index 证据

默认会读取 `model.safetensors.index.json` 的 `weight_map` key，作为模块级证据。
脚本只读取 key 名，不读取 tensor 内容。

常见判定规则：

- AWQ actual Marlin 需要看到 `qweight + qzeros + scales`。
- NVFP4 actual Marlin 需要看到 `weight + weight_scale + weight_scale_2`
  或等价 global scale 证据。
- FP8 actual Marlin 需要看到 `weight + weight_scale`。
- compressed-tensors WNA16 需要看到 packed weight/scale 证据，或者 config group
  明确且 index 粒度不足时标记 `config_derived_quant`。
- 如果只看到普通 `.weight`，判定为 `hypothetical_bf16`，并给出
  `index_disagrees_with_config`。

这个规则是为了避免把 ModelOpt/compressed-tensors 的模型级量化配置误套到
未量化模块上。

## 开发记录

本工具从一个实际问题出发：给定某个 vLLM serve 启动命令，希望知道 dense
Marlin 和 MoE Marlin kernel benchmark 应该覆盖哪些 MNK 参数组合。

关键开发决策：

- 从模型级量化收敛到模块级量化：`quant_method=modelopt` 或
  `compressed-tensors` 不能说明每个 linear 都真实量化。
- AWQ-Marlin 路径按 Marlin 目标收敛，BF16/FP16 只保留 shape 参考。
- FP8、NVFP4、MXFP4、compressed-tensors、modelopt 都可能进入 Marlin 路径；
  只有 BF16/FP16/unquantized 不能作为真实 Marlin op。
- ModelOpt `MIXED_PRECISION` 优先读取 `quantized_layers`，fused shard 混合
  FP8/NVFP4 时标记 `skipped_conflicting_quant`。
- compressed-tensors 按 `config_groups` 逐候选模块匹配，而不是使用第一个
  config group 全局套用。
- safetensors index 是强证据：能区分真实量化模块、普通 `.weight` 模块和
  config-derived 推断。
- `decode_concurrency=1,32,64` 用于表达 decode step 中 active sequence 数，
  MoE w2 的 `size_m` 会乘以 `top_k`。
- `moe_block_size` 按当前 fused Marlin MoE 逻辑枚举 `[8,16,32,48,64]`，
  不额外尝试 128/256。
- 输出会聚合同 shape/同量化判定的多层模块，并保留 `call_count` 和
  `layer_keys`。

## 验证记录

本次迁移环境中已执行：

```bash
./.venv/bin/python -m py_compile \
  benchmarks/marlin_gemm_shapes.py \
  tests/test_marlin_gemm_shapes.py
```

正式 pytest 覆盖了迁移后的独立测试，结果全部通过：

```text
PYTHONPATH=$PWD ./.venv/bin/python -m pytest tests/test_marlin_gemm_shapes.py -v
```

真实模型 smoke：

```text
Qwen3.5-122B-A10B-AWQ:
  dense 0
  moe 16
  warnings []

Step-3.7-Flash-NVFP4:
  dense 0
  moe 4
  warnings []
  moe_formats ['nvfp4']

Qwen3.6-35B-A3B-NVFP4:
  dense 8
  moe 4
  warnings []
  dense_formats ['fp8_e4m3', 'nvfp4']
  moe_formats ['nvfp4']
```

本环境中 `ruff` 暂不可用：

```text
/root/source/repos/marlin_v100/.venv/bin/python: No module named ruff
```

如果交付环境已安装 `ruff`，建议补跑：

```bash
./.venv/bin/python -m ruff check \
  benchmarks/marlin_gemm_shapes.py \
  tests/test_marlin_gemm_shapes.py \
  docs/20260612_057_marlin_gemm_shape_enumerator.md
```

## 交付注意事项

- 这是 shape/benchmark 参数设计工具，不是性能 benchmark。
- `actual_marlin` 表示静态证据显示该模块可走目标 Marlin op，不代表该 shape
  一定是性能热点。
- `hypothetical_bf16` 行用于保留 MNK 参考，不能当作实际 Marlin 调用；
  这类行的 `target_op` 为 `none`。
- 非 MoE AWQ dense 模型也可能在 Dense table 里显示
  `marlin_path=awq_marlin_wna16`。这表示 dense AWQ Marlin 路径，
  不是 `ops.moe_wna16_marlin_gemm`；应结合 `call_status=actual_marlin` 和
  `target_op=ops.marlin_gemm` 判断。
- `config_derived_quant` 表示缺少足够 index 粒度，需要结合 checkpoint 实际情况
  复核。
- EP 推导假设 `ep_size=tensor_parallel_size`，不模拟 router dispatch 的负载不均。
- 对新模型族，优先补模型结构字段归一化和候选 module key，再补量化证据规则。

## FAQ

### 非 MoE 模型为什么会看到 `awq_marlin_wna16`？

`awq_marlin_wna16` 是 AWQ Marlin 量化/内核家族描述，不代表该行一定来自 MoE。
如果它出现在 Dense table，并且 `call_status=actual_marlin`、
`target_op=ops.marlin_gemm`，说明这是 dense linear 的 AWQ Marlin shape。只有
MoE table 中 `call_status=actual_marlin`、
`target_op=ops.moe_wna16_marlin_gemm` 的行才表示 MoE Marlin GEMM。
