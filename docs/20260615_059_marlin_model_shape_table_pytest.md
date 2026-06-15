# pytest `--model` Marlin GEMM Table 测试说明

本文档说明 `tests/test_marlin_model_shapes_env.py` 提供的模型驱动 table
测试功能。这个测试把 `benchmarks/marlin_gemm_shapes.py` 的静态 shape
枚举结果接入 pytest，用一个用户显式指定的本地模型目录生成 Dense table / MoE
table，再对 table schema、pretty 输出、`actual_marlin` 行参数和 SM70 env
组合 runtime 精度进行验证。

## 目标

新增的是 pytest 测试入口，不改变 `benchmarks/marlin_gemm_shapes.py` 的核心用途。

测试目标：

- 用 pytest 参数 `--model <model_dir>` 指定一个本地模型 leaf directory。
- 自动读取 `<model_dir>/config.json`。
- 调用 `marlin_gemm_shapes.parse_args()` 和 `build_payload()` 生成结构化 payload。
- 对 Dense table / MoE table 做 schema 和 pretty 输出检查。
- 只对 `call_status == "actual_marlin"` 的行做 runtime 精度测试。
- exhaustive runtime 覆盖所有 SM70 dense / MoE env override 组合。
- Dense-only、MoE-only、Dense+MoE、无 actual row、warning 非空都应被正常处理。

非目标：

- 不扫描多个模型。
- 不支持把 `--model` 指向模型集合父目录后自动发现子模型。
- 不读取真实 checkpoint tensor payload。
- 不在默认 pytest run 中执行完整 env sweep。

## 文件入口

相关文件：

| 文件 | 职责 |
| --- | --- |
| `tests/conftest.py` | 注册 pytest `--model` 参数并校验模型目录。 |
| `tests/test_marlin_model_shapes_env.py` | 生成 table payload，执行 schema / pretty / runtime env sweep 测试。 |
| `benchmarks/marlin_gemm_shapes.py` | 仍然是 shape table 的唯一生成逻辑来源。 |
| `tests/sm70_env_sweep.py` | 提供 Dense/MoE env 组合枚举、env 设置和合法性判断。 |
| `tests/test_marlin_dense.py` | 提供 dense runtime synthetic tensor 和 reference helper。 |
| `tests/test_marlin_moe.py` | 提供 MoE runtime synthetic tensor 和 reference helper。 |

## `--model` 语义

pytest 参数：

```bash
--model /path/to/model_dir
```

`--model` 必须指向模型目录，不是 `config.json` 文件。

合法形式：

```bash
--model /mnt/huggingface/hub/models--QuantTrio--Qwen3.5-122B-A10B-AWQ/snapshots/af62f78061dacbebd03825ceae7d675609e66320
```

不作为目标适配：

```bash
--model /mnt/.../config.json
```

测试内部只做固定发现：

```python
config_path = Path(model_dir) / "config.json"
```

路径校验语义：

| 情况 | 行为 |
| --- | --- |
| 未传 `--model` | table-driven 测试 skip。 |
| `--model` 路径不存在 | pytest fail。 |
| `--model` 是文件 | pytest fail。 |
| 模型目录缺少 `config.json` | pytest fail。 |

未传参数时的 skip 提示：

```text
pass --model <model_dir> to run model-shape table tests
```

缺少 `config.json` 时的错误提示包含：

```text
--model must point to a model directory containing config.json
```

## 支持的模型目录形态

测试不绑定某一个根目录，也不扫描根目录。只要传入的是包含 `config.json` 的
leaf model directory，并且 `marlin_gemm_shapes.py` 支持其模型结构，就可以运行。

常见目录来源：

```text
/root/models/<org>/<model>
/mnt/modelscope/models/<org>/<model>
/mnt/huggingface/hub/models--<org>--<model>/snapshots/<revision>
```

示例：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /root/models/QuantTrio/Qwen3.6-27B-AWQ \
  -v
```

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/modelscope/models/stepfun-ai/Step-3___7-Flash-NVFP4 \
  -v
```

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.5-122B-A10B-AWQ/snapshots/af62f78061dacbebd03825ceae7d675609e66320 \
  -v
```

下面这些不是该测试的语义：

```bash
--model /root/models
--model /mnt/modelscope/models
--model /mnt/huggingface/hub
```

如果需要测多个模型，由调用者分别启动 pytest，每次传一个具体 leaf model
directory。

## 默认测试内容

默认运行不启用 exhaustive runtime。此时主要验证：

- `marlin_gemm_shapes.py` 能根据传入模型目录生成 payload。
- payload 顶层字段存在：
  `model`、`model_config`、`quantization`、`shape_inputs`、`scenarios`、
  `dense`、`moe`、`warnings`。
- Dense table 每行包含公共字段：
  `scenario`、`phase`、`layer_key`、`op`、`target_op`、`size_m`、
  `size_n`、`size_k`、`group_size`、`quant_method`、`quant_format`、
  `has_zp`、`marlin_path`、`call_status`、`call_count`。
- MoE table 每行额外包含：
  `moe_block_size`、`top_k`、`local_num_experts`、`global_num_experts`、
  `intermediate_size_per_partition`。
- `actual_marlin` Dense 行必须能映射到 `ops.marlin_gemm`。
- `actual_marlin` MoE 行必须能映射到 `ops.moe_wna16_marlin_gemm`。
- `actual_marlin` 行的 `size_m/size_n/size_k` 等 shape 参数必须为正数。
- `actual_marlin` 行的 `quant_format` 和 `group_size` 必须能被现有 runtime
  helper 支持。
- pretty 输出必须包含 `Model:`、`Config:`、`Warnings:`、`Dense table`、
  `MoE table`。
- 若某张表为空，pretty 输出中该表必须显示 `(no rows)`。

默认命令：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py -v
```

未传 `--model` 时，上述测试应 skip，不应失败。

## Runtime 精度测试

runtime 精度测试只覆盖 `call_status == "actual_marlin"` 的 table row。

不会进入 runtime 的行：

- `hypothetical_bf16`
- `skipped`
- router/gate 行
- 空 Dense table 或空 MoE table

runtime 使用 synthetic tensor，shape 完全来自 table row，不读取真实 checkpoint
tensor payload。

Dense 行使用现有 dense helper：

- `_make_dense_env_sweep_case`
- `_assert_dense_env_sweep_combo_matches_reference`

MoE 行使用现有 MoE helper：

- `_make_moe_env_sweep_inputs`
- `_moe_stage1_reference`
- `_moe_stage2_inputs_and_reference`
- `_run_moe_env_stage1_combo`
- `_run_moe_env_stage2_combo`

### Dense shape 映射

Dense table row 对应：

```text
activation: (size_m, size_k)
weight:     (size_k, size_n)
output:     (size_m, size_n)
```

目标 op：

```text
ops.marlin_gemm
```

### MoE shape 映射

MoE `w13` row 对应 stage1 GEMM：

```text
activation:     (size_m, size_k)
expert weight:  (local_num_experts, size_k, size_n)
output:         (size_m * top_k, size_n)
```

MoE `w2` row 对应 stage2 GEMM：

```text
activation:     (size_m, size_k)
expert weight:  (local_num_experts, size_k, size_n)
output:         (size_m, size_n)
top_k:          1
```

目标 op：

```text
ops.moe_wna16_marlin_gemm
```

## group_size 注意事项

`benchmarks/marlin_gemm_shapes.py` 的 table 会把配置里的 `group_size=-1`
展开为当前 row 的有效 `size_k`，因为 table 面向 benchmark 参数解释。

现有 runtime helper 内部仍用 `-1` 表示 full-K group。测试侧会在构造 runtime
key 时把：

```text
row["group_size"] == row["size_k"]
```

归一化回 helper 语义：

```text
group_size = -1
```

因此文档、pretty table、JSON payload 中看到的 full-K group 仍是有效
`size_k`，runtime helper 中看到的是 `-1`。

## SM70 env 全组合覆盖

runtime env sweep 由 marker 和环境变量共同控制。

测试 marker：

```text
sm70_env_exhaustive
```

必须设置：

```bash
MARLIN_EXHAUSTIVE_ENV_SWEEP=1
```

Dense runtime 遍历 `iter_env_combinations()`，覆盖：

```text
SM70_MARLIN_DENSE_CTA_GEOMETRY
SM70_MARLIN_DENSE_SPLIT_K
SM70_MARLIN_DENSE_METADATA_CACHE
```

MoE runtime 遍历 `iter_moe_env_combinations()`，覆盖：

```text
SM70_MARLIN_MOE_CTA_GEOMETRY
SM70_MARLIN_MOE_SPLIT_K
SM70_MARLIN_MOE_METADATA_CACHE
```

每个组合都会先通过现有合法性判断：

- Dense：`dense_env_combo_is_legal(...)`
- MoE：`moe_stage_env_combo_is_legal(...)`

合法组合必须运行并通过 `torch.testing.assert_close`。非法组合必须抛出匹配
`EXPLICIT_ENV_REJECTION_RE` 的 `RuntimeError`。

完整 sweep 命令：

```bash
PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_SWEEP=1 \
  ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  -m sm70_env_exhaustive \
  -v
```

分片或 smoke 命令：

```bash
PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_SWEEP=1 \
  MARLIN_EXHAUSTIVE_ENV_START=0 \
  MARLIN_EXHAUSTIVE_ENV_LIMIT=1000 \
  ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  -m sm70_env_exhaustive \
  -v
```

`MARLIN_EXHAUSTIVE_ENV_LIMIT=1` 可用于确认 runtime 接线，但不代表完整覆盖。

## 推荐验收命令

原有 shape enumerator 单测：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_gemm_shapes.py -v
```

新测试未传模型时应 skip：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py -v
```

MoE 模型 table smoke：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.5-122B-A10B-AWQ/snapshots/af62f78061dacbebd03825ceae7d675609e66320 \
  -v
```

Dense 模型 table smoke：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.6-27B-AWQ/snapshots/9b507bdc9afafb87b7898700cc2a591aa6639461 \
  -v
```

Dense 最小 runtime smoke：

```bash
PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_SWEEP=1 \
  MARLIN_EXHAUSTIVE_ENV_LIMIT=1 \
  ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.6-27B-AWQ/snapshots/9b507bdc9afafb87b7898700cc2a591aa6639461 \
  -m sm70_env_exhaustive \
  -v
```

MoE 最小 runtime smoke：

```bash
PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_SWEEP=1 \
  MARLIN_EXHAUSTIVE_ENV_LIMIT=1 \
  ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.5-122B-A10B-AWQ/snapshots/af62f78061dacbebd03825ceae7d675609e66320 \
  -m sm70_env_exhaustive \
  -v
```

注意：MoE runtime smoke 会按真实 table row 的 hidden/intermediate/expert
shape 构造 synthetic expert weights。大模型的第一条 actual row 也可能比较重，
`MARLIN_EXHAUSTIVE_ENV_LIMIT=1` 只限制 env 组合数量，不缩小 row 本身的
tensor shape。

## 常见结果解释

Dense-only 模型：

- Dense table 非空。
- MoE table 可以为空。
- 默认运行应是 table smoke pass，MoE runtime exhaustive 测试 skip。

MoE-only 模型：

- Dense table 可以为空。
- MoE table 非空。
- 默认运行应是 table smoke pass，Dense runtime exhaustive 测试 skip。

Dense + MoE 模型：

- 两张表都可以有 `actual_marlin` 行。
- 开启 exhaustive 时两类 runtime 都会覆盖。

两张表都没有 `actual_marlin`：

- schema 和 pretty smoke 仍应通过。
- runtime 测试 skip，并提示没有 actual row 可测。

缺少 `config.json`：

- 这是用户输入错误，测试应 fail。
- 错误信息应包含：

```text
--model must point to a model directory containing config.json
```

## 与 shape enumerator 文档的关系

`docs/20260612_057_marlin_gemm_shape_enumerator.md` 说明
`benchmarks/marlin_gemm_shapes.py` 如何枚举 table，以及 table 字段的静态含义。

本文档说明 pytest 如何消费这些 table，并把 `actual_marlin` 行接到 runtime
精度和 SM70 env sweep 测试中。
