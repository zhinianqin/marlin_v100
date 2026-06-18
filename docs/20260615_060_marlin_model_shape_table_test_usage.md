# Marlin 模型 Shape Table pytest 使用说明

本文档面向日常使用者，说明如何用 pytest 的 `--model` 参数运行
`marlin_gemm_shapes.py` 输出的 Dense / MoE table 测试。

如果你只想验证某个本地模型目录能否被当前 SM70 Marlin 测试系统正确识别，
从本文档开始即可。实现细节和测试覆盖设计见
`docs/20260615_059_marlin_model_shape_table_pytest.md`。

## 一句话用途

运行：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  -v
```

测试会做三件事：

- 读取 `/path/to/model_dir/config.json`。
- 调用 `benchmarks/marlin_gemm_shapes.py` 生成 Dense table / MoE table。
- 验证 table schema、pretty 输出，以及 `actual_marlin` 行是否能进入现有
  Marlin runtime 精度测试体系。

默认命令只跑轻量 table smoke。完整 SM70 env runtime sweep 需要额外打开
`MARLIN_EXHAUSTIVE_ENV_SWEEP=1`。

## 前置条件

从仓库根目录运行：

```bash
cd /root/source/repos/marlin_v100
```

推荐使用本仓库虚拟环境：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest --version
```

如果要运行 runtime 精度测试，还需要：

- 本地扩展已构建完成。
- 当前机器有 SM70 / V100 CUDA runtime。
- `vllm._C` / `vllm._moe_C` 能导入。
- `torch.ops._C.marlin_gemm` 和
  `torch.ops._moe_C.moe_wna16_marlin_gemm` 已注册。

快速导入检查：

```bash
PYTHONPATH=$PWD ./.venv/bin/python - <<'PY'
import torch
import vllm._C
import vllm._moe_C

assert hasattr(torch.ops._C, "marlin_gemm")
assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm")
print("marlin imports ok")
PY
```

## `--model` 应该传什么

`--model` 传的是模型目录，不是 `config.json` 文件。

正确：

```bash
--model /path/to/model_dir
```

其中测试会自动查找：

```text
/path/to/model_dir/config.json
```

错误：

```bash
--model /path/to/model_dir/config.json
```

也不要传模型集合父目录：

```bash
--model /root/models
--model /mnt/modelscope/models
--model /mnt/huggingface/hub
```

这个测试不做多模型扫描。要测哪个模型，就传哪个具体 leaf model directory。

## 常见模型目录

### `/root/models`

示例：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /root/models/QuantTrio/Qwen3.6-27B-AWQ \
  -v
```

只要该目录里存在 `config.json`，并且模型结构能被
`benchmarks/marlin_gemm_shapes.py` 识别，就可以作为 `--model` 参数。

### `/mnt/modelscope/models`

示例：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/modelscope/models/stepfun-ai/Step-3___7-Flash-NVFP4 \
  -v
```

注意 ModelScope 目录名可能包含转换后的下划线或三下划线，实际路径以本地
`config.json` 所在目录为准。

### `/mnt/huggingface/hub`

Hugging Face cache 通常需要传到 `snapshots/<revision>` 这一层：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.5-122B-A10B-AWQ/snapshots/af62f78061dacbebd03825ceae7d675609e66320 \
  -v
```

不要只传到：

```text
/mnt/huggingface/hub/models--QuantTrio--Qwen3.5-122B-A10B-AWQ
```

这一层通常没有直接的 `config.json`。

## 最常用命令

### 1. 不传模型，确认测试可收集

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py -v
```

预期结果：测试 skip。

skip 提示包含：

```text
pass --model <model_dir> to run model-shape table tests
```

这是正常行为，不代表失败。

### 2. 跑 MoE 模型 table smoke

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.5-122B-A10B-AWQ/snapshots/af62f78061dacbebd03825ceae7d675609e66320 \
  -v
```

常见结果：

- table / pretty smoke pass。
- Dense runtime 测试可能 skip，因为 Dense table 为空或没有 Dense
  `actual_marlin` 行。
- MoE runtime 测试默认 skip，因为未启用 exhaustive env sweep。

### 3. 跑 Dense 模型 table smoke

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.6-27B-AWQ/snapshots/9b507bdc9afafb87b7898700cc2a591aa6639461 \
  -v
```

常见结果：

- table / pretty smoke pass。
- MoE runtime 测试可能 skip，因为 MoE table 为空。
- Dense runtime 测试默认 skip，因为未启用 exhaustive env sweep。

### 4. 跑 Dense 最小 runtime smoke

```bash
PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_SWEEP=1 \
  MARLIN_EXHAUSTIVE_ENV_LIMIT=1 \
  ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.6-27B-AWQ/snapshots/9b507bdc9afafb87b7898700cc2a591aa6639461 \
  -m sm70_env_exhaustive \
  -v
```

`MARLIN_EXHAUSTIVE_ENV_LIMIT=1` 只跑第一个被选中的 env 组合，适合确认
runtime 接线和精度检查能进入 kernel。

### 5. 跑 MoE 最小 runtime smoke

```bash
PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_SWEEP=1 \
  MARLIN_EXHAUSTIVE_ENV_LIMIT=1 \
  ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.5-122B-A10B-AWQ/snapshots/af62f78061dacbebd03825ceae7d675609e66320 \
  -m sm70_env_exhaustive \
  -v
```

MoE smoke 可能比 Dense 慢很多，因为 synthetic expert weight 会按 table
row 的真实 `local_num_experts`、`size_k`、`size_n` 构造和量化。

### 6. 跑完整 env sweep

```bash
PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_SWEEP=1 \
  ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  -m sm70_env_exhaustive \
  -v
```

完整 sweep 会遍历所有 Dense 或 MoE env 组合，可能非常慢。

### 7. 分片跑完整 env sweep

```bash
PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_SWEEP=1 \
  MARLIN_EXHAUSTIVE_ENV_START=0 \
  MARLIN_EXHAUSTIVE_ENV_LIMIT=1000 \
  ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  -m sm70_env_exhaustive \
  -v
```

分片变量含义：

| 变量 | 含义 |
| --- | --- |
| `MARLIN_EXHAUSTIVE_ENV_START` | 从第几个 env case 开始选中。 |
| `MARLIN_EXHAUSTIVE_ENV_LIMIT` | 最多选中多少个 env case。 |

长跑时测试会主动输出 progress 行；测试会在输出 progress 时临时关闭
pytest capture，所以默认 `pytest -v` 下也能实时看见：

- start 摘要：模型路径、actual row 数、每个 row 的 env 组合数、`START/LIMIT`
- row 摘要：当前 Dense / MoE row 的 `scenario/phase/op/M/N/K/quant/group`
- heartbeat：每处理 64 个选中的 env 组合打印一次
  `checked/legal/rejected/total_index`
- summary：测试结束时打印 `checked/legal/rejected/total_seen`

建议先用 `MARLIN_EXHAUSTIVE_ENV_LIMIT=1` 验证 runtime 接线，再用
`MARLIN_EXHAUSTIVE_ENV_LIMIT=10` 看 row 级 progress 和耗时节奏。需要调试
pytest capture 本身时，也可以额外加 `-s` 或 `--capture=tee-sys`。`LIMIT=1000`
仍然是重测试，但 start、row、heartbeat 和 summary 会持续给出反馈。

## 默认 table smoke 会检查什么

默认 smoke 不运行 Marlin kernel，但会检查 table 是否能被测试系统消费。

检查内容包括：

- payload 顶层字段存在。
- Dense table 每行 schema 完整。
- MoE table 每行 schema 完整。
- pretty 输出包含 `Model:`、`Config:`、`Warnings:`、`Dense table`、
  `MoE table`。
- 空 Dense table 或空 MoE table 必须在 pretty 输出中显示 `(no rows)`。
- `actual_marlin` Dense 行必须指向 `ops.marlin_gemm`。
- `actual_marlin` MoE 行必须指向 `ops.moe_wna16_marlin_gemm`。
- `actual_marlin` 行的 `size_m/size_n/size_k` 等参数必须是正数。
- `quant_format` 必须能映射到现有 helper 支持的量化名称。
- `group_size` 必须能被对应 Dense/MoE helper 处理。

这些检查能尽早发现 table 字段变更、模型结构适配问题或 helper 不支持的新量化
组合。

## runtime 精度测试会检查什么

runtime 测试只对下面的行生效：

```text
call_status == "actual_marlin"
```

不会进入 runtime 的行：

| 行类型 | 原因 |
| --- | --- |
| `hypothetical_bf16` | 只是 BF16/FP16 shape 参考，不是目标 Marlin op。 |
| `skipped` | 被 ignore/exclude 或其他规则排除。 |
| router/gate | router 不是 Marlin GEMM benchmark 目标。 |
| 空 table | 没有 actual row 可测。 |

Dense runtime 使用 synthetic tensor：

```text
activation = (size_m, size_k)
weight     = (size_k, size_n)
output     = (size_m, size_n)
```

MoE `w13` runtime：

```text
activation    = (size_m, size_k)
expert weight = (local_num_experts, size_k, size_n)
output        = (size_m * top_k, size_n)
```

MoE `w2` runtime：

```text
activation    = (size_m, size_k)
expert weight = (local_num_experts, size_k, size_n)
output        = (size_m, size_n)
top_k         = 1
```

测试不读取 checkpoint 中的真实 tensor payload。

## env sweep 覆盖范围

Dense exhaustive 覆盖：

```text
SM70_MARLIN_DENSE_CTA_GEOMETRY
SM70_MARLIN_DENSE_SPLIT_K
SM70_MARLIN_DENSE_METADATA_CACHE
```

MoE exhaustive 覆盖：

```text
SM70_MARLIN_MOE_CTA_GEOMETRY
SM70_MARLIN_MOE_SPLIT_K
SM70_MARLIN_MOE_METADATA_CACHE
```

每个 env 组合分两类：

| 类型 | 预期 |
| --- | --- |
| 合法组合 | kernel 成功运行，并通过 `torch.testing.assert_close`。 |
| 非法组合 | 抛出匹配 `EXPLICIT_ENV_REJECTION_RE` 的 `RuntimeError`。 |

合法性判断复用：

- Dense：`dense_env_combo_is_legal(...)`
- MoE：`moe_stage_env_combo_is_legal(...)`

## 如何看 pytest 结果

### `1 passed, 2 skipped`

这是最常见的默认结果。

含义：

- table / pretty smoke 通过。
- Dense 或 MoE runtime 因未启用 `MARLIN_EXHAUSTIVE_ENV_SWEEP=1` 或无 actual
  row 而 skip。

### `3 skipped`

通常表示未传 `--model`。

检查命令中是否缺少：

```bash
--model /path/to/model_dir
```

### 缺少 `config.json`

错误示例：

```text
--model must point to a model directory containing config.json: /path/to/model_dir
```

处理方式：

- 确认传入的是模型 leaf directory。
- 对 Hugging Face cache，通常需要传到 `snapshots/<revision>`。
- 不要传 `/root/models`、`/mnt/modelscope/models` 或 `/mnt/huggingface/hub`
  这种集合父目录。

### `--model must be a model directory, not a file`

通常是把 `config.json` 文件路径传给了 `--model`。

改成传其父目录：

```bash
--model /path/to/model_dir
```

### `no actual dense rows to test`

说明当前模型的 Dense table 为空，或者 Dense table 没有
`call_status == "actual_marlin"` 的行。

这对 MoE-only 模型是正常现象。

### `no actual moe rows to test`

说明当前模型的 MoE table 为空，或者 MoE table 没有
`call_status == "actual_marlin"` 的行。

这对 Dense-only 模型是正常现象。

### runtime 很慢

常见于大型 MoE 模型。原因是测试会按 table row 构造 synthetic expert weights，
而不是缩小模型维度。

建议：

- 先跑默认 table smoke。
- 再用 `MARLIN_EXHAUSTIVE_ENV_LIMIT=1` 做 runtime 接线 smoke。
- 用 `MARLIN_EXHAUSTIVE_ENV_LIMIT=10` 确认 progress 输出和单 row 耗时。
- 完整 sweep 使用 `MARLIN_EXHAUSTIVE_ENV_START/LIMIT` 分片。

## 与 `marlin_gemm_shapes.py` 的关系

如果你只想看 table 内容，可以直接运行：

```bash
.venv/bin/python benchmarks/marlin_gemm_shapes.py \
  --model /path/to/model_dir \
  --moe-backend marlin \
  --format pretty
```

如果你想验证 table 是否能被当前测试系统消费，并确认 actual Marlin 行能进入
runtime 精度测试，就运行：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  -v
```

二者使用同一个 table 生成逻辑。pytest 不从 pretty 文本反解析参数，而是直接消费
`build_payload()` 返回的结构化 payload。

## 推荐工作流

新增或检查一个模型时，建议按下面顺序：

1. 确认模型目录存在 `config.json`。
2. 用 `benchmarks/marlin_gemm_shapes.py --format pretty` 人工看 Dense/MoE table。
3. 用 `pytest tests/test_marlin_model_shapes_env.py --model ...` 跑默认 table
   smoke。
4. 若 table 中有 Dense `actual_marlin` 行，跑 Dense 最小 runtime smoke。
5. 若 table 中有 MoE `actual_marlin` 行，跑 MoE 最小 runtime smoke。
6. 需要完整覆盖时，再开启完整或分片 exhaustive sweep。

最小命令组合：

```bash
.venv/bin/python benchmarks/marlin_gemm_shapes.py \
  --model /path/to/model_dir \
  --moe-backend marlin \
  --format pretty

PYTHONPATH=$PWD ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  -v

PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_SWEEP=1 \
  MARLIN_EXHAUSTIVE_ENV_LIMIT=1 \
  ./.venv/bin/pytest tests/test_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  -m sm70_env_exhaustive \
  -v
```
