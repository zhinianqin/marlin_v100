# Marlin 模型 Shape Table Benchmark 使用说明

本文档说明 `benchmarks/benchmark_marlin_model_shapes_env.py` 的使用方式。

这个入口是 benchmark，不是 pytest。它会读取 `benchmarks/marlin_gemm_shapes.py`
基于模型目录生成的 Dense / MoE shape table，只对 `call_status == "actual_marlin"`
的去重行做 runtime benchmark，并使用 synthetic tensor 构造输入，不读取 checkpoint
tensor payload。

如果你想看对应的 pytest 校验说明，可以先读：
`docs/20260615_060_marlin_model_shape_table_test_usage.md`。

## 默认行为

默认命令使用 `--kind auto`，会自动识别模型里实际存在的 table：

- 只有 Dense actual rows 时，只跑 Dense。
- 只有 MoE actual rows 时，只跑 MoE。
- Dense 和 MoE 都存在且都包含 actual rows 时，同一次运行同时跑两边。

如果某个模型同时含有 Dense table 和 MoE table，脚本会分别统计并输出两边的
actual row 数、去重后 actual row 数、unsupported row 数，再按 selected combo 顺序
benchmark。

## 常用命令

### 1. 自动识别 Dense / MoE

```bash
PYTHONPATH=$PWD ./.venv/bin/python benchmarks/benchmark_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  --check
```

### 2. 只跑 MoE

```bash
PYTHONPATH=$PWD ./.venv/bin/python benchmarks/benchmark_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.5-122B-A10B-AWQ/snapshots/af62f78061dacbebd03825ceae7d675609e66320 \
  --kind moe \
  --check
```

### 3. 只跑 Dense

```bash
PYTHONPATH=$PWD ./.venv/bin/python benchmarks/benchmark_marlin_model_shapes_env.py \
  --model /mnt/huggingface/hub/models--QuantTrio--Qwen3.6-27B-AWQ/snapshots/9b507bdc9afafb87b7898700cc2a591aa6639461 \
  --kind dense \
  --check
```

### 4. 同时跑 Dense + MoE

```bash
PYTHONPATH=$PWD ./.venv/bin/python benchmarks/benchmark_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  --kind both \
  --warmup-iters 1 \
  --iters 50 \
  --check
```

### 5. 分片跑大模型

```bash
PYTHONPATH=$PWD MARLIN_EXHAUSTIVE_ENV_START=0 MARLIN_EXHAUSTIVE_ENV_LIMIT=1000 \
  ./.venv/bin/python benchmarks/benchmark_marlin_model_shapes_env.py \
  --model /path/to/model_dir \
  --kind both
```

## 输出状态含义

- `OK`：合法 env 组合 benchmark 成功；如果启用 `--check`，reference check 也通过。
- `REJECTED`：非法 env 组合被正确识别并拒绝，属于预期结果。
- `UNSUPPORTED`：MoE runtime helper 明确不支持该 actual row，脚本记录原因但不失败。
- `MISMATCH`：`--check` 下合法组合的 reference close 失败。
- `ERR`：运行时出现其他错误，或者非法组合没有按预期抛出显式 env rejection。

## 主要参数

- `--model`：模型目录，目录里必须有 `config.json`。
- `--kind {auto,dense,moe,both}`：benchmark 范围。
- `--warmup-iters`：默认 `1`。
- `--iters`：默认 `50`。
- `--check`：先做 reference close，再做 benchmark；非法组合下还会验证显式 env rejection。
- `--max-cases`：限制实际执行的 expanded combo 数，便于调试和 smoke。
- `--csv`：CSV 输出路径；默认写到 `benchmarks/results/<timestamp>_model_shapes_env_benchmark.csv`。

## 组合与分片

Dense 使用 `tests/sm70_env_sweep.py` 里的 Dense env combo 枚举，MoE 使用 MoE
env combo 枚举。`MARLIN_EXHAUSTIVE_ENV_START` 和 `MARLIN_EXHAUSTIVE_ENV_LIMIT`
继续表示按 “unique actual row x env combo” 展开的全局 selected combo index。

脚本会在启动时打印：

- 模型路径
- requested kind
- auto detected kinds
- Dense / MoE table 的 row 数和 actual row 数
- Dense / MoE env combo 数
- START / LIMIT / selected
- CSV 路径

每 64 个 selected combo 会打印一次 heartbeat，格式类似：

```text
checked=<n> ok=<n> rejected=<n> unsupported=<n> err=<n>
```
