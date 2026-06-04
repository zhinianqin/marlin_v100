# 2026-06-04 SM70 Marlin 自动 CTA/Warps/Split-K 清理说明

## 摘要

本轮将 SM70 Marlin Dense/MoE 的手工运行时 geometry 与 split-K 控制面从当前生产源码、tests、Dense/MoE benchmark 中彻底移除。当前只保留自动策略：

- Dense/MoE kernel launch 不再读取环境变量决定 CTA geometry 或 split-K。
- Dense/MoE benchmark 不再提供 CTA 或 split-K 矩阵维度。
- Full matrix test/benchmark 的矩阵定义改为：

```text
Dense: class x quant x group x shape
MoE:   class x quant x group x shape
```

benchmark CSV 不再记录手工矩阵维度列，只记录实际解析结果：

```text
resolved_cta
resolved_split_k
```

历史 docs 暂不统一整理，避免把旧实验记录改写成新的语义；当前状态以本文档和源码为准。

## 自动策略

Dense CTA/warps：

- `CTA_N` 按 `256 -> 128 -> 64` 顺序选择能整除 `N` 的最大值。
- `CTA_M` 继续按当前 `M + CTA_N` 规则选择。
- warps 由 resolved `CTA_M/CTA_N` 唯一决定。

Dense split-K：

```text
if K < 4096 or K % 32 != 0:
  split_k = 1

tiles = ceil(M / CTA_M) * max(1, N / CTA_N)

if K >= 8192 and N <= 256:
  tiles <= 64  -> 8
  tiles <= 128 -> 4
  tiles <= 256 -> 2
  else         -> 1
else:
  tiles <= 16 -> 8
  tiles <= 32 -> 4
  tiles <= 64 -> 2
  else        -> 1
```

MoE CTA/warps：

```text
CTA_N=64  -> 64x64x4
CTA_N=128 -> tokens >= 4096 ? 64x128x8 : 32x128x4
CTA_N=256 -> tokens >= 1024 ? 64x256x4 : 32x256x4
```

MoE split-K：

```text
if K < 4096 or K % 32 != 0:
  split_k = 1

tiles = ceil((tokens * topk) / CTA_M) * max(1, N / CTA_N)

tiles <= 16 -> 8
tiles <= 32 -> 4
tiles <= 64 -> 2
else        -> 1
```

已有 active split-K clamp 仍保留，用于把 requested split-K 限制到实际 K tiles 范围内。`split_k > 1` 继续走 fp32 `c_tmp` accumulation 和 fp32-to-fp16 epilogue。

## 矩阵规模

M/tokens 覆盖：

```text
1, 8, 16, 24, 32, 48, 64, 1024, 2048, 4096, 5120
```

当前 streaming matrix summary：

| 方向 | total | supported | skipped | shape_count |
|---|---:|---:|---:|---:|
| Dense | 45,760 | 8,008 | 37,752 | 143 |
| MoE | 102,410 | 8,844 | 93,566 | 418 |

Dense resolved split-K coverage：

| split_k | cases |
|---:|---:|
| 1 | 32,960 |
| 2 | 2,240 |
| 4 | 960 |
| 8 | 9,600 |

Dense resolved CTA coverage：

| resolved_cta | cases |
|---|---:|
| `32x128x4` | 5,760 |
| `32x256x4` | 13,440 |
| `64x64x4` | 6,720 |
| `64x128x8` | 960 |
| `64x256x8` | 2,240 |
| `128x256x8` | 8,960 |
| `256x64x8` | 3,840 |
| `256x128x8` | 3,840 |

MoE resolved CTA coverage：

| resolved_cta | cases |
|---|---:|
| `32x128x4` | 13,230 |
| `32x256x4` | 44,590 |
| `64x64x4` | 16,170 |
| `64x128x8` | 2,940 |
| `64x256x4` | 25,480 |

MoE routing coverage：

| routing_profile | cases |
|---|---:|
| `uniform` | 51,205 |
| `zipfian` | 51,205 |

## 代码范围

生产源码：

- `csrc/quantization/marlin/sm70_marlin_common.cuh`
- `csrc/quantization/marlin/sm70_marlin_splitk.cuh`
- `csrc/quantization/marlin/sm70_marlin_gemm.cuh`
- `csrc/quantization/marlin/sm70_marlin_*_gemm.cu`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_gemm.cuh`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_*_gemm.cu`

测试、benchmark 和本地 wrapper：

- `python/marlin_v100/moe.py`
- `tests/writeback_marlin_cases.py`
- `tests/test_marlin_dense.py`
- `tests/test_marlin_moe.py`
- `tests/test_marlin_linear_kernel.py`
- `tests/test_marlin_moe_kernel.py`
- `benchmarks/benchmark_marlin_dense.py`
- `benchmarks/benchmark_marlin_moe.py`

## 复现命令

所有命令从 repo 根目录执行：

```bash
cd /root/source/repos/marlin_v100
export PYTHONPATH=$PWD/python:$PWD
```

源码/tests/Marlin benchmark 残留扫描：

```bash
rg -n "<deleted manual CTA/split-K control symbols>" \
  csrc tests benchmarks/benchmark_marlin_dense.py benchmarks/benchmark_marlin_moe.py \
  python/marlin_v100 -S
```

验收标准：无匹配。历史 docs 暂不纳入该扫描验收。

Python 编译：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/python -m py_compile \
  python/marlin_v100/moe.py \
  tests/writeback_marlin_cases.py \
  tests/test_marlin_dense.py \
  tests/test_marlin_moe.py \
  tests/test_marlin_linear_kernel.py \
  tests/test_marlin_moe_kernel.py \
  benchmarks/benchmark_marlin_dense.py \
  benchmarks/benchmark_marlin_moe.py
```

矩阵统计：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/python - <<'PY'
from tests.writeback_marlin_cases import (
    dense_writeback_matrix_summary,
    moe_writeback_matrix_summary,
)
print(dense_writeback_matrix_summary())
print(moe_writeback_matrix_summary())
PY
```

pytest collection：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/pytest --collect-only -q
```

导入检查：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python - <<'PY'
import marlin_v100
import marlin_v100._C
import marlin_v100._moe_C
print("imports ok")
PY
```

Numeric/regression tests：

```bash
./test.sh tests/test_marlin_dense.py -q
./test.sh tests/test_marlin_moe.py -q
```

Full matrix loop tests：

```bash
./test.sh tests/test_marlin_linear_kernel.py -q
./test.sh tests/test_marlin_moe_kernel.py -q
```

Strategy benchmark 使用新文件名，避免覆盖旧 full-matrix 结果：

```bash
BENCH_PRESET=full \
DENSE_ARGS='--shape-suite all --warmup-iters 0 --iters 1 --omit-skip --csv benchmarks/results/20260604_dense_auto_ctam_warps_splitk_iters1.csv' \
./benchmark.sh dense \
  > benchmarks/results/20260604_dense_auto_ctam_warps_splitk_iters1.log 2>&1

BENCH_PRESET=full \
MOE_ARGS='--shape-suite all --warmup-iters 0 --iters 1 --omit-skip --csv benchmarks/results/20260604_moe_auto_ctam_warps_splitk_iters1.csv' \
./benchmark.sh moe \
  > benchmarks/results/20260604_moe_auto_ctam_warps_splitk_iters1.log 2>&1
```

CSV validation：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/python - <<'PY'
import csv
from collections import Counter

checks = [
    ("benchmarks/results/20260604_dense_auto_ctam_warps_splitk_iters1.csv", "dense_class"),
    ("benchmarks/results/20260604_moe_auto_ctam_warps_splitk_iters1.csv", "method_class"),
]

for path, class_key in checks:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(path)
    print("rows", len(rows))
    for key in [
        class_key,
        "status",
        "quant",
        "group_size",
        "shape_id",
        "resolved_cta",
        "resolved_split_k",
    ]:
        counts = Counter(r[key] for r in rows)
        if key == "shape_id":
            print(key, "count", len(counts))
        else:
            print(key, dict(sorted(counts.items())))
    if rows and "routing_profile" in rows[0]:
        print("routing_profile", dict(sorted(Counter(r["routing_profile"] for r in rows).items())))
    print("skip_rows_in_csv", sum(r["status"] == "SKIP" for r in rows))
    print("errors", sum(r["status"] == "ERR" for r in rows))
    print("mismatch", sum(r["status"] == "MISMATCH" for r in rows))
    print(
        "missing_tflops",
        sum(
            not r.get("marlin_tflops") or r["marlin_tflops"] == "n/a"
            for r in rows
            if r["status"] == "OK"
        ),
    )
PY
```

## 实际验证结果

基础验证：

| 项 | 结果 |
|---|---|
| 残留扫描 | PASS，生产源码、tests、Dense/MoE benchmark、本地 Python wrapper 无匹配 |
| Python 编译 | PASS |
| import | PASS，`marlin_v100`、`marlin_v100._C`、`marlin_v100._moe_C` 均可导入 |
| pytest collection | `708 tests collected in 5.33s` |

Numeric/regression：

| 命令 | 结果 |
|---|---|
| `./test.sh tests/test_marlin_dense.py -q` | `345 passed in 302.97s` |
| `./test.sh tests/test_marlin_moe.py -q` | `180 passed, 2 skipped in 1827.06s` |

Full matrix class-path loop：

| 命令 | 结果 |
|---|---|
| `./test.sh tests/test_marlin_linear_kernel.py -q` | `58 passed, 14 warnings in 1541.87s` |
| `./test.sh tests/test_marlin_moe_kernel.py -q` | `14 passed, 14 warnings in 840.78s` |

Strategy benchmark 结果：

| 方向 | selected_cases | saved_rows | OK | SKIP | ERR | MISMATCH | CSV SKIP rows | missing_tflops |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense | 45,760 | 8,008 | 8,008 | 37,752 | 0 | 0 | 0 | 0 |
| MoE | 102,410 | 8,844 | 8,844 | 93,566 | 0 | 0 | 0 | 0 |

Strategy benchmark 产物：

| 方向 | CSV | Log |
|---|---|---|
| Dense | `benchmarks/results/20260604_dense_auto_ctam_warps_splitk_iters1.csv` | `benchmarks/results/20260604_dense_auto_ctam_warps_splitk_iters1.log` |
| MoE | `benchmarks/results/20260604_moe_auto_ctam_warps_splitk_iters1.csv` | `benchmarks/results/20260604_moe_auto_ctam_warps_splitk_iters1.log` |

旧 full-matrix benchmark 产物未覆盖：

```text
benchmarks/results/20260603_dense_writeback_full_matrix_iters1_no_skip_tflops.csv
benchmarks/results/20260603_moe_writeback_full_matrix_iters1_no_skip_tflops.csv
```

Dense strategy CSV 覆盖：

| 字段 | 覆盖 |
|---|---|
| rows | 8,008 |
| status | `OK`: 8,008 |
| quant | `uint4,uint4b8,uint8,uint8b128`: 各 1,716；`fp8`: 858；`nvfp4,mxfp4`: 各 143 |
| group_size | `-1`: 2,145；`16`: 143；`32`: 1,859；`64`: 1,716；`128`: 2,145 |
| shape_id | 143 |
| resolved_cta | `32x128x4,32x256x4,64x64x4,64x128x8,64x256x8,128x256x8,256x64x8,256x128x8` |
| resolved_split_k | `1,2,4,8` |

MoE strategy CSV 覆盖：

| 字段 | 覆盖 |
|---|---|
| rows | 8,844 |
| status | `OK`: 8,844 |
| method_class | `gptq_moe,awq_moe,compressed_tensors_wna16_moe`: 各 2,948 |
| quant | `uint4,uint8`: 各 1,474；`uint4b8,uint8b128`: 各 2,948 |
| group_size | `-1`: 2,508；`32`: 2,508；`64`: 2,112；`128`: 1,716 |
| shape_id | 418 |
| routing_profile | `uniform,zipfian`: 各 4,422 |
| resolved_cta | `32x128x4,32x256x4,64x64x4,64x128x8,64x256x4` |
| resolved_split_k | `1` 和 stage1/stage2 mixed `2/4/8` 组合 |

## 当前验收口径

- 生产源码、tests、Dense/MoE benchmark 中不再保留手工 CTA/split-K 环境变量控制面。
- Full matrix test/benchmark 不再把 CTA 或 split-K 作为矩阵维度。
- `resolved_cta` 和 `resolved_split_k` 是观测字段，不是输入维度。
- 当前 SM70 机器已经完成构建导入、数值回归、class-path loop 和 strategy benchmark 验证。
