# SM70 Dense Split-K All Quant Alignment

## 摘要

本轮以 dense `uint4` zero-point 的 split-K atomic fp32 reduce 路径为模板，把同类能力对齐到了其它 SM70 dense quant kernel：

- `uint4b8`
- `uint8`
- `uint8b128`
- `fp8`
- `nvfp4`
- `mxfp4`

同时将 `uint4` 自身的 split-K 公共逻辑抽出到 `csrc/quantization/marlin/sm70_dense_splitk.cuh`，让七个 dense kernel 共用同一套：

- split-K env parser
- 32-K tile partitioner
- 可复用 fp32 `c_tmp` 校验和 fallback allocation
- Volta accumulator reorder 后的 fp32 atomic epilogue
- `float4 -> half2 + half2` fp32-to-fp16 convert kernel

Public `marlin_gemm(...)` schema 没有变化，继续使用已有 `Tensor? c_tmp_or_none` 参数。MoE workspace 接口没有修改。

## 运行上下文

- Branch: `sm70-u4-splitk-atomic-reduce`
- Base commit before this change: `4516488`
- Working tree: tracked source/docs changes present; `benchmarks/results/` is ignored artifact
- GPU: `Tesla V100-SXM2-32GB`
- Capability: `sm70 (7.0)`
- Build target: local `marlin_v100` extension

## 实现内容

新增公共头文件：

```text
csrc/quantization/marlin/sm70_dense_splitk.cuh
```

该头文件提供：

| helper | 作用 |
|---|---|
| `parse_sm70_dense_split_k(env_name)` | 支持 unset/empty/`1`/`2`/`4`/`8`，其它值报错 |
| `sm70_dense_get_splitk_ctmp(...)` | 校验或 fallback 分配 fp32 `c_tmp` |
| `sm70_dense_splitk_partition<GroupSize>(...)` | 以 `CTA_K=32` 为单位做 group-aware split-K partition |
| `sm70_dense_active_split_k(...)` | 处理 `split_k > K/32` 时的实际非空 partition 数 |
| `Sm70DenseAtomicFp32Epilogue<Traits>` | 复用 CUTLASS Volta epilogue accumulator reorder，最后 atomicAdd 到 fp32 `c_tmp` |
| `launch_sm70_dense_fp32_to_fp16(...)` | vectorized fp32 `C_tmp` 转 fp16 `C` |

## Quant 对齐矩阵

| quant | split env | group 支持 | zero-point | 特殊处理 |
|---|---|---|---|---|
| `uint4` | `SM70_MARLIN_U4_SPLIT_K` | `-1/32/64/128` | fp16 zp | 已改为复用公共 helper |
| `uint4b8` | `SM70_MARLIN_U4B8_SPLIT_K` | `-1/32/64/128` | 不支持 zp | 新增 split-K variant |
| `uint8` | `SM70_MARLIN_U8_SPLIT_K` | `-1/32/64/128` | fp16 zp | 新增 split-K variant |
| `uint8b128` | `SM70_MARLIN_U8B128_SPLIT_K` | `-1/32/64/128` | 不支持 zp | 新增 split-K variant |
| `fp8` | `SM70_MARLIN_FP8_SPLIT_K` | `-1/128` | 不支持 zp | 新增 split-K variant |
| `nvfp4` | `SM70_MARLIN_NVFP4_SPLIT_K` | `16` | 不支持 zp | split-K path 在 atomic epilogue 前对 accumulator 乘 `global_scale[0]` |
| `mxfp4` | `SM70_MARLIN_MXFP4_SPLIT_K` | `32` | 不支持 zp | 新增 split-K variant |

所有 quant 的 no-split fast path 保持原 kernel launch；`split_k == 1` 或 env unset 时忽略 `c_tmp_or_none`，不经过 fp32 atomic reduce、`cudaMemsetAsync` 或 convert kernel。

## Split-K 语义

每个 quant 的 split-K path 统一使用：

```text
grid = (ceil(M / CTA_M), ceil(N / CTA_N), active_split_k)
active_split_k = min(requested_split_k, K / 32)
```

每个 `blockIdx.z` 通过公共 partitioner 得到：

```text
k_begin
partition_k
```

并保证：

- `K % 32 == 0` 是唯一 K 对齐要求。
- `k_begin % 32 == 0`。
- `partition_k % 32 == 0`。
- 所有非空 partition 连续覆盖 `[0, K)`。
- 当 `GroupSize >= 32` 时，partitioner 尽量让非最后分片按 group tile 对齐。
- 当 `GroupSize < 32`，例如 NVFP4 的 `group_size=16`，partitioner 退化为 32-tile 切分；因为 16 整除 32，`k_begin` 仍落在 scale group 边界上。

atomic epilogue 不包含 column guard。当前所有目标 dense path 在 dispatch 前都调用 `check_sm70_dense_n_tile_alignment(...)`，要求 `size_n % CTA_N == 0`。M tail 仍由 row guard 处理。

## 验证

已通过：

```text
./build.sh
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import torch, marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('import ok'); print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())"
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q tests/test_marlin_dense.py -k "split_k"
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q tests/test_marlin_dense.py
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
git diff --check
```

结果：

| 验证项 | 结果 |
|---|---|
| build | pass |
| import | pass, `Tesla V100-SXM2-32GB`, `(7, 0)` |
| `tests/test_marlin_dense.py -k "split_k"` | `70 passed, 247 deselected` |
| `tests/test_marlin_dense.py` | `317 passed` |
| `pytest --collect-only -q` | `503 tests collected` |
| `git diff --check` | pass |

新增 split-K correctness 覆盖：

- `split_k=2/4/8`：覆盖 U4B8/U8/U8B128/FP8/NVFP4/MXFP4。
- 非均匀 K smoke：
  - U4B8: `K=384, group_size=128`
  - U8: `K=352, group_size=32`
  - U8B128: `K=384, group_size=128`
  - FP8: `K=384, group_size=128`
  - NVFP4: `K=288, group_size=16`
  - MXFP4: `K=352, group_size=32`
- `c_tmp` 行为：
  - 每个新 quant 至少一个 split-K correctness case 传入足够大的 fp32 `c_tmp`。
  - 每个新 quant 验证 no-split env unset 时传入 `c_tmp` 不影响 fast path。
  - 每个新 split env 验证非法值 `3` 和 `abc` 报错。

U4 既有 `c_tmp=None` fallback、too-small、dtype、contiguous、device mismatch 等细粒度 rejection 仍保留。

## After Smoke Benchmark

本轮做了 after-only smoke benchmark，用来确认六个新增 quant 的 no-split 和 split-K path 都能实际 launch 并完成 timing。该 smoke 不替代完整 baseline/after 性能矩阵。

结果文件：

```text
benchmarks/results/20260528_222930_sm70_dense_splitk_all_quant_smoke_after/benchmark.log
benchmarks/results/20260528_222930_sm70_dense_splitk_all_quant_smoke_after/all_results.csv
```

口径：

- CTA: default CTA
- `M=1,128`
- `K=N=4096`
- split: `unset,4`
- group:
  - U4B8/U8/U8B128/FP8: `128`
  - NVFP4: `16`
  - MXFP4: `32`
- `warmup_iters=5`
- `iters=20`
- shared reusable `c_tmp = torch.empty((128 * 4096,), dtype=torch.float32, device="cuda")`

`kernel_like_us` smoke summary:

| quant | M | split | marlin_us | marlin_tflops | torch_us | speedup |
|---|---:|---|---:|---:|---:|---:|
| uint4b8 | 1 | unset | 305.152 | 0.110 | 59.392 | 0.195 |
| uint4b8 | 128 | unset | 307.200 | 13.981 | 115.712 | 0.377 |
| uint4b8 | 1 | 4 | 146.432 | 0.229 | 59.392 | 0.406 |
| uint4b8 | 128 | 4 | 215.040 | 19.973 | 115.712 | 0.538 |
| uint8 | 1 | unset | 313.328 | 0.107 | 59.392 | 0.190 |
| uint8 | 128 | unset | 323.584 | 13.273 | 116.736 | 0.361 |
| uint8 | 1 | 4 | 145.408 | 0.231 | 59.392 | 0.408 |
| uint8 | 128 | 4 | 214.016 | 20.068 | 115.712 | 0.541 |
| uint8b128 | 1 | unset | 306.176 | 0.110 | 58.880 | 0.192 |
| uint8b128 | 128 | unset | 312.320 | 13.752 | 115.712 | 0.370 |
| uint8b128 | 1 | 4 | 145.408 | 0.231 | 58.368 | 0.401 |
| uint8b128 | 128 | 4 | 212.992 | 20.165 | 115.712 | 0.543 |
| fp8 | 1 | unset | 308.736 | 0.109 | 59.392 | 0.192 |
| fp8 | 128 | unset | 321.536 | 13.358 | 115.712 | 0.360 |
| fp8 | 1 | 4 | 147.456 | 0.228 | 59.392 | 0.403 |
| fp8 | 128 | 4 | 216.064 | 19.878 | 115.712 | 0.536 |
| nvfp4 | 1 | unset | 307.200 | 0.109 | 59.888 | 0.195 |
| nvfp4 | 128 | unset | 311.824 | 13.774 | 115.712 | 0.371 |
| nvfp4 | 1 | 4 | 141.312 | 0.237 | 58.368 | 0.413 |
| nvfp4 | 128 | 4 | 209.920 | 20.460 | 115.712 | 0.551 |
| mxfp4 | 1 | unset | 811.520 | 0.041 | 59.392 | 0.073 |
| mxfp4 | 128 | unset | 819.200 | 5.243 | 115.712 | 0.141 |
| mxfp4 | 1 | 4 | 268.288 | 0.125 | 59.392 | 0.221 |
| mxfp4 | 128 | 4 | 340.992 | 12.596 | 116.736 | 0.342 |

这个 smoke 的主要结论是：六个新增 quant 的 `split_k=4` experimental path 都能实际执行，且在默认 CTA、`K=N=4096` 的 smoke 中没有出现 obvious failure。该表不是性能验收结论，因为没有改动前 baseline，也没有全 CTA sweep。

## 完整 Benchmark 状态

用户计划中的完整 benchmark 矩阵是：

```text
quant = uint4b8, uint8, uint8b128, fp8, nvfp4, mxfp4
M = 1,2,4,8,16,32,64,128
K = 4096,8192,16384
N = 4096
split_k = unset,2,4,8
CTA = 15 supported SM70 dense CTA
metrics = operator_us, kernel_like_us
```

这对应 `6 * 8 * 3 * 4 * 15 = 8640` 个 `run_case(...)`，每个 case 又包含 torch/marlin operator timing 和 kernel-like timing。按本轮 smoke 的耗时估计，完整 baseline+after 会是长时间 sweep，不适合在这次代码对齐收口中直接串行跑完。

因此本文件不写入完整性能结论，也不声称 split-K 对所有 quant/CTA 的 best latency 不回退。当前可确认的是：

- 功能路径已对齐。
- public schema 未变化。
- no-split fast path 仍存在，并且 `split_k=unset/1` 不使用 `c_tmp`。
- dense correctness 与 split-K 专项 tests 已通过。
- after smoke benchmark 已覆盖所有新增 quant 的实际 launch。

后续如需完整 benchmark，应在 clean baseline 和 after commit 上分别运行同一 runner，并生成 `compare_best_by_shape.csv` / `compare_same_config.csv` 后再下性能结论。

## 完整 Benchmark Runner 草案

下面 runner 可用于 clean baseline/after 两个 commit。Baseline 需要在本改动前的 commit 上运行；after 在包含本改动的 commit 上运行。

```bash
OUT_DIR="benchmarks/results/$(date +%Y%m%d_%H%M%S)_sm70_dense_splitk_all_quant_after"
mkdir -p "$OUT_DIR"
export OUT_DIR

PYTHONPATH=$PWD/python ./.venv/bin/python - <<'PY' 2>&1 | tee "$OUT_DIR/benchmark.log"
import csv
import os
from pathlib import Path

import torch

from benchmarks.benchmark_marlin_dense import run_case, tflops_from_us
from benchmarks.common import check_cuda_ready
from marlin_v100 import ops
from marlin_v100.dense import marlin_make_c_tmp

OUT_DIR = Path(os.environ["OUT_DIR"])
CTA_VALUES = [
    "32x128x4", "32x256x4", "64x64x4", "64x128x4", "64x128x8",
    "64x256x4", "64x256x8", "128x64x4", "128x64x8", "128x128x4",
    "128x128x8", "128x256x8", "256x64x4", "256x64x8", "256x128x8",
]
QUANT_CASES = [
    ("uint4b8", "SM70_MARLIN_U4B8_CTA", "SM70_MARLIN_U4B8_SPLIT_K", 128),
    ("uint8", "SM70_MARLIN_U8_CTA", "SM70_MARLIN_U8_SPLIT_K", 128),
    ("uint8b128", "SM70_MARLIN_U8B128_CTA", "SM70_MARLIN_U8B128_SPLIT_K", 128),
    ("fp8", "SM70_MARLIN_FP8_CTA", "SM70_MARLIN_FP8_SPLIT_K", 128),
    ("nvfp4", "SM70_MARLIN_NVFP4_CTA", "SM70_MARLIN_NVFP4_SPLIT_K", 16),
    ("mxfp4", "SM70_MARLIN_MXFP4_CTA", "SM70_MARLIN_MXFP4_SPLIT_K", 32),
]
M_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
K_VALUES = [4096, 8192, 16384]
N = 4096
SPLITS = ["unset", "2", "4", "8"]
WARMUP_ITERS = 20
ITERS = 100

check_cuda_ready()
ops._load_dense()
c_tmp = marlin_make_c_tmp(torch.device("cuda"), max(M_VALUES) * N)

rows = []
for quant_name, cta_env, split_env, group_size in QUANT_CASES:
    for cta in CTA_VALUES:
        os.environ[cta_env] = cta
        for split in SPLITS:
            if split == "unset":
                os.environ.pop(split_env, None)
            else:
                os.environ[split_env] = split
            for k in K_VALUES:
                for m in M_VALUES:
                    row = run_case(
                        model=f"custom_k{k}",
                        quant_name=quant_name,
                        group_size=group_size,
                        act_order=False,
                        is_k_full=True,
                        size_m=m,
                        size_k=k,
                        size_n=N,
                        reuse_output=True,
                        use_fp32_reduce=True,
                        warmup_iters=WARMUP_ITERS,
                        iters=ITERS,
                        c_tmp=c_tmp,
                    )
                    if row is None:
                        continue
                    flops = int(row["flops"])
                    for metric_name, metrics in row["results"].items():
                        rows.append({
                            "metric": metric_name,
                            "quant": quant_name,
                            "CTA": cta,
                            "split_k": split,
                            "M": m,
                            "K": k,
                            "N": N,
                            "group_size": group_size,
                            "torch_us": metrics["torch_us"],
                            "marlin_us": metrics["marlin_us"],
                            "speedup": metrics["speedup"],
                            "torch_tflops": tflops_from_us(flops, metrics["torch_us"]),
                            "marlin_tflops": tflops_from_us(flops, metrics["marlin_us"]),
                            "launch_dominated": row["launch_dominated"],
                        })

csv_path = OUT_DIR / "all_results.csv"
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"csv={csv_path}")
PY
```

## 结论

本轮完成了 SM70 dense split-K experimental path 的功能对齐：U4 的 post-`1274aeb...` 能力已经推广到 U4B8/U8/U8B128/FP8/NVFP4/MXFP4，并抽出了公共实现，减少后续 MoE 或更多 quant path 复用时的重复成本。

当前应保留该实现，原因是：

- no-split fast path 不变，仍由 env unset/`1` 承担默认性能路径。
- split-K path 只在显式 per-quant env 打开时启用。
- correctness 覆盖了所有新增 quant、非均匀 K、`c_tmp` 复用和 invalid env。
- after smoke benchmark 证明新增路径能在 V100 上实际运行。

尚未完成的是完整 baseline/after 性能矩阵。后续如果要把 split-K 作为某些 quant 的推荐配置，需要先跑完整 CTA/split sweep，特别关注 `M=64/128,K=8192/16384`，并将 `M<=16` 标注为 launch/fixed-overhead dominated。
