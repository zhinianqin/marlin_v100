# 20260604 SM70 Marlin 自动 CTA / Warps / split-K 策略 CSV 分析

## 结论

本轮基于四份既有 CSV 重新分析并校准了 SM70 Marlin 自动 `CTA_M/CTA_N/warps` 与 split-K 策略：

- Dense 当前策略在 `8008` 个可对齐 case 中，`7498` 个为最优，`7993` 个为 top-2，top-2 覆盖率为 `99.81%`。
- MoE 当前策略在 `5040` 个 exact-comparable case 中，`4311` 个为最优，`4862` 个为 top-2，top-2 覆盖率为 `96.47%`。
- MoE exact-comparable 当前策略 `policy_us / best_us` 的 `p95=1.0273`、`p99=1.0561`、`max=1.0976`，没有 `>1.10` 的 exact regression。
- MoE 原自动策略中 AWQ `uint4, group=-1` 大形状 `64x256x4` 的约 `1.7x` 退化已通过 quant/group CTA 特例消除。
- Dense 仍有少数 rank=3 case 和 rank=2 但相对最优差距较大的 case；继续按当前策略保留，不为少数 CSV 点牺牲整体鲁棒性。

本轮判断“最优/次优”使用 normalized strategy rank，而不是跨 run latency。`20260604 auto` 与 `20260603 full` 是不同运行批次，延迟可作为 sanity signal，但策略优劣以同一份 `20260603 full matrix` 内归一化后的 strategy rank 为准。

## 输入 CSV

分析使用以下四份 CSV：

```text
benchmarks/results/20260604_dense_auto_ctam_warps_splitk_iters1.csv
benchmarks/results/20260603_dense_writeback_full_matrix_iters1_no_skip_tflops.csv
benchmarks/results/20260604_moe_auto_ctam_warps_splitk_iters1.csv
benchmarks/results/20260603_moe_writeback_full_matrix_iters1_no_skip_tflops.csv
```

CSV 行数与 status：

| CSV | rows | status |
| --- | ---: | --- |
| Dense auto 20260604 | 8008 | `OK=8008` |
| Dense full 20260603 | 86240 | `OK=86240` |
| MoE auto 20260604 | 8844 | `OK=8844` |
| MoE full 20260603 | 168960 | `OK=168960` |

字段兼容说明：

- 旧 `20260604_dense_auto_ctam_warps_splitk_iters1.csv` 与 `20260604_moe_auto_ctam_warps_splitk_iters1.csv` 使用 `resolved_cta` / `resolved_split_k`。
- 当前 benchmark artifact 已改为 `auto_cta_geometry` / `auto_split_k`。
- 分析脚本兼容读取旧字段，但新输出和新 benchmark 只使用 `auto_*` 命名。

## 归一化口径

为了把旧 full matrix CSV 与当前自动策略可比，分析脚本按以下规则归一化：

- Dense 对齐 key：`dense_class, quant, group_size, shape_id`。
- MoE 对齐 key：`method_class, quant, group_size, shape_id, routing_profile`。
- `split_k=unset` 归一化为 `1`。
- 重复策略取同一 key 下最快的 `marlin_us`。
- `20260603 full matrix` 的 `cta=auto` 使用创建该 CSV 时的代码状态归一化。
- 历史代码状态固定为 commit `abeccd8fede450dc3e818998abf276040ac4ef31`。

历史 `cta=auto` 映射：

- Dense：使用 `abeccd8fede450dc3e818998abf276040ac4ef31` 时的 Dense auto `CTA_M/CTA_N/warps` 规则；当前 Dense auto CTA 规则未改变。
- MoE：使用 `abeccd8fede450dc3e818998abf276040ac4ef31` 时的 MoE stage geometry：
  - `CTA_N=64 -> 64x64x4`
  - `CTA_N=128 -> 32x128x4`
  - `CTA_N=256 -> 32x256x4`

Rank 定义：

- 对同一 key 下所有归一化 strategy 按 `marlin_us` 排序。
- 当前 policy 的 strategy 在 full matrix strategy set 中存在时，记为 exact-comparable。
- rank 为比当前 policy 更快的不同 latency 数量加一。
- top-2 表示当前 policy 是同一 key 下最优或次优 normalized strategy。
- MoE 当前策略可能产生 `stage1=...;stage2=...` 的混合 geometry 或 split-K；旧 full matrix 使用单一 `cta` / `split_k` 维度，不能精确表达所有 stage-pair，因此这些 case 记录为 mixed/missing，不强行赋 rank。

## 原策略问题

Dense 原自动策略：

| 指标 | 数值 |
| --- | ---: |
| rows | 8008 |
| exact comparable | 8008 |
| top1 / top2 / top4 | 6954 / 7628 / 7938 |
| top2 rate | 95.25% |
| median ratio | 1.0000 |
| p95 ratio | 1.2103 |
| p99 ratio | 1.3558 |
| max ratio | 1.6568 |
| `>1.10` | 645 |

Dense 主要问题：

- `dense_heavy_mlp_up_m64_k4096_n14336`：原策略选择 `split_k=2`，full matrix 显示 `split_k=1` 更快。
- `dense_heavy_gqa_kv_m1024_k4096_n1024` 与 `M>=2048,K=4096,N=1024`：原策略对部分大 M 仍 split，退化明显。
- `dense_stress_splitk_starve_m4096/5120_k14336_n256`：大 M 小 N 下 split-K 阈值需要更细。

MoE 原自动策略：

| 指标 | 数值 |
| --- | ---: |
| rows | 8844 |
| exact comparable | 5004 |
| mixed/missing | 3840 |
| top1 / top2 / top4 | 3972 / 4436 / 4822 |
| top2 rate | 88.65% |
| median ratio | 1.0000 |
| p95 ratio | 1.0724 |
| p99 ratio | 1.5899 |
| max ratio | 1.7284 |
| `>1.10` | 200 |

MoE 主要问题：

- AWQ `uint4, group=-1` 在大 `CTA_N=256` shape 上默认 `64x256x4`，full matrix 显示 `32x256x4` 或 `64x256x8` 明显更好，最大约 `1.73x`。
- AWQ `uint8, group=-1` 的部分大 `CTA_N=256` shape 更适合 `64x256x8`。
- 旧 auto CSV 中若干 tiny-token mixed stage split-K 无法直接映射到旧 full matrix 的单一 split-K；该类只能做风险分析，不能强行当 exact rank。

## 当前校准策略

### Dense CTA / Warps

Dense 保留既有自动 CTA 与 warps 规则：

- `CTA_N` 按 `N` 最大可整除选择：`256 -> 128 -> 64`。
- `CTA_M` 继续按当前 `M + CTA_N` 规则选择。
- warps 继续由 `CTA_M/CTA_N` 组合决定。

原因：

- CSV 显示 Dense 当前 CTA/warps 已接近全矩阵最优。
- 对少数 U8 `M=64,K=14336,N=4096` case，`64x256x4` 会比当前 `64x256x8` 更快；但全局切换会伤害 FP4/FP8 与其他 shape。
- 因此本轮只校准 Dense split-K，不改 Dense CTA/warps。

### Dense split-K

Dense split-K 从纯 tile-count 规则改为 CSV 校准的 shape-aware 规则：

```text
if K < 4096 or K % 32 != 0:
  split_k = 1

if K == 4096 and N == 1024:
  M >= 2048 -> 1
  M >= 1024 -> 2
  M >= 64   -> 8
  M >= 24   -> 4
  else      -> 8

if K == 4096 and N >= 8192:
  M >= 48 -> 1
  M >= 16 -> 2
  M == 1  -> 8
  else    -> 2

if K == 4096 and N >= 4096:
  M >= 1024 -> 1
  M >= 48   -> 4
  M <= 16   -> 8
  else      -> 4

if K >= 8192 and N <= 256:
  M >= 4096 -> 2
  M >= 2048 -> 4
  else      -> 8

if K >= 8192 and N >= 4096:
  M >= 1024 -> 1
  M >= 48   -> 4
  else      -> 8

otherwise:
  use the original tile-count rule
```

### MoE CTA / Warps

MoE 基础 stage geometry 规则保留：

```text
CTA_N = largest divisible value among 256, 128, 64

CTA_N=64:
  64x64x4

CTA_N=128:
  tokens >= 4096 -> 64x128x8
  otherwise      -> 32x128x4

CTA_N=256:
  tokens >= 1024 -> 64x256x4
  otherwise      -> 32x256x4
```

新增 quant/group 特例：

```text
uint4 with group_size=-1 and CTA_N=256:
  32x256x4

uint8 with group_size=-1 and CTA_N=256 and tokens >= 1024:
  64x256x8
```

注意：C++ op 层只能看到 quant type、group size、shape 参数，看不到 Python method class。因此该规则是 quant/group strategy，不是只针对 `AWQMarlinMoEMethod` 的 method-class strategy。当前 inventory 中 MoE `uint4/uint8` ZP production path 来自 AWQ class，所以 CSV 中主要表现为 AWQ 问题。

### MoE split-K

MoE split-K 改为更保守的 stage requested split-K：

```text
if K % 32 != 0:
  split_k = 1

if K == 2048:
  cta_tiles <= 64 -> 2
  else            -> 1

if K < 4096:
  split_k = 1

cta_tiles <= 16  -> 8
cta_tiles <= 32  -> 4
cta_tiles <= 128 -> 2
else             -> 1
```

这里的 `auto_split_k` 表示自动策略请求值。kernel launch 仍会通过 `sm70_active_split_k(K, requested_split_k)` clamp 到实际 active split-K。

## 当前策略结果

Dense 当前策略：

| 指标 | 数值 |
| --- | ---: |
| rows | 8008 |
| exact comparable | 8008 |
| mixed/missing | 0 |
| top1 / top2 / top4 | 7498 / 7993 / 8007 |
| top2 rate | 99.81% |
| median ratio | 1.0000 |
| p95 ratio | 1.0075 |
| p99 ratio | 1.2727 |
| max ratio | 1.4358 |
| `>1.10` | 191 |
| rank counts | `1:7498, 2:495, 3:14, 5:1` |

Dense 结论：

- `7993/8008` case 为最优或次优。
- 剩余非 top-2 只有 `15` 个，其中 `14` 个为 rank=3、`1` 个为 rank=5。
- `>1.10` 的 ratio 主要来自 rank=2 或少量 rank=3 case；rank 口径仍满足“最优或次优”主目标。
- 对 rank=3/5 继续过拟合需要引入 quant/class/shape 更细分支；结合用户要求，top-2 与 top-3/top-4 差距极小时优先保留鲁棒策略。

MoE 当前策略：

| 指标 | 数值 |
| --- | ---: |
| rows | 8844 |
| exact comparable | 5040 |
| mixed/missing | 3804 |
| top1 / top2 / top4 | 4311 / 4862 / 5032 |
| top2 rate | 96.47% |
| median ratio | 1.0000 |
| p95 ratio | 1.0273 |
| p99 ratio | 1.0561 |
| max ratio | 1.0976 |
| `>1.10` | 0 |
| rank counts | `1:4311, 2:551, 3:130, 4:40, 5:5, 6:1, 7:2` |

MoE 结论：

- exact-comparable case 中 `4862/5040` 为最优或次优。
- 所有 exact-comparable case 的 `policy_us / best_us <= 1.10`。
- 剩余 rank=3/4/5/6/7 case 与 best 的差距较小，`p95=1.0273`、`p99=1.0561`，符合“不为极小 top-2/top-3/top-4 差距过拟合”的策略。
- `3804` 个 mixed/missing case 来自 stage-pair geometry 或 stage-pair split-K；旧 full matrix 单一 `cta/split_k` 维度不能精确表达这些策略，因此不纳入 exact rank。

## 剩余风险

- Dense 存在 rank=2 但 ratio 较高的 case；如果未来要求 ratio 而非 rank，需要继续做 per-template 或 per-quant split-K/warps 分支。
- MoE mixed-stage case 仍需要用新 calibrated benchmark 做运行层 sanity；旧 full matrix 不能完整证明 stage-pair split-K 的 rank。
- 本轮策略来自当前 inventory CSV，不声明为所有未来 shape 的全局理论最优。

## 复现命令

所有命令从 repo 根目录执行：

```bash
cd /root/source/repos/marlin_v100
export PYTHONPATH=$PWD/python:$PWD
```

分析脚本：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/python benchmarks/analyze_marlin_auto_strategy.py \
  --dense-auto benchmarks/results/20260604_dense_auto_ctam_warps_splitk_iters1.csv \
  --dense-full benchmarks/results/20260603_dense_writeback_full_matrix_iters1_no_skip_tflops.csv \
  --moe-auto benchmarks/results/20260604_moe_auto_ctam_warps_splitk_iters1.csv \
  --moe-full benchmarks/results/20260603_moe_writeback_full_matrix_iters1_no_skip_tflops.csv \
  --output-json benchmarks/results/20260604_auto_strategy_csv_analysis.json \
  --output-md benchmarks/results/20260604_auto_strategy_csv_analysis.md
```

Python 静态检查：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/python -m py_compile \
  benchmarks/analyze_marlin_auto_strategy.py \
  tests/writeback_marlin_cases.py \
  benchmarks/benchmark_marlin_dense.py \
  benchmarks/benchmark_marlin_moe.py \
  tests/test_marlin_dense.py \
  tests/test_marlin_moe.py
```

构建与导入：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=8
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST='7.5'
export CMAKE_ARGS='-DCMAKE_CUDA_FLAGS=-gencode arch=compute_75,code=sm_75'

PYTHONPATH=$PWD/python ./.venv/bin/python setup.py build_ext --inplace \
  > benchmarks/results/20260604_build_auto_strategy_calibrated.log 2>&1

PYTHONPATH=$PWD/python ./.venv/bin/python - <<'PY'
import marlin_v100
import marlin_v100._C
import marlin_v100._moe_C
print("imports ok")
PY
```

focused correctness：

```bash
./test.sh tests/test_marlin_dense.py -q -k "auto_split_k or auto_cta"
./test.sh tests/test_marlin_moe.py -q -k "auto_split_k or auto_cta or stage1"
```

新 calibrated reduced full benchmark 路径，不覆盖旧 `20260603_*` 结果：

```bash
BENCH_PRESET=full \
DENSE_ARGS='--shape-suite all --warmup-iters 0 --iters 1 --omit-skip --csv benchmarks/results/20260604_dense_auto_strategy_calibrated_iters1.csv' \
./benchmark.sh dense \
  > benchmarks/results/20260604_dense_auto_strategy_calibrated_iters1.log 2>&1

BENCH_PRESET=full \
MOE_ARGS='--shape-suite all --warmup-iters 0 --iters 1 --omit-skip --csv benchmarks/results/20260604_moe_auto_strategy_calibrated_iters1.csv' \
./benchmark.sh moe \
  > benchmarks/results/20260604_moe_auto_strategy_calibrated_iters1.log 2>&1
```

CSV validation：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/python - <<'PY'
import csv
from collections import Counter

checks = [
    ("benchmarks/results/20260604_dense_auto_strategy_calibrated_iters1.csv", "dense_class"),
    ("benchmarks/results/20260604_moe_auto_strategy_calibrated_iters1.csv", "method_class"),
]

for path, class_key in checks:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(path)
    print("rows", len(rows))
    print("has_auto_cta_geometry", bool(rows and "auto_cta_geometry" in rows[0]))
    print("has_auto_split_k", bool(rows and "auto_split_k" in rows[0]))
    for key in [class_key, "status", "quant", "group_size", "shape_id", "auto_cta_geometry", "auto_split_k"]:
        print(key, dict(sorted(Counter(r[key] for r in rows).items())))
    if rows and "routing_profile" in rows[0]:
        print("routing_profile", dict(sorted(Counter(r["routing_profile"] for r in rows).items())))
    print("skip_rows_in_csv", sum(r["status"] == "SKIP" for r in rows))
    print("errors", sum(r["status"] == "ERR" for r in rows))
    print("mismatch", sum(r["status"] == "MISMATCH" for r in rows))
    print("missing_tflops", sum(
        not r.get("marlin_tflops") or r["marlin_tflops"] == "n/a"
        for r in rows
        if r["status"] == "OK"
    ))
PY
```

## 本轮已执行验证

已执行：

- Python 静态检查：通过。
- 分析脚本复现：通过，重新生成 `benchmarks/results/20260604_auto_strategy_csv_analysis.json` 与 `.md`。
- 构建：通过，日志 `benchmarks/results/20260604_build_auto_strategy_calibrated.log`。
- 构建日志 warning/error 扫描：未发现 `error:`, `fatal error`, `warning #...`, `nvcc warning`, `deprecated-gpu-targets`, `declared but never referenced`。
- 导入：`import marlin_v100`, `import marlin_v100._C`, `import marlin_v100._moe_C` 通过。

待执行或后续执行：

- focused correctness。
- 新 calibrated reduced full benchmark。
- 新 benchmark CSV validation。
