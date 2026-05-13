# SM70 Pure GEMM 全自动 Sweep 工具

## Summary

本轮新增一个无人值守的 SM70 pure GEMM tile/resource/benchmark sweep
工具，用来系统性比较 extracted CUTLASS threadblock pure GEMM 的
`kCtaM/kCtaN/kCtaK/kWarps` 组合。

工具目标不是替代 dense quant kernel benchmark，而是给后续 u4/u4b8
IteratorB 设计提供一个更完整的 pure GEMM 上限参考：

- 自动构建并保留 build log。
- 自动提取 `cuobjdump --dump-resource-usage` 资源信息。
- 自动解析 `ptxas` 的 stack/spill 信息。
- 自动运行所有候选 tile 组合。
- 每个候选做多次 independent repeat。
- 每个 repeat 完成后立即追加到 `raw_results.jsonl`。
- 对每个 MKN 输出两套排序表：
  - TFLOPs 优先排序。
  - `REG` 优先、TFLOPs 次优先的资源排序。
- unsupported / invalid / runtime failure 只写入结果，不中断整轮 sweep。

默认只扫 extracted CUTLASS threadblock pure GEMM 路径，不扫旧 CuTe
实验路径。

## Code Changes

### `sm70_cutlass_matmul_probe.cu`

文件：

```text
csrc/quantization/marlin/sm70_cutlass_matmul_probe.cu
```

保持 public op schema 不变：

```text
sm70_cutlass_matmul_probe(
  Tensor a,
  Tensor b,
  int cta_m,
  int cta_n,
  int cta_k,
  int warps,
  int stages,
  int a_path,
  int b_path
) -> Tensor
```

本轮只扩展 extracted CUTLASS threadblock pure GEMM 的模板实例。
仍使用：

- `cutlass::gemm::threadblock::DefaultMma`
- Volta TensorOp `8x8x4`
- CUTLASS predefined A/B shared-memory layout
- 2-stage pipeline
- `a_path = 2`
- `b_path = 0`

新增支持 `CTA_K=64/128`，并补齐合法的 4/8-warp 变体。

当前实例化的 canonical threadblock 形状为：

| CTA_M | CTA_N | Warps |
|---:|---:|---:|
| 64 | 64 | 4 |
| 64 | 128 | 4 |
| 64 | 128 | 8 |
| 64 | 256 | 8 |
| 128 | 64 | 4 |
| 128 | 64 | 8 |
| 128 | 128 | 4 |
| 128 | 128 | 8 |
| 128 | 256 | 8 |

每个形状都实例化：

```text
CTA_K = 32, 64, 128
```

也就是当前 C++ 侧实际生成 27 个 extracted CUTLASS threadblock pure
GEMM kernel 实例。

没有 canonical Volta warp shape 的组合不在 C++ 中实例化。sweep 脚本
会把它们记录为 `unsupported_geometry`，不会让整轮任务失败。

### `sweep_sm70_pure_gemm.py`

新增脚本：

```text
benchmarks/sweep_sm70_pure_gemm.py
```

默认行为：

1. 创建结果目录：

   ```text
   benchmarks/results/<timestamp>_sm70_pure_gemm_sweep/
   ```

2. 执行 `./build.sh`。

3. 默认删除 probe object 后再 build：

   ```text
   build/temp.*/CMakeFiles/_C.dir/csrc/quantization/marlin/sm70_cutlass_matmul_probe.cu.o
   ```

   这样可以强制重新编译 probe translation unit，使 `build.log` 中保留
   fresh `ptxas` stack/spill 输出。

4. 执行：

   ```bash
   /usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
     python/marlin_v100/_C.abi3.so | c++filt
   ```

5. 逐个 MKN 和 tile config benchmark。

6. 每个 repeat 完成后立即 append 一行 JSON 到 `raw_results.jsonl`。

7. 整轮完成后生成聚合 CSV、summary 和 per-MKN 排序表。

### `benchmark.sh`

新增 target：

```bash
./benchmark.sh pure-gemm-sweep
```

参数通过环境变量透传：

```bash
PURE_GEMM_SWEEP_ARGS="--repeats 5 --warmup-iters 10 --iters 30" \
  ./benchmark.sh pure-gemm-sweep
```

## Default Search Space

默认候选 tile 参数：

```text
CTA_M = 64, 128
CTA_N = 64, 128, 256
CTA_K = 32, 64, 128
Warps = 4, 8
```

其中只有 C++ 已实例化的 canonical 组合会实际运行。其他组合记录为：

```text
unsupported_geometry
```

默认 MKN 集合：

| Group | M | N | K |
|---|---:|---:|---:|
| base | 5120 | 4096 | 4096 |
| base | 4096 | 4096 | 4096 |
| M sweep | 2048 | 4096 | 4096 |
| M sweep | 1024 | 4096 | 4096 |
| M sweep | 512 | 4096 | 4096 |
| M sweep | 256 | 4096 | 4096 |
| M sweep | 128 | 4096 | 4096 |
| N sweep | 5120 | 8192 | 4096 |
| N sweep | 5120 | 2048 | 4096 |
| N sweep | 5120 | 1024 | 4096 |
| N sweep | 5120 | 512 | 4096 |
| N sweep | 5120 | 256 | 4096 |
| K sweep | 5120 | 4096 | 8192 |
| K sweep | 5120 | 4096 | 2048 |
| K sweep | 5120 | 4096 | 1024 |
| K sweep | 5120 | 4096 | 512 |
| K sweep | 5120 | 4096 | 256 |

这些默认 shape 都满足：

```text
M % 128 == 0
N % 256 == 0
K % 128 == 0
```

这样可以减少 tile 整除性对结果的干扰。

## Usage

### 完整默认 sweep

```bash
PURE_GEMM_SWEEP_ARGS="--repeats 5 --warmup-iters 10 --iters 30" \
  ./benchmark.sh pure-gemm-sweep
```

### 指定单个 MKN

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn 5120,4096,4096 \
  --repeats 5 \
  --warmup-iters 10 \
  --iters 30
```

### 指定多个 MKN

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn 5120,4096,4096 \
  --mkn 4096,4096,4096 \
  --repeats 5 \
  --warmup-iters 10 \
  --iters 30
```

### 从文件读取 MKN

`shapes.txt` 示例：

```text
5120,4096,4096
4096,4096,4096
# comment is allowed
5120,8192,4096
```

运行：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn-file shapes.txt
```

### 覆盖 tile 候选集合

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn 512,512,512 \
  --cta-m 64 128 \
  --cta-n 64 128 \
  --cta-k 32 64 128 \
  --warps 4 8 \
  --repeats 1 \
  --warmup-iters 2 \
  --iters 3
```

### 跳过 build

已经确认 `_C.abi3.so` 是最新时，可以跳过 build：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --skip-build \
  --mkn 512,512,512 \
  --cta-m 64 \
  --cta-n 64 \
  --cta-k 32 \
  --warps 4 \
  --repeats 1 \
  --warmup-iters 1 \
  --iters 1
```

注意：`--skip-build` 会导致 `build.log` 中没有 fresh `ptxas` 输出，因此
spill 信息只能来自已有 build log 或保持默认值。正式资源 sweep 推荐不使用
`--skip-build`。

### Resume

如果长时间 sweep 被打断，可以复用已有结果目录：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --resume \
  --output-dir benchmarks/results/20260514_XXXXXX_sm70_pure_gemm_sweep
```

如果只传 `--resume` 而不传 `--output-dir`，脚本会选择最新的
`*_sm70_pure_gemm_sweep` 目录。

resume 逻辑会读取已有 `raw_results.jsonl`，跳过已经完成的 ok repeat 和
terminal failure/unsupported config。

## Output Files

每次运行都会生成：

```text
build.log
resource_usage.txt
raw_results.jsonl
all_results.csv
summary.md
M*_N*_K*_tflops.md
M*_N*_K*_resource.md
```

### `raw_results.jsonl`

每个 repeat 完成后立即写入一行 JSON。这样即使任务中途被杀，也能保留已经完成
的测量结果。

status 可能为：

| Status | Meaning |
|---|---|
| `ok` | 单个 repeat 正常完成 |
| `unsupported_geometry` | 当前 tile 组合没有 C++ 模板实例 |
| `invalid_shape` | M/N/K 不能被 CTA_M/CTA_N/CTA_K 整除 |
| `failure` | kernel correctness 或 runtime launch 失败 |

### `all_results.csv`

聚合后的完整表，固定列为：

```text
M,N,K,CTA_M,CTA_N,CTA_K,Warps,status,avg_us,avg_tflops,repeats,REG,STACK,LOCAL,SHARED,spill_stores,spill_loads,max_abs_diff,notes
```

`avg_us` 是所有 successful repeat 的 repeat median latency 平均值。

`avg_tflops` 按下面公式由 `avg_us` 反算：

```text
avg_tflops = 2 * M * N * K / avg_us / 1e6
```

### Per-MKN TFLOPs 排序

文件名：

```text
M<M>_N<N>_K<K>_tflops.md
```

排序规则：

```text
avg_tflops desc, avg_us asc, REG asc
```

### Per-MKN 资源优先排序

文件名：

```text
M<M>_N<N>_K<K>_resource.md
```

排序规则：

```text
REG asc,
avg_tflops desc,
spill_stores asc,
spill_loads asc,
STACK asc,
LOCAL asc
```

这个排序用于寻找“寄存器更低但性能仍然接近”的候选，方便后续量化 IteratorB
设计时预留更多寄存器预算。

## Resource Parsing

脚本同时使用两类资源来源。

第一类是 `cuobjdump --dump-resource-usage`：

```text
REG
STACK
LOCAL
SHARED
```

第二类是 build log 中的 `ptxas` 输出：

```text
bytes stack frame
bytes spill stores
bytes spill loads
```

两者按 kernel name：

```text
sm70_cutlass_threadblock_gemm_kernel<CTA_M, CTA_N, CTA_K, Warps>
```

解析并合并。

## Smoke Validation

构建和导入：

```bash
./build.sh
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C; print('imports ok')"
```

结果：

```text
imports ok
```

probe 既有测试：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_matches_torch_mm \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_threadblock_path_matches_torch_mm \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_rejects_direct_a_path \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_rejects_non_pure_b_path
```

结果：

```text
4 passed in 1.46s
```

脚本 smoke：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn 512,512,512 \
  --cta-m 64 128 \
  --cta-n 64 128 \
  --cta-k 32 \
  --warps 4 8 \
  --repeats 1 \
  --warmup-iters 2 \
  --iters 3
```

输出目录：

```text
benchmarks/results/20260514_005644_sm70_pure_gemm_sweep/
```

结果摘要：

| Status | Count |
|---|---:|
| `ok` | 7 |
| `unsupported_geometry` | 1 |

该 smoke 证明：

- `raw_results.jsonl` 会逐 repeat 追加。
- `all_results.csv` 固定列完整。
- `summary.md` 正常生成。
- per-MKN 的 TFLOPs 排序表正常生成。
- per-MKN 的资源优先排序表正常生成。
- unsupported geometry 不会中断整轮 sweep。

`benchmark.sh` target smoke：

```bash
PURE_GEMM_SWEEP_ARGS="--skip-build --mkn 512,512,512 --cta-m 64 --cta-n 64 --cta-k 32 --warps 4 --repeats 1 --warmup-iters 1 --iters 1" \
  ./benchmark.sh pure-gemm-sweep
```

输出目录：

```text
benchmarks/results/20260514_011032_sm70_pure_gemm_sweep/
```

结果：

```text
status=ok
```

新增 `CTA_K=64/128` smoke：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --skip-build \
  --mkn 512,512,512 \
  --cta-m 128 \
  --cta-n 256 \
  --cta-k 64 128 \
  --warps 8 \
  --repeats 1 \
  --warmup-iters 1 \
  --iters 1
```

结果：

| CTA | Status | REG | STACK | Notes |
|---|---|---:|---:|---|
| `128x256x64/8` | `ok` | 248 | 0 | 正常运行 |
| `128x256x128/8` | `failure` | 255 | 128 | runtime `invalid argument`，脚本记录失败并继续 |

这说明部分高 K tile 虽然能编译并提取资源，但可能因为资源或 launch 约束在运行时失败。
这类失败会保留在结果中，方便后续人工筛选，不会阻塞长时间 sweep。

## Known Notes

- 当前工具只覆盖 extracted CUTLASS threadblock pure GEMM 默认路径。
- 旧 CuTe path 仍保留在 probe 中，但本脚本不会默认 sweep 它。
- `CTA_K=128` 的某些大 tile 会出现 stack/spill 或 runtime failure，这是 sweep
  要捕获的结果，不视为脚本失败。
- `--skip-build` 适合快速 smoke，不适合正式资源采集。
- 当前 full sweep 尚未在本文档中记录最终大表；正式长跑后应直接查看对应
  `benchmarks/results/<timestamp>_sm70_pure_gemm_sweep/summary.md` 和 per-MKN
  排序表。

## Suggested Full Run

推荐正式长跑命令：

```bash
PURE_GEMM_SWEEP_ARGS="--repeats 5 --warmup-iters 10 --iters 30" \
  ./benchmark.sh pure-gemm-sweep
```

长跑完成后，优先看：

```text
summary.md
M5120_N4096_K4096_tflops.md
M5120_N4096_K4096_resource.md
```

然后再看 M/N/K sweep 维度下每个 shape 的排序表。

