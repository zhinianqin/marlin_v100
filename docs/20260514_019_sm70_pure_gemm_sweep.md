# SM70 Pure GEMM 全自动 Sweep 工具

## Summary

`benchmarks/sweep_sm70_pure_gemm.py` 用于无人值守地扫描 extracted
CUTLASS threadblock pure GEMM 的 tile 资源和性能。工具会自动构建、提取
`cuobjdump` 资源信息、解析 `ptxas` stack/spill、运行 benchmark，并为每个
MKN 输出两套排序表。

本工具是后续 SM70 Marlin 量化 IteratorB 的 pure GEMM 上限参考，不替代
dense quant kernel benchmark。默认只扫 extracted CUTLASS threadblock 路径，
不扫旧 CuTe 实验路径。

本轮策略固定为：

```text
CTA_K = 32
CTA_M = 32, 64, 128, 256, 512
CTA_N = 64, 128, 256, 512
Warps = 4, 8
```

`CTA_K` 仍保留在输出列中，但它是兼容旧结果表结构的固定字段，不再是搜索维度。

## Probe 实例范围

文件：

```text
csrc/quantization/marlin/sm70_cutlass_matmul_probe.cu
```

public op schema 保持不变：

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

extracted CUTLASS threadblock 路径继续使用：

- `cutlass::gemm::threadblock::DefaultMma`
- Volta TensorOp `8x8x4`
- CUTLASS predefined A/B shared-memory layout
- 2-stage pipeline
- `a_path = 2`
- `b_path = 0`

threadblock path 只实例化 `CTA_K=32`，并且只实例化 per-warp M/N 都不超过
64、同时满足 CUTLASS thread map 每线程至少一次 128-bit access 的 canonical
shape。这样可以避免大 warp tile 或过小 A/B tile 带来不可控的编译时间、寄存器
占用和 spill。

当前支持的 threadblock shape 为：

| CTA_M | CTA_N | Warps | Warp_M | Warp_N |
|---:|---:|---:|---:|---:|
| 32 | 128 | 4 | 32 | 32 |
| 32 | 256 | 4 | 32 | 64 |
| 64 | 64 | 4 | 32 | 32 |
| 64 | 128 | 4 | 32 | 64 |
| 64 | 128 | 8 | 32 | 32 |
| 64 | 256 | 4 | 64 | 64 |
| 64 | 256 | 8 | 32 | 64 |
| 64 | 512 | 8 | 64 | 64 |
| 128 | 64 | 4 | 64 | 32 |
| 128 | 64 | 8 | 32 | 32 |
| 128 | 128 | 4 | 64 | 64 |
| 128 | 128 | 8 | 64 | 32 |
| 128 | 256 | 8 | 64 | 64 |
| 256 | 64 | 4 | 64 | 64 |
| 256 | 64 | 8 | 64 | 32 |
| 256 | 128 | 8 | 64 | 64 |
| 512 | 64 | 8 | 64 | 64 |

其他 `CTA_M/CTA_N/Warps` 候选会保留在 sweep 结果中并记录为
`unsupported_geometry`，不视为整轮失败。例如 `CTA_M=32` 配 8 warp 时，
A tile `32x32` 的 128-bit access 数少于 256 个线程，CUTLASS
`PitchLinearWarpRakedThreadMap` 会触发 static assertion，因此不实例化。

旧 CuTe path 仍保留在 probe 中，但本轮不扩展 CuTe 的 `CTA_M=256/512`
或 `CTA_N=512` 实例。

## Sweep 脚本行为

脚本：

```text
benchmarks/sweep_sm70_pure_gemm.py
```

默认行为：

1. 创建结果目录：

   ```text
   benchmarks/results/<timestamp>_sm70_pure_gemm_sweep/
   ```

2. 默认执行 `./build.sh`。

3. 默认删除 probe object 后再 build：

   ```text
   build/temp.*/CMakeFiles/_C.dir/csrc/quantization/marlin/sm70_cutlass_matmul_probe.cu.o
   ```

   这样可以强制重新编译 probe translation unit，使 `build.log` 中保留 fresh
   `ptxas` stack/spill 输出。

4. 执行：

   ```bash
   /usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
     python/marlin_v100/_C.abi3.so | c++filt
   ```

5. 逐个 MKN 和 tile config benchmark。

6. 每个 repeat 完成后立即 append 一行 JSON 到 `raw_results.jsonl`。

7. 整轮完成后生成 `all_results.csv`、`summary.md` 和 per-MKN 排序表。

`--cta-k` 是兼容参数，只接受 `32`。传入 `64` 或 `128` 会在脚本启动阶段直接报错：

```text
pure GEMM sweep only supports CTA_K=32
```

`--resume` 读取旧 `raw_results.jsonl` 时会忽略 `CTA_K != 32` 的历史记录，避免旧结果污染新的 summary。

## 默认 MKN

默认 MKN 由 dense benchmark 的模型形状派生，并额外补充 N/K sweep。

主集合来自：

```text
benchmarks/benchmark_shapes.py
DENSE_PRESETS["full"]
DENSE_WEIGHT_SHAPES
```

M 维使用：

```text
32, 64, 128, 256, 512, 1024, 2048, 4096, 5120
```

补充 N sweep：

```text
M = 5120
K = 4096
N = 64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 22016
```

补充 K sweep：

```text
M = 5120
N = 4096
K = 512, 1024, 2048, 4096, 8192, 11008
```

所有 shape 会 `sorted(set(...))` 去重。用户显式传入 `--mkn` 或
`--mkn-file` 时，脚本只使用显式 shape。

## 输出文件

每次运行生成：

```text
build.log
resource_usage.txt
raw_results.jsonl
all_results.csv
summary.md
M*_N*_K*_tflops.md
M*_N*_K*_resource.md
```

`raw_results.jsonl` 每个 repeat 立即写入，长任务中断后也能保留已有结果。

status 可能为：

| Status | Meaning |
|---|---|
| `ok` | 单个 repeat 正常完成 |
| `unsupported_geometry` | 当前 tile 组合没有 C++ 模板实例 |
| `invalid_shape` | M/N/K 不能被 CTA_M/CTA_N/CTA_K 整除 |
| `failure` | kernel correctness 或 runtime launch 失败 |

`all_results.csv` 固定列为：

```text
M,N,K,CTA_M,CTA_N,CTA_K,Warps,status,avg_us,avg_tflops,repeats,REG,STACK,LOCAL,SHARED,spill_stores,spill_loads,max_abs_diff,notes
```

`avg_us` 是 successful repeat 的 repeat median latency 平均值。

`avg_tflops` 按下面公式反算：

```text
avg_tflops = 2 * M * N * K / avg_us / 1e6
```

每个 MKN 会输出两套排序表：

- `M<M>_N<N>_K<K>_tflops.md`：
  `avg_tflops desc, avg_us asc, REG asc`
- `M<M>_N<N>_K<K>_resource.md`：
  `REG asc, avg_tflops desc, spill_stores asc, spill_loads asc, STACK asc, LOCAL asc`

资源优先排序用于寻找“寄存器更低但性能仍接近”的候选，方便量化 IteratorB
设计时预留寄存器预算。

## 使用方式

完整默认 sweep：

```bash
PURE_GEMM_SWEEP_ARGS="--repeats 5 --warmup-iters 10 --iters 30" \
  ./benchmark.sh pure-gemm-sweep
```

指定单个 MKN：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn 5120,4096,4096 \
  --repeats 5 \
  --warmup-iters 10 \
  --iters 30
```

覆盖 tile 候选集合：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn 512,512,512 \
  --cta-m 32 64 128 256 512 \
  --cta-n 64 128 256 512 \
  --warps 4 8 \
  --repeats 1 \
  --warmup-iters 1 \
  --iters 1
```

从文件读取 MKN：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn-file shapes.txt
```

跳过 build 只适合快速 smoke：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --skip-build \
  --mkn 512,512,512 \
  --cta-m 64 \
  --cta-n 64 \
  --warps 4 \
  --repeats 1 \
  --warmup-iters 1 \
  --iters 1
```

注意：`--skip-build` 会导致 `build.log` 中没有 fresh `ptxas` 输出，正式资源
sweep 推荐不使用。

Resume：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --resume \
  --output-dir benchmarks/results/20260514_XXXXXX_sm70_pure_gemm_sweep
```

如果只传 `--resume` 而不传 `--output-dir`，脚本会选择最新的
`*_sm70_pure_gemm_sweep` 目录。

## 验证命令

构建和导入：

```bash
./build.sh
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C; print('imports ok')"
```

资源检查：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_cutlass_threadblock_gemm_kernel"
```

预期 threadblock pure GEMM kernel name 只出现：

```text
sm70_cutlass_threadblock_gemm_kernel<*, *, 32, *>
```

probe 正确性：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_matches_torch_mm \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_threadblock_path_matches_torch_mm \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_extended_threadblock_shapes_match_torch_mm \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_rejects_unsupported_threadblock_shape \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_rejects_direct_a_path \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_rejects_non_pure_b_path
```

sweep smoke：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn 512,512,512 \
  --cta-m 32 64 128 256 512 \
  --cta-n 64 128 256 512 \
  --warps 4 8 \
  --repeats 1 \
  --warmup-iters 1 \
  --iters 1
```

CTA_K 参数校验：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python benchmarks/sweep_sm70_pure_gemm.py \
  --mkn 512,512,512 --cta-k 64
```

预期直接报错，不进入 build 或 benchmark。

## Notes

- 当前工具只覆盖 extracted CUTLASS threadblock pure GEMM 默认路径。
- 旧 CuTe path 仍保留在 probe 中，但本脚本不会默认 sweep 它。
- 非 canonical geometry 会记录为 `unsupported_geometry`，这是预期结果。
- `CTA_M=32/64` 形状用于观察 decode-ish small M 行为；真正的 M=1/16 decode
  latency 不适合当前 full-tile pure GEMM probe，需要后续专门的小 M/residue benchmark。
- 长跑完成后优先查看 `summary.md`、`M5120_N4096_K4096_tflops.md` 和
  `M5120_N4096_K4096_resource.md`。
