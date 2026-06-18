# 2026-05-11 SM70 TensorOp 8x8x4 Matmul Probe

## 改动内容

- 新增私有 CUDA op：
  - `sm70_cutlass_matmul_probe(Tensor a, Tensor b, int cta_m, int cta_n, int cta_k, int warps, int stages, int a_path, int b_path) -> Tensor`
- 接入两条 SM70 half GEMM probe 路径：
  - `a_path=0`: CUTLASS 3 CuTe 实验路径，A/B 都走手写 swizzled shared memory，保留用于后续 layout 实验对照。
  - `a_path=2`: 从 `cutlass::gemm::device::Gemm` 内部拆出的 SM70 threadblock 路径，不调用 `device::Gemm`，而是在自有 kernel 中直接组合 `DefaultMma`、`MmaPipelined`、`DefaultEpilogueVoltaTensorOp`。
- `a_path=2` 当前支持的 tile 配置：
  - `64x64x32/4`
  - `64x128x32/4`
  - `64x256x32/8`
  - `128x64x32/4`
  - `128x128x32/4`
  - `128x256x32/8`
- `a_path=2` 使用 Volta TensorOp `8x8x4`、2-stage pipeline、fp32 accumulate、fp16 output。
- 代码中新增 `static_assert`，锁定 `DefaultMma` 实际选择的 SM70 predefined shared-memory layout，避免后续 Marlin dequant 路径退回普通 row-major 或手写错误 swizzle。
- 这版 probe 先验证的是 dense B 的 CUTLASS threadblock 行为，不是最终 quantized B 路径。后续结论已经更新：不要在 dense B proxy 上保留额外 B 转换路径，真正 Marlin 主路径应让 packed B 以尽量短的寄存器生命周期完成 unpack / dequant / scale / zp，再直接落到 CUTLASS predefined Volta shared-memory B layout。
- 新增 `benchmarks/benchmark_sm70_matmul_probe.py` 和 `./benchmark.sh matmul` target。
- 新增 pytest：
  - CuTe shared path 对齐 `torch.mm`
  - extracted CUTLASS threadblock path 对齐 `torch.mm`
  - direct-global A path 预期拒绝

## `docs/functionality.md` 提取结论

`docs/functionality.md` 对本项目真正有用的部分集中在 TensorOp 8-by-8-by-4。SM70 上要冲 V100 Tensor Core 峰值，应只围绕这一条路线做优化：

- opcode class 使用 `TensorOp`，不是 `WmmaTensorOp`。WMMA 接口更通用，但 layout 和 pipeline 控制更粗，不适合作为 Marlin dequant 主循环基础。
- instruction shape 固定为 `8x8x4`。CUTLASS 默认 SM70 half TensorOp 配置也是 `ThreadblockShape=128x256x32`、`WarpShape=64x64x32`、`InstructionShape=8x8x4`、`Stages=2`。
- warp shape 候选集中在 `32x32x4`、`32x64x4`、`64x32x4`、`64x64x4` 组成的 Volta TensorOp family；在 CUTLASS threadblock path 中表现为 `WarpShape` 的 M/N 取 `32/64`，K 通常扩展到 `32`。
- Volta TensorOp 必须从 permuted shared memory layout 读取 operand。这个 layout 不是普通 row-major/column-major，也不应该在最终主路径中临时手写 `cute::Swizzle`。

`functionality.md` 表格中的 8x8x4 layout 与 CUTLASS 代码中的实际类型名对应如下：

| Operand | GMEM Layout | SMEM Layout in doc | CUTLASS SM70 type used by code |
| --- | --- | --- | --- |
| A | ColumnMajor | `ColumnMajorVoltaTensorOpCongruous<16>` | `ColumnMajorVoltaTensorOpMultiplicandCongruous<16>` |
| A | RowMajor | `RowMajorVoltaTensorOpCrosswise<16>` | `RowMajorVoltaTensorOpMultiplicandCrosswise<16, KBlock>` |
| B | ColumnMajor | `ColumnMajorVoltaTensorOpCrosswise<16>` | `ColumnMajorVoltaTensorOpMultiplicandCrosswise<16, KBlock>` |
| B | RowMajor | `RowMajorVoltaTensorOpCongruous<16>` | `RowMajorVoltaTensorOpMultiplicandBCongruous<16>` |

当前 probe 输入 A/B 都是 contiguous row-major，因此 `a_path=2` 的真实 layout 是：

- A shared memory: `RowMajorVoltaTensorOpMultiplicandCrosswise<16, CTA_K>`
- B shared memory: `RowMajorVoltaTensorOpMultiplicandBCongruous<16>`

这正是后续 Marlin 的关键优化点：B 权重应先短暂进入寄存器完成 unpack / dequant / scale / zp，再由 `SmemIteratorB` 直接写入 `Mma::SharedStorage` 里预定义好的 Volta layout，避免先写普通 row-major half tile 再二次搬运或重排。
如果后续 `ColumnMajorVoltaTensorOpMultiplicandCrosswise<16, KBlock>` 在真实 dequant-to-shared 流水里更优，它也应该作为 `SmemIteratorB` 的落点来评估，而不是拿 dense B 的 global 读取方式下结论。

## 优化思路

- 主路径采用 extracted CUTLASS threadblock/warp 组件，而不是 `cutlass::gemm::device::Gemm` 黑盒；这样可以在 B operand 的 global iterator / shared store 位置插入量化权重 unpack、zero-point、bias、scale。
- B dequant 应替换 `MmaPipelined` 当前的 `IteratorB.load -> SmemIteratorB.store` 数据源：packed int 权重从 global 读到寄存器，解包成 half fragment，乘 scale / 减 zero-point，然后直接写进 predefined Volta B layout。
- A 路径先沿用 CUTLASS `IteratorA -> SmemIteratorA`，因为它已经使用 128-bit global load 与 predefined crosswise SMEM layout；后续只需要 benchmark 是否值得为 A 做更轻的直接 shared layout 写入。
- 继续优先 benchmark `128x256x32/8`。该配置对应 CUTLASS SM70 默认大 tile，已经在大尺寸上达到 90+ TFLOPs；中小 M/N 再保留 `128x128x32/4` 和 `128x64x32/4` 作为备选。
- K tile 暂时固定为 `32`。这和 CUTLASS SM70 默认配置一致，寄存器/occupancy/SMEM 压力当前最可控；`64/128` 可留给 CuTe 实验路径，不进入 dequant 主路径。
- 不支持格式继续显式拒绝。把优化预算集中在 `kU4`、`kU4B8`、`kU8B128`，避免 bf16/fp8/nvfp4/mxfp4/act-order 把主循环复杂度拉散。

## 测试命令与结果

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import marlin_v100, marlin_v100._C, marlin_v100._moe_C"
```

结果：通过。

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
```

结果：通过，收集 `229` 个测试。

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_matches_torch_mm \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_threadblock_path_matches_torch_mm \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_rejects_direct_a_path
```

结果：通过，`3 passed in 1.46s`。

## Benchmark

小/中尺寸：

```bash
BENCH_PRESET=quick MATMUL_ARGS="--m 1024 --n 4096 --k 512 --warmup-iters 5 --iters 30 --a-paths cutlass_threadblock --cta-m 64 128 --cta-n 64 128 256 --cta-k 32 --warps 4 8" ./benchmark.sh matmul
```

结果摘要：

| CTA | warps | median us | TFLOPs | notes |
| --- | --- | --- | --- | --- |
| `128x128x32` | 4 | `93.18` | `46.09` | best for this shape |
| `128x256x32` | 8 | `95.23` | `45.10` | close second |

大尺寸：

```bash
BENCH_PRESET=quick MATMUL_ARGS="--m 4096 --n 4096 --k 4096 --warmup-iters 3 --iters 10 --a-paths cutlass_threadblock --cta-m 128 --cta-n 128 256 --cta-k 32 --warps 4 8" ./benchmark.sh matmul
```

结果摘要：

| CTA | warps | median us | TFLOPs | notes |
| --- | --- | --- | --- | --- |
| `128x128x32` | 4 | `1649.15` | `83.34` | no spills |
| `128x256x32` | 8 | `1597.44` | `86.04` | best for this shape |

饱和尺寸，达到 90+ TFLOPs：

```bash
BENCH_PRESET=quick MATMUL_ARGS="--m 5120 --n 4096 --k 4096 --warmup-iters 3 --iters 20 --a-paths cutlass_threadblock --cta-m 128 --cta-n 256 --cta-k 32 --warps 8 --atol 0.5 --rtol 0.05" ./benchmark.sh matmul
```

结果：

| CTA | warps | median us | TFLOPs | V100 peak |
| --- | --- | --- | --- | --- |
| `128x256x32` | 8 | `1851.39` | `92.79` | `74.24%` |

备注：`K=4096` 且 fp32 accumulate 后输出 fp16，和 `torch.mm` 对比时 `max_abs` 约 `0.25`，benchmark 需要 `--atol 0.5 --rtol 0.05`。

补充结论：dense B proxy 上的非纯 B 代理路径已被放弃并删除。原因是它会带来更高的寄存器压力或破坏后续 packed/dequant B 的合并访存假设；后续优化只围绕纯 row-major B 输入、CUTLASS predefined Volta B-congruous shared-memory layout，以及 shared-memory 读写侧的 swizzle/布局策略展开。

## ptxas 信息

extracted threadblock path：

| kernel config | registers | spill stores | spill loads |
| --- | --- | --- | --- |
| `128x256x32/8` | `216` | `0` | `0` |
| `128x128x32/4` | `236` | `0` | `0` |
| `128x64x32/4` | `154` | `0` | `0` |
| `64x256x32/8` | `140` | `0` | `0` |
| `64x128x32/4` | `162` | `0` | `0` |
| `64x64x32/4` | `104` | `0` | `0` |

CuTe experimental path 仍会编译较多配置，部分大 K tile 配置存在 spill；它不是当前高性能主路径。

## 已知问题

- 当前只是私有 matmul probe，尚未接入正式 `marlin_gemm` 或 `moe_wna16_marlin_gemm`。
- `a_path=2` 仍然从 global 读取 dense fp16 B；下一步需要把 B iterator / transform / store 替换为 packed quant 权重 dequant 到 `SmemIteratorB` 的路径，并重新 benchmark register count / spill / TFLOPs。
- `a_path=0` 编译配置过多，后续可以在主路径稳定后裁剪，避免 build 膨胀。
- 当前要求 M/N/K 能被 CTA tile 整除。
- 当前 epilogue 是普通 `LinearCombination`，还没有接入 Marlin workspace/reduction/MoE routing。

## 下一步

- 抽出可复用的 SM70 threadblock GEMM core。
- 在不改变 public op schema 的前提下，先实现一个 B dequant-to-shared prototype：packed B 从 global 合并读到寄存器，寄存器中流式完成 unpack / dequant / scale / zp，然后以最短生命周期写入 `RowMajorVoltaTensorOpMultiplicandBCongruous<16>` 对应的 shared-memory 布局。
- 第一版优先实现 `kU4B8` 或 `kU8B128`，验证 packed B -> predefined Volta B-congruous SMEM layout -> MMA 的完整正确性。
- 每加一类 quant 格式就运行对应 pytest 和 `BENCH_PRESET=quick ./benchmark.sh dense`，如果 median latency 相比 probe 主循环出现超过 10% 异常退化，先定位再继续叠功能。
