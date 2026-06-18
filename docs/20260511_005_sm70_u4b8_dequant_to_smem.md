# 2026-05-11 SM70 kU4B8 Dequant-To-SMEM Prototype

## 改动内容

- 新增 dense `kU4B8` 原型 kernel：`csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu`。
- 不调用 `cutlass::gemm::device::Gemm`，而是复用 CUTLASS SM70 threadblock 组件：
  - `DefaultMmaCore`
  - `MmaPipelined`
  - Volta TensorOp `8x8x4`
  - `DefaultEpilogueVoltaTensorOp`
- kernel 主配置为 `CTA=128x128x32`、`Warp=64x64x32`、`warps=4`、`stage=2`，fp32 accumulate，fp16 output。
- A 继续使用 CUTLASS predicated global iterator 写入 shared memory；B 使用自定义 iterator：
  - packed int4 global load
  - register unpack/dequant
  - fp16 scale multiply
  - 写入 CUTLASS `SmemIteratorB`
  - 最终 MMA B 来源仍是 shared memory
- B shared-memory layout 通过 `static_assert` 固定为 `RowMajorVoltaTensorOpMultiplicandBCongruous<16>`。
- dense `marlin_gemm` host 侧新增 SM70 `kU4B8` 分发，支持 `group_size=-1,32,64,128`，并显式拒绝当前原型未支持的功能：
  - 非 `kU4B8`
  - zero-point
  - act-order
  - bias
  - global scale
  - atomic-add
  - 非 fp16 A/C/scale
- `awq_marlin_repack` 和 `gptq_marlin_repack` 的 int4 non-A8 输出改为 SM70 原生 tile 布局：
  - shape 仍为 `{size_k / 16, size_n * 16 / 8}`
  - 每个 `16x64` tile 有 `128` 个 `uint32`
  - `word_offset = local_k * 8 + local_n_vec`
  - 每个 word 表示同一 K 行的 8 个连续 N 列
  - nibble pack order 为 `{0,2,4,6,1,3,5,7}`
- scale 布局改为 row-major `[num_groups, size_n]` contiguous，Python helper/reference 同步更新。
- 当前 dense 支持矩阵临时缩窄到 `uint4b8`，`uint4` zero-point 与 `uint8b128` 保留后续 TODO。

## 关键实现结论

- `TransformB` 和 B crosswise proxy 路径不适合真实 Marlin dequant，因为它们要么增加寄存器压力，要么破坏 packed qweight 的合并访存形态；本次实现维持 packed B global -> register dequant -> predefined B-congruous SMEM 的纯路径。
- `128x256x32/8` 是 pure fp16 B matmul 的最佳形态，但加入 dequant 后寄存器/指令压力过大；`128x128x32/4` 在 kU4B8 原型中表现最好。
- qweight offset 预计算数组会把寄存器从动态 offset 版本的约 `247` 提升到 `250/252`，但无 spill，并把 kernel-like benchmark 从约 `74.04 TFLOPs` 提升到本轮 `76.14 TFLOPs`，因此保留。
- 对 scale 做 `uint4` 风格向量化加载的实验正确性通过，但性能回退到约 `62.91 TFLOPs` kernel-like，已回退。
- 8-warp 的 `128x128` 变体正确性通过，但 kernel-like 只有约 `51 TFLOPs`，已回退。
- 当前实现超过了原型阶段 `70 TFLOPs` 暂停线，但仍低于 pure GEMM `91.98 TFLOPs` 基线，下一轮优化应继续围绕 `IteratorB::load()` 的指令数、scale load 和 qweight address arithmetic。

## 构建与导入

```bash
./build.sh
```

结果：通过，`python/marlin_v100/_C.abi3.so` 与 `python/marlin_v100/_moe_C.abi3.so` 已生成。

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：通过，输出 `imports ok`。

## 测试结果

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
```

结果：通过，收集 `198` 个测试。

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q tests/test_marlin_dense.py
```

结果：通过，`32 passed in 2.49s`。

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_calibration.py \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference
```

结果：通过，`6 passed in 1.45s`。

```bash
./test.sh \
  tests/test_marlin_dense.py \
  tests/test_calibration.py \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference
```

结果：通过，`38 passed in 2.51s`。

## Benchmark 结果

pure CUTLASS SM70 threadblock GEMM baseline：

```bash
MATMUL_ARGS="--m 5120 --n 4096 --k 4096 --cta-m 128 --cta-n 256 --cta-k 32 --warps 8 --a-paths cutlass_threadblock --rtol 5e-1 --atol 5e-1 --warmup-iters 10 --iters 30" BENCH_PRESET=quick ./benchmark.sh matmul
```

| path | MKN | median us | TFLOPs | V100 peak |
| --- | --- | --- | --- | --- |
| pure B shared-memory matmul | `5120x4096x4096` | `1867.78` | `91.98` | `73.58%` |

dense kU4B8 dequant-to-SMEM：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 --group-sizes 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" BENCH_PRESET=quick ./benchmark.sh dense
```

| metric | MKN | torch us | marlin us | torch TFLOPs | marlin TFLOPs | vs pure GEMM |
| --- | --- | --- | --- | --- | --- | --- |
| operator_us | `5120x4096x4096` | `1981.95` | `2292.74` | `86.68` | `74.93` | `81.46%` |
| kernel_like_us | `5120x4096x4096` | `1983.49` | `2256.38` | `86.61` | `76.14` | `82.78%` |

结论：原型达到 `76.14 TFLOPs` kernel-like，明显高于 `70 TFLOPs` 暂停线，但距离用户要求的 `90+ TFLOPs` 仍有约 `15.84 TFLOPs` 缺口。下一步不应扩展 MoE 或其他格式，应继续优化单格式 dense kU4B8。

## ptxas / Resource Usage

命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage python/marlin_v100/_C.abi3.so | c++filt | rg -A4 -B1 "sm70_marlin_u4b8_gemm_kernel"
```

| group_size specialization | registers | stack | local | spill |
| --- | --- | --- | --- | --- |
| `128` | `250` | `0` | `0` | `0` |
| `64` | `250` | `0` | `0` | `0` |
| `32` | `252` | `0` | `0` | `0` |
| `-1` | `247` | `0` | `0` | `0` |

说明：`./build.sh` 中旧 dense/MoE Marlin 模板仍会打印大量 255 registers 与 spill 信息，那些不是本次新增的 `sm70_marlin_u4b8_gemm_kernel`。

## 已知限制

- 只实现 dense `kU4B8`；MoE、`kU4` zero-point、`kU8B128` 后续再做。
- 只支持 fp16 A/C/scale。
- 不支持 act-order、bias、global scale、float zero-point、atomic-add。
- `size_k` 要求 `32` 对齐，`size_n` 要求 `64` 对齐。
- scale 当前是 row-major `[num_groups, size_n]` contiguous 内部 ABI。
- 当前性能仍低于 pure GEMM baseline，且还没有达到 `90+ TFLOPs` 目标。

## 下一步

- 继续优化 `IteratorB::load()`：
  - 检查 qweight global load 是否能稳定按 warp 合并为连续 `uint32`
  - 降低 qweight offset 预计算带来的寄存器占用
  - 尝试更短生命周期的 qword、dequant half2、scale cache
  - 用 SASS 检查 dequant/scale multiply 指令数
- 对 `128x128x32/4` 保持主线，同时只做小范围实验，不扩展接口面。
- 若能把 dense kU4B8 拉近到 pure GEMM 的 `90+ TFLOPs`，再补 `kU4` zero-point 与 `kU8B128`，最后进入 MoE 外层复用。
