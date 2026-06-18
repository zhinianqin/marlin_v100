# 2026-05-12 SM70 int4 qweight Macro-N 与 IteratorB 压缩结果

## 本轮目标

本轮继续优化 dense int4 路径，重点是 `uint4b8` 的 B 侧热路径。
统一 `128x256x32 / 8 warp` 之后，`uint4b8 group_size=-1` 仍在约
`79-80 TFLOPs`，离 pure GEMM baseline 约 `91.98 TFLOPs` 还有明显差距。

本轮按两阶段推进：

1. 先压缩当前 qweight layout 下的 `Sm70U4B8IteratorB` 指针模型和临时变量生命周期。
2. 如果仍低于 `85 TFLOPs`，再改变 int4 qweight 的物理顺序，把同一个
   `128x256x32` CTA 内 4 个 `N64` subtile 的同一个 `local_word` 连续放置，
   让 full-tile 热路径能用 `LDG.E.128` 一次读取 4 个 qword。

公共接口保持不变：

- 不修改 Torch op schema。
- 不修改 Python wrapper 参数。
- 不修改 `b_scales` shape/语义。
- 不修改 `b_zeros` shape/语义。
- `gptq_marlin_repack` / `awq_marlin_repack` 返回 tensor shape 不变。

## Stage 1: IteratorB 指针模型压缩

改动文件：

- `csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu`

主要变化：

- 用单个 `qweight_base_offset_` 替代 `qweight_offsets_[4]`。
- `operator++()` 只更新一个 base offset，不再更新 4 个数组元素。
- `group_size=-1` 的 scale cache 在 constructor 中一次性填充，`load()` 不再每个
  K tile 做 `cached_group_` 判断。
- `group_size=32/64/128` 保持 direct scale load，但把 `logical_k/group/scale_row`
  提升到 `c` loop 外。
- full/residue 热路径使用单个 `half2 deq[2]` 复用缓冲：

```text
qword -> low dequant -> 立即写 frag[0..1]
      -> high dequant -> 立即写 frag[2..3]
```

### Stage 1 资源

命令：

```bash
./build.sh
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4b8_gemm_kernel"
```

结果：

| specialization | REG | STACK | LOCAL | spill |
| --- | ---: | ---: | ---: | ---: |
| `<128, false>` | 238 | 0 | 0 | 0 |
| `<64, false>` | 238 | 0 | 0 | 0 |
| `<32, false>` | 238 | 0 | 0 | 0 |
| `<-1, false>` | 238 | 0 | 0 | 0 |
| `<128, true>` | 240 | 0 | 0 | 0 |
| `<64, true>` | 240 | 0 | 0 | 0 |
| `<32, true>` | 240 | 0 | 0 | 0 |
| `<-1, true>` | 235 | 0 | 0 | 0 |

所有实例仍为 `STACK=0`、`LOCAL=0`，无 spill。

### Stage 1 正确性

命令：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"

PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_calibration.py \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_helpers.py::test_uint4b8_act_order_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_n_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_k_single_group_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_accuracy
```

结果：

```text
imports ok
38 passed in 2.96s
```

### Stage 1 benchmark

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260512_221546_dense_quick.log
```

`kernel_like` 结果：

| quant | group_size | kernel_like TFLOPs |
| --- | ---: | ---: |
| `uint4b8` | `-1` | 78.62 |
| `uint4b8` | `32` | 79.06 |
| `uint4b8` | `64` | 79.29 |
| `uint4b8` | `128` | 79.97 |

结论：Stage 1 资源更轻，但 `group_size=-1` 仍低于 `85 TFLOPs`，继续 Stage 2。

## Stage 2: int4 qweight Macro-N 物理顺序

改动文件：

- `csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu`
- `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu`
- `csrc/quantization/marlin/gptq_marlin_repack.cu`
- `csrc/quantization/marlin/awq_marlin_repack.cu`
- `python/marlin_v100/quant_utils.py`
- `tests/helpers.py`

### 地址公式

旧 int4 物理顺序：

```text
[k_tile][n64_tile][local_k][local_n_vec]
```

其中：

```text
tile_words = 16 * 64 / 8 = 128
old_offset = k_tile * n_tiles * tile_words
           + n_tile * tile_words
           + local_k * 8
           + local_n_vec
```

新 int4 Macro-N 物理顺序：

```text
[k_tile][n256_macro][local_word][n64_subtile]
```

其中：

```text
macro_n_tile = n_tile / 4
macro_first_n_tile = macro_n_tile * 4
subtile = n_tile - macro_first_n_tile
subtile_count = min(4, n_tiles - macro_first_n_tile)
local_word = local_k * 8 + local_n_vec

new_offset = k_tile * n_tiles * tile_words
           + macro_n_tile * 4 * tile_words
           + local_word * subtile_count
           + subtile
```

对于 full-tile `size_n % 256 == 0` 的热路径，`subtile_count == 4`，
同一 `local_word` 的 4 个 `N64` qword 变成连续地址：

```text
base + 0, base + 1, base + 2, base + 3
```

因此 `uint4b8` full-tile path 可以从：

```text
4 个 scalar qweight load
```

改成：

```text
1 个 uint4 / LDG.E.128 qweight load
```

### 实现细节

`uint4b8`：

- `qweight_offset_from_logical()` 切到 Macro-N 地址公式。
- `qweight_offset(c)` 变成 `qweight_base_offset_ + c`。
- `load_full_tile()` 对 4 个 `N64` subtile 使用一次 `uint4` 读取：

```cpp
uint4 const qwords =
    *reinterpret_cast<uint4 const*>(qweight_ + qweight_base_offset_);
```

- `load_residue_tile()` 仍走同一 Macro-N layout，但只对有效 subtile 做 scalar load，
  保持 `size_n % 256 != 0` 的 residue 正确性。

`uint4`：

- 本轮只同步适配 `qweight_offset_from_logical()`，避免 int4 repack 物理顺序改变后
  dense `uint4` 正确性回退。
- 没有把 `uint4` 强行改成 256N，也没有重构 zero-point iterator。

repack / reference：

- `gptq_marlin_repack.cu` 与 `awq_marlin_repack.cu` 在 int4 分支写入 Macro-N 顺序。
- `python/marlin_v100/quant_utils.py` 的 `marlin_weights()` 在 `num_bits == 4`
  时生成同一 Macro-N 顺序。
- `tests/helpers.py` 的 unpack/reference 读取同一 Macro-N 顺序。
- 8-bit 路径未修改。

## Stage 2 资源

命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4b8_gemm_kernel|sm70_marlin_u4_gemm_kernel"
```

`uint4b8`：

| specialization | REG | STACK | LOCAL | spill |
| --- | ---: | ---: | ---: | ---: |
| `<128, false>` | 238 | 0 | 0 | 0 |
| `<64, false>` | 238 | 0 | 0 | 0 |
| `<32, false>` | 238 | 0 | 0 | 0 |
| `<-1, false>` | 238 | 0 | 0 | 0 |
| `<128, true>` | 252 | 0 | 0 | 0 |
| `<64, true>` | 252 | 0 | 0 | 0 |
| `<32, true>` | 252 | 0 | 0 | 0 |
| `<-1, true>` | 238 | 0 | 0 | 0 |

`uint4`：

| specialization | REG | STACK | LOCAL | spill |
| --- | ---: | ---: | ---: | ---: |
| `<128, false>` | 255 | 0 | 0 | 0 |
| `<64, false>` | 255 | 0 | 0 | 0 |
| `<32, false>` | 255 | 0 | 0 | 0 |
| `<-1, false>` | 254 | 0 | 0 | 0 |
| `<128, true>` | 255 | 0 | 0 | 0 |
| `<64, true>` | 255 | 0 | 0 | 0 |
| `<32, true>` | 255 | 0 | 0 | 0 |
| `<-1, true>` | 246 | 0 | 0 | 0 |

结论：

- 所有实例仍为 `STACK=0`、`LOCAL=0`，无 spill。
- `uint4b8<-1,true>` 保持 `REG=238`。
- `uint4b8<32/64/128,true>` 因 `uint4` vector load 上升到 `REG=252`，但没有 spill，
  benchmark 没有出现超过规则阈值的回退。
- `uint4` 仍然接近寄存器上限，本轮只是 layout 适配，不把它作为性能优化完成态。

## SASS 粗计数

命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-sass \
  python/marlin_v100/_C.abi3.so > /tmp/marlin_v100_all.sass

awk 'BEGIN{flag=0}
  /Function : _ZN57_GLOBAL__N__c67e0c23_24_sm70_marlin_u4b8_gemm_cu_b13c4b4328sm70_marlin_u4b8_gemm_kernelILin1ELb1EEEvPKN7cutlass6half_tEPKjS4_PS2_iiii/{flag=1}
  flag && /^\t\tFunction :/ && !/sm70_marlin_u4b8_gemm_kernelILin1ELb1EEEvPKN7cutlass6half_tEPKjS4_PS2_iiii/ {flag=0}
  flag {print}' \
  /tmp/marlin_v100_all.sass > /tmp/u4b8_minus1_true.sass
```

`uint4b8<-1,true>` 当前粗计数：

| instruction | static count |
| --- | ---: |
| `LDG` | 22 |
| `IMAD` | 108 |
| `LOP3` | 81 |
| `SHF` | 46 |
| `HMUL2` | 32 |
| `HFMA2` | 16 |
| `HADD2` | 18 |
| `HMMA` | 512 |
| `STS` | 80 |
| `LDS` | 100 |
| `LDGSTS` | 0 |

和 `docs/20260512_012_sm70_sass_optimization_analysis.md` 中旧记录对比：

| metric | old `u4b8<-1,true>` | current `u4b8<-1,true>` |
| --- | ---: | ---: |
| `REG` | 247 | 238 |
| `LDG` | 24 | 22 |
| `IMAD` | 138 | 108 |
| `LOP3` | 91 | 81 |
| `SHF` | 49 | 46 |
| `HMUL2` | 32 | 32 |
| `HFMA2` | 16 | 16 |
| `HADD2` | 18 | 18 |
| `HMMA` | 512 | 512 |
| `STS` | 88 | 80 |

当前 SASS 中能看到 qweight macro load：

```text
LDG.E.128.SYS
```

结论：Macro-N 主要减少了 qweight address / scalar load 相关压力；dequant 数学本身的
`HMUL2/HFMA2/HADD2` 没有减少，所以仍然和 pure GEMM 有明显差距。

## 正确性

导入：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：

```text
imports ok
```

定向测试：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_calibration.py \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_helpers.py::test_uint4b8_act_order_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_n_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_k_single_group_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_accuracy
```

结果：

```text
38 passed in 3.21s
```

## Benchmark

设备：

```text
Tesla V100-SXM2-32GB, sm70
```

### uint4b8

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260512_223013_dense_quick.log
```

结果：

| group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| `-1` | 2122.75 | 80.93 | 2078.72 | 82.65 |
| `32` | 2128.38 | 80.72 | 2159.10 | 79.57 |
| `64` | 2187.26 | 78.55 | 2158.59 | 79.59 |
| `128` | 2180.61 | 78.78 | 2174.46 | 79.01 |

### uint4

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260512_223100_dense_quick.log
```

结果：

| group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| `-1` | 2199.55 | 78.11 | 2207.74 | 77.82 |
| `32` | 2497.02 | 68.80 | 2521.60 | 68.13 |
| `64` | 2400.77 | 71.56 | 2469.89 | 69.56 |
| `128` | 2382.85 | 72.10 | 2448.38 | 70.17 |

## 对比与保留判断

| 实现/阶段 | quant | group_size | kernel_like TFLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| unified wide path | `uint4b8` | `-1` | 79.74 | `docs/20260512_013_sm70_u4b8_unified_wide_path.md` |
| Stage 1 IteratorB 压缩 | `uint4b8` | `-1` | 78.62 | 资源更轻，但吞吐没有提升 |
| Stage 2 Macro-N qweight | `uint4b8` | `-1` | 82.65 | 相对 unified wide path 约 `+3.65%` |
| Stage 2 Macro-N qweight | `uint4b8` | `128` | 79.01 | positive group 没有超过 5% 回退 |
| Stage 2 Macro-N qweight | `uint4` | `-1` | 77.82 | 正确性适配，性能基本维持 |
| pure GEMM baseline | n/a | n/a | 约 91.98 | 仍有约 10 TFLOPs 差距 |

保留判断：

- Stage 1 的指针模型压缩保留，原因是代码更简单、`u4b8<-1,true>` 寄存器更低，
  且无 spill。
- Stage 2 的 Macro-N qweight layout 保留，原因是 `uint4b8 group_size=-1`
  从 unified wide path 的 `79.74` 提升到 `82.65`，超过 `3%` 保留阈值。
- 虽然没有达到 `85+ TFLOPs`，但 SASS 已确认 qweight load 收缩为 vectorized load，
  是一个明确的正向优化点。
- `uint4` 只做 layout 同步，positive group 仍是后续优化对象。

## 已知限制

- 本轮改变了 int4 qweight 的内部物理顺序；dense `uint4b8` 和 dense `uint4`
  已同步适配并通过定向测试。
- MoE 本轮没有同步适配新 qweight layout。如果 MoE 依赖同一个 repack 输出，
  后续必须单独修改 `moe_wna16_marlin_gemm` 的 int4 qweight 读取公式后再声明支持。
- `uint4b8<32/64/128,true>` 因 full-tile vector load 资源上升到 `REG=252`，
  目前无 spill 且 benchmark 可接受；如果后续 positive group 是主目标，可以尝试
  仅对 `GroupSize=-1` 使用 vectorized qweight load。
- `uint4` 路径仍接近寄存器上限，特别是 positive group 的 zero-point 相关临时变量
  生命周期仍偏长。
- 这轮没有恢复 `BColumnMajorCrosswise`、`ExplicitTransformB`，也没有使用
  `cutlass::gemm::device::Gemm`。
- 已知精度失败的 `skip_flop=true` / `cached_bias=-8*scale` 没有重新尝试。

## 下一步

- 对 `uint4b8` 继续冲击 `85-90 TFLOPs` 时，优先看 full-tile 内 qword unpack 到
  `STS` 的调度，减少 `LOP3/SHF` 和 scale load 的依赖链，而不是恢复多 path。
- 对 positive group 可以测试 `GroupSize=-1` 专属 `LDG.E.128`、positive group scalar
  contiguous load 的折中，观察是否能降低 `REG=252`。
- 对 `uint4` 应单独做 zero-point path 压缩：direct zero/direct scale/direct bias
  的组合仍值得测试，但不能使用已失败的 `cached_bias=-8*scale` 语义。
- 如果 dense Macro-N layout 保留到主线，下一轮需要优先同步 MoE int4 qweight
  读取公式，避免 repack 输出被 dense/MoE 按不同物理语义解释。
