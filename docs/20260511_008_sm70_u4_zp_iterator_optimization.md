# SM70 kU4 Zero-Point IteratorB Repack Optimization

## 摘要

本轮继续优化 dense `kU4` integer zero-point 路径，重点放在
`Sm70U4ZpIteratorB::load()` 的热路径和一次性 repack 布局上。最终保留的
实现选择是：

- 回到 4-warp 高性能 tile：`CTA=128x128x32`，`Warp=64x64x32`，`kWarps=4`。
- `qweight` 主布局保持 16x64 tile 内 `local_k * 8 + local_n_vec` 的纯正路径。
- `b_zeros` 形状保持 `[num_groups, size_n / 8]`，但在每个 128-column CTA
  范围内做 zero word pair repack。
- `IteratorB` 预计算 `qweight_offsets_`，`operator++()` 只做固定 stride
  增量。
- 每次 `load()` 开头按当前 K group 刷新 metadata cache，避免内层循环重复
  刷新 zero/scale。
- metadata cache 直接保存 `scale` 与 `bias=-zero*scale`，主 qweight dequant
  后使用 `__hfma2(q, scale, bias)`。
- 添加 full-tile fast path，常规大 shape 跳过 residue 边界检查。

## Repack 布局结论

### 保留：zero-only pair repack

`MmaCore::IteratorThreadMapB` 当前满足：

- `Iterations::kContiguous == 2`
- `Delta::kContiguous == 64`
- `kElementsPerAccess == 8`

也就是说，同一个线程在一个 `load()` 中会读取两个 8-column zero word：

- `logical_n0`
- `logical_n0 + 64`

原始 `b_zeros` 在 128-column CTA 内的 16 个 word 是：

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
```

本轮改成物理顺序：

```text
0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15
```

这样线程读取 `logical_n0` 与 `logical_n0 + 64` 对应的 zero word 时可以通过
一次相邻 `uint2` load 完成。这个布局只改变 zero metadata 的物理解释，不改变
`b_zeros` 的 public shape，不引入展开后的 half zero/bias 中间结果。

### 丢弃：qweight pair repack

尝试过把 qweight 也按 128-column CTA 内的两个 64-column half 做 pair repack。
该方案虽然正确性通过，但 benchmark 回退：

- `uint4 group_size=128 kernel_like`: `3044.35 us`, `56.43 TFLOPs`
- zero-only pair 版本同类结果约为 `60-63 TFLOPs`

判断原因是 qweight 位于每个 K-step 的主热路径，pair repack 让每个线程的局部
访问更“顺手”，但破坏了 warp 级别更重要的全局内存合并/缓存访问形态，并增加
了寄存器生命周期压力。因此 qweight 主布局继续保持现有 16x64 native tile。

### 丢弃：skip_flop dequant fusion

尝试将 `kU4` qweight/qzero dequant 改成 `skip_flop=true`，并把修正折入 cached
bias。构建通过且资源未明显变化，但 dense `kU4` accuracy 失败，最大绝对误差
约 `2.5-4.1`，因此回退到当前稳定的：

```cpp
marlin::dequant<half2, vllm::kU4.id(), false>
```

## 资源信息

构建命令：

```bash
./build.sh
```

资源复核命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage python/marlin_v100/_C.abi3.so | c++filt | rg -A8 -B2 "sm70_marlin_u4_gemm_kernel|sm70_marlin_u4b8_gemm_kernel"
```

`kU4` 当前资源：

| kernel | registers | stack | ptxas spill stores | ptxas spill loads |
| --- | ---: | ---: | ---: | ---: |
| `sm70_marlin_u4_gemm_kernel<-1>` | 255 | 0 | 0 | 0 |
| `sm70_marlin_u4_gemm_kernel<32>` | 255 | 8 | 4 | 4 |
| `sm70_marlin_u4_gemm_kernel<64>` | 255 | 8 | 4 | 4 |
| `sm70_marlin_u4_gemm_kernel<128>` | 255 | 8 | 4 | 4 |

`kU4B8` 对照资源：

| kernel | registers | stack |
| --- | ---: | ---: |
| `sm70_marlin_u4b8_gemm_kernel<-1>` | 247 | 0 |
| `sm70_marlin_u4b8_gemm_kernel<32>` | 252 | 0 |
| `sm70_marlin_u4b8_gemm_kernel<64>` | 250 | 0 |
| `sm70_marlin_u4b8_gemm_kernel<128>` | 250 | 0 |

## 验证

导入：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：

```text
imports ok
```

pytest collect：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
```

结果：

```text
215 tests collected in 1.22s
```

targeted pytest：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_calibration.py \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_helpers.py::test_uint4b8_act_order_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_marlin_dequantize_uint4_zp_matches_quantize_helper_output \
  tests/test_marlin_dense.py
```

结果：

```text
58 passed in 3.79s
```

## Benchmark

单点 `uint4 group_size=128`：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

| metric | torch_us | marlin_us | torch TFLOPs | marlin TFLOPs |
| --- | ---: | ---: | ---: | ---: |
| operator | 1991.68 | 2961.92 | 86.26 | 58.00 |
| kernel_like | 1970.18 | 2860.54 | 87.20 | 60.06 |

`uint4` group sweep：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

| group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| -1 | 2545.15 | 67.50 | 2470.40 | 69.54 |
| 32 | 3044.86 | 56.42 | 2750.46 | 62.46 |
| 64 | 2878.98 | 59.67 | 2861.57 | 60.04 |
| 128 | 2771.46 | 61.99 | 2806.27 | 61.22 |

`uint4b8 group_size=128` 对照：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 --group-sizes 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

| metric | torch_us | marlin_us | torch TFLOPs | marlin TFLOPs |
| --- | ---: | ---: | ---: | ---: |
| operator | 1922.56 | 2273.28 | 89.36 | 75.57 |
| kernel_like | 1976.32 | 2236.42 | 86.93 | 76.82 |

## 结论

zero metadata pair repack 是当前最值得保留的 repack 布局优化：它不改变公共
接口和 tensor shape，不影响 qweight 主热路径的合并访存，同时减少 zero
metadata 的地址计算和 load 数量。

当前 dense `kU4` 最好结果仍只有约 `60-70 TFLOPs`，明显低于 pure GEMM baseline
约 `91.98 TFLOPs`，也低于 `kU4B8` 的 `76.82 TFLOPs`。剩余瓶颈主要在
`IteratorB` 内每个 qword 的 int4 dequant、zero-point bias 应用和 scale/bias
half2 生命周期。

下一步建议：

- 做 `Sm70U4ZpIteratorB::load()` 的 SASS 指令计数，区分 integer unpack、half2
  fma、metadata load 的比例。
- 继续尝试减少 `cached_scales_` 与 `cached_bias_` 的 live range，尤其是
  `cache_metadata_word()` 中 scale/zero/bias 的生成顺序。
- 谨慎探索 scale 的物理布局，但只有在不破坏外部 scale 语义或能通过一次性
  preprocess 明确接入时才落地。
- 暂不再改 qweight 主 repack，除非 SASS/内存事务数据显示现有 qweight 访问不是
  合并访存瓶颈。
