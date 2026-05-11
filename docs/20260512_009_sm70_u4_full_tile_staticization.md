# 2026-05-12 SM70 kU4 full_tile 模板静态化结果

## 改动内容

本轮只做 dense `kU4` zero-point 路径的 `full_tile` 静态化实验，核心改动位于
`csrc/quantization/marlin/sm70_marlin_u4_gemm.cu`：

- 将 `Sm70U4ZpIteratorB` 改成 `template <..., bool FullTile_>`。
- 删除运行时成员 `full_tile_`。
- `load()` 内使用 `if constexpr (FullTile_)` 分裂 full-tile / residue 路径。
- `refresh_metadata_cache()` 也按 `FullTile_` 静态分裂，full-tile 情况直接走无 residue 的 metadata 读取路径。
- host 侧在 launch 前按实际 shape 选择：
  - `size_k % 32 == 0 && size_n % 128 == 0` 走 `FullTile=true`
  - 否则走 `FullTile=false`

这次不改：

- `qweight` 主布局
- zero-only pair repack
- `cached_bias = -zero * scale`
- `__hfma2(q, scale, bias)` 计算方式
- 公共 Torch op schema
- `b_zeros` 形状

## 资源结果

构建命令：

```bash
./build.sh
```

资源复核命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4_gemm_kernel"
```

`cuobjdump` 看到的 `sm70_marlin_u4_gemm_kernel` 实例如下：

| specialization | REG | STACK | LOCAL | 结论 |
| --- | ---: | ---: | ---: | --- |
| `<-1, false>` | 254 | 0 | 0 | residue 专门实例，无 spill |
| `<32, false>` | 255 | 0 | 0 | residue 专门实例，无 spill |
| `<64, false>` | 255 | 0 | 0 | residue 专门实例，无 spill |
| `<128, false>` | 255 | 0 | 0 | residue 专门实例，无 spill |
| `<-1, true>` | 246 | 0 | 0 | full-tile 专门实例，无 spill |
| `<32, true>` | 255 | 0 | 0 | full-tile 专门实例，无 spill |
| `<64, true>` | 255 | 0 | 0 | full-tile 专门实例，无 spill |
| `<128, true>` | 255 | 0 | 0 | full-tile 专门实例，无 spill |

这轮最关键的结果是：

- full-tile 和 residue 两条路径都已经做到 `STACK:0`
- `LOCAL:0`
- 没有 spill stores / spill loads

相比前一轮 4-warp 版本里 `group_size=32/64/128` 的 `stack 8 + spill 4/4`，这次模板静态化把编译器的压力明显压下去了。

## 验证结果

导入：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：通过，输出 `imports ok`。

pytest 收集：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
```

结果：通过，收集 `215` 个测试。

定向测试：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_calibration.py \
  tests/test_marlin_helpers.py::test_marlin_dequantize_uint4_zp_matches_quantize_helper_output \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference \
  tests/test_marlin_dense.py
```

结果：通过，`54 passed in 3.78s`。

## Benchmark 结果

设备：

```text
Tesla V100-SXM2-32GB, sm70
```

### 单点：full-tile 典型形状

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

结果：

| metric | torch_us | marlin_us | torch TFLOPs | marlin TFLOPs |
| --- | ---: | ---: | ---: | ---: |
| operator | 1992.19 | 2465.79 | 86.24 | 69.67 |
| kernel_like | 1971.20 | 2406.40 | 87.15 | 71.39 |

### Sweep：确认其它 group_size 没有回退

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

结果：

| group_size | operator TFLOPs | kernel_like TFLOPs |
| ---: | ---: | ---: |
| `-1` | 86.39 | 79.01 |
| `32` | 87.45 | 71.58 |
| `64` | 87.84 | 71.74 |
| `128` | 87.27 | 72.82 |

## 结论

这次把 `full_tile_` 提升为模板参数是有效的：

- full-tile / residue 两条路径都被编译器拆成独立实例
- `STACK` 直接降到 `0`
- `group_size=32/64/128` 不再有 spill
- `uint4` dense benchmark 从上一轮约 `60 TFLOPs kernel_like` 提升到约 `71-73 TFLOPs kernel_like`

但它还没有把 dense `kU4` 拉到 pure GEMM baseline `91.98 TFLOPs`，说明后续瓶颈仍然在 `load_full_tile()` 热路径里的寄存器生命周期、dequant 顺序和 scale/bias 的活跃范围，而不是 runtime `full_tile` 分支本身。

## 下一步

- 继续压 `load_full_tile()` 的临时寄存器生命周期。
- 如果要继续尝试布局优化，优先从一次性 repack 里减少后续 load 指令数，而不是再引入更宽的运行时分支。
- 暂不碰 `kU4B8`、MoE 和公共 schema。
