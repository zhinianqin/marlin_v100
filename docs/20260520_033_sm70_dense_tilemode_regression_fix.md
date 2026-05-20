# SM70 dense TileMode 删除后的性能衰退修复

## 结论

`988b62b` 的性能衰退不是来自删除 residue 运行语义本身，而是删除
`TileMode` 时把原 FullTile 的编译期常量 qweight layout 路径改成了运行时
compact macro-N 路径。

旧 FullTile 路径中，8-bit qweight 的 macro-N word stride 等价于编译期常量
`kMacroNTiles == 4`。`988b62b` 后统一使用 compact 计算：

```cpp
subtile_count = min(size_n / 64 - macro_first_n_tile, 4)
```

对本次问题用例 `uint8 + CTA=128x256x8 + N=4096`，这个值运行时恒等于
`4`，但编译器不能稳定证明，因此 hot path 多出整数计算并增加寄存器压力。
ptxas 现象与此吻合：`uint8 128x256x8` positive group kernel 从 FullTile
baseline 的 `254 regs, 0 spill` 变成 `988b62b` 后的 `255 regs, 0 spill`。

本次修复恢复 full-tile 专用编译期常量 helper，不恢复 residue N/K 的运行
支持，也不重新引入 `Sm70TileMode` / `TileMode_` / `kResidue*` /
`load_full_tile` / `load_residue_tile`。

## 环境

| 项目 | 值 |
|---|---|
| 分支 | `work/sm70-dense-tilemode-removal` |
| 修复前 HEAD | `988b62b Remove SM70 dense TileMode residue paths` |
| GPU | `Tesla V100-SXM2-32GB` |
| Driver | `575.57.08` |
| CUDA_HOME | `/usr/local/cuda-12.8` |
| PyTorch | `2.10.0+cu128` |
| Torch CUDA | `12.8` |
| Build arch | `TORCH_CUDA_ARCH_LIST=7.0`, `sm_70` |

## 代码改动

核心改动在 `csrc/quantization/marlin/sm70_dense_iterator_utils.cuh`：

- 新增 `u4_full_tile_qweight_offset_from_logical(...)`
- 新增 `u8_full_tile_qweight_offset_from_logical(...)`
- 新增 `u8_full_tile_qweight_word_stride_from_logical(...)`
- full-tile offset 中固定使用 `local_word * kMacroNTiles`
- 8-bit word stride helper 直接返回 `kMacroNTiles`

所有 SM70 dense iterator 已切到这些 full-tile helper：

| quant | qweight helper |
|---|---|
| `uint4` | `u4_full_tile_qweight_offset_from_logical` |
| `uint4b8` | `u4_full_tile_qweight_offset_from_logical` |
| `nvfp4` | `u4_full_tile_qweight_offset_from_logical` |
| `mxfp4` | `u4_full_tile_qweight_offset_from_logical` |
| `uint8` | `u8_full_tile_qweight_offset_from_logical` + constant stride |
| `uint8b128` | `u8_full_tile_qweight_offset_from_logical` + constant stride |
| `fp8` | `u8_full_tile_qweight_offset_from_logical` + constant stride |

同时删除 residue K 删除后仍残留在 iterator 状态里的字段：

- `tile_k_end_`
- `next_k_advance_`
- iterator `Params::size_k`

kernel 构造从 `IteratorB::Params(k, n)` 收缩为 `IteratorB::Params(n)`。
入口仍保留 full-N/full-K contract，不改变 Python/Torch op ABI、benchmark CLI
或 quant 支持矩阵。

## ptxas 对比

目标 kernel：`uint8`, `CTA=128x256x8`, `sm_70`。

| group_size | FullTile baseline | `988b62b` 删除 TileMode 后 | 本次修复后 |
|---:|---:|---:|---:|
| -1 | 254 regs, 0 spill | 255 regs, 0 spill | 255 regs, 0 spill |
| 32 | 254 regs, 0 spill | 255 regs, 0 spill | 254 regs, 0 spill |
| 64 | 254 regs, 0 spill | 255 regs, 0 spill | 254 regs, 0 spill |
| 128 | 254 regs, 0 spill | 255 regs, 0 spill | 254 regs, 0 spill |

说明：

- positive group 的寄存器数已回到 FullTile baseline。
- `group_size=-1` 仍为 `255 regs`，但无 stack/spill；性能结果已回到历史
  FullTile 正常档。该路径还有额外 metadata cache 行为，后续如果要追平
  `254 regs` 可以继续单独看 SASS。
- 本次修复未引入 spill，所有目标组仍为 `0 bytes stack frame`,
  `0 bytes spill stores`, `0 bytes spill loads`。

日志来源：

- FullTile baseline: `benchmarks/results/sm70_tilemode_refactor/baseline_build.log`
- `988b62b`: `benchmarks/results/sm70_tilemode_refactor/after_build.log`
- 本次修复: `benchmarks/results/sm70_tilemode_regression_fix/build_final2.log`

## Benchmark 对比

复测只跑用户给定命令：

```bash
SM70_MARLIN_U8_CTA=128x256x8 \
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint8 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
BENCH_PRESET=quick ./benchmark.sh dense
```

形状：`MKN=5120x4096x4096`。

| group_size | operator_us | operator TFLOPs | kernel_like_us | kernel_like TFLOPs |
|---:|---:|---:|---:|---:|
| -1 | 2174.98 | 78.99 | 2125.82 | 80.82 |
| 32 | 2298.88 | 74.73 | 2269.70 | 75.69 |
| 64 | 2231.30 | 77.00 | 2260.99 | 75.98 |
| 128 | 2275.33 | 75.51 | 2233.34 | 76.92 |

历史 FullTile 正常档来自 `docs/20260516_025_sm70_dense_geometry_templates.md`：

| group_size | 历史 FullTile kernel_like TFLOPs | 本次修复 kernel_like TFLOPs |
|---:|---:|---:|
| -1 | 79.63 | 80.82 |
| 32 | 75.68 | 75.69 |
| 64 | 75.44 | 75.98 |
| 128 | 75.59 | 76.92 |

结论：本次修复后的 `uint8 + 128x256x8` 已回到历史 FullTile 正常性能范围，
positive group 略高于历史 smoke 数字；`group_size=-1` 也处于正常档。

日志来源：

- `benchmarks/results/sm70_tilemode_regression_fix/u8_128x256x8_benchmark_final.log`
- benchmark wrapper 同步写入
  `benchmarks/results/20260520_185849_dense_quick.log`
- 历史对照文档：
  `docs/20260516_025_sm70_dense_geometry_templates.md`

## 验证

静态确认 residue/TileMode 没有回归：

```bash
rg -n "Sm70TileMode|TileMode_|kTileMode|kResidue|load_full_tile|load_residue_tile|dispatch_tile_mode|initial_k_advance|tile_k_end_|next_k_advance_" csrc tests
```

结果：无匹配。

构建命令：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH" \
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST=7.0 \
CMAKE_ARGS='-DCMAKE_CUDA_FLAGS=-gencode arch=compute_70,code=sm_70' \
MAX_JOBS=8 NVCC_THREADS=1 \
./build.sh
```

导入检查：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH" \
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
PYTHONPATH=$PWD/python \
./.venv/bin/python -c "import torch; import marlin_v100; import marlin_v100._C; import marlin_v100._moe_C; print('ok', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

结果：

```text
ok 2.10.0+cu128 12.8 Tesla V100-SXM2-32GB
```

定向 pytest：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH" \
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
PYTHONPATH=$PWD/python \
./.venv/bin/pytest tests/test_marlin_dense.py -k "cta_geometry or residue_n or residue_k or small_tile or size_k"
```

结果：

```text
102 passed, 98 deselected
```

`git diff --check` 通过。

## 备注

- 本次没有因为 benchmark 波动回滚 `988b62b`。
- 本次没有恢复 residue N/K 的运行语义。
- 本次只修复 full-tile hot path 的非等价实现细节：恢复 qweight macro-N
  编译期常量路径，并移除已经无语义作用的 iterator 状态。
