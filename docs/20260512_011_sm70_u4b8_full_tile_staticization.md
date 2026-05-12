# 2026-05-12 SM70 uint4b8 FullTile 模板静态化结果

## 改动内容

本轮正式把 dense `uint4b8` 的 full-tile / residue 路径按 `uint4`
的方式拆成编译期模板实例，核心改动位于
`csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu`：

- `Sm70U4B8IteratorB` 增加 `bool FullTile_` 模板参数。
- `Sm70U4B8GemmTraits`、`sm70_marlin_u4b8_gemm_kernel`、host launch
  路径同步增加 `FullTile` 模板参数。
- `load()` 拆成 `load_full_tile()` 和 `load_residue_tile()`，并通过
  `if constexpr (FullTile_)` 静态分裂。
- `refresh_scale_cache()` 中 full-tile 路径删除 `logical_n` 边界判断。
- host 侧按 `size_k % 32 == 0 && size_n % 128 == 0` 选择
  `FullTile=true`，其它 shape 走 `FullTile=false`。

这次没有修改：

- Torch op schema
- `marlin_gemm(...)` Python/C++ 调用参数
- `b_q_weight` 主布局
- `b_scales` 语义和 shape
- repack helper 格式
- `uint4b8` 反量化数学
- MoE 或其它量化格式

## 资源结果

构建命令：

```bash
./build.sh
```

资源复核命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4b8_gemm_kernel"
```

`cuobjdump` 看到的 `sm70_marlin_u4b8_gemm_kernel` 实例如下：

| specialization | REG | STACK | LOCAL | 结论 |
| --- | ---: | ---: | ---: | --- |
| `<128, false>` | 250 | 0 | 0 | residue 专门实例，无 spill |
| `<64, false>` | 250 | 0 | 0 | residue 专门实例，无 spill |
| `<32, false>` | 252 | 0 | 0 | residue 专门实例，无 spill |
| `<-1, false>` | 247 | 0 | 0 | residue 专门实例，无 spill |
| `<128, true>` | 255 | 0 | 0 | full-tile 专门实例，无 spill |
| `<64, true>` | 255 | 0 | 0 | full-tile 专门实例，无 spill |
| `<32, true>` | 246 | 0 | 0 | full-tile 专门实例，无 spill |
| `<-1, true>` | 238 | 0 | 0 | full-tile 专门实例，无 spill |

关键变化：

- full-tile / residue 两套实例均保持 `STACK=0`、`LOCAL=0`。
- 没有 spill stores / spill loads。
- `GroupSize=-1, FullTile=true` 从原单实例 `REG=247` 降到 `REG=238`。
- `GroupSize=32, FullTile=true` 也低于 residue 实例。
- `GroupSize=64/128, FullTile=true` 升到 `REG=255`，后续如果继续优化这两个
  group size，需要关注寄存器压力。

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
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_helpers.py::test_uint4b8_act_order_repack_layout_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference
```

结果：通过，`18 passed in 2.06s`。

## Benchmark 结果

设备：

```text
Tesla V100-SXM2-32GB, sm70
```

### uint4b8 group sweep

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

结果：

| group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| `-1` | 2245.63 | 76.50 | 2225.15 | 77.21 |
| `32` | 2252.80 | 76.26 | 2253.31 | 76.24 |
| `64` | 2305.02 | 74.53 | 2324.48 | 73.91 |
| `128` | 2239.49 | 76.71 | 2306.56 | 74.48 |

相对上一份正式记录 `docs/20260512_010_sm70_u4b8_group_size_paths.md`
中的原实现结果：

| group_size | 旧 kernel_like TFLOPs | 本轮 kernel_like TFLOPs | 变化 |
| ---: | ---: | ---: | ---: |
| `-1` | 69.62 | 77.21 | +10.9% |
| `32` | 71.62 | 76.24 | +6.5% |
| `64` | 74.07 | 73.91 | -0.2% |
| `128` | 75.28 | 74.48 | -1.1% |

`group_size=-1` 是本轮最主要收益点：只做 FullTile 静态化，不改 scale
policy 和 repack，就从约 `69.62 TFLOPs` 提升到 `77.21 TFLOPs`。

### uint4 对照

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes -1 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

结果：

| quant | group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uint4` | `-1` | 2237.44 | 76.78 | 2211.33 | 77.69 |

本轮 `uint4b8 group_size=-1` 的 `77.21 TFLOPs` 已经贴近当前同环境
`uint4 group_size=-1` 的 `77.69 TFLOPs`，但仍低约 `0.6%`，没有真正超过。

## 结论

这次 `uint4b8` FullTile 模板静态化值得保留：

- full-tile 热路径在编译期删除 residue 边界检查。
- `group_size=-1` full-tile 实例寄存器从原 `247` 降到 `238`。
- `group_size=-1` benchmark 从旧记录约 `69.62 TFLOPs` 提升到
  `77.21 TFLOPs`。
- `group_size=32` 也有明显提升。
- `group_size=64/128` 没有明显收益，但回退幅度在 quick benchmark 噪声范围内；
  后续需要用 scale policy/repack 单独优化。

这次也说明：仅靠 FullTile 静态化还不足以让 `uint4b8 group_size=-1`
稳定超过 `uint4 group_size=-1`。如果下一步要继续拉高绝对性能，优先方向应是
`group_size=-1` 的 scale policy 或 scale-only repack，而不是继续扩大
FullTile 分裂范围。

## 下一步

- 单独实现并验证 `uint4b8 group_size=-1` 的虚拟 `128` scale refresh policy。
- 保持 `b_scales` 公共语义为单 group，不在接口层扩展 metadata。
- 对 `group_size=64/128` 关注 `FullTile=true` 的 `REG=255`，必要时拆更轻的
  full-tile scale cache 路径。
- 若 scale policy 仍不能稳定超过 `uint4 group_size=-1`，再设计 scale-only
  repack，让同一 CTA_N 内的 scale 访问更贴近 iterator 的读取顺序。
