# 2026-05-13 SM70 int4 GroupSize Metadata Cache 对齐结果

## 目标

本轮修复 `uint4` zero-point 路径在上一版结构对齐后的 benchmark 衰退，同时继续
约束 `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu` 和
`csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu` 的 IteratorB 结构保持同形。

最终采用的策略：

- `GroupSize=-1`: metadata 在 iterator constructor 中内联 cache 一次。
- `GroupSize=32/64/128`: metadata 在 `load()` 内、`load_full_tile()` /
  `load_residue_tile()` 外无条件刷新 cache。
- `load_full_tile()` / `load_residue_tile()` 不直接读取 positive group metadata。
  两条路径只读取 `cached_scales_`；`u4` 额外读取 `cached_bias_`。
- 不恢复 `cached_group_ == group` runtime early return。

本轮未修改 public Torch op schema、Python wrapper、qweight physical layout、
scale/zero tensor shape、MoE 或其他量化格式。

## 回溯保护

按计划先保存了性能衰退现场：

```bash
git stash push -m "wip u4-u4b8-cache-policy-alignment-regression" -- \
  csrc/quantization/marlin/sm70_marlin_u4_gemm.cu \
  csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu
git stash apply stash@{0}
```

当前保留 `stash@{0}`，用于回看本轮修复前的回归版本。

## 代码改动

### u4b8

`csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu`:

- 删除旧的 `kCacheScales` / `cache_single_group_scales()` 形态。
- `GroupSize=-1` 的 scale cache 保持在 constructor 内联执行，并增加注释说明：
  single-group scale row 对所有 K tile 都稳定，所以适合放在 MMA mainloop 之外。
- 新增 `cache_current_group_metadata(int group) const`，只供
  `GroupSize=32/64/128` 的 `load()` 调用。
- `load()` 结构固定为：

```cpp
if (!mask_enabled_) {
  return;
}

if constexpr (kGroupSize != -1) {
  int const first_logical_k = k_offset_ + thread_offset_.strided();
  cache_current_group_metadata(scale_group(first_logical_k));
}

if constexpr (kFullTile) {
  load_full_tile(frag);
} else {
  load_residue_tile(frag);
}
```

- `load_full_tile()` / `load_residue_tile()` 只从 `cached_scales_` 读取 scale。
- 保持 `128x256x32 / 8 warp`、`uint4` vectorized qweight load、FullTile/residue
  分裂和 macro-N qweight layout 不变。

### u4

`csrc/quantization/marlin/sm70_marlin_u4_gemm.cu`:

- `GroupSize=-1` 的 scale/zero metadata cache 保持在 constructor 内联执行。
  constructor 直接预计算：
  - `cached_scales_`
  - `cached_bias_ = -zero * scale`
- 删除旧的 runtime group reuse 分支，不保留 `cached_group_`。
- 删除旧 `refresh_metadata_cache()` 名称，统一使用
  `cache_current_group_metadata(int group) const`。
- `GroupSize=32/64/128` 在 `load()` 里调用 `cache_current_group_metadata()`；
  full/residue 内层不再直接读取 `scales_`、`qzeros_`，也不局部生成 bias。
- `load_full_tile()` / `load_residue_tile()` 与 `u4b8` 同样按
  qword -> low dequant -> 写低半 -> high dequant -> 写高半的顺序写 fragment。
- 保持 `128x128x32 / 4 warp` 几何不变。

保留的必要差异：

- `u4b8`: `128x256x32 / 8 warp`，`uint4` qweight access。
- `u4`: `128x128x32 / 4 warp`，`uint2` qweight access。
- `u4`: 需要 `b_zeros`，并缓存 `cached_bias_`。
- `u4`: 因当前 4-warp thread map 有多个 K-strided iteration，仍保留
  `qweight_strided_offsets_[s]`。
- `u4b8`: 只缓存 scale，热路径使用 `dequant<kU4B8>` + `__hmul2`。

## 对齐检查

命令：

```bash
rg -n "cached_group_|kCacheScales|refresh_metadata_cache|cache_single_group_scales|cache_current_group_metadata|void load\(|void load_full_tile|void load_residue_tile|scales_ \+ group|qzeros_\[qzeros_row|scale_row|zwords|uint2 zwords" \
  csrc/quantization/marlin/sm70_marlin_u4_gemm.cu \
  csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu
```

结论：

- 未出现 `cached_group_`。
- 未出现 `kCacheScales`。
- 未出现 `refresh_metadata_cache`。
- 未出现 `cache_single_group_scales`。
- `scales_ + group * params_.size_n` 只出现在
  `cache_current_group_metadata()`。
- `u4` 的 `qzeros_[qzeros_row + ...]` 只出现在
  `cache_current_group_metadata()`。
- 两边的 `load()` 控制流保持等价；差异集中在允许清单内。

## 构建与资源

构建命令：

```bash
./build.sh
```

结果：通过。

资源复核命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4_gemm_kernel|sm70_marlin_u4b8_gemm_kernel"
```

`u4b8` 资源：

| specialization | REG | STACK | LOCAL | spill |
| --- | ---: | ---: | ---: | ---: |
| `<128, false>` | 244 | 0 | 0 | 0 |
| `<64, false>` | 244 | 0 | 0 | 0 |
| `<32, false>` | 244 | 0 | 0 | 0 |
| `<-1, false>` | 238 | 0 | 0 | 0 |
| `<128, true>` | 250 | 0 | 0 | 0 |
| `<64, true>` | 250 | 0 | 0 | 0 |
| `<32, true>` | 250 | 0 | 0 | 0 |
| `<-1, true>` | 238 | 0 | 0 | 0 |

`u4` 资源：

| specialization | REG | STACK | LOCAL | spill |
| --- | ---: | ---: | ---: | ---: |
| `<128, false>` | 255 | 0 | 0 | 0 |
| `<64, false>` | 255 | 0 | 0 | 0 |
| `<32, false>` | 255 | 0 | 0 | 0 |
| `<-1, false>` | 253 | 0 | 0 | 0 |
| `<128, true>` | 255 | 0 | 0 | 0 |
| `<64, true>` | 255 | 0 | 0 | 0 |
| `<32, true>` | 255 | 0 | 0 | 0 |
| `<-1, true>` | 244 | 0 | 0 | 0 |

结论：

- dense `u4` / `u4b8` 目标 kernel 均为 `STACK=0`、`LOCAL=0`、无 spill。
- `u4` positive group 仍贴近 `REG=255`，但没有重新触发 stack/spill。
- `u4b8` positive group 的 cache 外提增加了 false 实例寄存器到 `REG=244`，
  true 实例为 `REG=250`，仍无 spill。

## 验证

导入：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：

```text
imports ok
```

pytest 收集：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
```

结果：

```text
226 tests collected in 1.19s
```

定向测试：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_helpers.py::test_uint4b8_act_order_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_marlin_dequantize_uint4_zp_matches_quantize_helper_output \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_8_row_bucket_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_size_m_24_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_small_tile_matches_reference
```

结果：

```text
33 passed in 3.26s
```

## Benchmark

设备：

```text
Tesla V100-SXM2-32GB, sm70
```

### uint4

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260513_221602_dense_quick.log
```

结果：

| group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| `-1` | 2251.78 | 76.29 | 2168.83 | 79.21 |
| `32` | 2259.97 | 76.02 | 2237.44 | 76.78 |
| `64` | 2189.31 | 78.47 | 2215.94 | 77.53 |
| `128` | 2168.83 | 79.21 | 2214.91 | 77.56 |

### uint4b8

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260513_221717_dense_quick.log
```

结果：

| group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| `-1` | 2084.86 | 82.40 | 2043.39 | 84.08 |
| `32` | 2074.11 | 82.83 | 2061.31 | 83.34 |
| `64` | 2061.31 | 83.34 | 2061.31 | 83.34 |
| `128` | 2012.16 | 85.38 | 2058.75 | 83.45 |

## 结论

- 本轮修复可以保留：`u4` 的 positive group metadata direct-load/bias 逻辑已经
  移出 full/residue 内层，benchmark 不再表现为严重衰退。
- `u4b8` positive group 也采用同样的 `load()` 外 cache 刷新后，没有出现稳定
  >2% benchmark 回退；因此本轮不需要为 `u4b8` 做性能例外。
- `GroupSize=-1` constructor inline cache 是更干净的固定策略，因为 metadata 在所有
  K tile 上不变；positive group 不做 runtime reuse 判断，避免把
  `cached_group_ == group` 分支重新带入热路径。
- 结构差异已经收敛到必要差异：量化数学、`u4` zero-point/bias、CTA/warp 几何、
  `uint2` vs `uint4` qweight access、以及 `u4` 的 `b_zeros` 参数。
- 后续如果继续压 `u4`，重点仍是降低 `REG=255` 附近的 zero-point/bias cache
  压力；当前版本至少维持了 `STACK=0`、`LOCAL=0`、无 spill 的稳定状态。
