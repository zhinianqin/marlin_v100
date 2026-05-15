# SM70 dense CTA geometry 模板化记录

## 背景

在前几轮实现中，dense SM70 CUTLASS kernel 已经逐步形成四条主路径：

- `uint4`：4-bit zero-point scale+bias，运行时数学为 `q * scale + zp_bias`。
- `uint4b8`：4-bit bias-8 scale-only，运行时数学为 `dequant(q) * scale`。
- `uint8`：8-bit zero-point scale+bias，运行时数学为 `q * scale + zp_bias`。
- `uint8b128`：8-bit bias-128 scale-only，运行时数学为 `(q - 128) * scale`。

这些路径此前默认都收敛到 `128x256x32 / 8 warp`，其中 `CTA_K=32` 固定。为了比较 small-M、large-M、不同 N tile 对寄存器、stack 和吞吐的影响，本轮把 `uint4` 已有的 CTA geometry 模板化方案同步到 `uint4b8 / uint8 / uint8b128`，并保留默认路径不变。

本轮不修改 public Torch op schema、Python wrapper、qweight repack layout、scale/bias metadata layout 或 benchmark CLI。新增控制面仅用于实验：

```bash
SM70_MARLIN_U4_CTA=128x256x8
SM70_MARLIN_U4B8_CTA=128x256x8
SM70_MARLIN_U8_CTA=128x256x8
SM70_MARLIN_U8B128_CTA=128x256x8
```

环境变量未设置时，四条路径仍默认使用 `128x256x32 / 8 warp`。

## 支持的 geometry

`CTA_K` 固定为 `32`。支持的 canonical geometry 来自 pure GEMM sweep 中 per-warp M/N 不超过 64 的组合：

| CTA_M | CTA_N | Warps | WarpShape |
|---:|---:|---:|---|
| 32 | 128 | 4 | `32x32x32` |
| 32 | 256 | 4 | `32x64x32` |
| 64 | 64 | 4 | `32x32x32` |
| 64 | 128 | 4 | `32x64x32` |
| 64 | 128 | 8 | `32x32x32` |
| 64 | 256 | 4 | `64x64x32` |
| 64 | 256 | 8 | `32x64x32` |
| 128 | 64 | 4 | `64x32x32` |
| 128 | 64 | 8 | `32x32x32` |
| 128 | 128 | 4 | `64x64x32` |
| 128 | 128 | 8 | `64x32x32` |
| 128 | 256 | 8 | `64x64x32` |
| 256 | 64 | 4 | `64x64x32` |
| 256 | 64 | 8 | `64x32x32` |
| 256 | 128 | 8 | `64x64x32` |

不支持 `CTA_M=512` 或 `CTA_N=512`，也不实例化 `32x64x4` 这类 non-canonical 组合。非法环境变量会直接报错，例如：

```text
Unsupported SM70_MARLIN_U8_CTA=32x64x4. Supported geometries are ...
```

环境变量格式支持：

```text
128x256x8
128*256*8
128,256,8
```

其中第三个数字是 warp 数，不包含 `CTA_K`，因为 `CTA_K=32` 固定。

## 实现要点

四个文件现在使用相同的模板形态：

```cpp
template <int CtaM, int CtaN, int Warps, int GroupSize,
          Sm70TileMode TileMode>
```

kernel launch 使用：

```cpp
__launch_bounds__(Warps * 32, 1)
ThreadblockShape = GemmShape<CtaM, CtaN, 32>
grid = ceil(M / CtaM) x ceil(N / CtaN)
```

每条路径都有各自的 `Sm70*WarpShape<CtaM, CtaN, Warps>` specialization，并通过 `static_assert` 保证：

```cpp
WarpShape::kM <= 64
WarpShape::kN <= 64
MmaCore::kThreads == Warps * 32
```

这样非法 geometry 不会被实例化，合法 geometry 的 warp shape、thread count、iterator shape 都由编译期决定。

## IteratorB 泛化

`CTA_N` 支持 `64 / 128 / 256`，因此 B iterator 不再假设 `ThreadMap::Iterations::kContiguous == 4`，而是约束为：

```cpp
ThreadMap::Iterations::kContiguous == CtaN / 64
```

full-tile qweight load 按 contiguous count 编译期分支：

| CTA_N | contiguous count | 4-bit qweight load | 8-bit qweight load |
|---:|---:|---|---|
| 64 | 1 | scalar `uint32_t` | two scalar `uint32_t` |
| 128 | 2 | `uint2` | two `uint2` |
| 256 | 4 | `uint4` | two `uint4` |

其中 8-bit 路径每个 B fragment access 需要两个 qword：

- `word0` 覆盖低 4 个 uint8。
- `word1` 覆盖高 4 个 uint8。

`uint4 / uint4b8` 每个 access 仍只需要一个 qword。

当 `ThreadMap::Iterations::kStrided > 1` 时，`qweight_offset(...)` 不复用单个 base offset 推导所有 strided fragment，而是按 logical K/N 重新计算 offset，保证不同 CTA geometry 下 CUTLASS thread map 的 strided 多迭代情况正确。

## partial macro-N 处理

qweight physical layout 仍是 256-column macro-N layout。即使 `CTA_N=64/128`，物理 layout 也没有改成更小 macro-N。为了避免 `size_n=64/128` 等 case 错误走 full-N fast path，host 侧 residue-N 判断统一为：

```cpp
residue_n = size_n % CtaN != 0 || size_n % 256 != 0;
```

含义是：

- `size_n % CtaN != 0`：当前 CTA 自身存在 N 边界。
- `size_n % 256 != 0`：物理 macro-N 不完整，需要动态 `subtile_count`。

只有当 `size_n` 同时整除 `CtaN` 和 `256` 时，才进入真正的 full-N fast path。这样 `CTA_N=64/128` 可以复用现有 qweight repack layout，不需要新增 repack 版本。

## 保留的功能差异

四条路径结构尽量对齐，但以下差异是功能必要差异：

| 路径 | qweight | metadata | 数学 |
|---|---|---|---|
| `uint4` | 4-bit，一个 qword/access | `scale + zp_bias` | `__hfma2(q, scale, bias)` |
| `uint4b8` | 4-bit，一个 qword/access | `scale` | `__hmul2(deq, scale)` |
| `uint8` | 8-bit，两个 qword/access | `scale + zp_bias` | `__hfma2(q, scale, bias)` |
| `uint8b128` | 8-bit，两个 qword/access | `scale` | `__hmul2(deq, scale)` |

因此 `u4/u8` 比 `u4b8/u8b128` 多一个 `b_zp_bias` plane、`cached_bias_` cache 和 `__hfma2`。`u8/u8b128` 比 `u4/u4b8` 多一次 qweight word load 和对应地址计算。

## 测试

构建和导入：

```bash
./build.sh
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：

```text
imports ok
```

`uint4b8 / uint8 / uint8b128` 定向测试：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_n_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_k_single_group_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_k_and_n_single_group_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_env_cta_geometry_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_env_cta_geometry_rejects_unsupported \
  tests/test_marlin_dense.py::test_marlin_dense_uint8_zp_bias_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint8_zp_bias_small_tile_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint8_zp_bias_residue_n_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint8_zp_bias_env_cta_geometry_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint8_zp_bias_env_cta_geometry_rejects_unsupported \
  tests/test_marlin_dense.py::test_marlin_dense_uint8b128_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint8b128_residue_n_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint8b128_env_cta_geometry_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint8b128_env_cta_geometry_rejects_unsupported
```

结果：

```text
90 passed in 6.46s
```

补跑 `uint4` 默认与 env geometry 回归：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_8_row_bucket_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_size_m_24_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_small_tile_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_residue_n_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_env_cta_geometry_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_env_cta_geometry_rejects_unsupported
```

结果：

```text
32 passed in 3.07s
```

## 资源

命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt
```

本轮共解析到 960 个目标 kernel 实例：

```text
4 quant types x 15 geometry x 4 group sizes x 4 TileMode = 960
```

全量实例资源范围：

| quant | REG range | max STACK | max LOCAL |
|---|---:|---:|---:|
| `u4` | 104-255 | 184 | 0 |
| `u4b8` | 94-255 | 56 | 0 |
| `u8` | 104-255 | 160 | 0 |
| `u8b128` | 96-255 | 56 | 0 |

默认 `128x256x8` 的 FullTile 资源：

| quant | REG | STACK | LOCAL |
|---|---:|---:|---:|
| `u4` | 255-255 | 0 | 0 |
| `u4b8` | 240-248 | 0 | 0 |
| `u8` | 254-255 | 0 | 0 |
| `u8b128` | 239-239 | 0 | 0 |

`256x128x8` override 的 FullTile 资源：

| quant | REG | STACK | LOCAL |
|---|---:|---:|---:|
| `u4` | 240-242 | 0 | 0 |
| `u4b8` | 232-234 | 0 | 0 |
| `u8` | 234-240 | 0 | 0 |
| `u8b128` | 226-232 | 0 | 0 |

观察：

- 默认 FullTile 没有 stack/local 退化。
- `256x128x8` FullTile 的寄存器显著低于默认 `128x256x8`，也没有 stack/local。
- 非默认高压组合，尤其 `64x256x4`，会出现 `REG=255` 和非零 `STACK`；这些实例主要用于实验，不影响默认路径。
- 所有目标实例 `LOCAL=0`。

## Benchmark smoke

默认 `128x256x8`：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 uint8 uint8b128 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

`5120x4096x4096`，kernel-like TFLOPs：

| group_size | `uint4b8` | `uint8` | `uint8b128` |
|---:|---:|---:|---:|
| -1 | 80.49 | 79.63 | 79.53 |
| 32 | 82.36 | 75.68 | 77.49 |
| 64 | 82.59 | 75.44 | 77.92 |
| 128 | 82.79 | 75.59 | 78.22 |

`256x128x8` override：

```bash
SM70_MARLIN_U4B8_CTA=256x128x8 \
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense

SM70_MARLIN_U8_CTA=256x128x8 \
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint8 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense

SM70_MARLIN_U8B128_CTA=256x128x8 \
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint8b128 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

`5120x4096x4096`，kernel-like TFLOPs：

| group_size | `uint4b8` | `uint8` | `uint8b128` |
|---:|---:|---:|---:|
| -1 | 80.97 | 76.45 | 75.86 |
| 32 | 75.81 | 71.56 | 74.43 |
| 64 | 76.17 | 71.58 | 74.42 |
| 128 | 75.71 | 72.55 | 73.60 |

结论：

- `256x128x8` FullTile 资源更低，但在当前 `5120x4096x4096` smoke 形状上并没有整体优于默认 `128x256x8`。
- 对 large-M large-N throughput，默认 `128x256x8` 仍应作为默认路径保留。
- `256x128x8` 值得继续用于 small-N、不同 M batch、decode-ish 或资源敏感场景的 sweep。

## 后续建议

- 使用 `benchmark_sm70_matmul_probe.py` 或 dense benchmark 增加按 M/N/K 维度的 geometry sweep，把 `128x256x8` 和 `256x128x8` 在 small-M、small-N、大 M、大 N 下分开比较。
- 对 `u4/u8` scale+bias 路径继续评估 metadata 合并布局，例如离线把 `scale + zp_bias` repack 成更容易 128-bit load 的结构，但这会改变 metadata layout，需要单独设计迁移。
- 对 `64x256x4` 这类有非零 STACK 的 geometry，默认不要启用；如需要保留实验入口，可以继续通过 env var 显式测试。
