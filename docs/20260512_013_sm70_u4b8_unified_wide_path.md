# 2026-05-12 SM70 uint4b8 统一 128x256x32 / 8-warp 路径结果

## 本轮目标

本轮把 dense `uint4b8` 从旧的 `128x128x32 / 4 warp` 主路径和临时
`WideGroupMinus1` 旁路整理成唯一实现：

```text
CTA = 128x256x32
Warp = 64x64x32
warps = 8
threads = 256
```

保留原有正式命名：

- `Sm70U4B8IteratorB`
- `Sm70U4B8GemmTraits`
- `sm70_marlin_u4b8_gemm_kernel`
- `launch_sm70_marlin_u4b8_gemm`

删除临时 wide 命名和 shape-level 双路径，降低后续维护成本。公共 Torch op schema、
Python wrapper 参数、`b_q_weight` repack 布局、`b_scales` shape/语义均不变。

本轮按要求不提交，完成后停在人工 review 状态。

## 代码改动

核心文件：

- `csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu`
- `csrc/quantization/marlin/marlin.cu`
- `tests/test_marlin_dense.py`

主要变化：

- `kCtaN` 从 `128` 改为 `256`。
- `kWarps` 从 `4` 改为 `8`，`kThreads = kWarps * 32` 变为 `256`。
- 删除 `kWideCtaN`、`kWideWarps`、`kWideThreads`、`WideGroupMinus1` 相关旁路。
- `Sm70U4B8IteratorB<Shape, ThreadMap, GroupSize, FullTile>` 成为唯一 256N
  量化 B iterator。
- iterator 增加静态约束：
  - `Shape::kN == 256`
  - `Shape::kK == 32`
  - `Iterations::kStrided == 1`
  - `Iterations::kContiguous == 4`
  - `Delta::kContiguous == 64`
  - `kElementsPerAccess == 8`
- `FullTile=true` 条件改为：

```cpp
size_k % 32 == 0 && size_n % 256 == 0
```

- `FullTile=false` 也使用同一套 `128x256x32 / 8 warp` GEMM shape，负责
  `size_n % 256 != 0` 和允许的 K residue。
- `group_size` 仍按 `-1 / 32 / 64 / 128` dispatch，但全部共享同一个 256N
  threadblock shape。

### Scale policy

统一 256N 后，如果所有 group 都沿用 scale cache，`group_size=64/128` 会明显掉速。
因此本轮保留在同一个 `Sm70U4B8IteratorB` 内的 group-specific policy，而不是恢复
旧 128N path：

| group_size | policy | 原因 |
| ---: | --- | --- |
| `-1` | cache scales | single group 语义，cache 命中稳定，寄存器最低 |
| `32` | direct scale load | 避免 cache 生命周期拉长 |
| `64` | direct scale load | cache 版本掉到约 60 TFLOPs |
| `128` | direct scale load | cache 版本掉到约 61 TFLOPs |

这个策略保留了单主线 shape，同时恢复 positive group 的吞吐。

### K residue

CUTLASS A iterator 的 K residue 采用 residue-first cadence。自定义 B iterator
需要同步这个节奏，否则 `size_k % 32 != 0` 时 A/B 会错位。

本轮在 B iterator 中加入：

- `tile_k_end_`
- `next_k_advance_`
- `initial_k_advance(size_k)`

`operator++()` 第一跳使用 residue K advance，之后回到固定 `32`。`load_residue_tile()`
用 `logical_k < tile_k_end_` 判断当前 K tile 是否有效，并对 invalid B fragment
显式写 0。

当前只允许 `uint4b8 group_size=-1` 使用非 32 对齐 K residue。`group_size=32/64/128`
仍要求 `size_k % 32 == 0`，因为这些 group metadata 的合法形状本身要求正向 group
整除 K，且本轮没有扩展正向 group 的 residue metadata 语义。

## 构建与资源

构建命令：

```bash
./build.sh
```

结果：通过。构建目标为 `TORCH_CUDA_ARCH_LIST=7.0`、`sm_70`。

资源复核命令：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4b8_gemm_kernel"
```

`cuobjdump` 结果：

| specialization | REG | STACK | LOCAL | spill |
| --- | ---: | ---: | ---: | ---: |
| `<128, false>` | 240 | 0 | 0 | 0 |
| `<64, false>` | 240 | 0 | 0 | 0 |
| `<32, false>` | 240 | 0 | 0 | 0 |
| `<-1, false>` | 241 | 0 | 0 | 0 |
| `<128, true>` | 242 | 0 | 0 | 0 |
| `<64, true>` | 242 | 0 | 0 | 0 |
| `<32, true>` | 242 | 0 | 0 | 0 |
| `<-1, true>` | 238 | 0 | 0 | 0 |

结论：

- 所有保留实例均为 `STACK=0`、`LOCAL=0`。
- 无 spill stores / spill loads。
- `group_size=-1, FullTile=true` 为 `REG=238`，仍是资源最轻的 full-tile 实例。
- positive group full-tile 从之前的 255 regs 降到 242 regs，统一 wide path
  不再顶满 SM70 寄存器上限。

## 验证

导入命令：

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
226 tests collected in 1.20s
```

定向测试：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_calibration.py \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_helpers.py::test_uint4b8_act_order_repack_layout_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_n_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_k_single_group_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_residue_k_rejects_multi_group_metadata
```

结果：

```text
29 passed in 2.39s
```

新增 residue 覆盖：

- `size_n=128` 覆盖 `size_n % 256 != 0`，覆盖 `group_size=-1/32/64/128`。
- `size_k=144, size_n=256, group_size=-1` 覆盖 single-group K residue。
- `size_k=144` 搭配多 group metadata 会被拒绝，错误信息包含
  `single-scale residue path`。

## Benchmark

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

日志：

```text
benchmarks/results/20260512_160926_dense_quick.log
```

结果：

| group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| ---: | ---: | ---: | ---: | ---: |
| `-1` | 2173.95 | 79.03 | 2154.50 | 79.74 |
| `32` | 2150.40 | 79.89 | 2205.18 | 77.91 |
| `64` | 2158.59 | 79.59 | 2212.86 | 77.64 |
| `128` | 2143.23 | 80.16 | 2189.31 | 78.47 |

### uint4 对照

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 --group-sizes -1 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

日志：

```text
benchmarks/results/20260512_161010_dense_quick.log
```

结果：

| quant | group_size | operator us | operator TFLOPs | kernel_like us | kernel_like TFLOPs |
| --- | ---: | ---: | ---: | ---: | ---: |
| `uint4` | `-1` | 2239.49 | 76.71 | 2207.74 | 77.82 |

本轮最终 `uint4b8 group_size=-1` 的 `79.74 TFLOPs kernel_like` 已经高于
同环境 `uint4 group_size=-1` 的 `77.82 TFLOPs kernel_like`。

## 对比

| 实现/记录 | group_size | kernel_like TFLOPs | 说明 |
| --- | ---: | ---: | --- |
| 旧 `128x128x32 / 4 warp` u4b8 | `-1` | 77.21 | `docs/20260512_011_sm70_u4b8_full_tile_staticization.md` |
| 临时 wide group=-1 原型 | `-1` | 约 81.56 | 只用于证明 256N 方向，不保留旁路命名 |
| 本轮统一 wide path | `-1` | 79.74 | 单主线、支持 `-1/32/64/128`、支持 residue |
| 本轮统一 wide path | `32` | 77.91 | direct scale load |
| 本轮统一 wide path | `64` | 77.64 | direct scale load |
| 本轮统一 wide path | `128` | 78.47 | direct scale load |
| 当前 `uint4` 对照 | `-1` | 77.82 | zero-point 路径 |
| pure GEMM baseline | n/a | 约 91.98 | `128x256x32 / 8 warp` probe |

与旧 `128x128` 正式路径相比，`group_size=-1` 从 `77.21` 提升到 `79.74`
TFLOPs，约 `+3.3%`。与之前单独 wide 原型的约 `81.56` 相比，本轮最终值略低，
但换来了单一路径、positive group 覆盖、residue 覆盖和无旁路命名的可维护结构。

第一版统一 256N 曾让 `group_size=64/128` 跌到约 `59.62/60.70` TFLOPs；切回
positive group direct scale load 后恢复到 `77.64/78.47` TFLOPs，因此不应恢复
所有 group 共享 scale cache 的写法。

## 保留与丢弃

保留：

- 单一 `128x256x32 / 8 warp` u4b8 path。
- 原有主线命名。
- `FullTile` 模板静态分裂。
- `group_size=-1` scale cache。
- `group_size=32/64/128` direct scale load。
- single-group K residue 支持。
- `size_n % 256 != 0` residue 支持。

丢弃：

- `WideGroupMinus1` 旁路命名。
- 旧 `128x128x32 / 4 warp` fallback。
- positive group 的统一 scale-cache policy。
- 已知精度失败的 `skip_flop=true` / `cached_bias=-8*scale` 方向。

## 已知限制

- 非 32 对齐的 K residue 目前只支持 `uint4b8 group_size=-1`。
- `group_size=32/64/128` 仍要求 `size_k % 32 == 0`。
- 当前 unified wide path 还没有达到 pure GEMM 约 `91.98 TFLOPs`，主要差距仍来自
  qweight unpack、bias-8 dequant、scale multiply、metadata address 计算，以及
  B dequant-to-smem 对主循环的额外指令。
- 本轮没有修改 qweight repack 或 scale metadata layout。

## 下一步

- 继续从 SASS/Nsight 角度看 256N path 的 dynamic instruction mix，而不是只看
  static 指令数量。
- 评估 qweight repack 是否能减少 `LOP3/SHF` 和 `qword >> 8` 相关指令。
- 如果继续冲击 `90 TFLOPs`，优先在同一个 `Sm70U4B8IteratorB` 内优化
  dequant-to-smem 的指令调度，不恢复多 path 结构。
- 对 `group_size=-1` 可以继续探索 scale-only repack，但必须保持公共 `b_scales`
  语义不变，且不能复制 metadata。
