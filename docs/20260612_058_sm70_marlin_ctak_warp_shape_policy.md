# SM70 Marlin CTA_K / Warp Shape / Packed Macro-N 策略

## 概要

本文记录 SM70 Marlin geometry 从旧的
`CTA_M x CTA_N x Warps` 形态扩展为完整常量 launch 形态：

```text
CTA_M x CTA_N x CTA_K x Warps x WarpM x WarpN x WarpK
```

该变更覆盖 dense 和 MoE 两类 SM70 Marlin CUDA 路径。不改变 Python
public API，不改变 Torch custom op schema，也不改变 repack op 的输入输出约定。

## 支持的 Shape 空间

Dense threadblock geometry 支持：

```text
CTA_M = 32, 64, 128, 256
CTA_N = 64, 128, 256
CTA_K = 16, 32, 64, 128
Warps = 4, 8
```

MoE threadblock geometry 收紧为：

```text
CTA_M = 32, 64
CTA_N = 64, 128, 256
CTA_K = 16, 32, 64, 128
Warps = 4, 8
```

也就是说，`CTA_M=128/256` 是 dense-only geometry。MoE runtime env、
auto policy mirror、tests 和 benchmark helpers 都只应枚举 `CTA_M=32/64`
的 geometry 子集。

Warp geometry 支持：

```text
WarpM = 32, 64
WarpN = 32, 64
WarpK = 16, 32
```

`Sm70WarpShape` 是显式的编译期校验器和 CUTLASS type adapter：

```cpp
template <int CtaM, int CtaN, int CtaK, int Warps,
          int WarpM, int WarpN, int WarpK>
struct Sm70WarpShape;
```

选择的 warp shape 必须能精确分解 CTA：

```text
CTA_M % WarpM == 0
CTA_N % WarpN == 0
CTA_K % WarpK == 0
(CTA_M / WarpM) * (CTA_N / WarpN) * (CTA_K / WarpK) == Warps
```

最终暴露给 CUTLASS 的 warp shape 为：

```cpp
cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>
```

## 非法组合与多解组合

有些理论 CTA 组合在当前支持的 warp shape 集合下没有合法分解。这类组合会在
host dispatch 阶段被拒绝；如果绕过 dispatch 直接实例化，也会被编译期
`static_assert` 拒绝。

无解组合示例：

```text
CTA_M=32, CTA_N=64, CTA_K=16, Warps=8
CTA_M=32, CTA_N=256, CTA_K=128, Warps=4
CTA_M=256, CTA_N=256, CTA_K=128, Warps=4
```

原因很直接：不存在一个受支持的 `WarpM/WarpN/WarpK` 三元组，能让 CTA
分解后的 warp 数等于请求的 `Warps`。

第二类更实际的拒绝条件是：某个 CTA/warp 组合虽然数学上可以分解，但当前
CUTLASS Volta row-major/row-major `DefaultMmaCore`、Volta epilogue
thread map 或 Marlin quantized B iterator contract 不支持它。

当前实现支持范围是以下条件的交集：

```text
合法 CTA/warp 分解
CUTLASS A thread map 产生非零迭代
CUTLASS B thread map 产生非零迭代
Marlin B iterator 对每个 64-column quant tile 看到一个 contiguous iteration
Volta epilogue 产生非零 row/column iteration
```

重要结论：

```text
CTA_K=16 不是全局非法项；只有无合法 warp 分解，或不满足 CUTLASS /
Marlin 静态 contract 的具体 CTA/warp 组合才会被拒绝。
```

早期 probe 曾经把部分 `WarpK=16` 失败误收敛到 `CTA_K=16` 不可用。
后续定位确认，真实问题是 Volta row-major crosswise warp iterator 在非零
K offset 时没有同步维护 `byte_offset_` phase。生产
`Sm70MarlinMmaPipelined` 已通过 phase-aware K offset helper 修复该问题；
因此已实例化列表中的 `WarpK=16` geometry 需要作为合法精度候选继续进入
env sweep，而不是被测试 helper 统一标记为非法。

还有一些大 `CTA_K` / 小 `CTA_N` / 高 warp 数组合，也会因为 Volta
epilogue 或 B 侧 thread map 产生零迭代而被拒绝。这些是编译期 CUTLASS
layout 约束，不是 runtime 数据相关失败。

有些 CTA 组合存在多个合法分解。当前实现不再推断其中哪一个更好。env
override 和 selector 输出都显式携带 `WarpM/WarpN/WarpK`，因此两个合法分解
会被视为不同的常量 kernel 实例。

当前实例化的组合为：

```text
32x64x32x4x32x32x16
32x64x64x4x32x32x32
32x64x64x4x32x64x16
32x64x128x4x32x64x32
32x128x32x4x32x32x32
32x128x32x4x32x64x16
32x128x64x4x32x64x32
32x128x64x8x32x32x32
32x128x64x8x32x64x16
32x128x128x8x32x64x32
32x256x32x4x32x64x32
32x256x64x8x32x64x32
64x64x32x4x32x32x32
64x64x32x4x32x64x16
64x64x32x4x64x32x16
64x64x32x8x32x32x16
64x64x64x4x32x64x32
64x64x64x4x64x32x32
64x64x64x4x64x64x16
64x64x64x8x32x32x32
64x64x64x8x32x64x16
64x64x128x4x64x64x32
64x64x128x8x32x64x32
64x128x32x4x32x64x32
64x128x32x4x64x32x32
64x128x32x4x64x64x16
64x128x32x8x32x32x32
64x128x32x8x32x64x16
64x128x32x8x64x32x16
64x128x64x4x64x64x32
64x128x64x8x32x64x32
64x128x64x8x64x32x32
64x128x64x8x64x64x16
64x128x128x8x64x64x32
64x256x32x4x64x64x32
64x256x32x8x32x64x32
64x256x32x8x64x32x32
64x256x32x8x64x64x16
64x256x64x8x64x64x32
128x64x32x4x32x64x32
128x64x32x4x64x32x32
128x64x32x4x64x64x16
128x64x32x8x32x32x32
128x64x32x8x32x64x16
128x64x32x8x64x32x16
128x64x64x4x64x64x32
128x64x64x8x32x64x32
128x64x64x8x64x32x32
128x64x64x8x64x64x16
128x64x128x8x64x64x32
128x128x32x4x64x64x32
128x128x32x8x32x64x32
128x128x32x8x64x32x32
128x128x32x8x64x64x16
128x128x64x8x64x64x32
128x256x32x8x64x64x32
256x64x32x4x64x64x32
256x64x32x8x32x64x32
256x64x32x8x64x32x32
256x64x32x8x64x64x16
256x64x64x8x64x64x32
256x128x32x8x64x64x32
```

## 通用回退策略

generic fallback 有意保持保守。`CTA_N` 仍然从 `size_n` 中选择最大的可用
divisor：

```text
size_n % 256 == 0 -> CTA_N=256
else % 128 == 0   -> CTA_N=128
else % 64 == 0    -> CTA_N=64
```

得到 `CTA_N` 后，fallback 使用：

```text
CTA_N=64  -> 64x64x32x4x32x32x32
CTA_N=128 -> 32x128x32x4x32x32x32
CTA_N=256 -> 32x256x32x4x32x64x32
split_k=1
UseMetadataVectorWords=true
```

这保证 unset env 的生产默认 geometry 仍留在规范的 `CTA_K=32` shape family，
同时允许 selector 或 debug env override 选择其它常量实例。

## PackedMacroN 与 CTA_N

repack layout macro-N 是 packed layout 的属性。它不是 GEMM launch 选择的
`CTA_N`。

repack 继续按 `size_n` 选择 packed macro-N：

```text
size_n % 256 == 0 -> PackedMacroN=256
else % 128 == 0   -> PackedMacroN=128
else % 64 == 0    -> PackedMacroN=64
```

repack fast path 不变。GEMM geometry env override 不会让 repack 重新按另一种
macro-N layout 打包权重。

GEMM B iterator 现在同时携带两个概念：

```text
Shape::kN    = 实际 GEMM CTA_N
PackedMacroN = packed qweight layout macro-N
```

合法读取组合为：

```text
CTA_N=64,  PackedMacroN=64/128/256
CTA_N=128, PackedMacroN=128/256
CTA_N=256, PackedMacroN=256
```

等价地，两个值都必须属于 `{64,128,256}`，并且满足：

```text
PackedMacroN % CTA_N == 0
```

自动 fast path 仍然保持 `CTA_N == PackedMacroN`。非 auto `CTA_N` 实例只改变
GEMM reader 对已 repacked layout 的读取解释，不改变 repack layout 本身。

## 环境变量覆盖约定

Dense env：

```text
SM70_MARLIN_DENSE_CTA_GEOMETRY
SM70_MARLIN_DENSE_SPLIT_K
SM70_MARLIN_DENSE_METADATA_CACHE
```

MoE env：

```text
SM70_MARLIN_MOE_CTA_GEOMETRY
SM70_MARLIN_MOE_SPLIT_K
SM70_MARLIN_MOE_METADATA_CACHE
```

Geometry 格式必须严格为：

```text
{CTA_M}x{CTA_N}x{CTA_K}x{Warps}x{WarpM}x{WarpN}x{WarpK}
```

示例：

```text
32x256x32x4x32x64x32
```

字段数量错误、字段值非法、warp 分解非法、不支持的 split-K 值、不支持的
metadata mode 都必须显式失败，不能静默 fallback 到 generic policy。

支持的 split-K 值：

```text
1, 2, 4, 8
```

支持的 metadata cache 值：

```text
vector_words
lane_vectors
```

metadata env 未设置或为空字符串时，等价于 `vector_words`。

## Focused Exact-MNK Env Sweep

为了验证非 auto `CTA_N` 读取端、`CTA_K/WarpK` 常量实例、split-K 与
metadata cache 组合，新增 focused direct-op env sweep。该 sweep 不替代完整
writeback matrix，而是用更小的固定 MNK 集合做快速但完整的 env 覆盖。

覆盖的 exact MNK 为：

```text
32x1024x1024
32x1088x1024
32x1152x1024
64x1024x1024
64x1088x1024
64x1152x1024
```

Dense 直接调用 `marlin_gemm`，其中 `M/N/K` 就是 op 的
`size_m/size_n/size_k`。

MoE 也直接调用 `moe_wna16_marlin_gemm`，本轮不是 stage1/stage2 模型语义
测试，不从 hidden/intermediate 派生额外 shape。每个 MNK 都构造 synthetic
single-stage MoE raw GEMM：

```text
activation:          [M, K]
per-expert weight:   [experts, K, N]
top_k:               1
output:              [M, N]
```

这样 `tokens=M` 且 `tokens * top_k == M`，不会因为 routed stage 的 top-k
展开改变 requested M。当前 synthetic 参数固定为 `experts=8, top_k=1`。

每个 supported `{quant, group_size, MNK}` 都乘上对应路径的完整 env tuple：

```text
dense: 62 geometry * 4 split_k * 2 metadata_cache = 496
MoE:   39 geometry * 4 split_k * 2 metadata_cache = 312
```

合法性判断仍只来自 kernel 明确约束：

```text
size_n % CTA_N == 0
PackedMacroN % CTA_N == 0
dense: size_k % 32 == 0
dense split_k > 1: size_k % CTA_K == 0
MoE: CTA_M in {32,64}
MoE: size_k % CTA_K == 0
```

若 focused sweep 发现精度失败或 illegal memory access，必须先定位到
reference、quant/repack、PackedMacroN offset、CTA_K/WarpK MMA、split-K
reduce 或 metadata cache 等具体路径。只有确认组合违反当前 kernel 显式约束
时，才允许更新 legality helper；不能把未知失败直接标记为非法。

## CTA_K 与 Split-K

split-K helper 必须使用当前选择的 `CTA_K`，不能继续使用旧的固定 `32`。

dense no-split 路径保留 partial-K 能力。dense split-K 和 MoE 在 launch
split-K work 前要求 K 与当前选择的 `CTA_K` 兼容。

`WarpK=16` 可能让 `group_size >= 32` 的路径在同一个 CTA 内对同一个 quant
group 多次调用 `cache_current_group_metadata`。这是允许的。它是暴露更小
warp-K partition 后的直接结果，同时 iterator 仍然在局部刷新 group metadata。

## 独立 WarpK=16 定位单元

为了避免每次定位 `WarpK=16` 都重新编译整个 Python extension，当前新增了一个
单文件 CUDA probe：

```text
tests/sm70_warpk16_probe.cu
```

它只依赖 CUTLASS 头文件、CUDA runtime 和本仓库的
`csrc/quantization/marlin/sm70_marlin_mma.cuh`，不依赖 PyTorch/C10 binding，
也不会触发完整项目构建。

独立编译命令：

```bash
mkdir -p build/sm70_probe
/usr/local/cuda-12.8/bin/nvcc -std=c++17 -O2 -arch=sm_70 \
  -I/root/source/repos/cutlass/include -Icsrc \
  -I/usr/local/cuda-12.8/include \
  tests/sm70_warpk16_probe.cu \
  -o build/sm70_probe/sm70_warpk16_probe
```

基础 smoke：

```bash
./build/sm70_probe/sm70_warpk16_probe --smoke
```

`WarpK=16` 定位套件：

```bash
./build/sm70_probe/sm70_warpk16_probe --diagnose-warpk16
```

单 case 调用格式：

```bash
./build/sm70_probe/sm70_warpk16_probe \
  32x64x32x4x32x32x16 single direct k1 local 32 64 32
```

当前定位结果：

```text
64x64x32x4x32x32x32 stock full K=128:
  通过，作为普通 WarpK=32 baseline。

32x64x64x4x32x32x32 custom/single full K=128:
  通过，说明当前 custom/single mainloop 能处理 WarpK=32 且
  kPartitionsK=2 的场景。

32x64x32x4x32x32x16 single full K=16:
  通过。

32x64x32x4x32x32x16 single atomic K=16:
  通过。

32x64x32x4x32x32x16 single full K=32:
  精度失败，约 1866/2048 mismatch。

32x64x32x4x32x32x16 single atomic K=32:
  同样精度失败，约 1866/2048 mismatch。

32x64x32x4x32x32x16 custom full K=32:
  同样精度失败，约 1866/2048 mismatch。

32x64x32x4x32x32x16 single direct K=32 k0 only:
  通过。direct 模式不走 Volta epilogue，而是用 warp MMA 自带的
  IteratorC 将每个 CTA 内 K partition 的寄存器 accumulator 分别写到
  global buffer，再在 host 侧相加。

32x64x32x4x32x32x16 single direct K=32 k1 only:
  失败，约 1866/2048 mismatch。actual_l1 / actual_l2 与 reference 完全
  对齐，说明不是漏算或清零；最近邻分析显示 actual 的每个元素都能在
  reference 中找到精确匹配，列坐标完全一致，行坐标只发生 8 行块交换：
  row_delta_hist 为 d8=1024, d24=1024。

32x64x64x4x32x32x32 single direct K=64 k1 only:
  通过，作为 WarpK=32 + kPartitionsK=2 的 direct 模式对照。

32x64x64x4x32x64x16 single direct K=64 k1 only:
  失败，约 1796/2048 mismatch；同样表现为列不变、行按 d8/d24 两个
  8 行块交换。

32x128x64x8x32x64x16 single direct K=64 k1 only:
  失败，约 3666/4096 mismatch；同样表现为列不变、行按 d8/d24 两个
  8 行块交换。
```

因此当前不能把 `WarpK=16` 简单归类为“单 partition 可用，只是 epilogue
合并失败”。更准确的边界是：

```text
WarpK=16 的第 0 个 CTA 内 warp-K partition 可以正确；
当第 1 个 CTA 内 warp-K partition 有有效数据时，寄存器 accumulator
本身已经发生 8 行块置换；
regular epilogue、atomic epilogue、direct accumulator dump 都复现；
因此问题不在最终 global store，也不只是 epilogue reduction 合并。
```

当前更接近的根因方向是：

```text
Volta row-major A crosswise shared-memory layout / warp tile iterator
在 warp-K partition offset 非 0 时的行组映射契约；
也就是旧定位版本中 SingleStage probe 与生产 Pipelined 构造函数里的：

  warp_tile_iterator_A_.add_tile_offset(
      {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});

配合 CUTLASS RowMajorVoltaTensorOpMultiplicandCrosswise 的 swizzle 后，
第 1 个 warp-K partition 读出的 A 侧行组与 accumulator 行组发生
8 行块置换。
```

进一步验证后，问题可以收窄到 Volta crosswise warp iterator 的 K 方向
`add_tile_offset` 不维护内部 phase：

```text
CUTLASS Volta crosswise warp iterator 在 operator++() 中同时推进 pointer_
和 byte_offset_ phase。跨到 k_group 4 或 0 时会执行一次 byte_offset_ XOR。

但 add_tile_offset({..., k_offset}) 只直接移动 pointer_，并把 k_group_idx_
重置为 0，不会补上跨 k-group 时本应发生的 byte_offset_ phase flip。

WarpK=32 时第 1 个 warp-K partition 的 offset 是 8 个 k-group，phase flip
发生两次后回到原态，因此旧路径刚好正确。

WarpK=16 时第 1 个 warp-K partition 的 offset 是 4 个 k-group，正确路径
应发生一次 phase flip，但 add_tile_offset 没有做，所以 A 侧行组 phase 错位，
表现为 8 行块置换。
```

probe 中新增了 `single_phase` 实验 mainloop：构造函数里不再用一次
`add_tile_offset` 设置非零 `warp_idx_k` 的 K 方向 offset，而是先设置
M/N 方向 offset，再用 `operator++()` 推进
`Base::kWarpGemmIterations * warp_idx_k` 次。该路径只用于定位，不代表生产
pipelined mainloop 已经修复。

验证命令：

```bash
./build/sm70_probe/sm70_warpk16_probe \
  32x64x32x4x32x32x16 single direct k1 local 32 64 32

./build/sm70_probe/sm70_warpk16_probe \
  32x64x32x4x32x32x16 single_phase direct k1 local 32 64 32

./build/sm70_probe/sm70_warpk16_probe \
  32x64x32x4x32x32x16 single_phase full all local 32 64 32

./build/sm70_probe/sm70_warpk16_probe \
  32x64x64x4x32x64x16 single_phase full all local 32 64 64

./build/sm70_probe/sm70_warpk16_probe \
  32x128x64x8x32x64x16 single_phase full all local 32 128 64
```

当前结果：

```text
single direct k1:
  仍失败，约 1866/2048 mismatch，nearest exact 全量匹配且列不变，
  行按 d8/d24 两个 8 行块置换。

single_phase direct k1:
  通过，mismatches=0/2048，direct partition1 的 actual_l1 / actual_l2
  与 reference 对齐。

single_phase full all:
  通过，mismatches=0/2048。

32x64x64x4x32x64x16 single_phase full all:
  通过，mismatches=0/2048。

32x128x64x8x32x64x16 single_phase full all:
  通过，mismatches=0/4096。
```

因此 `WarpK=16` 不是理论不可修的 shape。生产 `Sm70MarlinMmaPipelined`
修复采用如下策略：

```text
1. 删除生产 Sm70MarlinMmaSingleStage，只保留当前 dense/MoE 实际使用的
   Sm70MarlinMmaPipelined。
2. 为 Volta crosswise warp iterator 的非零 K offset 增加 phase-aware helper。
3. Base::kWarpGemmIterations != 4 继续走原 WarpK=32 fast path，避免影响
   当前默认路径性能。
4. Base::kWarpGemmIterations == 4 走 phase-aware reset + forward operator++()
   路径，用于维护 pointer_ 与 byte_offset_ phase。
5. Pipelined mainloop 的 constructor、stage advance 正 offset、stage advance
   负 offset、以及普通 iterator advance 都由 helper 管理。
6. 代码中保留 TODO：WarpK=16 完整验证后，后续尝试合并
   Base::kWarpGemmIterations != 4 与 == 4 两条路径。
```

已经排除或弱化的方向：

```text
1. 不是最终 global store 问题：atomic epilogue 同样失败。
2. 不是普通 epilogue reduction 独有问题：direct accumulator dump 同样失败。
3. 不是缺失第二个 partition：actual_l1 / actual_l2 与 reference 对齐。
4. 不是随机读错：nearest-reference 分析显示 exact value 全量匹配，
   且列坐标完全不变，只发生 8 行块交换。
5. 不是单纯 set_kgroup_index 使用局部/全局 k-group 的问题：
   probe-only global k-group 4..7 与 local k-group 0..3 结果一致。
```

生产修复后，`WarpK=16` geometry 可以重新作为 exhaustive env sweep 的合法
精度候选；旧 `single` probe 仍保留 expected-failure 证据，用于证明原始
row permutation 根因可复现。

## `kPartitionsK > 1` 的 stage wrap 结论

在 env sweep 定位过程中，`32x64x128x4x32x64x32`、`K=768` 曾经触发
`illegal memory access`。该组合满足当前显式 warp 分解规则：

```text
CTA_M=32, CTA_N=64, CTA_K=128, Warps=4
WarpM=32, WarpN=64, WarpK=32
kPartitionsK = CTA_K / WarpK = 4
```

因此不能把它标成非法组合。独立 probe 已确认根因是：

```text
stock CUTLASS MmaPipelined 的 read-stage wrap 会在 smem_write_stage_idx==0
时使用 -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations
回绕读指针。

当前 Marlin 自定义 mainloop 在每个 CTA_K iteration 末尾已经通过普通
operator++() 把 warp iterator 推进到下一个 stage 的局部位置；对
kPartitionsK > 1 的几何，再套 stock CUTLASS 的完整 circular-buffer
回绕会把读指针拉到负的 shared-memory K group，从而产生 illegal memory
access。
```

生产 `Sm70MarlinMmaPipelined` 必须使用 Marlin 的相对 stage offset：

```text
smem_write_stage_idx == 1:
  wrap write stage；
  如果 kPartitionsK > 1，读指针前进
  (kPartitionsK - 1) * Base::kWarpGemmIterations

smem_write_stage_idx == 0:
  读指针后退
  (kPartitionsK + 1) * Base::kWarpGemmIterations
```

这个相对 offset 规则同时覆盖：

```text
WarpK=32 / Base::kWarpGemmIterations=8:
  继续走直接 add_tile_offset fast path。

WarpK=16 / Base::kWarpGemmIterations=4:
  使用 phase-aware reset + forward operator++() helper，保证 pointer_
  与 byte_offset_ phase 同步。
```

已用 standalone probe 验证：

```bash
./build/sm70_probe/sm70_warpk16_probe \
  32x64x128x4x32x64x32 custom direct all local 32 64 768

./build/sm70_probe/sm70_warpk16_probe \
  32x64x128x4x32x64x32 custom atomic all local 32 64 768

./build/sm70_probe/sm70_warpk16_probe \
  32x64x128x4x32x64x32 custom full all local 32 64 768
```

三者均通过，说明 mainloop 的 K-partition 累加、atomic epilogue 与 regular
Volta epilogue 都能正确处理该组合。对照的 stock CUTLASS full path 仍会
触发 illegal memory access，因此这个测试用来防止后续误把生产路径改回
stock wrap 语义。

## 多 Strided B Iterator 的 Packed Offset 与 Metadata 修复

env smoke sweep 继续暴露了一个 dense 精度失败：

```text
quant=fp8, group_size=-1, M=1, N=192, K=768
geometry=32x64x128x4x32x64x32
split_k=1
metadata_cache=vector_words
```

同一模式也出现在 U8、U8B128、U4B8 和 FP8 group_size=128 上。独立 half-B
probe 对同一 geometry 通过，因此根因不在 `Sm70MarlinMmaPipelined` 的
K-partition 累加，也不在 Volta epilogue，而在 quantized B iterator。

根因是：`ThreadMap::Iterations::kStrided > 1` 时，旧代码用线性 word delta
计算后续 `s` 的 qweight 地址：

```text
qweight_base + s * kStridedQweightDeltaWords
```

这个公式只在 strided K delta 没有跨过 packed `kQuantTileK=16` 边界时成立。
Marlin packed macro-N layout 跨 16-K tile 时地址映射不是线性的；特别是
`CTA_K=128 / Warps=4` 这类 geometry 会让一个 iterator 的多个 strided
iteration 跨 packed tile，导致读取错误 qweight。

生产修复是：所有 dense/MoE quantized B iterator 在多 strided 路径里，都按
当前 `s` 的逻辑坐标重新计算 packed macro-N offset：

```text
logical_k_s = k_offset + thread_offset.strided + s * ThreadMap::Delta::kStrided
logical_n_base = n_offset + thread_offset.contiguous
qweight_offset_from_logical(PackedMacroN, logical_k_s, logical_n_base)
```

同一类问题也适用于 finite group metadata。旧路径只按第一个 strided K
缓存一次 scales / zero-points；当后续 `s` 跨过 group 边界时，会继续使用旧
group 的 metadata。修复后：

```text
group_size == -1:
  仍在 iterator 构造阶段缓存一次 metadata，因为 K 方向没有 group 变化。

ThreadMap::Iterations::kStrided == 1:
  仍在 load() 入口按 first_logical_k 缓存一次，保持原 fast path。

ThreadMap::Iterations::kStrided > 1 && finite group:
  每个 s 进入 dequant 前，按 logical_k_s 重新缓存当前 group metadata。
```

NVFP4 和 MoE MXFP4 已经使用 `ThreadMap::Iterations::kCount` 大小的 metadata
cache，并按每个 `s` 单独缓存 group；这两条路径不需要同样改动。dense MXFP4
原先仍是 `kContiguous` 大小 cache，因此跟随 finite group 修复，改为在每个
`s` 使用前刷新当前 group metadata。

## 后续选择器说明

selector 应输出完整 shape：

```text
CTA_M, CTA_N, CTA_K, Warps, WarpM, WarpN, WarpK, split_k,
UseMetadataVectorWords
```

Dense selector 输入：

```text
quant_format, group_size, size_m, size_n, size_k
```

MoE selector 输入：

```text
quant_format, group_size, moe_block_size, top_k, size_m, size_n, size_k
```

generic fallback 不追求全局最优。它只是 unmatched shape 的稳定基线。
model-specific 或 quant-specific selector 可以在它之上叠加，不需要改变 op
schema 或 repack contract。
