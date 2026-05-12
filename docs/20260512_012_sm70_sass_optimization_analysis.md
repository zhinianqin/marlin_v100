# 2026-05-12 SM70 SASS 优化空间分析结论

## 本轮性质

本轮只记录 SASS 形态分析结论，不修改 CUDA kernel。

目标是判断当前 dense `uint4` / `uint4b8` 路径距离 pure GEMM
`90+ TFLOPs` baseline 还有没有可继续优化的空间，并明确后续优先级。

## 分析对象

当前重点对象：

- pure GEMM probe: `sm70_cutlass_threadblock_gemm_kernel<128,256,32,8>`
- pure GEMM probe: `sm70_cutlass_threadblock_gemm_kernel<128,128,32,4>`
- dense `uint4b8`: `sm70_marlin_u4b8_gemm_kernel<-1,true>`
- dense `uint4b8`: `sm70_marlin_u4b8_gemm_kernel<128,true>`
- dense `uint4`: `sm70_marlin_u4_gemm_kernel<-1,true>`
- dense `uint4`: `sm70_marlin_u4_gemm_kernel<128,true>`

使用下面方式定位函数名：

```bash
/usr/local/cuda-12.8/bin/cuobjdump -ltext python/marlin_v100/_C.abi3.so | \
  rg "sm70_marlin_u4|sm70_cutlass_threadblock"
```

然后按单个 `--function` dump SASS 并做指令粗计数，避免旧 Marlin 模板和当前
SM70 CUTLASS kernel 混在一起。

## SASS 粗计数

| kernel | static inst | HMMA | LOP3 | SHF | IMAD | LDG | STS | HMUL2 | HFMA2 | HADD2 | REG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pure `128x256x32/8` | 1344 | 512 | 52 | 36 | 92 | 12 | 80 | 0 | 0 | 0 | 216 |
| pure `128x128x32/4` | 1416 | 512 | 60 | 39 | 111 | 16 | 88 | 0 | 0 | 0 | 236 |
| `uint4b8<-1,true>` | 1528 | 512 | 91 | 49 | 138 | 24 | 88 | 32 | 16 | 18 | 247 |
| `uint4b8<128,true>` | 1576 | 512 | 95 | 53 | 147 | 32 | 88 | 32 | 16 | 18 | 254 |
| `uint4<-1,true>` | 1568 | 512 | 99 | 52 | 141 | 25 | 88 | 0 | 60 | 22 | 246 |
| `uint4<128,true>` | 1664 | 512 | 111 | 69 | 153 | 34 | 88 | 0 | 72 | 27 | 255 |

资源计数来自：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4(_gemm_kernel|b8_gemm_kernel)"
```

## 主要结论

### 1. Shared-memory layout 不是当前第一瓶颈

`LDS=100` 与 `BAR=10` 在 pure GEMM 和 `uint4` / `uint4b8` 路径中基本一致。
`STS` 在当前量化路径中为 `88`，与 pure `128x128x32/4` 一致，只比 pure
`128x256x32/8` 多 `8` 条 static store 指令。

这说明当前 B operand 写入 CUTLASS predefined
`RowMajorVoltaTensorOpMultiplicandBCongruous<16>` shared-memory layout 的总体指令
形态是正常的。SASS 本身不能直接证明没有 bank conflict；如果后续要确认 bank
conflict，需要用 Nsight Compute 查看 shared load/store bank conflict 相关指标。

### 2. `uint4b8` 没有明显快过 `uint4` 的原因在 hot path

`uint4b8` 理论上没有 qzero metadata，应该比 `uint4` 更轻。但当前 SASS 显示
`uint4b8` hot path 仍包含：

- unpack 相关 `LOP3` / `SHF`
- bias-8 dequant 相关 `HADD2` / `HFMA2`
- scale multiply 相关 `HMUL2`

其中 `uint4b8<-1,true>` 仍有 `HMUL2=32`、`HFMA2=16`、`HADD2=18`。

而 `uint4` 已经通过 `cached_bias = -zero * scale` 把 hot path 折成
`HFMA2(q, scale, bias)`，因此虽然多了 zero metadata，但 `group_size=-1` 场景里
metadata 只刷新一次，实际性能可以贴近甚至略高于 `uint4b8`。

### 3. `skip_flop=true` / bias-cache 的 `uint4b8` 路径已试验并丢弃

曾尝试让 `uint4b8` 使用：

```cpp
dequant<half2, vllm::kU4B8.id(), true>
cached_bias = -8 * scale
frag = __hfma2(q_unsigned, scale, cached_bias)
```

该方向的目标是消除 hot path 中的 `HMUL2` 与部分 `HADD2`，把 `uint4b8`
转换为类似 `uint4` 的 fused `HFMA2` 形态。

实际试验结论：精度失败，因此该候选不保留。后续不要把这条路径作为正式工程
实现方向，除非先重新证明其数学语义和 reference 完全一致。

### 4. 单靠当前 `128x128x32/4` 形态很难达到 90 TFLOPs

pure GEMM 已验证超过 `90 TFLOPs` 的主形态是：

```text
CTA=128x256x32, warps=8
```

当前 dense `uint4` / `uint4b8` 主 kernel 仍是：

```text
CTA=128x128x32, warps=4
```

虽然 static SASS 中 HMMA 都是 `512`，但动态 tile 覆盖和 CTA 调度完全不同。
当前量化路径即使继续压掉部分 `IteratorB::load_full_tile()` 指令，也更可能只是接近
pure `128x128x32/4` 的性能上限，而不会自然达到 pure `128x256x32/8` 的
`90+ TFLOPs`。

### 5. `128x256x32/8` 不能直接套当前 IteratorB

当前 iterator 围绕 `kCtaN=128` 和两个 64-column contiguous access 设计。
如果直接把 CTA_N 改成 256，CUTLASS B thread map 很可能增加 contiguous access
数量，导致：

- `qweight_offsets_` 数量增加
- `cached_scales_` 或 `cached_bias_` 生命周期变长
- register 接近或超过 255
- 重新出现 spill 风险

因此冲击 `90 TFLOPs` 需要新写 `128x256x32/8` 专用 `IteratorB`，而不是只改
`kCtaN` / `kWarps` 常量。

## 后续优化排序

### 第一优先级：`uint4b8 group_size=-1` scale policy

`uint4b8 group_size=-1` 的正式工程优化仍应优先实现虚拟 `128` refresh cadence：

- 公共语义保持 single scale group。
- 不扩展 `b_scales` metadata。
- kernel 内部 cache key 按 `logical_k / 128` 变化。
- scale row 实际仍读第 0 行，或使用等价的 `scale_group_stride=0` policy。

此前临时实验已经证明该方向可以把 `group_size=-1` 从约 `69.6 TFLOPs` 提升到
约 `75.0 TFLOPs`，属于低风险、明确有效的工程优化。

### 第二优先级：`128x256x32/8` 专用量化 IteratorB

要继续接近 `90 TFLOPs`，需要把量化 B dequant-to-smem 路径迁移到 pure GEMM 已验证的
`128x256x32/8` 主形态。

关键约束：

- 不使用 `cutlass::gemm::device::Gemm`。
- 继续使用 CUTLASS threadblock / Volta TensorOp 组件。
- B 仍然是 packed global load -> register dequant -> B-congruous smem -> MMA。
- 设计 streaming iterator，避免 256N 的 qweight/scale metadata 同时长生命周期活跃。
- 硬门槛：`STACK=0`、无 spill。

建议先只做 `uint4b8 group_size=-1` 的 `128x256x32/8` 原型，因为它没有 qzero，
最适合验证 tile/iterator 结构是否能回到 pure GEMM 的高吞吐区间。

### 第三优先级：qweight repack 重新设计

SASS 中 `LOP3` / `SHF` 相比 pure GEMM 明显增加，说明 unpack 仍有成本。
但 qweight repack 会影响 helper/reference 和两套 repack kernel，风险高于 scale
policy 与 tile iterator。

只有在完成前两步后仍卡在 `80-85 TFLOPs` 区间，才建议重新设计 qweight physical
layout，目标是：

- 减少 `qword >> 8` 和相关 `SHF`
- 让 low/high nibble 顺序更贴近 `frag_vec[0..3]`
- 让 unpack 后写入 CUTLASS `SmemIteratorB` 的顺序更短

## 当前判断

还有继续优化的空间，但空间主要不在局部微调 `load_full_tile()` 的几行 half2 代码。

短期可做的是 `uint4b8 group_size=-1` scale policy，把已经验证有效的临时实验正式化。
中期要冲击 `90 TFLOPs`，必须开发 `128x256x32/8` 专用量化 `IteratorB`，并用
SASS / ptxas / benchmark 同时确认它没有把 register 和 spill 推爆。

`skip_flop=true` 的 `uint4b8` fused bias-cache 路径已因精度失败丢弃。
