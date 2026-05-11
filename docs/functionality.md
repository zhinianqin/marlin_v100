# CUTLASS Functionality Notes for SM70 Marlin

本文件只记录 `marlin_v100` 当前重构需要的 CUTLASS 功能结论，不复制 CUTLASS 仓库的完整文档。完整原始资料以外部 `CUTLASS_DIR` 为准，默认路径为 `/root/source/repos/cutlass`。

## 关键结论

- V100 / SM70 的 fp16 Tensor Core 主路径应使用 `OpClassTensorOp`，不是 `WmmaTensorOp`。
- SM70 half GEMM 的核心 instruction shape 是 `8x8x4`。
- CUTLASS SM70 half TensorOp 默认高性能形状集中在：
  - `ThreadblockShape = 128x256x32`
  - `WarpShape = 64x64x32`
  - `InstructionShape = 8x8x4`
  - `Stages = 2`
- 大尺寸饱和 GEMM 已验证 `128x256x32/8` 能达到 `92+ TFLOPs`，可以作为后续 dequant-to-shared 的基线。

## TensorOp 8x8x4 Shared-Memory Layout

Volta TensorOp 必须从 permuted/shared-memory layout 读取 operand，不能用普通 row-major/column-major shared tile 直接喂给 MMA。当前 probe 中需要重点锁定以下对应关系：

| Operand | GMEM Layout | SMEM Layout in CUTLASS docs | CUTLASS SM70 type used by probe |
| --- | --- | --- | --- |
| A | RowMajor | `RowMajorVoltaTensorOpCrosswise<16>` | `RowMajorVoltaTensorOpMultiplicandCrosswise<16, CTA_K>` |
| B | RowMajor | `RowMajorVoltaTensorOpCongruous<16>` | `RowMajorVoltaTensorOpMultiplicandBCongruous<16>` |
| A | ColumnMajor | `ColumnMajorVoltaTensorOpCongruous<16>` | 可作为后续对照 |
| B | ColumnMajor | `ColumnMajorVoltaTensorOpCrosswise<16>` | 不进入当前 pure B 主路径 |

当前已选路径是 A/B 都按 row-major global tensor 输入，B 由 CUTLASS iterator 写入 `RowMajorVoltaTensorOpMultiplicandBCongruous<16>` 对应的 shared-memory 布局。后续量化 B 不是从 dense fp16 global B 直接搬运，而是在寄存器中完成 unpack / dequant / scale / zero-point 后写入同一类 predefined Volta shared-memory layout。

## 对 Marlin 重构的约束

- 不调用 `cutlass::gemm::device::Gemm` 黑盒；只复用 CUTLASS threadblock/warp/epilogue 组件。
- B operand 的最终来源必须是 shared memory，因为真实 Marlin B 是量化权重 dequant 后的 half tile。
- B 的 packed global load 应保持合并访存，寄存器生命周期要短，dequant 后直接写入 TensorOp 需要的 shared-memory layout。
- 非纯 B proxy 路径不再保留。后续若要测试转置/布局变化，应发生在 shared-memory 写入或读取侧，并用 ptxas register、spill 与 TFLOPs 同时决策。
- A 路径暂时沿用 CUTLASS `IteratorA -> SmemIteratorA`，因为它已经具备 128-bit global load 和 Volta crosswise shared-memory layout。

## 后续 Benchmark 关注点

- `128x256x32/8`：大尺寸主基线。
- `128x128x32/4`、`128x64x32/4`：中小尺寸备选。
- 指标必须同时记录：
  - correctness vs `torch.mm`
  - median latency
  - TFLOPs 与 V100 125 TFLOPS 峰值比例
  - ptxas registers
  - spill stores / spill loads
  - 是否破坏 packed B 的合并读取
