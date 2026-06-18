# 2026-05-12 SM70 uint4b8 group_size=-1 分路径实验结论

## 本轮性质

本轮只用于获取结论，不作为正式工程实现。

临时改动过 `csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu`，用于验证
`uint4b8 group_size=-1` 的性能瓶颈来源。实验结束后已经恢复源码并重新构建，
最终提交只保留本文档，不提交 CUDA 实现改动。

## 问题背景

在 dense `uint4b8` 路径中，`group_size=-1` 的性能明显低于其它 group size。
此前 quick benchmark 中同一典型形状 `5120x4096x4096` 的结果如下：

| group_size | kernel_like TFLOPs |
| ---: | ---: |
| `-1` | 69.62 |
| `32` | 71.62 |
| `64` | 74.07 |
| `128` | 75.28 |

其中 `group_size=-1` 理论上 scale metadata 最少，不应该天然比
`group_size=64/128` 更慢，因此怀疑点集中在 kernel 的 group-size-specific
实现路径，而不是量化权重本身。

## 非修改诊断

先做了一个不改源码的诊断实验：

- 使用同一份 `group_size=-1` 量化得到的 `q_weight`。
- 将同一行 scales 重复成 32 行，让 host dispatch 强制走 `group_size=128`
  kernel。
- 比较原 `group_size=-1` 路径和 repeated-scales `128` 路径输出。

结果：

```text
max_diff = 0.0
group_size=-1 原路径: 2513.41 us, 68.35 TFLOPs
same qweight + repeated scales + 128 路径: 2319.36 us, 74.07 TFLOPs
```

这个结果说明：

- 同一份 `q_weight` 走 `128` 路径后可以明显变快。
- qweight repack 主布局不是当前第一瓶颈。
- 首要问题更可能是 `GroupSize=-1` 生成了更差的 iterator / scale cache /
  codegen 路径。

## 临时实验内容

临时实验版本只改 dense `uint4b8`，没有修改公共 Torch op schema，也没有修改
Python wrapper 参数。

主要实验点：

- 给 `Sm70U4B8IteratorB` 增加 group-size-specific scale policy。
- `GroupSize=-1` 走虚拟 `128` refresh cadence：
  - cache key 使用 `logical_k / 128`
  - scale row 语义仍读第 0 行
  - runtime `scale_group_stride=0`，不实际扩展 scale metadata
- `GroupSize=64/128` 保持真实 group row。
- `GroupSize=32` 临时改为 cache-per-CTA-K，测试是否优于原 direct-scale
  特殊路径。
- 额外实验 `FullTile` 模板分裂：
  - full-tile 删除 per-access bounds checks
  - residue 保留边界检查

这次没有改：

- qweight 主布局
- `_SM70_U4_PACK_ORDER`
- `gptq_marlin_repack.cu`
- `awq_marlin_repack.cu`
- 公共 op schema
- MoE

## 临时实验资源结果

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

临时实验版本的 `sm70_marlin_u4b8_gemm_kernel` 资源：

| specialization | REG | STACK | LOCAL | spill |
| --- | ---: | ---: | ---: | ---: |
| `<128, 128, false>` | 250 | 0 | 0 | 0 |
| `<64, 64, false>` | 250 | 0 | 0 | 0 |
| `<32, 32, false>` | 250 | 0 | 0 | 0 |
| `<-1, 128, false>` | 250 | 0 | 0 | 0 |
| `<128, 128, true>` | 255 | 0 | 0 | 0 |
| `<64, 64, true>` | 255 | 0 | 0 | 0 |
| `<32, 32, true>` | 255 | 0 | 0 | 0 |
| `<-1, 128, true>` | 255 | 0 | 0 | 0 |

结论：

- 临时 full/residue 两套实例都没有 stack/spill。
- `FullTile=true` 比 `FullTile=false` 多 5 个寄存器。
- uint4b8 原实现本身也已经是 `STACK=0`，所以 full-tile 静态化没有带来资源下降，
  不是这次性能提升的主要来源。

恢复源码并重新构建后的原实现资源：

| specialization | REG | STACK | LOCAL |
| --- | ---: | ---: | ---: |
| `<128>` | 250 | 0 | 0 |
| `<64>` | 250 | 0 | 0 |
| `<32>` | 252 | 0 | 0 |
| `<-1>` | 247 | 0 | 0 |

## 验证结果

临时实验版本定向测试：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference
```

结果：

```text
12 passed in 1.98s
```

实验结束后恢复源码并重新构建：

```bash
git checkout -- csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu
./build.sh
```

恢复后导入：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：

```text
imports ok
```

## Benchmark 结果

设备：

```text
Tesla V100-SXM2-32GB, sm70
```

命令：

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4b8 --group-sizes -1 32 64 128 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

临时实验版本结果：

| group_size | operator TFLOPs | kernel_like TFLOPs |
| ---: | ---: | ---: |
| `-1` | 74.15 | 75.00 |
| `32` | 73.55 | 72.85 |
| `64` | 75.61 | 74.33 |
| `128` | 76.04 | 75.20 |

与原实现对比：

| group_size | 原 kernel_like TFLOPs | 临时实验 kernel_like TFLOPs | 变化 |
| ---: | ---: | ---: | ---: |
| `-1` | 69.62 | 75.00 | +7.7% |
| `32` | 71.62 | 72.85 | +1.7% |
| `64` | 74.07 | 74.33 | +0.4% |
| `128` | 75.28 | 75.20 | -0.1% |

最关键现象是 `group_size=-1` 从约 `68.5-69.6 TFLOPs` 直接追到
`75.00 TFLOPs`，几乎等于 `group_size=128` 的 `75.20 TFLOPs`。

## 结论

`uint4b8 group_size=-1` 的主要瓶颈不是 qweight repack 主布局，而是
`GroupSize=-1` 的 iterator / scale policy / codegen 路径。

有效方向是给 `group_size=-1` 单独做虚拟 `128` scale refresh cadence：

- scale metadata 语义仍然是单 group。
- 不在公共接口层扩展 scales。
- 不复制 metadata。
- kernel 内部让 cache key 按 `logical_k / 128` 变化。
- scale row 实际仍映射到第 0 行，或使用等价的 `scale_group_stride=0` policy。

`FullTile` 模板分裂对 uint4b8 暂时不构成主要优化点：

- 原实现已经 `STACK=0` 且无 spill。
- 实验中 `FullTile=true` 反而增加到 `REG=255`。
- 后续如果要保留 full-tile 分裂，需要先用 SASS 或更细 benchmark 证明
  bounds check 指令成本确实超过寄存器压力。

`group_size=32` 的 cache-per-CTA-K 版本只有小幅收益，需要独立复测。它不应该和
`group_size=-1` 的虚拟 128 policy 绑在同一笔大改里。

## 下一步建议

正式工程实现建议拆成小步：

1. 只实现 `uint4b8 group_size=-1` 的虚拟 `128` scale policy，不改 qweight repack。
2. 保持 `group_size=64/128` 路径不变，避免影响当前最高性能路径。
3. `group_size=32` 单独做 A/B：
   - 原 direct-scale 特殊路径
   - cache-per-CTA-K 路径
   只有稳定高于原路径才保留。
4. 暂不引入 uint4b8 full-tile 模板分裂，除非后续 SASS 证明它能抵消寄存器增加。
5. 如果虚拟 128 policy 落地后 `group_size=-1` 仍比 `128` 慢超过 3%，再考虑
   scale-only repack，例如让同一 CTA_N 内 `n` 与 `n + 64` 的 scale chunk 更靠近；
   不要优先改 qweight repack。

