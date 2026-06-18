# 2026-05-14 Marlin 推理业务 M/N/K 边界分析

## 结论

后续 dense SM70 int4 Marlin 的主性能目标建议收窄到：

```text
M >= 128
N % 256 == 0
K % 32 == 0
N >= 1024
K >= 1024
group_size in {-1, 32, 64, 128}
```

主线 tile 建议优先围绕：

```text
CTA = 128x256x32
Warps = 8
```

这个边界覆盖普通 LLM block linear 的主要高吞吐场景：

- prefill
- chunked prefill
- 大 batch decode
- hidden / intermediate / fused MLP / fused QKV 等规整维度

`residue` 路径保留正确性和 API 覆盖即可，不应作为主要性能目标。除非线上模型形状证明大量命中 `N % 256 != 0` 或 `K % 32 != 0`，否则不要为 residue 开发复杂优化。

## GEMM 语义

Marlin dense 看到的 GEMM 通常是：

```text
A[M, K] * W[K, N] -> C[M, N]
```

其中：

```text
M = 本次 linear 一起处理的 token 数
K = 输入通道数，通常是 hidden_size 或 intermediate_size
N = 输出通道数，通常是 hidden_size / intermediate_size / fused projection 输出
```

因此：

- `M` 主要由推理调度方式决定。
- `K/N` 主要由模型结构决定。

## K/N 常见形状

Transformer block 内部 dense linear 的 `K/N` 通常很规整：

| 层 | K | N |
|---|---:|---:|
| `q_proj` | hidden | hidden |
| `k_proj/v_proj` | hidden | kv_hidden，GQA 下可能小于 hidden |
| fused `qkv` | hidden | q_out + k_out + v_out |
| `o_proj` | hidden | hidden |
| `gate_proj/up_proj` | hidden | intermediate |
| fused `gate_up` | hidden | 2 * intermediate |
| `down_proj` | intermediate | hidden |
| `lm_head` | hidden | vocab_size |

常见模型维度：

| 模型级别 | hidden | intermediate | 观察 |
|---|---:|---:|---|
| 1B-3B | 2048 / 2560 / 3072 | 5504 / 6912 / 8192 等 | 多数满足 64 或 256 对齐 |
| 7B | 4096 | 11008 | `11008 = 43 * 256` |
| 13B | 5120 | 13824 | hidden/intermediate 都规整 |
| 30B-34B | 6656 / 7168 / 8192 | 17920 / 22016 / 28672 等 | 通常 256 对齐 |
| 70B | 8192 | 28672 | hidden/intermediate 都规整 |

对普通 block linear，可以近似认为：

```text
K % 32 == 0 几乎总是成立
N % 64 == 0 几乎总是成立
N % 256 == 0 很常见
```

`N % 256 != 0` 主要可能来自：

- `lm_head` / vocab projection
- 特殊模型宽度
- 某些未 fused 的小 GQA projection

这些不应默认成为当前 dense Marlin 主优化目标。

## M 常见形状

`M` 是区分 prefill 和 decode 的关键。

### Prefill

prefill 中：

```text
M = sum(prompt_tokens_in_batch)
```

常见范围：

| 场景 | M |
|---|---:|
| 单请求短 prompt | 32 - 512 |
| 单请求长 prompt | 1024 - 8192+ |
| batch prefill | 512 - 8192+ |
| chunked prefill | 常见 cap 为 512 / 1024 / 2048 / 4096 |

prefill 特点：

```text
M 大
K/N 大
GEMM 更接近 compute-bound
大 tile 更容易发挥 Tensor Core 吞吐
```

因此，`128x256x32 / 8 warp` 是合理的主优化目标。

### Decode

decode 中每一步每个 sequence 通常只生成 1 个 token：

```text
M = 当前 step 的 active sequence 数
```

常见范围：

| 场景 | M |
|---|---:|
| 单请求低延迟 decode | 1 |
| 小 batch decode | 2 - 32 |
| 中等 batch decode | 32 - 256 |
| 高吞吐 continuous batching | 256 - 1024+ |
| speculative / multi-token decode | batch * draft_tokens，可能比普通 decode 更大 |

decode 特点：

```text
M 小或中等
kernel launch / memory / metadata overhead 更明显
CTA_M=128 时 M<128 会浪费 M 方向计算资源
```

如果业务主要是高吞吐 decode，`M >= 128` 时仍可复用主线大 tile。  
如果业务主要是低延迟 decode，`M = 1/2/4/8/16/32`，应单独设计 small-M kernel，不建议把复杂度塞进当前 dense 主线。

### MoE

MoE 不能直接套 dense 的 `M` 边界。每个 expert 看到的是：

```text
M_expert = 被路由到该 expert 的 token 数
```

特点是：

```text
expert 间 M 极不均匀
很多 expert M=0 或很小
少数 expert M 中等
```

因此 MoE 更接近 small-M / grouped GEMM 问题。dense `128x256x32 / 8 warp` 的业务边界不能直接迁移到 MoE。

## 建议开发边界

### 第一优先级：dense full-tile 主路径

```text
M >= 128
N % 256 == 0
K % 32 == 0
CTA = 128x256x32
Warps = 8
```

目标场景：

- prefill
- chunked prefill
- 大 batch decode
- block 内部规整 linear

这个范围能自然保证：

```text
ThreadMap::Iterations::kContiguous == 4
qweight full-tile 可 uint4 load
N residue 不进入主热路径
K tile 逻辑简单
```

### 第二优先级：中等 M decode 候选

```text
64 <= M < 128
N % 256 == 0
K % 32 == 0
CTA = 64x256x32
Warps = 4
```

这个组合适合业务上 decode batch 经常落在 `64-128` 的情况。  
建议作为单独 template specialization 评估，不要与 `128x256x32 / 8 warp` 混成复杂 runtime path。

### 第三优先级：correctness-only residue

```text
N % 64 == 0 but N % 256 != 0
K % 32 == 0
```

这类路径应保留正确性，但默认不追求 full-tile 同等级性能。  
原因是普通 large-N block linear 中，只有最后一个 N tile 会走 residue，整体占比通常很小。

## 暂不建议开发的范围

没有明确业务需求和 benchmark 证明前，不建议优先开发：

```text
M < 64 的 dense Marlin 高性能专用路径
K % 32 != 0 的优化路径
N % 64 != 0 的 dense int4 路径
CTA_K = 64 / 128 的量化 IteratorB 主路径
kCtaN = 64 / 128 还要求 uint4 qweight load 的路径
为 lm_head / vocab projection 专门调优
把 MoE small-M 问题混进 dense 主路径
```

这些场景不是没有价值，而是开发成本高、会显著增加路径复杂度。除非线上业务明确命中，否则不应牺牲当前 dense 主线的简洁性和可维护性。

## 与 qweight uint4 约束的关系

如果目标是 full-tile qweight `uint4` load，当前 int4 macro-N repack layout 下需要：

```text
kCtaN = 256
ThreadMap::Iterations::kContiguous = 4
ThreadMap::Delta::kContiguous = 64
ThreadMap::kElementsPerAccess = 8
```

推荐只开放：

```text
(kCtaM, kCtaN, kCtaK, kWarps) in {
  (64,  256, 32, 4),
  (128, 256, 32, 8),
}
```

其中主线优先：

```text
128x256x32 / 8 warp
```

`64x256x32 / 4 warp` 可作为中等 M decode 候选，但不应默认增加到主线，除非 benchmark 证明它服务真实业务。

## 最终建议

当前阶段把 dense SM70 int4 Marlin 的开发边界写死为：

```text
Primary performance target:
  M >= 128
  N % 256 == 0
  K % 32 == 0
  CTA = 128x256x32
  Warps = 8

Correctness coverage:
  N % 64 == 0
  K % 32 == 0
  residue path exists but is not the main performance target

Out of scope unless production requires it:
  M < 64 dense small-M
  K residue
  N not divisible by 64
  MoE small-M routing specialization
  lm_head/vocab-specific tuning
```

这能避免在没有业务需求的情况下开发大量组合，同时保留对主流 prefill / chunked prefill / 吞吐型 decode 的优化空间。
