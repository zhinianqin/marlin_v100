# SM70 Marlin Auto Params Selector 实现规则

本文记录 SM70 Marlin auto params selector 的当前实现约定。后续为具体模型补充
selector 时，应以本文和源码为准，不应直接套用历史 benchmark 文档里的旧策略。

## 决策优先级

Dense 和 MoE auto params 都按同一优先级决策：

1. env override 优先。
2. 模型专用 selector 第二。
3. 默认 fallback 最后。

也就是说，任何非空 env 都会完全跳过 selector。selector 只在没有 env override
时参与决策。

## Env 规则

Dense env 名称保持不变：

```text
SM70_MARLIN_DENSE_CTA_GEOMETRY
SM70_MARLIN_DENSE_SPLIT_K
SM70_MARLIN_DENSE_METADATA_CACHE
```

MoE env 名称保持不变：

```text
SM70_MARLIN_MOE_CTA_GEOMETRY
SM70_MARLIN_MOE_SPLIT_K
SM70_MARLIN_MOE_METADATA_CACHE
```

三项 env 中任一项非空就触发 env path。env path 从当前默认 params 起步，
只覆盖已经设置的字段，未设置字段继续使用默认值。完成解析和校验后立即
return，不进入 selector。

空字符串按未设置处理。非法 env 值必须继续抛出显式错误，便于 env sweep 和
benchmark 判断 `REJECTED` 状态。

## Selector 命名规则

selector 使用模型标识的 lowercase snake_case 转写作为函数名的一部分：

```text
/ -> _
. -> _
- -> _
```

示例：

```text
QuantTrio/Qwen3.6-27B-AWQ -> quanttrio_qwen3_6_27b_awq
```

当前预留的 selector 是：

```cpp
sm70_marlin_dense_try_select_quanttrio_qwen3_6_27b_awq_params(...)
sm70_marlin_moe_try_select_quanttrio_qwen3_6_35b_a3b_awq_params(...)
```

不要新增模糊的通用 selector 名称，例如 `try_select_model_shape_params`、
`try_select_small_m_params` 或 `try_select_block_shape_params`。需要新增规则时，
优先新增模型专用 selector，或者在已有模型 selector 内补充更具体的条件。

## Selector 编写规则

selector 必须是小型 `inline bool` 函数，签名遵循当前 Dense/MoE context：

```cpp
inline bool sm70_marlin_dense_try_select_<model_id>_params(
    Sm70MarlinDenseAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params);

inline bool sm70_marlin_moe_try_select_<model_id>_params(
    Sm70MarlinMoeAutoParamsContext const& ctx,
    Sm70MarlinAutoParams& params);
```

selector 只基于 context 判断，不直接读取 env。命中时必须写完整
`Sm70MarlinAutoParams`，包括：

```text
geometry
requested_split_k
use_metadata_vector_words
packed_macro_n
```

命中后返回 `true`，未命中返回 `false`。selector 不做 CUDA launch，不分配
tensor，不调用 benchmark 逻辑，不引入 Python 依赖。

同一个 selector 内部，规则顺序应保持更具体优先、更通用靠后。条件要保持可读，
不要把大段 benchmark 表原样塞进一个难以审查的表达式。

## Dense / MoE 边界

Dense selector 只能挂在 `sm70_marlin_dense_auto_params(...)` 的 selector 链上。
MoE selector 只能挂在 `sm70_marlin_moe_auto_stage_params(...)` 的 selector 链上。

MoE selector 不得返回 `CTA_M=128/256`。这两个 CTA_M 是 dense-only geometry。
MoE 当前只允许 `CTA_M=32/64`。

## 合法性要求

`packed_macro_n` 必须来自 `size_n`，不要在 selector 里另行推断或硬改。当前
公共入口通过 `sm70_marlin_auto_packed_macro_n(size_n)` 计算它。

selector 返回的 geometry 必须是当前 dispatch 已实例化的完整 7-field geometry：

```text
CTA_M x CTA_N x CTA_K x Warps x WarpM x WarpN x WarpK
```

返回 params 必须满足：

```text
requested_split_k in {1,2,4,8}
packed_macro_n in {64,128,256}
packed_macro_n % geometry.cta_n == 0
```

selector 命中后会通过公共 validation 检查 geometry、`requested_split_k`、
`packed_macro_n` 和 CTA_N 对齐关系。

selector 不应绕过现有 downstream validation。Dense 入口仍会检查 Dense geometry
和 `size_n` 对齐；MoE 入口仍会检查 MoE geometry、`size_n` 对齐和
`size_k % CTA_K == 0`。

## 添加新 Selector 的步骤

1. 先用 model shape table 或 benchmark 产出候选规则，并记录模型来源。
2. 新增一个模型专用 selector 函数，或在对应模型 selector 内补充规则。
3. 在 Dense 或 MoE auto params 入口里追加显式 `if (...) return params;`。
4. 保持 env path 在 selector 之前，保持默认 fallback 在 selector 之后。
5. 运行静态检查、构建、导入检查和最小 runtime smoke。

推荐检查命令：

```bash
git diff --check
./build.sh
PYTHONPATH=$PWD ./.venv/bin/python - <<'PY'
import torch
import vllm._C
import vllm._moe_C

assert hasattr(torch.ops._C, "marlin_gemm")
assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm")
print("marlin imports ok")
PY
```

最小 runtime smoke 可按任务需要运行 Dense 或 MoE 的 `env_smoke_single_shape`。

## 回写说明

本文档属于本地策略文档，不默认回写上游主树。上游回写仍只依据
`upstream_map.yaml` 中列出的源码映射执行。

历史日期型 benchmark / probe / bugfix 文档保留实验语义，不自动成为当前
selector 事实来源。当前事实以源码、测试 helper 和本文为准。
