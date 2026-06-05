# 20260605 SM70 Marlin Logical Metadata Layout Fix

## 结论

SM70 Marlin 新 CUTLASS kernel 的 metadata ABI 是 **logical N contiguous**：

```text
dense scales / float zero-point: [groups, N]
moe scales / float zero-point:   [experts, groups, N]
```

因此 Python 回写 class 在 `process_weights_after_loading` 或对应 prepare helper
中，不能再把 `scales` 和 float zero-point metadata 转成 legacy
`marlin_permute_scales` / `marlin_moe_permute_scales` layout。正确修复是让所有
SM70 回写路径输出 logical N contiguous metadata，并保留 CUDA kernel 对 metadata
的连续读取 hot path。

之前尝试的 CUDA-side gather 修复方向不正确：它让每次 metadata 读取都从
logical N 映射到 permuted storage N，会破坏 SM70 kernel 当前设计的连续 metadata
访问模式，并带来明显性能风险。该方案已回退。

## 回写 Class 范围

本轮范围从 commit `abeccd8fede450dc3e818998abf276040ac4ef31` 重新确认。权威
来源是该 commit 中的：

- `docs/20260603_049_upstream_py_writeback_review.md`
- `docs/20260603_050_full_py_writeback_test_validation.md`
- `upstream_map.yaml`
- `tests/writeback_marlin_cases.py`

Dense 必须覆盖 8 个 class：

| Class | Quant 范围 | ZP 范围 | 说明 |
| --- | --- | --- | --- |
| `MarlinLinearKernel` | `uint4,uint8,uint4b8,uint8b128,fp8,float4_e2m1f` | `uint4,uint8` | Dense mixed precision 主 kernel surface |
| `GPTQMarlinLinearMethod` | `uint4b8,uint8b128` | none | 通过 `MarlinLinearKernel` |
| `AWQMarlinLinearMethod` | `uint4,uint8` | `uint4,uint8` | 通过 `MarlinLinearKernel` |
| `CompressedTensorsWNA16` | `uint4,uint4b8,uint8,uint8b128` | `uint4,uint8` | 通过 `MarlinLinearKernel` |
| `MarlinFP8ScaledMMLinearKernel` | `fp8` | none | 通过 FP8 prepare helper |
| `CompressedTensorsW8A16Fp8` | `fp8` | none | 通过 FP8 prepare helper |
| `CompressedTensorsW4A16Fp4` | `nvfp4` | none | 通过 FP4 prepare helper |
| `CompressedTensorsW4A16Mxfp4` | `mxfp4` | none | 通过 FP4 prepare helper |

MoE 必须覆盖 7 个 class：

| Class | Quant 范围 | ZP 范围 | 说明 |
| --- | --- | --- | --- |
| `GPTQMarlinMoEMethod` | `uint4b8,uint8b128` | none | MoE WNA16 GPTQ |
| `AWQMarlinMoEMethod` | `uint4,uint8` | `uint4,uint8` | MoE AWQ asymmetric |
| `CompressedTensorsWNA16MarlinMoEMethod` | `uint4b8,uint8b128` | none | CT WNA16 MoE |
| `QuarkW8A8Fp8MoEMethod` | `fp8` | none | 通过 FP8 MoE prepare helper |
| `CompressedTensorsW8A8Fp8MoEMethod` | `fp8` | none | 通过 FP8 MoE prepare helper |
| `CompressedTensorsW4A4Nvfp4MoEMethod` | `nvfp4` | none | 通过 FP4 MoE prepare helper |
| `CompressedTensorsW4A4Mxfp4MoEMethod` | `mxfp4` | none | 通过 FP4 MoE prepare helper |

`MarlinExpertsBase` / `MarlinExperts` / `BatchedMarlinExperts` 是 MoE modular
execution helpers，不是独立 writeback class case；它们消费上述 MoE method 已经
准备好的 `scales` / ZP metadata，因此必须保持 ABI 兼容。

## 根因

SM70 kernel 中 metadata 读取按 logical output column 连续访问：

```cpp
scales[group * size_n + n]
zeros[group * size_n + n]
```

这要求 Python 端传入的 `b_scales` / float `b_zeros` 在 N 维就是 logical order。
但部分回写路径沿用了 legacy Marlin ABI：

```text
b_scales = marlin_permute_scales(scales, K, N, group_size, ...)
b_zeros  = marlin_permute_scales(zp * scales, K, N, group_size, ...)
```

MoE 路径也有等价的 expert 维版本：

```text
b_scales[e] = marlin_moe_permute_scales(scales[e], ...)
b_zeros[e]  = marlin_moe_permute_scales(zp[e] * scales[e], ...)
```

这样会让 SM70 kernel 在 logical column `n` 上读到 legacy permuted storage column
`n`，导致 scale 和 float zero-point 错列。对 AWQ U4/U8 ZP 来说，这会直接把
反量化公式中的 `(q - zp) * scale` 变成错误列的 metadata，模型级表现为随机英文、
符号或乱码输出。

同一个 ABI 错误不只影响 AWQ / U4 ZP。所有 SM70 kernel 使用的 scale metadata
都要求 logical N contiguous，因此影响面按 metadata path 判定：

- Dense WNA16 scale-only 和 U4/U8 float ZP。
- Dense FP8 / NVFP4 / MXFP4 scale metadata。
- MoE WNA16 scale-only 和 AWQ U4/U8 float ZP。
- MoE FP8 / NVFP4 / MXFP4 scale metadata。

不属于本次 bug 的范围：

- qweight repack layout。
- `a_scales` / activation scale。
- `global_scale` scalar。
- packed integer ZP。SM70 当前仍拒绝该路径。
- routing / top-k / fused MoE dispatch。

## 修复逻辑

本轮保留 legacy helpers，但把 SM70 production prepare path 显式切到新 helper：

```python
sm70_marlin_logical_scales(s, size_k, size_n, group_size, is_a_8bit=False)
sm70_marlin_moe_logical_scales(s, size_k, size_n, group_size, is_a_8bit=False)
```

helper 语义固定为 reshape 后输出 logical N contiguous：

```text
sm70_marlin_logical_scales      -> [groups, N]
sm70_marlin_moe_logical_scales  -> [experts, groups, N]
```

AWQ float zero-point 也使用独立 SM70 helper：

```python
awq_to_sm70_marlin_zero_points_float(...)
moe_awq_to_sm70_marlin_zero_points_float(...)
```

处理顺序固定为：

1. 从 checkpoint packed AWQ qzeros 解包。
2. 反转 AWQ nibble/byte interleave，得到 logical qzero。
3. 使用原始 logical scale 计算 `zp_float = zp * scale`。
4. 将 `zp_float` 保存为 SM70 logical N contiguous metadata。

本轮不修改：

- torch op schema。
- qweight repack layout。
- auto CTA / auto split-K 策略。
- empty `c_tmp` / output tensor split-K reduce 设计。
- CUDA metadata hot path。kernel 继续连续读取 logical metadata。

另外，真实 vLLM smoke 暴露出一个非 metadata ABI、但会阻断验证的 Python
capability gate：部分 Marlin writeback/dispatch surface 仍保留上游
`get_min_capability() == 75` 或 `device_capability < 75 -> []`，导致 V100 上
`AWQMarlinConfig.is_awq_marlin_compatible(...) == False`，HF checkpoint 中的
`quant_method=awq` 无法 override 成 `awq_marlin`。因此本轮同步将 Marlin 专属
surface 的 min capability 改为 SM70：

- `query_marlin_supported_quant_types(...)`
- `MarlinLinearKernel`
- `GPTQMarlinConfig`
- `AWQMarlinConfig`
- Dense compressed-tensors WNA16 / FP8 / NVFP4 / MXFP4 Marlin schemes

普通非 Marlin `awq` fallback gate 未修改，避免把未回写的高 SM fallback kernel
误开到 V100。

## 涉及文件

核心 helper：

- `vllm/model_executor/layers/quantization/utils/marlin_utils.py`

Dense path：

- `vllm/model_executor/kernels/linear/mixed_precision/marlin.py`
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py`

MoE path：

- `vllm/model_executor/layers/quantization/gptq_marlin.py`
- `vllm/model_executor/layers/quantization/awq_marlin.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`

## 验证计划与记录

静态确认 class 范围：

```bash
git show abeccd8fede450dc3e818998abf276040ac4ef31:tests/writeback_marlin_cases.py \
  | rg -n "class_name=|quant_names=|zp_quant_names=" -S
```

静态确认 CUDA gather 错误方案已回退：

```bash
rg -n "marlin_grouped_metadata_permuted_n|marlin_single_metadata_permuted_n|metadata_permuted_n|load_metadata_pair" \
  csrc/quantization/marlin csrc/moe/marlin_moe_wna16 -S
```

结果：无输出。

静态确认 SM70 production prepare path 不再调用 legacy scale permutation：

```bash
rg -n "marlin_permute_scales\\(|marlin_moe_permute_scales\\(" \
  vllm/model_executor/kernels/linear \
  vllm/model_executor/layers/quantization \
  -S
```

结果：只剩 `marlin_utils.py` 中 legacy helper 自身定义和 legacy float ZP helper
内部调用；15 个 SM70 production prepare surface 均已改用 `sm70_*logical*`
helper。

Python 静态编译：

```bash
PYTHONPATH=$PWD ./.venv/bin/python -m py_compile \
  vllm/model_executor/kernels/linear/mixed_precision/marlin.py \
  vllm/model_executor/kernels/linear/scaled_mm/marlin.py \
  vllm/model_executor/layers/quantization/utils/marlin_utils.py \
  vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py \
  vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py \
  vllm/model_executor/layers/quantization/gptq_marlin.py \
  vllm/model_executor/layers/quantization/awq_marlin.py \
  vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py \
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py \
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py \
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py \
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py \
  vllm/model_executor/layers/quantization/quark/quark_moe.py \
  tests/writeback_marlin_cases.py \
  tests/test_marlin_linear_kernel.py \
  tests/test_marlin_moe_kernel.py
```

结果：通过。

构建与导入：

```bash
PYTHONPATH=$PWD ./.venv/bin/python setup.py build_ext --inplace \
  > benchmarks/results/20260605_build_logical_metadata_layout_fix.log 2>&1

PYTHONPATH=$PWD ./.venv/bin/python - <<'PY'
import torch
import vllm._C
import vllm._moe_C
assert hasattr(torch.ops._C, "marlin_gemm")
assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm")
print("imports ok")
PY
```

结果：通过。构建日志未出现 `error:`、`fatal error`、`warning #...`、
`nvcc warning`、`declared but never referenced` 或 `deprecated-gpu-targets`。

Focused Dense ZP correctness：

```bash
./test.sh tests/test_marlin_dense.py -q -k "uint4_zp or uint8_zp"
```

结果：`62 passed, 283 deselected`。

Focused MoE metadata correctness：

```bash
./test.sh tests/test_marlin_moe.py -q \
  -k "uint4_zp or uint8_zp or fp8 or fp4 or nvfp4 or mxfp4"
```

结果：第一轮 `94 passed, 85 deselected`，另有 3 个负向 validation test 因测试
参数未设置 `is_zp_float=true` 先触发 ZP contract 错误。修正测试参数后，单独重跑
失败用例：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest -q \
  tests/test_marlin_moe.py::test_moe_wna16_uint4_zp_rejects_non_64_n_alignment \
  tests/test_marlin_moe.py::test_moe_wna16_uint4_zp_rejects_k_alignment \
  tests/test_marlin_moe.py::test_moe_wna16_uint4_zp_rejects_mismatched_zero_point_shape
```

结果：`3 passed`。

Full writeback class-path validation：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest -q \
  tests/test_marlin_linear_kernel.py \
  tests/test_marlin_moe_kernel.py
```

结果：`72 passed`，覆盖本轮确认的 8 个 Dense class 和 7 个 MoE class 的
post-load / apply surface。

本轮在 capability gate 修正后又补充了更小的 focused 回归：

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest -q \
  tests/test_marlin_linear_kernel.py::test_marlin_linear_kernel_scalar_support_matrix_includes_fp4 \
  tests/test_marlin_linear_kernel.py::test_marlin_linear_kernel_u4_u8_zp_preprocess_and_apply \
  tests/test_marlin_linear_kernel.py::test_awq_marlin_linear_method_class_path_converts_zp_and_uses_kernel_c_tmp
```

结果：`17 passed`。

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest -q \
  tests/test_marlin_moe_kernel.py::test_awq_marlin_moe_method_class_uses_owner \
  tests/test_marlin_moe.py::test_fused_marlin_moe_uint4_zp_accuracy \
  tests/test_marlin_moe.py::test_fused_marlin_moe_uint8_zp_accuracy
```

结果：`18 passed`，包含 `torch.jit.script_method` deprecation warnings。

```bash
PYTHONPATH=$PWD ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint8_zp_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_fp8_weight_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_nvfp4_weight_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_mxfp4_weight_accuracy
```

结果：`20 passed`。

vLLM 当前 checkout 同步与 rebuild/import：

```bash
cd /root/source/repos/vllm

PYTHONPATH=$PWD ./.venv/bin/python -m py_compile \
  vllm/model_executor/kernels/linear/mixed_precision/marlin.py \
  vllm/model_executor/kernels/linear/scaled_mm/marlin.py \
  vllm/model_executor/layers/quantization/utils/marlin_utils.py \
  vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py \
  vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py \
  vllm/model_executor/layers/quantization/gptq_marlin.py \
  vllm/model_executor/layers/quantization/awq_marlin.py \
  vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py \
  vllm/model_executor/layers/quantization/quark/quark_moe.py

uv pip install --no-build-isolation -e . --torch-backend=cu126 -v \
  > /tmp/20260605_vllm_logical_metadata_layout_fix_install.log 2>&1
```

结果：通过。安装耗时约 `33m04s`，生成并导入：

- `vllm/_C.abi3.so`
- `vllm/_C_stable_libtorch.abi3.so`
- `vllm/_moe_C.abi3.so`
- `vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`

导入检查：

```bash
PYTHONPATH=$PWD ./.venv/bin/python - <<'PY'
import torch
import vllm._C
import vllm._moe_C
import vllm.vllm_flash_attn._vllm_fa2_C
print("marlin_gemm", hasattr(torch.ops._C, "marlin_gemm"))
print("moe_wna16_marlin_gemm", hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"))
print("imports ok")
PY
```

结果：

```text
marlin_gemm True
moe_wna16_marlin_gemm True
imports ok
```

构建日志未出现 `error:`、`fatal error`、`warning #...`、`nvcc warning`、
`declared but never referenced` 或 `deprecated-gpu-targets`。

vLLM 当前 checkout capability probe：

```bash
cd /root/source/repos/vllm

PYTHONPATH=$PWD ./.venv/bin/python - <<'PY'
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.utils import marlin_utils as m
import json

cfg = json.load(open("/root/models/QuantTrio/Qwen3.6-27B-AWQ/config.json"))[
    "quantization_config"
]
qt = AWQMarlinConfig.TYPE_MAP[cfg["bits"]]
for cap in [70, 75, 80]:
    print("cap", cap, m._check_marlin_supported(
        qt, cfg["group_size"], cfg["zero_point"], cap))
print("compatible", AWQMarlinConfig.is_awq_marlin_compatible(cfg))
print("override none", AWQMarlinConfig.override_quantization_method(cfg, None))
print("override explicit",
      AWQMarlinConfig.override_quantization_method(cfg, "awq_marlin"))
PY
```

结果：

```text
cap 70 (True, None)
cap 75 (True, None)
cap 80 (True, None)
compatible True
override none awq_marlin
override explicit awq_marlin
```

真实 AWQ Marlin offline smoke：

```bash
cd /root/source/repos/vllm

SAFETENSORS_FAST_GPU=1 PYTHONPATH=$PWD ./.venv/bin/python \
  /tmp/vllm_awq_marlin_logical_metadata_smoke.py \
  > /tmp/20260605_vllm_awq_marlin_logical_metadata_smoke.log 2>&1
```

配置：

- model: `/root/models/QuantTrio/Qwen3.6-27B-AWQ`
- quantization: `awq_marlin`
- dtype: `float16`
- tensor_parallel_size: `4`
- attention backend: `FLASH_ATTN`
- max_model_len: `4096`

日志确认：

```text
The model is convertible to awq_marlin during runtime. Using awq_marlin kernel.
Using MarlinLinearKernel for AWQMarlinLinearMethod
Using FlashAttention version 2
```

生成结果：

```text
INPUT '1+1='
OUTPUT '2\n1+1=2\n1+1=2\n1+1=2\n1+1=2\n1+1=2\n'
```

第二轮使用 `enable_thinking=False` 的 chat template：

```bash
SAFETENSORS_FAST_GPU=1 PYTHONPATH=$PWD ./.venv/bin/python \
  /tmp/vllm_awq_marlin_logical_metadata_smoke_no_think.py \
  > /tmp/20260605_vllm_awq_marlin_logical_metadata_smoke_no_think.log 2>&1
```

结果：

```text
INPUT '1+1='
OUTPUT '2\n1+1=2\n1+1=2\n1+'

INPUT '<|im_start|>user\n请只回答：你好<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
OUTPUT '你好'
```

这两轮 smoke 说明原先的随机英文/符号乱码已消失。第一轮未关闭 thinking 时模型
输出英文思考模板，属于 Qwen reasoning template 行为，不是数值乱码；第二轮用
`enable_thinking=False` 后中文短答正常。
