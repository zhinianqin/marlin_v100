# 2026-06-03 Python Upstream Writeback Review

## 摘要

本记录审查当前 `marlin_v100` 中需要回写到上游 vLLM 的 Python 源码边界。

- 上游基线：`/root/source/repos/vllm-0.19.1`
- 回写依据：`upstream_map.yaml`
- 审查范围：`upstream_map.yaml` 中的 Python 映射文件，以及会直接或间接进入 `torch.ops._C.marlin_gemm`、`torch.ops._moe_C.moe_wna16_marlin_gemm` 的生产调用链。
- 不纳入 Python 回写清单：`tests/**`、`benchmarks/**`、`python/marlin_v100/**`、脚本、本地 standalone glue、docs。

当前结论：

- `upstream_map.yaml` 中共有 24 个 Python 映射文件，全部存在。
- 其中 15 个文件相对 `/root/source/repos/vllm-0.19.1` 有实质 diff，需要回写功能改动。
- 其中 9 个文件与上游一致，作为同路径依赖、接口基类或包边界镜像保留在 map 中。

## 审查方法

本次审查使用以下口径：

- 解析 `upstream_map.yaml`，只统计 `source` 以 `.py` 结尾的条目。
- 对每个映射文件检查本地文件是否存在。
- 使用 `diff -q /root/source/repos/vllm-0.19.1/<target> <source>` 判断本地文件与上游基线是否一致。
- 使用静态搜索复核目标调用链：
  - `torch.ops._C.marlin_gemm`
  - `torch.ops._moe_C.moe_wna16_marlin_gemm`
  - `ops.marlin_gemm`
  - `ops.moe_wna16_marlin_gemm`
  - `apply_gptq_marlin_linear`
  - `apply_awq_marlin_linear`
  - `apply_fp8_marlin_linear`
  - `apply_fp4_marlin_linear`
  - `fused_marlin_moe`
  - `batched_fused_marlin_moe`

注意：当前 `vllm/_custom_ops.py` 在 git 状态中显示为未跟踪文件，但它已经列入 `upstream_map.yaml`，因此本审查按回写映射文件处理。

## 有实质 Diff 的 Python 回写文件

这些文件相对 `/root/source/repos/vllm-0.19.1` 有实质差异，应作为本轮 Python 回写的核心文件。

| 文件 | Diff 规模 | 回写理由 |
|---|---:|---|
| `vllm/_custom_ops.py` | `+6/-6` | 目标 torch op wrapper/fake schema 使用 `c_tmp`，并透传 `is_zp_float`。这是 `torch.ops._C.marlin_gemm` 和 `torch.ops._moe_C.moe_wna16_marlin_gemm` 的唯一生产 Python wrapper。 |
| `vllm/model_executor/kernels/linear/mixed_precision/marlin.py` | `+56/-15` | `MarlinLinearKernel` 是 dense WNA16/GPTQ 主生产路径；在 `process_weights_after_loading` 持久化 `self.c_tmp`，对 U4/U8 ZP 做 fp16 float offset 预转换，并在 apply 时传 `is_zp_float=True`。 |
| `vllm/model_executor/kernels/linear/scaled_mm/marlin.py` | `+12/-1` | FP8 scaled-mm Marlin production path；把旧 `layer.workspace` 生命周期对齐为 `layer.c_tmp`，apply 时按 runtime shape 扩容并传给 `apply_fp8_marlin_linear`。 |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py` | `+90/-16` | Marlin dense/MoE 公共 helper；新增 `marlin_make_c_tmp`、`marlin_ensure_c_tmp`，让 `query_marlin_supported_quant_types(has_zp=True)` 支持 `uint4/uint8`，dense helper 透传 `c_tmp/is_zp_float`，并提供 AWQ MoE float ZP 转换 helper。 |
| `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py` | `+9/-10` | FP8 dense/MoE Marlin fallback prepare/apply helper；将旧 `workspace` 语义对齐为 layer 级 `c_tmp`，保持 FP8 路径 `is_zp_float=False`。 |
| `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py` | `+7/-9` | FP4/NVFP4/MXFP4 dense/MoE Marlin fallback prepare/apply helper；将旧 `workspace` 语义对齐为 layer 级 `c_tmp`，保持 FP4 系列路径 `is_zp_float=False`。 |
| `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py` | `+10/-1` | NVFP4 MARLIN backend 直接调用 `apply_fp4_marlin_linear`；apply 前按 `M * N` 扩容并传 `layer.c_tmp`。 |
| `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py` | `+59/-13` | MoE Marlin functional 主路径；将 `workspace` 改为 `c_tmp/c_tmp_owner`，按 runtime shape 扩容并写回 owner，两次 GEMM 分别传 `is_w1_zp_float` 和 `is_w2_zp_float`。 |
| `vllm/model_executor/layers/quantization/gptq_marlin.py` | `+6/-3` | GPTQ Marlin dense 继续委托 `MarlinLinearKernel`；GPTQ MoE 在 post-load 持久化 `layer.c_tmp`，apply 传给 `fused_marlin_moe`。 |
| `vllm/model_executor/layers/quantization/awq_marlin.py` | `+53/-27` | AWQ dense/MoE Marlin production path；AWQ MoE U4/U8 ZP 在 post-load 转为 fp16 float offset，持久化 `layer.c_tmp`，apply 显式传 `is_w1_zp_float=True` 和 `is_w2_zp_float=True`。 |
| `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py` | `+4/-3` | CompressedTensors WNA16 Marlin MoE production path；Marlin backend 在 post-load 持久化 `layer.c_tmp`，apply 传 `c_tmp=layer.c_tmp, c_tmp_owner=layer`。 |
| `vllm/model_executor/layers/quantization/quark/quark_moe.py` | `+2/-0` | Quark FP8 Marlin MoE 分支也进入 `fused_marlin_moe`；该分支传 `c_tmp=layer.c_tmp, c_tmp_owner=layer`，保持 FP8 非 ZP 语义。 |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py` | `+12/-1` | CompressedTensors FP8 dense Marlin entrance；apply 前扩容并传 `layer.c_tmp`，使该 production path 对齐当前 `marlin_gemm` 接口。 |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py` | `+12/-1` | CompressedTensors NVFP4 dense Marlin entrance；apply 前扩容并传 `layer.c_tmp`，保持 NVFP4 非 ZP 语义。 |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py` | `+12/-1` | CompressedTensors MXFP4 dense Marlin entrance；apply 前扩容并传 `layer.c_tmp`，保持 MXFP4 非 ZP 语义。 |

## 与上游一致但保留映射的 Python 文件

这些文件当前与 `/root/source/repos/vllm-0.19.1` 内容一致，但仍保留在 `upstream_map.yaml` 中。原因是它们属于上游同路径 import 边界、接口基类，或生产入口依赖。回写时保留这些映射可以避免只回写叶子文件造成路径或接口不完整。

| 文件 | 状态 | 保留理由 |
|---|---|---|
| `vllm/model_executor/__init__.py` | same | 包边界镜像；保持 `vllm/model_executor/**` 同路径可导入。 |
| `vllm/model_executor/kernels/__init__.py` | same | 包边界镜像；保持 kernel 子包同路径可导入。 |
| `vllm/model_executor/kernels/linear/mixed_precision/MPLinearKernel.py` | same | `MarlinLinearKernel` 的上游同路径基类。 |
| `vllm/model_executor/kernels/linear/scaled_mm/ScaledMMLinearKernel.py` | same | FP8 scaled-mm Marlin kernel 的上游同路径基类。 |
| `vllm/model_executor/layers/__init__.py` | same | 包边界镜像；保持 layer 子包同路径可导入。 |
| `vllm/model_executor/layers/quantization/utils/__init__.py` | same | quantization utils 包边界镜像。 |
| `vllm/model_executor/layers/fused_moe/fused_moe_method_base.py` | same | MoE quant method 基类依赖。 |
| `vllm/model_executor/layers/quantization/quark/__init__.py` | same | Quark quantization 包边界镜像。 |
| `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py` | same | 当前内容与上游一致，但它是 CompressedTensors WNA16 dense 进入 `MarlinLinearKernel` 的 production entrance；依赖已修改的 kernel 行为，保留映射便于回写时审查完整入口。 |

## 回写行为说明

### Dense 路径

当前 dense production path 已覆盖：

```text
CompressedTensorsWNA16 / GPTQ / FP8 / FP4 scheme
  -> process_weights_after_loading(...)
  -> MarlinLinearKernel 或 scaled-mm/FP helper
  -> apply_*_marlin_linear(...)
  -> ops.marlin_gemm(..., c_tmp=..., is_zp_float=...)
  -> torch.ops._C.marlin_gemm(...)
```

核心语义：

- 原上游 `self.workspace` 位置对齐为 `self.c_tmp`，用于 `MarlinLinearKernel`。
- 原上游 `layer.workspace` 位置对齐为 `layer.c_tmp`，用于 FP8/FP4/NVFP4/MXFP4 layer/scheme path。
- U4/U8 ZP 在 `process_weights_after_loading` 阶段预转换为 fp16 float zero-point offset。
- 只有已完成 float ZP 预转换的 U4/U8 path 传 `is_zp_float=True`。
- FP8、NVFP4、MXFP4、`uint4b8`、`uint8b128` 保持 `is_zp_float=False`。

### MoE 路径

当前 MoE production path 已覆盖：

```text
GPTQMarlinMoEMethod / AWQMarlinMoEMethod /
CompressedTensorsWNA16MarlinMoEMethod / Quark FP8 Marlin MoE
  -> process_weights_after_loading(...)
  -> layer.c_tmp = marlin_make_c_tmp(...)
  -> apply(...)
  -> fused_marlin_moe(..., c_tmp=layer.c_tmp, c_tmp_owner=layer, ...)
  -> ops.moe_wna16_marlin_gemm(..., c_tmp, is_zp_float=...)
  -> torch.ops._moe_C.moe_wna16_marlin_gemm(...)
```

核心语义：

- 原上游 method 级 `layer.workspace` 位置对齐为 `layer.c_tmp`。
- `fused_marlin_moe` functional fallback 原来临时创建 `workspace`，现在临时创建 fp32 `c_tmp`。
- 有 `c_tmp_owner` 时，`_ensure_marlin_moe_c_tmp` 会在 device/dtype/contiguous/容量不满足时扩容并写回 `owner.c_tmp`。
- AWQ MoE U4/U8 ZP 在 post-load 中转为 fp16 float offset，stage1/stage2 分别传 `is_zp_float=True`。
- GPTQ、CompressedTensors、Quark 的非 ZP/FP8 paths 保持 `is_zp_float=False`。
- Modular/LoRA `MarlinExperts` / `BatchedMarlinExperts` 没有新增 expert 对象级 `self.c_tmp`，保持上游旧 workspace 的 functional fallback 生命周期。

## 不进入 Python 回写清单的内容

以下内容不应作为上游 Python 源码回写：

- `tests/**`
- `benchmarks/**`
- `build.sh`
- `test.sh`
- `benchmark.sh`
- `python/marlin_v100/**`
- `docs/**`
- 本地 import/standalone glue，尤其是未进入 `upstream_map.yaml` 的 `vllm/model_executor/kernels/linear/__init__.py`
- `.venv/**`、`build/**`、扩展 `.so`、缓存文件

如果后续决定把 LoRA/oracle selector 也纳入“完整生产调用链”回写范围，应单独更新 `upstream_map.yaml` 并追加新的 docs 记录。本次文档只记录当前映射事实，不扩大回写边界。

## 复核命令

Python 映射文件存在性：

```bash
./.venv/bin/python - <<'PY'
import yaml
from pathlib import Path

entries = yaml.safe_load(Path("upstream_map.yaml").read_text())["upstream_map"]
py_entries = [entry for entry in entries if entry["source"].endswith(".py")]

print(f"python mappings: {len(py_entries)}")
for entry in py_entries:
    source = Path(entry["source"])
    print(f"{source}\t{'OK' if source.exists() else 'MISSING'}")
PY
```

相对上游基线的 diff 分类：

```bash
./.venv/bin/python - <<'PY'
import subprocess
import yaml
from pathlib import Path

upstream = Path("/root/source/repos/vllm-0.19.1")
entries = yaml.safe_load(Path("upstream_map.yaml").read_text())["upstream_map"]

for entry in entries:
    source = Path(entry["source"])
    if not str(source).endswith(".py"):
        continue
    target = upstream / entry["target"]
    if not source.exists():
        status = "LOCAL_MISSING"
    elif not target.exists():
        status = "UPSTREAM_MISSING"
    else:
        result = subprocess.run(
            ["diff", "-q", str(target), str(source)],
            capture_output=True,
            text=True,
        )
        status = "DIFF" if result.returncode == 1 else "SAME"
    print(f"{status}\t{source}\t{entry['target']}")
PY
```

文档静态检查：

```bash
sed -n '1,240p' docs/20260603_049_upstream_py_writeback_review.md
rg -n "vllm/_custom_ops.py|MarlinLinearKernel|c_tmp|is_zp_float|upstream_map" \
  docs/20260603_049_upstream_py_writeback_review.md
```
