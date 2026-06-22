# vLLM 0.19.1 SM70 Marlin + FlashAttention 回写与安装记录

日期：2026-06-21

目标是在 vLLM `v0.19.1` 源码上做最小范围 SM70/V100 补丁：回写 `marlin_v100` 的生产 Marlin dense/MoE 实现，删除旧生成式 Marlin 模板链，只解除 Marlin 与普通 FlashAttention 相关的计算能力门槛到 SM70，并使用支持 SM70 的 FlashAttention V100 源码构建 vLLM wheel 后安装到 vLLM 目录内 `.venv`。

当前手工补丁对照树为 `/root/source/repos/vllm`。自动复现脚本不再探测 `/root/source/repos/marlin_v100` 或 `/root/source/repos/flash-attention-v100`；默认完全以脚本自身所在目录 `$SCRIPT_DIR` 为工作根。

## 固定环境

- uv：`/root/.local/bin/uv`
- vLLM 源码默认目录：`$SCRIPT_DIR/vllm`
- vLLM 版本：`v0.19.1`
- vLLM clone 默认参数：`VLLM_CLONE_FLAGS="--depth 1"`，可通过环境变量覆盖
- marlin_v100 默认目录：`$SCRIPT_DIR/marlin_v100`
- marlin_v100 仓库：`http://openmediavault.lan:3000/admin/marlin_v100.git`，`main` 分支
- FlashAttention V100 默认目录：`$SCRIPT_DIR/flash-attention-v100`
- FlashAttention V100 仓库：`http://openmediavault.lan:3000/admin/flash-attention-v100.git`，`main` 分支
- CUDA：`/usr/local/cuda-12.6`
- Python：`3.12`
- PyTorch：`2.10.0`，cu126
- 目标计算能力：`7.0`
- vLLM wheel 流程：构建 wheel 后安装 wheel，不使用 `-e`

关键环境变量：

```bash
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$VLLM_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=7.0
export UV_TORCH_BACKEND=cu126
export VLLM_TARGET_DEVICE=cuda
export VLLM_FLASH_ATTN_SRC_DIR="$FLASH_ATTN_DIR"
export MAX_JOBS=${MAX_JOBS:-8}
export NVCC_THREADS=${NVCC_THREADS:-1}
export PYTHONPATH="$VLLM_DIR"
```

代理可选：

```bash
export HTTP_PROXY=http://192.168.2.1:8118
export HTTPS_PROXY=http://192.168.2.1:8118
```

## 安装命令记录

自动复现脚本：`install-vllm-0.19.1.sh`。

脚本包含以下安装与构建命令：

```bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
git clone --depth 1 --branch v0.19.1 https://github.com/vllm-project/vllm.git "$SCRIPT_DIR/vllm"
git clone --branch main http://openmediavault.lan:3000/admin/marlin_v100.git "$SCRIPT_DIR/marlin_v100"
git clone --branch main http://openmediavault.lan:3000/admin/flash-attention-v100.git "$SCRIPT_DIR/flash-attention-v100"
/root/.local/bin/uv venv --python 3.12 .venv
/root/.local/bin/uv pip install --python .venv/bin/python --torch-backend=cu126 "cmake>=3.26.1" ninja "packaging>=24.2" "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" wheel "jinja2>=3.1.6" regex build
/root/.local/bin/uv pip install --python .venv/bin/python --torch-backend=cu126 torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0
/root/.local/bin/uv pip install --python .venv/bin/python --torch-backend=cu126 -r requirements/common.txt -r requirements/cuda.txt
/root/.local/bin/uv build --wheel --no-build-isolation --out-dir dist .
/root/.local/bin/uv pip install --python .venv/bin/python --force-reinstall dist/vllm-*.whl
```

脚本不会执行任何 `apt` 安装。若缺少系统包，应停止并由用户确认后处理。

脚本目录策略：

- 默认 `VLLM_DIR=$SCRIPT_DIR/vllm`。
- 默认 `VLLM_CLONE_FLAGS="--depth 1"`，用于减少 vLLM tag clone 的网络传输；需要完整历史时可设置 `VLLM_CLONE_FLAGS=""`。
- 默认 `MARLIN_V100_DIR=$SCRIPT_DIR/marlin_v100`。
- 默认 `FLASH_ATTN_DIR=$SCRIPT_DIR/flash-attention-v100`。
- 默认 `LOG_DIR=$SCRIPT_DIR/logs`。
- 不自动探测 `/root/source/repos/marlin_v100` 或 `/root/source/repos/flash-attention-v100`。
- 若默认源码目录不存在，脚本从对应内网 Git 仓库 `main` 分支 clone 到 `$SCRIPT_DIR` 下。
- 若默认目录已存在但不是 git 仓库，脚本报错并要求显式设置对应 `*_DIR` 或移除该目录。
- 构建前脚本打印 `Using marlin_v100 source`、`Using FlashAttention V100 source`、`Using vLLM source/build dir` 及可读取的 git HEAD。

## 回写 Python 文件

以下文件从 `marlin_v100` 复制到 vLLM：

- `vllm/_custom_ops.py`
- `vllm/model_executor/kernels/linear/mixed_precision/marlin.py`
- `vllm/model_executor/kernels/linear/scaled_mm/marlin.py`
- `vllm/model_executor/layers/quantization/utils/marlin_utils.py`
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`
- `vllm/model_executor/layers/quantization/utils/nvfp4_utils.py`
- `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`
- `vllm/model_executor/layers/quantization/gptq_marlin.py`
- `vllm/model_executor/layers/quantization/awq_marlin.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`
- `vllm/model_executor/layers/quantization/quark/quark_moe.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py`

理由：这些文件承载当前生产 Marlin dense/MoE 行为，包括 `marlin_gemm` / `moe_wna16_marlin_gemm` 参数、`c_or_none` / `c_tmp` 生命周期、`global_scale`、`is_zp_float`、SM70 logical-scale、AWQ float zero-point，以及 FP8/FP4/NVFP4/MXFP4 Marlin 路径。

## 回写 dense CUDA/C++ 文件

以下文件从 `marlin_v100` 复制到 vLLM：

- `csrc/quantization/marlin/awq_marlin_repack.cu`
- `csrc/quantization/marlin/dequant.h`
- `csrc/quantization/marlin/gptq_marlin_repack.cu`
- `csrc/quantization/marlin/marlin.cu`
- `csrc/quantization/marlin/marlin.cuh`
- `csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu`
- `csrc/quantization/marlin/sm70_marlin_common.cuh`
- `csrc/quantization/marlin/sm70_marlin_fp8_gemm.cu`
- `csrc/quantization/marlin/sm70_marlin_gemm.cuh`
- `csrc/quantization/marlin/sm70_marlin_iterator_utils.cuh`
- `csrc/quantization/marlin/sm70_marlin_mma.cuh`
- `csrc/quantization/marlin/sm70_marlin_mxfp4_gemm.cu`
- `csrc/quantization/marlin/sm70_marlin_nvfp4_gemm.cu`
- `csrc/quantization/marlin/sm70_marlin_splitk.cuh`
- `csrc/quantization/marlin/sm70_marlin_u4_gemm.cu`
- `csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu`
- `csrc/quantization/marlin/sm70_marlin_u8_gemm.cu`
- `csrc/quantization/marlin/sm70_marlin_u8b128_gemm.cu`

明确排除：

- `csrc/quantization/marlin/sm70_cutlass_matmul_probe.cu`

排除理由：该文件是本地实验/benchmark probe，依赖 `marlin_v100` 最小项目独立 schema，vLLM 生产路径不调用；不复制、不注册 schema、不加入 CMake。

## 回写 MoE CUDA/C++ 文件

以下文件从 `marlin_v100` 复制到 vLLM：

- `csrc/moe/marlin_moe_wna16/ops.cu`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_gemm.cuh`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_fp8_gemm.cu`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_mxfp4_gemm.cu`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_nvfp4_gemm.cu`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_u4_gemm.cu`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_u4b8_gemm.cu`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_u8_gemm.cu`
- `csrc/moe/marlin_moe_wna16/sm70_marlin_u8b128_gemm.cu`

理由：这些文件提供当前生产 `moe_wna16_marlin_gemm` 的 SM70 显式 kernel 分发。

## 删除文件

删除 dense legacy generator/template 文件：

- `csrc/quantization/marlin/generate_kernels.py`
- `csrc/quantization/marlin/kernel.h`
- `csrc/quantization/marlin/marlin_dtypes.cuh`
- `csrc/quantization/marlin/marlin_mma.h`
- `csrc/quantization/marlin/marlin_template.h`

删除 MoE legacy generator/template 文件：

- `csrc/moe/marlin_moe_wna16/generate_kernels.py`
- `csrc/moe/marlin_moe_wna16/kernel.h`
- `csrc/moe/marlin_moe_wna16/marlin_template.h`

删除理由：当前 SM70 实现通过显式 `sm70_marlin_*` kernels 分发，不再使用旧 generator/template/sm75/sm80/sm89 生成链。

## vLLM 代码修改点与理由

- `CMakeLists.txt`
  - dense Marlin 段删除 generator 调用和 `sm75_kernel_*` / `sm80_kernel_*` / `sm89_kernel_*` glob。
  - 新增 `cuda_archs_loose_intersection(MARLIN_SM70_ARCHS "7.0" "${CUDA_ARCHS}")` 和显式 SM70 dense source list。
  - MoE Marlin 段删除 generator 调用和旧 glob。
  - 新增 `cuda_archs_loose_intersection(MARLIN_MOE_SM70_ARCHS "7.0" "${CUDA_ARCHS}")` 和显式 SM70 MoE source list。
  - 保留 vLLM 原有 CUTLASS FetchContent/include 机制，不引入 `marlin_v100` 独立 CMake 的 `/root/source/repos/cutlass` 假设。
- `setup.py`
  - 新增 `_is_sm70_flash_attn_source_build()`。
  - 当 `VLLM_FLASH_ATTN_SRC_DIR` 存在且 `TORCH_CUDA_ARCH_LIST` 为 `7.0`/`70` 时，不追加 `_vllm_fa3_C`。
  - 理由：FlashAttention V100 CMake 只定义 `_vllm_fa2_C`；vLLM 0.19.1 在 CUDA >= 12.3 默认请求 FA3，CUDA 12.6 + SM70 source build 必须跳过 FA3。
- `vllm/v1/attention/backends/flash_attn.py`
  - `FlashAttentionBackend.supports_compute_capability()` 从 `DeviceCapability(8, 0)` 改为 `DeviceCapability(7, 0)`。
  - 理由：FlashAttention V100 README 要求普通 FlashAttention backend 对 SM70 可用。
- `vllm/vllm_flash_attn/flash_attn_interface.py`
  - `has_device_capability(80)` 改为 `has_device_capability(70)`。
  - 理由：FlashAttention V100 仅提供 FA2；放宽到 SM70 允许 V100 通过 backend 能力检查。
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`
  - `is_fp8_marlin_supported()` 从 `has_device_capability(75)` 改为 `has_device_capability(70)`。
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`
  - `is_fp4_marlin_supported()` 从 `has_device_capability(75)` 改为 `has_device_capability(70)`。
- `vllm/model_executor/kernels/linear/scaled_mm/marlin.py`
  - FP8 Marlin 错误文案从 `7.5` 改为 `7.0`。
- `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`
  - `_supports_current_device()` 从 `(7, 5)` 改为 `(7, 0)`。
- `vllm/model_executor/layers/quantization/moe_wna16.py`
  - 普通 `awq` 继续使用 `AWQConfig.get_min_capability()`。
  - `awq_marlin` 不再被普通 AWQ 的 `75` 门槛提前拦截，改由 `AWQMarlinConfig.is_awq_marlin_compatible()` 判断。
  - 理由：普通 AWQ kernel 仍非本次解锁范围，AWQ-Marlin MoE 是 Marlin 路径。
- `csrc/quantization/gptq_allspark/allspark_utils.cuh`
  - 去掉对已删除 `../marlin/marlin_dtypes.cuh` 的 include。
  - 新增本地 `AllSparkScalarType<half/nv_bfloat16>`，只提供 AllSpark 已使用的 `num2float` / `float2num`。
- `csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu`
  - 将 `MarlinScalarType2<FType>` 替换为 `AllSparkScalarType<FType>`。
  - 理由：删除旧 Marlin header 后避免非 Marlin AllSpark 路径被悬空 include 破坏；不改变 AllSpark 的高能力 CMake gate。
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py`
  - `CompressedTensorsW4A4Fp4.get_min_capability()` 从 `75` 改为 `70`。
  - 理由：NVFP4 Marlin SM70 kernel 已通过 `sm70_marlin_nvfp4_gemm.cu` 提供，`75` 门槛会阻止 SM70 GPU 使用 CompressedTensors W4A4 NVFP4 量化路径。
- `vllm/model_executor/layers/quantization/modelopt.py`
  - `ModelOptNvFp4Config.get_min_capability()` 从 `75` 改为 `70`。
  - 理由：ModelOpt NVFP4 量化路径同样依赖 SM70 Marlin NVFP4 kernel，`75` 门槛不必要地阻止 SM70 GPU 加载 ModelOpt NVFP4 量化模型。
- `vllm/model_executor/layers/attention/attention.py`
  - 删除 `BaseKVCacheMethod` import（唯一使用处已被删除）。
  - 删除第一个废弃的 `quant_method =` 赋值（被 L156 遮蔽，本就 dead）。
  - 删除第二个 `quant_method =` 赋值及其后续 `should_load_quant_weights` 块（原 L156-172）。
  - 删除 `kv_cache_scheme` 覆写块（原 L228-241），硬编码 `use_per_head_quant_scales = False`。
  - 理由：V100 无 FP8 硬件，KV cache 量化在所有量化配置下均不应生效。`kv_cache_scheme` 块会篡改 `cache_config.cache_dtype` 为 `"fp8"`；`should_load_quant_weights` 块会创建 `k_scale`/`v_scale` 参数导致权重加载 KeyError。`set_default_quant_scales` 已为 `k_scale`/`v_scale` 注册 1.0 buffer 作为默认值。这两处是所有量化配置 KV cache 量化的汇聚点。注意：仅删除这两处不足以保证 SM70 运行——上游 `resolve_kv_cache_dtype_string()` 仍会将模型 `kv_cache_scheme` 解析为 `"fp8_e4m3"`，需配合 `torch_utils.py` 修改。
- `vllm/utils/torch_utils.py`
  - `resolve_kv_cache_dtype_string()` 删除 FP8 量化配置解析块（原 L333-339），`"auto"` 直接返回 `"auto"`。
  - 理由：V100 无 FP8 硬件。该块将模型 `config.json` 的 `kv_cache_scheme` 解析为 `"fp8_e4m3"`，导致 `CacheConfig.cache_dtype` 被设为 FP8，随后 FlashAttention backend 在 SM70 上拒绝启动（`kv_cache_dtype not supported`）。直接删除解析逻辑，与 attention.py 的删除策略一致。
- `vllm/model_executor/models/glm4_moe.py`
  - `Glm4MoeModel.load_weights()` 在 spec_layer 检查之后、stacked_params_mapping 循环之前，添加 `if name.endswith((".k_scale", ".v_scale", ".q_scale")): continue` 显式跳过 attention KV cache scale 权重。
  - 理由：attention.py 补丁删除 `create_weights` 后 `k_scale`/`v_scale`/`q_scale` 不再是 nn.Parameter。原 `stacked_params_mapping` 子串匹配 bug（`"v_proj" in "qkv_proj"`）会在这些权重上触发 KeyError。`.split(".")` 组件匹配虽然逻辑正确，但运行时可能因 `.pyc` 缓存未生效。在循环入口显式跳过是最稳健方案。
- `requirements/common.txt`
  - 添加 `fastapi[standard]` 版本上限 `< 0.137.0`。
  - 理由：`fastapi>=0.137.0` 移除了 `fastapi[standard]` extra，导致 vLLM 0.19.1 的 `pip install -r requirements/common.txt` 失败。

## 已确认保留的高能力限制

以下限制不修改：

- Machete
- CUTLASS W4A8
- AllSpark CMake 构建 arch：`8.0;8.6;8.7;8.9`
- FlashInfer
- DeepGEMM
- FlashMLA
- FlashAttention sink
- FlashAttention FP8
- FlashAttention MLA
- FA3/FA4/Hopper/Blackwell 专用路径
- PyTorch scaled-mm 高能力限制
- `vllm/platforms/cuda.py` 的 BF16 能力检查

理由：这些路径依赖其他高能力 kernel 或硬件特性，不属于本次 SM70 Marlin/普通 FA2 解锁范围。

## 不回写或不修改项

- 不复制 `marlin_v100/csrc/type_convert.cuh`。
  - 理由：它是 vLLM 全局 CUDA 头，影响 layernorm/qk-norm 等非 Marlin 内核；当前差异不是 Marlin 回写必需项，除非实际 SM70 build 暴露定点编译错误才单独处理并记录。
- 不新增 `csrc/torch_bindings_marlin.cpp` 或 `csrc/moe/torch_bindings_marlin.cpp`。
  - 理由：vLLM CMake 使用现有 `csrc/torch_bindings.cpp` 和 `csrc/moe/torch_bindings.cpp`，并且现有 schema 已包含 `marlin_gemm` / `moe_wna16_marlin_gemm` 的生产参数；独立项目 binding 包含 probe schema，不应进入 vLLM。

## 验证记录

本轮实际验证结果：

- vLLM 源码版本：`v0.19.1`，HEAD `b1388b1fbf5aaef47937fabe98931211684666a6`。
- FlashAttention V100 源码：脚本默认 `$SCRIPT_DIR/flash-attention-v100`，本轮手工构建使用的源码 HEAD 为 `c2eda5e6115b98c3ba4bfd181570668742eece22`。
- 手工构建与安装日志：`/root/source/repos/marlin_v100/logs/manual-vllm-build-20260621_204222.log`。
- 安装脚本：`/root/source/repos/marlin_v100/install-vllm-0.19.1.sh`，已设置可执行权限；`bash -n install-vllm-0.19.1.sh` 通过。

静态检查：

```bash
git -C /root/source/repos/vllm diff --check
git -C /root/source/repos/marlin_v100 diff --check
rg -n "generate_kernels|marlin_template|marlin_dtypes|marlin_mma\\.h|sm75_kernel|sm80_kernel|sm89_kernel|sm70_cutlass_matmul_probe|MarlinScalarType2" /root/source/repos/vllm/CMakeLists.txt /root/source/repos/vllm/csrc /root/source/repos/vllm/vllm
```

结果：`git diff --check` 通过；legacy generator/template、旧 sm75/sm80/sm89 generated kernel glob、`sm70_cutlass_matmul_probe`、`MarlinScalarType2` 均未在 vLLM `CMakeLists.txt` / `csrc` / `vllm` 中留下引用。

wheel 构建命令：

```bash
cd /root/source/repos/vllm
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=7.0
export UV_TORCH_BACKEND=cu126
export VLLM_TARGET_DEVICE=cuda
export VLLM_FLASH_ATTN_SRC_DIR="$FLASH_ATTN_DIR"
export MAX_JOBS=8
export NVCC_THREADS=1
export PYTHONPATH=$PWD
/root/.local/bin/uv build --wheel --no-build-isolation --out-dir dist .
```

结果：构建成功，生成 wheel：

```text
/root/source/repos/vllm/dist/vllm-0.19.2.dev0+gb1388b1fb.d20260621.cu126-cp312-cp312-linux_x86_64.whl
```

构建日志中确认：

- `Building SM70 Marlin kernels for archs: 7.0`
- `Building SM70 Marlin MOE kernels for archs: 7.0`
- `VLLM_FLASH_ATTN_SRC_DIR` 指向脚本解析出的 FlashAttention V100 源码目录
- wheel 包含 `vllm/_C.abi3.so`、`vllm/_moe_C.abi3.so`、`vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so`
- wheel 未包含 `_vllm_fa3_C`；SM70 FlashAttention source build gate 生效

wheel 安装命令：

```bash
cd /root/source/repos/vllm
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST=7.0
export UV_TORCH_BACKEND=cu126
export VLLM_TARGET_DEVICE=cuda
export VLLM_FLASH_ATTN_SRC_DIR="$FLASH_ATTN_DIR"
export PYTHONPATH=$PWD
/root/.local/bin/uv pip install --python .venv/bin/python --force-reinstall dist/vllm-*.whl
```

结果：安装成功，`.venv` 中安装的是本地 wheel：

```text
vllm==0.19.2.dev0+gb1388b1fb.d20260621.cu126
```

已安装 wheel smoke 命令从 `/tmp` 执行，并取消 `PYTHONPATH`，避免优先导入源码树：

```bash
cd /tmp
unset PYTHONPATH
/root/source/repos/vllm/.venv/bin/python - <<'PY'
import torch
import vllm
import vllm._C
import vllm._moe_C
import vllm.vllm_flash_attn
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("vllm", vllm.__version__, "from", vllm.__file__)
print("_C", vllm._C.__file__)
print("_moe_C", vllm._moe_C.__file__)
print("flash_attn", vllm.vllm_flash_attn.__file__)
print("marlin_gemm", hasattr(torch.ops._C, "marlin_gemm"))
print("moe_wna16_marlin_gemm", hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"))
print("flash_attn_sm70", FlashAttentionBackend.supports_compute_capability(DeviceCapability(7, 0)))
PY
```

结果：

```text
torch 2.10.0+cu126 cuda 12.6
vllm 0.19.2.dev0+gb1388b1fb.d20260621 from /root/source/repos/vllm/.venv/lib/python3.12/site-packages/vllm/__init__.py
_C /root/source/repos/vllm/.venv/lib/python3.12/site-packages/vllm/_C.abi3.so
_moe_C /root/source/repos/vllm/.venv/lib/python3.12/site-packages/vllm/_moe_C.abi3.so
flash_attn /root/source/repos/vllm/.venv/lib/python3.12/site-packages/vllm/vllm_flash_attn/__init__.py
marlin_gemm True
moe_wna16_marlin_gemm True
flash_attn_sm70 True
```

当前机器运行时限制：

- `nvidia-smi` 失败，错误为无法与 NVIDIA driver 通信。
- smoke 过程中 PyTorch 报告 `Can't initialize NVML`，vLLM 报告 `0 active driver(s) found` 并禁用 Triton。
- 因此本轮完成了构建、wheel 安装、扩展导入、op 注册和 SM70 能力门槛检查；未执行需要真实 GPU driver 的 Marlin dense/MoE runtime benchmark 或完整 pytest/exhaustive sweep。

## `/root` 独立运行验证记录

脚本复制到 `/root` 后的默认路径：

- 脚本：`/root/install-vllm-0.19.1.sh`
- vLLM：`/root/vllm`
- marlin_v100：`/root/marlin_v100`
- FlashAttention V100：`/root/flash-attention-v100`
- 日志：`/root/logs/install-vllm-0.19.1-*.log`

执行命令：

```bash
cp /root/source/repos/marlin_v100/install-vllm-0.19.1.sh /root/install-vllm-0.19.1.sh
chmod +x /root/install-vllm-0.19.1.sh
/root/install-vllm-0.19.1.sh
```

脚本会在构建前打印：

```text
Using marlin_v100 source: /root/marlin_v100
Using FlashAttention V100 source: /root/flash-attention-v100
Using vLLM source/build dir: /root/vllm
```

独立运行完成后，对比 `/root/vllm` 与当前手工补丁对照树 `/root/source/repos/vllm`：

```bash
git -C /root/vllm rev-parse HEAD
git -C /root/source/repos/vllm rev-parse HEAD
git -C /root/vllm describe --tags --exact-match HEAD
git -C /root/source/repos/vllm describe --tags --exact-match HEAD

diff -ruN --exclude=.git --exclude=.venv --exclude=build --exclude=dist \
  --exclude=__pycache__ --exclude='*.egg-info' --exclude=.pytest_cache \
  /root/source/repos/vllm/CMakeLists.txt /root/vllm/CMakeLists.txt

diff -ruN --exclude=.git --exclude=.venv --exclude=build --exclude=dist \
  --exclude=__pycache__ --exclude='*.egg-info' --exclude=.pytest_cache \
  /root/source/repos/vllm/setup.py /root/vllm/setup.py

diff -ruN --exclude=.git --exclude=.venv --exclude=build --exclude=dist \
  --exclude=__pycache__ --exclude='*.egg-info' --exclude=.pytest_cache \
  /root/source/repos/vllm/csrc /root/vllm/csrc

diff -ruN --exclude=.git --exclude=.venv --exclude=build --exclude=dist \
  --exclude=__pycache__ --exclude='*.egg-info' --exclude=.pytest_cache \
  /root/source/repos/vllm/vllm /root/vllm/vllm

diff -u \
  <(git -C /root/source/repos/vllm status --short | sort) \
  <(git -C /root/vllm status --short | sort)
```

结果待独立运行后补充。
