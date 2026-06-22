#!/usr/bin/env bash
set -euo pipefail

UV="${UV:-/root/.local/bin/uv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_TAG="${VLLM_TAG:-v0.19.1}"
VLLM_CLONE_FLAGS="${VLLM_CLONE_FLAGS:---depth 1}"
MARLIN_V100_REPO="${MARLIN_V100_REPO:-http://openmediavault.lan:3000/admin/marlin_v100.git}"
FLASH_ATTN_REPO="${FLASH_ATTN_REPO:-http://openmediavault.lan:3000/admin/flash-attention-v100.git}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="${VLLM_DIR:-$SCRIPT_DIR/vllm}"
MARLIN_V100_DIR="${MARLIN_V100_DIR:-$SCRIPT_DIR/marlin_v100}"
FLASH_ATTN_DIR="${FLASH_ATTN_DIR:-$SCRIPT_DIR/flash-attention-v100}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/install-vllm-0.19.1-$(date +%Y%m%d_%H%M%S).log}"

exec > >(tee -a "$LOG_FILE") 2>&1

run() {
  printf '\n+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

require_clean_repo() {
  local repo_dir="$1"
  local name="$2"
  if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
    echo "ERROR: $name has uncommitted changes: $repo_dir"
    echo "Refusing to overwrite an existing dirty worktree."
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: required path is missing: $path"
    exit 1
  fi
}

prepare_source_repo() {
  local name="$1"
  local repo_dir="$2"
  local repo_url="$3"
  local branch="$4"
  local required_marker="$5"

  if [[ -d "$repo_dir/.git" ]]; then
    require_clean_repo "$repo_dir" "$name"
    run git -C "$repo_dir" fetch origin "$branch"
    run git -C "$repo_dir" checkout "$branch"
    run git -C "$repo_dir" pull --ff-only origin "$branch"
  elif [[ -e "$repo_dir" ]]; then
    echo "ERROR: $name path exists but is not a git repository: $repo_dir"
    echo "Set ${name}_DIR explicitly or remove the existing path."
    exit 1
  else
    run git clone --branch "$branch" "$repo_url" "$repo_dir"
  fi

  require_file "$repo_dir/$required_marker"
}

prepare_flash_attn_dependencies() {
  local name="$1"
  local repo_dir="$2"

  if [[ ! -e "$repo_dir/csrc/cutlass/include/cutlass/numeric_types.h" ]]; then
    echo "Preparing $name CUTLASS submodule..."
    run git -C "$repo_dir" submodule update --init csrc/cutlass
  fi

  require_file "$repo_dir/csrc/cutlass/include/cutlass/numeric_types.h"
}

prepare_vllm_repo() {
  if [[ -d "$VLLM_DIR/.git" ]]; then
    require_clean_repo "$VLLM_DIR" "vLLM"
  elif [[ -e "$VLLM_DIR" ]]; then
    echo "ERROR: vLLM path exists but is not a git repository: $VLLM_DIR"
    echo "Set VLLM_DIR explicitly or remove the existing path."
    exit 1
  else
    local clone_flags=()
    if [[ -n "$VLLM_CLONE_FLAGS" ]]; then
      read -r -a clone_flags <<< "$VLLM_CLONE_FLAGS"
    fi
    run git clone "${clone_flags[@]}" --branch "$VLLM_TAG" "$VLLM_REPO" "$VLLM_DIR"
    require_clean_repo "$VLLM_DIR" "vLLM"
  fi

  local vllm_head
  local vllm_tag_commit
  vllm_head="$(git -C "$VLLM_DIR" rev-parse HEAD)"
  vllm_tag_commit="$(git -C "$VLLM_DIR" rev-list -n 1 "$VLLM_TAG")"
  if [[ "$vllm_head" != "$vllm_tag_commit" ]]; then
    echo "ERROR: $VLLM_DIR is not at $VLLM_TAG."
    echo "HEAD: $vllm_head"
    echo "$VLLM_TAG: $vllm_tag_commit"
    exit 1
  fi
}

print_git_head() {
  local name="$1"
  local repo_dir="$2"
  if [[ -d "$repo_dir/.git" ]]; then
    echo "$name HEAD: $(git -C "$repo_dir" rev-parse HEAD)"
  fi
}

echo "Log file: $LOG_FILE"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "MARLIN_V100_DIR=$MARLIN_V100_DIR"
echo "VLLM_DIR=$VLLM_DIR"
echo "VLLM_CLONE_FLAGS=$VLLM_CLONE_FLAGS"
echo "FLASH_ATTN_DIR=$FLASH_ATTN_DIR"
echo "CUDA_HOME=$CUDA_HOME"
require_file "$UV"
require_file "$CUDA_HOME/bin/nvcc"

prepare_vllm_repo
prepare_source_repo "MARLIN_V100" "$MARLIN_V100_DIR" "$MARLIN_V100_REPO" main "upstream_map.yaml"
prepare_source_repo "FLASH_ATTN" "$FLASH_ATTN_DIR" "$FLASH_ATTN_REPO" main "README.md"
prepare_flash_attn_dependencies "FlashAttention V100" "$FLASH_ATTN_DIR"

echo "Using marlin_v100 source: $MARLIN_V100_DIR"
echo "Using FlashAttention V100 source: $FLASH_ATTN_DIR"
echo "Using vLLM source/build dir: $VLLM_DIR"
print_git_head "marlin_v100" "$MARLIN_V100_DIR"
print_git_head "FlashAttention V100" "$FLASH_ATTN_DIR"
print_git_head "vLLM" "$VLLM_DIR"

python_files=(
  vllm/_custom_ops.py
  vllm/model_executor/kernels/linear/mixed_precision/marlin.py
  vllm/model_executor/kernels/linear/scaled_mm/marlin.py
  vllm/model_executor/layers/quantization/utils/marlin_utils.py
  vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py
  vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py
  vllm/model_executor/layers/quantization/utils/nvfp4_utils.py
  vllm/model_executor/layers/fused_moe/fused_marlin_moe.py
  vllm/model_executor/layers/quantization/gptq_marlin.py
  vllm/model_executor/layers/quantization/awq_marlin.py
  vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py
  vllm/model_executor/layers/quantization/quark/quark_moe.py
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a16_fp8.py
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_nvfp4.py
  vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a16_mxfp4.py
)

dense_files=(
  csrc/quantization/marlin/awq_marlin_repack.cu
  csrc/quantization/marlin/dequant.h
  csrc/quantization/marlin/gptq_marlin_repack.cu
  csrc/quantization/marlin/marlin.cu
  csrc/quantization/marlin/marlin.cuh
  csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu
  csrc/quantization/marlin/sm70_marlin_common.cuh
  csrc/quantization/marlin/sm70_marlin_fp8_gemm.cu
  csrc/quantization/marlin/sm70_marlin_gemm.cuh
  csrc/quantization/marlin/sm70_marlin_iterator_utils.cuh
  csrc/quantization/marlin/sm70_marlin_mma.cuh
  csrc/quantization/marlin/sm70_marlin_mxfp4_gemm.cu
  csrc/quantization/marlin/sm70_marlin_nvfp4_gemm.cu
  csrc/quantization/marlin/sm70_marlin_splitk.cuh
  csrc/quantization/marlin/sm70_marlin_u4_gemm.cu
  csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu
  csrc/quantization/marlin/sm70_marlin_u8_gemm.cu
  csrc/quantization/marlin/sm70_marlin_u8b128_gemm.cu
)

moe_files=(
  csrc/moe/marlin_moe_wna16/ops.cu
  csrc/moe/marlin_moe_wna16/sm70_marlin_fp8_gemm.cu
  csrc/moe/marlin_moe_wna16/sm70_marlin_gemm.cuh
  csrc/moe/marlin_moe_wna16/sm70_marlin_mxfp4_gemm.cu
  csrc/moe/marlin_moe_wna16/sm70_marlin_nvfp4_gemm.cu
  csrc/moe/marlin_moe_wna16/sm70_marlin_u4_gemm.cu
  csrc/moe/marlin_moe_wna16/sm70_marlin_u4b8_gemm.cu
  csrc/moe/marlin_moe_wna16/sm70_marlin_u8_gemm.cu
  csrc/moe/marlin_moe_wna16/sm70_marlin_u8b128_gemm.cu
)

echo "Copying Marlin production files into vLLM..."
for rel in "${python_files[@]}" "${dense_files[@]}" "${moe_files[@]}"; do
  require_file "$MARLIN_V100_DIR/$rel"
  run cp "$MARLIN_V100_DIR/$rel" "$VLLM_DIR/$rel"
done

echo "Removing legacy generated Marlin template files..."
run rm -f \
  "$VLLM_DIR/csrc/quantization/marlin/generate_kernels.py" \
  "$VLLM_DIR/csrc/quantization/marlin/kernel.h" \
  "$VLLM_DIR/csrc/quantization/marlin/marlin_dtypes.cuh" \
  "$VLLM_DIR/csrc/quantization/marlin/marlin_mma.h" \
  "$VLLM_DIR/csrc/quantization/marlin/marlin_template.h" \
  "$VLLM_DIR/csrc/moe/marlin_moe_wna16/generate_kernels.py" \
  "$VLLM_DIR/csrc/moe/marlin_moe_wna16/kernel.h" \
  "$VLLM_DIR/csrc/moe/marlin_moe_wna16/marlin_template.h"

echo "Applying vLLM SM70 patches..."
run python3 - "$VLLM_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])


def replace(path: str, old: str, new: str) -> None:
    p = root / path
    data = p.read_text()
    if old not in data:
        raise SystemExit(f"pattern not found in {path}")
    p.write_text(data.replace(old, new, 1))


cmake = root / "CMakeLists.txt"
data = cmake.read_text()
dense_start = data.index('  # Only build Marlin kernels if we are building for at least some compatible archs.')
dense_end = data.index('  # Only build AllSpark kernels if we are building for at least some compatible archs.', dense_start)
dense_block = '''  # SM70 Marlin uses explicit production kernels from marlin_v100 rather than
  # vLLM's legacy generated sm75/sm80/sm89 template path.
  cuda_archs_loose_intersection(MARLIN_SM70_ARCHS "7.0" "${CUDA_ARCHS}")
  if (MARLIN_SM70_ARCHS)
    set(MARLIN_SRCS
       "csrc/quantization/marlin/marlin.cu"
       "csrc/quantization/marlin/marlin_int4_fp8_preprocess.cu"
       "csrc/quantization/marlin/gptq_marlin_repack.cu"
       "csrc/quantization/marlin/awq_marlin_repack.cu"
       "csrc/quantization/marlin/sm70_marlin_u4_gemm.cu"
       "csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu"
       "csrc/quantization/marlin/sm70_marlin_u8_gemm.cu"
       "csrc/quantization/marlin/sm70_marlin_u8b128_gemm.cu"
       "csrc/quantization/marlin/sm70_marlin_fp8_gemm.cu"
       "csrc/quantization/marlin/sm70_marlin_nvfp4_gemm.cu"
       "csrc/quantization/marlin/sm70_marlin_mxfp4_gemm.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_SRCS}"
      CUDA_ARCHS "${MARLIN_SM70_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_SRCS}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_EXT_SRC "${MARLIN_SRCS}")

    message(STATUS "Building SM70 Marlin kernels for archs: ${MARLIN_SM70_ARCHS}")
  else()
    message(STATUS "Not building Marlin kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()

'''
data = data[:dense_start] + dense_block + data[dense_end:]
moe_start = data.index('  # moe marlin arches')
moe_end = data.index('  # DeepSeek V3 router GEMM kernel - requires SM90+', moe_start)
moe_block = '''  # SM70 Marlin MoE uses explicit production kernels from marlin_v100 rather
  # than vLLM's legacy generated sm75/sm80/sm89 template path.
  cuda_archs_loose_intersection(MARLIN_MOE_SM70_ARCHS "7.0" "${CUDA_ARCHS}")
  if (MARLIN_MOE_SM70_ARCHS)
    set(MARLIN_MOE_OTHER_SRC
       "csrc/moe/marlin_moe_wna16/ops.cu"
       "csrc/moe/marlin_moe_wna16/sm70_marlin_u4_gemm.cu"
       "csrc/moe/marlin_moe_wna16/sm70_marlin_u4b8_gemm.cu"
       "csrc/moe/marlin_moe_wna16/sm70_marlin_u8_gemm.cu"
       "csrc/moe/marlin_moe_wna16/sm70_marlin_u8b128_gemm.cu"
       "csrc/moe/marlin_moe_wna16/sm70_marlin_fp8_gemm.cu"
       "csrc/moe/marlin_moe_wna16/sm70_marlin_nvfp4_gemm.cu"
       "csrc/moe/marlin_moe_wna16/sm70_marlin_mxfp4_gemm.cu")
    set_gencode_flags_for_srcs(
      SRCS "${MARLIN_MOE_OTHER_SRC}"
      CUDA_ARCHS "${MARLIN_MOE_SM70_ARCHS}")
    if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
      set_source_files_properties(${MARLIN_MOE_OTHER_SRC}
        PROPERTIES COMPILE_FLAGS "-static-global-template-stub=false")
    endif()
    list(APPEND VLLM_MOE_EXT_SRC "${MARLIN_MOE_OTHER_SRC}")

    message(STATUS "Building SM70 Marlin MOE kernels for archs: ${MARLIN_MOE_SM70_ARCHS}")
  else()
    message(STATUS "Not building Marlin MOE kernels as no compatible archs found"
                   " in CUDA target architectures")
  endif()

'''
cmake.write_text(data[:moe_start] + moe_block + data[moe_end:])

replace("vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py",
        "return current_platform.has_device_capability(75)",
        "return current_platform.has_device_capability(70)")
replace("vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py",
        "return current_platform.has_device_capability(75)",
        "return current_platform.has_device_capability(70)")
replace("vllm/model_executor/kernels/linear/scaled_mm/marlin.py",
        "FP8 Marlin requires compute capability 7.5 or higher",
        "FP8 Marlin requires compute capability 7.0 or higher")
replace("vllm/model_executor/layers/fused_moe/fused_marlin_moe.py",
        "return p.is_cuda() and p.has_device_capability((7, 5))",
        "return p.is_cuda() and p.has_device_capability((7, 0))")
replace("vllm/v1/attention/backends/flash_attn.py",
        "return capability >= DeviceCapability(8, 0)",
        "return capability >= DeviceCapability(7, 0)")

replace("vllm/vllm_flash_attn/flash_attn_interface.py",
        "if not current_platform.has_device_capability(80):",
        "if not current_platform.has_device_capability(70):")

replace("vllm/model_executor/layers/quantization/moe_wna16.py",
        '''        elif self.linear_quant_method in ("awq", "awq_marlin"):
            capability_tuple = current_platform.get_device_capability()
            device_capability = (
                -1 if capability_tuple is None else capability_tuple.to_int()
            )
            awq_min_capability = AWQConfig.get_min_capability()
            if device_capability < awq_min_capability:
                raise ValueError(
                    "The quantization method moe_wna16 + awq is not supported "
                    "for the current GPU. "
                    f"Minimum capability: {awq_min_capability}. "
                    f"Current capability: {device_capability}."
                )
            self.use_marlin = AWQMarlinConfig.is_awq_marlin_compatible(full_config)
''',
        '''        elif self.linear_quant_method in ("awq", "awq_marlin"):
            if self.linear_quant_method == "awq":
                capability_tuple = current_platform.get_device_capability()
                device_capability = (
                    -1 if capability_tuple is None else capability_tuple.to_int()
                )
                awq_min_capability = AWQConfig.get_min_capability()
                if device_capability < awq_min_capability:
                    raise ValueError(
                        "The quantization method moe_wna16 + awq is not supported "
                        "for the current GPU. "
                        f"Minimum capability: {awq_min_capability}. "
                        f"Current capability: {device_capability}."
                    )
            self.use_marlin = AWQMarlinConfig.is_awq_marlin_compatible(full_config)
''')

replace("vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py",
        "return 75",
        "return 70")

replace("vllm/model_executor/layers/quantization/modelopt.py",
        "return 75",
        "return 70")

setup = root / "setup.py"
setup_data = setup.read_text()
if "_is_sm70_flash_attn_source_build" not in setup_data:
    ext_modules_marker = "ext_modules = []\n"
    if ext_modules_marker not in setup_data:
        raise SystemExit("pattern not found in setup.py: ext_modules = []")
    setup_data = setup_data.replace(
        ext_modules_marker,
        '''ext_modules = []


def _is_sm70_flash_attn_source_build() -> bool:
    if not os.getenv("VLLM_FLASH_ATTN_SRC_DIR"):
        return False
    torch_cuda_arch_list = os.getenv("TORCH_CUDA_ARCH_LIST", "")
    arch_tokens = torch_cuda_arch_list.replace(";", " ").replace(",", " ").split()
    return any(token.replace("+PTX", "") in {"7.0", "70"} for token in arch_tokens)


''',
        1,
    )
setup_data = setup_data.replace(
    '''        # FA3 requires CUDA 12.3 or later
        ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
''',
    '''        # The SM70 FlashAttention source used for V100 builds defines FA2 only.
        if not _is_sm70_flash_attn_source_build():
            # FA3 requires CUDA 12.3 or later
            ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C"))
''',
    1,
)
setup.write_text(setup_data)

allspark_utils = root / "csrc/quantization/gptq_allspark/allspark_utils.cuh"
allspark_data = allspark_utils.read_text()
allspark_old = '#include "../marlin/marlin_dtypes.cuh"\nusing marlin::MarlinScalarType2;\n\nnamespace allspark {\n'
if allspark_old not in allspark_data:
    raise SystemExit("pattern not found in csrc/quantization/gptq_allspark/allspark_utils.cuh")
allspark_data = allspark_data.replace(allspark_old,
                                      '''namespace allspark {

template <typename FType>
class AllSparkScalarType {};

template <>
class AllSparkScalarType<half> {
 public:
  static __device__ float inline num2float(const half x) {
    return __half2float(x);
  }

  static __host__ __device__ half inline float2num(const float x) {
    return __float2half(x);
  }
};

template <>
class AllSparkScalarType<nv_bfloat16> {
 public:
  static __device__ float inline num2float(const nv_bfloat16 x) {
    return __bfloat162float(x);
  }

  static __host__ __device__ nv_bfloat16 inline float2num(const float x) {
    return __float2bfloat16(x);
  }
};

''',
                                      1)
allspark_data = allspark_data.replace("MarlinScalarType2<FType>::", "AllSparkScalarType<FType>::")
allspark_utils.write_text(allspark_data)

allspark_kernel = root / "csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu"
allspark_kernel_data = allspark_kernel.read_text()
if "MarlinScalarType2<FType>::" not in allspark_kernel_data:
    raise SystemExit("pattern not found in csrc/quantization/gptq_allspark/allspark_qgemm_w8a16.cu")
allspark_kernel.write_text(allspark_kernel_data.replace("MarlinScalarType2<FType>::", "AllSparkScalarType<FType>::"))

replace("requirements/common.txt",
        "fastapi[standard] >= 0.115.0 # Required by FastAPI's form models",
        "fastapi[standard] >= 0.115.0, < 0.137.0 # Required by FastAPI's form models")
PY

export CUDA_HOME="$CUDA_HOME"
export PATH="$VLLM_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu126}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export VLLM_FLASH_ATTN_SRC_DIR="$FLASH_ATTN_DIR"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export PYTHONPATH="$VLLM_DIR"

cd "$VLLM_DIR"

run git diff --check
run "$UV" venv --python 3.12

run "$UV" pip install --python .venv/bin/python --torch-backend=cu126 \
  "cmake>=3.26.1" ninja "packaging>=24.2" \
  "setuptools>=77.0.3,<81.0.0" "setuptools-scm>=8.0" \
  wheel "jinja2>=3.1.6" regex build
run "$UV" pip install --python .venv/bin/python --torch-backend=cu126 \
  torch==2.10.0 torchaudio==2.10.0 torchvision==0.25.0
run "$UV" pip install --python .venv/bin/python --torch-backend=cu126 \
  -r requirements/common.txt -r requirements/cuda.txt

run "$UV" build --wheel --no-build-isolation --out-dir dist .
run "$UV" pip install --python .venv/bin/python --force-reinstall dist/vllm-*.whl

cd "$SCRIPT_DIR" && PYTHONPATH="" run "$VLLM_DIR/.venv/bin/python" - <<'PY'
import torch
import vllm._C
import vllm._moe_C
import vllm.vllm_flash_attn
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("marlin_gemm", hasattr(torch.ops._C, "marlin_gemm"))
print("moe_wna16_marlin_gemm", hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm"))
print("flash_attn_sm70", FlashAttentionBackend.supports_compute_capability(DeviceCapability(7, 0)))
PY

echo "Done. Log file: $LOG_FILE"
