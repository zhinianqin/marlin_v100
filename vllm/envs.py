from __future__ import annotations

import os

VLLM_MARLIN_USE_ATOMIC_ADD = (
    os.environ.get("VLLM_MARLIN_USE_ATOMIC_ADD", "0").lower()
    in {"1", "true", "yes", "on"}
)
VLLM_MARLIN_INPUT_DTYPE = os.environ.get("VLLM_MARLIN_INPUT_DTYPE")
VLLM_DISABLED_KERNELS = tuple(
    item.strip()
    for item in os.environ.get("VLLM_DISABLED_KERNELS", "").split(",")
    if item.strip()
)
VLLM_USE_FBGEMM = (
    os.environ.get("VLLM_USE_FBGEMM", "0").lower() in {"1", "true", "yes", "on"}
)
VLLM_USE_TRITON_AWQ = (
    os.environ.get("VLLM_USE_TRITON_AWQ", "0").lower()
    in {"1", "true", "yes", "on"}
)
VLLM_USE_NVFP4_CT_EMULATIONS = (
    os.environ.get("VLLM_USE_NVFP4_CT_EMULATIONS", "0").lower()
    in {"1", "true", "yes", "on"}
)
VLLM_NVFP4_GEMM_BACKEND = os.environ.get("VLLM_NVFP4_GEMM_BACKEND")
VLLM_ROCM_FP8_MFMA_PAGE_ATTN = (
    os.environ.get("VLLM_ROCM_FP8_MFMA_PAGE_ATTN", "0").lower()
    in {"1", "true", "yes", "on"}
)
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE = int(
    os.environ.get("VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE", "16384")
)
VLLM_MOE_DP_CHUNK_SIZE = int(os.environ.get("VLLM_MOE_DP_CHUNK_SIZE", "0"))
VLLM_BATCH_INVARIANT = (
    os.environ.get("VLLM_BATCH_INVARIANT", "0").lower()
    in {"1", "true", "yes", "on"}
)
VLLM_TEST_FORCE_FP8_MARLIN = (
    os.environ.get("VLLM_TEST_FORCE_FP8_MARLIN", "0").lower()
    in {"1", "true", "yes", "on"}
)
