from __future__ import annotations

import importlib

import torch

_dense_loaded = False
_moe_loaded = False


def _load_dense() -> None:
    global _dense_loaded
    if not _dense_loaded:
        importlib.import_module("marlin_v100._C")
        _dense_loaded = True


def _load_moe() -> None:
    global _moe_loaded
    if not _moe_loaded:
        importlib.import_module("marlin_v100._moe_C")
        _moe_loaded = True


def marlin_gemm(*args, **kwargs) -> torch.Tensor:
    """Low-level dense binding without Python-side support-matrix validation."""
    _load_dense()
    return torch.ops._C.marlin_gemm(*args, **kwargs)


def gptq_marlin_repack(*args, **kwargs) -> torch.Tensor:
    _load_dense()
    return torch.ops._C.gptq_marlin_repack(*args, **kwargs)


def awq_marlin_repack(*args, **kwargs) -> torch.Tensor:
    _load_dense()
    return torch.ops._C.awq_marlin_repack(*args, **kwargs)


def marlin_int4_fp8_preprocess(*args, **kwargs) -> torch.Tensor:
    _load_dense()
    return torch.ops._C.marlin_int4_fp8_preprocess(*args, **kwargs)


def topk_softmax(*args, **kwargs) -> None:
    _load_moe()
    return torch.ops._moe_C.topk_softmax(*args, **kwargs)


def topk_sigmoid(*args, **kwargs) -> None:
    _load_moe()
    return torch.ops._moe_C.topk_sigmoid(*args, **kwargs)


def grouped_topk(*args, **kwargs):
    _load_moe()
    return torch.ops._moe_C.grouped_topk(*args, **kwargs)


def moe_align_block_size(*args, **kwargs) -> None:
    _load_moe()
    return torch.ops._moe_C.moe_align_block_size(*args, **kwargs)


def batched_moe_align_block_size(*args, **kwargs) -> None:
    _load_moe()
    return torch.ops._moe_C.batched_moe_align_block_size(*args, **kwargs)


def moe_wna16_marlin_gemm(*args, **kwargs) -> torch.Tensor:
    """Low-level MoE binding without Python-side support-matrix validation."""
    _load_moe()
    return torch.ops._moe_C.moe_wna16_marlin_gemm(*args, **kwargs)
