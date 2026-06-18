from __future__ import annotations

from enum import Enum


class Mxfp4MoeBackend(Enum):
    NONE = "None"
    FLASHINFER_TRTLLM_MXFP4_MXFP8 = "FLASHINFER_TRTLLM_MXFP4_MXFP8"
    FLASHINFER_TRTLLM_MXFP4_BF16 = "FLASHINFER_TRTLLM_MXFP4_BF16"
    FLASHINFER_CUTLASS_MXFP4_MXFP8 = "FLASHINFER_CUTLASS_MXFP4_MXFP8"
    FLASHINFER_CUTLASS_MXFP4_BF16 = "FLASHINFER_CUTLASS_MXFP4_BF16"
    BATCHED_MARLIN = "BATCHED_MARLIN"
    MARLIN = "MARLIN"
    CK = "CK"
    TRITON = "TRITON"
    TRITON_UNFUSED = "TRITON_UNFUSED"
    XPU = "XPU"


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def select_mxfp4_moe_backend(*args, **kwargs):
    del args, kwargs
    return Mxfp4MoeBackend.NONE, None


def mxfp4_round_up_hidden_size_and_intermediate_size(
    backend: Mxfp4MoeBackend,
    hidden_size: int,
    intermediate_size: int,
) -> tuple[int, int]:
    if backend in (Mxfp4MoeBackend.MARLIN, Mxfp4MoeBackend.BATCHED_MARLIN):
        return _round_up(hidden_size, 256), _round_up(intermediate_size, 128)
    if backend in (
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_MXFP8,
        Mxfp4MoeBackend.FLASHINFER_TRTLLM_MXFP4_BF16,
    ):
        return _round_up(hidden_size, 256), _round_up(intermediate_size, 256)
    if backend in (
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_MXFP8,
        Mxfp4MoeBackend.FLASHINFER_CUTLASS_MXFP4_BF16,
    ):
        return _round_up(hidden_size, 128), _round_up(intermediate_size, 128)
    return hidden_size, _round_up(intermediate_size, 64)


def make_mxfp4_moe_kernel(*args, **kwargs):
    raise NotImplementedError


def make_mxfp4_moe_quant_config(*args, **kwargs):
    raise NotImplementedError
