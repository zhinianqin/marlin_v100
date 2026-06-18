from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GroupShape:
    row: int | str = "per_token"
    col: int | None = None


GroupShape.PER_TOKEN = GroupShape("per_token")  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ScaleDesc:
    dtype: torch.dtype
    is_static: bool
    group_shape: GroupShape

    @property
    def static(self) -> bool:
        return self.is_static


@dataclass(frozen=True)
class QuantKey:
    dtype: torch.dtype
    scale: ScaleDesc | None = None
    symmetric: bool = False
    scale2: ScaleDesc | None = None


_STATIC_TENSOR_SCALE = ScaleDesc(torch.float32, True, GroupShape("tensor"))
_STATIC_CHANNEL_SCALE = ScaleDesc(torch.float32, True, GroupShape("channel"))
_STATIC_128_BLOCK_SCALE = ScaleDesc(torch.float32, True, GroupShape(128, 128))
_STATIC_NVFP4_SCALE = ScaleDesc(torch.float8_e4m3fn, True, GroupShape(1, 16))
_STATIC_MXFP4_SCALE = ScaleDesc(torch.float8_e8m0fnu, True, GroupShape(1, 32))

kFp8StaticTensorSym = QuantKey(torch.float8_e4m3fn, _STATIC_TENSOR_SCALE, True)
kFp8StaticChannelSym = QuantKey(torch.float8_e4m3fn, _STATIC_CHANNEL_SCALE, True)
kFp8Static128BlockSym = QuantKey(torch.float8_e4m3fn, _STATIC_128_BLOCK_SCALE, True)
kNvfp4Static = QuantKey(torch.float8_e4m3fn, _STATIC_NVFP4_SCALE, True)
kMxfp4Static = QuantKey(torch.float8_e8m0fnu, _STATIC_MXFP4_SCALE, True)
kFp8Dynamic128Sym = kFp8Static128BlockSym
kFp8DynamicTokenSym = QuantKey(torch.float8_e4m3fn, ScaleDesc(torch.float32, False, GroupShape.PER_TOKEN), True)
kNvfp4Dynamic = kNvfp4Static


def convert_bf16_scales_to_fp8(scales: torch.Tensor) -> torch.Tensor:
    return scales


def convert_packed_uint4b8_to_signed_int4_inplace(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def is_layer_skipped(prefix: str, modules_to_not_convert: list[str] | None) -> bool:
    if not modules_to_not_convert:
        return False
    return any(module_name in prefix for module_name in modules_to_not_convert)


def pack_cols(q_w, num_bits: int, size_k: int, size_n: int):
    import torch

    pack_factor = 32 // num_bits
    q_w = q_w.to(torch.int64)
    packed = torch.zeros(
        (size_k, size_n // pack_factor),
        dtype=torch.int64,
        device=q_w.device,
    )
    for idx in range(pack_factor):
        packed |= q_w[:, idx::pack_factor] << (num_bits * idx)
    return packed.to(torch.int32).contiguous()


def unpack_cols(q_w, num_bits: int, size_k: int, size_n: int):
    import torch

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    unpacked = torch.empty((size_k, size_n), dtype=torch.int32, device=q_w.device)
    words = q_w.to(torch.int64)
    for idx in range(pack_factor):
        unpacked[:, idx::pack_factor] = (
            (words >> (num_bits * idx)) & mask
        ).to(torch.int32)
    return unpacked.contiguous()
