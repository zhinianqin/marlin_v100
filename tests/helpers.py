from __future__ import annotations

from dataclasses import dataclass

import torch

from marlin_v100 import dense, quant_utils


@dataclass(frozen=True)
class ScalarType:
    exponent: int
    mantissa: int
    signed: bool
    bias: int = 0
    finite_values_only: bool = False
    nan_repr: int = 1

    @property
    def size_bits(self) -> int:
        return self.exponent + self.mantissa + int(self.signed)

    @property
    def id(self) -> int:
        val = 0
        offset = 0

        def add_field(member: int, bit_width: int) -> None:
            nonlocal val, offset
            bit_mask = (1 << bit_width) - 1
            val |= (int(member) & bit_mask) << offset
            offset += bit_width

        add_field(self.exponent, 8)
        add_field(self.mantissa, 8)
        add_field(self.signed, 1)
        add_field(self.bias, 32)
        add_field(self.finite_values_only, 1)
        add_field(self.nan_repr, 8)
        return val


class scalar_types:
    uint4 = ScalarType(0, 4, False, 0)
    uint4b8 = ScalarType(0, 4, False, 8)


def marlin_make_workspace_new(device: torch.device, max_blocks_per_sm: int = 1) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int, device=device)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int, device=device)


def _quantize_uint4b8(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    size_k, size_n = weight.shape
    if group_size == -1:
        group_size = size_k
    if size_k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide size_k={size_k}")

    groups = size_k // group_size
    reshaped = weight.reshape(groups, group_size, size_n)
    max_abs = reshaped.abs().amax(dim=1, keepdim=False).clamp_min(1e-6)
    scales = max_abs / 7.0
    scales = scales.to(weight.dtype)

    q = torch.round(reshaped / scales.unsqueeze(1)).clamp(-8, 7).to(torch.int32)
    q = (q + 8).reshape(size_k, size_n)
    return q, scales


def marlin_quantize(
    weight: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if act_order:
        raise ValueError("Local marlin_v100 helper does not support act_order yet.")
    if quant_type != scalar_types.uint4b8:
        raise ValueError("Local marlin_v100 helper currently supports uint4b8 only.")

    size_k, size_n = weight.shape
    q_weight, scales = _quantize_uint4b8(weight, group_size)
    weight_perm = quant_utils.get_weight_perm(quant_type.size_bits, is_a_8bit=False)
    marlin_q_weight = quant_utils.marlin_weights(
        q_weight,
        size_k,
        size_n,
        quant_type.size_bits,
        weight_perm,
        is_a_8bit=False,
    )
    marlin_scales = dense.marlin_permute_scales(
        scales,
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    g_idx = marlin_make_empty_g_idx(weight.device)
    sort_indices = torch.empty(0, dtype=torch.int, device=weight.device)
    rand_perm = torch.arange(size_k, dtype=torch.int, device=weight.device)
    return weight, marlin_q_weight, marlin_scales, g_idx, sort_indices, rand_perm
