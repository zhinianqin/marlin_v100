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
    def is_a_8bit(self) -> bool:
        return self.size_bits == 8

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
    uint8b128 = ScalarType(0, 8, False, 128)


def marlin_make_workspace_new(device: torch.device, max_blocks_per_sm: int = 1) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int, device=device)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int, device=device)


def _supported_quant_types() -> tuple[ScalarType, ...]:
    return (scalar_types.uint4b8, scalar_types.uint8b128)


def _quantize_unsigned_with_bias(
    weight: torch.Tensor, group_size: int, bias: int
) -> tuple[torch.Tensor, torch.Tensor]:
    size_k, size_n = weight.shape
    if group_size == -1:
        group_size = size_k
    if size_k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide size_k={size_k}")

    groups = size_k // group_size
    reshaped = weight.reshape(groups, group_size, size_n)
    max_abs = reshaped.abs().amax(dim=1, keepdim=False).clamp_min(1e-6)
    scales = max_abs / float(bias - 1)
    scales = scales.to(weight.dtype)

    q = torch.round(reshaped / scales.unsqueeze(1)).clamp(-bias, bias - 1).to(torch.int32)
    q = (q + bias).reshape(size_k, size_n)
    return q, scales


def marlin_quantize(
    weight: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if act_order:
        raise ValueError("Local marlin_v100 helper does not support act_order yet.")
    if quant_type not in _supported_quant_types():
        raise ValueError("Local marlin_v100 helper currently supports uint4b8 and uint8b128 only.")

    size_k, size_n = weight.shape
    q_weight, scales = _quantize_unsigned_with_bias(weight, group_size, quant_type.bias)
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


def marlin_dequantize(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    quant_type: ScalarType,
) -> torch.Tensor:
    if quant_type not in _supported_quant_types():
        raise ValueError("Local marlin_v100 helper currently supports uint4b8 and uint8b128 only.")
    unpacked = marlin_unpack(q_weight, size_k, size_n, quant_type).to(torch.float32)
    unpermuted_scales = marlin_unpermute_scales(scales, size_k, size_n, group_size, quant_type)
    if group_size == -1:
        group_size = size_k
    expanded_scales = unpermuted_scales.repeat_interleave(group_size, dim=0)[:size_k]
    return ((unpacked - float(quant_type.bias)) * expanded_scales.to(torch.float32)).to(
        torch.float16
    )


def marlin_unpack(
    q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    quant_type: ScalarType,
) -> torch.Tensor:
    if quant_type not in _supported_quant_types():
        raise ValueError("Local marlin_v100 helper currently supports uint4b8 and uint8b128 only.")
    perm = quant_utils.get_weight_perm(quant_type.size_bits, is_a_8bit=False).to(
        q_weight.device, dtype=torch.long
    )
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=q_weight.device, dtype=torch.long)

    packed = q_weight.to(torch.int64) & 0xFFFFFFFF
    pack_factor = quant_utils.get_pack_factor(quant_type.size_bits)
    unpacked = torch.stack(
        [
            (packed >> (quant_type.size_bits * i)) & ((1 << quant_type.size_bits) - 1)
            for i in range(pack_factor)
        ],
        dim=-1,
    )
    unpacked = unpacked.to(torch.int32).reshape(q_weight.shape[0], q_weight.shape[1] * pack_factor)
    unpacked = unpacked.reshape(-1, perm.numel())[:, inv_perm].reshape(size_k // 16, size_n * 16)
    unpacked = unpacked.reshape(size_k // 16, size_n // 16, 16, 16)
    return unpacked.permute(0, 2, 1, 3).reshape(size_k, size_n)


def marlin_unpermute_scales(
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    quant_type: ScalarType,
) -> torch.Tensor:
    if quant_type not in _supported_quant_types():
        raise ValueError("Local marlin_v100 helper currently supports uint4b8 and uint8b128 only.")
    scale_perm, scale_perm_single = dense.get_scale_perms()
    perm = scale_perm if group_size < size_k and group_size != -1 else scale_perm_single
    perm = torch.tensor(perm, device=scales.device, dtype=torch.long)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.numel(), device=scales.device, dtype=torch.long)
    return scales.reshape(-1, perm.numel())[:, inv_perm].reshape(-1, size_n).contiguous()


def marlin_dense_reference(
    a: torch.Tensor,
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    quant_type: ScalarType,
) -> torch.Tensor:
    weight = marlin_dequantize(q_weight, scales, size_k, size_n, group_size, quant_type)
    return torch.matmul(a.to(torch.float32), weight.to(torch.float32))


def marlin_quantize_experts(
    weights: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_weights = []
    scales = []
    dequantized = []
    for expert in range(weights.shape[0]):
        _, q_weight, scale, _g_idx, _sort_indices, _rand_perm = marlin_quantize(
            weights[expert], quant_type, group_size, act_order
        )
        q_weights.append(q_weight)
        scales.append(scale)
        dequantized.append(
            marlin_dequantize(
                q_weight,
                scale,
                weights.shape[1],
                weights.shape[2],
                group_size,
                quant_type,
            )
        )
    return torch.stack(q_weights), torch.stack(scales), torch.stack(dequantized)


def marlin_moe_reference(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    tokens, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    outputs = []
    for token_idx in range(tokens):
        token_out = torch.zeros((hidden,), device=hidden_states.device, dtype=torch.float32)
        hidden_row = hidden_states[token_idx : token_idx + 1].to(torch.float32)
        for route_idx in range(topk):
            expert = int(topk_ids[token_idx, route_idx].item())
            gate_up = torch.matmul(hidden_row, w1[expert].to(torch.float32))
            gate, up = gate_up.chunk(2, dim=-1)
            activated = torch.nn.functional.silu(gate) * up
            route_out = torch.matmul(activated, w2[expert].to(torch.float32))[0]
            token_out += route_out * topk_weights[token_idx, route_idx].to(torch.float32)
        outputs.append(token_out)
    return torch.stack(outputs, dim=0)
