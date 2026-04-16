from __future__ import annotations

from dataclasses import dataclass
import math

import pytest
import torch

from marlin_v100.calibration import validate_dense_group_size
from marlin_v100 import dense, ops, quant_utils


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


_REPACK_IMPL_CASES = (
    pytest.param("gptq", id="gptq_marlin_repack"),
    pytest.param("awq", id="awq_marlin_repack"),
)


def marlin_make_workspace_new(device: torch.device, max_blocks_per_sm: int = 1) -> torch.Tensor:
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    return torch.zeros(sms * max_blocks_per_sm, dtype=torch.int, device=device)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int, device=device)


def _supported_quant_types() -> tuple[ScalarType, ...]:
    return (scalar_types.uint4b8, scalar_types.uint8b128)


def pack_rows(
    q_weight: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    if q_weight.shape != (size_k, size_n):
        raise ValueError(f"Expected q_weight.shape == {(size_k, size_n)}, got {tuple(q_weight.shape)}")

    pack_factor = quant_utils.get_pack_factor(num_bits)
    if size_k % pack_factor != 0:
        raise ValueError(f"size_k={size_k} must be divisible by pack_factor={pack_factor}")

    packed = torch.zeros((size_k // pack_factor, size_n), dtype=torch.int64, device=q_weight.device)
    for idx in range(pack_factor):
        packed |= q_weight[idx::pack_factor, :].to(torch.int64) << (num_bits * idx)
    return packed.to(torch.int32).contiguous()


def gptq_pack(
    q_weight: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    return pack_rows(q_weight, num_bits, size_k, size_n)


def pack_cols(
    q_weight: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    if q_weight.shape != (size_k, size_n):
        raise ValueError(f"Expected q_weight.shape == {(size_k, size_n)}, got {tuple(q_weight.shape)}")

    pack_factor = quant_utils.get_pack_factor(num_bits)
    if size_n % pack_factor != 0:
        raise ValueError(f"size_n={size_n} must be divisible by pack_factor={pack_factor}")

    packed = torch.zeros((size_k, size_n // pack_factor), dtype=torch.int64, device=q_weight.device)
    for idx in range(pack_factor):
        packed |= q_weight[:, idx::pack_factor].to(torch.int64) << (num_bits * idx)
    return packed.to(torch.int32).contiguous()


def awq_pack(
    q_weight: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    if num_bits == 4:
        interleave = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=q_weight.device, dtype=torch.long)
    elif num_bits == 8:
        interleave = torch.tensor([0, 2, 1, 3], device=q_weight.device, dtype=torch.long)
    else:
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")

    q_weight = q_weight.reshape((-1, interleave.numel()))[:, interleave].reshape(-1, size_n)
    return pack_cols(q_weight.contiguous(), num_bits, size_k, size_n)


def _deterministic_repack_input(
    quant_type: ScalarType,
    *,
    act_order: bool,
    size_k: int,
    size_n: int,
    group_size: int,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_weight = torch.arange(size_k * size_n, device=device, dtype=torch.int32).reshape(size_k, size_n)
    q_weight = q_weight.remainder(1 << quant_type.size_bits).contiguous()
    sort_indices = torch.empty(0, dtype=torch.int, device=device)
    sorted_q_weight = q_weight
    if act_order:
        _g_idx, sort_indices = _make_act_order_metadata(size_k, group_size, torch.device(device))
        sorted_q_weight = q_weight.index_select(0, sort_indices.to(torch.long)).contiguous()
    return q_weight, sorted_q_weight, sort_indices


def assert_repack_layout_matches_reference(
    repack_impl: str,
    *,
    quant_type: ScalarType,
    act_order: bool = False,
    size_k: int = 128,
    size_n: int = 64,
    group_size: int = 64,
) -> None:
    if repack_impl not in {"gptq", "awq"}:
        raise ValueError(f"Unsupported repack_impl={repack_impl!r}")
    if size_k % 16 != 0 or size_n % 64 != 0:
        raise ValueError(f"Marlin repack expects size_k%16==0 and size_n%64==0, got {(size_k, size_n)}")

    ops._load_dense()
    q_weight, sorted_q_weight, sort_indices = _deterministic_repack_input(
        quant_type,
        act_order=act_order,
        size_k=size_k,
        size_n=size_n,
        group_size=group_size,
        device="cuda",
    )
    weight_perm = quant_utils.get_weight_perm(quant_type.size_bits, is_a_8bit=False).to(q_weight.device)
    expected = quant_utils.marlin_weights(
        sorted_q_weight,
        size_k,
        size_n,
        quant_type.size_bits,
        weight_perm,
        is_a_8bit=False,
    )

    if repack_impl == "gptq":
        packed = gptq_pack(q_weight, quant_type.size_bits, size_k, size_n)
        actual = ops.gptq_marlin_repack(
            packed,
            sort_indices,
            size_k,
            size_n,
            quant_type.size_bits,
            False,
        )
    else:
        source = sorted_q_weight if act_order else q_weight
        packed = awq_pack(source, quant_type.size_bits, size_k, size_n)
        actual = ops.awq_marlin_repack(
            packed,
            size_k,
            size_n,
            quant_type.size_bits,
            False,
        )

    assert torch.equal(actual, expected)


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


def _make_group_ids(size_k: int, group_size: int, device: torch.device) -> torch.Tensor:
    actual_group_size = size_k if group_size == -1 else group_size
    return torch.arange(size_k, device=device, dtype=torch.int) // actual_group_size


def _make_act_order_metadata(
    size_k: int,
    group_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    actual_group_size = size_k if group_size == -1 else group_size
    groups = size_k // actual_group_size
    perm = (
        torch.arange(size_k, device=device, dtype=torch.int)
        .reshape(groups, actual_group_size)
        .flip(1)
        .reshape(-1)
        .contiguous()
    )
    g_idx = _make_group_ids(size_k, group_size, device)[perm.to(torch.long)].contiguous()
    return g_idx, perm


def marlin_quantize(
    weight: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if quant_type not in _supported_quant_types():
        raise ValueError("Local marlin_v100 helper currently supports uint4b8 and uint8b128 only.")
    validate_dense_group_size(group_size)

    size_k, size_n = weight.shape
    g_idx = marlin_make_empty_g_idx(weight.device)
    sort_indices = torch.empty(0, dtype=torch.int, device=weight.device)
    rand_perm = torch.arange(size_k, dtype=torch.int, device=weight.device)
    quant_weight = weight
    if act_order:
        g_idx, sort_indices = _make_act_order_metadata(size_k, group_size, weight.device)
        rand_perm = sort_indices.clone()
        quant_weight = weight.index_select(0, sort_indices.to(torch.long)).contiguous()

    q_weight, scales = _quantize_unsigned_with_bias(quant_weight, group_size, quant_type.bias)
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
    return weight, marlin_q_weight, marlin_scales, g_idx, sort_indices, rand_perm


def marlin_dequantize(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    quant_type: ScalarType,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    if quant_type not in _supported_quant_types():
        raise ValueError("Local marlin_v100 helper currently supports uint4b8 and uint8b128 only.")
    unpacked = marlin_unpack(q_weight, size_k, size_n, quant_type).to(torch.float32)
    unpermuted_scales = marlin_unpermute_scales(scales, size_k, size_n, group_size, quant_type)
    if group_size == -1:
        group_size = size_k
    expanded_scales = unpermuted_scales.repeat_interleave(group_size, dim=0)[:size_k]
    dequantized = ((unpacked - float(quant_type.bias)) * expanded_scales.to(torch.float32)).to(
        torch.float16
    )
    if perm is not None and perm.numel() > 0:
        logical = torch.empty_like(dequantized)
        logical[perm.to(torch.long)] = dequantized
        return logical
    return dequantized


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
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = marlin_dequantize(q_weight, scales, size_k, size_n, group_size, quant_type, perm=perm)
    return torch.matmul(a.to(torch.float32), weight.to(torch.float32))


def marlin_quantize_experts(
    weights: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_weights, scales, dequantized, _g_idx, _perm = marlin_quantize_experts_with_metadata(
        weights, quant_type, group_size, act_order
    )
    return q_weights, scales, dequantized


def marlin_quantize_experts_with_metadata(
    weights: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_weights = []
    scales = []
    dequantized = []
    g_indices = []
    perms = []
    for expert in range(weights.shape[0]):
        _, q_weight, scale, g_idx, sort_indices, _rand_perm = marlin_quantize(
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
                perm=sort_indices,
            )
        )
        g_indices.append(g_idx)
        perms.append(sort_indices)
    return (
        torch.stack(q_weights),
        torch.stack(scales),
        torch.stack(dequantized),
        torch.stack(g_indices),
        torch.stack(perms),
    )


def make_moe_model_like_inputs(
    tokens: int,
    hidden: int,
    intermediate: int,
    experts: int,
    topk: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn((tokens, hidden), device=device, dtype=dtype)
    topk_weights = torch.rand((tokens, topk), device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    topk_ids = torch.empty((tokens, topk), device=device, dtype=torch.int32)
    for token_idx in range(tokens):
        for route_idx in range(topk):
            topk_ids[token_idx, route_idx] = (token_idx + route_idx) % experts

    # Use fan-in-scaled weights so the local MoE benchmark checks resemble the
    # activation ranges seen in real models instead of overflowing fp16 paths
    # with unit-variance synthetic weights.
    w1 = torch.randn((experts, hidden, 2 * intermediate), device=device, dtype=dtype)
    w1 = w1 * (1.0 / math.sqrt(hidden))
    w2 = torch.randn((experts, intermediate, hidden), device=device, dtype=dtype)
    w2 = w2 * (1.0 / math.sqrt(intermediate))
    return hidden_states, topk_weights, topk_ids, w1, w2


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
