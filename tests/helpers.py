from __future__ import annotations

from dataclasses import dataclass
import math

import pytest
import torch

from tests import ops, quant_utils
from tests.calibration import validate_dense_group_size, validate_dense_marlin_call, validate_moe_marlin_call


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
    uint8 = ScalarType(0, 8, False, 0)
    uint8b128 = ScalarType(0, 8, False, 128)
    float4_e2m1f = ScalarType(2, 1, True, 0, True, 0)
    float8_e4m3fn = ScalarType(4, 3, True, 0, True, 2)
    float8_e8m0fnu = ScalarType(8, 0, False, 0, True, 2)


_REPACK_IMPL_CASES = (
    pytest.param("gptq", id="gptq_marlin_repack"),
    pytest.param("awq", id="awq_marlin_repack"),
)
_SM70_ROW_GROUPS = (
    (0, 1, 8, 9),
    (2, 3, 10, 11),
    (4, 5, 12, 13),
    (6, 7, 14, 15),
)


def marlin_make_c_tmp(
    device: torch.device,
    numel_or_shape: int | tuple[int, ...] = 0,
) -> torch.Tensor:
    if isinstance(numel_or_shape, tuple):
        return torch.empty(numel_or_shape, dtype=torch.float32, device=device)
    return torch.empty((numel_or_shape,), dtype=torch.float32, device=device)


def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.empty(0, dtype=torch.int, device=device)


def get_scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def marlin_permute_scales(
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    del size_k, group_size, is_a_8bit
    return scales.reshape((-1, size_n)).contiguous()


def marlin_permute_bias(bias: torch.Tensor) -> torch.Tensor:
    origin_shape = bias.shape
    _, scale_perm_single = get_scale_perms()
    bias = bias.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return bias.reshape(*origin_shape).contiguous()


def run_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    b_type_id: int,
    size_m: int,
    size_n: int,
    size_k: int,
    c_tmp: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    b_bias: torch.Tensor | None = None,
    a_scales: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    b_zeros: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = False,
    is_zp_float: bool | None = None,
) -> torch.Tensor:
    if is_zp_float is None:
        is_zp_float = b_zeros is not None
    validate_dense_marlin_call(
        b_type_id=b_type_id,
        size_k=size_k,
        num_groups=int(b_scales.size(0)),
        g_idx=g_idx,
        perm=perm,
        is_k_full=is_k_full,
    )
    return ops.marlin_gemm(
        a,
        c,
        b_q_weight,
        b_bias,
        b_scales,
        a_scales,
        global_scale,
        b_zeros,
        g_idx,
        perm,
        c_tmp,
        b_type_id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = max_num_tokens_padded // block_size + 1
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        expert_map,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def fused_marlin_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    quant_type_id: int,
    moe_block_size: int = 16,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    c_tmp: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
    is_w1_zp_float: bool = False,
    is_w2_zp_float: bool = False,
    is_k_full: bool = True,
) -> torch.Tensor:
    m, k = hidden_states.shape
    topk = topk_ids.shape[1]
    validate_moe_marlin_call(
        b_type_id=quant_type_id,
        size_k=k,
        num_groups=int(w1_scale.size(1)),
        g_idx=g_idx1,
        perm=sort_indices1,
        is_k_full=is_k_full,
    )
    validate_moe_marlin_call(
        b_type_id=quant_type_id,
        size_k=int(w2.shape[1] * 16),
        num_groups=int(w2_scale.size(1)),
        g_idx=g_idx2,
        perm=sort_indices2,
        is_k_full=is_k_full,
    )
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids, moe_block_size, w1.shape[0]
    )

    gate_up_width = int(w1_scale.shape[-1])
    if gate_up_width % 2 != 0:
        raise ValueError(f"w1_scale output width must be even, got {gate_up_width}")
    n = gate_up_width // 2
    intermediate = torch.empty((m * topk, gate_up_width), dtype=hidden_states.dtype, device=hidden_states.device)
    intermediate = ops.moe_wna16_marlin_gemm(
        hidden_states,
        intermediate,
        w1,
        bias1,
        w1_scale,
        None,
        global_scale1,
        w1_zeros,
        g_idx1,
        sort_indices1,
        c_tmp,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        moe_block_size,
        topk,
        False,
        quant_type_id,
        m,
        gate_up_width,
        k,
        is_k_full,
        False,
        False,
        is_w1_zp_float,
        -1,
        -1,
        -1,
    )
    gate, up = intermediate.view(m * topk, gate_up_width).chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    output = torch.empty((m * topk, k), dtype=hidden_states.dtype, device=hidden_states.device)
    output = ops.moe_wna16_marlin_gemm(
        activated,
        output,
        w2,
        bias2,
        w2_scale,
        None,
        global_scale2,
        w2_zeros,
        g_idx2,
        sort_indices2,
        c_tmp,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        moe_block_size,
        1,
        True,
        quant_type_id,
        m * topk,
        k,
        n,
        is_k_full,
        False,
        False,
        is_w2_zp_float,
        -1,
        -1,
        -1,
    )
    return output.view(m, topk, k).sum(dim=1)


def topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = False,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=gating_output.device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=gating_output.device)
    token_expert_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=gating_output.device)
    ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        bias,
    )
    return topk_weights, topk_ids, token_expert_indices


def topk_sigmoid(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = False,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = gating_output.shape[0]
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=gating_output.device)
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=gating_output.device)
    token_expert_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=gating_output.device)
    ops.topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        bias,
    )
    return topk_weights, topk_ids, token_expert_indices


def grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ops.grouped_topk(
        scores,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


def _supported_quant_types() -> tuple[ScalarType, ...]:
    return (scalar_types.uint4b8, scalar_types.uint8b128)


def _supported_unpack_quant_types() -> tuple[ScalarType, ...]:
    return (
        scalar_types.uint4,
        scalar_types.uint4b8,
        scalar_types.uint8,
        scalar_types.uint8b128,
        scalar_types.float4_e2m1f,
        scalar_types.float8_e4m3fn,
    )


def _is_fp8_quant_type(quant_type: ScalarType) -> bool:
    return quant_type == scalar_types.float8_e4m3fn


def _is_nvfp4_quant_type(quant_type: ScalarType) -> bool:
    return quant_type == scalar_types.float4_e2m1f


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


def unpack_cols(
    q_packed: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    if q_packed.shape != (size_k, size_n // quant_utils.get_pack_factor(num_bits)):
        raise ValueError(
            "Expected q_packed.shape == "
            f"{(size_k, size_n // quant_utils.get_pack_factor(num_bits))}, got {tuple(q_packed.shape)}"
        )

    pack_factor = quant_utils.get_pack_factor(num_bits)
    unpacked = torch.empty((size_k, size_n), dtype=torch.int32, device=q_packed.device)
    mask = (1 << num_bits) - 1
    for idx in range(pack_factor):
        unpacked[:, idx::pack_factor] = ((q_packed.to(torch.int64) >> (num_bits * idx)) & mask).to(
            torch.int32
        )
    return unpacked.contiguous()


def pack_uint4_zero_points(zero_points: torch.Tensor, size_k: int, size_n: int) -> torch.Tensor:
    if zero_points.shape != (size_k, size_n):
        raise ValueError(
            f"Expected zero_points.shape == {(size_k, size_n)}, got {tuple(zero_points.shape)}"
        )

    order = torch.tensor(quant_utils._SM70_U4_PACK_ORDER, device=zero_points.device, dtype=torch.long)
    reshaped = zero_points.reshape(size_k, size_n // 8, 8).index_select(2, order)
    packed = torch.zeros((size_k, size_n // 8), dtype=torch.int64, device=zero_points.device)
    for idx in range(8):
        packed |= reshaped[:, :, idx].to(torch.int64) << (4 * idx)

    words_per_tile = quant_utils._SM70_U4_ZERO_WORDS_PER_CTA_N
    full_tiles = packed.shape[1] // words_per_tile
    if full_tiles > 0:
        pair_order = torch.tensor(
            quant_utils._SM70_U4_ZERO_WORD_PAIR_ORDER,
            device=zero_points.device,
            dtype=torch.long,
        )
        tiled = packed[:, : full_tiles * words_per_tile].reshape(
            size_k, full_tiles, words_per_tile
        )
        paired = tiled.index_select(2, pair_order).reshape(
            size_k, full_tiles * words_per_tile
        )
        if full_tiles * words_per_tile != packed.shape[1]:
            packed = torch.cat((paired, packed[:, full_tiles * words_per_tile :]), dim=1)
        else:
            packed = paired
    return packed.to(torch.int32).contiguous()


def unpack_uint4_zero_points(
    packed_zero_points: torch.Tensor,
    size_k: int,
    size_n: int,
) -> torch.Tensor:
    if packed_zero_points.shape != (size_k, size_n // 8):
        raise ValueError(
            "Expected packed_zero_points.shape == "
            f"{(size_k, size_n // 8)}, got {tuple(packed_zero_points.shape)}"
        )

    unpacked = torch.empty((size_k, size_n), dtype=torch.int32, device=packed_zero_points.device)
    words = packed_zero_points.to(torch.int64)
    words_per_tile = quant_utils._SM70_U4_ZERO_WORDS_PER_CTA_N
    full_tiles = words.shape[1] // words_per_tile
    if full_tiles > 0:
        pair_order = torch.tensor(
            quant_utils._SM70_U4_ZERO_WORD_PAIR_ORDER,
            device=packed_zero_points.device,
            dtype=torch.long,
        )
        inverse_order = torch.empty_like(pair_order)
        inverse_order[pair_order] = torch.arange(
            words_per_tile, device=packed_zero_points.device, dtype=torch.long
        )
        tiled = words[:, : full_tiles * words_per_tile].reshape(
            size_k, full_tiles, words_per_tile
        )
        unpaired = tiled.index_select(2, inverse_order).reshape(
            size_k, full_tiles * words_per_tile
        )
        if full_tiles * words_per_tile != words.shape[1]:
            words = torch.cat((unpaired, words[:, full_tiles * words_per_tile :]), dim=1)
        else:
            words = unpaired
    for word_idx in range(size_n // 8):
        packed_vals = [((words[:, word_idx] >> (4 * idx)) & 0xF).to(torch.int32) for idx in range(8)]
        logical_vals = [torch.empty_like(packed_vals[0]) for _ in range(8)]
        for out_idx, src_idx in enumerate(quant_utils._SM70_U4_PACK_ORDER):
            logical_vals[src_idx] = packed_vals[out_idx]
        for idx, values in enumerate(logical_vals):
            unpacked[:, word_idx * 8 + idx] = values
    return unpacked.contiguous()


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
    size_n: int = 128,
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


def _quantize_uint4_with_zero_point(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size_k, size_n = weight.shape
    if group_size == -1:
        group_size = size_k
    if size_k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide size_k={size_k}")

    groups = size_k // group_size
    reshaped = weight.to(torch.float32).reshape(groups, group_size, size_n)
    mins = reshaped.amin(dim=1)
    maxs = reshaped.amax(dim=1)
    scales = ((maxs - mins) / 15.0).clamp_min(1e-6)
    zero_points = torch.round(-mins / scales).clamp(0, 15).to(torch.int32)
    q = torch.round(reshaped / scales.unsqueeze(1) + zero_points.unsqueeze(1))
    q = q.clamp(0, 15).to(torch.int32).reshape(size_k, size_n)
    return q, scales.to(weight.dtype), zero_points.contiguous()


def _quantize_uint8_with_zero_point(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size_k, size_n = weight.shape
    if group_size == -1:
        group_size = size_k
    if size_k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide size_k={size_k}")

    groups = size_k // group_size
    reshaped = weight.to(torch.float32).reshape(groups, group_size, size_n)
    mins = reshaped.amin(dim=1)
    maxs = reshaped.amax(dim=1)
    scales = ((maxs - mins) / 255.0).clamp_min(1e-6)
    zero_points = torch.round(-mins / scales).clamp(0, 255).to(torch.int32)
    q = torch.round(reshaped / scales.unsqueeze(1) + zero_points.unsqueeze(1))
    q = q.clamp(0, 255).to(torch.int32).reshape(size_k, size_n)
    return q, scales.to(weight.dtype), zero_points.contiguous()


def _fp8_fused_exponent_bias_into_scales(scales: torch.Tensor) -> torch.Tensor:
    return scales * 256.0


def fp8_weight_to_marlin_weight(fp8_weight: torch.Tensor) -> torch.Tensor:
    if fp8_weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Expected torch.float8_e4m3fn weight, got {fp8_weight.dtype}")
    if fp8_weight.ndim != 2:
        raise ValueError(f"Expected 2D FP8 weight, got rank {fp8_weight.ndim}")

    size_k, size_n = fp8_weight.shape
    raw_bytes = fp8_weight.contiguous().view(torch.uint8).to(torch.int32)
    weight_perm = quant_utils.get_weight_perm(8, is_a_8bit=False)
    return quant_utils.marlin_weights(
        raw_bytes,
        size_k,
        size_n,
        8,
        weight_perm,
        is_a_8bit=False,
    )


def fp4_e2m1_weight_to_marlin_weight(fp4_weight: torch.Tensor) -> torch.Tensor:
    if fp4_weight.ndim != 2:
        raise ValueError(f"Expected 2D FP4 nibble weight, got rank {fp4_weight.ndim}")
    if fp4_weight.min().item() < 0 or fp4_weight.max().item() > 15:
        raise ValueError("FP4 nibble weight values must be in [0, 15].")

    size_k, size_n = fp4_weight.shape
    weight_perm = quant_utils.get_weight_perm(4, is_a_8bit=False)
    return quant_utils.marlin_weights(
        fp4_weight.to(torch.int32),
        size_k,
        size_n,
        4,
        weight_perm,
        is_a_8bit=False,
    )


def _fp4_e2m1_values(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        dtype=torch.float32,
        device=device,
    )


def _quantize_to_fp4_e2m1(values: torch.Tensor) -> torch.Tensor:
    fp4_values = _fp4_e2m1_values(values.device)
    distances = (values.to(torch.float32).unsqueeze(-1) - fp4_values).abs()
    return distances.argmin(dim=-1).to(torch.int32)


def _nvfp4_compute_scale_factor(
    marlin_scales: torch.Tensor,
    a_dtype: torch.dtype | None = None,
) -> float:
    if a_dtype is not None and a_dtype == torch.float16:
        return 1.0

    ws_float = marlin_scales.float() * (2**7)
    nonzero_mask = ws_float > 0
    if nonzero_mask.any():
        min_val = ws_float[nonzero_mask].min()
        if min_val < 2:
            return (2 / min_val).log2().ceil().exp2().item()
    return 1.0


def _nvfp4_marlin_process_scales(
    marlin_scales: torch.Tensor,
    scale_factor: float | None = None,
    a_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, float]:
    marlin_scales = marlin_scales.to(torch.float16)
    marlin_scales = marlin_scales.view(-1, 4)[:, [0, 2, 1, 3]].reshape(
        marlin_scales.shape
    )

    if scale_factor is None:
        scale_factor = _nvfp4_compute_scale_factor(marlin_scales, a_dtype)
    if scale_factor > 1.0:
        marlin_scales = (marlin_scales.float() * scale_factor).to(torch.float16)

    marlin_scales = (marlin_scales * (2**7)).view(torch.int16) << 1
    marlin_scales = marlin_scales.view(torch.float8_e4m3fn)
    marlin_scales = marlin_scales[:, 1::2].contiguous()
    return marlin_scales, scale_factor


def _nvfp4_marlin_process_global_scale(
    global_scale: torch.Tensor,
    a_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if a_dtype is None:
        a_dtype = torch.float16
    if a_dtype != torch.float16:
        raise ValueError("SM70 NVFP4 dense helper currently expects fp16 activations.")

    fp4_exponent = 2
    target_exponent = 5
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


def _decode_nvfp4_marlin_fast_scales(scales: torch.Tensor) -> torch.Tensor:
    scale_bits = scales.view(torch.uint8).to(torch.int16) << 7
    decoded = scale_bits.contiguous().view(torch.float16).to(torch.float32)
    return decoded.view(-1, 4)[:, [0, 2, 1, 3]].reshape(scales.shape)


def _quantize_nvfp4_weight(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_size != 16:
        raise ValueError("SM70 NVFP4 dense helper supports only group_size=16.")

    size_k, size_n = weight.shape
    if size_k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide size_k={size_k}")

    groups = size_k // group_size
    reshaped = weight.to(torch.float32).reshape(groups, group_size, size_n)
    scales = (reshaped.abs().amax(dim=1) / 6.0).clamp_min(1e-6)
    global_scale = (scales.max() / 448.0).to(torch.float32).reshape(1)
    fp8_scales = (scales / global_scale).to(torch.float8_e4m3fn)
    effective_scales = fp8_scales.to(torch.float32) * global_scale
    q_weight = _quantize_to_fp4_e2m1(
        reshaped / effective_scales.unsqueeze(1).clamp_min(1e-12)
    ).reshape(size_k, size_n)
    fp4_values = _fp4_e2m1_values(weight.device)
    repeated_scales = effective_scales.repeat_interleave(group_size, dim=0)
    dequantized = (fp4_values[q_weight.to(torch.long)] * repeated_scales).to(torch.float16)
    return q_weight, fp8_scales, global_scale, dequantized


def _quantize_mxfp4_weight(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if group_size != 32:
        raise ValueError("SM70 MXFP4 dense helper supports only group_size=32.")

    size_k, size_n = weight.shape
    if size_k % group_size != 0:
        raise ValueError(f"group_size={group_size} must divide size_k={size_k}")

    groups = size_k // group_size
    reshaped = weight.to(torch.float32).reshape(groups, group_size, size_n)
    max_abs = reshaped.abs().amax(dim=1)
    scale_inputs = torch.where(
        max_abs > 0,
        (max_abs / 6.0).clamp_min(2.0**-127),
        torch.ones_like(max_abs),
    )
    scale_exponents = torch.ceil(torch.log2(scale_inputs)).clamp(-127, 127)
    power_of_two_scales = torch.pow(torch.full_like(scale_exponents, 2.0), scale_exponents)
    fp8_scales = power_of_two_scales.to(torch.float8_e8m0fnu)
    effective_scales = fp8_scales.to(torch.float32)

    q_weight = _quantize_to_fp4_e2m1(
        reshaped / effective_scales.unsqueeze(1).clamp_min(1e-12)
    ).reshape(size_k, size_n)
    fp4_values = _fp4_e2m1_values(weight.device)
    repeated_scales = effective_scales.repeat_interleave(group_size, dim=0)
    dequantized = (fp4_values[q_weight.to(torch.long)] * repeated_scales).to(torch.float16)
    return q_weight, fp8_scales, dequantized


def _quantize_fp8_weight(
    weight: torch.Tensor, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    size_k, size_n = weight.shape
    if group_size == -1:
        runtime_group_size = size_k
    elif group_size == 128:
        runtime_group_size = group_size
    else:
        raise ValueError("FP8 dense helper supports only group_size=-1 or 128.")
    if size_k % runtime_group_size != 0:
        raise ValueError(f"group_size={runtime_group_size} must divide size_k={size_k}")

    groups = size_k // runtime_group_size
    reshaped = weight.to(torch.float32).reshape(groups, runtime_group_size, size_n)
    scales = (reshaped.abs().amax(dim=1) / 448.0).clamp_min(1e-6)
    fp8_weight = (reshaped / scales.unsqueeze(1)).to(torch.float8_e4m3fn).reshape(
        size_k, size_n
    )
    repeated_scales = scales.repeat_interleave(runtime_group_size, dim=0)[:size_k]
    dequantized = (fp8_weight.to(torch.float32) * repeated_scales).to(torch.float16)
    return fp8_weight, scales.to(weight.dtype), dequantized


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
    if _is_fp8_quant_type(quant_type):
        return marlin_quantize_fp8(weight, group_size, act_order)

    if quant_type not in _supported_quant_types():
        raise ValueError("Local test helper currently supports uint4b8 and uint8b128 only.")
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
    marlin_scales = marlin_permute_scales(
        scales,
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    return weight, marlin_q_weight, marlin_scales, g_idx, sort_indices, rand_perm


def marlin_quantize_fp8(
    weight: torch.Tensor,
    group_size: int,
    act_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if act_order:
        raise ValueError("SM70 FP8 dense helper does not support act_order.")
    if group_size not in (-1, 128):
        raise ValueError("SM70 FP8 dense helper supports only group_size=-1 or 128.")

    size_k, size_n = weight.shape
    fp8_weight, scales, _dequantized = _quantize_fp8_weight(weight, group_size)
    marlin_q_weight = fp8_weight_to_marlin_weight(fp8_weight)
    marlin_scales = marlin_permute_scales(
        _fp8_fused_exponent_bias_into_scales(scales).to(torch.float16),
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    g_idx = marlin_make_empty_g_idx(weight.device)
    sort_indices = torch.empty(0, dtype=torch.int, device=weight.device)
    rand_perm = torch.arange(size_k, dtype=torch.int, device=weight.device)
    return weight, marlin_q_weight, marlin_scales, g_idx, sort_indices, rand_perm


def marlin_quantize_nvfp4(
    weight: torch.Tensor,
    group_size: int,
    act_order: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if act_order:
        raise ValueError("SM70 NVFP4 dense helper does not support act_order.")
    if group_size != 16:
        raise ValueError("SM70 NVFP4 dense helper supports only group_size=16.")

    size_k, size_n = weight.shape
    q_weight, fp8_scales, global_scale, dequantized = _quantize_nvfp4_weight(
        weight,
        group_size,
    )
    marlin_q_weight = fp4_e2m1_weight_to_marlin_weight(q_weight)
    raw_marlin_scales = marlin_permute_scales(
        fp8_scales,
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    marlin_scales, scale_factor = _nvfp4_marlin_process_scales(
        raw_marlin_scales,
        a_dtype=weight.dtype,
    )
    marlin_global_scale = _nvfp4_marlin_process_global_scale(
        global_scale.reshape(1).contiguous(),
        weight.dtype,
    ).to(torch.float32)
    marlin_global_scale = (marlin_global_scale / scale_factor).contiguous()
    dequantized = marlin_dequantize_nvfp4(
        marlin_q_weight,
        marlin_scales,
        marlin_global_scale,
        size_k,
        size_n,
        group_size,
    )
    g_idx = marlin_make_empty_g_idx(weight.device)
    sort_indices = torch.empty(0, dtype=torch.int, device=weight.device)
    rand_perm = torch.arange(size_k, dtype=torch.int, device=weight.device)
    return (
        dequantized,
        marlin_q_weight,
        marlin_scales,
        marlin_global_scale,
        g_idx,
        sort_indices,
        rand_perm,
    )


def marlin_quantize_mxfp4(
    weight: torch.Tensor,
    group_size: int,
    act_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if act_order:
        raise ValueError("SM70 MXFP4 dense helper does not support act_order.")
    if group_size != 32:
        raise ValueError("SM70 MXFP4 dense helper supports only group_size=32.")

    size_k, size_n = weight.shape
    q_weight, fp8_scales, dequantized = _quantize_mxfp4_weight(weight, group_size)
    marlin_q_weight = fp4_e2m1_weight_to_marlin_weight(q_weight)
    marlin_scales = marlin_permute_scales(
        fp8_scales,
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    g_idx = marlin_make_empty_g_idx(weight.device)
    sort_indices = torch.empty(0, dtype=torch.int, device=weight.device)
    rand_perm = torch.arange(size_k, dtype=torch.int, device=weight.device)
    return dequantized, marlin_q_weight, marlin_scales, g_idx, sort_indices, rand_perm


def marlin_dequantize(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    quant_type: ScalarType,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    if _is_nvfp4_quant_type(quant_type):
        raise ValueError(
            "Use marlin_dequantize_nvfp4 or marlin_dequantize_mxfp4 for "
            "preconverted FP4 weights."
        )
    if quant_type not in _supported_quant_types() and not _is_fp8_quant_type(quant_type):
        raise ValueError("Local test helper currently supports uint4b8, uint8b128, and fp8 only.")
    unpacked = marlin_unpack(q_weight, size_k, size_n, quant_type).to(torch.float32)
    unpermuted_scales = marlin_unpermute_scales(scales, size_k, size_n, group_size, quant_type)
    if group_size == -1:
        group_size = size_k
    expanded_scales = unpermuted_scales.repeat_interleave(group_size, dim=0)[:size_k]
    if _is_fp8_quant_type(quant_type):
        fp8_values = unpacked.to(torch.uint8).contiguous().view(torch.float8_e4m3fn)
        dequantized = (fp8_values.to(torch.float32) * (expanded_scales.to(torch.float32) / 256.0)).to(
            torch.float16
        )
        if perm is not None and perm.numel() > 0:
            logical = torch.empty_like(dequantized)
            logical[perm.to(torch.long)] = dequantized
            return logical
        return dequantized
    dequantized = ((unpacked - float(quant_type.bias)) * expanded_scales.to(torch.float32)).to(
        torch.float16
    )
    if perm is not None and perm.numel() > 0:
        logical = torch.empty_like(dequantized)
        logical[perm.to(torch.long)] = dequantized
        return logical
    return dequantized


def marlin_dequantize_nvfp4(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    global_scale: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    if group_size != 16:
        raise ValueError("SM70 NVFP4 dense helper supports only group_size=16.")
    unpacked = marlin_unpack(q_weight, size_k, size_n, scalar_types.float4_e2m1f)
    fp4_values = _fp4_e2m1_values(q_weight.device) / float(2**14)
    decoded_scales = _decode_nvfp4_marlin_fast_scales(scales)
    unpermuted_scales = marlin_unpermute_scales(
        decoded_scales,
        size_k,
        size_n,
        group_size,
        scalar_types.float4_e2m1f,
    ).to(torch.float32)
    expanded_scales = unpermuted_scales.repeat_interleave(group_size, dim=0)[:size_k]
    dequantized = (
        fp4_values[unpacked.to(torch.long)]
        * expanded_scales
        * global_scale.reshape(-1)[0].to(torch.float32)
    )
    return dequantized.to(torch.float16)


def marlin_dequantize_mxfp4(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    if group_size != 32:
        raise ValueError("SM70 MXFP4 dense helper supports only group_size=32.")
    unpacked = marlin_unpack(q_weight, size_k, size_n, scalar_types.float4_e2m1f)
    fp4_values = _fp4_e2m1_values(q_weight.device)
    unpermuted_scales = marlin_unpermute_scales(
        scales,
        size_k,
        size_n,
        group_size,
        scalar_types.float4_e2m1f,
    ).to(torch.float32)
    expanded_scales = unpermuted_scales.repeat_interleave(group_size, dim=0)[:size_k]
    dequantized = fp4_values[unpacked.to(torch.long)] * expanded_scales
    return dequantized.to(torch.float16)


def marlin_quantize_uint4_packed_zp(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_dense_group_size(group_size)
    size_k, size_n = weight.shape
    q_weight, scales, zero_points = _quantize_uint4_with_zero_point(weight, group_size)
    weight_perm = quant_utils.get_weight_perm(scalar_types.uint4.size_bits, is_a_8bit=False)
    marlin_q_weight = quant_utils.marlin_weights(
        q_weight,
        size_k,
        size_n,
        scalar_types.uint4.size_bits,
        weight_perm,
        is_a_8bit=False,
    )
    marlin_scales = marlin_permute_scales(
        scales,
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    packed_zero_points = pack_uint4_zero_points(
        zero_points,
        zero_points.shape[0],
        size_n,
    )
    dequantized = marlin_dequantize_uint4_packed_zp(
        marlin_q_weight,
        marlin_scales,
        packed_zero_points,
        size_k,
        size_n,
        group_size,
    )
    return weight, marlin_q_weight, marlin_scales, packed_zero_points, dequantized


def marlin_quantize_uint4_zp(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_dense_group_size(group_size)
    size_k, size_n = weight.shape
    q_weight, scales, zero_points = _quantize_uint4_with_zero_point(weight, group_size)
    weight_perm = quant_utils.get_weight_perm(scalar_types.uint4.size_bits, is_a_8bit=False)
    marlin_q_weight = quant_utils.marlin_weights(
        q_weight,
        size_k,
        size_n,
        scalar_types.uint4.size_bits,
        weight_perm,
        is_a_8bit=False,
    )
    marlin_scales = marlin_permute_scales(
        scales,
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    zp = (zero_points.to(torch.float32) * scales.to(torch.float32)).to(weight.dtype)
    marlin_zp = marlin_permute_scales(
        zp,
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    dequantized = marlin_dequantize_uint4_zp(
        marlin_q_weight,
        marlin_scales,
        marlin_zp,
        size_k,
        size_n,
        group_size,
    )
    return weight, marlin_q_weight, marlin_scales, marlin_zp, dequantized


def marlin_quantize_uint8_zp(
    weight: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    validate_dense_group_size(group_size)
    size_k, size_n = weight.shape
    q_weight, scales, zero_points = _quantize_uint8_with_zero_point(weight, group_size)
    weight_perm = quant_utils.get_weight_perm(scalar_types.uint8.size_bits, is_a_8bit=False)
    marlin_q_weight = quant_utils.marlin_weights(
        q_weight,
        size_k,
        size_n,
        scalar_types.uint8.size_bits,
        weight_perm,
        is_a_8bit=False,
    )
    marlin_scales = marlin_permute_scales(
        scales,
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    zp = (zero_points.to(torch.float32) * scales.to(torch.float32)).to(weight.dtype)
    marlin_zp = marlin_permute_scales(
        zp,
        size_k,
        size_n,
        group_size,
        is_a_8bit=False,
    )
    dequantized = marlin_dequantize_uint8_zp(
        marlin_q_weight,
        marlin_scales,
        marlin_zp,
        size_k,
        size_n,
        group_size,
    )
    return weight, marlin_q_weight, marlin_scales, marlin_zp, dequantized


@dataclass
class MarlinLinearKernelCase:
    quant_name: str
    group_size: int
    num_groups: int
    kernel: object
    layer: torch.nn.Module
    activation: torch.Tensor
    weight: torch.Tensor
    q_weight: torch.Tensor
    scales: torch.Tensor
    zero_points: torch.Tensor
    dequantized: torch.Tensor
    reference: torch.Tensor | None = None
    output: torch.Tensor | None = None


def _marlin_linear_kernel_quant_config(quant_name: str):
    from vllm.scalar_type import scalar_types as vllm_scalar_types

    if quant_name == "uint4":
        return (
            scalar_types.uint4,
            vllm_scalar_types.uint4,
            _quantize_uint4_with_zero_point,
            marlin_dequantize_uint4_zp,
            True,
        )
    if quant_name == "uint8":
        return (
            scalar_types.uint8,
            vllm_scalar_types.uint8,
            _quantize_uint8_with_zero_point,
            marlin_dequantize_uint8_zp,
            True,
        )
    if quant_name == "uint4b8":
        return (
            scalar_types.uint4b8,
            vllm_scalar_types.uint4b8,
            lambda weight, group_size: (
                *_quantize_unsigned_with_bias(weight, group_size, scalar_types.uint4b8.bias),
                None,
            ),
            marlin_dequantize,
            False,
        )
    if quant_name == "uint8b128":
        return (
            scalar_types.uint8b128,
            vllm_scalar_types.uint8b128,
            lambda weight, group_size: (
                *_quantize_unsigned_with_bias(weight, group_size, scalar_types.uint8b128.bias),
                None,
            ),
            marlin_dequantize,
            False,
        )
    if quant_name == "fp8":
        def quantize_fp8_for_kernel(
            weight: torch.Tensor,
            group_size: int,
        ) -> tuple[torch.Tensor, torch.Tensor, None]:
            fp8_weight, scales, _dequantized = _quantize_fp8_weight(
                weight,
                group_size,
            )
            raw_q_weight = fp8_weight.contiguous().view(torch.uint8).to(torch.int32)
            fused_scales = _fp8_fused_exponent_bias_into_scales(scales).to(
                weight.dtype
            )
            return raw_q_weight, fused_scales, None

        return (
            scalar_types.float8_e4m3fn,
            vllm_scalar_types.float8_e4m3fn,
            quantize_fp8_for_kernel,
            marlin_dequantize,
            False,
        )
    raise ValueError(f"Unsupported MarlinLinearKernel quant_name={quant_name!r}")


def prepare_marlin_linear_kernel_case(
    *,
    quant_name: str,
    group_size: int,
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> MarlinLinearKernelCase:
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.marlin import (
        MarlinLinearKernel,
    )
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    validate_dense_group_size(group_size)
    if activation.dtype != torch.float16:
        raise ValueError(f"Expected fp16 activation, got {activation.dtype}")
    if weight.dtype != torch.float16:
        raise ValueError(f"Expected fp16 weight, got {weight.dtype}")

    size_k, size_n = weight.shape
    if activation.shape[-1] != size_k:
        raise ValueError(
            f"activation.shape[-1]={activation.shape[-1]} does not match size_k={size_k}"
        )

    local_quant_type, vllm_quant_type, quantize, dequantize, has_zp = (
        _marlin_linear_kernel_quant_config(quant_name)
    )
    quantized = quantize(weight, group_size)
    q_weight = quantized[0]
    scales = quantized[1]
    zero_points = quantized[2]
    num_bits = local_quant_type.size_bits
    num_groups = scales.shape[0]
    packed_weight = gptq_pack(q_weight, num_bits, size_k, size_n)

    layer = torch.nn.Module()
    layer.register_parameter(
        "qweight",
        PackedvLLMParameter(
            packed_weight.contiguous(),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
        ),
    )
    layer.register_parameter(
        "scales",
        GroupQuantScaleParameter(
            scales.contiguous(),
            input_dim=0,
            output_dim=1,
        ),
    )
    if has_zp:
        assert zero_points is not None
        packed_zero_points = pack_cols(
            zero_points,
            num_bits,
            num_groups,
            size_n,
        ).t().contiguous()
        layer.register_parameter(
            "qzeros",
            PackedvLLMParameter(
                packed_zero_points,
                input_dim=1,
                output_dim=0,
                packed_dim=0,
            ),
        )

    config = MPLinearLayerConfig(
        full_weight_shape=(size_k, size_n),
        partition_weight_shape=(size_k, size_n),
        weight_type=vllm_quant_type,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=has_zp,
        has_g_idx=False,
    )
    kernel = MarlinLinearKernel(
        config,
        "qweight",
        "scales",
        "qzeros" if has_zp else None,
    )
    kernel.process_weights_after_loading(layer)

    if has_zp:
        dequantized = dequantize(
            layer.qweight,
            layer.scales,
            layer.qzeros,
            size_k,
            size_n,
            group_size,
        )
    else:
        dequantized = dequantize(
            layer.qweight,
            layer.scales,
            size_k,
            size_n,
            group_size,
            local_quant_type,
        )
    return MarlinLinearKernelCase(
        quant_name=quant_name,
        group_size=group_size,
        num_groups=num_groups,
        kernel=kernel,
        layer=layer,
        activation=activation,
        weight=weight,
        q_weight=q_weight,
        scales=scales,
        zero_points=zero_points
        if zero_points is not None
        else torch.empty(0, dtype=torch.int32, device=weight.device),
        dequantized=dequantized,
    )


def run_marlin_linear_kernel_case(
    *,
    quant_name: str,
    group_size: int,
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> MarlinLinearKernelCase:
    case = prepare_marlin_linear_kernel_case(
        quant_name=quant_name,
        group_size=group_size,
        activation=activation,
        weight=weight,
    )
    case.output = case.kernel.apply_weights(case.layer, activation)
    case.reference = torch.matmul(
        activation.to(torch.float32),
        case.dequantized.to(torch.float32),
    ).to(torch.float16)
    return case


def marlin_dequantize_uint4_packed_zp(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    packed_zero_points: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    unpacked = _marlin_unpack_impl(q_weight, size_k, size_n, scalar_types.uint4).to(torch.float32)
    unpermuted_scales = _marlin_unpermute_scales_impl(
        scales,
        size_k,
        size_n,
        group_size,
    ).to(torch.float32)
    unpacked_zero_points = unpack_uint4_zero_points(
        packed_zero_points,
        unpermuted_scales.shape[0],
        size_n,
    ).to(torch.float32)
    if group_size == -1:
        group_size = size_k
    expanded_scales = unpermuted_scales.repeat_interleave(group_size, dim=0)[:size_k]
    expanded_zero_points = unpacked_zero_points.repeat_interleave(group_size, dim=0)[:size_k]
    dequantized = ((unpacked - expanded_zero_points) * expanded_scales).to(torch.float16)
    if perm is not None and perm.numel() > 0:
        logical = torch.empty_like(dequantized)
        logical[perm.to(torch.long)] = dequantized
        return logical
    return dequantized


def marlin_dequantize_uint4_zp(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zp: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    unpacked = _marlin_unpack_impl(q_weight, size_k, size_n, scalar_types.uint4).to(torch.float32)
    unpermuted_scales = _marlin_unpermute_scales_impl(
        scales,
        size_k,
        size_n,
        group_size,
    ).to(torch.float32)
    unpermuted_zp = _marlin_unpermute_scales_impl(
        zp,
        size_k,
        size_n,
        group_size,
    ).to(torch.float32)
    if group_size == -1:
        group_size = size_k
    expanded_scales = unpermuted_scales.repeat_interleave(group_size, dim=0)[:size_k]
    expanded_zp = unpermuted_zp.repeat_interleave(group_size, dim=0)[:size_k]
    dequantized = (unpacked * expanded_scales - expanded_zp).to(torch.float16)
    if perm is not None and perm.numel() > 0:
        logical = torch.empty_like(dequantized)
        logical[perm.to(torch.long)] = dequantized
        return logical
    return dequantized


def marlin_dequantize_uint8_zp(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zp: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    unpacked = _marlin_unpack_impl(q_weight, size_k, size_n, scalar_types.uint8).to(torch.float32)
    unpermuted_scales = _marlin_unpermute_scales_impl(
        scales,
        size_k,
        size_n,
        group_size,
    ).to(torch.float32)
    unpermuted_zp = _marlin_unpermute_scales_impl(
        zp,
        size_k,
        size_n,
        group_size,
    ).to(torch.float32)
    if group_size == -1:
        group_size = size_k
    expanded_scales = unpermuted_scales.repeat_interleave(group_size, dim=0)[:size_k]
    expanded_zp = unpermuted_zp.repeat_interleave(group_size, dim=0)[:size_k]
    dequantized = (unpacked * expanded_scales - expanded_zp).to(torch.float16)
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
    if quant_type not in _supported_unpack_quant_types():
        raise ValueError(
            "Local test helper currently supports uint4, uint4b8, uint8, uint8b128, and fp8 unpacking."
        )
    return _marlin_unpack_impl(q_weight, size_k, size_n, quant_type)


def _marlin_unpack_impl(
    q_weight: torch.Tensor,
    size_k: int,
    size_n: int,
    quant_type: ScalarType,
) -> torch.Tensor:
    num_bits = quant_type.size_bits
    pack_factor = quant_utils.get_pack_factor(num_bits)
    tile_words = (16 * 64) // pack_factor
    n_tiles = size_n // 64
    packed_words = (
        q_weight.detach().cpu().numpy().astype("uint32", copy=False)
        & 0xFFFFFFFF
    )
    unpacked = torch.empty((size_k, size_n), dtype=torch.int32)
    unpacked_np = unpacked.numpy()

    if num_bits == 4:
        packed = packed_words.reshape(size_k // 16, n_tiles * tile_words)
        for k_tile in range(size_k // 16):
            row_start = 16 * k_tile
            for n_tile in range(n_tiles):
                col_tile_start = 64 * n_tile
                for local_k in range(16):
                    for local_n_vec in range(8):
                        local_word = local_k * 8 + local_n_vec
                        word_offset = quant_utils._sm70_u4_cta_n_offset(
                            n_tiles,
                            n_tile,
                            local_word,
                        )
                        word = int(packed[k_tile, word_offset])
                        packed_vals = [(word >> (num_bits * idx)) & 0xF for idx in range(8)]
                        logical_vals = [0] * 8
                        for out_idx, src_idx in enumerate(quant_utils._SM70_U4_PACK_ORDER):
                            logical_vals[src_idx] = packed_vals[out_idx]
                        for idx, value in enumerate(logical_vals):
                            unpacked_np[
                                row_start + local_k,
                                col_tile_start + local_n_vec * 8 + idx,
                            ] = value
    else:
        packed = packed_words.reshape(size_k // 16, n_tiles * tile_words)
        for k_tile in range(size_k // 16):
            row_start = 16 * k_tile
            for n_tile in range(n_tiles):
                col_tile_start = 64 * n_tile
                for local_k in range(16):
                    for local_n_word in range(16):
                        local_word = local_k * 16 + local_n_word
                        word_offset = quant_utils._sm70_u8_cta_n_offset(
                            n_tiles,
                            n_tile,
                            local_word,
                        )
                        word = int(packed[k_tile, word_offset])
                        packed_vals = [(word >> (num_bits * idx)) & 0xFF for idx in range(4)]
                        logical_vals = [0] * 4
                        for out_idx, src_idx in enumerate(quant_utils._SM70_U8_PACK_ORDER):
                            logical_vals[src_idx] = packed_vals[out_idx]
                        for idx, value in enumerate(logical_vals):
                            unpacked_np[
                                row_start + local_k,
                                col_tile_start + local_n_word * 4 + idx,
                            ] = value

    return unpacked.to(q_weight.device)


def marlin_unpermute_scales(
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
    quant_type: ScalarType,
) -> torch.Tensor:
    if quant_type not in _supported_unpack_quant_types():
        raise ValueError(
            "Local test helper currently supports uint4, uint4b8, uint8, uint8b128, and fp8 scale unpermute."
        )
    return _marlin_unpermute_scales_impl(scales, size_k, size_n, group_size)


def _marlin_unpermute_scales_impl(
    scales: torch.Tensor,
    size_k: int,
    size_n: int,
    group_size: int,
) -> torch.Tensor:
    return scales.reshape(-1, size_n).contiguous()


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


def marlin_quantize_experts_uint4_zp_with_metadata(
    weights: torch.Tensor,
    group_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    q_weights = []
    scales = []
    zero_points = []
    dequantized = []
    g_indices = []
    perms = []
    for expert in range(weights.shape[0]):
        _, q_weight, scale, zero_point, expert_dequantized = marlin_quantize_uint4_zp(
            weights[expert],
            group_size,
        )
        q_weights.append(q_weight)
        scales.append(scale)
        zero_points.append(zero_point)
        dequantized.append(expert_dequantized)
        g_indices.append(torch.empty((0,), dtype=torch.int, device=weights.device))
        perms.append(torch.empty((0,), dtype=torch.int, device=weights.device))
    return (
        torch.stack(q_weights),
        torch.stack(scales),
        torch.stack(zero_points),
        torch.stack(dequantized),
        torch.stack(g_indices),
        torch.stack(perms),
    )


def marlin_quantize_experts_uint8_zp_with_metadata(
    weights: torch.Tensor,
    group_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    q_weights = []
    scales = []
    zero_points = []
    dequantized = []
    g_indices = []
    perms = []
    for expert in range(weights.shape[0]):
        _, q_weight, scale, zero_point, expert_dequantized = marlin_quantize_uint8_zp(
            weights[expert],
            group_size,
        )
        q_weights.append(q_weight)
        scales.append(scale)
        zero_points.append(zero_point)
        dequantized.append(expert_dequantized)
        g_indices.append(torch.empty((0,), dtype=torch.int, device=weights.device))
        perms.append(torch.empty((0,), dtype=torch.int, device=weights.device))
    return (
        torch.stack(q_weights),
        torch.stack(scales),
        torch.stack(zero_points),
        torch.stack(dequantized),
        torch.stack(g_indices),
        torch.stack(perms),
    )


def marlin_quantize_experts_nvfp4_with_metadata(
    weights: torch.Tensor,
    group_size: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    q_weights = []
    scales = []
    global_scales = []
    dequantized = []
    g_indices = []
    perms = []
    for expert in range(weights.shape[0]):
        (
            expert_dequantized,
            q_weight,
            scale,
            global_scale,
            _g_idx,
            _sort_indices,
            _rand_perm,
        ) = marlin_quantize_nvfp4(weights[expert], group_size)
        q_weights.append(q_weight)
        scales.append(scale)
        global_scales.append(global_scale.reshape(-1)[0])
        dequantized.append(expert_dequantized)
        g_indices.append(torch.empty((0,), dtype=torch.int, device=weights.device))
        perms.append(torch.empty((0,), dtype=torch.int, device=weights.device))
    return (
        torch.stack(q_weights),
        torch.stack(scales),
        torch.stack(global_scales).contiguous(),
        torch.stack(dequantized),
        torch.stack(g_indices),
        torch.stack(perms),
    )


def marlin_quantize_experts_mxfp4_with_metadata(
    weights: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_weights = []
    scales = []
    dequantized = []
    g_indices = []
    perms = []
    for expert in range(weights.shape[0]):
        expert_dequantized, q_weight, scale, _g_idx, _sort_indices, _rand_perm = (
            marlin_quantize_mxfp4(weights[expert], group_size)
        )
        q_weights.append(q_weight)
        scales.append(scale)
        dequantized.append(expert_dequantized)
        g_indices.append(torch.empty((0,), dtype=torch.int, device=weights.device))
        perms.append(torch.empty((0,), dtype=torch.int, device=weights.device))
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
    routing_profile: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_states = torch.randn((tokens, hidden), device=device, dtype=dtype)
    if routing_profile is None:
        topk_weights = torch.rand((tokens, topk), device=device, dtype=torch.float32)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        topk_ids = torch.empty((tokens, topk), device=device, dtype=torch.int32)
        for token_idx in range(tokens):
            for route_idx in range(topk):
                topk_ids[token_idx, route_idx] = (token_idx + route_idx) % experts
    else:
        topk_weights, topk_ids = make_moe_routing_tensors(
            tokens=tokens,
            experts=experts,
            topk=topk,
            device=device,
            routing_profile=routing_profile,
        )

    # Use fan-in-scaled weights so the local MoE benchmark checks resemble the
    # activation ranges seen in real models instead of overflowing fp16 paths
    # with unit-variance synthetic weights.
    w1 = torch.randn((experts, hidden, 2 * intermediate), device=device, dtype=dtype)
    w1 = w1 * (1.0 / math.sqrt(hidden))
    w2 = torch.randn((experts, intermediate, hidden), device=device, dtype=dtype)
    w2 = w2 * (1.0 / math.sqrt(intermediate))
    return hidden_states, topk_weights, topk_ids, w1, w2


def make_moe_routing_tensors(
    *,
    tokens: int,
    experts: int,
    topk: int,
    device: torch.device | str,
    routing_profile: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_weights = torch.full(
        (tokens, topk),
        1.0 / float(topk),
        device=device,
        dtype=torch.float32,
    )
    topk_ids = torch.empty((tokens, topk), device=device, dtype=torch.int32)
    if routing_profile == "uniform":
        for token_idx in range(tokens):
            for route_idx in range(topk):
                topk_ids[token_idx, route_idx] = (token_idx + route_idx) % experts
        return topk_weights, topk_ids

    if routing_profile != "zipfian":
        raise ValueError(f"Unsupported routing_profile={routing_profile!r}")

    if experts <= 1:
        topk_ids.zero_()
        return topk_weights, topk_ids

    hot_tokens = (tokens * 3 + 4) // 5
    cold_experts = experts - 1
    for token_idx in range(tokens):
        for route_idx in range(topk):
            if route_idx == 0 and token_idx < hot_tokens:
                expert = 0
            else:
                offset = token_idx + route_idx
                expert = 1 + (offset % cold_experts)
            topk_ids[token_idx, route_idx] = expert
    return topk_weights, topk_ids


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
