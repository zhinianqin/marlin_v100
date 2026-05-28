from __future__ import annotations

import torch

from . import ops
from .calibration import validate_moe_marlin_call


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
    workspace: torch.Tensor | None = None,
    global_scale1: torch.Tensor | None = None,
    global_scale2: torch.Tensor | None = None,
    g_idx1: torch.Tensor | None = None,
    g_idx2: torch.Tensor | None = None,
    sort_indices1: torch.Tensor | None = None,
    sort_indices2: torch.Tensor | None = None,
    w1_zeros: torch.Tensor | None = None,
    w2_zeros: torch.Tensor | None = None,
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
    if c_tmp is None:
        c_tmp = workspace

    # Local quantized expert helpers preserve the logical output width in the
    # scale tensors, which is the most reliable source for the dense width here.
    n = w2_scale.shape[-1]
    intermediate = torch.empty((m * topk, 2 * n), dtype=hidden_states.dtype, device=hidden_states.device)
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
        2 * n,
        k,
        is_k_full,
        False,
        True,
        False,
        -1,
        -1,
        -1,
    )
    gate, up = intermediate.view(m * topk, 2 * n).chunk(2, dim=-1)
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
        True,
        False,
        -1,
        -1,
        -1,
    )
    return output.view(m, topk, k).sum(dim=1)
