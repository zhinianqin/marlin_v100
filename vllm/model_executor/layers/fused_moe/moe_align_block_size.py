from __future__ import annotations

import torch

import vllm._custom_ops as ops


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: torch.Tensor | None = None,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    expert_ids = torch.empty(
        (max_num_tokens_padded // block_size + 1,),
        dtype=torch.int32,
        device=topk_ids.device,
    )
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


def batched_moe_align_block_size(
    max_tokens_per_batch: int,
    block_size: int,
    expert_num_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_experts = expert_num_tokens.numel()
    max_num_tokens_padded = num_experts * (
        max_tokens_per_batch + block_size - 1
    )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=expert_num_tokens.device
    )
    expert_ids = torch.empty(
        (max_num_tokens_padded // block_size + 1,),
        dtype=torch.int32,
        device=expert_num_tokens.device,
    )
    num_tokens_post_pad = torch.empty(
        (num_experts,), dtype=torch.int32, device=expert_num_tokens.device
    )
    ops.batched_moe_align_block_size(
        max_tokens_per_batch,
        block_size,
        expert_num_tokens,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad
