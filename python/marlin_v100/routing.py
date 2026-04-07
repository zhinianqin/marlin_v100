from __future__ import annotations

import torch

from . import ops


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
