from __future__ import annotations

import torch

from . import ops
from .calibration import validate_dense_marlin_call


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
    return scales.reshape((-1, size_n)).contiguous()


def marlin_permute_bias(bias: torch.Tensor) -> torch.Tensor:
    origin_shape = bias.shape
    _, scale_perm_single = get_scale_perms()
    bias = bias.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return bias.reshape(*origin_shape).contiguous()


def marlin_make_workspace(
    device: torch.device,
    size: int = 0,
    max_blocks_per_sm: int = 4,
) -> torch.Tensor:
    if size <= 0:
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        size = sms * max_blocks_per_sm
    return torch.zeros(size, dtype=torch.int, device=device)


def run_marlin_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_scales: torch.Tensor,
    b_type_id: int,
    size_m: int,
    size_n: int,
    size_k: int,
    workspace: torch.Tensor | None = None,
    c: torch.Tensor | None = None,
    b_bias: torch.Tensor | None = None,
    a_scales: torch.Tensor | None = None,
    global_scale: torch.Tensor | None = None,
    b_zeros: torch.Tensor | None = None,
    g_idx: torch.Tensor | None = None,
    perm: torch.Tensor | None = None,
    is_k_full: bool = True,
    use_atomic_add: bool = False,
    use_fp32_reduce: bool = True,
    is_zp_float: bool = False,
) -> torch.Tensor:
    if workspace is None:
        workspace = marlin_make_workspace(a.device)
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
        workspace,
        b_type_id,
        size_m,
        size_n,
        size_k,
        is_k_full,
        use_atomic_add,
        use_fp32_reduce,
        is_zp_float,
    )
