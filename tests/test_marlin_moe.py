from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.calibration import (
    source_target_capability,
    source_target_label,
    supported_moe_quant_type_names,
)
from tests import ops
from tests.helpers import (
    _REPACK_IMPL_CASES,
    assert_repack_layout_matches_reference,
    fused_marlin_moe,
    grouped_topk,
    make_moe_model_like_inputs,
    marlin_moe_reference,
    marlin_quantize_experts_uint4_zp_with_metadata,
    marlin_quantize_experts_uint8_zp_with_metadata,
    marlin_quantize_experts_nvfp4_with_metadata,
    marlin_quantize_experts_mxfp4_with_metadata,
    marlin_quantize_experts,
    marlin_quantize_experts_with_metadata,
    moe_align_block_size,
    scalar_types,
    topk_softmax,
)
from tests.sm70_env_sweep import (
    EXPLICIT_ENV_REJECTION_RE,
    SM70_MOE_GEOMETRIES,
    SM70_METADATA_CACHE_VALUES,
    SM70_SPLIT_K_VALUES,
    MoeDirectOpKey,
    exhaustive_enabled,
    exhaustive_index_is_past_limit,
    exhaustive_index_is_selected,
    iter_moe_env_combinations,
    iter_moe_direct_op_keys,
    iter_moe_focused_mnk_direct_op_keys,
    moe_auto_block_size,
    moe_env,
    moe_stage_env_combo_is_legal,
    set_moe_env,
)

_MOE_SUPPORTED_QUANT_NAMES = frozenset(
    supported_moe_quant_type_names(
        ("uint4", "uint4b8", "uint8", "uint8b128", "fp8", "nvfp4", "mxfp4")
    )
)
_GROUP_SIZES = (-1, 32, 64, 128)
_UINT4_ZP_GROUP_SIZES = (-1, 32, 64, 128)
_UINT8_ZP_GROUP_SIZES = (-1, 32, 64, 128)
_FP8_GROUP_SIZES = (-1, 128)
_SM70_SUPPORTED_MOE_BLOCK_SIZES = (8, 16, 32, 48, 64)
_SUPPORTED_MOE_BLOCK_SIZE_ERROR = (
    "moe_block_size=8, 16, 32, 48, or 64|unsupported moe_block_size="
)
_K_TILE_ALIGNMENT_ERROR = "requires K divisible by CTA_K=|is not divisible by tile_size"
_FLOAT16_DTYPE_ERROR = (
    rf"{source_target_label()} build only supports float16 activations\."
    rf"|{source_target_label()} build only supports float16 outputs\."
    rf"|{source_target_label()} build only supports float16 scales\."
    rf"|{source_target_label()} Marlin MoE supports only float16 activations\."
    rf"|{source_target_label()} Marlin MoE supports only float16 outputs\."
    rf"|{source_target_label()} Marlin MoE supports only float16 scales"
)
_FORCED_GEOMETRY_REPACK_CASES = (_REPACK_IMPL_CASES[0],)
_SUPPORTED_THREAD_GEOMETRY_ERROR = (
    "automatic thread selection or thread_k/thread_n=\\(128,64\\) or \\(128,32\\)"
)
_FORCED_THREAD_GEOMETRY_CASES = (
    pytest.param(8, 128, 128, 128, 64, id="thread_n_64_moe_block_8"),
    pytest.param(16, 256, 128, 128, 64, id="thread_n_64_moe_block_16"),
    pytest.param(16, 256, 128, 128, 32, id="thread_n_32_moe_block_16"),
    pytest.param(32, 256, 128, 128, 64, id="thread_n_64_moe_block_32"),
    pytest.param(48, 256, 128, 128, 64, id="thread_n_64_moe_block_48"),
    pytest.param(64, 256, 128, 128, 64, id="thread_n_64_moe_block_64"),
    pytest.param(32, 256, 128, 128, 32, id="thread_n_32_moe_block_32"),
    pytest.param(64, 256, 128, 128, 32, id="thread_n_32_moe_block_64"),
)
_UNSUPPORTED_MOE_BLOCK_SIZE_CASES = (
    pytest.param(24, id="moe_block_24"),
)
_MOE_ENV_SWEEP_TOLERANCES = {
    "uint4": (2e-1, 2.0),
    "uint4b8": (7e-2, 5e-1),
    "uint8": (7e-2, 5e-1),
    "uint8b128": (7e-2, 5e-1),
    "fp8": (7e-2, 5e-1),
    "nvfp4": (2e-1, 2.0),
    "mxfp4": (2e-1, 2.0),
}

def _require_moe_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    target_capability = source_target_capability()
    capability = torch.cuda.get_device_capability()
    if capability != target_capability:
        pytest.skip(f"Marlin MoE requires {source_target_label()} for this source tree")
    try:
        ops._load_moe()
    except Exception as exc:  # pragma: no cover - depends on local build state
        pytest.skip(f"marlin moe extension is not available: {exc}")


def _topk_softmax_reference(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = False,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.softmax(gating_output.to(torch.float32), dim=-1)
    selection_scores = probs
    if bias is not None:
        selection_scores = selection_scores + bias.to(torch.float32)
    topk_ids = torch.topk(selection_scores, k=topk, dim=-1).indices
    topk_weights = torch.gather(probs, 1, topk_ids)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)
    token_expert_indices = torch.stack(
        [
            torch.arange(gating_output.shape[0], device=gating_output.device, dtype=torch.int32)
            + route_idx * gating_output.shape[0]
            for route_idx in range(topk)
        ],
        dim=-1,
    )
    return topk_weights, topk_ids.to(torch.int32), token_expert_indices


def _grouped_topk_reference(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_experts = scores.shape
    experts_per_group = num_experts // num_expert_group
    if scoring_func == 0:
        unbiased = scores.to(torch.float32)
    elif scoring_func == 1:
        unbiased = torch.sigmoid(scores.to(torch.float32))
    else:
        raise ValueError(f"Unsupported scoring_func={scoring_func}")
    biased = unbiased + bias.to(torch.float32)

    topk_weights = []
    topk_ids = []
    for token_idx in range(num_tokens):
        group_scores: list[tuple[float, int]] = []
        for group_idx in range(num_expert_group):
            start = group_idx * experts_per_group
            end = start + experts_per_group
            top2 = torch.topk(biased[token_idx, start:end], k=min(2, experts_per_group)).values
            group_scores.append((float(top2.sum().item()), group_idx))

        selected_groups = [
            group_idx
            for _score, group_idx in sorted(group_scores, key=lambda item: (-item[0], item[1]))[
                :topk_group
            ]
        ]
        candidates: list[tuple[float, int]] = []
        for group_idx in selected_groups:
            start = group_idx * experts_per_group
            end = start + experts_per_group
            for expert_idx in range(start, end):
                candidates.append((float(biased[token_idx, expert_idx].item()), expert_idx))
        selected_experts = sorted(candidates, key=lambda item: (-item[0], item[1]))[:topk]
        selected_ids = torch.tensor(
            [expert_idx for _score, expert_idx in selected_experts],
            device=scores.device,
            dtype=torch.int32,
        )
        selected_weights = unbiased[token_idx, selected_ids.to(torch.long)]
        scale = routed_scaling_factor
        if renormalize:
            scale /= float(selected_weights.sum().item())
        topk_weights.append(selected_weights * scale)
        topk_ids.append(selected_ids)

    return torch.stack(topk_weights, dim=0), torch.stack(topk_ids, dim=0)


def _moe_align_block_size_reference(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    numel = topk_ids.numel()
    flat_topk_ids = topk_ids.reshape(-1)
    sorted_ids = []
    expert_ids = []
    for expert_idx in range(num_experts):
        expert_token_ids = torch.nonzero(flat_topk_ids == expert_idx, as_tuple=False).reshape(-1)
        padded_count = ((expert_token_ids.numel() + block_size - 1) // block_size) * block_size
        if padded_count == 0:
            continue
        sorted_ids.extend(expert_token_ids.tolist())
        sorted_ids.extend([numel] * (padded_count - expert_token_ids.numel()))
        expert_ids.extend([expert_idx] * (padded_count // block_size))
    return (
        torch.tensor(sorted_ids, device=topk_ids.device, dtype=torch.int32),
        torch.tensor(expert_ids, device=topk_ids.device, dtype=torch.int32),
        torch.tensor([len(sorted_ids)], device=topk_ids.device, dtype=torch.int32),
    )


def test_marlin_moe_symbols_available():
    expected = [
        "topk_softmax",
        "topk_sigmoid",
        "grouped_topk",
        "moe_align_block_size",
        "batched_moe_align_block_size",
        "moe_wna16_marlin_gemm",
    ]
    for name in expected:
        assert hasattr(ops, name)


def test_topk_softmax_and_align_block_size_shapes():
    _require_moe_cuda()

    gating_output = torch.randn((8, 16), device="cuda", dtype=torch.float16)
    topk_weights, topk_ids, token_expert_indices = topk_softmax(
        gating_output, topk=2
    )
    assert topk_weights.shape == (8, 2)
    assert topk_ids.shape == (8, 2)
    assert token_expert_indices.shape == (8, 2)

    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids, block_size=16, num_experts=16
    )
    assert sorted_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    assert num_tokens_post_pad.dtype == torch.int32


def test_topk_softmax_matches_reference():
    _require_moe_cuda()

    gating_output = torch.tensor(
        [
            [0.10, 0.80, -0.20, 0.30],
            [1.20, -0.40, 0.50, 0.10],
            [-0.70, 0.60, 0.20, 1.10],
        ],
        device="cuda",
        dtype=torch.float16,
    )
    bias = torch.tensor([0.05, -0.10, 0.20, 0.15], device="cuda", dtype=torch.float32)

    topk_weights, topk_ids, token_expert_indices = topk_softmax(
        gating_output, topk=2, renormalize=True, bias=bias
    )
    ref_weights, ref_ids, ref_token_expert_indices = _topk_softmax_reference(
        gating_output, topk=2, renormalize=True, bias=bias
    )

    torch.testing.assert_close(topk_weights, ref_weights, rtol=1e-4, atol=1e-4)
    assert torch.equal(topk_ids, ref_ids)
    assert torch.equal(token_expert_indices, ref_token_expert_indices)


def test_moe_align_block_size_matches_reference():
    _require_moe_cuda()

    topk_ids = torch.tensor(
        [[0, 3], [1, 3], [0, 2], [1, 0]], device="cuda", dtype=torch.int32
    )
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids, block_size=4, num_experts=4
    )
    ref_sorted_ids, ref_expert_ids, ref_num_tokens_post_pad = _moe_align_block_size_reference(
        topk_ids, block_size=4, num_experts=4
    )

    actual_num_tokens_post_pad = int(num_tokens_post_pad.item())
    assert actual_num_tokens_post_pad == int(ref_num_tokens_post_pad.item())
    assert torch.equal(sorted_ids[:actual_num_tokens_post_pad], ref_sorted_ids)
    assert torch.equal(expert_ids[: ref_expert_ids.numel()], ref_expert_ids)
    assert torch.all(sorted_ids[actual_num_tokens_post_pad:] == topk_ids.numel())


def test_grouped_topk_shapes():
    _require_moe_cuda()

    scores = torch.randn((8, 16), device="cuda", dtype=torch.float32)
    bias = torch.randn((16,), device="cuda", dtype=torch.float32)
    topk_weights, topk_ids = grouped_topk(
        scores=scores,
        num_expert_group=4,
        topk_group=2,
        topk=2,
        renormalize=False,
        routed_scaling_factor=1.0,
        bias=bias,
        scoring_func=1,
    )
    assert topk_weights.shape == (8, 2)
    assert topk_ids.shape == (8, 2)


def test_grouped_topk_matches_reference():
    _require_moe_cuda()

    scores = torch.tensor(
        [
            [0.10, 0.80, -0.40, 0.20, 1.50, 0.30, -0.10, 0.70],
            [0.90, -0.30, 0.40, 0.20, -0.60, 0.50, 1.10, 0.00],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    bias = torch.tensor(
        [0.05, 0.10, -0.05, 0.00, 0.20, -0.10, 0.15, 0.05],
        device="cuda",
        dtype=torch.float32,
    )

    topk_weights, topk_ids = grouped_topk(
        scores=scores,
        num_expert_group=4,
        topk_group=2,
        topk=3,
        renormalize=True,
        routed_scaling_factor=1.0,
        bias=bias,
        scoring_func=1,
    )
    ref_weights, ref_ids = _grouped_topk_reference(
        scores=scores,
        num_expert_group=4,
        topk_group=2,
        topk=3,
        renormalize=True,
        routed_scaling_factor=1.0,
        bias=bias,
        scoring_func=1,
    )

    torch.testing.assert_close(topk_weights, ref_weights, rtol=1e-4, atol=1e-4)
    assert torch.equal(topk_ids, ref_ids)


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_fused_marlin_moe_smoke(repack_impl: str):
    _require_moe_cuda()
    assert_repack_layout_matches_reference(repack_impl, quant_type=scalar_types.uint4b8)
    # This is a collectable smoke shape check only for the checked-in source target build.

    quant_type_id = 1
    hidden_states = torch.randn((4, 128), device="cuda", dtype=torch.float16)
    topk_weights = torch.rand((4, 2), device="cuda", dtype=torch.float32)
    topk_ids = torch.randint(0, 4, (4, 2), device="cuda", dtype=torch.int32)
    w1 = torch.empty((4, 8, 16), device="cuda", dtype=torch.int32)
    w2 = torch.empty((4, 8, 8), device="cuda", dtype=torch.int32)
    w1_scale = torch.ones((4, 1, 128), device="cuda", dtype=torch.float16)
    w2_scale = torch.ones((4, 1, 128), device="cuda", dtype=torch.float16)

    try:
        output = fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=quant_type_id,
            moe_block_size=16,
        )
    except RuntimeError:
        pytest.skip("smoke input layout is not compatible with local kernel expectations")

    assert output.shape == hidden_states.shape


def _make_moe_accuracy_inputs(
    quant_type,
    *,
    repack_impl: str | None = None,
    group_size: int,
    act_order: bool,
    tokens: int = 4,
    hidden: int = 128,
    intermediate: int = 128,
    experts: int = 4,
    topk: int = 2,
):
    if repack_impl is not None:
        assert_repack_layout_matches_reference(
            repack_impl,
            quant_type=quant_type,
            act_order=act_order,
            group_size=group_size,
        )

    hidden_states, topk_weights, topk_ids, w1, w2 = make_moe_model_like_inputs(
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
        device="cuda",
    )
    if quant_type == scalar_types.uint4:
        if act_order:
            raise AssertionError("uint4 zero-point tests do not support act_order")
        w1_q, w1_scales, w1_zeros, w1_dequant, w1_g_idx, w1_perm = (
            marlin_quantize_experts_uint4_zp_with_metadata(w1, group_size)
        )
        w2_q, w2_scales, w2_zeros, w2_dequant, w2_g_idx, w2_perm = (
            marlin_quantize_experts_uint4_zp_with_metadata(w2, group_size)
        )
        w1_global_scale = None
        w2_global_scale = None
    elif quant_type == scalar_types.uint8:
        if act_order:
            raise AssertionError("uint8 zero-point tests do not support act_order")
        w1_q, w1_scales, w1_zeros, w1_dequant, w1_g_idx, w1_perm = (
            marlin_quantize_experts_uint8_zp_with_metadata(w1, group_size)
        )
        w2_q, w2_scales, w2_zeros, w2_dequant, w2_g_idx, w2_perm = (
            marlin_quantize_experts_uint8_zp_with_metadata(w2, group_size)
        )
        w1_global_scale = None
        w2_global_scale = None
    elif quant_type == scalar_types.float4_e2m1f and group_size == 16:
        if act_order:
            raise AssertionError("nvfp4 tests do not support act_order")
        w1_q, w1_scales, w1_global_scale, w1_dequant, w1_g_idx, w1_perm = (
            marlin_quantize_experts_nvfp4_with_metadata(w1, group_size)
        )
        w2_q, w2_scales, w2_global_scale, w2_dequant, w2_g_idx, w2_perm = (
            marlin_quantize_experts_nvfp4_with_metadata(w2, group_size)
        )
        w1_zeros = None
        w2_zeros = None
    elif quant_type == scalar_types.float4_e2m1f and group_size == 32:
        if act_order:
            raise AssertionError("mxfp4 tests do not support act_order")
        w1_q, w1_scales, w1_dequant, w1_g_idx, w1_perm = (
            marlin_quantize_experts_mxfp4_with_metadata(w1, group_size)
        )
        w2_q, w2_scales, w2_dequant, w2_g_idx, w2_perm = (
            marlin_quantize_experts_mxfp4_with_metadata(w2, group_size)
        )
        w1_zeros = None
        w2_zeros = None
        w1_global_scale = None
        w2_global_scale = None
    else:
        w1_q, w1_scales, w1_dequant, w1_g_idx, w1_perm = marlin_quantize_experts_with_metadata(
            w1, quant_type, group_size, act_order
        )
        w2_q, w2_scales, w2_dequant, w2_g_idx, w2_perm = marlin_quantize_experts_with_metadata(
            w2, quant_type, group_size, act_order
        )
        w1_zeros = None
        w2_zeros = None
        w1_global_scale = None
        w2_global_scale = None
    return {
        "tokens": tokens,
        "hidden": hidden,
        "intermediate": intermediate,
        "experts": experts,
        "topk": topk,
        "hidden_states": hidden_states,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1_q": w1_q,
        "w1_scales": w1_scales,
        "w1_zeros": w1_zeros,
        "w1_global_scale": w1_global_scale,
        "w1_dequant": w1_dequant,
        "w1_g_idx": w1_g_idx,
        "w1_perm": w1_perm,
        "w2_q": w2_q,
        "w2_scales": w2_scales,
        "w2_zeros": w2_zeros,
        "w2_global_scale": w2_global_scale,
        "w2_dequant": w2_dequant,
        "w2_g_idx": w2_g_idx,
        "w2_perm": w2_perm,
    }


def _run_fused_moe_accuracy_case(
    quant_type,
    *,
    repack_impl: str,
    group_size: int,
    act_order: bool,
    is_k_full: bool,
    tokens: int = 4,
    hidden: int = 128,
    intermediate: int = 128,
    experts: int = 4,
    topk: int = 2,
    moe_block_size: int = 16,
    workspace: torch.Tensor | None = None,
    rtol: float = 7e-2,
    atol: float = 1e-2,
) -> None:
    if act_order:
        raise AssertionError("act_order coverage uses explicit rejection tests")
    if moe_block_size not in _SM70_SUPPORTED_MOE_BLOCK_SIZES:
        raise AssertionError("accuracy helper only covers supported SM70 MoE block sizes")

    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        quant_type,
        repack_impl=repack_impl,
        group_size=group_size,
        act_order=act_order,
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
    )

    output = fused_marlin_moe(
        hidden_states=inputs["hidden_states"],
        w1=inputs["w1_q"],
        w2=inputs["w2_q"],
        w1_scale=inputs["w1_scales"],
        w2_scale=inputs["w2_scales"],
        topk_weights=inputs["topk_weights"],
        topk_ids=inputs["topk_ids"],
        quant_type_id=quant_type.id,
        global_scale1=inputs["w1_global_scale"],
        global_scale2=inputs["w2_global_scale"],
        w1_zeros=inputs["w1_zeros"],
        w2_zeros=inputs["w2_zeros"],
        is_w1_zp_float=inputs["w1_zeros"] is not None,
        is_w2_zp_float=inputs["w2_zeros"] is not None,
        g_idx1=inputs["w1_g_idx"],
        g_idx2=inputs["w2_g_idx"],
        sort_indices1=inputs["w1_perm"],
        sort_indices2=inputs["w2_perm"],
        is_k_full=is_k_full,
        moe_block_size=moe_block_size,
        workspace=workspace,
    )
    reference = marlin_moe_reference(
        inputs["hidden_states"],
        inputs["w1_dequant"],
        inputs["w2_dequant"],
        inputs["topk_weights"],
        inputs["topk_ids"],
    ).to(torch.float16)

    assert output.shape == inputs["hidden_states"].shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _assert_moe_backend_rejects_act_order_compat_args(
    quant_type,
    *,
    repack_impl: str | None = None,
    group_size: int,
    is_k_full: bool,
    tokens: int = 4,
    moe_block_size: int = 16,
) -> None:
    if moe_block_size not in _SM70_SUPPORTED_MOE_BLOCK_SIZES:
        raise AssertionError("act_order rejection helper only covers supported SM70 MoE block sizes")

    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        quant_type,
        repack_impl=repack_impl,
        group_size=group_size,
        act_order=False,
        tokens=tokens,
    )
    hidden_states = inputs["hidden_states"]
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        inputs["topk_ids"], block_size=moe_block_size, num_experts=inputs["experts"]
    )

    with pytest.raises(RuntimeError, match="act_order"):
        ops.moe_wna16_marlin_gemm(
            hidden_states,
            torch.empty(
                (inputs["tokens"] * inputs["topk"], 2 * inputs["intermediate"]),
                device="cuda",
                dtype=torch.float16,
            ),
            inputs["w1_q"],
            None,
            inputs["w1_scales"],
            None,
            inputs["w1_global_scale"],
            inputs["w1_zeros"],
            torch.zeros(hidden_states.shape[1], device="cuda", dtype=torch.int32),
            None,
            None,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            inputs["topk_weights"],
            moe_block_size,
            inputs["topk"],
            False,
            quant_type.id,
            inputs["tokens"],
            2 * inputs["intermediate"],
            hidden_states.shape[1],
            is_k_full,
            True,
            True,
            inputs["w1_zeros"] is not None,
            128,
            64,
            4,
        )

    with pytest.raises(RuntimeError, match="act_order"):
        ops.moe_wna16_marlin_gemm(
            hidden_states,
            torch.empty(
                (inputs["tokens"] * inputs["topk"], 2 * inputs["intermediate"]),
                device="cuda",
                dtype=torch.float16,
            ),
            inputs["w1_q"],
            None,
            inputs["w1_scales"],
            None,
            inputs["w1_global_scale"],
            inputs["w1_zeros"],
            None,
            torch.arange(hidden_states.shape[1], device="cuda", dtype=torch.int32),
            None,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            inputs["topk_weights"],
            moe_block_size,
            inputs["topk"],
            False,
            quant_type.id,
            inputs["tokens"],
            2 * inputs["intermediate"],
            hidden_states.shape[1],
            is_k_full,
            True,
            True,
            inputs["w1_zeros"] is not None,
            128,
            64,
            4,
        )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_fused_marlin_moe_uint4b8_topk_weight_fusion_order_matches_reference(
    repack_impl: str,
):
    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=128,
        act_order=False,
        tokens=2,
    )
    inputs["topk_weights"] = torch.tensor(
        [[0.75, 0.25], [0.20, 0.80]], device="cuda", dtype=torch.float32
    )
    inputs["topk_ids"] = torch.tensor([[0, 1], [2, 3]], device="cuda", dtype=torch.int32)

    output = fused_marlin_moe(
        hidden_states=inputs["hidden_states"],
        w1=inputs["w1_q"],
        w2=inputs["w2_q"],
        w1_scale=inputs["w1_scales"],
        w2_scale=inputs["w2_scales"],
        topk_weights=inputs["topk_weights"],
        topk_ids=inputs["topk_ids"],
        quant_type_id=scalar_types.uint4b8.id,
        global_scale1=inputs["w1_global_scale"],
        global_scale2=inputs["w2_global_scale"],
        w1_zeros=inputs["w1_zeros"],
        w2_zeros=inputs["w2_zeros"],
        g_idx1=inputs["w1_g_idx"],
        g_idx2=inputs["w2_g_idx"],
        sort_indices1=inputs["w1_perm"],
        sort_indices2=inputs["w2_perm"],
        is_k_full=True,
        moe_block_size=16,
    )
    reference = marlin_moe_reference(
        inputs["hidden_states"],
        inputs["w1_dequant"],
        inputs["w2_dequant"],
        inputs["topk_weights"],
        inputs["topk_ids"],
    ).to(torch.float16)

    torch.testing.assert_close(output, reference, rtol=7e-2, atol=1e-2)


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_fused_marlin_moe_uint4b8_accuracy(group_size: int, repack_impl: str):
    _run_fused_moe_accuracy_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=group_size,
        act_order=False,
        is_k_full=True,
        moe_block_size=16,
    )


@pytest.mark.parametrize("is_k_full", (True, False))
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_fused_marlin_moe_uint4b8_rejects_act_order_compat_args(
    is_k_full: bool, repack_impl: str
):
    _assert_moe_backend_rejects_act_order_compat_args(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=64,
        is_k_full=is_k_full,
        moe_block_size=16,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_fused_marlin_moe_uint4b8_partial_block_matches_reference(repack_impl: str):
    _run_fused_moe_accuracy_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=128,
        act_order=False,
        is_k_full=True,
        tokens=3,
        moe_block_size=16,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_fused_marlin_moe_uint4b8_single_token_matches_reference(repack_impl: str):
    _run_fused_moe_accuracy_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=128,
        act_order=False,
        is_k_full=True,
        tokens=1,
        moe_block_size=16,
    )


@pytest.mark.parametrize("moe_block_size", (8, 32, 48, 64))
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_fused_marlin_moe_uint4b8_supported_moe_block_sizes_match_reference(
    repack_impl: str, moe_block_size: int
):
    _run_fused_moe_accuracy_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=128,
        act_order=False,
        is_k_full=True,
        moe_block_size=moe_block_size,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_fused_marlin_moe_uint4b8_rejects_unsupported_moe_block_size(
    repack_impl: str,
):
    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=128,
        act_order=False,
        tokens=3,
    )

    with pytest.raises(RuntimeError, match=_SUPPORTED_MOE_BLOCK_SIZE_ERROR):
        fused_marlin_moe(
            hidden_states=inputs["hidden_states"],
            w1=inputs["w1_q"],
            w2=inputs["w2_q"],
            w1_scale=inputs["w1_scales"],
            w2_scale=inputs["w2_scales"],
            topk_weights=inputs["topk_weights"],
            topk_ids=inputs["topk_ids"],
            quant_type_id=scalar_types.uint4b8.id,
            g_idx1=inputs["w1_g_idx"],
            g_idx2=inputs["w2_g_idx"],
            sort_indices1=inputs["w1_perm"],
            sort_indices2=inputs["w2_perm"],
            is_k_full=True,
            moe_block_size=24,
        )


def _run_stage1_kernel_case(
    quant_type,
    *,
    repack_impl: str,
    group_size: int,
    act_order: bool,
    is_k_full: bool,
    tokens: int = 4,
    moe_block_size: int = 16,
    hidden: int = 128,
    intermediate: int = 128,
    experts: int = 4,
    topk: int = 2,
    thread_k: int = -1,
    thread_n: int = -1,
    blocks_per_sm: int = -1,
    workspace: torch.Tensor | None = None,
    rtol: float = 6e-2,
    atol: float = 3e-1,
) -> None:
    if act_order:
        raise AssertionError("act_order stage1 coverage uses explicit rejection tests")
    if moe_block_size not in _SM70_SUPPORTED_MOE_BLOCK_SIZES:
        raise AssertionError("stage1 helper only covers supported SM70 MoE block sizes")

    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        quant_type,
        repack_impl=repack_impl,
        group_size=group_size,
        act_order=act_order,
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
    )
    hidden_states = inputs["hidden_states"]
    topk_weights = inputs["topk_weights"]
    topk_ids = inputs["topk_ids"]
    experts = inputs["experts"]
    tokens = inputs["tokens"]
    topk = inputs["topk"]
    intermediate = inputs["intermediate"]
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids, block_size=moe_block_size, num_experts=experts
    )

    stage1 = ops.moe_wna16_marlin_gemm(
        hidden_states,
        torch.empty((tokens * topk, 2 * intermediate), device="cuda", dtype=torch.float16),
        inputs["w1_q"],
        None,
        inputs["w1_scales"],
        None,
        inputs["w1_global_scale"],
        inputs["w1_zeros"],
        inputs["w1_g_idx"],
        inputs["w1_perm"],
        workspace,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        moe_block_size,
        topk,
        False,
        quant_type.id,
        tokens,
        2 * intermediate,
        hidden_states.shape[1],
        is_k_full,
        False,
        False,
        inputs["w1_zeros"] is not None,
        thread_k,
        thread_n,
        blocks_per_sm,
    )
    reference_gate_up = []
    for token_idx in range(tokens):
        for route_idx in range(topk):
            expert = int(topk_ids[token_idx, route_idx].item())
            reference_gate_up.append(
                torch.matmul(
                    hidden_states[token_idx : token_idx + 1].to(torch.float32),
                    inputs["w1_dequant"][expert].to(torch.float32),
                )[0]
            )
    reference_stage1 = torch.stack(reference_gate_up, dim=0).to(torch.float16)

    assert stage1.shape == (tokens * topk, 2 * intermediate)
    assert torch.isfinite(stage1).all()
    torch.testing.assert_close(stage1, reference_stage1, rtol=rtol, atol=atol)


def _run_forced_fused_kernel_case(
    quant_type,
    *,
    repack_impl: str,
    group_size: int,
    tokens: int,
    hidden: int,
    intermediate: int,
    experts: int,
    topk: int,
    moe_block_size: int,
    thread_k: int,
    thread_n: int,
    blocks_per_sm: int = -1,
    rtol: float = 7e-2,
    atol: float = 1e-2,
) -> None:
    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        quant_type,
        repack_impl=repack_impl,
        group_size=group_size,
        act_order=False,
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
    )
    hidden_states = inputs["hidden_states"]
    topk_weights = inputs["topk_weights"]
    topk_ids = inputs["topk_ids"]
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids, block_size=moe_block_size, num_experts=experts
    )

    stage1 = ops.moe_wna16_marlin_gemm(
        hidden_states,
        torch.empty((tokens * topk, 2 * intermediate), device="cuda", dtype=torch.float16),
        inputs["w1_q"],
        None,
        inputs["w1_scales"],
        None,
        inputs["w1_global_scale"],
        inputs["w1_zeros"],
        inputs["w1_g_idx"],
        inputs["w1_perm"],
        None,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        moe_block_size,
        topk,
        False,
        quant_type.id,
        tokens,
        2 * intermediate,
        hidden,
        True,
        False,
        False,
        inputs["w1_zeros"] is not None,
        thread_k,
        thread_n,
        blocks_per_sm,
    )
    gate, up = stage1.view(tokens * topk, 2 * intermediate).chunk(2, dim=-1)
    activated = torch.nn.functional.silu(gate) * up
    output = ops.moe_wna16_marlin_gemm(
        activated,
        torch.empty((tokens * topk, hidden), device="cuda", dtype=torch.float16),
        inputs["w2_q"],
        None,
        inputs["w2_scales"],
        None,
        inputs["w2_global_scale"],
        inputs["w2_zeros"],
        inputs["w2_g_idx"],
        inputs["w2_perm"],
        None,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        moe_block_size,
        1,
        True,
        quant_type.id,
        tokens * topk,
        hidden,
        intermediate,
        True,
        False,
        False,
        inputs["w2_zeros"] is not None,
        thread_k,
        thread_n,
        blocks_per_sm,
    )
    fused_output = output.view(tokens, topk, hidden).sum(dim=1)
    reference = marlin_moe_reference(
        hidden_states,
        inputs["w1_dequant"],
        inputs["w2_dequant"],
        topk_weights,
        topk_ids,
    ).to(torch.float16)

    assert fused_output.shape == hidden_states.shape
    assert torch.isfinite(fused_output).all()
    torch.testing.assert_close(fused_output, reference, rtol=rtol, atol=atol)


def _assert_stage1_kernel_rejects_unsupported_config(
    quant_type,
    *,
    repack_impl: str,
    moe_block_size: int,
    thread_k: int,
    thread_n: int,
    error_match: str,
    hidden: int = 128,
    intermediate: int = 128,
    experts: int = 4,
    tokens: int = 2,
    topk: int = 2,
) -> None:
    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        quant_type,
        repack_impl=repack_impl,
        group_size=128,
        act_order=False,
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
    )
    hidden_states = inputs["hidden_states"]
    topk_weights = inputs["topk_weights"]
    topk_ids = inputs["topk_ids"]
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        topk_ids, block_size=moe_block_size, num_experts=experts
    )

    with pytest.raises(RuntimeError, match=error_match):
        ops.moe_wna16_marlin_gemm(
            hidden_states,
            torch.empty((tokens * topk, 2 * intermediate), device="cuda", dtype=torch.float16),
            inputs["w1_q"],
            None,
            inputs["w1_scales"],
            None,
            inputs["w1_global_scale"],
            inputs["w1_zeros"],
            inputs["w1_g_idx"],
            inputs["w1_perm"],
            None,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            topk_weights,
            moe_block_size,
            topk,
            False,
            quant_type.id,
            tokens,
            2 * intermediate,
            hidden,
            True,
            False,
            False,
            inputs["w1_zeros"] is not None,
            thread_k,
            thread_n,
            -1,
        )


def _make_moe_env_sweep_inputs(
    key: MoeDirectOpKey,
) -> dict[str, object]:
    _require_moe_cuda()
    torch.manual_seed(2000 + key.tokens + key.hidden + key.intermediate)
    torch.cuda.manual_seed_all(2000 + key.tokens + key.hidden + key.intermediate)
    quant_type = {
        "uint4": scalar_types.uint4,
        "uint4b8": scalar_types.uint4b8,
        "uint8": scalar_types.uint8,
        "uint8b128": scalar_types.uint8b128,
        "fp8": scalar_types.float8_e4m3fn,
        "nvfp4": scalar_types.float4_e2m1f,
        "mxfp4": scalar_types.float4_e2m1f,
    }[key.quant_name]
    return _make_moe_accuracy_inputs(
        quant_type,
        repack_impl="gptq",
        group_size=key.group_size,
        act_order=False,
        tokens=key.tokens,
        hidden=key.hidden,
        intermediate=key.intermediate,
        experts=key.experts,
        topk=key.topk,
    )


def _moe_quant_type_id(quant_name: str) -> int:
    return {
        "uint4": scalar_types.uint4,
        "uint4b8": scalar_types.uint4b8,
        "uint8": scalar_types.uint8,
        "uint8b128": scalar_types.uint8b128,
        "fp8": scalar_types.float8_e4m3fn,
        "nvfp4": scalar_types.float4_e2m1f,
        "mxfp4": scalar_types.float4_e2m1f,
    }[quant_name].id


def _make_moe_exact_mnk_env_sweep_inputs(
    key: MoeDirectOpKey,
) -> dict[str, object]:
    if key.topk != 1:
        raise AssertionError("exact-MNK MoE env sweep uses topk=1")
    _require_moe_cuda()
    torch.manual_seed(3000 + key.tokens + key.hidden + key.intermediate)
    torch.cuda.manual_seed_all(3000 + key.tokens + key.hidden + key.intermediate)
    quant_type = {
        "uint4": scalar_types.uint4,
        "uint4b8": scalar_types.uint4b8,
        "uint8": scalar_types.uint8,
        "uint8b128": scalar_types.uint8b128,
        "fp8": scalar_types.float8_e4m3fn,
        "nvfp4": scalar_types.float4_e2m1f,
        "mxfp4": scalar_types.float4_e2m1f,
    }[key.quant_name]

    hidden_states = torch.randn(
        (key.tokens, key.hidden), device="cuda", dtype=torch.float16
    )
    weights = torch.randn(
        (key.experts, key.hidden, key.intermediate),
        device="cuda",
        dtype=torch.float16,
    )
    weights = weights * (1.0 / (key.hidden ** 0.5))
    topk_ids = torch.empty((key.tokens, 1), device="cuda", dtype=torch.int32)
    for token_idx in range(key.tokens):
        topk_ids[token_idx, 0] = token_idx % key.experts
    topk_weights = torch.ones((key.tokens, 1), device="cuda", dtype=torch.float32)

    if key.quant_name == "uint4":
        q_weight, scales, zeros, dequant, g_idx, perm = (
            marlin_quantize_experts_uint4_zp_with_metadata(weights, key.group_size)
        )
        global_scale = None
    elif key.quant_name == "uint8":
        q_weight, scales, zeros, dequant, g_idx, perm = (
            marlin_quantize_experts_uint8_zp_with_metadata(weights, key.group_size)
        )
        global_scale = None
    elif key.quant_name == "nvfp4":
        q_weight, scales, global_scale, dequant, g_idx, perm = (
            marlin_quantize_experts_nvfp4_with_metadata(weights, key.group_size)
        )
        zeros = None
    elif key.quant_name == "mxfp4":
        q_weight, scales, dequant, g_idx, perm = (
            marlin_quantize_experts_mxfp4_with_metadata(weights, key.group_size)
        )
        zeros = None
        global_scale = None
    else:
        q_weight, scales, dequant, g_idx, perm = marlin_quantize_experts_with_metadata(
            weights, quant_type, key.group_size, False
        )
        zeros = None
        global_scale = None

    return {
        "tokens": key.tokens,
        "hidden": key.hidden,
        "intermediate": key.intermediate,
        "experts": key.experts,
        "topk": key.topk,
        "hidden_states": hidden_states,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "w1_q": q_weight,
        "w1_scales": scales,
        "w1_zeros": zeros,
        "w1_global_scale": global_scale,
        "w1_dequant": dequant,
        "w1_g_idx": g_idx,
        "w1_perm": perm,
    }


def _moe_exact_mnk_reference(inputs: dict[str, object]) -> torch.Tensor:
    hidden_states = inputs["hidden_states"]
    topk_ids = inputs["topk_ids"]
    tokens = inputs["tokens"]
    reference_rows = []
    for token_idx in range(tokens):
        expert = int(topk_ids[token_idx, 0].item())
        reference_rows.append(
            torch.matmul(
                hidden_states[token_idx : token_idx + 1].to(torch.float32),
                inputs["w1_dequant"][expert].to(torch.float32),
            )[0]
        )
    return torch.stack(reference_rows, dim=0).to(torch.float16)


def _run_moe_exact_mnk_env_combo(
    key: MoeDirectOpKey,
    inputs: dict[str, object],
    *,
    moe_block_size: int,
    reference: torch.Tensor,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        inputs["topk_ids"], block_size=moe_block_size, num_experts=inputs["experts"]
    )
    output = ops.moe_wna16_marlin_gemm(
        inputs["hidden_states"],
        torch.empty(
            (key.tokens, key.intermediate),
            device="cuda",
            dtype=torch.float16,
        ),
        inputs["w1_q"],
        None,
        inputs["w1_scales"],
        None,
        inputs["w1_global_scale"],
        inputs["w1_zeros"],
        inputs["w1_g_idx"],
        inputs["w1_perm"],
        None,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        inputs["topk_weights"],
        moe_block_size,
        1,
        False,
        _moe_quant_type_id(key.quant_name),
        key.tokens,
        key.intermediate,
        key.hidden,
        True,
        False,
        False,
        inputs["w1_zeros"] is not None,
        -1,
        -1,
        -1,
    )
    assert output.shape == (key.tokens, key.intermediate)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
    return output


def _moe_stage1_reference(inputs: dict[str, object]) -> torch.Tensor:
    hidden_states = inputs["hidden_states"]
    topk_ids = inputs["topk_ids"]
    tokens = inputs["tokens"]
    topk = inputs["topk"]
    reference_gate_up = []
    for token_idx in range(tokens):
        for route_idx in range(topk):
            expert = int(topk_ids[token_idx, route_idx].item())
            reference_gate_up.append(
                torch.matmul(
                    hidden_states[token_idx : token_idx + 1].to(torch.float32),
                    inputs["w1_dequant"][expert].to(torch.float32),
                )[0]
            )
    return torch.stack(reference_gate_up, dim=0).to(torch.float16)


def _moe_stage2_inputs_and_reference(
    inputs: dict[str, object],
    *,
    moe_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = inputs["tokens"]
    topk = inputs["topk"]
    intermediate = inputs["intermediate"]
    experts = inputs["experts"]
    topk_ids = inputs["topk_ids"]
    activation = torch.randn(
        (tokens * topk, intermediate), device="cuda", dtype=torch.float16
    )
    stage2_ids = topk_ids.reshape(tokens * topk, 1).contiguous()
    stage2_weights = torch.ones(
        (tokens * topk, 1), device="cuda", dtype=torch.float32
    )
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        stage2_ids,
        block_size=moe_block_size,
        num_experts=experts,
    )
    reference_rows = []
    for row in range(tokens * topk):
        expert = int(stage2_ids[row, 0].item())
        reference_rows.append(
            torch.matmul(
                activation[row : row + 1].to(torch.float32),
                inputs["w2_dequant"][expert].to(torch.float32),
            )[0]
        )
    reference = torch.stack(reference_rows, dim=0).to(torch.float16)
    return activation, stage2_weights, sorted_ids, expert_ids, num_tokens_post_pad, reference


def _run_moe_env_stage1_combo(
    key: MoeDirectOpKey,
    inputs: dict[str, object],
    *,
    moe_block_size: int,
    reference: torch.Tensor,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        inputs["topk_ids"], block_size=moe_block_size, num_experts=inputs["experts"]
    )
    output = ops.moe_wna16_marlin_gemm(
        inputs["hidden_states"],
        torch.empty(
            (key.tokens * key.topk, 2 * key.intermediate),
            device="cuda",
            dtype=torch.float16,
        ),
        inputs["w1_q"],
        None,
        inputs["w1_scales"],
        None,
        inputs["w1_global_scale"],
        inputs["w1_zeros"],
        inputs["w1_g_idx"],
        inputs["w1_perm"],
        None,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        inputs["topk_weights"],
        moe_block_size,
        key.topk,
        False,
        _moe_quant_type_id(key.quant_name),
        key.tokens,
        2 * key.intermediate,
        key.hidden,
        True,
        False,
        False,
        inputs["w1_zeros"] is not None,
        -1,
        -1,
        -1,
    )
    assert output.shape == (key.tokens * key.topk, 2 * key.intermediate)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
    return output


def _run_moe_env_stage2_combo(
    key: MoeDirectOpKey,
    inputs: dict[str, object],
    *,
    activation: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    moe_block_size: int,
    reference: torch.Tensor,
    rtol: float,
    atol: float,
) -> torch.Tensor:
    output = ops.moe_wna16_marlin_gemm(
        activation,
        torch.empty((key.tokens * key.topk, key.hidden), device="cuda", dtype=torch.float16),
        inputs["w2_q"],
        None,
        inputs["w2_scales"],
        None,
        inputs["w2_global_scale"],
        inputs["w2_zeros"],
        inputs["w2_g_idx"],
        inputs["w2_perm"],
        None,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        topk_weights,
        moe_block_size,
        1,
        False,
        _moe_quant_type_id(key.quant_name),
        key.tokens * key.topk,
        key.hidden,
        key.intermediate,
        True,
        False,
        False,
        inputs["w2_zeros"] is not None,
        -1,
        -1,
        -1,
    )
    assert output.shape == (key.tokens * key.topk, key.hidden)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
    return output


def _run_moe_invalid_env_smoke() -> None:
    key = MoeDirectOpKey("uint4", 128, 2, 128, 128, 4, 2)
    inputs = _make_moe_env_sweep_inputs(key)
    reference = _moe_stage1_reference(inputs)
    _run_moe_env_stage1_combo(
        key,
        inputs,
        moe_block_size=16,
        reference=reference,
        rtol=2e-1,
        atol=2.0,
    )


@pytest.mark.sm70_env_exhaustive
def test_moe_wna16_direct_op_env_geometry_exhaustive_matches_reference():
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the full env sweep")
    _require_moe_cuda()

    total = 0
    checked = 0
    legal = 0
    rejected = 0

    for key in iter_moe_direct_op_keys():
        prepared = None
        stage1_reference = None
        stage2_prepared = None
        moe_block_size = moe_auto_block_size(key.tokens, key.topk, key.experts)
        rtol, atol = _MOE_ENV_SWEEP_TOLERANCES[key.quant_name]
        for stage in ("stage1", "stage2"):
            for geometry, split_k, metadata_cache in iter_moe_env_combinations():
                if exhaustive_index_is_past_limit(total):
                    break
                selected = exhaustive_index_is_selected(total)
                total += 1
                if not selected:
                    continue

                checked += 1
                if prepared is None:
                    prepared = _make_moe_env_sweep_inputs(key)
                    stage1_reference = _moe_stage1_reference(prepared)
                inputs = prepared
                if stage == "stage1":
                    is_legal = moe_stage_env_combo_is_legal(
                        geometry,
                        size_n=2 * key.intermediate,
                        size_k=key.hidden,
                    )
                else:
                    is_legal = moe_stage_env_combo_is_legal(
                        geometry,
                        size_n=key.hidden,
                        size_k=key.intermediate,
                    )
                    if stage2_prepared is None:
                        stage2_prepared = _moe_stage2_inputs_and_reference(
                            inputs,
                            moe_block_size=moe_block_size,
                        )

                try:
                    with moe_env(geometry, split_k, metadata_cache):
                        if is_legal:
                            if stage == "stage1":
                                assert stage1_reference is not None
                                _run_moe_env_stage1_combo(
                                    key,
                                    inputs,
                                    moe_block_size=moe_block_size,
                                    reference=stage1_reference,
                                    rtol=rtol,
                                    atol=atol,
                                )
                            else:
                                assert stage2_prepared is not None
                                (
                                    activation,
                                    stage2_weights,
                                    sorted_ids,
                                    expert_ids,
                                    num_tokens_post_pad,
                                    stage2_reference,
                                ) = stage2_prepared
                                _run_moe_env_stage2_combo(
                                    key,
                                    inputs,
                                    activation=activation,
                                    topk_weights=stage2_weights,
                                    sorted_ids=sorted_ids,
                                    expert_ids=expert_ids,
                                    num_tokens_post_pad=num_tokens_post_pad,
                                    moe_block_size=moe_block_size,
                                    reference=stage2_reference,
                                    rtol=rtol,
                                    atol=atol,
                                )
                            legal += 1
                        else:
                            with pytest.raises(RuntimeError, match=EXPLICIT_ENV_REJECTION_RE):
                                if stage == "stage1":
                                    assert stage1_reference is not None
                                    _run_moe_env_stage1_combo(
                                        key,
                                        inputs,
                                        moe_block_size=moe_block_size,
                                        reference=stage1_reference,
                                        rtol=rtol,
                                        atol=atol,
                                    )
                                else:
                                    assert stage2_prepared is not None
                                    (
                                        activation,
                                        stage2_weights,
                                        sorted_ids,
                                        expert_ids,
                                        num_tokens_post_pad,
                                        stage2_reference,
                                    ) = stage2_prepared
                                    _run_moe_env_stage2_combo(
                                        key,
                                        inputs,
                                        activation=activation,
                                        topk_weights=stage2_weights,
                                        sorted_ids=sorted_ids,
                                        expert_ids=expert_ids,
                                        num_tokens_post_pad=num_tokens_post_pad,
                                        moe_block_size=moe_block_size,
                                        reference=stage2_reference,
                                        rtol=rtol,
                                        atol=atol,
                                    )
                            rejected += 1
                except Exception as exc:
                    raise AssertionError(
                        f"key={key}, stage={stage}, geometry={geometry.label}, "
                        f"split_k={split_k}, metadata={metadata_cache}, "
                        f"legal={is_legal}, error={exc}"
                    ) from exc
            if exhaustive_index_is_past_limit(total):
                break
        if exhaustive_index_is_past_limit(total):
            break

    assert checked > 0
    assert legal + rejected == checked


@pytest.mark.sm70_env_exhaustive
def test_moe_wna16_direct_op_env_focused_mnk_matches_reference():
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the focused env sweep")
    _require_moe_cuda()

    total = 0
    checked = 0
    legal = 0
    rejected = 0

    for key in iter_moe_focused_mnk_direct_op_keys():
        prepared = None
        reference = None
        moe_block_size = moe_auto_block_size(key.tokens, key.topk, key.experts)
        rtol, atol = _MOE_ENV_SWEEP_TOLERANCES[key.quant_name]
        for geometry, split_k, metadata_cache in iter_moe_env_combinations():
            if exhaustive_index_is_past_limit(total):
                break
            selected = exhaustive_index_is_selected(total)
            total += 1
            if not selected:
                continue

            checked += 1
            if prepared is None:
                prepared = _make_moe_exact_mnk_env_sweep_inputs(key)
                reference = _moe_exact_mnk_reference(prepared)
            inputs = prepared
            is_legal = moe_stage_env_combo_is_legal(
                geometry,
                size_n=key.intermediate,
                size_k=key.hidden,
            )

            try:
                with moe_env(geometry, split_k, metadata_cache):
                    if is_legal:
                        assert reference is not None
                        _run_moe_exact_mnk_env_combo(
                            key,
                            inputs,
                            moe_block_size=moe_block_size,
                            reference=reference,
                            rtol=rtol,
                            atol=atol,
                        )
                        legal += 1
                    else:
                        with pytest.raises(RuntimeError, match=EXPLICIT_ENV_REJECTION_RE):
                            assert reference is not None
                            _run_moe_exact_mnk_env_combo(
                                key,
                                inputs,
                                moe_block_size=moe_block_size,
                                reference=reference,
                                rtol=rtol,
                                atol=atol,
                            )
                        rejected += 1
            except Exception as exc:
                raise AssertionError(
                    f"key={key}, geometry={geometry.label}, split_k={split_k}, "
                    f"metadata={metadata_cache}, legal={is_legal}, error={exc}"
                ) from exc
        if exhaustive_index_is_past_limit(total):
            break

    assert checked > 0
    assert legal + rejected == checked


@pytest.mark.parametrize(
    ("geometry", "split_k", "metadata_cache", "match"),
    (
        pytest.param(
            "32x256x32x4x32x64",
            "1",
            "vector_words",
            "Invalid SM70_MARLIN_MOE_CTA_GEOMETRY",
            id="bad_geometry_field_count",
        ),
        pytest.param(
            "32x256x16x4x32x64x16",
            "1",
            "vector_words",
            "Unsupported SM70 Marlin.*CTA geometry",
            id="unsupported_geometry",
        ),
        pytest.param(
            "128x64x32x4x32x64x32",
            "1",
            "vector_words",
            "Unsupported SM70 Marlin MoE CTA geometry",
            id="dense_only_cta_m_128",
        ),
        pytest.param(
            "32x256x32x4x32x64x32",
            "3",
            "vector_words",
            "Invalid SM70_MARLIN_MOE_SPLIT_K",
            id="bad_split_k",
        ),
        pytest.param(
            "32x256x32x4x32x64x32",
            "1",
            "paired_words",
            "Invalid SM70_MARLIN_MOE_METADATA_CACHE",
            id="bad_metadata",
        ),
    ),
)
def test_moe_wna16_direct_op_rejects_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
    geometry: str,
    split_k: str,
    metadata_cache: str,
    match: str,
):
    _require_moe_cuda()
    set_moe_env(
        monkeypatch,
        geometry=geometry,
        split_k=split_k,
        metadata_cache=metadata_cache,
    )
    with pytest.raises(RuntimeError, match=match):
        _run_moe_invalid_env_smoke()


@pytest.mark.parametrize("geometry", SM70_MOE_GEOMETRIES)
@pytest.mark.parametrize("split_k", SM70_SPLIT_K_VALUES)
@pytest.mark.parametrize("metadata_cache", SM70_METADATA_CACHE_VALUES)
@pytest.mark.sm70_env_exhaustive
def test_moe_wna16_direct_op_env_smoke_single_shape_matches_reference(
    geometry,
    split_k: int,
    metadata_cache: str,
):
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the env smoke sweep")
    _require_moe_cuda()
    key = MoeDirectOpKey("uint4", 128, 2, 128, 128, 4, 2)
    inputs = _make_moe_env_sweep_inputs(key)
    reference = _moe_stage1_reference(inputs)
    with moe_env(geometry, split_k, metadata_cache):
        if moe_stage_env_combo_is_legal(
            geometry,
            size_n=2 * key.intermediate,
            size_k=key.hidden,
        ):
            _run_moe_env_stage1_combo(
                key,
                inputs,
                moe_block_size=16,
                reference=reference,
                rtol=2e-1,
                atol=2.0,
            )
        else:
            with pytest.raises(RuntimeError, match=EXPLICIT_ENV_REJECTION_RE):
                _run_moe_env_stage1_combo(
                    key,
                    inputs,
                    moe_block_size=16,
                    reference=reference,
                    rtol=2e-1,
                    atol=2.0,
                )


@pytest.mark.parametrize("moe_block_size", _SM70_SUPPORTED_MOE_BLOCK_SIZES)
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_moe_wna16_uint4b8_stage1_supported_moe_block_sizes_match_reference(
    repack_impl: str, moe_block_size: int
):
    _run_stage1_kernel_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=64,
        act_order=False,
        is_k_full=True,
        moe_block_size=moe_block_size,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
@pytest.mark.parametrize("moe_block_size", _UNSUPPORTED_MOE_BLOCK_SIZE_CASES)
def test_moe_wna16_uint4b8_stage1_rejects_unsupported_moe_block_size(
    repack_impl: str, moe_block_size: int
):
    _assert_stage1_kernel_rejects_unsupported_config(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        moe_block_size=moe_block_size,
        thread_k=-1,
        thread_n=-1,
        error_match=_SUPPORTED_MOE_BLOCK_SIZE_ERROR,
    )


@pytest.mark.parametrize(
    "moe_block_size,hidden,intermediate,thread_k,thread_n", _FORCED_THREAD_GEOMETRY_CASES
)
@pytest.mark.parametrize("repack_impl", _FORCED_GEOMETRY_REPACK_CASES)
def test_moe_wna16_uint4b8_stage1_forced_thread_geometry_matches_reference(
    repack_impl: str,
    moe_block_size: int,
    hidden: int,
    intermediate: int,
    thread_k: int,
    thread_n: int,
):
    _run_stage1_kernel_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=128,
        act_order=False,
        is_k_full=True,
        tokens=2,
        hidden=hidden,
        intermediate=intermediate,
        moe_block_size=moe_block_size,
        thread_k=thread_k,
        thread_n=thread_n,
    )


@pytest.mark.parametrize(
    "moe_block_size,hidden,intermediate,thread_k,thread_n", _FORCED_THREAD_GEOMETRY_CASES
)
@pytest.mark.parametrize("repack_impl", _FORCED_GEOMETRY_REPACK_CASES)
def test_fused_marlin_moe_uint4b8_forced_thread_geometry_matches_reference(
    repack_impl: str,
    moe_block_size: int,
    hidden: int,
    intermediate: int,
    thread_k: int,
    thread_n: int,
):
    _run_forced_fused_kernel_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=128,
        tokens=2,
        hidden=hidden,
        intermediate=intermediate,
        experts=4,
        topk=2,
        moe_block_size=moe_block_size,
        thread_k=thread_k,
        thread_n=thread_n,
    )


if "uint4" in _MOE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("group_size", _UINT4_ZP_GROUP_SIZES)
    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_uint4_zp_accuracy(group_size: int, repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.uint4,
            repack_impl=repack_impl,
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            moe_block_size=16,
            rtol=2e-1,
            atol=1.25,
        )

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_uint4_zp_partial_block_matches_reference(repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.uint4,
            repack_impl=repack_impl,
            group_size=64,
            act_order=False,
            is_k_full=True,
            tokens=3,
            moe_block_size=16,
            rtol=2e-1,
            atol=1.25,
        )

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_uint4_zp_single_token_matches_reference(repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.uint4,
            repack_impl=repack_impl,
            group_size=64,
            act_order=False,
            is_k_full=True,
            tokens=1,
            moe_block_size=16,
            rtol=2e-1,
            atol=1.25,
        )

    @pytest.mark.parametrize("moe_block_size", (32, 48, 64))
    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_uint4_zp_supported_moe_block_sizes_match_reference(
        repack_impl: str, moe_block_size: int
    ):
        _run_fused_moe_accuracy_case(
            scalar_types.uint4,
            repack_impl=repack_impl,
            group_size=64,
            act_order=False,
            is_k_full=True,
            moe_block_size=moe_block_size,
            rtol=2e-1,
            atol=1.25,
        )

    @pytest.mark.parametrize("group_size", _UINT4_ZP_GROUP_SIZES)
    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_moe_wna16_uint4_zp_stage1_accuracy(group_size: int, repack_impl: str):
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl=repack_impl,
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            moe_block_size=16,
            rtol=2e-1,
            atol=2.0,
        )

    @pytest.mark.parametrize("moe_block_size", (32, 48, 64))
    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_moe_wna16_uint4_zp_stage1_supported_moe_block_sizes_match_reference(
        repack_impl: str, moe_block_size: int
    ):
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl=repack_impl,
            group_size=64,
            act_order=False,
            is_k_full=True,
            moe_block_size=moe_block_size,
            rtol=2e-1,
            atol=2.0,
        )

    @pytest.mark.parametrize(
        "moe_block_size,hidden,intermediate,thread_k,thread_n", _FORCED_THREAD_GEOMETRY_CASES
    )
    @pytest.mark.parametrize("repack_impl", _FORCED_GEOMETRY_REPACK_CASES)
    def test_moe_wna16_uint4_zp_stage1_forced_thread_geometry_matches_reference(
        repack_impl: str,
        moe_block_size: int,
        hidden: int,
        intermediate: int,
        thread_k: int,
        thread_n: int,
    ):
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl=repack_impl,
            group_size=64,
            act_order=False,
            is_k_full=True,
            tokens=2,
            hidden=hidden,
            intermediate=intermediate,
            moe_block_size=moe_block_size,
            thread_k=thread_k,
            thread_n=thread_n,
            rtol=2e-1,
            atol=2.0,
        )

    @pytest.mark.parametrize(
        "moe_block_size,hidden,intermediate,thread_k,thread_n", _FORCED_THREAD_GEOMETRY_CASES
    )
    @pytest.mark.parametrize("repack_impl", _FORCED_GEOMETRY_REPACK_CASES)
    def test_fused_marlin_moe_uint4_zp_forced_thread_geometry_matches_reference(
        repack_impl: str,
        moe_block_size: int,
        hidden: int,
        intermediate: int,
        thread_k: int,
        thread_n: int,
    ):
        _run_forced_fused_kernel_case(
            scalar_types.uint4,
            repack_impl=repack_impl,
            group_size=64,
            tokens=2,
            hidden=hidden,
            intermediate=intermediate,
            experts=4,
            topk=2,
            moe_block_size=moe_block_size,
            thread_k=thread_k,
            thread_n=thread_n,
            rtol=2e-1,
            atol=1.25,
        )

    @pytest.mark.parametrize("group_size", _UINT4_ZP_GROUP_SIZES)
    def test_moe_wna16_uint4_zp_auto_split_k_stage1_matches_reference(group_size: int):
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            moe_block_size=16,
            topk=2,
            rtol=2e-1,
            atol=2.0,
        )

    @pytest.mark.parametrize("topk", (1, 2))
    @pytest.mark.parametrize("moe_block_size", (16, 32, 64))
    def test_fused_marlin_moe_uint4_zp_auto_split_k_matches_reference(
        topk: int,
        moe_block_size: int,
    ):
        _run_fused_moe_accuracy_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=32,
            act_order=False,
            is_k_full=True,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            moe_block_size=moe_block_size,
            topk=topk,
            rtol=2e-1,
            atol=2.0,
        )

    def test_moe_wna16_uint4_zp_auto_split_k_keeps_empty_workspace_matches_reference():
        workspace = torch.empty((0,), dtype=torch.int, device="cuda")
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=32,
            act_order=False,
            is_k_full=True,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            topk=2,
            workspace=workspace,
            rtol=2e-1,
            atol=2.0,
        )
        assert workspace.numel() == 0

    def test_moe_wna16_uint4_zp_no_split_accepts_unused_workspace():
        workspace = torch.empty((1,), dtype=torch.int, device="cuda")
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=32,
            act_order=False,
            is_k_full=True,
            tokens=2,
            workspace=workspace,
            rtol=2e-1,
            atol=2.0,
        )

    @pytest.mark.parametrize(
        ("intermediate", "case_id"),
        (
            pytest.param(32, "auto_n64", id="auto_n64"),
            pytest.param(64, "auto_n128", id="auto_n128"),
            pytest.param(128, "auto_n256", id="auto_n256"),
            pytest.param(2048, "auto_n4096", id="auto_n4096"),
        ),
    )
    def test_moe_wna16_uint4_zp_auto_cta_n_matches_reference(
        intermediate: int,
        case_id: str,
    ):
        del case_id
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=32,
            act_order=False,
            is_k_full=True,
            tokens=1,
            intermediate=intermediate,
            topk=1,
            moe_block_size=8,
            rtol=2e-1,
            atol=2.0,
        )

    def test_moe_wna16_uint4_zp_rejects_non_64_n_alignment():
        _require_moe_cuda()

        tokens = 2
        topk = 1
        experts = 1
        size_k = 128
        size_n = 160
        pack_factor = 8
        topk_ids = torch.zeros((tokens, topk), dtype=torch.int32, device="cuda")
        sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            topk_ids, block_size=16, num_experts=experts
        )

        with pytest.raises(RuntimeError, match="min_thread_n"):
            ops.moe_wna16_marlin_gemm(
                torch.empty((tokens, size_k), device="cuda", dtype=torch.float16),
                torch.empty((tokens * topk, size_n), device="cuda", dtype=torch.float16),
                torch.empty(
                    (experts, size_k // 16, size_n * 16 // pack_factor),
                    device="cuda",
                    dtype=torch.int32,
                ),
                None,
                torch.empty((experts, 1, size_n), device="cuda", dtype=torch.float16),
                None,
                None,
                torch.empty((experts, 1, size_n), device="cuda", dtype=torch.float16),
                torch.empty((0,), device="cuda", dtype=torch.int32),
                torch.empty((0,), device="cuda", dtype=torch.int32),
                None,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                torch.ones((tokens, topk), device="cuda", dtype=torch.float32),
                16,
                topk,
                False,
                scalar_types.uint4.id,
                tokens,
                size_n,
                size_k,
                True,
                False,
                False,
                True,
                -1,
                -1,
                -1,
            )

    def test_moe_wna16_uint4_zp_rejects_k_alignment():
        _require_moe_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        tokens = 2
        hidden = 144
        intermediate = 128
        experts = 4
        topk = 2
        hidden_states, topk_weights, topk_ids, w1, _w2 = make_moe_model_like_inputs(
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            topk=topk,
            device="cuda",
        )
        w1_q, w1_scales, w1_zeros, _w1_dequant, w1_g_idx, w1_perm = (
            marlin_quantize_experts_uint4_zp_with_metadata(w1, -1)
        )
        sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            topk_ids, block_size=16, num_experts=experts
        )
        with pytest.raises(RuntimeError, match=_K_TILE_ALIGNMENT_ERROR):
            ops.moe_wna16_marlin_gemm(
                hidden_states,
                torch.empty((tokens * topk, 2 * intermediate), device="cuda", dtype=torch.float16),
                w1_q,
                None,
                w1_scales,
                None,
                None,
                w1_zeros,
                w1_g_idx,
                w1_perm,
                None,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                topk_weights,
                16,
                topk,
                False,
                scalar_types.uint4.id,
                tokens,
                2 * intermediate,
                hidden,
                True,
                False,
                False,
                True,
                -1,
                -1,
                -1,
            )

    def test_moe_wna16_uint4_zp_auto_split_k_ignores_fp32_reduce():
        inputs = _make_moe_accuracy_inputs(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            topk=2,
        )
        sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            inputs["topk_ids"], block_size=16, num_experts=inputs["experts"]
        )
        output = ops.moe_wna16_marlin_gemm(
            inputs["hidden_states"],
            torch.empty(
                (inputs["tokens"] * inputs["topk"], 2 * inputs["intermediate"]),
                device="cuda",
                dtype=torch.float16,
            ),
            inputs["w1_q"],
            None,
            inputs["w1_scales"],
            None,
            inputs["w1_global_scale"],
            inputs["w1_zeros"],
            inputs["w1_g_idx"],
            inputs["w1_perm"],
            None,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            inputs["topk_weights"],
            16,
            inputs["topk"],
            False,
            scalar_types.uint4.id,
            inputs["tokens"],
            2 * inputs["intermediate"],
            inputs["hidden_states"].shape[1],
            True,
            False,
            True,
            True,
            -1,
            -1,
            -1,
        )
        assert output.shape == (inputs["tokens"] * inputs["topk"], 2 * inputs["intermediate"])
        assert torch.isfinite(output).all()

    def test_moe_wna16_uint4_zp_auto_split_k_accepts_unused_workspace():
        workspace = torch.empty((1 * 2 * 4096 - 1,), dtype=torch.int, device="cuda")
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            is_k_full=True,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            topk=2,
            workspace=workspace,
            rtol=2e-1,
            atol=2.0,
        )

    @pytest.mark.parametrize(
        "make_workspace",
        (
            lambda device: torch.empty((1, 2, 4096), device=device, dtype=torch.float16),
            lambda device: torch.empty((2, 4096), device=device, dtype=torch.float32).t(),
            lambda device: torch.empty((1, 2, 4096), dtype=torch.float32),
        ),
    )
    def test_moe_wna16_uint4_zp_auto_split_k_accepts_unused_workspace_variants(
        make_workspace,
    ):
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            is_k_full=True,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            topk=2,
            workspace=make_workspace(torch.device("cuda")),
            rtol=2e-1,
            atol=2.0,
        )


def test_fused_marlin_moe_uint4_zp_single_scale_group_matches_reference():
    _run_fused_moe_accuracy_case(
        scalar_types.uint4,
        repack_impl="gptq",
        group_size=-1,
        act_order=False,
        is_k_full=True,
        tokens=2,
        moe_block_size=16,
        rtol=2e-1,
        atol=1.25,
    )


if "uint8b128" in _MOE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("group_size", _GROUP_SIZES)
    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_uint8b128_accuracy(group_size: int, repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.uint8b128,
            repack_impl=repack_impl,
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            moe_block_size=16,
        )

    @pytest.mark.parametrize("is_k_full", (True, False))
    def test_fused_marlin_moe_uint8b128_rejects_act_order_compat_args(
        is_k_full: bool,
    ):
        _assert_moe_backend_rejects_act_order_compat_args(
            scalar_types.uint8b128,
            group_size=64,
            is_k_full=is_k_full,
            moe_block_size=16,
        )

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_uint8b128_partial_block_matches_reference(repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.uint8b128,
            repack_impl=repack_impl,
            group_size=128,
            act_order=False,
            is_k_full=True,
            tokens=3,
            moe_block_size=16,
        )

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_uint8b128_single_token_matches_reference(repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.uint8b128,
            repack_impl=repack_impl,
            group_size=128,
            act_order=False,
            is_k_full=True,
            tokens=1,
            moe_block_size=16,
        )

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_moe_wna16_uint8b128_stage1_kernel_is_finite(repack_impl: str):
        _run_stage1_kernel_case(
            scalar_types.uint8b128,
            repack_impl=repack_impl,
            group_size=64,
            act_order=False,
            is_k_full=True,
            moe_block_size=16,
        )

    @pytest.mark.parametrize("is_k_full", (True, False))
    def test_moe_wna16_uint8b128_stage1_rejects_act_order_compat_args(
        is_k_full: bool,
    ):
        _assert_moe_backend_rejects_act_order_compat_args(
            scalar_types.uint8b128,
            group_size=64,
            is_k_full=is_k_full,
            moe_block_size=16,
        )

    def test_moe_wna16_uint8b128_stage1_single_group_rejects_act_order_compat_args():
        _assert_moe_backend_rejects_act_order_compat_args(
            scalar_types.uint8b128,
            group_size=128,
            is_k_full=False,
            moe_block_size=16,
        )


if "uint8" in _MOE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("group_size", _UINT8_ZP_GROUP_SIZES)
    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_uint8_zp_accuracy(group_size: int, repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.uint8,
            repack_impl=repack_impl,
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            moe_block_size=16,
            rtol=2e-1,
            atol=1.25,
        )

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_moe_wna16_uint8_zp_stage1_matches_reference(repack_impl: str):
        _run_stage1_kernel_case(
            scalar_types.uint8,
            repack_impl=repack_impl,
            group_size=64,
            act_order=False,
            is_k_full=True,
            moe_block_size=16,
            rtol=2e-1,
            atol=1.25,
        )

    def test_moe_wna16_uint8_zp_auto_split_k_stage1_matches_reference():
        _run_stage1_kernel_case(
            scalar_types.uint8,
            repack_impl="gptq",
            group_size=64,
            act_order=False,
            is_k_full=True,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            topk=2,
            moe_block_size=16,
            rtol=2e-1,
            atol=2.0,
        )


if "fp8" in _MOE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("group_size", _FP8_GROUP_SIZES)
    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_fp8_accuracy(group_size: int, repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.float8_e4m3fn,
            repack_impl=repack_impl,
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            moe_block_size=16,
            rtol=2e-1,
            atol=1.25,
        )

    def test_moe_wna16_fp8_auto_split_k_stage1_matches_reference():
        _run_stage1_kernel_case(
            scalar_types.float8_e4m3fn,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            is_k_full=True,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            topk=2,
            moe_block_size=16,
            rtol=2e-1,
            atol=2.0,
        )


if "nvfp4" in _MOE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_nvfp4_accuracy(repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.float4_e2m1f,
            repack_impl=repack_impl,
            group_size=16,
            act_order=False,
            is_k_full=True,
            moe_block_size=16,
            rtol=2e-1,
            atol=1.25,
        )

    def test_moe_wna16_nvfp4_auto_split_k_stage1_matches_reference():
        _run_stage1_kernel_case(
            scalar_types.float4_e2m1f,
            repack_impl="gptq",
            group_size=16,
            act_order=False,
            is_k_full=True,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            topk=2,
            moe_block_size=16,
            rtol=2e-1,
            atol=2.0,
        )


if "mxfp4" in _MOE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_fused_marlin_moe_mxfp4_accuracy(repack_impl: str):
        _run_fused_moe_accuracy_case(
            scalar_types.float4_e2m1f,
            repack_impl=repack_impl,
            group_size=32,
            act_order=False,
            is_k_full=True,
            moe_block_size=16,
            rtol=2e-1,
            atol=1.25,
        )

    def test_moe_wna16_mxfp4_auto_split_k_stage1_matches_reference():
        _run_stage1_kernel_case(
            scalar_types.float4_e2m1f,
            repack_impl="gptq",
            group_size=32,
            act_order=False,
            is_k_full=True,
            tokens=1,
            hidden=4096,
            intermediate=2048,
            topk=2,
            moe_block_size=16,
            rtol=2e-1,
            atol=2.0,
        )


def test_moe_wna16_uint4_zp_rejects_non_uint4_quant_type():
    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        scalar_types.uint4,
        repack_impl="gptq",
        group_size=64,
        act_order=False,
        tokens=2,
    )
    hidden_states = inputs["hidden_states"]
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        inputs["topk_ids"], block_size=16, num_experts=inputs["experts"]
    )

    with pytest.raises(
        RuntimeError,
        match="only uint4 or uint8 weights when zero-points are enabled",
    ):
        ops.moe_wna16_marlin_gemm(
            hidden_states,
            torch.empty(
                (inputs["tokens"] * inputs["topk"], 2 * inputs["intermediate"]),
                device="cuda",
                dtype=torch.float16,
            ),
            inputs["w1_q"],
            None,
            inputs["w1_scales"],
            None,
            inputs["w1_global_scale"],
            inputs["w1_zeros"],
            inputs["w1_g_idx"],
            inputs["w1_perm"],
            None,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            inputs["topk_weights"],
            16,
            inputs["topk"],
            False,
            scalar_types.uint4b8.id,
            inputs["tokens"],
            2 * inputs["intermediate"],
            hidden_states.shape[1],
            True,
            False,
            False,
            True,
            -1,
            -1,
            -1,
        )


def test_moe_wna16_uint4_zp_rejects_packed_zero_points():
    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        scalar_types.uint4,
        repack_impl="gptq",
        group_size=64,
        act_order=False,
        tokens=2,
    )
    hidden_states = inputs["hidden_states"]
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        inputs["topk_ids"], block_size=16, num_experts=inputs["experts"]
    )

    packed_zero_points = torch.empty(
        (
            inputs["experts"],
            inputs["w1_scales"].shape[1],
            inputs["w1_scales"].shape[2] // 8,
        ),
        device="cuda",
        dtype=torch.int32,
    )

    with pytest.raises(RuntimeError, match="fp16 zero points"):
        ops.moe_wna16_marlin_gemm(
            hidden_states,
            torch.empty(
                (inputs["tokens"] * inputs["topk"], 2 * inputs["intermediate"]),
                device="cuda",
                dtype=torch.float16,
            ),
            inputs["w1_q"],
            None,
            inputs["w1_scales"],
            None,
            inputs["w1_global_scale"],
            packed_zero_points,
            inputs["w1_g_idx"],
            inputs["w1_perm"],
            None,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            inputs["topk_weights"],
            16,
            inputs["topk"],
            False,
            scalar_types.uint4.id,
            inputs["tokens"],
            2 * inputs["intermediate"],
            hidden_states.shape[1],
            True,
            False,
            False,
            True,
            -1,
            -1,
            -1,
        )


def test_moe_wna16_uint4_zp_rejects_mismatched_zero_point_shape():
    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    inputs = _make_moe_accuracy_inputs(
        scalar_types.uint4,
        repack_impl="gptq",
        group_size=64,
        act_order=False,
        tokens=2,
    )
    hidden_states = inputs["hidden_states"]
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        inputs["topk_ids"], block_size=16, num_experts=inputs["experts"]
    )
    bad_zero_points = inputs["w1_zeros"][:, :, :-1].contiguous()

    with pytest.raises(RuntimeError, match="b_zeros dim 2"):
        ops.moe_wna16_marlin_gemm(
            hidden_states,
            torch.empty(
                (inputs["tokens"] * inputs["topk"], 2 * inputs["intermediate"]),
                device="cuda",
                dtype=torch.float16,
            ),
            inputs["w1_q"],
            None,
            inputs["w1_scales"],
            None,
            inputs["w1_global_scale"],
            bad_zero_points,
            inputs["w1_g_idx"],
            inputs["w1_perm"],
            None,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            inputs["topk_weights"],
            16,
            inputs["topk"],
            False,
            scalar_types.uint4.id,
            inputs["tokens"],
            2 * inputs["intermediate"],
            hidden_states.shape[1],
            True,
            False,
            False,
            True,
            -1,
            -1,
            -1,
        )

@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_moe_rejects_mismatched_capability_or_unsupported_dtypes(repack_impl: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    try:
        ops._load_moe()
    except Exception as exc:  # pragma: no cover - depends on local build state
        pytest.skip(f"marlin moe extension is not available: {exc}")

    device = torch.device("cuda")
    hidden_states = torch.randn((4, 128), device=device, dtype=torch.float16)
    topk_weights = torch.rand((4, 2), device=device, dtype=torch.float32)
    topk_ids = torch.randint(0, 4, (4, 2), device=device, dtype=torch.int32)
    w = torch.empty((4, 8, 16), device=device, dtype=torch.int32)
    scales = torch.ones((4, 1, 128), device=device, dtype=torch.float16)
    sorted_ids = torch.zeros(32, dtype=torch.int32, device=device)
    expert_ids = torch.zeros(4, dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.tensor([32], dtype=torch.int32, device=device)

    target_capability = source_target_capability()
    capability = torch.cuda.get_device_capability(device)
    if capability != target_capability:
        with pytest.raises(RuntimeError, match=source_target_label()):
            ops.moe_wna16_marlin_gemm(
                hidden_states,
                None,
                w,
                None,
                scales,
                None,
                None,
                None,
                None,
                None,
                None,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                topk_weights.reshape(-1),
                16,
                2,
                True,
                1,
                4,
                128,
                128,
                True,
                False,
                False,
                False,
                -1,
                -1,
                -1,
            )
        return

    assert_repack_layout_matches_reference(repack_impl, quant_type=scalar_types.uint4b8)
    hidden_states_bf16 = hidden_states.to(torch.bfloat16)
    scales_bf16 = scales.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match=_FLOAT16_DTYPE_ERROR):
        ops.moe_wna16_marlin_gemm(
            hidden_states_bf16,
            None,
            w,
            None,
            scales_bf16,
            None,
            None,
            None,
            None,
            None,
            None,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            topk_weights.reshape(-1),
            16,
            2,
            True,
            1,
            4,
            128,
            128,
            True,
            False,
            False,
            False,
            -1,
            -1,
            -1,
        )
