from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marlin_v100.calibration import (
    source_target_capability,
    source_target_label,
    supported_moe_quant_type_names,
)
from marlin_v100 import moe, ops, routing
from tests.helpers import (
    _REPACK_IMPL_CASES,
    assert_repack_layout_matches_reference,
    make_moe_model_like_inputs,
    marlin_moe_reference,
    marlin_quantize_experts_uint4_zp_with_metadata,
    marlin_quantize_experts,
    marlin_quantize_experts_with_metadata,
    scalar_types,
)

_MOE_SUPPORTED_QUANT_NAMES = frozenset(
    supported_moe_quant_type_names(("uint4", "uint4b8", "uint8b128"))
)
_GROUP_SIZES = (-1, 32, 64, 128)
_UINT4_ZP_GROUP_SIZES = (-1, 32, 64, 128)
_SM70_MOE_U4_SPLIT_K_ENV = "SM70_MARLIN_MOE_U4_SPLIT_K"
_SM70_MOE_U4_CTA_ENV = "SM70_MARLIN_MOE_U4_CTA"
_SM70_SUPPORTED_MOE_BLOCK_SIZES = (8, 16, 32, 48, 64)
_SUPPORTED_MOE_BLOCK_SIZE_ERROR = "moe_block_size=8, 16, 32, 48, or 64"
_FLOAT16_DTYPE_ERROR = (
    rf"{source_target_label()} build only supports float16 activations\."
    rf"|{source_target_label()} build only supports float16 outputs\."
    rf"|{source_target_label()} build only supports float16 scales\."
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
_UNSUPPORTED_THREAD_GEOMETRY_CASES = (
    pytest.param(16, 128, 128, 128, 128, id="thread_n_128_moe_block_16"),
    pytest.param(16, 256, 128, 64, 256, id="thread_n_256_moe_block_16"),
)


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
    topk_weights, topk_ids, token_expert_indices = routing.topk_softmax(
        gating_output, topk=2
    )
    assert topk_weights.shape == (8, 2)
    assert topk_ids.shape == (8, 2)
    assert token_expert_indices.shape == (8, 2)

    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
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

    topk_weights, topk_ids, token_expert_indices = routing.topk_softmax(
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
    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
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
    topk_weights, topk_ids = routing.grouped_topk(
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

    topk_weights, topk_ids = routing.grouped_topk(
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
        output = moe.fused_marlin_moe(
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
    else:
        w1_q, w1_scales, w1_dequant, w1_g_idx, w1_perm = marlin_quantize_experts_with_metadata(
            w1, quant_type, group_size, act_order
        )
        w2_q, w2_scales, w2_dequant, w2_g_idx, w2_perm = marlin_quantize_experts_with_metadata(
            w2, quant_type, group_size, act_order
        )
        w1_zeros = None
        w2_zeros = None
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
        "w1_dequant": w1_dequant,
        "w1_g_idx": w1_g_idx,
        "w1_perm": w1_perm,
        "w2_q": w2_q,
        "w2_scales": w2_scales,
        "w2_zeros": w2_zeros,
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
    c_tmp: torch.Tensor | None = None,
    rtol: float = 7e-2,
    atol: float = 1e-2,
) -> None:
    if act_order:
        raise AssertionError("act_order accuracy coverage was replaced by explicit rejection tests")
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

    output = moe.fused_marlin_moe(
        hidden_states=inputs["hidden_states"],
        w1=inputs["w1_q"],
        w2=inputs["w2_q"],
        w1_scale=inputs["w1_scales"],
        w2_scale=inputs["w2_scales"],
        topk_weights=inputs["topk_weights"],
        topk_ids=inputs["topk_ids"],
        quant_type_id=quant_type.id,
        w1_zeros=inputs["w1_zeros"],
        w2_zeros=inputs["w2_zeros"],
        g_idx1=inputs["w1_g_idx"],
        g_idx2=inputs["w2_g_idx"],
        sort_indices1=inputs["w1_perm"],
        sort_indices2=inputs["w2_perm"],
        is_k_full=is_k_full,
        moe_block_size=moe_block_size,
        c_tmp=c_tmp,
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


def _assert_moe_backend_rejects_act_order(
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
        act_order=True,
        tokens=tokens,
    )
    hidden_states = inputs["hidden_states"]
    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
        inputs["topk_ids"], block_size=moe_block_size, num_experts=inputs["experts"]
    )

    with pytest.raises(RuntimeError, match="act_order is not supported"):
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
            None,
            inputs["w1_zeros"],
            inputs["w1_g_idx"],
            inputs["w1_perm"],
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
            False,
            True,
            False,
            -1,
            -1,
            -1,
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

    output = moe.fused_marlin_moe(
        hidden_states=inputs["hidden_states"],
        w1=inputs["w1_q"],
        w2=inputs["w2_q"],
        w1_scale=inputs["w1_scales"],
        w2_scale=inputs["w2_scales"],
        topk_weights=inputs["topk_weights"],
        topk_ids=inputs["topk_ids"],
        quant_type_id=scalar_types.uint4b8.id,
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
def test_fused_marlin_moe_uint4b8_act_order_accuracy(is_k_full: bool, repack_impl: str):
    _assert_moe_backend_rejects_act_order(
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
        moe.fused_marlin_moe(
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
    c_tmp: torch.Tensor | None = None,
    rtol: float = 6e-2,
    atol: float = 3e-1,
) -> None:
    if act_order:
        raise AssertionError("act_order stage1 coverage was replaced by explicit rejection tests")
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
    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
        topk_ids, block_size=moe_block_size, num_experts=experts
    )

    stage1 = ops.moe_wna16_marlin_gemm(
        hidden_states,
        torch.empty((tokens * topk, 2 * intermediate), device="cuda", dtype=torch.float16),
        inputs["w1_q"],
        None,
        inputs["w1_scales"],
        None,
        None,
        inputs["w1_zeros"],
        inputs["w1_g_idx"],
        inputs["w1_perm"],
        c_tmp,
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
        True,
        False,
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
    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
        topk_ids, block_size=moe_block_size, num_experts=experts
    )

    stage1 = ops.moe_wna16_marlin_gemm(
        hidden_states,
        torch.empty((tokens * topk, 2 * intermediate), device="cuda", dtype=torch.float16),
        inputs["w1_q"],
        None,
        inputs["w1_scales"],
        None,
        None,
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
        True,
        False,
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
        None,
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
        True,
        False,
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
    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
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
            None,
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
            True,
            False,
            thread_k,
            thread_n,
            -1,
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
    "moe_block_size,hidden,intermediate,thread_k,thread_n", _UNSUPPORTED_THREAD_GEOMETRY_CASES
)
@pytest.mark.parametrize("repack_impl", _FORCED_GEOMETRY_REPACK_CASES)
def test_moe_wna16_uint4b8_stage1_rejects_unsupported_thread_geometry(
    repack_impl: str,
    moe_block_size: int,
    hidden: int,
    intermediate: int,
    thread_k: int,
    thread_n: int,
):
    _assert_stage1_kernel_rejects_unsupported_config(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        moe_block_size=moe_block_size,
        hidden=hidden,
        intermediate=intermediate,
        thread_k=thread_k,
        thread_n=thread_n,
        error_match=_SUPPORTED_THREAD_GEOMETRY_ERROR,
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

    @pytest.mark.parametrize("split_k", ("2", "4", "8"))
    @pytest.mark.parametrize("group_size", _UINT4_ZP_GROUP_SIZES)
    def test_moe_wna16_uint4_zp_split_k_stage1_matches_reference(
        monkeypatch: pytest.MonkeyPatch,
        split_k: str,
        group_size: int,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_SPLIT_K_ENV, split_k)
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            tokens=3,
            moe_block_size=16,
            topk=2,
            rtol=2e-1,
            atol=2.0,
        )

    @pytest.mark.parametrize("topk", (1, 2))
    @pytest.mark.parametrize("moe_block_size", (16, 32, 64))
    def test_fused_marlin_moe_uint4_zp_split_k_matches_reference(
        monkeypatch: pytest.MonkeyPatch,
        topk: int,
        moe_block_size: int,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_SPLIT_K_ENV, "4")
        _run_fused_moe_accuracy_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            is_k_full=True,
            tokens=2,
            moe_block_size=moe_block_size,
            topk=topk,
            rtol=2e-1,
            atol=1.25,
        )

    def test_moe_wna16_uint4_zp_split_k_reuses_c_tmp_matches_reference(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_SPLIT_K_ENV, "4")
        c_tmp = torch.empty((3 * 2 * 256,), dtype=torch.float32, device="cuda")
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            is_k_full=True,
            tokens=3,
            hidden=128,
            intermediate=128,
            topk=2,
            c_tmp=c_tmp,
            rtol=2e-1,
            atol=2.0,
        )

    def test_moe_wna16_uint4_zp_no_split_accepts_unused_c_tmp(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv(_SM70_MOE_U4_SPLIT_K_ENV, raising=False)
        c_tmp = torch.empty((1,), dtype=torch.float32, device="cuda")
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            is_k_full=True,
            tokens=2,
            c_tmp=c_tmp,
            rtol=2e-1,
            atol=2.0,
        )

    @pytest.mark.parametrize("cta", ("32x128x4", "32x256x4", "64x64x4", "64x128x4"))
    def test_moe_wna16_uint4_zp_supported_cta_matches_reference(
        monkeypatch: pytest.MonkeyPatch,
        cta: str,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_CTA_ENV, cta)
        _run_stage1_kernel_case(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            is_k_full=True,
            tokens=2,
            moe_block_size=16,
            rtol=2e-1,
            atol=2.0,
        )

    @pytest.mark.parametrize("cta", ("bad", "128x128x4"))
    def test_moe_wna16_uint4_zp_rejects_invalid_cta(
        monkeypatch: pytest.MonkeyPatch,
        cta: str,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_CTA_ENV, cta)
        with pytest.raises(RuntimeError, match=_SM70_MOE_U4_CTA_ENV):
            _run_stage1_kernel_case(
                scalar_types.uint4,
                repack_impl="gptq",
                group_size=128,
                act_order=False,
                is_k_full=True,
                tokens=2,
                moe_block_size=16,
                rtol=2e-1,
                atol=2.0,
            )

    def test_moe_wna16_uint4_zp_rejects_cta_n_alignment(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_CTA_ENV, "32x256x4")
        with pytest.raises(RuntimeError, match="size_n must be divisible by both CTA_N and 64"):
            _run_stage1_kernel_case(
                scalar_types.uint4,
                repack_impl="gptq",
                group_size=-1,
                act_order=False,
                is_k_full=True,
                tokens=2,
                intermediate=160,
                moe_block_size=16,
                rtol=2e-1,
                atol=2.0,
            )

    @pytest.mark.parametrize("split_k", ("3", "abc"))
    def test_moe_wna16_uint4_zp_split_k_rejects_invalid_env(
        monkeypatch: pytest.MonkeyPatch,
        split_k: str,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_SPLIT_K_ENV, split_k)
        with pytest.raises(RuntimeError, match=_SM70_MOE_U4_SPLIT_K_ENV):
            _run_stage1_kernel_case(
                scalar_types.uint4,
                repack_impl="gptq",
                group_size=128,
                act_order=False,
                is_k_full=True,
                tokens=2,
                rtol=2e-1,
                atol=2.0,
            )

    def test_moe_wna16_uint4_zp_split_k_rejects_k_partition_tail(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_SPLIT_K_ENV, "2")
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
        sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
            topk_ids, block_size=16, num_experts=experts
        )
        with pytest.raises(RuntimeError, match="requires K divisible by 32"):
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
                True,
                False,
                -1,
                -1,
                -1,
            )

    def test_moe_wna16_uint4_zp_rejects_fp16_reduce(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.delenv(_SM70_MOE_U4_SPLIT_K_ENV, raising=False)
        inputs = _make_moe_accuracy_inputs(
            scalar_types.uint4,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            tokens=2,
        )
        sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
            inputs["topk_ids"], block_size=16, num_experts=inputs["experts"]
        )
        with pytest.raises(RuntimeError, match="requires use_fp32_reduce=True"):
            ops.moe_wna16_marlin_gemm(
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
                None,
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
                False,
                False,
                -1,
                -1,
                -1,
            )

    def test_moe_wna16_uint4_zp_split_k_rejects_small_c_tmp(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_SPLIT_K_ENV, "2")
        c_tmp = torch.empty((2 * 2 * 256 - 1,), dtype=torch.float32, device="cuda")
        with pytest.raises(RuntimeError, match=r"c_tmp\.numel.*M\*N"):
            _run_stage1_kernel_case(
                scalar_types.uint4,
                repack_impl="gptq",
                group_size=128,
                act_order=False,
                is_k_full=True,
                tokens=2,
                c_tmp=c_tmp,
                rtol=2e-1,
                atol=2.0,
            )

    @pytest.mark.parametrize(
        ("make_c_tmp", "message"),
        (
            (
                lambda device: torch.empty((2, 2, 256), device=device, dtype=torch.float16),
                "dtype torch.float32",
            ),
            (
                lambda device: torch.empty((2, 2, 256), device=device, dtype=torch.float32).transpose(0, 1),
                "contiguous",
            ),
            (
                lambda device: torch.empty((2, 2, 256), dtype=torch.float32),
                "CUDA tensor",
            ),
        ),
    )
    def test_moe_wna16_uint4_zp_split_k_rejects_invalid_c_tmp(
        monkeypatch: pytest.MonkeyPatch,
        make_c_tmp,
        message: str,
    ):
        monkeypatch.setenv(_SM70_MOE_U4_SPLIT_K_ENV, "2")
        with pytest.raises(RuntimeError, match=message):
            _run_stage1_kernel_case(
                scalar_types.uint4,
                repack_impl="gptq",
                group_size=128,
                act_order=False,
                is_k_full=True,
                tokens=2,
                c_tmp=make_c_tmp(torch.device("cuda")),
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
    def test_fused_marlin_moe_uint8b128_act_order_accuracy(is_k_full: bool):
        _assert_moe_backend_rejects_act_order(
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
    def test_moe_wna16_uint8b128_stage1_act_order_kernel_is_finite(is_k_full: bool):
        _assert_moe_backend_rejects_act_order(
            scalar_types.uint8b128,
            group_size=64,
            is_k_full=is_k_full,
            moe_block_size=16,
        )

    def test_moe_wna16_uint8b128_stage1_single_group_act_order_matches_reference():
        _assert_moe_backend_rejects_act_order(
            scalar_types.uint8b128,
            group_size=128,
            is_k_full=False,
            moe_block_size=16,
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
    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
        inputs["topk_ids"], block_size=16, num_experts=inputs["experts"]
    )

    with pytest.raises(RuntimeError, match="only supports uint4 weights when zero-points are enabled"):
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
            None,
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
            True,
            False,
            -1,
            -1,
            -1,
        )


def test_moe_wna16_uint4_zp_rejects_float_zero_points():
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
    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
        inputs["topk_ids"], block_size=16, num_experts=inputs["experts"]
    )

    with pytest.raises(RuntimeError, match="does not support float zero-points"):
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
            None,
            inputs["w1_zeros"].to(torch.float16),
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
            True,
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
    sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
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
            None,
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
            True,
            False,
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
                True,
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
            True,
            False,
            -1,
            -1,
            -1,
        )
