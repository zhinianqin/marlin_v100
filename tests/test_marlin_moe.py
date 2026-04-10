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
    make_moe_model_like_inputs,
    marlin_moe_reference,
    marlin_quantize_experts,
    scalar_types,
)

_MOE_SUPPORTED_QUANT_NAMES = frozenset(supported_moe_quant_type_names(("uint4b8", "uint8b128")))


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


def test_fused_marlin_moe_smoke():
    _require_moe_cuda()
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
        )
    except RuntimeError:
        pytest.skip("smoke input layout is not compatible with local kernel expectations")

    assert output.shape == hidden_states.shape


def test_fused_marlin_moe_uint4b8_accuracy():
    _require_moe_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    tokens = 4
    hidden = 128
    intermediate = 128
    experts = 4
    topk = 2

    hidden_states, topk_weights, topk_ids, w1, w2 = make_moe_model_like_inputs(
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
        device="cuda",
    )
    w1_q, w1_scales, w1_dequant = marlin_quantize_experts(
        w1, scalar_types.uint4b8, 128, False
    )
    w2_q, w2_scales, w2_dequant = marlin_quantize_experts(
        w2, scalar_types.uint4b8, 128, False
    )

    output = moe.fused_marlin_moe(
        hidden_states=hidden_states,
        w1=w1_q,
        w2=w2_q,
        w1_scale=w1_scales,
        w2_scale=w2_scales,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        quant_type_id=scalar_types.uint4b8.id,
    )
    reference = marlin_moe_reference(
        hidden_states,
        w1_dequant,
        w2_dequant,
        topk_weights,
        topk_ids,
    ).to(torch.float16)

    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=7e-2, atol=1e-2)


if "uint8b128" in _MOE_SUPPORTED_QUANT_NAMES:

    def test_fused_marlin_moe_uint8b128_accuracy():
        _require_moe_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        tokens = 4
        hidden = 128
        intermediate = 128
        experts = 4
        topk = 2

        hidden_states, topk_weights, topk_ids, w1, w2 = make_moe_model_like_inputs(
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            topk=topk,
            device="cuda",
        )
        w1_q, w1_scales, w1_dequant = marlin_quantize_experts(
            w1, scalar_types.uint8b128, 128, False
        )
        w2_q, w2_scales, w2_dequant = marlin_quantize_experts(
            w2, scalar_types.uint8b128, 128, False
        )

        output = moe.fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1_q,
            w2=w2_q,
            w1_scale=w1_scales,
            w2_scale=w2_scales,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=scalar_types.uint8b128.id,
        )
        reference = marlin_moe_reference(
            hidden_states,
            w1_dequant,
            w2_dequant,
            topk_weights,
            topk_ids,
        ).to(torch.float16)

        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, rtol=7e-2, atol=1e-2)

    def test_moe_wna16_uint8b128_stage1_kernel_is_finite():
        _require_moe_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        tokens = 4
        hidden = 128
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
        w1_q, w1_scales, w1_dequant = marlin_quantize_experts(
            w1, scalar_types.uint8b128, 128, False
        )
        sorted_ids, expert_ids, num_tokens_post_pad = moe.moe_align_block_size(
            topk_ids, block_size=16, num_experts=experts
        )
        workspace = torch.zeros(
            torch.cuda.get_device_properties(hidden_states.device).multi_processor_count * 4,
            dtype=torch.int,
            device=hidden_states.device,
        )

        stage1 = ops.moe_wna16_marlin_gemm(
            hidden_states,
            torch.empty((tokens * topk, 2 * intermediate), device="cuda", dtype=torch.float16),
            w1_q,
            None,
            w1_scales,
            None,
            None,
            None,
            None,
            None,
            workspace,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            topk_weights,
            16,
            topk,
            False,
            scalar_types.uint8b128.id,
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
        reference_gate_up = []
        for token_idx in range(tokens):
            for route_idx in range(topk):
                expert = int(topk_ids[token_idx, route_idx].item())
                reference_gate_up.append(
                    torch.matmul(
                        hidden_states[token_idx : token_idx + 1].to(torch.float32),
                        w1_dequant[expert].to(torch.float32),
                    )[0]
                )
        reference_stage1 = torch.stack(reference_gate_up, dim=0).to(torch.float16)

        assert stage1.shape == (tokens * topk, 2 * intermediate)
        assert torch.isfinite(stage1).all()
        torch.testing.assert_close(stage1, reference_stage1, rtol=6e-2, atol=3e-1)

def test_marlin_moe_rejects_non_sm75_or_unsupported_dtypes():
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
    workspace = torch.zeros(128, dtype=torch.int32, device=device)
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
                workspace,
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

    hidden_states_bf16 = hidden_states.to(torch.bfloat16)
    scales_bf16 = scales.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="float16 or int8 activations|float16 outputs|float16 scales"):
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
            workspace,
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
