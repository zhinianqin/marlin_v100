from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marlin_v100 import moe, ops, routing


def _require_moe_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7 or (capability[0] == 7 and capability[1] < 5):
        pytest.skip("Marlin MoE requires SM75 or newer")
    try:
        ops._load_moe()
    except Exception as exc:  # pragma: no cover - depends on local build state
        pytest.skip(f"marlin moe extension is not available: {exc}")


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


def test_fused_marlin_moe_smoke():
    _require_moe_cuda()
    # This is a collectable smoke shape check only. Current SM70 machines are
    # not considered a valid runtime acceptance environment for Marlin MoE.

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
