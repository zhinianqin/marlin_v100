from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marlin_v100 import dense, moe
from marlin_v100.calibration import supported_dense_quant_type_names
from tests.helpers import (
    marlin_dequantize,
    marlin_quantize,
    marlin_quantize_experts,
    scalar_types,
)


_ROUNDTRIP_TOLERANCES = {
    "uint4b8": (scalar_types.uint4b8, 3.0e-1, 2.0e-1),
    "uint8b128": (scalar_types.uint8b128, 2.0e-2, 2.0e-2),
}
_GROUP_SIZES = (-1, 32, 64, 128)
_ROUNDTRIP_CASES = [
    pytest.param(
        _ROUNDTRIP_TOLERANCES[name][0],
        _ROUNDTRIP_TOLERANCES[name][1],
        _ROUNDTRIP_TOLERANCES[name][2],
        id=name,
    )
    for name in supported_dense_quant_type_names(_ROUNDTRIP_TOLERANCES)
]


@pytest.mark.parametrize(
    ("quant_type", "atol", "rtol"),
    _ROUNDTRIP_CASES,
)
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_marlin_quantize_round_trip_matches_original_weight(
    quant_type,
    atol: float,
    rtol: float,
    group_size: int,
):
    torch.manual_seed(0)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _, q_weight, scales, _g_idx, _sort_indices, _rand_perm = marlin_quantize(
        weight, quant_type, group_size, False
    )
    dequantized = marlin_dequantize(
        q_weight,
        scales,
        size_k=weight.shape[0],
        size_n=weight.shape[1],
        group_size=group_size,
        quant_type=quant_type,
    )

    assert dequantized.shape == weight.shape
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, weight, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("quant_type", "atol", "rtol"),
    _ROUNDTRIP_CASES,
)
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_marlin_quantize_experts_round_trip_matches_original_weights(
    quant_type,
    atol: float,
    rtol: float,
    group_size: int,
):
    torch.manual_seed(0)
    weights = torch.randn((2, 128, 256), dtype=torch.float16)
    q_weights, scales, dequantized = marlin_quantize_experts(
        weights, quant_type, group_size, False
    )
    expected_groups = 1 if group_size == -1 else weights.shape[1] // group_size

    assert q_weights.shape[0] == weights.shape[0]
    assert scales.shape == (weights.shape[0], expected_groups, weights.shape[2])
    assert dequantized.shape == weights.shape
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, weights, atol=atol, rtol=rtol)


@pytest.mark.parametrize(("quant_type", "atol", "rtol"), _ROUNDTRIP_CASES)
@pytest.mark.parametrize("group_size", (32, 64, 128))
def test_marlin_quantize_act_order_round_trip_matches_original_weight(
    quant_type,
    atol: float,
    rtol: float,
    group_size: int,
):
    torch.manual_seed(0)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _, q_weight, scales, g_idx, sort_indices, _rand_perm = marlin_quantize(
        weight, quant_type, group_size, True
    )
    dequantized = marlin_dequantize(
        q_weight,
        scales,
        size_k=weight.shape[0],
        size_n=weight.shape[1],
        group_size=group_size,
        quant_type=quant_type,
        perm=sort_indices,
    )

    assert g_idx.shape == (weight.shape[0],)
    assert sort_indices.shape == (weight.shape[0],)
    assert not torch.equal(sort_indices, torch.arange(weight.shape[0], dtype=torch.int))
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, weight, atol=atol, rtol=rtol)


def test_marlin_quantize_rejects_group_size_zero_outside_runtime_act_order_path():
    weight = torch.randn((128, 256), dtype=torch.float16)
    with pytest.raises(ValueError, match="Unsupported dense group_size=0"):
        marlin_quantize(weight, scalar_types.uint4b8, 0, False)


def test_dense_wrapper_rejects_incomplete_act_order_metadata_before_loading_extension():
    a = torch.randn((4, 128), dtype=torch.float16)
    b_q_weight = torch.zeros((8, 256), dtype=torch.int32)
    b_scales = torch.ones((2, 256), dtype=torch.float16)
    workspace = torch.zeros((1,), dtype=torch.int32)
    g_idx = torch.arange(128, dtype=torch.int32)

    with pytest.raises(ValueError, match="g_idx and perm must be provided together"):
        dense.run_marlin_gemm(
            a,
            b_q_weight,
            b_scales,
            scalar_types.uint4b8.id,
            size_m=a.shape[0],
            size_n=256,
            size_k=a.shape[1],
            workspace=workspace,
            g_idx=g_idx,
            perm=None,
            is_k_full=True,
        )


def test_moe_wrapper_rejects_incomplete_act_order_metadata_before_loading_extension():
    hidden_states = torch.randn((4, 128), dtype=torch.float16)
    w1 = torch.zeros((2, 8, 256), dtype=torch.int32)
    w2 = torch.zeros((2, 8, 128), dtype=torch.int32)
    w1_scale = torch.ones((2, 2, 128), dtype=torch.float16)
    w2_scale = torch.ones((2, 2, 128), dtype=torch.float16)
    topk_weights = torch.ones((4, 2), dtype=torch.float32) / 2
    topk_ids = torch.zeros((4, 2), dtype=torch.int32)
    g_idx = torch.arange(128, dtype=torch.int32).repeat(2, 1)

    with pytest.raises(ValueError, match="g_idx and perm must be provided together"):
        moe.fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=scalar_types.uint4b8.id,
            g_idx1=g_idx,
            sort_indices1=None,
        )
