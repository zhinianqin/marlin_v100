from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marlin_v100.calibration import supported_dense_quant_type_names
from tests.helpers import marlin_dequantize, marlin_quantize, marlin_quantize_experts, scalar_types


_ROUNDTRIP_TOLERANCES = {
    "uint4b8": (scalar_types.uint4b8, 3.0e-1, 2.0e-1),
    "uint8b128": (scalar_types.uint8b128, 2.0e-2, 2.0e-2),
}
_ROUNDTRIP_CASES = [
    pytest.param(_ROUNDTRIP_TOLERANCES[name][0], _ROUNDTRIP_TOLERANCES[name][1], _ROUNDTRIP_TOLERANCES[name][2], id=name)
    for name in supported_dense_quant_type_names(_ROUNDTRIP_TOLERANCES)
]


@pytest.mark.parametrize(
    ("quant_type", "atol", "rtol"),
    _ROUNDTRIP_CASES,
)
def test_marlin_quantize_round_trip_matches_original_weight(
    quant_type,
    atol: float,
    rtol: float,
):
    torch.manual_seed(0)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _, q_weight, scales, _g_idx, _sort_indices, _rand_perm = marlin_quantize(
        weight, quant_type, 128, False
    )
    dequantized = marlin_dequantize(
        q_weight,
        scales,
        size_k=weight.shape[0],
        size_n=weight.shape[1],
        group_size=128,
        quant_type=quant_type,
    )

    assert dequantized.shape == weight.shape
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, weight, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    ("quant_type", "atol", "rtol"),
    _ROUNDTRIP_CASES,
)
def test_marlin_quantize_experts_round_trip_matches_original_weights(
    quant_type,
    atol: float,
    rtol: float,
):
    torch.manual_seed(0)
    weights = torch.randn((2, 128, 256), dtype=torch.float16)
    q_weights, scales, dequantized = marlin_quantize_experts(weights, quant_type, 128, False)

    assert q_weights.shape[0] == weights.shape[0]
    assert scales.shape == (weights.shape[0], 1, weights.shape[2])
    assert dequantized.shape == weights.shape
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, weights, atol=atol, rtol=rtol)
