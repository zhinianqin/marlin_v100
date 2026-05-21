from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marlin_v100 import dense, moe, ops
from marlin_v100.calibration import (
    quant_type_name_from_id,
    source_target_capability,
    source_target_label,
    supported_dense_quant_type_names,
)
from tests.helpers import (
    _REPACK_IMPL_CASES,
    assert_repack_layout_matches_reference,
    fp4_e2m1_weight_to_marlin_weight,
    fp8_weight_to_marlin_weight,
    make_moe_model_like_inputs,
    marlin_dequantize_mxfp4,
    marlin_dequantize_nvfp4,
    marlin_dequantize_uint4_zp,
    marlin_dequantize_uint4_packed_zp,
    marlin_dequantize_uint8_zp,
    marlin_dequantize,
    marlin_unpack,
    marlin_quantize,
    marlin_quantize_fp8,
    marlin_quantize_mxfp4,
    marlin_quantize_nvfp4,
    marlin_quantize_experts_uint4_zp_with_metadata,
    marlin_quantize_experts,
    marlin_quantize_experts_with_metadata,
    marlin_quantize_uint4_zp,
    marlin_quantize_uint4_packed_zp,
    marlin_quantize_uint8_zp,
    scalar_types,
)


_ROUNDTRIP_TOLERANCES = {
    "uint4b8": (scalar_types.uint4b8, 3.0e-1, 2.0e-1),
    "uint8b128": (scalar_types.uint8b128, 2.0e-2, 2.0e-2),
}
_REPACK_QUANT_TYPES = {
    "uint4b8": scalar_types.uint4b8,
    "uint8": scalar_types.uint8,
    "uint8b128": scalar_types.uint8b128,
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
_REPACK_QUANT_CASES = [
    pytest.param(_REPACK_QUANT_TYPES[name], id=name)
    for name in supported_dense_quant_type_names(_REPACK_QUANT_TYPES)
]


def test_fp8_scalar_type_id_matches_vllm_encoding():
    expected_id = (4 << 0) | (3 << 8) | (1 << 16) | (1 << 49) | (2 << 50)

    assert scalar_types.float8_e4m3fn.id == expected_id
    assert quant_type_name_from_id(scalar_types.float8_e4m3fn.id) == "fp8"


def test_nvfp4_scalar_type_id_matches_vllm_encoding():
    expected_id = (2 << 0) | (1 << 8) | (1 << 16) | (1 << 49)

    assert scalar_types.float4_e2m1f.id == expected_id
    assert quant_type_name_from_id(scalar_types.float4_e2m1f.id) == "nvfp4"


def test_mxfp4_scale_scalar_type_id_matches_vllm_encoding():
    expected_id = (8 << 0) | (1 << 49) | (2 << 50)

    assert scalar_types.float8_e8m0fnu.id == expected_id


@pytest.mark.parametrize("size_n", (64, 128, 256))
def test_fp8_marlin_weight_pack_unpack_preserves_raw_bytes(size_n: int):
    size_k = 16
    raw = torch.arange(size_k * size_n, dtype=torch.uint8).reshape(size_k, size_n)
    fp8_weight = raw.view(torch.float8_e4m3fn)

    packed = fp8_weight_to_marlin_weight(fp8_weight)
    unpacked = marlin_unpack(
        packed,
        size_k,
        size_n,
        scalar_types.float8_e4m3fn,
    )

    assert torch.equal(unpacked.to(torch.uint8), raw)


@pytest.mark.parametrize("size_n", (64, 128, 256))
def test_nvfp4_marlin_weight_pack_unpack_preserves_raw_nibbles(size_n: int):
    size_k = 16
    raw = torch.arange(size_k * size_n, dtype=torch.int32).reshape(size_k, size_n) % 16

    packed = fp4_e2m1_weight_to_marlin_weight(raw)
    unpacked = marlin_unpack(
        packed,
        size_k,
        size_n,
        scalar_types.float4_e2m1f,
    )

    assert torch.equal(unpacked, raw)


@pytest.mark.parametrize(
    ("group_size", "expected_groups"),
    ((-1, 1), (128, 2)),
)
def test_marlin_quantize_fp8_uses_fused_scales_and_dequantizes(
    group_size: int,
    expected_groups: int,
):
    torch.manual_seed(0)
    weight = torch.randn((256, 128), dtype=torch.float16)

    _weight, q_weight, scales, g_idx, sort_indices, rand_perm = marlin_quantize_fp8(
        weight,
        group_size,
    )
    dequantized = marlin_dequantize(
        q_weight,
        scales,
        size_k=weight.shape[0],
        size_n=weight.shape[1],
        group_size=group_size,
        quant_type=scalar_types.float8_e4m3fn,
    )

    assert scales.shape == (expected_groups, weight.shape[1])
    assert scales.dtype == torch.float16
    assert g_idx.numel() == 0
    assert sort_indices.numel() == 0
    assert torch.equal(rand_perm, torch.arange(weight.shape[0], dtype=torch.int))
    assert torch.isfinite(dequantized).all()
    assert (scales > 1.0).all()
    torch.testing.assert_close(dequantized, weight, atol=5.0e-1, rtol=5.0e-1)


def test_marlin_quantize_nvfp4_uses_fp8_scales_and_global_scale():
    torch.manual_seed(0)
    weight = torch.randn((256, 128), dtype=torch.float16)

    weight_ref, q_weight, scales, global_scale, g_idx, sort_indices, rand_perm = (
        marlin_quantize_nvfp4(weight, 16)
    )
    dequantized = marlin_dequantize_nvfp4(
        q_weight,
        scales,
        global_scale,
        size_k=weight.shape[0],
        size_n=weight.shape[1],
        group_size=16,
    )

    assert scales.shape == (weight.shape[0] // 16, weight.shape[1])
    assert scales.dtype == torch.float8_e4m3fn
    assert global_scale.shape == (1,)
    assert global_scale.dtype == torch.float32
    assert g_idx.numel() == 0
    assert sort_indices.numel() == 0
    assert torch.equal(rand_perm, torch.arange(weight.shape[0], dtype=torch.int))
    assert torch.isfinite(weight_ref).all()
    assert torch.isfinite(dequantized).all()
    assert (scales.view(torch.uint8) > 0).all()
    torch.testing.assert_close(dequantized, weight_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(weight_ref, weight, atol=5.0e-1, rtol=5.0e-1)


def test_marlin_quantize_mxfp4_uses_raw_e8m0_scales():
    torch.manual_seed(0)
    weight = torch.randn((256, 128), dtype=torch.float16)

    weight_ref, q_weight, scales, g_idx, sort_indices, rand_perm = marlin_quantize_mxfp4(
        weight,
        32,
    )
    dequantized = marlin_dequantize_mxfp4(
        q_weight,
        scales,
        size_k=weight.shape[0],
        size_n=weight.shape[1],
        group_size=32,
    )

    assert scales.shape == (weight.shape[0] // 32, weight.shape[1])
    assert scales.dtype == torch.float8_e8m0fnu
    assert g_idx.numel() == 0
    assert sort_indices.numel() == 0
    assert torch.equal(rand_perm, torch.arange(weight.shape[0], dtype=torch.int))
    assert torch.isfinite(weight_ref).all()
    assert torch.isfinite(dequantized).all()
    assert (scales.to(torch.float32) > 0.0).all()
    torch.testing.assert_close(dequantized, weight_ref, atol=0.0, rtol=0.0)
    torch.testing.assert_close(weight_ref, weight, atol=7.5e-1, rtol=7.5e-1)


def test_marlin_quantize_mxfp4_rejects_unsupported_metadata():
    weight = torch.randn((256, 128), dtype=torch.float16)

    with pytest.raises(ValueError, match="group_size=32"):
        marlin_quantize_mxfp4(weight, 16)
    with pytest.raises(ValueError, match="does not support act_order"):
        marlin_quantize_mxfp4(weight, 32, act_order=True)


def _require_repack_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    target_capability = source_target_capability()
    capability = torch.cuda.get_device_capability()
    if capability != target_capability:
        pytest.skip(f"Marlin repack requires {source_target_label()} for this source tree")
    try:
        ops._load_dense()
    except Exception as exc:  # pragma: no cover - depends on local build state
        pytest.skip(f"marlin dense extension is not available: {exc}")


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


@pytest.mark.parametrize("quant_type", _REPACK_QUANT_CASES)
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_repack_layout_matches_reference_for_supported_dense_quant_types(
    quant_type,
    repack_impl: str,
):
    _require_repack_cuda()
    assert_repack_layout_matches_reference(repack_impl, quant_type=quant_type)


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_uint4b8_act_order_repack_layout_matches_reference(repack_impl: str):
    _require_repack_cuda()
    assert_repack_layout_matches_reference(
        repack_impl,
        quant_type=scalar_types.uint4b8,
        act_order=True,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_uint4_repack_layout_matches_reference(repack_impl: str):
    _require_repack_cuda()
    assert_repack_layout_matches_reference(
        repack_impl,
        quant_type=scalar_types.uint4,
        act_order=False,
    )


def test_marlin_quantize_rejects_group_size_zero_outside_runtime_act_order_path():
    weight = torch.randn((128, 256), dtype=torch.float16)
    with pytest.raises(ValueError, match="Unsupported dense group_size=0"):
        marlin_quantize(weight, scalar_types.uint4b8, 0, False)


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_marlin_quantize_uint4_packed_zp_round_trip_matches_original_weight(group_size: int):
    torch.manual_seed(0)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _weight, q_weight, scales, zero_points, dequantized = marlin_quantize_uint4_packed_zp(
        weight, group_size
    )

    assert q_weight.shape == (weight.shape[0] // 16, weight.shape[1] * 16 // 8)
    assert scales.shape == (
        1 if group_size == -1 else weight.shape[0] // group_size,
        weight.shape[1],
    )
    assert zero_points.shape == (
        1 if group_size == -1 else weight.shape[0] // group_size,
        weight.shape[1] // 8,
    )
    assert zero_points.dtype == torch.int32
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, weight, atol=4.0e-1, rtol=3.0e-1)


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_marlin_quantize_uint4_zp_round_trip_matches_original_weight(group_size: int):
    torch.manual_seed(0)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _weight, q_weight, scales, zp, dequantized = marlin_quantize_uint4_zp(weight, group_size)

    assert q_weight.shape == (weight.shape[0] // 16, weight.shape[1] * 16 // 8)
    assert scales.shape == (
        1 if group_size == -1 else weight.shape[0] // group_size,
        weight.shape[1],
    )
    assert zp.shape == scales.shape
    assert zp.dtype == torch.float16
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, weight, atol=4.0e-1, rtol=3.0e-1)


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_marlin_quantize_uint8_zp_round_trip_matches_original_weight(group_size: int):
    torch.manual_seed(0)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _weight, q_weight, scales, zp, dequantized = marlin_quantize_uint8_zp(weight, group_size)

    assert q_weight.shape == (weight.shape[0] // 16, weight.shape[1] * 16 // 4)
    assert scales.shape == (
        1 if group_size == -1 else weight.shape[0] // group_size,
        weight.shape[1],
    )
    assert zp.shape == scales.shape
    assert zp.dtype == torch.float16
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, weight, atol=5.0e-2, rtol=5.0e-2)


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_marlin_quantize_experts_uint4_zp_round_trip_matches_original_weights(group_size: int):
    torch.manual_seed(0)
    weights = torch.randn((2, 128, 256), dtype=torch.float16)
    q_weights, scales, zero_points, dequantized, g_idx, perm = (
        marlin_quantize_experts_uint4_zp_with_metadata(weights, group_size)
    )
    expected_groups = 1 if group_size == -1 else weights.shape[1] // group_size

    assert q_weights.shape[0] == weights.shape[0]
    assert scales.shape == (weights.shape[0], expected_groups, weights.shape[2])
    assert zero_points.shape == (weights.shape[0], expected_groups, weights.shape[2] // 8)
    assert dequantized.shape == weights.shape
    assert g_idx.shape == (weights.shape[0], 0)
    assert perm.shape == (weights.shape[0], 0)
    assert zero_points.dtype == torch.int32
    assert torch.isfinite(dequantized).all()
    torch.testing.assert_close(dequantized, weights, atol=4.0e-1, rtol=3.0e-1)


def test_marlin_dequantize_uint4_packed_zp_matches_quantize_helper_output():
    torch.manual_seed(0)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _weight, q_weight, scales, zero_points, dequantized = marlin_quantize_uint4_packed_zp(
        weight, 64
    )
    roundtrip = marlin_dequantize_uint4_packed_zp(
        q_weight,
        scales,
        zero_points,
        size_k=weight.shape[0],
        size_n=weight.shape[1],
        group_size=64,
    )

    assert zero_points.dtype == torch.int32
    torch.testing.assert_close(roundtrip, dequantized, atol=0.0, rtol=0.0)


def test_marlin_dequantize_uint4_zp_matches_quantize_helper_output():
    torch.manual_seed(0)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _weight, q_weight, scales, zp, dequantized = marlin_quantize_uint4_zp(weight, 64)
    roundtrip = marlin_dequantize_uint4_zp(
        q_weight,
        scales,
        zp,
        size_k=weight.shape[0],
        size_n=weight.shape[1],
        group_size=64,
    )

    assert zp.dtype == torch.float16
    torch.testing.assert_close(roundtrip, dequantized, atol=0.0, rtol=0.0)


def test_marlin_dequantize_uint8_zp_matches_quantize_helper_output():
    torch.manual_seed(0)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _weight, q_weight, scales, zp, dequantized = marlin_quantize_uint8_zp(weight, 64)
    roundtrip = marlin_dequantize_uint8_zp(
        q_weight,
        scales,
        zp,
        size_k=weight.shape[0],
        size_n=weight.shape[1],
        group_size=64,
    )

    assert zp.dtype == torch.float16
    torch.testing.assert_close(roundtrip, dequantized, atol=0.0, rtol=0.0)


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


def test_dense_wrapper_rejects_act_order_metadata_before_loading_extension():
    a = torch.randn((4, 128), dtype=torch.float16)
    weight = torch.randn((128, 256), dtype=torch.float16)
    _, q_weight, scales, g_idx, sort_indices, _ = marlin_quantize(
        weight, scalar_types.uint4b8, 64, True
    )
    workspace = torch.zeros((1,), dtype=torch.int32)

    with pytest.raises(ValueError, match="act_order is not supported"):
        dense.run_marlin_gemm(
            a,
            q_weight,
            scales,
            scalar_types.uint4b8.id,
            size_m=a.shape[0],
            size_n=weight.shape[1],
            size_k=weight.shape[0],
            workspace=workspace,
            g_idx=g_idx,
            perm=sort_indices,
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


def test_moe_wrapper_rejects_act_order_metadata_before_loading_extension():
    hidden_states, topk_weights, topk_ids, w1, w2 = make_moe_model_like_inputs(
        tokens=4,
        hidden=128,
        intermediate=128,
        experts=2,
        topk=2,
        device="cpu",
    )
    w1_q, w1_scale, _w1_dequant, g_idx1, sort_indices1 = marlin_quantize_experts_with_metadata(
        w1, scalar_types.uint4b8, 64, True
    )
    w2_q, w2_scale, _w2_dequant, _g_idx2, _sort_indices2 = marlin_quantize_experts_with_metadata(
        w2, scalar_types.uint4b8, 64, False
    )

    with pytest.raises(ValueError, match="act_order is not supported"):
        moe.fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1_q,
            w2=w2_q,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=scalar_types.uint4b8.id,
            g_idx1=g_idx1,
            g_idx2=None,
            sort_indices1=sort_indices1,
            sort_indices2=None,
            is_k_full=True,
        )
