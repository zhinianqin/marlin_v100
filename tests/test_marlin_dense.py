from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marlin_v100.calibration import (
    source_target_capability,
    source_target_label,
    supported_dense_quant_type_names,
)
from marlin_v100 import dense, ops
from tests.helpers import (
    _REPACK_IMPL_CASES,
    assert_repack_layout_matches_reference,
    marlin_dense_reference,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_quantize,
    marlin_quantize_mxfp4,
    marlin_quantize_nvfp4,
    marlin_quantize_uint4_zp,
    marlin_quantize_uint4_zp_bias,
    marlin_quantize_uint8_zp_bias,
    scalar_types,
)

_DENSE_SUPPORTED_QUANT_NAMES = frozenset(
    supported_dense_quant_type_names(
        ("uint4", "uint4b8", "uint8", "uint8b128", "fp8", "nvfp4", "mxfp4")
    )
)
_GROUP_SIZES = (-1, 32, 64, 128)
_CTA_GEOMETRY_CASES = (
    ("32x128x4", 32, 256),
    ("32x256x4", 32, 256),
    ("64x64x4", 64, 256),
    ("64x128x4", 64, 256),
    ("64x128x8", 64, 256),
    ("64x256x4", 64, 256),
    ("64x256x8", 64, 256),
    ("128x64x4", 128, 256),
    ("128x64x8", 128, 256),
    ("128x128x4", 128, 256),
    ("128x128x8", 128, 256),
    ("128x256x8", 128, 256),
    ("256x64x4", 256, 256),
    ("256x64x8", 256, 256),
    ("256x128x8", 256, 256),
)
_FP8_CTA_GEOMETRY_CASES = (
    ("64x128x4", 64, 256),
    ("128x256x8", 128, 256),
    ("256x64x8", 256, 256),
)
_FLOAT16_ACTIVATION_ERROR = (
    rf"{source_target_label()} build only supports float16 activations\."
)
_FLOAT16_DTYPE_ERROR = (
    rf"{source_target_label()} build only supports float16 activations\."
    rf"|{source_target_label()} build only supports float16 outputs\."
    rf"|{source_target_label()} build only supports float16 scales\."
)
_FULL_N_TILE_ERROR = "requires full-N tiles"


def _require_marlin_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    target_capability = source_target_capability()
    capability = torch.cuda.get_device_capability()
    if capability != target_capability:
        pytest.skip(f"Marlin requires {source_target_label()} for this source tree")
    try:
        ops._load_dense()
    except Exception as exc:  # pragma: no cover - depends on local build state
        pytest.skip(f"marlin dense extension is not available: {exc}")


def test_marlin_dense_symbols_available():
    expected = [
        "marlin_gemm",
        "gptq_marlin_repack",
        "awq_marlin_repack",
        "marlin_int4_fp8_preprocess",
        "sm70_cutlass_matmul_probe",
    ]
    for name in expected:
        assert hasattr(ops, name)


def test_sm70_cutlass_matmul_probe_matches_torch_mm():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)

    output = ops.sm70_cutlass_matmul_probe(a, b, 32, 64, 32, 4, 2, 0, 0)
    reference = torch.mm(a, b)

    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, rtol=5e-2, atol=5e-2)


def test_sm70_cutlass_matmul_probe_threadblock_path_matches_torch_mm():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    a = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 256), device="cuda", dtype=torch.float16)

    output = ops.sm70_cutlass_matmul_probe(a, b, 128, 256, 32, 8, 2, 2, 0)
    reference = torch.mm(a, b)

    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize(
    ("cta_m", "cta_n", "warps"),
    [
        (32, 256, 4),
        (256, 128, 8),
        (512, 64, 8),
    ],
)
def test_sm70_cutlass_matmul_probe_extended_threadblock_shapes_match_torch_mm(
    cta_m: int, cta_n: int, warps: int
):
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    a = torch.randn((cta_m, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, cta_n), device="cuda", dtype=torch.float16)

    output = ops.sm70_cutlass_matmul_probe(a, b, cta_m, cta_n, 32, warps, 2, 2, 0)
    reference = torch.mm(a, b)

    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, rtol=5e-2, atol=5e-2)


def test_sm70_cutlass_matmul_probe_rejects_unsupported_threadblock_shape():
    _require_marlin_cuda()
    a = torch.randn((512, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 512), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="unsupported extracted CUTLASS"):
        ops.sm70_cutlass_matmul_probe(a, b, 512, 512, 32, 8, 2, 2, 0)


def test_sm70_cutlass_matmul_probe_rejects_direct_a_path():
    _require_marlin_cuda()
    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="A direct-global path is TODO"):
        ops.sm70_cutlass_matmul_probe(a, b, 32, 64, 64, 4, 2, 1, 0)


def test_sm70_cutlass_matmul_probe_rejects_non_pure_b_path():
    _require_marlin_cuda()
    a = torch.randn((128, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 256), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="unknown B path id"):
        ops.sm70_cutlass_matmul_probe(a, b, 128, 256, 32, 8, 2, 2, 1)


def test_marlin_int4_fp8_preprocess_without_zp():
    _require_marlin_cuda()

    qweight_unpacked = torch.randint(
        0, 16, size=(2048, 2048), dtype=torch.int32, device="cuda"
    )
    qweight_packed = qweight_unpacked[:, ::2] * 16 + qweight_unpacked[:, 1::2]
    qweight_packed = qweight_packed.to(torch.int8).view(torch.int32)

    cuda_res = ops.marlin_int4_fp8_preprocess(qweight_packed, None, True)
    torch_res = torch.where(
        qweight_unpacked >= 8, qweight_unpacked - 8, 15 - qweight_unpacked
    )
    torch_res = torch_res[:, ::2] * 16 + torch_res[:, 1::2]
    torch_res = torch_res.to(torch.int8).view(torch.int32)
    assert torch.equal(cuda_res, torch_res)


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_smoke_local_helpers(repack_impl: str):
    _require_marlin_cuda()
    assert_repack_layout_matches_reference(repack_impl, quant_type=scalar_types.uint4b8)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint4b8, 128, False
    )
    output = ops.marlin_gemm(
        a,
        None,
        q_w,
        None,
        scales,
        None,
        None,
        marlin_make_empty_g_idx(a.device),
        g_idx,
        sort_indices,
        marlin_make_workspace_new(a.device),
        scalar_types.uint4b8.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        True,
        False,
        True,
        False,
    )
    assert output.shape == (a.shape[0], w.shape[1])
    assert torch.isfinite(output).all()


def _run_dense_accuracy_case(
    quant_type,
    *,
    repack_impl: str,
    group_size: int,
    act_order: bool,
    is_k_full: bool,
    rtol: float,
    atol: float,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
) -> None:
    if act_order:
        raise AssertionError("act_order accuracy coverage was replaced by explicit rejection tests")

    _require_marlin_cuda()
    assert_repack_layout_matches_reference(
        repack_impl,
        quant_type=quant_type,
        act_order=act_order,
        group_size=group_size,
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, quant_type, group_size, act_order
    )
    output = ops.marlin_gemm(
        a,
        None,
        q_w,
        None,
        scales,
        None,
        None,
        marlin_make_empty_g_idx(a.device),
        g_idx,
        sort_indices,
        marlin_make_workspace_new(a.device),
        quant_type.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        is_k_full,
        False,
        True,
        False,
    )
    reference = marlin_dense_reference(
        a,
        q_w,
        scales,
        size_k=w.shape[0],
        size_n=w.shape[1],
        group_size=group_size,
        quant_type=quant_type,
        perm=sort_indices,
    ).to(torch.float16)

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _run_fp8_dense_accuracy_case(
    *,
    group_size: int,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
    rtol: float = 4e-2,
    atol: float = 2e-1,
) -> None:
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w,
        scalar_types.float8_e4m3fn,
        group_size,
        False,
    )
    output = ops.marlin_gemm(
        a,
        None,
        q_w,
        None,
        scales,
        None,
        None,
        None,
        g_idx,
        sort_indices,
        marlin_make_workspace_new(a.device),
        scalar_types.float8_e4m3fn.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        True,
        False,
        True,
        False,
    )
    reference = marlin_dense_reference(
        a,
        q_w,
        scales,
        size_k=w.shape[0],
        size_n=w.shape[1],
        group_size=group_size,
        quant_type=scalar_types.float8_e4m3fn,
    ).to(torch.float16)

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _run_nvfp4_dense_accuracy_case(
    *,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
    rtol: float = 5e-2,
    atol: float = 2.5e-1,
) -> None:
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    weight_ref, q_w, scales, global_scale, g_idx, sort_indices, _ = (
        marlin_quantize_nvfp4(w, 16)
    )
    output = dense.run_marlin_gemm(
        a,
        q_w,
        scales,
        scalar_types.float4_e2m1f.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        workspace=marlin_make_workspace_new(a.device),
        global_scale=global_scale,
        g_idx=g_idx,
        perm=sort_indices,
        is_k_full=True,
        use_fp32_reduce=True,
    )
    reference = torch.matmul(
        a.to(torch.float32),
        weight_ref.to(torch.float32),
    ).to(torch.float16)

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _run_mxfp4_dense_accuracy_case(
    *,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
    rtol: float = 5e-2,
    atol: float = 2.5e-1,
) -> None:
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    weight_ref, q_w, scales, g_idx, sort_indices, _ = marlin_quantize_mxfp4(w, 32)
    output = dense.run_marlin_gemm(
        a,
        q_w,
        scales,
        scalar_types.float4_e2m1f.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        workspace=marlin_make_workspace_new(a.device),
        g_idx=g_idx,
        perm=sort_indices,
        is_k_full=True,
        use_fp32_reduce=True,
    )
    reference = torch.matmul(
        a.to(torch.float32),
        weight_ref.to(torch.float32),
    ).to(torch.float16)

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _assert_dense_backend_rejects_act_order(
    quant_type,
    *,
    repack_impl: str | None = None,
    group_size: int,
    is_k_full: bool,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
) -> None:
    _require_marlin_cuda()
    if repack_impl is not None:
        assert_repack_layout_matches_reference(
            repack_impl,
            quant_type=quant_type,
            act_order=True,
            group_size=group_size,
        )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(w, quant_type, group_size, True)

    with pytest.raises(RuntimeError, match="act_order is not supported"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            None,
            g_idx,
            sort_indices,
            marlin_make_workspace_new(a.device),
            quant_type.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            is_k_full,
            False,
            True,
            False,
        )


def _run_dense_uint4_zp_accuracy_case(
    *,
    repack_impl: str,
    group_size: int,
    rtol: float,
    atol: float,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
) -> None:
    _require_marlin_cuda()
    assert_repack_layout_matches_reference(
        repack_impl,
        quant_type=scalar_types.uint4,
        act_order=False,
        group_size=64 if group_size == -1 else group_size,
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    _w, q_w, scales, zp_bias, dequantized = marlin_quantize_uint4_zp_bias(
        w, group_size
    )
    output = ops.marlin_gemm(
        a,
        None,
        q_w,
        None,
        scales,
        None,
        None,
        zp_bias,
        None,
        None,
        marlin_make_workspace_new(a.device),
        scalar_types.uint4.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        True,
        False,
        True,
        True,
    )
    reference = torch.matmul(a.to(torch.float32), dequantized.to(torch.float32)).to(
        torch.float16
    )

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _run_dense_uint8_zp_bias_accuracy_case(
    *,
    repack_impl: str,
    group_size: int,
    rtol: float,
    atol: float,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
) -> None:
    _require_marlin_cuda()
    assert_repack_layout_matches_reference(
        repack_impl,
        quant_type=scalar_types.uint8,
        act_order=False,
        group_size=64 if group_size == -1 else group_size,
    )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    _w, q_w, scales, zp_bias, dequantized = marlin_quantize_uint8_zp_bias(
        w, group_size
    )
    output = ops.marlin_gemm(
        a,
        None,
        q_w,
        None,
        scales,
        None,
        None,
        zp_bias,
        None,
        None,
        marlin_make_workspace_new(a.device),
        scalar_types.uint8.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        True,
        False,
        True,
        True,
    )
    reference = torch.matmul(a.to(torch.float32), dequantized.to(torch.float32)).to(
        torch.float16
    )

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_accuracy(group_size: int, repack_impl: str):
    _run_dense_accuracy_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=group_size,
        act_order=False,
        is_k_full=True,
        rtol=5e-2,
        atol=2.5e-1,
    )


@pytest.mark.parametrize("is_k_full", (True, False))
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_act_order_accuracy(is_k_full: bool, repack_impl: str):
    _assert_dense_backend_rejects_act_order(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=64,
        is_k_full=is_k_full,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_8_row_bucket_matches_reference(repack_impl: str):
    _run_dense_accuracy_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=128,
        act_order=False,
        is_k_full=True,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=8,
        size_k=256,
        size_n=256,
    )


@pytest.mark.parametrize("group_size", (-1, 128))
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference(
    group_size: int,
    repack_impl: str,
):
    _run_dense_accuracy_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=group_size,
        act_order=False,
        is_k_full=True,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=8,
        size_k=128,
        size_n=256,
    )


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_residue_n_rejects_full_tile_contract(
    group_size: int,
    repack_impl: str,
):
    with pytest.raises(RuntimeError, match=_FULL_N_TILE_ERROR):
        _run_dense_accuracy_case(
            scalar_types.uint4b8,
            repack_impl=repack_impl,
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            rtol=5e-2,
            atol=2.5e-1,
            size_m=8,
            size_k=256,
            size_n=128,
        )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_residue_k_single_group_rejects_full_tile_contract(
    repack_impl: str,
):
    with pytest.raises(RuntimeError, match="requires size_k % 32 == 0"):
        _run_dense_accuracy_case(
            scalar_types.uint4b8,
            repack_impl=repack_impl,
            group_size=-1,
            act_order=False,
            is_k_full=True,
            rtol=5e-2,
            atol=2.5e-1,
            size_m=8,
            size_k=144,
            size_n=256,
        )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_residue_k_and_n_single_group_rejects_full_tile_contract(
    repack_impl: str,
):
    with pytest.raises(RuntimeError, match="requires size_k % 32 == 0"):
        _run_dense_accuracy_case(
            scalar_types.uint4b8,
            repack_impl=repack_impl,
            group_size=-1,
            act_order=False,
            is_k_full=True,
            rtol=5e-2,
            atol=2.5e-1,
            size_m=8,
            size_k=144,
            size_n=128,
        )


def test_marlin_dense_uint4b8_residue_k_rejects_multi_group_metadata():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    size_m = 8
    size_k = 144
    size_n = 256
    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    _w, q_w, _scales, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint4b8, -1, False
    )
    scales = torch.ones((3, size_n), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="requires size_k % 32 == 0"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            marlin_make_empty_g_idx(a.device),
            g_idx,
            sort_indices,
            marlin_make_workspace_new(a.device),
            scalar_types.uint4b8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            False,
        )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_sm70_act_order_group_switch_matches_reference(repack_impl: str):
    _assert_dense_backend_rejects_act_order(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=64,
        is_k_full=False,
        size_m=8,
        size_k=128,
        size_n=128,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_size_m_24_uses_32_row_bucket_matches_reference(
    repack_impl: str,
):
    _run_dense_accuracy_case(
        scalar_types.uint4b8,
        repack_impl=repack_impl,
        group_size=128,
        act_order=False,
        is_k_full=True,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=24,
        size_k=256,
        size_n=256,
    )


@pytest.mark.parametrize(("cta_geometry", "size_m", "size_n"), _CTA_GEOMETRY_CASES)
def test_marlin_dense_uint4b8_env_cta_geometry_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    cta_geometry: str,
    size_m: int,
    size_n: int,
):
    monkeypatch.setenv("SM70_MARLIN_U4B8_CTA", cta_geometry)
    _run_dense_accuracy_case(
        scalar_types.uint4b8,
        repack_impl="gptq",
        group_size=128,
        act_order=False,
        is_k_full=True,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=size_m,
        size_k=256,
        size_n=size_n,
    )


def test_marlin_dense_uint4b8_env_cta_geometry_rejects_unsupported(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SM70_MARLIN_U4B8_CTA", "32x64x4")
    with pytest.raises(RuntimeError, match="Unsupported SM70_MARLIN_U4B8_CTA"):
        _run_dense_accuracy_case(
            scalar_types.uint4b8,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            is_k_full=True,
            rtol=5e-2,
            atol=2.5e-1,
            size_m=32,
            size_k=256,
            size_n=64,
        )


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4_zp_accuracy(group_size: int, repack_impl: str):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl=repack_impl,
        group_size=group_size,
        rtol=5e-2,
        atol=2.5e-1,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4_zp_8_row_bucket_matches_reference(repack_impl: str):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl=repack_impl,
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=8,
        size_k=256,
        size_n=256,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4_zp_size_m_24_matches_reference(repack_impl: str):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl=repack_impl,
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=24,
        size_k=256,
        size_n=256,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4_zp_small_tile_matches_reference(repack_impl: str):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl=repack_impl,
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=8,
        size_k=128,
        size_n=256,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4_zp_residue_n_rejects_full_tile_contract(
    repack_impl: str,
):
    with pytest.raises(RuntimeError, match=_FULL_N_TILE_ERROR):
        _run_dense_uint4_zp_accuracy_case(
            repack_impl=repack_impl,
            group_size=128,
            rtol=5e-2,
            atol=2.5e-1,
            size_m=8,
            size_k=256,
            size_n=128,
        )


@pytest.mark.parametrize(("cta_geometry", "size_m", "size_n"), _CTA_GEOMETRY_CASES)
def test_marlin_dense_uint4_zp_env_cta_geometry_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    cta_geometry: str,
    size_m: int,
    size_n: int,
):
    monkeypatch.setenv("SM70_MARLIN_U4_CTA", cta_geometry)
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=size_m,
        size_k=256,
        size_n=size_n,
    )


def test_marlin_dense_uint4_zp_env_cta_geometry_rejects_unsupported(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SM70_MARLIN_U4_CTA", "32x64x4")
    with pytest.raises(RuntimeError, match="Unsupported SM70_MARLIN_U4_CTA"):
        _run_dense_uint4_zp_accuracy_case(
            repack_impl="gptq",
            group_size=128,
            rtol=5e-2,
            atol=2.5e-1,
            size_m=32,
            size_k=256,
            size_n=64,
        )


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint8_zp_bias_accuracy(group_size: int, repack_impl: str):
    _run_dense_uint8_zp_bias_accuracy_case(
        repack_impl=repack_impl,
        group_size=group_size,
        rtol=5e-2,
        atol=2.5e-1,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint8_zp_bias_small_tile_matches_reference(repack_impl: str):
    _run_dense_uint8_zp_bias_accuracy_case(
        repack_impl=repack_impl,
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=8,
        size_k=128,
        size_n=256,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint8_zp_bias_residue_n_rejects_full_tile_contract(
    repack_impl: str,
):
    with pytest.raises(RuntimeError, match=_FULL_N_TILE_ERROR):
        _run_dense_uint8_zp_bias_accuracy_case(
            repack_impl=repack_impl,
            group_size=128,
            rtol=5e-2,
            atol=2.5e-1,
            size_m=8,
            size_k=256,
            size_n=128,
        )


@pytest.mark.parametrize(("cta_geometry", "size_m", "size_n"), _CTA_GEOMETRY_CASES)
def test_marlin_dense_uint8_zp_bias_env_cta_geometry_matches_reference(
    monkeypatch: pytest.MonkeyPatch,
    cta_geometry: str,
    size_m: int,
    size_n: int,
):
    monkeypatch.setenv("SM70_MARLIN_U8_CTA", cta_geometry)
    _run_dense_uint8_zp_bias_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=size_m,
        size_k=256,
        size_n=size_n,
    )


def test_marlin_dense_uint8_zp_bias_env_cta_geometry_rejects_unsupported(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SM70_MARLIN_U8_CTA", "32x64x4")
    with pytest.raises(RuntimeError, match="Unsupported SM70_MARLIN_U8_CTA"):
        _run_dense_uint8_zp_bias_accuracy_case(
            repack_impl="gptq",
            group_size=128,
            rtol=5e-2,
            atol=2.5e-1,
            size_m=32,
            size_k=256,
            size_n=64,
        )


def test_marlin_dense_uint4_zp_requires_bias():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, _zp_bias, _dequantized = marlin_quantize_uint4_zp_bias(w, 128)

    with pytest.raises(RuntimeError, match="requires fp16 precomputed zero-point bias"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            None,
            None,
            None,
            marlin_make_workspace_new(a.device),
            scalar_types.uint4.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            False,
        )


def test_marlin_dense_uint8_zp_bias_requires_bias():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, _zp_bias, _dequantized = marlin_quantize_uint8_zp_bias(w, 128)

    with pytest.raises(RuntimeError, match="requires fp16 precomputed zero-point bias"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            None,
            None,
            None,
            marlin_make_workspace_new(a.device),
            scalar_types.uint8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            False,
        )


def test_marlin_dense_uint4_zp_rejects_packed_zero_points():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, packed_zero_points, _dequantized = marlin_quantize_uint4_zp(w, 128)

    with pytest.raises(RuntimeError, match="fp16 precomputed zero-point bias"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            packed_zero_points,
            None,
            None,
            marlin_make_workspace_new(a.device),
            scalar_types.uint4.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            True,
        )


def test_marlin_dense_uint8_zp_bias_rejects_packed_zero_points():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, _zp_bias, _dequantized = marlin_quantize_uint8_zp_bias(w, 128)
    packed_zero_points = torch.zeros(
        (scales.shape[0], w.shape[1] // 4),
        device="cuda",
        dtype=torch.int32,
    )

    with pytest.raises(RuntimeError, match="fp16 precomputed zero-point bias"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            packed_zero_points,
            None,
            None,
            marlin_make_workspace_new(a.device),
            scalar_types.uint8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            True,
        )


def test_marlin_dense_uint4b8_rejects_zp_bias_metadata():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint4b8, 128, False
    )
    zp_bias = torch.zeros_like(scales)

    with pytest.raises(RuntimeError, match="zero-point bias metadata"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp_bias,
            g_idx,
            sort_indices,
            marlin_make_workspace_new(a.device),
            scalar_types.uint4b8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            True,
        )


def test_marlin_dense_uint8_zp_bias_rejects_bias_without_flag():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, zp_bias, _dequantized = marlin_quantize_uint8_zp_bias(w, 128)

    with pytest.raises(RuntimeError, match="use_zp_bias is false"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp_bias,
            None,
            None,
            marlin_make_workspace_new(a.device),
            scalar_types.uint8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            False,
        )


def test_marlin_dense_uint8b128_rejects_zp_bias_metadata():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint8b128, 128, False
    )
    zp_bias = torch.zeros_like(scales)

    with pytest.raises(RuntimeError, match="zero-point bias metadata"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp_bias,
            g_idx,
            sort_indices,
            marlin_make_workspace_new(a.device),
            scalar_types.uint8b128.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            True,
        )


def test_marlin_dense_uint8b128_rejects_use_zp_bias_without_metadata():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint8b128, 128, False
    )

    with pytest.raises(RuntimeError, match="use_zp_bias is true"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            None,
            g_idx,
            sort_indices,
            marlin_make_workspace_new(a.device),
            scalar_types.uint8b128.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            True,
        )


def test_marlin_dense_uint4_zp_rejects_bias_without_flag():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, zp_bias, _dequantized = marlin_quantize_uint4_zp_bias(w, 128)

    with pytest.raises(RuntimeError, match="use_zp_bias is false"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp_bias,
            None,
            None,
            marlin_make_workspace_new(a.device),
            scalar_types.uint4.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            False,
        )


def test_marlin_dense_uint4_zp_rejects_act_order():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    size_k = 256
    a = torch.randn((16, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, zp_bias, _dequantized = marlin_quantize_uint4_zp_bias(w, 64)
    g_idx = (torch.arange(size_k, device=a.device, dtype=torch.int32) // 64).contiguous()
    perm = torch.arange(size_k, device=a.device, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="act_order is not supported"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp_bias,
            g_idx,
            perm,
            marlin_make_workspace_new(a.device),
            scalar_types.uint4.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            True,
        )


if "uint8b128" in _DENSE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("group_size", _GROUP_SIZES)
    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_marlin_dense_uint8b128_accuracy(group_size: int, repack_impl: str):
        _run_dense_accuracy_case(
            scalar_types.uint8b128,
            repack_impl=repack_impl,
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            rtol=4e-2,
            atol=2e-1,
        )

    @pytest.mark.parametrize("is_k_full", (True, False))
    def test_marlin_dense_uint8b128_act_order_accuracy(is_k_full: bool):
        _assert_dense_backend_rejects_act_order(
            scalar_types.uint8b128,
            group_size=64,
            is_k_full=is_k_full,
        )

    def test_marlin_dense_uint8b128_act_order_single_group_small_k_matches_reference():
        _assert_dense_backend_rejects_act_order(
            scalar_types.uint8b128,
            group_size=128,
            is_k_full=False,
            size_m=1,
            size_k=128,
            size_n=256,
        )

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_marlin_dense_uint8b128_8_row_bucket_matches_reference(repack_impl: str):
        _run_dense_accuracy_case(
            scalar_types.uint8b128,
            repack_impl=repack_impl,
            group_size=128,
            act_order=False,
            is_k_full=True,
            rtol=4e-2,
            atol=2e-1,
            size_m=8,
            size_k=256,
            size_n=256,
        )

    @pytest.mark.parametrize("group_size", (-1, 128))
    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_marlin_dense_uint8b128_sm70_scale_zp_math_consistency_matches_reference(
        group_size: int,
        repack_impl: str,
    ):
        _run_dense_accuracy_case(
            scalar_types.uint8b128,
            repack_impl=repack_impl,
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            rtol=4e-2,
            atol=2e-1,
            size_m=8,
            size_k=128,
            size_n=256,
        )

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_marlin_dense_uint8b128_residue_n_rejects_full_tile_contract(
        repack_impl: str,
    ):
        with pytest.raises(RuntimeError, match=_FULL_N_TILE_ERROR):
            _run_dense_accuracy_case(
                scalar_types.uint8b128,
                repack_impl=repack_impl,
                group_size=128,
                act_order=False,
                is_k_full=True,
                rtol=4e-2,
                atol=2e-1,
                size_m=8,
                size_k=256,
                size_n=128,
            )

    @pytest.mark.parametrize(("cta_geometry", "size_m", "size_n"), _CTA_GEOMETRY_CASES)
    def test_marlin_dense_uint8b128_env_cta_geometry_matches_reference(
        monkeypatch: pytest.MonkeyPatch,
        cta_geometry: str,
        size_m: int,
        size_n: int,
    ):
        monkeypatch.setenv("SM70_MARLIN_U8B128_CTA", cta_geometry)
        _run_dense_accuracy_case(
            scalar_types.uint8b128,
            repack_impl="gptq",
            group_size=128,
            act_order=False,
            is_k_full=True,
            rtol=4e-2,
            atol=2e-1,
            size_m=size_m,
            size_k=256,
            size_n=size_n,
        )

    def test_marlin_dense_uint8b128_env_cta_geometry_rejects_unsupported(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("SM70_MARLIN_U8B128_CTA", "32x64x4")
        with pytest.raises(RuntimeError, match="Unsupported SM70_MARLIN_U8B128_CTA"):
            _run_dense_accuracy_case(
                scalar_types.uint8b128,
                repack_impl="gptq",
                group_size=128,
                act_order=False,
                is_k_full=True,
                rtol=4e-2,
                atol=2e-1,
                size_m=32,
                size_k=256,
                size_n=64,
            )


if "fp8" in _DENSE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("group_size", (-1, 128))
    def test_marlin_dense_fp8_weight_accuracy(group_size: int):
        _run_fp8_dense_accuracy_case(group_size=group_size)

    def test_marlin_dense_fp8_weight_residue_n_rejects_full_tile_contract():
        with pytest.raises(RuntimeError, match=_FULL_N_TILE_ERROR):
            _run_fp8_dense_accuracy_case(
                group_size=128,
                size_m=8,
                size_k=256,
                size_n=128,
            )

    @pytest.mark.parametrize(("cta_geometry", "size_m", "size_n"), _FP8_CTA_GEOMETRY_CASES)
    def test_marlin_dense_fp8_env_cta_geometry_matches_reference(
        monkeypatch: pytest.MonkeyPatch,
        cta_geometry: str,
        size_m: int,
        size_n: int,
    ):
        monkeypatch.setenv("SM70_MARLIN_FP8_CTA", cta_geometry)
        _run_fp8_dense_accuracy_case(
            group_size=128,
            size_m=size_m,
            size_k=256,
            size_n=size_n,
        )

    def test_marlin_dense_fp8_env_cta_geometry_rejects_unsupported(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("SM70_MARLIN_FP8_CTA", "32x64x4")
        with pytest.raises(RuntimeError, match="Unsupported SM70_MARLIN_FP8_CTA"):
            _run_fp8_dense_accuracy_case(
                group_size=128,
                size_m=32,
                size_k=256,
                size_n=64,
            )

    @pytest.mark.parametrize("bad_group_size", (32, 64))
    def test_marlin_dense_fp8_rejects_unsupported_group_size(bad_group_size: int):
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
            w,
            scalar_types.float8_e4m3fn,
            128,
            False,
        )
        bad_num_groups = w.shape[0] // bad_group_size
        bad_scales = scales.repeat_interleave(bad_num_groups // scales.shape[0], dim=0)

        with pytest.raises(RuntimeError, match="supports only group_size -1 or 128"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                bad_scales,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                marlin_make_workspace_new(a.device),
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

    def test_marlin_dense_fp8_rejects_unsupported_k_and_n_shapes():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        bad_k = 144
        size_n = 128
        a_bad_k = torch.randn((8, bad_k), device="cuda", dtype=torch.float16)
        w_bad_k = torch.randn((bad_k, size_n), device="cuda", dtype=torch.float16)
        _, q_w_bad_k, scales_bad_k, g_idx, sort_indices, _ = marlin_quantize(
            w_bad_k,
            scalar_types.float8_e4m3fn,
            -1,
            False,
        )
        with pytest.raises(RuntimeError, match="requires size_k % 32 == 0"):
            ops.marlin_gemm(
                a_bad_k,
                None,
                q_w_bad_k,
                None,
                scales_bad_k,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                marlin_make_workspace_new(a_bad_k.device),
                scalar_types.float8_e4m3fn.id,
                a_bad_k.shape[0],
                size_n,
                bad_k,
                True,
                False,
                True,
                False,
            )

        size_k = 256
        bad_n = 96
        a_bad_n = torch.randn((8, size_k), device="cuda", dtype=torch.float16)
        q_w_bad_n = torch.empty((size_k // 16, bad_n * 16 // 4), device="cuda", dtype=torch.int32)
        scales_bad_n = torch.ones((2, bad_n), device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="requires size_n % 64 == 0"):
            ops.marlin_gemm(
                a_bad_n,
                None,
                q_w_bad_n,
                None,
                scales_bad_n,
                None,
                None,
                None,
                marlin_make_empty_g_idx(a_bad_n.device),
                torch.empty(0, device="cuda", dtype=torch.int),
                marlin_make_workspace_new(a_bad_n.device),
                scalar_types.float8_e4m3fn.id,
                a_bad_n.shape[0],
                bad_n,
                size_k,
                True,
                False,
                True,
                False,
            )

    def test_marlin_dense_fp8_rejects_unsupported_dtypes_and_metadata():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
            w,
            scalar_types.float8_e4m3fn,
            128,
            False,
        )
        workspace = marlin_make_workspace_new(a.device)
        common_args = (
            q_w,
            None,
            scales,
            None,
            None,
            None,
            g_idx,
            sort_indices,
            workspace,
            scalar_types.float8_e4m3fn.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            False,
        )

        with pytest.raises(RuntimeError, match=_FLOAT16_ACTIVATION_ERROR):
            ops.marlin_gemm(a.to(torch.bfloat16), None, *common_args)

        with pytest.raises(RuntimeError, match="SM70 build only supports float16 activations"):
            ops.marlin_gemm(a.to(torch.float8_e4m3fn), None, *common_args)

        with pytest.raises(RuntimeError, match="SM70 build only supports float16 scales"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales.to(torch.bfloat16),
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        c_bf16 = torch.empty((a.shape[0], w.shape[1]), device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="SM70 build only supports float16 outputs"):
            ops.marlin_gemm(a, c_bf16, *common_args)

        with pytest.raises(RuntimeError, match="act_order is not supported"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                torch.zeros(w.shape[0], device="cuda", dtype=torch.int),
                torch.arange(w.shape[0], device="cuda", dtype=torch.int),
                workspace,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="does not support bias"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                torch.zeros(w.shape[1], device="cuda", dtype=torch.float16),
                scales,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="supports global_scale only for preconverted nvfp4"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                torch.ones(1, device="cuda", dtype=torch.float16),
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        zp_bias = torch.zeros_like(scales)
        with pytest.raises(RuntimeError, match="zero-point bias metadata"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                zp_bias,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                True,
            )

        with pytest.raises(RuntimeError, match="use_zp_bias is true"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                True,
            )

        with pytest.raises(RuntimeError, match="requires full-K"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                False,
                False,
                True,
                False,
            )


if "nvfp4" in _DENSE_SUPPORTED_QUANT_NAMES:

    def test_marlin_dense_nvfp4_weight_accuracy():
        _run_nvfp4_dense_accuracy_case()

    def test_marlin_dense_nvfp4_weight_residue_n_rejects_full_tile_contract():
        with pytest.raises(RuntimeError, match=_FULL_N_TILE_ERROR):
            _run_nvfp4_dense_accuracy_case(
                size_m=8,
                size_k=256,
                size_n=128,
            )

    @pytest.mark.parametrize(("cta_geometry", "size_m", "size_n"), _FP8_CTA_GEOMETRY_CASES)
    def test_marlin_dense_nvfp4_env_cta_geometry_matches_reference(
        monkeypatch: pytest.MonkeyPatch,
        cta_geometry: str,
        size_m: int,
        size_n: int,
    ):
        monkeypatch.setenv("SM70_MARLIN_NVFP4_CTA", cta_geometry)
        _run_nvfp4_dense_accuracy_case(
            size_m=size_m,
            size_k=256,
            size_n=size_n,
        )

    def test_marlin_dense_nvfp4_env_cta_geometry_rejects_unsupported(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("SM70_MARLIN_NVFP4_CTA", "32x64x4")
        with pytest.raises(RuntimeError, match="Unsupported SM70_MARLIN_NVFP4_CTA"):
            _run_nvfp4_dense_accuracy_case(
                size_m=32,
                size_k=256,
                size_n=64,
            )

    def test_marlin_dense_nvfp4_rejects_raw_fp8_scales_and_missing_global_scale():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        _weight_ref, q_w, scales, global_scale, g_idx, sort_indices, _ = (
            marlin_quantize_nvfp4(w, 16)
        )
        raw_fp8_scales = (scales.to(torch.float32) / 128.0).to(torch.float8_e4m3fn)
        workspace = marlin_make_workspace_new(a.device)

        with pytest.raises(RuntimeError, match="preconverted float16 NVFP4/MXFP4 scales"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                raw_fp8_scales,
                None,
                global_scale,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="mxfp4 prototype supports only group_size 32"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="expects fp32 global_scale"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                global_scale.to(torch.float16),
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

    def test_marlin_dense_nvfp4_rejects_unsupported_group_size_and_metadata():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        _weight_ref, q_w, scales, global_scale, g_idx, sort_indices, _ = (
            marlin_quantize_nvfp4(w, 16)
        )
        workspace = marlin_make_workspace_new(a.device)

        with pytest.raises(RuntimeError, match="supports only group_size 16"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales[:8].contiguous(),
                None,
                global_scale,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="act_order is not supported"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                global_scale,
                None,
                torch.zeros(w.shape[0], device="cuda", dtype=torch.int),
                torch.arange(w.shape[0], device="cuda", dtype=torch.int),
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="does not support bias"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                torch.zeros(w.shape[1], device="cuda", dtype=torch.float16),
                scales,
                None,
                global_scale,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="zero-point bias metadata"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                global_scale,
                torch.zeros_like(scales),
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                True,
            )

        with pytest.raises(RuntimeError, match="requires full-K"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                global_scale,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                False,
                False,
                True,
                False,
            )


if "mxfp4" in _DENSE_SUPPORTED_QUANT_NAMES:

    def test_marlin_dense_mxfp4_weight_accuracy():
        _run_mxfp4_dense_accuracy_case()

    def test_marlin_dense_mxfp4_weight_residue_n_rejects_full_tile_contract():
        with pytest.raises(RuntimeError, match=_FULL_N_TILE_ERROR):
            _run_mxfp4_dense_accuracy_case(
                size_m=8,
                size_k=256,
                size_n=128,
            )

    @pytest.mark.parametrize(("cta_geometry", "size_m", "size_n"), _FP8_CTA_GEOMETRY_CASES)
    def test_marlin_dense_mxfp4_env_cta_geometry_matches_reference(
        monkeypatch: pytest.MonkeyPatch,
        cta_geometry: str,
        size_m: int,
        size_n: int,
    ):
        monkeypatch.setenv("SM70_MARLIN_MXFP4_CTA", cta_geometry)
        _run_mxfp4_dense_accuracy_case(
            size_m=size_m,
            size_k=256,
            size_n=size_n,
        )

    def test_marlin_dense_mxfp4_env_cta_geometry_rejects_unsupported(
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setenv("SM70_MARLIN_MXFP4_CTA", "32x64x4")
        with pytest.raises(RuntimeError, match="Unsupported SM70_MARLIN_MXFP4_CTA"):
            _run_mxfp4_dense_accuracy_case(
                size_m=32,
                size_k=256,
                size_n=64,
            )

    def test_marlin_dense_mxfp4_rejects_raw_e8m0_scales_and_wrong_routing():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        _weight_ref, q_w, scales, g_idx, sort_indices, _ = marlin_quantize_mxfp4(
            w,
            32,
        )
        workspace = marlin_make_workspace_new(a.device)

        with pytest.raises(RuntimeError, match="preconverted float16 NVFP4/MXFP4 scales"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales.to(torch.float8_e8m0fnu),
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="nvfp4 prototype supports only group_size 16"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                torch.ones(1, device="cuda", dtype=torch.float32),
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        _nv_weight_ref, nv_q_w, nv_scales, _global_scale, nv_g_idx, nv_sort_indices, _ = (
            marlin_quantize_nvfp4(w, 16)
        )
        with pytest.raises(RuntimeError, match="mxfp4 prototype supports only group_size 32"):
            ops.marlin_gemm(
                a,
                None,
                nv_q_w,
                None,
                nv_scales,
                None,
                None,
                None,
                nv_g_idx,
                nv_sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

    def test_marlin_dense_mxfp4_rejects_unsupported_dtypes_and_metadata():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        _weight_ref, q_w, scales, g_idx, sort_indices, _ = marlin_quantize_mxfp4(
            w,
            32,
        )
        workspace = marlin_make_workspace_new(a.device)
        common_args = (
            q_w,
            None,
            scales,
            None,
            None,
            None,
            g_idx,
            sort_indices,
            workspace,
            scalar_types.float4_e2m1f.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            False,
        )

        with pytest.raises(RuntimeError, match=_FLOAT16_ACTIVATION_ERROR):
            ops.marlin_gemm(a.to(torch.bfloat16), None, *common_args)

        c_bf16 = torch.empty((a.shape[0], w.shape[1]), device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="SM70 build only supports float16 outputs"):
            ops.marlin_gemm(a, c_bf16, *common_args)

        with pytest.raises(RuntimeError, match="SM70 build only supports float16 scales"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales.to(torch.bfloat16),
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="act_order is not supported"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                torch.zeros(w.shape[0], device="cuda", dtype=torch.int),
                torch.arange(w.shape[0], device="cuda", dtype=torch.int),
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="does not support bias"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                torch.zeros(w.shape[1], device="cuda", dtype=torch.float16),
                scales,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        with pytest.raises(RuntimeError, match="zero-point bias metadata"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                torch.zeros_like(scales),
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                True,
            )

        with pytest.raises(RuntimeError, match="requires full-K"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                False,
                False,
                True,
                False,
            )

    def test_marlin_dense_mxfp4_rejects_unsupported_k_and_n_shapes():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        bad_k = 144
        size_n = 128
        a_bad_k = torch.randn((8, bad_k), device="cuda", dtype=torch.float16)
        q_w_bad_k = torch.empty(
            (bad_k // 16, size_n * 16 // 8),
            device="cuda",
            dtype=torch.int32,
        )
        scales_bad_k = torch.ones((1, size_n), device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="requires size_k % 32 == 0"):
            ops.marlin_gemm(
                a_bad_k,
                None,
                q_w_bad_k,
                None,
                scales_bad_k,
                None,
                None,
                None,
                marlin_make_empty_g_idx(a_bad_k.device),
                torch.empty(0, device="cuda", dtype=torch.int),
                marlin_make_workspace_new(a_bad_k.device),
                scalar_types.float4_e2m1f.id,
                a_bad_k.shape[0],
                size_n,
                bad_k,
                True,
                False,
                True,
                False,
            )

        size_k = 256
        bad_n = 96
        a_bad_n = torch.randn((8, size_k), device="cuda", dtype=torch.float16)
        q_w_bad_n = torch.empty((size_k // 16, bad_n * 16 // 8), device="cuda", dtype=torch.int32)
        scales_bad_n = torch.ones((size_k // 32, bad_n), device="cuda", dtype=torch.float16)
        with pytest.raises(RuntimeError, match="requires size_n % 64 == 0"):
            ops.marlin_gemm(
                a_bad_n,
                None,
                q_w_bad_n,
                None,
                scales_bad_n,
                None,
                None,
                None,
                marlin_make_empty_g_idx(a_bad_n.device),
                torch.empty(0, device="cuda", dtype=torch.int),
                marlin_make_workspace_new(a_bad_n.device),
                scalar_types.float4_e2m1f.id,
                a_bad_n.shape[0],
                bad_n,
                size_k,
                True,
                False,
                True,
                False,
            )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_rejects_mismatched_capability_or_unsupported_dtypes(repack_impl: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    try:
        ops._load_dense()
    except Exception as exc:  # pragma: no cover - depends on local build state
        pytest.skip(f"marlin dense extension is not available: {exc}")

    device = torch.device("cuda")
    a = torch.randn((16, 256), device=device, dtype=torch.float16)
    w = torch.randn((256, 256), device=device, dtype=torch.float16)
    _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint4b8, 128, False
    )
    workspace = marlin_make_workspace_new(device)

    target_capability = source_target_capability()
    capability = torch.cuda.get_device_capability(device)
    if capability != target_capability:
        with pytest.raises(RuntimeError, match=source_target_label()):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.uint4b8.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )
        return

    assert_repack_layout_matches_reference(repack_impl, quant_type=scalar_types.uint4b8)
    a_bf16 = a.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match=_FLOAT16_ACTIVATION_ERROR):
        ops.marlin_gemm(
            a_bf16,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            None,
            g_idx,
            sort_indices,
            workspace,
            scalar_types.uint4b8.id,
            a_bf16.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            True,
            False,
        )


if "uint8b128" in _DENSE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
    def test_marlin_dense_uint8b128_rejects_unsupported_dtypes(repack_impl: str):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required")

        try:
            ops._load_dense()
        except Exception as exc:  # pragma: no cover - depends on local build state
            pytest.skip(f"marlin dense extension is not available: {exc}")

        device = torch.device("cuda")
        a = torch.randn((16, 256), device=device, dtype=torch.float16)
        w = torch.randn((256, 256), device=device, dtype=torch.float16)
        _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
            w, scalar_types.uint8b128, 128, False
        )
        workspace = marlin_make_workspace_new(device)

        target_capability = source_target_capability()
        capability = torch.cuda.get_device_capability(device)
        if capability != target_capability:
            with pytest.raises(RuntimeError, match=source_target_label()):
                ops.marlin_gemm(
                    a,
                    None,
                    q_w,
                    None,
                    scales,
                    None,
                    None,
                    None,
                    g_idx,
                    sort_indices,
                    workspace,
                    scalar_types.uint8b128.id,
                    a.shape[0],
                    w.shape[1],
                    w.shape[0],
                    True,
                    False,
                    True,
                    False,
                )
            return

        assert_repack_layout_matches_reference(repack_impl, quant_type=scalar_types.uint8b128)
        a_bf16 = a.to(torch.bfloat16)
        with pytest.raises(RuntimeError, match=_FLOAT16_ACTIVATION_ERROR):
            ops.marlin_gemm(
                a_bf16,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.uint8b128.id,
                a_bf16.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        a_bf16 = a.to(torch.bfloat16)
        scales_bf16 = scales.to(torch.bfloat16)
        with pytest.raises(
            RuntimeError,
            match=_FLOAT16_DTYPE_ERROR,
        ):
            ops.marlin_gemm(
                a_bf16,
                None,
                q_w,
                None,
                scales_bf16,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.uint8b128.id,
                a_bf16.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )

        a_bf16 = a.to(torch.bfloat16)
        scales_bf16 = scales.to(torch.bfloat16)
        with pytest.raises(
            RuntimeError,
            match=_FLOAT16_DTYPE_ERROR,
        ):
            ops.marlin_gemm(
                a_bf16,
                None,
                q_w,
                None,
                scales_bf16,
                None,
                None,
                None,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.uint4b8.id,
                a_bf16.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                True,
                False,
            )
