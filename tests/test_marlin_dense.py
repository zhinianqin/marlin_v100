from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from tests.calibration import (
    source_target_capability,
    source_target_label,
    supported_dense_quant_type_names,
)
from tests import ops
from tests.helpers import (
    _REPACK_IMPL_CASES,
    assert_repack_layout_matches_reference,
    marlin_dense_reference,
    marlin_make_workspace,
    marlin_make_empty_g_idx,
    marlin_permute_bias,
    marlin_quantize,
    marlin_quantize_mxfp4,
    marlin_quantize_nvfp4,
    marlin_quantize_uint4_zp,
    marlin_quantize_uint4_packed_zp,
    marlin_quantize_uint8_zp,
    run_marlin_gemm,
    run_marlin_linear_kernel_case,
    scalar_types,
)
from tests.sm70_env_sweep import (
    EXPLICIT_ENV_REJECTION_RE,
    SM70_GEOMETRIES,
    SM70_METADATA_CACHE_VALUES,
    SM70_SPLIT_K_VALUES,
    DenseDirectOpKey,
    dense_env,
    dense_env_combo_is_legal,
    exhaustive_enabled,
    exhaustive_index_is_past_limit,
    exhaustive_index_is_selected,
    iter_dense_direct_op_keys,
    iter_dense_focused_mnk_direct_op_keys,
    iter_env_combinations,
    set_dense_env,
)

_DENSE_SUPPORTED_QUANT_NAMES = frozenset(
    supported_dense_quant_type_names(
        ("uint4", "uint4b8", "uint8", "uint8b128", "fp8", "nvfp4", "mxfp4")
    )
)
_GROUP_SIZES = (-1, 32, 64, 128)
_SM70_CUTE_NATIVE_CASES = tuple(
    (cta_m, cta_n, warps)
    for cta_m in (8, 16, 32, 48, 64)
    for cta_n in (64, 128, 256)
    for warps in (4, 8)
)
_FLOAT16_ACTIVATION_ERROR = (
    rf"{source_target_label()} build only supports float16 activations\."
)
_FLOAT16_DTYPE_ERROR = (
    rf"{source_target_label()} build only supports float16 activations\."
    rf"|{source_target_label()} build only supports float16 outputs\."
    rf"|{source_target_label()} build only supports float16 scales\."
)
_N_TILE_ALIGNMENT_ERROR = "requires size_n divisible by 64"
_K_TILE_ALIGNMENT_ERROR = (
    "requires size_k % 32 == 0|requires K divisible by CTA_K="
)
_SPLIT_K_QUANT_CASES = (
    ("uint4b8", 128, 4096, 5e-2, 5e-1),
    ("uint8", 32, 4096, 5e-2, 5e-1),
    ("uint8b128", 128, 4096, 4e-2, 4e-1),
    ("fp8", 128, 4096, 4e-2, 4e-1),
    ("nvfp4", 16, 4096, 5e-2, 5e-1),
    ("mxfp4", 32, 4096, 5e-2, 5e-1),
)
_AUTO_CTA_N_QUANT_CASES = (
    ("uint4b8", 128, 5e-2, 2.5e-1),
    ("uint4", 128, 5e-2, 2.5e-1),
    ("uint8", 128, 5e-2, 2.5e-1),
    ("uint8b128", 128, 4e-2, 2e-1),
    ("fp8", 128, 4e-2, 2e-1),
    ("nvfp4", 16, 5e-2, 2.5e-1),
    ("mxfp4", 32, 5e-2, 2.5e-1),
)
_DENSE_ENV_SWEEP_TOLERANCES = {
    "uint4": (5e-2, 3.5e-1),
    "uint4b8": (5e-2, 3.5e-1),
    "uint8": (5e-2, 3.5e-1),
    "uint8b128": (4e-2, 3.5e-1),
    "fp8": (4e-2, 4e-1),
    "nvfp4": (5e-2, 5e-1),
    "mxfp4": (5e-2, 5e-1),
}


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
        "sm70_cutlass_matmul_explicit_warp_probe",
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


@pytest.mark.parametrize(("cta_m", "cta_n", "warps"), _SM70_CUTE_NATIVE_CASES)
def test_sm70_cutlass_matmul_probe_cute_native_shapes_match_torch_mm(
    cta_m: int, cta_n: int, warps: int
):
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    m = cta_m * 2
    n = cta_n
    k = 64
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    output = ops.sm70_cutlass_matmul_probe(a, b, cta_m, cta_n, 32, warps, 2, 0, 0)
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
    ("cta_m", "cta_n", "cta_k", "warps", "warp_m", "warp_n", "warp_k"),
    [
        (64, 64, 32, 4, 32, 32, 32),
    ],
)
def test_sm70_cutlass_explicit_warp_probe_matches_torch_mm(
    cta_m: int,
    cta_n: int,
    cta_k: int,
    warps: int,
    warp_m: int,
    warp_n: int,
    warp_k: int,
):
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    m = cta_m * 2
    n = cta_n * 2
    k = cta_k * 4
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    output = ops.sm70_cutlass_matmul_explicit_warp_probe(
        a, b, cta_m, cta_n, cta_k, warps, warp_m, warp_n, warp_k
    )
    reference = torch.mm(a, b)

    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, rtol=5e-2, atol=5e-2)


def test_sm70_cutlass_explicit_warp_probe_rejects_warpk16_without_phase_helper():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="WarpK=16 is rejected"):
        ops.sm70_cutlass_matmul_explicit_warp_probe(
            a, b, 32, 64, 32, 4, 32, 32, 16
        )


@pytest.mark.parametrize(
    ("m", "cta_m", "cta_n", "warps"),
    [
        (1, 32, 128, 4),
        (2, 32, 128, 4),
        (4, 32, 128, 4),
        (8, 64, 128, 4),
        (16, 64, 128, 4),
    ],
)
def test_sm70_cutlass_matmul_probe_threadblock_small_m_matches_torch_mm(
    m: int, cta_m: int, cta_n: int, warps: int
):
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    n = 4096
    k = 4096
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    output = ops.sm70_cutlass_matmul_probe(a, b, cta_m, cta_n, 32, warps, 2, 2, 0)
    reference = torch.mm(a, b)

    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, rtol=5e-2, atol=5e-2)


@pytest.mark.parametrize(
    ("m", "cta_m"),
    [
        (8, 8),
        (16, 16),
        (64, 16),
        (128, 16),
    ],
)
def test_sm70_cutlass_matmul_probe_sm70_atom_path_matches_torch_mm(
    m: int, cta_m: int
):
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    n = 4096
    k = 4096
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    b = torch.randn((k, n), device="cuda", dtype=torch.float16)

    output = ops.sm70_cutlass_matmul_probe(a, b, cta_m, 64, 128, 4, 2, 3, 0)
    reference = torch.mm(a, b)

    assert output.shape == reference.shape
    torch.testing.assert_close(output, reference, rtol=5e-2, atol=5e-1)


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


@pytest.mark.parametrize(
    ("cta_m", "cta_n", "cta_k", "warps"),
    [
        (32, 32, 32, 4),
        (32, 64, 64, 4),
    ],
)
def test_sm70_cutlass_matmul_probe_rejects_unsupported_cute_native_shape(
    cta_m: int, cta_n: int, cta_k: int, warps: int
):
    _require_marlin_cuda()
    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="unsupported CUTLASS 3 CuTe native"):
        ops.sm70_cutlass_matmul_probe(a, b, cta_m, cta_n, cta_k, warps, 2, 0, 0)


def test_sm70_cutlass_matmul_probe_rejects_direct_a_path():
    _require_marlin_cuda()
    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="A direct-global path is TODO"):
        ops.sm70_cutlass_matmul_probe(a, b, 32, 64, 64, 4, 2, 1, 0)


@pytest.mark.parametrize(
    ("cta_m", "cta_n", "cta_k", "warps", "match"),
    [
        (8, 64, 32, 4, "unsupported SM70 atom config"),
        (8, 64, 128, 8, "unsupported SM70 atom config"),
        (8, 128, 128, 4, "unsupported SM70 atom config"),
        (32, 64, 128, 4, "unsupported SM70 atom config"),
    ],
)
def test_sm70_cutlass_matmul_probe_rejects_unsupported_sm70_atom_shape(
    cta_m: int, cta_n: int, cta_k: int, warps: int, match: str
):
    _require_marlin_cuda()
    a = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match=match):
        ops.sm70_cutlass_matmul_probe(a, b, cta_m, cta_n, cta_k, warps, 2, 3, 0)


def test_sm70_cutlass_matmul_probe_rejects_sm70_atom_non_divisible_shape():
    _require_marlin_cuda()
    a = torch.randn((24, 128), device="cuda", dtype=torch.float16)
    b = torch.randn((128, 128), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="requires M divisible by cta_m"):
        ops.sm70_cutlass_matmul_probe(a, b, 16, 64, 128, 4, 2, 3, 0)


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
        None,
        scalar_types.uint4b8.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        True,
        False,
        False,
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
    workspace: torch.Tensor | None = None,
) -> None:
    if act_order:
        raise AssertionError("act_order coverage uses explicit rejection tests")

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
        workspace,
        quant_type.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        is_k_full,
        False,
        False,
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


def _run_dense_bias_accuracy_case(
    quant_type,
    *,
    group_size: int,
    rtol: float,
    atol: float,
    size_m: int = 8,
    size_k: int = 256,
    size_n: int = 256,
) -> None:
    _require_marlin_cuda()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    raw_bias = torch.randn((size_n,), device="cuda", dtype=torch.float16)
    b_bias = marlin_permute_bias(raw_bias)

    if quant_type == scalar_types.uint4b8:
        _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
            w,
            quant_type,
            group_size,
            False,
        )
        b_zeros = None
        is_zp_float = False
        reference = marlin_dense_reference(
            a,
            q_w,
            scales,
            size_k=size_k,
            size_n=size_n,
            group_size=group_size,
            quant_type=quant_type,
            perm=sort_indices,
        ).to(torch.float32)
    elif quant_type == scalar_types.uint4:
        _, q_w, scales, b_zeros, dequantized = marlin_quantize_uint4_zp(
            w,
            group_size,
        )
        g_idx = marlin_make_empty_g_idx(a.device)
        sort_indices = torch.empty(0, dtype=torch.int, device=a.device)
        is_zp_float = True
        reference = torch.matmul(
            a.to(torch.float32),
            dequantized.to(torch.float32),
        )
    else:
        raise AssertionError(f"Unsupported dense bias quant type: {quant_type}")

    output = ops.marlin_gemm(
        a,
        None,
        q_w,
        b_bias,
        scales,
        None,
        None,
        b_zeros,
        g_idx,
        sort_indices,
        None,
        quant_type.id,
        size_m,
        size_n,
        size_k,
        True,
        False,
        False,
        is_zp_float,
    )
    reference = (reference + raw_bias.to(torch.float32)).to(torch.float16)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _dense_geometry(label: str):
    return next(geometry for geometry in SM70_GEOMETRIES if geometry.label == label)


def test_marlin_dense_uint4b8_bias_matches_reference():
    _run_dense_bias_accuracy_case(
        scalar_types.uint4b8,
        group_size=128,
        rtol=5e-2,
        atol=5e-1,
    )


def test_marlin_dense_uint4_zp_bias_matches_reference():
    _run_dense_bias_accuracy_case(
        scalar_types.uint4,
        group_size=128,
        rtol=7e-2,
        atol=6e-1,
    )


def test_marlin_dense_uint4b8_split_k_bias_matches_reference():
    with dense_env(_dense_geometry("32x128x32x4x32x32x32"), 2, "vector_words"):
        _run_dense_bias_accuracy_case(
            scalar_types.uint4b8,
            group_size=128,
            size_k=512,
            rtol=5e-2,
            atol=5e-1,
        )


def test_marlin_dense_bias_validation_errors():
    _require_marlin_cuda()
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    size_m, size_k, size_n = 8, 256, 256
    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w,
        scalar_types.uint4b8,
        128,
        False,
    )
    common_args = (
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
        None,
        scalar_types.uint4b8.id,
        size_m,
        size_n,
        size_k,
        True,
        False,
        False,
        False,
    )

    with pytest.raises(RuntimeError, match="b_bias.size\\(0\\) != size_n"):
        args = list(common_args)
        args[3] = torch.zeros((size_n - 1,), device="cuda", dtype=torch.float16)
        ops.marlin_gemm(*args)

    with pytest.raises(RuntimeError, match="b_bias is not contiguous"):
        args = list(common_args)
        args[3] = torch.zeros((size_n, 2), device="cuda", dtype=torch.float16)[:, 0]
        ops.marlin_gemm(*args)

    with pytest.raises(RuntimeError, match="SM70 Marlin bias must be float16"):
        args = list(common_args)
        args[3] = torch.zeros((size_n,), device="cuda", dtype=torch.float32)
        ops.marlin_gemm(*args)


def _run_fp8_dense_accuracy_case(
    *,
    group_size: int,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
    rtol: float = 4e-2,
    atol: float = 2e-1,
    workspace: torch.Tensor | None = None,
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
        workspace,
        scalar_types.float8_e4m3fn.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        True,
        False,
        False,
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
    workspace: torch.Tensor | None = None,
) -> None:
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    weight_ref, q_w, scales, global_scale, g_idx, sort_indices, _ = (
        marlin_quantize_nvfp4(w, 16)
    )
    output = run_marlin_gemm(
        a,
        q_w,
        scales,
        scalar_types.float4_e2m1f.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        workspace=workspace,
        global_scale=global_scale,
        g_idx=g_idx,
        perm=sort_indices,
        is_k_full=True,
        use_fp32_reduce=False,
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
    workspace: torch.Tensor | None = None,
) -> None:
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    weight_ref, q_w, scales, g_idx, sort_indices, _ = marlin_quantize_mxfp4(w, 32)
    output = run_marlin_gemm(
        a,
        q_w,
        scales,
        scalar_types.float4_e2m1f.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        workspace=workspace,
        g_idx=g_idx,
        perm=sort_indices,
        is_k_full=True,
        use_fp32_reduce=False,
    )
    reference = torch.matmul(
        a.to(torch.float32),
        weight_ref.to(torch.float32),
    ).to(torch.float16)

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _assert_dense_backend_rejects_act_order_compat_args(
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
            act_order=False,
            group_size=group_size,
        )
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)
    _, q_w, scales, _g_idx, _sort_indices, _ = marlin_quantize(
        w, quant_type, group_size, False
    )
    with pytest.raises(RuntimeError, match="act_order"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            None,
            torch.zeros(w.shape[0], device="cuda", dtype=torch.int32),
            None,
            None,
            quant_type.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            is_k_full,
            True,
            True,
            False,
        )

    with pytest.raises(RuntimeError, match="act_order"):
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
            torch.arange(w.shape[0], device="cuda", dtype=torch.int32),
            None,
            quant_type.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            is_k_full,
            True,
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
    use_fp32_reduce: bool = False,
    workspace: torch.Tensor | None = None,
) -> None:
    if use_fp32_reduce or workspace is not None:
        _run_dense_uint4_zp_raw_accuracy_case(
            repack_impl=repack_impl,
            group_size=group_size,
            rtol=rtol,
            atol=atol,
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            use_fp32_reduce=use_fp32_reduce,
            workspace=workspace,
        )
        return

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
    case = run_marlin_linear_kernel_case(
        quant_name="uint4",
        group_size=group_size,
        activation=a,
        weight=w,
    )

    assert case.output is not None
    assert case.reference is not None
    assert torch.isfinite(case.output).all()
    assert not torch.all(case.output == 0)
    assert case.output.float().std().item() > 0
    torch.testing.assert_close(case.output, case.reference, rtol=rtol, atol=atol)


def _run_dense_uint4_zp_raw_accuracy_case(
    *,
    repack_impl: str,
    group_size: int,
    rtol: float,
    atol: float,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
    use_fp32_reduce: bool = False,
    workspace: torch.Tensor | None = None,
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
    _w, q_w, scales, zp, dequantized = marlin_quantize_uint4_zp(
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
        zp,
        None,
        None,
        workspace,
        scalar_types.uint4.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        True,
        False,
        use_fp32_reduce,
        True,
    )
    reference = torch.matmul(a.to(torch.float32), dequantized.to(torch.float32)).to(
        torch.float16
    )

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _run_dense_uint8_zp_accuracy_case(
    *,
    repack_impl: str,
    group_size: int,
    rtol: float,
    atol: float,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
    workspace: torch.Tensor | None = None,
) -> None:
    if workspace is not None:
        _run_dense_uint8_zp_raw_accuracy_case(
            repack_impl=repack_impl,
            group_size=group_size,
            rtol=rtol,
            atol=atol,
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            workspace=workspace,
        )
        return

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
    case = run_marlin_linear_kernel_case(
        quant_name="uint8",
        group_size=group_size,
        activation=a,
        weight=w,
    )

    assert case.output is not None
    assert case.reference is not None
    assert torch.isfinite(case.output).all()
    assert not torch.all(case.output == 0)
    assert case.output.float().std().item() > 0
    torch.testing.assert_close(case.output, case.reference, rtol=rtol, atol=atol)


def _run_dense_uint8_zp_raw_accuracy_case(
    *,
    repack_impl: str,
    group_size: int,
    rtol: float,
    atol: float,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
    workspace: torch.Tensor | None = None,
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
    _w, q_w, scales, zp, dequantized = marlin_quantize_uint8_zp(
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
        zp,
        None,
        None,
        workspace,
        scalar_types.uint8.id,
        a.shape[0],
        w.shape[1],
        w.shape[0],
        True,
        False,
        False,
        True,
    )
    reference = torch.matmul(a.to(torch.float32), dequantized.to(torch.float32)).to(
        torch.float16
    )

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)


def _run_split_k_quant_accuracy_case(
    quant_name: str,
    *,
    group_size: int,
    size_m: int = 16,
    size_k: int = 256,
    size_n: int = 256,
    rtol: float,
    atol: float,
    workspace: torch.Tensor | None = None,
) -> None:
    if quant_name == "uint4b8":
        _run_dense_accuracy_case(
            scalar_types.uint4b8,
            repack_impl="gptq",
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            rtol=rtol,
            atol=atol,
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            workspace=workspace,
        )
    elif quant_name == "uint8":
        _run_dense_uint8_zp_accuracy_case(
            repack_impl="gptq",
            group_size=group_size,
            rtol=rtol,
            atol=atol,
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            workspace=workspace,
        )
    elif quant_name == "uint8b128":
        _run_dense_accuracy_case(
            scalar_types.uint8b128,
            repack_impl="gptq",
            group_size=group_size,
            act_order=False,
            is_k_full=True,
            rtol=rtol,
            atol=atol,
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            workspace=workspace,
        )
    elif quant_name == "fp8":
        _run_fp8_dense_accuracy_case(
            group_size=group_size,
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            rtol=rtol,
            atol=atol,
            workspace=workspace,
        )
    elif quant_name == "nvfp4":
        _run_nvfp4_dense_accuracy_case(
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            rtol=rtol,
            atol=atol,
            workspace=workspace,
        )
    elif quant_name == "mxfp4":
        _run_mxfp4_dense_accuracy_case(
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            rtol=rtol,
            atol=atol,
            workspace=workspace,
        )
    else:
        raise AssertionError(f"Unsupported split-K quant_name={quant_name}")


def _run_auto_cta_n_quant_accuracy_case(
    quant_name: str,
    *,
    group_size: int,
    size_m: int,
    size_k: int,
    size_n: int,
    rtol: float,
    atol: float,
) -> None:
    if quant_name == "uint4":
        _run_dense_uint4_zp_accuracy_case(
            repack_impl="gptq",
            group_size=group_size,
            rtol=rtol,
            atol=atol,
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
        )
    else:
        _run_split_k_quant_accuracy_case(
            quant_name,
            group_size=group_size,
            size_m=size_m,
            size_k=size_k,
            size_n=size_n,
            rtol=rtol,
            atol=atol,
        )


def _make_dense_env_sweep_case(
    key: DenseDirectOpKey,
) -> tuple[tuple, torch.Tensor, torch.Tensor, float, float]:
    _require_marlin_cuda()
    torch.manual_seed(1000 + key.size_m + key.size_n + key.size_k)
    torch.cuda.manual_seed_all(1000 + key.size_m + key.size_n + key.size_k)

    a = torch.randn((key.size_m, key.size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((key.size_k, key.size_n), device="cuda", dtype=torch.float16)
    workspace = marlin_make_workspace(torch.device("cuda"))
    c = torch.empty((key.size_m, key.size_n), device="cuda", dtype=torch.float16)
    rtol, atol = _DENSE_ENV_SWEEP_TOLERANCES[key.quant_name]

    if key.quant_name == "uint4":
        _w, q_w, scales, zp, dequantized = marlin_quantize_uint4_zp(
            w, key.group_size
        )
        reference = torch.matmul(
            a.to(torch.float32), dequantized.to(torch.float32)
        ).to(torch.float16)
        args = (
            a,
            c,
            q_w,
            None,
            scales,
            None,
            None,
            zp,
            None,
            None,
            workspace,
            scalar_types.uint4.id,
            key.size_m,
            key.size_n,
            key.size_k,
            True,
            False,
            False,
            True,
        )
        return args, c, reference, rtol, atol

    if key.quant_name == "uint8":
        _w, q_w, scales, zp, dequantized = marlin_quantize_uint8_zp(
            w, key.group_size
        )
        reference = torch.matmul(
            a.to(torch.float32), dequantized.to(torch.float32)
        ).to(torch.float16)
        args = (
            a,
            c,
            q_w,
            None,
            scales,
            None,
            None,
            zp,
            None,
            None,
            workspace,
            scalar_types.uint8.id,
            key.size_m,
            key.size_n,
            key.size_k,
            True,
            False,
            False,
            True,
        )
        return args, c, reference, rtol, atol

    if key.quant_name in {"uint4b8", "uint8b128"}:
        quant_type = (
            scalar_types.uint4b8
            if key.quant_name == "uint4b8"
            else scalar_types.uint8b128
        )
        _w, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
            w, quant_type, key.group_size, False
        )
        reference = marlin_dense_reference(
            a,
            q_w,
            scales,
            size_k=key.size_k,
            size_n=key.size_n,
            group_size=key.group_size,
            quant_type=quant_type,
            perm=sort_indices,
        ).to(torch.float16)
        args = (
            a,
            c,
            q_w,
            None,
            scales,
            None,
            None,
            marlin_make_empty_g_idx(a.device),
            g_idx,
            sort_indices,
            workspace,
            quant_type.id,
            key.size_m,
            key.size_n,
            key.size_k,
            True,
            False,
            False,
            False,
        )
        return args, c, reference, rtol, atol

    if key.quant_name == "fp8":
        _w, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
            w, scalar_types.float8_e4m3fn, key.group_size, False
        )
        reference = marlin_dense_reference(
            a,
            q_w,
            scales,
            size_k=key.size_k,
            size_n=key.size_n,
            group_size=key.group_size,
            quant_type=scalar_types.float8_e4m3fn,
        ).to(torch.float16)
        args = (
            a,
            c,
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
            key.size_m,
            key.size_n,
            key.size_k,
            True,
            False,
            False,
            False,
        )
        return args, c, reference, rtol, atol

    if key.quant_name == "nvfp4":
        weight_ref, q_w, scales, global_scale, g_idx, sort_indices, _ = (
            marlin_quantize_nvfp4(w, key.group_size)
        )
        reference = torch.matmul(
            a.to(torch.float32), weight_ref.to(torch.float32)
        ).to(torch.float16)
        args = (
            a,
            c,
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
            key.size_m,
            key.size_n,
            key.size_k,
            True,
            False,
            False,
            False,
        )
        return args, c, reference, rtol, atol

    if key.quant_name == "mxfp4":
        weight_ref, q_w, scales, g_idx, sort_indices, _ = marlin_quantize_mxfp4(
            w, key.group_size
        )
        reference = torch.matmul(
            a.to(torch.float32), weight_ref.to(torch.float32)
        ).to(torch.float16)
        args = (
            a,
            c,
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
            key.size_m,
            key.size_n,
            key.size_k,
            True,
            False,
            False,
            False,
        )
        return args, c, reference, rtol, atol

    raise AssertionError(f"Unsupported dense env sweep quant={key.quant_name!r}")


def _assert_dense_env_sweep_combo_matches_reference(
    key: DenseDirectOpKey,
    args: tuple,
    output: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    output.zero_()
    result = ops.marlin_gemm(*args)
    assert result is output
    assert result.shape == (key.size_m, key.size_n)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, reference, rtol=rtol, atol=atol)


def _run_dense_invalid_env_smoke() -> None:
    key = DenseDirectOpKey("uint4b8", 128, 8, 256, 256)
    args, _output, _reference, _rtol, _atol = _make_dense_env_sweep_case(key)
    ops.marlin_gemm(*args)


@pytest.mark.sm70_env_exhaustive
def test_marlin_dense_direct_op_env_geometry_exhaustive_matches_reference():
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the full env sweep")
    _require_marlin_cuda()

    total = 0
    checked = 0
    legal = 0
    rejected = 0
    first_failure: str | None = None

    for key in iter_dense_direct_op_keys():
        prepared: tuple[tuple, torch.Tensor, torch.Tensor, float, float] | None = None
        for geometry, split_k, metadata_cache in iter_env_combinations():
            if exhaustive_index_is_past_limit(total):
                break
            selected = exhaustive_index_is_selected(total)
            total += 1
            if not selected:
                continue

            checked += 1
            if prepared is None:
                prepared = _make_dense_env_sweep_case(key)
            args, output, reference, rtol, atol = prepared
            is_legal = dense_env_combo_is_legal(
                geometry,
                split_k,
                size_n=key.size_n,
                size_k=key.size_k,
            )
            try:
                with dense_env(geometry, split_k, metadata_cache):
                    if is_legal:
                        _assert_dense_env_sweep_combo_matches_reference(
                            key,
                            args,
                            output,
                            reference,
                            rtol=rtol,
                            atol=atol,
                        )
                        legal += 1
                    else:
                        with pytest.raises(RuntimeError, match=EXPLICIT_ENV_REJECTION_RE):
                            ops.marlin_gemm(*args)
                        rejected += 1
            except Exception as exc:
                first_failure = (
                    f"key={key}, geometry={geometry.label}, split_k={split_k}, "
                    f"metadata={metadata_cache}, legal={is_legal}, error={exc}"
                )
                raise
        if exhaustive_index_is_past_limit(total):
            break

    assert checked > 0
    assert legal + rejected == checked
    assert first_failure is None


@pytest.mark.sm70_env_exhaustive
def test_marlin_dense_direct_op_env_focused_mnk_matches_reference():
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the focused env sweep")
    _require_marlin_cuda()

    total = 0
    checked = 0
    legal = 0
    rejected = 0
    first_failure: str | None = None

    for key in iter_dense_focused_mnk_direct_op_keys():
        prepared: tuple[tuple, torch.Tensor, torch.Tensor, float, float] | None = None
        for geometry, split_k, metadata_cache in iter_env_combinations():
            if exhaustive_index_is_past_limit(total):
                break
            selected = exhaustive_index_is_selected(total)
            total += 1
            if not selected:
                continue

            checked += 1
            if prepared is None:
                prepared = _make_dense_env_sweep_case(key)
            args, output, reference, rtol, atol = prepared
            is_legal = dense_env_combo_is_legal(
                geometry,
                split_k,
                size_n=key.size_n,
                size_k=key.size_k,
            )
            try:
                with dense_env(geometry, split_k, metadata_cache):
                    if is_legal:
                        _assert_dense_env_sweep_combo_matches_reference(
                            key,
                            args,
                            output,
                            reference,
                            rtol=rtol,
                            atol=atol,
                        )
                        legal += 1
                    else:
                        with pytest.raises(RuntimeError, match=EXPLICIT_ENV_REJECTION_RE):
                            ops.marlin_gemm(*args)
                        rejected += 1
            except Exception as exc:
                first_failure = (
                    f"key={key}, geometry={geometry.label}, split_k={split_k}, "
                    f"metadata={metadata_cache}, legal={is_legal}, error={exc}"
                )
                raise
        if exhaustive_index_is_past_limit(total):
            break

    assert checked > 0
    assert legal + rejected == checked
    assert first_failure is None


@pytest.mark.parametrize(
    ("geometry", "split_k", "metadata_cache", "match"),
    (
        pytest.param(
            "32x256x32x4x32x64",
            "1",
            "vector_words",
            "Invalid SM70_MARLIN_DENSE_CTA_GEOMETRY",
            id="bad_geometry_field_count",
        ),
        pytest.param(
            "32x256x16x4x32x64x16",
            "1",
            "vector_words",
            "Unsupported SM70 Marlin CTA geometry",
            id="unsupported_geometry",
        ),
        pytest.param(
            "32x256x32x4x32x64x32",
            "3",
            "vector_words",
            "Invalid SM70_MARLIN_DENSE_SPLIT_K",
            id="bad_split_k",
        ),
        pytest.param(
            "32x256x32x4x32x64x32",
            "1",
            "paired_words",
            "Invalid SM70_MARLIN_DENSE_METADATA_CACHE",
            id="bad_metadata",
        ),
    ),
)
def test_marlin_dense_direct_op_rejects_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
    geometry: str,
    split_k: str,
    metadata_cache: str,
    match: str,
):
    _require_marlin_cuda()
    set_dense_env(
        monkeypatch,
        geometry=geometry,
        split_k=split_k,
        metadata_cache=metadata_cache,
    )
    with pytest.raises(RuntimeError, match=match):
        _run_dense_invalid_env_smoke()


@pytest.mark.parametrize("geometry", SM70_GEOMETRIES)
@pytest.mark.parametrize("split_k", SM70_SPLIT_K_VALUES)
@pytest.mark.parametrize("metadata_cache", SM70_METADATA_CACHE_VALUES)
@pytest.mark.sm70_env_exhaustive
def test_marlin_dense_direct_op_env_smoke_single_shape_matches_reference(
    geometry,
    split_k: int,
    metadata_cache: str,
):
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the env smoke sweep")
    _require_marlin_cuda()
    key = DenseDirectOpKey("uint4b8", 128, 8, 256, 256)
    args, output, reference, rtol, atol = _make_dense_env_sweep_case(key)
    with dense_env(geometry, split_k, metadata_cache):
        if dense_env_combo_is_legal(
            geometry,
            split_k,
            size_n=key.size_n,
            size_k=key.size_k,
        ):
            _assert_dense_env_sweep_combo_matches_reference(
                key,
                args,
                output,
                reference,
                rtol=rtol,
                atol=atol,
            )
        else:
            with pytest.raises(RuntimeError, match=EXPLICIT_ENV_REJECTION_RE):
                ops.marlin_gemm(*args)


@pytest.mark.parametrize(
    ("quant_name", "group_size", "rtol", "atol"),
    _AUTO_CTA_N_QUANT_CASES,
)
@pytest.mark.parametrize("size_n", (64, 128, 192, 320))
@pytest.mark.parametrize("size_m", (16, 64, 128, 5120))
def test_marlin_dense_auto_cta_mn_partial_n_matches_reference(
    quant_name: str,
    group_size: int,
    rtol: float,
    atol: float,
    size_m: int,
    size_n: int,
):
    if quant_name not in _DENSE_SUPPORTED_QUANT_NAMES:
        pytest.skip(f"{quant_name} dense path is not supported in this build")
    _run_auto_cta_n_quant_accuracy_case(
        quant_name,
        group_size=group_size,
        size_m=size_m,
        size_k=256,
        size_n=size_n,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    ("quant_name", "group_size", "size_k", "rtol", "atol"),
    _SPLIT_K_QUANT_CASES,
)
@pytest.mark.parametrize(("size_m", "size_n"), ((1, 4096), (16, 4096), (64, 1024), (128, 256)))
def test_marlin_dense_auto_split_k_quant_matches_reference(
    quant_name: str,
    group_size: int,
    size_k: int,
    rtol: float,
    atol: float,
    size_m: int,
    size_n: int,
):
    if quant_name not in _DENSE_SUPPORTED_QUANT_NAMES:
        pytest.skip(f"{quant_name} dense path is not supported in this build")
    _run_split_k_quant_accuracy_case(
        quant_name,
        group_size=group_size,
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize(
    ("quant_name", "group_size", "size_k", "rtol", "atol"),
    _SPLIT_K_QUANT_CASES,
)
def test_marlin_dense_auto_split_k_quant_keeps_empty_workspace_matches_reference(
    quant_name: str,
    group_size: int,
    size_k: int,
    rtol: float,
    atol: float,
):
    if quant_name not in _DENSE_SUPPORTED_QUANT_NAMES:
        pytest.skip(f"{quant_name} dense path is not supported in this build")
    workspace = marlin_make_workspace(torch.device("cuda"))
    _run_split_k_quant_accuracy_case(
        quant_name,
        group_size=group_size,
        size_m=16,
        size_k=size_k,
        size_n=256,
        rtol=rtol,
        atol=atol,
        workspace=workspace,
    )
    assert workspace.numel() == 0


@pytest.mark.parametrize(
    ("quant_name", "group_size", "_size_k", "rtol", "atol"),
    _SPLIT_K_QUANT_CASES,
)
def test_marlin_dense_no_split_quant_accepts_unused_workspace(
    quant_name: str,
    group_size: int,
    _size_k: int,
    rtol: float,
    atol: float,
):
    if quant_name not in _DENSE_SUPPORTED_QUANT_NAMES:
        pytest.skip(f"{quant_name} dense path is not supported in this build")
    workspace = marlin_make_workspace(torch.device("cuda"), 1)
    _run_split_k_quant_accuracy_case(
        quant_name,
        group_size=group_size,
        size_m=16,
        size_k=256,
        size_n=256,
        rtol=rtol,
        atol=atol,
        workspace=workspace,
    )


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
def test_marlin_dense_uint4b8_rejects_act_order_compat_args(
    is_k_full: bool, repack_impl: str
):
    _assert_dense_backend_rejects_act_order_compat_args(
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
def test_marlin_dense_uint4b8_partial_n_auto_cta_matches_reference(
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
        size_k=256,
        size_n=128,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_residue_k_single_group_rejects_k_tile_alignment_contract(
    repack_impl: str,
):
    with pytest.raises(RuntimeError, match=_K_TILE_ALIGNMENT_ERROR):
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
def test_marlin_dense_uint4b8_residue_k_and_n_single_group_rejects_k_tile_alignment_contract(
    repack_impl: str,
):
    with pytest.raises(RuntimeError, match=_K_TILE_ALIGNMENT_ERROR):
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

    with pytest.raises(RuntimeError, match=_K_TILE_ALIGNMENT_ERROR):
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
            None,
            scalar_types.uint4b8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            False,
        )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint4b8_sm70_rejects_act_order_group_switch_args(
    repack_impl: str,
):
    _assert_dense_backend_rejects_act_order_compat_args(
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
def test_marlin_dense_uint4_zp_partial_n_auto_cta_matches_reference(
    repack_impl: str,
):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl=repack_impl,
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=8,
        size_k=256,
        size_n=128,
    )


@pytest.mark.parametrize("size_m", (1, 8, 16, 24, 32, 48, 64))
def test_marlin_dense_uint4_zp_auto_split_k_small_m_matches_reference(size_m: int):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=5e-1,
        size_m=size_m,
        size_k=4096,
        size_n=4096,
    )


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_marlin_dense_uint4_zp_auto_split_k_group_sizes_match_reference(group_size: int):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=group_size,
        rtol=5e-2,
        atol=5e-1,
        size_m=16,
        size_k=4096,
        size_n=256,
    )


@pytest.mark.parametrize(
    ("group_size", "size_k"),
    (
        (128, 384),
        (-1, 352),
        (32, 288),
    ),
)
def test_marlin_dense_uint4_zp_no_split_nonuniform_k_matches_reference(
    group_size: int,
    size_k: int,
):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=group_size,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=16,
        size_k=size_k,
        size_n=256,
    )


@pytest.mark.parametrize(
    ("size_m", "size_n"),
    (
        (1, 4096),
        (16, 4096),
        (64, 1024),
        (128, 256),
    ),
)
def test_marlin_dense_uint4_zp_auto_cta_split_k_shapes_match_reference(
    size_m: int,
    size_n: int,
):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=5e-1,
        size_m=size_m,
        size_k=4096,
        size_n=size_n,
    )


def test_marlin_dense_uint4_zp_auto_split_k_large_k_smoke_matches_reference():
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=5e-1,
        size_m=1,
        size_k=4096,
        size_n=4096,
    )


def test_marlin_dense_uint4_zp_auto_split_k_keeps_empty_workspace_matches_reference():
    workspace = marlin_make_workspace(torch.device("cuda"))
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=5e-1,
        size_m=16,
        size_k=4096,
        size_n=256,
        workspace=workspace,
    )
    assert workspace.numel() == 0


def test_marlin_dense_uint4_zp_no_split_accepts_unused_workspace():
    workspace = marlin_make_workspace(torch.device("cuda"), 1)
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=16,
        size_k=256,
        size_n=256,
        workspace=workspace,
    )


def test_marlin_dense_uint4_zp_auto_split_k_ignores_fp32_reduce():
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=5e-1,
        size_m=16,
        size_k=4096,
        size_n=256,
        use_fp32_reduce=True,
    )


def test_marlin_dense_uint4_zp_auto_split_k_accepts_unused_workspace():
    workspace = marlin_make_workspace(torch.device("cuda"), 16 * 256 - 1)
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=5e-1,
        size_m=16,
        size_k=4096,
        size_n=256,
        workspace=workspace,
    )


@pytest.mark.parametrize(
    "make_workspace",
    (
        lambda device: torch.empty((16, 256), device=device, dtype=torch.float16),
        lambda device: torch.empty((16, 256), device=device, dtype=torch.float32).t(),
        lambda device: torch.empty((16, 256), dtype=torch.float32),
    ),
)
def test_marlin_dense_uint4_zp_auto_split_k_accepts_unused_workspace_variants(
    make_workspace,
):
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=5e-1,
        size_m=16,
        size_k=4096,
        size_n=256,
        workspace=make_workspace(torch.device("cuda")),
    )


def test_marlin_dense_uint4_zp_ignores_fp32_reduce_without_split_k():
    _run_dense_uint4_zp_accuracy_case(
        repack_impl="gptq",
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=8,
        size_k=256,
        size_n=256,
        use_fp32_reduce=True,
    )


@pytest.mark.parametrize("group_size", _GROUP_SIZES)
@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint8_zp_accuracy(group_size: int, repack_impl: str):
    _run_dense_uint8_zp_accuracy_case(
        repack_impl=repack_impl,
        group_size=group_size,
        rtol=5e-2,
        atol=2.5e-1,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint8_zp_small_tile_matches_reference(repack_impl: str):
    _run_dense_uint8_zp_accuracy_case(
        repack_impl=repack_impl,
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=8,
        size_k=128,
        size_n=256,
    )


@pytest.mark.parametrize("repack_impl", _REPACK_IMPL_CASES)
def test_marlin_dense_uint8_zp_partial_n_auto_cta_matches_reference(
    repack_impl: str,
):
    _run_dense_uint8_zp_accuracy_case(
        repack_impl=repack_impl,
        group_size=128,
        rtol=5e-2,
        atol=2.5e-1,
        size_m=8,
        size_k=256,
        size_n=128,
    )


def test_marlin_dense_uint4_zp_requires_zeros():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, _zp, _dequantized = marlin_quantize_uint4_zp(w, 128)

    with pytest.raises(RuntimeError, match="requires fp16 zero points"):
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
            None,
            scalar_types.uint4.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            False,
        )


def test_marlin_dense_uint8_zp_requires_zeros():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, _zp, _dequantized = marlin_quantize_uint8_zp(w, 128)

    with pytest.raises(RuntimeError, match="requires fp16 zero points"):
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
            None,
            scalar_types.uint8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            False,
        )


def test_marlin_dense_uint4_zp_rejects_packed_zero_points():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, packed_zero_points, _dequantized = marlin_quantize_uint4_packed_zp(
        w, 128
    )

    with pytest.raises(RuntimeError, match="fp16 zero points"):
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
            None,
            scalar_types.uint4.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            True,
        )


def test_marlin_dense_uint8_zp_rejects_packed_zero_points():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, _zp, _dequantized = marlin_quantize_uint8_zp(w, 128)
    packed_zero_points = torch.zeros(
        (scales.shape[0], w.shape[1] // 4),
        device="cuda",
        dtype=torch.int32,
    )

    with pytest.raises(RuntimeError, match="fp16 zero points"):
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
            None,
            scalar_types.uint8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            True,
        )


def test_marlin_dense_uint4b8_rejects_zp_metadata():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint4b8, 128, False
    )
    zp = torch.zeros_like(scales)

    with pytest.raises(RuntimeError, match="zero-point metadata"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp,
            g_idx,
            sort_indices,
            None,
            scalar_types.uint4b8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            True,
        )


def test_marlin_dense_uint8_zp_rejects_zeros_without_flag():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, zp, _dequantized = marlin_quantize_uint8_zp(w, 128)

    with pytest.raises(RuntimeError, match="is_zp_float is false"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp,
            None,
            None,
            None,
            scalar_types.uint8.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            False,
        )


def test_marlin_dense_uint8b128_rejects_zp_metadata():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint8b128, 128, False
    )
    zp = torch.zeros_like(scales)

    with pytest.raises(RuntimeError, match="zero-point metadata"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp,
            g_idx,
            sort_indices,
            None,
            scalar_types.uint8b128.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            True,
        )


def test_marlin_dense_uint8b128_rejects_is_zp_float_without_metadata():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
        w, scalar_types.uint8b128, 128, False
    )

    with pytest.raises(RuntimeError, match="is_zp_float is true"):
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
            None,
            scalar_types.uint8b128.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            True,
        )


def test_marlin_dense_uint4_zp_rejects_zeros_without_flag():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
    w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, zp, _dequantized = marlin_quantize_uint4_zp(w, 128)

    with pytest.raises(RuntimeError, match="is_zp_float is false"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp,
            None,
            None,
            None,
            scalar_types.uint4.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            True,
            False,
            False,
            False,
        )


def test_marlin_dense_uint4_zp_rejects_act_order_compat_args():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    size_k = 256
    a = torch.randn((16, size_k), device="cuda", dtype=torch.float16)
    w = torch.randn((size_k, 256), device="cuda", dtype=torch.float16)
    _w, q_w, scales, zp, _dequantized = marlin_quantize_uint4_zp(w, 64)
    g_idx = (torch.arange(size_k, device=a.device, dtype=torch.int32) // 64).contiguous()
    perm = torch.arange(size_k, device=a.device, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="act_order"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp,
            g_idx,
            None,
            None,
            scalar_types.uint4.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            False,
            True,
            True,
            True,
        )

    with pytest.raises(RuntimeError, match="act_order"):
        ops.marlin_gemm(
            a,
            None,
            q_w,
            None,
            scales,
            None,
            None,
            zp,
            None,
            perm,
            None,
            scalar_types.uint4.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            False,
            True,
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
    def test_marlin_dense_uint8b128_rejects_act_order_compat_args(
        is_k_full: bool,
    ):
        _assert_dense_backend_rejects_act_order_compat_args(
            scalar_types.uint8b128,
            group_size=64,
            is_k_full=is_k_full,
        )

    def test_marlin_dense_uint8b128_rejects_single_group_act_order_compat_args():
        _assert_dense_backend_rejects_act_order_compat_args(
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
    def test_marlin_dense_uint8b128_partial_n_auto_cta_matches_reference(
        repack_impl: str,
    ):
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

if "fp8" in _DENSE_SUPPORTED_QUANT_NAMES:

    @pytest.mark.parametrize("group_size", (-1, 128))
    def test_marlin_dense_fp8_weight_accuracy(group_size: int):
        _run_fp8_dense_accuracy_case(group_size=group_size)

    def test_marlin_dense_fp8_weight_partial_n_auto_cta_matches_reference():
        _run_fp8_dense_accuracy_case(
            group_size=128,
            size_m=8,
            size_k=256,
            size_n=128,
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
                None,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                False,
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
        with pytest.raises(RuntimeError, match=_K_TILE_ALIGNMENT_ERROR):
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
                None,
                scalar_types.float8_e4m3fn.id,
                a_bad_k.shape[0],
                size_n,
                bad_k,
                True,
                False,
                False,
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
                None,
                scalar_types.float8_e4m3fn.id,
                a_bad_n.shape[0],
                bad_n,
                size_k,
                True,
                False,
                False,
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
        workspace = None
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
            False,
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
                False,
                False,
            )

        c_bf16 = torch.empty((a.shape[0], w.shape[1]), device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="SM70 build only supports float16 outputs"):
            ops.marlin_gemm(a, c_bf16, *common_args)

        with pytest.raises(RuntimeError, match="act_order"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                torch.zeros(w.shape[0], device="cuda", dtype=torch.int32),
                torch.arange(w.shape[0], device="cuda", dtype=torch.int32),
                workspace,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                False,
                True,
                True,
                False,
            )

        bias_output = ops.marlin_gemm(
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
            False,
            False,
        )
        assert bias_output.shape == (a.shape[0], w.shape[1])
        assert torch.isfinite(bias_output).all()

        with pytest.raises(RuntimeError, match="supports global_scale only for nvfp4 format"):
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
                False,
                False,
            )

        zp = torch.zeros_like(scales)
        with pytest.raises(RuntimeError, match="zero-point metadata"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                zp,
                g_idx,
                sort_indices,
                workspace,
                scalar_types.float8_e4m3fn.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                True,
                False,
                False,
                True,
            )

        with pytest.raises(RuntimeError, match="is_zp_float is true"):
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
                False,
                True,
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
            workspace,
            scalar_types.float8_e4m3fn.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            False,
            False,
            False,
            False,
        )
        assert output.shape == (a.shape[0], w.shape[1])
        assert torch.isfinite(output).all()


if "nvfp4" in _DENSE_SUPPORTED_QUANT_NAMES:

    def test_marlin_dense_nvfp4_weight_accuracy():
        _run_nvfp4_dense_accuracy_case()

    def test_marlin_dense_nvfp4_weight_partial_n_auto_cta_matches_reference():
        _run_nvfp4_dense_accuracy_case(
            size_m=8,
            size_k=256,
            size_n=128,
        )

    def test_marlin_dense_nvfp4_requires_fp8_scales_and_global_scale():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        _weight_ref, q_w, scales, global_scale, g_idx, sort_indices, _ = (
            marlin_quantize_nvfp4(w, 16)
        )
        fp16_scales = scales.to(torch.float32).to(torch.float16)
        workspace = None

        with pytest.raises(RuntimeError, match="global_scale parameter must be passed"):
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
                False,
                False,
            )

        with pytest.raises(RuntimeError, match="b_scales must be float8_e4m3fn"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                fp16_scales,
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
                False,
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
                False,
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
        workspace = None

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
                False,
                False,
            )

        with pytest.raises(RuntimeError, match="act_order"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                global_scale,
                None,
                torch.zeros(w.shape[0], device="cuda", dtype=torch.int32),
                torch.arange(w.shape[0], device="cuda", dtype=torch.int32),
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                False,
                True,
                True,
                False,
            )

        bias_output = ops.marlin_gemm(
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
            False,
            False,
        )
        assert bias_output.shape == (a.shape[0], w.shape[1])
        assert torch.isfinite(bias_output).all()

        with pytest.raises(RuntimeError, match="zero-point metadata"):
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
                False,
                True,
            )

        output = ops.marlin_gemm(
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
            False,
            False,
        )
        assert output.shape == (a.shape[0], w.shape[1])
        assert torch.isfinite(output).all()


if "mxfp4" in _DENSE_SUPPORTED_QUANT_NAMES:

    def test_marlin_dense_mxfp4_weight_accuracy():
        _run_mxfp4_dense_accuracy_case()

    def test_marlin_dense_mxfp4_weight_partial_n_auto_cta_matches_reference():
        _run_mxfp4_dense_accuracy_case(
            size_m=8,
            size_k=256,
            size_n=128,
        )

    def test_marlin_dense_mxfp4_rejects_fp16_scales_and_wrong_routing():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        _weight_ref, q_w, scales, g_idx, sort_indices, _ = marlin_quantize_mxfp4(
            w,
            32,
        )
        workspace = None

        with pytest.raises(RuntimeError, match="float8_e8m0fnu"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales.to(torch.float32).to(torch.float16),
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
                False,
                False,
            )

        with pytest.raises(RuntimeError, match="supports global_scale only for nvfp4"):
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
                False,
                False,
            )

        _nv_weight_ref, nv_q_w, nv_scales, _global_scale, nv_g_idx, nv_sort_indices, _ = (
            marlin_quantize_nvfp4(w, 16)
        )
        with pytest.raises(RuntimeError, match="global_scale parameter must be passed"):
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
                False,
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
        workspace = None
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
            False,
            False,
        )

        with pytest.raises(RuntimeError, match=_FLOAT16_ACTIVATION_ERROR):
            ops.marlin_gemm(a.to(torch.bfloat16), None, *common_args)

        c_bf16 = torch.empty((a.shape[0], w.shape[1]), device="cuda", dtype=torch.bfloat16)
        with pytest.raises(RuntimeError, match="SM70 build only supports float16 outputs"):
            ops.marlin_gemm(a, c_bf16, *common_args)

        with pytest.raises(RuntimeError, match="float8_e8m0fnu"):
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
                False,
                False,
            )

        with pytest.raises(RuntimeError, match="act_order"):
            ops.marlin_gemm(
                a,
                None,
                q_w,
                None,
                scales,
                None,
                None,
                None,
                torch.zeros(w.shape[0], device="cuda", dtype=torch.int32),
                torch.arange(w.shape[0], device="cuda", dtype=torch.int32),
                workspace,
                scalar_types.float4_e2m1f.id,
                a.shape[0],
                w.shape[1],
                w.shape[0],
                False,
                True,
                True,
                False,
            )

        bias_output = ops.marlin_gemm(
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
            False,
            False,
        )
        assert bias_output.shape == (a.shape[0], w.shape[1])
        assert torch.isfinite(bias_output).all()

        with pytest.raises(RuntimeError, match="zero-point metadata"):
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
                False,
                True,
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
            workspace,
            scalar_types.float4_e2m1f.id,
            a.shape[0],
            w.shape[1],
            w.shape[0],
            False,
            False,
            False,
            False,
        )
        assert output.shape == (a.shape[0], w.shape[1])
        assert torch.isfinite(output).all()

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
        scales_bad_k = torch.ones((1, size_n), device="cuda", dtype=torch.float32).to(
            torch.float8_e8m0fnu
        )
        with pytest.raises(RuntimeError, match=_K_TILE_ALIGNMENT_ERROR):
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
                None,
                scalar_types.float4_e2m1f.id,
                a_bad_k.shape[0],
                size_n,
                bad_k,
                True,
                False,
                False,
                False,
            )

        size_k = 256
        bad_n = 96
        a_bad_n = torch.randn((8, size_k), device="cuda", dtype=torch.float16)
        q_w_bad_n = torch.empty((size_k // 16, bad_n * 16 // 8), device="cuda", dtype=torch.int32)
        scales_bad_n = torch.ones(
            (size_k // 32, bad_n), device="cuda", dtype=torch.float32
        ).to(torch.float8_e8m0fnu)
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
                None,
                scalar_types.float4_e2m1f.id,
                a_bad_n.shape[0],
                bad_n,
                size_k,
                True,
                False,
                False,
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
    workspace = None

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
                False,
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
            False,
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
        workspace = None

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
                    False,
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
                False,
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
                False,
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
                False,
                False,
            )
