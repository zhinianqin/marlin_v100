from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marlin_v100.calibration import (
    source_target_capability,
    source_target_label,
    supported_dense_quant_type_names,
)
from marlin_v100 import ops
from tests.helpers import (
    marlin_dense_reference,
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_quantize,
    scalar_types,
)

_DENSE_SUPPORTED_QUANT_NAMES = frozenset(
    supported_dense_quant_type_names(("uint4b8", "uint8b128"))
)


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
    ]
    for name in expected:
        assert hasattr(ops, name)


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


def test_marlin_dense_smoke_local_helpers():
    _require_marlin_cuda()

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


def test_marlin_dense_uint4b8_accuracy():
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

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
    reference = marlin_dense_reference(
        a,
        q_w,
        scales,
        size_k=w.shape[0],
        size_n=w.shape[1],
        group_size=128,
        quant_type=scalar_types.uint4b8,
    ).to(torch.float16)

    assert torch.isfinite(output).all()
    assert not torch.all(output == 0)
    assert output.float().std().item() > 0
    torch.testing.assert_close(output, reference, rtol=5e-2, atol=2.5e-1)


if "uint8b128" in _DENSE_SUPPORTED_QUANT_NAMES:

    def test_marlin_dense_uint8b128_accuracy():
        _require_marlin_cuda()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

        a = torch.randn((16, 256), device="cuda", dtype=torch.float16)
        w = torch.randn((256, 256), device="cuda", dtype=torch.float16)
        _, q_w, scales, g_idx, sort_indices, _ = marlin_quantize(
            w, scalar_types.uint8b128, 128, False
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
            scalar_types.uint8b128.id,
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
            group_size=128,
            quant_type=scalar_types.uint8b128,
        ).to(torch.float16)

        assert torch.isfinite(output).all()
        assert not torch.all(output == 0)
        assert output.float().std().item() > 0
        torch.testing.assert_close(output, reference, rtol=4e-2, atol=2e-1)


def test_marlin_dense_rejects_non_sm75_or_unsupported_dtypes():
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

    a_bf16 = a.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="float16 or int8 activations"):
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

    def test_marlin_dense_uint8b128_rejects_unsupported_dtypes():
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

        a_bf16 = a.to(torch.bfloat16)
        with pytest.raises(RuntimeError, match="float16 or int8 activations"):
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
            match="float16 or int8 activations|float16 outputs|float16 scales",
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
            match="float16 or int8 activations|float16 outputs|float16 scales",
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
