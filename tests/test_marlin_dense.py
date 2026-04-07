from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from marlin_v100 import ops
from tests.helpers import (
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_quantize,
    scalar_types,
)


def _require_marlin_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability[0] < 7 or (capability[0] == 7 and capability[1] < 5):
        pytest.skip("Marlin requires SM75 or newer")
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
