from __future__ import annotations

import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tests.calibration import source_target_capability, source_target_label
from tests import ops
from tests import quant_utils
from tests.helpers import (
    _quantize_fp8_weight,
    _quantize_mxfp4_weight,
    _quantize_nvfp4_weight,
    _quantize_unsigned_with_bias,
    _quantize_uint4_with_zero_point,
    _quantize_uint8_with_zero_point,
    awq_pack,
    gptq_pack,
    pack_cols,
    prepare_marlin_linear_kernel_case,
    run_marlin_linear_kernel_case,
)
from vllm.scalar_type import scalar_types
from tests.writeback_marlin_cases import (
    DENSE_ALL_QUANT_NAMES,
    DENSE_IRREGULAR_SHAPE_CASES,
    DENSE_REGULAR_SHAPE_CASES,
    DENSE_WRITEBACK_CLASS_CASES,
    DENSE_WRITEBACK_CLASS_CASE_BY_NAME,
    DenseWritebackMatrixCase,
    iter_dense_writeback_matrix,
    is_dense_group_size_supported,
)


_GROUP_SIZES = (-1, 32, 64, 128)
_DENSE_CLASS_TEST_COVERAGE = {
    "marlin_linear_kernel",
    "gptq_marlin_linear_method",
    "awq_marlin_linear_method",
    "compressed_tensors_wna16",
    "marlin_fp8_scaled_mm",
    "compressed_tensors_w8a16_fp8",
    "compressed_tensors_w4a16_nvfp4",
    "compressed_tensors_w4a16_mxfp4",
}
_DENSE_FULL_MATRIX_JSONL = Path(
    "benchmarks/results/20260604_test_dense_writeback_full_matrix.jsonl"
)
_DENSE_FULL_MATRIX_SUMMARY = Path(
    "benchmarks/results/20260604_test_dense_writeback_full_matrix_summary.json"
)


def _counter_to_json(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def _dense_irregular_numeric_params():
    shapes = DENSE_REGULAR_SHAPE_CASES + DENSE_IRREGULAR_SHAPE_CASES
    params = []
    for class_case in DENSE_WRITEBACK_CLASS_CASES:
        for quant_name in DENSE_ALL_QUANT_NAMES:
            for group_size in class_case.default_group_sizes:
                for shape in shapes:
                    marks = []
                    if quant_name not in class_case.quant_names:
                        marks.append(
                            pytest.mark.skip(
                                reason=(
                                    "unsupported dense writeback class/quant "
                                    "combination"
                                )
                            )
                        )
                    elif quant_name == "float4_e2m1f":
                        marks.append(
                            pytest.mark.skip(
                                reason=(
                                    "direct float4_e2m1f scalar support is "
                                    "asserted separately; production FP4 "
                                    "numeric paths are NVFP4/MXFP4 schemes"
                                )
                            )
                        )
                    elif not is_dense_group_size_supported(
                        quant_name,
                        group_size,
                        shape.size_k,
                    ):
                        marks.append(
                            pytest.mark.skip(
                                reason=(
                                    "unsupported class/quant/group/shape "
                                    "alignment combination"
                                )
                            )
                        )
                    elif class_case.name in {
                        "marlin_fp8_scaled_mm",
                        "compressed_tensors_w8a16_fp8",
                    } and group_size != -1:
                        marks.append(
                            pytest.mark.skip(
                                reason=(
                                    "local class-path FP8 numeric helper "
                                    "covers channel-wise group_size=-1; "
                                    "block/group FP8 is reported as unsupported"
                                )
                            )
                        )
                    params.append(
                        pytest.param(
                            class_case.name,
                            quant_name,
                            group_size,
                            shape,
                            marks=marks,
                            id=(
                                f"{class_case.name}_{quant_name}_g{group_size}_"
                                f"{shape.name}"
                            ),
                        )
                    )
    return params


def test_dense_writeback_class_inventory_has_test_coverage() -> None:
    assert {case.name for case in DENSE_WRITEBACK_CLASS_CASES} == _DENSE_CLASS_TEST_COVERAGE


@dataclass
class _RecordedDenseGemm:
    c_tmp: torch.Tensor
    size_m: int
    size_n: int
    size_k: int
    use_fp32_reduce: bool
    is_zp_float: bool
    b_q_type: Any


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


@pytest.mark.parametrize("quant_name", ("uint4", "uint8"))
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_marlin_linear_kernel_u4_u8_zp_preprocess_and_apply(
    quant_name: str,
    group_size: int,
):
    _require_marlin_cuda()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    size_m = 16
    size_k = 256
    size_n = 256
    activation = torch.randn((size_m, size_k), device="cuda", dtype=torch.float16)
    weight = torch.randn((size_k, size_n), device="cuda", dtype=torch.float16)

    case = run_marlin_linear_kernel_case(
        quant_name=quant_name,
        group_size=group_size,
        activation=activation,
        weight=weight,
    )

    assert case.layer.qzeros.dtype == torch.float16
    assert case.layer.qzeros.ndim == 2
    assert case.layer.qzeros.shape == (case.num_groups, size_n)
    assert case.layer.qzeros.is_contiguous()
    assert case.kernel.is_zp_float is True
    assert case.kernel.c_tmp.dtype == torch.float32
    assert case.kernel.c_tmp.device == activation.device
    assert case.kernel.c_tmp.numel() == 0

    assert case.output is not None
    assert case.reference is not None
    assert torch.isfinite(case.output).all()
    assert not torch.all(case.output == 0)
    assert case.output.float().std().item() > 0
    torch.testing.assert_close(case.output, case.reference, rtol=5e-2, atol=2.5e-1)


def _install_dense_marlin_cpu_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fast_repack: bool = False,
) -> list[_RecordedDenseGemm]:
    import vllm.model_executor.kernels.linear.mixed_precision.marlin as mp_marlin_mod
    import vllm.model_executor.layers.quantization.utils.marlin_utils as marlin_utils_mod
    import vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 as fp4_mod
    import vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 as fp8_mod

    records: list[_RecordedDenseGemm] = []

    def fake_repack(
        b_q_weight: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        perm = kwargs.get("perm", args[0] if len(args) > 0 else None)
        size_k = kwargs.get("size_k", args[1] if len(args) > 1 else None)
        size_n = kwargs.get("size_n", args[2] if len(args) > 2 else None)
        num_bits = kwargs.get("num_bits", args[3] if len(args) > 3 else None)
        is_a_8bit = kwargs.get("is_a_8bit", args[4] if len(args) > 4 else False)
        if size_k is None or size_n is None or num_bits is None:
            return b_q_weight.detach().clone().contiguous()
        if fast_repack:
            return torch.zeros(
                (
                    int(size_k) // 16,
                    int(size_n) * (int(num_bits) // 2),
                ),
                dtype=torch.int32,
                device=b_q_weight.device,
            )

        pack_factor = 32 // int(num_bits)
        unpacked = torch.empty(
            (int(size_k), int(size_n)),
            dtype=torch.int32,
            device=b_q_weight.device,
        )
        words = b_q_weight.to(torch.int64)
        mask = (1 << int(num_bits)) - 1
        for idx in range(pack_factor):
            unpacked[idx::pack_factor, :] = (
                (words >> (int(num_bits) * idx)) & mask
            ).to(torch.int32)
        if perm is not None and perm.numel() > 0:
            unpacked = unpacked.index_select(0, perm.to(torch.long)).contiguous()
        weight_perm = quant_utils.get_weight_perm(int(num_bits), is_a_8bit=is_a_8bit)
        return quant_utils.marlin_weights(
            unpacked,
            int(size_k),
            int(size_n),
            int(num_bits),
            weight_perm.to(unpacked.device),
            is_a_8bit=is_a_8bit,
        )

    def fake_marlin_gemm(*args, **kwargs) -> torch.Tensor:
        a = kwargs.get("a", args[0] if len(args) > 0 else None)
        c_tmp = kwargs.get("c_tmp", args[10] if len(args) > 10 else None)
        b_q_type = kwargs.get("b_q_type", args[11] if len(args) > 11 else None)
        size_m = kwargs.get("size_m", args[12] if len(args) > 12 else None)
        size_n = kwargs.get("size_n", args[13] if len(args) > 13 else None)
        size_k = kwargs.get("size_k", args[14] if len(args) > 14 else None)
        use_fp32_reduce = kwargs.get(
            "use_fp32_reduce", args[17] if len(args) > 17 else False
        )
        is_zp_float = kwargs.get(
            "is_zp_float", args[18] if len(args) > 18 else False
        )

        assert a is not None
        assert c_tmp is not None
        assert b_q_type is not None
        assert size_m is not None
        assert size_n is not None
        assert size_k is not None
        records.append(
            _RecordedDenseGemm(
                c_tmp=c_tmp,
                size_m=size_m,
                size_n=size_n,
                size_k=size_k,
                use_fp32_reduce=use_fp32_reduce,
                is_zp_float=is_zp_float,
                b_q_type=b_q_type,
            )
        )
        return torch.zeros((size_m, size_n), dtype=a.dtype, device=a.device)

    monkeypatch.setattr(mp_marlin_mod.ops, "gptq_marlin_repack", fake_repack)
    monkeypatch.setattr(mp_marlin_mod.ops, "marlin_gemm", fake_marlin_gemm)
    monkeypatch.setattr(marlin_utils_mod.ops, "gptq_marlin_repack", fake_repack)
    monkeypatch.setattr(marlin_utils_mod.ops, "marlin_gemm", fake_marlin_gemm)
    monkeypatch.setattr(fp8_mod.ops, "gptq_marlin_repack", fake_repack)
    monkeypatch.setattr(fp8_mod.ops, "marlin_gemm", fake_marlin_gemm)
    monkeypatch.setattr(fp4_mod.ops, "gptq_marlin_repack", fake_repack)
    monkeypatch.setattr(fp4_mod.ops, "marlin_gemm", fake_marlin_gemm)
    return records


def _assert_fresh_layer_c_tmp(layer: torch.nn.Module) -> None:
    assert hasattr(layer, "c_tmp")
    assert layer.c_tmp.dtype == torch.float32
    assert layer.c_tmp.is_contiguous()
    assert layer.c_tmp.numel() == 0
    assert not layer.c_tmp.requires_grad


def _assert_layer_apply_keeps_empty_c_tmp(
    apply_fn: Any,
    layer: torch.nn.Module,
    records: list[_RecordedDenseGemm],
    *,
    size_k: int,
    size_n: int,
) -> None:
    records.clear()
    x = torch.randn(2, size_k, dtype=torch.float16)
    output = apply_fn(x)
    assert output.shape == (2, size_n)
    assert layer.c_tmp.dtype == torch.float32
    assert layer.c_tmp.is_contiguous()
    assert layer.c_tmp.numel() == 0
    assert len(records) == 1
    assert records[0].c_tmp is layer.c_tmp
    assert records[0].size_m == 2
    assert records[0].size_n == size_n
    assert records[0].size_k == size_k
    assert records[0].use_fp32_reduce is False
    assert records[0].is_zp_float is False

    first_c_tmp = layer.c_tmp
    records.clear()
    output = apply_fn(torch.randn(1, size_k, dtype=torch.float16))
    assert output.shape == (1, size_n)
    assert layer.c_tmp is first_c_tmp
    assert len(records) == 1
    assert records[0].c_tmp is first_c_tmp
    assert records[0].use_fp32_reduce is False
    assert records[0].is_zp_float is False

    records.clear()
    output = apply_fn(torch.randn(5, size_k, dtype=torch.float16))
    assert output.shape == (5, size_n)
    assert layer.c_tmp is first_c_tmp
    assert layer.c_tmp.numel() == 0
    assert len(records) == 1
    assert records[0].c_tmp is layer.c_tmp
    assert records[0].use_fp32_reduce is False
    assert records[0].is_zp_float is False


def _assert_fresh_kernel_c_tmp(kernel: Any) -> None:
    assert hasattr(kernel, "c_tmp")
    assert kernel.c_tmp.dtype == torch.float32
    assert kernel.c_tmp.is_contiguous()
    assert kernel.c_tmp.numel() == 0
    assert not kernel.c_tmp.requires_grad


def _assert_kernel_apply_keeps_empty_c_tmp(
    apply_fn: Any,
    kernel: Any,
    records: list[_RecordedDenseGemm],
    *,
    size_k: int,
    size_n: int,
    expected_is_zp_float: bool,
) -> None:
    records.clear()
    output = apply_fn(torch.randn(2, size_k, dtype=torch.float16))
    assert output.shape == (2, size_n)
    assert kernel.c_tmp.dtype == torch.float32
    assert kernel.c_tmp.is_contiguous()
    assert kernel.c_tmp.numel() == 0
    assert len(records) == 1
    assert records[0].c_tmp is kernel.c_tmp
    assert records[0].size_m == 2
    assert records[0].size_n == size_n
    assert records[0].size_k == size_k
    assert records[0].use_fp32_reduce is False
    assert records[0].is_zp_float is expected_is_zp_float

    first_c_tmp = kernel.c_tmp
    records.clear()
    output = apply_fn(torch.randn(1, size_k, dtype=torch.float16))
    assert output.shape == (1, size_n)
    assert kernel.c_tmp is first_c_tmp
    assert len(records) == 1
    assert records[0].c_tmp is first_c_tmp
    assert records[0].use_fp32_reduce is False
    assert records[0].is_zp_float is expected_is_zp_float

    records.clear()
    output = apply_fn(torch.randn(5, size_k, dtype=torch.float16))
    assert output.shape == (5, size_n)
    assert kernel.c_tmp is first_c_tmp
    assert kernel.c_tmp.numel() == 0
    assert len(records) == 1
    assert records[0].c_tmp is kernel.c_tmp
    assert records[0].use_fp32_reduce is False
    assert records[0].is_zp_float is expected_is_zp_float


def _make_compressed_tensors_wna16_layer_case(
    *,
    quant_name: str,
    group_size: int,
    size_k: int = 128,
    size_n: int = 128,
) -> tuple[torch.nn.Module, Any, int]:
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
        CompressedTensorsWNA16,
    )

    symmetric = quant_name in {"uint4b8", "uint8b128"}
    if quant_name == "uint4":
        num_bits = 4
        quantize = _quantize_uint4_with_zero_point
    elif quant_name == "uint8":
        num_bits = 8
        quantize = _quantize_uint8_with_zero_point
    elif quant_name == "uint4b8":
        num_bits = 4
        quantize = lambda weight, gs: (
            *_quantize_unsigned_with_bias(weight, gs, scalar_types.uint4b8.bias),
            None,
        )
    elif quant_name == "uint8b128":
        num_bits = 8
        quantize = lambda weight, gs: (
            *_quantize_unsigned_with_bias(weight, gs, scalar_types.uint8b128.bias),
            None,
        )
    else:
        raise ValueError(f"Unsupported quant_name={quant_name!r}")

    strategy = "channel" if group_size == -1 else "group"
    scheme = CompressedTensorsWNA16(
        strategy=strategy,
        num_bits=num_bits,
        group_size=None if group_size == -1 else group_size,
        symmetric=symmetric,
    )
    layer = torch.nn.Module()
    scheme.create_weights(
        layer=layer,
        output_size=size_n,
        input_size=size_k,
        output_partition_sizes=[size_n],
        input_size_per_partition=size_k,
        params_dtype=torch.float16,
        weight_loader=lambda param, loaded_weight: None,
    )

    weight = torch.randn(size_k, size_n, dtype=torch.float16)
    q_weight, scales, zero_points = quantize(weight, group_size)
    num_groups = scales.shape[0]
    packed_weight = gptq_pack(q_weight, num_bits, size_k, size_n)

    layer.weight_packed.data.copy_(packed_weight.t().contiguous())
    layer.weight_scale.data.copy_(scales.t().contiguous())
    if zero_points is not None:
        packed_zero_points = pack_cols(
            zero_points,
            num_bits,
            num_groups,
            size_n,
        )
        layer.weight_zero_point.data.copy_(packed_zero_points.t().contiguous())
    layer.weight_shape.data.copy_(torch.tensor([size_k, size_n], dtype=torch.int64))
    return layer, scheme, num_groups


def _make_gptq_marlin_linear_method_case(
    *,
    quant_name: str,
    group_size: int,
    size_k: int = 128,
    size_n: int = 128,
) -> tuple[torch.nn.Module, Any]:
    from vllm.model_executor.layers.quantization.gptq_marlin import (
        GPTQMarlinConfig,
        GPTQMarlinLinearMethod,
    )

    if quant_name == "uint4b8":
        num_bits = 4
        bias = scalar_types.uint4b8.bias
    elif quant_name == "uint8b128":
        num_bits = 8
        bias = scalar_types.uint8b128.bias
    else:
        raise ValueError(f"Unsupported GPTQ linear quant_name={quant_name!r}")

    config = GPTQMarlinConfig(
        weight_bits=num_bits,
        group_size=group_size,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )
    method = GPTQMarlinLinearMethod(config)
    layer = torch.nn.Module()
    method.create_weights(
        layer=layer,
        input_size_per_partition=size_k,
        output_partition_sizes=[size_n],
        input_size=size_k,
        output_size=size_n,
        params_dtype=torch.float16,
        weight_loader=lambda param, loaded_weight: None,
    )

    weight = torch.randn(size_k, size_n, dtype=torch.float16)
    q_weight, scales = _quantize_unsigned_with_bias(weight, group_size, bias)
    layer.qweight.data.copy_(gptq_pack(q_weight, num_bits, size_k, size_n))
    layer.scales.data.copy_(scales)
    layer.qzeros.data.zero_()
    method.process_weights_after_loading(layer)
    return layer, method


def _make_awq_marlin_linear_method_case(
    *,
    quant_name: str,
    group_size: int,
    size_k: int = 128,
    size_n: int = 128,
) -> tuple[torch.nn.Module, Any, int]:
    from vllm.model_executor.layers.quantization.awq_marlin import (
        AWQMarlinConfig,
        AWQMarlinLinearMethod,
    )

    if quant_name == "uint4":
        num_bits = 4
        quantize = _quantize_uint4_with_zero_point
    elif quant_name == "uint8":
        num_bits = 8
        quantize = _quantize_uint8_with_zero_point
    else:
        raise ValueError(f"Unsupported AWQ linear quant_name={quant_name!r}")

    config = AWQMarlinConfig(
        weight_bits=num_bits,
        group_size=group_size,
        zero_point=True,
        lm_head_quantized=False,
        modules_to_not_convert=None,
        full_config={},
    )
    method = AWQMarlinLinearMethod(config)
    layer = torch.nn.Module()
    method.create_weights(
        layer=layer,
        input_size_per_partition=size_k,
        output_partition_sizes=[size_n],
        input_size=size_k,
        output_size=size_n,
        params_dtype=torch.float16,
        weight_loader=lambda param, loaded_weight: None,
    )

    weight = torch.randn(size_k, size_n, dtype=torch.float16)
    q_weight, scales, zero_points = quantize(weight, group_size)
    num_groups = zero_points.shape[0]
    layer.qweight.data.copy_(awq_pack(q_weight, num_bits, size_k, size_n))
    layer.scales.data.copy_(scales)
    layer.qzeros.data.copy_(awq_pack(zero_points, num_bits, num_groups, size_n))
    method.process_weights_after_loading(layer)
    return layer, method, num_groups


@pytest.mark.parametrize("quant_name", ("uint4", "uint8"))
@pytest.mark.parametrize("group_size", _GROUP_SIZES)
def test_compressed_tensors_wna16_asymmetric_uses_marlin_kernel_zp_and_c_tmp(
    monkeypatch: pytest.MonkeyPatch,
    quant_name: str,
    group_size: int,
):
    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 128
    size_n = 128
    layer, scheme, num_groups = _make_compressed_tensors_wna16_layer_case(
        quant_name=quant_name,
        group_size=group_size,
        size_k=size_k,
        size_n=size_n,
    )

    scheme.process_weights_after_loading(layer)
    assert layer.weight_zero_point.dtype == torch.float16
    assert layer.weight_zero_point.ndim == 2
    assert layer.weight_zero_point.shape == (num_groups, size_n)
    assert layer.weight_zero_point.is_contiguous()
    assert scheme.kernel.is_zp_float is True
    _assert_fresh_kernel_c_tmp(scheme.kernel)

    _assert_kernel_apply_keeps_empty_c_tmp(
        lambda x: scheme.apply_weights(layer, x, None),
        scheme.kernel,
        records,
        size_k=size_k,
        size_n=size_n,
        expected_is_zp_float=True,
    )


def test_marlin_linear_kernel_scalar_support_matrix_includes_fp4() -> None:
    case = DENSE_WRITEBACK_CLASS_CASE_BY_NAME["marlin_linear_kernel"]
    assert {
        "uint4",
        "uint8",
        "uint4b8",
        "uint8b128",
        "float8_e4m3fn",
        "float4_e2m1f",
    }.issubset(set(case.scalar_type_names))
    assert "float4_e2m1f" in case.quant_names
    assert "nvfp4" not in case.quant_names
    assert "mxfp4" not in case.quant_names


@pytest.mark.parametrize(
    ("quant_name", "group_size"),
    [
        pytest.param(quant_name, group_size, id=f"{quant_name}_g{group_size}")
        for quant_name in ("uint4b8", "uint8b128", "fp8")
        for group_size in DENSE_WRITEBACK_CLASS_CASE_BY_NAME[
            "marlin_linear_kernel"
        ].default_group_sizes
        if is_dense_group_size_supported(quant_name, group_size, 128)
    ],
)
def test_marlin_linear_kernel_non_zp_quant_uses_c_tmp_and_false_zp_flag(
    monkeypatch: pytest.MonkeyPatch,
    quant_name: str,
    group_size: int,
):
    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 128
    size_n = 128
    activation = torch.randn(2, size_k, dtype=torch.float16)
    weight = torch.randn(size_k, size_n, dtype=torch.float16)

    case = prepare_marlin_linear_kernel_case(
        quant_name=quant_name,
        group_size=group_size,
        activation=activation,
        weight=weight,
    )

    assert getattr(case.kernel, "is_zp_float", None) is False
    assert case.layer.w_zp.numel() == 0
    _assert_fresh_kernel_c_tmp(case.kernel)
    _assert_kernel_apply_keeps_empty_c_tmp(
        lambda x: case.kernel.apply_weights(case.layer, x),
        case.kernel,
        records,
        size_k=size_k,
        size_n=size_n,
        expected_is_zp_float=False,
    )


@pytest.mark.parametrize(
    ("quant_name", "group_size"),
    [
        pytest.param(quant_name, group_size, id=f"{quant_name}_g{group_size}")
        for quant_name in DENSE_WRITEBACK_CLASS_CASE_BY_NAME[
            "gptq_marlin_linear_method"
        ].quant_names
        for group_size in DENSE_WRITEBACK_CLASS_CASE_BY_NAME[
            "gptq_marlin_linear_method"
        ].default_group_sizes
        if is_dense_group_size_supported(quant_name, group_size, 128)
    ],
)
def test_gptq_marlin_linear_method_class_path_uses_kernel_c_tmp(
    monkeypatch: pytest.MonkeyPatch,
    quant_name: str,
    group_size: int,
):
    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 128
    size_n = 128
    layer, method = _make_gptq_marlin_linear_method_case(
        quant_name=quant_name,
        group_size=group_size,
        size_k=size_k,
        size_n=size_n,
    )

    assert method.kernel.is_zp_float is False
    assert layer.qzeros.numel() == 0
    _assert_fresh_kernel_c_tmp(method.kernel)
    _assert_kernel_apply_keeps_empty_c_tmp(
        lambda x: method.apply(layer, x, None),
        method.kernel,
        records,
        size_k=size_k,
        size_n=size_n,
        expected_is_zp_float=False,
    )


@pytest.mark.parametrize(
    ("quant_name", "group_size"),
    [
        pytest.param(quant_name, group_size, id=f"{quant_name}_g{group_size}")
        for quant_name in DENSE_WRITEBACK_CLASS_CASE_BY_NAME[
            "awq_marlin_linear_method"
        ].quant_names
        for group_size in DENSE_WRITEBACK_CLASS_CASE_BY_NAME[
            "awq_marlin_linear_method"
        ].default_group_sizes
        if is_dense_group_size_supported(quant_name, group_size, 128)
    ],
)
def test_awq_marlin_linear_method_class_path_converts_zp_and_uses_kernel_c_tmp(
    monkeypatch: pytest.MonkeyPatch,
    quant_name: str,
    group_size: int,
):
    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 128
    size_n = 128
    layer, method, num_groups = _make_awq_marlin_linear_method_case(
        quant_name=quant_name,
        group_size=group_size,
        size_k=size_k,
        size_n=size_n,
    )

    assert layer.qzeros.dtype == torch.float16
    assert layer.qzeros.shape == (num_groups, size_n)
    assert layer.qzeros.is_contiguous()
    assert method.kernel.is_zp_float is True
    _assert_fresh_kernel_c_tmp(method.kernel)
    _assert_kernel_apply_keeps_empty_c_tmp(
        lambda x: method.apply(layer, x, None),
        method.kernel,
        records,
        size_k=size_k,
        size_n=size_n,
        expected_is_zp_float=True,
    )


@pytest.mark.parametrize(
    ("quant_name", "group_size"),
    [
        pytest.param(quant_name, group_size, id=f"{quant_name}_g{group_size}")
        for quant_name in ("uint4b8", "uint8b128")
        for group_size in _GROUP_SIZES
        if is_dense_group_size_supported(quant_name, group_size, 128)
    ],
)
def test_compressed_tensors_wna16_symmetric_uses_marlin_kernel_c_tmp(
    monkeypatch: pytest.MonkeyPatch,
    quant_name: str,
    group_size: int,
):
    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 128
    size_n = 128
    layer, scheme, _num_groups = _make_compressed_tensors_wna16_layer_case(
        quant_name=quant_name,
        group_size=group_size,
        size_k=size_k,
        size_n=size_n,
    )

    scheme.process_weights_after_loading(layer)
    assert scheme.kernel.is_zp_float is False
    assert getattr(layer, scheme.kernel.w_zp_name).numel() == 0
    _assert_fresh_kernel_c_tmp(scheme.kernel)
    _assert_kernel_apply_keeps_empty_c_tmp(
        lambda x: scheme.apply_weights(layer, x, None),
        scheme.kernel,
        records,
        size_k=size_k,
        size_n=size_n,
        expected_is_zp_float=False,
    )


def _make_fp8_layer(*, size_k: int = 64, size_n: int = 64) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.logical_widths = [size_n]
    layer.orig_dtype = torch.float16
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.zeros(size_n, size_k, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones(size_n, dtype=torch.float32), requires_grad=False),
    )
    return layer


def _make_fp8_block_layer(*, size_k: int, size_n: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.logical_widths = [size_n]
    layer.orig_dtype = torch.float16
    layer.weight_block_size = [64, 128]
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.zeros(size_n, size_k, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(
            torch.ones(
                (size_n + 63) // 64,
                size_k // 128,
                dtype=torch.float32,
            ),
            requires_grad=False,
        ),
    )
    return layer


def test_fp8_scaled_mm_kernel_persists_c_tmp_and_apply_uses_it(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
        FP8ScaledMMLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
        MarlinFP8ScaledMMLinearKernel,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kFp8DynamicTokenSym,
        kFp8StaticChannelSym,
    )

    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 64
    size_n = 64
    layer = _make_fp8_layer(size_k=size_k, size_n=size_n)
    kernel = MarlinFP8ScaledMMLinearKernel(
        FP8ScaledMMLinearLayerConfig(
            weight_quant_key=kFp8StaticChannelSym,
            activation_quant_key=kFp8DynamicTokenSym,
            out_dtype=None,
        ),
        ["weight", "weight_scale", "input_scale", "input_scale_ub"],
    )

    kernel.process_weights_after_loading(layer)
    _assert_fresh_layer_c_tmp(layer)
    _assert_layer_apply_keeps_empty_c_tmp(
        lambda x: kernel.apply_weights(layer, x),
        layer,
        records,
        size_k=size_k,
        size_n=size_n,
    )


def test_compressed_tensors_w8a16_fp8_persists_c_tmp_and_apply_uses_it(
    monkeypatch: pytest.MonkeyPatch,
):
    from compressed_tensors.quantization import QuantizationStrategy

    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
        CompressedTensorsW8A16Fp8,
    )

    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 64
    size_n = 64
    layer = _make_fp8_layer(size_k=size_k, size_n=size_n)
    scheme = CompressedTensorsW8A16Fp8(
        SimpleNamespace(strategy=QuantizationStrategy.CHANNEL, block_structure=None),
        is_static_input_scheme=False,
    )

    scheme.process_weights_after_loading(layer)
    _assert_fresh_layer_c_tmp(layer)
    _assert_layer_apply_keeps_empty_c_tmp(
        lambda x: scheme.apply_weights(layer, x),
        layer,
        records,
        size_k=size_k,
        size_n=size_n,
    )


def _make_nvfp4_layer(*, size_k: int = 64, size_n: int = 64) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.logical_widths = [size_n]
    layer.params_dtype = torch.float16
    layer.orig_dtype = torch.float16
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(
            torch.zeros(size_n, size_k // 2, dtype=torch.uint8),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_global_scale",
        torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            torch.ones(
                size_n,
                size_k // 16,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "input_global_scale_inv",
        torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False),
    )
    layer.register_parameter(
        "alpha",
        torch.nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False),
    )
    return layer


def _make_mxfp4_layer(*, size_k: int = 64, size_n: int = 64) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.logical_widths = [size_n]
    layer.params_dtype = torch.float16
    layer.orig_dtype = torch.float16
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(
            torch.zeros(size_n, size_k // 2, dtype=torch.uint8),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            torch.ones(size_n, size_k // 32, dtype=torch.uint8),
            requires_grad=False,
        ),
    )
    return layer


def test_compressed_tensors_w4a16_nvfp4_persists_c_tmp_and_apply_uses_it(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import (  # noqa: E501
        CompressedTensorsW4A16Fp4,
    )

    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 64
    size_n = 64
    layer = _make_nvfp4_layer(size_k=size_k, size_n=size_n)
    scheme = CompressedTensorsW4A16Fp4()

    scheme.process_weights_after_loading(layer)
    _assert_fresh_layer_c_tmp(layer)
    _assert_layer_apply_keeps_empty_c_tmp(
        lambda x: scheme.apply_weights(layer, x),
        layer,
        records,
        size_k=size_k,
        size_n=size_n,
    )


def test_compressed_tensors_w4a16_mxfp4_persists_c_tmp_and_apply_uses_it(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_mxfp4 import (  # noqa: E501
        CompressedTensorsW4A16Mxfp4,
    )

    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 64
    size_n = 64
    layer = _make_mxfp4_layer(size_k=size_k, size_n=size_n)
    scheme = CompressedTensorsW4A16Mxfp4()

    scheme.process_weights_after_loading(layer)
    _assert_fresh_layer_c_tmp(layer)
    _assert_layer_apply_keeps_empty_c_tmp(
        lambda x: scheme.apply_weights(layer, x),
        layer,
        records,
        size_k=size_k,
        size_n=size_n,
    )


def test_nvfp4_utils_marlin_backend_reuses_layer_c_tmp(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        NvFp4LinearBackend,
        apply_nvfp4_linear,
        convert_to_nvfp4_linear_kernel_format,
    )

    records = _install_dense_marlin_cpu_mocks(monkeypatch)
    size_k = 64
    size_n = 64
    layer = _make_nvfp4_layer(size_k=size_k, size_n=size_n)

    layer.weight = torch.nn.Parameter(layer.weight_packed.data, requires_grad=False)
    del layer.weight_packed
    convert_to_nvfp4_linear_kernel_format(NvFp4LinearBackend.MARLIN, layer)
    _assert_fresh_layer_c_tmp(layer)
    _assert_layer_apply_keeps_empty_c_tmp(
        lambda x: apply_nvfp4_linear(NvFp4LinearBackend.MARLIN, layer, x),
        layer,
        records,
        size_k=size_k,
        size_n=size_n,
    )


_DENSE_MATRIX_CACHE_LIMIT = 16
_DENSE_MATRIX_CACHE: OrderedDict[
    tuple[str, str, int, int, int],
    tuple[Any, Any, bool],
] = OrderedDict()


def _make_compressed_tensors_fp8_layer_case(
    *,
    group_size: int,
    size_k: int,
    size_n: int,
) -> tuple[torch.nn.Module, Any]:
    from compressed_tensors.quantization import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )

    from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a16_fp8 import (  # noqa: E501
        CompressedTensorsW8A16Fp8,
    )

    if group_size == -1:
        weight_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.CHANNEL,
        )
    elif group_size == 128:
        weight_quant = QuantizationArgs(
            num_bits=8,
            type=QuantizationType.FLOAT,
            symmetric=True,
            strategy=QuantizationStrategy.BLOCK,
            block_structure=[64, 128],
            dynamic=False,
        )
    else:
        raise ValueError(f"Unsupported FP8 group_size={group_size}")

    scheme = CompressedTensorsW8A16Fp8(
        weight_quant,
        is_static_input_scheme=False,
    )
    layer = torch.nn.Module()
    scheme.create_weights(
        layer=layer,
        input_size_per_partition=size_k,
        output_partition_sizes=[size_n],
        input_size=size_k,
        output_size=size_n,
        params_dtype=torch.float16,
        weight_loader=lambda param, loaded_weight: None,
    )
    layer.weight.data.zero_()
    layer.weight_scale.data.fill_(1.0)
    scheme.process_weights_after_loading(layer)
    return layer, scheme


def _prepare_dense_matrix_case(
    case: DenseWritebackMatrixCase,
) -> tuple[Any, Any, bool]:
    key = (
        case.class_case.name,
        case.quant_name,
        case.group_size,
        case.shape.size_k,
        case.shape.size_n,
    )
    cached = _DENSE_MATRIX_CACHE.get(key)
    if cached is not None:
        _DENSE_MATRIX_CACHE.move_to_end(key)
        return cached

    size_k = case.shape.size_k
    size_n = case.shape.size_n
    activation = torch.zeros(1, size_k, dtype=torch.float16)
    weight = torch.zeros(size_k, size_n, dtype=torch.float16)
    expected_is_zp_float = case.quant_name in case.class_case.zp_quant_names

    if case.class_case.name == "marlin_linear_kernel":
        prepared = prepare_marlin_linear_kernel_case(
            quant_name=case.quant_name,
            group_size=case.group_size,
            activation=activation,
            weight=weight,
        )
        result = (
            lambda x, prepared=prepared: prepared.kernel.apply_weights(
                prepared.layer, x
            ),
            prepared.kernel,
            expected_is_zp_float,
        )
    elif case.class_case.name == "gptq_marlin_linear_method":
        layer, method = _make_gptq_marlin_linear_method_case(
            quant_name=case.quant_name,
            group_size=case.group_size,
            size_k=size_k,
            size_n=size_n,
        )
        result = (
            lambda x, layer=layer, method=method: method.apply(layer, x, None),
            method.kernel,
            expected_is_zp_float,
        )
    elif case.class_case.name == "awq_marlin_linear_method":
        layer, method, _num_groups = _make_awq_marlin_linear_method_case(
            quant_name=case.quant_name,
            group_size=case.group_size,
            size_k=size_k,
            size_n=size_n,
        )
        result = (
            lambda x, layer=layer, method=method: method.apply(layer, x, None),
            method.kernel,
            expected_is_zp_float,
        )
    elif case.class_case.name == "compressed_tensors_wna16":
        layer, scheme, _num_groups = _make_compressed_tensors_wna16_layer_case(
            quant_name=case.quant_name,
            group_size=case.group_size,
            size_k=size_k,
            size_n=size_n,
        )
        scheme.process_weights_after_loading(layer)
        result = (
            lambda x, layer=layer, scheme=scheme: scheme.apply_weights(
                layer, x, None
            ),
            scheme.kernel,
            expected_is_zp_float,
        )
    elif case.class_case.name == "marlin_fp8_scaled_mm":
        from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (  # noqa: E501
            FP8ScaledMMLinearLayerConfig,
        )
        from vllm.model_executor.kernels.linear.scaled_mm.marlin import (
            MarlinFP8ScaledMMLinearKernel,
        )
        from vllm.model_executor.layers.quantization.utils.quant_utils import (
            kFp8DynamicTokenSym,
            kFp8Static128BlockSym,
            kFp8StaticChannelSym,
        )

        block_quant = case.group_size == 128
        layer = (
            _make_fp8_block_layer(size_k=size_k, size_n=size_n)
            if block_quant
            else _make_fp8_layer(size_k=size_k, size_n=size_n)
        )
        kernel = MarlinFP8ScaledMMLinearKernel(
            FP8ScaledMMLinearLayerConfig(
                weight_quant_key=(
                    kFp8Static128BlockSym if block_quant else kFp8StaticChannelSym
                ),
                activation_quant_key=kFp8DynamicTokenSym,
                out_dtype=None,
            ),
            ["weight", "weight_scale", "input_scale", "input_scale_ub"],
        )
        kernel.process_weights_after_loading(layer)
        result = (
            lambda x, layer=layer, kernel=kernel: kernel.apply_weights(layer, x),
            layer,
            expected_is_zp_float,
        )
    elif case.class_case.name == "compressed_tensors_w8a16_fp8":
        layer, scheme = _make_compressed_tensors_fp8_layer_case(
            group_size=case.group_size,
            size_k=size_k,
            size_n=size_n,
        )
        result = (
            lambda x, layer=layer, scheme=scheme: scheme.apply_weights(layer, x),
            layer,
            expected_is_zp_float,
        )
    elif case.class_case.name == "compressed_tensors_w4a16_nvfp4":
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import (  # noqa: E501
            CompressedTensorsW4A16Fp4,
        )

        layer = _make_nvfp4_layer(size_k=size_k, size_n=size_n)
        scheme = CompressedTensorsW4A16Fp4()
        scheme.process_weights_after_loading(layer)
        result = (
            lambda x, layer=layer, scheme=scheme: scheme.apply_weights(layer, x),
            layer,
            expected_is_zp_float,
        )
    elif case.class_case.name == "compressed_tensors_w4a16_mxfp4":
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_mxfp4 import (  # noqa: E501
            CompressedTensorsW4A16Mxfp4,
        )

        layer = _make_mxfp4_layer(size_k=size_k, size_n=size_n)
        scheme = CompressedTensorsW4A16Mxfp4()
        scheme.process_weights_after_loading(layer)
        result = (
            lambda x, layer=layer, scheme=scheme: scheme.apply_weights(layer, x),
            layer,
            expected_is_zp_float,
        )
    else:
        raise AssertionError(f"Unhandled dense class {case.class_case.name!r}")

    _DENSE_MATRIX_CACHE[key] = result
    _DENSE_MATRIX_CACHE.move_to_end(key)
    while len(_DENSE_MATRIX_CACHE) > _DENSE_MATRIX_CACHE_LIMIT:
        _DENSE_MATRIX_CACHE.popitem(last=False)
    return result


def _dense_matrix_row(
    case: DenseWritebackMatrixCase,
    *,
    status: str,
    expected_is_zp_float: bool | None = None,
    c_tmp_numel: int | None = None,
    reason: str = "",
) -> dict[str, object]:
    return {
        "case_id": case.id,
        "dense_class": case.class_case.name,
        "quant": case.quant_name,
        "group_size": case.group_size,
        "shape_id": case.shape.name,
        "M": case.shape.size_m,
        "K": case.shape.size_k,
        "N": case.shape.size_n,
        "status": status,
        "is_zp_float": expected_is_zp_float,
        "c_tmp_numel": c_tmp_numel,
        "reason": reason,
    }


def test_dense_writeback_class_full_matrix_post_load_apply_path(
    monkeypatch: pytest.MonkeyPatch,
):
    records = _install_dense_marlin_cpu_mocks(monkeypatch, fast_repack=True)
    _DENSE_FULL_MATRIX_JSONL.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    status_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    coverage: dict[str, Counter[Any]] = {
        "dense_class": Counter(),
        "quant": Counter(),
        "group_size": Counter(),
        "shape_id": Counter(),
    }
    ok_coverage: dict[str, Counter[Any]] = {
        "dense_class": Counter(),
        "quant": Counter(),
        "group_size": Counter(),
        "shape_id": Counter(),
    }
    first_failure: dict[str, object] | None = None
    max_c_tmp_numel = 0

    with _DENSE_FULL_MATRIX_JSONL.open("w", encoding="utf-8") as jsonl:
        for index, matrix_case in enumerate(iter_dense_writeback_matrix(), start=1):
            status_counts["total"] += 1
            coverage["dense_class"][matrix_case.class_case.name] += 1
            coverage["quant"][matrix_case.quant_name] += 1
            coverage["group_size"][matrix_case.group_size] += 1
            coverage["shape_id"][matrix_case.shape.name] += 1

            if not matrix_case.supported:
                status_counts["SKIP"] += 1
                skip_reasons[matrix_case.reason] += 1
                continue

            expected_is_zp_float: bool | None = None
            c_tmp_numel: int | None = None
            try:
                records.clear()
                apply_fn, owner, expected_is_zp_float = _prepare_dense_matrix_case(
                    matrix_case
                )
                x = torch.zeros(
                    matrix_case.shape.size_m,
                    matrix_case.shape.size_k,
                    dtype=torch.float16,
                )
                output = apply_fn(x)

                assert output.shape == (
                    matrix_case.shape.size_m,
                    matrix_case.shape.size_n,
                )
                assert len(records) == 1
                assert records[0].size_m == matrix_case.shape.size_m
                assert records[0].size_k == matrix_case.shape.size_k
                assert records[0].size_n == matrix_case.shape.size_n
                assert records[0].use_fp32_reduce is False
                assert records[0].is_zp_float is expected_is_zp_float

                assert hasattr(owner, "c_tmp")
                assert owner.c_tmp.dtype == torch.float32
                assert owner.c_tmp.is_contiguous()
                assert owner.c_tmp.numel() == 0
                assert records[0].c_tmp is owner.c_tmp
                c_tmp_numel = int(owner.c_tmp.numel())
                max_c_tmp_numel = max(max_c_tmp_numel, c_tmp_numel)
                status = "OK"
                reason = ""
                status_counts["OK"] += 1
                ok_coverage["dense_class"][matrix_case.class_case.name] += 1
                ok_coverage["quant"][matrix_case.quant_name] += 1
                ok_coverage["group_size"][matrix_case.group_size] += 1
                ok_coverage["shape_id"][matrix_case.shape.name] += 1
            except Exception as exc:
                status = "ERR"
                reason = str(exc).splitlines()[0]
                status_counts["ERR"] += 1
                if first_failure is None:
                    first_failure = _dense_matrix_row(
                        matrix_case,
                        status=status,
                        expected_is_zp_float=expected_is_zp_float,
                        c_tmp_numel=c_tmp_numel,
                        reason=reason,
                    )

            jsonl.write(
                json.dumps(
                    _dense_matrix_row(
                        matrix_case,
                        status=status,
                        expected_is_zp_float=expected_is_zp_float,
                        c_tmp_numel=c_tmp_numel,
                        reason=reason,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            if index % 10000 == 0:
                jsonl.flush()

    elapsed = time.time() - started
    summary = {
        "matrix": "class x quant x group x shape",
        "total": status_counts["total"],
        "OK": status_counts["OK"],
        "SKIP": status_counts["SKIP"],
        "ERR": status_counts["ERR"],
        "skip_reasons": _counter_to_json(skip_reasons),
        "coverage": {key: _counter_to_json(counter) for key, counter in coverage.items()},
        "ok_coverage": {
            key: _counter_to_json(counter) for key, counter in ok_coverage.items()
        },
        "max_c_tmp_numel": max_c_tmp_numel,
        "first_failure": first_failure,
        "jsonl": str(_DENSE_FULL_MATRIX_JSONL),
        "elapsed_seconds": round(elapsed, 3),
    }
    _DENSE_FULL_MATRIX_SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _DENSE_MATRIX_CACHE.clear()

    if first_failure is not None:
        pytest.fail(
            "dense writeback full matrix loop had "
            f"{status_counts['ERR']} ERR cases; first={first_failure}"
        )
