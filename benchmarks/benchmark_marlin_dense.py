from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)

try:
    from common import (
        banner,
        require_matching_cuda_benchmark_runtime,
        format_float,
        time_cuda_callable,
        timestamp,
    )
except ModuleNotFoundError:
    from benchmarks.common import (
        banner,
        require_matching_cuda_benchmark_runtime,
        format_float,
        time_cuda_callable,
        timestamp,
    )

from tests.calibration import (
    format_capability,
    runtime_capability,
    source_target_cuda_arch_arg,
    source_target_label,
)
from tests.helpers import (
    _quantize_unsigned_with_bias,
    _quantize_uint4_with_zero_point,
    _quantize_uint8_with_zero_point,
    awq_pack,
    gptq_pack,
    pack_cols,
    prepare_marlin_linear_kernel_case,
)
from tests.writeback_marlin_cases import (
    DENSE_ALL_QUANT_NAMES,
    DENSE_BENCHMARK_SHAPE_CASES,
    DENSE_IRREGULAR_SHAPE_CASES,
    DENSE_REGULAR_SHAPE_CASES,
    DENSE_WRITEBACK_CLASS_CASES,
    DenseShapeCase,
    DenseWritebackMatrixCase,
    WRITEBACK_GROUP_SIZE_VALUES,
    dense_auto_cta_geometry_label,
    dense_auto_split_k,
    iter_dense_writeback_matrix,
)
from vllm.scalar_type import scalar_types


_DENSE_CLASS_CHOICES = tuple(case.name for case in DENSE_WRITEBACK_CLASS_CASES)
_SHAPE_SUITE_CASES = {
    "regular": DENSE_REGULAR_SHAPE_CASES,
    "irregular": DENSE_IRREGULAR_SHAPE_CASES,
    "all": DENSE_BENCHMARK_SHAPE_CASES,
}
_DENSE_CSV_FIELDNAMES = [
    "dense_class",
    "quant",
    "group_size",
    "shape_id",
    "M",
    "K",
    "N",
    "auto_cta_geometry",
    "auto_split_k",
    "status",
    "marlin_us",
    "flops",
    "marlin_tflops",
    "all_finite",
    "reason",
]


def _dense_flops(shape: DenseShapeCase) -> int:
    return 2 * shape.size_m * shape.size_k * shape.size_n


def _format_tflops(flops: int, latency_us: float) -> str:
    return f"{flops / (latency_us * 1_000_000):.6f}"


def _set_parameter_data(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    param = getattr(layer, name)
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(
            f"{name} expected shape {tuple(param.shape)}, got {tuple(value.shape)}"
        )
    param.data.copy_(value.to(device=param.device, dtype=param.dtype))


def _make_compressed_tensors_wna16_layer_case(
    *,
    quant_name: str,
    group_size: int,
    size_k: int,
    size_n: int,
) -> tuple[torch.nn.Module, Any]:
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

        def quantize(weight: torch.Tensor, gs: int):
            q_weight, scales = _quantize_unsigned_with_bias(
                weight, gs, scalar_types.uint4b8.bias
            )
            return q_weight, scales, None

    elif quant_name == "uint8b128":
        num_bits = 8

        def quantize(weight: torch.Tensor, gs: int):
            q_weight, scales = _quantize_unsigned_with_bias(
                weight, gs, scalar_types.uint8b128.bias
            )
            return q_weight, scales, None

    else:
        raise ValueError(f"Unsupported quant_name={quant_name!r}")

    scheme = CompressedTensorsWNA16(
        strategy="channel" if group_size == -1 else "group",
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
    weight = torch.zeros(size_k, size_n, dtype=torch.float16, device="cuda")
    q_weight, scales, zero_points = quantize(weight, group_size)
    packed_weight = gptq_pack(q_weight, num_bits, size_k, size_n)
    _set_parameter_data(layer, "weight_packed", packed_weight.t().contiguous())
    _set_parameter_data(layer, "weight_scale", scales.t().contiguous())
    _set_parameter_data(
        layer, "weight_shape", torch.tensor([size_k, size_n], dtype=torch.int64)
    )
    if zero_points is not None:
        packed_zero_points = pack_cols(
            zero_points, num_bits, scales.shape[0], size_n
        )
        _set_parameter_data(
            layer, "weight_zero_point", packed_zero_points.t().contiguous()
        )
    layer.to("cuda")
    scheme.process_weights_after_loading(layer)
    return layer, scheme


def _make_gptq_marlin_linear_method_case(
    *,
    quant_name: str,
    group_size: int,
    size_k: int,
    size_n: int,
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
        raise ValueError(f"Unsupported GPTQ dense quant_name={quant_name!r}")

    method = GPTQMarlinLinearMethod(
        GPTQMarlinConfig(
            weight_bits=num_bits,
            group_size=group_size,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config={},
        )
    )
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
    weight = torch.zeros(size_k, size_n, dtype=torch.float16, device="cuda")
    q_weight, scales = _quantize_unsigned_with_bias(weight, group_size, bias)
    _set_parameter_data(layer, "qweight", gptq_pack(q_weight, num_bits, size_k, size_n))
    _set_parameter_data(layer, "scales", scales)
    layer.to("cuda")
    method.process_weights_after_loading(layer)
    return layer, method


def _make_awq_marlin_linear_method_case(
    *,
    quant_name: str,
    group_size: int,
    size_k: int,
    size_n: int,
) -> tuple[torch.nn.Module, Any]:
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
        raise ValueError(f"Unsupported AWQ dense quant_name={quant_name!r}")

    method = AWQMarlinLinearMethod(
        AWQMarlinConfig(
            weight_bits=num_bits,
            group_size=group_size,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=None,
            full_config={},
        )
    )
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
    weight = torch.zeros(size_k, size_n, dtype=torch.float16, device="cuda")
    q_weight, scales, zero_points = quantize(weight, group_size)
    num_groups = zero_points.shape[0]
    _set_parameter_data(layer, "qweight", awq_pack(q_weight, num_bits, size_k, size_n))
    _set_parameter_data(layer, "scales", scales)
    _set_parameter_data(
        layer, "qzeros", awq_pack(zero_points, num_bits, num_groups, size_n)
    )
    layer.to("cuda")
    method.process_weights_after_loading(layer)
    return layer, method


def _make_fp8_layer(*, size_k: int, size_n: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.logical_widths = [size_n]
    layer.orig_dtype = torch.float16
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(
            torch.zeros(size_n, size_k, dtype=torch.float8_e4m3fn, device="cuda"),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(torch.ones(size_n, dtype=torch.float32, device="cuda"), requires_grad=False),
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
            torch.zeros(size_n, size_k, dtype=torch.float8_e4m3fn, device="cuda"),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale_inv",
        torch.nn.Parameter(
            torch.ones((size_n + 63) // 64, size_k // 128, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        ),
    )
    return layer


def _make_compressed_tensors_fp8_layer_case(
    *,
    group_size: int,
    size_k: int,
    size_n: int,
) -> tuple[torch.nn.Module, Any]:
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
    scheme = CompressedTensorsW8A16Fp8(weight_quant, is_static_input_scheme=False)
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
    layer.to("cuda")
    layer.weight.data.zero_()
    layer.weight_scale.data.fill_(1.0)
    scheme.process_weights_after_loading(layer)
    return layer, scheme


def _make_nvfp4_layer(*, size_k: int, size_n: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.logical_widths = [size_n]
    layer.params_dtype = torch.float16
    layer.orig_dtype = torch.float16
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(
            torch.zeros(size_n, size_k // 2, dtype=torch.uint8, device="cuda"),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_global_scale",
        torch.nn.Parameter(torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            torch.ones(size_n, size_k // 16, dtype=torch.float8_e4m3fn, device="cuda"),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "input_global_scale_inv",
        torch.nn.Parameter(torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False),
    )
    layer.register_parameter(
        "alpha",
        torch.nn.Parameter(torch.ones(1, dtype=torch.float32, device="cuda"), requires_grad=False),
    )
    return layer


def _make_mxfp4_layer(*, size_k: int, size_n: int) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.input_size_per_partition = size_k
    layer.output_size_per_partition = size_n
    layer.logical_widths = [size_n]
    layer.params_dtype = torch.float16
    layer.orig_dtype = torch.float16
    layer.register_parameter(
        "weight_packed",
        torch.nn.Parameter(
            torch.zeros(size_n, size_k // 2, dtype=torch.uint8, device="cuda"),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        torch.nn.Parameter(
            torch.ones(size_n, size_k // 32, dtype=torch.uint8, device="cuda"),
            requires_grad=False,
        ),
    )
    return layer


def _prepare_dense_class_case(
    case: DenseWritebackMatrixCase,
) -> Callable[[torch.Tensor], torch.Tensor]:
    size_k = case.shape.size_k
    size_n = case.shape.size_n
    weight = torch.zeros(size_k, size_n, dtype=torch.float16, device="cuda")
    activation = torch.zeros(1, size_k, dtype=torch.float16, device="cuda")

    if case.class_case.name == "marlin_linear_kernel":
        prepared = prepare_marlin_linear_kernel_case(
            quant_name=case.quant_name,
            group_size=case.group_size,
            activation=activation,
            weight=weight,
        )
        return lambda x, prepared=prepared: prepared.kernel.apply_weights(
            prepared.layer, x
        )
    if case.class_case.name == "gptq_marlin_linear_method":
        layer, method = _make_gptq_marlin_linear_method_case(
            quant_name=case.quant_name,
            group_size=case.group_size,
            size_k=size_k,
            size_n=size_n,
        )
        return lambda x, layer=layer, method=method: method.apply(layer, x, None)
    if case.class_case.name == "awq_marlin_linear_method":
        layer, method = _make_awq_marlin_linear_method_case(
            quant_name=case.quant_name,
            group_size=case.group_size,
            size_k=size_k,
            size_n=size_n,
        )
        return lambda x, layer=layer, method=method: method.apply(layer, x, None)
    if case.class_case.name == "compressed_tensors_wna16":
        layer, scheme = _make_compressed_tensors_wna16_layer_case(
            quant_name=case.quant_name,
            group_size=case.group_size,
            size_k=size_k,
            size_n=size_n,
        )
        return lambda x, layer=layer, scheme=scheme: scheme.apply_weights(layer, x, None)
    if case.class_case.name == "marlin_fp8_scaled_mm":
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
        return lambda x, layer=layer, kernel=kernel: kernel.apply_weights(layer, x)
    if case.class_case.name == "compressed_tensors_w8a16_fp8":
        layer, scheme = _make_compressed_tensors_fp8_layer_case(
            group_size=case.group_size,
            size_k=size_k,
            size_n=size_n,
        )
        return lambda x, layer=layer, scheme=scheme: scheme.apply_weights(layer, x)
    if case.class_case.name == "compressed_tensors_w4a16_nvfp4":
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import (  # noqa: E501
            CompressedTensorsW4A16Fp4,
        )

        layer = _make_nvfp4_layer(size_k=size_k, size_n=size_n)
        scheme = CompressedTensorsW4A16Fp4()
        scheme.process_weights_after_loading(layer)
        return lambda x, layer=layer, scheme=scheme: scheme.apply_weights(layer, x)
    if case.class_case.name == "compressed_tensors_w4a16_mxfp4":
        from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_mxfp4 import (  # noqa: E501
            CompressedTensorsW4A16Mxfp4,
        )

        layer = _make_mxfp4_layer(size_k=size_k, size_n=size_n)
        scheme = CompressedTensorsW4A16Mxfp4()
        scheme.process_weights_after_loading(layer)
        return lambda x, layer=layer, scheme=scheme: scheme.apply_weights(layer, x)
    raise AssertionError(f"Unhandled dense class {case.class_case.name!r}")


def _iter_filtered_matrix(
    args: argparse.Namespace,
) -> Iterator[DenseWritebackMatrixCase]:
    allowed_shapes = {shape.name for shape in _SHAPE_SUITE_CASES[args.shape_suite]}
    class_cases = tuple(
        case for case in DENSE_WRITEBACK_CLASS_CASES if case.name in args.dense_classes
    )
    shapes = tuple(
        sorted(
            (
                shape
                for shape in _SHAPE_SUITE_CASES[args.shape_suite]
                if shape.name in allowed_shapes
            ),
            key=lambda shape: (
                shape.size_k,
                shape.size_n,
                shape.size_m,
                shape.name,
            ),
        )
    )
    yielded = 0
    for case in iter_dense_writeback_matrix(
        class_cases=class_cases,
        quant_names=tuple(args.quant_types),
        group_sizes=tuple(args.group_sizes),
        shapes=shapes,
    ):
        yield case
        yielded += 1
        if args.max_cases is not None and yielded >= args.max_cases:
            return


def _count_filtered_matrix(args: argparse.Namespace) -> int:
    return sum(1 for _case in _iter_filtered_matrix(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dense Marlin writeback class matrix."
    )
    parser.add_argument("--preset", default="full", help="Compatibility option.")
    parser.add_argument(
        "--dense-classes",
        nargs="+",
        choices=_DENSE_CLASS_CHOICES,
        default=list(_DENSE_CLASS_CHOICES),
    )
    parser.add_argument(
        "--quant-types",
        nargs="+",
        default=list(DENSE_ALL_QUANT_NAMES),
    )
    parser.add_argument(
        "--group-sizes",
        nargs="+",
        type=int,
        default=list(WRITEBACK_GROUP_SIZE_VALUES),
    )
    parser.add_argument(
        "--shape-suite",
        choices=("regular", "irregular", "all"),
        default="all",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--omit-skip",
        action="store_true",
        help="Do not save or print SKIP rows; still count them in the summary.",
    )
    return parser.parse_args()


def _skip_row(case: DenseWritebackMatrixCase, reason: str) -> dict[str, str]:
    return {
        "dense_class": case.class_case.name,
        "quant": case.quant_name,
        "group_size": str(case.group_size),
        "shape_id": case.shape.name,
        "M": str(case.shape.size_m),
        "K": str(case.shape.size_k),
        "N": str(case.shape.size_n),
        "auto_cta_geometry": "n/a",
        "auto_split_k": "n/a",
        "status": "SKIP",
        "marlin_us": "n/a",
        "flops": "n/a",
        "marlin_tflops": "n/a",
        "all_finite": "n/a",
        "reason": reason,
    }


def _error_row(case: DenseWritebackMatrixCase, exc: BaseException) -> dict[str, str]:
    row = _skip_row(case, str(exc).splitlines()[0])
    row["status"] = "ERR"
    row["auto_cta_geometry"] = dense_auto_cta_geometry_label(case.shape)
    row["auto_split_k"] = str(dense_auto_split_k(case.shape))
    row["flops"] = "n/a"
    row["marlin_tflops"] = "n/a"
    return row


def _run_matrix_case(
    case: DenseWritebackMatrixCase,
    *,
    warmup_iters: int,
    iters: int,
    check: bool,
    prepared_cache: dict[str, object],
) -> dict[str, str]:
    if not case.supported:
        return _skip_row(case, case.reason)

    try:
        prepare_key = (
            case.class_case.name,
            case.quant_name,
            case.group_size,
            case.shape.size_k,
            case.shape.size_n,
        )
        if prepared_cache.get("prepare_key") != prepare_key:
            prepared_cache.clear()
            run_marlin = _prepare_dense_class_case(case)
            prepared_cache["prepare_key"] = prepare_key
            prepared_cache["run_marlin"] = run_marlin

        input_key = (case.shape.size_m, case.shape.size_k)
        if prepared_cache.get("input_key") != input_key:
            x = torch.zeros(
                case.shape.size_m,
                case.shape.size_k,
                dtype=torch.float16,
                device="cuda",
            )
            prepared_cache["input_key"] = input_key
            prepared_cache["x"] = x
        else:
            x = prepared_cache["x"]
        run_marlin = prepared_cache["run_marlin"]

        all_finite = "n/a"
        if check:
            output = run_marlin(x)
            all_finite = "yes" if torch.isfinite(output).all().item() else "no"
            if output.shape != (case.shape.size_m, case.shape.size_n):
                raise AssertionError(
                    f"output shape {tuple(output.shape)} != "
                    f"{(case.shape.size_m, case.shape.size_n)}"
                )
        stats = time_cuda_callable(
            lambda: run_marlin(x),
            warmup_iters=warmup_iters,
            iters=iters,
        )
    except Exception as exc:
        return _error_row(case, exc)

    flops = _dense_flops(case.shape)
    return {
        "dense_class": case.class_case.name,
        "quant": case.quant_name,
        "group_size": str(case.group_size),
        "shape_id": case.shape.name,
        "M": str(case.shape.size_m),
        "K": str(case.shape.size_k),
        "N": str(case.shape.size_n),
        "auto_cta_geometry": dense_auto_cta_geometry_label(case.shape),
        "auto_split_k": str(dense_auto_split_k(case.shape)),
        "status": "OK",
        "marlin_us": format_float(stats["median_us"]),
        "flops": str(flops),
        "marlin_tflops": _format_tflops(flops, stats["median_us"]),
        "all_finite": all_finite,
        "reason": "",
    }


def main() -> None:
    args = parse_args()
    require_matching_cuda_benchmark_runtime()
    selected_cases = _count_filtered_matrix(args)
    csv_path = args.csv or Path("benchmarks/results") / (
        f"{timestamp().replace(' ', '_').replace(':', '')}_dense_writeback_matrix.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    banner(f"Marlin Dense Writeback Matrix Benchmark ({timestamp()})")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"capability={format_capability(runtime_capability(0))}")
    print(f"build_target={source_target_label()} ({source_target_cuda_arch_arg()})")
    print("matrix=class x quant x group x shape")
    print(f"preset={args.preset}")
    print(f"dense_classes={args.dense_classes}")
    print(f"quant_types={args.quant_types}")
    print(f"group_sizes={args.group_sizes}")
    print(f"shape_suite={args.shape_suite}")
    print(f"warmup_iters={args.warmup_iters}, iters={args.iters}, check={args.check}")
    print(f"omit_skip={args.omit_skip}")
    print(f"selected_cases={selected_cases}")
    print(f"csv={csv_path}")

    status_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    saved_rows = 0
    prepared_cache: dict[str, object] = {}
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_DENSE_CSV_FIELDNAMES)
        writer.writeheader()
        for index, case in enumerate(_iter_filtered_matrix(args), start=1):
            row = _run_matrix_case(
                case,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
                check=args.check,
                prepared_cache=prepared_cache,
            )
            status_counts[row["status"]] += 1
            if row["status"] == "SKIP":
                skip_reasons[row["reason"]] += 1
                if args.omit_skip:
                    continue

            writer.writerow(row)
            saved_rows += 1
            if saved_rows % 100 == 0 or row["status"] in {"ERR", "MISMATCH"}:
                f.flush()
            if (
                saved_rows % 1000 == 0
                or row["status"] in {"ERR", "MISMATCH"}
                or index == selected_cases
            ):
                print(
                    f"case {index}/{selected_cases} saved {saved_rows}: "
                    f"{case.id} [{row['status']}]",
                    flush=True,
                )

    print()
    print(
        "summary: "
        f"selected_cases={selected_cases}, saved_rows={saved_rows}, "
        f"OK={status_counts['OK']}, SKIP={status_counts['SKIP']}, "
        f"ERR={status_counts['ERR']}, MISMATCH={status_counts['MISMATCH']}"
    )
    if skip_reasons:
        print("top_skip_reasons:")
        for reason, count in skip_reasons.most_common(10):
            print(f"{count}\t{reason}")
        print()
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
