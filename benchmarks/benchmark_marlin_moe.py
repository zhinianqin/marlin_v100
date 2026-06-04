from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import torch

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
    supported_moe_quant_type_names,
    source_target_cuda_arch_arg,
    source_target_label,
)
from tests import ops
from tests.helpers import (
    _quantize_uint4_with_zero_point,
    _quantize_uint8_with_zero_point,
    _quantize_unsigned_with_bias,
    awq_pack,
    gptq_pack,
    make_moe_routing_tensors,
    marlin_moe_reference,
    scalar_types,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.awq_marlin import (
    AWQMarlinConfig,
    AWQMarlinMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsWNA16MarlinMoEMethod,
)
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig,
    GPTQMarlinMoEMethod,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
    QuantizationType,
)
from tests.writeback_marlin_cases import (
    MOE_ALL_QUANT_NAMES,
    MOE_BENCHMARK_SHAPE_CASES,
    MOE_IRREGULAR_SHAPE_CASES,
    MOE_REGULAR_SHAPE_CASES,
    MOE_WRITEBACK_CLASS_CASES,
    MoeWritebackMatrixCase,
    WRITEBACK_GROUP_SIZE_VALUES,
    iter_moe_writeback_matrix,
    moe_case_auto_cta_geometry_label,
    moe_case_auto_split_k_label,
)

_MOE_QUANT_TYPE_CANDIDATES = {
    "uint4": scalar_types.uint4,
    "uint4b8": scalar_types.uint4b8,
    "uint8": scalar_types.uint8,
    "uint8b128": scalar_types.uint8b128,
}
QUANT_TYPES = {
    name: _MOE_QUANT_TYPE_CANDIDATES[name]
    for name in supported_moe_quant_type_names(_MOE_QUANT_TYPE_CANDIDATES)
}
METHOD_CLASS_CHOICES = tuple(case.name for case in MOE_WRITEBACK_CLASS_CASES)
_SHAPE_SUITE_CASES = {
    "regular": MOE_REGULAR_SHAPE_CASES,
    "irregular": MOE_IRREGULAR_SHAPE_CASES,
    "all": MOE_BENCHMARK_SHAPE_CASES,
}
_MOE_CSV_FIELDNAMES = [
    "method_class",
    "quant",
    "group_size",
    "shape_id",
    "tokens",
    "hidden",
    "intermediate",
    "experts",
    "topk",
    "routing_profile",
    "auto_cta_geometry",
    "auto_split_k",
    "status",
    "marlin_us",
    "flops",
    "marlin_tflops",
    "all_finite",
    "check_pass",
    "max_abs_err",
    "reason",
]


@dataclass
class _PreparedMoeCase:
    run_marlin: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    w1_dequant: torch.Tensor | None
    w2_dequant: torch.Tensor | None


def _actual_group_size(group_size: int, size_k: int) -> int:
    return size_k if group_size == -1 else group_size


def _moe_flops(case: MoeWritebackMatrixCase) -> int:
    return (
        6
        * case.shape.tokens
        * case.shape.topk
        * case.shape.hidden
        * case.shape.intermediate
    )


def _format_tflops(flops: int, latency_us: float) -> str:
    return f"{flops / (latency_us * 1_000_000):.6f}"


def _set_parameter_data(layer: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    param = getattr(layer, name)
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(
            f"{name} expected shape {tuple(param.shape)}, got {tuple(value.shape)}"
        )
    param.data.copy_(value.to(device=param.device, dtype=param.dtype))


def _packed_repeated_value(num_bits: int, value: int) -> int:
    word = 0
    for idx in range(32 // num_bits):
        word |= int(value) << (idx * num_bits)
    if word >= 2**31:
        word -= 2**32
    return word


def _zero_reference_weights(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
    device: torch.device,
    enabled: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not enabled:
        return None, None
    return (
        torch.zeros(
            (experts, hidden, 2 * intermediate),
            dtype=torch.float16,
            device=device,
        ),
        torch.zeros(
            (experts, intermediate, hidden),
            dtype=torch.float16,
            device=device,
        ),
    )


def _make_case_inputs(
    case: MoeWritebackMatrixCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device("cuda")
    hidden_states = torch.zeros(
        (case.shape.tokens, case.shape.hidden),
        dtype=torch.float16,
        device=device,
    )
    topk_weights, topk_ids = make_moe_routing_tensors(
        tokens=case.shape.tokens,
        experts=case.shape.experts,
        topk=case.shape.topk,
        device=device,
        routing_profile=case.shape.routing_profile,
    )
    return hidden_states, topk_weights, topk_ids


def _dequant_unsigned_with_bias(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    bias: int,
) -> torch.Tensor:
    size_k = q_weight.shape[0]
    expanded_scales = scales.to(torch.float32).repeat_interleave(
        _actual_group_size(group_size, size_k), dim=0
    )[:size_k]
    return ((q_weight.to(torch.float32) - float(bias)) * expanded_scales).to(
        torch.float16
    )


def _dequant_zero_point(
    q_weight: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    size_k = q_weight.shape[0]
    actual_group_size = _actual_group_size(group_size, size_k)
    expanded_scales = scales.to(torch.float32).repeat_interleave(
        actual_group_size, dim=0
    )[:size_k]
    expanded_zero_points = zero_points.to(torch.float32).repeat_interleave(
        actual_group_size, dim=0
    )[:size_k]
    return ((q_weight.to(torch.float32) - expanded_zero_points) * expanded_scales).to(
        torch.float16
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MoE Marlin writeback class matrix."
    )
    parser.add_argument(
        "--preset",
        default="full",
        help="Compatibility option. The writeback matrix is driven by --shape-suite.",
    )
    parser.add_argument("--cases", nargs="+", help=argparse.SUPPRESS)
    parser.add_argument("--tokens", nargs="+", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--quant-types",
        nargs="+",
        default=list(MOE_ALL_QUANT_NAMES),
        help="Quantized weight types to include in the matrix.",
    )
    parser.add_argument(
        "--group-sizes",
        nargs="+",
        type=int,
        default=list(WRITEBACK_GROUP_SIZE_VALUES),
        help="Group sizes to include in the matrix.",
    )
    parser.add_argument(
        "--shape-suite",
        choices=("regular", "irregular", "all"),
        default="all",
        help="Shape suite to include in the matrix.",
    )
    parser.add_argument(
        "--method-classes",
        nargs="+",
        choices=METHOD_CLASS_CHOICES,
        default=list(METHOD_CLASS_CHOICES),
        help="Production MoE quant-method classes to include in the matrix.",
    )
    parser.add_argument("--act-order", choices=("off", "on", "all"), help=argparse.SUPPRESS)
    parser.add_argument("--is-k-full", choices=("true", "false", "all"), help=argparse.SUPPRESS)
    parser.add_argument("--path", choices=("method", "raw"), default="method", help=argparse.SUPPRESS)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a correctness sanity check before timing each supported case.",
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument(
        "--omit-skip",
        action="store_true",
        help="Do not save or print SKIP rows; still count them in the summary.",
    )
    return parser.parse_args()


def _iter_filtered_matrix(args: argparse.Namespace) -> Iterator[MoeWritebackMatrixCase]:
    allowed_shapes = {shape.name for shape in _SHAPE_SUITE_CASES[args.shape_suite]}
    class_cases = tuple(
        case for case in MOE_WRITEBACK_CLASS_CASES if case.name in args.method_classes
    )
    shapes = tuple(
        sorted(
            (
                shape
                for shape in _SHAPE_SUITE_CASES[args.shape_suite]
                if shape.name in allowed_shapes
            ),
            key=lambda shape: (
                shape.hidden,
                shape.intermediate,
                shape.experts,
                shape.topk,
                shape.tokens,
                shape.routing_profile,
                shape.name,
            ),
        )
    )
    yielded = 0
    for case in iter_moe_writeback_matrix(
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


def _make_method_layer(
    *,
    experts: int,
    hidden: int,
    intermediate: int,
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.apply_router_weight_on_input = False
    layer.global_num_experts = experts
    layer.expert_map = None
    layer.activation = MoEActivation.SILU
    layer.intermediate_size_per_partition = intermediate
    return layer


def _make_moe_config() -> FusedMoEConfig:
    return FusedMoEConfig(
        hidden_dim=0,
        intermediate_size_per_partition=0,
        disable_inplace=True,
        is_act_and_mul=True,
    )


def _prepare_gptq_method_case(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    experts: int,
    hidden: int,
    intermediate: int,
    quant_name: str,
    group_size: int,
    act_order: bool,
    is_k_full: bool,
    prepare_reference: bool,
) -> _PreparedMoeCase:
    quant_type = QUANT_TYPES[quant_name]
    if quant_type not in (scalar_types.uint4b8, scalar_types.uint8b128):
        raise ValueError(f"GPTQ method path does not support {quant_name!r}")

    config = GPTQMarlinConfig(
        weight_bits=quant_type.size_bits,
        group_size=group_size,
        desc_act=act_order,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )
    method = GPTQMarlinMoEMethod(config, _make_moe_config())
    layer = _make_method_layer(experts=experts, hidden=hidden, intermediate=intermediate)
    method.create_weights(
        layer,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.float16,
        intermediate_size_full=intermediate,
    )
    layer.to(hidden_states.device)
    method.is_k_full = is_k_full

    w1_packed = []
    w1_scales = []
    w1_dequant = [] if prepare_reference else None
    w2_packed = []
    w2_scales = []
    w2_dequant = [] if prepare_reference else None
    for expert in range(experts):
        q1, s1 = _quantize_unsigned_with_bias(w1[expert], group_size, quant_type.bias)
        q2, s2 = _quantize_unsigned_with_bias(w2[expert], group_size, quant_type.bias)
        w1_packed.append(gptq_pack(q1, quant_type.size_bits, hidden, 2 * intermediate))
        w2_packed.append(gptq_pack(q2, quant_type.size_bits, intermediate, hidden))
        w1_scales.append(s1)
        w2_scales.append(s2)
        if prepare_reference:
            assert w1_dequant is not None
            assert w2_dequant is not None
            w1_dequant.append(
                _dequant_unsigned_with_bias(q1, s1, group_size, quant_type.bias)
            )
            w2_dequant.append(
                _dequant_unsigned_with_bias(q2, s2, group_size, quant_type.bias)
            )

    _set_parameter_data(layer, "w13_qweight", torch.stack(w1_packed))
    _set_parameter_data(layer, "w2_qweight", torch.stack(w2_packed))
    _set_parameter_data(layer, "w13_scales", torch.stack(w1_scales))
    _set_parameter_data(layer, "w2_scales", torch.stack(w2_scales))

    if act_order:
        raise ValueError("SM70 method benchmark keeps act_order disabled.")

    _set_parameter_data(
        layer,
        "w13_g_idx",
        torch.zeros((experts, hidden), dtype=torch.int32, device=hidden_states.device),
    )
    _set_parameter_data(
        layer,
        "w2_g_idx",
        torch.zeros(
            (experts, intermediate), dtype=torch.int32, device=hidden_states.device
        ),
    )
    method.process_weights_after_loading(layer)

    def run_marlin() -> torch.Tensor:
        return method.apply(layer, hidden_states, topk_weights, topk_ids, None)

    run_marlin()
    return _PreparedMoeCase(
        run_marlin=run_marlin,
        w1_dequant=(
            torch.stack(w1_dequant) if w1_dequant is not None else None
        ),
        w2_dequant=(
            torch.stack(w2_dequant) if w2_dequant is not None else None
        ),
    )


def _prepare_awq_method_case(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    experts: int,
    hidden: int,
    intermediate: int,
    quant_name: str,
    group_size: int,
    prepare_reference: bool,
) -> _PreparedMoeCase:
    if quant_name == "uint4":
        weight_bits = 4
        quantize = _quantize_uint4_with_zero_point
    elif quant_name == "uint8":
        weight_bits = 8
        quantize = _quantize_uint8_with_zero_point
    else:
        raise ValueError(f"AWQ method path does not support {quant_name!r}")

    config = AWQMarlinConfig(
        weight_bits=weight_bits,
        group_size=group_size,
        zero_point=True,
        lm_head_quantized=False,
        modules_to_not_convert=None,
        full_config={},
    )
    method = AWQMarlinMoEMethod(config, _make_moe_config())
    layer = _make_method_layer(experts=experts, hidden=hidden, intermediate=intermediate)
    method.create_weights(
        layer,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.float16,
        intermediate_size_full=intermediate,
    )
    layer.to(hidden_states.device)
    method.is_k_full = True

    w1_packed = []
    w1_scales = []
    w1_zeros = []
    w1_dequant = [] if prepare_reference else None
    w2_packed = []
    w2_scales = []
    w2_zeros = []
    w2_dequant = [] if prepare_reference else None
    for expert in range(experts):
        q1, s1, z1 = quantize(w1[expert], group_size)
        q2, s2, z2 = quantize(w2[expert], group_size)
        w1_packed.append(awq_pack(q1, weight_bits, hidden, 2 * intermediate))
        w2_packed.append(awq_pack(q2, weight_bits, intermediate, hidden))
        w1_scales.append(s1)
        w2_scales.append(s2)
        w1_zeros.append(awq_pack(z1, weight_bits, z1.shape[0], 2 * intermediate))
        w2_zeros.append(awq_pack(z2, weight_bits, z2.shape[0], hidden))
        if prepare_reference:
            assert w1_dequant is not None
            assert w2_dequant is not None
            w1_dequant.append(_dequant_zero_point(q1, s1, z1, group_size))
            w2_dequant.append(_dequant_zero_point(q2, s2, z2, group_size))

    _set_parameter_data(layer, "w13_qweight", torch.stack(w1_packed))
    _set_parameter_data(layer, "w2_qweight", torch.stack(w2_packed))
    _set_parameter_data(layer, "w13_scales", torch.stack(w1_scales))
    _set_parameter_data(layer, "w2_scales", torch.stack(w2_scales))
    _set_parameter_data(layer, "w13_qzeros", torch.stack(w1_zeros))
    _set_parameter_data(layer, "w2_qzeros", torch.stack(w2_zeros))

    method.process_weights_after_loading(layer)

    def run_marlin() -> torch.Tensor:
        return method.apply(layer, hidden_states, topk_weights, topk_ids, None)

    run_marlin()
    return _PreparedMoeCase(
        run_marlin=run_marlin,
        w1_dequant=(
            torch.stack(w1_dequant) if w1_dequant is not None else None
        ),
        w2_dequant=(
            torch.stack(w2_dequant) if w2_dequant is not None else None
        ),
    )


def _prepare_compressed_tensors_wna16_method_case(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    experts: int,
    hidden: int,
    intermediate: int,
    quant_name: str,
    group_size: int,
    act_order: bool,
    is_k_full: bool,
    prepare_reference: bool,
) -> _PreparedMoeCase:
    quant_type = QUANT_TYPES[quant_name]
    if quant_type not in (scalar_types.uint4b8, scalar_types.uint8b128):
        raise ValueError(
            "CompressedTensorsWNA16MarlinMoEMethod path supports "
            f"uint4b8/uint8b128, got {quant_name!r}"
        )

    strategy = (
        QuantizationStrategy.CHANNEL
        if group_size == -1
        else QuantizationStrategy.GROUP
    )
    weight_quant = QuantizationArgs(
        num_bits=quant_type.size_bits,
        type=QuantizationType.INT,
        symmetric=True,
        group_size=None if group_size == -1 else group_size,
        strategy=strategy,
        actorder="group" if act_order else None,
    )
    method = CompressedTensorsWNA16MarlinMoEMethod(
        weight_quant,
        None,
        _make_moe_config(),
    )
    layer = _make_method_layer(experts=experts, hidden=hidden, intermediate=intermediate)
    method.create_weights(
        layer,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=intermediate,
        params_dtype=torch.float16,
        intermediate_size_full=intermediate,
    )
    layer.to(hidden_states.device)
    method.is_k_full = is_k_full

    w1_packed = []
    w1_scales = []
    w1_dequant = [] if prepare_reference else None
    w2_packed = []
    w2_scales = []
    w2_dequant = [] if prepare_reference else None
    for expert in range(experts):
        q1, s1 = _quantize_unsigned_with_bias(w1[expert], group_size, quant_type.bias)
        q2, s2 = _quantize_unsigned_with_bias(w2[expert], group_size, quant_type.bias)
        w1_packed.append(gptq_pack(q1, quant_type.size_bits, hidden, 2 * intermediate))
        w2_packed.append(gptq_pack(q2, quant_type.size_bits, intermediate, hidden))
        w1_scales.append(s1)
        w2_scales.append(s2)
        if prepare_reference:
            assert w1_dequant is not None
            assert w2_dequant is not None
            w1_dequant.append(
                _dequant_unsigned_with_bias(q1, s1, group_size, quant_type.bias)
            )
            w2_dequant.append(
                _dequant_unsigned_with_bias(q2, s2, group_size, quant_type.bias)
            )

    _set_parameter_data(layer, "w13_weight_packed", torch.stack(w1_packed))
    _set_parameter_data(layer, "w2_weight_packed", torch.stack(w2_packed))
    _set_parameter_data(layer, "w13_weight_scale", torch.stack(w1_scales))
    _set_parameter_data(layer, "w2_weight_scale", torch.stack(w2_scales))
    _set_parameter_data(
        layer,
        "w13_weight_shape",
        torch.tensor([[hidden, 2 * intermediate]] * experts, device=hidden_states.device),
    )
    _set_parameter_data(
        layer,
        "w2_weight_shape",
        torch.tensor([[intermediate, hidden]] * experts, device=hidden_states.device),
    )

    if act_order:
        raise ValueError("SM70 method benchmark keeps act_order disabled.")

    _set_parameter_data(
        layer,
        "w13_weight_g_idx",
        torch.zeros((experts, hidden), dtype=torch.int32, device=hidden_states.device),
    )
    _set_parameter_data(
        layer,
        "w2_weight_g_idx",
        torch.zeros(
            (experts, intermediate), dtype=torch.int32, device=hidden_states.device
        ),
    )
    method.process_weights_after_loading(layer)

    def run_marlin() -> torch.Tensor:
        return method.apply(layer, hidden_states, topk_weights, topk_ids, None)

    run_marlin()
    return _PreparedMoeCase(
        run_marlin=run_marlin,
        w1_dequant=(
            torch.stack(w1_dequant) if w1_dequant is not None else None
        ),
        w2_dequant=(
            torch.stack(w2_dequant) if w2_dequant is not None else None
        ),
    )


def _prepare_method_case(
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    experts: int,
    hidden: int,
    intermediate: int,
    quant_name: str,
    group_size: int,
    act_order: bool,
    is_k_full: bool,
    method_class: str,
    prepare_reference: bool,
) -> _PreparedMoeCase:
    if method_class == "awq_moe":
        return _prepare_awq_method_case(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1=w1,
            w2=w2,
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            quant_name=quant_name,
            group_size=group_size,
            prepare_reference=prepare_reference,
        )
    if method_class == "gptq_moe":
        return _prepare_gptq_method_case(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1=w1,
            w2=w2,
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            quant_name=quant_name,
            group_size=group_size,
            act_order=act_order,
            is_k_full=is_k_full,
            prepare_reference=prepare_reference,
        )
    if method_class == "compressed_tensors_wna16_moe":
        return _prepare_compressed_tensors_wna16_method_case(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w1=w1,
            w2=w2,
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
            quant_name=quant_name,
            group_size=group_size,
            act_order=act_order,
            is_k_full=is_k_full,
            prepare_reference=prepare_reference,
        )
    raise ValueError(f"Unsupported method_class={method_class!r}")


def _prepare_benchmark_method_case(
    case: MoeWritebackMatrixCase,
    *,
    prepare_reference: bool,
) -> _PreparedMoeCase:
    device = torch.device("cuda")
    experts = case.shape.experts
    hidden = case.shape.hidden
    intermediate = case.shape.intermediate
    group_size = case.group_size
    quant_name = case.quant_name

    if case.class_case.name == "awq_moe":
        if quant_name == "uint4":
            weight_bits = 4
        elif quant_name == "uint8":
            weight_bits = 8
        else:
            raise ValueError(f"AWQ method path does not support {quant_name!r}")

        method = AWQMarlinMoEMethod(
            AWQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=True,
                lm_head_quantized=False,
                modules_to_not_convert=None,
                full_config={},
            ),
            _make_moe_config(),
        )
        layer = _make_method_layer(
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
        )
        method.create_weights(
            layer,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            params_dtype=torch.float16,
            intermediate_size_full=intermediate,
        )
        layer.to(device)
        method.is_k_full = True
        layer.w13_qweight.data.zero_()
        layer.w2_qweight.data.zero_()
        layer.w13_scales.data.fill_(1.0)
        layer.w2_scales.data.fill_(1.0)
        layer.w13_qzeros.data.zero_()
        layer.w2_qzeros.data.zero_()
        method.process_weights_after_loading(layer)

    elif case.class_case.name == "gptq_moe":
        quant_type = QUANT_TYPES[quant_name]
        if quant_type not in (scalar_types.uint4b8, scalar_types.uint8b128):
            raise ValueError(f"GPTQ method path does not support {quant_name!r}")

        method = GPTQMarlinMoEMethod(
            GPTQMarlinConfig(
                weight_bits=quant_type.size_bits,
                group_size=group_size,
                desc_act=False,
                is_sym=True,
                lm_head_quantized=False,
                dynamic={},
                full_config={},
            ),
            _make_moe_config(),
        )
        layer = _make_method_layer(
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
        )
        method.create_weights(
            layer,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            params_dtype=torch.float16,
            intermediate_size_full=intermediate,
        )
        layer.to(device)
        method.is_k_full = True
        packed_zero = _packed_repeated_value(
            quant_type.size_bits,
            quant_type.bias,
        )
        layer.w13_qweight.data.fill_(packed_zero)
        layer.w2_qweight.data.fill_(packed_zero)
        layer.w13_scales.data.fill_(1.0)
        layer.w2_scales.data.fill_(1.0)
        _set_parameter_data(
            layer,
            "w13_g_idx",
            torch.zeros((experts, hidden), dtype=torch.int32, device=device),
        )
        _set_parameter_data(
            layer,
            "w2_g_idx",
            torch.zeros((experts, intermediate), dtype=torch.int32, device=device),
        )
        method.process_weights_after_loading(layer)

    elif case.class_case.name == "compressed_tensors_wna16_moe":
        quant_type = QUANT_TYPES[quant_name]
        if quant_type not in (scalar_types.uint4b8, scalar_types.uint8b128):
            raise ValueError(
                "CompressedTensorsWNA16MarlinMoEMethod path supports "
                f"uint4b8/uint8b128, got {quant_name!r}"
            )

        strategy = (
            QuantizationStrategy.CHANNEL
            if group_size == -1
            else QuantizationStrategy.GROUP
        )
        weight_quant = QuantizationArgs(
            num_bits=quant_type.size_bits,
            type=QuantizationType.INT,
            symmetric=True,
            group_size=None if group_size == -1 else group_size,
            strategy=strategy,
            actorder=None,
        )
        method = CompressedTensorsWNA16MarlinMoEMethod(
            weight_quant,
            None,
            _make_moe_config(),
        )
        layer = _make_method_layer(
            experts=experts,
            hidden=hidden,
            intermediate=intermediate,
        )
        method.create_weights(
            layer,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            params_dtype=torch.float16,
            intermediate_size_full=intermediate,
        )
        layer.to(device)
        method.is_k_full = True
        packed_zero = _packed_repeated_value(
            quant_type.size_bits,
            quant_type.bias,
        )
        layer.w13_weight_packed.data.fill_(packed_zero)
        layer.w2_weight_packed.data.fill_(packed_zero)
        layer.w13_weight_scale.data.fill_(1.0)
        layer.w2_weight_scale.data.fill_(1.0)
        _set_parameter_data(
            layer,
            "w13_weight_shape",
            torch.tensor([[hidden, 2 * intermediate]] * experts, device=device),
        )
        _set_parameter_data(
            layer,
            "w2_weight_shape",
            torch.tensor([[intermediate, hidden]] * experts, device=device),
        )
        _set_parameter_data(
            layer,
            "w13_weight_g_idx",
            torch.zeros((experts, hidden), dtype=torch.int32, device=device),
        )
        _set_parameter_data(
            layer,
            "w2_weight_g_idx",
            torch.zeros((experts, intermediate), dtype=torch.int32, device=device),
        )
        method.process_weights_after_loading(layer)

    else:
        raise ValueError(f"Unsupported method_class={case.class_case.name!r}")

    def run_marlin(
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        return method.apply(layer, hidden_states, topk_weights, topk_ids, None)

    w1_dequant, w2_dequant = _zero_reference_weights(
        experts=experts,
        hidden=hidden,
        intermediate=intermediate,
        device=device,
        enabled=prepare_reference,
    )
    return _PreparedMoeCase(
        run_marlin=run_marlin,
        w1_dequant=w1_dequant,
        w2_dequant=w2_dequant,
    )


def _base_row(case: MoeWritebackMatrixCase) -> dict[str, str]:
    return {
        "method_class": case.class_case.name,
        "quant": case.quant_name,
        "group_size": str(case.group_size),
        "shape_id": case.shape.name,
        "tokens": str(case.shape.tokens),
        "hidden": str(case.shape.hidden),
        "intermediate": str(case.shape.intermediate),
        "experts": str(case.shape.experts),
        "topk": str(case.shape.topk),
        "routing_profile": case.shape.routing_profile,
        "auto_cta_geometry": "n/a",
        "auto_split_k": "n/a",
        "status": "",
        "marlin_us": "",
        "flops": "",
        "marlin_tflops": "",
        "all_finite": "",
        "check_pass": "",
        "max_abs_err": "",
        "reason": "",
    }


def _skip_row(case: MoeWritebackMatrixCase, reason: str) -> dict[str, str]:
    row = _base_row(case)
    row.update(
        {
            "status": "SKIP",
            "marlin_us": "n/a",
            "flops": "n/a",
            "marlin_tflops": "n/a",
            "all_finite": "n/a",
            "check_pass": "n/a",
            "max_abs_err": "n/a",
            "reason": reason,
        }
    )
    return row


def _error_row(case: MoeWritebackMatrixCase, exc: BaseException) -> dict[str, str]:
    row = _skip_row(case, str(exc).splitlines()[0])
    row["status"] = "ERR"
    row["auto_cta_geometry"] = moe_case_auto_cta_geometry_label(case)
    row["auto_split_k"] = moe_case_auto_split_k_label(case)
    row["flops"] = "n/a"
    row["marlin_tflops"] = "n/a"
    return row


def _run_matrix_case(
    case: MoeWritebackMatrixCase,
    *,
    warmup_iters: int,
    iters: int,
    check: bool,
    prepared_cache: dict[str, object],
) -> dict[str, str]:
    if not case.supported:
        return _skip_row(case, case.reason)

    all_finite = "n/a"
    check_pass = "n/a"
    max_abs_err = "n/a"
    reason = ""

    try:
        prepare_key = (
            case.class_case.name,
            case.quant_name,
            case.group_size,
            case.shape.hidden,
            case.shape.intermediate,
            case.shape.experts,
            check,
        )
        if prepared_cache.get("prepare_key") != prepare_key:
            prepared_cache.clear()
            prepared = _prepare_benchmark_method_case(
                case,
                prepare_reference=check,
            )
            prepared_cache["prepare_key"] = prepare_key
            prepared_cache["prepared"] = prepared
        else:
            prepared = prepared_cache["prepared"]

        input_key = (
            case.shape.tokens,
            case.shape.hidden,
            case.shape.experts,
            case.shape.topk,
            case.shape.routing_profile,
        )
        if prepared_cache.get("input_key") != input_key:
            hidden_states, topk_weights, topk_ids = _make_case_inputs(case)
            prepared_cache["input_key"] = input_key
            prepared_cache["hidden_states"] = hidden_states
            prepared_cache["topk_weights"] = topk_weights
            prepared_cache["topk_ids"] = topk_ids
        else:
            hidden_states = prepared_cache["hidden_states"]
            topk_weights = prepared_cache["topk_weights"]
            topk_ids = prepared_cache["topk_ids"]

        assert isinstance(prepared, _PreparedMoeCase)
        run_marlin = prepared.run_marlin

        if check:
            output = run_marlin(hidden_states, topk_weights, topk_ids)
            all_finite = "yes" if torch.isfinite(output).all().item() else "no"
            if output.shape != hidden_states.shape:
                raise AssertionError(
                    f"output shape {tuple(output.shape)} != "
                    f"{tuple(hidden_states.shape)}"
                )
            if prepared.w1_dequant is None or prepared.w2_dequant is None:
                raise AssertionError("reference weights were not prepared")
            reference = marlin_moe_reference(
                hidden_states,
                prepared.w1_dequant,
                prepared.w2_dequant,
                topk_weights,
                topk_ids,
            ).to(torch.float16)
            if all_finite == "yes":
                diff = (output - reference).abs().to(torch.float32)
                max_abs_err = format_float(float(diff.max().item()))
                try:
                    if case.quant_name in {"uint4", "uint8"}:
                        torch.testing.assert_close(
                            output,
                            reference,
                            rtol=2e-1,
                            atol=1.25,
                        )
                    else:
                        torch.testing.assert_close(
                            output,
                            reference,
                            rtol=7e-2,
                            atol=1e-2,
                        )
                    check_pass = "yes"
                except AssertionError as exc:
                    check_pass = "no"
                    reason = str(exc).splitlines()[0]
            else:
                check_pass = "no"
                max_abs_err = "inf"
                reason = "output contains non-finite values"

        stats = time_cuda_callable(
            lambda: run_marlin(hidden_states, topk_weights, topk_ids),
            warmup_iters=warmup_iters,
            iters=iters,
        )
    except Exception as exc:
        return _error_row(case, exc)

    row = _base_row(case)
    row.update(
        {
            "status": "OK" if check_pass != "no" else "MISMATCH",
            "marlin_us": format_float(stats["median_us"]),
            "flops": str(_moe_flops(case)),
            "marlin_tflops": _format_tflops(_moe_flops(case), stats["median_us"]),
            "auto_cta_geometry": moe_case_auto_cta_geometry_label(case),
            "auto_split_k": moe_case_auto_split_k_label(case),
            "all_finite": all_finite,
            "check_pass": check_pass,
            "max_abs_err": max_abs_err,
            "reason": reason,
        }
    )
    return row


def main() -> None:
    args = parse_args()
    require_matching_cuda_benchmark_runtime()
    ops._load_dense()
    ops._load_moe()

    selected_cases = _count_filtered_matrix(args)
    csv_path = args.csv or Path("benchmarks/results") / (
        f"{timestamp().replace(' ', '_').replace(':', '')}_moe_writeback_matrix.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    banner(f"Marlin MoE Writeback Matrix Benchmark ({timestamp()})")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"capability={format_capability(runtime_capability(0))}")
    print(f"build_target={source_target_label()} ({source_target_cuda_arch_arg()})")
    print("matrix=class x quant x group x shape")
    print(f"preset={args.preset}")
    if args.path != "method":
        print("path=raw was requested, but writeback matrix benchmark uses method path.")
    print(f"quant_types={args.quant_types}")
    print(f"group_sizes={args.group_sizes}")
    print(f"shape_suite={args.shape_suite}")
    print(f"method_classes={args.method_classes}")
    print(f"warmup_iters={args.warmup_iters}, iters={args.iters}, check={args.check}")
    print(f"omit_skip={args.omit_skip}")
    print(f"selected_cases={selected_cases}")
    print(f"csv={csv_path}")

    status_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    saved_rows = 0
    prepared_cache: dict[str, object] = {}
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_MOE_CSV_FIELDNAMES)
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
