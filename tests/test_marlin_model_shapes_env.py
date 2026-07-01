from __future__ import annotations

import importlib.util
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

torch = pytest.importorskip("torch")

from tests import ops, quant_utils
from tests.sm70_env_sweep import (
    EXPLICIT_ENV_REJECTION_RE,
    DenseDirectOpKey,
    MoeDirectOpKey,
    dense_env,
    dense_env_combo_is_legal,
    exhaustive_enabled,
    exhaustive_index_is_past_limit,
    exhaustive_index_is_selected,
    exhaustive_start_limit,
    iter_env_combinations,
    iter_moe_env_combinations,
    moe_env,
    moe_stage_env_combo_is_legal,
)
from tests.helpers import (
    _fp8_fused_exponent_bias_into_scales,
    _nvfp4_marlin_process_global_scale,
    _nvfp4_marlin_process_scales,
    _quantize_fp8_weight,
    _quantize_mxfp4_weight,
    _quantize_nvfp4_weight,
    _quantize_uint4_with_zero_point,
    _quantize_uint8_with_zero_point,
    _quantize_unsigned_with_bias,
    fp4_e2m1_weight_to_marlin_weight,
    fp8_weight_to_marlin_weight,
    marlin_permute_bias,
    marlin_permute_scales,
    moe_align_block_size,
    scalar_types,
)
from tests.writeback_marlin_cases import (
    is_dense_group_size_supported,
)
from tests.test_marlin_dense import (
    _assert_dense_env_sweep_combo_matches_reference,
    _make_dense_env_sweep_case,
)
from tests.test_marlin_moe import (
    _MOE_ENV_SWEEP_TOLERANCES,
    _require_moe_cuda,
    _run_moe_env_stage1_combo,
    _run_moe_env_stage2_combo,
)


SCRIPT = Path(__file__).parents[1] / "benchmarks/marlin_gemm_shapes.py"
SPEC = importlib.util.spec_from_file_location("marlin_gemm_shapes", SCRIPT)
assert SPEC is not None
marlin_gemm_shapes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = marlin_gemm_shapes
SPEC.loader.exec_module(marlin_gemm_shapes)


_DENSE_ROW_KEYS = (
    "scenario",
    "phase",
    "layer_key",
    "op",
    "target_op",
    "size_m",
    "size_n",
    "size_k",
    "group_size",
    "quant_method",
    "quant_format",
    "has_zp",
    "has_bias",
    "marlin_path",
    "call_status",
    "call_count",
)

_MOE_ROW_KEYS = _DENSE_ROW_KEYS + (
    "moe_block_size",
    "top_k",
    "local_num_experts",
    "global_num_experts",
    "intermediate_size_per_partition",
)

_ACTUAL_DENSE_CONTEXT_KEYS = (
    "model",
    "scenario",
    "phase",
    "op",
    "size_m",
    "size_n",
    "size_k",
    "quant_format",
    "group_size",
)

_ACTUAL_MOE_CONTEXT_KEYS = (
    "model",
    "scenario",
    "phase",
    "op",
    "size_m",
    "size_n",
    "size_k",
    "moe_block_size",
    "top_k",
    "quant_format",
    "group_size",
)

_PROGRESS_HEARTBEAT_INTERVAL = 64
_MOE_RUNTIME_EXPERT_SUBSET = 8
_MoeQuantizedWeight = dict[str, object]
_MoeStage1RuntimeCase = tuple[MoeDirectOpKey, dict[str, object], torch.Tensor, float, float]
_MoeStage2RuntimeCase = tuple[
    MoeDirectOpKey,
    dict[str, object],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    float,
    float,
]


@dataclass(frozen=True)
class RuntimeSupport:
    supported: bool
    reason: str = ""


def _quant_name(quant_format: str) -> str:
    if quant_format in {"uint4", "uint8", "uint4b8", "uint8b128", "nvfp4", "mxfp4"}:
        return quant_format
    if quant_format.startswith("fp8"):
        return "fp8"
    raise AssertionError(f"Unsupported quant_format={quant_format!r}")


def _load_payload(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        pytest.fail(
            f"--model must point to a model directory containing config.json: {model_dir}",
            pytrace=False,
        )
    args = marlin_gemm_shapes.parse_args(
        [
            "--model",
            str(model_dir),
            "--moe-backend",
            "marlin",
            "--format",
            "json",
        ]
    )
    return marlin_gemm_shapes.build_payload(args)


def _capture_pretty(model_dir: Path, capsys: pytest.CaptureFixture[str]) -> str:
    rc = marlin_gemm_shapes.main(
        [
            "--model",
            str(model_dir),
            "--moe-backend",
            "marlin",
            "--format",
            "pretty",
        ]
    )
    assert rc == 0
    return capsys.readouterr().out


def _emit_progress(request: pytest.FixtureRequest, message: str) -> None:
    capturemanager = request.config.pluginmanager.get_plugin("capturemanager")
    reporter = request.config.pluginmanager.get_plugin("terminalreporter")
    try:
        if capturemanager is not None:
            with capturemanager.global_and_fixture_disabled():
                if reporter is not None:
                    reporter.write_line(message)
                    terminal_writer = getattr(reporter, "_tw", None)
                    flush = getattr(terminal_writer, "flush", None)
                    if callable(flush):
                        flush()
                    return
                print(message, flush=True)
                return
        if reporter is not None:
            reporter.write_line(message)
            terminal_writer = getattr(reporter, "_tw", None)
            flush = getattr(terminal_writer, "flush", None)
            if callable(flush):
                flush()
            return
        print(message, flush=True)
    except Exception:
        print(message, flush=True)


def _format_limit(limit: int | None) -> str:
    return "unbounded" if limit is None else str(limit)


def _row_context(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    return ", ".join(f"{key}={row.get(key)!r}" for key in keys)


def _int_field(row: dict[str, Any], key: str, context_keys: tuple[str, ...]) -> int:
    try:
        return int(row[key])
    except (TypeError, ValueError) as exc:
        raise AssertionError(
            f"row field {key!r} must be an int-compatible value: "
            f"{_row_context(row, context_keys)}"
        ) from exc


def _assert_positive_int_field(
    row: dict[str, Any],
    key: str,
    context_keys: tuple[str, ...],
) -> int:
    value = _int_field(row, key, context_keys)
    assert value > 0, f"row field {key!r} must be > 0: {_row_context(row, context_keys)}"
    return value


def _assert_payload_shape(payload: dict[str, Any]) -> None:
    for key in (
        "model",
        "model_config",
        "quantization",
        "shape_inputs",
        "scenarios",
        "dense",
        "moe",
        "warnings",
    ):
        assert key in payload
    assert isinstance(payload["dense"], list)
    assert isinstance(payload["moe"], list)
    assert isinstance(payload["warnings"], list)


def _assert_dense_row_schema(row: dict[str, Any]) -> None:
    for key in _DENSE_ROW_KEYS:
        assert key in row, f"dense row missing {key!r}: {row}"


def _assert_moe_row_schema(row: dict[str, Any]) -> None:
    for key in _MOE_ROW_KEYS:
        assert key in row, f"moe row missing {key!r}: {row}"


def _validate_row_schema(payload: dict[str, Any]) -> None:
    for row in payload["dense"]:
        _assert_dense_row_schema(row)
        if row["call_status"] == "actual_marlin":
            _assert_dense_actual_row_supported(row)
    for row in payload["moe"]:
        _assert_moe_row_schema(row)
        if row["call_status"] == "actual_marlin":
            _assert_moe_actual_row_supported(row)


def _assert_pretty_output(payload: dict[str, Any], output: str) -> None:
    for needle in ("Model:", "Config:", "Warnings:", "Dense table", "MoE table"):
        assert needle in output
    assert "has_bias" in output
    if not payload["dense"]:
        assert "Dense table\n  (no rows)" in output
    if not payload["moe"]:
        assert "MoE table\n  (no rows)" in output


def _assert_dense_actual_row_supported(row: dict[str, Any]) -> None:
    assert row["target_op"] == "ops.marlin_gemm"
    _assert_positive_int_field(row, "size_m", _ACTUAL_DENSE_CONTEXT_KEYS)
    _assert_positive_int_field(row, "size_n", _ACTUAL_DENSE_CONTEXT_KEYS)
    _assert_positive_int_field(row, "size_k", _ACTUAL_DENSE_CONTEXT_KEYS)
    _int_field(row, "group_size", _ACTUAL_DENSE_CONTEXT_KEYS)
    try:
        quant_name = _quant_name(str(row["quant_format"]))
    except AssertionError as exc:
        raise AssertionError(
            "unsupported dense quant in actual row: "
            f"{_row_context(row, _ACTUAL_DENSE_CONTEXT_KEYS)}"
        ) from exc
    if not is_dense_group_size_supported(
        quant_name,
        _helper_group_size(row),
        int(row["size_k"]),
    ):
        raise AssertionError(
            "unsupported dense group_size in actual row: "
            f"{_row_context(row, _ACTUAL_DENSE_CONTEXT_KEYS)}"
        )


def _assert_moe_actual_row_supported(row: dict[str, Any]) -> None:
    assert row["target_op"] == "ops.moe_wna16_marlin_gemm"
    assert row["op"] in {"w13", "w2"}
    _assert_positive_int_field(row, "moe_block_size", _ACTUAL_MOE_CONTEXT_KEYS)
    top_k = _assert_positive_int_field(row, "top_k", _ACTUAL_MOE_CONTEXT_KEYS)
    _assert_positive_int_field(row, "size_m", _ACTUAL_MOE_CONTEXT_KEYS)
    _assert_positive_int_field(row, "size_n", _ACTUAL_MOE_CONTEXT_KEYS)
    _assert_positive_int_field(row, "size_k", _ACTUAL_MOE_CONTEXT_KEYS)
    _assert_positive_int_field(row, "local_num_experts", _ACTUAL_MOE_CONTEXT_KEYS)
    _int_field(row, "group_size", _ACTUAL_MOE_CONTEXT_KEYS)
    if row["op"] == "w2":
        assert top_k == 1, f"w2 row top_k must be 1: {_row_context(row, _ACTUAL_MOE_CONTEXT_KEYS)}"
    try:
        quant_name = _quant_name(str(row["quant_format"]))
    except AssertionError as exc:
        raise AssertionError(
            "unsupported MoE quant in actual row: "
            f"{_row_context(row, _ACTUAL_MOE_CONTEXT_KEYS)}"
        ) from exc


def _moe_runtime_support(row: dict[str, Any]) -> RuntimeSupport:
    try:
        quant_name = _quant_name(str(row["quant_format"]))
        group_size = _helper_group_size(row)
        size_k = int(row["size_k"])
    except (AssertionError, TypeError, ValueError) as exc:
        return RuntimeSupport(False, f"invalid runtime row metadata: {exc}")

    if group_size == -1:
        return RuntimeSupport(True)

    if size_k % group_size != 0:
        if quant_name == "fp8":
            return RuntimeSupport(
                False,
                "fp8 group_size=128 requires row size_k divisible by 128 "
                "in current SM70 MoE runtime helper/op",
            )
        return RuntimeSupport(
            False,
            f"{quant_name} group_size={group_size} requires row size_k divisible "
            "by group_size in current SM70 MoE runtime helper/op",
        )

    if quant_name == "nvfp4":
        if group_size == 16:
            return RuntimeSupport(True)
        return RuntimeSupport(False, "nvfp4 supports only group_size=16")
    if quant_name == "mxfp4":
        if group_size == 32:
            return RuntimeSupport(True)
        return RuntimeSupport(False, "mxfp4 supports only group_size=32")
    if quant_name == "fp8":
        if group_size == 128:
            return RuntimeSupport(True)
        return RuntimeSupport(False, "fp8 supports only group_size=-1 or 128")
    if group_size in (32, 64, 128):
        return RuntimeSupport(True)
    return RuntimeSupport(
        False,
        f"{quant_name} supports only group_size=-1, 32, 64, or 128",
    )


def _helper_group_size(row: dict[str, Any]) -> int:
    group_size = int(row["group_size"])
    size_k = int(row["size_k"])
    if group_size == size_k:
        return -1
    return group_size


def _moe_scalar_type(quant_name: str) -> Any:
    return {
        "uint4": scalar_types.uint4,
        "uint4b8": scalar_types.uint4b8,
        "uint8": scalar_types.uint8,
        "uint8b128": scalar_types.uint8b128,
        "fp8": scalar_types.float8_e4m3fn,
        "nvfp4": scalar_types.float4_e2m1f,
        "mxfp4": scalar_types.float4_e2m1f,
    }[quant_name]


def _moe_runtime_expert_count(row: dict[str, Any]) -> int:
    experts = int(row["local_num_experts"])
    topk = int(row["top_k"])
    return min(experts, max(topk, _MOE_RUNTIME_EXPERT_SUBSET))


def _make_uniform_topk_routing(
    *,
    tokens: int,
    experts: int,
    topk: int,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_weights = torch.rand((tokens, topk), device=device, dtype=torch.float32)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_ids = torch.empty((tokens, topk), device=device, dtype=torch.int32)
    for token_idx in range(tokens):
        for route_idx in range(topk):
            topk_ids[token_idx, route_idx] = (token_idx + route_idx) % experts
    return topk_weights, topk_ids


def _quantize_moe_expert_weights(
    weights: torch.Tensor,
    *,
    quant_name: str,
    group_size: int,
) -> _MoeQuantizedWeight:
    q_weights: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    zeros: list[torch.Tensor] = []
    global_scales: list[torch.Tensor] = []
    dequantized: list[torch.Tensor] = []
    g_indices: list[torch.Tensor] = []
    perms: list[torch.Tensor] = []

    quant_type = _moe_scalar_type(quant_name)
    size_k = int(weights.shape[1])
    size_n = int(weights.shape[2])
    for expert in range(weights.shape[0]):
        weight = weights[expert]
        if quant_name == "uint4":
            q_weight, scale, zero_point = _quantize_uint4_with_zero_point(
                weight,
                group_size,
            )
            q_marlin = quant_utils.marlin_weights(
                q_weight,
                size_k,
                size_n,
                scalar_types.uint4.size_bits,
                quant_utils.get_weight_perm(scalar_types.uint4.size_bits, is_a_8bit=False),
                is_a_8bit=False,
            )
            marlin_scale = marlin_permute_scales(
                scale,
                size_k,
                size_n,
                group_size,
                is_a_8bit=False,
            )
            zp = (zero_point.to(torch.float32) * scale.to(torch.float32)).to(weight.dtype)
            marlin_zero = marlin_permute_scales(
                zp,
                size_k,
                size_n,
                group_size,
                is_a_8bit=False,
            )
            runtime_group_size = size_k if group_size == -1 else group_size
            expanded_scale = scale.to(torch.float32).repeat_interleave(runtime_group_size, dim=0)[
                :size_k
            ]
            expanded_zp = zp.to(torch.float32).repeat_interleave(runtime_group_size, dim=0)[
                :size_k
            ]
            dequant = (q_weight.to(torch.float32) * expanded_scale - expanded_zp).to(
                torch.float16
            )
            zeros.append(marlin_zero)
            global_scale = None
        elif quant_name == "uint8":
            q_weight, scale, zero_point = _quantize_uint8_with_zero_point(
                weight,
                group_size,
            )
            q_marlin = quant_utils.marlin_weights(
                q_weight,
                size_k,
                size_n,
                scalar_types.uint8.size_bits,
                quant_utils.get_weight_perm(scalar_types.uint8.size_bits, is_a_8bit=False),
                is_a_8bit=False,
            )
            marlin_scale = marlin_permute_scales(
                scale,
                size_k,
                size_n,
                group_size,
                is_a_8bit=False,
            )
            zp = (zero_point.to(torch.float32) * scale.to(torch.float32)).to(weight.dtype)
            marlin_zero = marlin_permute_scales(
                zp,
                size_k,
                size_n,
                group_size,
                is_a_8bit=False,
            )
            runtime_group_size = size_k if group_size == -1 else group_size
            expanded_scale = scale.to(torch.float32).repeat_interleave(runtime_group_size, dim=0)[
                :size_k
            ]
            expanded_zp = zp.to(torch.float32).repeat_interleave(runtime_group_size, dim=0)[
                :size_k
            ]
            dequant = (q_weight.to(torch.float32) * expanded_scale - expanded_zp).to(
                torch.float16
            )
            zeros.append(marlin_zero)
            global_scale = None
        elif quant_name == "nvfp4":
            q_weight, fp8_scales, raw_global_scale, dequant = _quantize_nvfp4_weight(
                weight,
                group_size,
            )
            q_marlin = fp4_e2m1_weight_to_marlin_weight(q_weight)
            raw_marlin_scale = marlin_permute_scales(
                fp8_scales,
                size_k,
                size_n,
                group_size,
                is_a_8bit=False,
            )
            marlin_scale, scale_factor = _nvfp4_marlin_process_scales(
                raw_marlin_scale,
                a_dtype=weight.dtype,
            )
            global_scale = _nvfp4_marlin_process_global_scale(
                raw_global_scale.reshape(1).contiguous(),
                weight.dtype,
            ).to(torch.float32)
            global_scale = (global_scale / scale_factor).contiguous()
        elif quant_name == "mxfp4":
            q_weight, fp8_scales, dequant = _quantize_mxfp4_weight(weight, group_size)
            q_marlin = fp4_e2m1_weight_to_marlin_weight(q_weight)
            marlin_scale = marlin_permute_scales(
                fp8_scales,
                size_k,
                size_n,
                group_size,
                is_a_8bit=False,
            )
            global_scale = None
        elif quant_name == "fp8":
            fp8_weight, scale, dequant = _quantize_fp8_weight(weight, group_size)
            q_marlin = fp8_weight_to_marlin_weight(fp8_weight)
            marlin_scale = marlin_permute_scales(
                _fp8_fused_exponent_bias_into_scales(scale).to(torch.float16),
                size_k,
                size_n,
                group_size,
                is_a_8bit=False,
            )
            global_scale = None
        else:
            q_weight, scale = _quantize_unsigned_with_bias(
                weight,
                group_size,
                quant_type.bias,
            )
            q_marlin = quant_utils.marlin_weights(
                q_weight,
                size_k,
                size_n,
                quant_type.size_bits,
                quant_utils.get_weight_perm(quant_type.size_bits, is_a_8bit=False),
                is_a_8bit=False,
            )
            marlin_scale = marlin_permute_scales(
                scale,
                size_k,
                size_n,
                group_size,
                is_a_8bit=False,
            )
            runtime_group_size = size_k if group_size == -1 else group_size
            expanded_scale = scale.to(torch.float32).repeat_interleave(runtime_group_size, dim=0)[
                :size_k
            ]
            dequant = (
                (q_weight.to(torch.float32) - float(quant_type.bias)) * expanded_scale
            ).to(torch.float16)
            global_scale = None

        q_weights.append(q_marlin)
        scales.append(marlin_scale)
        dequantized.append(dequant)
        if global_scale is not None:
            global_scales.append(global_scale.reshape(-1)[0])
        g_indices.append(torch.empty((0,), dtype=torch.int, device=weights.device))
        perms.append(torch.empty((0,), dtype=torch.int, device=weights.device))

    zeros_tensor = torch.stack(zeros) if zeros else None
    global_scale_tensor = torch.stack(global_scales).contiguous() if global_scales else None
    return {
        "q_weight": torch.stack(q_weights),
        "scales": torch.stack(scales),
        "zeros": zeros_tensor,
        "global_scale": global_scale_tensor,
        "dequant": dequantized,
        "g_idx": torch.stack(g_indices),
        "perm": torch.stack(perms),
    }


def _repeat_moe_quantized_result(
    quantized: _MoeQuantizedWeight,
    expert_count: int,
) -> _MoeQuantizedWeight:
    current = int(cast(torch.Tensor, quantized["q_weight"]).shape[0])
    if current == expert_count:
        return quantized
    repeats = math.ceil(expert_count / current)

    def _repeat_tensor(value: torch.Tensor) -> torch.Tensor:
        dims = (repeats,) + (1,) * (value.dim() - 1)
        return value.repeat(dims)[:expert_count].contiguous()

    repeated: _MoeQuantizedWeight = {
        "q_weight": _repeat_tensor(cast(torch.Tensor, quantized["q_weight"])),
        "scales": _repeat_tensor(cast(torch.Tensor, quantized["scales"])),
        "zeros": None,
        "global_scale": None,
        "dequant": cast(list[torch.Tensor], quantized["dequant"]) * repeats,
        "g_idx": _repeat_tensor(cast(torch.Tensor, quantized["g_idx"])),
        "perm": _repeat_tensor(cast(torch.Tensor, quantized["perm"])),
    }
    repeated["dequant"] = cast(list[torch.Tensor], repeated["dequant"])[:expert_count]
    zeros = quantized["zeros"]
    if isinstance(zeros, torch.Tensor):
        repeated["zeros"] = _repeat_tensor(zeros)
    global_scale = quantized["global_scale"]
    if isinstance(global_scale, torch.Tensor):
        repeated["global_scale"] = _repeat_tensor(global_scale)
    return repeated


def _moe_stage1_reference_fast(inputs: dict[str, object]) -> torch.Tensor:
    hidden_states = inputs["hidden_states"]
    topk_ids = inputs["topk_ids"]
    tokens = int(inputs["tokens"])
    topk = int(inputs["topk"])
    dequantized = inputs["w1_dequant"]
    bias = inputs.get("w1_bias_raw")
    assert isinstance(hidden_states, torch.Tensor)
    assert isinstance(topk_ids, torch.Tensor)
    assert isinstance(dequantized, list)
    assert bias is None or isinstance(bias, torch.Tensor)
    route_count = tokens * topk
    flat_topk_ids = topk_ids.reshape(route_count)
    route_token_ids = (
        torch.arange(tokens, device=hidden_states.device, dtype=torch.long)
        .repeat_interleave(topk)
        .contiguous()
    )
    output = torch.empty(
        (route_count, dequantized[0].shape[1]),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    for expert_tensor in torch.unique(flat_topk_ids).tolist():
        expert = int(expert_tensor)
        route_mask = flat_topk_ids == expert
        route_indices = torch.nonzero(route_mask, as_tuple=False).reshape(-1)
        token_indices = route_token_ids.index_select(0, route_indices)
        expert_output = torch.matmul(
            hidden_states.index_select(0, token_indices).to(torch.float32),
            dequantized[expert].to(torch.float32),
        )
        if bias is not None:
            expert_output = expert_output + bias[expert].to(torch.float32)
        output.index_copy_(0, route_indices, expert_output)
    return output.to(torch.float16)


def _moe_stage2_inputs_and_reference_fast(
    inputs: dict[str, object],
    *,
    moe_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = int(inputs["tokens"])
    topk = int(inputs["topk"])
    intermediate = int(inputs["intermediate"])
    experts = int(inputs["experts"])
    topk_ids = inputs["topk_ids"]
    dequantized = inputs["w2_dequant"]
    bias = inputs.get("w2_bias_raw")
    assert isinstance(topk_ids, torch.Tensor)
    assert isinstance(dequantized, list)
    assert bias is None or isinstance(bias, torch.Tensor)
    activation = torch.randn(
        (tokens * topk, intermediate), device="cuda", dtype=torch.float16
    )
    stage2_ids = topk_ids.reshape(tokens * topk, 1).contiguous()
    stage2_weights = torch.ones(
        (tokens * topk, 1), device="cuda", dtype=torch.float32
    )
    sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
        stage2_ids,
        block_size=moe_block_size,
        num_experts=experts,
    )
    route_count = tokens * topk
    flat_stage2_ids = stage2_ids.reshape(route_count)
    reference = torch.empty(
        (route_count, dequantized[0].shape[1]),
        device=activation.device,
        dtype=torch.float32,
    )
    for expert_tensor in torch.unique(flat_stage2_ids).tolist():
        expert = int(expert_tensor)
        route_indices = torch.nonzero(
            flat_stage2_ids == expert, as_tuple=False
        ).reshape(-1)
        expert_output = torch.matmul(
            activation.index_select(0, route_indices).to(torch.float32),
            dequantized[expert].to(torch.float32),
        )
        if bias is not None:
            expert_output = expert_output + bias[expert].to(torch.float32)
        reference.index_copy_(0, route_indices, expert_output)
    reference = reference.to(torch.float16)
    return activation, stage2_weights, sorted_ids, expert_ids, num_tokens_post_pad, reference


def _moe_stage1_runtime_case(row: dict[str, Any]) -> _MoeStage1RuntimeCase:
    key = _moe_key_from_row(row)
    quant_name = _quant_name(str(row["quant_format"]))
    try:
        rtol, atol = _MOE_ENV_SWEEP_TOLERANCES[quant_name]
        seed = 2000 + key.tokens + key.hidden + key.intermediate + key.experts + key.topk
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        hidden_states = torch.randn((key.tokens, key.hidden), device="cuda", dtype=torch.float16)
        routing_experts = _moe_runtime_expert_count(row)
        topk_weights, topk_ids = _make_uniform_topk_routing(
            tokens=key.tokens,
            experts=routing_experts,
            topk=key.topk,
        )
        weights = torch.randn(
            (routing_experts, key.hidden, 2 * key.intermediate),
            device="cuda",
            dtype=torch.float16,
        )
        weights = weights * (1.0 / math.sqrt(key.hidden))
        quantized = _repeat_moe_quantized_result(
            _quantize_moe_expert_weights(
                weights,
                quant_name=quant_name,
                group_size=key.group_size,
            ),
            key.experts,
        )
        inputs: dict[str, object] = {
            "tokens": key.tokens,
            "hidden": key.hidden,
            "intermediate": key.intermediate,
            "experts": key.experts,
            "topk": key.topk,
            "hidden_states": hidden_states,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "w1_q": quantized["q_weight"],
            "w1_scales": quantized["scales"],
            "w1_zeros": quantized["zeros"],
            "w1_global_scale": quantized["global_scale"],
            "w1_dequant": quantized["dequant"],
            "w1_g_idx": quantized["g_idx"],
            "w1_perm": quantized["perm"],
        }
        if bool(row.get("has_bias", False)):
            raw_bias = torch.randn(
                (key.experts, 2 * key.intermediate),
                device="cuda",
                dtype=torch.float16,
            )
            inputs["w1_bias_raw"] = raw_bias
            inputs["w1_bias"] = marlin_permute_bias(raw_bias)
        reference = _moe_stage1_reference_fast(inputs)
        return key, inputs, reference, rtol, atol
    except (AssertionError, KeyError, ValueError, RuntimeError, TypeError) as exc:
        raise AssertionError(
            "MoE stage1 helper does not support actual row: "
            f"{_row_context(row, _ACTUAL_MOE_CONTEXT_KEYS)}"
        ) from exc


def _moe_stage2_runtime_case(
    row: dict[str, Any],
) -> _MoeStage2RuntimeCase:
    key = _moe_key_from_row(row)
    moe_block_size = int(row["moe_block_size"])
    quant_name = _quant_name(str(row["quant_format"]))
    try:
        rtol, atol = _MOE_ENV_SWEEP_TOLERANCES[quant_name]
        seed = 3000 + key.tokens + key.hidden + key.intermediate + key.experts + key.topk
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        weights = torch.randn(
            (_moe_runtime_expert_count(row), key.intermediate, key.hidden),
            device="cuda",
            dtype=torch.float16,
        )
        weights = weights * (1.0 / math.sqrt(key.intermediate))
        quantized = _repeat_moe_quantized_result(
            _quantize_moe_expert_weights(
                weights,
                quant_name=quant_name,
                group_size=key.group_size,
            ),
            key.experts,
        )
        topk_ids = torch.empty((key.tokens, key.topk), device="cuda", dtype=torch.int32)
        for token_idx in range(key.tokens):
            for route_idx in range(key.topk):
                topk_ids[token_idx, route_idx] = (
                    token_idx + route_idx
                ) % quantized["q_weight"].shape[0]
        inputs: dict[str, object] = {
            "tokens": key.tokens,
            "intermediate": key.intermediate,
            "experts": key.experts,
            "topk": key.topk,
            "topk_ids": topk_ids,
            "w2_q": quantized["q_weight"],
            "w2_scales": quantized["scales"],
            "w2_zeros": quantized["zeros"],
            "w2_global_scale": quantized["global_scale"],
            "w2_dequant": quantized["dequant"],
            "w2_g_idx": quantized["g_idx"],
            "w2_perm": quantized["perm"],
        }
        if bool(row.get("has_bias", False)):
            raw_bias = torch.randn(
                (key.experts, key.hidden),
                device="cuda",
                dtype=torch.float16,
            )
            inputs["w2_bias_raw"] = raw_bias
            inputs["w2_bias"] = marlin_permute_bias(raw_bias)
        activation, topk_weights, sorted_ids, expert_ids, num_tokens_post_pad, reference = (
            _moe_stage2_inputs_and_reference_fast(inputs, moe_block_size=moe_block_size)
        )
        return (
            key,
            inputs,
            activation,
            topk_weights,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            reference,
            rtol,
            atol,
        )
    except (AssertionError, KeyError, ValueError, RuntimeError, TypeError) as exc:
        raise AssertionError(
            "MoE stage2 helper does not support actual row: "
            f"{_row_context(row, _ACTUAL_MOE_CONTEXT_KEYS)}"
        ) from exc


def _dense_key_from_row(row: dict[str, Any]) -> DenseDirectOpKey:
    _assert_dense_actual_row_supported(row)
    return DenseDirectOpKey(
        _quant_name(str(row["quant_format"])),
        _helper_group_size(row),
        int(row["size_m"]),
        int(row["size_n"]),
        int(row["size_k"]),
    )


def _moe_key_from_row(row: dict[str, Any]) -> MoeDirectOpKey:
    _assert_moe_actual_row_supported(row)
    if row["op"] == "w13":
        size_n = int(row["size_n"])
        if size_n % 2 != 0:
            raise AssertionError(
                "w13 row size_n must be even because it represents gate_up: "
                f"{_row_context(row, _ACTUAL_MOE_CONTEXT_KEYS)}"
            )
        return MoeDirectOpKey(
            _quant_name(str(row["quant_format"])),
            _helper_group_size(row),
            int(row["size_m"]),
            int(row["size_k"]),
            size_n // 2,
            int(row["local_num_experts"]),
            int(row["top_k"]),
        )
    return MoeDirectOpKey(
        _quant_name(str(row["quant_format"])),
        _helper_group_size(row),
        int(row["size_m"]),
        int(row["size_n"]),
        int(row["size_k"]),
        int(row["local_num_experts"]),
        1,
    )


def _unique_actual_dense_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in payload["dense"]:
        _assert_dense_row_schema(row)
        if row["call_status"] != "actual_marlin":
            continue
        key = _dense_key_from_row(row)
        marker = (
            key.quant_name,
            key.group_size,
            key.size_m,
            key.size_n,
            key.size_k,
            bool(row.get("has_bias", False)),
        )
        if marker in seen:
            continue
        seen.add(marker)
        out.append(row)
    return out


def _unique_actual_moe_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in payload["moe"]:
        _assert_moe_row_schema(row)
        if row["call_status"] != "actual_marlin":
            continue
        key = _moe_key_from_row(row)
        marker = (
            row["op"],
            key.quant_name,
            key.group_size,
            int(row["moe_block_size"]),
            key.tokens,
            key.hidden,
            key.intermediate,
            key.experts,
            key.topk,
            bool(row.get("has_bias", False)),
        )
        if marker in seen:
            continue
        seen.add(marker)
        out.append(row)
    return out


def _make_dense_runtime_case(row: dict[str, Any]) -> tuple[DenseDirectOpKey, tuple]:
    key = _dense_key_from_row(row)
    try:
        args, output, reference, rtol, atol = _make_dense_env_sweep_case(key)
        if bool(row.get("has_bias", False)):
            raw_bias = torch.randn(
                (key.size_n,),
                device="cuda",
                dtype=torch.float16,
            )
            args_list = list(args)
            args_list[3] = marlin_permute_bias(raw_bias)
            args = tuple(args_list)
            reference = (reference.to(torch.float32) + raw_bias.to(torch.float32)).to(
                torch.float16
            )
        return key, (args, output, reference, rtol, atol)
    except (AssertionError, KeyError, ValueError) as exc:
        raise AssertionError(
            "dense helper does not support actual row: "
            f"{_row_context(row, _ACTUAL_DENSE_CONTEXT_KEYS)}"
        ) from exc


def _make_moe_runtime_case(row: dict[str, Any]) -> tuple[MoeDirectOpKey, Callable[[], None]]:
    support = _moe_runtime_support(row)
    if not support.supported:
        raise AssertionError(
            "MoE helper does not support actual row: "
            f"{_row_context(row, _ACTUAL_MOE_CONTEXT_KEYS)}; reason={support.reason}"
        )

    moe_block_size = int(row["moe_block_size"])
    if row["op"] == "w13":
        key, inputs, reference, rtol, atol = _moe_stage1_runtime_case(row)

        def run() -> None:
            _run_moe_env_stage1_combo(
                key,
                inputs,
                moe_block_size=moe_block_size,
                reference=reference,
                rtol=rtol,
                atol=atol,
            )

        return key, run

    (
        key,
        inputs,
        activation,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        reference,
        rtol,
        atol,
    ) = _moe_stage2_runtime_case(row)

    def run() -> None:
        _run_moe_env_stage2_combo(
            key,
            inputs,
            activation=activation,
            topk_weights=topk_weights,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_pad=num_tokens_post_pad,
            moe_block_size=moe_block_size,
            reference=reference,
            rtol=rtol,
            atol=atol,
        )

    return key, run


def _dense_bias_smoke_row() -> dict[str, Any]:
    return {
        "model": "synthetic",
        "scenario": "tp1",
        "phase": "smoke",
        "layer_key": "model.layers.0.mlp.gate_up_proj",
        "op": "gate_up_proj",
        "target_op": "ops.marlin_gemm",
        "size_m": 4,
        "size_n": 128,
        "size_k": 128,
        "group_size": 128,
        "quant_method": "gptq",
        "quant_format": "uint4b8",
        "has_zp": False,
        "has_bias": True,
        "marlin_path": "wna16_marlin",
        "call_status": "actual_marlin",
        "call_count": 1,
    }


def _dense_fp8_per_tensor_smoke_row() -> dict[str, Any]:
    return {
        "model": "synthetic",
        "scenario": "tp1",
        "phase": "smoke",
        "layer_key": "model.layers.0.self_attn.o_proj",
        "op": "o_proj",
        "target_op": "ops.marlin_gemm",
        "size_m": 4,
        "size_n": 128,
        "size_k": 256,
        "group_size": 256,
        "quant_method": "modelopt",
        "quant_format": "fp8_e4m3",
        "has_zp": False,
        "has_bias": False,
        "marlin_path": "fp8_marlin",
        "call_status": "actual_marlin",
        "call_count": 1,
    }


def _moe_bias_smoke_row(op: str) -> dict[str, Any]:
    if op == "w13":
        size_m = 4
        size_n = 256
        size_k = 128
        top_k = 2
    else:
        size_m = 8
        size_n = 128
        size_k = 128
        top_k = 1
    return {
        "model": "synthetic",
        "scenario": "tp1",
        "phase": "smoke",
        "layer_key": "model.layers.0.mlp.experts",
        "op": op,
        "target_op": "ops.moe_wna16_marlin_gemm",
        "moe_block_size": 16,
        "top_k": top_k,
        "size_m": size_m,
        "size_n": size_n,
        "size_k": size_k,
        "group_size": 128,
        "quant_method": "gptq",
        "quant_format": "uint4b8",
        "has_zp": False,
        "has_bias": True,
        "marlin_path": "wna16_marlin",
        "local_num_experts": 4,
        "global_num_experts": 4,
        "intermediate_size_per_partition": 128,
        "call_status": "actual_marlin",
        "call_count": 1,
    }


def test_model_shape_dense_fp8_size_k_group_maps_to_runtime_minus_one() -> None:
    row = _dense_fp8_per_tensor_smoke_row()
    key = _dense_key_from_row(row)
    assert key.group_size == -1

    bad_row = dict(row, group_size=None)
    with pytest.raises(AssertionError, match="group_size"):
        _dense_key_from_row(bad_row)


def test_model_shape_dense_bias_runtime_helper_matches_reference() -> None:
    row = _dense_bias_smoke_row()
    key, prepared = _make_dense_runtime_case(row)
    args, output, reference, rtol, atol = prepared
    assert args[3] is not None

    legal_combo = next(
        (geometry, split_k, metadata_cache)
        for geometry, split_k, metadata_cache in iter_env_combinations()
        if dense_env_combo_is_legal(
            geometry,
            split_k,
            size_n=key.size_n,
            size_k=key.size_k,
        )
    )
    with dense_env(*legal_combo):
        _assert_dense_env_sweep_combo_matches_reference(
            key,
            args,
            output,
            reference,
            rtol=rtol,
            atol=atol,
        )


def test_model_shape_moe_bias_runtime_helpers_match_reference() -> None:
    _require_moe_cuda()

    stage1_row = _moe_bias_smoke_row("w13")
    key, inputs, reference, rtol, atol = _moe_stage1_runtime_case(stage1_row)
    assert inputs["w1_bias"] is not None
    legal_stage1 = next(
        (geometry, split_k, metadata_cache)
        for geometry, split_k, metadata_cache in iter_moe_env_combinations()
        if moe_stage_env_combo_is_legal(
            geometry,
            size_n=2 * key.intermediate,
            size_k=key.hidden,
        )
    )
    with moe_env(*legal_stage1):
        _run_moe_env_stage1_combo(
            key,
            inputs,
            moe_block_size=int(stage1_row["moe_block_size"]),
            reference=reference,
            rtol=rtol,
            atol=atol,
        )

    stage2_row = _moe_bias_smoke_row("w2")
    (
        key,
        inputs,
        activation,
        topk_weights,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        reference,
        rtol,
        atol,
    ) = _moe_stage2_runtime_case(stage2_row)
    assert inputs["w2_bias"] is not None
    legal_stage2 = next(
        (geometry, split_k, metadata_cache)
        for geometry, split_k, metadata_cache in iter_moe_env_combinations()
        if moe_stage_env_combo_is_legal(
            geometry,
            size_n=key.hidden,
            size_k=key.intermediate,
        )
    )
    with moe_env(*legal_stage2):
        _run_moe_env_stage2_combo(
            key,
            inputs,
            activation=activation,
            topk_weights=topk_weights,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            num_tokens_post_pad=num_tokens_post_pad,
            moe_block_size=int(stage2_row["moe_block_size"]),
            reference=reference,
            rtol=rtol,
            atol=atol,
        )


def _selected_env_summary(rows: list[dict[str, Any]], combos_per_row: int) -> tuple[int, int, int | None]:
    start, limit = exhaustive_start_limit()
    possible = len(rows) * combos_per_row
    if start >= possible:
        return possible, start, 0
    if limit is None:
        selected = possible - start
    else:
        selected = min(limit, possible - start)
    return possible, start, selected


def _env_row_range_is_selected(row_start: int, combos_per_row: int) -> bool:
    row_end = row_start + combos_per_row
    start, limit = exhaustive_start_limit()
    if limit is None:
        return row_end > start
    return row_start < start + limit and row_end > start


def _dense_row_progress(row: dict[str, Any], row_index: int, row_count: int) -> str:
    return (
        f"[marlin-model-shapes][dense] row {row_index}/{row_count}: "
        f"scenario={row.get('scenario')} phase={row.get('phase')} op={row.get('op')} "
        f"m={row.get('size_m')} n={row.get('size_n')} k={row.get('size_k')} "
        f"quant={row.get('quant_format')} group={row.get('group_size')} "
        f"has_bias={bool(row.get('has_bias', False))}"
    )


def _moe_row_progress(row: dict[str, Any], row_index: int, row_count: int) -> str:
    return (
        f"[marlin-model-shapes][moe] row {row_index}/{row_count}: "
        f"scenario={row.get('scenario')} phase={row.get('phase')} op={row.get('op')} "
        f"m={row.get('size_m')} n={row.get('size_n')} k={row.get('size_k')} "
        f"top_k={row.get('top_k')} experts={row.get('local_num_experts')} "
        f"block={row.get('moe_block_size')} quant={row.get('quant_format')} "
        f"group={row.get('group_size')} has_bias={bool(row.get('has_bias', False))}"
    )


def _moe_unsupported_row_progress(
    row: dict[str, Any],
    row_index: int,
    row_count: int,
    reason: str,
) -> str:
    return (
        f"[marlin-model-shapes][moe] unsupported runtime row "
        f"{row_index}/{row_count}: "
        f"scenario={row.get('scenario')} phase={row.get('phase')} op={row.get('op')} "
        f"m={row.get('size_m')} n={row.get('size_n')} k={row.get('size_k')} "
        f"top_k={row.get('top_k')} experts={row.get('local_num_experts')} "
        f"block={row.get('moe_block_size')} quant={row.get('quant_format')} "
        f"group={row.get('group_size')} has_bias={bool(row.get('has_bias', False))} "
        f"reason={reason}"
    )


def _heartbeat(
    request: pytest.FixtureRequest,
    *,
    kind: str,
    checked: int,
    legal: int,
    rejected: int,
    total_index: int,
) -> None:
    if checked == 0 or checked % _PROGRESS_HEARTBEAT_INTERVAL != 0:
        return
    _emit_progress(
        request,
        f"[marlin-model-shapes][{kind}] heartbeat: "
        f"checked={checked} legal={legal} rejected={rejected} total_index={total_index}",
    )


def test_marlin_model_table_payload_and_pretty_smoke(
    model_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _load_payload(model_dir)
    _assert_payload_shape(payload)
    _validate_row_schema(payload)

    pretty_output = _capture_pretty(model_dir, capsys)
    _assert_pretty_output(payload, pretty_output)


@pytest.mark.sm70_env_exhaustive
def test_marlin_model_dense_table_runtime_matches_reference(
    model_dir: Path,
    request: pytest.FixtureRequest,
) -> None:
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the full env sweep")

    _emit_progress(request, f"[marlin-model-shapes][dense] loading payload: model={model_dir}")
    payload = _load_payload(model_dir)
    actual_rows = _unique_actual_dense_rows(payload)
    if not actual_rows:
        pytest.skip("no actual dense rows to test")

    combos_per_row = sum(1 for _ in iter_env_combinations())
    possible, start, selected = _selected_env_summary(actual_rows, combos_per_row)
    _emit_progress(
        request,
        f"[marlin-model-shapes][dense] start: model={model_dir} "
        f"actual_rows={len(actual_rows)} combos_per_row={combos_per_row} "
        f"possible={possible} start={start} limit={_format_limit(exhaustive_start_limit()[1])} "
        f"selected={selected}",
    )

    total = 0
    checked = 0
    legal = 0
    rejected = 0

    combos = tuple(iter_env_combinations())
    for row_index, row in enumerate(actual_rows, start=1):
        if not _env_row_range_is_selected(total, combos_per_row):
            total += combos_per_row
            if exhaustive_index_is_past_limit(total):
                break
            continue
        _emit_progress(request, _dense_row_progress(row, row_index, len(actual_rows)))
        key, prepared = _make_dense_runtime_case(row)
        args, output, reference, rtol, atol = prepared
        for geometry, split_k, metadata_cache in combos:
            if exhaustive_index_is_past_limit(total):
                break
            selected = exhaustive_index_is_selected(total)
            total += 1
            if not selected:
                continue

            checked += 1
            is_legal = dense_env_combo_is_legal(
                geometry,
                split_k,
                size_n=key.size_n,
                size_k=key.size_k,
            )
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
                        _assert_dense_env_sweep_combo_matches_reference(
                            key,
                            args,
                            output,
                            reference,
                            rtol=rtol,
                            atol=atol,
                        )
                    rejected += 1
            _heartbeat(
                request,
                kind="dense",
                checked=checked,
                legal=legal,
                rejected=rejected,
                total_index=total,
            )
        if exhaustive_index_is_past_limit(total):
            break

    _emit_progress(
        request,
        f"[marlin-model-shapes][dense] summary: checked={checked} legal={legal} "
        f"rejected={rejected} total_seen={total}",
    )
    assert checked > 0
    assert legal + rejected == checked


@pytest.mark.sm70_env_exhaustive
def test_marlin_model_moe_table_runtime_matches_reference(
    model_dir: Path,
    request: pytest.FixtureRequest,
) -> None:
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the full env sweep")

    _emit_progress(request, f"[marlin-model-shapes][moe] loading payload: model={model_dir}")
    payload = _load_payload(model_dir)
    actual_rows = _unique_actual_moe_rows(payload)
    if not actual_rows:
        pytest.skip("no actual moe rows to test")

    supported_rows: list[dict[str, Any]] = []
    unsupported_rows: list[tuple[int, dict[str, Any], str]] = []
    for row_index, row in enumerate(actual_rows, start=1):
        support = _moe_runtime_support(row)
        if support.supported:
            supported_rows.append(row)
        else:
            unsupported_rows.append((row_index, row, support.reason))

    for row_index, row, reason in unsupported_rows:
        _emit_progress(
            request,
            _moe_unsupported_row_progress(row, row_index, len(actual_rows), reason),
        )

    if not supported_rows:
        pytest.skip("no supported actual moe runtime rows to test")

    combos_per_row = sum(1 for _ in iter_moe_env_combinations())
    possible, start, selected = _selected_env_summary(supported_rows, combos_per_row)
    _emit_progress(
        request,
        f"[marlin-model-shapes][moe] start: model={model_dir} "
        f"actual_rows={len(actual_rows)} supported_runtime_rows={len(supported_rows)} "
        f"unsupported_runtime_rows={len(unsupported_rows)} "
        f"combos_per_row={combos_per_row} "
        f"possible={possible} start={start} limit={_format_limit(exhaustive_start_limit()[1])} "
        f"selected={selected}",
    )

    total = 0
    checked = 0
    legal = 0
    rejected = 0

    _require_moe_cuda()
    combos = tuple(iter_moe_env_combinations())
    for row_index, row in enumerate(supported_rows, start=1):
        if not _env_row_range_is_selected(total, combos_per_row):
            total += combos_per_row
            if exhaustive_index_is_past_limit(total):
                break
            continue
        _emit_progress(request, _moe_row_progress(row, row_index, len(supported_rows)))
        key, run = _make_moe_runtime_case(row)
        for geometry, split_k, metadata_cache in combos:
            if exhaustive_index_is_past_limit(total):
                break
            selected = exhaustive_index_is_selected(total)
            total += 1
            if not selected:
                continue

            checked += 1
            is_legal = moe_stage_env_combo_is_legal(
                geometry,
                size_n=int(row["size_n"]),
                size_k=int(row["size_k"]),
            )
            with moe_env(geometry, split_k, metadata_cache):
                if is_legal:
                    run()
                    legal += 1
                else:
                    with pytest.raises(RuntimeError, match=EXPLICIT_ENV_REJECTION_RE):
                        run()
                    rejected += 1
            _heartbeat(
                request,
                kind="moe",
                checked=checked,
                legal=legal,
                rejected=rejected,
                total_index=total,
            )
        if exhaustive_index_is_past_limit(total):
            break

    _emit_progress(
        request,
        f"[marlin-model-shapes][moe] summary: checked={checked} legal={legal} "
        f"rejected={rejected} unsupported_rows={len(unsupported_rows)} "
        f"total_seen={total}",
    )
    assert checked > 0
    assert legal + rejected == checked
