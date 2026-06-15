from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tests import ops
from tests.sm70_env_sweep import (
    EXPLICIT_ENV_REJECTION_RE,
    DenseDirectOpKey,
    MoeDirectOpKey,
    dense_env,
    dense_env_combo_is_legal,
    exhaustive_enabled,
    exhaustive_index_is_past_limit,
    exhaustive_index_is_selected,
    iter_env_combinations,
    iter_moe_env_combinations,
    moe_env,
    moe_stage_env_combo_is_legal,
)
from tests.writeback_marlin_cases import (
    is_dense_group_size_supported,
    is_moe_group_size_supported,
)
from tests.test_marlin_dense import (
    _assert_dense_env_sweep_combo_matches_reference,
    _make_dense_env_sweep_case,
)
from tests.test_marlin_moe import (
    _MOE_ENV_SWEEP_TOLERANCES,
    _make_moe_env_sweep_inputs,
    _moe_stage1_reference,
    _moe_stage2_inputs_and_reference,
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
    if row["op"] == "w13":
        hidden = int(row["size_k"])
        intermediate = int(row["size_n"]) // 2
    else:
        hidden = int(row["size_n"])
        intermediate = int(row["size_k"])
    if not is_moe_group_size_supported(
        quant_name,
        _helper_group_size(row),
        hidden,
        intermediate,
    ):
        raise AssertionError(
            "unsupported MoE group_size in actual row: "
            f"{_row_context(row, _ACTUAL_MOE_CONTEXT_KEYS)}"
        )


def _helper_group_size(row: dict[str, Any]) -> int:
    group_size = int(row["group_size"])
    size_k = int(row["size_k"])
    if group_size == size_k:
        return -1
    return group_size


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
        )
        if marker in seen:
            continue
        seen.add(marker)
        out.append(row)
    return out


def _make_dense_runtime_case(row: dict[str, Any]) -> tuple[DenseDirectOpKey, tuple]:
    key = _dense_key_from_row(row)
    try:
        return key, _make_dense_env_sweep_case(key)
    except (AssertionError, KeyError, ValueError) as exc:
        raise AssertionError(
            "dense helper does not support actual row: "
            f"{_row_context(row, _ACTUAL_DENSE_CONTEXT_KEYS)}"
        ) from exc


def _make_moe_runtime_case(row: dict[str, Any]) -> tuple[MoeDirectOpKey, Callable[[], None]]:
    key = _moe_key_from_row(row)
    moe_block_size = int(row["moe_block_size"])
    quant_name = _quant_name(str(row["quant_format"]))
    try:
        rtol, atol = _MOE_ENV_SWEEP_TOLERANCES[quant_name]
        inputs = _make_moe_env_sweep_inputs(key)
        if row["op"] == "w13":
            reference = _moe_stage1_reference(inputs)

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

        stage2 = _moe_stage2_inputs_and_reference(
            inputs,
            moe_block_size=moe_block_size,
        )
        (
            activation,
            topk_weights,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            reference,
        ) = stage2

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
    except (AssertionError, KeyError, ValueError) as exc:
        raise AssertionError(
            "MoE helper does not support actual row: "
            f"{_row_context(row, _ACTUAL_MOE_CONTEXT_KEYS)}"
        ) from exc


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
def test_marlin_model_dense_table_runtime_matches_reference(model_dir: Path) -> None:
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the full env sweep")

    payload = _load_payload(model_dir)
    actual_rows = _unique_actual_dense_rows(payload)
    if not actual_rows:
        pytest.skip("no actual dense rows to test")

    total = 0
    checked = 0
    legal = 0
    rejected = 0

    for row in actual_rows:
        key, prepared = _make_dense_runtime_case(row)
        args, output, reference, rtol, atol = prepared
        for geometry, split_k, metadata_cache in iter_env_combinations():
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
        if exhaustive_index_is_past_limit(total):
            break

    assert checked > 0
    assert legal + rejected == checked


@pytest.mark.sm70_env_exhaustive
def test_marlin_model_moe_table_runtime_matches_reference(model_dir: Path) -> None:
    if not exhaustive_enabled():
        pytest.skip("set MARLIN_EXHAUSTIVE_ENV_SWEEP=1 to run the full env sweep")

    payload = _load_payload(model_dir)
    actual_rows = _unique_actual_moe_rows(payload)
    if not actual_rows:
        pytest.skip("no actual moe rows to test")

    total = 0
    checked = 0
    legal = 0
    rejected = 0

    for row in actual_rows:
        key, run = _make_moe_runtime_case(row)
        for geometry, split_k, metadata_cache in iter_moe_env_combinations():
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
        if exhaustive_index_is_past_limit(total):
            break

    assert checked > 0
    assert legal + rejected == checked
