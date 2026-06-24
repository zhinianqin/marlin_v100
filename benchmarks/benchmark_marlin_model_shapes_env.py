from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

try:
    from common import (
        banner,
        format_float,
        require_matching_cuda_benchmark_runtime,
        time_cuda_callable,
        timestamp,
    )
except ModuleNotFoundError:
    from benchmarks.common import (
        banner,
        format_float,
        require_matching_cuda_benchmark_runtime,
        time_cuda_callable,
        timestamp,
    )

from tests import ops
from tests.helpers import moe_align_block_size
from tests.sm70_env_sweep import (
    EXPLICIT_ENV_REJECTION_RE,
    DenseDirectOpKey,
    MoeDirectOpKey,
    dense_env,
    dense_env_combo_is_legal,
    exhaustive_start_limit,
    iter_env_combinations,
    iter_moe_env_combinations,
    moe_env,
    moe_stage_env_combo_is_legal,
)
from tests.test_marlin_model_shapes_env import (
    _assert_dense_env_sweep_combo_matches_reference,
    _dense_row_progress,
    _make_dense_runtime_case,
    _moe_row_progress,
    _moe_runtime_support,
    _moe_stage1_runtime_case,
    _moe_stage2_runtime_case,
    _moe_unsupported_row_progress,
    _unique_actual_dense_rows,
    _unique_actual_moe_rows,
)
from tests.test_marlin_moe import (
    _moe_quant_type_id,
    _run_moe_env_stage1_combo,
    _run_moe_env_stage2_combo,
)


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "benchmarks" / "marlin_gemm_shapes.py"
SPEC = importlib.util.spec_from_file_location(
    "benchmark_marlin_gemm_shapes",
    SCRIPT,
)
assert SPEC is not None
marlin_gemm_shapes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = marlin_gemm_shapes
SPEC.loader.exec_module(marlin_gemm_shapes)


CSV_FIELDNAMES = [
    "kind",
    "model",
    "scenario",
    "phase",
    "op",
    "layer_key",
    "row_call_count",
    "quant_method",
    "quant_format",
    "group_size",
    "has_bias",
    "size_m",
    "size_n",
    "size_k",
    "env_cta_geometry",
    "env_split_k",
    "env_metadata_cache",
    "env_legal",
    "status",
    "marlin_us",
    "flops",
    "marlin_tflops",
    "check_pass",
    "reason",
    "moe_block_size",
    "top_k",
    "local_num_experts",
    "global_num_experts",
    "intermediate_size_per_partition",
]
_HEARTBEAT_INTERVAL = 64


@dataclass(frozen=True)
class DensePrepared:
    key: DenseDirectOpKey
    check: Callable[[], None]
    run: Callable[[], object]


@dataclass(frozen=True)
class MoePrepared:
    key: MoeDirectOpKey
    check: Callable[[], None]
    run: Callable[[], object]


@dataclass(frozen=True)
class TableStats:
    rows: int
    actual_rows: int
    unique_actual_rows: int
    non_actual_rows: int
    unsupported_runtime_rows: int = 0
    supported_runtime_rows: int = 0


@dataclass
class RunState:
    checked: int = 0
    selected_seen: int = 0
    saved_rows: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Marlin model-shape table rows across SM70 env combos."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Local model directory containing config.json.",
    )
    parser.add_argument(
        "--kind",
        choices=("auto", "dense", "moe", "both"),
        default="auto",
        help="Which table kind to benchmark. auto detects actual Dense/MoE rows.",
    )
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--max-cases", type=int)
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    model_dir = args.model.expanduser()
    if not model_dir.exists():
        raise FileNotFoundError(f"--model path does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"--model must be a model directory: {model_dir}")
    if not (model_dir / "config.json").exists():
        raise FileNotFoundError(
            "--model must point to a model directory containing config.json: "
            f"{model_dir}"
        )
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be non-negative.")
    if args.iters <= 0:
        raise ValueError("--iters must be positive.")
    if args.max_cases is not None and args.max_cases < 0:
        raise ValueError("--max-cases must be non-negative.")


def load_payload(model_dir: Path) -> dict[str, Any]:
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


def _default_csv_path() -> Path:
    stamp = timestamp().replace(" ", "_").replace(":", "")
    return Path("benchmarks/results") / f"{stamp}_model_shapes_env_benchmark.csv"


def _format_tflops(flops: int, latency_us: float) -> str:
    return f"{flops / (latency_us * 1_000_000):.6f}"


def _first_line(exc: BaseException) -> str:
    return str(exc).splitlines()[0] if str(exc).splitlines() else repr(exc)


def _bool_cell(value: bool) -> str:
    return "true" if value else "false"


def _select_field(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""


def _status_row(
    *,
    kind: str,
    model: str,
    row: dict[str, Any],
    status: str,
    env_legal: bool | None = None,
    geometry_label: str = "",
    split_k: int | str = "",
    metadata_cache: str = "",
    marlin_us: float | None = None,
    check_pass: str = "",
    reason: str = "",
) -> dict[str, str]:
    flops = _row_flops(kind, row)
    return {
        "kind": kind,
        "model": model,
        "scenario": str(row.get("scenario", "")),
        "phase": str(row.get("phase", "")),
        "op": str(row.get("op", "")),
        "layer_key": _select_field(row, "layer_key"),
        "row_call_count": _select_field(row, "call_count", "row_call_count"),
        "quant_method": str(row.get("quant_method", "")),
        "quant_format": str(row.get("quant_format", "")),
        "group_size": str(row.get("group_size", "")),
        "has_bias": _bool_cell(bool(row.get("has_bias", False))),
        "size_m": str(row.get("size_m", "")),
        "size_n": str(row.get("size_n", "")),
        "size_k": str(row.get("size_k", "")),
        "env_cta_geometry": geometry_label,
        "env_split_k": str(split_k),
        "env_metadata_cache": metadata_cache,
        "env_legal": "" if env_legal is None else _bool_cell(env_legal),
        "status": status,
        "marlin_us": "" if marlin_us is None else format_float(marlin_us),
        "flops": str(flops),
        "marlin_tflops": "" if marlin_us is None else _format_tflops(flops, marlin_us),
        "check_pass": check_pass,
        "reason": reason,
        "moe_block_size": str(row.get("moe_block_size", "")) if kind == "moe" else "",
        "top_k": _select_field(row, "top_k") if kind == "moe" else "",
        "local_num_experts": _select_field(row, "local_num_experts") if kind == "moe" else "",
        "global_num_experts": _select_field(row, "global_num_experts") if kind == "moe" else "",
        "intermediate_size_per_partition": (
            _select_field(row, "intermediate_size_per_partition")
            if kind == "moe"
            else ""
        ),
    }


def _row_flops(kind: str, row: dict[str, Any]) -> int:
    del kind
    return 2 * int(row["size_m"]) * int(row["size_n"]) * int(row["size_k"])


def _write_row(writer: csv.DictWriter, state: RunState, row: dict[str, str]) -> None:
    writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})
    state.saved_rows += 1
    state.status_counts[row["status"]] += 1


def _count_actual(rows: Iterable[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("call_status") == "actual_marlin")


def _selected_count(possible: int, start: int, limit: int | None, max_cases: int | None) -> int:
    if start >= possible:
        selected = 0
    elif limit is None:
        selected = possible - start
    else:
        selected = min(limit, possible - start)
    if max_cases is not None:
        selected = min(selected, max_cases)
    return selected


def _index_is_selected(index: int, start: int, limit: int | None) -> bool:
    return index >= start and (limit is None or index < start + limit)


def _past_limit(index: int, start: int, limit: int | None) -> bool:
    return limit is not None and index >= start + limit


def _max_cases_reached(state: RunState, max_cases: int | None) -> bool:
    return max_cases is not None and state.selected_seen >= max_cases


def _prepare_dense_runtime(row: dict[str, Any]) -> DensePrepared:
    key, prepared = _make_dense_runtime_case(row)
    args, output, reference, rtol, atol = prepared

    def check() -> None:
        _assert_dense_env_sweep_combo_matches_reference(
            key,
            args,
            output,
            reference,
            rtol=rtol,
            atol=atol,
        )

    def run() -> object:
        output.zero_()
        return ops.marlin_gemm(*args)

    return DensePrepared(key=key, check=check, run=run)


def _prepare_moe_runtime(row: dict[str, Any]) -> MoePrepared:
    moe_block_size = int(row["moe_block_size"])
    if row["op"] == "w13":
        key, inputs, reference, rtol, atol = _moe_stage1_runtime_case(row)
        sorted_ids, expert_ids, num_tokens_post_pad = moe_align_block_size(
            inputs["topk_ids"],
            block_size=moe_block_size,
            num_experts=inputs["experts"],
        )
        output = torch.empty(
            (key.tokens * key.topk, 2 * key.intermediate),
            device="cuda",
            dtype=torch.float16,
        )

        def check() -> None:
            _run_moe_env_stage1_combo(
                key,
                inputs,
                moe_block_size=moe_block_size,
                reference=reference,
                rtol=rtol,
                atol=atol,
            )

        def run() -> object:
            return ops.moe_wna16_marlin_gemm(
                inputs["hidden_states"],
                output,
                inputs["w1_q"],
                inputs.get("w1_bias"),
                inputs["w1_scales"],
                None,
                inputs["w1_global_scale"],
                inputs["w1_zeros"],
                inputs["w1_g_idx"],
                inputs["w1_perm"],
                None,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                inputs["topk_weights"],
                moe_block_size,
                key.topk,
                False,
                _moe_quant_type_id(key.quant_name),
                key.tokens,
                2 * key.intermediate,
                key.hidden,
                True,
                False,
                False,
                inputs["w1_zeros"] is not None,
                -1,
                -1,
                -1,
            )

        return MoePrepared(key=key, check=check, run=run)

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
    output = torch.empty(
        (key.tokens * key.topk, key.hidden),
        device="cuda",
        dtype=torch.float16,
    )

    def check() -> None:
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

    def run() -> object:
        return ops.moe_wna16_marlin_gemm(
            activation,
            output,
            inputs["w2_q"],
            inputs.get("w2_bias"),
            inputs["w2_scales"],
            None,
            inputs["w2_global_scale"],
            inputs["w2_zeros"],
            inputs["w2_g_idx"],
            inputs["w2_perm"],
            None,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            topk_weights,
            moe_block_size,
            1,
            False,
            _moe_quant_type_id(key.quant_name),
            key.tokens * key.topk,
            key.hidden,
            key.intermediate,
            True,
            False,
            False,
            inputs["w2_zeros"] is not None,
            -1,
            -1,
            -1,
        )

    return MoePrepared(key=key, check=check, run=run)


def _expect_runtime_rejection(fn: Callable[[], object]) -> tuple[bool, str]:
    try:
        fn()
    except RuntimeError as exc:
        if re.search(EXPLICIT_ENV_REJECTION_RE, str(exc)):
            return True, ""
        return False, f"unexpected RuntimeError: {_first_line(exc)}"
    except Exception as exc:
        return False, f"unexpected exception: {_first_line(exc)}"
    return False, "illegal env combo did not raise expected RuntimeError"


def _run_checked_timed_combo(
    prepared: DensePrepared | MoePrepared,
    *,
    warmup_iters: int,
    iters: int,
    check: bool,
) -> tuple[str, float | None, str, str]:
    if check:
        try:
            prepared.check()
        except AssertionError as exc:
            return "MISMATCH", None, "no", _first_line(exc)
        except Exception as exc:
            return "ERR", None, "no", _first_line(exc)
        check_pass = "yes"
    else:
        check_pass = ""

    try:
        stats = time_cuda_callable(
            prepared.run,
            warmup_iters=warmup_iters,
            iters=iters,
        )
    except Exception as exc:
        return "ERR", None, check_pass, _first_line(exc)
    return "OK", stats["median_us"], check_pass, ""


def _selected_kinds(
    requested: str,
    dense_rows: list[dict[str, Any]],
    moe_rows: list[dict[str, Any]],
) -> list[str]:
    if requested == "dense":
        return ["dense"]
    if requested == "moe":
        return ["moe"]
    if requested == "both":
        return ["dense", "moe"]
    out = []
    if dense_rows:
        out.append("dense")
    if moe_rows:
        out.append("moe")
    return out


def _format_kinds(kinds: list[str]) -> str:
    return ",".join(kinds) if kinds else "none"


def _table_stats(
    payload: dict[str, Any],
    dense_rows: list[dict[str, Any]],
    moe_rows: list[dict[str, Any]],
    supported_moe_rows: list[dict[str, Any]],
    unsupported_moe_rows: list[tuple[int, dict[str, Any], str]],
) -> tuple[TableStats, TableStats]:
    dense_total = len(payload["dense"])
    moe_total = len(payload["moe"])
    dense_actual = _count_actual(payload["dense"])
    moe_actual = _count_actual(payload["moe"])
    return (
        TableStats(
            rows=dense_total,
            actual_rows=dense_actual,
            unique_actual_rows=len(dense_rows),
            non_actual_rows=dense_total - dense_actual,
        ),
        TableStats(
            rows=moe_total,
            actual_rows=moe_actual,
            unique_actual_rows=len(moe_rows),
            non_actual_rows=moe_total - moe_actual,
            unsupported_runtime_rows=len(unsupported_moe_rows),
            supported_runtime_rows=len(supported_moe_rows),
        ),
    )


def _print_startup(
    *,
    model: str,
    requested_kind: str,
    detected_kinds: list[str],
    dense_stats: TableStats,
    moe_stats: TableStats,
    dense_combo_count: int,
    moe_combo_count: int,
    possible: int,
    start: int,
    limit: int | None,
    selected: int,
    csv_path: Path,
    warmup_iters: int,
    iters: int,
    check: bool,
) -> None:
    banner(f"Marlin Model Shape Env Benchmark ({timestamp()})")
    print(f"model={model}")
    print(f"requested_kind={requested_kind}")
    print(f"detected_kinds={_format_kinds(detected_kinds)}")
    print(
        "dense_rows="
        f"{dense_stats.rows} dense_actual_rows={dense_stats.actual_rows} "
        f"dense_unique_actual_rows={dense_stats.unique_actual_rows} "
        f"dense_non_actual_rows={dense_stats.non_actual_rows}"
    )
    print(
        "moe_rows="
        f"{moe_stats.rows} moe_actual_rows={moe_stats.actual_rows} "
        f"moe_unique_actual_rows={moe_stats.unique_actual_rows} "
        f"moe_supported_runtime_rows={moe_stats.supported_runtime_rows} "
        f"moe_unsupported_runtime_rows={moe_stats.unsupported_runtime_rows} "
        f"moe_non_actual_rows={moe_stats.non_actual_rows}"
    )
    print(f"dense_env_combos={dense_combo_count}")
    print(f"moe_env_combos={moe_combo_count}")
    print(f"possible_selected_combos={possible}")
    print(f"start={start}")
    print(f"limit={'unbounded' if limit is None else limit}")
    print(f"selected={selected}")
    print(f"warmup_iters={warmup_iters}, iters={iters}, check={check}")
    print(f"csv={csv_path}")


def _maybe_heartbeat(state: RunState) -> None:
    if state.checked == 0 or state.checked % _HEARTBEAT_INTERVAL != 0:
        return
    print(
        f"checked={state.checked} "
        f"ok={state.status_counts['OK']} "
        f"rejected={state.status_counts['REJECTED']} "
        f"unsupported={state.status_counts['UNSUPPORTED']} "
        f"err={state.status_counts['ERR']}",
        flush=True,
    )


def _handle_dense_combo(
    *,
    writer: csv.DictWriter,
    state: RunState,
    model: str,
    row: dict[str, Any],
    geometry: Any,
    split_k: int,
    metadata_cache: str,
    prepared: DensePrepared | None,
    prepare_error: BaseException | None,
    warmup_iters: int,
    iters: int,
    check: bool,
) -> tuple[DensePrepared | None, BaseException | None]:
    is_legal = dense_env_combo_is_legal(
        geometry,
        split_k,
        size_n=int(row["size_n"]),
        size_k=int(row["size_k"]),
    )
    if prepared is None and prepare_error is None and (is_legal or check):
        try:
            prepared = _prepare_dense_runtime(row)
        except Exception as exc:
            prepare_error = exc

    if prepare_error is not None and (is_legal or check):
        out = _status_row(
            kind="dense",
            model=model,
            row=row,
            geometry_label=geometry.label,
            split_k=split_k,
            metadata_cache=metadata_cache,
            env_legal=is_legal,
            status="ERR",
            reason=_first_line(prepare_error),
        )
        _write_row(writer, state, out)
        return prepared, prepare_error

    with dense_env(geometry, split_k, metadata_cache):
        if not is_legal:
            check_pass = ""
            reason = ""
            status = "REJECTED"
            if check:
                assert prepared is not None
                ok, reason = _expect_runtime_rejection(prepared.check)
                check_pass = "yes" if ok else "no"
                status = "REJECTED" if ok else "ERR"
            out = _status_row(
                kind="dense",
                model=model,
                row=row,
                geometry_label=geometry.label,
                split_k=split_k,
                metadata_cache=metadata_cache,
                env_legal=False,
                status=status,
                check_pass=check_pass,
                reason=reason,
            )
            _write_row(writer, state, out)
            return prepared, prepare_error

        assert prepared is not None
        status, marlin_us, check_pass, reason = _run_checked_timed_combo(
            prepared,
            warmup_iters=warmup_iters,
            iters=iters,
            check=check,
        )
        out = _status_row(
            kind="dense",
            model=model,
            row=row,
            geometry_label=geometry.label,
            split_k=split_k,
            metadata_cache=metadata_cache,
            env_legal=True,
            status=status,
            marlin_us=marlin_us,
            check_pass=check_pass,
            reason=reason,
        )
        _write_row(writer, state, out)
        return prepared, prepare_error


def _handle_moe_combo(
    *,
    writer: csv.DictWriter,
    state: RunState,
    model: str,
    row: dict[str, Any],
    geometry: Any,
    split_k: int,
    metadata_cache: str,
    prepared: MoePrepared | None,
    prepare_error: BaseException | None,
    warmup_iters: int,
    iters: int,
    check: bool,
) -> tuple[MoePrepared | None, BaseException | None]:
    is_legal = moe_stage_env_combo_is_legal(
        geometry,
        size_n=int(row["size_n"]),
        size_k=int(row["size_k"]),
    )
    if prepared is None and prepare_error is None and (is_legal or check):
        try:
            prepared = _prepare_moe_runtime(row)
        except Exception as exc:
            prepare_error = exc

    if prepare_error is not None and (is_legal or check):
        out = _status_row(
            kind="moe",
            model=model,
            row=row,
            geometry_label=geometry.label,
            split_k=split_k,
            metadata_cache=metadata_cache,
            env_legal=is_legal,
            status="ERR",
            reason=_first_line(prepare_error),
        )
        _write_row(writer, state, out)
        return prepared, prepare_error

    with moe_env(geometry, split_k, metadata_cache):
        if not is_legal:
            check_pass = ""
            reason = ""
            status = "REJECTED"
            if check:
                assert prepared is not None
                ok, reason = _expect_runtime_rejection(prepared.check)
                check_pass = "yes" if ok else "no"
                status = "REJECTED" if ok else "ERR"
            out = _status_row(
                kind="moe",
                model=model,
                row=row,
                geometry_label=geometry.label,
                split_k=split_k,
                metadata_cache=metadata_cache,
                env_legal=False,
                status=status,
                check_pass=check_pass,
                reason=reason,
            )
            _write_row(writer, state, out)
            return prepared, prepare_error

        assert prepared is not None
        status, marlin_us, check_pass, reason = _run_checked_timed_combo(
            prepared,
            warmup_iters=warmup_iters,
            iters=iters,
            check=check,
        )
        out = _status_row(
            kind="moe",
            model=model,
            row=row,
            geometry_label=geometry.label,
            split_k=split_k,
            metadata_cache=metadata_cache,
            env_legal=True,
            status=status,
            marlin_us=marlin_us,
            check_pass=check_pass,
            reason=reason,
        )
        _write_row(writer, state, out)
        return prepared, prepare_error


def _ensure_runtime(selected_kinds: list[str]) -> None:
    require_matching_cuda_benchmark_runtime()
    if "dense" in selected_kinds:
        ops._load_dense()
    if "moe" in selected_kinds:
        ops._load_moe()


def _selected_combo_kinds(
    *,
    dense_possible: int,
    moe_possible: int,
    start: int,
    limit: int | None,
    max_cases: int | None,
) -> list[str]:
    possible = dense_possible + moe_possible
    selected = _selected_count(possible, start, limit, max_cases)
    if selected <= 0:
        return []
    selected_start = min(start, possible)
    selected_end = min(selected_start + selected, possible)
    out: list[str] = []
    if dense_possible > 0 and selected_start < dense_possible and selected_end > 0:
        out.append("dense")
    moe_start = dense_possible
    if moe_possible > 0 and selected_start < possible and selected_end > moe_start:
        out.append("moe")
    return out


def run_benchmark(args: argparse.Namespace) -> Path:
    _validate_args(args)
    model_dir = args.model.expanduser().resolve()
    payload = load_payload(model_dir)
    model = str(payload.get("model", model_dir))

    all_dense_rows = _unique_actual_dense_rows(payload)
    all_moe_rows = _unique_actual_moe_rows(payload)
    selected_kinds = _selected_kinds(args.kind, all_dense_rows, all_moe_rows)
    dense_rows = list(all_dense_rows)
    moe_rows = list(all_moe_rows)

    supported_moe_rows: list[dict[str, Any]] = []
    unsupported_moe_rows: list[tuple[int, dict[str, Any], str]] = []
    for row_index, row in enumerate(moe_rows, start=1):
        support = _moe_runtime_support(row)
        if support.supported:
            supported_moe_rows.append(row)
        else:
            unsupported_moe_rows.append((row_index, row, support.reason))
    all_supported_moe_rows = list(supported_moe_rows)
    all_unsupported_moe_rows = list(unsupported_moe_rows)

    if "dense" not in selected_kinds:
        dense_rows = []
    if "moe" not in selected_kinds:
        supported_moe_rows = []
        unsupported_moe_rows = []

    dense_combos = tuple(iter_env_combinations())
    moe_combos = tuple(iter_moe_env_combinations())
    start, limit = exhaustive_start_limit()
    dense_possible = len(dense_rows) * len(dense_combos)
    moe_possible = len(supported_moe_rows) * len(moe_combos)
    possible = dense_possible + moe_possible
    selected_combo_count = _selected_count(possible, start, limit, args.max_cases)
    runtime_kinds = _selected_combo_kinds(
        dense_possible=dense_possible,
        moe_possible=moe_possible,
        start=start,
        limit=limit,
        max_cases=args.max_cases,
    )
    csv_path = args.csv or _default_csv_path()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    dense_stats, moe_stats = _table_stats(
        payload,
        all_dense_rows,
        all_moe_rows,
        all_supported_moe_rows,
        all_unsupported_moe_rows,
    )
    detected_kinds = [
        kind
        for kind in ("dense", "moe")
        if (kind == "dense" and dense_stats.unique_actual_rows > 0)
        or (kind == "moe" and moe_stats.unique_actual_rows > 0)
    ]
    _print_startup(
        model=model,
        requested_kind=args.kind,
        detected_kinds=detected_kinds,
        dense_stats=dense_stats,
        moe_stats=moe_stats,
        dense_combo_count=len(dense_combos),
        moe_combo_count=len(moe_combos),
        possible=possible,
        start=start,
        limit=limit,
        selected=selected_combo_count,
        csv_path=csv_path,
        warmup_iters=args.warmup_iters,
        iters=args.iters,
        check=args.check,
    )

    if runtime_kinds:
        _ensure_runtime(runtime_kinds)

    state = RunState()
    global_index = 0

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for row_index, row in enumerate(dense_rows, start=1):
            if _max_cases_reached(state, args.max_cases):
                break
            prepared: DensePrepared | None = None
            prepare_error: BaseException | None = None
            row_logged = False
            for geometry, split_k, metadata_cache in dense_combos:
                if _max_cases_reached(state, args.max_cases):
                    break
                if _past_limit(global_index, start, limit):
                    break
                selected = _index_is_selected(global_index, start, limit)
                global_index += 1
                if not selected:
                    continue
                state.selected_seen += 1
                state.checked += 1
                if not row_logged:
                    print(_dense_row_progress(row, row_index, len(dense_rows)), flush=True)
                    row_logged = True
                prepared, prepare_error = _handle_dense_combo(
                    writer=writer,
                    state=state,
                    model=model,
                    row=row,
                    geometry=geometry,
                    split_k=split_k,
                    metadata_cache=metadata_cache,
                    prepared=prepared,
                    prepare_error=prepare_error,
                    warmup_iters=args.warmup_iters,
                    iters=args.iters,
                    check=args.check,
                )
                _maybe_heartbeat(state)
            if _past_limit(global_index, start, limit) or _max_cases_reached(
                state,
                args.max_cases,
            ):
                break

        if "moe" in selected_kinds:
            for row_index, row, reason in unsupported_moe_rows:
                print(
                    _moe_unsupported_row_progress(row, row_index, len(moe_rows), reason),
                    flush=True,
                )
                out = _status_row(
                    kind="moe",
                    model=model,
                    row=row,
                    status="UNSUPPORTED",
                    reason=reason,
                )
                _write_row(writer, state, out)

        for row_index, row in enumerate(supported_moe_rows, start=1):
            if _max_cases_reached(state, args.max_cases):
                break
            prepared: MoePrepared | None = None
            prepare_error: BaseException | None = None
            row_logged = False
            for geometry, split_k, metadata_cache in moe_combos:
                if _max_cases_reached(state, args.max_cases):
                    break
                if _past_limit(global_index, start, limit):
                    break
                selected = _index_is_selected(global_index, start, limit)
                global_index += 1
                if not selected:
                    continue
                state.selected_seen += 1
                state.checked += 1
                if not row_logged:
                    print(_moe_row_progress(row, row_index, len(supported_moe_rows)), flush=True)
                    row_logged = True
                prepared, prepare_error = _handle_moe_combo(
                    writer=writer,
                    state=state,
                    model=model,
                    row=row,
                    geometry=geometry,
                    split_k=split_k,
                    metadata_cache=metadata_cache,
                    prepared=prepared,
                    prepare_error=prepare_error,
                    warmup_iters=args.warmup_iters,
                    iters=args.iters,
                    check=args.check,
                )
                _maybe_heartbeat(state)
            if _past_limit(global_index, start, limit) or _max_cases_reached(
                state,
                args.max_cases,
            ):
                break

    print()
    print(
        "summary: "
        f"selected_combos={state.selected_seen}, saved_rows={state.saved_rows}, "
        f"OK={state.status_counts['OK']}, "
        f"REJECTED={state.status_counts['REJECTED']}, "
        f"UNSUPPORTED={state.status_counts['UNSUPPORTED']}, "
        f"ERR={state.status_counts['ERR']}, "
        f"MISMATCH={state.status_counts['MISMATCH']}"
    )
    print(f"csv={csv_path}")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
