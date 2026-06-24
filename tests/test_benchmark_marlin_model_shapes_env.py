from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import benchmarks.benchmark_marlin_model_shapes_env as bench


@dataclass(frozen=True)
class FakeGeometry:
    label: str
    cta_m: int
    cta_n: int
    cta_k: int


LEGAL_DENSE = FakeGeometry("dense_legal", 32, 64, 32)
ILLEGAL_DENSE = FakeGeometry("dense_illegal", 32, 96, 32)
LEGAL_MOE = FakeGeometry("moe_legal", 32, 64, 32)
ILLEGAL_MOE = FakeGeometry("moe_illegal", 128, 64, 32)


def _model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "fake"}), encoding="utf-8")
    return model_dir


def _dense_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": "tp4",
        "phase": "prefill",
        "layer_key": "model.layers.0.mlp.gate_up_proj",
        "op": "gate_up_proj",
        "target_op": "ops.marlin_gemm",
        "size_m": 32,
        "size_n": 128,
        "size_k": 256,
        "group_size": 128,
        "quant_method": "awq",
        "quant_format": "uint4",
        "has_zp": True,
        "has_bias": False,
        "marlin_path": "awq_marlin_wna16",
        "call_status": "actual_marlin",
        "call_count": 3,
    }
    row.update(overrides)
    return row


def _moe_row(**overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": "tp4+ep",
        "phase": "decode",
        "layer_key": "model.layers.0.mlp.experts",
        "op": "w13",
        "target_op": "ops.moe_wna16_marlin_gemm",
        "size_m": 32,
        "size_n": 256,
        "size_k": 256,
        "group_size": 128,
        "quant_method": "awq",
        "quant_format": "uint4",
        "has_zp": True,
        "has_bias": False,
        "marlin_path": "awq_marlin_moe_wna16",
        "call_status": "actual_marlin",
        "call_count": 4,
        "moe_block_size": 16,
        "top_k": 2,
        "local_num_experts": 8,
        "global_num_experts": 16,
        "intermediate_size_per_partition": 128,
    }
    row.update(overrides)
    return row


def _payload(
    model_dir: Path,
    *,
    dense: list[dict[str, Any]] | None = None,
    moe: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "model": str(model_dir),
        "model_config": {},
        "quantization": {},
        "shape_inputs": {},
        "scenarios": [],
        "dense": dense or [],
        "moe": moe or [],
        "warnings": [],
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _patch_common(monkeypatch, payload: dict[str, Any]) -> list[str]:
    runtime_kinds: list[str] = []
    monkeypatch.setattr(bench, "load_payload", lambda _model_dir: payload)
    monkeypatch.setattr(bench, "_ensure_runtime", lambda kinds: runtime_kinds.extend(kinds))
    monkeypatch.setattr(
        bench,
        "iter_env_combinations",
        lambda: iter(((LEGAL_DENSE, 1, "vector_words"), (ILLEGAL_DENSE, 1, "vector_words"))),
    )
    monkeypatch.setattr(
        bench,
        "iter_moe_env_combinations",
        lambda: iter(((LEGAL_MOE, 1, "lane_vectors"), (ILLEGAL_MOE, 1, "lane_vectors"))),
    )
    monkeypatch.setattr(
        bench,
        "time_cuda_callable",
        lambda fn, warmup_iters, iters: (fn(), {"median_us": 10.0})[1],
    )

    def make_dense(_row: dict[str, Any]) -> bench.DensePrepared:
        return bench.DensePrepared(key=None, check=lambda: None, run=lambda: None)

    def make_moe(_row: dict[str, Any]) -> bench.MoePrepared:
        return bench.MoePrepared(key=None, check=lambda: None, run=lambda: None)

    monkeypatch.setattr(bench, "_prepare_dense_runtime", make_dense)
    monkeypatch.setattr(bench, "_prepare_moe_runtime", make_moe)
    return runtime_kinds


def _run(
    tmp_path: Path,
    monkeypatch,
    payload: dict[str, Any],
    *extra_args: str,
) -> tuple[list[dict[str, str]], str, list[str]]:
    csv_path = tmp_path / "out.csv"
    runtime_kinds = _patch_common(monkeypatch, payload)
    args = bench.parse_args(
        [
            "--model",
            str(Path(payload["model"])),
            "--csv",
            str(csv_path),
            "--warmup-iters",
            "0",
            "--iters",
            "1",
            *extra_args,
        ]
    )
    bench.run_benchmark(args)
    return _read_csv(csv_path), csv_path.read_text(encoding="utf-8"), runtime_kinds


def test_auto_dense_and_moe_runs_both_and_dedups(tmp_path: Path, monkeypatch, capsys) -> None:
    model_dir = _model_dir(tmp_path)
    dense = _dense_row()
    duplicate_dense = _dense_row(layer_key="model.layers.1.mlp.gate_up_proj")
    skipped_dense = _dense_row(call_status="skipped", marlin_path="none")
    moe = _moe_row()
    payload = _payload(model_dir, dense=[dense, duplicate_dense, skipped_dense], moe=[moe])

    rows, _csv_text, runtime_kinds = _run(tmp_path, monkeypatch, payload)

    assert runtime_kinds == ["dense", "moe"]
    assert [row["kind"] for row in rows] == ["dense", "dense", "moe", "moe"]
    assert [row["status"] for row in rows] == ["OK", "REJECTED", "OK", "REJECTED"]
    assert rows[0]["env_cta_geometry"] == "dense_legal"
    assert rows[2]["env_cta_geometry"] == "moe_legal"
    assert rows[0]["marlin_us"] == "10.00"
    assert rows[0]["marlin_tflops"] == "0.209715"
    assert rows[2]["moe_block_size"] == "16"
    assert rows[2]["global_num_experts"] == "16"

    out = capsys.readouterr().out
    assert "detected_kinds=dense,moe" in out
    assert "dense_actual_rows=2" in out
    assert "dense_unique_actual_rows=1" in out


def test_has_bias_is_written_to_csv(tmp_path: Path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path)
    payload = _payload(model_dir, dense=[_dense_row(has_bias=True)], moe=[])

    rows, csv_text, _runtime_kinds = _run(
        tmp_path,
        monkeypatch,
        payload,
        "--kind",
        "dense",
    )

    assert "has_bias" in csv_text.splitlines()[0].split(",")
    assert {row["has_bias"] for row in rows} == {"true"}


def test_has_bias_participates_in_unique_row_dedup(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    model_dir = _model_dir(tmp_path)
    payload = _payload(
        model_dir,
        dense=[
            _dense_row(layer_key="model.layers.0.mlp.gate_up_proj"),
            _dense_row(layer_key="model.layers.1.mlp.gate_up_proj", has_bias=True),
        ],
        moe=[
            _moe_row(layer_key="model.layers.0.mlp.experts"),
            _moe_row(layer_key="model.layers.1.mlp.experts", has_bias=True),
        ],
    )

    rows, _csv_text, _runtime_kinds = _run(tmp_path, monkeypatch, payload)

    assert [row["kind"] for row in rows].count("dense") == 4
    assert [row["kind"] for row in rows].count("moe") == 4
    out = capsys.readouterr().out
    assert "dense_unique_actual_rows=2" in out
    assert "moe_unique_actual_rows=2" in out


def test_has_bias_true_rows_reach_prepare_path(tmp_path: Path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path)
    payload = _payload(
        model_dir,
        dense=[_dense_row(has_bias=True)],
        moe=[_moe_row(has_bias=True)],
    )
    captured_dense: list[dict[str, Any]] = []
    captured_moe: list[dict[str, Any]] = []
    _patch_common(monkeypatch, payload)
    monkeypatch.setattr(
        bench,
        "_prepare_dense_runtime",
        lambda row: (
            captured_dense.append(row),
            bench.DensePrepared(key=None, check=lambda: None, run=lambda: None),
        )[1],
    )
    monkeypatch.setattr(
        bench,
        "_prepare_moe_runtime",
        lambda row: (
            captured_moe.append(row),
            bench.MoePrepared(key=None, check=lambda: None, run=lambda: None),
        )[1],
    )
    csv_path = tmp_path / "prepare.csv"
    args = bench.parse_args(
        [
            "--model",
            str(model_dir),
            "--csv",
            str(csv_path),
            "--warmup-iters",
            "0",
            "--iters",
            "1",
            "--kind",
            "both",
            "--max-cases",
            "3",
        ]
    )

    bench.run_benchmark(args)

    assert captured_dense and captured_dense[0]["has_bias"] is True
    assert captured_moe and captured_moe[0]["has_bias"] is True


def test_kind_auto_handles_dense_only_and_empty_payload(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    model_dir = _model_dir(tmp_path)
    dense_rows, _csv_text, runtime_kinds = _run(
        tmp_path,
        monkeypatch,
        _payload(model_dir, dense=[_dense_row()], moe=[]),
    )
    assert runtime_kinds == ["dense"]
    assert [row["kind"] for row in dense_rows] == ["dense", "dense"]

    empty_dir = _model_dir(tmp_path / "empty")
    empty_csv = tmp_path / "empty.csv"
    runtime_kinds = _patch_common(
        monkeypatch,
        _payload(empty_dir, dense=[_dense_row(call_status="skipped", marlin_path="none")], moe=[]),
    )
    args = bench.parse_args(
        [
            "--model",
            str(empty_dir),
            "--csv",
            str(empty_csv),
            "--warmup-iters",
            "0",
            "--iters",
            "1",
        ]
    )
    bench.run_benchmark(args)
    assert runtime_kinds == []
    assert _read_csv(empty_csv) == []
    assert empty_csv.read_text(encoding="utf-8").startswith("kind,model,scenario")
    assert "selected=0" in capsys.readouterr().out


def test_explicit_kind_filters_tables(tmp_path: Path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path)
    payload = _payload(model_dir, dense=[_dense_row()], moe=[_moe_row()])

    dense_rows, _csv_text, dense_runtime = _run(
        tmp_path,
        monkeypatch,
        payload,
        "--kind",
        "dense",
    )
    assert {row["kind"] for row in dense_rows} == {"dense"}
    assert dense_runtime == ["dense"]

    moe_rows, _csv_text, moe_runtime = _run(
        tmp_path,
        monkeypatch,
        payload,
        "--kind",
        "moe",
    )
    assert {row["kind"] for row in moe_rows} == {"moe"}
    assert moe_runtime == ["moe"]


def test_global_sharding_uses_dense_then_moe_order(tmp_path: Path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path)
    payload = _payload(model_dir, dense=[_dense_row()], moe=[_moe_row()])
    monkeypatch.setenv("MARLIN_EXHAUSTIVE_ENV_START", "1")
    monkeypatch.setenv("MARLIN_EXHAUSTIVE_ENV_LIMIT", "2")

    rows, _csv_text, runtime_kinds = _run(tmp_path, monkeypatch, payload)

    assert runtime_kinds == ["dense", "moe"]
    assert [(row["kind"], row["env_cta_geometry"]) for row in rows] == [
        ("dense", "dense_illegal"),
        ("moe", "moe_legal"),
    ]


def test_max_cases_limits_selected_expanded_combos(tmp_path: Path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path)
    payload = _payload(model_dir, dense=[_dense_row()], moe=[_moe_row()])

    rows, _csv_text, runtime_kinds = _run(
        tmp_path,
        monkeypatch,
        payload,
        "--max-cases",
        "1",
    )

    assert runtime_kinds == ["dense"]
    assert [(row["kind"], row["env_cta_geometry"]) for row in rows] == [
        ("dense", "dense_legal")
    ]


def test_unsupported_moe_row_is_written_without_env_index(tmp_path: Path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path)
    unsupported = _moe_row(group_size=7)
    supported = _moe_row(layer_key="model.layers.1.mlp.experts")
    payload = _payload(model_dir, dense=[], moe=[unsupported, supported])
    monkeypatch.setenv("MARLIN_EXHAUSTIVE_ENV_LIMIT", "1")

    rows, _csv_text, runtime_kinds = _run(tmp_path, monkeypatch, payload)

    assert runtime_kinds == ["moe"]
    assert [row["status"] for row in rows] == ["UNSUPPORTED", "OK"]
    assert rows[0]["env_cta_geometry"] == ""
    assert rows[1]["env_cta_geometry"] == "moe_legal"


def test_check_expected_rejection_failure_writes_err(tmp_path: Path, monkeypatch) -> None:
    model_dir = _model_dir(tmp_path)
    payload = _payload(model_dir, dense=[_dense_row()], moe=[])
    _patch_common(monkeypatch, payload)
    monkeypatch.setattr(
        bench,
        "iter_env_combinations",
        lambda: iter(((ILLEGAL_DENSE, 1, "vector_words"),)),
    )
    monkeypatch.setattr(
        bench,
        "_prepare_dense_runtime",
        lambda _row: bench.DensePrepared(key=None, check=lambda: None, run=lambda: None),
    )
    csv_path = tmp_path / "check.csv"
    args = bench.parse_args(
        [
            "--model",
            str(model_dir),
            "--csv",
            str(csv_path),
            "--warmup-iters",
            "0",
            "--iters",
            "1",
            "--check",
        ]
    )

    bench.run_benchmark(args)

    rows = _read_csv(csv_path)
    assert len(rows) == 1
    assert rows[0]["status"] == "ERR"
    assert rows[0]["check_pass"] == "no"
    assert "did not raise expected RuntimeError" in rows[0]["reason"]
