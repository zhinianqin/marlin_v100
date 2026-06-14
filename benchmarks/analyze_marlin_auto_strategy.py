from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tests.writeback_marlin_cases import (
    DENSE_BENCHMARK_SHAPE_CASES,
    DENSE_WRITEBACK_CLASS_CASE_BY_NAME,
    MOE_BENCHMARK_SHAPE_CASES,
    MOE_WRITEBACK_CLASS_CASE_BY_NAME,
    DenseShapeCase,
    MoeShapeCase,
    MoeWritebackMatrixCase,
    ResolvedCta,
    dense_auto_cta_geometry_label,
    dense_auto_split_k,
    moe_case_auto_cta_geometry_label,
    moe_case_auto_split_k_label,
)


# 3-field CTA label → 7-field CTA label mapping, consistent with
# sm70_marlin_geometry_from_legacy_cta in sm70_marlin_common.cuh.
_LEGACY_CTA_TO_7FIELD_LABEL: dict[str, str] = {
    "32x128x4":  "32x128x32x4x32x32x32",
    "32x256x4":  "32x256x32x4x32x64x32",
    "64x64x4":   "64x64x32x4x32x32x32",
    "64x128x4":  "64x128x32x4x32x64x32",
    "64x128x8":  "64x128x32x8x32x32x32",
    "64x256x4":  "64x256x32x4x64x64x32",
    "64x256x8":  "64x256x32x8x32x64x32",
    "128x64x4":  "128x64x32x4x64x32x32",
    "128x64x8":  "128x64x32x8x32x32x32",
    "128x128x4": "128x128x32x4x64x64x32",
    "128x128x8": "128x128x32x8x64x32x32",
    "128x256x8": "128x256x32x8x64x64x32",
    "256x64x4":  "256x64x32x4x64x64x32",
    "256x64x8":  "256x64x32x8x64x32x32",
    "256x128x8": "256x128x32x8x64x64x32",
}


def normalize_cta_label_to_7field(cta_label: str) -> str:
    """Normalize a 3-field or 7-field CTA label to the canonical 7-field form."""
    if cta_label in ("n/a", "auto"):
        return cta_label
    if cta_label.count("x") == 6:
        return cta_label
    return _LEGACY_CTA_TO_7FIELD_LABEL.get(cta_label, cta_label)


DEFAULT_DENSE_AUTO = Path(
    "benchmarks/results/20260604_dense_auto_ctam_warps_splitk_iters1.csv"
)
DEFAULT_DENSE_FULL = Path(
    "benchmarks/results/20260603_dense_writeback_full_matrix_iters1_no_skip_tflops.csv"
)
DEFAULT_MOE_AUTO = Path(
    "benchmarks/results/20260604_moe_auto_ctam_warps_splitk_iters1.csv"
)
DEFAULT_MOE_FULL = Path(
    "benchmarks/results/20260603_moe_writeback_full_matrix_iters1_no_skip_tflops.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze SM70 Marlin auto CTA/warp/split-K strategy CSVs."
    )
    parser.add_argument("--dense-auto", type=Path, default=DEFAULT_DENSE_AUTO)
    parser.add_argument("--dense-full", type=Path, default=DEFAULT_DENSE_FULL)
    parser.add_argument("--moe-auto", type=Path, default=DEFAULT_MOE_AUTO)
    parser.add_argument("--moe-full", type=Path, default=DEFAULT_MOE_FULL)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--top-bad", type=int, default=20)
    return parser.parse_args()


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def ok_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "OK"]


def latency_us(row: dict[str, str]) -> float:
    try:
        return float(row["marlin_us"])
    except (KeyError, ValueError):
        return math.inf


def auto_field(row: dict[str, str], name: str) -> str:
    if name in row:
        return row[name]
    if name == "auto_cta_geometry":
        return row["resolved_cta"]
    if name == "auto_split_k":
        return row["resolved_split_k"]
    raise KeyError(name)


def split_label(value: str) -> str:
    return "1" if value == "unset" else value


def cta_label(cta: ResolvedCta | None) -> str:
    if cta is None:
        return "n/a"
    return (
        f"{cta.cta_m}x{cta.cta_n}x{cta.cta_k}x{cta.warps}x"
        f"{cta.warp_m}x{cta.warp_n}x{cta.warp_k}"
    )


def historical_dense_auto_cta_label(shape: DenseShapeCase) -> str:
    return dense_auto_cta_geometry_label(shape)


def historical_moe_stage_cta(size_n: int) -> ResolvedCta | None:
    if size_n % 256 == 0:
        return ResolvedCta(32, 256, 32, 4, 32, 64, 32)
    if size_n % 128 == 0:
        return ResolvedCta(32, 128, 32, 4, 32, 32, 32)
    if size_n % 64 == 0:
        return ResolvedCta(64, 64, 32, 4, 32, 32, 32)
    return None


def stage_pair_label(stage1: ResolvedCta | None, stage2: ResolvedCta | None) -> str:
    stage1_label = cta_label(stage1)
    stage2_label = cta_label(stage2)
    if stage1_label == stage2_label:
        return stage1_label
    return f"stage1={stage1_label};stage2={stage2_label}"


def historical_moe_auto_cta_label(shape: MoeShapeCase) -> str:
    return stage_pair_label(
        historical_moe_stage_cta(2 * shape.intermediate),
        historical_moe_stage_cta(shape.hidden),
    )


def dense_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["dense_class"], row["quant"], row["group_size"], row["shape_id"])


def moe_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["method_class"],
        row["quant"],
        row["group_size"],
        row["shape_id"],
        row["routing_profile"],
    )


def dense_full_strategy(
    row: dict[str, str], shapes: dict[str, DenseShapeCase]
) -> tuple[str, str]:
    shape = shapes[row["shape_id"]]
    cta = (
        historical_dense_auto_cta_label(shape)
        if row["cta"] == "auto"
        else normalize_cta_label_to_7field(row["cta"])
    )
    return (cta, split_label(row["split_k"]))


def moe_full_strategy(
    row: dict[str, str], shapes: dict[str, MoeShapeCase]
) -> tuple[str, str]:
    shape = shapes[row["shape_id"]]
    cta = (
        historical_moe_auto_cta_label(shape)
        if row["cta"] == "auto"
        else normalize_cta_label_to_7field(row["cta"])
    )
    return (cta, split_label(row["split_k"]))


def dense_observed_auto_strategy(row: dict[str, str]) -> tuple[str, str]:
    return (
        auto_field(row, "auto_cta_geometry"),
        auto_field(row, "auto_split_k"),
    )


def moe_observed_auto_strategy(row: dict[str, str]) -> tuple[str, str]:
    return (
        auto_field(row, "auto_cta_geometry"),
        auto_field(row, "auto_split_k"),
    )


def dense_current_policy_strategy(
    row: dict[str, str], shapes: dict[str, DenseShapeCase]
) -> tuple[str, str]:
    shape = shapes[row["shape_id"]]
    return (dense_auto_cta_geometry_label(shape), str(dense_auto_split_k(shape)))


def moe_current_policy_strategy(
    row: dict[str, str], shapes: dict[str, MoeShapeCase]
) -> tuple[str, str]:
    class_case = MOE_WRITEBACK_CLASS_CASE_BY_NAME[row["method_class"]]
    case = MoeWritebackMatrixCase(
        class_case=class_case,
        quant_name=row["quant"],
        group_size=int(row["group_size"]),
        shape=shapes[row["shape_id"]],
        supported=True,
        reason="",
    )
    return (
        moe_case_auto_cta_geometry_label(case),
        moe_case_auto_split_k_label(case),
    )


def build_full_map(
    rows: list[dict[str, str]],
    key_fn,
    strategy_fn,
) -> dict[tuple[str, ...], dict[tuple[str, str], float]]:
    strategies: dict[tuple[str, ...], dict[tuple[str, str], float]] = defaultdict(dict)
    for row in ok_rows(rows):
        key = key_fn(row)
        strategy = strategy_fn(row)
        current = strategies[key].get(strategy, math.inf)
        strategies[key][strategy] = min(current, latency_us(row))
    return strategies


def percentile(sorted_values: list[float], p: float) -> float | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, math.ceil(p * len(sorted_values)) - 1))
    return sorted_values[index]


def ratio_summary(ratios: list[float]) -> dict[str, float | int | None]:
    ratios = sorted(ratios)
    return {
        "count": len(ratios),
        "median": percentile(ratios, 0.50),
        "p90": percentile(ratios, 0.90),
        "p95": percentile(ratios, 0.95),
        "p99": percentile(ratios, 0.99),
        "max": ratios[-1] if ratios else None,
        "gt_1_10": sum(ratio > 1.10 for ratio in ratios),
    }


def rank_policy(
    *,
    rows: list[dict[str, str]],
    key_fn,
    policy_fn,
    full_map: dict[tuple[str, ...], dict[tuple[str, str], float]],
    shape_lookup: dict[str, Any],
    top_bad: int,
    observed_latency: bool,
) -> dict[str, Any]:
    ranks: Counter[int] = Counter()
    ratios: list[float] = []
    missing: list[dict[str, Any]] = []
    bad: list[dict[str, Any]] = []
    cross_run_ratios: list[float] = []

    for row in ok_rows(rows):
        key = key_fn(row)
        strategies = full_map[key]
        best_strategy, best_us = min(strategies.items(), key=lambda item: item[1])
        policy_strategy = policy_fn(row)

        if observed_latency:
            cross_run_ratios.append(latency_us(row) / best_us)

        if policy_strategy not in strategies:
            missing.append(
                {
                    "key": list(key),
                    "policy_strategy": list(policy_strategy),
                    "best_strategy": list(best_strategy),
                    "best_us": best_us,
                    "observed_us": latency_us(row) if observed_latency else None,
                    "observed_over_best": (
                        latency_us(row) / best_us if observed_latency else None
                    ),
                }
            )
            continue

        policy_us = strategies[policy_strategy]
        rank = 1 + sum(
            strategy_us < policy_us - 1e-9
            for strategy_us in set(strategies.values())
        )
        ranks[rank] += 1
        ratio = policy_us / best_us
        ratios.append(ratio)
        if rank > 2:
            shape_id = row["shape_id"]
            shape = shape_lookup[shape_id]
            shape_dims = {
                key: getattr(shape, key)
                for key in (
                    "size_m",
                    "size_k",
                    "size_n",
                    "tokens",
                    "hidden",
                    "intermediate",
                    "experts",
                    "topk",
                )
                if hasattr(shape, key)
            }
            bad.append(
                {
                    "ratio": ratio,
                    "rank": rank,
                    "key": list(key),
                    "shape_dims": shape_dims,
                    "policy_strategy": list(policy_strategy),
                    "best_strategy": list(best_strategy),
                    "policy_us": policy_us,
                    "best_us": best_us,
                }
            )

    exact_count = sum(ranks.values())
    top1 = ranks[1]
    top2 = sum(count for rank, count in ranks.items() if rank <= 2)
    top4 = sum(count for rank, count in ranks.items() if rank <= 4)

    def aggregate_bad(field_index: int) -> dict[str, int]:
        counter = Counter(item["key"][field_index] for item in bad)
        return dict(counter.most_common(20))

    return {
        "rows": len(ok_rows(rows)),
        "exact_comparable": exact_count,
        "mixed_or_missing_strategy": len(missing),
        "top1": top1,
        "top2": top2,
        "top4": top4,
        "top2_rate": top2 / exact_count if exact_count else None,
        "rank_counts": dict(sorted((str(rank), count) for rank, count in ranks.items())),
        "policy_over_best": ratio_summary(ratios),
        "observed_over_full_best": ratio_summary(cross_run_ratios)
        if observed_latency
        else None,
        "bad_by_shape": aggregate_bad(3),
        "bad_by_quant": aggregate_bad(1),
        "bad_by_group": aggregate_bad(2),
        "top_bad": sorted(bad, key=lambda item: item["ratio"], reverse=True)[
            :top_bad
        ],
        "top_mixed": sorted(
            missing,
            key=lambda item: item["observed_over_best"] or 0.0,
            reverse=True,
        )[:top_bad],
    }


def csv_status(path: Path, rows: list[dict[str, str]], fields: list[str]) -> dict[str, Any]:
    return {
        "path": str(path),
        "rows": len(rows),
        "fields": fields,
        "status": dict(sorted(Counter(row.get("status", "") for row in rows).items())),
    }


def make_report(args: argparse.Namespace) -> dict[str, Any]:
    dense_shapes = {shape.name: shape for shape in DENSE_BENCHMARK_SHAPE_CASES}
    moe_shapes = {shape.name: shape for shape in MOE_BENCHMARK_SHAPE_CASES}

    dense_auto_rows, dense_auto_fields = read_rows(args.dense_auto)
    dense_full_rows, dense_full_fields = read_rows(args.dense_full)
    moe_auto_rows, moe_auto_fields = read_rows(args.moe_auto)
    moe_full_rows, moe_full_fields = read_rows(args.moe_full)

    dense_full_map = build_full_map(
        dense_full_rows,
        dense_key,
        lambda row: dense_full_strategy(row, dense_shapes),
    )
    moe_full_map = build_full_map(
        moe_full_rows,
        moe_key,
        lambda row: moe_full_strategy(row, moe_shapes),
    )

    return {
        "normalization": {
            "historical_full_matrix_commit": "abeccd8fede450dc3e818998abf276040ac4ef31",
            "split_k": "unset is normalized to 1",
            "dense_cta_auto": (
                "20260603 cta=auto is mapped through the historical dense "
                "CTA_M/CTA_N/CTA_K/warps/WarpM/WarpN/WarpK auto geometry."
            ),
            "moe_cta_auto": (
                "20260603 cta=auto is mapped through the historical MoE "
                "default stage geometry: "
                "CTA_N=64 -> 64x64x32x4x32x32x32, "
                "CTA_N=128 -> 32x128x32x4x32x32x32, "
                "CTA_N=256 -> 32x256x32x4x32x64x32."
            ),
            "rank": (
                "Rank is computed within the normalized 20260603 full matrix "
                "strategy set for the same class/quant/group/shape key."
            ),
        },
        "inputs": {
            "dense_auto": csv_status(args.dense_auto, dense_auto_rows, dense_auto_fields),
            "dense_full": csv_status(args.dense_full, dense_full_rows, dense_full_fields),
            "moe_auto": csv_status(args.moe_auto, moe_auto_rows, moe_auto_fields),
            "moe_full": csv_status(args.moe_full, moe_full_rows, moe_full_fields),
        },
        "dense": {
            "observed_auto_csv": rank_policy(
                rows=dense_auto_rows,
                key_fn=dense_key,
                policy_fn=dense_observed_auto_strategy,
                full_map=dense_full_map,
                shape_lookup=dense_shapes,
                top_bad=args.top_bad,
                observed_latency=True,
            ),
            "current_code_policy": rank_policy(
                rows=dense_auto_rows,
                key_fn=dense_key,
                policy_fn=lambda row: dense_current_policy_strategy(row, dense_shapes),
                full_map=dense_full_map,
                shape_lookup=dense_shapes,
                top_bad=args.top_bad,
                observed_latency=False,
            ),
        },
        "moe": {
            "observed_auto_csv": rank_policy(
                rows=moe_auto_rows,
                key_fn=moe_key,
                policy_fn=moe_observed_auto_strategy,
                full_map=moe_full_map,
                shape_lookup=moe_shapes,
                top_bad=args.top_bad,
                observed_latency=True,
            ),
            "current_code_policy": rank_policy(
                rows=moe_auto_rows,
                key_fn=moe_key,
                policy_fn=lambda row: moe_current_policy_strategy(row, moe_shapes),
                full_map=moe_full_map,
                shape_lookup=moe_shapes,
                top_bad=args.top_bad,
                observed_latency=False,
            ),
        },
    }


def markdown_summary(report: dict[str, Any]) -> str:
    dense_observed = report["dense"]["observed_auto_csv"]
    dense_policy = report["dense"]["current_code_policy"]
    moe_observed = report["moe"]["observed_auto_csv"]
    moe_policy = report["moe"]["current_code_policy"]

    lines = [
        "# SM70 Marlin Auto Strategy CSV Analysis",
        "",
        "## Normalization",
        "",
        f"- Historical full-matrix commit: `{report['normalization']['historical_full_matrix_commit']}`.",
        f"- {report['normalization']['split_k']}.",
        f"- {report['normalization']['dense_cta_auto']}",
        f"- {report['normalization']['moe_cta_auto']}",
        f"- {report['normalization']['rank']}",
        "",
        "## CSV Inputs",
        "",
    ]
    for name, info in report["inputs"].items():
        lines.append(f"- `{name}`: `{info['path']}`, rows={info['rows']}, status={info['status']}")

    def add_policy(section: str, data: dict[str, Any]) -> None:
        ratio = data["policy_over_best"]
        lines.extend(
            [
                "",
                f"## {section}",
                "",
                f"- rows: {data['rows']}",
                f"- exact comparable: {data['exact_comparable']}",
                f"- mixed or missing strategy: {data['mixed_or_missing_strategy']}",
                f"- top1/top2/top4: {data['top1']} / {data['top2']} / {data['top4']}",
                f"- top2 rate: {data['top2_rate']}",
                f"- policy/best ratio: median={ratio['median']}, p95={ratio['p95']}, p99={ratio['p99']}, max={ratio['max']}, >1.10={ratio['gt_1_10']}",
                f"- rank counts: {data['rank_counts']}",
                f"- bad by quant: {data['bad_by_quant']}",
                f"- bad by group: {data['bad_by_group']}",
                f"- bad by shape: {data['bad_by_shape']}",
            ]
        )
        observed = data.get("observed_over_full_best")
        if observed is not None:
            lines.append(
                "- observed auto CSV latency/full best ratio: "
                f"median={observed['median']}, p95={observed['p95']}, "
                f"p99={observed['p99']}, max={observed['max']}, "
                f">1.10={observed['gt_1_10']}"
            )
        if data["top_bad"]:
            lines.extend(["", "Top non-top2 exact-comparable cases:"])
            for item in data["top_bad"][:10]:
                lines.append(
                    f"- ratio={item['ratio']:.6g}, rank={item['rank']}, "
                    f"key={item['key']}, policy={item['policy_strategy']}, "
                    f"best={item['best_strategy']}"
                )
        if data["top_mixed"]:
            lines.extend(["", "Top mixed/missing strategy cases:"])
            for item in data["top_mixed"][:10]:
                lines.append(
                    f"- key={item['key']}, policy={item['policy_strategy']}, "
                    f"best={item['best_strategy']}, "
                    f"observed/best={item['observed_over_best']}"
                )

    add_policy("Dense Observed Auto CSV", dense_observed)
    add_policy("Dense Current Code Policy", dense_policy)
    add_policy("MoE Observed Auto CSV", moe_observed)
    add_policy("MoE Current Code Policy", moe_policy)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = make_report(args)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_summary(report), encoding="utf-8")

    print(markdown_summary(report))


if __name__ == "__main__":
    main()
