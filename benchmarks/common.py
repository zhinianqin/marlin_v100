from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Callable, Iterable

import torch

from tests.calibration import (
    format_capability,
    runtime_capability,
    source_target_capability,
    source_target_label,
)


ROOT = Path(__file__).resolve().parent.parent


def require_matching_cuda_benchmark_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Marlin benchmarks.")
    capability = runtime_capability()
    target = source_target_capability()
    if capability != target:
        raise RuntimeError(
            "Current Marlin benchmark scripts require a runtime GPU that matches the "
            f"checked-in {source_target_label()} build. Found capability="
            f"{format_capability(capability)}."
        )


def time_cuda_callable(
    fn: Callable[[], object],
    warmup_iters: int,
    iters: int,
) -> dict[str, float]:
    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    latencies_us: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        latencies_us.append(start.elapsed_time(end) * 1000.0)

    return {
        "median_us": statistics.median(latencies_us),
        "mean_us": statistics.fmean(latencies_us),
        "min_us": min(latencies_us),
        "max_us": max(latencies_us),
    }


def format_float(value: float) -> str:
    return f"{value:.2f}"


def print_table(headers: list[str], rows: Iterable[list[str]]) -> None:
    rows = list(rows)
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def render(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def banner(title: str) -> None:
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
