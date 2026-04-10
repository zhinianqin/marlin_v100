from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from benchmark_shapes import DENSE_PRESETS, DENSE_WEIGHT_SHAPES
from common import (
    banner,
    check_cuda_ready,
    format_float,
    print_table,
    time_cuda_callable,
    timestamp,
)
from marlin_v100 import ops
from tests.helpers import (
    marlin_make_empty_g_idx,
    marlin_make_workspace_new,
    marlin_quantize,
    scalar_types,
)


QUANT_TYPES = {
    "uint4b8": scalar_types.uint4b8,
    "uint8b128": scalar_types.uint8b128,
}
LAUNCH_DOMINATED_FLOPS = 1_000_000_000


def parse_bool_arg(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def dense_flops(size_m: int, size_k: int, size_n: int) -> int:
    return 2 * size_m * size_k * size_n


def format_flops(value: int) -> str:
    for unit, scale in (("T", 10**12), ("G", 10**9), ("M", 10**6), ("K", 10**3)):
        if value >= scale:
            return f"{value / scale:.2f}{unit}"
    return str(value)


def tflops_from_us(flops: int, latency_us: float) -> float:
    if latency_us <= 0.0:
        return 0.0
    return flops / (latency_us * 1_000_000.0)


def is_launch_dominated(flops: int) -> bool:
    return flops < LAUNCH_DOMINATED_FLOPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Marlin dense kernels.")
    parser.add_argument(
        "--preset",
        choices=sorted(DENSE_PRESETS.keys()),
        default="full",
        help="Benchmark preset to run.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(DENSE_WEIGHT_SHAPES.keys()),
        help="Dense shape presets to run. Defaults to the selected preset.",
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        help="Batch sizes (M dimension). Defaults to the selected preset.",
    )
    parser.add_argument(
        "--quant-types",
        nargs="+",
        choices=sorted(QUANT_TYPES.keys()),
        default=list(QUANT_TYPES.keys()),
        help="Quantized weight types to benchmark.",
    )
    parser.add_argument(
        "--group-sizes",
        nargs="+",
        type=int,
        default=[128, -1],
        help="Group sizes to benchmark.",
    )
    parser.add_argument(
        "--reuse-output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also report kernel_like_us using preallocated output buffers.",
    )
    parser.add_argument(
        "--use-fp32-reduce",
        type=parse_bool_arg,
        default=True,
        metavar="{true,false}",
        help="Whether to enable Marlin fp32 reduction.",
    )
    parser.add_argument(
        "--report-tflops",
        action="store_true",
        help="Report derived TFLOPs alongside latency.",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def run_case(
    model: str,
    quant_name: str,
    group_size: int,
    size_m: int,
    size_k: int,
    size_n: int,
    reuse_output: bool,
    use_fp32_reduce: bool,
    warmup_iters: int,
    iters: int,
) -> dict[str, object] | None:
    if group_size != -1 and size_k % group_size != 0:
        return None

    quant_type = QUANT_TYPES[quant_name]
    device = torch.device("cuda")
    a = torch.randn((size_m, size_k), device=device, dtype=torch.float16)
    weight = torch.randn((size_k, size_n), device=device, dtype=torch.float16)
    weight_ref, q_weight, scales, g_idx, sort_indices, _ = marlin_quantize(
        weight, quant_type, group_size, False
    )
    workspace = marlin_make_workspace_new(device)
    empty_g_idx = marlin_make_empty_g_idx(device)
    torch_output = torch.empty((size_m, size_n), device=device, dtype=torch.float16)
    marlin_output = torch.empty((size_m, size_n), device=device, dtype=torch.float16)
    g_idx_or_empty = empty_g_idx if g_idx.numel() == 0 else g_idx
    flops = dense_flops(size_m, size_k, size_n)

    def run_torch_operator() -> torch.Tensor:
        return torch.matmul(a, weight_ref)

    def run_marlin_operator() -> torch.Tensor:
        return ops.marlin_gemm(
            a,
            None,
            q_weight,
            None,
            scales,
            None,
            None,
            None,
            g_idx_or_empty,
            sort_indices,
            workspace,
            quant_type.id,
            size_m,
            size_n,
            size_k,
            True,
            False,
            use_fp32_reduce,
            False,
        )

    results: dict[str, dict[str, float]] = {}
    torch_stats = time_cuda_callable(
        run_torch_operator, warmup_iters=warmup_iters, iters=iters
    )
    marlin_stats = time_cuda_callable(
        run_marlin_operator, warmup_iters=warmup_iters, iters=iters
    )
    results["operator_us"] = {
        "torch_us": torch_stats["median_us"],
        "marlin_us": marlin_stats["median_us"],
        "speedup": torch_stats["median_us"] / marlin_stats["median_us"],
    }

    if reuse_output:
        def run_torch_kernel_like() -> torch.Tensor:
            return torch.mm(a, weight_ref, out=torch_output)

        def run_marlin_kernel_like() -> torch.Tensor:
            return ops.marlin_gemm(
                a,
                marlin_output,
                q_weight,
                None,
                scales,
                None,
                None,
                None,
                g_idx_or_empty,
                sort_indices,
                workspace,
                quant_type.id,
                size_m,
                size_n,
                size_k,
                True,
                False,
                use_fp32_reduce,
                False,
            )

        torch_stats = time_cuda_callable(
            run_torch_kernel_like, warmup_iters=warmup_iters, iters=iters
        )
        marlin_stats = time_cuda_callable(
            run_marlin_kernel_like, warmup_iters=warmup_iters, iters=iters
        )
        results["kernel_like_us"] = {
            "torch_us": torch_stats["median_us"],
            "marlin_us": marlin_stats["median_us"],
            "speedup": torch_stats["median_us"] / marlin_stats["median_us"],
        }

    return {
        "model": model,
        "quant": quant_name,
        "group_size": group_size,
        "mkn": f"{size_m}x{size_k}x{size_n}",
        "flops": flops,
        "launch_dominated": is_launch_dominated(flops),
        "results": results,
    }


def build_rows(
    rows: list[dict[str, object]],
    metric_name: str,
    report_tflops: bool,
) -> list[list[str]]:
    rendered: list[list[str]] = []
    for row in rows:
        metrics = row["results"][metric_name]
        table_row = [
            str(row["model"]),
            str(row["quant"]),
            str(row["mkn"]),
            format_flops(int(row["flops"])),
            "yes" if bool(row["launch_dominated"]) else "no",
            format_float(metrics["torch_us"]),
            format_float(metrics["marlin_us"]),
            f"{metrics['speedup']:.2f}x",
        ]
        if report_tflops:
            flops = int(row["flops"])
            table_row.extend(
                [
                    format_float(tflops_from_us(flops, metrics["torch_us"])),
                    format_float(tflops_from_us(flops, metrics["marlin_us"])),
                ]
            )
        rendered.append(table_row)
    return rendered


def render_metric_table(
    rows: list[dict[str, object]],
    metric_name: str,
    report_tflops: bool,
) -> None:
    headers = [
        "model",
        "quant",
        "MKN",
        "flops",
        "launch_dominated",
        "torch_us",
        "marlin_us",
        "speedup",
    ]
    if report_tflops:
        headers.extend(["torch_tflops", "marlin_tflops"])
    print(metric_name)
    print_table(headers=headers, rows=build_rows(rows, metric_name, report_tflops))


def main() -> None:
    args = parse_args()
    check_cuda_ready()
    ops._load_dense()

    preset = DENSE_PRESETS[args.preset]
    models = args.models or list(preset["models"])
    batch_sizes = args.batch_sizes or list(preset["batch_sizes"])

    banner(f"Marlin Dense Benchmark ({timestamp()})")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"capability={torch.cuda.get_device_capability(0)}")
    print("note=current results reflect the checked-in SM70-only build on this machine.")
    print("note=do not treat these numbers as final SM75-target Marlin performance.")
    print(f"preset={args.preset}")
    print(f"models={models}")
    print(f"batch_sizes={batch_sizes}")
    print(f"quant_types={args.quant_types}")
    print(f"group_sizes={args.group_sizes}")
    print(f"reuse_output={args.reuse_output}")
    print(f"use_fp32_reduce={args.use_fp32_reduce}")
    print(f"report_tflops={args.report_tflops}")
    print(f"launch_dominated_flops<{LAUNCH_DOMINATED_FLOPS}")
    print(f"warmup_iters={args.warmup_iters}, iters={args.iters}")

    rows: list[dict[str, object]] = []
    for model in models:
        for size_k, size_n in DENSE_WEIGHT_SHAPES[model]:
            for quant_name in args.quant_types:
                for group_size in args.group_sizes:
                    for size_m in batch_sizes:
                        row = run_case(
                            model=model,
                            quant_name=quant_name,
                            group_size=group_size,
                            size_m=size_m,
                            size_k=size_k,
                            size_n=size_n,
                            reuse_output=args.reuse_output,
                            use_fp32_reduce=args.use_fp32_reduce,
                            warmup_iters=args.warmup_iters,
                            iters=args.iters,
                        )
                        if row is not None:
                            rows.append(row)

    grouped_rows: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped_rows[int(row["group_size"])].append(row)

    for group_size in args.group_sizes:
        group_rows = grouped_rows.get(group_size, [])
        if not group_rows:
            continue
        print()
        print(f"group_size={group_size}")
        render_metric_table(group_rows, metric_name="operator_us", report_tflops=args.report_tflops)
        if args.reuse_output:
            print()
            render_metric_table(
                group_rows,
                metric_name="kernel_like_us",
                report_tflops=args.report_tflops,
            )


if __name__ == "__main__":
    main()
