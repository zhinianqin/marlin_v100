from __future__ import annotations

import argparse

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
    warmup_iters: int,
    iters: int,
) -> list[str] | None:
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

    def run_torch() -> torch.Tensor:
        return torch.matmul(a, weight_ref)

    def run_marlin() -> torch.Tensor:
        return ops.marlin_gemm(
            a,
            None,
            q_weight,
            None,
            scales,
            None,
            None,
            None,
            empty_g_idx if g_idx.numel() == 0 else g_idx,
            sort_indices,
            workspace,
            quant_type.id,
            size_m,
            size_n,
            size_k,
            True,
            False,
            True,
            False,
        )

    torch_stats = time_cuda_callable(run_torch, warmup_iters=warmup_iters, iters=iters)
    marlin_stats = time_cuda_callable(run_marlin, warmup_iters=warmup_iters, iters=iters)
    speedup = torch_stats["median_us"] / marlin_stats["median_us"]
    return [
        model,
        quant_name,
        str(group_size),
        f"{size_m}x{size_k}x{size_n}",
        format_float(torch_stats["median_us"]),
        format_float(marlin_stats["median_us"]),
        f"{speedup:.2f}x",
    ]


def main() -> None:
    args = parse_args()
    check_cuda_ready()
    ops._load_dense()

    preset = DENSE_PRESETS[args.preset]
    models = args.models or list(preset["models"])
    batch_sizes = args.batch_sizes or list(preset["batch_sizes"])

    banner(f"Marlin Dense Benchmark ({timestamp()})")
    print(f"preset={args.preset}")
    print(f"models={models}")
    print(f"batch_sizes={batch_sizes}")
    print(f"quant_types={args.quant_types}")
    print(f"group_sizes={args.group_sizes}")
    print(f"warmup_iters={args.warmup_iters}, iters={args.iters}")

    rows: list[list[str]] = []
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
                            warmup_iters=args.warmup_iters,
                            iters=args.iters,
                        )
                        if row is not None:
                            rows.append(row)

    print()
    print_table(
        headers=[
            "model",
            "quant",
            "group",
            "MKN",
            "torch_us",
            "marlin_us",
            "speedup",
        ],
        rows=rows,
    )


if __name__ == "__main__":
    main()
