from __future__ import annotations

import argparse
from itertools import product

import torch

from common import (
    banner,
    check_cuda_ready,
    format_float,
    print_table,
    time_cuda_callable,
    timestamp,
)
from marlin_v100 import ops


CTA_M_CANDIDATES = (8, 16, 32, 64, 128)
CTA_N_CANDIDATES = (32, 64, 128, 256)
CTA_K_CANDIDATES = (64, 128)
WARP_CANDIDATES = (4, 8)
STAGE_CANDIDATES = (2,)
A_PATH_IDS = {
    "cutlass_shared": 0,
    "direct_global": 1,
    "cutlass_threadblock": 2,
}
B_PATH_IDS = {
    "cutlass_shared": 0,
}
V100_PEAK_TFLOPS = 125.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the private SM70 CUTLASS matmul probe."
    )
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--cta-m", nargs="+", type=int, help="CTA M candidates.")
    parser.add_argument("--cta-n", nargs="+", type=int, help="CTA N candidates.")
    parser.add_argument("--cta-k", nargs="+", type=int, help="CTA K candidates.")
    parser.add_argument("--warps", nargs="+", type=int, help="Warp-count candidates.")
    parser.add_argument(
        "--a-paths",
        nargs="+",
        choices=sorted(A_PATH_IDS),
        help="A input paths to benchmark. Defaults depend on --preset.",
    )
    parser.add_argument(
        "--b-paths",
        nargs="+",
        choices=sorted(B_PATH_IDS),
        help="B input paths to benchmark. Only the pure row-major CUTLASS path is kept.",
    )
    parser.add_argument(
        "--include-unsupported",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Print unsupported candidate rows instead of silently skipping them.",
    )
    return parser.parse_args()


def tflops_from_us(m: int, n: int, k: int, latency_us: float) -> float:
    if latency_us <= 0.0:
        return 0.0
    return (2.0 * m * n * k) / (latency_us * 1_000_000.0)


def max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs.float() - rhs.float()).abs().max().item()


def candidate_configs(args: argparse.Namespace):
    if args.preset == "quick":
        cta_m_values = (64, 128)
        cta_n_values = (64, 128, 256)
        cta_k_values = (32,)
        warp_values = (4, 8)
        a_paths = args.a_paths or ("cutlass_threadblock", "cutlass_shared")
    else:
        cta_m_values = CTA_M_CANDIDATES
        cta_n_values = CTA_N_CANDIDATES
        cta_k_values = CTA_K_CANDIDATES
        warp_values = WARP_CANDIDATES
        a_paths = args.a_paths or ("cutlass_shared", "direct_global")

    cta_m_values = tuple(args.cta_m or cta_m_values)
    cta_n_values = tuple(args.cta_n or cta_n_values)
    cta_k_values = tuple(args.cta_k or cta_k_values)
    warp_values = tuple(args.warps or warp_values)

    requested_b_paths = tuple(args.b_paths) if args.b_paths else None

    for cta_m, cta_n, cta_k, warps, stages, a_path in product(
        cta_m_values,
        cta_n_values,
        cta_k_values,
        warp_values,
        STAGE_CANDIDATES,
        a_paths,
    ):
        if requested_b_paths is not None:
            b_paths = requested_b_paths
        else:
            b_paths = ("cutlass_shared",)

        for b_path in b_paths:
            yield {
                "cta_m": cta_m,
                "cta_n": cta_n,
                "cta_k": cta_k,
                "warps": warps,
                "stages": stages,
                "a_path": a_path,
                "b_path": b_path,
            }


def run_probe_case(
    a: torch.Tensor,
    b_by_path: dict[str, torch.Tensor],
    reference_by_path: dict[str, torch.Tensor],
    cfg: dict[str, object],
    warmup_iters: int,
    iters: int,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    a_path_id = A_PATH_IDS[str(cfg["a_path"])]
    b_path = str(cfg["b_path"])
    b_path_id = B_PATH_IDS[b_path]
    b = b_by_path[b_path]
    reference = reference_by_path[b_path]

    def run() -> torch.Tensor:
        return ops.sm70_cutlass_matmul_probe(
            a,
            b,
            int(cfg["cta_m"]),
            int(cfg["cta_n"]),
            int(cfg["cta_k"]),
            int(cfg["warps"]),
            int(cfg["stages"]),
            a_path_id,
            b_path_id,
        )

    output = run()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)

    stats = time_cuda_callable(run, warmup_iters=warmup_iters, iters=iters)
    tflops = tflops_from_us(a.size(0), b.size(1), a.size(1), stats["median_us"])
    return {
        "status": "ok",
        "median_us": stats["median_us"],
        "mean_us": stats["mean_us"],
        "tflops": tflops,
        "peak_pct": 100.0 * tflops / V100_PEAK_TFLOPS,
        "max_abs_diff": max_abs_diff(output, reference),
    }


def main() -> None:
    args = parse_args()
    check_cuda_ready()
    ops._load_dense()

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    a = torch.randn((args.m, args.k), device=device, dtype=torch.float16)
    b_row_major = torch.randn((args.k, args.n), device=device, dtype=torch.float16)
    b_by_path = {
        "cutlass_shared": b_row_major,
    }

    reference_by_path = {
        name: torch.mm(a, b) for name, b in b_by_path.items()
    }
    torch.cuda.synchronize()

    banner("SM70 CUTLASS matmul probe")
    print(f"Timestamp: {timestamp()}")
    print(f"Problem: M={args.m}, N={args.n}, K={args.k}")
    print(f"Preset: {args.preset}")
    print()

    rows: list[list[str]] = []
    for cfg in candidate_configs(args):
        try:
            result = run_probe_case(
                a,
                b_by_path,
                reference_by_path,
                cfg,
                warmup_iters=args.warmup_iters,
                iters=args.iters,
                rtol=args.rtol,
                atol=args.atol,
            )
        except Exception as exc:
            if not args.include_unsupported:
                continue
            rows.append(
                [
                    str(cfg["cta_m"]),
                    str(cfg["cta_n"]),
                    str(cfg["cta_k"]),
                    str(cfg["warps"]),
                    str(cfg["stages"]),
                    str(cfg["a_path"]),
                    str(cfg["b_path"]),
                    "unsupported",
                    "-",
                    "-",
                    "-",
                    str(exc).splitlines()[0][:96],
                ]
            )
            continue

        rows.append(
            [
                str(cfg["cta_m"]),
                str(cfg["cta_n"]),
                str(cfg["cta_k"]),
                str(cfg["warps"]),
                str(cfg["stages"]),
                str(cfg["a_path"]),
                str(cfg["b_path"]),
                "ok",
                format_float(float(result["median_us"])),
                format_float(float(result["tflops"])),
                format_float(float(result["peak_pct"])),
                f"max_abs={result['max_abs_diff']:.4f}",
            ]
        )

    print_table(
        [
            "cta_m",
            "cta_n",
            "cta_k",
            "warps",
            "stage",
            "a_path",
            "b_path",
            "status",
            "median_us",
            "TFLOPs",
            "%peak",
            "notes",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
