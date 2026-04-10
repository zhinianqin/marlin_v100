from __future__ import annotations

import argparse

import torch

from benchmark_shapes import MOE_CASES, MOE_PRESETS
from common import (
    banner,
    check_cuda_ready,
    format_float,
    print_table,
    time_cuda_callable,
    timestamp,
)
from marlin_v100 import moe, ops
from tests.helpers import (
    marlin_moe_reference,
    marlin_quantize_experts,
    scalar_types,
)


QUANT_TYPES = {
    "uint4b8": scalar_types.uint4b8,
    "uint8b128": scalar_types.uint8b128,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark local Marlin MoE kernels.")
    parser.add_argument(
        "--preset",
        choices=sorted(MOE_PRESETS.keys()),
        default="full",
        help="Benchmark preset to run.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(MOE_CASES.keys()),
        help="MoE shape presets to run. Defaults to the selected preset.",
    )
    parser.add_argument(
        "--tokens",
        nargs="+",
        type=int,
        help="Token counts to benchmark. Defaults to the selected preset.",
    )
    parser.add_argument(
        "--quant-types",
        nargs="+",
        choices=sorted(QUANT_TYPES.keys()),
        default=list(QUANT_TYPES.keys()),
        help="Quantized weight types to benchmark.",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a small correctness sanity check before timing each case.",
    )
    return parser.parse_args()


def make_routing(tokens: int, experts: int, topk: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.empty((tokens, topk), dtype=torch.int32, device=device)
    for token in range(tokens):
        for route in range(topk):
            topk_ids[token, route] = (token + route) % experts
    topk_weights = torch.tensor([0.6, 0.4], device=device, dtype=torch.float32)
    topk_weights = topk_weights.repeat(tokens, 1)
    topk_weights = topk_weights[:, :topk]
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_ids, topk_weights


def run_case(
    case_name: str,
    tokens: int,
    quant_name: str,
    warmup_iters: int,
    iters: int,
    check: bool,
) -> list[str]:
    case = MOE_CASES[case_name]
    experts = case["experts"]
    topk = case["topk"]
    hidden = case["hidden"]
    intermediate = case["intermediate"]

    quant_type = QUANT_TYPES[quant_name]
    device = torch.device("cuda")
    hidden_states = torch.randn((tokens, hidden), device=device, dtype=torch.float16)
    topk_ids, topk_weights = make_routing(tokens, experts, topk, device)

    w1 = torch.randn((experts, hidden, 2 * intermediate), device=device, dtype=torch.float16)
    w2 = torch.randn((experts, intermediate, hidden), device=device, dtype=torch.float16)
    w1_q, w1_scales, w1_dequant = marlin_quantize_experts(w1, quant_type, 128, False)
    w2_q, w2_scales, w2_dequant = marlin_quantize_experts(w2, quant_type, 128, False)

    def run_marlin() -> torch.Tensor:
        return moe.fused_marlin_moe(
            hidden_states=hidden_states,
            w1=w1_q,
            w2=w2_q,
            w1_scale=w1_scales,
            w2_scale=w2_scales,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=quant_type.id,
            is_k_full=True,
        )

    if check:
        output = run_marlin()
        reference = marlin_moe_reference(
            hidden_states,
            w1_dequant,
            w2_dequant,
            topk_weights,
            topk_ids,
        ).to(torch.float16)
        torch.testing.assert_close(output, reference, rtol=8e-2, atol=1.1)

    stats = time_cuda_callable(run_marlin, warmup_iters=warmup_iters, iters=iters)
    return [
        case_name,
        quant_name,
        str(tokens),
        str(experts),
        str(hidden),
        str(intermediate),
        format_float(stats["median_us"]),
    ]


def main() -> None:
    args = parse_args()
    check_cuda_ready()
    ops._load_moe()

    preset = MOE_PRESETS[args.preset]
    cases = args.cases or list(preset["cases"])
    tokens_list = args.tokens or list(preset["tokens"])

    banner(f"Marlin MoE Benchmark ({timestamp()})")
    print(f"preset={args.preset}")
    print(f"cases={cases}")
    print(f"tokens={tokens_list}")
    print(f"quant_types={args.quant_types}")
    print(f"warmup_iters={args.warmup_iters}, iters={args.iters}, check={args.check}")

    rows: list[list[str]] = []
    for case_name in cases:
        for tokens in tokens_list:
            for quant_name in args.quant_types:
                rows.append(
                    run_case(
                        case_name=case_name,
                        tokens=tokens,
                        quant_name=quant_name,
                        warmup_iters=args.warmup_iters,
                        iters=args.iters,
                        check=args.check,
                    )
                )

    print()
    print_table(
        headers=[
            "case",
            "quant",
            "tokens",
            "experts",
            "hidden",
            "intermediate",
            "marlin_us",
        ],
        rows=rows,
    )


if __name__ == "__main__":
    main()
