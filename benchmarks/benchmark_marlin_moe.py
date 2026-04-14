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
from marlin_v100.calibration import (
    architecture_support,
    format_capability,
    runtime_capability,
    supported_dense_group_sizes,
    supported_moe_quant_type_names,
    source_target_cuda_arch_arg,
    source_target_label,
)
from marlin_v100 import moe, ops
from tests.helpers import (
    make_moe_model_like_inputs,
    marlin_moe_reference,
    marlin_quantize_experts_with_metadata,
    scalar_types,
)

_MOE_QUANT_TYPE_CANDIDATES = {
    "uint4b8": scalar_types.uint4b8,
    "uint8b128": scalar_types.uint8b128,
}
QUANT_TYPES = {
    name: _MOE_QUANT_TYPE_CANDIDATES[name]
    for name in supported_moe_quant_type_names(_MOE_QUANT_TYPE_CANDIDATES)
}
GROUP_SIZES = supported_dense_group_sizes((-1, 32, 64, 128))


def parse_act_order_values(mode: str) -> list[bool]:
    if mode == "off":
        return [False]
    if mode == "on":
        return [True]
    if mode == "all":
        return [False, True]
    raise ValueError(f"Unsupported act_order mode: {mode!r}")


def parse_is_k_full_values(mode: str) -> list[bool]:
    if mode == "true":
        return [True]
    if mode == "false":
        return [False]
    if mode == "all":
        return [True, False]
    raise ValueError(f"Unsupported is_k_full mode: {mode!r}")


def parse_args() -> argparse.Namespace:
    support = architecture_support()
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
        help="Quantized weight types to benchmark for the current source target.",
    )
    parser.add_argument(
        "--group-sizes",
        nargs="+",
        type=int,
        choices=list(GROUP_SIZES),
        default=list(GROUP_SIZES),
        help=(
            "Group sizes to benchmark for the current source target "
            f"({source_target_label()}; supported defaults={list(support.dense_group_sizes)})."
        ),
    )
    parser.add_argument(
        "--act-order",
        choices=("off", "on", "all"),
        default="off",
        help="Whether to benchmark act_order off only, on only, or both.",
    )
    parser.add_argument(
        "--is-k-full",
        choices=("true", "false", "all"),
        default="all",
        help="For act_order cases, benchmark full-K, non-full-K, or both.",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a small correctness sanity check before timing each case.",
    )
    return parser.parse_args()


def run_case(
    case_name: str,
    tokens: int,
    quant_name: str,
    group_size: int,
    act_order: bool,
    is_k_full: bool,
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
    if act_order and is_k_full and group_size == -1:
        return [
            case_name,
            quant_name,
            str(group_size),
            "yes",
            "yes",
            str(tokens),
            str(experts),
            str(hidden),
            str(intermediate),
            "SKIP",
            "unsupported",
            "n/a",
            "n/a",
            "n/a",
            "act_order with is_k_full=True requires more than one scale group",
        ]
    hidden_states, topk_weights, topk_ids, w1, w2 = make_moe_model_like_inputs(
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=experts,
        topk=topk,
        device=device,
    )
    w1_q, w1_scales, w1_dequant, w1_g_idx, w1_perm = marlin_quantize_experts_with_metadata(
        w1, quant_type, group_size, act_order
    )
    w2_q, w2_scales, w2_dequant, w2_g_idx, w2_perm = marlin_quantize_experts_with_metadata(
        w2, quant_type, group_size, act_order
    )

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
            g_idx1=w1_g_idx,
            g_idx2=w2_g_idx,
            sort_indices1=w1_perm,
            sort_indices2=w2_perm,
            is_k_full=is_k_full,
        )

    status = "unchecked"
    all_finite = "n/a"
    check_pass = "n/a"
    max_abs_err = "n/a"
    error = ""

    if check:
        output = run_marlin()
        reference = marlin_moe_reference(
            hidden_states,
            w1_dequant,
            w2_dequant,
            topk_weights,
            topk_ids,
        ).to(torch.float16)
        finite = bool(torch.isfinite(output).all().item())
        all_finite = "yes" if finite else "no"
        if finite:
            diff = (output - reference).abs().to(torch.float32)
            max_abs_err = format_float(float(diff.max().item()))
            try:
                torch.testing.assert_close(output, reference, rtol=7e-2, atol=1e-2)
                status = "ok"
                check_pass = "yes"
            except AssertionError as exc:
                status = "mismatch"
                check_pass = "no"
                error = str(exc).splitlines()[0]
        else:
            status = "non_finite"
            check_pass = "no"
            max_abs_err = "inf"

    try:
        stats = time_cuda_callable(run_marlin, warmup_iters=warmup_iters, iters=iters)
        marlin_us = format_float(stats["median_us"])
    except Exception as exc:
        status = "error"
        all_finite = "n/a" if not check else all_finite
        check_pass = "n/a" if not check else check_pass
        max_abs_err = "n/a" if not check else max_abs_err
        marlin_us = "ERR"
        error = str(exc).splitlines()[0]

    return [
        case_name,
        quant_name,
        str(group_size),
        "yes" if act_order else "no",
        "yes" if is_k_full else "no",
        str(tokens),
        str(experts),
        str(hidden),
        str(intermediate),
        marlin_us,
        status,
        all_finite,
        check_pass,
        max_abs_err,
        error,
    ]


def main() -> None:
    args = parse_args()
    check_cuda_ready()
    ops._load_moe()

    preset = MOE_PRESETS[args.preset]
    cases = args.cases or list(preset["cases"])
    tokens_list = args.tokens or list(preset["tokens"])

    banner(f"Marlin MoE Benchmark ({timestamp()})")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"capability={format_capability(runtime_capability(0))}")
    print(f"build_target={source_target_label()} ({source_target_cuda_arch_arg()})")
    print(f"preset={args.preset}")
    print(f"cases={cases}")
    print(f"tokens={tokens_list}")
    print(f"quant_types={args.quant_types}")
    print(f"group_sizes={args.group_sizes}")
    print(f"act_order={args.act_order}")
    print(f"is_k_full={args.is_k_full}")
    print(f"warmup_iters={args.warmup_iters}, iters={args.iters}, check={args.check}")

    rows: list[list[str]] = []
    act_order_values = parse_act_order_values(args.act_order)
    requested_is_k_full = parse_is_k_full_values(args.is_k_full)
    for case_name in cases:
        for tokens in tokens_list:
            for quant_name in args.quant_types:
                for group_size in args.group_sizes:
                    for act_order in act_order_values:
                        is_k_full_values = requested_is_k_full if act_order else [True]
                        for is_k_full in is_k_full_values:
                            rows.append(
                                run_case(
                                    case_name=case_name,
                                    tokens=tokens,
                                    quant_name=quant_name,
                                    group_size=group_size,
                                    act_order=act_order,
                                    is_k_full=is_k_full,
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
            "group_size",
            "act_order",
            "is_k_full",
            "tokens",
            "experts",
            "hidden",
            "intermediate",
            "marlin_us",
            "status",
            "all_finite",
            "check_pass",
            "max_abs_err",
            "error",
        ],
        rows=rows,
    )


if __name__ == "__main__":
    main()
