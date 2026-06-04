from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any

import torch

from benchmark_shapes import DENSE_PRESETS, DENSE_WEIGHT_SHAPES
from common import require_matching_cuda_benchmark_runtime, timestamp
from marlin_v100 import ops


ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = ROOT / "benchmarks" / "results"
EXTENSION_PATH = ROOT / "python" / "marlin_v100" / "_C.abi3.so"

THREADBLOCK_CTA_M_CANDIDATES = (32, 64, 128, 256, 512)
THREADBLOCK_CTA_N_CANDIDATES = (64, 128, 256, 512)
CUTE_CTA_M_CANDIDATES = (8, 16, 32, 48, 64)
CUTE_CTA_N_CANDIDATES = (64, 128, 256)
CTA_K = 32
SM70_ATOM_CTA_K = 128
CTA_K_CANDIDATES = (CTA_K,)
WARP_CANDIDATES = (4, 8)
STAGES = 2
A_PATH_IDS = {
    "cute_shared": 0,
    "cutlass_threadblock": 2,
    "sm70_atom": 3,
}
B_PATH_CUTLASS_SHARED = 0
V100_PEAK_TFLOPS = 125.0

DEFAULT_BATCH_SIZES = (32, 64, 128, 256, 512, 1024, 2048, 4096, 5120)
DEFAULT_N_SWEEP = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 12288, 22016)
DEFAULT_K_SWEEP = (512, 1024, 2048, 4096, 8192, 11008)


def default_mkn_shapes() -> tuple[tuple[int, int, int], ...]:
    shapes: set[tuple[int, int, int]] = set()
    dense_models = DENSE_PRESETS["full"]["models"]
    for model in dense_models:
        for size_k, size_n in DENSE_WEIGHT_SHAPES[str(model)]:
            for size_m in DEFAULT_BATCH_SIZES:
                shapes.add((size_m, size_n, size_k))

    shapes.update((5120, size_n, 4096) for size_n in DEFAULT_N_SWEEP)
    shapes.update((5120, 4096, size_k) for size_k in DEFAULT_K_SWEEP)
    return tuple(sorted(shapes))


DEFAULT_MKN = default_mkn_shapes()
DEFAULT_CUTE_MKN = (
    (384, 512, 512),
    (768, 1024, 1024),
    (1536, 4096, 4096),
    (3072, 4096, 4096),
)

SUPPORTED_THREADBLOCK_CONFIGS = {
    (cta_m, cta_n, CTA_K, warps)
    for cta_m, cta_n, warps in (
        (32, 128, 4),
        (32, 256, 4),
        (64, 64, 4),
        (64, 128, 4),
        (64, 128, 8),
        (64, 256, 4),
        (64, 256, 8),
        (64, 512, 8),
        (128, 64, 4),
        (128, 64, 8),
        (128, 128, 4),
        (128, 128, 8),
        (128, 256, 8),
        (256, 64, 4),
        (256, 64, 8),
        (256, 128, 8),
        (512, 64, 8),
    )
}
SUPPORTED_CUTE_CONFIGS = {
    (cta_m, cta_n, CTA_K, warps)
    for cta_m, cta_n, warps in product(
        CUTE_CTA_M_CANDIDATES,
        CUTE_CTA_N_CANDIDATES,
        WARP_CANDIDATES,
    )
}
SUPPORTED_SM70_ATOM_CONFIGS = {
    (8, 64, SM70_ATOM_CTA_K, 4),
    (16, 64, SM70_ATOM_CTA_K, 4),
}

RESULT_COLUMNS = (
    "M",
    "N",
    "K",
    "CTA_M",
    "CTA_N",
    "CTA_K",
    "Warps",
    "a_path",
    "status",
    "avg_us",
    "avg_tflops",
    "repeats",
    "REG",
    "STACK",
    "LOCAL",
    "SHARED",
    "spill_stores",
    "spill_loads",
    "max_abs_diff",
    "notes",
)

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
THREADBLOCK_KERNEL_RE = re.compile(
    r"sm70_cutlass_threadblock_gemm_kernel<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>"
)
CUTE_KERNEL_RE = re.compile(
    r"sm70_cute_gemm_kernel<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>"
)
SM70_ATOM_KERNEL_RE = re.compile(
    r"sm70_atom_gemm_kernel<\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*>"
)
RESOURCE_RE = re.compile(r"([A-Z]+(?:\[\d+\])?):([0-9]+)")
SPILL_RE = re.compile(
    r"(\d+)\s+bytes\s+stack\s+frame,\s+(\d+)\s+bytes\s+spill\s+stores,\s+(\d+)\s+bytes\s+spill\s+loads"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unattended SM70 pure GEMM tile/resource/benchmark sweep."
    )
    parser.add_argument(
        "--a-path",
        choices=sorted(A_PATH_IDS),
        default="cutlass_threadblock",
        help="Probe path to sweep. Defaults to the extracted CUTLASS threadblock path.",
    )
    parser.add_argument("--mkn", action="append", help="Shape as M,N,K. May be repeated.")
    parser.add_argument("--mkn-file", type=Path, help="Text file with one M,N,K per line.")
    parser.add_argument("--cta-m", nargs="+", type=int)
    parser.add_argument("--cta-n", nargs="+", type=int)
    parser.add_argument(
        "--cta-k",
        nargs="+",
        type=int,
        default=None,
        help="CTA K candidates. cute_shared/cutlass_threadblock use CTA_K=32; sm70_atom uses CTA_K=128.",
    )
    parser.add_argument("--warps", nargs="+", type=int, default=list(WARP_CANDIDATES))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="Resume from --output-dir, or latest sweep dir.")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument(
        "--force-probe-rebuild",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove the probe object before build so ptxas resource/spill lines are emitted.",
    )
    parser.add_argument(
        "--include-unsupported",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record unsupported geometry and invalid shape rows.",
    )
    args = parser.parse_args()
    if args.cta_m is None:
        if args.a_path == "cute_shared":
            args.cta_m = list(CUTE_CTA_M_CANDIDATES)
        elif args.a_path == "sm70_atom":
            args.cta_m = [8, 16]
        else:
            args.cta_m = list(THREADBLOCK_CTA_M_CANDIDATES)
    if args.cta_n is None:
        if args.a_path == "cute_shared":
            args.cta_n = list(CUTE_CTA_N_CANDIDATES)
        elif args.a_path == "sm70_atom":
            args.cta_n = [64]
        else:
            args.cta_n = list(THREADBLOCK_CTA_N_CANDIDATES)
    if args.a_path == "sm70_atom" and args.warps == list(WARP_CANDIDATES):
        args.warps = [4]
    if args.cta_k is None:
        args.cta_k = [SM70_ATOM_CTA_K] if args.a_path == "sm70_atom" else list(CTA_K_CANDIDATES)
    supported_cta_k = SM70_ATOM_CTA_K if args.a_path == "sm70_atom" else CTA_K
    if any(cta_k != supported_cta_k for cta_k in args.cta_k):
        parser.error(f"{args.a_path} pure GEMM sweep only supports CTA_K={supported_cta_k}")
    return args


def parse_mkn(value: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected M,N,K, got {value!r}")
    try:
        m, n, k = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer M,N,K, got {value!r}") from exc
    if m <= 0 or n <= 0 or k <= 0:
        raise argparse.ArgumentTypeError(f"M,N,K must be positive, got {value!r}")
    return m, n, k


def load_shapes(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    shapes: list[tuple[int, int, int]] = []
    if args.mkn:
        shapes.extend(parse_mkn(value) for value in args.mkn)
    if args.mkn_file:
        for line_number, raw_line in enumerate(args.mkn_file.read_text().splitlines(), 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            try:
                shapes.append(parse_mkn(line))
            except argparse.ArgumentTypeError as exc:
                raise ValueError(f"{args.mkn_file}:{line_number}: {exc}") from exc
    if not shapes:
        shapes.extend(DEFAULT_CUTE_MKN if args.a_path == "cute_shared" else DEFAULT_MKN)
    return sorted(set(shapes))


def latest_sweep_dir() -> Path | None:
    if not RESULTS_ROOT.exists():
        return None
    candidates = sorted(RESULTS_ROOT.glob("*_sm70_pure_gemm_sweep"))
    return candidates[-1] if candidates else None


def output_dir_for_run(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.resume:
        latest = latest_sweep_dir()
        if latest is not None:
            return latest
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return RESULTS_ROOT / f"{stamp}_sm70_pure_gemm_sweep"


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def run_logged_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    print(f"$ {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        status = process.wait()
    if status != 0:
        raise RuntimeError(f"Command failed with exit code {status}: {' '.join(cmd)}")


def remove_probe_objects() -> list[Path]:
    removed: list[Path] = []
    for path in (ROOT / "build").glob(
        "temp.*/CMakeFiles/_C.dir/csrc/quantization/marlin/sm70_cutlass_matmul_probe.cu.o"
    ):
        path.unlink(missing_ok=True)
        removed.append(path)
    return removed


def run_build(args: argparse.Namespace, out_dir: Path, env: dict[str, str]) -> str:
    build_log = out_dir / "build.log"
    if args.skip_build:
        message = "Build skipped by --skip-build.\n"
        build_log.write_text(message, encoding="utf-8")
        print(message, end="")
        return message
    if args.force_probe_rebuild:
        removed = remove_probe_objects()
        if removed:
            print(f"Removed {len(removed)} probe object(s) to force ptxas output.")
    run_logged_command(["./build.sh"], build_log, env)
    return build_log.read_text(encoding="utf-8", errors="replace")


def run_cuobjdump(out_dir: Path, env: dict[str, str]) -> str:
    cuda_home = Path(env.get("CUDA_HOME", "/usr/local/cuda-12.8"))
    cuobjdump = cuda_home / "bin" / "cuobjdump"
    if not EXTENSION_PATH.exists():
        raise FileNotFoundError(f"Missing extension: {EXTENSION_PATH}")
    cmd = f"{cuobjdump} --dump-resource-usage {EXTENSION_PATH} | c++filt"
    print(f"$ {cmd}")
    result = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout
    (out_dir / "resource_usage.txt").write_text(output, encoding="utf-8")
    return output


KernelKey = tuple[str, int, int, int, int]


def shape_from_text(text: str) -> KernelKey | None:
    match = THREADBLOCK_KERNEL_RE.search(text)
    if match:
        return ("cutlass_threadblock", *(int(group) for group in match.groups()))
    match = CUTE_KERNEL_RE.search(text)
    if match:
        return ("cute_shared", *(int(group) for group in match.groups()))
    match = SM70_ATOM_KERNEL_RE.search(text)
    if match:
        return ("sm70_atom", *(int(group) for group in match.groups()))
    return None


def parse_resource_usage(text: str) -> dict[KernelKey, dict[str, int]]:
    resources: dict[KernelKey, dict[str, int]] = {}
    current_key: KernelKey | None = None
    for line in text.splitlines():
        key = shape_from_text(line)
        if key is not None:
            current_key = key
            continue
        if current_key is None or "REG:" not in line:
            continue
        values = {name: int(value) for name, value in RESOURCE_RE.findall(line)}
        resources[current_key] = {
            "REG": values.get("REG", -1),
            "STACK": values.get("STACK", -1),
            "LOCAL": values.get("LOCAL", -1),
            "SHARED": values.get("SHARED", -1),
        }
        current_key = None
    return resources


def demangle_names(names: list[str]) -> list[str]:
    if not names:
        return []
    result = subprocess.run(
        ["c++filt"],
        input="\n".join(names),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.splitlines()


def parse_ptxas_spills(text: str) -> dict[KernelKey, dict[str, int]]:
    clean_lines = strip_ansi(text).splitlines()
    entries: list[tuple[str, tuple[int, int, int]]] = []
    for index, line in enumerate(clean_lines):
        if "Function properties for " not in line:
            continue
        name = line.split("Function properties for ", 1)[1].strip().rstrip(":")
        window = " ".join(clean_lines[index + 1 : index + 6])
        match = SPILL_RE.search(window)
        if not match:
            continue
        stack, stores, loads = (int(group) for group in match.groups())
        entries.append((name, (stack, stores, loads)))

    demangled = demangle_names([name for name, _ in entries])
    spills: dict[KernelKey, dict[str, int]] = {}
    for (_, values), demangled_name in zip(entries, demangled):
        key = shape_from_text(demangled_name)
        if key is None:
            continue
        stack, stores, loads = values
        spills[key] = {
            "ptxas_stack": stack,
            "spill_stores": stores,
            "spill_loads": loads,
        }
    return spills


def tflops_from_us(m: int, n: int, k: int, latency_us: float) -> float:
    if latency_us <= 0.0:
        return 0.0
    return (2.0 * m * n * k) / (latency_us * 1_000_000.0)


def max_abs_diff(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs.float() - rhs.float()).abs().max().item()


def time_probe(fn, warmup_iters: int, iters: int) -> dict[str, float]:
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


def candidate_configs(args: argparse.Namespace) -> list[tuple[int, int, int, int]]:
    return [
        (cta_m, cta_n, cta_k, warps)
        for cta_m, cta_n, cta_k, warps in product(
            args.cta_m,
            args.cta_n,
            args.cta_k,
            args.warps,
        )
    ]


def raw_key(record: dict[str, Any]) -> tuple[str, int, int, int, int, int, int, int, int]:
    return (
        str(record.get("a_path", "cutlass_threadblock")),
        int(record["M"]),
        int(record["N"]),
        int(record["K"]),
        int(record["CTA_M"]),
        int(record["CTA_N"]),
        int(record["CTA_K"]),
        int(record["Warps"]),
        int(record.get("repeat", -1)),
    )


def terminal_key(record: dict[str, Any]) -> tuple[str, int, int, int, int, int, int, int]:
    return raw_key(record)[:8]


def read_raw_records(raw_path: Path) -> list[dict[str, Any]]:
    if not raw_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records.append(record)
    return records


def append_raw_record(raw_path: Path, record: dict[str, Any]) -> None:
    with raw_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def resource_for(
    a_path: str,
    cfg: tuple[int, int, int, int],
    resources: dict[KernelKey, dict[str, int]],
    spills: dict[KernelKey, dict[str, int]],
) -> dict[str, int]:
    key = (a_path, *cfg)
    resource = dict(resources.get(key, {}))
    spill = spills.get(key, {})
    resource["spill_stores"] = spill.get("spill_stores", 0)
    resource["spill_loads"] = spill.get("spill_loads", 0)
    return resource


def base_record(
    a_path: str,
    m: int,
    n: int,
    k: int,
    cfg: tuple[int, int, int, int],
    repeat: int,
    status: str,
    resources: dict[str, int],
    notes: str = "",
) -> dict[str, Any]:
    cta_m, cta_n, cta_k, warps = cfg
    return {
        "M": m,
        "N": n,
        "K": k,
        "CTA_M": cta_m,
        "CTA_N": cta_n,
        "CTA_K": cta_k,
        "Warps": warps,
        "a_path": a_path,
        "repeat": repeat,
        "status": status,
        "latency_us": None,
        "tflops": None,
        "REG": resources.get("REG", -1),
        "STACK": resources.get("STACK", -1),
        "LOCAL": resources.get("LOCAL", -1),
        "SHARED": resources.get("SHARED", -1),
        "spill_stores": resources.get("spill_stores", 0),
        "spill_loads": resources.get("spill_loads", 0),
        "max_abs_diff": None,
        "notes": notes,
    }


def run_config_repeats(
    raw_path: Path,
    completed_repeats: set[tuple[str, int, int, int, int, int, int, int, int]],
    terminal_configs: set[tuple[str, int, int, int, int, int, int, int]],
    m: int,
    n: int,
    k: int,
    cfg: tuple[int, int, int, int],
    args: argparse.Namespace,
    resources: dict[str, int],
    a: torch.Tensor,
    b: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    cta_m, cta_n, cta_k, warps = cfg
    a_path = str(args.a_path)
    term_key = (a_path, m, n, k, cta_m, cta_n, cta_k, warps)
    if term_key in terminal_configs:
        return

    supported_configs = (
        SUPPORTED_CUTE_CONFIGS
        if a_path == "cute_shared"
        else SUPPORTED_SM70_ATOM_CONFIGS
        if a_path == "sm70_atom"
        else SUPPORTED_THREADBLOCK_CONFIGS
    )
    if cfg not in supported_configs:
        if args.include_unsupported:
            record = base_record(
                a_path,
                m,
                n,
                k,
                cfg,
                -1,
                "unsupported_geometry",
                resources,
                f"No SM70 {a_path} kernel is instantiated for this config.",
            )
            append_raw_record(raw_path, record)
            terminal_configs.add(term_key)
        return

    if (
        (a_path != "cutlass_threadblock" and m % cta_m != 0)
        or n % cta_n != 0
        or k % cta_k != 0
    ):
        if args.include_unsupported:
            record = base_record(
                a_path,
                m,
                n,
                k,
                cfg,
                -1,
                "invalid_shape",
                resources,
                "M/N/K must be divisible by CTA_M/CTA_N/CTA_K for this probe.",
            )
            append_raw_record(raw_path, record)
            terminal_configs.add(term_key)
        return

    def run() -> torch.Tensor:
        return ops.sm70_cutlass_matmul_probe(
            a,
            b,
            cta_m,
            cta_n,
            cta_k,
            warps,
            STAGES,
            A_PATH_IDS[a_path],
            B_PATH_CUTLASS_SHARED,
        )

    try:
        output = run()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, reference, rtol=args.rtol, atol=args.atol)
        diff = max_abs_diff(output, reference)
    except Exception as exc:
        record = base_record(a_path, m, n, k, cfg, -1, "failure", resources, str(exc).splitlines()[0][:240])
        append_raw_record(raw_path, record)
        terminal_configs.add(term_key)
        return

    for repeat in range(args.repeats):
        key = (a_path, m, n, k, cta_m, cta_n, cta_k, warps, repeat)
        if key in completed_repeats:
            continue
        try:
            stats = time_probe(run, warmup_iters=args.warmup_iters, iters=args.iters)
            latency_us = float(stats["median_us"])
            record = base_record(a_path, m, n, k, cfg, repeat, "ok", resources)
            record.update(
                {
                    "latency_us": latency_us,
                    "tflops": tflops_from_us(m, n, k, latency_us),
                    "mean_us": float(stats["mean_us"]),
                    "min_us": float(stats["min_us"]),
                    "max_us": float(stats["max_us"]),
                    "max_abs_diff": diff,
                }
            )
        except Exception as exc:
            record = base_record(a_path, m, n, k, cfg, repeat, "failure", resources, str(exc).splitlines()[0][:240])
        append_raw_record(raw_path, record)
        completed_repeats.add(key)


def aggregate_records(records: list[dict[str, Any]], target_repeats: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int, int, int, int, int], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(terminal_key(record), []).append(record)

    rows: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        a_path, m, n, k, cta_m, cta_n, cta_k, warps = key
        ok_records = [record for record in group if record["status"] == "ok"]
        first = group[0]
        row = {
            "M": m,
            "N": n,
            "K": k,
            "CTA_M": cta_m,
            "CTA_N": cta_n,
            "CTA_K": cta_k,
            "Warps": warps,
            "a_path": a_path,
            "REG": first.get("REG", -1),
            "STACK": first.get("STACK", -1),
            "LOCAL": first.get("LOCAL", -1),
            "SHARED": first.get("SHARED", -1),
            "spill_stores": first.get("spill_stores", 0),
            "spill_loads": first.get("spill_loads", 0),
            "notes": first.get("notes", ""),
        }
        if ok_records:
            latencies = [float(record["latency_us"]) for record in ok_records]
            avg_us = statistics.fmean(latencies)
            row.update(
                {
                    "status": "ok" if len(ok_records) >= target_repeats else "partial_ok",
                    "avg_us": avg_us,
                    "avg_tflops": tflops_from_us(m, n, k, avg_us),
                    "repeats": len(ok_records),
                    "max_abs_diff": max(float(record.get("max_abs_diff") or 0.0) for record in ok_records),
                }
            )
        else:
            terminal = next((record for record in group if int(record.get("repeat", -1)) == -1), first)
            row.update(
                {
                    "status": terminal["status"],
                    "avg_us": "",
                    "avg_tflops": "",
                    "repeats": 0,
                    "max_abs_diff": "",
                    "notes": terminal.get("notes", ""),
                }
            )
        rows.append(row)
    return rows


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})


def markdown_table(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> str:
    rendered = [[format_cell(row.get(column, "")) for column in columns] for row in rows]
    widths = [len(column) for column in columns]
    for row in rendered:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(width) for cell, width in zip(cells, widths)) + " |"

    lines = [render_row(list(columns))]
    lines.append("| " + " | ".join("-" * width for width in widths) + " |")
    lines.extend(render_row(row) for row in rendered)
    return "\n".join(lines) + "\n"


def write_sorted_tables(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "a_path",
        "CTA_M",
        "CTA_N",
        "CTA_K",
        "Warps",
        "avg_tflops",
        "avg_us",
        "REG",
        "STACK",
        "LOCAL",
        "spill_stores",
        "spill_loads",
        "repeats",
    )
    shapes = sorted(
        {
            (str(row.get("a_path", "cutlass_threadblock")), int(row["M"]), int(row["N"]), int(row["K"]))
            for row in rows
        }
    )
    for a_path, m, n, k in shapes:
        ok_rows = [
            row
            for row in rows
            if row.get("a_path", "cutlass_threadblock") == a_path
            and row["M"] == m
            and row["N"] == n
            and row["K"] == k
            and row["status"] in {"ok", "partial_ok"}
        ]
        by_tflops = sorted(
            ok_rows,
            key=lambda row: (-float(row["avg_tflops"]), float(row["avg_us"]), int(row["REG"])),
        )
        by_resource = sorted(
            ok_rows,
            key=lambda row: (
                int(row["REG"]),
                -float(row["avg_tflops"]),
                int(row["spill_stores"]),
                int(row["spill_loads"]),
                int(row["STACK"]),
                int(row["LOCAL"]),
            ),
        )
        stem = f"{a_path}_M{m}_N{n}_K{k}"
        (out_dir / f"{stem}_tflops.md").write_text(
            f"# {stem} TFLOPs sort\n\n" + markdown_table(by_tflops, columns),
            encoding="utf-8",
        )
        (out_dir / f"{stem}_resource.md").write_text(
            f"# {stem} resource sort\n\n" + markdown_table(by_resource, columns),
            encoding="utf-8",
        )


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or "unknown"


def write_summary(out_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ok_rows = [row for row in rows if row["status"] in {"ok", "partial_ok"}]
    best_by_shape: list[dict[str, Any]] = []
    for shape in sorted({(row["a_path"], row["M"], row["N"], row["K"]) for row in ok_rows}):
        shape_rows = [
            row
            for row in ok_rows
            if (row["a_path"], row["M"], row["N"], row["K"]) == shape
        ]
        best_by_shape.append(max(shape_rows, key=lambda row: float(row["avg_tflops"])))

    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1

    lines = [
        "# SM70 pure GEMM sweep",
        "",
        f"- Timestamp: {timestamp()}",
        f"- Git commit: {git_commit()}",
        f"- A path: {args.a_path}",
        f"- Repeats: {args.repeats}",
        f"- Warmup iters: {args.warmup_iters}",
        f"- Timed iters: {args.iters}",
        f"- Status counts: {status_counts}",
        "",
        "## Best TFLOPs Per MKN",
        "",
    ]
    lines.append(
        markdown_table(
            best_by_shape,
            (
                "M",
                "N",
                "K",
                "a_path",
                "CTA_M",
                "CTA_N",
                "CTA_K",
                "Warps",
                "avg_tflops",
                "avg_us",
                "REG",
                "STACK",
                "LOCAL",
                "spill_stores",
                "spill_loads",
            ),
        )
    )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `build.log`",
            "- `resource_usage.txt`",
            "- `raw_results.jsonl`",
            "- `all_results.csv`",
            "- `M*_N*_K*_tflops.md` and `M*_N*_K*_resource.md`",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.repeats <= 0 or args.warmup_iters < 0 or args.iters <= 0:
        raise ValueError("--repeats and --iters must be positive; --warmup-iters must be non-negative.")

    out_dir = output_dir_for_run(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "raw_results.jsonl"
    print(f"Output directory: {out_dir}")

    env = os.environ.copy()
    env.setdefault("CUDA_HOME", "/usr/local/cuda-12.8")
    env["PATH"] = f"{ROOT / '.venv' / 'bin'}:{Path(env['CUDA_HOME']) / 'bin'}:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{Path(env['CUDA_HOME']) / 'lib64'}:{env.get('LD_LIBRARY_PATH', '')}"
    env.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
    env["PYTHONPATH"] = f"{ROOT / 'python'}:{env.get('PYTHONPATH', '')}"

    build_text = run_build(args, out_dir, env)
    resource_text = run_cuobjdump(out_dir, env)
    resources = parse_resource_usage(resource_text)
    spills = parse_ptxas_spills(build_text)

    require_matching_cuda_benchmark_runtime()
    ops._load_dense()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    existing_records = read_raw_records(raw_path) if args.resume else []
    completed_repeats = {raw_key(record) for record in existing_records if record.get("status") == "ok"}
    terminal_configs = {
        terminal_key(record)
        for record in existing_records
        if int(record.get("repeat", -1)) == -1 and record.get("status") != "ok"
    }

    shapes = load_shapes(args)
    configs = candidate_configs(args)
    print(f"A path: {args.a_path}")
    print(f"Shapes: {len(shapes)}")
    print(f"Candidate configs: {len(configs)}")

    for m, n, k in shapes:
        print(f"\n=== M={m} N={n} K={k} ===")
        a = torch.randn((m, k), device=device, dtype=torch.float16)
        b = torch.randn((k, n), device=device, dtype=torch.float16)
        reference = torch.mm(a, b)
        torch.cuda.synchronize()
        for cfg in configs:
            cta_m, cta_n, cta_k, warps = cfg
            print(f"Config CTA={cta_m}x{cta_n}x{cta_k}/{warps}w")
            run_config_repeats(
                raw_path,
                completed_repeats,
                terminal_configs,
                m,
                n,
                k,
                cfg,
                args,
                resource_for(str(args.a_path), cfg, resources, spills),
                a,
                b,
                reference,
            )
        del a, b, reference
        torch.cuda.empty_cache()

    all_records = read_raw_records(raw_path)
    rows = aggregate_records(all_records, args.repeats)
    write_csv(out_dir / "all_results.csv", rows)
    write_sorted_tables(out_dir, rows)
    write_summary(out_dir, rows, args)
    print(f"\nWrote sweep results to {out_dir}")


if __name__ == "__main__":
    main()
