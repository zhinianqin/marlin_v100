from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_FILES = (
    ROOT / "csrc/quantization/marlin/generate_kernels.py",
    ROOT / "csrc/quantization/marlin/kernel.h",
    ROOT / "csrc/quantization/marlin/kernel_selector.h",
    ROOT / "csrc/quantization/marlin/marlin_dtypes.cuh",
    ROOT / "csrc/quantization/marlin/marlin_mma.h",
    ROOT / "csrc/quantization/marlin/marlin_template.h",
    ROOT / "csrc/moe/marlin_moe_wna16/generate_kernels.py",
    ROOT / "csrc/moe/marlin_moe_wna16/kernel.h",
    ROOT / "csrc/moe/marlin_moe_wna16/kernel_selector.h",
    ROOT / "csrc/moe/marlin_moe_wna16/marlin_template.h",
)

FORBIDDEN_GLOBS = (
    "csrc/quantization/marlin/sm70_kernel_*.cu",
    "csrc/moe/marlin_moe_wna16/sm70_kernel_*.cu",
)

FORBIDDEN_TOKENS = (
    "generate_kernels.py",
    "marlin_template.h",
    "marlin_mma.h",
    "kernel_selector.h",
    "sm70_kernel_",
    "MarlinDefault",
    "get_marlin_kernel",
    "MarlinFuncPtr",
    "marlin_mm(",
    "MARLIN_KERNEL_PARAMS",
    "marlin_dtypes.cuh",
)

TEXT_SUFFIXES = {
    ".cmake",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".cpp",
    ".txt",
}


def _source_files_to_scan() -> list[Path]:
    paths = [ROOT / "CMakeLists.txt"]
    paths.extend(
        path
        for path in (ROOT / "csrc").rglob("*")
        if path.is_file() and path.suffix in TEXT_SUFFIXES
    )
    return paths


def test_legacy_marlin_template_generator_chain_is_removed():
    existing_files = [path.relative_to(ROOT).as_posix() for path in FORBIDDEN_FILES if path.exists()]
    assert not existing_files, "\n".join(existing_files)

    generated_files = []
    for pattern in FORBIDDEN_GLOBS:
        generated_files.extend(path.relative_to(ROOT).as_posix() for path in ROOT.glob(pattern))
    assert not generated_files, "\n".join(sorted(generated_files))

    offenders = []
    for path in _source_files_to_scan():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: {token}")

    assert not offenders, "\n".join(sorted(offenders))
