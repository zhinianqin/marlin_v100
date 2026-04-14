from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jinja2")

from marlin_v100.calibration import source_target_cuda_arch_arg, source_target_sm_tag


ROOT = Path(__file__).resolve().parents[1]


def _run_generator(script: Path, arch: str) -> None:
    subprocess.check_call([sys.executable, str(script), arch], cwd=script.parent)


def test_dense_generator_outputs_are_stable():
    script = ROOT / "csrc" / "quantization" / "marlin" / "generate_kernels.py"
    target_arch = source_target_cuda_arch_arg()
    target_sm = source_target_sm_tag()
    _run_generator(script, target_arch)
    first = sorted(script.parent.glob(f"{target_sm}_kernel_*.cu"))
    assert first, "expected dense marlin kernels to be generated"
    assert (script.parent / "kernel_selector.h").exists()
    assert [path.name for path in first] == [
        path.name for path in sorted(script.parent.glob("sm*_kernel_*.cu"))
    ]

    _run_generator(script, target_arch)
    second = sorted(script.parent.glob(f"{target_sm}_kernel_*.cu"))
    assert [path.name for path in first] == [path.name for path in second]


def test_moe_generator_outputs_are_stable():
    script = ROOT / "csrc" / "moe" / "marlin_moe_wna16" / "generate_kernels.py"
    target_arch = source_target_cuda_arch_arg()
    target_sm = source_target_sm_tag()
    _run_generator(script, target_arch)
    first = sorted(script.parent.glob(f"{target_sm}_kernel_*.cu"))
    assert first, "expected moe marlin kernels to be generated"
    assert (script.parent / "kernel_selector.h").exists()
    assert [path.name for path in first] == [
        path.name for path in sorted(script.parent.glob("sm*_kernel_*.cu"))
    ]

    _run_generator(script, target_arch)
    second = sorted(script.parent.glob(f"{target_sm}_kernel_*.cu"))
    assert [path.name for path in first] == [path.name for path in second]
