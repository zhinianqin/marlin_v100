from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("jinja2")


ROOT = Path(__file__).resolve().parents[1]


def _run_generator(script: Path, arch: str) -> None:
    subprocess.check_call([sys.executable, str(script), arch], cwd=script.parent)


def test_dense_generator_outputs_are_stable():
    script = ROOT / "csrc" / "quantization" / "marlin" / "generate_kernels.py"
    _run_generator(script, "7.5")
    first = sorted(script.parent.glob("sm75_kernel_*.cu"))
    assert first, "expected dense marlin kernels to be generated"
    assert (script.parent / "kernel_selector.h").exists()
    assert not list(script.parent.glob("sm80_kernel_*.cu"))
    assert not list(script.parent.glob("sm89_kernel_*.cu"))

    _run_generator(script, "7.5")
    second = sorted(script.parent.glob("sm75_kernel_*.cu"))
    assert [path.name for path in first] == [path.name for path in second]


def test_moe_generator_outputs_are_stable():
    script = ROOT / "csrc" / "moe" / "marlin_moe_wna16" / "generate_kernels.py"
    _run_generator(script, "7.5")
    first = sorted(script.parent.glob("sm75_kernel_*.cu"))
    assert first, "expected moe marlin kernels to be generated"
    assert (script.parent / "kernel_selector.h").exists()
    assert not list(script.parent.glob("sm80_kernel_*.cu"))
    assert not list(script.parent.glob("sm89_kernel_*.cu"))

    _run_generator(script, "7.5")
    second = sorted(script.parent.glob("sm75_kernel_*.cu"))
    assert [path.name for path in first] == [path.name for path in second]
