from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Local model directory for marlin_gemm_shapes table-driven tests.",
    )


@pytest.fixture(scope="session")
def model_dir(request: pytest.FixtureRequest) -> Path:
    model = request.config.getoption("--model")
    if not model:
        pytest.skip("pass --model <model_dir> to run model-shape table tests")

    path = Path(str(model)).expanduser()
    if not path.exists():
        pytest.fail(f"--model path does not exist: {path}", pytrace=False)
    if not path.is_dir():
        pytest.fail(
            f"--model must be a model directory, not a file: {path}",
            pytrace=False,
        )
    if not (path / "config.json").exists():
        pytest.fail(
            "--model must point to a model directory containing config.json: "
            f"{path}",
            pytrace=False,
        )
    return path.resolve()
