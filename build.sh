#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing local virtualenv python at $ROOT_DIR/.venv/bin/python"
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$ROOT_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
export PYTHONPATH="$ROOT_DIR"

export CMAKE_ARGS="$(
  "$ROOT_DIR/.venv/bin/python" - <<'PY'
import os
import shlex


def parse_cmake_args(raw_args: str) -> list[str]:
    args = shlex.split(raw_args)
    normalized_args: list[str] = []
    for arg in args:
        if (
            normalized_args
            and not arg.startswith("-")
            and normalized_args[-1].startswith("-D")
            and "=" in normalized_args[-1]
        ):
            normalized_args[-1] = f"{normalized_args[-1]} {arg}"
            continue
        normalized_args.append(arg)
    return normalized_args


default_cuda_flags = "-gencode arch=compute_70,code=sm_70"
ptxas_flag = "-Xptxas=-v"
raw_args = os.environ.get("CMAKE_ARGS", "")

if not raw_args:
    print(shlex.join([f"-DCMAKE_CUDA_FLAGS={default_cuda_flags} {ptxas_flag}"]))
    raise SystemExit

cmake_args = parse_cmake_args(raw_args)
for index, arg in enumerate(cmake_args):
    if not arg.startswith("-DCMAKE_CUDA_FLAGS="):
        continue

    cuda_flags = arg.split("=", 1)[1]
    if ptxas_flag not in cuda_flags:
        cuda_flags = f"{cuda_flags} {ptxas_flag}".strip()
        cmake_args[index] = f"-DCMAKE_CUDA_FLAGS={cuda_flags}"
    print(shlex.join(cmake_args))
    raise SystemExit

cmake_args.append(f"-DCMAKE_CUDA_FLAGS={ptxas_flag}")
print(shlex.join(cmake_args))
PY
)"

echo "Build root: $ROOT_DIR"
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "CMAKE_ARGS: $CMAKE_ARGS"
echo

set +e
"$ROOT_DIR/.venv/bin/python" setup.py build_ext --inplace 2>&1 | \
  sed -E 's/^(.*([1-9][0-9]* bytes (stack frame|spill stores|spill loads)).*)$/\x1b[31m\1\x1b[0m/g'
build_status=${PIPESTATUS[0]}
set -e

exit "$build_status"
