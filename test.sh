#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing local virtualenv python at $ROOT_DIR/.venv/bin/python"
  exit 1
fi

if [[ ! -x "$ROOT_DIR/.venv/bin/pytest" ]]; then
  echo "Missing local virtualenv pytest at $ROOT_DIR/.venv/bin/pytest"
  exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$ROOT_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
export PYTHONPATH="$ROOT_DIR/python:$ROOT_DIR"

if [[ ! -f "$ROOT_DIR/python/marlin_v100/_C.abi3.so" || ! -f "$ROOT_DIR/python/marlin_v100/_moe_C.abi3.so" ]]; then
  echo "Extensions not found. Building first..."
  "$ROOT_DIR/build.sh"
fi

if [[ "$#" -eq 0 ]]; then
  PYTEST_ARGS=(-q)
else
  PYTEST_ARGS=("$@")
fi

echo "Test root: $ROOT_DIR"
echo "CUDA_HOME: $CUDA_HOME"
echo "TORCH_CUDA_ARCH_LIST: $TORCH_CUDA_ARCH_LIST"
echo "Pytest: $ROOT_DIR/.venv/bin/pytest"
echo "Args: ${PYTEST_ARGS[*]}"
echo

"$ROOT_DIR/.venv/bin/pytest" "${PYTEST_ARGS[@]}"
