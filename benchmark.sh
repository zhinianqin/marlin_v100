#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-all}"
BENCH_PRESET="${BENCH_PRESET:-full}"
RESULTS_DIR="$ROOT_DIR/benchmarks/results"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$RESULTS_DIR/${TIMESTAMP}_${TARGET}_${BENCH_PRESET}.log"

case "$TARGET" in
  matmul|dense|moe|all)
    ;;
  *)
    echo "Usage: ./benchmark.sh [matmul|dense|moe|all]"
    exit 1
    ;;
esac

if [[ ! -x "$ROOT_DIR/.venv/bin/python" ]]; then
  echo "Missing local virtualenv at $ROOT_DIR/.venv"
  exit 1
fi

mkdir -p "$RESULTS_DIR"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$ROOT_DIR/.venv/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS="${MAX_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
export CMAKE_ARGS="${CMAKE_ARGS:--DCMAKE_CUDA_FLAGS=-gencode arch=compute_70,code=sm_70}"
export PYTHONPATH="$ROOT_DIR/python"

read -r -a DENSE_EXTRA_ARGS <<<"${DENSE_ARGS:-}"
read -r -a MOE_EXTRA_ARGS <<<"${MOE_ARGS:-}"
read -r -a MATMUL_EXTRA_ARGS <<<"${MATMUL_ARGS:-}"

if [[ ! -f "$ROOT_DIR/python/marlin_v100/_C.abi3.so" || ! -f "$ROOT_DIR/python/marlin_v100/_moe_C.abi3.so" ]]; then
  echo "Extensions not found. Building first..."
  "$ROOT_DIR/.venv/bin/python" setup.py build_ext --inplace
fi

exec > >(tee "$LOG_FILE") 2>&1

echo "Benchmark root: $ROOT_DIR"
echo "Target: $TARGET"
echo "Preset: $BENCH_PRESET"
echo "Log: $LOG_FILE"
echo

run_matmul() {
  "$ROOT_DIR/.venv/bin/python" benchmarks/benchmark_sm70_matmul_probe.py \
    --preset "$BENCH_PRESET" \
    "${MATMUL_EXTRA_ARGS[@]}"
}

run_dense() {
  "$ROOT_DIR/.venv/bin/python" benchmarks/benchmark_marlin_dense.py \
    --preset "$BENCH_PRESET" \
    "${DENSE_EXTRA_ARGS[@]}"
}

run_moe() {
  "$ROOT_DIR/.venv/bin/python" benchmarks/benchmark_marlin_moe.py \
    --preset "$BENCH_PRESET" \
    "${MOE_EXTRA_ARGS[@]}"
}

case "$TARGET" in
  matmul)
    run_matmul
    ;;
  dense)
    run_dense
    ;;
  moe)
    run_moe
    ;;
  all)
    run_matmul
    echo
    run_dense
    echo
    run_moe
    ;;
esac
