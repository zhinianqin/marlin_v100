# CUTLASS build baseline

## Summary

Added the external CUTLASS build hook. `CUTLASS_DIR` is a CMake cache path and
can also be supplied through the environment. If neither is set, it defaults to
`/root/source/repos/cutlass`.

CMake now validates these headers before building:

- `include/cute/tensor.hpp`
- `include/cutlass/cutlass.h`

Both extension targets receive `${CUTLASS_DIR}/include` as a private include
directory.

## PyTorch baseline

The PyTorch CUDA path for regular 2D fp16 `torch.matmul` / `torch.mm` is:

`matmul -> mm_out_cuda -> addmm_out_cuda_impl -> at::cuda::blas::gemm<Half> -> cublasGemmEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP)`

This is a behavior and performance baseline only. The SM70 Marlin rewrite will
use CUTLASS/CUTE Volta tensor-op primitives rather than copying a PyTorch kernel
body.

## Validation

Passed for this checkpoint:

```bash
./build.sh
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import marlin_v100, marlin_v100._C, marlin_v100._moe_C"
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
```

Results:

- `./build.sh`: passed. CMake configured with
  `marlin_v100 CUTLASS_DIR: /root/source/repos/cutlass`, built `_C.abi3.so`
  and `_moe_C.abi3.so`, and copied both extensions into
  `python/marlin_v100/`.
- Import check: passed.
- Pytest collection: passed, `226 tests collected`.

## ptxas notes

No new kernel path was introduced in this checkpoint, so the ptxas profile still
reflects the inherited generated dense/MoE kernels. The build output confirms
why the next checkpoint needs a new SM70 matmul core:

- Many generated SM70 kernels use the maximum reported `255 registers`.
- Several generated MoE instances spill heavily. Representative lines observed
  during this build include `800 bytes spill stores, 750 bytes spill loads`,
  `314 bytes spill stores, 442 bytes spill loads`, and `426 bytes spill stores,
  454 bytes spill loads`.
- Smaller specializations are still present and commonly report roughly
  `128-177 registers` with no spills, but these are not the dominant high-work
  paths.

## Benchmark

Not run. No kernel implementation changed in this checkpoint.

## Known issues

- CUTLASS is only wired into the extension targets; no CUTLASS/CUTE matmul
  implementation is active yet.
- Existing generated kernels still include the old slow path and retain the
  high register/spill profile noted above.

## Next step

Add a private SM70 half GEMM probe that can instantiate the planned CTA tile,
warp count, K tile, and stage combinations, then benchmark A-global/direct
versus A-shared-memory paths while keeping B sourced from shared memory.
