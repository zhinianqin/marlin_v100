# Restore validation harness

## Summary

Created `refactor/cutlass-sm70-marlin-v100` from
`a7dd5607f5c6506ccb9d96d01654b5fd5fee19a2` and restored the known-good local
validation harness from `a79b60a69f701f293af7251a5749465ed1deb22b`.

Restored paths:

- `benchmarks/`
- `tests/`
- `python/marlin_v100/calibration.py`
- `python/marlin_v100/dense.py`
- `python/marlin_v100/quant_utils.py`

## Validation

Ran:

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
```

Result: `226 tests collected`.

## Benchmark

Not run. Kernel code remains the `a7dd5607` compile-only baseline at this
checkpoint.

## Notes

- Public Torch op schemas are unchanged.
- No CUTLASS integration has been added yet.
- The restored tests are expected to define the correctness target; the
  baseline kernels are not expected to satisfy the full restored test matrix.
