# SM70 u4 Paired Metadata Cache

## Summary

This change continues from `366f9b1 Align SM70 int4 group cache policy`.
The goal was to keep the u4/u4b8 outer IteratorB structure aligned while
recovering the u4 positive-group performance that regressed when the old
paired qzero metadata path was replaced by a generic per-contiguous loop.

Final kept shape:

- `GroupSize=-1`: constructor inline cache for scale/zero metadata.
- `GroupSize=32/64/128`: refresh metadata once in `load()`, before
  `load_full_tile()` / `load_residue_tile()`.
- No `cached_group_` runtime early return.
- No old `refresh_metadata_cache` name.
- `load_full_tile()` / `load_residue_tile()` only consume cached scale/bias.
- u4b8 functional code is unchanged in this round.

The main u4-specific recovery is restoring the paired qzero metadata cache:
full-tile positive groups load a `uint2` qzero pair for the two 64-column
contiguous halves and dispatch through `cache_metadata_word<0/1>()`. Residue
keeps the paired load when both halves are valid and falls back to scalar qzero
only for the final partial half.

## Diagnosis

Reference logs:

| Build | log | `-1` | `32` | `64` | `128` |
|---|---|---:|---:|---:|---:|
| `366f9b1` | `benchmarks/results/20260513_221831_dense_quick.log` | 78.40 | 77.46 | 77.51 | 77.51 |
| `c4085ed` + local no early return | `benchmarks/results/20260513_223338_dense_quick.log` | 77.39 | 78.51 | 78.40 | 79.19 |

The regression was not a spill problem. The relevant dense u4 kernels stayed at
`STACK=0`, `LOCAL=0`, and no spill. The performance difference came from the
B-side metadata/load shape:

- `366f9b1` fixed `GroupSize=-1` by constructor caching, but positive groups
  used a generic per-`c` qzero path with more address arithmetic.
- The faster local `c4085ed` experiment used the old paired qzero metadata load
  and removed the runtime `cached_group_ == group` early-return branch.

## Kept Implementation

`csrc/quantization/marlin/sm70_marlin_u4_gemm.cu` now uses:

- `cache_current_group_metadata(int group) const` for positive groups only.
- `cache_metadata_word<C>()` as the specialized per-contiguous cache primitive.
- Full-tile positive metadata:
  - compute the first logical N once,
  - load one `uint2 zpair`,
  - cache `C=0` and `C=1` with template-specialized calls.
- Residue metadata:
  - if both 64-column halves are valid, use the same `uint2 zpair` path,
  - otherwise cache only `C=0` from a scalar qzero word.
- `cached_scales_` and `cached_bias_ = -zero * scale` remain the only metadata
  consumed by `load_full_tile()` and `load_residue_tile()`.

One u4-specific performance exception was retained: the qweight offset model was
changed back to per-access offsets:

- `qweight_offsets_[ThreadMap::Iterations::kCount]`
- `operator++()` advances every precomputed offset by the fixed K-tile qword
  increment.
- full/residue load scalar `qweight_[qweight_offsets_[idx]]`.

This is less visually aligned with u4b8 than the `qweight_base_offset_ +
qweight_strided_offsets_[s] + c` model, but the benchmark below showed the
per-access model was materially better for u4 positive groups.

## Discarded Variant

Before restoring per-access qweight offsets, I tested:

- paired qzero metadata cache,
- `GroupSize=-1` constructor cache,
- `366f9b1` qweight model with `qweight_base_offset_`,
  `qweight_strided_offsets_[s]`, and full-tile `uint2` qweight vector load.

It built cleanly and passed the same targeted tests, but positive groups were
still below the faster reference:

| log | `-1` | `32` | `64` | `128` |
|---|---:|---:|---:|---:|
| `benchmarks/results/20260513_231318_dense_quick.log` | 78.22 | 77.49 | 76.87 | 78.11 |
| `benchmarks/results/20260513_231445_dense_quick.log` | 78.22 | 77.55 | 77.19 | 77.58 |

Conclusion: paired metadata alone fixed the structure and resource profile, but
did not fully recover positive-group scheduling. The final version keeps the
paired metadata and restores per-access qweight offsets for u4.

## Resource Usage

Command:

```bash
./build.sh
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage \
  python/marlin_v100/_C.abi3.so | c++filt | \
  rg -A8 -B2 "sm70_marlin_u4_gemm_kernel|sm70_marlin_u4b8_gemm_kernel"
```

Final dense u4b8 resources, unchanged by this round:

| Kernel | REG | STACK | LOCAL |
|---|---:|---:|---:|
| `u4b8<128,false>` | 244 | 0 | 0 |
| `u4b8<64,false>` | 244 | 0 | 0 |
| `u4b8<32,false>` | 244 | 0 | 0 |
| `u4b8<-1,false>` | 238 | 0 | 0 |
| `u4b8<128,true>` | 250 | 0 | 0 |
| `u4b8<64,true>` | 250 | 0 | 0 |
| `u4b8<32,true>` | 250 | 0 | 0 |
| `u4b8<-1,true>` | 238 | 0 | 0 |

Final dense u4 resources:

| Kernel | REG | STACK | LOCAL |
|---|---:|---:|---:|
| `u4<128,false>` | 254 | 0 | 0 |
| `u4<64,false>` | 254 | 0 | 0 |
| `u4<32,false>` | 254 | 0 | 0 |
| `u4<-1,false>` | 252 | 0 | 0 |
| `u4<128,true>` | 250 | 0 | 0 |
| `u4<64,true>` | 250 | 0 | 0 |
| `u4<32,true>` | 250 | 0 | 0 |
| `u4<-1,true>` | 244 | 0 | 0 |

All listed dense u4/u4b8 target kernels have no spill stores/loads.

## Correctness

Import:

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c \
  "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

Result:

```text
imports ok
```

Targeted pytest:

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_helpers.py::test_uint4_repack_layout_matches_reference \
  tests/test_marlin_helpers.py::test_marlin_dequantize_uint4_zp_matches_quantize_helper_output \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_8_row_bucket_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_size_m_24_matches_reference \
  tests/test_marlin_dense.py::test_marlin_dense_uint4_zp_small_tile_matches_reference \
  tests/test_marlin_helpers.py::test_repack_layout_matches_reference_for_supported_dense_quant_types \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_accuracy \
  tests/test_marlin_dense.py::test_marlin_dense_uint4b8_sm70_scale_zp_math_consistency_matches_reference
```

Result:

```text
31 passed in 3.22s
```

Alignment hygiene check:

```bash
rg -n "cached_group_|kCacheScales|refresh_metadata_cache|cache_single_group_scales" \
  csrc/quantization/marlin/sm70_marlin_u4_gemm.cu \
  csrc/quantization/marlin/sm70_marlin_u4b8_gemm.cu
```

Result: no matches.

## Benchmark

Command:

```bash
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types uint4 \
  --group-sizes -1 32 64 128 --act-order off --is-k-full true \
  --report-tflops --warmup-iters 10 --iters 30" \
  BENCH_PRESET=quick ./benchmark.sh dense
```

Final log:

```text
benchmarks/results/20260513_233854_dense_quick.log
```

Final uint4 results, `5120x4096x4096`, kernel-like TFLOPs:

| group_size | kernel_like TFLOPs |
|---:|---:|
| `-1` | 79.10 |
| `32` | 78.05 |
| `64` | 78.69 |
| `128` | 78.78 |

Comparison against the two reference logs:

| group_size | `366f9b1` | `c4085ed` + no early return | final |
|---:|---:|---:|---:|
| `-1` | 78.40 | 77.39 | 79.10 |
| `32` | 77.46 | 78.51 | 78.05 |
| `64` | 77.51 | 78.40 | 78.69 |
| `128` | 77.51 | 79.19 | 78.78 |

The final version recovers the positive-group regression relative to
`366f9b1`, stays within noise of the faster `c4085ed` positive-group reference,
and improves `GroupSize=-1` over both references in this run.

## Remaining Differences From u4b8

These differences are intentional:

- u4 has zero-point metadata and precomputes `bias = -zero * scale`.
- u4 uses `128x128x32 / 4 warp`; u4b8 uses `128x256x32 / 8 warp`.
- u4 uses scalar per-access qweight offsets in the final kept version because
  it benchmarked better for positive groups.
- u4b8 has no `b_zeros` argument and no bias cache.

The shared structure is still aligned at the policy level:

- `GroupSize=-1` metadata is constructor-cached.
- Positive groups prepare metadata once in `load()`.
- full/residue paths consume cached metadata and keep their hot loops free of
  scale/qzero direct-load policy branches.

## Conclusion

Keep this version. It is a small u4-only performance exception inside the
broader u4/u4b8 alignment effort: paired qzero metadata cache plus per-access
qweight offsets gives the best measured balance of structure, resources, and
uint4 TFLOPs so far.
