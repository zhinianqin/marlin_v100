# 2026-06-03 Python 回写全量测试验证记录

## 摘要

本文档记录 `/root/source/repos/marlin_v100` 中 Marlin Python 回写 tests/benchmarks 的 streaming loop-runner 重构和完整验证结果。

完整矩阵定义保持不变：

```text
class x quant x group x shape x split-K x CTA
```

本轮关键变化是：完整矩阵不再表示为百万级 pytest item、不再 import-time 物化 case tuple，也不再把 CSV rows 全量保存在内存中。Tests 和 benchmarks 现在都从 `tests/writeback_marlin_cases.py` 流式读取同一个共享 inventory。

最终状态：

| 方向 | 全量 selected | Supported/OK | Unsupported/SKIP | ERR | MISMATCH | 状态 |
|---|---:|---:|---:|---:|---:|---|
| Dense loop test | 3,660,800 | 86,240 | 3,574,560 | 0 | n/a | PASS |
| MoE loop test | 4,096,400 | 168,960 | 3,927,440 | 0 | n/a | PASS |
| Dense benchmark | 3,660,800 | 86,240 | 3,574,560 | 0 | 0 | PASS |
| MoE benchmark | 4,096,400 | 168,960 | 3,927,440 | 0 | 0 | PASS |

固定 benchmark CSV 产物：

| 方向 | CSV | Log |
|---|---|---|
| Dense | `benchmarks/results/20260603_dense_writeback_full_matrix_iters1_no_skip_tflops.csv` | `benchmarks/results/20260603_dense_writeback_full_matrix_iters1_no_skip_tflops.log` |
| MoE | `benchmarks/results/20260603_moe_writeback_full_matrix_iters1_no_skip_tflops.csv` | `benchmarks/results/20260603_moe_writeback_full_matrix_iters1_no_skip_tflops.log` |

固定 loop test 产物：

| 方向 | JSONL | Summary JSON | Pytest log |
|---|---|---|---|
| Dense | `benchmarks/results/20260604_test_dense_writeback_full_matrix.jsonl` | `benchmarks/results/20260604_test_dense_writeback_full_matrix_summary.json` | `benchmarks/results/20260604_test_marlin_linear_kernel_shapeid_redesign.log` |
| MoE | `benchmarks/results/20260604_test_moe_writeback_full_matrix.jsonl` | `benchmarks/results/20260604_test_moe_writeback_full_matrix_summary.json` | `benchmarks/results/20260604_test_marlin_moe_kernel_shapeid_redesign.log` |

## 环境

| 项 | 值 |
|---|---|
| cwd | `/root/source/repos/marlin_v100` |
| PYTHONPATH | `$PWD/python:$PWD` |
| Python | `./.venv/bin/python` |
| Pytest | 通过 `./test.sh` 使用 `./.venv/bin/pytest` |
| CUDA_HOME | `/usr/local/cuda-12.8` |
| Runtime GPU | `Tesla V100-SXM2-32GB` |
| Runtime capability | `sm70 (7.0)` |
| Build target | `SM70 (7.0)` |
| Matrix source | `tests/writeback_marlin_cases.py` |

## 旧 OOM 记录

旧版 MoE class-path full matrix 使用百万级 pytest item 表示完整矩阵，未能完成：

| 事实 | 值 |
|---|---|
| MoE itemized matrix size | 4,096,400 pytest cases |
| 被 kill 前大致进度 | 约 6% |
| 退出方式 | kernel OOM killer，exit code 137 |
| dmesg RSS 观察 | pytest `anon-rss` 约 523 GiB |
| GPU 显存状态 | idle，不是限制因素 |
| 根因 | pytest item metadata、skip report、物化矩阵/result 结构导致 CPU 内存膨胀 |

这不是 CUDA kernel 显存 OOM。本轮规避方式：

- `iter_dense_writeback_matrix()` 和 `iter_moe_writeback_matrix()` 是真正的 generator。
- 不再 import-time 物化 `DENSE_WRITEBACK_MATRIX_CASES` / `MOE_WRITEBACK_MATRIX_CASES`。
- Dense/MoE full matrix pytest 各使用单个 loop-runner test。
- Unsupported case 只累计 counter 和 reason counter，不调用 `pytest.skip()`，也不生成 pytest report。
- Benchmarks 流式遍历 case，并且每个 non-SKIP result 立即写入 CSV。
- Benchmarks 不保存完整 `cases` list，也不保存完整 `rows` list。
- `--omit-skip` 只影响 CSV 输出，不影响 selected-case/skip summary 的完整矩阵计数。

## Shape 设计

完整 Dense `M` / MoE `tokens` 覆盖：

```text
1, 8, 16, 24, 32, 48, 64, 1024, 2048, 4096, 5120
```

Dense shape suites：

| Suite | Templates | Count |
|---|---:|---:|
| heavy | 4 | 44 |
| alignment | 7 | 77 |
| stress | 2 | 22 |
| all | 13 | 143 |

Dense templates：

| Template | K | N | 目标 auto CTA_N 覆盖 |
|---|---:|---:|---:|
| `dense_heavy_qo_m{m}_k4096_n4096` | 4096 | 4096 | 256 |
| `dense_heavy_gqa_kv_m{m}_k4096_n1024` | 4096 | 1024 | 256 |
| `dense_heavy_mlp_up_m{m}_k4096_n14336` | 4096 | 14336 | 256 |
| `dense_heavy_mlp_down_m{m}_k14336_n4096` | 14336 | 4096 | 256 |
| `dense_align_cta64_narrow_m{m}_k768_n192` | 768 | 192 | 64 |
| `dense_align_cta64_partial_n_m{m}_k1152_n320` | 1152 | 320 | 64 |
| `dense_align_cta64_residue_m{m}_k1792_n832` | 1792 | 832 | 64 |
| `dense_align_cta128_mid_n_m{m}_k768_n384` | 768 | 384 | 128 |
| `dense_align_cta128_partial_n_m{m}_k1152_n640` | 1152 | 640 | 128 |
| `dense_align_cta128_residue_m{m}_k1792_n1152` | 1792 | 1152 | 128 |
| `dense_align_cta256_tiny_square_m{m}_k512_n256` | 512 | 256 | 256 |
| `dense_stress_cache_thrash_m{m}_k1024_n256` | 1024 | 256 | 256 |
| `dense_stress_splitk_starve_m{m}_k14336_n256` | 14336 | 256 | 256 |

Dense auto CTA_N 覆盖：

| auto CTA_N | Shape count |
|---:|---:|
| 64 | 33 |
| 128 | 33 |
| 256 | 77 |

MoE shape suites：

| Suite | Templates | Routing profiles | Count |
|---|---:|---:|---:|
| production | 5 | 2 | 110 |
| alignment | 11 | 2 | 242 |
| stress | 3 | 2 | 66 |
| all | 19 | 2 | 418 |

MoE routing profile 写入 `shape_id`，不新增矩阵维度：

| Profile | 构造方式 |
|---|---|
| `uniform` | 专家间确定性 round-robin |
| `zipfian` | 确定性热点专家倾斜，expert 0 接收较重的第一路由 share |

MoE templates：

| Template | hidden | intermediate | experts | topk | Stage CTA_N |
|---|---:|---:|---:|---:|---|
| `moe_prod_mixtral_up_m{m}_h4096_i14336_e8_topk2_route_{r}` | 4096 | 14336 | 8 | 2 | 256/256 |
| `moe_prod_mixtral_down_m{m}_h14336_i4096_e8_topk2_route_{r}` | 14336 | 4096 | 8 | 2 | 256/256 |
| `moe_prod_deepseek_tp_m{m}_h7168_i2048_e8_topk2_route_{r}` | 7168 | 2048 | 8 | 2 | 256/256 |
| `moe_prod_small_square_m{m}_h2048_i2048_e8_topk2_route_{r}` | 2048 | 2048 | 8 | 2 | 256/256 |
| `moe_prod_70b_tp_m{m}_h8192_i3584_e8_topk2_route_{r}` | 8192 | 3584 | 8 | 2 | 256/256 |
| `moe_align_cta64_tiny_m{m}_h192_i96_e8_topk2_route_{r}` | 192 | 96 | 8 | 2 | 64/64 |
| `moe_align_cta64_partial_m{m}_h320_i160_e8_topk2_route_{r}` | 320 | 160 | 8 | 2 | 64/64 |
| `moe_align_cta64_residue_m{m}_h832_i416_e8_topk2_route_{r}` | 832 | 416 | 8 | 2 | 64/64 |
| `moe_align_cta128_tiny_m{m}_h384_i192_e8_topk2_route_{r}` | 384 | 192 | 8 | 2 | 128/128 |
| `moe_align_cta128_partial_m{m}_h640_i320_e8_topk2_route_{r}` | 640 | 320 | 8 | 2 | 128/128 |
| `moe_align_cta128_residue_m{m}_h1152_i576_e8_topk2_route_{r}` | 1152 | 576 | 8 | 2 | 128/128 |
| `moe_align_k_tail_m{m}_h3584_i4096_e8_topk2_route_{r}` | 3584 | 4096 | 8 | 2 | 256/256 |
| `moe_align_irregular_i_m{m}_h4096_i5120_e8_topk2_route_{r}` | 4096 | 5120 | 8 | 2 | 256/256 |
| `moe_align_thin_gate_m{m}_h4096_i1024_e8_topk2_route_{r}` | 4096 | 1024 | 8 | 2 | 256/256 |
| `moe_align_many_experts16_m{m}_h4096_i4096_e16_topk2_route_{r}` | 4096 | 4096 | 16 | 2 | 256/256 |
| `moe_align_many_experts64_m{m}_h4096_i4096_e64_topk2_route_{r}` | 4096 | 4096 | 64 | 2 | 256/256 |
| `moe_stress_draft_decode_m{m}_h2048_i8192_e8_topk2_route_{r}` | 2048 | 8192 | 8 | 2 | 256/256 |
| `moe_stress_topk1_latency_m{m}_h4096_i14336_e8_topk1_route_{r}` | 4096 | 14336 | 8 | 1 | 256/256 |
| `moe_stress_degenerate_dense_m{m}_h4096_i4096_e1_topk1_route_{r}` | 4096 | 4096 | 1 | 1 | 256/256 |

MoE stage CTA_N 覆盖：

| Stage1/Stage2 auto CTA_N | Shape count |
|---|---:|
| 64/64 | 66 |
| 128/128 | 66 |
| 256/256 | 286 |

## Class Quant Matrix

Dense：

| Case | Class | 支持 quant | ZP flag 预期 |
|---|---|---|---|
| `marlin_linear_kernel` | `MarlinLinearKernel` | `uint4,uint8,uint4b8,uint8b128,fp8,float4_e2m1f` | `uint4,uint8` 为 true；其余为 false |
| `gptq_marlin_linear_method` | `GPTQMarlinLinearMethod` | `uint4b8,uint8b128` | false |
| `awq_marlin_linear_method` | `AWQMarlinLinearMethod` | `uint4,uint8` | true |
| `compressed_tensors_wna16` | `CompressedTensorsWNA16` | `uint4,uint8,uint4b8,uint8b128` | `uint4,uint8` 为 true；symmetric 为 false |
| `marlin_fp8_scaled_mm` | `MarlinFP8ScaledMMLinearKernel` | `fp8` | false |
| `compressed_tensors_w8a16_fp8` | `CompressedTensorsW8A16Fp8` | `fp8` | false |
| `compressed_tensors_w4a16_nvfp4` | `CompressedTensorsW4A16Fp4` | `nvfp4` | false |
| `compressed_tensors_w4a16_mxfp4` | `CompressedTensorsW4A16Mxfp4` | `mxfp4` | false |

MoE：

| Case | Class | 支持 quant | Full matrix benchmark mode |
|---|---|---|---|
| `gptq_moe` | `GPTQMarlinMoEMethod` | `uint4b8,uint8b128` | timed |
| `awq_moe` | `AWQMarlinMoEMethod` | `uint4,uint8` | timed |
| `compressed_tensors_wna16_moe` | `CompressedTensorsWNA16MarlinMoEMethod` | `uint4b8,uint8b128` | timed |
| `quark_w8a8_fp8_moe` | `QuarkW8A8Fp8MoEMethod` | `fp8` | class-path smoke；standalone oracle 下 full matrix 不支持 |
| `compressed_tensors_w8a8_fp8_moe` | `CompressedTensorsW8A8Fp8MoEMethod` | `fp8` | class-path smoke；standalone oracle 下 full matrix 不支持 |
| `compressed_tensors_w4a4_nvfp4_moe` | `CompressedTensorsW4A4Nvfp4MoEMethod` | `nvfp4` | class-path smoke；standalone oracle 下 full matrix 不支持 |
| `compressed_tensors_w4a4_mxfp4_moe` | `CompressedTensorsW4A4Mxfp4MoEMethod` | `mxfp4` | class-path smoke；standalone oracle 下 full matrix 不支持 |

## 矩阵统计

矩阵统计日志：

```text
benchmarks/results/20260604_matrix_summary_shapeid_redesign.log
```

Dense full matrix：

| Counter | Value |
|---|---:|
| total | 3,660,800 |
| supported | 86,240 |
| skipped | 3,574,560 |
| shape_count | 143 |

Dense selected-dimension counters：

| Dimension | Counts |
|---|---|
| class | 8 个 class 各 457,600 |
| quant | 8 个 quant name 各 457,600 |
| group | `-1,16,32,64,128` 各 732,160 |
| split-K | `unset,1,2,4,8` 各 732,160 |
| CTA | 16 个 CTA choice 各 228,800 |

Dense top skip reasons：

| Count | Reason |
|---:|---|
| 2,631,200 | unsupported dense writeback class/quant combination |
| 320,320 | group_size is not a supported default for this dense class |
| 57,200 | unsupported dense quant/group/shape alignment combination |
| 11,440 | direct `float4_e2m1f` scalar support is inventory-only; production FP4 paths are NVFP4/MXFP4 schemes |

MoE full matrix：

| Counter | Value |
|---|---:|
| total | 4,096,400 |
| supported | 168,960 |
| skipped | 3,927,440 |
| shape_count | 418 |

MoE selected-dimension counters：

| Dimension | Counts |
|---|---|
| class | 7 个 class 各 585,200 |
| quant | 7 个 quant name 各 585,200 |
| group | `-1,16,32,64,128` 各 819,280 |
| split-K | `unset,1,2,4,8` 各 819,280 |
| CTA | 8 个 CTA choice 各 512,050 |
| routing_profile | `uniform`: 2,048,200；`zipfian`: 2,048,200 |

MoE top skip reasons：

| Count | Reason |
|---:|---|
| 3,260,400 | unsupported MoE writeback class/quant combination |
| 334,400 | MoE modular/smoke class is covered by dedicated class-path smoke tests; standalone oracle stubs do not support full matrix execution |
| 102,960 | explicit CTA_N=128 does not match both stage1/stage2 auto CTA_N=256 |
| 100,320 | group_size is not a supported default for this MoE class |
| 47,520 | unsupported MoE quant/group/shape alignment combination |
| 34,320 | explicit CTA_N=64 does not match both stage1/stage2 auto CTA_N=256 |

## 测试结果

静态检查：

| Command | Result | Log |
|---|---|---|
| inventory、tests、benchmarks 的 `py_compile` | PASS，log 为空 | `benchmarks/results/20260604_py_compile_shapeid_redesign.log` |

Pytest collection：

| Command | Result | Log |
|---|---|---|
| `PYTHONPATH=$PWD/python:$PWD ./.venv/bin/pytest --collect-only -q` | 892 tests collected in 5.82s | `benchmarks/results/20260604_collect_shapeid_redesign.log` |

Full matrix loop tests：

| Test | Result | Matrix total | OK | SKIP | ERR | Elapsed | JSONL rows | Max c_tmp numel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense class-path loop | 58 passed, 14 warnings | 3,660,800 | 86,240 | 3,574,560 | 0 | 1428.403s | 86,240 | 73,400,320 |
| MoE class-path loop | 38 passed, 14 warnings | 4,096,400 | 168,960 | 3,927,440 | 0 | 2648.112s | 168,960 | 293,601,280 |

数值/回归测试：

| Test | Result | Log |
|---|---|---|
| `./test.sh tests/test_marlin_dense.py -q` | 475 passed in 38.47s | `benchmarks/results/20260604_test_marlin_dense_shapeid_redesign.log` |
| `./test.sh tests/test_marlin_moe.py -q` | 209 passed, 2 skipped in 58.66s | `benchmarks/results/20260604_test_marlin_moe_shapeid_redesign.log` |

Full suite sanity：

| Command | Result | Log |
|---|---|---|
| `./test.sh -q` | 890 passed, 2 skipped, 14 warnings in 4170.85s | `benchmarks/results/20260604_test_full_shapeid_redesign.log` |

Summary JSON 记录的 full loop test 覆盖：

| 方向 | Class/method 覆盖 | Quant 覆盖 | Group 覆盖 | Shape 覆盖 | Split-K 覆盖 | CTA 覆盖 |
|---|---|---|---|---:|---|---|
| Dense | 8 个 dense case 全覆盖 | `uint4,uint4b8,uint8,uint8b128,fp8,nvfp4,mxfp4` | 按支持组合覆盖 `-1,16,32,64,128` | 143 | `unset,1,2,4,8` | `auto` 加支持的 explicit CTAs |
| MoE | `gptq_moe,awq_moe,compressed_tensors_wna16_moe` full matrix；FP8/FP4 modular classes 有 smoke/path 覆盖 | `uint4,uint4b8,uint8,uint8b128` full matrix | 按支持组合覆盖 `-1,32,64,128` | 418 | `unset,1,2,4,8` | `auto` 加支持的 explicit CTAs |

MoE full loop 额外覆盖：

| Dimension | Covered values |
|---|---|
| tokens | `1,8,16,24,32,48,64,1024,2048,4096,5120` |
| routing_profile | `uniform,zipfian` |
| experts | `1,8,16,64` |
| topk | `1,2` |

## Benchmark 结果

Benchmark CSV validation log：

```text
benchmarks/results/20260604_csv_validation_no_skip_tflops.log
```

TFLOPS 计算口径：

| 方向 | FLOPS 定义 |
|---|---|
| Dense | `2 * M * K * N` |
| MoE | `6 * tokens * topk * hidden * intermediate`，只统计两次 Marlin GEMM |
| TFLOPS | `flops / (marlin_us * 1_000_000)` |

Dense benchmark：

| Metric | Value |
|---|---:|
| selected_cases | 3,660,800 |
| saved_rows | 86,240 |
| OK | 86,240 |
| summary 中计数的 SKIP | 3,574,560 |
| CSV 中的 SKIP rows | 0 |
| ERR | 0 |
| MISMATCH | 0 |
| OK rows 中缺失 tflops | 0 |
| shape_id count | 143 |

Dense benchmark OK 覆盖：

| Dimension | Counts |
|---|---|
| dense_class | `marlin_linear_kernel`: 27,720；`compressed_tensors_wna16`: 24,640；`gptq_marlin_linear_method`: 12,320；`awq_marlin_linear_method`: 12,320；`marlin_fp8_scaled_mm`: 3,080；`compressed_tensors_w8a16_fp8`: 3,080；`compressed_tensors_w4a16_nvfp4`: 1,540；`compressed_tensors_w4a16_mxfp4`: 1,540 |
| quant | `uint4`: 18,480；`uint4b8`: 18,480；`uint8`: 18,480；`uint8b128`: 18,480；`fp8`: 9,240；`nvfp4`: 1,540；`mxfp4`: 1,540 |
| group_size | `-1`: 23,100；`32`: 20,020；`64`: 18,480；`128`: 23,100；`16`: 1,540 |
| split-K | `unset,1,2,4,8` 各 17,248 |
| CTA | `auto`: 40,040；`32x256x4`: 11,760；`128x256x8`: 7,840；`64x64x4`: 5,880；`32x128x4`: 5,040；`256x64x4/256x64x8/256x128x8`: 各 3,360；`64x256x4/64x256x8`: 各 1,960；`64x128x4/64x128x8`: 各 840 |

MoE benchmark：

| Metric | Value |
|---|---:|
| selected_cases | 4,096,400 |
| saved_rows | 168,960 |
| OK | 168,960 |
| summary 中计数的 SKIP | 3,927,440 |
| CSV 中的 SKIP rows | 0 |
| ERR | 0 |
| MISMATCH | 0 |
| OK rows 中缺失 tflops | 0 |
| shape_id count | 418 |

MoE benchmark OK 覆盖：

| Dimension | Counts |
|---|---|
| method_class | `gptq_moe`: 56,320；`awq_moe`: 56,320；`compressed_tensors_wna16_moe`: 56,320 |
| quant | `uint4`: 28,160；`uint8`: 28,160；`uint4b8`: 56,320；`uint8b128`: 56,320 |
| group_size | `-1`: 46,200；`32`: 46,200；`64`: 42,240；`128`: 34,320 |
| split-K | `unset,1,2,4,8` 各 33,792 |
| routing_profile | `uniform`: 84,480；`zipfian`: 84,480 |
| CTA | `auto`: 44,220；`32x256x4/64x256x4/64x256x8`: 各 34,320；`32x128x4/64x128x4/64x128x8`: 各 5,940；`64x64x4`: 3,960 |

## 路径断言

Full matrix loop tests 走 class production path，不用 raw op 替代：

| 方向 | 覆盖行为 |
|---|---|
| Dense `process_weights_after_loading` | `MarlinLinearKernel`、GPTQ、AWQ、CompressedTensors WNA16、FP8、NVFP4、MXFP4 class paths |
| MoE `process_weights_after_loading` | GPTQ MoE、AWQ MoE、CompressedTensors WNA16 MoE full matrix paths；FP8/FP4 modular classes 有 smoke/path 覆盖 |
| U4/U8 ZP | Dense AWQ/WNA16 asymmetric 和 MoE AWQ 断言 `is_zp_float=True` |
| Non-ZP | symmetric int、FP8、NVFP4、MXFP4 断言 `is_zp_float=False` |
| Dense `c_tmp` | kernel-owned `self.c_tmp` 和 layer-owned `layer.c_tmp` 的 resize/reuse paths |
| MoE `c_tmp` | owner `layer.c_tmp` 被 stage1/stage2 records 共享，并覆盖 resize/reuse paths |
| Functional fallback `c_tmp` | dedicated tests 覆盖临时 fallback `c_tmp` 的创建和使用 |

Full loop source points：

| File | Function |
|---|---|
| `tests/writeback_marlin_cases.py` | `iter_dense_writeback_matrix`, `iter_moe_writeback_matrix`, `dense_writeback_matrix_summary`, `moe_writeback_matrix_summary` |
| `tests/test_marlin_linear_kernel.py` | `test_dense_writeback_class_full_matrix_post_load_apply_path` |
| `tests/test_marlin_moe_kernel.py` | `test_moe_writeback_method_full_matrix_post_load_apply_path` |
| `benchmarks/benchmark_marlin_dense.py` | `_iter_filtered_matrix`, `main` streaming CSV writer |
| `benchmarks/benchmark_marlin_moe.py` | `_iter_filtered_matrix`, `main` streaming CSV writer |

## 复现命令

所有命令都从 repo 根目录执行：

```bash
cd /root/source/repos/marlin_v100
export PYTHONPATH=$PWD/python:$PWD
```

静态检查：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/python -m py_compile \
  tests/writeback_marlin_cases.py \
  tests/test_marlin_linear_kernel.py \
  tests/test_marlin_moe_kernel.py \
  benchmarks/benchmark_marlin_dense.py \
  benchmarks/benchmark_marlin_moe.py \
  > benchmarks/results/20260604_py_compile_shapeid_redesign.log 2>&1
```

矩阵统计：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/python - <<'PY' \
  > benchmarks/results/20260604_matrix_summary_shapeid_redesign.log 2>&1
from collections import Counter
from tests.writeback_marlin_cases import (
    dense_writeback_matrix_summary,
    moe_writeback_matrix_summary,
    DENSE_BENCHMARK_SHAPE_CASES,
    MOE_BENCHMARK_SHAPE_CASES,
    dense_auto_cta,
    moe_auto_cta,
)

def compact(name, summary):
    print(name)
    for key in ["total", "supported", "skipped", "shape_count"]:
        print(key, summary[key])
    for key in ["class", "quant", "group_size", "split_k", "cta"]:
        print(key, dict(sorted(summary[key].items())))
    if "routing_profile" in summary:
        print("routing_profile", dict(sorted(summary["routing_profile"].items())))
    print("top_skip_reasons")
    for reason, count in sorted(summary["skip_reasons"].items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(count, reason)
    print()

dense = dense_writeback_matrix_summary()
moe = moe_writeback_matrix_summary()
compact("dense", dense)
compact("moe", moe)

print("dense M", sorted({s.size_m for s in DENSE_BENCHMARK_SHAPE_CASES}))
print("dense shape_count", len({s.name for s in DENSE_BENCHMARK_SHAPE_CASES}))
print("dense auto CTA_N", dict(sorted(Counter(
    dense_auto_cta(s.size_m, s.size_n).cta_n
    for s in DENSE_BENCHMARK_SHAPE_CASES
).items())))
print("moe tokens", sorted({s.tokens for s in MOE_BENCHMARK_SHAPE_CASES}))
print("moe shape_count", len({s.name for s in MOE_BENCHMARK_SHAPE_CASES}))
print("moe routing", dict(sorted(Counter(
    s.routing_profile for s in MOE_BENCHMARK_SHAPE_CASES
).items())))
print("moe stage CTA_N", dict(sorted(Counter(
    (moe_auto_cta(2 * s.intermediate).cta_n, moe_auto_cta(s.hidden).cta_n)
    for s in MOE_BENCHMARK_SHAPE_CASES
).items())))
PY
```

Collection：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/pytest --collect-only -q \
  > benchmarks/results/20260604_collect_shapeid_redesign.log 2>&1
```

Full matrix loop tests：

```bash
./test.sh tests/test_marlin_linear_kernel.py -q \
  > benchmarks/results/20260604_test_marlin_linear_kernel_shapeid_redesign.log 2>&1

./test.sh tests/test_marlin_moe_kernel.py -q \
  > benchmarks/results/20260604_test_marlin_moe_kernel_shapeid_redesign.log 2>&1
```

数值/回归测试：

```bash
./test.sh tests/test_marlin_dense.py -q \
  > benchmarks/results/20260604_test_marlin_dense_shapeid_redesign.log 2>&1

./test.sh tests/test_marlin_moe.py -q \
  > benchmarks/results/20260604_test_marlin_moe_shapeid_redesign.log 2>&1
```

Full suite：

```bash
./test.sh -q \
  > benchmarks/results/20260604_test_full_shapeid_redesign.log 2>&1
```

Dense benchmark：

```bash
BENCH_PRESET=full \
DENSE_ARGS='--shape-suite all --warmup-iters 0 --iters 1 --omit-skip --csv benchmarks/results/20260603_dense_writeback_full_matrix_iters1_no_skip_tflops.csv' \
./benchmark.sh dense \
  > benchmarks/results/20260603_dense_writeback_full_matrix_iters1_no_skip_tflops.log 2>&1
```

MoE benchmark：

```bash
BENCH_PRESET=full \
MOE_ARGS='--shape-suite all --warmup-iters 0 --iters 1 --omit-skip --csv benchmarks/results/20260603_moe_writeback_full_matrix_iters1_no_skip_tflops.csv' \
./benchmark.sh moe \
  > benchmarks/results/20260603_moe_writeback_full_matrix_iters1_no_skip_tflops.log 2>&1
```

CSV validation：

```bash
PYTHONPATH=$PWD/python:$PWD ./.venv/bin/python - <<'PY' \
  > benchmarks/results/20260604_csv_validation_no_skip_tflops.log 2>&1
import csv
from collections import Counter

checks = [
    ("benchmarks/results/20260603_dense_writeback_full_matrix_iters1_no_skip_tflops.csv", "dense_class"),
    ("benchmarks/results/20260603_moe_writeback_full_matrix_iters1_no_skip_tflops.csv", "method_class"),
]

for path, class_key in checks:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    print(path)
    print("rows", len(rows))
    for key in [class_key, "status", "quant", "group_size", "shape_id", "split_k", "cta"]:
        counts = Counter(r[key] for r in rows)
        if key == "shape_id":
            print(key, "count", len(counts))
        else:
            print(key, dict(sorted(counts.items())))
    if rows and "routing_profile" in rows[0]:
        print("routing_profile", dict(sorted(Counter(r["routing_profile"] for r in rows).items())))
    print("skip_rows_in_csv", sum(r["status"] == "SKIP" for r in rows))
    print("errors", sum(r["status"] == "ERR" for r in rows))
    print("mismatch", sum(r["status"] == "MISMATCH" for r in rows))
    print("missing_tflops", sum(
        not r.get("marlin_tflops") or r["marlin_tflops"] == "n/a"
        for r in rows
        if r["status"] == "OK"
    ))
    print()
PY
```

## 阻塞状态

本轮验证没有剩余 blocker。

| 方向 | First failure | Blocking |
|---|---|---|
| Dense loop test | none | no |
| MoE loop test | none | no |
| Dense benchmark | none | no |
| MoE benchmark | none | no |

