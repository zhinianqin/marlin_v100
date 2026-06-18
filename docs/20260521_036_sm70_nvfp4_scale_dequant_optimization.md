# SM70 Dense NVFP4 Scale Dequant 三阶段优化实验

日期：2026-05-21

## 结论摘要

本次实验以 `work/sm70-nvfp4-fp8-scales` 当前 clean HEAD `57f57d297f22d8c2c8222abcbae4eb81caba46e8` 作为 baseline。该 baseline 已经是 SM70 dense NVFP4 `b_scales=torch.float8_e4m3fn`、raw fp32 `global_scale` 的 raw FP8 actual-value 语义。

三阶段结果：

- 阶段 A：用逐 byte integer helper 直接生成 actual-value `half2` scales，移除了 `__hmul2(*256)` 和 low/high lane 重排。功能正确，但性能全面严重退化，`64x256x4` 出现 stack/spill。
- 阶段 B：改为 word-level bit packing，仍保持 raw FP8 actual-value 语义，避免 low/high 拆装和 `*256` 热区乘法。功能正确，资源恢复到 0 spill，但多数 CTA 仍慢于 baseline。
- 阶段 C：实验原版 Marlin fast encoding。`b_scales.dtype` 仍为 `torch.float8_e4m3fn`，但 bytes 不再是普通 raw FP8 actual-value，而是接近原版 Marlin 的 fast encoding；qweight dequant 切到 `dequant<half2, vllm::kFE2M1f.id(), true>`，scale cache 直接使用 `dequant_fp8_scales` 原生输出。功能正确，15 个 CTA 中无明确衰退，12 个改善，2 个轻微退化，1 个基本持平。

按 `kernel_like_us` 判定，阶段 C 是本次三组实验中最优的 codegen 形态。不过阶段 C 的输入 bytes 语义已经从 “raw FP8 actual-value scales” 变为 “Marlin fast encoding scales”，是否采用应由后续人工决定。本次实验只记录事实；benchmark 退化未触发自动回滚，当前阶段 C 代码保留在工作区中，未自动提交。

## 环境信息

- 分支：`work/sm70-nvfp4-fp8-scales`
- baseline commit：`57f57d297f22d8c2c8222abcbae4eb81caba46e8`
- 当前状态：同一 commit 上的未提交工作区改动
- GPU：4 × `Tesla V100-SXM2-32GB`
- Driver：`575.57.08`
- PyTorch：`2.10.0+cu128`
- CUDA：`12.8`
- nvcc：`Build cuda_12.8.r12.8/compiler.35583870_0`
- 构建目标：`TORCH_CUDA_ARCH_LIST=7.0`，`-gencode arch=compute_70,code=sm_70`

当前阶段 C 工作区修改文件：

- `csrc/quantization/marlin/sm70_marlin_nvfp4_gemm.cu`
- `tests/helpers.py`
- `tests/test_marlin_helpers.py`

## 阶段说明

### Baseline

baseline 为 raw FP8 actual-value 语义：

- `b_scales.dtype == torch.float8_e4m3fn`
- `global_scale.dtype == torch.float32`
- kernel metadata cache 阶段调用 `dequant_fp8_scales` 后做 low/high lane 重排
- kernel 中通过 `__hmul2(..., 256.0f)` 补偿 FP8 exponent bias
- qweight dequant 使用 `dequant<half2, vllm::kFE2M1f.id(), false>`

### 阶段 A

阶段 A 保持 public contract 不变，新增逐 byte integer FP8 E4M3FN 到 FP16 helper，直接向 `cached_scales_` 写入 actual-value `half2`。该阶段移除了热区 `__hmul2(*256)` 和 low/high lane 重排，但整数 helper 进入 metadata cache 后明显扩大了指令压力和寄存器生命周期。

### 阶段 B

阶段 B 仍保持 raw FP8 actual-value 语义，把阶段 A 的逐 byte helper 收敛成 word-level bit packing helper，直接生成当前 `load_macro_n_aligned()` 需要的 cache 顺序。该阶段消除了阶段 A 的 spill，但性能仍普遍低于 baseline，说明 raw actual-value 在 kernel 内恢复的 bit-level 成本仍然偏高。

### 阶段 C

阶段 C 是独立 fast encoding 实验：

- `cache_metadata_fp8_scales()` 直接缓存 `dequant_fp8_scales<half2, vllm::kFE4M3fn.id()>` 原生输出顺序
- 删除 `__low2half` / `__high2half` / `__halves2half2`
- 删除 kernel 内 `*256`
- qweight dequant 改为 `dequant<half2, vllm::kFE2M1f.id(), true>`
- Python helper 将 `b_scales` bytes 按原版 Marlin fast encoding 处理，`global_scale` 做对应补偿
- 参考 dequant 用 fast encoding decode 逻辑，不再把 bytes 当普通 raw FP8 actual-value scale

阶段 C 保持 dtype 对外仍是 `torch.float8_e4m3fn`，但 scale bytes 语义与阶段 A/B 不同。

## 验证结果

三阶段均完成构建、targeted helper 测试、targeted dense 测试和 15 个 CTA quick benchmark。

功能验证结果：

- 阶段 A：`tests/test_marlin_helpers.py -k "nvfp4"`：`5 passed`
- 阶段 A：`tests/test_marlin_dense.py -k "nvfp4 or cta_geometry or residue_n or small_tile or size_k"`：`100 passed`
- 阶段 B：同上两组测试通过
- 阶段 C：同上两组测试通过

静态检查：

- 阶段 C 当前文件中没有 `__hmul2(...256.0f)` / `__float2half2_rn(256.0f)` / `__low2half` / `__high2half` / `__halves2half2`
- 阶段 C 当前文件中没有 `dequant<half2, vllm::kFE2M1f.id(), false>`
- `git diff --check` 通过
- 旧 `preconverted nvfp4` / `_preconvert_nvfp4_scales_to_fp16` / `_preconvert_nvfp4_global_scale` 语义未回归

## Benchmark 对比

判定规则：

- `kernel_like_us` 变慢 `> 3%`：明确衰退
- `kernel_like_us` 变慢 `1% - 3%`：轻微退化
- `kernel_like_us` 在 `±1%` 内：基本持平
- `kernel_like_us` 变快 `> 1%`：改善

| CTA | baseline op/us | baseline kernel/us | baseline TF | A kernel/us | A TF | B kernel/us | B TF | C kernel/us | C TF | C kernel Δ | C TF Δ | C 判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32x128x4 | 3760.13 | 3734.02 | 46.01 | 8845.31 | 19.42 | 4214.78 | 40.76 | 3425.28 | 50.16 | -8.3% | +9.0% | 改善 |
| 32x256x4 | 3924.48 | 4003.33 | 42.91 | 14194.69 | 12.10 | 4524.03 | 37.97 | 3588.10 | 47.88 | -10.4% | +11.6% | 改善 |
| 64x64x4 | 3292.16 | 3344.90 | 51.36 | 6113.28 | 28.10 | 3511.30 | 48.93 | 3246.08 | 52.92 | -3.0% | +3.0% | 改善 |
| 64x128x4 | 2821.63 | 2809.86 | 61.14 | 8961.54 | 19.17 | 3126.27 | 54.95 | 2677.76 | 64.16 | -4.7% | +4.9% | 改善 |
| 64x128x8 | 3258.37 | 3246.08 | 52.92 | 6441.47 | 26.67 | 3370.50 | 50.97 | 3337.22 | 51.48 | +2.8% | -2.7% | 轻微退化 |
| 64x256x4 | 3062.27 | 3113.47 | 55.18 | 9538.56 | 18.01 | 3769.34 | 45.58 | 2934.27 | 58.55 | -5.8% | +6.1% | 改善 |
| 64x256x8 | 3119.10 | 3042.30 | 56.47 | 10321.41 | 16.64 | 3408.90 | 50.40 | 2948.10 | 58.27 | -3.1% | +3.2% | 改善 |
| 128x64x4 | 2720.26 | 2751.49 | 62.44 | 4973.06 | 34.55 | 2845.18 | 60.38 | 2739.20 | 62.72 | -0.4% | +0.4% | 基本持平 |
| 128x64x8 | 3101.70 | 3139.07 | 54.73 | 4904.96 | 35.03 | 3222.53 | 53.31 | 3091.46 | 55.57 | -1.5% | +1.5% | 改善 |
| 128x128x4 | 2427.90 | 2416.13 | 71.10 | 5897.73 | 29.13 | 2598.91 | 66.10 | 2314.24 | 74.24 | -4.2% | +4.4% | 改善 |
| 128x128x8 | 3315.71 | 3285.50 | 52.29 | 6868.48 | 25.01 | 3328.00 | 51.62 | 3161.60 | 54.34 | -3.8% | +3.9% | 改善 |
| 128x256x8 | 2292.74 | 2335.74 | 73.55 | 5721.09 | 30.03 | 2461.18 | 69.80 | 2143.23 | 80.16 | -8.2% | +9.0% | 改善 |
| 256x64x4 | 2378.75 | 2395.14 | 71.73 | 4692.99 | 36.61 | 2520.58 | 68.16 | 2448.38 | 70.17 | +2.2% | -2.2% | 轻微退化 |
| 256x64x8 | 3424.26 | 3487.23 | 49.27 | 5885.44 | 29.19 | 3283.97 | 52.31 | 3433.98 | 50.03 | -1.5% | +1.5% | 改善 |
| 256x128x8 | 2318.34 | 2297.34 | 74.78 | 4417.54 | 38.89 | 2324.48 | 73.91 | 2270.21 | 75.68 | -1.2% | +1.2% | 改善 |

## 分类汇总

阶段 A：

- 明确衰退：15 个 CTA 全部明确衰退
- 轻微退化：无
- 基本持平：无
- 改善：无

阶段 B：

- 明确衰退：`32x128x4`、`32x256x4`、`64x64x4`、`64x128x4`、`64x128x8`、`64x256x4`、`64x256x8`、`128x64x4`、`128x128x4`、`128x256x8`、`256x64x4`
- 轻微退化：`128x64x8`、`128x128x8`、`256x128x8`
- 基本持平：无
- 改善：`256x64x8`

阶段 C：

- 明确衰退：无
- 轻微退化：`64x128x8`、`256x64x4`
- 基本持平：`128x64x4`
- 改善：`32x128x4`、`32x256x4`、`64x64x4`、`64x128x4`、`64x256x4`、`64x256x8`、`128x64x8`、`128x128x4`、`128x128x8`、`128x256x8`、`256x64x8`、`256x128x8`

## ptxas 资源对比

格式为 `registers/stack/spill stores/spill loads`。

说明：本次 `sm70_nvfp4_scale_opt/baseline/build.log` 是增量构建日志，未包含目标 `sm70_marlin_nvfp4_gemm.cu` 的 ptxas 小节。表中的 baseline 资源使用上一轮 raw-FP8 after build 的完整 ptxas 日志补齐，路径为 `benchmarks/results/sm70_nvfp4_fp8_scales/after_build.log`；该代码状态对应本次 baseline 的 raw FP8 actual-value 实现。阶段 A/B/C 资源均来自本次实验各自目录下的 `build.log`。

| CTA | baseline | A | B | C |
|---|---:|---:|---:|---:|
| 32x128x4 | 115/0/0/0 | 127/0/0/0 | 115/0/0/0 | 113/0/0/0 |
| 32x256x4 | 177/0/0/0 | 201/0/0/0 | 176/0/0/0 | 168/0/0/0 |
| 64x64x4 | 108/0/0/0 | 122/0/0/0 | 112/0/0/0 | 110/0/0/0 |
| 64x128x4 | 163/0/0/0 | 178/0/0/0 | 162/0/0/0 | 160/0/0/0 |
| 64x128x8 | 110/0/0/0 | 109/0/0/0 | 109/0/0/0 | 96/0/0/0 |
| 64x256x4 | 255/0/0/0 | 255/72/68/88 | 255/0/0/0 | 255/0/0/0 |
| 64x256x8 | 168/0/0/0 | 173/0/0/0 | 168/0/0/0 | 164/0/0/0 |
| 128x64x4 | 157/0/0/0 | 168/0/0/0 | 160/0/0/0 | 157/0/0/0 |
| 128x64x8 | 106/0/0/0 | 110/0/0/0 | 106/0/0/0 | 108/0/0/0 |
| 128x128x4 | 241/0/0/0 | 254/0/0/0 | 242/0/0/0 | 240/0/0/0 |
| 128x128x8 | 155/0/0/0 | 157/0/0/0 | 156/0/0/0 | 152/0/0/0 |
| 128x256x8 | 245/0/0/0 | 253/0/0/0 | 249/0/0/0 | 244/0/0/0 |
| 256x64x4 | 246/0/0/0 | 254/0/0/0 | 245/0/0/0 | 246/0/0/0 |
| 256x64x8 | 154/0/0/0 | 158/0/0/0 | 154/0/0/0 | 156/0/0/0 |
| 256x128x8 | 230/0/0/0 | 237/0/0/0 | 234/0/0/0 | 227/0/0/0 |

资源观察：

- 阶段 A 的逐 byte helper 在 `64x256x4` 触发 `72 bytes stack frame, 68 bytes spill stores, 88 bytes spill loads`，与该阶段大幅退化一致。
- 阶段 B 去掉了 spill，说明 word-level bit packing 解决了阶段 A 的最坏 codegen 风险，但 raw actual-value 路径仍然慢于 baseline。
- 阶段 C 全部目标 CTA 0 spill，且多数组合寄存器数小于或等于 baseline；`64x256x4` 仍在 255 regs 边界，但无 stack/spill。

## 重点 CTA

`32x256x4`：

- baseline：`4003.33 us / 42.91 TFLOPs`
- 阶段 A：`14194.69 us / 12.10 TFLOPs`
- 阶段 B：`4524.03 us / 37.97 TFLOPs`
- 阶段 C：`3588.10 us / 47.88 TFLOPs`
- 结论：阶段 C 相对 baseline `kernel_like_us` 改善 `10.4%`，TFLOPs 提升 `11.6%`。

`64x256x4`：

- baseline：`3113.47 us / 55.18 TFLOPs`
- 阶段 A：`9538.56 us / 18.01 TFLOPs`，且出现 spill
- 阶段 B：`3769.34 us / 45.58 TFLOPs`
- 阶段 C：`2934.27 us / 58.55 TFLOPs`
- 结论：阶段 C 相对 baseline `kernel_like_us` 改善 `5.8%`，TFLOPs 提升 `6.1%`。

`128x256x8`：

- baseline：`2335.74 us / 73.55 TFLOPs`
- 阶段 A：`5721.09 us / 30.03 TFLOPs`
- 阶段 B：`2461.18 us / 69.80 TFLOPs`
- 阶段 C：`2143.23 us / 80.16 TFLOPs`
- 结论：阶段 C 相对 baseline `kernel_like_us` 改善 `8.2%`，TFLOPs 提升 `9.0%`。

## 解释与建议

阶段 A 证明“把 `*256` 从 `__hmul2` 热区移走”不能简单理解为任何 integer dequant 都更快。逐 byte 的 bit-level E4M3 到 FP16 转换给 ptxas 带来了更差的生命周期和调度，尤其在 `64x256x4` 这类本来接近 255 regs 的实例上直接触发 spill。

阶段 B 表明，把 scale cache 的 bit 操作收敛到 word-level 可以显著改善资源和性能，但 raw FP8 actual-value 语义仍要求 kernel 内恢复实际 scale 值；这部分成本在多数 CTA 中仍高于 baseline 的 `dequant_fp8_scales + lane 重排 + *256` 组合。

阶段 C 最接近原版 Marlin fast-path：scale bytes 预编码后，kernel 内不用恢复普通 raw FP8 actual-value，qweight dequant 也切回 fast encoding 对应的 `true` 路径。这个形态在 SM70 上 codegen 最好，也是本次唯一做到“无明确衰退且大多数 CTA 改善”的方案。

如果必须保持外部 bytes 为普通 raw FP8 actual-value，阶段 B 是相对更安全的实现基线，但仍有性能成本。如果可以接受 “dtype 仍为 `float8_e4m3fn`，bytes 为 Marlin fast encoding” 的内部 contract，阶段 C 是当前性能上最值得继续推进的方向。

## 日志路径

本次实验日志目录：

- baseline：`benchmarks/results/sm70_nvfp4_scale_opt/baseline/`
- 阶段 A：`benchmarks/results/sm70_nvfp4_scale_opt/stage_a/`
- 阶段 B：`benchmarks/results/sm70_nvfp4_scale_opt/stage_b/`
- 阶段 C：`benchmarks/results/sm70_nvfp4_scale_opt/stage_c/`

每个目录包含：

- `build.log`
- `resource_usage.txt`
- `nvfp4_<CTA>.log`

阶段 diff：

- `benchmarks/results/sm70_nvfp4_scale_opt/stage_a/diff.patch`
- `benchmarks/results/sm70_nvfp4_scale_opt/stage_b/diff.patch`
- `benchmarks/results/sm70_nvfp4_scale_opt/stage_c/diff.patch`

baseline 资源补充来源：

- `benchmarks/results/sm70_nvfp4_fp8_scales/after_build.log`

## 复现命令

构建命令：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH" \
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
TORCH_CUDA_ARCH_LIST=7.0 \
CMAKE_ARGS='-DCMAKE_CUDA_FLAGS=-gencode arch=compute_70,code=sm_70' \
MAX_JOBS=8 NVCC_THREADS=1 \
./build.sh
```

benchmark 命令模板：

```bash
SM70_MARLIN_NVFP4_CTA=<CTA> \
DENSE_ARGS="--models ideal --batch-sizes 5120 --quant-types nvfp4 --group-sizes 16 --act-order off --is-k-full true --report-tflops --warmup-iters 10 --iters 30" \
BENCH_PRESET=quick ./benchmark.sh dense
```

CTA 覆盖：

```text
32x128x4
32x256x4
64x64x4
64x128x4
64x128x8
64x256x4
64x256x8
128x64x4
128x64x8
128x128x4
128x128x8
128x256x8
256x64x4
256x64x8
256x128x8
```

测试命令：

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH" \
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
PYTHONPATH=$PWD/python \
./.venv/bin/pytest tests/test_marlin_helpers.py -k "nvfp4"
```

```bash
CUDA_HOME=/usr/local/cuda-12.8 \
PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH" \
LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" \
PYTHONPATH=$PWD/python \
./.venv/bin/pytest tests/test_marlin_dense.py -k "nvfp4 or cta_geometry or residue_n or small_tile or size_k"
```
