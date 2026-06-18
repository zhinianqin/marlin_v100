# 2026-05-11 Pure B SMEM MMA Probe

## 改动内容

- 删除 probe 中非纯 B 的实验分支：不再保留 B crosswise 代理路径，也不再保留显式 B 转换路径。
- `sm70_cutlass_matmul_probe` 仍保留 `b_path` 参数以维持 op schema 不变，但当前只接受 `b_path=0`。
- `a_path=2` 继续使用从 CUTLASS SM70 threadblock GEMM 中拆出的组件：`DefaultMma`、Volta TensorOp `8x8x4`、2-stage pipeline、fp32 accumulate、fp16 output。
- B 输入回到纯 row-major global tensor，由 CUTLASS iterator 写入 `RowMajorVoltaTensorOpMultiplicandBCongruous<16>` 对应的 shared-memory layout，再进入 MMA。
- benchmark 只枚举 `cutlass_shared` B 路径；pytest 增加了非纯 B 路径拒绝测试。

## 决策原因

- 显式 B 转换方向会增加寄存器压力，或让 packed/dequant B 很难保持合并访存。
- Marlin 的真实 B 是量化权重，最终必须在 shared memory 中形成 MMA operand；dense fp16 B 的额外转换 proxy 不能代表真实 dequant-to-shared 流水。
- 后续优化应把重点放在：packed B 合并读取、寄存器中流式 dequant、短生命周期写入 predefined Volta B-congruous SMEM layout，以及必要时在 shared-memory 读写侧做布局/Swizzle 验证。

## 构建与导入

```bash
./build.sh
```

结果：通过，`python/marlin_v100/_C.abi3.so` 和 `python/marlin_v100/_moe_C.abi3.so` 已生成。

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -c "import marlin_v100, marlin_v100._C, marlin_v100._moe_C; print('imports ok')"
```

结果：通过，输出 `imports ok`。

## 测试结果

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest -q \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_matches_torch_mm \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_threadblock_path_matches_torch_mm \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_rejects_direct_a_path \
  tests/test_marlin_dense.py::test_sm70_cutlass_matmul_probe_rejects_non_pure_b_path
```

结果：通过，`4 passed in 1.46s`。

```bash
PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q
```

结果：通过，收集 `230` 个测试。

```bash
./test.sh
```

结果：当前阶段未通过，`52 passed, 178 failed in 12.73s`。失败集中在旧 dense/MoE repack、量化 GEMM、MoE kernel 验证面，典型首个失败为 `assert_repack_layout_matches_reference` 中 `torch.equal(actual, expected)` 不成立。该结果与本次 pure B matmul probe 的目标验证分开记录；正式 dense/MoE kernel 仍待后续重构接入。

## Benchmark 结果

中等规模：

```bash
BENCH_PRESET=quick MATMUL_ARGS="--m 1024 --n 4096 --k 512 --warmup-iters 5 --iters 30 --a-paths cutlass_threadblock --b-paths cutlass_shared --cta-m 128 --cta-n 128 256 --cta-k 32 --warps 4 8" ./benchmark.sh matmul
```

| CTA | warps | status | median us | TFLOPs | V100 peak | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `128x128x32` | 4 | ok | `94.21` | `45.59` | `36.47%` | `max_abs=0.0000` |
| `128x128x32` | 8 | unsupported | - | - | - | 当前未实例化 |
| `128x256x32` | 4 | unsupported | - | - | - | 当前未实例化 |
| `128x256x32` | 8 | ok | `96.26` | `44.62` | `35.70%` | `max_abs=0.0000` |

饱和规模：

```bash
BENCH_PRESET=quick MATMUL_ARGS="--m 5120 --n 4096 --k 4096 --warmup-iters 3 --iters 20 --a-paths cutlass_threadblock --b-paths cutlass_shared --cta-m 128 --cta-n 256 --cta-k 32 --warps 8 --atol 0.5 --rtol 0.05" ./benchmark.sh matmul
```

| CTA | warps | status | median us | TFLOPs | V100 peak | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `128x256x32` | 8 | ok | `1850.37` | `92.85` | `74.28%` | `max_abs=0.2500` |

结论：纯 B shared-memory 路径在饱和尺寸继续超过 `90 TFLOPs`，可作为下一步 dequant-to-shared prototype 的基线。

## ptxas / Resource Usage

`./build.sh` 的 ptxas 输出，以及下面命令的资源信息一致：

```bash
/usr/local/cuda-12.8/bin/cuobjdump --dump-resource-usage python/marlin_v100/_C.abi3.so | c++filt | rg -A1 "sm70_cutlass_threadblock_gemm_kernel"
```

| kernel config | registers | stack | spill stores | spill loads |
| --- | --- | --- | --- | --- |
| `128x256x32/8` | `216` | `0` | `0` | `0` |
| `128x128x32/4` | `236` | `0` | `0` | `0` |
| `128x64x32/4` | `154` | `0` | `0` | `0` |
| `64x256x32/8` | `140` | `0` | `0` | `0` |
| `64x128x32/4` | `162` | `0` | `0` | `0` |
| `64x64x32/4` | `104` | `0` | `0` | `0` |

## 下一步

- 把 `128x256x32/8` pure B path 抽成 dense/MoE 可复用的 SM70 GEMM core。
- 第一版 dequant-to-shared prototype 只做 packed B global load -> register unpack/dequant -> B-congruous SMEM store -> MMA，避免任何额外 dense B 转换 proxy。
- 优先实现 `kU4B8` 或 `kU8B128` 的单格式原型，记录寄存器、spill、median latency，再扩展到 `kU4`。
- 每次引入 dequant、scale、zero-point 或 MoE routing 后都先跑对应 pytest 与 `BENCH_PRESET=quick ./benchmark.sh dense|moe`，如果相对 `92.85 TFLOPs` matmul probe 出现异常断崖，先定位寄存器生命周期和 shared-memory 布局。
