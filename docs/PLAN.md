# marlin_v100 基于 CUTLASS 的 SM70 重构方案

## 摘要

- 从 `a7dd5607f5c6506ccb9d96d01654b5fd5fee19a2` 创建新分支，例如 `refactor/cutlass-sm70-marlin-v100`，因为该 commit 能编译且接口未被破坏，适合作为新实现基础。
- 第一笔提交从 `a79b60a69f701f293af7251a5749465ed1deb22b` 恢复当前确认准确的 `benchmarks/`、`tests/`，以及这些测试依赖的本地 Python 辅助代码。
- CUTLASS 作为外部依赖接入，不复制进仓库；CMake 使用 `CUTLASS_DIR`，默认 `/root/source/repos/cutlass`。
- 保持 `marlin_gemm` 和 `moe_wna16_marlin_gemm` 的 Torch op schema、Python 调用参数和现有接口不变，便于直接替换。
- 项目只面向 Tesla V100 / SM70，不再保留非 SM70 的实现路径。
- 最终正式支持 `kU4`、`kU4B8`、`kU8B128`，并且 dense 与 MoE 都支持；其他量化格式保留接口检查，但删除实现代码并加 TODO。

## 实现计划

- 分支与提交流程：
  - 执行分支创建：`git switch -c refactor/cutlass-sm70-marlin-v100 a7dd5607f5c6506ccb9d96d01654b5fd5fee19a2`。
  - 第一笔提交只恢复准确测试和 benchmark，不开始重写 kernel。
  - 每次取得确定成果前，在 `docs/` 下新增一个 Markdown 记录：改动内容、测试命令、benchmark 结果、ptxas 寄存器/溢出信息、已知问题、下一步。
  - 每次提交保持小步明确，不提交 `.venv/`、`build/`、`.so`、生成的 kernel 文件或缓存。

- PyTorch 与 CUTLASS 基线：
  - 记录 PyTorch 中 `torch.matmul/mm` 的真实 CUDA 路径：`matmul -> mm_out_cuda -> addmm_out_cuda_impl -> at::cuda::blas::gemm<Half> -> cublasGemmEx(..., CUBLAS_GEMM_DEFAULT_TENSOR_OP)`。
  - 该路径作为行为与性能基线，不直接搬运为 kernel 主体。
  - 在 CMake 中校验 `CUTLASS_DIR/include/cute/tensor.hpp` 和 `CUTLASS_DIR/include/cutlass/cutlass.h` 是否存在，缺失时给出明确错误。
  - CUTLASS include 只接入本地扩展目标，仓库保持独立可构建。

- SM70 matmul 框架：
  - 先实现私有的 SM70 half GEMM 探针/基准 kernel，使用 CUTLASS/CUTE 的 Volta tensor-op 组件，例如 `SM70_8x8x4_F32F16F16F32_*` 和 Volta shared-memory layout。
  - 框架必须支持编译期配置：CTA tile、warp tile、K tile、warp 数量、pipeline stage。
  - 初始参数矩阵：
    - CTA M：`8, 16, 32, 64`
    - CTA N：`32, 64, 128`
    - CTA K：`64, 128`
    - warp 数量：`4, 8`
    - stage：`2`
  - 对 A 输入测试两条路径：
    - 从显存直接加载到寄存器，再做寄存器布局修正。
    - 先加载到 shared memory，通过 shared memory 调整布局，再从 shared memory 读入 MMA。
  - 对 A shared-memory 布局调整测试多种方式：写入 shared memory 时计算地址调整、`__shfl_sync` 辅助调整、`cute::copy` 调整。
  - 对 B 输入固定以 shared memory 作为 MMA 来源，因为后续 Marlin kernel 的 B 来自 dequant 结果。
  - 已放弃在 CUTLASS `TransformB` 插入 dequant：该路径会增加寄存器压力，且难以同时保证后续 packed/dequant 数据的合并访存。后续应维持纯正 CUTLASS row-major B 主路径，把 B 的布局/转置处理约束在 shared-memory 写入或读取侧验证。
  - 以正确性、median latency、TFLOPs、寄存器数量、spill stores/loads 选择最终技术路径。

- dense `marlin_gemm`：
  - 用选出的 SM70 matmul 主循环替换当前慢速/占位实现。
  - 保持 Marlin 的循环方向：最外层循环 B/N tile，内层循环 A/M tile。
  - B 权重的 unpack / dequant / scale / zp 不走 CUTLASS `TransformB`。后续实现应先以最短寄存器生命周期写入 shared memory，再围绕 shared-memory 读写布局做转置/Swizzle 验证。
  - 实现三类格式：
    - `kU4`：支持整数 zero-point。
    - `kU4B8`：支持 bias-8 语义。
    - `kU8B128`：支持 bias-128 语义。
  - A 只支持 fp16，C 只支持 fp16，scale 只支持 fp16。
  - bf16、fp8、nvfp4、mxfp4、float zero-point、global scale 等路径保留接口检查并明确报错，代码处加 TODO。
  - act-order 继续显式拒绝，除非后续单独 benchmark 证明可高性能支持。
  - kernel generator 只生成 SM70 与上述三种量化格式的实例。

- MoE `moe_wna16_marlin_gemm`：
  - 复用 dense 的 SM70 matmul/dequant 核心，只在外层增加 expert/token routing、`top_k`、`mul_topk_weights`、workspace/lock 处理。
  - 保持现有 MoE 参数不变，包括 `moe_block_size`、`top_k`、`thread_k`、`thread_n`、`blocks_per_sm`。
  - 保持支持的 `moe_block_size`：`8, 16, 32, 48, 64`。
  - 保留强制几何参数支持：`thread_k/thread_n=(128,64)` 和 `(128,32)`；自动选择路径根据 benchmark 选择最快配置。
  - MoE 同样支持 `kU4`、`kU4B8`、`kU8B128`。
  - float zero-point、bf16/fp8/nvfp4/mxfp4、act-order 继续明确拒绝并加 TODO。

## 公共接口与边界

- 不修改现有 Torch op schema：
  - `marlin_gemm(...) -> Tensor`
  - `moe_wna16_marlin_gemm(...) -> Tensor`
  - repack、preprocess、topk、align 等辅助 op schema 保持不变。
- 新增唯一构建接口：`CUTLASS_DIR`，默认 `/root/source/repos/cutlass`，可由环境变量或 CMake cache 覆盖。
- 本地 Python wrapper 仍只做薄封装；只允许为测试/benchmark 发现 `kU4` dense 支持而做必要更新。
- `upstream_map.yaml` 继续作为回写主树唯一依据。
- 新增 CUDA 头文件若位于 `csrc/quantization/marlin/**` 或 `csrc/moe/marlin_moe_wna16/**`，属于可回写源码范围；`docs/`、`tests/`、`benchmarks/`、本地 Python wrapper、CMake 本地配置不默认回写。

## 测试与 Benchmark

- 每次提交前最低验证：
  - `./build.sh`
  - `PYTHONPATH=$PWD/python ./.venv/bin/python -c "import marlin_v100, marlin_v100._C, marlin_v100._moe_C"`
  - `PYTHONPATH=$PWD/python ./.venv/bin/pytest --collect-only -q`
  - 针对被改子系统运行对应 pytest。
  - 在 `docs/` 记录 ptxas 输出中的 register、spill stores、spill loads。

- matmul 框架阶段：
  - 对所有候选 tile/warp 配置运行私有 matmul benchmark。
  - 与 `torch.mm` 对比正确性。
  - 记录 median latency、TFLOPs、寄存器、spill、A/B 路径选择。
  - 选出 dense 与 MoE 后续共用的主循环实现。

- dense 阶段：
  - 运行 `tests/test_marlin_dense.py`、`tests/test_marlin_helpers.py`、`tests/test_calibration.py`。
  - 补齐 dense `kU4` zero-point 测试与 benchmark。
  - 稳定节点运行 `BENCH_PRESET=quick ./benchmark.sh dense`。
  - 重要节点运行 `BENCH_PRESET=full ./benchmark.sh dense`。

- MoE 阶段：
  - 运行 `tests/test_marlin_moe.py` 以及 routing/align 相关测试。
  - 稳定节点运行 `BENCH_PRESET=quick ./benchmark.sh moe`。
  - 重要节点运行 `BENCH_PRESET=full ./benchmark.sh moe`。

- 最终验收：
  - `./test.sh` 在 V100 SM70 机器上通过。
  - `BENCH_PRESET=full ./benchmark.sh all` 完成并产生记录。
  - 任一阶段相对上一最佳实现出现超过 10% 的 median latency 退化时，暂停继续叠功能，先定位原因并重构。
  - 最终文档记录相对 `torch.mm` 的性能、相对 V100 125 TFLOPS 峰值的比例、剩余瓶颈与后续 TODO。

## 已确定假设

- CUTLASS 使用外部依赖，默认路径 `/root/source/repos/cutlass`。
- `kU4` 是 dense 与 MoE 的正式支持目标，不仅限于 MoE。
- 当前唯一目标设备是 Tesla V100-SXM2-32GB，SM70。
- `a79b60a` 的 tests 和 benchmarks 是正确性与性能记录基准。
- 第一笔恢复测试的提交可以包含测试依赖的本地 Python 辅助代码，因为这些属于本工作区验证链，不属于默认上游回写范围。
- 不支持的量化格式不做隐藏 fallback；保留接口识别、显式报错、TODO 注释。
