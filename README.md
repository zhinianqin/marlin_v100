# marlin_v100

## 项目简介

`marlin_v100` 是从 vLLM 主树中抽出的最小 Marlin 开发工作区，用于独立开发、整理和验证 SM70 / V100 上的 Marlin dense 与 Marlin MoE CUDA 路径。

这个仓库不是完整 vLLM 副本。它只保留 Marlin 相关 CUDA/C++ 源码、最小上游式 `vllm` Python package、本地测试与 benchmark 辅助代码，以及回写主树所需的 `upstream_map.yaml`。

当前项目按“仓库外可用”维护：即使脱离父级 vLLM 目录单独存在，也应该能独立构建、独立导入扩展、独立执行 pytest 收集。

扩展加载路径与上游对齐：

- `vllm._C` -> `vllm/_C.abi3.so`
- `vllm._moe_C` -> `vllm/_moe_C.abi3.so`

加载扩展后，torch op namespace 为：

- `torch.ops._C`
- `torch.ops._moe_C`

## 当前能力

- SM70 Marlin dense 和 MoE kernel 源码位于 `csrc/`。
- 当前几何策略使用 7-field CTA geometry：
  `CTA_M x CTA_N x CTA_K x Warps x WarpM x WarpN x WarpK`。
- dense 支持 `CTA_M={32,64,128,256}`；MoE 支持 `CTA_M={32,64}`。
- dense / MoE 都支持 debug-only env override：
  - `SM70_MARLIN_DENSE_CTA_GEOMETRY`
  - `SM70_MARLIN_DENSE_SPLIT_K`
  - `SM70_MARLIN_DENSE_METADATA_CACHE`
  - `SM70_MARLIN_MOE_CTA_GEOMETRY`
  - `SM70_MARLIN_MOE_SPLIT_K`
  - `SM70_MARLIN_MOE_METADATA_CACHE`
- metadata cache env 取值为 `vector_words` 或 `lane_vectors`。
- repack layout 的 `PackedMacroN` 与 GEMM launch 的实际 `CTA_N` 已分离；repack 继续按 `size_n` 自动选择 packed macro-N。

当前 geometry / env / PackedMacroN 规则以 `docs/20260612_058_sm70_marlin_ctak_warp_shape_policy.md` 为准。

## 目录说明

- `csrc/`
  Marlin dense、Marlin MoE 和最小 binding 所需的 CUDA/C++ 源码。
- `vllm/`
  最小上游式 Python package 与扩展落位目录，负责通过 `vllm._C` / `vllm._moe_C` 加载本地扩展。
- `tests/`
  本地独立测试，包括 direct-op、wrapper、writeback matrix 和 SM70 env sweep 辅助代码。
- `benchmarks/`
  本地 benchmark、shape 枚举和分析脚本。
- `docs/`
  当前策略文档与历史实验记录。日期型实验文档保留历史语义，不一定代表当前实现。
- `cmake/`
  本地 CMake 辅助宏，不依赖父级 vLLM 仓库。
- `upstream_map.yaml`
  回写主树时使用的文件映射，是回写范围的唯一依据。

## 环境与依赖

推荐使用本目录下的虚拟环境工具：

- `./.venv/bin/python`
- `./.venv/bin/pytest`
- `./.venv/bin/cmake`
- `./.venv/bin/ninja`

推荐先使用 `uv` 创建 Python 3.12 虚拟环境，并安装最小构建依赖：

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install "cmake>=3.26.1" ninja "packaging>=24.2" \
  "setuptools>=77.0.3,<81.0.0" wheel jinja2 pytest numpy

uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

当前默认 CUDA 工具链来自 `/usr/local/cuda-12.8`。Linux 下必须使用 `LD_LIBRARY_PATH`；不要把 `D_LIBRARY_PATH` 当成可生效的替代变量。

## 构建方法

推荐直接使用仓库脚本构建。脚本默认设置 SM70 / V100 架构，并打开 ptxas verbose 输出：

```bash
./build.sh
```

等价的关键环境变量是：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=8
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST='7.0'
export CMAKE_ARGS='-DCMAKE_CUDA_FLAGS=-gencode arch=compute_70,code=sm_70 -Xptxas=-v'
PYTHONPATH=$PWD ./.venv/bin/python setup.py build_ext --inplace
```

构建成功后，产物会落在：

- `vllm/_C.abi3.so`
- `vllm/_moe_C.abi3.so`

## 验证方法

最小导入与 op 注册检查：

```bash
PYTHONPATH=$PWD ./.venv/bin/python - <<'PY'
import torch
import vllm._C
import vllm._moe_C

assert hasattr(torch.ops._C, "marlin_gemm")
assert hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm")
print(vllm._C.__file__)
print(vllm._moe_C.__file__)
print("imports ok")
PY
```

常用测试分层：

- 快速结构检查：
  `PYTHONPATH=$PWD ./.venv/bin/pytest --collect-only -q`
- 普通本地测试：
  `PYTHONPATH=$PWD ./.venv/bin/pytest -q tests`
- SM70 env sweep smoke：
  设置 `MARLIN_EXHAUSTIVE_ENV_SWEEP=1` 和 `MARLIN_EXHAUSTIVE_ENV_LIMIT=<N>` 后运行相关 marked tests。
- full env sweep：
  只在明确需要完整覆盖时执行；这类测试会很慢。

当前项目目标就是 SM70 Marlin。当前 SM70 机器可用于构建、导入、op 注册和必要的 SM70 runtime smoke；大型 pytest / exhaustive sweep 不是每次文档或小改动的默认硬门槛。

## 与主树同步方式

`marlin_v100` 是本地独立开发工作区，不等于主树上游包结构。

回写主树时只依据 `upstream_map.yaml` 中列出的路径执行。默认不回写：

- 本地文档
- 本地测试
- benchmark artifact
- 本地构建辅助文件

回写前建议先做 dry-run diff，确认仅覆盖映射内文件。`upstream_map.yaml` 中的目标路径按“相对于上游 vLLM 仓库根目录”理解，不绑定某台机器的绝对路径。

## Git 与产物约定

不要把下面内容纳入版本管理：

- `.venv/`
- `build/`
- `vllm/_C*.so`
- `vllm/_moe_C*.so`
- `__pycache__/`
- pytest/cache 产物

建议纳入版本管理的内容包括：

- `csrc/`
- `vllm/`
- `tests/`
- `benchmarks/`
- `docs/`
- `setup.py`
- `CMakeLists.txt`
- `cmake/`
- `upstream_map.yaml`
- `README.md`
- `AGENTS.md`
- `.gitignore`
