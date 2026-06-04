# marlin_v100

## 项目简介

`marlin_v100` 是从 `vllm-0.17.1` 主树中抽出的最小 Marlin 开发工作区，用于独立开发和验证 Marlin dense 与 Marlin MoE 两部分能力。

这个目录的目标不是复制完整的 vLLM，而是保留 Marlin 相关 CUDA 源码、最小上游式 `vllm` Python package、最小测试集合，以及与主树同步所需的映射信息，方便在一个更小的工作区里迭代。

当前项目按“仓库外可用”维护：将 `marlin_v100/` 单独 clone 到父级 vLLM 仓库之外后，仍应可以独立完成构建、扩展导入和测试收集。

当前扩展加载路径与上游对齐，构建产物为：

- `vllm._C` -> `vllm/_C.abi3.so`
- `vllm._moe_C` -> `vllm/_moe_C.abi3.so`

加载扩展后，底层 torch op namespace 仍为：

- `torch.ops._C`
- `torch.ops._moe_C`

## 目录说明

- `csrc/`
  Marlin dense、Marlin MoE 以及最小 binding 所需的 CUDA/C++ 源码。
- `vllm/`
  最小上游式 Python package 与扩展落位目录，负责通过 `vllm._C` / `vllm._moe_C` 加载本地扩展。
- `tests/`
  生成器测试、dense 轻量测试、MoE 轻量测试。
- `setup.py`
  本地扩展构建入口，调用 CMake + Ninja 完成构建。
- `CMakeLists.txt`
  最小 CUDA/CMake 构建定义。
- `cmake/`
  本地 CMake 辅助宏，不依赖父级 vLLM 仓库。
- `upstream_map.yaml`
  回写主树时使用的文件映射。
- `pytest.ini`
  本地 pytest 配置，确保仓库外运行时 rootdir 指向当前项目。

## 环境与依赖

当前工作区默认依赖本目录下的虚拟环境工具：

- `./.venv/bin/python`
- `./.venv/bin/pytest`
- `./.venv/bin/cmake`
- `./.venv/bin/ninja`

推荐先使用 `uv` 创建 Python 3.12 虚拟环境，并安装当前已验证过的最小构建依赖：

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install "cmake>=3.26.1" ninja "packaging>=24.2" \
  "setuptools>=77.0.3,<81.0.0" wheel jinja2 pytest numpy

uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

当前文档按下面这组依赖前提维护：

- 已安装 `torch`
- 已安装 `jinja2`
- 已安装 `pytest`
- 已安装 `numpy`
- CUDA 工具链来自 `/usr/local/cuda-12.8`

构建前推荐设置 CUDA 相关环境变量：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=8
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST='7.5'
export CMAKE_ARGS='-DCMAKE_CUDA_FLAGS=-gencode arch=compute_75,code=sm_75'
```

注意：动态库环境变量应使用 `LD_LIBRARY_PATH`。如果你手头的命令里写的是 `D_LIBRARY_PATH`，请改成 `LD_LIBRARY_PATH`。

## 构建方法

进入目录后，使用下面的命令构建：

```bash
PYTHONPATH=$PWD ./.venv/bin/python setup.py build_ext --inplace
```

构建成功后，产物会落在：

- `vllm/_C.abi3.so`
- `vllm/_moe_C.abi3.so`

可用下面的方式做最小导入检查：

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

## 测试方法

当前阶段默认只做构建与导入验收，不执行 `pytest`。

建议的验收步骤是：

- `PYTHONPATH=$PWD ./.venv/bin/python setup.py build_ext --inplace`
- `import vllm._C`
- `import vllm._moe_C`
- `hasattr(torch.ops._C, "marlin_gemm")`
- `hasattr(torch.ops._moe_C, "moe_wna16_marlin_gemm")`

## 当前限制

当前工作区已经固定为 `SM75` 单架构构建。当前没有 `SM75` 机器时，只适合作为构建链与导入链验证环境，不适合作为 Marlin 内核运行验收环境。

这意味着：

- 可以验证目录结构、构建脚本、扩展落位与导入
- `pytest` 当前阶段不作为默认验收项
- 不应把 dense / moe 的数值运行结果作为当前机器上的最终通过标准
- 真正的 Marlin 运行验证应放到 `SM75` 机器执行

## 与主树同步方式

`marlin_v100` 是本地独立开发工作区，不等于主树上游包结构。

当前默认对齐的上游代码基线为 `vllm-0.17.1`，回写和对比时应按这一版本的目录结构理解映射关系。

回写主树时应以 `upstream_map.yaml` 为准，只同步映射中明确列出的上游源码文件。下面这些内容默认不回写主树：

- 本目录的文档文件
- 本目录的测试文件
- 本目录的本地构建辅助文件

回写前建议先做一次 dry-run diff，确认仅覆盖映射内文件。

`upstream_map.yaml` 中的目标路径使用相对描述，表示“相对于上游 vLLM 仓库根目录”的目标位置，而不是某台机器上的绝对路径。

## Git 初始化建议

本目录设计为可独立纳入 git 管理。推荐初始化步骤：

```bash
cd marlin_v100
git init
git add .
git status --short
git commit -m "Initialize marlin_v100 workspace"
```

初始化后重点确认以下内容不会被纳入版本管理：

- `.venv/`
- `build/`
- `vllm/_C*.so`
- `vllm/_moe_C*.so`

建议纳入版本管理的内容包括：

- `csrc/`
- `vllm/`
- `tests/`
- `setup.py`
- `CMakeLists.txt`
- `upstream_map.yaml`
- `README.md`
- `AGENTS.md`
- `.gitignore`
