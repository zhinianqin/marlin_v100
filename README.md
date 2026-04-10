# marlin_v100

## 项目简介

`marlin_v100` 是从 `vllm-0.17.1` 主树中抽出的最小 Marlin 开发工作区，用于独立开发和验证 Marlin dense 与 Marlin MoE 两部分能力。

这个目录的目标不是复制完整的 vLLM，而是保留 Marlin 相关 CUDA 源码、最小 Python 薄封装、最小测试集合，以及与主树同步所需的映射信息，方便在一个更小的工作区里迭代。

当前项目按“仓库外可用”维护：将 `marlin_v100/` 单独 clone 到父级 vLLM 仓库之外后，仍应可以独立完成构建、扩展导入和测试收集。

当前包名为 `marlin_v100`，构建产物为：

- `marlin_v100._C`
- `marlin_v100._moe_C`

## 目录说明

- `csrc/`
  Marlin dense、Marlin MoE 以及最小 binding 所需的 CUDA/C++ 源码。
- `python/marlin_v100/`
  最小 Python 薄封装，负责加载扩展、封装 dense 与 MoE 调用。
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
export TORCH_CUDA_ARCH_LIST='7.0'
export CMAKE_ARGS='-DCMAKE_CUDA_FLAGS=-gencode arch=compute_70,code=sm_70'
```

注意：动态库环境变量应使用 `LD_LIBRARY_PATH`。如果你手头的命令里写的是 `D_LIBRARY_PATH`，请改成 `LD_LIBRARY_PATH`。

## 构建方法

进入目录后，使用下面的命令构建：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python setup.py build_ext --inplace
```

构建成功后，产物会落在：

- `python/marlin_v100/_C.abi3.so`
- `python/marlin_v100/_moe_C.abi3.so`

可用下面的方式做最小导入检查：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python - <<'PY'
import marlin_v100
import marlin_v100._C
import marlin_v100._moe_C
print(marlin_v100.__file__)
print(marlin_v100._C.__file__)
print(marlin_v100._moe_C.__file__)
PY
```

## 测试方法

当前阶段默认只做构建与导入验收，不执行 `pytest`。

建议的验收步骤是：

- `PYTHONPATH=$PWD/python ./.venv/bin/python setup.py build_ext --inplace`
- `import marlin_v100`
- `import marlin_v100._C`
- `import marlin_v100._moe_C`

## 当前限制

当前仓库里提交的 Marlin dense / MoE kernel 生成与 benchmark 路径是 `SM70` 单架构前提。当前 benchmark 脚本在运行时也会显式检查 `SM70`。

这意味着：

- 可以在当前 `SM70` 机器上验证目录结构、构建脚本、扩展落位、导入与当前实现的实际运行特征
- `pytest` 当前阶段不作为默认验收项
- 当前 `SM70` benchmark 结果只代表这份 checked-in build 的现状，不应直接外推成未来 `SM75` 目标实现的最终性能结论
- 如果后续恢复 `SM75` 目标构建，应在 `SM75` 机器上用同一套 benchmark 脚本重新测量

当前 dense benchmark 默认将性能分析聚焦在 `ideal` 与 `llama2_7b_tp1` 大 shape 上，并额外区分：

- `operator_us`：包含 Python 调度与算子内部临时分配
- `kernel_like_us`：预分配输出并复用 workspace，用来弱化脚本侧固定开销

`smoke` preset 仅用于冒烟，不适合用来判断真实吞吐。

## 与主树同步方式

`marlin_v100` 是本地独立开发工作区，不等于主树上游包结构。

当前默认对齐的上游代码基线为 `vllm-0.17.1`，回写和对比时应按这一版本的目录结构理解映射关系。

回写主树时应以 `upstream_map.yaml` 为准，只同步映射中明确列出的上游源码文件。下面这些内容默认不回写主树：

- 本目录的文档文件
- 本目录的测试文件
- 本目录的 Python 薄封装
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
- `python/marlin_v100/_C*.so`
- `python/marlin_v100/_moe_C*.so`

建议纳入版本管理的内容包括：

- `csrc/`
- `python/marlin_v100/*.py`
- `tests/`
- `setup.py`
- `CMakeLists.txt`
- `upstream_map.yaml`
- `README.md`
- `AGENTS.md`
- `.gitignore`
