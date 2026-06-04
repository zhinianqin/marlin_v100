# marlin_v100 协作约定

## 工作目标

`marlin_v100` 是一个独立的最小 Marlin 开发项目，用于在较小工作区内开发、整理和验证以下能力：

- Marlin dense
- Marlin MoE
- 最小构建链
- 最小测试链

在这里工作的主要目标是提升 Marlin 重构效率，而不是复制完整的 vLLM 主树能力。

当前项目应保持“仓库外可用”状态：即使脱离父级 vLLM 目录单独存在，也应该能独立构建、独立导入扩展、独立执行 pytest 收集。

## 目录职责

- `csrc/`
  放置需要回写主树的 Marlin 相关 CUDA/C++ 源码与最小 binding。
- `vllm/`
  放置最小上游式 Python package 与扩展落位目录。扩展通过 `vllm._C` / `vllm._moe_C` 加载。
- `tests/`
  放置本地独立测试，不默认回写主树。
- `upstream_map.yaml`
  放置主树回写映射，是回写时的唯一依据。
- `cmake/`
  放置本地最小 CMake 宏，不应重新依赖父级仓库的 `cmake/utils.cmake`。
- `README.md`、`AGENTS.md`
  放置本地工作区文档，不回写主树。

## 开发约定

- 优先使用本目录下的工具链：
  `./.venv/bin/python`、`./.venv/bin/pytest`、`./.venv/bin/cmake`、`./.venv/bin/ninja`
- Python 相关命令统一带上：
  `PYTHONPATH=$PWD`
- Python 扩展加载路径与上游对齐为 `vllm._C` / `vllm._moe_C`；不要重新引入 `marlin_v100` Python package
- 不要重新引入对父级 vLLM 仓库路径、pytest 配置或 Python helper 的硬依赖
- 不要把 `.venv/`、`build/`、`*.so`、`__pycache__/` 纳入版本管理
- 修改上游可回写源码时，要同步检查 `upstream_map.yaml` 是否仍然准确

## 构建与验证约定

推荐先按下面的方式准备构建环境：

```bash
uv venv --python 3.12
source .venv/bin/activate

uv pip install "cmake>=3.26.1" ninja "packaging>=24.2" \
  "setuptools>=77.0.3,<81.0.0" wheel jinja2 pytest numpy

uv pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

推荐构建命令：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=8
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST='7.5'
export CMAKE_ARGS='-DCMAKE_CUDA_FLAGS=-gencode arch=compute_75,code=sm_75'
PYTHONPATH=$PWD ./.venv/bin/python setup.py build_ext --inplace
```

注意：Linux 下必须使用 `LD_LIBRARY_PATH`。不要把 `D_LIBRARY_PATH` 当成可生效的替代变量。

构建前建议检查：

- 当前是否在 `marlin_v100/` 目录
- `CUDA_HOME` 与 `LD_LIBRARY_PATH` 是否正确
- 是否使用本目录 `.venv` 中的 Python 与构建工具

构建后建议检查：

- `vllm/_C.abi3.so` 是否存在
- `vllm/_moe_C.abi3.so` 是否存在
- `import vllm._C`、`import vllm._moe_C` 是否成功
- `torch.ops._C.marlin_gemm` 与 `torch.ops._moe_C.moe_wna16_marlin_gemm` 是否注册成功

当前阶段的验收分层如下：

- 当前 SM70 机器：
  只看构建与导入
- 支持 Marlin 的 SM75 机器：
  再执行运行验证

## 回写主树约定

回写主树时只依据 `upstream_map.yaml` 中列出的文件和路径执行。

默认可回写的是 Marlin 相关上游源码，默认不回写的是：

- `tests/`
- `.gitignore`
- `README.md`
- `AGENTS.md`

回写前必须先确认：

- 改动确实发生在需要回写的源码范围内
- 本地工作区专用命名没有误带入主树无关位置
- 回写 diff 仅覆盖映射文件
- `upstream_map.yaml` 中的目标路径按“相对于上游仓库根目录”理解，不绑定某台开发机的绝对路径

## 禁止事项

- 不要把 `.venv/`、`build/`、扩展 `.so` 文件提交到 git
- 不要在当前 SM70 机器上把 Marlin 运行结果当成最终数值验收
- 不要把 `pytest` 当成当前阶段的硬性通过标准
- 不要绕过 `upstream_map.yaml` 直接大范围覆盖主树
- 不要把本地文档、测试、辅助封装当作上游源码一并回写
