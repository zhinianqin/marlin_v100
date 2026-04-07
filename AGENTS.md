# marlin_v100 协作约定

## 工作目标

`marlin_v100` 是一个独立的最小 Marlin 开发项目，用于在较小工作区内开发、整理和验证以下能力：

- Marlin dense
- Marlin MoE
- 最小构建链
- 最小测试链

在这里工作的主要目标是提升 Marlin 重构效率，而不是复制完整的 vLLM 主树能力。

## 目录职责

- `csrc/`
  放置需要回写主树的 Marlin 相关 CUDA/C++ 源码与最小 binding。
- `python/marlin_v100/`
  放置本地开发使用的 Python 薄封装，只服务于本工作区。
- `tests/`
  放置本地独立测试，不默认回写主树。
- `upstream_map.yaml`
  放置主树回写映射，是回写时的唯一依据。
- `README.md`、`AGENTS.md`
  放置本地工作区文档，不回写主树。

## 开发约定

- 优先使用本目录下的工具链：
  `./.venv/bin/python`、`./.venv/bin/cmake`、`./.venv/bin/ninja`
- Python 相关命令统一带上：
  `PYTHONPATH=$PWD/python`
- 不要把本地工作区名 `marlin_v100` 传播到主树的上游 Python 包结构中
- 不要把 `.venv/`、`build/`、`*.so`、`__pycache__/` 纳入版本管理
- 修改上游可回写源码时，要同步检查 `upstream_map.yaml` 是否仍然准确

## 构建与验证约定

推荐构建命令：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$PWD/.venv/bin:/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS=8
export NVCC_THREADS=1
export TORCH_CUDA_ARCH_LIST='8.0'
export CMAKE_ARGS='-DCMAKE_CUDA_FLAGS=-gencode arch=compute_80,code=sm_80'
PYTHONPATH=$PWD/python ./.venv/bin/python setup.py build_ext --inplace
```

推荐轻量验证命令：

```bash
PYTHONPATH=$PWD/python ./.venv/bin/python -m pytest tests/test_marlin_generators.py -q
PYTHONPATH=$PWD/python ./.venv/bin/python -m pytest --collect-only tests
```

构建前建议检查：

- 当前是否在 `marlin_v100/` 目录
- `CUDA_HOME` 与 `LD_LIBRARY_PATH` 是否正确
- 是否使用本目录 `.venv` 中的 Python 与构建工具

构建后建议检查：

- `python/marlin_v100/_C.abi3.so` 是否存在
- `python/marlin_v100/_moe_C.abi3.so` 是否存在
- `import marlin_v100`、`import marlin_v100._C`、`import marlin_v100._moe_C` 是否成功

## 回写主树约定

回写主树时只依据 `upstream_map.yaml` 中列出的文件和路径执行。

默认可回写的是 Marlin 相关上游源码，默认不回写的是：

- `python/marlin_v100/`
- `tests/`
- `.gitignore`
- `README.md`
- `AGENTS.md`

回写前必须先确认：

- 改动确实发生在需要回写的源码范围内
- 本地工作区专用命名没有误带入主树无关位置
- 回写 diff 仅覆盖映射文件

## 禁止事项

- 不要把 `.venv/`、`build/`、扩展 `.so` 文件提交到 git
- 不要在当前 SM70 机器上把 Marlin 运行结果当成最终数值验收
- 不要绕过 `upstream_map.yaml` 直接大范围覆盖主树
- 不要把本地文档、测试、辅助封装当作上游源码一并回写
