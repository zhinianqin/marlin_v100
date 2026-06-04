from __future__ import annotations

from vllm.model_executor.layers.linear import set_weight_attrs
from vllm.model_executor.layers.quantization.utils import replace_parameter

__all__ = ["replace_parameter", "set_weight_attrs"]
