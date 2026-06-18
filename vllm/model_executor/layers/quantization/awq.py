from __future__ import annotations

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig


class AWQConfig(QuantizationConfig):
    @staticmethod
    def get_min_capability() -> int:
        return 75
