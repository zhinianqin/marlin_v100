from __future__ import annotations

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)

    def create_weights(self, *args, **kwargs):
        return None

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        return None

    def apply(self, *args, **kwargs):
        raise NotImplementedError
