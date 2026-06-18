from __future__ import annotations

import torch


class QuantizeMethodBase:
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return None


class QuantizationConfig:
    def get_name(self) -> str:
        return self.__class__.__name__
