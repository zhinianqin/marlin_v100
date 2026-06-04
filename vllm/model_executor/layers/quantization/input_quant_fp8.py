from __future__ import annotations

import torch


class QuantFP8:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.ones(
            (x.shape[0], 1),
            device=x.device,
            dtype=torch.float32,
        )
        return x.to(torch.float8_e4m3fn), scale

