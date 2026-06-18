from __future__ import annotations

import torch


def per_token_quant_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-6) / 127.0
    q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float32)

