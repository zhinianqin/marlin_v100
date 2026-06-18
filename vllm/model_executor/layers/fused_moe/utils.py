from __future__ import annotations

import torch


def _resize_cache(cache: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return cache[: int(torch.tensor(shape).prod().item())].view(shape)


def disable_inplace() -> bool:
    return False
