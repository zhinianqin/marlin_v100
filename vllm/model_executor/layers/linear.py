from __future__ import annotations

import torch


class LinearBase(torch.nn.Module):
    pass


class LinearMethodBase:
    pass


class UnquantizedLinearMethod(LinearMethodBase):
    pass


def set_weight_attrs(weight: torch.Tensor, weight_attrs: dict | None) -> None:
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        setattr(weight, key, value)
