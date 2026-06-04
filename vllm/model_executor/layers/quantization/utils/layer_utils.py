from __future__ import annotations

import torch


def update_tensor_inplace(dst: torch.Tensor, src: torch.Tensor):
    assert dst.dtype == src.dtype, "Tensors must have the same dtype"
    dst.as_strided_(src.shape, src.stride())
    if dst.data_ptr() != src.data_ptr():
        dst.copy_(src)
        del src


def replace_parameter(
    mod: torch.nn.Module,
    name: str,
    new: torch.Tensor | torch.nn.Parameter,
) -> None:
    old = getattr(mod, name)
    if (
        type(old) is type(new)
        and old.dtype == new.dtype
        and old.untyped_storage().nbytes() == new.untyped_storage().nbytes()
    ):
        update_tensor_inplace(old, new)
        return
    if not isinstance(new, torch.nn.Parameter):
        new = torch.nn.Parameter(new, requires_grad=False)
    mod.register_parameter(name, torch.nn.Parameter(new, requires_grad=False))

