from __future__ import annotations

from collections.abc import Callable

import torch
from torch.nn import Parameter


class BasevLLMParameter(Parameter):
    def __new__(cls, data: torch.Tensor | None, **kwargs):
        return super().__new__(cls, data=data, requires_grad=False)

    def __init__(self, data: torch.Tensor, weight_loader: Callable | None = None):
        self._weight_loader = weight_loader or (lambda param, loaded_weight: None)

    @property
    def weight_loader(self) -> Callable:
        return self._weight_loader

    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        self.data.copy_(loaded_weight)

    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        self.data.copy_(loaded_weight)


class ModelWeightParameter(BasevLLMParameter):
    def __init__(
        self,
        data: torch.Tensor,
        weight_loader: Callable | None = None,
        input_dim: int = 0,
        output_dim: int = 1,
    ):
        self._input_dim = input_dim
        self._output_dim = output_dim
        super().__init__(data=data, weight_loader=weight_loader)

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim


class GroupQuantScaleParameter(ModelWeightParameter):
    pass


class ChannelQuantScaleParameter(ModelWeightParameter):
    pass


class PerTensorScaleParameter(BasevLLMParameter):
    pass


class BlockQuantScaleParameter(ModelWeightParameter):
    pass


class RowvLLMParameter(BasevLLMParameter):
    def __init__(
        self,
        data: torch.Tensor,
        weight_loader: Callable | None = None,
        input_dim: int = 0,
    ):
        self._input_dim = input_dim
        super().__init__(data=data, weight_loader=weight_loader)

    @property
    def input_dim(self) -> int:
        return self._input_dim


class PackedvLLMParameter(ModelWeightParameter):
    def __init__(
        self,
        data: torch.Tensor,
        weight_loader: Callable | None = None,
        input_dim: int = 0,
        output_dim: int = 1,
        packed_dim: int = 0,
        packed_factor: int = 1,
    ):
        self._packed_dim = packed_dim
        self._packed_factor = packed_factor
        super().__init__(
            data=data,
            weight_loader=weight_loader,
            input_dim=input_dim,
            output_dim=output_dim,
        )

    @property
    def packed_dim(self) -> int:
        return self._packed_dim

    @property
    def packed_factor(self) -> int:
        return self._packed_factor


class PackedColumnParameter(PackedvLLMParameter):
    pass


def permute_param_layout_(
    param: BasevLLMParameter,
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> BasevLLMParameter:
    curr_input_dim = getattr(param, "input_dim", None)
    curr_output_dim = getattr(param, "output_dim", None)

    if curr_input_dim is None or curr_output_dim is None:
        assert param.data.dim() == 2, (
            "permute_param_layout_ only supports 2D parameters when either "
            "input_dim or output_dim is not set"
        )

    if curr_input_dim is None:
        assert curr_output_dim is not None
        curr_input_dim = (curr_output_dim + 1) % 2
    if curr_output_dim is None:
        assert curr_input_dim is not None
        curr_output_dim = (curr_input_dim + 1) % 2

    perm = [
        i for i in range(param.data.dim()) if i not in [curr_input_dim, curr_output_dim]
    ]
    perm.insert(input_dim, curr_input_dim)
    perm.insert(output_dim, curr_output_dim)

    if "packed_dim" in kwargs:
        assert (
            hasattr(param, "packed_dim")
            and param.packed_dim == perm[kwargs["packed_dim"]]
        ), "permute_param_layout_ currently doesn't support repacking"

    param.data = param.data.permute(*perm)
    if hasattr(param, "_input_dim"):
        param._input_dim = input_dim
    if hasattr(param, "_output_dim"):
        param._output_dim = output_dim
    if "packed_dim" in kwargs and hasattr(param, "_packed_dim"):
        param._packed_dim = kwargs["packed_dim"]

    return param
