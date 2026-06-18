from __future__ import annotations

from enum import Enum

import torch
import torch.nn.functional as F


class MoEActivation(Enum):
    SILU = "silu"
    GELU = "gelu"
    RELU2 = "relu2"
    SWIGLUOAI = "swigluoai"
    SWIGLUSTEP = "swiglustep"
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"

    @property
    def is_gated(self) -> bool:
        return not self.value.endswith("_no_mul")

    def without_mul(self) -> "MoEActivation":
        return _WITHOUT_MUL.get(self, self)

    @classmethod
    def from_str(cls, value: str) -> "MoEActivation":
        for activation in cls:
            if activation.value == value:
                return activation
        raise ValueError(f"Unknown MoE activation: {value!r}")


_WITHOUT_MUL = {
    MoEActivation.SILU: MoEActivation.SILU_NO_MUL,
    MoEActivation.GELU: MoEActivation.GELU_NO_MUL,
    MoEActivation.RELU2: MoEActivation.RELU2_NO_MUL,
}


def activation_without_mul(activation: str) -> str:
    return MoEActivation.from_str(activation).without_mul().value


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    if activation == MoEActivation.SILU:
        gate, up = input.chunk(2, dim=-1)
        output.copy_(F.silu(gate) * up)
    elif activation == MoEActivation.GELU:
        gate, up = input.chunk(2, dim=-1)
        output.copy_(F.gelu(gate) * up)
    elif activation == MoEActivation.RELU2:
        gate, up = input.chunk(2, dim=-1)
        output.copy_(F.relu(gate).square() * up)
    elif activation == MoEActivation.SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == MoEActivation.GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == MoEActivation.RELU2_NO_MUL:
        output.copy_(F.relu(input).square())
    else:
        raise NotImplementedError(f"Unsupported MoE activation: {activation}")
    return output
