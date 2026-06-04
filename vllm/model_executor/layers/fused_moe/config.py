from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import torch


class RoutingMethodType(IntEnum):
    Default = 0
    Renormalize = 1
    DeepSeekV3 = 2
    Llama4 = 3
    RenormalizeNaive = 4
    TopK = 5
    Custom = 6
    Simulated = 7
    Unspecified = 8


@dataclass
class FusedMoEConfig:
    hidden_dim: int = 0
    intermediate_size_per_partition: int = 0
    disable_inplace: bool = False
    is_act_and_mul: bool = True


@dataclass
class FusedMoEParallelConfig:
    pass


@dataclass
class FusedMoEQuantConfig:
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    w1_zp: torch.Tensor | None = None
    w2_zp: torch.Tensor | None = None
    block_shape: list[int] | None = None


class FusedMoEQuantFormat(Enum):
    INT = "int"
    FP8 = "fp8"


def int4_w4a16_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def int8_w8a16_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def int4_w4afp8_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def int8_w8afp8_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def int8_w8a8_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def fp8_w8a8_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def fp8_w8a16_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def nvfp4_w4a16_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def mxfp4_w4a16_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def mxfp4_w4a8_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)


def ocp_mx_moe_quant_config(**kwargs) -> FusedMoEQuantConfig:
    return FusedMoEQuantConfig(**kwargs)
