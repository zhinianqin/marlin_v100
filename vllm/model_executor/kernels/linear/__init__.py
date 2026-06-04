from __future__ import annotations

from vllm.model_executor.kernels.linear.mixed_precision import (
    MPLinearKernel,
    MPLinearLayerConfig,
    MarlinLinearKernel,
)


def choose_mp_linear_kernel(*args, **kwargs):
    del args, kwargs
    return MarlinLinearKernel


__all__ = [
    "MPLinearKernel",
    "MPLinearLayerConfig",
    "MarlinLinearKernel",
    "choose_mp_linear_kernel",
]
