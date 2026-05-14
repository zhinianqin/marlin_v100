from __future__ import annotations

import pytest

from marlin_v100.calibration import (
    act_order_runtime_group_size,
    architecture_support,
    supported_dense_group_sizes,
    supported_act_order_quant_type_names,
    supported_dense_quant_type_names,
    supported_moe_quant_type_names,
)


_QUANT_CANDIDATES = ("uint4", "uint4b8", "uint8", "uint8b128", "fp8", "nvfp4", "mxfp4")


def test_sm70_support_matrix_filters_out_fp8_family_candidates():
    target = (7, 0)

    assert supported_dense_quant_type_names(_QUANT_CANDIDATES, target) == (
        "uint4",
        "uint4b8",
        "uint8",
        "uint8b128",
    )
    assert supported_moe_quant_type_names(_QUANT_CANDIDATES, target) == (
        "uint4",
        "uint4b8",
    )
    assert supported_act_order_quant_type_names(_QUANT_CANDIDATES, target) == ()
    assert supported_dense_group_sizes((-1, 0, 32, 64, 128), target) == (-1, 32, 64, 128)
    with pytest.raises(ValueError, match="act_order is not supported"):
        act_order_runtime_group_size(64, is_k_full=False, target_capability=target)

    support = architecture_support(target)
    assert support.allow_fp8_kernels is False
    assert support.allow_nvfp4_global_scale is False
    assert support.allow_mxfp4 is False


def test_unknown_capability_falls_back_to_sm70_style_quant_candidates():
    target = (8, 0)

    assert supported_dense_quant_type_names(_QUANT_CANDIDATES, target) == (
        "uint4",
        "uint4b8",
        "uint8",
        "uint8b128",
    )
    assert supported_moe_quant_type_names(_QUANT_CANDIDATES, target) == (
        "uint4",
        "uint4b8",
    )
    assert supported_act_order_quant_type_names(_QUANT_CANDIDATES, target) == ()
    assert supported_dense_group_sizes((-1, 0, 32, 64, 128), target) == (-1, 32, 64, 128)
    with pytest.raises(ValueError, match="act_order is not supported"):
        act_order_runtime_group_size(64, is_k_full=False, target_capability=target)

    support = architecture_support(target)
    assert support.allow_fp8_kernels is False
    assert support.allow_nvfp4_global_scale is False
    assert support.allow_mxfp4 is False
