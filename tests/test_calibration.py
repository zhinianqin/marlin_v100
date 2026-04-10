from __future__ import annotations

from marlin_v100.calibration import (
    architecture_support,
    supported_dense_group_sizes,
    supported_dense_quant_type_names,
    supported_moe_quant_type_names,
)


_QUANT_CANDIDATES = ("uint4b8", "uint8b128", "fp8", "nvfp4", "mxfp4")


def test_sm70_support_matrix_filters_out_fp8_family_candidates():
    target = (7, 0)

    assert supported_dense_quant_type_names(_QUANT_CANDIDATES, target) == (
        "uint4b8",
        "uint8b128",
    )
    assert supported_moe_quant_type_names(_QUANT_CANDIDATES, target) == (
        "uint4b8",
        "uint8b128",
    )
    assert supported_dense_group_sizes((128, -1), target) == (128, -1)

    support = architecture_support(target)
    assert support.allow_fp8_kernels is False
    assert support.allow_nvfp4_global_scale is False
    assert support.allow_mxfp4 is False


def test_sm75_support_matrix_keeps_current_workspace_quant_candidates():
    target = (7, 5)

    assert supported_dense_quant_type_names(_QUANT_CANDIDATES, target) == (
        "uint4b8",
        "uint8b128",
    )
    assert supported_moe_quant_type_names(_QUANT_CANDIDATES, target) == (
        "uint4b8",
        "uint8b128",
    )
    assert supported_dense_group_sizes((128, -1), target) == (128, -1)

    support = architecture_support(target)
    assert support.allow_fp8_kernels is False
    assert support.allow_nvfp4_global_scale is False
    assert support.allow_mxfp4 is False
