from __future__ import annotations


def convert_to_nvfp4_moe_kernel_format(*args, **kwargs):
    raise NotImplementedError


def is_global_sf_supported_for_nvfp4_backend(*args, **kwargs):
    return False


def make_nvfp4_moe_kernel(*args, **kwargs):
    raise NotImplementedError


def make_nvfp4_moe_quant_config(*args, **kwargs):
    raise NotImplementedError


def select_nvfp4_moe_backend(*args, **kwargs):
    raise NotImplementedError
