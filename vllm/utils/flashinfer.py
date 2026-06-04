from __future__ import annotations


def has_flashinfer() -> bool:
    return False


def flashinfer_scaled_fp4_mm(*args, **kwargs):
    raise NotImplementedError("FlashInfer FP4 MM is not available in marlin_v100")


def flashinfer_quant_nvfp4_8x4_sf_layout(*args, **kwargs):
    raise NotImplementedError(
        "FlashInfer NVFP4 quantization is not available in marlin_v100"
    )
