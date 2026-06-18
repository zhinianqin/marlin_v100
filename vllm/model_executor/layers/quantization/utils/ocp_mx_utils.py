from __future__ import annotations

from enum import Enum

OCP_MX_BLOCK_SIZE = 32

OCP_MX_DTYPES = {
    "mxfp4",
    "mxfp6_e3m2",
    "mxfp6_e2m3",
    "mxfp8_e4m3",
    "mxfp8_e5m2",
    "mxint8",
}


class OCP_MX_Scheme(str, Enum):
    w_mxfp4 = "w_mxfp4"
    w_mxfp4_a_mxfp4 = "w_mxfp4_a_mxfp4"
    w_mxfp4_a_mxfp6_e3m2 = "w_mxfp4_a_mxfp6_e3m2"
    w_mxfp4_a_mxfp6_e2m3 = "w_mxfp4_a_mxfp6_e2m3"
    w_mxfp4_a_fp8 = "w_mxfp4_a_fp8"
    w_mxfp6_e3m2 = "w_mxfp6_e3m2"
    w_mxfp6_e3m2_a_mxfp6_e3m2 = "w_mxfp6_e3m2_a_mxfp6_e3m2"
    w_mxfp6_e3m2_a_fp8 = "w_mxfp6_e3m2_a_fp8"
    w_mxfp6_e2m3 = "w_mxfp6_e2m3"
    w_mxfp6_e2m3_a_mxfp6_e2m3 = "w_mxfp6_e2m3_a_mxfp6_e2m3"
    w_mxfp6_e2m3_a_fp8 = "w_mxfp6_e2m3_a_fp8"

    @classmethod
    def from_quant_dtype(cls, input_dtype: str | None, weight_dtype: str | None):
        if input_dtype not in OCP_MX_DTYPES and weight_dtype not in OCP_MX_DTYPES:
            return None
        if input_dtype is None and weight_dtype == "mxfp4":
            return cls.w_mxfp4
        if input_dtype is None and weight_dtype == "mxfp6_e3m2":
            return cls.w_mxfp6_e3m2
        if input_dtype is None and weight_dtype == "mxfp6_e2m3":
            return cls.w_mxfp6_e2m3
        if input_dtype == "mxfp4" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_mxfp4
        if input_dtype == "mxfp6_e3m2" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_mxfp6_e3m2
        if input_dtype == "mxfp6_e2m3" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_mxfp6_e2m3
        if input_dtype == "fp8" and weight_dtype == "mxfp4":
            return cls.w_mxfp4_a_fp8
        if input_dtype == "mxfp6_e3m2" and weight_dtype == "mxfp6_e3m2":
            return cls.w_mxfp6_e3m2_a_mxfp6_e3m2
        if input_dtype == "fp8" and weight_dtype == "mxfp6_e3m2":
            return cls.w_mxfp6_e3m2_a_fp8
        if input_dtype == "mxfp6_e2m3" and weight_dtype == "mxfp6_e2m3":
            return cls.w_mxfp6_e2m3_a_mxfp6_e2m3
        if input_dtype == "fp8" and weight_dtype == "mxfp6_e2m3":
            return cls.w_mxfp6_e2m3_a_fp8
        return None
