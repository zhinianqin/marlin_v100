from __future__ import annotations

import functools
from dataclasses import dataclass
from enum import Enum

_SCALAR_TYPES_ID_MAP = {}


class NanRepr(Enum):
    NONE = 0
    IEEE_754 = 1
    EXTD_RANGE_MAX_MIN = 2


@dataclass(frozen=True)
class ScalarType:
    exponent: int
    mantissa: int
    signed: bool
    bias: int = 0
    _finite_values_only: bool = False
    nan_repr: NanRepr = NanRepr.IEEE_754

    @functools.cached_property
    def id(self) -> int:
        val = 0
        offset = 0

        def add_field(member, bit_width: int) -> None:
            nonlocal val, offset
            bit_mask = (1 << bit_width) - 1
            val |= (int(member) & bit_mask) << offset
            offset += bit_width

        add_field(self.exponent, 8)
        add_field(self.mantissa, 8)
        add_field(self.signed, 1)
        add_field(self.bias, 32)
        add_field(self._finite_values_only, 1)
        add_field(self.nan_repr.value, 8)
        _SCALAR_TYPES_ID_MAP[val] = self
        return val

    @property
    def size_bits(self) -> int:
        return self.exponent + self.mantissa + int(self.signed)

    def __str__(self) -> str:
        for name, value in vars(scalar_types).items():
            if not name.startswith("_") and value == self:
                return name
        return (
            f"ScalarType(exponent={self.exponent}, mantissa={self.mantissa}, "
            f"signed={self.signed}, bias={self.bias})"
        )

    @classmethod
    def int_(cls, size_bits: int, bias: int | None = None) -> "ScalarType":
        return cls(0, size_bits - 1, True, 0 if bias is None else bias)

    @classmethod
    def uint(cls, size_bits: int, bias: int | None = None) -> "ScalarType":
        return cls(0, size_bits, False, 0 if bias is None else bias)

    @classmethod
    def float_(
        cls,
        exponent: int,
        mantissa: int,
        finite_values_only: bool,
        nan_repr: NanRepr,
    ) -> "ScalarType":
        return cls(exponent, mantissa, True, 0, finite_values_only, nan_repr)

    @classmethod
    def from_id(cls, scalar_type_id: int) -> "ScalarType":
        if scalar_type_id not in _SCALAR_TYPES_ID_MAP:
            raise ValueError(f"scalar_type_id {scalar_type_id} doesn't exist.")
        return _SCALAR_TYPES_ID_MAP[scalar_type_id]


class scalar_types:
    uint4 = ScalarType.uint(4, None)
    uint4b8 = ScalarType.uint(4, 8)
    uint8 = ScalarType.uint(8, None)
    uint8b128 = ScalarType.uint(8, 128)
    float4_e2m1f = ScalarType.float_(2, 1, True, NanRepr.NONE)
    float8_e4m3fn = ScalarType.float_(4, 3, True, NanRepr.EXTD_RANGE_MAX_MIN)
    float8_e8m0fnu = ScalarType(8, 0, False, 0, True, NanRepr.EXTD_RANGE_MAX_MIN)


for _scalar_type in (
    scalar_types.uint4,
    scalar_types.uint4b8,
    scalar_types.uint8,
    scalar_types.uint8b128,
    scalar_types.float4_e2m1f,
    scalar_types.float8_e4m3fn,
    scalar_types.float8_e8m0fnu,
):
    _ = _scalar_type.id

