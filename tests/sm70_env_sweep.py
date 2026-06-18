from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os

from tests.writeback_marlin_cases import (
    DENSE_ALL_QUANT_NAMES,
    MOE_ALL_QUANT_NAMES,
    WRITEBACK_GROUP_SIZE_VALUES,
    iter_dense_writeback_matrix,
    iter_moe_writeback_matrix,
    is_dense_group_size_supported,
    is_moe_group_size_supported,
)


@dataclass(frozen=True, order=True)
class Sm70Geometry:
    cta_m: int
    cta_n: int
    cta_k: int
    warps: int
    warp_m: int
    warp_n: int
    warp_k: int

    @classmethod
    def parse(cls, label: str) -> "Sm70Geometry":
        parts = label.split("x")
        if len(parts) != 7:
            raise ValueError(f"Expected 7-field SM70 geometry, got {label!r}")
        return cls(*(int(part) for part in parts))

    @property
    def label(self) -> str:
        return (
            f"{self.cta_m}x{self.cta_n}x{self.cta_k}x{self.warps}x"
            f"{self.warp_m}x{self.warp_n}x{self.warp_k}"
        )


SM70_GEOMETRY_LABELS: tuple[str, ...] = (
    "32x64x32x4x32x32x16",
    "32x64x64x4x32x32x32",
    "32x64x64x4x32x64x16",
    "32x64x128x4x32x64x32",
    "32x128x32x4x32x32x32",
    "32x128x32x4x32x64x16",
    "32x128x64x4x32x64x32",
    "32x128x64x8x32x32x32",
    "32x128x64x8x32x64x16",
    "32x128x128x8x32x64x32",
    "32x256x32x4x32x64x32",
    "32x256x64x8x32x64x32",
    "64x64x32x4x32x32x32",
    "64x64x32x4x32x64x16",
    "64x64x32x4x64x32x16",
    "64x64x32x8x32x32x16",
    "64x64x64x4x32x64x32",
    "64x64x64x4x64x32x32",
    "64x64x64x4x64x64x16",
    "64x64x64x8x32x32x32",
    "64x64x64x8x32x64x16",
    "64x64x128x4x64x64x32",
    "64x64x128x8x32x64x32",
    "64x128x32x4x32x64x32",
    "64x128x32x4x64x32x32",
    "64x128x32x4x64x64x16",
    "64x128x32x8x32x32x32",
    "64x128x32x8x32x64x16",
    "64x128x32x8x64x32x16",
    "64x128x64x4x64x64x32",
    "64x128x64x8x32x64x32",
    "64x128x64x8x64x32x32",
    "64x128x64x8x64x64x16",
    "64x128x128x8x64x64x32",
    "64x256x32x4x64x64x32",
    "64x256x32x8x32x64x32",
    "64x256x32x8x64x32x32",
    "64x256x32x8x64x64x16",
    "64x256x64x8x64x64x32",
    "128x64x32x4x32x64x32",
    "128x64x32x4x64x32x32",
    "128x64x32x4x64x64x16",
    "128x64x32x8x32x32x32",
    "128x64x32x8x32x64x16",
    "128x64x32x8x64x32x16",
    "128x64x64x4x64x64x32",
    "128x64x64x8x32x64x32",
    "128x64x64x8x64x32x32",
    "128x64x64x8x64x64x16",
    "128x64x128x8x64x64x32",
    "128x128x32x4x64x64x32",
    "128x128x32x8x32x64x32",
    "128x128x32x8x64x32x32",
    "128x128x32x8x64x64x16",
    "128x128x64x8x64x64x32",
    "128x256x32x8x64x64x32",
    "256x64x32x4x64x64x32",
    "256x64x32x8x32x64x32",
    "256x64x32x8x64x32x32",
    "256x64x32x8x64x64x16",
    "256x64x64x8x64x64x32",
    "256x128x32x8x64x64x32",
)
SM70_GEOMETRIES: tuple[Sm70Geometry, ...] = tuple(
    Sm70Geometry.parse(label) for label in SM70_GEOMETRY_LABELS
)
SM70_MOE_GEOMETRIES: tuple[Sm70Geometry, ...] = tuple(
    geometry for geometry in SM70_GEOMETRIES if geometry.cta_m in (32, 64)
)
SM70_SPLIT_K_VALUES: tuple[int, ...] = (1, 2, 4, 8)
SM70_METADATA_CACHE_VALUES: tuple[str, ...] = ("vector_words", "lane_vectors")
FOCUSED_MNK_CASES: tuple[tuple[int, int, int], ...] = (
    (32, 1024, 1024),
    (32, 1088, 1024),
    (32, 1152, 1024),
    (64, 1024, 1024),
    (64, 1088, 1024),
    (64, 1152, 1024),
)
FOCUSED_DENSE_QUANT_NAMES: tuple[str, ...] = (
    "uint4",
    "uint4b8",
    "uint8",
    "uint8b128",
    "fp8",
    "nvfp4",
    "mxfp4",
)
FOCUSED_MOE_EXPERTS = 8
FOCUSED_MOE_TOPK = 1

_DENSE_ENV_NAMES = (
    "SM70_MARLIN_DENSE_CTA_GEOMETRY",
    "SM70_MARLIN_DENSE_SPLIT_K",
    "SM70_MARLIN_DENSE_METADATA_CACHE",
)
_MOE_ENV_NAMES = (
    "SM70_MARLIN_MOE_CTA_GEOMETRY",
    "SM70_MARLIN_MOE_SPLIT_K",
    "SM70_MARLIN_MOE_METADATA_CACHE",
)

EXPLICIT_ENV_REJECTION_RE = (
    "Invalid SM70_MARLIN_.*"
    "|Unsupported SM70 Marlin.*CTA geometry"
    "|requires PackedMacroN divisible by CTA_N"
    "|requires size_n divisible by both CTA_N and 64"
    "|requires K divisible by CTA_K="
    "|requires size_k % 32 == 0"
    "|is not divisible by tile_size"
)


@dataclass(frozen=True, order=True)
class DenseDirectOpKey:
    quant_name: str
    group_size: int
    size_m: int
    size_n: int
    size_k: int


@dataclass(frozen=True, order=True)
class MoeDirectOpKey:
    quant_name: str
    group_size: int
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    topk: int


def moe_auto_block_size(tokens: int, topk: int, experts: int) -> int:
    block_size_m = 64
    for candidate in (8, 16, 32, 48, 64):
        block_size_m = candidate
        if tokens * topk / experts / candidate < 0.9:
            break
    return block_size_m


def exhaustive_enabled() -> bool:
    return os.getenv("MARLIN_EXHAUSTIVE_ENV_SWEEP") == "1"


def exhaustive_start_limit() -> tuple[int, int | None]:
    start = int(os.getenv("MARLIN_EXHAUSTIVE_ENV_START", "0"))
    limit_value = os.getenv("MARLIN_EXHAUSTIVE_ENV_LIMIT")
    limit = None if not limit_value else int(limit_value)
    if start < 0:
        raise ValueError("MARLIN_EXHAUSTIVE_ENV_START must be non-negative.")
    if limit is not None and limit < 0:
        raise ValueError("MARLIN_EXHAUSTIVE_ENV_LIMIT must be non-negative.")
    return start, limit


def exhaustive_index_is_selected(index: int) -> bool:
    start, limit = exhaustive_start_limit()
    if index < start:
        return False
    return limit is None or index < start + limit


def exhaustive_index_is_past_limit(index: int) -> bool:
    start, limit = exhaustive_start_limit()
    return limit is not None and index >= start + limit


def packed_macro_n(size_n: int) -> int:
    if size_n % 256 == 0:
        return 256
    if size_n % 128 == 0:
        return 128
    if size_n % 64 == 0:
        return 64
    raise ValueError(f"SM70 Marlin requires size_n divisible by 64, got {size_n}.")


def dense_env_combo_is_legal(
    geometry: Sm70Geometry,
    split_k: int,
    *,
    size_n: int,
    size_k: int,
) -> bool:
    macro_n = packed_macro_n(size_n)
    if size_n % geometry.cta_n != 0:
        return False
    if macro_n % geometry.cta_n != 0:
        return False
    # The current dense binding still has a format-independent SM70 size_k % 32
    # validation before it reaches the per-geometry launcher.
    if size_k % 32 != 0:
        return False
    if split_k > 1 and size_k % geometry.cta_k != 0:
        return False
    return True


def moe_stage_env_combo_is_legal(
    geometry: Sm70Geometry,
    *,
    size_n: int,
    size_k: int,
) -> bool:
    if geometry.cta_m not in (32, 64):
        return False
    macro_n = packed_macro_n(size_n)
    if size_n % geometry.cta_n != 0:
        return False
    if macro_n % geometry.cta_n != 0:
        return False
    return size_k % geometry.cta_k == 0


@contextmanager
def dense_env(
    geometry: Sm70Geometry,
    split_k: int,
    metadata_cache: str,
) -> Iterator[None]:
    old_values = {name: os.environ.get(name) for name in _DENSE_ENV_NAMES}
    os.environ["SM70_MARLIN_DENSE_CTA_GEOMETRY"] = geometry.label
    os.environ["SM70_MARLIN_DENSE_SPLIT_K"] = str(split_k)
    os.environ["SM70_MARLIN_DENSE_METADATA_CACHE"] = metadata_cache
    try:
        yield
    finally:
        _restore_env(old_values)


@contextmanager
def moe_env(
    geometry: Sm70Geometry,
    split_k: int,
    metadata_cache: str,
) -> Iterator[None]:
    old_values = {name: os.environ.get(name) for name in _MOE_ENV_NAMES}
    os.environ["SM70_MARLIN_MOE_CTA_GEOMETRY"] = geometry.label
    os.environ["SM70_MARLIN_MOE_SPLIT_K"] = str(split_k)
    os.environ["SM70_MARLIN_MOE_METADATA_CACHE"] = metadata_cache
    try:
        yield
    finally:
        _restore_env(old_values)


def set_dense_env(
    monkeypatch,
    *,
    geometry: str,
    split_k: str,
    metadata_cache: str,
) -> None:
    monkeypatch.setenv("SM70_MARLIN_DENSE_CTA_GEOMETRY", geometry)
    monkeypatch.setenv("SM70_MARLIN_DENSE_SPLIT_K", split_k)
    monkeypatch.setenv("SM70_MARLIN_DENSE_METADATA_CACHE", metadata_cache)


def set_moe_env(
    monkeypatch,
    *,
    geometry: str,
    split_k: str,
    metadata_cache: str,
) -> None:
    monkeypatch.setenv("SM70_MARLIN_MOE_CTA_GEOMETRY", geometry)
    monkeypatch.setenv("SM70_MARLIN_MOE_SPLIT_K", split_k)
    monkeypatch.setenv("SM70_MARLIN_MOE_METADATA_CACHE", metadata_cache)


def iter_dense_direct_op_keys() -> tuple[DenseDirectOpKey, ...]:
    seen: set[DenseDirectOpKey] = set()
    for case in iter_dense_writeback_matrix():
        if not case.supported:
            continue
        shape = case.shape
        seen.add(
            DenseDirectOpKey(
                case.quant_name,
                case.group_size,
                shape.size_m,
                shape.size_n,
                shape.size_k,
            )
        )
    return tuple(sorted(seen))


def iter_dense_focused_mnk_direct_op_keys() -> tuple[DenseDirectOpKey, ...]:
    keys: set[DenseDirectOpKey] = set()
    for size_m, size_n, size_k in FOCUSED_MNK_CASES:
        for quant_name in DENSE_ALL_QUANT_NAMES:
            if quant_name not in FOCUSED_DENSE_QUANT_NAMES:
                continue
            for group_size in WRITEBACK_GROUP_SIZE_VALUES:
                if not is_dense_group_size_supported(quant_name, group_size, size_k):
                    continue
                keys.add(DenseDirectOpKey(quant_name, group_size, size_m, size_n, size_k))
    return tuple(sorted(keys))


def iter_moe_direct_op_keys() -> tuple[MoeDirectOpKey, ...]:
    seen: set[MoeDirectOpKey] = set()
    for case in iter_moe_writeback_matrix():
        if not case.supported:
            continue
        shape = case.shape
        seen.add(
            MoeDirectOpKey(
                case.quant_name,
                case.group_size,
                shape.tokens,
                shape.hidden,
                shape.intermediate,
                shape.experts,
                shape.topk,
            )
        )
    return tuple(sorted(seen))


def iter_moe_focused_mnk_direct_op_keys() -> tuple[MoeDirectOpKey, ...]:
    keys: set[MoeDirectOpKey] = set()
    for size_m, size_n, size_k in FOCUSED_MNK_CASES:
        for quant_name in MOE_ALL_QUANT_NAMES:
            for group_size in WRITEBACK_GROUP_SIZE_VALUES:
                if not is_moe_group_size_supported(
                    quant_name,
                    group_size,
                    size_k,
                    size_n,
                ):
                    continue
                keys.add(
                    MoeDirectOpKey(
                        quant_name,
                        group_size,
                        size_m,
                        size_k,
                        size_n,
                        FOCUSED_MOE_EXPERTS,
                        FOCUSED_MOE_TOPK,
                    )
                )
    return tuple(sorted(keys))


def iter_env_combinations() -> Iterator[tuple[Sm70Geometry, int, str]]:
    for geometry in SM70_GEOMETRIES:
        for split_k in SM70_SPLIT_K_VALUES:
            for metadata_cache in SM70_METADATA_CACHE_VALUES:
                yield geometry, split_k, metadata_cache


def iter_moe_env_combinations() -> Iterator[tuple[Sm70Geometry, int, str]]:
    for geometry in SM70_MOE_GEOMETRIES:
        for split_k in SM70_SPLIT_K_VALUES:
            for metadata_cache in SM70_METADATA_CACHE_VALUES:
                yield geometry, split_k, metadata_cache


def _restore_env(old_values: dict[str, str | None]) -> None:
    for name, value in old_values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
