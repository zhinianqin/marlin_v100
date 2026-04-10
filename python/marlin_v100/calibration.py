from __future__ import annotations

from dataclasses import dataclass
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
_CMAKE_LISTS = ROOT / "CMakeLists.txt"
_CUDA_ARCH_RE = re.compile(r'set\(CUDA_ARCHS "(\d+)\.(\d+)"\)')


@dataclass(frozen=True)
class QuantTypeSupport:
    name: str
    dense_supported: bool
    moe_supported: bool
    requires_fp8_kernels: bool = False
    requires_nvfp4_global_scale: bool = False
    requires_mxfp4: bool = False


@dataclass(frozen=True)
class ArchitectureSupport:
    target_capability: tuple[int, int]
    dense_group_sizes: tuple[int, ...]
    allow_fp8_kernels: bool
    allow_nvfp4_global_scale: bool
    allow_mxfp4: bool


QUANT_TYPE_SUPPORT: dict[str, QuantTypeSupport] = {
    "uint4b8": QuantTypeSupport(name="uint4b8", dense_supported=True, moe_supported=True),
    "uint8b128": QuantTypeSupport(name="uint8b128", dense_supported=True, moe_supported=True),
    "fp8": QuantTypeSupport(
        name="fp8",
        dense_supported=True,
        moe_supported=True,
        requires_fp8_kernels=True,
    ),
    "nvfp4": QuantTypeSupport(
        name="nvfp4",
        dense_supported=True,
        moe_supported=True,
        requires_fp8_kernels=True,
        requires_nvfp4_global_scale=True,
    ),
    "mxfp4": QuantTypeSupport(
        name="mxfp4",
        dense_supported=True,
        moe_supported=True,
        requires_mxfp4=True,
    ),
}

_ARCHITECTURE_SUPPORT: dict[tuple[int, int], ArchitectureSupport] = {
    # 2b1fd987 keeps SM70 limited to the uint4b8/uint8b128-style paths. Kernel
    # generators and runtime checks explicitly reject fp8 and nvfp4-specific flows.
    (7, 0): ArchitectureSupport(
        target_capability=(7, 0),
        dense_group_sizes=(128, -1),
        allow_fp8_kernels=False,
        allow_nvfp4_global_scale=False,
        allow_mxfp4=False,
    ),
    # The current minimal SM75 workspace still carries the same exclusions in its
    # generators/runtime checks, even though the build target is different.
    (7, 5): ArchitectureSupport(
        target_capability=(7, 5),
        dense_group_sizes=(128, -1),
        allow_fp8_kernels=False,
        allow_nvfp4_global_scale=False,
        allow_mxfp4=False,
    ),
}


@lru_cache(maxsize=1)
def source_target_capability() -> tuple[int, int]:
    match = _CUDA_ARCH_RE.search(_CMAKE_LISTS.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"Could not determine CUDA target arch from {_CMAKE_LISTS}.")
    return int(match.group(1)), int(match.group(2))


def source_target_cuda_arch_arg() -> str:
    major, minor = source_target_capability()
    return f"{major}.{minor}"


def source_target_sm_tag() -> str:
    major, minor = source_target_capability()
    return f"sm{major}{minor}"


def source_target_label() -> str:
    _major, _minor = source_target_capability()
    return source_target_sm_tag().upper()


def architecture_support(
    target_capability: tuple[int, int] | None = None,
) -> ArchitectureSupport:
    capability = source_target_capability() if target_capability is None else target_capability
    return _ARCHITECTURE_SUPPORT.get(
        capability,
        ArchitectureSupport(
            target_capability=capability,
            dense_group_sizes=(128, -1),
            allow_fp8_kernels=False,
            allow_nvfp4_global_scale=False,
            allow_mxfp4=False,
        ),
    )


def _normalize_candidate_names(
    candidates: Iterable[str] | Mapping[str, object],
) -> tuple[str, ...]:
    if isinstance(candidates, Mapping):
        return tuple(candidates.keys())
    return tuple(candidates)


def _quant_type_supported(
    quant_name: str,
    subsystem: str,
    target_capability: tuple[int, int] | None = None,
) -> bool:
    support = QUANT_TYPE_SUPPORT.get(quant_name)
    if support is None:
        return False

    if subsystem == "dense":
        subsystem_supported = support.dense_supported
    elif subsystem == "moe":
        subsystem_supported = support.moe_supported
    else:
        raise ValueError(f"Unsupported subsystem: {subsystem!r}")

    if not subsystem_supported:
        return False

    arch = architecture_support(target_capability)
    if support.requires_fp8_kernels and not arch.allow_fp8_kernels:
        return False
    if support.requires_nvfp4_global_scale and not arch.allow_nvfp4_global_scale:
        return False
    if support.requires_mxfp4 and not arch.allow_mxfp4:
        return False
    return True


def supported_dense_quant_type_names(
    candidates: Iterable[str] | Mapping[str, object],
    target_capability: tuple[int, int] | None = None,
) -> tuple[str, ...]:
    return tuple(
        name
        for name in _normalize_candidate_names(candidates)
        if _quant_type_supported(name, "dense", target_capability)
    )


def supported_moe_quant_type_names(
    candidates: Iterable[str] | Mapping[str, object],
    target_capability: tuple[int, int] | None = None,
) -> tuple[str, ...]:
    return tuple(
        name
        for name in _normalize_candidate_names(candidates)
        if _quant_type_supported(name, "moe", target_capability)
    )


def supported_dense_group_sizes(
    group_sizes: Iterable[int],
    target_capability: tuple[int, int] | None = None,
) -> tuple[int, ...]:
    supported = set(architecture_support(target_capability).dense_group_sizes)
    return tuple(group_size for group_size in group_sizes if group_size in supported)


def runtime_capability(device: int | torch.device | None = None) -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to query runtime capability.")
    target_device = 0 if device is None else device
    return torch.cuda.get_device_capability(target_device)


def runtime_matches_source_target(device: int | torch.device | None = None) -> bool:
    if not torch.cuda.is_available():
        return False
    return runtime_capability(device) == source_target_capability()


def format_capability(capability: tuple[int, int]) -> str:
    return f"sm{capability[0]}{capability[1]} ({capability[0]}.{capability[1]})"
