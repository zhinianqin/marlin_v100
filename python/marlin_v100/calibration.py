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
    allow_act_order: bool = False


QUANT_TYPE_SUPPORT: dict[str, QuantTypeSupport] = {
    "uint4": QuantTypeSupport(name="uint4", dense_supported=True, moe_supported=True),
    "uint4b8": QuantTypeSupport(name="uint4b8", dense_supported=True, moe_supported=True),
    "uint8": QuantTypeSupport(name="uint8", dense_supported=True, moe_supported=False),
    "uint8b128": QuantTypeSupport(name="uint8b128", dense_supported=True, moe_supported=False),
    "fp8": QuantTypeSupport(
        name="fp8",
        dense_supported=True,
        moe_supported=False,
    ),
    "nvfp4": QuantTypeSupport(
        name="nvfp4",
        dense_supported=True,
        moe_supported=False,
        requires_nvfp4_global_scale=True,
    ),
    "mxfp4": QuantTypeSupport(
        name="mxfp4",
        dense_supported=True,
        moe_supported=False,
        requires_mxfp4=True,
    ),
}

_ARCHITECTURE_SUPPORT: dict[tuple[int, int], ArchitectureSupport] = {
    # SM70 supports int quantized paths plus dense FP8/NVFP4/MXFP4
    # weight-only dequant into the regular FP16 MMA path. FP8/FP4 activation
    # paths remain absent.
    (7, 0): ArchitectureSupport(
        target_capability=(7, 0),
        dense_group_sizes=(-1, 32, 64, 128),
        allow_fp8_kernels=False,
        allow_nvfp4_global_scale=True,
        allow_mxfp4=True,
        allow_act_order=False,
    ),
}

_ACT_ORDER_SUPPORTED_QUANT_NAMES = ("uint4b8", "uint8b128")
_ACT_ORDER_UNSUPPORTED_ERROR = "act_order is not supported for this SM70 Marlin build."


def _encode_scalar_type_id(
    exponent: int,
    mantissa: int,
    signed: bool,
    bias: int = 0,
    finite_values_only: bool = False,
    nan_repr: int = 1,
) -> int:
    val = 0
    offset = 0

    def add_field(member: int, bit_width: int) -> None:
        nonlocal val, offset
        bit_mask = (1 << bit_width) - 1
        val |= (int(member) & bit_mask) << offset
        offset += bit_width

    add_field(exponent, 8)
    add_field(mantissa, 8)
    add_field(signed, 1)
    add_field(bias, 32)
    add_field(finite_values_only, 1)
    add_field(nan_repr, 8)
    return val


_QUANT_TYPE_IDS = {
    "uint4": _encode_scalar_type_id(0, 4, False, 0),
    "uint4b8": _encode_scalar_type_id(0, 4, False, 8),
    "uint8": _encode_scalar_type_id(0, 8, False, 0),
    "uint8b128": _encode_scalar_type_id(0, 8, False, 128),
    "fp8": _encode_scalar_type_id(4, 3, True, 0, True, 2),
    "nvfp4": _encode_scalar_type_id(2, 1, True, 0, True, 0),
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
            dense_group_sizes=(-1, 32, 64, 128),
            allow_fp8_kernels=False,
            allow_nvfp4_global_scale=False,
            allow_mxfp4=False,
            allow_act_order=False,
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


def supported_act_order_quant_type_names(
    candidates: Iterable[str] | Mapping[str, object],
    target_capability: tuple[int, int] | None = None,
) -> tuple[str, ...]:
    if not architecture_support(target_capability).allow_act_order:
        return ()
    dense_supported = set(supported_dense_quant_type_names(candidates, target_capability))
    moe_supported = set(supported_moe_quant_type_names(candidates, target_capability))
    return tuple(
        name
        for name in _normalize_candidate_names(candidates)
        if name in _ACT_ORDER_SUPPORTED_QUANT_NAMES
        and name in dense_supported
        and name in moe_supported
    )


def supported_dense_group_sizes(
    group_sizes: Iterable[int],
    target_capability: tuple[int, int] | None = None,
) -> tuple[int, ...]:
    supported = set(architecture_support(target_capability).dense_group_sizes)
    return tuple(group_size for group_size in group_sizes if group_size in supported)


def validate_dense_group_size(
    group_size: int,
    target_capability: tuple[int, int] | None = None,
) -> int:
    supported = set(architecture_support(target_capability).dense_group_sizes)
    if group_size not in supported:
        raise ValueError(
            f"Unsupported dense group_size={group_size}. Supported values are "
            f"{sorted(supported)}."
        )
    return group_size


def resolve_dense_runtime_group_size(
    group_size: int,
    *,
    act_order: bool,
    is_k_full: bool,
    target_capability: tuple[int, int] | None = None,
) -> int:
    validate_dense_group_size(group_size, target_capability)
    if act_order and not architecture_support(target_capability).allow_act_order:
        raise ValueError(_ACT_ORDER_UNSUPPORTED_ERROR)
    if act_order and not is_k_full:
        return 0
    return group_size


def act_order_runtime_group_size(
    group_size: int,
    *,
    is_k_full: bool,
    target_capability: tuple[int, int] | None = None,
) -> int:
    return resolve_dense_runtime_group_size(
        group_size,
        act_order=True,
        is_k_full=is_k_full,
        target_capability=target_capability,
    )


def is_act_order_runtime_group_size(group_size: int) -> bool:
    return group_size == 0


def infer_dense_group_size(size_k: int, num_groups: int) -> int:
    if num_groups <= 1:
        return -1
    if size_k % num_groups != 0:
        raise ValueError(f"size_k={size_k} must be divisible by num_groups={num_groups}.")
    return size_k // num_groups


def quant_type_name_from_id(type_id: int) -> str | None:
    for name, quant_type_id in _QUANT_TYPE_IDS.items():
        if quant_type_id == type_id:
            return name
    return None


def is_supported_act_order_type_id(type_id: int) -> bool:
    quant_name = quant_type_name_from_id(type_id)
    return quant_name in _ACT_ORDER_SUPPORTED_QUANT_NAMES


def has_nonempty_metadata(tensor: torch.Tensor | None) -> bool:
    return tensor is not None and tensor.numel() > 0


def validate_act_order_metadata(
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    *,
    size_k: int,
) -> bool:
    has_g_idx = has_nonempty_metadata(g_idx)
    has_perm = has_nonempty_metadata(perm)
    if has_g_idx != has_perm:
        raise ValueError("g_idx and perm must be provided together for act_order.")

    if not has_g_idx:
        return False

    assert g_idx is not None
    assert perm is not None
    if g_idx.size(-1) != size_k or perm.size(-1) != size_k:
        raise ValueError(
            "g_idx and perm must have last dimension equal to size_k when act_order is enabled."
        )
    return True


def validate_dense_marlin_call(
    *,
    b_type_id: int,
    size_k: int,
    num_groups: int,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    is_k_full: bool,
    target_capability: tuple[int, int] | None = None,
) -> dict[str, int | bool]:
    quant_group_size = infer_dense_group_size(size_k, num_groups)
    act_order = validate_act_order_metadata(g_idx, perm, size_k=size_k)
    quant_name = quant_type_name_from_id(b_type_id)
    if quant_name == "nvfp4":
        if act_order:
            raise ValueError("fp4 dense weight-only path does not support act_order.")
        if not is_k_full:
            raise ValueError("fp4 dense weight-only path requires is_k_full=True.")
        if quant_group_size not in (16, 32):
            raise ValueError(
                "fp4 dense weight-only path supports only group_size 16 "
                "(NVFP4) or 32 (MXFP4)."
            )
        runtime_group_size = quant_group_size
    else:
        runtime_group_size = resolve_dense_runtime_group_size(
            quant_group_size,
            act_order=act_order,
            is_k_full=is_k_full,
            target_capability=target_capability,
        )
    if act_order:
        if not is_supported_act_order_type_id(b_type_id):
            raise ValueError(
                "act_order is only supported for uint4b8 and uint8b128 in this workspace."
            )
        if is_k_full and num_groups <= 1:
            raise ValueError("act_order with is_k_full=True requires more than one scale group.")
    if quant_name == "fp8" and runtime_group_size not in (-1, 128):
        raise ValueError("fp8 dense weight-only path supports only group_size -1 or 128.")
    return {
        "act_order": act_order,
        "quant_group_size": quant_group_size,
        "runtime_group_size": runtime_group_size,
    }


def validate_moe_marlin_call(
    *,
    b_type_id: int,
    size_k: int,
    num_groups: int,
    g_idx: torch.Tensor | None,
    perm: torch.Tensor | None,
    is_k_full: bool,
    target_capability: tuple[int, int] | None = None,
) -> dict[str, int | bool]:
    return validate_dense_marlin_call(
        b_type_id=b_type_id,
        size_k=size_k,
        num_groups=num_groups,
        g_idx=g_idx,
        perm=perm,
        is_k_full=is_k_full,
        target_capability=target_capability,
    )


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
