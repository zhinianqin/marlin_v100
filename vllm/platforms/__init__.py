from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class PlatformEnum(Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    XPU = "xpu"
    CPU = "cpu"
    UNSPECIFIED = "unspecified"


class CpuArchEnum(Enum):
    X86 = "x86"
    ARM = "arm"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DeviceCapability:
    major: int
    minor: int

    def to_int(self) -> int:
        return self.major * 10 + self.minor

    def __getitem__(self, index: int) -> int:
        return (self.major, self.minor)[index]


class _CurrentPlatform:
    _enum = PlatformEnum.CUDA

    def is_cuda(self) -> bool:
        return True

    def is_cuda_alike(self) -> bool:
        return True

    def is_rocm(self) -> bool:
        return False

    def is_xpu(self) -> bool:
        return False

    def is_cpu(self) -> bool:
        return False

    def get_device_capability(self) -> DeviceCapability | None:
        override = torch.cuda.is_available()
        if override:
            major, minor = torch.cuda.get_device_capability()
            if major * 10 + minor >= 75:
                return DeviceCapability(major, minor)
        return DeviceCapability(7, 5)

    def is_device_capability(self, capability: int) -> bool:
        current = self.get_device_capability()
        if current is None:
            return False
        if isinstance(capability, tuple):
            return (current.major, current.minor) == capability
        return current.to_int() == capability

    def is_device_capability_family(self, family: int) -> bool:
        current = self.get_device_capability()
        return current is not None and current.major == family

    def has_device_capability(
        self,
        capability: tuple[int, int] | int,
        device_id: int = 0,
    ) -> bool:
        current = self.get_device_capability()
        if current is None:
            return False
        if isinstance(capability, tuple):
            return (current.major, current.minor) >= capability
        return current.to_int() >= capability

    def fp8_dtype(self) -> torch.dtype:
        return torch.float8_e4m3fn

    def is_fp8_fnuz(self) -> bool:
        return False

    def supports_mx(self) -> bool:
        return False

    def get_cpu_architecture(self) -> CpuArchEnum:
        return CpuArchEnum.X86

    def num_compute_units(self, device_id: int = 0) -> int:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(device_id).multi_processor_count
        return 1

    def import_kernels(self) -> None:
        import vllm._C  # noqa: F401
        import vllm._moe_C  # noqa: F401

    def use_sync_weight_loader(self) -> bool:
        return False

    def make_synced_weight_loader(self, weight_loader):
        return weight_loader


current_platform = _CurrentPlatform()
