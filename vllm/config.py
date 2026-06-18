from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace


@dataclass
class _ModelConfig:
    hf_config: object = field(default_factory=SimpleNamespace)


@dataclass
class _VllmConfig:
    model_config: _ModelConfig = field(default_factory=_ModelConfig)


_CURRENT_CONFIG = _VllmConfig()


def get_current_vllm_config() -> _VllmConfig:
    return _CURRENT_CONFIG
