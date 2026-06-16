#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Enumerate Marlin GEMM shapes for local model checkpoints.

This is a static analysis helper for benchmark design. It reads local model
configs and, when available, safetensors index metadata. It does not construct a
vLLM engine and does not load tensor payloads.

See docs/20260612_057_marlin_gemm_shape_enumerator.md for delivery notes and
usage.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

QUANT_SUFFIXES = {
    "qweight",
    "qzeros",
    "scales",
    "w13_weight",
    "w2_weight",
    "w13_weight_packed",
    "w2_weight_packed",
    "weight_scale",
    "weight_scale_2",
    "weight_global_scale",
    "input_scale",
    "weight_packed",
    "w13_weight_scale",
    "w2_weight_scale",
    "w13_weight_scale_2",
    "w2_weight_scale_2",
    "w13_weight_global_scale",
    "w2_weight_global_scale",
}


@dataclass(frozen=True)
class Scenario:
    tp_size: int
    enable_expert_parallel: bool

    @property
    def label(self) -> str:
        suffix = "+ep" if self.enable_expert_parallel else ""
        return f"tp{self.tp_size}{suffix}"


@dataclass
class QuantInfo:
    method: str = "unquantized"
    raw: dict[str, Any] = field(default_factory=dict)
    bits: int | None = None
    group_size: int | str | None = None
    zero_point: bool = False
    ignore: list[str] = field(default_factory=list)
    quant_algo: str | None = None
    quantized_layers: dict[str, dict[str, Any]] = field(default_factory=dict)
    format: str | None = None
    weight_args: dict[str, Any] = field(default_factory=dict)

    @property
    def is_unquantized(self) -> bool:
        return self.method in {"", "unquantized", "none", "bf16", "fp16"}


@dataclass
class ModelSpec:
    model_dir: Path
    model_name: str
    model_type: str
    architecture_family: str
    raw_config: dict[str, Any]
    text_config: dict[str, Any]
    quant: QuantInfo
    hidden_size: int
    intermediate_size: int | None
    moe_intermediate_size: int | None
    shared_expert_intermediate_size: int | None
    num_hidden_layers: int
    num_experts: int | None
    top_k: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    layer_types: list[str]
    moe_layer_indices: list[int]
    dense_layer_indices: list[int]
    attention_other_setting: dict[str, Any] | None = None


@dataclass
class Candidate:
    kind: str
    op: str
    layer_idx: int
    aliases: list[str]
    prefix_groups: list[list[str]]
    size_kind: str
    intermediate_size: int | None = None
    attn_heads: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    q_multiplier: int = 1

    @property
    def canonical_key(self) -> str:
        return self.aliases[0]


@dataclass
class QuantDecision:
    call_status: str
    quant_method: str
    quant_format: str
    group_size: int | str | None
    has_zp: bool
    marlin_path: str
    warning: str = ""


class IndexEvidence:
    def __init__(
        self,
        keys: set[str] | None,
        dtypes: dict[str, str] | None = None,
    ):
        self.keys = keys
        self.dtypes = dtypes or {}
        self._suffix_cache: dict[str, set[str]] = {}
        self._has_any_cache: dict[str, bool] = {}

    @property
    def available(self) -> bool:
        return self.keys is not None

    def suffixes_for_prefix(self, prefix: str) -> set[str]:
        if self.keys is None:
            return set()
        if prefix in self._suffix_cache:
            return self._suffix_cache[prefix]
        suffixes: set[str] = set()
        exact = prefix + "."
        for key in self.keys:
            if key.startswith(exact):
                suffixes.add(key.rsplit(".", 1)[-1])
                if self._key_is_fp8_weight(key):
                    suffixes.add("fp8_weight")
        self._suffix_cache[prefix] = suffixes
        return suffixes

    def has_any_for_prefix(self, prefix: str) -> bool:
        if self.keys is None:
            return False
        if prefix in self._has_any_cache:
            return self._has_any_cache[prefix]
        exact = prefix + "."
        has_any = any(key.startswith(exact) for key in self.keys)
        self._has_any_cache[prefix] = has_any
        return has_any

    def evidence_for_group(self, prefixes: list[str]) -> set[str]:
        suffixes: set[str] = set()
        for prefix in prefixes:
            suffixes.update(self.suffixes_for_prefix(prefix))
        return suffixes

    def has_any_for_group(self, prefixes: list[str]) -> bool:
        return any(self.has_any_for_prefix(prefix) for prefix in prefixes)

    def _key_is_fp8_weight(self, key: str) -> bool:
        if not key.endswith(".weight"):
            return False
        return self.dtypes.get(key, "").upper() in {"F8_E4M3", "F8_E4M3FN"}


def _as_int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_int_list(values: list[str] | None, default: list[int]) -> list[int]:
    if not values:
        return default
    out: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out or default


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_quant_config(model_dir: Path, config: dict[str, Any],
                       text_config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("quantization_config"), dict):
        return config["quantization_config"]
    if isinstance(text_config.get("quantization_config"), dict):
        return text_config["quantization_config"]
    if isinstance(config.get("compression_config"), dict):
        return config["compression_config"]
    if isinstance(text_config.get("compression_config"), dict):
        return text_config["compression_config"]
    quantize_config = model_dir / "quantize_config.json"
    if quantize_config.exists():
        return _read_json(quantize_config)
    return {}


def _parse_quant(model_dir: Path, config: dict[str, Any],
                 text_config: dict[str, Any]) -> QuantInfo:
    raw = _load_quant_config(model_dir, config, text_config)
    if not raw:
        return QuantInfo()

    method = str(raw.get("quant_method") or raw.get("quantization_method")
                 or "unknown").lower()
    if method == "unknown" and "config_groups" in raw:
        method = "compressed-tensors"
    if method == "compressed_tensors":
        method = "compressed-tensors"
    ignore: list[str] = []
    for key in ["modules_to_not_convert", "ignore", "exclude_modules"]:
        values = raw.get(key) or []
        if isinstance(values, str):
            ignore.append(values)
        else:
            ignore.extend(str(value) for value in values)
    ignore = list(dict.fromkeys(ignore))
    bits = _as_int(raw.get("bits") or raw.get("num_bits"))
    group_size: int | str | None = _as_int(raw.get("group_size"))
    zero_point = bool(raw.get("zero_point", False))
    quant_algo = None
    if raw.get("quant_algo") is not None:
        quant_algo = str(raw["quant_algo"]).upper()
    quantized_layers = raw.get("quantized_layers") or {}
    fmt = raw.get("format") or raw.get("fmt")

    weight_args: dict[str, Any] = {}
    groups = raw.get("config_groups")
    if isinstance(groups, dict) and groups:
        first_group = next(iter(groups.values()))
        if isinstance(first_group, dict):
            weight_args = dict(first_group.get("weights") or {})
            if group_size is None:
                group_size = _as_int(weight_args.get("group_size"))
            if bits is None:
                bits = _as_int(weight_args.get("num_bits"))
    weight_block_size = raw.get("weight_block_size")
    if group_size is None and isinstance(weight_block_size, list):
        if weight_block_size:
            if method == "fp8" and len(weight_block_size) >= 2:
                group_size = _as_int(weight_block_size[1])
            else:
                group_size = "x".join(str(x) for x in weight_block_size)

    return QuantInfo(
        method=method,
        raw=raw,
        bits=bits,
        group_size=group_size,
        zero_point=zero_point,
        ignore=ignore,
        quant_algo=quant_algo,
        quantized_layers=quantized_layers,
        format=str(fmt) if fmt is not None else None,
        weight_args=weight_args,
    )


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def _architecture_family(model_type: str) -> str:
    if model_type in {"qwen3_5_text", "qwen3_5_moe_text", "qwen3_next"}:
        return model_type
    if model_type == "minimax_m2":
        return "minimax_m2"
    if model_type == "glm4_moe":
        return "glm4_moe"
    if model_type in {"step3p5", "step3p7"}:
        return "step3p5"
    return model_type or "unknown"


def _get_num_hidden_layers(text: dict[str, Any]) -> int:
    n = _as_int(text.get("num_hidden_layers"))
    if n is not None:
        return n
    layer_types = text.get("layer_types")
    if isinstance(layer_types, list):
        return len(layer_types)
    raise ValueError("Unable to infer num_hidden_layers from config.")


def _parse_moe_layers(family: str, text: dict[str, Any],
                      num_hidden_layers: int) -> list[int]:
    explicit = text.get("moe_layer_indices") or text.get("moe_layers")
    if isinstance(explicit, list):
        return [int(x) for x in explicit]
    if isinstance(explicit, str) and explicit.strip():
        return [int(x) for x in explicit.split(",") if x.strip()]

    layer_types = text.get("layer_types")
    if isinstance(layer_types, list) and any("moe" in str(x) for x in layer_types):
        return [i for i, layer_type in enumerate(layer_types) if "moe" in str(layer_type)]

    if family == "glm4_moe":
        first = _as_int(text.get("first_k_dense_replace"), 0) or 0
        return list(range(first, num_hidden_layers))
    if family == "step3p5":
        enum = text.get("moe_layers_enum")
        if isinstance(enum, list):
            return [int(x) for x in enum]
        if isinstance(enum, str) and enum.strip():
            return [int(x) for x in enum.split(",") if x.strip()]
        return list(range(1, num_hidden_layers))
    first_k_dense_replace = _as_int(text.get("first_k_dense_replace"))
    if first_k_dense_replace is not None and (
        text.get("num_experts") or text.get("n_routed_experts")
    ):
        return list(range(first_k_dense_replace, num_hidden_layers))
    if family == "minimax_m2":
        return list(range(num_hidden_layers))
    if family in {"qwen3_5_moe_text", "qwen3_next"}:
        if text.get("num_experts") or text.get("moe_intermediate_size"):
            return list(range(num_hidden_layers))
    return []


def load_model_spec(model_dir: Path) -> ModelSpec:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} does not exist")
    config = _read_json(config_path)
    text = _text_config(config)
    model_type = str(text.get("model_type") or config.get("model_type") or "")
    family = _architecture_family(model_type)
    quant = _parse_quant(model_dir, config, text)

    hidden_size = _as_int(text.get("hidden_size"))
    if hidden_size is None:
        raise ValueError("Unable to infer hidden_size from config.")
    num_hidden_layers = _get_num_hidden_layers(text)
    layer_types = text.get("layer_types")
    if not isinstance(layer_types, list):
        layer_types = ["full_attention"] * num_hidden_layers
    layer_types = [str(x) for x in layer_types[:num_hidden_layers]]

    intermediate_size = _as_int(text.get("intermediate_size"))
    moe_intermediate_size = _as_int(text.get("moe_intermediate_size"))
    shared_expert_intermediate_size = _as_int(
        text.get("shared_expert_intermediate_size")
    )
    if shared_expert_intermediate_size is None:
        shared_expert_intermediate_size = _as_int(text.get("share_expert_dim"))
    if family == "glm4_moe":
        n_shared = _as_int(text.get("n_shared_experts"))
        if n_shared and moe_intermediate_size:
            shared_expert_intermediate_size = moe_intermediate_size * n_shared
    if family == "minimax_m2" and moe_intermediate_size is None:
        moe_intermediate_size = intermediate_size

    num_experts = _as_int(
        text.get("num_experts")
        or text.get("n_routed_experts")
        or text.get("num_local_experts")
        or text.get("moe_num_experts")
    )
    top_k = _as_int(
        text.get("num_experts_per_tok")
        or text.get("num_experts_per_token")
        or text.get("moe_top_k")
        or text.get("top_k")
    )
    num_attention_heads = _as_int(text.get("num_attention_heads"))
    num_key_value_heads = _as_int(
        text.get("num_key_value_heads") or text.get("num_attention_groups")
    )
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    head_dim = _as_int(text.get("head_dim"))
    if head_dim is None and hidden_size and num_attention_heads:
        head_dim = hidden_size // num_attention_heads

    moe_layer_indices = _parse_moe_layers(family, text, num_hidden_layers)
    dense_layer_indices = [
        i for i in range(num_hidden_layers) if i not in set(moe_layer_indices)
    ]

    return ModelSpec(
        model_dir=model_dir,
        model_name=model_dir.name,
        model_type=model_type,
        architecture_family=family,
        raw_config=config,
        text_config=text,
        quant=quant,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_expert_intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_experts=num_experts,
        top_k=top_k,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        layer_types=layer_types,
        moe_layer_indices=moe_layer_indices,
        dense_layer_indices=dense_layer_indices,
        attention_other_setting=text.get("attention_other_setting"),
    )


def _layer_roots(spec: ModelSpec) -> list[str]:
    if spec.architecture_family in {"qwen3_5_text", "qwen3_5_moe_text",
                                    "qwen3_next", "step3p5"}:
        return ["model.language_model.layers", "model.layers"]
    return ["model.layers", "model.language_model.layers"]


def _alias_groups(aliases: list[str], fused_suffix: str | None = None,
                  moe_parent: bool = False) -> list[list[str]]:
    groups: list[list[str]] = []
    for alias in aliases:
        groups.append([alias])
        if fused_suffix and fused_suffix in PACKED_MODULES:
            base = alias.rsplit(".", 1)[0]
            groups.append([f"{base}.{part}" for part in PACKED_MODULES[fused_suffix]])
        if moe_parent:
            groups.append([
                f"{alias}.gate_proj",
                f"{alias}.up_proj",
                f"{alias}.down_proj",
            ])
            groups.append([f"{alias}.w1", f"{alias}.w3", f"{alias}.w2"])
    return groups


def _attention_dims(spec: ModelSpec, layer_idx: int) -> tuple[int, int, int]:
    heads = spec.num_attention_heads
    kv_heads = spec.num_key_value_heads
    head_dim = spec.head_dim
    other = spec.attention_other_setting
    if (
        spec.architecture_family == "step3p5"
        and isinstance(other, dict)
        and layer_idx < len(spec.layer_types)
        and spec.layer_types[layer_idx] == other.get("attention_type")
    ):
        heads = _as_int(other.get("num_attention_heads"), heads)
        kv_heads = _as_int(other.get("num_attention_groups"), kv_heads)
        head_dim = _as_int(other.get("head_dim"), head_dim)
    if heads is None or kv_heads is None or head_dim is None:
        raise ValueError("Attention dimensions are incomplete in config.")
    return heads, kv_heads, head_dim


def _is_attention_layer(spec: ModelSpec, layer_idx: int) -> bool:
    if layer_idx >= len(spec.layer_types):
        return True
    layer_type = spec.layer_types[layer_idx]
    if spec.architecture_family in {"qwen3_5_moe_text", "qwen3_next",
                                    "qwen3_5_text"}:
        return layer_type == "full_attention"
    return "attention" in layer_type


def build_dense_candidates(spec: ModelSpec) -> list[Candidate]:
    candidates: list[Candidate] = []
    roots = _layer_roots(spec)
    for layer_idx in range(spec.num_hidden_layers):
        if _is_attention_layer(spec, layer_idx):
            heads, kv_heads, head_dim = _attention_dims(spec, layer_idx)
            q_multiplier = 2 if (
                spec.architecture_family in {"qwen3_5_moe_text", "qwen3_next",
                                             "qwen3_5_text"}
                and bool(spec.text_config.get("attn_output_gate", True))
            ) else 1
            qkv_aliases = [
                f"{root}.{layer_idx}.self_attn.qkv_proj" for root in roots
            ]
            candidates.append(
                Candidate(
                    kind="dense",
                    op="qkv_proj",
                    layer_idx=layer_idx,
                    aliases=qkv_aliases,
                    prefix_groups=_alias_groups(qkv_aliases, "qkv_proj"),
                    size_kind="qkv",
                    attn_heads=heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    q_multiplier=q_multiplier,
                )
            )
            o_aliases = [f"{root}.{layer_idx}.self_attn.o_proj" for root in roots]
            candidates.append(
                Candidate(
                    kind="dense",
                    op="o_proj",
                    layer_idx=layer_idx,
                    aliases=o_aliases,
                    prefix_groups=_alias_groups(o_aliases),
                    size_kind="o_proj",
                    attn_heads=heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                )
            )

        if layer_idx in spec.dense_layer_indices and spec.intermediate_size:
            base_aliases = [f"{root}.{layer_idx}.mlp" for root in roots]
            _append_mlp_candidates(
                candidates, layer_idx, base_aliases, spec.intermediate_size
            )

        if (layer_idx in spec.moe_layer_indices
                and spec.shared_expert_intermediate_size):
            if spec.architecture_family == "glm4_moe":
                suffix = "mlp.shared_experts"
            elif spec.architecture_family == "step3p5":
                suffix = "share_expert"
            else:
                suffix = "mlp.shared_expert"
            base_aliases = [f"{root}.{layer_idx}.{suffix}" for root in roots]
            _append_mlp_candidates(
                candidates,
                layer_idx,
                base_aliases,
                spec.shared_expert_intermediate_size,
                shared=True,
            )
    return candidates


def _append_mlp_candidates(candidates: list[Candidate], layer_idx: int,
                           base_aliases: list[str], intermediate_size: int,
                           shared: bool = False) -> None:
    gate_aliases = [f"{base}.gate_up_proj" for base in base_aliases]
    candidates.append(
        Candidate(
            kind="dense",
            op="shared_gate_up_proj" if shared else "gate_up_proj",
            layer_idx=layer_idx,
            aliases=gate_aliases,
            prefix_groups=_alias_groups(gate_aliases, "gate_up_proj"),
            size_kind="gate_up",
            intermediate_size=intermediate_size,
        )
    )
    down_aliases = [f"{base}.down_proj" for base in base_aliases]
    candidates.append(
        Candidate(
            kind="dense",
            op="shared_down_proj" if shared else "down_proj",
            layer_idx=layer_idx,
            aliases=down_aliases,
            prefix_groups=_alias_groups(down_aliases),
            size_kind="down",
            intermediate_size=intermediate_size,
        )
    )


def build_moe_candidates(spec: ModelSpec) -> list[Candidate]:
    if not (spec.num_experts and spec.top_k and spec.moe_intermediate_size):
        return []
    roots = _layer_roots(spec)
    candidates: list[Candidate] = []
    for layer_idx in spec.moe_layer_indices:
        aliases: list[str] = []
        for root in roots:
            if spec.architecture_family == "minimax_m2":
                aliases.append(f"{root}.{layer_idx}.block_sparse_moe.experts")
                aliases.append(f"{root}.{layer_idx}.mlp.experts")
            elif spec.architecture_family == "step3p5":
                aliases.append(f"{root}.{layer_idx}.moe")
            else:
                aliases.append(f"{root}.{layer_idx}.mlp.experts")
        candidates.append(
            Candidate(
                kind="moe",
                op="experts",
                layer_idx=layer_idx,
                aliases=list(dict.fromkeys(aliases)),
                prefix_groups=_alias_groups(
                    list(dict.fromkeys(aliases)), moe_parent=True
                ),
                size_kind="moe",
            )
        )
    return candidates


def build_router_candidates(spec: ModelSpec) -> list[Candidate]:
    if not spec.num_experts:
        return []
    roots = _layer_roots(spec)
    candidates: list[Candidate] = []
    for layer_idx in spec.moe_layer_indices:
        if spec.architecture_family == "step3p5":
            suffix = "moe.gate"
        elif spec.architecture_family == "minimax_m2":
            suffix = "block_sparse_moe.gate"
        else:
            suffix = "mlp.gate"
        aliases = [f"{root}.{layer_idx}.{suffix}" for root in roots]
        candidates.append(
            Candidate(
                kind="router",
                op="router",
                layer_idx=layer_idx,
                aliases=aliases,
                prefix_groups=_alias_groups(aliases),
                size_kind="router",
            )
        )
    return candidates


def _normalize_prefix(prefix: str) -> str:
    return prefix.replace("model.language_model.", "model.")


def _regex_or_exact(value: str, target: str) -> bool:
    if target.startswith("re:"):
        return re.match(target[3:], value) is not None
    return value == target


def _prefix_matches_ignore_target(prefix: str, target: str) -> bool:
    normalized = _normalize_prefix(prefix)
    candidates = [prefix, normalized]
    if target.startswith("re:"):
        return any(_regex_or_exact(value, target) for value in candidates)

    target = target.rstrip(".")
    if not target:
        return False
    for value in candidates:
        if (
            value == target
            or value.startswith(target + ".")
            or fnmatch.fnmatch(value, target)
        ):
            return True
        parts = value.split(".")
        for idx in range(len(parts)):
            suffix = ".".join(parts[idx:])
            if (
                suffix == target
                or suffix.startswith(target + ".")
                or fnmatch.fnmatch(suffix, target)
            ):
                return True
    return False


def _candidate_group_ignored(candidate: Candidate, ignore: list[str]) -> bool:
    for group in candidate.prefix_groups:
        if all(
            any(_prefix_matches_ignore_target(prefix, target)
                for target in ignore)
            for prefix in group
        ):
            return True
    return False


def _awq_ignored(candidate: Candidate, ignore: list[str]) -> bool:
    return _candidate_group_ignored(candidate, ignore)


def _ct_ignored(candidate: Candidate, ignore: list[str]) -> bool:
    return _candidate_group_ignored(candidate, ignore)


def _modelopt_ignored(candidate: Candidate, ignore: list[str]) -> bool:
    return _candidate_group_ignored(candidate, ignore)


def _modelopt_prefix_ignored(prefix: str, ignore: list[str]) -> bool:
    for target in ignore:
        if _prefix_matches_ignore_target(prefix, target):
            return True
    return False


def _all_prefixes(candidate: Candidate) -> list[str]:
    prefixes: list[str] = []
    for group in candidate.prefix_groups:
        prefixes.extend(group)
    return list(dict.fromkeys(prefixes))


_CT_DENSE_TARGETS = {
    "Linear",
    "ColumnParallelLinear",
    "QKVParallelLinear",
    "MergedColumnParallelLinear",
    "RowParallelLinear",
    "ReplicatedLinear",
}

_CT_MOE_TARGETS = {"FusedMoE", "FusedMoEGroupedGEMM", "MoE"}


def _ct_prefix_matches_target(prefix: str, target: str) -> bool:
    return _regex_or_exact(prefix, target) or _regex_or_exact(
        _normalize_prefix(prefix), target
    )


def _ct_group_match_rank(candidate: Candidate, targets: list[str]) -> int | None:
    has_prefix_match = False
    for group in candidate.prefix_groups:
        if all(any(_ct_prefix_matches_target(prefix, target)
                   for target in targets) for prefix in group):
            has_prefix_match = True
            break
    if has_prefix_match:
        return 0
    if candidate.kind == "moe" and any(target in _CT_MOE_TARGETS for target in targets):
        return 1
    if candidate.kind == "moe" and any(target in _CT_DENSE_TARGETS for target in targets):
        return 2
    if candidate.kind == "dense" and any(target in _CT_DENSE_TARGETS for target in targets):
        return 1
    return None


def _ct_quant_for_candidate(candidate: Candidate,
                            quant: QuantInfo) -> QuantInfo | None:
    groups = quant.raw.get("config_groups")
    if not isinstance(groups, dict) or not groups:
        return quant

    best_rank: int | None = None
    best_group: dict[str, Any] | None = None
    for group_config in groups.values():
        if not isinstance(group_config, dict):
            continue
        targets = [str(x) for x in group_config.get("targets") or []]
        if not targets:
            continue
        rank = _ct_group_match_rank(candidate, targets)
        if rank is None:
            continue
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_group = group_config

    if best_group is not None:
        group_config = best_group

        weights = dict(group_config.get("weights") or {})
        bits = _as_int(weights.get("num_bits") or weights.get("bits"), quant.bits)
        group_size = _as_int(weights.get("group_size"), quant.group_size)
        fmt = group_config.get("format") or group_config.get("fmt") or quant.format
        return QuantInfo(
            method=quant.method,
            raw=group_config,
            bits=bits,
            group_size=group_size,
            zero_point=bool(weights.get("zero_point", quant.zero_point)),
            ignore=quant.ignore,
            quant_algo=quant.quant_algo,
            quantized_layers=quant.quantized_layers,
            format=str(fmt) if fmt is not None else None,
            weight_args=weights,
        )

    return None


def _is_ignored(candidate: Candidate, quant: QuantInfo) -> bool:
    if not quant.ignore:
        return False
    if quant.method in {"awq", "awq_marlin"}:
        return _awq_ignored(candidate, quant.ignore)
    if quant.method == "compressed-tensors":
        return _ct_ignored(candidate, quant.ignore)
    if quant.method == "modelopt":
        return _modelopt_ignored(candidate, quant.ignore)
    return _awq_ignored(candidate, quant.ignore)


def _choose_layer_key(candidate: Candidate, evidence: IndexEvidence) -> str:
    if evidence.available:
        for group in candidate.prefix_groups:
            if evidence.has_any_for_group(group):
                return group[0]
    return candidate.canonical_key


def _resolve_modelopt_mixed_algo(candidate: Candidate,
                                 quant: QuantInfo) -> tuple[str | None, str]:
    if not quant.quantized_layers:
        return None, ""
    for group in candidate.prefix_groups:
        algos: set[str] = set()
        for prefix in group:
            info = quant.quantized_layers.get(prefix)
            if info is not None:
                algos.add(str(info.get("quant_algo", "")).upper())
                continue
            prefix_dot = prefix + "."
            for key, child_info in quant.quantized_layers.items():
                if key.startswith(prefix_dot):
                    algos.add(str(child_info.get("quant_algo", "")).upper())
        algos.discard("")
        if len(algos) == 1:
            return next(iter(algos)), ""
        if len(algos) > 1:
            return None, f"skipped_conflicting_quant:{sorted(algos)}"
    return None, "modelopt_mixed_bf16_module"


def _configured_quant_format(quant: QuantInfo,
                             algo_override: str | None = None) -> QuantDecision:
    method = quant.method
    algo = (algo_override or quant.quant_algo or "").upper()
    weight_type = str(quant.weight_args.get("type", "")).lower()
    bits = quant.bits
    fmt = (quant.format or "").lower()

    if method in {"awq", "awq_marlin"}:
        if bits == 8:
            qfmt = "uint8"
        else:
            qfmt = "uint4"
        return QuantDecision(
            "actual_marlin",
            method,
            qfmt,
            quant.group_size,
            quant.zero_point,
            "awq_marlin_wna16",
        )
    if method in {"gptq", "gptq_marlin"}:
        return QuantDecision(
            "actual_marlin",
            method,
            "uint8b128" if bits == 8 else "uint4b8",
            quant.group_size,
            False,
            "wna16_marlin",
        )
    if method == "fp8" or algo == "FP8":
        return QuantDecision(
            "actual_marlin",
            method,
            f"fp8_{quant.format}" if quant.format else "fp8_e4m3",
            quant.group_size,
            False,
            "fp8_marlin",
        )
    if method == "modelopt":
        if "MXFP4" in algo:
            return QuantDecision(
                "actual_marlin",
                method,
                "mxfp4",
                quant.group_size or 32,
                False,
                "fp4_marlin",
            )
        if algo == "NVFP4" or "NVFP4" in algo or "FP4" in algo:
            return QuantDecision(
                "actual_marlin",
                method,
                "nvfp4",
                quant.group_size or 16,
                False,
                "fp4_marlin",
            )
        if algo == "FP8":
            return QuantDecision(
                "actual_marlin",
                method,
                "fp8_e4m3",
                quant.group_size,
                False,
                "fp8_marlin",
            )
    if method == "compressed-tensors":
        if "nvfp4" in fmt:
            return QuantDecision(
                "actual_marlin",
                method,
                "nvfp4",
                quant.group_size or 16,
                False,
                "fp4_marlin",
            )
        if "mxfp4" in fmt:
            return QuantDecision(
                "actual_marlin",
                method,
                "mxfp4",
                quant.group_size or 32,
                False,
                "fp4_marlin",
            )
        if weight_type == "float" and bits == 4:
            return QuantDecision(
                "actual_marlin",
                method,
                "nvfp4",
                quant.group_size or 16,
                False,
                "fp4_marlin",
            )
        if weight_type == "float" and bits == 8:
            return QuantDecision(
                "actual_marlin",
                method,
                "fp8_e4m3",
                quant.group_size,
                False,
                "fp8_marlin",
            )
        if weight_type == "int" or bits in {4, 8}:
            return QuantDecision(
                "actual_marlin",
                method,
                "uint8b128" if bits == 8 else "uint4b8",
                quant.group_size,
                False,
                "wna16_marlin",
            )
    return QuantDecision(
        "hypothetical_bf16",
        method,
        "bf16_or_fp16",
        None,
        False,
        "none",
        "hypothetical_bf16",
    )


def _bf16_decision(quant: QuantInfo, warning: str) -> QuantDecision:
    return QuantDecision(
        "hypothetical_bf16",
        quant.method,
        "bf16_or_fp16",
        None,
        False,
        "none",
        warning,
    )


def _evidence_suffixes(candidate: Candidate, evidence: IndexEvidence) -> set[str]:
    suffixes: set[str] = set()
    for group in candidate.prefix_groups:
        suffixes.update(evidence.evidence_for_group(group))
    return suffixes


def _suffixes_satisfy_quant(qfmt: str, suffixes: set[str]) -> bool:
    has_awq = {"qweight", "qzeros", "scales"}.issubset(suffixes)
    has_weight = bool(
        suffixes
        & {
            "weight",
            "weight_packed",
            "w13_weight",
            "w2_weight",
            "w13_weight_packed",
            "w2_weight_packed",
        }
    )
    has_scale = bool(
        suffixes & {"weight_scale", "w13_weight_scale", "w2_weight_scale"}
    )
    has_global_scale = bool(
        suffixes
        & {
            "weight_scale_2",
            "weight_global_scale",
            "w13_weight_scale_2",
            "w2_weight_scale_2",
            "w13_weight_global_scale",
            "w2_weight_global_scale",
        }
    )
    has_nvfp4 = has_weight and has_scale and has_global_scale
    has_mxfp4 = has_weight and has_scale
    has_fp8 = (has_weight and has_scale) or "fp8_weight" in suffixes
    has_wna16 = (
        has_awq
        or "weight_packed" in suffixes
        or {"qweight", "scales"}.issubset(suffixes)
        or has_fp8
    )
    if qfmt in {"uint4", "uint8"}:
        return has_awq
    if qfmt in {"uint4b8", "uint8b128"}:
        return has_wna16
    if qfmt == "nvfp4":
        return has_nvfp4
    if qfmt == "mxfp4":
        return has_mxfp4
    if qfmt.startswith("fp8"):
        return has_fp8
    return False


def _prefix_is_plain_weight(suffixes: set[str]) -> bool:
    return "weight" in suffixes and not (suffixes & QUANT_SUFFIXES)


def _group_evidence_status(
    candidate: Candidate, evidence: IndexEvidence, qfmt: str
) -> str:
    """Return actual/plain/disagrees/none for a candidate's index evidence.

    Fused groups such as qkv_proj only count as actual when every shard has
    compatible quantization evidence. A union of q/k/v suffixes is not enough.
    """
    found_any = False
    found_plain = False
    found_disagreement = False

    for group in candidate.prefix_groups:
        suffixes_by_prefix = [evidence.suffixes_for_prefix(p) for p in group]
        if not any(suffixes_by_prefix):
            continue
        found_any = True

        if len(group) > 1 and not all(suffixes_by_prefix):
            found_disagreement = True
            continue

        if all(_suffixes_satisfy_quant(qfmt, s) for s in suffixes_by_prefix):
            return "actual"
        if all(_prefix_is_plain_weight(s) for s in suffixes_by_prefix):
            found_plain = True
        else:
            found_disagreement = True

    if found_disagreement:
        return "disagrees"
    if found_plain:
        return "plain"
    if found_any:
        return "disagrees"
    return "none"


def decide_quant(candidate: Candidate, spec: ModelSpec,
                 evidence: IndexEvidence) -> QuantDecision:
    quant = spec.quant
    if _is_ignored(candidate, quant):
        return QuantDecision(
            "skipped",
            quant.method,
            "excluded",
            None,
            False,
            "none",
            "excluded_quant_module",
        )
    if quant.is_unquantized:
        return _bf16_decision(quant, "hypothetical_bf16")

    algo_override: str | None = None
    if quant.method == "modelopt" and quant.quant_algo == "MIXED_PRECISION":
        algo_override, warning = _resolve_modelopt_mixed_algo(candidate, quant)
        if warning.startswith("skipped_conflicting_quant"):
            return QuantDecision(
                "skipped",
                quant.method,
                "conflicting",
                None,
                False,
                "none",
                warning,
            )
        if algo_override is None:
            return _bf16_decision(quant, warning or "modelopt_mixed_bf16_module")

    configured_quant = quant
    if quant.method == "compressed-tensors":
        ct_quant = _ct_quant_for_candidate(candidate, quant)
        if ct_quant is None:
            return _bf16_decision(quant, "hypothetical_bf16")
        configured_quant = ct_quant

    configured = _configured_quant_format(configured_quant, algo_override)
    if configured.call_status != "actual_marlin":
        return configured

    if not evidence.available:
        configured.warning = "config_derived_quant"
        return configured

    evidence_status = _group_evidence_status(
        candidate, evidence, configured.quant_format
    )
    if evidence_status == "actual":
        return configured
    if evidence_status == "plain":
        return _bf16_decision(quant, "index_disagrees_with_config")
    if evidence_status == "disagrees":
        return _bf16_decision(quant, "index_disagrees_with_config")
    configured.warning = "config_derived_quant"
    return configured


def _dense_shape(candidate: Candidate, spec: ModelSpec,
                 tp_size: int) -> tuple[int, int]:
    if candidate.size_kind == "qkv":
        assert candidate.attn_heads and candidate.kv_heads and candidate.head_dim
        q_heads = candidate.attn_heads // tp_size
        kv_heads = (
            candidate.kv_heads // tp_size
            if candidate.kv_heads >= tp_size else 1
        )
        q_size = q_heads * candidate.head_dim
        kv_size = kv_heads * candidate.head_dim
        return spec.hidden_size, candidate.q_multiplier * q_size + 2 * kv_size
    if candidate.size_kind == "o_proj":
        assert candidate.attn_heads and candidate.head_dim
        return (candidate.attn_heads // tp_size) * candidate.head_dim, spec.hidden_size
    if candidate.size_kind == "gate_up":
        assert candidate.intermediate_size
        return spec.hidden_size, 2 * (candidate.intermediate_size // tp_size)
    if candidate.size_kind == "down":
        assert candidate.intermediate_size
        return candidate.intermediate_size // tp_size, spec.hidden_size
    raise ValueError(f"Unknown dense size kind: {candidate.size_kind}")


def _effective_group_size(group_size: int | str | None, size_k: int) -> int | str | None:
    if group_size == -1:
        return size_k
    return group_size


def _target_op_for_decision(kind: str, decision: QuantDecision) -> str:
    if decision.call_status != "actual_marlin" or decision.marlin_path == "none":
        return "none"
    if kind == "dense":
        return "ops.marlin_gemm"
    if kind == "moe":
        return "ops.moe_wna16_marlin_gemm"
    return "none"


def _moe_block_size(m: int, top_k: int, local_num_experts: int,
                    input_itemsize: int = 2) -> int:
    block_size_m = 64
    for candidate in [8, 16, 32, 48, 64]:
        block_size_m = candidate
        if m * top_k / local_num_experts / candidate < 0.9:
            break
    if input_itemsize == 1:
        block_size_m = max(block_size_m, 16)
    return block_size_m


def _phases(max_tokens: list[int], decode_concurrency: list[int]
            ) -> list[tuple[str, int | None, int | None, int]]:
    phases: list[tuple[str, int | None, int | None, int]] = []
    for m in max_tokens:
        phases.append(("prefill", m, None, m))
    for c in decode_concurrency:
        phases.append(("decode", None, c, c))
    return phases


def _candidate_rows_label(layer_keys: list[str]) -> str:
    unique = list(dict.fromkeys(layer_keys))
    if len(unique) <= 3:
        return ";".join(unique)
    return f"{unique[0]};{unique[1]};...(+{len(unique) - 2} more)"


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: dict[tuple[Any, ...], dict[str, Any]] = {}
    skip = {"layer_key", "layer_keys", "description"}
    for row in rows:
        key = tuple((k, json.dumps(v, sort_keys=True))
                    for k, v in sorted(row.items()) if k not in skip)
        if key not in aggregated:
            new_row = dict(row)
            new_row["layer_keys"] = [row["layer_key"]]
            new_row["call_count"] = 1
            aggregated[key] = new_row
        else:
            item = aggregated[key]
            item["layer_keys"].append(row["layer_key"])
            item["call_count"] += 1
    result: list[dict[str, Any]] = []
    for row in aggregated.values():
        row["layer_keys"] = list(dict.fromkeys(row["layer_keys"]))
        row["layer_key"] = _candidate_rows_label(row["layer_keys"])
        result.append(row)
    return sorted(result, key=lambda r: (
        r.get("scenario", ""),
        r.get("phase", ""),
        r.get("op", ""),
        r.get("size_m", 0),
        str(r.get("layer_key", "")),
    ))


def enumerate_dense_rows(spec: ModelSpec, evidence: IndexEvidence,
                         scenarios: list[Scenario], max_tokens: list[int],
                         decode_concurrency: list[int],
                         include_skipped: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if include_skipped:
        for candidate in build_router_candidates(spec):
            rows.append({
                "model": spec.model_name,
                "scenario": "n/a",
                "phase": "skipped_router",
                "max_num_batched_tokens": None,
                "decode_concurrency": None,
                "layer_key": _choose_layer_key(candidate, evidence),
                "op": candidate.op,
                "target_op": "none",
                "size_m": None,
                "size_n": None,
                "size_k": None,
                "group_size": None,
                "quant_method": spec.quant.method,
                "quant_format": "router",
                "has_zp": False,
                "marlin_path": "none",
                "call_status": "skipped",
                "description": (
                    "skipped_router: router/gate is not a Marlin GEMM "
                    "benchmark target"
                ),
                "warning": "skipped_router",
            })
    for candidate in build_dense_candidates(spec):
        decision = decide_quant(candidate, spec, evidence)
        if decision.call_status == "skipped" and not include_skipped:
            continue
        layer_key = _choose_layer_key(candidate, evidence)
        for scenario in scenarios:
            try:
                size_k, size_n = _dense_shape(candidate, spec, scenario.tp_size)
            except Exception as exc:
                if include_skipped:
                    rows.append({
                        "model": spec.model_name,
                        "scenario": scenario.label,
                        "phase": "shape_error",
                        "layer_key": layer_key,
                        "op": candidate.op,
                        "target_op": "none",
                        "size_m": None,
                        "size_n": None,
                        "size_k": None,
                        "quant_method": spec.quant.method,
                        "quant_format": "unknown",
                        "group_size": None,
                        "has_zp": False,
                        "marlin_path": "none",
                        "call_status": "skipped",
                        "description": f"shape_error: {exc}",
                        "warning": "shape_error",
                    })
                continue
            for phase, max_num_tokens, concurrency, m in _phases(
                max_tokens, decode_concurrency
            ):
                rows.append({
                    "model": spec.model_name,
                    "scenario": scenario.label,
                    "phase": phase,
                    "max_num_batched_tokens": max_num_tokens,
                    "decode_concurrency": concurrency,
                    "layer_key": layer_key,
                    "op": candidate.op,
                    "target_op": _target_op_for_decision("dense", decision),
                    "size_m": m,
                    "size_n": size_n,
                    "size_k": size_k,
                    "group_size": _effective_group_size(
                        decision.group_size, size_k
                    ),
                    "quant_method": decision.quant_method,
                    "quant_format": decision.quant_format,
                    "has_zp": decision.has_zp,
                    "marlin_path": decision.marlin_path,
                    "call_status": decision.call_status,
                    "description": (
                        f"{decision.call_status}: {candidate.op}, "
                        f"{scenario.label}, {phase} M={m}; "
                        f"N={size_n}; K={size_k}"
                    ),
                    "warning": decision.warning,
                })
    return aggregate_rows(rows)


def enumerate_moe_rows(spec: ModelSpec, evidence: IndexEvidence,
                       scenarios: list[Scenario], max_tokens: list[int],
                       decode_concurrency: list[int],
                       include_skipped: bool) -> list[dict[str, Any]]:
    if not (spec.moe_intermediate_size and spec.num_experts and spec.top_k):
        return []
    rows: list[dict[str, Any]] = []
    for candidate in build_moe_candidates(spec):
        decision = decide_quant(candidate, spec, evidence)
        if decision.call_status == "skipped" and not include_skipped:
            continue
        layer_key = _choose_layer_key(candidate, evidence)
        for scenario in scenarios:
            if scenario.enable_expert_parallel:
                local_num_experts = spec.num_experts // scenario.tp_size
                intermediate = spec.moe_intermediate_size
            else:
                local_num_experts = spec.num_experts
                intermediate = spec.moe_intermediate_size // scenario.tp_size
            for phase, max_num_tokens, concurrency, m in _phases(
                max_tokens, decode_concurrency
            ):
                block_size = _moe_block_size(m, spec.top_k, local_num_experts)
                for op in ["w13", "w2"]:
                    if op == "w13":
                        size_m = m
                        top_k = spec.top_k
                        size_n = 2 * intermediate
                        size_k = spec.hidden_size
                    else:
                        size_m = m * spec.top_k
                        top_k = 1
                        size_n = spec.hidden_size
                        size_k = intermediate
                    rows.append({
                        "model": spec.model_name,
                        "scenario": scenario.label,
                        "phase": phase,
                        "max_num_batched_tokens": max_num_tokens,
                        "decode_concurrency": concurrency,
                        "layer_key": layer_key,
                        "op": op,
                        "target_op": _target_op_for_decision("moe", decision),
                        "moe_block_size": block_size,
                        "top_k": top_k,
                        "size_m": size_m,
                        "size_n": size_n,
                        "size_k": size_k,
                        "group_size": _effective_group_size(
                            decision.group_size, size_k
                        ),
                        "quant_method": decision.quant_method,
                        "quant_format": decision.quant_format,
                        "has_zp": decision.has_zp,
                        "marlin_path": decision.marlin_path,
                        "local_num_experts": local_num_experts,
                        "global_num_experts": spec.num_experts,
                        "intermediate_size_per_partition": intermediate,
                        "call_status": decision.call_status,
                        "description": (
                            f"{decision.call_status}: routed MoE {op}, "
                            f"{scenario.label}, {phase} M={m}; "
                            f"N={size_n}; K={size_k}; block={block_size}"
                        ),
                        "warning": decision.warning,
                    })
    return aggregate_rows(rows)


def load_index(model_dir: Path, verify: bool) -> IndexEvidence:
    if not verify:
        return IndexEvidence(None)
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        return IndexEvidence(None)
    data = _read_json(index_path)
    weight_map = data.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return IndexEvidence(set())
    keys = set(str(key) for key in weight_map)
    return IndexEvidence(keys, _load_safetensors_dtypes(model_dir, weight_map))


def _load_safetensors_dtypes(
    model_dir: Path, weight_map: dict[str, Any]
) -> dict[str, str]:
    try:
        from safetensors import safe_open
    except Exception:
        return {}

    dtypes: dict[str, str] = {}
    keys_by_file: dict[str, list[str]] = {}
    for key, filename in weight_map.items():
        if not isinstance(key, str) or not isinstance(filename, str):
            continue
        if filename.endswith(".safetensors"):
            keys_by_file.setdefault(filename, []).append(key)

    for filename, keys in keys_by_file.items():
        path = model_dir / filename
        if not path.exists():
            continue
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for key in keys:
                    if key in available:
                        dtypes[key] = str(handle.get_slice(key).get_dtype())
        except Exception:
            continue
    return dtypes


def _scenarios(tp_sizes: list[int], ep_modes: list[str]) -> list[Scenario]:
    out: list[Scenario] = []
    include_tp = "tp" in ep_modes
    include_ep = "tp_ep" in ep_modes or "ep" in ep_modes
    for tp in tp_sizes:
        if include_tp:
            out.append(Scenario(tp, False))
        if include_ep:
            out.append(Scenario(tp, True))
    return out


def _warnings(rows: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for row in rows:
        warning = row.get("warning")
        if warning:
            warnings.append(str(warning))
    return sorted(set(warnings))


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"--model must be a local directory: {model_dir}")
    spec = load_model_spec(model_dir)
    evidence = load_index(model_dir, args.verify_safetensors_index)
    scenarios = _scenarios(
        _parse_int_list(args.tp_sizes, [4, 8]),
        args.ep_modes.split(",") if isinstance(args.ep_modes, str)
        else args.ep_modes,
    )
    max_tokens = _parse_int_list(args.max_num_batched_tokens, [2048, 4096])
    decode_concurrency = _parse_int_list(args.decode_concurrency, [1, 32, 64])
    dense = enumerate_dense_rows(
        spec, evidence, scenarios, max_tokens, decode_concurrency,
        args.include_skipped
    )
    moe = enumerate_moe_rows(
        spec, evidence, scenarios, max_tokens, decode_concurrency,
        args.include_skipped
    )
    return {
        "model": str(model_dir),
        "model_config": {
            "model_type": spec.model_type,
            "architecture_family": spec.architecture_family,
            "hidden_size": spec.hidden_size,
            "intermediate_size": spec.intermediate_size,
            "moe_intermediate_size": spec.moe_intermediate_size,
            "shared_expert_intermediate_size": (
                spec.shared_expert_intermediate_size
            ),
            "num_experts": spec.num_experts,
            "top_k": spec.top_k,
            "num_hidden_layers": spec.num_hidden_layers,
            "moe_layer_indices": spec.moe_layer_indices,
        },
        "quantization": {
            "quant_method": spec.quant.method,
            "quant_algo": spec.quant.quant_algo,
            "quant_format": _configured_quant_format(spec.quant).quant_format,
            "group_size": spec.quant.group_size,
            "zero_point": spec.quant.zero_point,
        },
        "shape_inputs": {
            "max_num_batched_tokens": max_tokens,
            "decode_concurrency": decode_concurrency,
        },
        "scenarios": [
            {
                "tp_size": s.tp_size,
                "enable_expert_parallel": s.enable_expert_parallel,
            } for s in scenarios
        ],
        "dense": dense,
        "moe": moe,
        "warnings": _warnings(dense + moe),
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def print_table(title: str, rows: list[dict[str, Any]],
                columns: list[str]) -> None:
    print(title)
    if not rows:
        print("  (no rows)")
        print()
        return
    widths = {
        col: max(len(col), *(len(_format_value(row.get(col))) for row in rows))
        for col in columns
    }
    header = "  " + "  ".join(col.ljust(widths[col]) for col in columns)
    sep = "  " + "  ".join("-" * widths[col] for col in columns)
    print(header)
    print(sep)
    for row in rows:
        print("  " + "  ".join(
            _format_value(row.get(col)).ljust(widths[col]) for col in columns
        ))
    print()


def print_pretty(payload: dict[str, Any]) -> None:
    print(f"Model: {payload['model']}")
    print(
        "Config: "
        f"{payload['model_config']['architecture_family']} "
        f"hidden={payload['model_config']['hidden_size']} "
        f"moe_intermediate={payload['model_config']['moe_intermediate_size']} "
        f"experts={payload['model_config']['num_experts']} "
        f"top_k={payload['model_config']['top_k']}"
    )
    print(f"Warnings: {', '.join(payload['warnings']) or '-'}")
    print()
    dense_columns = [
        "scenario", "phase", "layer_key", "op", "target_op", "size_m",
        "size_n", "size_k", "group_size", "quant_method", "quant_format",
        "has_zp", "marlin_path", "call_status", "call_count", "warning",
    ]
    moe_columns = [
        "scenario", "phase", "layer_key", "op", "target_op",
        "moe_block_size", "top_k", "size_m", "size_n", "size_k", "group_size",
        "quant_method", "quant_format", "has_zp", "marlin_path",
        "local_num_experts", "global_num_experts",
        "intermediate_size_per_partition", "call_status", "call_count",
        "warning",
    ]
    print_table("Dense table", payload["dense"], dense_columns)
    print_table("MoE table", payload["moe"], moe_columns)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate Marlin GEMM shapes from a local model directory."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-num-batched-tokens", action="append")
    parser.add_argument("--decode-concurrency", action="append")
    parser.add_argument("--tp-sizes", action="append")
    parser.add_argument("--ep-modes", default="tp,tp_ep")
    parser.add_argument("--moe-backend", default="marlin")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--format", choices=["pretty", "json", "both"],
                        default="both")
    parser.add_argument("--json-out")
    parser.add_argument("--include-skipped", action="store_true")
    parser.add_argument("--verify-safetensors-index", action=argparse.BooleanOptionalAction,
                        default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.moe_backend != "marlin":
        print(
            "warning: this tool enumerates Marlin target shapes; "
            f"received --moe-backend={args.moe_backend!r}",
            file=sys.stderr,
        )
    payload = build_payload(args)
    if args.format in {"pretty", "both"}:
        print_pretty(payload)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.format in {"json", "both"}:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
