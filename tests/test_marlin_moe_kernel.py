from __future__ import annotations

import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from tests.helpers import make_moe_routing_tensors
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.quantization.awq_marlin import (
    AWQMarlinConfig,
    AWQMarlinMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsWNA16MarlinMoEMethod,
)
from vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig,
    GPTQMarlinMoEMethod,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkW8A8Fp8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_c_tmp,
)
from vllm.scalar_type import scalar_types
from tests.writeback_marlin_cases import (
    MOE_WRITEBACK_CLASS_CASES,
    MOE_WRITEBACK_CLASS_CASE_BY_NAME,
    MoeWritebackMatrixCase,
    iter_moe_writeback_matrix,
)


_MOE_METHOD_CLASS_TEST_COVERAGE = {
    "gptq_moe",
    "awq_moe",
    "compressed_tensors_wna16_moe",
    "quark_w8a8_fp8_moe",
    "compressed_tensors_w8a8_fp8_moe",
    "compressed_tensors_w4a4_nvfp4_moe",
    "compressed_tensors_w4a4_mxfp4_moe",
}
_MOE_FULL_MATRIX_JSONL = Path(
    "benchmarks/results/20260604_test_moe_writeback_full_matrix.jsonl"
)
_MOE_FULL_MATRIX_SUMMARY = Path(
    "benchmarks/results/20260604_test_moe_writeback_full_matrix_summary.json"
)


def _counter_to_json(counter: Counter[Any]) -> dict[str, int]:
    return {
        str(key): value
        for key, value in sorted(counter.items(), key=lambda item: str(item[0]))
    }


def test_moe_writeback_class_inventory_has_test_coverage() -> None:
    assert {case.name for case in MOE_WRITEBACK_CLASS_CASES} == (
        _MOE_METHOD_CLASS_TEST_COVERAGE
    )


@dataclass
class _RecordedGemm:
    c_tmp: torch.Tensor
    b_qzeros: torch.Tensor | None
    size_m: int
    size_n: int
    size_k: int
    top_k: int
    is_zp_float: bool


@dataclass
class _CachedMoeMatrixCase:
    method: Any
    layer: torch.nn.Module
    expected_zp_float: bool
    estimated_bytes: int


def _make_minimal_moe_layer(
    *,
    num_experts: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 16,
    num_bits: int = 4,
    group_size: int = -1,
    prefix: str = "gptq",
) -> torch.nn.Module:
    pack_factor = 32 // num_bits
    num_groups_w13 = 1 if group_size == -1 else hidden_size // group_size
    num_groups_w2 = 1 if group_size == -1 else intermediate_size // group_size
    layer = torch.nn.Module()
    layer.apply_router_weight_on_input = False
    layer.global_num_experts = num_experts
    layer.expert_map = None
    layer.activation = MoEActivation.SILU
    layer.intermediate_size_per_partition = intermediate_size
    layer.num_groups_w13 = num_groups_w13
    layer.num_groups_w2 = num_groups_w2

    if prefix in {"gptq", "awq"}:
        layer.register_parameter(
            "w13_qweight",
            torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size // 16,
                    2 * intermediate_size * (num_bits // 2),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_qweight",
            torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    intermediate_size // 16,
                    hidden_size * (num_bits // 2),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w13_scales",
            torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    num_groups_w13,
                    2 * intermediate_size,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_scales",
            torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    num_groups_w2,
                    hidden_size,
                    dtype=torch.float16,
                ),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w13_g_idx",
            torch.nn.Parameter(
                torch.empty(num_experts, 0, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_g_idx",
            torch.nn.Parameter(
                torch.empty(num_experts, 0, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w13_g_idx_sort_indices",
            torch.nn.Parameter(
                torch.empty(num_experts, 0, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "w2_g_idx_sort_indices",
            torch.nn.Parameter(
                torch.empty(num_experts, 0, dtype=torch.int32),
                requires_grad=False,
            ),
        )
        if prefix == "awq":
            layer.register_parameter(
                "w13_qzeros",
                torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        num_groups_w13,
                        2 * intermediate_size // pack_factor,
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                ),
            )
            layer.register_parameter(
                "w2_qzeros",
                torch.nn.Parameter(
                    torch.empty(
                        num_experts,
                        num_groups_w2,
                        hidden_size // pack_factor,
                        dtype=torch.int32,
                    ),
                    requires_grad=False,
                ),
            )
        return layer

    layer.register_parameter(
        "w13_weight_packed",
        torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // 16,
                2 * intermediate_size * (num_bits // 2),
                dtype=torch.int32,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight_packed",
        torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size // 16,
                hidden_size * (num_bits // 2),
                dtype=torch.int32,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                2 * intermediate_size,
                dtype=torch.float16,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w2,
                hidden_size,
                dtype=torch.float16,
            ),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_g_idx",
        torch.nn.Parameter(torch.empty(num_experts, 0, dtype=torch.int32), requires_grad=False),
    )
    layer.register_parameter(
        "w2_weight_g_idx",
        torch.nn.Parameter(torch.empty(num_experts, 0, dtype=torch.int32), requires_grad=False),
    )
    layer.register_parameter(
        "w13_g_idx_sort_indices",
        torch.nn.Parameter(torch.empty(num_experts, 0, dtype=torch.int32), requires_grad=False),
    )
    layer.register_parameter(
        "w2_g_idx_sort_indices",
        torch.nn.Parameter(torch.empty(num_experts, 0, dtype=torch.int32), requires_grad=False),
    )
    return layer


def _make_minimal_quark_fp8_moe_layer(
    *,
    num_experts: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 16,
) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.apply_router_weight_on_input = False
    layer.global_num_experts = num_experts
    layer.expert_map = None
    layer.activation = MoEActivation.SILU
    layer.num_experts = num_experts
    layer.local_num_experts = num_experts
    layer.hidden_size = hidden_size
    layer.intermediate_size_per_partition = intermediate_size
    layer.weight_block_size = None
    layer.orig_dtype = torch.float16
    layer.w13_input_scale = None
    layer.w2_input_scale = None
    layer.w13_bias = None
    layer.w2_bias = None

    layer.register_parameter(
        "w13_weight",
        torch.nn.Parameter(
            torch.empty(num_experts, 2 * intermediate_size, hidden_size),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight",
        torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, intermediate_size),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_weight_scale",
        torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size, dtype=torch.float32),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_weight_scale",
        torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        ),
    )
    return layer


def _install_fused_marlin_cpu_mocks(monkeypatch: pytest.MonkeyPatch) -> list[_RecordedGemm]:
    import vllm.model_executor.layers.fused_moe.fused_marlin_moe as fused_mod

    records: list[_RecordedGemm] = []

    def fake_align(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
        expert_map: torch.Tensor | None = None,
        ignore_invalid_experts: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del block_size, num_experts, expert_map, ignore_invalid_experts
        return (
            torch.arange(topk_ids.numel(), dtype=torch.int32, device=topk_ids.device),
            topk_ids.reshape(-1).to(torch.int32).contiguous(),
            torch.tensor([topk_ids.numel()], dtype=torch.int32, device=topk_ids.device),
        )

    def fake_gemm(
        input: torch.Tensor,
        output: torch.Tensor | None,
        b_qweight: torch.Tensor,
        b_bias: torch.Tensor | None,
        b_scales: torch.Tensor,
        a_scales: torch.Tensor | None,
        global_scale: torch.Tensor | None,
        b_qzeros: torch.Tensor | None,
        g_idx: torch.Tensor | None,
        perm: torch.Tensor | None,
        c_tmp: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_past_padded: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        moe_block_size: int,
        top_k: int,
        mul_topk_weights: bool,
        b_q_type: Any,
        size_m: int,
        size_n: int,
        size_k: int,
        is_k_full: bool,
        use_atomic_add: bool,
        use_fp32_reduce: bool,
        is_zp_float: bool,
        thread_k: int = -1,
        thread_n: int = -1,
        blocks_per_sm: int = -1,
    ) -> torch.Tensor:
        del (
            input,
            b_qweight,
            b_bias,
            b_scales,
            a_scales,
            global_scale,
            g_idx,
            perm,
            sorted_token_ids,
            expert_ids,
            num_tokens_past_padded,
            topk_weights,
            moe_block_size,
            mul_topk_weights,
            b_q_type,
            is_k_full,
            use_atomic_add,
            use_fp32_reduce,
            thread_k,
            thread_n,
            blocks_per_sm,
        )
        records.append(
            _RecordedGemm(
                c_tmp=c_tmp,
                b_qzeros=b_qzeros,
                size_m=size_m,
                size_n=size_n,
                size_k=size_k,
                top_k=top_k,
                is_zp_float=is_zp_float,
            )
        )
        output_rows = size_m * top_k
        if output is None:
            output = torch.empty(output_rows, size_n, dtype=torch.float16)
        else:
            output = output.view(output_rows, size_n)
        output.zero_()
        return output

    monkeypatch.setattr(fused_mod, "moe_align_block_size", fake_align)
    monkeypatch.setattr(fused_mod.ops, "moe_wna16_marlin_gemm", fake_gemm)
    return records


def _gptq_config_for_quant_name(quant_name: str) -> GPTQMarlinConfig:
    if quant_name == "uint4b8":
        weight_bits = 4
    elif quant_name == "uint8b128":
        weight_bits = 8
    else:
        raise ValueError(f"Unsupported GPTQ MoE quant_name={quant_name!r}")
    return GPTQMarlinConfig(
        weight_bits=weight_bits,
        group_size=-1,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )


def _patch_moe_post_load_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    class_name: str,
    num_bits: int,
) -> None:
    del num_bits

    def fake_repack(weight: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        return weight.detach().clone().contiguous()

    def fake_permute_scales(s: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return s.detach().clone().contiguous()

    def fake_awq_float_zp(
        q_zp_packed: torch.Tensor,
        scales: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        del q_zp_packed, args, kwargs
        return torch.zeros_like(scales, dtype=torch.float16).contiguous()

    if class_name == "gptq_moe":
        import vllm.model_executor.layers.quantization.gptq_marlin as gptq_mod

        monkeypatch.setattr(gptq_mod.ops, "gptq_marlin_moe_repack", fake_repack)
        monkeypatch.setattr(gptq_mod, "marlin_moe_permute_scales", fake_permute_scales)
    elif class_name == "awq_moe":
        import vllm.model_executor.layers.quantization.awq_marlin as awq_mod

        monkeypatch.setattr(awq_mod.ops, "awq_marlin_moe_repack", fake_repack)
        monkeypatch.setattr(awq_mod, "marlin_moe_permute_scales", fake_permute_scales)
        monkeypatch.setattr(
            awq_mod,
            "moe_awq_to_marlin_zero_points_float",
            fake_awq_float_zp,
        )
    elif class_name == "compressed_tensors_wna16_moe":
        import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe as ct_mod

        monkeypatch.setattr(ct_mod.ops, "gptq_marlin_moe_repack", fake_repack)
        monkeypatch.setattr(ct_mod, "marlin_moe_permute_scales", fake_permute_scales)
    else:
        raise AssertionError(f"Unhandled MoE class {class_name!r}")


def _install_all_moe_post_load_mocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_repack(weight: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        out = weight.detach()
        return out if out.is_contiguous() else out.contiguous()

    def fake_permute_scales(s: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        out = s.detach()
        return out if out.is_contiguous() else out.contiguous()

    def fake_awq_float_zp(
        q_zp_packed: torch.Tensor,
        scales: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        del q_zp_packed, args, kwargs
        return torch.zeros_like(scales, dtype=torch.float16).contiguous()

    import vllm.model_executor.layers.quantization.awq_marlin as awq_mod
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe as ct_mod
    import vllm.model_executor.layers.quantization.gptq_marlin as gptq_mod

    monkeypatch.setattr(gptq_mod.ops, "gptq_marlin_moe_repack", fake_repack)
    monkeypatch.setattr(gptq_mod, "marlin_moe_permute_scales", fake_permute_scales)
    monkeypatch.setattr(awq_mod.ops, "awq_marlin_moe_repack", fake_repack)
    monkeypatch.setattr(awq_mod, "marlin_moe_permute_scales", fake_permute_scales)
    monkeypatch.setattr(awq_mod, "moe_awq_to_marlin_zero_points_float", fake_awq_float_zp)
    monkeypatch.setattr(ct_mod.ops, "gptq_marlin_moe_repack", fake_repack)
    monkeypatch.setattr(ct_mod, "marlin_moe_permute_scales", fake_permute_scales)


def _make_gptq_moe_matrix_case(
    case: MoeWritebackMatrixCase,
) -> tuple[Any, torch.nn.Module, bool]:
    method = GPTQMarlinMoEMethod(
        GPTQMarlinConfig(
            weight_bits=4 if case.quant_name == "uint4b8" else 8,
            group_size=case.group_size,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config={},
        ),
        FusedMoEConfig(disable_inplace=True),
    )
    method.is_k_full = True
    layer = _make_minimal_moe_layer(
        prefix="gptq",
        num_experts=case.shape.experts,
        hidden_size=case.shape.hidden,
        intermediate_size=case.shape.intermediate,
        num_bits=method.quant_type.size_bits,
        group_size=case.group_size,
    )
    method.process_weights_after_loading(layer)
    return method, layer, False


def _make_awq_moe_matrix_case(
    case: MoeWritebackMatrixCase,
) -> tuple[Any, torch.nn.Module, bool]:
    num_bits = 4 if case.quant_name == "uint4" else 8
    method = AWQMarlinMoEMethod(
        AWQMarlinConfig(
            weight_bits=num_bits,
            group_size=case.group_size,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=None,
            full_config={},
        ),
        FusedMoEConfig(disable_inplace=True),
    )
    method.is_k_full = True
    layer = _make_minimal_moe_layer(
        prefix="awq",
        num_experts=case.shape.experts,
        hidden_size=case.shape.hidden,
        intermediate_size=case.shape.intermediate,
        num_bits=num_bits,
        group_size=case.group_size,
    )
    method.process_weights_after_loading(layer)
    assert layer.w13_qzeros.dtype == torch.float16
    assert layer.w2_qzeros.dtype == torch.float16
    assert layer.w13_qzeros.is_contiguous()
    assert layer.w2_qzeros.is_contiguous()
    return method, layer, True


def _make_compressed_tensors_wna16_moe_matrix_case(
    case: MoeWritebackMatrixCase,
) -> tuple[Any, torch.nn.Module, bool]:
    num_bits = 4 if case.quant_name == "uint4b8" else 8
    method = object.__new__(CompressedTensorsWNA16MarlinMoEMethod)
    method.moe = FusedMoEConfig(disable_inplace=True)
    method.kernel_backend = "Marlin"
    method.num_bits = num_bits
    method.packed_factor = 32 // num_bits
    method.strategy = "channel" if case.group_size == -1 else "group"
    method.group_size = case.group_size
    method.actorder = None
    method.quant_type = (
        scalar_types.uint4b8 if case.quant_name == "uint4b8" else scalar_types.uint8b128
    )
    method.marlin_input_dtype = None
    method.is_k_full = True
    method.moe_quant_config = None
    method.moe_kernel = None

    layer = _make_minimal_moe_layer(
        prefix="compressed_tensors",
        num_experts=case.shape.experts,
        hidden_size=case.shape.hidden,
        intermediate_size=case.shape.intermediate,
        num_bits=num_bits,
        group_size=case.group_size,
    )
    layer.marlin_state = SimpleNamespace()
    method.process_weights_after_loading(layer)
    return method, layer, False


_MOE_MATRIX_CACHE_LIMIT = 24
_MOE_MATRIX_CACHE_BYTES_LIMIT = 16 * 1024**3
_MOE_MATRIX_CACHE: OrderedDict[
    tuple[str, str, int, int, int, int, int],
    _CachedMoeMatrixCase,
] = OrderedDict()


def _moe_matrix_cache_key(
    case: MoeWritebackMatrixCase,
) -> tuple[str, str, int, int, int, int, int]:
    return (
        case.class_case.name,
        case.quant_name,
        case.group_size,
        case.shape.hidden,
        case.shape.intermediate,
        case.shape.experts,
        case.shape.topk,
    )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _add_tensor_nbytes(
    tensor: torch.Tensor,
    seen: set[int],
) -> int:
    identity = id(tensor)
    if identity in seen:
        return 0
    seen.add(identity)
    return _tensor_nbytes(tensor)


def _module_tensor_nbytes(module: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters(recurse=True):
        total += _add_tensor_nbytes(parameter, seen)
    for buffer in module.buffers(recurse=True):
        total += _add_tensor_nbytes(buffer, seen)
    for value in vars(module).values():
        if isinstance(value, torch.Tensor):
            total += _add_tensor_nbytes(value, seen)
    return total


def _trim_moe_matrix_cache() -> None:
    while len(_MOE_MATRIX_CACHE) > _MOE_MATRIX_CACHE_LIMIT:
        _MOE_MATRIX_CACHE.popitem(last=False)

    while len(_MOE_MATRIX_CACHE) > 1:
        total_bytes = sum(entry.estimated_bytes for entry in _MOE_MATRIX_CACHE.values())
        if total_bytes <= _MOE_MATRIX_CACHE_BYTES_LIMIT:
            break
        _MOE_MATRIX_CACHE.popitem(last=False)


def _refresh_moe_matrix_cache_entry(case: MoeWritebackMatrixCase) -> None:
    cached = _MOE_MATRIX_CACHE.get(_moe_matrix_cache_key(case))
    if cached is None:
        return
    cached.estimated_bytes = _module_tensor_nbytes(cached.layer)
    _MOE_MATRIX_CACHE.move_to_end(_moe_matrix_cache_key(case))
    _trim_moe_matrix_cache()


def _prepare_moe_matrix_case(case: MoeWritebackMatrixCase) -> tuple[
    Any,
    torch.nn.Module,
    bool,
]:
    key = _moe_matrix_cache_key(case)
    cached = _MOE_MATRIX_CACHE.get(key)
    if cached is not None:
        _MOE_MATRIX_CACHE.move_to_end(key)
        return cached.method, cached.layer, cached.expected_zp_float

    if case.class_case.name == "gptq_moe":
        result = _make_gptq_moe_matrix_case(case)
    elif case.class_case.name == "awq_moe":
        result = _make_awq_moe_matrix_case(case)
    elif case.class_case.name == "compressed_tensors_wna16_moe":
        result = _make_compressed_tensors_wna16_moe_matrix_case(case)
    else:
        raise AssertionError(f"Unhandled MoE class {case.class_case.name!r}")

    method, layer, expected_zp_float = result
    _MOE_MATRIX_CACHE[key] = _CachedMoeMatrixCase(
        method=method,
        layer=layer,
        expected_zp_float=expected_zp_float,
        estimated_bytes=_module_tensor_nbytes(layer),
    )
    _MOE_MATRIX_CACHE.move_to_end(key)
    _trim_moe_matrix_cache()
    return result


def _moe_matrix_row(
    case: MoeWritebackMatrixCase,
    *,
    status: str,
    expected_is_zp_float: bool | None = None,
    c_tmp_numel: int | None = None,
    reason: str = "",
) -> dict[str, object]:
    return {
        "case_id": case.id,
        "method_class": case.class_case.name,
        "quant": case.quant_name,
        "group_size": case.group_size,
        "shape_id": case.shape.name,
        "tokens": case.shape.tokens,
        "hidden": case.shape.hidden,
        "intermediate": case.shape.intermediate,
        "experts": case.shape.experts,
        "topk": case.shape.topk,
        "routing_profile": case.shape.routing_profile,
        "status": status,
        "is_zp_float": expected_is_zp_float,
        "c_tmp_numel": c_tmp_numel,
        "reason": reason,
    }


def _assert_fresh_c_tmp(layer: torch.nn.Module) -> None:
    assert hasattr(layer, "c_tmp")
    assert layer.c_tmp.dtype == torch.float32
    assert layer.c_tmp.is_contiguous()
    assert layer.c_tmp.numel() == 0
    assert not layer.c_tmp.requires_grad


def _assert_apply_resizes_and_reuses_c_tmp(
    method: Any,
    layer: torch.nn.Module,
    records: list[_RecordedGemm],
    *,
    hidden_size: int = 32,
    intermediate_size: int = 16,
    num_experts: int = 2,
    topk: int = 2,
    expected_zp_float: bool = False,
) -> None:
    hidden_states = torch.randn(2, hidden_size, dtype=torch.float16)
    topk_weights = torch.full((2, topk), 0.5, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    output = method.apply(layer, hidden_states, topk_weights, topk_ids, None)

    required = hidden_states.shape[0] * topk * max(2 * intermediate_size, hidden_size)
    assert output.shape == hidden_states.shape
    assert layer.c_tmp.numel() >= required
    assert layer.c_tmp.dtype == torch.float32
    assert layer.c_tmp.is_contiguous()
    assert len(records) == 2
    assert records[0].c_tmp is layer.c_tmp
    assert records[1].c_tmp is layer.c_tmp
    assert records[0].is_zp_float is expected_zp_float
    assert records[1].is_zp_float is expected_zp_float
    if expected_zp_float:
        assert records[0].b_qzeros is not None
        assert records[1].b_qzeros is not None
        assert records[0].b_qzeros.dtype == torch.float16
        assert records[1].b_qzeros.dtype == torch.float16

    first_c_tmp = layer.c_tmp
    records.clear()
    smaller_hidden = torch.randn(1, hidden_size, dtype=torch.float16)
    smaller_weights = torch.ones(1, topk, dtype=torch.float32)
    smaller_ids = torch.tensor([[0, 1]], dtype=torch.int32)
    method.apply(layer, smaller_hidden, smaller_weights, smaller_ids, None)
    assert layer.c_tmp is first_c_tmp
    assert len(records) == 2
    assert records[0].c_tmp is first_c_tmp
    assert records[1].c_tmp is first_c_tmp
    assert records[0].is_zp_float is expected_zp_float
    assert records[1].is_zp_float is expected_zp_float

    records.clear()
    larger_hidden = torch.randn(5, hidden_size, dtype=torch.float16)
    larger_weights = torch.full((5, topk), 0.5, dtype=torch.float32)
    larger_ids = torch.tensor([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=torch.int32)
    method.apply(layer, larger_hidden, larger_weights, larger_ids, None)
    larger_required = larger_hidden.shape[0] * topk * max(2 * intermediate_size, hidden_size)
    assert layer.c_tmp is not first_c_tmp
    assert layer.c_tmp.numel() >= larger_required
    assert len(records) == 2
    assert records[0].c_tmp is layer.c_tmp
    assert records[1].c_tmp is layer.c_tmp
    assert records[0].is_zp_float is expected_zp_float
    assert records[1].is_zp_float is expected_zp_float


def test_moe_writeback_method_full_matrix_post_load_apply_path(
    monkeypatch: pytest.MonkeyPatch,
):
    records = _install_fused_marlin_cpu_mocks(monkeypatch)
    _install_all_moe_post_load_mocks(monkeypatch)
    _MOE_FULL_MATRIX_JSONL.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    status_counts: Counter[str] = Counter()
    skip_reasons: Counter[str] = Counter()
    coverage: dict[str, Counter[Any]] = {
        "method_class": Counter(),
        "quant": Counter(),
        "group_size": Counter(),
        "shape_id": Counter(),
        "tokens": Counter(),
        "hidden": Counter(),
        "intermediate": Counter(),
        "experts": Counter(),
        "topk": Counter(),
        "routing_profile": Counter(),
    }
    ok_coverage: dict[str, Counter[Any]] = {
        key: Counter() for key in coverage
    }
    first_failure: dict[str, object] | None = None
    max_c_tmp_numel = 0

    with _MOE_FULL_MATRIX_JSONL.open("w", encoding="utf-8") as jsonl:
        for index, matrix_case in enumerate(iter_moe_writeback_matrix(), start=1):
            status_counts["total"] += 1
            coverage["method_class"][matrix_case.class_case.name] += 1
            coverage["quant"][matrix_case.quant_name] += 1
            coverage["group_size"][matrix_case.group_size] += 1
            coverage["shape_id"][matrix_case.shape.name] += 1
            coverage["tokens"][matrix_case.shape.tokens] += 1
            coverage["hidden"][matrix_case.shape.hidden] += 1
            coverage["intermediate"][matrix_case.shape.intermediate] += 1
            coverage["experts"][matrix_case.shape.experts] += 1
            coverage["topk"][matrix_case.shape.topk] += 1
            coverage["routing_profile"][matrix_case.shape.routing_profile] += 1

            if not matrix_case.supported:
                status_counts["SKIP"] += 1
                skip_reasons[matrix_case.reason] += 1
                continue

            expected_zp_float: bool | None = None
            c_tmp_numel: int | None = None
            try:
                records.clear()
                method, layer, expected_zp_float = _prepare_moe_matrix_case(
                    matrix_case
                )
                hidden_states = torch.zeros(
                    matrix_case.shape.tokens,
                    matrix_case.shape.hidden,
                    dtype=torch.float16,
                )
                topk_weights, topk_ids = make_moe_routing_tensors(
                    tokens=matrix_case.shape.tokens,
                    experts=matrix_case.shape.experts,
                    topk=matrix_case.shape.topk,
                    device=torch.device("cpu"),
                    routing_profile=matrix_case.shape.routing_profile,
                )
                output = method.apply(
                    layer,
                    hidden_states,
                    topk_weights,
                    topk_ids,
                    None,
                )

                assert output.shape == hidden_states.shape
                assert len(records) == 2

                stage1, stage2 = records
                assert stage1.c_tmp is layer.c_tmp
                assert stage2.c_tmp is layer.c_tmp
                assert stage1.size_m == matrix_case.shape.tokens
                assert stage1.size_k == matrix_case.shape.hidden
                assert stage1.size_n == 2 * matrix_case.shape.intermediate
                assert stage1.top_k == matrix_case.shape.topk
                assert stage2.size_m == (
                    matrix_case.shape.tokens * matrix_case.shape.topk
                )
                assert stage2.size_k == matrix_case.shape.intermediate
                assert stage2.size_n == matrix_case.shape.hidden
                assert stage2.top_k == 1
                assert stage1.is_zp_float is expected_zp_float
                assert stage2.is_zp_float is expected_zp_float

                if expected_zp_float:
                    assert stage1.b_qzeros is not None
                    assert stage2.b_qzeros is not None
                    assert stage1.b_qzeros.dtype == torch.float16
                    assert stage2.b_qzeros.dtype == torch.float16
                else:
                    assert stage1.b_qzeros is None
                    assert stage2.b_qzeros is None

                required_c_tmp = (
                    matrix_case.shape.tokens
                    * matrix_case.shape.topk
                    * max(2 * matrix_case.shape.intermediate, matrix_case.shape.hidden)
                )
                assert layer.c_tmp.dtype == torch.float32
                assert layer.c_tmp.is_contiguous()
                assert layer.c_tmp.numel() >= required_c_tmp
                assert not layer.c_tmp.requires_grad

                c_tmp_numel = int(layer.c_tmp.numel())
                max_c_tmp_numel = max(max_c_tmp_numel, c_tmp_numel)
                status = "OK"
                reason = ""
                status_counts["OK"] += 1
                ok_coverage["method_class"][matrix_case.class_case.name] += 1
                ok_coverage["quant"][matrix_case.quant_name] += 1
                ok_coverage["group_size"][matrix_case.group_size] += 1
                ok_coverage["shape_id"][matrix_case.shape.name] += 1
                ok_coverage["tokens"][matrix_case.shape.tokens] += 1
                ok_coverage["hidden"][matrix_case.shape.hidden] += 1
                ok_coverage["intermediate"][matrix_case.shape.intermediate] += 1
                ok_coverage["experts"][matrix_case.shape.experts] += 1
                ok_coverage["topk"][matrix_case.shape.topk] += 1
                ok_coverage["routing_profile"][
                    matrix_case.shape.routing_profile
                ] += 1
            except Exception as exc:
                status = "ERR"
                reason = str(exc).splitlines()[0]
                status_counts["ERR"] += 1
                if first_failure is None:
                    first_failure = _moe_matrix_row(
                        matrix_case,
                        status=status,
                        expected_is_zp_float=expected_zp_float,
                        c_tmp_numel=c_tmp_numel,
                        reason=reason,
                    )

            _refresh_moe_matrix_cache_entry(matrix_case)
            jsonl.write(
                json.dumps(
                    _moe_matrix_row(
                        matrix_case,
                        status=status,
                        expected_is_zp_float=expected_zp_float,
                        c_tmp_numel=c_tmp_numel,
                        reason=reason,
                    ),
                    sort_keys=True,
                )
                + "\n"
            )
            if index % 10000 == 0:
                jsonl.flush()

    elapsed = time.time() - started
    summary = {
        "matrix": "class x quant x group x shape",
        "total": status_counts["total"],
        "OK": status_counts["OK"],
        "SKIP": status_counts["SKIP"],
        "ERR": status_counts["ERR"],
        "skip_reasons": _counter_to_json(skip_reasons),
        "coverage": {key: _counter_to_json(counter) for key, counter in coverage.items()},
        "ok_coverage": {
            key: _counter_to_json(counter) for key, counter in ok_coverage.items()
        },
        "max_c_tmp_numel": max_c_tmp_numel,
        "first_failure": first_failure,
        "jsonl": str(_MOE_FULL_MATRIX_JSONL),
        "elapsed_seconds": round(elapsed, 3),
    }
    _MOE_FULL_MATRIX_SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _MOE_MATRIX_CACHE.clear()

    if first_failure is not None:
        pytest.fail(
            "MoE writeback full matrix loop had "
            f"{status_counts['ERR']} ERR cases; first={first_failure}"
        )


def test_fused_marlin_moe_resizes_and_persists_c_tmp(monkeypatch: pytest.MonkeyPatch):
    from vllm.model_executor.layers.fused_moe.fused_marlin_moe import fused_marlin_moe

    records = _install_fused_marlin_cpu_mocks(monkeypatch)
    owner = torch.nn.Module()
    owner.c_tmp = marlin_make_c_tmp(torch.device("cpu"))
    hidden_states = torch.randn(2, 32, dtype=torch.float16)
    topk_weights = torch.full((2, 2), 0.5, dtype=torch.float32)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
    w1 = torch.empty(2, 2, 64, dtype=torch.int32)
    w2 = torch.empty(2, 1, 64, dtype=torch.int32)
    w1_scale = torch.ones(2, 1, 32, dtype=torch.float16)
    w2_scale = torch.ones(2, 1, 32, dtype=torch.float16)

    output = fused_marlin_moe(
        hidden_states,
        w1,
        w2,
        None,
        None,
        w1_scale,
        w2_scale,
        topk_weights,
        topk_ids,
        quant_type_id=scalar_types.uint4b8.id,
        c_tmp=owner.c_tmp,
        c_tmp_owner=owner,
    )

    assert output.shape == hidden_states.shape
    assert owner.c_tmp.dtype == torch.float32
    assert owner.c_tmp.is_contiguous()
    assert owner.c_tmp.numel() >= 2 * 2 * max(2 * 16, 32)
    assert len(records) == 2
    assert records[0].c_tmp is owner.c_tmp
    assert records[1].c_tmp is owner.c_tmp
    assert records[0].is_zp_float is False
    assert records[1].is_zp_float is False


def test_gptq_marlin_moe_method_persists_c_tmp_and_apply_uses_owner(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.model_executor.layers.quantization.gptq_marlin as gptq_mod

    monkeypatch.setattr(
        gptq_mod.ops,
        "gptq_marlin_moe_repack",
        lambda weight, *args, **kwargs: weight.detach().clone().contiguous(),
    )
    monkeypatch.setattr(
        gptq_mod,
        "marlin_moe_permute_scales",
        lambda s, *args, **kwargs: s.detach().clone().contiguous(),
    )
    records = _install_fused_marlin_cpu_mocks(monkeypatch)

    config = GPTQMarlinConfig(
        weight_bits=4,
        group_size=-1,
        desc_act=False,
        is_sym=True,
        lm_head_quantized=False,
        dynamic={},
        full_config={},
    )
    method = GPTQMarlinMoEMethod(config, FusedMoEConfig(disable_inplace=True))
    method.is_k_full = True
    layer = _make_minimal_moe_layer(prefix="gptq")
    method.process_weights_after_loading(layer)
    _assert_fresh_c_tmp(layer)
    _assert_apply_resizes_and_reuses_c_tmp(method, layer, records)


@pytest.mark.parametrize(
    "quant_name",
    MOE_WRITEBACK_CLASS_CASE_BY_NAME["gptq"].quant_names,
)
def test_gptq_marlin_moe_method_class_uses_owner(
    monkeypatch: pytest.MonkeyPatch,
    quant_name: str,
):
    import vllm.model_executor.layers.quantization.gptq_marlin as gptq_mod

    monkeypatch.setattr(
        gptq_mod.ops,
        "gptq_marlin_moe_repack",
        lambda weight, *args, **kwargs: weight.detach().clone().contiguous(),
    )
    monkeypatch.setattr(
        gptq_mod,
        "marlin_moe_permute_scales",
        lambda s, *args, **kwargs: s.detach().clone().contiguous(),
    )
    records = _install_fused_marlin_cpu_mocks(monkeypatch)

    method = GPTQMarlinMoEMethod(
        _gptq_config_for_quant_name(quant_name),
        FusedMoEConfig(disable_inplace=True),
    )
    method.is_k_full = True
    layer = _make_minimal_moe_layer(
        prefix="gptq",
        num_bits=method.quant_type.size_bits,
    )
    method.process_weights_after_loading(layer)
    _assert_fresh_c_tmp(layer)
    _assert_apply_resizes_and_reuses_c_tmp(
        method,
        layer,
        records,
        expected_zp_float=False,
    )


@pytest.mark.parametrize(
    ("num_bits", "expected_quant_type"),
    (
        pytest.param(4, scalar_types.uint4, id="uint4_zp"),
        pytest.param(8, scalar_types.uint8, id="uint8_zp"),
    ),
)
def test_awq_marlin_moe_method_persists_c_tmp_and_apply_uses_owner(
    monkeypatch: pytest.MonkeyPatch,
    num_bits: int,
    expected_quant_type: Any,
):
    import vllm.model_executor.layers.quantization.awq_marlin as awq_mod

    monkeypatch.setattr(
        awq_mod.ops,
        "awq_marlin_moe_repack",
        lambda weight, *args, **kwargs: weight.detach().clone().contiguous(),
    )
    monkeypatch.setattr(
        awq_mod,
        "marlin_moe_permute_scales",
        lambda s, *args, **kwargs: s.detach().clone().contiguous(),
    )
    records = _install_fused_marlin_cpu_mocks(monkeypatch)

    config = AWQMarlinConfig(
        weight_bits=num_bits,
        group_size=32,
        zero_point=True,
        lm_head_quantized=False,
        modules_to_not_convert=None,
        full_config={},
    )
    method = AWQMarlinMoEMethod(config, FusedMoEConfig(disable_inplace=True))
    method.is_k_full = True
    assert method.quant_type == expected_quant_type
    layer = _make_minimal_moe_layer(prefix="awq", num_bits=num_bits)
    method.process_weights_after_loading(layer)
    _assert_fresh_c_tmp(layer)
    assert layer.w13_qzeros.dtype == torch.float16
    assert layer.w2_qzeros.dtype == torch.float16
    assert layer.w13_qzeros.is_contiguous()
    assert layer.w2_qzeros.is_contiguous()
    _assert_apply_resizes_and_reuses_c_tmp(
        method, layer, records, expected_zp_float=True
    )


@pytest.mark.parametrize(
    "quant_name",
    MOE_WRITEBACK_CLASS_CASE_BY_NAME["awq_moe"].quant_names,
)
def test_awq_marlin_moe_method_class_uses_owner(
    monkeypatch: pytest.MonkeyPatch,
    quant_name: str,
):
    import vllm.model_executor.layers.quantization.awq_marlin as awq_mod

    monkeypatch.setattr(
        awq_mod.ops,
        "awq_marlin_moe_repack",
        lambda weight, *args, **kwargs: weight.detach().clone().contiguous(),
    )
    monkeypatch.setattr(
        awq_mod,
        "marlin_moe_permute_scales",
        lambda s, *args, **kwargs: s.detach().clone().contiguous(),
    )
    records = _install_fused_marlin_cpu_mocks(monkeypatch)

    num_bits = 4 if quant_name == "uint4" else 8
    method = AWQMarlinMoEMethod(
        AWQMarlinConfig(
            weight_bits=num_bits,
            group_size=32,
            zero_point=True,
            lm_head_quantized=False,
            modules_to_not_convert=None,
            full_config={},
        ),
        FusedMoEConfig(disable_inplace=True),
    )
    method.is_k_full = True
    layer = _make_minimal_moe_layer(prefix="awq", num_bits=num_bits)
    method.process_weights_after_loading(layer)
    _assert_fresh_c_tmp(layer)
    assert layer.w13_qzeros.dtype == torch.float16
    assert layer.w2_qzeros.dtype == torch.float16
    _assert_apply_resizes_and_reuses_c_tmp(
        method,
        layer,
        records,
        expected_zp_float=True,
    )


def test_compressed_tensors_wna16_marlin_moe_method_persists_c_tmp_and_apply_uses_owner(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe as ct_mod

    monkeypatch.setattr(
        ct_mod.ops,
        "gptq_marlin_moe_repack",
        lambda weight, *args, **kwargs: weight.detach().clone().contiguous(),
    )
    monkeypatch.setattr(
        ct_mod,
        "marlin_moe_permute_scales",
        lambda s, *args, **kwargs: s.detach().clone().contiguous(),
    )
    records = _install_fused_marlin_cpu_mocks(monkeypatch)

    method = object.__new__(CompressedTensorsWNA16MarlinMoEMethod)
    method.moe = FusedMoEConfig(disable_inplace=True)
    method.kernel_backend = "Marlin"
    method.num_bits = 4
    method.packed_factor = 8
    method.strategy = "channel"
    method.group_size = -1
    method.actorder = None
    method.quant_type = scalar_types.uint4
    method.marlin_input_dtype = None
    method.is_k_full = True
    method.moe_quant_config = None
    method.moe_kernel = None

    layer = _make_minimal_moe_layer(prefix="compressed_tensors")
    layer.marlin_state = SimpleNamespace()
    method.process_weights_after_loading(layer)
    _assert_fresh_c_tmp(layer)
    _assert_apply_resizes_and_reuses_c_tmp(method, layer, records)


@pytest.mark.parametrize(
    "quant_name",
    MOE_WRITEBACK_CLASS_CASE_BY_NAME["compressed_tensors_wna16_moe"].quant_names,
)
def test_compressed_tensors_wna16_marlin_moe_method_class_uses_owner(
    monkeypatch: pytest.MonkeyPatch,
    quant_name: str,
):
    import vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe as ct_mod

    monkeypatch.setattr(
        ct_mod.ops,
        "gptq_marlin_moe_repack",
        lambda weight, *args, **kwargs: weight.detach().clone().contiguous(),
    )
    monkeypatch.setattr(
        ct_mod,
        "marlin_moe_permute_scales",
        lambda s, *args, **kwargs: s.detach().clone().contiguous(),
    )
    records = _install_fused_marlin_cpu_mocks(monkeypatch)

    num_bits = 4 if quant_name == "uint4b8" else 8
    method = object.__new__(CompressedTensorsWNA16MarlinMoEMethod)
    method.moe = FusedMoEConfig(disable_inplace=True)
    method.kernel_backend = "Marlin"
    method.num_bits = num_bits
    method.packed_factor = 32 // num_bits
    method.strategy = "channel"
    method.group_size = -1
    method.actorder = None
    method.quant_type = (
        scalar_types.uint4b8 if quant_name == "uint4b8" else scalar_types.uint8b128
    )
    method.marlin_input_dtype = None
    method.is_k_full = True
    method.moe_quant_config = None
    method.moe_kernel = None

    layer = _make_minimal_moe_layer(
        prefix="compressed_tensors",
        num_bits=num_bits,
    )
    layer.marlin_state = SimpleNamespace()
    method.process_weights_after_loading(layer)
    _assert_fresh_c_tmp(layer)
    _assert_apply_resizes_and_reuses_c_tmp(
        method,
        layer,
        records,
        expected_zp_float=False,
    )


def test_quark_w8a8_fp8_marlin_moe_method_persists_c_tmp_and_apply_uses_owner(
    monkeypatch: pytest.MonkeyPatch,
):
    import vllm.model_executor.layers.quantization.quark.quark_moe as quark_mod

    hidden_size = 32
    intermediate_size = 16
    num_experts = 2

    def fake_prepare_fp8_moe_layer_for_marlin(
        layer: torch.nn.Module,
        w13_weight: torch.Tensor,
        w2_weight: torch.Tensor,
        w13_weight_scale: torch.Tensor,
        w2_weight_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del w13_weight, w2_weight, w13_weight_scale, w2_weight_scale
        layer.c_tmp = marlin_make_c_tmp(layer.w13_weight.device)
        w13_packed = torch.empty(
            num_experts,
            hidden_size // 16,
            2 * intermediate_size * 4,
            dtype=torch.int32,
        )
        w2_packed = torch.empty(
            num_experts,
            intermediate_size // 16,
            hidden_size * 4,
            dtype=torch.int32,
        )
        w13_scale = torch.ones(
            num_experts,
            1,
            2 * intermediate_size,
            dtype=torch.float16,
        )
        w2_scale = torch.ones(num_experts, 1, hidden_size, dtype=torch.float16)
        return (
            w13_packed.contiguous(),
            w2_packed.contiguous(),
            w13_scale.contiguous(),
            w2_scale.contiguous(),
        )

    monkeypatch.setattr(
        quark_mod,
        "prepare_fp8_moe_layer_for_marlin",
        fake_prepare_fp8_moe_layer_for_marlin,
    )
    records = _install_fused_marlin_cpu_mocks(monkeypatch)

    method = object.__new__(QuarkW8A8Fp8MoEMethod)
    method.moe = FusedMoEConfig(disable_inplace=True)
    method.weight_qscheme = "per_channel"
    method.input_qscheme = "per_tensor"
    method.act_quant_group_shape = None
    method.static_input_scales = False
    method.rocm_aiter_moe_enabled = False
    method.use_marlin = True
    method.model_type = None
    method.moe_quant_config = None

    layer = _make_minimal_quark_fp8_moe_layer(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    method.process_weights_after_loading(layer)
    _assert_fresh_c_tmp(layer)
    _assert_apply_resizes_and_reuses_c_tmp(
        method,
        layer,
        records,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        expected_zp_float=False,
    )
