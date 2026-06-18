# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "benchmarks/marlin_gemm_shapes.py"
SPEC = importlib.util.spec_from_file_location("marlin_gemm_shapes", SCRIPT)
assert SPEC is not None
marlin_gemm_shapes = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = marlin_gemm_shapes
SPEC.loader.exec_module(marlin_gemm_shapes)


def write_model(tmp_path: Path, config: dict, keys: list[str] | None = None) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if keys is not None:
        index = {
            "metadata": {},
            "weight_map": {key: "model.safetensors" for key in keys},
        }
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
    return model_dir


def payload_for(model_dir: Path, *extra_args: str) -> dict:
    args = marlin_gemm_shapes.parse_args(["--model", str(model_dir), *extra_args])
    return marlin_gemm_shapes.build_payload(args)


def has_layer_key(row: dict, needle: str) -> bool:
    return any(needle in key for key in row["layer_keys"])


def qwen_moe_config(quant_config: dict) -> dict:
    return {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "quantization_config": quant_config,
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 3072,
            "moe_intermediate_size": 1024,
            "shared_expert_intermediate_size": 1024,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_types": ["linear_attention", "full_attention"],
            "attn_output_gate": True,
        },
    }


def qwen_dense_config(quant_config: dict) -> dict:
    return {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "quantization_config": quant_config,
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "num_hidden_layers": 2,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "layer_types": ["linear_attention", "full_attention"],
            "attn_output_gate": True,
        },
    }


def test_qwen_awq_dense_model_is_not_moe(tmp_path: Path):
    config = qwen_dense_config({
        "quant_method": "awq",
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "modules_to_not_convert": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "model.layers.0.",
        ],
    })
    keys = [
        "model.language_model.layers.1.mlp.gate_proj.qweight",
        "model.language_model.layers.1.mlp.gate_proj.qzeros",
        "model.language_model.layers.1.mlp.gate_proj.scales",
        "model.language_model.layers.1.mlp.up_proj.qweight",
        "model.language_model.layers.1.mlp.up_proj.qzeros",
        "model.language_model.layers.1.mlp.up_proj.scales",
        "model.language_model.layers.1.mlp.down_proj.qweight",
        "model.language_model.layers.1.mlp.down_proj.qzeros",
        "model.language_model.layers.1.mlp.down_proj.scales",
    ]
    model_dir = write_model(tmp_path, config, keys)

    payload = payload_for(
        model_dir,
        "--format",
        "json",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
    )

    assert payload["model_config"]["architecture_family"] == "qwen3_5_text"
    assert payload["model_config"]["moe_intermediate_size"] is None
    assert payload["model_config"]["num_experts"] is None
    assert payload["moe"] == []
    dense = next(r for r in payload["dense"] if r["op"] == "gate_up_proj")
    assert dense["target_op"] == "ops.marlin_gemm"
    assert dense["marlin_path"] == "awq_marlin_wna16"
    assert dense["call_status"] == "actual_marlin"


def test_qwen_awq_moe_decode_concurrency_block_sizes(tmp_path: Path):
    config = qwen_moe_config({
        "quant_method": "awq",
        "bits": 4,
        "group_size": 128,
        "zero_point": True,
        "modules_to_not_convert": [
            "self_attn",
            "shared_expert",
            "mlp.gate",
            "model.layers.0.",
        ],
    })
    keys = [
        "model.language_model.layers.1.mlp.experts.0.gate_proj.qweight",
        "model.language_model.layers.1.mlp.experts.0.gate_proj.qzeros",
        "model.language_model.layers.1.mlp.experts.0.gate_proj.scales",
        "model.language_model.layers.1.mlp.experts.0.up_proj.qweight",
        "model.language_model.layers.1.mlp.experts.0.up_proj.qzeros",
        "model.language_model.layers.1.mlp.experts.0.up_proj.scales",
        "model.language_model.layers.1.mlp.experts.0.down_proj.qweight",
        "model.language_model.layers.1.mlp.experts.0.down_proj.qzeros",
        "model.language_model.layers.1.mlp.experts.0.down_proj.scales",
    ]
    model_dir = write_model(tmp_path, config, keys)

    payload = payload_for(model_dir, "--format", "json")

    assert payload["dense"] == []
    row = next(
        r for r in payload["moe"]
        if r["scenario"] == "tp8+ep"
        and r["phase"] == "decode"
        and r["decode_concurrency"] == 64
        and r["op"] == "w13"
    )
    assert row["call_status"] == "actual_marlin"
    assert row["target_op"] == "ops.moe_wna16_marlin_gemm"
    assert row["quant_format"] == "uint4"
    assert row["group_size"] == 128
    assert row["moe_block_size"] == 32
    assert row["size_m"] == 64
    assert row["size_n"] == 2048
    assert row["size_k"] == 3072

    row = next(
        r for r in payload["moe"]
        if r["scenario"] == "tp4+ep"
        and r["phase"] == "decode"
        and r["decode_concurrency"] == 64
        and r["op"] == "w2"
    )
    assert row["moe_block_size"] == 16
    assert row["size_m"] == 512
    assert row["size_k"] == 1024


def test_qwen3_moe_experts_are_not_dense_mlp(tmp_path: Path):
    config = {
        "model_type": "qwen3_moe",
        "architectures": ["Qwen3MoeForCausalLM"],
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 8,
            "group_size": 32,
            "format": "gptq",
        },
        "hidden_size": 4096,
        "intermediate_size": 12288,
        "moe_intermediate_size": 1536,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 2,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
    }
    keys = [
        "model.layers.0.mlp.experts.0.gate_proj.qweight",
        "model.layers.0.mlp.experts.0.gate_proj.qzeros",
        "model.layers.0.mlp.experts.0.gate_proj.scales",
        "model.layers.0.mlp.experts.0.up_proj.qweight",
        "model.layers.0.mlp.experts.0.up_proj.qzeros",
        "model.layers.0.mlp.experts.0.up_proj.scales",
        "model.layers.0.mlp.experts.0.down_proj.qweight",
        "model.layers.0.mlp.experts.0.down_proj.qzeros",
        "model.layers.0.mlp.experts.0.down_proj.scales",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
    ]
    model_dir = write_model(tmp_path, config, keys)

    payload = payload_for(
        model_dir,
        "--format",
        "json",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
    )

    assert payload["model_config"]["architecture_family"] == "qwen3_moe"
    assert payload["model_config"]["moe_layer_indices"] == [0, 1]
    assert not any(
        row["op"] in {"gate_up_proj", "down_proj"}
        for row in payload["dense"]
    )

    w13 = next(
        row for row in payload["moe"]
        if row["op"] == "w13" and row["phase"] == "prefill"
    )
    w2 = next(
        row for row in payload["moe"]
        if row["op"] == "w2" and row["phase"] == "prefill"
    )
    for row in [w13, w2]:
        assert row["call_status"] == "actual_marlin"
        assert row["target_op"] == "ops.moe_wna16_marlin_gemm"
        assert row["quant_format"] == "uint8b128"
        assert row["group_size"] == 32
        assert row["marlin_path"] == "wna16_marlin"
        assert has_layer_key(row, "model.layers.0.mlp.experts")

    qkv = next(row for row in payload["dense"] if row["op"] == "qkv_proj")
    assert qkv["call_status"] == "hypothetical_bf16"
    assert qkv["quant_method"] == "unquantized"
    assert qkv["quant_format"] == "bf16_or_fp16"
    assert qkv["target_op"] == "none"


def step_config(quant_config: dict) -> dict:
    return {
        "model_type": "step3p7",
        "architectures": ["Step3p7ForConditionalGeneration"],
        "quantization_config": quant_config,
        "text_config": {
            "model_type": "step3p5",
            "hidden_size": 4096,
            "intermediate_size": 11264,
            "moe_intermediate_size": 1280,
            "moe_num_experts": 288,
            "moe_top_k": 8,
            "share_expert_dim": 1280,
            "num_hidden_layers": 4,
            "num_attention_heads": 64,
            "num_attention_groups": 8,
            "head_dim": 128,
            "layer_types": [
                "full_attention",
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            ],
            "attention_other_setting": {
                "attention_type": "sliding_attention",
                "num_attention_heads": 96,
                "num_attention_groups": 8,
                "head_dim": 128,
            },
            "moe_layers_enum": "2,3",
        },
    }


def test_modelopt_nvfp4_mixed_modules_fall_back_to_bf16(tmp_path: Path):
    config = step_config({
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "ignore": [
            "model.language_model.layers.1.self_attn*",
            "model.language_model.layers.2.moe.gate",
        ],
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "dynamic": False,
                },
                "input_activations": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "dynamic": False,
                },
            }
        },
    })
    keys = [
        # Quantized qkv shards on layer 0.
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight_scale",
        "model.language_model.layers.0.self_attn.q_proj.weight_scale_2",
        "model.language_model.layers.0.self_attn.k_proj.weight",
        "model.language_model.layers.0.self_attn.k_proj.weight_scale",
        "model.language_model.layers.0.self_attn.k_proj.weight_scale_2",
        "model.language_model.layers.0.self_attn.v_proj.weight",
        "model.language_model.layers.0.self_attn.v_proj.weight_scale",
        "model.language_model.layers.0.self_attn.v_proj.weight_scale_2",
        # Plain BF16 o_proj on the same model.
        "model.language_model.layers.0.self_attn.o_proj.weight",
        # Ignored attention layer.
        "model.language_model.layers.1.self_attn.q_proj.weight",
        "model.language_model.layers.1.self_attn.q_proj.weight_scale",
        "model.language_model.layers.1.self_attn.q_proj.weight_scale_2",
        "model.language_model.layers.1.self_attn.k_proj.weight",
        "model.language_model.layers.1.self_attn.k_proj.weight_scale",
        "model.language_model.layers.1.self_attn.k_proj.weight_scale_2",
        "model.language_model.layers.1.self_attn.v_proj.weight",
        "model.language_model.layers.1.self_attn.v_proj.weight_scale",
        "model.language_model.layers.1.self_attn.v_proj.weight_scale_2",
        # Router gate is ignored, but the FusedMoE expert parent is quantized.
        "model.language_model.layers.2.moe.weight",
        "model.language_model.layers.2.moe.weight_scale",
        "model.language_model.layers.2.moe.weight_scale_2",
    ]
    model_dir = write_model(tmp_path, config, keys)

    payload = payload_for(
        model_dir,
        "--format",
        "json",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
    )

    qkv = next(
        r for r in payload["dense"]
        if has_layer_key(r, "layers.0.self_attn.q_proj") and r["op"] == "qkv_proj"
    )
    assert qkv["call_status"] == "actual_marlin"
    assert qkv["target_op"] == "ops.marlin_gemm"
    assert qkv["quant_format"] == "nvfp4"
    assert qkv["size_n"] == 2560  # Sliding override does not apply to layer 0.

    o_proj = next(
        r for r in payload["dense"]
        if r["op"] == "o_proj"
        and has_layer_key(r, "layers.0.self_attn.o_proj")
    )
    assert o_proj["call_status"] == "hypothetical_bf16"
    assert o_proj["quant_method"] == "unquantized"
    assert o_proj["target_op"] == "none"
    assert o_proj["quant_format"] == "bf16_or_fp16"
    assert o_proj["marlin_path"] == "none"
    assert o_proj["warning"] == "index_disagrees_with_config"

    moe = next(
        r for r in payload["moe"]
        if r["op"] == "w13" and has_layer_key(r, "layers.2.moe")
    )
    assert moe["call_status"] == "actual_marlin"
    assert moe["target_op"] == "ops.moe_wna16_marlin_gemm"
    assert moe["quant_format"] == "nvfp4"

    payload_with_skips = payload_for(
        model_dir,
        "--format",
        "json",
        "--include-skipped",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
    )
    skipped = [
        r for r in payload_with_skips["dense"]
        if r["call_status"] == "skipped"
        and has_layer_key(r, "layers.1.self_attn.q_proj")
    ]
    assert skipped


def test_step_fp8_moe_weights_use_safetensors_dtype_evidence(tmp_path: Path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is unavailable")

    config = step_config({
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.1.self_attn.qkv_proj",
            "model.layers.1.self_attn.q_proj",
            "model.layers.1.self_attn.k_proj",
            "model.layers.1.self_attn.v_proj",
            "model.layers.1.self_attn.o_proj",
            "model.layers.1.mlp.gate_up_proj",
            "model.layers.1.mlp.gate_proj",
            "model.layers.1.mlp.up_proj",
            "model.layers.1.mlp.down_proj",
            "model.layers.2.self_attn.qkv_proj",
            "model.layers.2.self_attn.q_proj",
            "model.layers.2.self_attn.k_proj",
            "model.layers.2.self_attn.v_proj",
            "model.layers.2.self_attn.o_proj",
            "model.layers.2.moe.gate",
            "model.layers.2.share_expert.gate_up_proj",
            "model.layers.2.share_expert.gate_proj",
            "model.layers.2.share_expert.up_proj",
            "model.layers.2.share_expert.down_proj",
            "model.layers.3.self_attn.qkv_proj",
            "model.layers.3.self_attn.q_proj",
            "model.layers.3.self_attn.k_proj",
            "model.layers.3.self_attn.v_proj",
            "model.layers.3.self_attn.o_proj",
            "model.layers.3.moe.gate",
            "model.layers.3.share_expert.gate_up_proj",
            "model.layers.3.share_expert.gate_proj",
            "model.layers.3.share_expert.up_proj",
            "model.layers.3.share_expert.down_proj",
        ],
    })
    config["text_config"]["moe_layers_enum"] = "2"
    keys = [
        "model.layers.2.moe.gate.weight",
        "model.layers.2.moe.gate_proj.weight",
        "model.layers.2.moe.up_proj.weight",
        "model.layers.2.moe.down_proj.weight",
        "model.layers.2.self_attn.q_proj.weight",
        "model.layers.2.share_expert.gate_proj.weight",
    ]
    model_dir = write_model(tmp_path, config, keys)
    safetensors_torch.save_file(
        {
            "model.layers.2.moe.gate.weight": torch.empty(
                (1, 1), dtype=torch.bfloat16
            ),
            "model.layers.2.moe.gate_proj.weight": torch.empty(
                (1, 1), dtype=torch.float8_e4m3fn
            ),
            "model.layers.2.moe.up_proj.weight": torch.empty(
                (1, 1), dtype=torch.float8_e4m3fn
            ),
            "model.layers.2.moe.down_proj.weight": torch.empty(
                (1, 1), dtype=torch.float8_e4m3fn
            ),
            "model.layers.2.self_attn.q_proj.weight": torch.empty(
                (1, 1), dtype=torch.bfloat16
            ),
            "model.layers.2.share_expert.gate_proj.weight": torch.empty(
                (1, 1), dtype=torch.bfloat16
            ),
        },
        str(model_dir / "model.safetensors"),
    )

    payload = payload_for(
        model_dir,
        "--format",
        "json",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
    )

    moe_w13 = next(
        row for row in payload["moe"]
        if row["op"] == "w13" and row["phase"] == "prefill"
    )
    moe_w2 = next(
        row for row in payload["moe"]
        if row["op"] == "w2" and row["phase"] == "prefill"
    )
    for row in [moe_w13, moe_w2]:
        assert row["call_status"] == "actual_marlin"
        assert row["quant_method"] == "fp8"
        assert row["quant_format"] == "fp8_e4m3"
        assert row["group_size"] == 128
        assert row["marlin_path"] == "fp8_marlin"
        assert row["target_op"] == "ops.moe_wna16_marlin_gemm"
        assert row["warning"] == ""
    assert not any(row["warning"] == "excluded_quant_module"
                   for row in payload["moe"])


def test_modelopt_mixed_precision_layer_map(tmp_path: Path):
    config = qwen_moe_config({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.language_model.layers.1.self_attn.q_proj": {
                "quant_algo": "FP8"
            },
            "model.language_model.layers.1.self_attn.k_proj": {
                "quant_algo": "FP8"
            },
            "model.language_model.layers.1.self_attn.v_proj": {
                "quant_algo": "FP8"
            },
            "model.language_model.layers.1.mlp.experts.0.gate_proj": {
                "quant_algo": "NVFP4",
                "group_size": 16,
            },
        },
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "dynamic": False,
                },
            }
        },
    })
    keys = [
        "model.language_model.layers.1.self_attn.q_proj.weight",
        "model.language_model.layers.1.self_attn.q_proj.weight_scale",
        "model.language_model.layers.1.self_attn.k_proj.weight",
        "model.language_model.layers.1.self_attn.k_proj.weight_scale",
        "model.language_model.layers.1.self_attn.v_proj.weight",
        "model.language_model.layers.1.self_attn.v_proj.weight_scale",
        "model.language_model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.1.mlp.experts.0.gate_proj.weight_scale",
        "model.language_model.layers.1.mlp.experts.0.gate_proj.weight_scale_2",
        "model.language_model.layers.1.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.1.mlp.experts.0.up_proj.weight_scale",
        "model.language_model.layers.1.mlp.experts.0.up_proj.weight_scale_2",
        "model.language_model.layers.1.mlp.experts.0.down_proj.weight",
        "model.language_model.layers.1.mlp.experts.0.down_proj.weight_scale",
        "model.language_model.layers.1.mlp.experts.0.down_proj.weight_scale_2",
    ]
    model_dir = write_model(tmp_path, config, keys)

    payload = payload_for(
        model_dir,
        "--format",
        "json",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
    )

    qkv = next(r for r in payload["dense"] if r["op"] == "qkv_proj")
    assert qkv["call_status"] == "actual_marlin"
    assert qkv["target_op"] == "ops.marlin_gemm"
    assert qkv["quant_format"] == "fp8_e4m3"

    moe = next(
        r for r in payload["moe"]
        if r["op"] == "w13"
        and r["phase"] == "prefill"
        and has_layer_key(r, "layers.1.mlp.experts")
    )
    assert moe["call_status"] == "actual_marlin"
    assert moe["target_op"] == "ops.moe_wna16_marlin_gemm"
    assert moe["quant_format"] == "nvfp4"

    unlisted_shared = [
        r for r in payload["dense"]
        if r["op"] == "shared_gate_up_proj"
    ]
    assert unlisted_shared
    assert all(r["call_status"] == "hypothetical_bf16" for r in unlisted_shared)
    assert all(r["quant_method"] == "unquantized" for r in unlisted_shared)
    assert all(r["target_op"] == "none" for r in unlisted_shared)


def test_modelopt_mixed_precision_conflicting_fused_shards(tmp_path: Path):
    config = qwen_moe_config({
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "quantized_layers": {
            "model.language_model.layers.1.self_attn.q_proj": {
                "quant_algo": "FP8"
            },
            "model.language_model.layers.1.self_attn.k_proj": {
                "quant_algo": "NVFP4"
            },
            "model.language_model.layers.1.self_attn.v_proj": {
                "quant_algo": "FP8"
            },
        },
    })
    model_dir = write_model(tmp_path, config, None)

    payload = payload_for(
        model_dir,
        "--format",
        "json",
        "--include-skipped",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
        "--no-verify-safetensors-index",
    )

    qkv = next(r for r in payload["dense"] if r["op"] == "qkv_proj")
    assert qkv["call_status"] == "skipped"
    assert qkv["quant_format"] == "conflicting"
    assert qkv["warning"].startswith("skipped_conflicting_quant")


def test_compressed_tensors_config_groups_and_fused_ignore(tmp_path: Path):
    config = qwen_moe_config({
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "ignore": [
            "model.language_model.layers.1.self_attn.q_proj",
            "model.language_model.layers.1.self_attn.k_proj",
            "model.language_model.layers.1.self_attn.v_proj",
        ],
        "config_groups": {
            "dense_wna16": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "group_size": 128,
                    "symmetric": True,
                },
            },
            "moe_nvfp4": {
                "targets": ["FusedMoE"],
                "format": "nvfp4-pack-quantized",
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "group_size": 16,
                    "dynamic": False,
                },
            },
        },
    })
    model_dir = write_model(tmp_path, config, None)

    payload = payload_for(
        model_dir,
        "--format",
        "json",
        "--include-skipped",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
        "--no-verify-safetensors-index",
    )

    qkv = next(r for r in payload["dense"] if r["op"] == "qkv_proj")
    assert qkv["call_status"] == "skipped"
    assert qkv["target_op"] == "none"
    assert qkv["warning"] == "excluded_quant_module"
    assert any(r["phase"] == "skipped_router" for r in payload["dense"])

    o_proj = next(r for r in payload["dense"] if r["op"] == "o_proj")
    assert o_proj["call_status"] == "actual_marlin"
    assert o_proj["target_op"] == "ops.marlin_gemm"
    assert o_proj["quant_format"] == "uint4b8"
    assert o_proj["group_size"] == 128

    moe = next(
        r for r in payload["moe"]
        if r["op"] == "w13" and r["phase"] == "prefill"
    )
    assert moe["call_status"] == "actual_marlin"
    assert moe["target_op"] == "ops.moe_wna16_marlin_gemm"
    assert moe["quant_format"] == "nvfp4"
    assert moe["group_size"] == 16


def test_minimax_moe_linear_targets_are_recognized(tmp_path: Path):
    config = {
        "model_type": "minimax_m2",
        "architectures": ["MiniMaxM2ForCausalLM"],
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "pack-quantized",
            "ignore": [
                "model.layers.0.block_sparse_moe.gate",
                "lm_head",
            ],
            "config_groups": {
                "dense_linear": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 4,
                        "type": "int",
                        "group_size": 32,
                        "symmetric": True,
                    },
                }
            },
        },
        "text_config": {
            "model_type": "minimax_m2",
            "hidden_size": 3072,
            "intermediate_size": 1536,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "layer_types": ["full_attention", "full_attention"],
        },
    }
    keys = [
        "model.layers.0.block_sparse_moe.experts.0.w1.weight_packed",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale",
        "model.layers.0.block_sparse_moe.experts.0.w1.weight_shape",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight_packed",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight_scale",
        "model.layers.0.block_sparse_moe.experts.0.w2.weight_shape",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight_packed",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight_scale",
        "model.layers.0.block_sparse_moe.experts.0.w3.weight_shape",
        "model.layers.0.self_attn.q_proj.qweight",
        "model.layers.0.self_attn.q_proj.qzeros",
        "model.layers.0.self_attn.q_proj.scales",
        "model.layers.0.self_attn.k_proj.qweight",
        "model.layers.0.self_attn.k_proj.qzeros",
        "model.layers.0.self_attn.k_proj.scales",
        "model.layers.0.self_attn.v_proj.qweight",
        "model.layers.0.self_attn.v_proj.qzeros",
        "model.layers.0.self_attn.v_proj.scales",
        "model.layers.0.self_attn.o_proj.qweight",
        "model.layers.0.self_attn.o_proj.qzeros",
        "model.layers.0.self_attn.o_proj.scales",
    ]
    model_dir = write_model(tmp_path, config, keys)

    payload = payload_for(
        model_dir,
        "--format",
        "json",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp,tp_ep",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
    )

    moe_w13 = next(
        row for row in payload["moe"]
        if row["op"] == "w13" and row["phase"] == "prefill"
    )
    moe_w2 = next(
        row for row in payload["moe"]
        if row["op"] == "w2" and row["phase"] == "prefill"
    )
    assert moe_w13["call_status"] == "actual_marlin"
    assert moe_w13["quant_method"] == "compressed-tensors"
    assert moe_w13["quant_format"] == "uint4b8"
    assert moe_w13["group_size"] == 32
    assert moe_w13["marlin_path"] == "wna16_marlin"
    assert moe_w2["call_status"] == "actual_marlin"
    assert moe_w2["quant_format"] == "uint4b8"
    assert moe_w2["group_size"] == 32
    assert moe_w2["marlin_path"] == "wna16_marlin"

    dense = next(
        row for row in payload["dense"]
        if row["op"] == "qkv_proj" and row["phase"] == "prefill"
    )
    assert dense["call_status"] == "actual_marlin"
    assert dense["quant_format"] == "uint4b8"
    assert dense["group_size"] == 32


def test_architecture_specific_moe_specs(tmp_path: Path):
    minimax_dir = write_model(
        tmp_path,
        {
            "model_type": "minimax_m2",
            "hidden_size": 3072,
            "intermediate_size": 1536,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rotary_dim": 128,
            "rope_parameters": {},
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            },
        },
        None,
    )
    minimax_payload = payload_for(
        minimax_dir,
        "--format",
        "json",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
        "--no-verify-safetensors-index",
    )
    assert minimax_payload["model_config"]["num_experts"] == 256
    assert minimax_payload["model_config"]["moe_intermediate_size"] == 1536
    row = next(r for r in minimax_payload["dense"] if r["op"] == "qkv_proj")
    assert row["size_n"] == 2048

    glm_dir = write_model(
        tmp_path / "glm",
        {
            "model_type": "glm4_moe",
            "hidden_size": 5120,
            "intermediate_size": 12288,
            "moe_intermediate_size": 1536,
            "n_routed_experts": 160,
            "num_experts_per_tok": 8,
            "n_shared_experts": 1,
            "num_hidden_layers": 4,
            "num_attention_heads": 96,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "first_k_dense_replace": 3,
            "quantization_config": {
                "quant_method": "awq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
            },
        },
        None,
    )
    glm_payload = payload_for(
        glm_dir,
        "--format",
        "json",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
        "--no-verify-safetensors-index",
    )
    assert glm_payload["model_config"]["moe_layer_indices"] == [3]
    assert glm_payload["model_config"]["shared_expert_intermediate_size"] == 1536
    assert any(r["op"] == "gate_up_proj" for r in glm_payload["dense"])
    assert any(r["op"] == "w13" for r in glm_payload["moe"])


def test_cli_json_smoke(tmp_path: Path, capsys):
    model_dir = write_model(tmp_path, qwen_moe_config({}), None)
    rc = marlin_gemm_shapes.main([
        "--model",
        str(model_dir),
        "--format",
        "json",
        "--tp-sizes",
        "4",
        "--ep-modes",
        "tp",
        "--max-num-batched-tokens",
        "2048",
        "--decode-concurrency",
        "1",
        "--no-verify-safetensors-index",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "dense" in parsed
    assert "moe" in parsed
    assert parsed["warnings"] == ["hypothetical_bf16"]
    assert parsed["dense"]
    assert parsed["moe"]
    assert all(r["call_status"] == "hypothetical_bf16" for r in parsed["dense"])
    assert all(r["quant_method"] == "unquantized" for r in parsed["dense"])
    assert all(r["target_op"] == "none" for r in parsed["dense"])
    assert all(r["quant_format"] == "bf16_or_fp16" for r in parsed["dense"])
    assert all(r["marlin_path"] == "none" for r in parsed["dense"])
    assert all(r["call_status"] == "hypothetical_bf16" for r in parsed["moe"])
    assert all(r["quant_method"] == "unquantized" for r in parsed["moe"])
    assert all(r["target_op"] == "none" for r in parsed["moe"])
    assert all(r["quant_format"] == "bf16_or_fp16" for r in parsed["moe"])
    assert all(r["marlin_path"] == "none" for r in parsed["moe"])
