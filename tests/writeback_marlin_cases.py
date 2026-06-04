from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class DenseShapeCase:
    name: str
    size_m: int
    size_k: int
    size_n: int


@dataclass(frozen=True)
class MoeShapeCase:
    name: str
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    topk: int
    routing_profile: str


@dataclass(frozen=True)
class CtaCase:
    name: str
    value: str | None
    cta_m: int | None = None
    cta_n: int | None = None
    warps: int | None = None

    @property
    def is_auto(self) -> bool:
        return self.value is None


@dataclass(frozen=True)
class DenseWritebackClassCase:
    name: str
    class_name: str
    quant_names: tuple[str, ...]
    scalar_type_names: tuple[str, ...]
    zp_quant_names: tuple[str, ...]
    split_k_quant_names: tuple[str, ...]
    default_group_sizes: tuple[int, ...]
    benchmark_mode: str
    production_scope: str
    notes: str = ""


@dataclass(frozen=True)
class MoeWritebackClassCase:
    name: str
    class_name: str
    quant_names: tuple[str, ...]
    scalar_type_names: tuple[str, ...]
    zp_quant_names: tuple[str, ...]
    split_k_quant_names: tuple[str, ...]
    default_group_sizes: tuple[int, ...]
    benchmark_mode: str
    production_scope: str
    notes: str = ""


@dataclass(frozen=True)
class DenseWritebackMatrixCase:
    class_case: DenseWritebackClassCase
    quant_name: str
    group_size: int
    shape: DenseShapeCase
    split_k: str
    cta: CtaCase
    supported: bool
    reason: str = ""

    @property
    def id(self) -> str:
        return (
            f"{self.class_case.name}_{self.quant_name}_g{self.group_size}_"
            f"{self.shape.name}_sk{self.split_k}_cta{self.cta.name}"
        )


@dataclass(frozen=True)
class MoeWritebackMatrixCase:
    class_case: MoeWritebackClassCase
    quant_name: str
    group_size: int
    shape: MoeShapeCase
    split_k: str
    cta: CtaCase
    supported: bool
    reason: str = ""

    @property
    def id(self) -> str:
        return (
            f"{self.class_case.name}_{self.quant_name}_g{self.group_size}_"
            f"{self.shape.name}_sk{self.split_k}_cta{self.cta.name}"
        )


DENSE_M_VALUES = (1, 8, 16, 24, 32, 48, 64, 1024, 2048, 4096, 5120)
MOE_M_VALUES = DENSE_M_VALUES
MOE_ROUTING_PROFILES = ("uniform", "zipfian")


def _make_dense_shapes(
    templates: tuple[tuple[str, int, int], ...],
) -> tuple[DenseShapeCase, ...]:
    return tuple(
        DenseShapeCase(template.format(m=size_m), size_m, size_k, size_n)
        for size_m in DENSE_M_VALUES
        for template, size_k, size_n in templates
    )


def _make_moe_shapes(
    templates: tuple[tuple[str, int, int, int, int], ...],
) -> tuple[MoeShapeCase, ...]:
    return tuple(
        MoeShapeCase(
            template.format(m=tokens, r=routing_profile),
            tokens,
            hidden,
            intermediate,
            experts,
            topk,
            routing_profile,
        )
        for tokens in MOE_M_VALUES
        for template, hidden, intermediate, experts, topk in templates
        for routing_profile in MOE_ROUTING_PROFILES
    )


DENSE_HEAVY_SHAPE_CASES = _make_dense_shapes(
    (
        ("dense_heavy_qo_m{m}_k4096_n4096", 4096, 4096),
        ("dense_heavy_gqa_kv_m{m}_k4096_n1024", 4096, 1024),
        ("dense_heavy_mlp_up_m{m}_k4096_n14336", 4096, 14336),
        ("dense_heavy_mlp_down_m{m}_k14336_n4096", 14336, 4096),
    )
)

DENSE_ALIGNMENT_SHAPE_CASES = _make_dense_shapes(
    (
        ("dense_align_cta64_narrow_m{m}_k768_n192", 768, 192),
        ("dense_align_cta64_partial_n_m{m}_k1152_n320", 1152, 320),
        ("dense_align_cta64_residue_m{m}_k1792_n832", 1792, 832),
        ("dense_align_cta128_mid_n_m{m}_k768_n384", 768, 384),
        ("dense_align_cta128_partial_n_m{m}_k1152_n640", 1152, 640),
        ("dense_align_cta128_residue_m{m}_k1792_n1152", 1792, 1152),
        ("dense_align_cta256_tiny_square_m{m}_k512_n256", 512, 256),
    )
)

DENSE_STRESS_SHAPE_CASES = _make_dense_shapes(
    (
        ("dense_stress_cache_thrash_m{m}_k1024_n256", 1024, 256),
        ("dense_stress_splitk_starve_m{m}_k14336_n256", 14336, 256),
    )
)

DENSE_REGULAR_SHAPE_CASES = DENSE_HEAVY_SHAPE_CASES
DENSE_IRREGULAR_SHAPE_CASES = (
    DENSE_ALIGNMENT_SHAPE_CASES + DENSE_STRESS_SHAPE_CASES
)
DENSE_BENCHMARK_SHAPE_CASES = DENSE_REGULAR_SHAPE_CASES + DENSE_IRREGULAR_SHAPE_CASES


MOE_PRODUCTION_SHAPE_CASES = _make_moe_shapes(
    (
        ("moe_prod_mixtral_up_m{m}_h4096_i14336_e8_topk2_route_{r}", 4096, 14336, 8, 2),
        ("moe_prod_mixtral_down_m{m}_h14336_i4096_e8_topk2_route_{r}", 14336, 4096, 8, 2),
        ("moe_prod_deepseek_tp_m{m}_h7168_i2048_e8_topk2_route_{r}", 7168, 2048, 8, 2),
        ("moe_prod_small_square_m{m}_h2048_i2048_e8_topk2_route_{r}", 2048, 2048, 8, 2),
        ("moe_prod_70b_tp_m{m}_h8192_i3584_e8_topk2_route_{r}", 8192, 3584, 8, 2),
    )
)

MOE_ALIGNMENT_SHAPE_CASES = _make_moe_shapes(
    (
        ("moe_align_cta64_tiny_m{m}_h192_i96_e8_topk2_route_{r}", 192, 96, 8, 2),
        ("moe_align_cta64_partial_m{m}_h320_i160_e8_topk2_route_{r}", 320, 160, 8, 2),
        ("moe_align_cta64_residue_m{m}_h832_i416_e8_topk2_route_{r}", 832, 416, 8, 2),
        ("moe_align_cta128_tiny_m{m}_h384_i192_e8_topk2_route_{r}", 384, 192, 8, 2),
        ("moe_align_cta128_partial_m{m}_h640_i320_e8_topk2_route_{r}", 640, 320, 8, 2),
        ("moe_align_cta128_residue_m{m}_h1152_i576_e8_topk2_route_{r}", 1152, 576, 8, 2),
        ("moe_align_k_tail_m{m}_h3584_i4096_e8_topk2_route_{r}", 3584, 4096, 8, 2),
        ("moe_align_irregular_i_m{m}_h4096_i5120_e8_topk2_route_{r}", 4096, 5120, 8, 2),
        ("moe_align_thin_gate_m{m}_h4096_i1024_e8_topk2_route_{r}", 4096, 1024, 8, 2),
        ("moe_align_many_experts16_m{m}_h4096_i4096_e16_topk2_route_{r}", 4096, 4096, 16, 2),
        ("moe_align_many_experts64_m{m}_h4096_i4096_e64_topk2_route_{r}", 4096, 4096, 64, 2),
    )
)

MOE_STRESS_SHAPE_CASES = _make_moe_shapes(
    (
        ("moe_stress_draft_decode_m{m}_h2048_i8192_e8_topk2_route_{r}", 2048, 8192, 8, 2),
        ("moe_stress_topk1_latency_m{m}_h4096_i14336_e8_topk1_route_{r}", 4096, 14336, 8, 1),
        ("moe_stress_degenerate_dense_m{m}_h4096_i4096_e1_topk1_route_{r}", 4096, 4096, 1, 1),
    )
)

MOE_REGULAR_SHAPE_CASES = MOE_PRODUCTION_SHAPE_CASES
MOE_IRREGULAR_SHAPE_CASES = MOE_ALIGNMENT_SHAPE_CASES + MOE_STRESS_SHAPE_CASES
MOE_BENCHMARK_SHAPE_CASES = MOE_REGULAR_SHAPE_CASES + MOE_IRREGULAR_SHAPE_CASES


WRITEBACK_GROUP_SIZE_VALUES = (-1, 16, 32, 64, 128)
WRITEBACK_SPLIT_K_VALUES = ("unset", "1", "2", "4", "8")

DENSE_ALL_QUANT_NAMES = (
    "uint4",
    "uint4b8",
    "uint8",
    "uint8b128",
    "fp8",
    "nvfp4",
    "mxfp4",
    "float4_e2m1f",
)

MOE_ALL_QUANT_NAMES = (
    "uint4",
    "uint4b8",
    "uint8",
    "uint8b128",
    "fp8",
    "nvfp4",
    "mxfp4",
)

DENSE_SPLIT_K_ENV_BY_QUANT = {
    "uint4": "SM70_MARLIN_U4_SPLIT_K",
    "uint4b8": "SM70_MARLIN_U4B8_SPLIT_K",
    "uint8": "SM70_MARLIN_U8_SPLIT_K",
    "uint8b128": "SM70_MARLIN_U8B128_SPLIT_K",
    "fp8": "SM70_MARLIN_FP8_SPLIT_K",
    "nvfp4": "SM70_MARLIN_NVFP4_SPLIT_K",
    "mxfp4": "SM70_MARLIN_MXFP4_SPLIT_K",
    "float4_e2m1f": "SM70_MARLIN_MXFP4_SPLIT_K",
}

MOE_SPLIT_K_ENV_BY_QUANT = {
    "uint4": "SM70_MARLIN_MOE_U4_SPLIT_K",
    "uint4b8": "SM70_MARLIN_MOE_U4B8_SPLIT_K",
    "uint8": "SM70_MARLIN_MOE_U8_SPLIT_K",
    "uint8b128": "SM70_MARLIN_MOE_U8B128_SPLIT_K",
    "fp8": "SM70_MARLIN_MOE_FP8_SPLIT_K",
    "nvfp4": "SM70_MARLIN_MOE_NVFP4_SPLIT_K",
    "mxfp4": "SM70_MARLIN_MOE_MXFP4_SPLIT_K",
}

DENSE_CTA_ENV_BY_QUANT = {
    "uint4": "SM70_MARLIN_U4_CTA",
    "uint4b8": "SM70_MARLIN_U4B8_CTA",
    "uint8": "SM70_MARLIN_U8_CTA",
    "uint8b128": "SM70_MARLIN_U8B128_CTA",
    "fp8": "SM70_MARLIN_FP8_CTA",
    "nvfp4": "SM70_MARLIN_NVFP4_CTA",
    "mxfp4": "SM70_MARLIN_MXFP4_CTA",
    "float4_e2m1f": "SM70_MARLIN_MXFP4_CTA",
}

MOE_CTA_ENV_BY_QUANT = {
    "uint4": "SM70_MARLIN_MOE_U4_CTA",
    "uint4b8": "SM70_MARLIN_MOE_U4B8_CTA",
    "uint8": "SM70_MARLIN_MOE_U8_CTA",
    "uint8b128": "SM70_MARLIN_MOE_U8B128_CTA",
    "fp8": "SM70_MARLIN_MOE_FP8_CTA",
    "nvfp4": "SM70_MARLIN_MOE_NVFP4_CTA",
    "mxfp4": "SM70_MARLIN_MOE_MXFP4_CTA",
}

DENSE_CTA_CASES = (
    CtaCase("auto", None),
    CtaCase("32x128x4", "32x128x4", 32, 128, 4),
    CtaCase("32x256x4", "32x256x4", 32, 256, 4),
    CtaCase("64x64x4", "64x64x4", 64, 64, 4),
    CtaCase("64x128x4", "64x128x4", 64, 128, 4),
    CtaCase("64x128x8", "64x128x8", 64, 128, 8),
    CtaCase("64x256x4", "64x256x4", 64, 256, 4),
    CtaCase("64x256x8", "64x256x8", 64, 256, 8),
    CtaCase("128x64x4", "128x64x4", 128, 64, 4),
    CtaCase("128x64x8", "128x64x8", 128, 64, 8),
    CtaCase("128x128x4", "128x128x4", 128, 128, 4),
    CtaCase("128x128x8", "128x128x8", 128, 128, 8),
    CtaCase("128x256x8", "128x256x8", 128, 256, 8),
    CtaCase("256x64x4", "256x64x4", 256, 64, 4),
    CtaCase("256x64x8", "256x64x8", 256, 64, 8),
    CtaCase("256x128x8", "256x128x8", 256, 128, 8),
)

MOE_CTA_CASES = (
    CtaCase("auto", None),
    CtaCase("32x128x4", "32x128x4", 32, 128, 4),
    CtaCase("32x256x4", "32x256x4", 32, 256, 4),
    CtaCase("64x64x4", "64x64x4", 64, 64, 4),
    CtaCase("64x128x4", "64x128x4", 64, 128, 4),
    CtaCase("64x128x8", "64x128x8", 64, 128, 8),
    CtaCase("64x256x4", "64x256x4", 64, 256, 4),
    CtaCase("64x256x8", "64x256x8", 64, 256, 8),
)


DENSE_WRITEBACK_CLASS_CASES = (
    DenseWritebackClassCase(
        name="marlin_linear_kernel",
        class_name="MarlinLinearKernel",
        quant_names=(
            "uint4",
            "uint8",
            "uint4b8",
            "uint8b128",
            "fp8",
            "float4_e2m1f",
        ),
        scalar_type_names=(
            "uint4",
            "uint8",
            "uint4b8",
            "uint8b128",
            "float8_e4m3fn",
            "float4_e2m1f",
        ),
        zp_quant_names=("uint4", "uint8"),
        split_k_quant_names=(
            "uint4",
            "uint8",
            "uint4b8",
            "uint8b128",
            "fp8",
            "float4_e2m1f",
        ),
        default_group_sizes=(-1, 32, 64, 128),
        benchmark_mode="timed",
        production_scope="dense_mixed_precision_kernel",
        notes=(
            "float4_e2m1f is a direct scalar type supported by the kernel. "
            "User-facing NVFP4/MXFP4 production checkpoints are covered through "
            "their compressed-tensors schemes because those paths also prepare "
            "the required FP4 scales/global scales."
        ),
    ),
    DenseWritebackClassCase(
        name="gptq_marlin_linear_method",
        class_name="GPTQMarlinLinearMethod",
        quant_names=("uint4b8", "uint8b128"),
        scalar_type_names=("uint4b8", "uint8b128"),
        zp_quant_names=(),
        split_k_quant_names=("uint4b8", "uint8b128"),
        default_group_sizes=(-1, 32, 64, 128),
        benchmark_mode="timed",
        production_scope="dense_gptq_symmetric_method",
    ),
    DenseWritebackClassCase(
        name="awq_marlin_linear_method",
        class_name="AWQMarlinLinearMethod",
        quant_names=("uint4", "uint8"),
        scalar_type_names=("uint4", "uint8"),
        zp_quant_names=("uint4", "uint8"),
        split_k_quant_names=("uint4", "uint8"),
        default_group_sizes=(-1, 32, 64, 128),
        benchmark_mode="timed",
        production_scope="dense_awq_asymmetric_zp_method",
    ),
    DenseWritebackClassCase(
        name="compressed_tensors_wna16",
        class_name="CompressedTensorsWNA16",
        quant_names=("uint4", "uint4b8", "uint8", "uint8b128"),
        scalar_type_names=("uint4", "uint4b8", "uint8", "uint8b128"),
        zp_quant_names=("uint4", "uint8"),
        split_k_quant_names=("uint4", "uint4b8", "uint8", "uint8b128"),
        default_group_sizes=(-1, 32, 64, 128),
        benchmark_mode="timed",
        production_scope="dense_wna16_compressed_tensors_scheme",
    ),
    DenseWritebackClassCase(
        name="marlin_fp8_scaled_mm",
        class_name="MarlinFP8ScaledMMLinearKernel",
        quant_names=("fp8",),
        scalar_type_names=("float8_e4m3fn",),
        zp_quant_names=(),
        split_k_quant_names=("fp8",),
        default_group_sizes=(-1, 128),
        benchmark_mode="timed",
        production_scope="dense_fp8_scaled_mm_kernel",
    ),
    DenseWritebackClassCase(
        name="compressed_tensors_w8a16_fp8",
        class_name="CompressedTensorsW8A16Fp8",
        quant_names=("fp8",),
        scalar_type_names=("float8_e4m3fn",),
        zp_quant_names=(),
        split_k_quant_names=("fp8",),
        default_group_sizes=(-1, 128),
        benchmark_mode="timed",
        production_scope="dense_fp8_compressed_tensors_scheme",
    ),
    DenseWritebackClassCase(
        name="compressed_tensors_w4a16_nvfp4",
        class_name="CompressedTensorsW4A16Fp4",
        quant_names=("nvfp4",),
        scalar_type_names=("float4_e2m1f",),
        zp_quant_names=(),
        split_k_quant_names=("nvfp4",),
        default_group_sizes=(16,),
        benchmark_mode="timed",
        production_scope="dense_nvfp4_compressed_tensors_scheme",
    ),
    DenseWritebackClassCase(
        name="compressed_tensors_w4a16_mxfp4",
        class_name="CompressedTensorsW4A16Mxfp4",
        quant_names=("mxfp4",),
        scalar_type_names=("float4_e2m1f",),
        zp_quant_names=(),
        split_k_quant_names=("mxfp4",),
        default_group_sizes=(32,),
        benchmark_mode="timed",
        production_scope="dense_mxfp4_compressed_tensors_scheme",
    ),
)


MOE_WRITEBACK_CLASS_CASES = (
    MoeWritebackClassCase(
        name="gptq_moe",
        class_name="GPTQMarlinMoEMethod",
        quant_names=("uint4b8", "uint8b128"),
        scalar_type_names=("uint4b8", "uint8b128"),
        zp_quant_names=(),
        split_k_quant_names=("uint4b8", "uint8b128"),
        default_group_sizes=(-1, 32, 64, 128),
        benchmark_mode="timed",
        production_scope="moe_gptq_symmetric_method",
    ),
    MoeWritebackClassCase(
        name="awq_moe",
        class_name="AWQMarlinMoEMethod",
        quant_names=("uint4", "uint8"),
        scalar_type_names=("uint4", "uint8"),
        zp_quant_names=("uint4", "uint8"),
        split_k_quant_names=("uint4", "uint8"),
        default_group_sizes=(-1, 32, 64, 128),
        benchmark_mode="timed",
        production_scope="moe_awq_asymmetric_zp_method",
    ),
    MoeWritebackClassCase(
        name="compressed_tensors_wna16_moe",
        class_name="CompressedTensorsWNA16MarlinMoEMethod",
        quant_names=("uint4b8", "uint8b128"),
        scalar_type_names=("uint4b8", "uint8b128"),
        zp_quant_names=(),
        split_k_quant_names=("uint4b8", "uint8b128"),
        default_group_sizes=(-1, 32, 64, 128),
        benchmark_mode="timed",
        production_scope="moe_compressed_tensors_wna16_symmetric_method",
    ),
    MoeWritebackClassCase(
        name="quark_w8a8_fp8_moe",
        class_name="QuarkW8A8Fp8MoEMethod",
        quant_names=("fp8",),
        scalar_type_names=("float8_e4m3fn",),
        zp_quant_names=(),
        split_k_quant_names=(),
        default_group_sizes=(-1, 128),
        benchmark_mode="smoke_only",
        production_scope="moe_quark_fp8_marlin_method",
    ),
    MoeWritebackClassCase(
        name="compressed_tensors_w8a8_fp8_moe",
        class_name="CompressedTensorsW8A8Fp8MoEMethod",
        quant_names=("fp8",),
        scalar_type_names=("float8_e4m3fn",),
        zp_quant_names=(),
        split_k_quant_names=("fp8",),
        default_group_sizes=(-1, 128),
        benchmark_mode="pytest_only",
        production_scope="moe_compressed_tensors_fp8_method",
    ),
    MoeWritebackClassCase(
        name="compressed_tensors_w4a4_nvfp4_moe",
        class_name="CompressedTensorsW4A4Nvfp4MoEMethod",
        quant_names=("nvfp4",),
        scalar_type_names=("float4_e2m1f",),
        zp_quant_names=(),
        split_k_quant_names=("nvfp4",),
        default_group_sizes=(16,),
        benchmark_mode="pytest_only",
        production_scope="moe_compressed_tensors_nvfp4_method",
    ),
    MoeWritebackClassCase(
        name="compressed_tensors_w4a4_mxfp4_moe",
        class_name="CompressedTensorsW4A4Mxfp4MoEMethod",
        quant_names=("mxfp4",),
        scalar_type_names=("float4_e2m1f",),
        zp_quant_names=(),
        split_k_quant_names=("mxfp4",),
        default_group_sizes=(32,),
        benchmark_mode="pytest_only",
        production_scope="moe_compressed_tensors_mxfp4_method",
    ),
)


DENSE_WRITEBACK_CLASS_CASE_BY_NAME = {
    case.name: case for case in DENSE_WRITEBACK_CLASS_CASES
}
MOE_WRITEBACK_CLASS_CASE_BY_NAME = {
    case.name: case for case in MOE_WRITEBACK_CLASS_CASES
}
MOE_WRITEBACK_CLASS_CASE_BY_NAME.update(
    {
        "gptq": MOE_WRITEBACK_CLASS_CASE_BY_NAME["gptq_moe"],
        "awq": MOE_WRITEBACK_CLASS_CASE_BY_NAME["awq_moe"],
        "compressed_tensors_wna16": MOE_WRITEBACK_CLASS_CASE_BY_NAME[
            "compressed_tensors_wna16_moe"
        ],
    }
)


DENSE_WRITEBACK_CLASS_ALIASES = {
    case.name: case.name for case in DENSE_WRITEBACK_CLASS_CASES
}
DENSE_WRITEBACK_CLASS_ALIASES.update(
    {
        "kernel": "marlin_linear_kernel",
        "gptq": "gptq_marlin_linear_method",
        "awq": "awq_marlin_linear_method",
        "ct_wna16": "compressed_tensors_wna16",
        "ct_fp8": "compressed_tensors_w8a16_fp8",
        "ct_nvfp4": "compressed_tensors_w4a16_nvfp4",
        "ct_mxfp4": "compressed_tensors_w4a16_mxfp4",
    }
)

MOE_WRITEBACK_CLASS_ALIASES = {
    case.name: case.name for case in MOE_WRITEBACK_CLASS_CASES
}
MOE_WRITEBACK_CLASS_ALIASES.update(
    {
        "gptq": "gptq_moe",
        "awq": "awq_moe",
        "compressed_tensors_wna16": "compressed_tensors_wna16_moe",
        "ct_wna16": "compressed_tensors_wna16_moe",
        "quark_fp8": "quark_w8a8_fp8_moe",
        "ct_fp8": "compressed_tensors_w8a8_fp8_moe",
        "ct_nvfp4": "compressed_tensors_w4a4_nvfp4_moe",
        "ct_mxfp4": "compressed_tensors_w4a4_mxfp4_moe",
    }
)


def normalize_dense_class_name(case_name: str) -> str:
    try:
        return DENSE_WRITEBACK_CLASS_ALIASES[case_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dense writeback class case: {case_name!r}") from exc


def normalize_moe_class_name(case_name: str) -> str:
    try:
        return MOE_WRITEBACK_CLASS_ALIASES[case_name]
    except KeyError as exc:
        raise ValueError(f"Unknown MoE writeback class case: {case_name!r}") from exc


def dense_case_supports_quant(case_name: str, quant_name: str) -> bool:
    case_name = normalize_dense_class_name(case_name)
    return quant_name in DENSE_WRITEBACK_CLASS_CASE_BY_NAME[case_name].quant_names


def moe_case_supports_quant(case_name: str, quant_name: str) -> bool:
    case_name = normalize_moe_class_name(case_name)
    return quant_name in MOE_WRITEBACK_CLASS_CASE_BY_NAME[case_name].quant_names


def is_dense_group_size_supported(quant_name: str, group_size: int, size_k: int) -> bool:
    if group_size != -1 and size_k % group_size != 0:
        return False
    if quant_name == "nvfp4":
        return group_size == 16
    if quant_name == "mxfp4":
        return group_size == 32
    if quant_name == "float4_e2m1f":
        return group_size in (16, 32)
    if group_size == 16:
        return False
    if quant_name == "fp8":
        return group_size in (-1, 128)
    return group_size in (-1, 32, 64, 128)


def is_moe_group_size_supported(
    quant_name: str,
    group_size: int,
    hidden: int,
    intermediate: int,
) -> bool:
    if quant_name == "nvfp4":
        return group_size == 16 and hidden % 16 == 0 and intermediate % 16 == 0
    if quant_name == "mxfp4":
        return group_size == 32 and hidden % 32 == 0 and intermediate % 32 == 0
    if quant_name == "fp8":
        return group_size in (-1, 128) and (
            group_size == -1
            or (hidden % group_size == 0 and intermediate % group_size == 0)
        )
    return group_size in (-1, 32, 64, 128) and (
        group_size == -1
        or (hidden % group_size == 0 and intermediate % group_size == 0)
    )


def dense_auto_cta(size_m: int, size_n: int) -> CtaCase | None:
    if size_n % 256 == 0:
        cta_n = 256
    elif size_n % 128 == 0:
        cta_n = 128
    elif size_n % 64 == 0:
        cta_n = 64
    else:
        return None

    if cta_n == 64:
        cta_m = 256 if size_m >= 256 else 128 if size_m >= 128 else 64
    elif cta_n == 128:
        if size_m >= 256:
            cta_m = 256
        elif size_m >= 128:
            cta_m = 128
        elif size_m >= 64:
            cta_m = 64
        else:
            cta_m = 32
    else:
        cta_m = 128 if size_m >= 128 else 64 if size_m >= 64 else 32

    if cta_n == 64:
        warps = 4 if cta_m == 64 else 8
    elif cta_n == 128:
        warps = 4 if cta_m == 32 else 8
    else:
        warps = 4 if cta_m == 32 else 8
    value = f"{cta_m}x{cta_n}x{warps}"
    return CtaCase(value, value, cta_m, cta_n, warps)


def moe_auto_cta(size_n: int) -> CtaCase | None:
    if size_n % 256 == 0:
        cta_n = 256
        value = "32x256x4"
    elif size_n % 128 == 0:
        cta_n = 128
        value = "32x128x4"
    elif size_n % 64 == 0:
        cta_n = 64
        value = "64x64x4"
    else:
        return None
    cta_m, _cta_n, warps = (int(part) for part in value.split("x"))
    assert _cta_n == cta_n
    return CtaCase(value, value, cta_m, cta_n, warps)


def dense_cta_case_supported(cta: CtaCase, shape: DenseShapeCase) -> tuple[bool, str]:
    auto = dense_auto_cta(shape.size_m, shape.size_n)
    if auto is None:
        return False, "shape size_n is not divisible by 64"
    if cta.is_auto:
        return True, ""
    if cta.cta_m == auto.cta_m and cta.cta_n == auto.cta_n:
        return True, ""
    return (
        False,
        (
            f"dense explicit CTA {cta.name} does not match auto "
            f"CTA_M/CTA_N {auto.cta_m}x{auto.cta_n} for shape {shape.name}"
        ),
    )


def moe_cta_case_supported(cta: CtaCase, shape: MoeShapeCase) -> tuple[bool, str]:
    stage1_n = 2 * shape.intermediate
    stage2_n = shape.hidden
    auto_stage1 = moe_auto_cta(stage1_n)
    auto_stage2 = moe_auto_cta(stage2_n)
    if auto_stage1 is None or auto_stage2 is None:
        return False, "MoE stage N dimension is not divisible by 64"
    if cta.is_auto:
        return True, ""
    assert cta.cta_n is not None
    if cta.cta_n != auto_stage1.cta_n or cta.cta_n != auto_stage2.cta_n:
        return (
            False,
            (
                f"MoE explicit CTA_N={cta.cta_n} does not match both "
                f"stage1 auto CTA_N={auto_stage1.cta_n} and stage2 "
                f"auto CTA_N={auto_stage2.cta_n}; one CTA env is applied "
                "to both Marlin MoE GEMMs"
            ),
        )
    return True, ""


def dense_matrix_support_reason(
    class_case: DenseWritebackClassCase,
    quant_name: str,
    group_size: int,
    shape: DenseShapeCase,
    split_k: str,
    cta: CtaCase,
) -> tuple[bool, str]:
    if quant_name not in class_case.quant_names:
        return False, "unsupported dense writeback class/quant combination"
    if group_size not in class_case.default_group_sizes:
        return False, "group_size is not a supported default for this dense class"
    if not is_dense_group_size_supported(quant_name, group_size, shape.size_k):
        return False, "unsupported dense quant/group/shape alignment combination"
    if split_k != "unset" and quant_name not in class_case.split_k_quant_names:
        return False, "split-K is not supported for this dense class/quant"
    if quant_name == "float4_e2m1f":
        return (
            False,
            (
                "direct float4_e2m1f scalar support is inventory-only; "
                "production FP4 paths are NVFP4/MXFP4 schemes"
            ),
        )
    return dense_cta_case_supported(cta, shape)


def moe_matrix_support_reason(
    class_case: MoeWritebackClassCase,
    quant_name: str,
    group_size: int,
    shape: MoeShapeCase,
    split_k: str,
    cta: CtaCase,
) -> tuple[bool, str]:
    if quant_name not in class_case.quant_names:
        return False, "unsupported MoE writeback class/quant combination"
    if class_case.benchmark_mode in {"pytest_only", "smoke_only"}:
        return (
            False,
            (
                "MoE modular/smoke class is covered by dedicated class-path "
                "smoke tests; the standalone oracle kernel stubs do not "
                "support full matrix execution"
            ),
        )
    if group_size not in class_case.default_group_sizes:
        return False, "group_size is not a supported default for this MoE class"
    if not is_moe_group_size_supported(
        quant_name,
        group_size,
        shape.hidden,
        shape.intermediate,
    ):
        return False, "unsupported MoE quant/group/shape alignment combination"
    if split_k != "unset" and quant_name not in class_case.split_k_quant_names:
        return False, "split-K is not supported for this MoE class/quant"
    return moe_cta_case_supported(cta, shape)


def iter_dense_writeback_matrix(
    *,
    class_cases: tuple[DenseWritebackClassCase, ...] = DENSE_WRITEBACK_CLASS_CASES,
    quant_names: tuple[str, ...] = DENSE_ALL_QUANT_NAMES,
    group_sizes: tuple[int, ...] = WRITEBACK_GROUP_SIZE_VALUES,
    shapes: tuple[DenseShapeCase, ...] = DENSE_BENCHMARK_SHAPE_CASES,
    split_k_values: tuple[str, ...] = WRITEBACK_SPLIT_K_VALUES,
    cta_cases: tuple[CtaCase, ...] = DENSE_CTA_CASES,
) -> Iterator[DenseWritebackMatrixCase]:
    for class_case in class_cases:
        for quant_name in quant_names:
            for group_size in group_sizes:
                for shape in shapes:
                    for split_k in split_k_values:
                        for cta in cta_cases:
                            supported, reason = dense_matrix_support_reason(
                                class_case,
                                quant_name,
                                group_size,
                                shape,
                                split_k,
                                cta,
                            )
                            yield DenseWritebackMatrixCase(
                                class_case=class_case,
                                quant_name=quant_name,
                                group_size=group_size,
                                shape=shape,
                                split_k=split_k,
                                cta=cta,
                                supported=supported,
                                reason=reason,
                            )


def iter_moe_writeback_matrix(
    *,
    class_cases: tuple[MoeWritebackClassCase, ...] = MOE_WRITEBACK_CLASS_CASES,
    quant_names: tuple[str, ...] = MOE_ALL_QUANT_NAMES,
    group_sizes: tuple[int, ...] = WRITEBACK_GROUP_SIZE_VALUES,
    shapes: tuple[MoeShapeCase, ...] = MOE_BENCHMARK_SHAPE_CASES,
    split_k_values: tuple[str, ...] = WRITEBACK_SPLIT_K_VALUES,
    cta_cases: tuple[CtaCase, ...] = MOE_CTA_CASES,
) -> Iterator[MoeWritebackMatrixCase]:
    for class_case in class_cases:
        for quant_name in quant_names:
            for group_size in group_sizes:
                for shape in shapes:
                    for split_k in split_k_values:
                        for cta in cta_cases:
                            supported, reason = moe_matrix_support_reason(
                                class_case,
                                quant_name,
                                group_size,
                                shape,
                                split_k,
                                cta,
                            )
                            yield MoeWritebackMatrixCase(
                                class_case=class_case,
                                quant_name=quant_name,
                                group_size=group_size,
                                shape=shape,
                                split_k=split_k,
                                cta=cta,
                                supported=supported,
                                reason=reason,
                            )


def dense_writeback_matrix_summary(
    **kwargs,
) -> dict[str, object]:
    status = Counter()
    classes = Counter()
    quants = Counter()
    groups = Counter()
    shapes = Counter()
    split_k = Counter()
    ctas = Counter()
    skip_reasons = Counter()
    for case in iter_dense_writeback_matrix(**kwargs):
        status["total"] += 1
        classes[case.class_case.name] += 1
        quants[case.quant_name] += 1
        groups[case.group_size] += 1
        shapes[case.shape.name] += 1
        split_k[case.split_k] += 1
        ctas[case.cta.name] += 1
        if case.supported:
            status["supported"] += 1
        else:
            status["skipped"] += 1
            skip_reasons[case.reason] += 1
    return {
        "total": status["total"],
        "supported": status["supported"],
        "skipped": status["skipped"],
        "shape_count": len(shapes),
        "class": dict(sorted(classes.items())),
        "quant": dict(sorted(quants.items())),
        "group_size": dict(sorted(groups.items())),
        "split_k": dict(sorted(split_k.items())),
        "cta": dict(sorted(ctas.items())),
        "skip_reasons": dict(sorted(skip_reasons.items())),
    }


def moe_writeback_matrix_summary(
    **kwargs,
) -> dict[str, object]:
    status = Counter()
    classes = Counter()
    quants = Counter()
    groups = Counter()
    shapes = Counter()
    split_k = Counter()
    ctas = Counter()
    routing = Counter()
    skip_reasons = Counter()
    for case in iter_moe_writeback_matrix(**kwargs):
        status["total"] += 1
        classes[case.class_case.name] += 1
        quants[case.quant_name] += 1
        groups[case.group_size] += 1
        shapes[case.shape.name] += 1
        split_k[case.split_k] += 1
        ctas[case.cta.name] += 1
        routing[case.shape.routing_profile] += 1
        if case.supported:
            status["supported"] += 1
        else:
            status["skipped"] += 1
            skip_reasons[case.reason] += 1
    return {
        "total": status["total"],
        "supported": status["supported"],
        "skipped": status["skipped"],
        "shape_count": len(shapes),
        "class": dict(sorted(classes.items())),
        "quant": dict(sorted(quants.items())),
        "group_size": dict(sorted(groups.items())),
        "split_k": dict(sorted(split_k.items())),
        "cta": dict(sorted(ctas.items())),
        "routing_profile": dict(sorted(routing.items())),
        "skip_reasons": dict(sorted(skip_reasons.items())),
    }
