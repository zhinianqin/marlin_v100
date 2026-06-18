from __future__ import annotations

from enum import Enum


class FusedMoEActivationFormat(Enum):
    Standard = "standard"
    BatchedExperts = "batched_experts"


class FusedMoEPrepareAndFinalizeModular:
    def activation_format(self) -> FusedMoEActivationFormat:
        return FusedMoEActivationFormat.Standard

    def topk_indices_dtype(self):
        return None


class FusedMoEExpertsModular:
    @classmethod
    def is_monolithic(cls) -> bool:
        return False


class FusedMoEKernel:
    shared_experts = None
    is_monolithic = False

    def __init__(self, *args, **kwargs):
        pass


class TopKWeightAndReduce:
    pass


class FusedMoEActivation:
    pass


class ExpertTokensMetadata:
    pass
