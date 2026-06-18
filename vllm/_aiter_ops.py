from __future__ import annotations


class _RocmAiterOps:
    def is_fused_moe_enabled(self) -> bool:
        return False

    def shuffle_weights(self, *args, **kwargs):
        raise NotImplementedError("ROCm AITER is not available in marlin_v100")


rocm_aiter_ops = _RocmAiterOps()
