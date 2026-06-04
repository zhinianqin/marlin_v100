from __future__ import annotations


def get_tensor_model_parallel_world_size() -> int:
    return 1


def get_tensor_model_parallel_rank() -> int:
    return 0


class _DummyGroup:
    rank_in_group = 0
    world_size = 1

    def reduce_scatter(self, tensor, dim=0):
        return tensor


def get_dp_group() -> _DummyGroup:
    return _DummyGroup()


def get_pcp_group() -> _DummyGroup:
    return _DummyGroup()
