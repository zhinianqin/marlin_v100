from __future__ import annotations

import torch


def num_compute_units(device_id: int = 0) -> int:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(device_id).multi_processor_count
    return 1

