from __future__ import annotations

import numpy as np
import torch


def get_pack_factor(num_bits: int) -> int:
    if 32 % num_bits != 0:
        raise ValueError(f"Unsupported num_bits={num_bits}")
    return 32 // num_bits


def get_weight_perm(num_bits: int, is_a_8bit: bool = False) -> torch.Tensor:
    perm_list: list[int] = []
    if is_a_8bit:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3,
                    4 * (i % 4 + 4),
                    4 * (i % 4 + 4) + 1,
                    4 * (i % 4 + 4) + 2,
                    4 * (i % 4 + 4) + 3,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(2):
                perm_list.extend([p + 512 * j for p in perm1])
    else:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)
    if num_bits == 4:
        interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7]) if is_a_8bit else np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        interleave = np.array([0, 1, 2, 3]) if is_a_8bit else np.array([0, 2, 1, 3])
    else:
        raise ValueError(f"Unsupported num_bits={num_bits}")
    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    return torch.from_numpy(perm)


def marlin_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    perm: torch.Tensor,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    if is_a_8bit:
        q_w = q_w.reshape((size_k // 32, 32, size_n // 16, 16))
    else:
        q_w = q_w.reshape((size_k // 16, 16, size_n // 16, 16))
    q_w = q_w.permute((0, 2, 1, 3)).reshape((size_k // 16, size_n * 16))
    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    pack_factor = get_pack_factor(num_bits)
    q_w_np = q_w.cpu().numpy().astype(np.uint32)
    q_packed = np.zeros((q_w_np.shape[0], q_w_np.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w_np[:, i::pack_factor] << num_bits * i
    return torch.from_numpy(q_packed.astype(np.int32)).to(q_w.device)
