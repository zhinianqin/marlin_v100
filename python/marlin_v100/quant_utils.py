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


_SM70_ROW_GROUPS = (
    (0, 1, 8, 9),
    (2, 3, 10, 11),
    (4, 5, 12, 13),
    (6, 7, 14, 15),
)
_SM70_U4_PACK_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)
_SM70_U4_MACRO_N_TILES = 4
_SM70_U4_ZERO_WORDS_PER_CTA_N = 16
_SM70_U4_ZERO_WORD_PAIR_ORDER = (
    0,
    8,
    1,
    9,
    2,
    10,
    3,
    11,
    4,
    12,
    5,
    13,
    6,
    14,
    7,
    15,
)
_SM70_U8_PACK_ORDER = (0, 2, 1, 3)


def _legacy_marlin_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    perm: torch.Tensor,
    is_a_8bit: bool,
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


def _pack_sm70_native_tile(q_tile: np.ndarray, num_bits: int) -> np.ndarray:
    if q_tile.shape != (16, 64):
        raise ValueError(f"Expected a 16x64 tile, got {q_tile.shape}")

    if num_bits == 4:
        packed = np.empty((16, 8), dtype=np.uint32)
        for local_k in range(16):
            for local_n_vec in range(8):
                vals = [
                    int(q_tile[local_k, local_n_vec * 8 + n])
                    for n in range(8)
                ]
                word = 0
                for out_idx, src_idx in enumerate(_SM70_U4_PACK_ORDER):
                    word |= vals[src_idx] << (num_bits * out_idx)
                packed[local_k, local_n_vec] = np.uint32(word)
        return packed.reshape(-1)

    if num_bits == 8:
        packed = np.empty((16, 16), dtype=np.uint32)
        for local_k in range(16):
            for local_n_word in range(16):
                vals = [
                    int(q_tile[local_k, local_n_word * 4 + n])
                    for n in range(4)
                ]
                word = 0
                for out_idx, src_idx in enumerate(_SM70_U8_PACK_ORDER):
                    word |= vals[src_idx] << (num_bits * out_idx)
                packed[local_k, local_n_word] = np.uint32(word)
        return packed.reshape(-1)

    raise ValueError(f"Unsupported num_bits={num_bits}")


def _sm70_u4_macro_n_offset(
    n_tiles: int,
    n_tile: int,
    local_word: int,
) -> int:
    macro_n_tile = n_tile // _SM70_U4_MACRO_N_TILES
    macro_first_n_tile = macro_n_tile * _SM70_U4_MACRO_N_TILES
    subtile = n_tile - macro_first_n_tile
    subtile_count = min(_SM70_U4_MACRO_N_TILES, n_tiles - macro_first_n_tile)
    tile_words = 16 * 64 // get_pack_factor(4)
    return (
        macro_n_tile * _SM70_U4_MACRO_N_TILES * tile_words
        + local_word * subtile_count
        + subtile
    )


def _sm70_u8_macro_n_offset(
    n_tiles: int,
    n_tile: int,
    local_word: int,
) -> int:
    macro_n_tile = n_tile // _SM70_U4_MACRO_N_TILES
    macro_first_n_tile = macro_n_tile * _SM70_U4_MACRO_N_TILES
    subtile = n_tile - macro_first_n_tile
    subtile_count = min(_SM70_U4_MACRO_N_TILES, n_tiles - macro_first_n_tile)
    tile_words = 16 * 64 // get_pack_factor(8)
    return (
        macro_n_tile * _SM70_U4_MACRO_N_TILES * tile_words
        + local_word * subtile_count
        + subtile
    )


def marlin_weights(
    q_w: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    perm: torch.Tensor,
    is_a_8bit: bool = False,
) -> torch.Tensor:
    if is_a_8bit:
        return _legacy_marlin_weights(q_w, size_k, size_n, num_bits, perm, is_a_8bit)

    if size_k % 16 != 0 or size_n % 64 != 0:
        raise ValueError(f"SM70 native Marlin layout expects size_k%16==0 and size_n%64==0, got {(size_k, size_n)}")
    if num_bits not in (4, 8):
        raise ValueError(f"Unsupported num_bits={num_bits}")

    q_w_np = q_w.detach().cpu().numpy().astype(np.uint32, copy=False)
    pack_factor = get_pack_factor(num_bits)
    tile_words = (16 * 64) // pack_factor
    n_tiles = size_n // 64

    packed = np.empty((size_k // 16, n_tiles * tile_words), dtype=np.uint32)
    for k_tile in range(size_k // 16):
        row_start = 16 * k_tile
        row_stop = row_start + 16
        for n_tile in range(n_tiles):
            col_start = 64 * n_tile
            col_stop = col_start + 64
            tile = _pack_sm70_native_tile(
                q_w_np[row_start:row_stop, col_start:col_stop],
                num_bits,
            )
            for local_word, word in enumerate(tile):
                if num_bits == 4:
                    word_offset = _sm70_u4_macro_n_offset(
                        n_tiles,
                        n_tile,
                        local_word,
                    )
                else:
                    word_offset = _sm70_u8_macro_n_offset(
                        n_tiles,
                        n_tile,
                        local_word,
                    )
                packed[k_tile, word_offset] = word
    return torch.from_numpy(packed.astype(np.int32)).to(q_w.device)
