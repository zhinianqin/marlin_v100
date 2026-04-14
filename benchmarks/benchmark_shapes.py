from __future__ import annotations

DENSE_WEIGHT_SHAPES: dict[str, list[tuple[int, int]]] = {
    "smoke": [
        (256, 256),
        (512, 512),
    ],
    "ideal": [
        (4096, 4096),
    ],
    "llama2_7b_tp1": [
        (4096, 12288),
        (4096, 4096),
        (4096, 22016),
        (11008, 4096),
    ],
}

DENSE_PRESETS: dict[str, dict[str, list[int] | list[str]]] = {
    "smoke": {
        "models": ["smoke"],
        "batch_sizes": [1, 16, 64],
    },
    "quick": {
        "models": ["ideal"],
        "batch_sizes": [1, 16, 64, 256],
    },
    "full": {
        "models": ["ideal", "llama2_7b_tp1"],
        "batch_sizes": [1, 16, 64, 256, 1024, 4096],
    },
}

MOE_CASES: dict[str, dict[str, int]] = {
    "smoke": {
        "experts": 4,
        "topk": 2,
        "hidden": 128,
        "intermediate": 128,
    },
    "medium": {
        "experts": 8,
        "topk": 2,
        "hidden": 512,
        "intermediate": 512,
    },
    "large": {
        "experts": 8,
        "topk": 2,
        "hidden": 1024,
        "intermediate": 1024,
    },
}

MOE_PRESETS: dict[str, dict[str, list[int] | list[str]]] = {
    "quick": {
        "cases": ["smoke"],
        "tokens": [4, 16],
    },
    "full": {
        "cases": ["smoke", "medium", "large"],
        "tokens": [4, 16, 64, 128],
    },
}
