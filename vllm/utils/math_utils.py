from __future__ import annotations


def round_up(value: int, multiple: int) -> int:
    if multiple == 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
