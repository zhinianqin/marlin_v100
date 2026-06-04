from __future__ import annotations


def is_list_of(value, typ) -> bool:
    return isinstance(value, list) and all(isinstance(item, typ) for item in value)
