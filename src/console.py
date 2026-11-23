from __future__ import annotations


def counted(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"
