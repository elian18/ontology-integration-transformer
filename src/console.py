"""Small helpers shared by the console texts of every module."""

from __future__ import annotations


def counted(n: int, one: str, many: str) -> str:
    """'1 llamada', '3 llamadas': the number with the word that agrees with it."""
    return f"{n} {one if n == 1 else many}"
