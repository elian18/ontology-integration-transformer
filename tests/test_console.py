"""The console says '1 llamada' and '2 llamadas': every AI line agrees with its number."""

from __future__ import annotations

import pytest

from src.alignment import law_definitions as ld
from src.alignment import user_alignment as ua
from src.console import counted
from src.core import law_concepts as lc
from src.transform import labels as lab
from src.validation import ai_auditor as au
from src.validation import query_drafter as qd
from src.validation import question_explainer as qe
from src.validation import shapes as sh


@pytest.mark.parametrize("n, text", [(0, "0 llamadas"), (1, "1 llamada"), (2, "2 llamadas"),
                                     (11, "11 llamadas")])
def test_counted_agrees_with_the_number(n, text):
    assert counted(n, "llamada", "llamadas") == text


def test_the_old_helpers_are_the_shared_one():
    assert sh.counted is counted and ld.counted is counted


def _ai(**values) -> dict:
    return {"available": True, "needed": True, "calls": 1, "cached": 1, "invalid": 1,
            "errors": [], "quota_exhausted": False, "call_limit_reached": False, **values}


def test_one_of_each_is_singular():
    assert ua.decider_ai_text(_ai()) == "1 llamada, 1 respuesta desde la caché"
    assert lab.ai_line({"ai": _ai()}) == ("1 llamada, 1 desde la caché, 1 respuesta inválida")
    assert au.ai_text(_ai()) == "1 llamada, 1 desde la caché, 1 inválida"
    assert qe.ai_text({**_ai(), "questions": 1}) == "1 llamada, 1 desde la caché, 1 inválida"
    assert qd.ai_text(_ai()) == "1 llamada; 1 borrador desde la caché; 1 inválido"
    assert ld.ai_line({"law_concepts": {"ai": _ai()}, "ai": _ai()}) == (
        "extracción 1 llamada; definiciones 1 llamada, 1 desde la caché")
    assert lc._ai_text({"ai": _ai(), "extraction": {"batches_with_errors": 1}}) == (
        "1 llamada en esta corrida, 1 lote con error")


def test_two_of_each_is_plural():
    two = _ai(calls=2, cached=2, invalid=2)
    assert ua.decider_ai_text(two) == "2 llamadas, 2 respuestas desde la caché"
    assert lab.ai_line({"ai": two}) == "2 llamadas, 2 desde la caché, 2 respuestas inválidas"
    assert au.ai_text(two) == "2 llamadas, 2 desde la caché, 2 inválidas"
    assert qe.ai_text({**two, "questions": 2}) == "2 llamadas, 2 desde la caché, 2 inválidas"
    assert qd.ai_text(two) == "2 llamadas; 2 borradores desde la caché; 2 inválidos"
    assert ld.ai_line({"law_concepts": {"ai": two}, "ai": two}) == (
        "extracción 2 llamadas; definiciones 2 llamadas, 2 desde la caché")
    assert lc._ai_text({"ai": two, "extraction": {"batches_with_errors": 2}}) == (
        "2 llamadas en esta corrida, 2 lotes con error")


def test_zero_is_plural():
    zero = _ai(calls=0, cached=0, invalid=0)
    assert ua.decider_ai_text(zero) == "0 llamadas, 0 respuestas desde la caché"
    assert au.ai_text(zero) == "0 llamadas, 0 desde la caché, 0 inválidas"
    assert lc._ai_text({"ai": zero, "extraction": {"batches_with_errors": 0}}) == (
        "0 llamadas en esta corrida, 0 lotes con error")
