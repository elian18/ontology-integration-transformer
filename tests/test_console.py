"""The console says '1 llamada' and '2 llamadas': every AI line agrees with its number.

S10-T01: the last plurals written by hand also use counted, and the embedding model loads without
the warning of the HF Hub and the "Loading weights" bar."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys

import pytest

from app.services import align_ontology as ao
from app.services import base_questions as bq
from src.alignment import law_definitions as ld
from src.alignment import user_alignment as ua
from src.config import QUIET_HUB, ROOT
from src.console import counted
from src.core import law_concepts as lc
from src.transform import labels as lab
from src.transform import roundtrip as rt
from src.validation import ai_auditor as au
from src.validation import query_drafter as qd
from src.validation import question_catalog as qc
from src.validation import question_explainer as qe
from src.validation import questions as qs
from src.validation import shapes as sh

# The files of S10-T01 and the hand-written plural they had: concepto{'' if n == 1 else 's'}
CLEANED = ("app/services/align_ontology.py", "app/services/base_questions.py",
           "src/validation/questions.py", "src/validation/question_catalog.py",
           "src/transform/roundtrip.py")
HAND_PLURAL = re.compile(r"\{'(s|n|es|)' if [^{}]* else '(s|n|es|)'\}|def _n\(")
HUB_WARNING = "aviso del Hub"


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


# --------------------------------------------------------------------------- S10-T01

def test_the_last_hand_written_plurals_use_counted():
    for module in (ao, bq, qs, qc, rt):
        assert module.counted is counted
        assert not hasattr(module, "_n")
    for path in CLEANED:
        assert not HAND_PLURAL.search((ROOT / path).read_text(encoding="utf-8")), path


@pytest.mark.parametrize("n, words", [(1, "1 concepto"), (2, "2 conceptos")])
def test_answer_and_criterion_agree_with_the_number(n, words):
    assert qs.answer_text({"kind": qc.AT_LEAST, "color": qs.GREEN, "answer": [],
                           "count": n}) == words
    question = qc.Question(id="G99", group=qc.GENERIC, question="¿?", source="generic",
                           reference="", modes=("base",), kind=qc.AT_LEAST, min=n)
    assert qc.criterion_text(question) == (f"Verde si la respuesta trae al menos {words}; "
                                           "amarillo si no trae ninguno.")


@pytest.mark.parametrize("n, words", [(1, "1 no aplica"), (2, "2 no aplican")])
def test_not_applicable_agrees_with_the_number(n, words):
    counts = {qs.GREEN: 1, qs.YELLOW: 0, qs.GRAY: 0, qs.NOT_APPLICABLE: n}
    assert qs.counts_text(counts).endswith(f"; {words}")


@pytest.mark.parametrize("n, words", [(1, "(1 archivo, 1 tripleta)"),
                                      (2, "(2 archivos, 2 tripletas)")])
def test_round_trip_result_agrees_with_the_number(n, words):
    reports = [rt.FileReport(f"f{i}.ttl", f"f{i}.ttl", "turtle", 1, 0) for i in range(n)]
    assert words in rt.render_console(reports, [])


def _load_hub(**env) -> tuple[str, str]:
    """Imports src.config and then huggingface_hub in a fresh process, as the console does."""
    clean = {k: v for k, v in os.environ.items() if k not in QUIET_HUB}
    code = ("import logging, src.config\n"
            "from huggingface_hub.utils import are_progress_bars_disabled\n"
            f"logging.getLogger('huggingface_hub.utils._http').warning('{HUB_WARNING}')\n"
            "print(are_progress_bars_disabled())")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**clean, **env},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return done.stdout.strip(), done.stderr


def test_the_hub_loads_without_warning_or_bars():
    shown, errors = _load_hub()
    assert shown == "True" and HUB_WARNING not in errors


def test_env_can_bring_the_hub_messages_back():
    shown, errors = _load_hub(HF_HUB_VERBOSITY="warning", HF_HUB_DISABLE_PROGRESS_BARS="0")
    assert shown == "False" and HUB_WARNING in errors


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None,
                    reason="sin transformers (lo trae sentence-transformers)")
def test_the_loading_weights_bar_is_off():
    clean = {k: v for k, v in os.environ.items() if k not in QUIET_HUB}
    code = ("import src.config\n"
            "from transformers.utils import logging as tl\n"
            "print(tl.is_progress_bar_enabled())")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=clean,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().splitlines()[-1] == "False"
