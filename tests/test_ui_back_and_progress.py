"""S10-T11: while it runs, the assistant shows a progress bar with each step in plain words; the
uploaded files are kept, so "Atrás" goes back to them without uploading again or computing
again. The LLM and the embedder are fakes."""
from __future__ import annotations

import importlib

import pytest

from app.services import align_ontology as svc
from src.workspace.store import ALL_STEPS, STEP_LABELS, WorkspaceError, steps_for
from tests.test_alignment_propose import TTL, env, targets  # noqa: F401
from tests.test_law_evidence import LAW_E2E
from tests.test_ui_align_ontology import (TITLE, _pick, _titles, _zips, cfg,  # noqa: F401
                                          live, page)


def test_every_step_in_plain_words():
    assert set(svc.STEP_TEXT) == set(ALL_STEPS) == set(STEP_LABELS)
    assert svc.progress_of("law", STEP_LABELS["segment"]) == ("Leyendo tu ley…", 0.0)
    steps = steps_for("ontology+law")
    text, done = svc.progress_of("ontology+law", STEP_LABELS["package"])
    assert (text, done) == ("Preparando la descarga…", (len(steps) - 1) / len(steps))
    assert svc.progress_of("ontology", STEP_LABELS["propose"]) == (
        "Eligiendo con la IA el mejor equivalente…", 2 / 10)
    assert svc.progress_of("ontology", STEP_LABELS["segment"])[1] == 0.0    # not its step
    assert svc.progress_of("law", "Otra cosa") == ("Otra cosa…", 0.0)
    assert not any(w in " ".join(svc.STEP_TEXT.values())
                   for w in ("SHACL", "DPV", "razonador", "Indexar", "OntoPriv"))


def test_sizes():
    assert svc.file_size(378_449) == "378,4 KB"                # sample.ttl
    assert svc.file_size(882_927) == "882,9 KB"                # ontopriv.rdf
    assert svc.file_size(999_949) == "999,9 KB" and svc.file_size(1_000_000) == "1,0 MB"


@pytest.fixture
def counted(live, monkeypatch):
    """The live page, counting the runs and the uploaders drawn."""
    import streamlit
    page, uploads = live
    view_svc = importlib.import_module("services.align_ontology")
    calls, real = [], view_svc.run
    monkeypatch.setattr(view_svc, "run", lambda *a, **k: calls.append(a[0]) or real(*a, **k))
    drawn, uploader = [], streamlit.file_uploader
    monkeypatch.setattr(streamlit, "file_uploader",
                        lambda label, *a, **k: drawn.append(label) or uploader(label, *a, **k))
    return page, uploads, calls, drawn


READY = "¡Tu ontología está lista!"


def _kept(at) -> list[str]:
    return [m.value for m in at.markdown if m.value.startswith(":material/description:")]


def _files(at) -> list[str]:
    """The kinds with a kept file (the fake uploader of ``live`` draws nothing)."""
    return list(at.session_state["align_files"]) if "align_files" in at.session_state else []


def test_back_keeps_the_files_and_computes_nothing(counted):
    page, uploads, calls, drawn = counted
    at = page()
    at.run()
    _pick(at, "ontology")
    uploads["ontology"] = ("Mi GDPR.ttl", TTL.encode("utf-8"))
    at.run()
    assert calls == ["ontology"] and _titles(at) == [READY]
    drawn.clear()
    at.button(key="align_result_back").click().run()          # the result: "Atrás"
    assert not at.exception and _titles(at) == ["Mi ontología"]
    assert not drawn and _kept(at) == [f":material/description: **Mi GDPR.ttl** · "
                                                f"{svc.file_size(len(TTL.encode('utf-8')))}"]
    assert at.button(key="align_next").label == svc.NEXT_LABEL
    assert calls == ["ontology"] and not at.status
    at.button(key="align_next").click().run()
    assert _titles(at) == [READY] and calls == ["ontology"]    # the same result, no new run
    at.button(key="align_result_back").click().run()
    at.button(key="align_back").click().run()                  # the modes
    _pick(at, "ontology+law")
    assert len(_kept(at)) == 1 and drawn[-1] == "Tu ley (.txt o .pdf)"
    uploads["law"] = ("ley.txt", LAW_E2E.encode("utf-8"))
    at.run()
    assert not at.exception and _titles(at) == [READY] and calls == ["ontology", "ontology+law"]
    at.button(key="align_review_open").click().run()
    assert _zips(at)
    at.button(key="align_again").click().run()                 # start over: the files go
    assert _titles(at) == [TITLE] and "align_files" not in at.session_state
    uploads.clear()
    _pick(at, "ontology")
    assert not _kept(at) and not _files(at) and [c.value for c in at.caption] == [
        svc.MODE_WAIT["ontology"]]


def test_change_puts_a_new_uploader(counted):
    page, uploads, calls, drawn = counted
    at = page()
    at.run()
    _pick(at, "ontology")
    uploads["ontology"] = ("Mi GDPR.ttl", TTL.encode("utf-8"))
    at.run()
    at.button(key="align_result_back").click().run()
    round_ = at.session_state["align_round"]
    uploads.clear()
    at.button(key="align_change_ontology").click().run()
    assert not _kept(at) and not _files(at) and drawn[-1].startswith("Tu ontología")
    assert at.session_state["align_round"] == round_ + 1      # a new key (S7-T12)
    assert not [b for b in at.button if b.key == "align_next"]
    uploads["ontology"] = ("Mi GDPR.ttl", TTL.encode("utf-8"))  # the same file again
    at.run()
    assert [b.key for b in at.button if b.key == "align_next"] and calls == ["ontology"]


def test_the_progress_bar_says_each_step(live, monkeypatch):
    page, uploads = live
    view_svc = importlib.import_module("services.align_ontology")

    def run(mode, files, progress, **kw):
        progress(STEP_LABELS["segment"])
        progress(STEP_LABELS["extract"])
        raise WorkspaceError("Algo falló.")
    monkeypatch.setattr(view_svc, "run", run)
    at = page()
    at.run()
    _pick(at, "law")
    uploads["law"] = ("ley.txt", LAW_E2E.encode("utf-8"))
    at.run()
    assert not at.exception and [e.value for e in at.error] == ["Algo falló."]
    bar = at.get("progress")[0]
    steps = steps_for("law")
    assert bar.proto.text == "Buscando con la IA los conceptos de tu ley…"
    assert bar.proto.value == round(100 * steps.index("extract") / len(steps))
    assert [m.value for m in at.status[0].markdown] == [
        "Leyendo tu ley…", "Buscando con la IA los conceptos de tu ley…"]
    at.run()                                                     # the file stays, shown
    assert _kept(at) and [e.value for e in at.error] == ["Algo falló."]
    assert at.button(key="align_change_law") and at.button(key="align_retry")
