"""S4-T08: the AI proposes the SKOS type of each DPV candidate and justifies it (fake LLM)."""
import json
import os
import re

import pandas as pd
import pytest

from src.alignment.export import COLUMNS, REVIEW_PENDING, CANDIDATES_JSON, CANDIDATES_CSV
from src.alignment.justify import (
    justify_candidates, justify_settings, normalize_relation, relation_counts, render_console,
    clean_justification, JustifySettings, RELATION_UNTYPED,
)

FAST = JustifySettings(batch_size=2, max_calls=20, article_chars=300, max_tokens=1024,
                       pause_seconds=0)
NO_SLEEP = lambda _s: None                                              # noqa: E731

ARTICLES = {
    4: {"number": 4, "title": "Terminos y definiciones", "text": "Titular: persona natural..."},
    17: {"number": 17, "title": "Derecho a la portabilidad", "text": "El titular tiene derecho..."},
}


def _rows(concept_key, name, origin, candidates, needs=True, articles=(), label=None,
          definition=None, kinds=("class",)):
    rows = []
    for rank, (dpv_name, parents) in enumerate(candidates, start=1):
        row = {c: None for c in COLUMNS}
        row.update({
            "concept_key": concept_key, "origin": origin, "concept_name": name,
            "concept_label": label, "concept_kinds": list(kinds), "concept_definition": definition,
            "concept_articles": list(articles), "rank": rank,
            "dpv_iri": f"https://w3id.org/dpv#{dpv_name}", "dpv_name": dpv_name,
            "dpv_label": re.sub(r"(?<=[a-z])(?=[A-Z])", " ", dpv_name),
            "dpv_kind": "class", "dpv_definition": f"Definition of {dpv_name}.",
            "dpv_parents": list(parents), "lexical": 0.5, "semantic": 0.5, "score": 0.5,
            "needs_justification": needs, "review_status": REVIEW_PENDING,
        })
        rows.append(row)
    return rows


def _data():
    rows = []
    rows += _rows("http://ex.org#Data_subject", "Data_subject", "ontology",
                  [("DataSubject", []), ("DataSubjectRight", []), ("Child", ["DataSubject"])])
    rows += _rows("ai:PortabilityRight", "PortabilityRight", "ai",
                  [("DataSubjectRight", []), ("RightNotice", []), ("DigitalLiteracy", [])],
                  articles=[17], label="Derecho a la portabilidad",
                  definition="Derecho del titular a recibir sus datos.")
    rows += _rows("ai:DataSubject", "DataSubject", "ai",                 # possible duplicate
                  [("DataSubject", []), ("Child", []), ("DataController", [])],
                  needs=False, articles=[4], label="titular")
    rows += _rows("http://ex.org#Banking", "Banking", "ontology",
                  [("Sector", []), ("Finance", ["Sector"]), ("Purpose", [])])
    return {"metadata": {"settings": {"top_k": 3}}, "columns": COLUMNS, "rows": rows}


class FakeLLM:
    """Answers every candidate it sees in the prompt; a rule decides the relation."""

    def __init__(self, rule=None, fail=None, skip=None):
        self.contexts = []
        self.rule = rule or (lambda concept, dpv: "exactMatch" if concept.replace("_", "")
                             .lower() == dpv.lower() else "relatedMatch")
        self.fail = fail or []            # list of exceptions to raise on successive calls
        self.skip = skip or set()         # (concept_name, dpv) pairs not answered

    def ask(self, prompt, context="", max_tokens=1024):
        self.contexts.append(context)
        if self.fail:
            exc = self.fail.pop(0)
            if exc is not None:
                raise exc
        lines = []
        for block in context.split("=== ")[1:]:
            cid = block.split(" ===", 1)[0]
            concept = re.search(r"Concepto: (\S+)", block).group(1)
            for dpv in re.findall(r"^- (\S+) \|", block, flags=re.M):
                if (concept, dpv) in self.skip:
                    continue
                lines.append(json.dumps({"concept": cid, "dpv": dpv,
                                         "relation": self.rule(concept, dpv),
                                         "justification": f"{concept} frente a {dpv}."},
                                        ensure_ascii=False))
        return "\n".join(lines)


def _retrieve(query):
    return [{"number": 4, "title": "Terminos y definiciones", "text": f"(recuperado) {query}",
             "source": "lopdp", "distance": 0.1}]


def _run(data, llm, settings=FAST, out_dir=None):
    return justify_candidates(data, llm=llm, article_lookup=ARTICLES.get, retrieve_fn=_retrieve,
                              settings=settings, out_dir=out_dir, sleep_fn=NO_SLEEP)


# --- settings and normalization -------------------------------------------------------------

def test_settings_from_config():
    assert justify_settings({}) == JustifySettings()
    assert JustifySettings().max_calls == 60 and JustifySettings().pause_seconds == 5.0
    s = justify_settings({"alignment": {"justify_batch_size": 5, "justify_max_calls": 3,
                                        "justify_pause_seconds": 2}})
    assert (s.batch_size, s.max_calls, s.pause_seconds) == (5, 3, 2.0)
    with pytest.raises(ValueError):
        justify_settings({"alignment": {"justify_batch_size": 0}})
    with pytest.raises(ValueError):
        justify_settings({"alignment": {"justify_pause_seconds": -1}})


def test_pause_between_calls_not_before_the_first():
    waits = []
    settings = JustifySettings(batch_size=1, max_calls=20, article_chars=300, max_tokens=1024,
                               pause_seconds=5)
    report = justify_candidates(_data(), llm=FakeLLM(), article_lookup=ARTICLES.get,
                                retrieve_fn=_retrieve, settings=settings, sleep_fn=waits.append)
    assert report.calls == 3 and waits == [5, 5]


def test_normalize_relation():
    assert normalize_relation("skos:ExactMatch") == "skos:exactMatch"
    assert normalize_relation("broadMatch") == "skos:broadMatch"
    assert normalize_relation("NONE") == "none"
    assert normalize_relation("sin_correspondencia") == "none"
    assert normalize_relation("equivalentClass") == RELATION_UNTYPED
    assert normalize_relation(None) == RELATION_UNTYPED


def test_clean_justification_drops_batch_ids():
    assert clean_justification("Las medidas de seguridad (C4) son el medio.") == \
        "Las medidas de seguridad son el medio."
    assert clean_justification("( C12 ) Igual.") == "Igual."
    assert clean_justification("Articulo 4 (Terminos)") == "Articulo 4 (Terminos)"
    assert clean_justification(None) is None and clean_justification("(C1)") is None


def test_ids_are_cleaned_in_new_answers_and_old_rows():
    data = _data()
    old = data["rows"][0]
    old.update(proposed_relation="skos:exactMatch", justification="Coinciden (C1) en todo.")
    _run(data, FakeLLM(rule=lambda c, d: "relatedMatch"))
    assert old["justification"] == "Coinciden en todo."


# --- prompt ---------------------------------------------------------------------------------

def test_prompt_has_definitions_parents_and_evidence():
    llm = FakeLLM()
    _run(_data(), llm)
    first = llm.contexts[0]
    assert "=== C1 ===" in first and "=== C2 ===" in first
    assert "Concepto: Data_subject" in first
    assert "(recuperado) Data subject" in first                  # OntoPriv -> RAG evidence
    assert "Articulo 17 de la ley (Derecho a la portabilidad)" in first   # AI -> own article
    assert "padres en el DPV: DataSubject" in first               # parents travel to the AI
    assert "Definition of DataSubjectRight." in first


def test_duplicates_are_never_sent():
    llm = FakeLLM()
    data = _data()
    _run(data, llm)
    assert all("Concepto: DataSubject " not in c for c in llm.contexts)
    dup_rows = [r for r in data["rows"] if r["concept_key"] == "ai:DataSubject"]
    assert all(r["proposed_relation"] is None for r in dup_rows)


# --- filling rows ---------------------------------------------------------------------------

def test_rows_are_filled_with_type_justification_and_evidence():
    data = _data()
    report = _run(data, FakeLLM())
    assert report.calls == 2 and report.concepts_sent == 3
    assert report.rows_filled == 9 and report.remaining_concepts == 0
    ds = next(r for r in data["rows"] if r["concept_name"] == "Data_subject" and r["rank"] == 1)
    assert ds["proposed_relation"] == "skos:exactMatch"
    assert ds["justification"] == "Data_subject frente a DataSubject."
    assert ds["evidence_article"] == 4
    port = next(r for r in data["rows"] if r["concept_name"] == "PortabilityRight")
    assert port["evidence_article"] == 17
    assert {r["review_status"] for r in data["rows"]} == {REVIEW_PENDING}   # nothing approved


def test_invalid_relation_becomes_untyped_without_crashing():
    data = _data()
    report = _run(data, FakeLLM(rule=lambda c, d: "owl:equivalentClass"))
    assert report.rows_untyped == 9
    assert all(r["proposed_relation"] == RELATION_UNTYPED
               for r in data["rows"] if r["needs_justification"])
    assert relation_counts(data["rows"])["sin tipo valido"] == 9


def test_unanswered_rows_stay_pending_and_resume_only_sends_them():
    data = _data()
    first = _run(data, FakeLLM(skip={("Banking", "Purpose")}))
    assert first.rows_unanswered == 1 and first.remaining_concepts == 1
    llm = FakeLLM()
    second = _run(data, llm)
    assert second.calls == 1 and second.rows_filled == 1 and second.remaining_concepts == 0
    assert len(llm.contexts) == 1 and "Concepto: Banking" in llm.contexts[0]
    assert "Concepto: Data_subject" not in llm.contexts[0]


# --- quota, limits and errors ---------------------------------------------------------------

def test_call_limit_stops_and_next_run_continues():
    data = _data()
    one = JustifySettings(batch_size=1, max_calls=2, article_chars=300, max_tokens=1024,
                          pause_seconds=0)
    r1 = _run(data, FakeLLM(), settings=one)
    assert r1.calls == 2 and r1.call_limit_reached and r1.remaining_concepts == 1
    r2 = _run(data, FakeLLM(), settings=one)
    assert r2.calls == 1 and r2.remaining_concepts == 0


def test_quota_waits_once_then_continues():
    data = _data()
    llm = FakeLLM(fail=[RuntimeError("Error code: 429 RESOURCE_EXHAUSTED")])
    report = _run(data, llm)
    assert not report.quota_exhausted and report.remaining_concepts == 0


def test_persistent_quota_stops_cleanly_and_keeps_work(tmp_path):
    data = _data()
    quota = RuntimeError("429 quota exceeded")
    llm = FakeLLM(fail=[None, quota, quota])            # 1st batch ok, 2nd fails twice
    report = _run(data, llm, out_dir=tmp_path)
    assert report.quota_exhausted and report.rows_filled == 6 and report.remaining_concepts == 1
    saved = json.loads((tmp_path / CANDIDATES_JSON).read_text(encoding="utf-8"))
    assert sum(1 for r in saved["rows"] if r["proposed_relation"]) == 6


def test_server_overload_503_waits_once_then_continues():
    data = _data()
    waits = []
    busy = RuntimeError("Error code: 503 - This model is currently experiencing high demand")
    report = justify_candidates(data, llm=FakeLLM(fail=[busy]), article_lookup=ARTICLES.get,
                                retrieve_fn=_retrieve, settings=FAST, sleep_fn=waits.append)
    assert waits == [20] and not report.errors and report.remaining_concepts == 0


def test_persistent_503_skips_only_that_batch():
    data = _data()
    busy = RuntimeError("Error code: 503 - UNAVAILABLE")
    report = _run(data, FakeLLM(fail=[busy, busy]))
    assert len(report.errors) == 1 and not report.quota_exhausted
    assert report.rows_filled == 3 and report.remaining_concepts == 2   # retried next run


def test_other_errors_skip_the_batch_and_continue():
    data = _data()
    report = _run(data, FakeLLM(fail=[ValueError("respuesta vacia")]))
    assert len(report.errors) == 1 and not report.quota_exhausted
    assert report.rows_filled == 3 and report.remaining_concepts == 2


# --- files ----------------------------------------------------------------------------------

def test_files_are_rewritten_with_the_new_columns(tmp_path):
    data = _data()
    report = _run(data, FakeLLM(), out_dir=tmp_path)
    frame = pd.read_csv(tmp_path / CANDIDATES_CSV, encoding="utf-8-sig")
    assert list(frame.columns) == COLUMNS
    assert frame["proposed_relation"].notna().sum() == 9
    assert set(frame.loc[frame["concept_name"] == "Banking", "evidence_article"]) == {4}
    saved = json.loads((tmp_path / CANDIDATES_JSON).read_text(encoding="utf-8"))
    assert saved["metadata"]["justification"]["remaining_concepts"] == 0
    text = render_console(report, data["rows"])
    assert "equivalente: 1" in text and "Conceptos pendientes para la proxima corrida: 0" in text


def test_old_file_without_evidence_column_still_works():
    data = _data()
    for r in data["rows"]:
        r.pop("evidence_article")
    _run(data, FakeLLM())
    assert all("evidence_article" in r for r in data["rows"])


@pytest.mark.slow
def test_real_llm_one_concept():
    """One real LLM call (spends quota): the answer parses into a valid relation."""
    if not os.environ.get("LLM_API_KEY"):
        from src import config  # noqa: F401  (loads .env)
    if not os.environ.get("LLM_API_KEY"):
        pytest.skip("Falta LLM_API_KEY en .env")
    data = {"metadata": {}, "columns": COLUMNS,
            "rows": _rows("ai:DataController", "DataController", "ai",
                          [("DataController", ["Organisation"]), ("JointDataControllers", []),
                           ("DataSubject", [])],
                          label="responsable del tratamiento",
                          definition="Persona natural o juridica que decide sobre la finalidad "
                                     "y el tratamiento de datos personales.")}
    report = justify_candidates(data, article_lookup=lambda n: None, retrieve_fn=None,
                                settings=JustifySettings(batch_size=1, max_calls=1,
                                                         pause_seconds=0))
    assert report.calls == 1 and not report.errors
    first = data["rows"][0]
    assert first["proposed_relation"] in ("skos:exactMatch", "skos:closeMatch")
    assert first["justification"]