"""S6-T05: an uploaded ontology inherits the validated decisions of the base (Sprint 5).

The small base comes from the S5-T06 fixture: Consent aligned (exactMatch Consent, plus a
relatedMatch confirmed from the assistant), Turnover "sin correspondencia" (with the AI
duplicate 'BusinessVolume' / 'volumen de negocio' as alias) and the new concept
ImpactAssessment aligned with DPIA."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.emit import PROFILE_IRI
from src.alignment import inherit as inh
from src.alignment.inherit import (METHOD_SAME_IRI, METHOD_SAME_NAME, METHOD_SIMILAR,
                                   RESULT_FILE, BaseConcept, BaseIndex, inherit,
                                   inherit_for_workspace, load_base, load_result,
                                   render_console, user_concepts)
from src.alignment.sources import SourceConcept
from src.workspace.store import DONE, ERROR, open_ontology
from tests.test_alignment_write import DPV, ONTO, _decide_all

ROOT = Path(__file__).resolve().parents[1]
EX = "http://example.org/gdpr#"


@pytest.fixture
def base_files(tmp_path):
    data, log = _decide_all(tmp_path)
    cand = tmp_path / "output" / "alignment-candidates.json"
    cand.parent.mkdir(parents=True, exist_ok=True)
    cand.write_text(json.dumps(data), encoding="utf-8")
    return cand, log


@pytest.fixture
def base(base_files):
    return load_base(*base_files)


def _user(name, kinds=("class",), iri=None, label=None, definition=None):
    iri = iri or EX + name
    return SourceConcept(key=iri, origin="ontology", name=name, kinds=tuple(kinds),
                         label=label, definition=definition, iri=iri)


def _fake_embed(texts):
    """'consent' texts point one way, everything else elsewhere (deterministic)."""
    out = []
    for i, t in enumerate(texts):
        v = np.zeros(8)
        if "consent" in t.lower():
            v[0] = 1.0
        else:
            v[1 + (sum(map(ord, t)) % 7)] = 1.0
        out.append(v)
    return np.vstack(out)


# ------------------------------------------------------------------ the base

def test_load_base_reads_the_validated_decisions(base):
    by = {c.name: c for c in base.concepts}
    assert set(by) == {"Consent", "Turnover", "ImpactAssessment"} and base.pending == 0
    assert by["Consent"].status == "aligned"
    rels = {(m["dpv_iri"], m["relation"], m["decided_by"]) for m in by["Consent"].mappings}
    assert rels == {(DPV + "Consent", "skos:exactMatch", "person"),
                    (DPV + "ConsentRecord", "skos:relatedMatch", "assistant_confirmed")}
    assert by["Turnover"].status == "no_match" and by["Turnover"].mappings == []
    assert by["Turnover"].aliases == ["BusinessVolume", "volumen de negocio"]
    assert by["ImpactAssessment"].iri == PROFILE_IRI + "#ImpactAssessment"
    assert by["ImpactAssessment"].origin == "ai"


def test_undecided_base_concepts_are_not_offered(tmp_path):
    data, log = _decide_all(tmp_path, leave_pending=True)
    cand = tmp_path / "c.json"
    cand.write_text(json.dumps(data), encoding="utf-8")
    base = load_base(cand, log)
    assert base.pending == 1 and "Turnover" not in {c.name for c in base.concepts}


def test_missing_candidates_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="candidatos de la base"):
        load_base(tmp_path / "no.json", tmp_path / "log.jsonl")


# ------------------------------------------------------------------ the three rules

def test_same_iri_inherits_everything(base):
    report = inherit([_user("Consent", iri=ONTO + "Consent")], base, use_embeddings=False)
    (i,) = report.inherited
    assert i.method == METHOD_SAME_IRI and i.score == 1.0 and i.status == "aligned"
    assert len(i.mappings) == 2 and i.base_key == ONTO + "Consent"


def test_same_name_label_and_alias(base):
    users = [_user("CONSENT"),                                  # other namespace, same name
             _user("C17", label="Impact assessment"),           # by label
             _user("BusinessVolume")]                           # alias of Turnover
    report = inherit(users, base, use_embeddings=False)
    got = {i.name: (i.method, i.base_name, i.status) for i in report.inherited}
    assert got == {"CONSENT": (METHOD_SAME_NAME, "Consent", "aligned"),
                   "C17": (METHOD_SAME_NAME, "ImpactAssessment", "aligned"),
                   "BusinessVolume": (METHOD_SAME_NAME, "Turnover", "no_match")}


def test_kinds_must_be_compatible(base):
    report = inherit([_user("Consent", kinds=("property",)), _user("Ex1", kinds=("individual",))],
                     base, use_embeddings=False)
    assert report.inherited == []
    assert {n["name"]: n["reason"] for n in report.not_inherited} == {"Consent": "kind",
                                                                      "Ex1": "kind"}


def test_similar_needs_embeddings(base):
    user = _user("GivenConsentStatement", definition="Consent given by the data subject")
    with_emb = inherit([user], base, embed_fn=_fake_embed)
    (i,) = with_emb.inherited
    assert i.method == METHOD_SIMILAR and i.base_name == "Consent" and i.score >= 0.75
    without = inherit([user], base, use_embeddings=False)
    assert without.inherited == [] and without.not_inherited[0]["reason"] == "no_match"
    assert 0 < without.not_inherited[0]["best_score"] < 0.95


def test_unrelated_concept_goes_to_proposals(base):
    report = inherit([_user("Pizza")], base, embed_fn=_fake_embed)
    assert report.inherited == [] and report.not_inherited[0]["reason"] == "no_match"


def test_tie_with_different_decisions_is_ambiguous():
    def bc(key, status, maps=()):
        return BaseConcept(key=key, iri=key, origin="ontology", name="Confidentiality",
                           label=None, definition=None, kinds=("class",), status=status,
                           mappings=[{"dpv_iri": d, "relation": "skos:closeMatch"}
                                     for d in maps])
    split = BaseIndex([bc("a#C", "no_match"), bc("b#C", "aligned", [DPV + "Confidentiality"])],
                      "x")
    report = inherit([_user("Confidentiality")], split, use_embeddings=False)
    assert report.not_inherited[0]["reason"] == "ambiguous"
    agree = BaseIndex([bc("a#C", "no_match"), bc("b#C", "no_match")], "x")
    assert inherit([_user("Confidentiality")], agree,
                   use_embeddings=False).inherited[0].method == METHOD_SAME_NAME


def test_summary_counts(base):
    users = [_user("Consent", iri=ONTO + "Consent"), _user("BusinessVolume"), _user("Pizza")]
    s = inherit(users, base, use_embeddings=False).summary()
    assert s["concepts"] == 3 and s["inherited"] == 2 and s["not_inherited"] == 1
    assert (s["inherited_aligned"], s["inherited_no_match"], s["inherited_mappings"]) == (1, 1, 2)
    assert s["by_method"] == {METHOD_SAME_IRI: 1, METHOD_SAME_NAME: 1, METHOD_SIMILAR: 0}


# ------------------------------------------------------------------ the uploaded ontology

TTL = f"""
@prefix ex: <{EX}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
<http://example.org/gdpr> a owl:Ontology .
ex:Consent a owl:Class ; rdfs:label "Consent"@en .
ex:BusinessVolume a owl:Class .
ex:Pizza a owl:Class .
ex:hasPurpose a owl:ObjectProperty .
ex:consent001 a owl:NamedIndividual, ex:Consent .
"""


def test_user_concepts_never_bring_the_lopdp_proposals(tmp_path):
    path = tmp_path / "gdpr.ttl"
    path.write_text(TTL, encoding="utf-8")
    keys = [c.key for c in user_concepts(path)]
    assert sorted(keys) == sorted(EX + n for n in ("Consent", "BusinessVolume", "Pizza",
                                                   "hasPurpose"))
    assert not any(k.startswith("ai:") for k in keys)          # no LOPDP concepts mixed in


def test_workspace_run_saves_and_marks_the_step(tmp_path, base):
    cfg = {"workspaces": {"dir": str(tmp_path / "ws")}}
    ws = open_ontology("gdpr.ttl", TTL.encode("utf-8"), cfg=cfg)
    report = inherit_for_workspace(ws, base=base, use_embeddings=False)
    assert report.summary()["inherited"] == 2
    saved = load_result(ws)
    assert saved["summary"]["inherited"] == 2 and (ws.work_dir / RESULT_FILE).exists()
    step = ws.step("inherit")
    assert step["status"] == DONE and step["inherited"] == 2 and step["not_inherited"] == 2
    assert "2 de 4 conceptos heredados" in step["detail"]


def test_workspace_run_records_errors(tmp_path):
    cfg = {"workspaces": {"dir": str(tmp_path / "ws")}}
    ws = open_ontology("gdpr.ttl", TTL.encode("utf-8"), cfg=cfg)
    broken = {"outputs": {"dir": str(tmp_path / "nada")},
              "review": {"decisions_file": str(tmp_path / "log.jsonl")}}
    with pytest.raises(FileNotFoundError):
        inherit_for_workspace(ws, cfg=broken, use_embeddings=False)
    assert ws.step("inherit")["status"] == ERROR


def test_cli_prints_the_summary(tmp_path, monkeypatch, capsys, base):
    monkeypatch.setattr("src.config.load_config",
                        lambda: {"workspaces": {"dir": str(tmp_path / "ws")}})
    monkeypatch.setattr(inh, "load_base", lambda **kw: base)
    path = tmp_path / "gdpr.ttl"
    path.write_text(TTL, encoding="utf-8")
    assert inh._run([str(path), "--sin-embeddings"]) == 0
    out = capsys.readouterr().out
    assert "Herencia de lo validado (S6-T05)" in out
    assert "Heredan la decisión de la base: 2" in out and "por mismo nombre: 2" in out
    assert "Consent <- Consent (mismo nombre): skos:exactMatch dpv:Consent" in out
    assert inh._run([str(tmp_path / "no.ttl")]) == 1


# ------------------------------------------------------------------ real base (control)

def test_real_sample_inherits_every_ontopriv_decision(tmp_path):
    sample = ROOT / "data" / "input" / "sample.ttl"
    try:
        real = load_base()
    except FileNotFoundError:
        pytest.skip("Falta data/output/alignment-candidates.json (base del Sprint 4-5)")
    if not sample.exists():
        pytest.skip("Falta data/input/sample.ttl")
    report = inherit(user_concepts(sample), real, use_embeddings=False)
    s = report.summary()
    onto = [c for c in real.concepts if c.origin == "ontology"]
    assert s["concepts"] == len(onto) == s["inherited"] == s["by_method"][METHOD_SAME_IRI]
    assert s["inherited_aligned"] == sum(c.status == "aligned" for c in onto)
    assert s["inherited_mappings"] == sum(len(c.mappings) for c in onto)
    assert "Ejemplos:" in render_console(report)