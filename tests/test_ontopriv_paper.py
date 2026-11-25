from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdflib import Graph

from src.alignment import ontopriv_paper as op
from src.publish.base_doc import MODULES

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "output"
VOCAB = ROOT / "vocab" / "dpv.ttl"
needs_base = pytest.mark.skipif(
    not VOCAB.exists() or not all((OUTPUT / f"{m}.ttl").exists() for m in MODULES),
    reason="Falta la base en data/output o el DPV en vocab")
PREFIXES = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix dpv: <https://w3id.org/dpv#> .
@prefix lopdp: <http://www.semanticweb.org/ley-organica-proteccion-datos-personales#> .
@prefix ec: <http://www.semanticweb.org/profiles/ecuador-lopdp#> .
@prefix other: <http://example.org/other#> .
"""
BASE = PREFIXES + """
lopdp:Personal_data a owl:Class ; skos:exactMatch dpv:PersonalData .
lopdp:Revocation_of_consent a owl:Class ; skos:closeMatch dpv:WithdrawConsent .
lopdp:Security a owl:Class ; skos:relatedMatch dpv:EnforceSecurity .
lopdp:Purpose a owl:Class .
lopdp:Anonymization a owl:Class ; skos:exactMatch dpv:Anonymisation .
ec:LegalBasis a owl:Class ; skos:broadMatch dpv:LegalBasis .
lopdp:Sensitive_data a owl:Class ; skos:closeMatch dpv:SensitivePersonalData .
other:Risk_analysis a owl:Class ; skos:exactMatch dpv:Risk .
lopdp:hasDuration a owl:ObjectProperty ; skos:exactMatch dpv:Duration .
lopdp:Retention a owl:Class ; skos:broadMatch dpv:Duration .
"""
VOCABULARY = PREFIXES + "".join(
    f"dpv:{t} a skos:Concept .\n" for t in (
        "PersonalData", "SensitivePersonalData", "WithdrawConsent", "EnforceSecurity",
        "DataSecurityManagement", "Purpose", "Anonymisation", "LegalBasis", "Risk", "Duration"))


def _row(n, ontopriv, dpv, axiom) -> op.PaperRow:
    subject, symbol, target = op._AXIOM_RE.fullmatch(axiom).groups()
    return op.PaperRow(n, ontopriv, dpv, axiom, subject, symbol, target)


ROWS = [_row(1, "Personal data", "PersonalData", "Personal data ≡ dpv:PersonalData"),
        _row(2, "Revocation of consent", "WithdrawConsent",
             "Revocation of consent ⊑ dpv:WithdrawConsent"),
        _row(3, "Security", "DataSecurityManagement", "Security ⊑ dpv:DataSecurityManagement"),
        _row(4, "Purpose", "Purpose", "Purpose ≡ dpv:Purpose"),
        _row(5, "Anonymization", "Anonymization", "Anonymization ≡ dpv:Anonymization"),
        _row(6, "Legal basis", "LegalBasis", "Legal basis ⊑ dpv:LegalBasis"),
        _row(7, "Sensitive data", "SensitivePersonalData", "Sensitive data ⊑ dpv:PersonalData"),
        _row(8, "Risk analysis", "Risk", "Risk analysis ⊑ dpv:Risk"),
        _row(9, "Duration", "Duration", "NotificationTime ⊑ dpv:Duration"),
        _row(10, "Retention", "Duration", "StorageTime ⊑ dpv:Duration")]


@pytest.fixture
def found() -> dict:
    rows = op.compare(ROWS, Graph().parse(data=BASE, format="turtle"),
                      Graph().parse(data=VOCABULARY, format="turtle"))
    return {r["n"]: r for r in rows}


def test_each_result_of_a_row(found):
    assert [found[n]["result"] for n in range(1, 11)] == [
        op.SAME,                 # ≡ and exactMatch to the same term
        op.OTHER_RELATION,       # ⊑ is broadMatch, the base says closeMatch
        op.OTHER_TERM,
        op.NO_MAPPING,
        op.SAME,                 # Anonymization is written Anonymisation in the DPV 2.3
        op.SAME,                 # "Legal basis" is ec:LegalBasis; ⊑ and broadMatch
        op.OTHER_RELATION,       # the term of the column, not the parent of the axiom
        op.NEW_CONCEPT,          # only in a namespace that is not the base's
        op.NEW_CONCEPT,          # a property of the base, not a class
        op.SAME]                 # the concept of the first column, not the one of the axiom


def test_what_the_dpv_23_has(found):
    assert found[5]["dpv23"] == [{"term": "Anonymization", "exists": False,
                                  "spelling": "Anonymisation"}]
    assert found[7]["dpv23"] == [
        {"term": "SensitivePersonalData", "exists": True, "spelling": None},
        {"term": "PersonalData", "exists": True, "spelling": None}]
    assert op.in_dpv("BiometricData", {"Biometrics", "Data"}) is None
    assert op.in_dpv("Organization", {"Organisation", "OrganiSation"}) is None   # not one


def test_a_row_whose_axiom_is_of_another_concept(found):
    assert [n for n, r in found.items() if not r["coherent"]] == [9, 10]
    assert op._paper_text(found[9]) == ("NotificationTime ⊑ dpv:Duration "
                                        "(la fila es de «Duration»)")
    assert op._paper_text(found[7]) == ("Sensitive data ⊑ dpv:PersonalData "
                                        "(término: dpv:SensitivePersonalData)")


def test_the_summary(found):
    s = op.summarize(list(found.values()))
    assert s["in_base"] == 8 and s["same_term"] == 6 and s["missing_in_dpv23"] == [5]
    assert s["by_result"] == {op.SAME: 4, op.OTHER_RELATION: 2, op.OTHER_TERM: 1,
                              op.NO_MAPPING: 1, op.NEW_CONCEPT: 2}


def test_an_axiom_that_is_not_understood(tmp_path):
    path = tmp_path / "tabla.yaml"
    path.write_text('source: {}\nrows:\n  - {ontopriv: A, dpv: "dpv:B", axiom: "A = dpv:B"}\n',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="fila 1: no se entiende el axioma «A = dpv:B»"):
        op.load_reference(path)


# --------------------------------------------------------------------------- the real base

def test_the_reference_is_table_10_of_the_article():
    source, rows = op.load_reference(ROOT / op.REFERENCE)
    assert len(rows) == 14 and source["dpv_version"] == "2.0"
    assert [r.symbol for r in rows].count("≡") == 7
    assert (rows[11].ontopriv, rows[11].subject, rows[11].target) == (
        "Duration", "NotificationTime", "Duration")


@needs_base
def test_the_base_against_table_10(tmp_path, capsys):
    assert op._run(["--salida", str(tmp_path)]) == 0
    data = json.loads((tmp_path / op.RESULT_JSON).read_text(encoding="utf-8"))
    assert [r["result"] for r in data["rows"]] == [
        op.SAME, op.OTHER_RELATION, op.SAME, op.SAME, op.OTHER_RELATION, op.OTHER_TERM,
        op.OTHER_TERM, op.OTHER_TERM, op.NEW_CONCEPT, op.OTHER_RELATION, op.NEW_CONCEPT,
        op.NEW_CONCEPT, op.SAME, op.NEW_CONCEPT]
    assert data["summary"]["missing_in_dpv23"] == [3, 6, 13]
    assert data["summary"]["not_coherent"] == [12]
    table = (tmp_path / op.RESULT_MD).read_text(encoding="utf-8")
    assert table.count("\n| ") == 15                       # the header and the 14 rows
    assert "| 13 | Anonymization ≡ dpv:Anonymization | exactMatch dpv:Anonymisation |" in table
    out = capsys.readouterr().out
    assert "De esas 10: coinciden 4, mismo término con otra relación 3, otro término 3" in out
    assert "≡" not in out and "⊑" not in out


@needs_base
def test_the_concepts_the_base_lacks_are_not_in_ontopriv_either():
    ontopriv = Graph().parse(ROOT / "data" / "input" / "ontopriv.rdf")
    assert op.concepts(ontopriv, "Revocation of consent")              # it does look there
    for name in ("Legal basis", "Risk analysis", "NotificationTime", "Duration",
                 "Automated decision"):
        assert op.concepts(ontopriv, name) == [], name
