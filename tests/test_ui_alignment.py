"""Tests for the alignment UI service (Sprint 4, S4-T10)."""
import json

from app.services import alignment

ROWS = [
    {"concept_key": "o:A", "origin": "ontology", "concept_name": "Data_subject",
     "concept_label": None, "rank": 1, "dpv_name": "DataSubject", "dpv_label": "Data Subject",
     "score": 0.91, "lexical": 1.0, "semantic": 0.85, "proposed_relation": "skos:exactMatch",
     "justification": "Mismo rol.", "evidence_article": 4, "duplicate_status": None},
    {"concept_key": "o:A", "origin": "ontology", "concept_name": "Data_subject",
     "concept_label": None, "rank": 2, "dpv_name": "Child", "dpv_label": "Child",
     "score": 0.40, "lexical": 0.2, "semantic": 0.5, "proposed_relation": "skos:narrowMatch",
     "justification": "Un nino es un titular.", "evidence_article": 4, "duplicate_status": None},
    {"concept_key": "ai:P", "origin": "ai", "concept_name": "PortabilityRight",
     "concept_label": "Derecho a la portabilidad", "rank": 1, "dpv_name": "DataSubjectRight",
     "dpv_label": "Data Subject Right", "score": 0.56, "lexical": 0.46, "semantic": 0.63,
     "proposed_relation": "skos:broadMatch", "justification": "Es un derecho del titular.",
     "evidence_article": 17, "duplicate_status": "new"},
    {"concept_key": "ai:D", "origin": "ai", "concept_name": "DataController",
     "concept_label": "responsable", "rank": 1, "dpv_name": "DataController",
     "dpv_label": "Data Controller", "score": 0.93, "lexical": 1.0, "semantic": 0.88,
     "proposed_relation": None, "justification": None, "evidence_article": None,
     "duplicate_status": "possible_duplicate"},
]


def _write(tmp_path, payload, name="c.json"):
    f = tmp_path / name
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return f


def test_candidates_reads_the_file(tmp_path):
    f = _write(tmp_path, {"metadata": {"settings": {"top_k": 3}}, "counts": {"rows": 4},
                          "rows": ROWS})
    data = alignment.candidates(path=f)
    assert data["counts"]["rows"] == 4 and len(data["rows"]) == 4
    assert data["relations"] == sorted(["equivalente", "el DPV es mas especifico",
                                        "el DPV es mas general", "sin tipo aun"])


def test_missing_or_broken_files_return_none(tmp_path):
    assert alignment.candidates(path=tmp_path / "no.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{no es json", encoding="utf-8")
    assert alignment.candidates(path=bad) is None
    assert alignment.evaluation(path=tmp_path / "no.json") is None


def test_relation_labels():
    assert alignment.relation_label("skos:broadMatch") == "el DPV es mas general"
    assert alignment.relation_label("none") == "sin correspondencia"
    assert alignment.relation_label("untyped") == "sin tipo valido"
    assert alignment.relation_label(None) == "sin tipo aun"


def test_filters():
    assert len(alignment.filter_rows(ROWS, origin="ai")) == 2
    assert [r["concept_name"] for r in alignment.filter_rows(ROWS, duplicate="new")] == \
        ["PortabilityRight"]
    assert len(alignment.filter_rows(ROWS, best_only=True)) == 3
    assert len(alignment.filter_rows(ROWS, relations=["equivalente"])) == 1
    assert [r["dpv_name"] for r in alignment.filter_rows(ROWS, query="portabilidad")] == \
        ["DataSubjectRight"]                                  # searches the Spanish label too
    assert len(alignment.filter_rows(ROWS, query="data subject")) == 2   # DPV label


def test_table_rows_follow_the_scrumban_columns():
    table = alignment.table_rows(ROWS[:1] + ROWS[2:3])
    assert list(table[0])[:4] == ["Concepto", "Candidato DPV", "Similitud", "Tipo propuesto"]
    assert table[0]["Tipo propuesto"] == "equivalente" and table[0]["Origen"] == "OntoPriv"
    assert table[1]["Marca (IA)"] == "nuevo" and table[1]["Artículo"] == 17


def test_evaluation_reads_the_metrics(tmp_path):
    f = _write(tmp_path, {"reference": "ref.csv", "metrics": {"hit_at_1_rate": 0.474, "k": 3}},
               name="e.json")
    assert alignment.evaluation(path=f)["metrics"]["hit_at_1_rate"] == 0.474