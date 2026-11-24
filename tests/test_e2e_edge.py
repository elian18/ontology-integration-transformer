from __future__ import annotations

from pathlib import Path

import pytest
from pypdf import PdfReader
from rdflib import Graph
from rdflib.namespace import OWL, RDF

from app.services import align_ontology as svc
from src.e2e import edge
from src.e2e import runner as rn

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def material(tmp_path_factory):
    folder = tmp_path_factory.mktemp("material")
    edge.write_material(folder, max_mb=1)
    return folder


def test_the_material_is_what_each_case_needs(material):
    assert sorted(p.name for p in material.iterdir()) == [
        "danada.ttl", "escaneada.pdf", "grande.ttl", "mi-ontologia.docx",
        "mini-privacidad.ttl", "vacia.ttl"]
    assert (material / "vacia.ttl").stat().st_size == 0
    assert 1024 * 1024 < (material / "grande.ttl").stat().st_size < 1024 * 1024 + 200
    mini = Graph().parse(material / "mini-privacidad.ttl", format="turtle")
    assert len(set(mini.subjects(RDF.type, OWL.Class))) == len(edge.CLASSES) == 6
    assert len(Graph().parse(material / "grande.ttl", format="turtle")) == len(mini)
    with pytest.raises(Exception):
        Graph().parse(material / "danada.ttl", format="turtle")
    scanned = PdfReader(material / "escaneada.pdf")
    assert len(scanned.pages) == 1 and not scanned.pages[0].extract_text().strip()


def test_the_command_makes_the_material(tmp_path, capsys):
    assert edge._run(["--carpeta", str(tmp_path)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "Archivos de los casos límite (S11-T10)"
    assert "  grande.ttl: 25.00 MB" in printed or "  grande.ttl: 25.01 MB" in printed


@pytest.mark.parametrize("name, message", [
    ("vacia.ttl", "El archivo vacia.ttl está vacío."),
    ("grande.ttl", "El archivo grande.ttl pesa 1.1 MB; el máximo es 0.5 MB."),
    ("mi-ontologia.docx", "Formato de ontología no soportado: '.docx'. Sube la ontología en "
                          "RDF/XML (.rdf, .owl), Turtle (.ttl) o JSON-LD (.jsonld)."),
    ("danada.ttl", "No se pudo leer la ontología danada.ttl: ")])
def test_what_cannot_be_read_is_said_before_any_work(material, name, message, tmp_path):
    cfg = {"workspaces": {"dir": str(tmp_path / "ws"), "max_upload_mb": 0.5}}
    r = rn.run_case(rn.Case("x", svc.MODE_ONTOLOGY, {"ontology": material / name}), cfg=cfg)
    assert r["result"] == "error" and r["error"].startswith(message) and r["ai_calls"] == 0
    assert not list((tmp_path / "ws").glob("*/workspace.json"))            # nothing was made


def test_a_scanned_law_says_it_has_no_text(material, tmp_path):
    cfg = {"workspaces": {"dir": str(tmp_path / "ws")}}
    r = rn.run_case(rn.Case("x", svc.MODE_LAW, {"law": material / "escaneada.pdf"}), cfg=cfg)
    assert r["error"].startswith("escaneada.pdf no tiene texto que se pueda leer: parece una "
                                 "imagen escaneada.")


def test_without_key_or_quota_the_client_fails_as_it_really_would():
    from src.ai import llm_client
    with rn.under(rn.NO_KEY):
        with pytest.raises(KeyError, match="LLM_API_KEY"):
            llm_client.LLMClient(client=object())
    with rn.under(rn.NO_QUOTA):
        with pytest.raises(RuntimeError, match="429 - RESOURCE_EXHAUSTED"):
            llm_client.LLMClient(client=object(), model="m").ask("hola")
        assert llm_client._is_quota_or_server_error(rn.QUOTA_ERROR)   # seen as a quota error
    assert llm_client.LLMClient(client=object(), model="m").model == "m"    # back as it was


def test_each_condition_gets_its_own_settings(tmp_path):
    cfg = {"workspaces": {"dir": str(tmp_path / "ws")}, "validation": {"engine": "auto"}}
    (tmp_path / "ws" / rn.ISOLATED / "x" / "viejo").mkdir(parents=True)
    alone = rn.case_cfg(rn.Case("x", "ontology", isolated=True), cfg)
    assert Path(alone["workspaces"]["dir"]) == tmp_path / "ws" / rn.ISOLATED / "x"
    assert not (tmp_path / "ws" / rn.ISOLATED / "x").exists()          # empty, every time
    java = rn.case_cfg(rn.Case("y", "ontology", condition=rn.NO_JAVA), cfg)
    assert java["validation"] == {"engine": "auto", "java": "java-no-instalado"}
    assert cfg["validation"] == {"engine": "auto"}                      # not changed
    assert rn.case_cfg(rn.Case("z", "ontology"), cfg) == cfg


def test_an_unknown_condition_is_refused(tmp_path):
    path = tmp_path / "casos.yaml"
    path.write_text("cases:\n  - {id: x, mode: ontology, condition: sin-luz}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="caso x: condición desconocida sin-luz"):
        rn.load_cases(path, tmp_path)


def test_the_conditions_show_in_the_summary_and_on_screen():
    base = {"case": "sin-java", "mode": "ontology", "control": False, "result": "ok",
            "condition": rn.NO_JAVA, "concepts": 6, "base": 2, "rule": 1, "suggested": 0,
            "no_match": 3, "no_proposal": 0, "ai_calls": 4, "articles": None, "seconds": 9.0,
            "validation": {"status": "verde"}, "engines": {"logic": "owlrl 7.6.2"},
            "without_ai": [], "download": {"same": True, "triples": {"turtle": 40}}}
    assert rn.row(base)[2] == "ok (sin-java)"
    assert rn.render(base).endswith("; 4 llamadas a la IA; 9.0 s; lógica con owlrl 7.6.2")
    quota = {**base, "case": "cuota", "condition": rn.NO_QUOTA,
             "without_ai": ["ai_audit", "decider", "labels"]}
    assert rn.render(quota).endswith("; sin IA en: ai_audit, decider, labels")
    busy = {**base, "condition": "", "ai_errors": {"decider": "Error code: 503 - mucha demanda",
                                                   "labels": "Error code: 503 - mucha demanda"}}
    assert rn.render(busy).endswith("; la IA falló en decider, labels (Error code: 503 - mucha "
                                    "demanda)")
    assert rn.row(busy)[10] == "4 (errores en decider, labels)"
    failed = {**base, "result": "error", "error": "Falta la clave.", "condition": rn.NO_KEY}
    assert rn.row(failed)[2] == "error (sin-clave): Falta la clave."


def test_the_edge_cases_are_in_the_cases_of_the_sprint():
    cases = {c.id: c for c in rn.load_cases(ROOT / "data/reference/e2e-s11.yaml", ROOT)}
    assert {"vacia", "danada", "escaneada", "grande", "extension", "dos-veces-1",
            "dos-veces-2", "sin-clave", "sin-clave-ley", "sin-java", "cuota-agotada"} <= set(cases)
    assert [c.condition for c in cases.values() if c.condition] == [
        rn.NO_KEY, rn.NO_KEY, rn.NO_JAVA, rn.NO_QUOTA]
    assert all(c.isolated for c in cases.values() if c.condition)
    assert cases["dos-veces-1"].fresh and not cases["dos-veces-2"].fresh
    assert cases["dos-veces-1"].files == cases["dos-veces-2"].files


def test_the_errors_of_the_ai_are_read_from_each_step():
    manifest = {"decider": {"ai": {"available": True, "errors": ["Error code: 503 - busy"]}},
                "labels": {"ai": {"available": True, "errors": []}},
                "ai_audit": {"ai": {"available": False, "errors": []}}, "summary": {}}
    assert rn._ai_blocks(manifest) == {k: v["ai"] for k, v in manifest.items() if k != "summary"}
