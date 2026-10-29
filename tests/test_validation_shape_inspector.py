from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest
from rdflib import Graph

from src.validation import shape_inspector as si
from src.validation.shape_inspector import (AMBER, BASE, GREEN, KIND_BASE, LOG_FILE, NEW,
                                            ONTOLOGY, RED, REPORT_FILE, Part, inspect,
                                            inspect_base, inspect_workspace, load_result,
                                            status_of, workspace_parts)
from src.workspace.store import load_workspace
from tests.test_alignment_export import _DPV
from tests.test_alignment_propose import targets  # noqa: F401
from tests.test_law_definitions import LawLLM, _align, env  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
DPV = "https://w3id.org/dpv#"
EX = "http://example.org/gdpr#"
PREFIXES = f"""
@prefix ex: <{EX}> .
@prefix dpv: <{DPV}> .
@prefix core: <http://www.semanticweb.org/ontopriv-core#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix skos: <http://www.w3.org/2004/02/skos/core#> .
@prefix al: <http://example.org/gdpr-dpv-alignment#> .
"""
ONTOLOGY_TTL = """
<http://example.org/gdpr> a owl:Ontology ; owl:imports <http://example.org/missing> .
ex:A a owl:Class ; rdfs:label "untagged" ; skos:exactMatch dpv:Consent ;
    ex:note "nota"^^rdf:langString .
ex:B a owl:Class ; rdfs:label "B"@es, "B"@en ; rdfs:comment "B."@es ;
    skos:closeMatch dpv:Inventado .
ex:D skos:exactMatch dpv:Consent .
<http://xmlns.com/foaf/0.1/Person> a owl:Class .
"""
MODULE_TTL = """
<http://example.org/gdpr> rdfs:label "GDPR"@es .
ex:A skos:broadMatch dpv:Consent ; rdfs:label "A"@es, "A"@en .
ex:B rdfs:label " "@es ; skos:exactMatch dpv:Consent .
ex:C skos:exactMatch dpv:Otro .
ex:M a owl:Class ; rdfs:label "M"@es, "M"@en .
[] a owl:Axiom ; owl:annotatedSource ex:A ; owl:annotatedProperty skos:broadMatch ;
   owl:annotatedTarget dpv:Consent ; owl:versionInfo "1" .
"""
CORE_TTL = """
core:X a owl:Class ; core:note "texto"^^rdf:langString .
"""


def _graph(ttl: str) -> Graph:
    return Graph().parse(data=PREFIXES + ttl, format="turtle")


def _terms(*extra: str) -> Graph:
    return _graph("\n".join(f"dpv:{t} rdfs:isDefinedBy dpv: ." for t in ("Consent", *extra)))


def _parts() -> list[Part]:
    return [Part("ontology", "input/gdpr.ttl", ONTOLOGY, _graph(ONTOLOGY_TTL)),
            Part("alignment", "output/gdpr-dpv-alignment.ttl", NEW, _graph(MODULE_TTL)),
            Part("core", "output/ontopriv-core.ttl", BASE, _graph(CORE_TTL))]


def _origins(report: dict) -> Counter:
    return Counter((f["rule"], f["origin"]) for f in report["findings"])


# ------------------------------------------------------------------ where each finding comes from

def test_each_finding_says_where_it_comes_from():
    report = inspect(_parts(), dpv=_terms(), name="gdpr")
    assert _origins(report) == Counter({
        ("ImportShape", ONTOLOGY): 1,                    # the user's ontology imports it
        ("LabelShape", ONTOLOGY): 1,                     # "untagged" is the user's
        ("LabelShape", NEW): 1,                          # " "@es: the system wrote it
        ("MappingConflictShape", NEW): 1,                # exactMatch (user) + broadMatch (system)
        ("DpvTermShape", ONTOLOGY): 1,                   # dpv:Inventado, in the user's file
        ("DpvTermShape", NEW): 1,                        # dpv:Otro, in the module
        ("AlignedConceptShape", NEW): 1,                 # ex:C, aligned but never declared
        ("AlignedConceptShape", ONTOLOGY): 1,            # ex:D, the same in the user's file
        ("MappingAnnotationShape", NEW): 1,              # its annotation says nobody decided it
        ("LangStringShape", ONTOLOGY): 1,                # ex:A
        ("LangStringShape", BASE): 1,                    # in OntoPriv-Core
        ("DefinitionCompletenessShape", ONTOLOGY): 1,    # ex:A
        ("DefinitionCompletenessShape", NEW): 1,         # ex:M, a concept of the module
        ("SpanishLabelShape", BASE): 1, ("EnglishLabelShape", BASE): 1,
        ("DefinitionCompletenessShape", BASE): 1})       # core:X
    s = report["summary"]
    assert s["violations"] == {NEW: 5, ONTOLOGY: 5, BASE: 1}
    assert s["warnings"] == {NEW: 1, ONTOLOGY: 1, BASE: 3}
    assert s["rules"]["LabelShape"] == {"name": "Cada etiqueta es un texto con su idioma",
                                        "severity": "violación", NEW: 1, ONTOLOGY: 1, BASE: 0}
    assert report["status"] == RED
    assert report["completeness"] == {"concepts": 3, "label_es": 3, "label_en": 3,
                                      "definition": 1}      # A, B and M: not D, foaf, core:X
    assert report["triples"] == sum(p["triples"] for p in report["parts"])
    assert report["engine"]["name"] == "pySHACL" and report["shapes"]["version"] == "1"


def test_what_the_profile_restates_of_the_core_is_the_systems():
    profile = _graph('core:Y a owl:Class ; rdfs:label "Y"@es, "Y"@en .')
    core = _graph("core:Y a owl:Class . core:Z a owl:Class .")
    parts = [Part("ontology", "input/ley-profile.ttl", NEW, profile),
             Part("core", "output/ontopriv-core.ttl", BASE, core)]
    report = inspect(parts, dpv=_terms())
    definitions = {f["focus"].rsplit("#", 1)[1]: f["origin"] for f in report["findings"]
                   if f["rule"] == "DefinitionCompletenessShape"}
    assert definitions == {"Y": NEW, "Z": BASE}
    assert report["completeness"] == {"concepts": 1, "label_es": 1, "label_en": 1,
                                      "definition": 0}
    assert si.origin([], parts) == NEW                   # nothing behind it: the system's


def test_status():
    zero = {NEW: 0, ONTOLOGY: 0, BASE: 0}
    assert status_of({"violations": zero, "warnings": zero}) == GREEN
    assert status_of({"violations": zero, "warnings": {**zero, BASE: 2}}) == AMBER
    assert status_of({"violations": {**zero, ONTOLOGY: 1}, "warnings": zero}) == AMBER
    assert status_of({"violations": {**zero, NEW: 1}, "warnings": zero}) == RED


def test_nothing_wrong_is_green():
    ontology = _graph("""
        ex:A a owl:Class ; rdfs:label "A"@es, "A"@en ; rdfs:comment "A."@es ;
            skos:exactMatch dpv:Consent .
        """)
    report = inspect([Part("ontology", "gdpr.ttl", ONTOLOGY, ontology)], dpv=_terms())
    assert report["status"] == GREEN and report["findings"] == []
    assert "Resultado: verde: 0 violaciones nuevas, 0 heredadas, 0 advertencias" in (
        si.render_log(report))


def test_in_the_base_what_ontopriv_had_is_inherited():
    seed = _graph('ex:A a owl:Class . [] rdfs:label "Q_31" .')
    copy = _graph('ex:A a owl:Class ; rdfs:label "A"@es, "A"@en ; rdfs:comment "A."@es .'
                  ' [] rdfs:label "Q_31" . ex:B a owl:Class ; rdfs:label "B" .')
    report = inspect([Part("core", "data/output/ontopriv-core.rdf", NEW, copy)],
                     [Part("ontopriv", "data/input/ontopriv.rdf", ONTOLOGY, seed)],
                     dpv=_terms(), kind=KIND_BASE)
    labels = {f["value"]: f["origin"] for f in report["findings"] if f["rule"] == "LabelShape"}
    assert labels == {'"Q_31"': ONTOLOGY, '"B"': NEW}    # the blank node: same label in OntoPriv
    log = si.render_log(report)
    assert log.startswith("Inspector de forma (SHACL) de la base del sistema")
    assert "    - data/input/ontopriv.rdf (solo para saber qué es heredado)" in log
    assert ("    [rojo] Cada etiqueta es un texto con su idioma: 2 (1 del sistema, 1 de "
            "OntoPriv)") in log


# ------------------------------------------------------------------ the log

def test_the_log_lists_everything_and_the_console_a_few():
    report = inspect(_parts(), dpv=_terms(), name="gdpr")
    log = si.render_log(report)
    assert log.startswith("Inspector de forma (SHACL) del espacio gdpr\n")
    assert "    - input/gdpr.ttl (tu ontología, " in log
    assert "    - output/gdpr-dpv-alignment.ttl (escrito por el sistema, " in log
    assert "    - output/ontopriv-core.ttl (la base del sistema, " in log
    assert "  Resultado: rojo: 5 violaciones nuevas, 6 heredadas, 5 advertencias (" in log
    assert ("  Completitud de 3 conceptos: etiqueta en español 3 (100,0 %), en inglés 3 "
            "(100,0 %), definición 1 (33,3 %)") in log
    assert ("    [rojo] Cada etiqueta es un texto con su idioma: 2 (1 nuevo, 1 de tu "
            "ontología)") in log
    assert ("    [ámbar] Ningún texto rdf:langString sin idioma: 2 (1 de tu ontología, 1 de "
            "la base)") in log
    assert ("    [ámbar] Cada concepto tiene definición: 3 (1 nuevo, 1 de tu ontología, 1 de "
            "la base)") in log
    assert "    [ámbar] Cada importación viene en el paquete: 1 (1 de tu ontología)" in log
    assert "    [ok] Cada definición tomada de la ley dice su artículo: 0" in log
    assert ("         X: Su valor de note está marcado como rdf:langString pero no tiene "
            "idioma. [de la base]") in log
    short = si.render_log(report, examples=1)
    assert "         … y 1 más (en shacl-log.txt)" in short
    assert short.count("La etiqueta") == 1


# ------------------------------------------------------------------ a real upload

def _cfg_with_dpv(env, tmp_path):  # noqa: F811
    path = tmp_path / "dpv-terms.ttl"
    test_dpv = Graph().parse(data=_DPV, format="turtle")
    names = {str(t)[len(DPV):] for t in test_dpv.subjects() if str(t).startswith(DPV)}
    _terms("ConsentRecord", *names).serialize(path, format="turtle")   # + the test base's
    return {**env.cfg, "inputs": {**env.cfg.get("inputs", {}), "dpv": str(path)}}


def test_the_upload_of_ontology_and_law(env, tmp_path):  # noqa: F811
    run = _align(env, LawLLM())
    cfg = _cfg_with_dpv(env, tmp_path)
    report = inspect_workspace(run.ws, cfg)
    assert [p["kind"] for p in report["parts"]] == ["ontology", "labels", "definitions",
                                                    "alignment"]
    assert {p["kind"]: p["origin"] for p in report["parts"]}["ontology"] == ONTOLOGY
    first = report["parts"][0]
    assert first["sha256"] == hashlib.sha256((run.ws.path / first["file"]).read_bytes()).hexdigest()
    assert report["status"] == AMBER
    assert report["summary"]["violations"] == {NEW: 0, ONTOLOGY: 1, BASE: 0}   # "erasure"
    assert report["summary"]["warnings"] == {NEW: 0, ONTOLOGY: 1, BASE: 0}     # Pizza
    assert report["completeness"] == {"concepts": 5, "label_es": 5, "label_en": 5,
                                      "definition": 4}
    assert load_result(run.ws) == json.loads(
        (run.ws.work_dir / REPORT_FILE).read_text(encoding="utf-8"))
    log = (run.ws.work_dir / LOG_FILE).read_text(encoding="utf-8")
    assert "         Pizza: No tiene definición. [de tu ontología]" in log

    module = run.ws.path / run.manifest["files"]["alignment"]       # a defect of the system
    module.write_text(module.read_text(encoding="utf-8") +
                      f"\n<{EX}Pizza> <http://www.w3.org/2004/02/skos/core#exactMatch> "
                      f"<{DPV}Inventada> .\n", encoding="utf-8")
    report = inspect_workspace(run.ws, cfg)
    assert report["status"] == RED
    assert report["summary"]["rules"]["DpvTermShape"][NEW] == 1


def _workspace(tmp_path, *, generated_from=None, files=None) -> Path:
    path = tmp_path / "ws" / "mi-ley-1234"
    (path / "input").mkdir(parents=True)
    (path / "output").mkdir()
    entry = {"original_name": "x.ttl", "stored_as": "x.ttl"}
    if generated_from:
        entry["generated_from"] = generated_from
    (path / "workspace.json").write_text(json.dumps({"id": path.name,
                                                     "inputs": {"ontology": entry}}),
                                         encoding="utf-8")
    (path / "input" / "x.ttl").write_text(PREFIXES + "ex:A a owl:Class .", encoding="utf-8")
    if files is not None:
        for rel in files.values():
            if not (path / rel).exists():
                (path / rel).write_text(PREFIXES + CORE_TTL, encoding="utf-8")
        (path / "output" / "alignment-manifest.json").write_text(
            json.dumps({"files": files}), encoding="utf-8")
    return path


def test_a_law_profile_is_the_systems_and_the_core_the_bases(tmp_path):
    path = _workspace(tmp_path, generated_from="law",
                      files={"ontology": "input/x.ttl", "alignment": "output/x-dpv.ttl",
                             "core": "output/ontopriv-core.ttl"})
    parts = workspace_parts(load_workspace(path.name, root=path.parent))
    assert [(p.kind, p.file, p.origin) for p in parts] == [
        ("ontology", "input/x.ttl", NEW), ("alignment", "output/x-dpv.ttl", NEW),
        ("core", "output/ontopriv-core.ttl", BASE)]


def test_without_the_alignment_there_is_nothing_to_inspect(tmp_path):
    path = _workspace(tmp_path)
    with pytest.raises(ValueError, match="Primero alinea la ontología"):
        workspace_parts(load_workspace(path.name, root=path.parent))


# ------------------------------------------------------------------ the CLI and the real base

def test_the_cli(tmp_path, capsys, monkeypatch):
    path = _workspace(tmp_path, files={"ontology": "input/x.ttl"})
    assert si._run([str(path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Inspector de forma (SHACL) del espacio mi-ley-1234")
    assert "  Resultado: ámbar: 0 violaciones nuevas, 0 heredadas, 3 advertencias" in out
    assert f"  Guardado en: {path / 'work' / REPORT_FILE} y {path / 'work' / LOG_FILE}" in out
    assert (path / "work" / LOG_FILE).exists()
    monkeypatch.chdir(tmp_path)
    assert si._run(["ws/mi-ley-1234"]) == 0                # a folder, relative to here
    capsys.readouterr()
    assert si._run([]) == 1 and si._run(["--base", "x"]) == 1
    assert "Error: indica un espacio de trabajo o --base." in capsys.readouterr().out
    assert si._run(["no-existe"]) == 1
    assert "Error: No existe el espacio de trabajo no-existe." in capsys.readouterr().out
    assert si._run([str(_workspace(tmp_path / "otro"))]) == 1
    assert "Error: Primero alinea la ontología" in capsys.readouterr().out


BASE_FILES = [ROOT / "data" / "output" / name for name in
              ("ontopriv-core.rdf", "profile-ecuador-lopdp.rdf", "ontopriv-dpv-alignment.rdf")]


@pytest.mark.skipif(not all(p.exists() for p in BASE_FILES), reason="Falta la base en data/output")
def test_the_real_base(tmp_path):
    report = inspect_base(out_dir=tmp_path)
    assert report["status"] == AMBER
    assert report["summary"]["violations"] == {NEW: 0, ONTOLOGY: 1, BASE: 0}
    assert report["summary"]["rules"]["LangStringShape"][ONTOLOGY] == 1
    assert report["completeness"] == {"concepts": 552, "label_es": 25, "label_en": 0,
                                      "definition": 26}
    assert [r["file"] for r in report["references"]] == ["data/input/ontopriv.rdf"]
    assert (tmp_path / REPORT_FILE).exists() and (tmp_path / LOG_FILE).exists()