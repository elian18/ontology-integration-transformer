from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from rdflib import Graph

from src.validation import logic_inspector as li
from src.validation.logic_inspector import (JENA, LOG_FILE, OWLRL, REPORT_FILE, Settings,
                                            choose_engine, inspect, inspect_base, java_version,
                                            kind_of, settings)
from src.validation.shape_inspector import AMBER, BASE, GREEN, NEW, ONTOLOGY, RED, Part
from tests.test_validation_shape_inspector import _workspace

ROOT = Path(__file__).resolve().parents[1]
EX = "http://example.org/gdpr#"
PREFIXES = f"""
@prefix ex: <{EX}> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
"""
ONTOLOGY_TTL = """
<http://example.org/gdpr> a owl:Ontology .
ex:A a owl:Class . ex:B a owl:Class ; owl:disjointWith ex:A .
ex:x a ex:A .
ex:y a ex:A , ex:B .
ex:p a owl:DatatypeProperty . ex:q a owl:ObjectProperty .
ex:z ex:p ex:w ; ex:q "texto" .
"""
MODULE_TTL = """
ex:x a ex:B .
ex:A rdfs:label "A"@es .
"""


def _java() -> str | None:
    if not shutil.which("java"):
        return None
    done = subprocess.run(["java", "-version"], capture_output=True, text=True)
    return java_version(done.stderr + done.stdout)


JAVA = _java()
HAS_JENA = (JAVA is not None and int(JAVA.split(".")[0]) >= li.MIN_JAVA
            and li.jena_version(li._home(Settings())) is not None)
needs_jena = pytest.mark.skipif(not HAS_JENA, reason="Falta Java 21+ o Apache Jena en tools/")


def _parts(tmp_path, ontology=ONTOLOGY_TTL, module=MODULE_TTL) -> list[Part]:
    parts = []
    for kind, ttl, origin in (("ontology", ontology, ONTOLOGY), ("alignment", module, NEW)):
        path = tmp_path / f"{kind}.ttl"
        path.write_text(PREFIXES + ttl, encoding="utf-8")
        parts.append(Part(kind, path.name, origin, Graph().parse(path), "", path))
    return parts


def _by(report: dict) -> dict:
    found = {}
    for f in report["findings"]:
        found.setdefault((f["kind"], f["origin"], f["error"]), []).append(f)
    return found


# ------------------------------------------------------------------ choosing the engine

def test_java_version():
    assert java_version('java version "24.0.2" 2025-07-15') == "24.0.2"
    assert java_version('openjdk version "21.0.12.1" 2026-08-18') == "21.0.12.1"
    assert java_version("no es java") is None


def test_choose_engine(tmp_path):
    home = tmp_path / "jena" / "lib"
    home.mkdir(parents=True)
    (home / "jena-core-6.2.0.jar").write_bytes(b"")
    s = Settings(jena_home=str(tmp_path / "jena"))
    assert choose_engine(s, "24.0.2") == {"name": JENA, "version": "6.2.0", "java": "24.0.2",
                                          "reasoner": "owl-micro", "home": tmp_path / "jena"}
    assert choose_engine(s, "17.0.2") == {
        "name": OWLRL, "reason": "Java 17.0.2: Jena 6 necesita Java 21 o superior"}
    assert choose_engine(Settings(jena_home=str(tmp_path / "nada")), "24.0.2") == {
        "name": OWLRL, "reason": f"falta Apache Jena en {tmp_path / 'nada'} (ver README)"}
    assert choose_engine(Settings(java="java-que-no-existe"))["reason"] == (
        "no se encontró Java (java-que-no-existe)")
    assert choose_engine(Settings(engine=OWLRL), "24.0.2")["name"] == OWLRL
    with pytest.raises(RuntimeError, match="No se puede usar Jena: Java 17"):
        choose_engine(Settings(engine=JENA, jena_home=str(tmp_path / "jena")), "17")


def test_settings():
    assert settings({"validation": {"reasoner": "rdfs"}}).reasoner == "rdfs"
    assert settings({}) == Settings()
    with pytest.raises(ValueError, match="bloque validation"):
        settings({"validation": {"engine": "hermit"}})


def test_kinds_of_each_engine():
    def k(description, error=True, kind="x"):
        return kind_of({"error": error, "type": kind, "description": description})
    assert k('"Individual a member of disjoint classes"\nCulprit = <a>') == "disjoint"
    assert k("Disjoint classes A and B have a common individual x") == "disjoint"
    assert k('"Object value for datatype property (prop, value)"') == "object_value"
    assert k("Property p has a typed range but was given a non literal value x",
             kind="dtRange") == "object_value"
    assert k('"Literal value for object property (prop, value)"', False) == "literal_value"
    assert k('"Incorrectly typed literal due to range (prop, value)"') == "literal_type"
    assert k("Property p has a typed range Datatype[xsd:string]", kind="dtRange") == (
        "literal_type")
    assert k("Lexical value of the literal 'x' does not match its datatype") == "literal_type"
    assert k("Irreflexive property used on x with y") == "other"


def test_the_same_report_with_other_blank_nodes_is_the_same():
    one = {"error": True, "type": "x", "description": "Culprit = 0123456789abcdef0123456789abcdef"}
    two = {**one, "description": "Culprit = fedcba9876543210fedcba9876543210"}
    assert li.key_of(one) == li.key_of(two)
    assert li.key_of(one) != li.key_of({**one, "description": "Culprit = <http://a#b>"})


# ------------------------------------------------------------------ owlrl (no Java needed)

def test_owlrl_tells_new_from_inherited(tmp_path):
    report = inspect(_parts(tmp_path), engine=OWLRL, name="gdpr")
    by = _by(report)
    assert [f["nodes"][-1] for f in by[("disjoint", NEW, True)]] == [EX + "x"]
    assert [f["nodes"][-1] for f in by[("disjoint", ONTOLOGY, True)]] == [EX + "y"]
    assert report["summary"]["errors"] == {NEW: 1, ONTOLOGY: 1, BASE: 0}
    assert report["status"] == RED and report["consistent"] is False
    assert report["engine"] == {"name": OWLRL, "reason": "elegido en config.yaml "
                                "(validation.engine)", "version": li.version("owlrl")}
    assert report["runs"]["inherited"]["triples"] < report["runs"]["full"]["triples"]
    assert [p["file"] for p in report["inherited_parts"]] == ["ontology.ttl"]


def test_without_contradictions_it_is_green(tmp_path):
    report = inspect(_parts(tmp_path, ontology="ex:A a owl:Class .", module="ex:x a ex:A ."),
                     engine=OWLRL)
    assert report["status"] == GREEN and report["consistent"] and report["findings"] == []
    assert "    [ok] El razonador no encontró contradicciones." in li.render_log(report)


def test_the_log(tmp_path):
    report = inspect(_parts(tmp_path), engine=OWLRL, name="gdpr")
    log = li.render_log(report)
    assert log.startswith("Inspector de lógica (razonador) del espacio gdpr\n")
    assert "  Motor: owlrl " in log and "(OWL 2 RL en Python): elegido en config.yaml" in log
    assert "  Para saber qué es heredado razonó también sobre: ontology.ttl" in log
    assert "  Resultado: rojo: 1 inconsistencia nueva, 1 heredada, 0 advertencias (" in log
    assert ("    [rojo] Un individuo está en dos clases disjuntas (inconsistencias): 2 (1 nuevo, "
            "1 de tu ontología)") in log
    assert "         x: A y B [nuevo]" in log and "         y: A y B [de tu ontología]" in log
    short = li.render_log({**report, "findings": report["findings"] * 3}, examples=1)
    assert "         … y 5 más (en reasoner-log.txt)" in short


def test_a_jena_failure_falls_back_to_owlrl(tmp_path, monkeypatch):
    def broken(engine, sets, s):
        raise RuntimeError("Jena tardó más de 1 s")
    monkeypatch.setattr(li, "_jena", broken)
    monkeypatch.setattr(li, "choose_engine", lambda s, java=None: (
        {"name": JENA, "version": "6.2.0", "java": "24", "reasoner": "owl-micro"}
        if s.engine != OWLRL else {"name": OWLRL, "reason": "x"}))
    report = inspect(_parts(tmp_path))
    assert report["engine"]["name"] == OWLRL
    assert report["engine"]["reason"] == "Jena tardó más de 1 s"
    with pytest.raises(RuntimeError, match="tardó"):
        inspect(_parts(tmp_path), engine=JENA)


# ------------------------------------------------------------------ Jena

@needs_jena
def test_jena_tells_new_from_inherited(tmp_path):
    report = inspect(_parts(tmp_path), engine=JENA, name="gdpr")
    by = _by(report)
    assert [f["nodes"][0] for f in by[("disjoint", NEW, True)]] == [EX + "x"]
    assert [f["nodes"][0] for f in by[("disjoint", ONTOLOGY, True)]] == [EX + "y"]
    assert by[("object_value", ONTOLOGY, True)][0]["nodes"][:2] == [EX + "z", EX + "p"]
    assert by[("literal_value", ONTOLOGY, False)][0]["nodes"][:2] == [EX + "z", EX + "q"]
    assert report["summary"]["errors"][NEW] == 1 and report["summary"]["warnings"][NEW] == 0
    assert report["status"] == RED
    e = report["engine"]
    assert (e["name"], e["version"], e["reasoner"]) == (JENA, "6.2.0", "owl-micro")
    assert li.engine_text(e) == f"Apache Jena 6.2.0, razonador OWL Micro (Java {e['java']})"


@needs_jena
def test_jena_reads_every_format_and_reports_the_parser(tmp_path):
    rdf = tmp_path / "x.rdf"
    Graph().parse(data=PREFIXES + ONTOLOGY_TTL, format="turtle").serialize(rdf, format="xml")
    weird = tmp_path / "weird.ttl"
    weird.write_text(PREFIXES + 'ex:w ex:note "texto"^^'
                     '<http://www.w3.org/1999/02/22-rdf-syntax-ns#langString> .',
                     encoding="utf-8")
    parts = [Part("ontology", "x.rdf", ONTOLOGY, Graph().parse(rdf), "", rdf),
             Part("labels", "weird.ttl", NEW, Graph().parse(weird), "", weird)]
    report = inspect(parts, engine=JENA)
    assert report["runs"]["full"]["triples"] == len(parts[0].graph) + 1
    assert any("langString but no language tag" in n for n in report["parser_notes"])


@needs_jena
def test_jena_literal_range_artifact_is_dismissed(tmp_path):
    ontology = 'ex:A a owl:Class ; rdfs:comment "Una cita."^^rdfs:Literal ; rdfs:label "A" .'
    report = inspect(_parts(tmp_path, ontology=ontology, module='ex:A rdfs:label "A"@es .'),
                     engine=JENA)
    assert report["findings"] == [] and report["dismissed"] > 0
    assert "  Descartadas: " in li.render_log(report)


# ------------------------------------------------------------------ the real base and the CLI

BASE_FILES = [ROOT / "data" / "output" / name for name in
              ("ontopriv-core.rdf", "profile-ecuador-lopdp.rdf", "ontopriv-dpv-alignment.rdf")]
needs_base = pytest.mark.skipif(not all(p.exists() for p in BASE_FILES),
                                reason="Falta la base en data/output")


@needs_base
def test_the_real_base_with_owlrl(tmp_path):
    report = inspect_base(out_dir=tmp_path, engine=OWLRL)
    assert report["summary"]["errors"] == {NEW: 0, ONTOLOGY: 419, BASE: 0}
    assert report["summary"]["kinds"]["disjoint"]["errors"] == 419
    assert report["status"] == AMBER
    assert [p["file"] for p in report["inherited_parts"]] == ["data/input/ontopriv.rdf"]
    assert (tmp_path / REPORT_FILE).exists() and (tmp_path / LOG_FILE).exists()


@needs_base
@needs_jena
def test_the_real_base_with_jena(tmp_path):
    report = inspect_base(out_dir=tmp_path, engine=JENA)
    s = report["summary"]
    assert s["errors"][NEW] == 0 and s["warnings"][NEW] == 0
    assert s["kinds"]["disjoint"]["errors"] == 419                # the same as owlrl
    assert s["errors"][ONTOLOGY] == 830 and s["warnings"][ONTOLOGY] == 1310
    assert report["status"] == AMBER


def test_the_cli(tmp_path, capsys):
    path = _workspace(tmp_path, files={"ontology": "input/x.ttl"})
    assert li._run([str(path), "--motor", OWLRL]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Inspector de lógica (razonador) del espacio mi-ley-1234")
    assert "  Resultado: verde: 0 inconsistencias nuevas, 0 heredadas, 0 advertencias" in out
    assert f"  Guardado en: {path / 'work' / REPORT_FILE} y {path / 'work' / LOG_FILE}" in out
    saved = json.loads((path / "work" / REPORT_FILE).read_text(encoding="utf-8"))
    assert li.load_result(li.si.open_workspace(str(path), {})) == saved
    assert li._run([]) == 1 and li._run(["--base", "x"]) == 1
    assert "Error: indica un espacio de trabajo o --base." in capsys.readouterr().out
    assert li._run(["no-existe"]) == 1
    assert "Error: No existe el espacio de trabajo no-existe." in capsys.readouterr().out