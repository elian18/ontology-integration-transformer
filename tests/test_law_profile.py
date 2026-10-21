"""S7-T10: the "solo ley" mode. The accepted concepts of a law become a profile that imports
OntoPriv-Core (decision D6: reuse the IRI of the core, link the very similar ones, create the
rest), and the profile goes through the alignment flow of S7-T06/T08 into a zip that also
carries the core.

Fast tests: the small validated base of the decider tests (OntoPriv Consent and Turnover, the
AI concept ImpactAssessment), a small core graph (Consent is in the core, Turnover is not, as if
it lived in the Ecuador profile), a small law in .txt and a fake LLM that answers the three
prompts of the flow (extraction, decider, labels)."""
from __future__ import annotations

import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from src.core import law_profile as lp
from src.core.emit import CORE_FILE, CORE_IRI
from src.core.law_concepts import RESULT_FILE as CONCEPTS_FILE, prepare_law_concepts
from src.core.law_profile import (LINKED, NEW, REUSED, align_law, is_current, match_core,
                                  profile_for_workspace, profile_graph, render_console)
from src.workspace.store import DONE, ERROR, WorkspaceError, open_law
from tests.test_alignment_bundle import _resolve
from tests.test_alignment_propose import DeciderLLM, make_env, targets  # noqa: F401

ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
LAW = ("Art. 1.- Consentimiento.- El consentimiento del titular es libre e informado.\n"
       "Art. 2.- Volumen.- El volumen de negocio sirve para graduar las multas.\n"
       "Art. 3.- Evaluación.- La evaluación de impacto se hace antes del tratamiento.\n"
       "Art. 4.- Registro.- El registro guarda las cosas del responsable.\n")
CONCEPTS = {
    1: [{"name": "Consent", "label": "consentimiento", "type": "class",
         "definition": "Manifestación de voluntad libre del titular.", "article": 1},
        {"name": "ConsentGiven", "label": "consentimiento otorgado", "type": "class",
         "definition": "El consent que el titular ya otorgó.", "article": 1},
        {"name": "Permission", "label": "Consent", "type": "class",
         "definition": "Otra manera de nombrar el consentimiento del titular.", "article": 1}],
    2: [{"name": "BusinessVolume", "label": "volumen de negocio", "type": "class",
         "definition": "Cifra de negocios usada para graduar las multas.", "article": 2}],
    3: [{"name": "ImpactAssessment", "label": "evaluación de impacto", "type": "class",
         "definition": "Análisis de los riesgos de un tratamiento.", "article": 3}],
    4: [{"name": "hasThing", "label": "tiene cosa", "type": "property",
         "definition": "Relaciona el registro con las cosas que guarda.", "article": 4}],
}
CORE_TTL = f"""
@prefix owl: <http://www.w3.org/2002/07/owl#> .
<{CORE_IRI}> a owl:Ontology .
<{ONTO}Consent> a owl:Class .
<{ONTO}Other> a owl:Class, owl:DatatypeProperty .
<{ONTO}isSupervisedBy> a owl:ObjectProperty .
"""


class LawLLM(DeciderLLM):
    """Answers the extraction prompt (S7-T09) per article; the decider and labels as
    DeciderLLM."""

    def ask(self, prompt, context="", max_tokens=1024, json_mode=False):
        if "Extrae los conceptos" in prompt:
            self.calls += 1
            numbers = [int(n) for n in re.findall(r"=== ARTICULO (\d+) ", context)]
            return "\n".join(json.dumps(c, ensure_ascii=False)
                             for n in numbers for c in CONCEPTS.get(n, []))
        return super().ask(prompt, context, max_tokens, json_mode)


class Embed:
    """Texts about consent share one direction; every other text gets its own."""

    def __init__(self):
        self.seen, self.calls = {}, 0

    def __call__(self, texts):
        self.calls += 1
        out = []
        for t in texts:
            v = np.zeros(64)
            if "consent" in t.lower():
                v[0] = 1.0
            else:
                v[1 + self.seen.setdefault(t, len(self.seen)) % 63] = 1.0
            out.append(v)
        return out


@pytest.fixture
def env(tmp_path, targets):  # noqa: F811
    env = make_env(tmp_path)
    env.cfg = {**env.cfg, "extraction": {"pause_seconds": 0}}
    out = Path(env.cfg["outputs"]["dir"])
    Graph().parse(data=CORE_TTL, format="turtle").serialize(out / CORE_FILE, format="xml")
    env.targets = targets
    return env


def _extracted(env, llm=None):
    ws = open_law("Ley de prueba.txt", LAW.encode("utf-8"), cfg=env.cfg)
    prepare_law_concepts(ws, llm=llm or LawLLM(), cfg=env.cfg, sleep_fn=lambda s: None)
    return ws


def _kw(env, **extra):
    return dict(base=env.base, targets=env.targets, embed_fn=Embed(), cfg=env.cfg,
                chroma_path=env.chroma, sleep_fn=lambda s: None, **extra)


# ------------------------------------------------------------------ D6

def test_d6_reuses_links_or_creates(env):
    ws = _extracted(env)
    concepts = json.loads((ws.work_dir / CONCEPTS_FILE).read_text(encoding="utf-8"))
    core = lp.load_core(env.cfg)
    assert lp.core_entities(core) == {ONTO + "Consent", ONTO + "Other", ONTO + "isSupervisedBy"}
    matched = match_core(concepts["entries"], env.base, lp.core_entities(core),
                         embed_fn=Embed(), cfg=env.cfg)
    entries = {e["name"]: e for e in matched}
    assert entries["Consent"]["status"] == REUSED                     # same name, in the core
    assert entries["Consent"]["core_iri"] == ONTO + "Consent"
    assert entries["Consent"]["base_match"]["method"] == "same_name"
    assert entries["Permission"]["status"] == REUSED                  # its label is "Consent"
    assert entries["ConsentGiven"]["status"] == LINKED                # very similar
    assert entries["ConsentGiven"]["base_match"]["method"] == "similar"
    assert entries["ConsentGiven"]["base_match"]["score"] >= 0.75
    assert entries["BusinessVolume"]["status"] == NEW                 # alias of Turnover, but
    assert entries["BusinessVolume"]["base_match"]["name"] == "Turnover"   # not in the core
    assert entries["ImpactAssessment"]["status"] == NEW               # Ecuador profile
    assert entries["ImpactAssessment"]["base_match"]["origin"] == "ai"
    assert entries["hasThing"]["status"] == NEW and entries["hasThing"]["base_match"] is None
    assert entries["hasThing"]["core_iri"] is None and entries["hasThing"]["kind"] == "property"
    assert lp.summarize(list(entries.values())) == {
        "concepts": 6, "reused": 2, "linked": 1, "new": 3, "core_entities": 1, "classes": 5,
        "properties": 1}


def test_only_accepted_concepts_go_to_the_profile():
    entries = [{"name": "A", "kind": "class", "label": "a", "definition": "d", "articles": [1],
                "status": "accepted"},
               {"name": "B", "kind": "class", "label": "b", "definition": "d", "articles": [1],
                "status": "repeated"}]
    assert [s.key for s in lp.law_concepts_as_sources(entries)] == ["law:A"]
    s = lp.law_concepts_as_sources([{**entries[0], "label": "", "definition": ""}])[0]
    assert (s.label, s.definition, s.kinds, s.origin, s.articles) == (None, None, ("class",),
                                                                      "ai", [1])


def test_the_profile_graph():
    core = Graph().parse(data=CORE_TTL, format="turtle")
    entries = [
        {"name": "Consent", "kind": "class", "label": "consent", "definition": "Free will.",
         "articles": [1, 7], "status": REUSED, "core_iri": ONTO + "Other"},
        {"name": "ConsentGiven", "kind": "class", "label": "consent given", "definition": None,
         "articles": [1], "status": LINKED, "core_iri": ONTO + "Consent"},
        {"name": "hasThing", "kind": "property", "label": None, "definition": "Relates.",
         "articles": [], "status": NEW, "core_iri": None}]
    iri = "http://www.semanticweb.org/profiles/x"
    g = profile_graph(entries, core, iri=iri, law="GDPR", language="en", created="2026-10-20")
    node = URIRef(iri)
    assert g.value(node, OWL.imports) == URIRef(CORE_IRI)
    assert str(g.value(node, RDFS.label)) == "Perfil de GDPR"
    assert "reutilizan su IRI" in str(g.value(node, RDFS.comment))
    assert g.value(node, DCTERMS.source) == Literal("GDPR")
    assert g.value(node, OWL.versionInfo) == Literal("1")
    other = URIRef(ONTO + "Other")
    assert set(g.objects(other, RDF.type)) == {OWL.Class, OWL.DatatypeProperty}   # the core's
    assert g.value(other, RDFS.label) == Literal("consent", lang="en")
    assert g.value(other, SKOS.definition) == Literal("Free will.", lang="en")
    assert g.value(other, SKOS.exactMatch) is None                      # reused: no link
    sources = {Literal("GDPR, art. 1", lang="es"), Literal("GDPR, art. 7", lang="es")}
    assert set(g.objects(other, DCTERMS.source)) == sources
    given = URIRef(iri + "#ConsentGiven")
    assert (given, RDF.type, OWL.Class) in g
    assert g.value(given, SKOS.exactMatch) == URIRef(ONTO + "Consent")
    assert g.value(given, SKOS.definition) is None
    thing = URIRef(iri + "#hasThing")
    assert (thing, RDF.type, OWL.ObjectProperty) in g and g.value(thing, RDFS.label) is None
    assert g.value(thing, SKOS.exactMatch) is None
    assert [e["iri"] for e in entries] == [ONTO + "Other", iri + "#ConsentGiven",
                                           iri + "#hasThing"]
    prefixes = dict(g.namespaces())
    assert str(prefixes["core"]) == ONTO and str(prefixes["profile"]) == iri + "#"
    assert lp._namespace(Graph()) == CORE_IRI + "#"                    # an empty core


# ------------------------------------------------------------------ the workspace

def test_profile_for_workspace(env):
    ws = _extracted(env)
    stale = ws.work_dir / "inherited.json"
    stale.write_text("{}", encoding="utf-8")
    r = profile_for_workspace(ws, base=env.base, embed_fn=Embed(), cfg=env.cfg)
    assert r["profile_iri"] == "http://www.semanticweb.org/profiles/" + ws.id
    assert r["core_iri"] == CORE_IRI and r["language"] == "es"
    assert r["law"] == "Ley de prueba.txt" and r["file"] == "input/ley-de-prueba-profile.ttl"
    assert r["summary"]["reused"] == 2 and r["summary"]["new"] == 3
    assert lp.load_result(ws) == r and not stale.exists()             # stale results go
    entry = ws.manifest["inputs"]["ontology"]
    assert entry["original_name"] == "Perfil de Ley de prueba.txt"
    assert entry["generated_from"] == "law" and entry["stored_as"] == "ley-de-prueba-profile.ttl"
    assert ws.input_path() == ws.input_dir / "ley-de-prueba-profile.ttl"
    g = Graph().parse(ws.input_path())
    assert len(g) == r["triples"]
    assert g.value(URIRef(ONTO + "Consent"), RDFS.label) is not None
    assert set(g.objects(URIRef(ONTO + "Consent"), RDFS.label)) == {
        Literal("consentimiento", lang="es"), Literal("Consent", lang="es")}  # two law terms
    step = ws.step("profile")
    assert step["status"] == DONE
    assert step["detail"] == "2 reutilizan OntoPriv-Core, 1 enlazados y 3 nuevos"
    before = ws.input_path().read_bytes()
    stale.write_text("{}", encoding="utf-8")
    real = lp.datetime

    class Later:
        @staticmethod
        def now(tz=None):
            return real(2030, 1, 1, tzinfo=tz)
    lp.datetime = Later                                                # another date only
    try:
        profile_for_workspace(ws, base=env.base, embed_fn=Embed(), cfg=env.cfg)
    finally:
        lp.datetime = real
    assert ws.input_path().read_bytes() == before and stale.exists()  # same content: kept
    ws.input_path().unlink()
    profile_for_workspace(ws, base=env.base, embed_fn=Embed(), cfg=env.cfg)
    assert ws.input_path().exists()                                    # missing: written again


def test_without_concepts_there_is_no_profile(env):
    ws = open_law("Ley de prueba.txt", LAW.encode("utf-8"), cfg=env.cfg)
    with pytest.raises(ValueError, match="extraer y aceptar"):
        profile_for_workspace(ws, base=env.base, cfg=env.cfg)
    off = {**env.cfg, "auto": {**env.cfg["auto"], "use_llm": False}}
    prepare_law_concepts(ws, cfg=off)
    with pytest.raises(WorkspaceError, match="la IA no estuvo disponible para extraerlos"):
        profile_for_workspace(ws, base=env.base, cfg=env.cfg)
    assert ws.step("profile")["status"] == ERROR
    path = ws.work_dir / CONCEPTS_FILE
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["ai"] = {"available": True}
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(WorkspaceError, match="la regla no aceptó ninguno"):
        profile_for_workspace(ws, base=env.base, cfg=env.cfg)


def test_errors_mark_the_step_and_the_core_must_exist(env, tmp_path):
    ws = _extracted(env)
    no_core = {**env.cfg, "outputs": {"dir": str(tmp_path / "vacio")}}
    with pytest.raises(FileNotFoundError, match="OntoPriv-Core"):
        profile_for_workspace(ws, base=env.base, embed_fn=Embed(), cfg=no_core)
    assert ws.step("profile")["status"] == ERROR and "OntoPriv-Core" in ws.step("profile")[
        "detail"]


def test_is_current(env):
    ws = _extracted(env)
    concepts = json.loads((ws.work_dir / CONCEPTS_FILE).read_text(encoding="utf-8"))
    assert not is_current(None, concepts)
    r = profile_for_workspace(ws, base=env.base, embed_fn=Embed(), cfg=env.cfg)
    assert is_current(r, concepts) and not is_current(r, None)
    assert not is_current({**r, "method_version": 0}, concepts)
    changed = json.loads(json.dumps(concepts))
    changed["entries"][0]["definition"] = "Otra definición del concepto."
    assert not is_current(r, changed)
    changed["entries"][0] = {**concepts["entries"][0], "status": "rejected"}
    assert not is_current(r, changed)


# ------------------------------------------------------------------ the whole mode

def test_align_law_end_to_end(env, tmp_path):
    llm, seen = LawLLM(), []
    ws, m, path = align_law("Ley de prueba.txt", LAW.encode("utf-8"), llm=llm,
                            progress=seen.append, **_kw(env))
    assert seen == ["Separar la ley en artículos", "Indexar los artículos de la ley",
                    "Extraer los conceptos de la ley con la IA",
                    "Aceptar los conceptos por regla",
                    "Armar el perfil de la ley sobre OntoPriv-Core", "Heredar lo validado",
                    "Buscar candidatos del DPV", "Completar las etiquetas en español e inglés",
                    "Escribir la ontología alineada", "Preparar la descarga"]
    assert all(s["status"] == DONE for s in ws.progress())
    assert m["generated_from"] == "law" and m["ontology_iri"] == lp.profile_iri(ws)
    assert m["modules"]["core"] == CORE_IRI and m["files"]["core"] == "output/ontopriv-core.ttl"
    assert m["triples"]["core"] == 5
    assert m["profile"]["reused"] == 2 and m["profile"]["law"] == "Ley de prueba.txt"
    assert m["summary"]["concepts"] == 5             # 6 accepted, 2 of them reuse one IRI
    assert m["profile"]["law_concepts"]["accepted"] == 6
    assert m["profile"]["extraction"]["complete"] is True
    assert m["settings"]["prompts"]["extract"]["version"] == 2
    inherited = {i["name"]: i for i in
                 json.loads((ws.work_dir / "inherited.json").read_text())["inherited"]}
    assert inherited["Consent"]["method"] == "same_iri"               # reused: by its IRI
    assert path.name == "ley-de-prueba-profile-alineada-turtle.zip"
    names = zipfile.ZipFile(path).namelist()
    assert {"ley-de-prueba-profile-dpv-alignment.ttl", "ley-de-prueba-profile-labels.ttl",
            "ley-de-prueba-profile.ttl", "ontopriv-core.ttl"} <= set(names)
    folder = tmp_path / "z"
    zipfile.ZipFile(path).extractall(folder)
    g = _resolve(folder, m["modules"]["alignment"])
    assert (URIRef(ONTO + "Other"), RDF.type, OWL.DatatypeProperty) in g    # the core resolves
    assert (URIRef(lp.profile_iri(ws) + "#hasThing"), RDF.type, OWL.ObjectProperty) in g
    assert (URIRef(ONTO + "Consent"), SKOS.exactMatch,
            URIRef("https://w3id.org/dpv#Consent")) in g                # inherited mapping
    readme = (folder / "LEEME.txt").read_text(encoding="utf-8")
    assert ("  - ley-de-prueba-profile.ttl: el perfil de tu ley (6 conceptos que la IA extrajo "
            "y la regla aceptó: 2 reutilizan OntoPriv-Core, 1 se enlazan con él y 3 son "
            "nuevos). Importa OntoPriv-Core.") in readme
    assert ("  - ontopriv-core.ttl: el núcleo OntoPriv-Core de la base del sistema, sin "
            "cambios.") in readme
    assert "Tu ontología, sin cambios" not in readme
    from src.alignment.user_alignment import render_console as ua_console
    assert "  Núcleo OntoPriv-Core incluido: output/ontopriv-core.ttl (5 tripletas)" in \
        ua_console(ws, m, path)
    calls, seen = llm.calls, []
    again = align_law("Ley de prueba.txt", LAW.encode("utf-8"), llm=llm, progress=seen.append,
                      **_kw(env))
    assert llm.calls == calls and again[0].path == ws.path            # nothing recomputed
    assert "Armar el perfil de la ley sobre OntoPriv-Core" not in seen


def test_a_core_already_loaded_is_used(env):
    core = Graph().parse(data=CORE_TTL, format="turtle")
    core.remove((URIRef(ONTO + "Other"), None, None))
    _, m, _ = align_law("Ley de prueba.txt", LAW.encode("utf-8"), llm=LawLLM(), core=core,
                        **_kw(env))
    assert m["triples"]["core"] == len(core) == 3


def test_a_new_extraction_rebuilds_the_profile(env):
    ws, _, _ = align_law("Ley de prueba.txt", LAW.encode("utf-8"), llm=LawLLM(), **_kw(env))
    first = lp.load_result(ws)
    path = ws.work_dir / CONCEPTS_FILE
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["entries"] = [e for e in saved["entries"] if e["name"] != "hasThing"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    seen = []
    _, m, _ = align_law("Ley de prueba.txt", LAW.encode("utf-8"), llm=LawLLM(),
                        progress=seen.append, **_kw(env))
    assert "Armar el perfil de la ley sobre OntoPriv-Core" in seen
    assert "Heredar lo validado" in seen                               # the profile changed
    assert lp.load_result(ws)["summary"]["new"] == first["summary"]["new"] - 1
    assert m["summary"]["concepts"] == 4             # Consent and Permission share one IRI


def test_console(env):
    ws = _extracted(env)
    r = profile_for_workspace(ws, base=env.base, embed_fn=Embed(), cfg=env.cfg)
    text = render_console(r)
    assert text.splitlines()[:5] == [
        "Perfil de la ley (S7-T10)",
        "  Ley de prueba.txt: 6 conceptos aceptados (5 clases, 1 propiedades)",
        "    reutilizan el IRI de OntoPriv-Core: 2 (en 1 entidades del núcleo)",
        "    nuevos enlazados al núcleo (muy parecidos, skos:exactMatch): 1",
        "    nuevos en el perfil: 3"]
    assert (f"  Perfil: input/ley-de-prueba-profile.ttl ({r['triples']} tripletas), importa "
            f"{CORE_IRI}") in text
    assert "  Reutiliza el iri de ontopriv-core:" in text
    assert '    Consent "consentimiento" -> Consent (same name, 1.00)' in text
    assert "  Nuevo, enlazado a ontopriv-core (muy parecido):" in text
    assert '    ConsentGiven "consentimiento otorgado" -> Consent (similar, 0.88)' in text
    assert len(render_console(r, examples=0).splitlines()) == 6


def test_cli(env, tmp_path, monkeypatch, capsys):
    fake = types.ModuleType("src.ai.rag.embeddings")
    fake.embed = Embed()
    monkeypatch.setitem(sys.modules, "src.ai.rag.embeddings", fake)
    monkeypatch.setattr("src.ai.rag.vector_store._PATH", str(tmp_path / "chroma-cli"))
    monkeypatch.setattr("src.config.load_config", lambda: env.cfg)
    import src.ai.llm_client as client
    from src.alignment import inherit as inh, propose as prop
    monkeypatch.setattr(client, "LLMClient", LawLLM)
    monkeypatch.setattr(inh, "load_base", lambda **kw: env.base)
    monkeypatch.setattr(prop, "load_dpv_targets", lambda cfg=None: env.targets)
    law = tmp_path / "ley.txt"
    law.write_text(LAW, encoding="utf-8")
    assert lp._run([str(law), "--formato", "turtle"]) == 0
    out = capsys.readouterr().out
    assert "Perfil de la ley (S7-T10)" in out and "Ontología alineada (S7-T08)" in out
    assert "ley-profile-alineada-turtle.zip" in out
    owx = tmp_path / "owx.txt"
    owx.write_text(LAW.replace("Volumen", "Monto"), encoding="utf-8")
    assert lp._run([str(owx), "--formato", "owx"]) == 1
    assert not any("owx" in p.name for p in Path(env.cfg["workspaces"]["dir"]).iterdir())
    assert lp._run([str(tmp_path / "no.txt")]) == 1
    other = tmp_path / "otra.txt"
    other.write_text(LAW.replace("Registro", "Archivo"), encoding="utf-8")
    assert lp._run([str(other), "--sin-ia"]) == 1
    assert "No hay conceptos aceptados de la ley (la IA no estuvo disponible" in \
        capsys.readouterr().out


# ------------------------------------------------------------------ real base and embeddings

@pytest.mark.slow
def test_real_d6_on_a_few_gdpr_concepts():
    """The real base, OntoPriv-Core and embedding model (no LLM call)."""
    from src.alignment.inherit import load_base
    try:
        base, core = load_base(), lp.load_core()
    except FileNotFoundError:
        pytest.skip("Faltan los archivos de la base del Sprint 3-5")
    entries = [{"name": n, "kind": k, "label": label, "definition": d, "articles": [4],
                "status": "accepted"} for n, k, label, d in [
        ("DataSubject", "class", "data subject",
         "An identified or identifiable natural person."),
        ("Controller", "class", "controller",
         "The natural or legal person which determines the purposes and means of the "
         "processing of personal data."),
        ("Consent", "class", "consent",
         "Any freely given, specific, informed and unambiguous indication of the data "
         "subject's wishes."),
        ("SupervisoryAuthority", "class", "supervisory authority",
         "An independent public authority established by a Member State.")]]
    matched = {e["name"]: e for e in match_core(entries, base, lp.core_entities(core))}
    print("\n" + "\n".join(f"  {e['name']}: {e['status']} -> {e['base_match']}"
                           for e in matched.values()))
    assert matched["DataSubject"]["status"] == REUSED            # alias of Data_Owner
    assert matched["DataSubject"]["core_iri"].endswith("#Data_Owner")
    assert matched["Consent"]["status"] == REUSED
    for e in matched.values():
        if e["status"] == LINKED:
            assert e["base_match"]["score"] >= 0.75 and e["core_iri"] in lp.core_entities(core)