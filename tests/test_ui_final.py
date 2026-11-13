"""S10-T15: the last screen gives one file per format (the modules put together, no imports, no
provenance) and, from version 2 on, the previous version; the technical details only when the
person asks for them. The LLM and the embedder are fakes."""
from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace

from rdflib import Graph, Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import OWL, RDF, RDFS, SKOS

from app.services import align_ontology as svc
from src.alignment import user_alignment as ua
from src.alignment import user_decisions as ud
from src.transform.convert import parse_bytes
from src.transform.merge import merge, package_graphs, removed
from src.transform.package import catalog_xml
from src.workspace.store import load_workspace
from tests.test_alignment_propose import env, targets  # noqa: F401
from tests.test_ui_align_ontology import _titles, _zips, cfg, page  # noqa: F401
from tests.test_upload_questions import aligned, doubtful  # noqa: F401

EX = "http://example.org/"
DPV = "https://w3id.org/dpv#"


def _g(ttl: str) -> Graph:
    return Graph().parse(data=f"@prefix ex: <{EX}> . @prefix owl: <{OWL}> . "
                              f"@prefix rdfs: <{RDFS}> . @prefix skos: <{SKOS}> . " + ttl,
                         format="turtle")


ONTO = _g("ex:onto a owl:Ontology ; rdfs:label 'Mía' ; owl:imports ex:other . "
          "ex:A a owl:Class ; rdfs:subClassOf [ a owl:Restriction ] .")
ALIGN = _g("ex:onto-align a owl:Ontology ; rdfs:label 'Alineación' ; owl:imports ex:onto , "
           "ex:onto-labels . ex:A skos:exactMatch <https://w3id.org/dpv#Consent> .")
LABELS = _g("ex:onto-labels a owl:Ontology ; owl:imports ex:onto . ex:A rdfs:label 'A'@es .")


def test_the_modules_put_together():
    one = merge([ALIGN, LABELS, ONTO], EX + "onto")
    assert set(one.subjects(RDF.type, OWL.Ontology)) == {URIRef(EX + "onto")}
    assert set(one.objects(None, OWL.imports)) == {URIRef(EX + "other")}   # not a module
    assert (URIRef(EX + "onto"), RDFS.label, Literal("Mía")) in one
    assert (URIRef(EX + "A"), SKOS.exactMatch, URIRef(DPV + "Consent")) in one
    assert (URIRef(EX + "A"), RDFS.label, Literal("A", lang="es")) in one
    assert len(list(one.objects(URIRef(EX + "A"), RDFS.subClassOf))) == 1  # blank node kept
    assert {(str(s), str(p)) for s, p, _ in removed([ALIGN, LABELS, ONTO], EX + "onto")} == {
        (EX + "onto-align", str(RDF.type)), (EX + "onto-align", str(RDFS.label)),
        (EX + "onto-align", str(OWL.imports)), (EX + "onto-labels", str(RDF.type)),
        (EX + "onto-labels", str(OWL.imports))}
    assert len(one) == len(ALIGN) + len(LABELS) + len(ONTO) - 6
    assert dict(one.namespaces())["ex"] == URIRef(EX)                     # its prefixes


def test_the_ontology_files_of_a_package():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("onto.rdf", ONTO.serialize(format="xml"))
        z.writestr("onto-align.ttl", ALIGN.serialize(format="turtle"))
        z.writestr("onto-provenance.ttl", "this is not read")
        z.writestr("otro.ttl", "not in the catalog: not read")
        z.writestr("catalog-v001.xml", catalog_xml({EX + "onto": "onto.rdf",
                                                    EX + "onto-align": "onto-align.ttl",
                                                    EX + "onto-provenance": "onto-provenance.ttl"}))
        z.writestr("LEEME.txt", "léeme")
        z.writestr("alignment.csv", "a,b")
    graphs = package_graphs(buf.getvalue(), skip="-provenance")
    assert [len(g) for g in graphs] == [len(ALIGN), len(ONTO)]            # sorted by name
    assert isomorphic(merge(graphs, EX + "onto"), merge([ONTO, ALIGN], EX + "onto"))


def _changed(ws, env, targets):  # noqa: F811
    """Version 2 of the test upload: PersonalData only related to the DPV term."""
    out = svc.apply_answers(ws.id, {1: "a"}, targets=targets)
    assert out["version"] == 2
    return load_workspace(ws.id, cfg=env.cfg)


def test_the_name_is_what_the_person_uploaded():
    law = SimpleNamespace(manifest={"inputs": {
        "law": {"original_name": "gdpr-en.pdf", "stored_as": "gdpr-en.pdf"},
        "ontology": {"original_name": "Perfil de gdpr-en.pdf", "stored_as": "gdpr-en-profile.ttl",
                     "generated_from": "law"}}})
    onto = SimpleNamespace(manifest={"inputs": {
        "law": {"original_name": "lopdp.pdf", "stored_as": "lopdp.pdf"},
        "ontology": {"original_name": "Mi GDPR.ttl", "stored_as": "mi-gdpr.ttl"}}})
    assert (svc.source_name(law), svc.aligned_stem(law)) == ("gdpr-en.pdf", "gdpr-en-dpv-aligned")
    assert (svc.source_name(onto), svc.aligned_stem(onto)) == ("Mi GDPR.ttl", "mi-gdpr-dpv-aligned")


def test_the_previous_version(doubtful, env, targets, monkeypatch):  # noqa: F811
    from src.alignment import user_questions as uq
    monkeypatch.setattr(svc, "workspace_cfg", lambda: env.cfg)
    uq.refresh_questions(doubtful, env.cfg, targets=targets)
    before = svc.final_bytes(doubtful.id, "turtle", env.cfg)
    ws = _changed(doubtful, env, targets)
    final = svc.final_downloads(ws.id)
    assert [b["file_name"] for b in final["buttons"]] == [
        "mi-gdpr-dpv-aligned-v2.rdf", "mi-gdpr-dpv-aligned-v2.owl", "mi-gdpr-dpv-aligned-v2.ttl",
        "mi-gdpr-dpv-aligned-v2.jsonld"]
    previous = final["previous"]
    assert previous["label"] == "Descargar la versión anterior (versión 1, Turtle)"
    assert previous["file_name"] == "mi-gdpr-dpv-aligned-v1.ttl"
    old, now = parse_bytes(previous["data"](), "turtle"), parse_bytes(
        final["buttons"][2]["data"](), "turtle")
    assert isomorphic(old, parse_bytes(before, "turtle"))               # what v1 gave
    personal = URIRef("http://example.org/gdpr#PersonalData")
    assert (personal, SKOS.exactMatch, URIRef(DPV + "PersonalData")) in old
    assert (personal, SKOS.relatedMatch, URIRef(DPV + "PersonalData")) in now
    (ud.history_dir(ws, 1) / ua.package_name(ws, "turtle", 1)).unlink()
    assert svc.final_downloads(ws.id)["previous"] is None              # nothing kept: no link
    assert ua.previous_single_file(ws, 1) is None


def test_the_last_screen_from_version_2(page, doubtful, env, targets):  # noqa: F811
    from src.alignment import user_questions as uq
    uq.refresh_questions(doubtful, env.cfg, targets=targets)
    svc_cfg = svc.workspace_cfg
    svc.workspace_cfg = lambda: env.cfg
    try:
        _changed(doubtful, env, targets)
    finally:
        svc.workspace_cfg = svc_cfg
    at = page()
    at.session_state["align_ws"], at.session_state["align_review"] = doubtful.id, True
    at.run()
    assert not at.exception and _titles(at) == [svc.FINAL_TITLE]
    assert _zips(at) == ["RDF/XML", "OWL", "Turtle", "JSON-LD"]
    assert [d.proto.type for d in at.download_button] == ["primary"] * 4 + ["tertiary"]
    [previous] = [d for d in at.download_button if d.key == "align_dl_previous"]
    assert previous.label == "Descargar la versión anterior (versión 1, Turtle)"
    assert not at.tabs and not at.metric                                # details: off
    at.toggle(key="align_details").set_value(True).run()
    assert [t.label for t in at.tabs][:3] == ["Resumen", "Conceptos", "Para revisar"]
    at.button(key="align_review_back").click().run()                    # back to "Tú decides"
    assert _titles(at)[0] in (svc.DECIDE_TITLE,)
