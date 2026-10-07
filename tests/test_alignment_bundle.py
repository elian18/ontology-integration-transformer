"""S5-T07: the zip for Protégé carries the three modules and a catalog that resolves the imports
alignment -> profile -> core by IRI."""
import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, SKOS

from src.core.emit import CORE_IRI, PROFILE_IRI, CORE_FILE, PROFILE_FILE
from src.alignment.write import ALIGNMENT_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV, alignment_iri
from src.alignment.bundle import (build_bundle, catalog_xml, CATALOG_FILE, README_FILE,
                                  BUNDLE_FILE)

ROOT = Path(__file__).resolve().parents[1]
CAT = "{urn:oasis:names:tc:entity:xmlns:xml:catalog}"
ONTO = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"


def _module(path, iri, imports=None, body=""):
    imp = f'<owl:imports rdf:resource="{imports}"/>' if imports else ""
    path.write_text(f"""<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:skos="http://www.w3.org/2004/02/skos/core#">
  <owl:Ontology rdf:about="{iri}">{imp}</owl:Ontology>
  {body}
</rdf:RDF>""", encoding="utf-8")


def _out(tmp_path, with_modules=True):
    out = tmp_path / "out"
    out.mkdir()
    _module(out / ALIGNMENT_FILE, alignment_iri(PROFILE_IRI), PROFILE_IRI,
            f'<rdf:Description rdf:about="{ONTO}Consent">'
            f'<skos:exactMatch rdf:resource="https://w3id.org/dpv#Consent"/></rdf:Description>')
    (out / ALIGNMENT_MANIFEST).write_text("{}", encoding="utf-8")
    (out / ALIGNMENT_CSV).write_text("a,b\n", encoding="utf-8")
    if with_modules:
        _module(out / PROFILE_FILE, PROFILE_IRI, CORE_IRI)
        _module(out / CORE_FILE, CORE_IRI, body=f'<owl:Class rdf:about="{ONTO}Consent"/>')
    return out


def _resolve(folder: Path, start: str) -> Graph:
    """Load a module and its imports the way Protégé does: IRI -> file through the catalog."""
    root = ET.parse(folder / CATALOG_FILE).getroot()
    files = {u.get("name"): u.get("uri") for u in root.iter(f"{CAT}uri")}
    g, todo, seen = Graph(), [start], set()
    while todo:
        iri = todo.pop()
        if iri in seen:
            continue
        seen.add(iri)
        part = Graph().parse(folder / files[iri], format="xml")    # KeyError = unresolved
        g += part
        todo += [str(o) for o in part.objects(URIRef(iri), OWL.imports)]
    return g


def test_needs_the_alignment_file(tmp_path):
    with pytest.raises(ValueError, match="escribelo primero"):
        build_bundle(tmp_path)


def test_catalog_maps_each_iri_to_its_file():
    xml = catalog_xml({"http://a/x": "x.rdf", "http://a/y": "y & z.rdf"})
    root = ET.fromstring(xml)
    assert {u.get("name"): u.get("uri") for u in root.iter(f"{CAT}uri")} == {
        "http://a/x": "x.rdf", "http://a/y": "y & z.rdf"}


def test_zip_contents_and_import_chain(tmp_path):
    data, included, missing = build_bundle(_out(tmp_path))
    assert missing == []
    names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
    assert names == {ALIGNMENT_FILE, PROFILE_FILE, CORE_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV,
                     CATALOG_FILE, README_FILE} == set(included)
    folder = tmp_path / "unzipped"
    zipfile.ZipFile(io.BytesIO(data)).extractall(folder)
    g = _resolve(folder, alignment_iri(PROFILE_IRI))
    assert {str(s) for s in g.subjects(RDF.type, OWL.Ontology)} == {
        alignment_iri(PROFILE_IRI), PROFILE_IRI, CORE_IRI}
    assert (URIRef(ONTO + "Consent"), RDF.type, OWL.Class) in g          # from the core
    readme = (folder / README_FILE).read_text(encoding="utf-8")
    assert ALIGNMENT_FILE in readme and "no se importa" in readme
    assert BUNDLE_FILE == "ontopriv-dpv-alignment.zip"


def test_missing_core_and_profile_are_reported(tmp_path):
    data, included, missing = build_bundle(_out(tmp_path, with_modules=False))
    assert missing == [PROFILE_FILE, CORE_FILE]
    zf = zipfile.ZipFile(io.BytesIO(data))
    root = ET.fromstring(zf.read(CATALOG_FILE))
    assert [u.get("uri") for u in root.iter(f"{CAT}uri")] == [ALIGNMENT_FILE]
    assert "py -m src.core.emit" in zf.read(README_FILE).decode("utf-8")


def test_real_bundle_resolves_every_mapped_entity(tmp_path):
    out = ROOT / "data/output"
    if not all((out / f).exists() for f in (ALIGNMENT_FILE, PROFILE_FILE, CORE_FILE)):
        pytest.skip("Faltan el grafo alineado o el nucleo/perfil en data/output")
    data, _, missing = build_bundle(out)
    assert missing == []
    folder = tmp_path / "real"
    zipfile.ZipFile(io.BytesIO(data)).extractall(folder)
    g = _resolve(folder, alignment_iri(PROFILE_IRI))
    mapped = {s for p in (SKOS.exactMatch, SKOS.closeMatch, SKOS.broadMatch, SKOS.narrowMatch,
                          SKOS.relatedMatch) for s in g.subjects(p, None)}
    declared = {s for s, p, o in g if p == RDF.type and o != OWL.Ontology}
    assert mapped and mapped <= declared          # no correspondence points to a missing entity