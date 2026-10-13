"""S5-T07: the zip for Protégé carries the three modules and a catalog that resolves the imports
alignment -> profile -> core by IRI. S6-T03: the same in every output format (RDF/XML, OWL,
Turtle, JSON-LD), plus the fix of the catalog left in data/output."""
import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pytest
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, SKOS

from src.core.emit import CORE_IRI, PROFILE_IRI, CORE_FILE, PROFILE_FILE
from src.alignment.write import ALIGNMENT_FILE, ALIGNMENT_MANIFEST, ALIGNMENT_CSV, alignment_iri
from rdflib.compare import isomorphic

from src.alignment import bundle
from src.alignment.bundle import (build_bundle, build_package, bundle_name, catalog_xml,
                                  write_base_catalog, CATALOG_FILE, README_FILE, BUNDLE_FILE,
                                  PROTEGE_JSONLD_NOTE)
from src.transform.convert import load_graph

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
        part = load_graph(folder / files[iri])           # KeyError = unresolved; any format
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

# ---------------------------------------------------------------- S6-T03: every format
FORMATS = {"rdfxml": ".rdf", "owl": ".owl", "turtle": ".ttl", "jsonld": ".jsonld"}


def test_bundle_name_per_format():
    assert bundle_name() == bundle_name("rdfxml") == BUNDLE_FILE
    assert bundle_name("turtle") == "ontopriv-dpv-alignment-turtle.zip"
    assert bundle_name("json-ld") == "ontopriv-dpv-alignment-jsonld.zip"
    assert bundle_name("owl") == "ontopriv-dpv-alignment-owl.zip"


@pytest.mark.parametrize("fmt,ext", FORMATS.items())
def test_each_format_resolves_the_import_chain(tmp_path, fmt, ext):
    out = _out(tmp_path)
    package = build_package(out, fmt=fmt)
    stems = [Path(f).stem for f in (ALIGNMENT_FILE, PROFILE_FILE, CORE_FILE)]
    assert package.main_file == stems[0] + ext
    assert [f for f in package.included if f.endswith(ext)] == [s + ext for s in stems]
    folder = tmp_path / f"unzipped-{fmt}"
    zipfile.ZipFile(io.BytesIO(package.data)).extractall(folder)
    root = ET.parse(folder / CATALOG_FILE).getroot()
    assert all(u.get("uri").endswith(ext) for u in root.iter(f"{CAT}uri"))
    g = _resolve(folder, alignment_iri(PROFILE_IRI))
    assert {str(s) for s in g.subjects(RDF.type, OWL.Ontology)} == {
        alignment_iri(PROFILE_IRI), PROFILE_IRI, CORE_IRI}
    assert (URIRef(ONTO + "Consent"), SKOS.exactMatch, URIRef("https://w3id.org/dpv#Consent")) in g
    readme = (folder / README_FILE).read_text(encoding="utf-8")
    if fmt == "jsonld":                       # Protégé does not open JSON-LD (S6-T03)
        assert f"Archivo principal: {stems[0]}{ext}." in readme
    else:
        assert f"Abre {stems[0]}{ext} en Protégé" in readme


def test_every_format_carries_the_same_graph(tmp_path):
    out = _out(tmp_path)
    graphs = {}
    for fmt in FORMATS:
        folder = tmp_path / f"u-{fmt}"
        zipfile.ZipFile(io.BytesIO(build_bundle(out, fmt=fmt)[0])).extractall(folder)
        graphs[fmt] = _resolve(folder, alignment_iri(PROFILE_IRI))
    for fmt in ("owl", "turtle", "jsonld"):
        assert isomorphic(graphs["rdfxml"], graphs[fmt]), fmt


def test_rdfxml_package_copies_the_files_unchanged(tmp_path):
    out = _out(tmp_path)
    zf = zipfile.ZipFile(io.BytesIO(build_bundle(out)[0]))
    for name in (ALIGNMENT_FILE, PROFILE_FILE, CORE_FILE):
        assert zf.read(name) == (out / name).read_bytes()


def test_write_base_catalog_fixes_the_protege_catalog(tmp_path):
    out = _out(tmp_path)
    (out / CATALOG_FILE).write_text(        # the wrong one Protégé left (core IRI -> profile)
        '<?xml version="1.0"?><catalog xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">'
        f'<uri name="{CORE_IRI}" uri="{PROFILE_FILE}"/></catalog>', encoding="utf-8")
    path = write_base_catalog(out)
    root = ET.parse(path).getroot()
    assert {u.get("name"): u.get("uri") for u in root.iter(f"{CAT}uri")} == {
        alignment_iri(PROFILE_IRI): ALIGNMENT_FILE, PROFILE_IRI: PROFILE_FILE,
        CORE_IRI: CORE_FILE}


def test_cli_writes_every_package_and_the_catalog(tmp_path, monkeypatch, capsys):
    out = _out(tmp_path)
    monkeypatch.setattr("src.config.load_config", lambda: {"outputs": {"dir": str(out)}})
    assert bundle._run([]) == 0
    text = capsys.readouterr().out
    for fmt in FORMATS:
        assert (out / bundle_name(fmt)).exists()
    assert "Paquetes para Protégé (S6-T03)" in text and "Catálogo corregido" in text
    assert bundle._run(["--formato", "owx"]) == 1
    assert "Error:" in capsys.readouterr().out


@pytest.mark.parametrize("fmt", ["turtle", "jsonld"])
def test_real_bundle_in_other_formats_resolves(tmp_path, fmt):
    out = ROOT / "data/output"
    if not all((out / f).exists() for f in (ALIGNMENT_FILE, PROFILE_FILE, CORE_FILE)):
        pytest.skip("Faltan el grafo alineado o el nucleo/perfil en data/output")
    folder = tmp_path / fmt
    zipfile.ZipFile(io.BytesIO(build_bundle(out, fmt=fmt)[0])).extractall(folder)
    g = _resolve(folder, alignment_iri(PROFILE_IRI))
    expected = Graph()
    for name in (ALIGNMENT_FILE, PROFILE_FILE, CORE_FILE):
        expected += load_graph(out / name)
    assert isomorphic(g, expected)


def test_only_the_jsonld_readme_warns_about_protege(tmp_path):
    out = _out(tmp_path)
    for fmt in FORMATS:
        zf = zipfile.ZipFile(io.BytesIO(build_bundle(out, fmt=fmt)[0]))
        readme = zf.read(README_FILE).decode("utf-8")
        assert (PROTEGE_JSONLD_NOTE in readme) == (fmt == "jsonld"), fmt