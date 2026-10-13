"""S6-T03: generic packaging of modules in one format with a Protégé catalog."""
from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from rdflib import Graph
from rdflib.compare import isomorphic

from src.transform.convert import load_graph, serialize
from src.transform.package import (CATALOG_FILE, README_FILE, Module, module_bytes,
                                   module_file, package_modules, write_catalog)
from tests.test_transform_convert import FIXTURE_TTL

CAT = "{urn:oasis:names:tc:entity:xmlns:xml:catalog}"


def _source(folder: Path, name: str) -> Path:
    g = Graph()
    g.parse(data=FIXTURE_TTL, format="turtle")
    path = folder / name
    path.write_bytes(serialize(g, {".rdf": "rdfxml", ".ttl": "turtle"}[path.suffix]))
    return path


def _catalog(data: bytes) -> dict[str, str]:
    root = ET.fromstring(data)
    return {u.get("name"): u.get("uri") for u in root.iter(f"{CAT}uri")}


def test_module_file_uses_the_format_extension(tmp_path):
    m = Module("http://e/a", tmp_path / "a.rdf", "a")
    assert module_file(m, "turtle") == "a.ttl"
    assert module_file(m, "jsonld") == "a.jsonld"
    assert module_file(m, "owl") == "a.owl"


def test_module_bytes_copies_same_format_and_converts_otherwise(tmp_path):
    src = _source(tmp_path, "a.ttl")
    m = Module("http://e/a", src, "a")
    assert module_bytes(m, "turtle") == src.read_bytes()          # copied as is
    converted = Graph().parse(data=module_bytes(m, "rdfxml"), format="xml")
    assert isomorphic(converted, load_graph(src))


def test_package_skips_missing_modules_and_files(tmp_path):
    a = _source(tmp_path, "a.rdf")
    modules = [Module("http://e/missing", tmp_path / "nope.rdf", "nope"),
               Module("http://e/a", a, "a")]
    extras = {"notes.txt": b"hola", "absent.csv": tmp_path / "absent.csv"}
    package = package_modules(modules, "turtle", extras=extras,
                              readme=lambda p: f"abre {p.main_file}; faltan {p.missing}")
    assert package.missing == ["nope.rdf"]
    assert package.main_file == "a.ttl"                           # first module present
    assert package.included == ["a.ttl", "notes.txt", CATALOG_FILE, README_FILE]
    zf = zipfile.ZipFile(io.BytesIO(package.data))
    assert _catalog(zf.read(CATALOG_FILE)) == {"http://e/a": "a.ttl"}
    assert zf.read(README_FILE).decode("utf-8") == "abre a.ttl; faltan ['nope.rdf']"
    assert package.format.label == "Turtle"


def test_package_without_readme(tmp_path):
    a = _source(tmp_path, "a.rdf")
    package = package_modules([Module("http://e/a", a, "a")], "jsonld")
    assert README_FILE not in package.included
    assert package.included == ["a.jsonld", CATALOG_FILE]


def test_write_catalog_lists_only_existing_files(tmp_path):
    _source(tmp_path, "a.rdf")
    modules = [Module("http://e/a", tmp_path / "a.rdf", "a"),
               Module("http://e/b", tmp_path / "b.rdf", "b")]
    path = write_catalog(tmp_path, modules)
    assert path.name == CATALOG_FILE
    assert _catalog(path.read_bytes()) == {"http://e/a": "a.rdf"}
    assert _catalog(write_catalog(tmp_path, modules, "turtle").read_bytes()) == {}