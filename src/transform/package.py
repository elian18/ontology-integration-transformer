"""Package ontology modules in one output format with a Protégé catalog (S6-T03).

A module imports another by IRI (``owl:imports <http://.../ecuador-lopdp>``), not by file name,
so the imports do not change between RDF/XML, Turtle and JSON-LD. What changes is the file
that holds each IRI: ``profile-ecuador-lopdp.rdf`` in RDF/XML, ``profile-ecuador-lopdp.ttl`` in
Turtle. Protégé finds those files through an XML catalog (``catalog-v001.xml``) placed next to
them, so each package carries a catalog written for its own format.

This module is generic: the base package (alignment -> profile -> core, ``src.alignment.bundle``)
and the user packages of S6-T07 use the same code with different modules.

When a module is already stored in the requested format, its bytes are copied as they are (the
file on disk is the reference); otherwise it is converted with ``src.transform.convert``, which
keeps the same graph (S6-T01, checked on the real files in S6-T02).
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import quoteattr

from src.transform.convert import (OutputFormat, input_format, load_graph, output_format,
                                   serialize)

CATALOG_FILE = "catalog-v001.xml"
README_FILE = "LEEME.txt"


@dataclass(frozen=True)
class Module:
    iri: str        # ontology IRI (what owl:imports points to)
    source: Path    # file on disk, in any accepted input format
    stem: str       # file name without extension inside the package


@dataclass
class Package:
    data: bytes                     # the zip
    format: OutputFormat
    included: list[str] = field(default_factory=list)   # files inside the zip
    missing: list[str] = field(default_factory=list)    # module sources not found
    main_file: str | None = None    # the file to open in Protégé (first module present)


def catalog_xml(entries: dict[str, str]) -> str:
    """Protégé XML catalog: ontology IRI -> local file name."""
    lines = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
             '<catalog prefer="public" xmlns="urn:oasis:names:tc:entity:xmlns:xml:catalog">']
    for i, (iri, file_name) in enumerate(entries.items(), start=1):
        lines.append(f'    <uri id={quoteattr(f"modulo-{i}")} name={quoteattr(iri)} '
                     f'uri={quoteattr(file_name)}/>')
    lines.append("</catalog>")
    return "\n".join(lines) + "\n"


def module_file(module: Module, fmt: str | OutputFormat) -> str:
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    return f"{module.stem}{spec.extension}"


def module_bytes(module: Module, fmt: str | OutputFormat) -> bytes:
    """The module in ``fmt``: copied if the source already has that format, else converted."""
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    source = Path(module.source)
    if input_format(source) == spec.rdflib_name:
        return source.read_bytes()
    return serialize(load_graph(source), spec)


def package_modules(modules: list[Module], fmt: str | OutputFormat,
                    extras: dict[str, bytes | Path] | None = None,
                    readme: Callable[[Package], str] | None = None) -> Package:
    """Zip the modules in one format with their catalog, the extra files and a LEEME.txt.

    ``modules`` go in order; the first one present is the file to open. A module whose source
    does not exist is listed in ``missing`` (by its source name) and left out of the catalog.
    ``readme`` receives the package (included files, missing, main file) and returns its text."""
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    result = Package(data=b"", format=spec)
    entries: dict[str, str] = {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for module in modules:
            if not Path(module.source).exists():
                result.missing.append(Path(module.source).name)
                continue
            name = module_file(module, spec)
            zf.writestr(name, module_bytes(module, spec))
            entries[module.iri] = name
            result.included.append(name)
            result.main_file = result.main_file or name
        for name, content in (extras or {}).items():
            if isinstance(content, Path):
                if not content.exists():
                    continue
                content = content.read_bytes()
            zf.writestr(name, content)
            result.included.append(name)
        zf.writestr(CATALOG_FILE, catalog_xml(entries))
        result.included.append(CATALOG_FILE)
        if readme is not None:
            result.included.append(README_FILE)
            zf.writestr(README_FILE, readme(result))
    result.data = buf.getvalue()
    return result


def write_catalog(folder: str | Path, modules: list[Module],
                  fmt: str | OutputFormat = "rdfxml") -> Path:
    """Write ``catalog-v001.xml`` in ``folder`` mapping each module IRI to its file there
    (only modules whose file exists in that format). Replaces any previous catalog."""
    folder = Path(folder)
    entries = {m.iri: module_file(m, fmt) for m in modules
               if (folder / module_file(m, fmt)).exists()}
    path = folder / CATALOG_FILE
    path.write_text(catalog_xml(entries), encoding="utf-8")
    return path