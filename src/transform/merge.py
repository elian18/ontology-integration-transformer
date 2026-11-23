from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from src.transform.convert import input_format
from src.transform.package import CATALOG_FILE


def merge(graphs: Iterable[Graph], keep: str) -> Graph:
    """The union of ``graphs``; ``keep`` (the person's ontology) stays the only owl:Ontology:
    the headers of the other modules and every owl:imports between modules are removed."""
    out = Graph()
    for g in graphs:
        for prefix, ns in g.namespaces():
            out.bind(prefix, ns, override=False)
        out += g
    modules = {s for s in out.subjects(RDF.type, OWL.Ontology) if isinstance(s, URIRef)}
    for module in modules - {URIRef(keep)}:
        out.remove((module, None, None))
    for module in modules:
        out.remove((None, OWL.imports, module))
    return out


def removed(graphs: Iterable[Graph], keep: str) -> set:
    """What ``merge`` leaves out of the union (for the tests and the checks)."""
    graphs = list(graphs)
    union = set().union(*(set(g) for g in graphs))
    return union - set(merge(graphs, keep))


def package_graphs(data: bytes, skip: str | None = None) -> list[Graph]:
    """The modules of a package (.zip): the files its Protégé catalog lists, each parsed; a
    file whose stem ends with ``skip`` (the provenance, which nothing imports) is left out."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        files = re.findall(r'uri="([^"]+)"', z.read(CATALOG_FILE).decode("utf-8"))
        return [Graph().parse(data=z.read(name), format=input_format(name))
                for name in sorted(files) if not (skip and Path(name).stem.endswith(skip))]
