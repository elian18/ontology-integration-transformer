"""Convert an ontology between RDF/XML, Turtle and JSON-LD without losing triples (S6-T01).

Stage 3 of the architecture (Transformation). The same rdflib graph is written in any of the
output formats and can be read back from any of the input formats:

    output formats: RDF/XML (.rdf), OWL (.owl, which is RDF/XML with the extension Protégé
                    expects), Turtle (.ttl) and JSON-LD (.jsonld)
    input formats:  the same plus .xml and .json; OWL/XML (.owx) and PDF stay input-only for
                    other stages (OWL/XML is converted once with Protégé, Sprint 1)

Measured on the real files before writing this module: plain rdflib loses or rewrites data in
two cases, so this module fixes both (each one has a regression test):

1. Turtle rewrites the lexical form of some numbers. OntoPriv has two values typed
   ``"15"^^xsd:decimal``; rdflib writes them with the Turtle shorthand ``15.0``, which reads
   back as ``"15.0"^^xsd:decimal`` (same value, different text, so not the same graph).
   Fix: a numeric or boolean literal uses the shorthand only when the shorthand keeps its exact
   text; otherwise it is written explicitly (``"15"^^xsd:decimal``).
2. JSON-LD drops the explicit ``rdf:type rdf:List`` of a list node. OntoPriv's SWRL rules have
   12 lists typed that way; rdflib turns them into ``@list`` and the 12 triples disappear.
   Fix: a list whose nodes carry ``rdf:type rdf:List`` is written node by node (``rdf:first`` /
   ``rdf:rest``), so the type survives. Lists without that type still use ``@list``.
   Also, JSON-LD is written with typed values (``{"@type": "xsd:integer", "@value": "15"}``),
   never native JSON numbers, so integers, doubles and ``xsd:string`` keep their exact form.

The JSON-LD ``@context`` only lists the prefixes the graph actually uses (rdflib binds about 30
common prefixes to every graph, most of them unused), so the file stays readable. DPV, SKOS and
Dublin Core terms that only appear as objects get their usual prefix (``dpv:Consent`` instead of
the full IRI); this adds prefixes only and never changes a triple.

``owl:imports`` points to ontology IRIs, not to files, so it does not change between formats;
mapping those IRIs to local files is the job of the Protégé catalog (S6-T03).

CLI: ``py -m src.transform.convert ARCHIVO [ARCHIVO ...] --to turtle|jsonld|rdfxml|owl|all``.
"""
from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass
from pathlib import Path

from rdflib import BNode, Graph, Literal, URIRef
from rdflib.namespace import RDF, XSD
from rdflib.plugins.serializers.jsonld import Converter as _JsonLdConverter
from rdflib.plugins.serializers.turtle import TurtleSerializer
from rdflib.plugins.shared.jsonld.context import Context


@dataclass(frozen=True)
class OutputFormat:
    key: str            # identifier used in code, the CLI and the web
    label: str          # name shown to the user (Spanish UI)
    extension: str      # file extension written
    media_type: str     # MIME type for downloads
    rdflib_name: str    # rdflib parser name used to read it back


OUTPUT_FORMATS: dict[str, OutputFormat] = {
    "rdfxml": OutputFormat("rdfxml", "RDF/XML", ".rdf", "application/rdf+xml", "xml"),
    "owl": OutputFormat("owl", "OWL (RDF/XML)", ".owl", "application/rdf+xml", "xml"),
    "turtle": OutputFormat("turtle", "Turtle", ".ttl", "text/turtle", "turtle"),
    "jsonld": OutputFormat("jsonld", "JSON-LD", ".jsonld", "application/ld+json", "json-ld"),
}

_ALIASES = {"rdf": "rdfxml", "rdf/xml": "rdfxml", "xml": "rdfxml", "ttl": "turtle",
            "json-ld": "jsonld", "json": "jsonld"}

# Input suffix -> rdflib parser (same table as src/ingest/ontology_loader.py).
INPUT_SUFFIXES = {".rdf": "xml", ".owl": "xml", ".xml": "xml", ".ttl": "turtle",
                  ".jsonld": "json-ld", ".json": "json-ld"}
_INPUT_LABELS = {"xml": "RDF/XML", "turtle": "Turtle", "json-ld": "JSON-LD"}

# Vocabularies whose IRIs appear as plain objects (RDF/XML does not declare a prefix for them,
# so after reading RDF/XML they have none). Bound before writing, only when used and unbound.
KNOWN_PREFIXES = {"dpv": "https://w3id.org/dpv#", "skos": "http://www.w3.org/2004/02/skos/core#",
                  "dct": "http://purl.org/dc/terms/"}

# Datatypes that Turtle may write as a bare number/boolean (the shorthand that can rewrite text).
_SHORTHAND_TYPES = {XSD.integer, XSD.decimal, XSD.double, XSD.boolean}
_XML_NAMESPACE = "http://www.w3.org/XML/1998/namespace"


class ConversionError(ValueError):
    """A file cannot be read or written in the requested format (message in Spanish)."""


@dataclass
class ConversionResult:
    source: Path
    target: Path
    source_label: str
    target_label: str
    triples_in: int
    triples_out: int        # triples read back from the written file

    @property
    def complete(self) -> bool:
        return self.triples_in == self.triples_out


# --------------------------------------------------------------------------- formats

def output_format(key: str) -> OutputFormat:
    """The output format for a key or a common alias ('ttl', 'json-ld', 'rdf/xml'...)."""
    norm = str(key or "").strip().lower()
    norm = _ALIASES.get(norm, norm)
    if norm not in OUTPUT_FORMATS:
        valid = ", ".join(OUTPUT_FORMATS)
        raise ConversionError(f"Formato de salida no soportado: '{key}'. Usa uno de: {valid}.")
    return OUTPUT_FORMATS[norm]


def input_format(path: str | Path) -> str:
    """rdflib parser name for an input file, from its extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".owx":
        raise ConversionError(
            "OWL/XML (.owx) es solo de entrada: conviértelo una vez a RDF/XML con Protégé "
            "(File > Save as > RDF/XML) y usa ese archivo.")
    if suffix not in INPUT_SUFFIXES:
        raise ConversionError(f"Formato de ontología no soportado: '{suffix or path}'.")
    return INPUT_SUFFIXES[suffix]


def load_graph(path: str | Path) -> Graph:
    """Parse an ontology file in any accepted input format."""
    path = Path(path)
    fmt = input_format(path)            # format first: a .owx gets the Protégé hint
    if not path.exists():
        raise ConversionError(f"No existe el archivo: {path}")
    graph = Graph()
    try:
        graph.parse(str(path), format=fmt)
    except Exception as exc:  # rdflib raises many parser-specific errors
        raise ConversionError(
            f"No se pudo leer {path.name} como {_INPUT_LABELS[fmt]}: {exc}") from exc
    return graph


def parse_bytes(data: bytes, fmt: str | OutputFormat) -> Graph:
    """Read back serialized bytes (used by the round-trip checks)."""
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    graph = Graph()
    graph.parse(data=data, format=spec.rdflib_name)
    return graph


# --------------------------------------------------------------------------- prefixes

def _iris(graph: Graph) -> set[str]:
    found: set[str] = set()
    for triple in graph:
        for node in triple:
            if isinstance(node, URIRef):
                found.add(str(node))
            elif isinstance(node, Literal) and node.datatype is not None:
                found.add(str(node.datatype))
    return found


def used_prefixes(graph: Graph) -> dict[str, str]:
    """Bound prefixes whose namespace is used by at least one IRI of the graph.

    Skips the empty prefix and ``xml``, and namespaces that do not end in '#' or '/' (JSON-LD
    1.1 only treats those as prefixes when expanding a compact IRI)."""
    iris = _iris(graph)
    result: dict[str, str] = {}
    for prefix, namespace in graph.namespaces():
        ns = str(namespace)
        if not prefix or prefix == "xml" or ns == _XML_NAMESPACE:
            continue
        if not ns.endswith(("#", "/")):
            continue
        if any(iri.startswith(ns) and len(iri) > len(ns) for iri in iris):
            result[str(prefix)] = ns
    return dict(sorted(result.items()))


def bind_known_prefixes(graph: Graph) -> list[str]:
    """Bind KNOWN_PREFIXES that the graph uses but has no prefix for; returns the ones added."""
    bound = {str(ns) for _, ns in graph.namespaces()}
    taken = {str(p) for p, _ in graph.namespaces()}
    iris = None
    added = []
    for prefix, ns in KNOWN_PREFIXES.items():
        if ns in bound or prefix in taken:
            continue
        iris = iris if iris is not None else _iris(graph)
        if any(iri.startswith(ns) for iri in iris):
            graph.bind(prefix, ns)
            added.append(prefix)
    return added


def jsonld_context(graph: Graph) -> dict[str, str]:
    """The JSON-LD ``@context``: only the prefixes the graph uses."""
    return used_prefixes(graph)


# --------------------------------------------------------------------------- serializers

class _ExactTurtleSerializer(TurtleSerializer):
    """Turtle that never rewrites the text of a number or boolean (fix 1, see module doc)."""

    def label(self, node, position):
        if isinstance(node, Literal) and node.datatype in _SHORTHAND_TYPES:
            shorthand = node._literal_n3(use_plain=True)
            if shorthand != str(node):
                return node._literal_n3(
                    use_plain=False,
                    qname_callback=lambda dt: self.get_pname(dt, False))
        return super().label(node, position)


class _LosslessJsonLdConverter(_JsonLdConverter):
    """JSON-LD converter that keeps typed lists and exact literal text (fix 2)."""

    def __init__(self, context: Context):
        super().__init__(context, use_native_types=False, use_rdf_type=False)
        self.use_native_types = False      # rdflib forces native JSON values with a context

    def to_collection(self, graph: Graph, l_):
        node, seen = l_, set()
        while isinstance(node, BNode) and node not in seen:
            seen.add(node)
            if (node, RDF.type, RDF.List) in graph:
                return None                # written node by node, so the type survives
            node = graph.value(node, RDF.rest)
        return super().to_collection(graph, l_)


def _turtle(graph: Graph) -> bytes:
    buf = io.BytesIO()
    _ExactTurtleSerializer(graph).serialize(buf, encoding="utf-8")
    return buf.getvalue()


def _jsonld(graph: Graph) -> bytes:
    context_data = jsonld_context(graph)
    context = Context(context_data)
    result = _LosslessJsonLdConverter(context).convert(graph)
    if context.active:
        if isinstance(result, list):
            result = {"@graph": result}
        result["@context"] = context_data
    text = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    return text.encode("utf-8")


def serialize(graph: Graph, fmt: str | OutputFormat) -> bytes:
    """The graph written in the given output format, as UTF-8 bytes."""
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    bind_known_prefixes(graph)          # only adds prefixes; never changes a triple
    try:
        if spec.rdflib_name == "turtle":
            return _turtle(graph)
        if spec.rdflib_name == "json-ld":
            return _jsonld(graph)
        return graph.serialize(format="xml", encoding="utf-8")
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"No se pudo escribir en {spec.label}: {exc}") from exc


# --------------------------------------------------------------------------- files

def target_path(source: str | Path, fmt: str | OutputFormat,
                out_dir: str | Path | None = None, stem: str | None = None) -> Path:
    """Where a conversion is written: same name with the new extension (default: same folder)."""
    source = Path(source)
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    folder = Path(out_dir) if out_dir is not None else source.parent
    return folder / f"{stem or source.stem}{spec.extension}"


def convert_file(source: str | Path, fmt: str | OutputFormat,
                 out_dir: str | Path | None = None, stem: str | None = None,
                 graph: Graph | None = None) -> ConversionResult:
    """Convert one file and read the result back to count its triples.

    Refuses to overwrite the source file itself. ``graph`` may be given when the source was
    already parsed (avoids parsing it again for each format)."""
    source = Path(source)
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    target = target_path(source, spec, out_dir, stem)
    if target.resolve() == source.resolve():
        raise ConversionError(f"{source.name} ya está en {spec.label}: no se sobrescribe.")
    if graph is None:
        graph = load_graph(source)
    data = serialize(graph, spec)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    back = parse_bytes(data, spec)
    return ConversionResult(
        source=source, target=target,
        source_label=_INPUT_LABELS[input_format(source)], target_label=spec.label,
        triples_in=len(graph), triples_out=len(back))


def formats_for(key: str, source: str | Path) -> list[OutputFormat]:
    """Formats requested on the CLI; 'all' means every output format except the source's own
    extension (a .rdf source is not rewritten as .rdf)."""
    if str(key).strip().lower() == "all":
        suffix = Path(source).suffix.lower()
        return [f for f in OUTPUT_FORMATS.values() if f.extension != suffix]
    return [output_format(key)]


# --------------------------------------------------------------------------- console

def _size(path: Path) -> str:
    kb = path.stat().st_size / 1024
    return f"{kb:,.0f} KB".replace(",", ".")


def render_console(results: list[ConversionResult], skipped: list[str] | None = None) -> str:
    lines = ["Conversión de formatos (S6-T01)"]
    current = None
    for r in results:
        if r.source != current:
            current = r.source
            lines.append(f"  {r.source.name}  ({r.source_label}, {r.triples_in} tripletas)")
        state = "completo" if r.complete else f"FALTAN {r.triples_in - r.triples_out} tripletas"
        lines.append(f"    -> {r.target.name:<38} {r.target_label:<14} "
                     f"{r.triples_out:>6} tripletas  {_size(r.target):>8}  {state}")
    for note in skipped or []:
        lines.append(f"  (se omite) {note}")
    incomplete = [r for r in results if not r.complete]
    lines.append("")
    if incomplete:
        lines.append(f"Resultado: {len(incomplete)} conversiones con tripletas perdidas.")
    else:
        lines.append(f"Resultado: {len(results)} archivos escritos, ninguna tripleta perdida.")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="py -m src.transform.convert",
        description="Convierte ontologías entre RDF/XML, Turtle y JSON-LD (S6-T01).")
    parser.add_argument("files", nargs="+", help="archivos .rdf, .owl, .ttl o .jsonld")
    parser.add_argument("--to", default="all",
                        help="turtle, jsonld, rdfxml, owl o all (todos menos el propio)")
    parser.add_argument("--out", default=None, help="carpeta de salida (por defecto, la misma)")
    args = parser.parse_args(argv)

    results: list[ConversionResult] = []
    skipped: list[str] = []
    try:
        for name in args.files:
            source = Path(name)
            graph = load_graph(source)
            for spec in formats_for(args.to, source):
                target = target_path(source, spec, args.out)
                if target.resolve() == source.resolve():
                    skipped.append(f"{source.name}: ya está en {spec.label}")
                    continue
                results.append(convert_file(source, spec, args.out, graph=graph))
    except ConversionError as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(results, skipped))
    return 0 if all(r.complete for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(_run())