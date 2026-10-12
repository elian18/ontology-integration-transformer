"""Round-trip check on the real files: convert, read back, compare (S6-T02).

For every file and every output format the check writes the graph with ``convert.serialize``,
reads it back and compares it with the original using graph isomorphism
(``rdflib.compare.isomorphic``), which also matches blank nodes (OWL restrictions, SWRL lists,
``owl:Axiom`` annotations), whose internal ids change on every parse. "Same graph" therefore
means: the same triples, with the same literal text, language tags and datatypes.

Two extra measurements make the result usable in the thesis:
- the chain RDF/XML -> Turtle -> JSON-LD -> RDF/XML, each step reading the previous output;
- the same round trip with plain rdflib (no S6-T01 fixes), so the report shows what was lost
  before (two rewritten decimals in Turtle, 12 dropped ``rdf:type rdf:List`` in JSON-LD on
  OntoPriv) and that the fixes remove it.

The ``owl`` output is not checked separately: it uses the RDF/XML serializer and produces the
same bytes as ``rdfxml`` (a test locks this).

Default files: OntoPriv (``inputs.ontology``), the three base modules in the outputs folder
(core, Ecuador profile, DPV alignment), ``data/input/sample.ttl`` if present (a Turtle source)
and the DPV (``inputs.dpv``) as a large external vocabulary. Missing files are reported, not
fatal. The report is written to ``<outputs>/roundtrip-report.json`` for the CHANGELOG.

CLI: ``py -m src.transform.roundtrip [ARCHIVO ...] [--sin-dpv]``.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rdflib import BNode, Graph
from rdflib.compare import graph_diff, isomorphic, to_isomorphic

from src.transform.convert import (ConversionError, OutputFormat, input_format, load_graph,
                                   output_format, parse_bytes, serialize)

CHECKED_FORMATS = ("rdfxml", "turtle", "jsonld")
CHAIN = ("rdfxml", "turtle", "jsonld", "rdfxml")
PLAIN_FORMATS = {"turtle": "turtle", "jsonld": "json-ld"}   # rdflib's own serializers
REPORT_FILE = "roundtrip-report.json"
SAMPLE_FILE = "data/input/sample.ttl"
_EXAMPLES = 3


@dataclass
class RoundTrip:
    format: str                  # output format key, or "chain"
    triples_before: int
    triples_after: int
    same_graph: bool
    missing: int = 0             # triples WITHOUT blank nodes only in the original
    extra: int = 0               # triples WITHOUT blank nodes only in the result
    examples: list[str] = field(default_factory=list)   # a few differing triples (no bnodes)
    seconds: float = 0.0


@dataclass
class FileReport:
    name: str
    path: str
    source_format: str
    triples: int
    blank_node_triples: int
    checks: list[RoundTrip] = field(default_factory=list)
    chain: RoundTrip | None = None
    plain: list[RoundTrip] = field(default_factory=list)   # plain rdflib, for comparison

    @property
    def ok(self) -> bool:
        return all(c.same_graph for c in self.checks) and (self.chain is None
                                                           or self.chain.same_graph)


# --------------------------------------------------------------------------- comparison

def _blank_node_triples(graph: Graph) -> int:
    return sum(1 for t in graph if any(isinstance(n, BNode) for n in t))


def compare(original: Graph, result: Graph, label: str, seconds: float = 0.0) -> RoundTrip:
    """Compare two graphs; on a difference, count missing/extra triples and show examples."""
    same = isomorphic(original, result)
    check = RoundTrip(format=label, triples_before=len(original), triples_after=len(result),
                      same_graph=same, seconds=round(seconds, 2))
    if not same:
        # Only triples without blank nodes are counted one by one: a blank-node difference
        # changes the canonical ids of every related blank node, so graph_diff would report
        # hundreds of "different" triples for a handful of real ones. Those show up as a
        # change in the triple count instead (see _state).
        _, only_original, only_result = graph_diff(to_isomorphic(original),
                                                   to_isomorphic(result))
        for prefix, graph in (("falta", only_original), ("sobra", only_result)):
            named = sorted(t for t in graph if not any(isinstance(n, BNode) for n in t))
            if prefix == "falta":
                check.missing = len(named)
            else:
                check.extra = len(named)
            for s, p, o in named[:_EXAMPLES]:
                check.examples.append(f"{prefix}: {s.n3()} {p.n3()} {o.n3()}")
    return check


def check_round_trip(graph: Graph, fmt: str | OutputFormat) -> RoundTrip:
    """Write the graph in ``fmt`` with the S6-T01 converter, read it back and compare."""
    spec = fmt if isinstance(fmt, OutputFormat) else output_format(fmt)
    start = time.perf_counter()
    back = parse_bytes(serialize(graph, spec), spec)
    return compare(graph, back, spec.key, time.perf_counter() - start)


def check_plain(graph: Graph, fmt: str) -> RoundTrip:
    """Same round trip with rdflib's own serializer (what S6-T01 had to fix)."""
    start = time.perf_counter()
    data = graph.serialize(format=PLAIN_FORMATS[fmt], encoding="utf-8")
    back = Graph()
    back.parse(data=data, format=PLAIN_FORMATS[fmt])
    return compare(graph, back, fmt, time.perf_counter() - start)


def check_chain(graph: Graph, chain: tuple[str, ...] = CHAIN) -> RoundTrip:
    """Convert step by step along ``chain`` (each step reads the previous output)."""
    start = time.perf_counter()
    current = graph
    for key in chain:
        current = parse_bytes(serialize(current, key), key)
    return compare(graph, current, "chain", time.perf_counter() - start)


def check_file(path: str | Path, formats: tuple[str, ...] = CHECKED_FORMATS,
               with_chain: bool = True, with_plain: bool = True) -> FileReport:
    path = Path(path)
    graph = load_graph(path)
    report = FileReport(name=path.name, path=str(path),
                        source_format={"xml": "RDF/XML", "turtle": "Turtle",
                                       "json-ld": "JSON-LD"}[input_format(path)],
                        triples=len(graph), blank_node_triples=_blank_node_triples(graph))
    report.checks = [check_round_trip(graph, f) for f in formats]
    if with_chain:
        report.chain = check_chain(graph)
    if with_plain:
        report.plain = [check_plain(graph, f) for f in PLAIN_FORMATS]
    return report


# --------------------------------------------------------------------------- default files

def default_files(cfg: dict | None = None, include_dpv: bool = True,
                  root: Path | None = None) -> list[Path]:
    """OntoPriv, the three base modules, sample.ttl and the DPV (paths from config.yaml)."""
    from src.config import ROOT, load_config
    from src.core.emit import CORE_FILE, PROFILE_FILE
    from src.alignment.write import ALIGNMENT_FILE

    cfg = cfg if cfg is not None else load_config()
    root = root or ROOT
    inputs = cfg.get("inputs", {})
    out_dir = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    files = [Path(inputs.get("ontology", "data/input/ontopriv.rdf")),
             out_dir / CORE_FILE, out_dir / PROFILE_FILE, out_dir / ALIGNMENT_FILE,
             Path(SAMPLE_FILE)]
    if include_dpv:
        files.append(Path(inputs.get("dpv", "vocab/dpv.ttl")))
    return [f if f.is_absolute() else root / f for f in files]


def run(files: list[Path], with_plain: bool = True) -> tuple[list[FileReport], list[str]]:
    """Check every existing file; return the reports and the files that were not found."""
    reports, missing = [], []
    for path in files:
        if not Path(path).exists():
            missing.append(str(path))
            continue
        reports.append(check_file(path, with_plain=with_plain))
    return reports, missing


# --------------------------------------------------------------------------- report

def summary(reports: list[FileReport]) -> dict:
    checks = [c for r in reports for c in r.checks]
    chains = [r.chain for r in reports if r.chain is not None]
    plain = [p for r in reports for p in r.plain]
    return {
        "files": len(reports),
        "round_trips": len(checks) + len(chains),
        "same_graph": sum(c.same_graph for c in checks + chains),
        "differences": sum(not c.same_graph for c in checks + chains),
        "triples_checked": sum(r.triples for r in reports),
        "plain_rdflib_failures": sum(not p.same_graph for p in plain),
        "plain_rdflib_checks": len(plain),
    }


def write_report(reports: list[FileReport], missing: list[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "formats": list(CHECKED_FORMATS),
        "chain": list(CHAIN),
        "summary": summary(reports),
        "files": [asdict(r) | {"ok": r.ok} for r in reports],
        "missing_files": missing,
        "notes": [
            "Mismo grafo = isomorfismo de grafos RDF (rdflib.compare), incluidos nodos en "
            "blanco, texto de los literales, idioma y tipo de dato.",
            "El formato 'owl' usa el serializador RDF/XML y produce los mismos bytes que "
            "'rdfxml'.",
            "'plain' = el mismo viaje de ida y vuelta con el serializador propio de rdflib, "
            "sin las correcciones de S6-T01.",
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


_IRI = re.compile(r"<[^>]*[#/]([^>#/]+)>")


def _short(example: str) -> str:
    """Console form of an example triple: IRIs reduced to their local name."""
    return _IRI.sub(r"\1", example)


def _state(check: RoundTrip) -> str:
    if check.same_graph:
        return "mismo grafo"
    parts = []
    if check.missing or check.extra:
        parts.append(f"faltan {check.missing}, sobran {check.extra}")
    delta = check.triples_after - check.triples_before
    if delta:
        parts.append(f"{abs(delta)} tripletas {'menos' if delta < 0 else 'más'}")
    if not parts:
        parts.append("difieren nodos en blanco")
    return "DIFERENTE (" + "; ".join(parts) + ")"


def render_console(reports: list[FileReport], missing: list[str],
                   report_path: Path | None = None) -> str:
    lines = ["Prueba de ida y vuelta (S6-T02)"]
    for r in reports:
        lines.append(f"  {r.name}  ({r.source_format}, {r.triples} tripletas, "
                     f"{r.blank_node_triples} con nodos en blanco)")
        for c in r.checks + ([r.chain] if r.chain else []):
            name = "cadena RDF/XML>Turtle>JSON-LD>RDF/XML" if c.format == "chain" else c.format
            lines.append(f"    {name:<38} {c.triples_after:>6} tripletas  {_state(c)}"
                         f"  ({c.seconds:.1f} s)")
            lines.extend(f"      {_short(e)}" for e in c.examples)
        for p in r.plain:
            if not p.same_graph:
                lines.append(f"    rdflib sin corrección, {p.format:<14} {_state(p)}")
                for side in ("falta", "sobra"):     # one example per side is enough here
                    first = next((e for e in p.examples if e.startswith(side)), None)
                    if first:
                        lines.append(f"      {_short(first)}")
    for m in missing:
        lines.append(f"  (no encontrado, se omite) {m}")
    s = summary(reports)
    lines.append("")
    lines.append(f"Resultado: {s['same_graph']} de {s['round_trips']} viajes de ida y vuelta "
                 f"dan el mismo grafo ({s['files']} archivo{'s' if s['files'] != 1 else ''}, "
                 f"{s['triples_checked']} tripletas).")
    lines.append(f"Sin las correcciones de S6-T01, rdflib falla en {s['plain_rdflib_failures']} "
                 f"de {s['plain_rdflib_checks']} viajes (Turtle y JSON-LD).")
    if report_path is not None:
        lines.append(f"Reporte: {report_path}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.config import ROOT, load_config

    parser = argparse.ArgumentParser(
        prog="py -m src.transform.roundtrip",
        description="Comprueba que convertir y volver a leer da el mismo grafo (S6-T02).")
    parser.add_argument("files", nargs="*", help="archivos a comprobar (por defecto, la base)")
    parser.add_argument("--sin-dpv", action="store_true", help="no comprobar el DPV")
    parser.add_argument("--sin-comparar", action="store_true",
                        help="no medir rdflib sin correcciones (más rápido)")
    args = parser.parse_args(argv)

    cfg = load_config()
    files = ([Path(f) for f in args.files] if args.files
             else default_files(cfg, include_dpv=not args.sin_dpv))
    try:
        reports, missing = run(files, with_plain=not args.sin_comparar)
    except ConversionError as exc:
        print(f"Error: {exc}")
        return 1
    out_dir = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    path = write_report(reports, missing, out_dir / REPORT_FILE)
    try:
        shown = path.relative_to(ROOT)
    except ValueError:
        shown = path
    print(render_console(reports, missing, shown))
    return 0 if all(r.ok for r in reports) else 2


if __name__ == "__main__":
    raise SystemExit(_run())