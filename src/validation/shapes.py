from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pyshacl
from rdflib import BNode, Graph
from rdflib.namespace import RDF, RDFS, SH

SHAPES_FILE = Path(__file__).with_name("shapes.ttl")
SHAPES_NS = "http://www.semanticweb.org/ontology-integration-transformer/shapes#"
SEVERITIES = {SH.Violation: "violación", SH.Warning: "advertencia", SH.Info: "aviso"}
VIOLATION, WARNING = SEVERITIES[SH.Violation], SEVERITIES[SH.Warning]
EXAMPLES = 3


@dataclass(frozen=True)
class Rule:
    id: str
    name: str
    description: str
    group: str
    group_order: int
    order: int
    severity: str


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    focus: str
    message: str
    path: str | None = None
    value: str | None = None


def load_shapes(path: str | Path = SHAPES_FILE) -> Graph:
    return Graph().parse(str(path), format="turtle")


def _id(node) -> str:
    return str(node).rsplit("#", 1)[-1]


def _text(node) -> str | None:
    if node is None:
        return None
    return f"_:{node}" if isinstance(node, BNode) else str(node)


def rules(shapes: Graph | None = None) -> list[Rule]:
    g = shapes if shapes is not None else load_shapes()
    found = []
    for shape in set(g.subjects(RDF.type, SH.NodeShape)):
        name = g.value(shape, RDFS.label)
        if name is None:                               # helper shapes used inside the rules
            continue
        group = g.value(shape, SH.group)
        found.append(Rule(id=_id(shape), name=str(name),
                          description=str(g.value(shape, RDFS.comment) or ""),
                          group=str(g.value(group, RDFS.label)),
                          group_order=int(g.value(group, SH.order)),
                          order=int(g.value(shape, SH.order)),
                          severity=SEVERITIES[g.value(shape, SH.severity, default=SH.Violation)]))
    return sorted(found, key=lambda r: (r.group_order, r.order))


def rule_of(shapes: Graph, source_shape) -> str | None:
    node = source_shape
    while isinstance(node, BNode):                     # a property shape: go up to its rule
        node = shapes.value(None, SH.property, node)
    return _id(node) if node is not None else None


def dpv_terms(path: str | Path) -> Graph:
    dpv, terms = Graph().parse(str(path), format="turtle"), Graph()
    for triple in dpv.triples((None, RDFS.isDefinedBy, None)):
        terms.add(triple)
    return terms


def check(data: Graph, shapes: Graph | None = None) -> list[Finding]:
    g = shapes if shapes is not None else load_shapes()
    _, report, _ = pyshacl.validate(data, shacl_graph=g, inference="none", advanced=True)
    findings = []
    for result in report.subjects(RDF.type, SH.ValidationResult):
        path = report.value(result, SH.resultPath)
        findings.append(Finding(
            rule=rule_of(g, report.value(result, SH.sourceShape)),
            severity=SEVERITIES[report.value(result, SH.resultSeverity)],
            focus=_text(report.value(result, SH.focusNode)),
            message=str(report.value(result, SH.resultMessage) or ""),
            path=None if isinstance(path, BNode) else _text(path),
            value=_text(report.value(result, SH.value))))
    return sorted(findings, key=lambda f: (f.rule, f.focus, f.message))


# --------------------------------------------------------------------------- console

def _local(node: str) -> str:
    if node.startswith("_:"):
        return "(nodo sin nombre)"
    return node.rstrip("#/").rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def counted(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def render_rules(catalog: list[Rule]) -> str:
    lines = [f"Reglas de forma (SHACL) de src/validation/{SHAPES_FILE.name}: "
             f"{counted(len(catalog), 'regla', 'reglas')}"]
    group = None
    for number, rule in enumerate(catalog, 1):
        if rule.group != group:
            group = rule.group
            lines.append(f"\n{group}")
        lines.append(f"  {number:>2}. {rule.name} ({rule.severity})")
        lines.append(f"      {rule.description}")
    return "\n".join(lines)


def render_findings(findings: list[Finding], catalog: list[Rule], files: list[Path],
                    triples: int) -> str:
    by_rule = Counter(f.rule for f in findings)
    severity = Counter(f.severity for f in findings)
    lines = [f"Reglas de forma sobre {counted(len(files), 'archivo', 'archivos')} "
             f"({triples} tripletas): "
             f"{counted(severity[VIOLATION], 'violación', 'violaciones')}, "
             f"{counted(severity[WARNING], 'advertencia', 'advertencias')}"]
    for rule in catalog:
        n = by_rule.get(rule.id, 0)
        lines.append(f"  [{'ok' if n == 0 else rule.severity}] {rule.name}: {n}")
        for f in [f for f in findings if f.rule == rule.id][:EXAMPLES]:
            lines.append(f"       {_local(f.focus)}: {f.message}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.config import load_config
    from src.transform.convert import ConversionError, load_graph
    parser = argparse.ArgumentParser(prog="py -m src.validation.shapes",
                                     description="Muestra las reglas de forma (SHACL) o las "
                                                 "corre sobre una ontología y sus módulos.")
    parser.add_argument("files", nargs="*", help="archivos de la ontología y sus módulos")
    args = parser.parse_args(argv)
    shapes = load_shapes()
    catalog = rules(shapes)
    if not args.files:
        print(render_rules(catalog))
        return 0
    files = [Path(f) for f in args.files]
    data = Graph()
    try:
        for f in files:
            data += load_graph(f)
    except ConversionError as exc:
        print(f"Error: {exc}")
        return 1
    triples = len(data)
    data += dpv_terms(load_config()["inputs"]["dpv"])
    print(render_findings(check(data, shapes), catalog, files, triples))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())