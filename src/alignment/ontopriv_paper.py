"""The base against the alignment the authors of OntoPriv proposed in their extended article
(S11-T07): its Table 10 has 14 correspondences with the DPV 2.0, written as DL axioms. Each
row says whether the base agrees, against the DPV 2.3 the system uses."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF

from src.publish.base_doc import BASE_NAMESPACES, DPV, dpv_links, load_modules

REFERENCE = "data/reference/ontopriv-paper-dpv.yaml"
RESULT_JSON = "ontopriv-paper-comparison.json"
RESULT_MD = "ontopriv-paper-comparison.md"
# An axiom of the article as a correspondence of the base: X ≡ Y is skos:exactMatch, and
# X ⊑ Y (Y is broader) is skos:broadMatch.
AXIOMS = {"≡": "exactMatch", "⊑": "broadMatch"}
WORDS = {"≡": "es igual a", "⊑": "es un tipo de"}   # no ≡ or ⊑ in the console (Windows)
SAME = "coincide"
OTHER_RELATION = "mismo término, otra relación"
OTHER_TERM = "otro término"
NO_MAPPING = "sin correspondencia en la base"
NEW_CONCEPT = "concepto que la base no tiene"
RESULTS = (SAME, OTHER_RELATION, OTHER_TERM, NO_MAPPING, NEW_CONCEPT)
_AXIOM_RE = re.compile(r"\s*(.+?)\s*([≡⊑])\s*dpv:(\w+)\s*")


@dataclass(frozen=True)
class PaperRow:
    n: int
    ontopriv: str              # column "OntoPriv Term"
    dpv: str                   # column "DPV Term", without "dpv:"
    axiom: str                 # column "Formal Mapping (DL Axiom)", as printed
    subject: str               # of the axiom
    symbol: str                # ≡ or ⊑
    target: str                # DPV term of the axiom, without "dpv:"

    @property
    def coherent(self) -> bool:
        """The axiom is about the concept of the first column (not so in row 12)."""
        return self.subject == self.ontopriv

    @property
    def terms(self) -> list[str]:
        """The DPV terms of the row: the column and, when the axiom names its parent, it too."""
        return list(dict.fromkeys((self.dpv, self.target)))


def load_reference(path: str | Path) -> tuple[dict, list[PaperRow]]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rows = []
    for n, raw in enumerate(data["rows"], 1):
        found = _AXIOM_RE.fullmatch(raw["axiom"])
        if not found:
            raise ValueError(f"fila {n}: no se entiende el axioma «{raw['axiom']}»")
        rows.append(PaperRow(n, raw["ontopriv"], raw["dpv"].removeprefix("dpv:"), raw["axiom"],
                             *found.groups()))
    return data["source"], rows


def _key(name: str) -> str:
    return re.sub(r"[\s_]", "", name).lower()       # "Legal basis" = Legal_basis = LegalBasis


def concepts(g: Graph, name: str) -> list[URIRef]:
    """The classes of the base with the name of the article (case and separators aside)."""
    return sorted(s for s in g.subjects(RDF.type, OWL.Class)
                  if isinstance(s, URIRef) and str(s).startswith(BASE_NAMESPACES)
                  and _key(str(s).rsplit("#", 1)[-1]) == _key(name))


def dpv_names(vocab: Graph) -> set[str]:
    return {str(s)[len(DPV):] for s in vocab.subjects(RDF.type, None)
            if isinstance(s, URIRef) and str(s).startswith(DPV)}


def in_dpv(name: str, names: set[str]) -> str | None:
    """The term of the DPV 2.3: the same name or its British spelling (Anonymization)."""
    if name in names:
        return name
    british = [n for n in names if n.lower() == name.lower().replace("z", "s")]
    return british[0] if len(british) == 1 else None


def _in_23(term: str, names: set[str]) -> dict:
    found = in_dpv(term, names)
    return {"term": term, "exists": found == term,
            "spelling": found if found not in (None, term) else None}


def compare_row(row: PaperRow, g: Graph, links: dict, names: set[str]) -> dict:
    found = concepts(g, row.ontopriv) or concepts(g, row.subject)
    wanted = set(row.terms) | {t for t in (in_dpv(x, names) for x in row.terms) if t}
    base = sorted({(rel, str(o)[len(DPV):]) for c in found for rel, o in links.get(c, [])})
    same = [rel for rel, term in base if term in wanted]
    if not found:
        result = NEW_CONCEPT
    elif not base:
        result = NO_MAPPING
    elif not same:
        result = OTHER_TERM
    else:
        result = SAME if AXIOMS[row.symbol] in same else OTHER_RELATION
    return {"n": row.n, "ontopriv": row.ontopriv, "dpv": row.dpv, "axiom": row.axiom,
            "symbol": row.symbol, "target": row.target, "coherent": row.coherent,
            "concepts": [str(c) for c in found],
            "base": [{"relation": rel, "dpv": term} for rel, term in base],
            "dpv23": [_in_23(t, names) for t in row.terms], "result": result}


def compare(rows: list[PaperRow], g: Graph, vocab: Graph) -> list[dict]:
    links, names = dpv_links(g), dpv_names(vocab)
    return [compare_row(r, g, links, names) for r in rows]


def summarize(found: list[dict]) -> dict:
    counts = Counter(r["result"] for r in found)
    return {"rows": len(found), "in_base": len(found) - counts[NEW_CONCEPT],
            "by_result": {k: counts.get(k, 0) for k in RESULTS},
            "same_term": counts[SAME] + counts[OTHER_RELATION],
            "missing_in_dpv23": [r["n"] for r in found
                                 if not all(t["exists"] for t in r["dpv23"])],
            "not_coherent": [r["n"] for r in found if not r["coherent"]]}


# --------------------------------------------------------------------------- output

def _base_text(row: dict) -> str:
    return ", ".join(f"{m['relation']} dpv:{m['dpv']}" for m in row["base"]) or "—"


def _dpv23_text(row: dict) -> str:
    missing = [t for t in row["dpv23"] if not t["exists"]]
    return "; ".join(f"no dpv:{t['term']}" + (f" (se escribe dpv:{t['spelling']})"
                                              if t["spelling"] else "")
                     for t in missing) or "sí"


def _paper_text(row: dict) -> str:
    """The axiom, with the term of the column when the axiom names its parent instead."""
    term = f" (término: dpv:{row['dpv']})" if row["dpv"] != row["target"] else ""
    other = (f" (la fila es de «{row['ontopriv']}»)" if not row["coherent"] else "")
    return f"{row['axiom']}{term}{other}"


def markdown(source: dict, found: list[dict], summary: dict) -> str:
    lines = ["# La base frente a la Tabla 10 del artículo de OntoPriv",
             "",
             f"{source['authors']}, «{source['title']}» ({source['year']}), {source['where']}: "
             f"{summary['rows']} correspondencias con el DPV {source['dpv_version']}. La base "
             "usa el DPV 2.3. Un axioma ≡ equivale a skos:exactMatch y ⊑ a skos:broadMatch.",
             "",
             "| # | Artículo (DPV 2.0) | Base (DPV 2.3) | ¿Está en el DPV 2.3? | Resultado |",
             "|---|---|---|---|---|"]
    for r in found:
        lines.append(f"| {r['n']} | {_paper_text(r)} | {_base_text(r)} | {_dpv23_text(r)} | "
                     f"{r['result']} |")
    return "\n".join(lines + ["", _totals(summary)]) + "\n"


def _totals(s: dict) -> str:
    b = s["by_result"]
    return (f"De las {s['rows']} filas, {s['in_base']} son conceptos que están en la base "
            f"(los otros {b[NEW_CONCEPT]} los propone el artículo y la base no los tiene). De "
            f"esas {s['in_base']}: coinciden {b[SAME]}, mismo término con otra relación "
            f"{b[OTHER_RELATION]}, otro término {b[OTHER_TERM]} y sin correspondencia en la base "
            f"{b[NO_MAPPING]}. Usan el mismo término del DPV en {s['same_term']} de "
            f"{s['in_base']}.")


def render_console(source: dict, found: list[dict], summary: dict, out: Path) -> str:
    lines = ["Comparación con la Tabla 10 del artículo de OntoPriv (S11-T07)",
             f"  Fuente: {source['authors']} ({source['year']}), {source['where']}; "
             f"DPV {source['dpv_version']} (la base usa el 2.3)"]
    for r in found:
        said = _paper_text(r).replace(r["symbol"], WORDS[r["symbol"]])
        lines.append(f"  {r['n']:>2}. {said} | base: {_base_text(r)} | {r['result']}")
    missing = [r for r in found if r["n"] in summary["missing_in_dpv23"]]
    lines += ["  " + _totals(summary),
              "  Términos del artículo que no están en el DPV 2.3: "
              + ("; ".join(f"fila {r['n']}, {_dpv23_text(r)}" for r in missing) or "—"),
              "  Filas cuyo axioma es de otro concepto que el de su primera columna: "
              f"{', '.join(map(str, summary['not_coherent'])) or '—'}",
              f"  Guardado en: {out / RESULT_MD} y {RESULT_JSON}"]
    return "\n".join(lines)


def run(folder: Path, vocab_path: Path, reference: Path, out: Path) -> tuple[dict, list, dict]:
    source, rows = load_reference(reference)
    g = Graph()
    for module in load_modules(folder):
        g += module
    found = compare(rows, g, Graph().parse(vocab_path, format="turtle"))
    summary = summarize(found)
    out.mkdir(parents=True, exist_ok=True)
    (out / RESULT_JSON).write_text(json.dumps(
        {"source": source, "summary": summary, "rows": found}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out / RESULT_MD).write_text(markdown(source, found, summary), encoding="utf-8")
    return source, found, summary


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import base_paths
    from src.config import ROOT, load_config
    folder = base_paths()["candidates"].parent
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.ontopriv_paper",
        description="Compara la base con las 14 correspondencias que las autoras de OntoPriv "
                    "propusieron en su artículo extendido (Tabla 10).")
    parser.add_argument("--salida", type=Path, default=folder,
                        help="carpeta donde se escriben la tabla (.md) y el detalle (.json)")
    args = parser.parse_args(argv)
    vocab = ROOT / load_config().get("inputs", {}).get("dpv", "vocab/dpv.ttl")
    try:
        source, found, summary = run(folder, vocab, ROOT / REFERENCE, args.salida)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(source, found, summary, args.salida))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
