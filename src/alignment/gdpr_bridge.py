# The bridge of the base with the GDPR (S9-T09): the pairs OntoPriv <-> GDPRtEXT that reach the
# same DPV term with exactMatch or closeMatch on both sides, written as skos:closeMatch in a
# module of their own, each one marked "encadenado por el DPV" with the term it goes through.
# The alignment of GDPRtEXT it uses is fixed in data/reference, so the bridge is rebuilt without
# the AI and always gives the same pairs. The reviewing AI explains each pair in a CSV apart:
# nothing it says enters the ontology.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from src.ai.prompts import load_prompt
from src.validation import ai_auditor as au

PROMPT = load_prompt("bridge", 1)
BRIDGE_IRI = "http://www.semanticweb.org/profiles/ontopriv-gdpr-alignment"
BRIDGE = Namespace(BRIDGE_IRI + "#")
GDPRTEXT_IRI = "http://purl.org/adaptcentre/openscience/ontologies/GDPRtEXT"
DPV = "https://w3id.org/dpv#"
MODULE = "ontopriv-gdpr-alignment"
FORMATS = ("rdfxml", "owl", "turtle", "jsonld")
CSV_FILE = "ontopriv-gdpr-bridge.csv"
RESULT_FILE = "ontopriv-gdpr-bridge.json"
CACHE_FILE = "bridge-cache.jsonl"
STRICT = (SKOS.exactMatch, SKOS.closeMatch)
LOOSE = STRICT + (SKOS.broadMatch, SKOS.narrowMatch, SKOS.relatedMatch)
DEFAULTS = {"gdprtext": "data/input/gdprtext-ontology.ttl",
            "gdprtext_alignment": "data/reference/gdprtext-dpv-alignment.ttl"}
CSV_COLUMNS = ["ontopriv", "ontopriv_label", "ontopriv_source", "gdprtext", "gdprtext_label",
               "gdpr_articles", "dpv_terms", "ontopriv_relation", "gdprtext_relation",
               "verdict", "ai_relation", "explanation"]


@dataclass(frozen=True)
class Pair:
    ontopriv: str
    gdprtext: str
    terms: tuple            # ((dpv term, relation of OntoPriv, relation of GDPRtEXT), ...)

    @property
    def key(self) -> str:
        return f"{self.ontopriv} {self.gdprtext}"


def local(iri: str) -> str:
    return iri.replace("/", "#").rsplit("#", 1)[-1]


# --------------------------------------------------------------------------- the pairs

def links(g: Graph, relations=STRICT) -> dict[str, set]:
    """DPV term -> {(concept, relation)} of an alignment module."""
    out: dict[str, set] = {}
    for r in relations:
        for s, o in g.subject_objects(r):
            if isinstance(s, URIRef) and isinstance(o, URIRef) and str(o).startswith(DPV):
                out.setdefault(str(o), set()).add((str(s), local(str(r))))
    return out


def chain(base: Graph, gdprtext: Graph, relations=STRICT) -> list[Pair]:
    """The pairs that reach the same DPV term, both sides with one of ``relations``."""
    ours, theirs = links(base, relations), links(gdprtext, relations)
    pairs: dict[tuple, list] = {}
    for term in sorted(ours.keys() & theirs.keys()):
        for a, ra in sorted(ours[term]):
            for b, rb in sorted(theirs[term]):
                pairs.setdefault((a, b), []).append((term, ra, rb))
    return [Pair(a, b, tuple(sorted(ts))) for (a, b), ts in sorted(pairs.items())]


def _axiom(pair: Pair) -> BNode:
    """The same blank node every time, so rebuilding writes the same files."""
    return BNode("pair" + hashlib.sha256(pair.key.encode("utf-8")).hexdigest()[:24])


def build_graph(pairs: list[Pair]) -> Graph:
    from src.alignment.write import alignment_iri
    from src.core.emit import PROFILE_IRI
    g = Graph()
    for prefix, ns in (("bridge", BRIDGE), ("skos", SKOS), ("owl", OWL), ("dct", DCTERMS),
                       ("dpv", Namespace(DPV)), ("gdprtext", Namespace(GDPRTEXT_IRI + "#"))):
        g.bind(prefix, ns)
    onto = URIRef(BRIDGE_IRI)
    g.add((onto, RDF.type, OWL.Ontology))
    g.add((onto, RDFS.label, Literal("Puente de la base con el GDPR: OntoPriv y GDPRtEXT "
                                     "encadenados por el DPV", lang="es")))
    g.add((onto, RDFS.comment, Literal(
        "Pares de un concepto de la base (OntoPriv + LOPDP) y uno de GDPRtEXT que se "
        "corresponden con el mismo término del DPV con exactMatch o closeMatch en los dos lados. "
        "Cada par se escribe con skos:closeMatch y su anotación dice el término del DPV por el "
        "que pasa. GDPRtEXT y el DPV se referencian por IRI; no se importan. La explicación de "
        f"la IA revisora va aparte ({CSV_FILE}), no en la ontología.", lang="es")))
    g.add((onto, OWL.imports, URIRef(alignment_iri(PROFILE_IRI))))
    g.add((onto, DCTERMS.references, URIRef(GDPRTEXT_IRI)))
    g.add((onto, DCTERMS.references, URIRef(DPV.rstrip("#"))))
    g.add((onto, OWL.versionInfo, Literal("1")))
    g.add((BRIDGE.chainedBy, RDF.type, OWL.AnnotationProperty))
    g.add((BRIDGE.chainedBy, RDFS.label, Literal("encadenado por el DPV", lang="es")))
    g.add((BRIDGE.chainedBy, RDFS.comment, Literal(
        "Término del DPV con el que se corresponden los dos conceptos del par.", lang="es")))
    for pair in pairs:
        a, b = URIRef(pair.ontopriv), URIRef(pair.gdprtext)
        g.add((a, SKOS.closeMatch, b))
        x = _axiom(pair)
        g.add((x, RDF.type, OWL.Axiom))
        g.add((x, OWL.annotatedSource, a))
        g.add((x, OWL.annotatedProperty, SKOS.closeMatch))
        g.add((x, OWL.annotatedTarget, b))
        for term, ra, rb in pair.terms:
            g.add((x, BRIDGE.chainedBy, URIRef(term)))
        g.add((x, RDFS.comment, Literal("Encadenado por el DPV: " + "; ".join(
            f"OntoPriv {local(pair.ontopriv)} {ra} dpv:{local(term)} y GDPRtEXT "
            f"{local(pair.gdprtext)} {rb} dpv:{local(term)}" for term, ra, rb in pair.terms)
            + ".", lang="es")))
        g.add((x, OWL.versionInfo, Literal("1")))
    return g


def write_module(graph: Graph, folder: Path) -> list[Path]:
    from src.transform.convert import output_format, serialize
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in FORMATS:
        path = folder / f"{MODULE}{output_format(key).extension}"
        path.write_bytes(serialize(graph, key))
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- what each side says

@dataclass
class Side:
    iri: str
    kind: str
    label: str | None
    definition: str | None
    sources: list = field(default_factory=list)


def _text(g: Graph, node, *predicates) -> str | None:
    """The first value of the first predicate that has one: Spanish, then English, then any."""
    for p in predicates:
        values = [o for o in g.objects(node, p) if isinstance(o, Literal)]
        for lang in ("es", "en", None):
            for v in sorted(values, key=str):
                if lang is None or v.language == lang:
                    return str(v)
    return None


def describe(g: Graph, iri: str, *, gdpr: bool = False) -> Side:
    node = URIRef(iri)
    kinds = set(g.objects(node, RDF.type))
    kind = "propiedad" if kinds & {OWL.ObjectProperty, OWL.DatatypeProperty} else "clase"
    if gdpr:
        sources = sorted(local(str(o)) for o in g.objects(node, RDFS.isDefinedBy))
        return Side(iri, kind, _text(g, node, RDFS.label), _text(g, node, RDFS.comment), sources)
    return Side(iri, kind, _text(g, node, RDFS.label),
                _text(g, node, SKOS.definition, RDFS.comment),
                sorted(str(o) for o in g.objects(node, DCTERMS.source)))


@dataclass
class BridgeItem:
    key: str
    name: str
    ontopriv: Side
    gdprtext: Side
    terms: list                 # [(name, definition, relation of OntoPriv, relation of GDPRtEXT)]


def render_item(item: BridgeItem, chars: int = 400) -> str:
    one = au._one_line
    o, g = item.ontopriv, item.gdprtext
    lines = [f"OntoPriv: {local(o.iri)} ({o.kind})", f"Etiqueta: {one(o.label) or '-'}",
             f"Definicion: {one(o.definition, chars) or '-'}",
             f"Articulo de la LOPDP: {', '.join(o.sources) or '-'}",
             f"GDPRtEXT: {local(g.iri)} ({g.kind})", f"Etiqueta: {one(g.label) or '-'}",
             f"Descripcion: {one(g.definition, chars) or '-'}",
             f"Definido en el GDPR: {', '.join(g.sources) or '-'}"]
    for name, definition, ra, rb in item.terms:
        lines += [f"Termino DPV: {name} (OntoPriv {ra}, GDPRtEXT {rb})",
                  f"Definicion DPV: {one(definition, chars) or '-'}"]
    return "\n".join(lines)


def items_for(pairs: list[Pair], base: Graph, gdprtext: Graph, targets) -> list[BridgeItem]:
    terms = {t.iri: t for t in targets.targets} if targets is not None else {}
    out = []
    for p in pairs:
        named = [(local(t), getattr(terms.get(t), "definition", None), ra, rb)
                 for t, ra, rb in p.terms]
        out.append(BridgeItem(p.key, f"{local(p.ontopriv)} ↔ {local(p.gdprtext)}",
                              describe(base, p.ontopriv), describe(gdprtext, p.gdprtext, gdpr=True),
                              named))
    return out


# --------------------------------------------------------------------------- validating

def validate(parts: list, bridge, cfg: dict | None, engine: str | None) -> dict:
    """Forma and Lógica on the base plus the bridge: what the base already had is inherited,
    what only the bridge brings is new."""
    from src.validation import logic_inspector as li
    from src.validation import shape_inspector as si
    base = [replace(p, origin=si.BASE) for p in parts]
    name = "Base del sistema + puente con el GDPR"
    shapes = si.inspect([*base, bridge], dpv=si._dpv(cfg), kind=si.KIND_WORKSPACE, name=name)
    logic = li.inspect([*base, bridge], cfg=cfg, engine=engine, kind=si.KIND_WORKSPACE, name=name)
    v, w = shapes["summary"]["violations"], shapes["summary"]["warnings"]
    e, lw = logic["summary"]["errors"], logic["summary"]["warnings"]
    inherited = lambda c: sum(n for k, n in c.items() if k != si.NEW)       # noqa: E731
    return {"shapes": {"engine": shapes["engine"], "new_violations": v[si.NEW],
                       "new_warnings": w[si.NEW], "inherited_violations": inherited(v),
                       "inherited_warnings": inherited(w),
                       "new": [f for f in shapes["findings"] if f["origin"] == si.NEW]},
            "logic": {"engine": logic["engine"], "new_errors": e[si.NEW],
                      "new_warnings": lw[si.NEW], "inherited_errors": inherited(e),
                      "inherited_warnings": inherited(lw),
                      "new": [f for f in logic["findings"] if f["origin"] == si.NEW]}}


def clean(validation: dict | None) -> bool:
    if not validation:
        return False
    s, lg = validation["shapes"], validation["logic"]
    return not (s["new_violations"] or s["new_warnings"] or lg["new_errors"]
                or lg["new_warnings"])


# --------------------------------------------------------------------------- building

def paths(cfg: dict | None = None) -> dict[str, Path]:
    from src.config import ROOT, load_config
    cfg = cfg if cfg is not None else load_config()
    block = {**DEFAULTS, **(cfg.get("bridge") or {})}
    return {k: (Path(v) if Path(v).is_absolute() else ROOT / v) for k, v in block.items()
            if k in DEFAULTS}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:   # BOM: Excel shows accents
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) if r.get(c) is not None else "" for c in CSV_COLUMNS})


def build(cfg: dict | None = None, *, llm=None, use_llm: bool | None = None,
          engine: str | None = None, out_dir: str | Path | None = None, check: bool = True,
          targets=None, sleep_fn=time.sleep) -> dict:
    """Pairs, module in four formats, Forma and Lógica, and the explanation of each pair."""
    from src.alignment.inherit import base_paths
    from src.config import ROOT, load_config
    from src.transform.convert import load_graph
    from src.validation import shape_inspector as si
    cfg = cfg if cfg is not None else load_config()
    where = paths(cfg)
    for path in where.values():
        if not path.exists():
            raise FileNotFoundError(f"Falta {path}")
    parts, _ = si.base_parts(cfg)
    base = Graph()
    for p in parts:
        base += p.graph
    aligned = next(p.graph for p in parts if p.kind == "alignment")
    theirs = load_graph(where["gdprtext_alignment"])
    gdprtext = load_graph(where["gdprtext"])
    pairs = chain(aligned, theirs)
    loose = chain(aligned, theirs, LOOSE)
    folder = Path(out_dir) if out_dir else base_paths(cfg)["candidates"].parent
    graph = build_graph(pairs)
    files = write_module(graph, folder)
    turtle = next(f for f in files if f.suffix == ".ttl")
    shown = si._shown(turtle, ROOT)
    bridge = si.Part("bridge", shown, si.NEW, graph, _sha256(turtle), turtle)
    validation = validate(parts, bridge, cfg, engine) if check else None

    settings = au.audit_settings(cfg)
    available = au._use_llm(cfg, use_llm) and settings.enabled
    error = None if available else ("se pidió --sin-ia" if settings.enabled
                                    else "desactivada en config.yaml (audit.enabled)")
    if available and llm is None:
        try:
            from src.ai.llm_client import LLMClient
            llm = LLMClient()
        except KeyError as exc:                        # LLM_API_KEY / LLM_MODEL missing
            available, error = False, f"Falta la variable {exc} en .env"
    if targets is None:
        from src.alignment.propose import load_dpv_targets
        targets = load_dpv_targets(cfg)
    items = items_for(pairs, base, gdprtext, targets)
    report = au.ask_audit(items, llm=llm if available else None, settings=settings,
                          cache_path=folder / CACHE_FILE, sleep_fn=sleep_fn, prompt=PROMPT,
                          render=render_item)
    rows = []
    for pair, item in zip(pairs, items):
        answer = report.answers.get(pair.key) or {}
        rows.append({"ontopriv": local(pair.ontopriv), "ontopriv_iri": pair.ontopriv,
                     "ontopriv_label": item.ontopriv.label,
                     "ontopriv_source": "; ".join(item.ontopriv.sources) or None,
                     "gdprtext": local(pair.gdprtext), "gdprtext_iri": pair.gdprtext,
                     "gdprtext_label": item.gdprtext.label,
                     "gdpr_articles": "; ".join(item.gdprtext.sources) or None,
                     "dpv_terms": "; ".join(f"dpv:{local(t)}" for t, _, _ in pair.terms),
                     "ontopriv_relation": "; ".join(ra for _, ra, _ in pair.terms),
                     "gdprtext_relation": "; ".join(rb for _, _, rb in pair.terms),
                     "verdict": answer.get("verdict"), "ai_relation": answer.get("relation"),
                     "explanation": answer.get("explanation")})
    write_csv(rows, folder / CSV_FILE)
    counts = {v: sum(r["verdict"] == v for r in rows) for v in (*au.VERDICTS, au.INVALID)}
    out = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "gdprtext": {"file": si._shown(where["gdprtext"], ROOT),
                        "sha256": _sha256(where["gdprtext"])},
           "gdprtext_alignment": {"file": si._shown(where["gdprtext_alignment"], ROOT),
                                  "sha256": _sha256(where["gdprtext_alignment"])},
           "pairs": len(pairs), "terms": len({t for p in pairs for t, _, _ in p.terms}),
           "loose_pairs": len(loose), "triples": len(graph),
           "files": [si._shown(f, ROOT) for f in files], "validation": validation,
           "prompt": PROMPT.manifest(),
           "ai": {**report.ai(available), **({"error": error} if error else {})},
           "summary": {**counts, "pending": sum(r["verdict"] is None for r in rows)},
           "rows": rows}
    (folder / RESULT_FILE).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    return out


def load_result(cfg: dict | None = None) -> dict | None:
    """The last bridge built (for the page of the base), or None."""
    from src.alignment.inherit import base_paths
    path = base_paths(cfg)["candidates"].parent / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- console

def _n(n: int, one: str, many: str) -> str:
    return f"{n} {one if n == 1 else many}"


def validation_lines(v: dict | None) -> list[str]:
    if v is None:
        return ["  Forma y Lógica: no se revisaron"]
    s, lg = v["shapes"], v["logic"]
    from src.validation.logic_inspector import engine_text
    return [f"  Forma (pySHACL {s['engine']['version']}): "
            f"{_n(s['new_violations'], 'violación nueva', 'violaciones nuevas')}, "
            f"{_n(s['new_warnings'], 'aviso nuevo', 'avisos nuevos')} · heredados de la base: "
            f"{_n(s['inherited_violations'], 'violación', 'violaciones')}, "
            f"{_n(s['inherited_warnings'], 'aviso', 'avisos')}",
            f"  Lógica ({engine_text(lg['engine'])}): "
            f"{_n(lg['new_errors'], 'inconsistencia nueva', 'inconsistencias nuevas')}, "
            f"{_n(lg['new_warnings'], 'aviso nuevo', 'avisos nuevos')} · heredados de la base: "
            f"{_n(lg['inherited_errors'], 'inconsistencia', 'inconsistencias')}, "
            f"{_n(lg['inherited_warnings'], 'aviso', 'avisos')}",
            "  " + ("Sin hallazgos nuevos: el puente no rompe ninguna regla ni agrega "
                    "contradicciones." if clean(v) else
                    "El puente trae hallazgos nuevos: revisa los reportes.")]


def render(out: dict) -> str:
    p, ai, s = out["prompt"], out["ai"], out["summary"]
    lines = ["Puente de la base con el GDPR (S9-T09)",
             f"  Alineación de GDPRtEXT usada (fija): {out['gdprtext_alignment']['file']} "
             f"(sha256 {out['gdprtext_alignment']['sha256'][:12]})",
             f"  Pares OntoPriv ↔ GDPRtEXT con exactMatch o closeMatch en los dos lados: "
             f"{out['pairs']}, por {_n(out['terms'], 'término', 'términos')} del DPV "
             f"(con cualquier relación SKOS serían {out['loose_pairs']}; no entran)",
             f"  Módulo: {', '.join(out['files'])} ({out['triples']} tripletas)",
             *validation_lines(out["validation"]),
             f"  IA revisora ({p['name']}.v{p['version']}): {au.ai_text(ai)} · "
             f"{_n(s[au.COHERENT], 'coherente', 'coherentes')}, "
             f"{_n(s[au.DOUBTFUL], 'dudosa', 'dudosas')}, "
             f"{_n(s[au.INCOHERENT], 'incoherente', 'incoherentes')}"
             + (f", {s['pending']} sin revisar" if s["pending"] else "")
             + "; solo explica, no cambia los pares", "", "  Pares:"]
    for r in out["rows"]:
        verdict = r["verdict"] or "sin revisar"
        lines.append(f"    [{verdict}] {r['ontopriv']} ↔ {r['gdprtext']} · por {r['dpv_terms']} "
                     f"({r['ontopriv_relation']} / {r['gdprtext_relation']})")
        if r["explanation"]:
            lines.append(f"         {r['explanation']}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import base_paths
    from src.config import load_config
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.gdpr_bridge",
        description="Escribe el puente de la base con el GDPR (pares OntoPriv ↔ GDPRtEXT "
                    "encadenados por el DPV), lo valida y la IA revisora explica cada par.")
    parser.add_argument("--sin-ia", action="store_true", help="sin la IA revisora (solo caché)")
    parser.add_argument("--motor", choices=["jena", "owlrl"], help="motor de la lógica")
    args = parser.parse_args(argv)
    cfg = load_config()
    try:
        out = build(cfg, use_llm=False if args.sin_ia else None, engine=args.motor)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render(out))
    folder = base_paths(cfg)["candidates"].parent
    print(f"  Explicaciones: {folder / CSV_FILE}")
    print(f"  Guardado en: {folder / RESULT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
