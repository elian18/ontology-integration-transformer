"""Automatic acceptance of the concepts the AI extracts from a law (S7-T09, decision D1).

In Sprint 5 a person decided the 93 concepts the AI extracted from the LOPDP: 25 approved as
new, 66 confirmed as the same as an OntoPriv entity, 2 rejected. For a law a person uploads
nobody reviews them one by one any more (the architecture of Sprint 7): a FIXED rule decides,
and the person comments afterwards (Sprint 10). The rule, as decision D1 states it, accepts a
concept when it

1. CITES AN ARTICLE OF THE LAW: at least one article number, and every number is an article of
   the uploaded law (provenance for PROV-O, Sprint 10);
2. IS WELL FORMED: a technical name of 3 to 60 ASCII letters or digits starting with a letter
   (the kind comes from it, as in S4-T09: UpperCamelCase = class, lowerCamelCase = property),
   a label of one line up to 150 characters and a definition of 20 to 600 characters;
3. IS NOT REPEATED: two concepts of the same kind with the same label (ignoring case, accents
   and punctuation) are one concept. The first one extracted stays; the later one is
   "repeated": it is not written, and its articles are added to the first one. A class and a
   property are never the same concept (on the GDPR the LLM labelled the property
   ``hasController`` "controller", like the class ``Controller``).

The rule has no free threshold to tune, so its calibration is a measurement: it is applied to
the 93 concepts of the LOPDP and compared with the decisions of Sprint 5
(``py -m src.core.acceptance``). A repeated concept agrees with the person when the person
rejected it or confirmed it as the same OntoPriv entity as the one it joins.

Being "the same as an OntoPriv entity" is not a reason to reject: S7-T10 reuses the IRI of
OntoPriv-Core for those (decision D6).
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from src.alignment.sources import kind_from_name

RULE_VERSION = 1
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{2,59}$")
MAX_LABEL_CHARS = 150
MIN_DEFINITION_CHARS = 20
MAX_DEFINITION_CHARS = 600
ACCEPTED, REPEATED, REJECTED = "accepted", "repeated", "rejected"
STATUS_LABELS = {ACCEPTED: "aceptado por la regla", REPEATED: "repetido (se une al primero)",
                 REJECTED: "rechazado por la regla"}
REASONS = {"no_article": "no cita un artículo de la ley",
           "unknown_article": "cita un artículo que la ley no tiene",
           "bad_name": "nombre técnico inválido",
           "no_label": "sin etiqueta",
           "long_label": "etiqueta de más de 150 caracteres o de varias líneas",
           "no_definition": "sin definición",
           "short_definition": "definición de menos de 20 caracteres",
           "long_definition": "definición de más de 600 caracteres"}
RULE_TEXT = ("cita un artículo de la ley, está bien formado (nombre técnico, etiqueta y "
             "definición) y no se repite")
CALIBRATION_FILE = "acceptance-calibration.json"


def label_key(label: str) -> str:
    """The label without case, accents or punctuation: what "the same label" means."""
    text = unicodedata.normalize("NFKD", str(label or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(re.sub(r"[^\w]+", " ", text).split())


def check(concept: dict, law_articles: set[int] | None = None) -> list[str]:
    """Reasons (codes of ``REASONS``) why the rule rejects one concept; [] if it passes."""
    reasons = []
    articles = concept.get("articles") or []
    if not articles:
        reasons.append("no_article")
    elif law_articles is not None and not set(articles) <= law_articles:
        reasons.append("unknown_article")
    if not NAME_RE.match(str(concept.get("name") or "")):
        reasons.append("bad_name")
    label = str(concept.get("label") or "").strip()
    if not label:
        reasons.append("no_label")
    elif len(label) > MAX_LABEL_CHARS or "\n" in label:
        reasons.append("long_label")
    definition = str(concept.get("definition") or "").strip()
    if not definition:
        reasons.append("no_definition")
    elif len(definition) < MIN_DEFINITION_CHARS:
        reasons.append("short_definition")
    elif len(definition) > MAX_DEFINITION_CHARS:
        reasons.append("long_definition")
    return reasons


def apply_rule(concepts: list[dict], law_articles: set[int] | None = None) -> list[dict]:
    """One entry per extracted concept, in extraction order, with the verdict of the rule."""
    entries, first = [], {}
    for c in concepts:
        entry = {"name": c.get("name"), "label": (c.get("label") or "").strip(),
                 "definition": (c.get("definition") or "").strip(),
                 "declared_type": c.get("type"), "kind": None,
                 "articles": list(c.get("articles") or []), "status": ACCEPTED,
                 "reasons": check(c, law_articles), "repeated_of": None, "joined": []}
        if entry["reasons"]:
            entry["status"] = REJECTED
        else:
            entry["kind"] = kind_from_name(entry["name"])
            key = (entry["kind"], label_key(entry["label"]))
            kept = first.get(key)
            if kept is None:
                first[key] = entry
            else:
                entry["status"], entry["repeated_of"] = REPEATED, kept["name"]
                kept["joined"].append(entry["name"])
                kept["articles"] += [a for a in entry["articles"] if a not in kept["articles"]]
        entries.append(entry)
    return entries


def summarize(entries: list[dict]) -> dict:
    accepted = [e for e in entries if e["status"] == ACCEPTED]
    reasons = {code: sum(code in e["reasons"] for e in entries) for code in REASONS}
    return {"concepts": len(entries), "accepted": len(accepted),
            "repeated": sum(e["status"] == REPEATED for e in entries),
            "rejected": sum(e["status"] == REJECTED for e in entries),
            "classes": sum(e["kind"] == "class" for e in accepted),
            "properties": sum(e["kind"] == "property" for e in accepted),
            "reasons": {code: n for code, n in reasons.items() if n}}


# --------------------------------------------------------------------------- calibration

def agreement(entries: list[dict], decisions: dict) -> dict:
    """The rule against the decisions of Sprint 5 (``concept_key`` -> Decision)."""
    from src.alignment.decisions import DECISION_DUPLICATE, DECISION_REJECTED
    rows = []
    for e in entries:
        d = decisions.get(f"ai:{e['name']}")
        if d is None:
            continue                                   # not decided by the person
        rejected = d.decision == DECISION_REJECTED
        if e["status"] == ACCEPTED:
            agrees = not rejected
        elif e["status"] == REJECTED:
            agrees = rejected
        else:
            kept = decisions.get(f"ai:{e['repeated_of']}")
            agrees = rejected or (d.decision == DECISION_DUPLICATE and kept is not None
                                  and kept.same_as == d.same_as)      # the same entity
        rows.append({"name": e["name"], "label": e["label"], "rule": e["status"],
                     "reasons": e["reasons"], "repeated_of": e["repeated_of"],
                     "person": d.decision, "same_as": d.same_as, "agrees": agrees})
    n = len(rows)
    agree = sum(r["agrees"] for r in rows)
    return {"decided": n, "agree": agree, "agreement": agree / n if n else None,
            "accepted_but_rejected": [r["name"] for r in rows
                                      if r["rule"] == ACCEPTED and r["person"] == "rejected"],
            "rejected_but_kept": [r["name"] for r in rows
                                  if r["rule"] == REJECTED and r["person"] != "rejected"],
            "repeated_disagree": [r["name"] for r in rows
                                  if r["rule"] == REPEATED and not r["agrees"]],
            "rows": rows}


def base_files(cfg: dict | None = None) -> dict[str, Path]:
    """The AI concepts of the LOPDP (S3-T05), the decisions log (Sprint 5) and the output."""
    from src.alignment.inherit import base_paths
    from src.core.extract import PROPOSALS_FILE
    paths = base_paths(cfg)
    out_dir = paths["candidates"].parent
    return {"proposals": out_dir / PROPOSALS_FILE, "log": paths["log"],
            "output": out_dir / CALIBRATION_FILE}


def calibrate(cfg: dict | None = None) -> dict:
    """Apply the rule to the 93 concepts of the LOPDP and compare with Sprint 5."""
    from src.alignment.ai_concepts import concept_decisions
    from src.alignment.decisions import DEFAULT_SOURCE_ID, read_log
    files = base_files(cfg)
    path = files["proposals"]
    if not path.exists():
        raise FileNotFoundError(f"No está el archivo de conceptos de la IA de la base: {path}")
    proposals = json.loads(path.read_text(encoding="utf-8"))
    entries = apply_rule(proposals.get("concepts", []))
    decisions = concept_decisions(read_log(files["log"]), DEFAULT_SOURCE_ID)
    return {"rule_version": RULE_VERSION, "rule": RULE_TEXT, "source": str(path.name),
            "summary": summarize(entries), "agreement": agreement(entries, decisions)}


def _pct(value) -> str:
    return "—" if value is None else f"{100 * value:.1f}".replace(".", ",") + " %"


def render_calibration(result: dict) -> str:
    s, a = result["summary"], result["agreement"]
    by_name = {r["name"]: r for r in a["rows"]}
    lines = ["Regla de aceptación de conceptos (S7-T09)",
             f"  Regla: {result['rule']}",
             f"  Conceptos de la IA en la base (LOPDP): {s['concepts']} · aceptados: "
             f"{s['accepted']} · repetidos: {s['repeated']} · rechazados: {s['rejected']}",
             f"  Coincide con tus decisiones del Sprint 5 en {a['agree']} de {a['decided']} "
             f"({_pct(a['agreement'])})"]
    for r in a["rows"]:
        if r["rule"] == REPEATED:
            who = ("lo rechazaste" if r["person"] == "rejected" else
                   f"lo confirmaste como {(r['same_as'] or '').rsplit('#', 1)[-1]}")
            lines.append(f"    repetido: {r['name']} se une a {r['repeated_of']} "
                         f"(\"{r['label']}\"); {who}")
    for name in a["accepted_but_rejected"]:
        lines.append(f"    aceptado por la regla pero lo rechazaste: {name} "
                     f"(\"{by_name[name]['label']}\")")
    for name in a["rejected_but_kept"]:
        why = ", ".join(REASONS[c] for c in by_name[name]["reasons"])
        lines.append(f"    rechazado por la regla ({why}) pero lo conservaste: {name}")
    for name in a["repeated_disagree"]:
        lines.append(f"    repetido según la regla pero lo decidiste distinto: {name}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.config import load_config
    parser = argparse.ArgumentParser(prog="py -m src.core.acceptance",
                                     description="Mide la regla de aceptación de conceptos con "
                                                 "las decisiones del Sprint 5 (S7-T09).")
    parser.parse_args(argv)
    cfg = load_config()
    try:
        result = calibrate(cfg)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    out = base_files(cfg)["output"]
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(render_calibration(result))
    print(f"  Guardado en: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())