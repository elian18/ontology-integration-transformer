"""OOPS! (OntOlogy Pitfall Scanner!) on OntoPriv and on the base, and the comparison of both
(S11-T06). The request is the one WIDOCO 1.4.26 sends; the answers are kept in docs/oops."""
from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from rdflib import Graph, Namespace
from rdflib.namespace import RDF

URL = "https://oops.linkeddata.es/rest"
OOPS = Namespace("http://oops.linkeddata.es/def#")
SUBJECTS = ("ontopriv", "base")
RESULT_FILE = "evaluation.json"
LEVELS = {"critical": 0, "important": 1, "minor": 2}
LEVEL_WORDS = {"es": {"critical": "crítico", "important": "importante", "minor": "menor"},
               "en": {"critical": "critical", "important": "important", "minor": "minor"}}
CHANGE_WORDS = {
    "es": {"fixed": "corregido", "fewer": "menos casos", "same": "igual", "more": "más casos",
           "new": "nuevo"},
    "en": {"fixed": "fixed", "fewer": "fewer cases", "same": "same", "more": "more cases",
           "new": "new"}}
TEXT = {
    "es": {"title": "Evaluación con OOPS!", "date": "OOPS! revisó OntoPriv original y la base "
           "el {date}. La base tiene más elementos que OntoPriv (la alineación, el puente y los "
           "conceptos de la LOPDP), así que una parte de los casos nuevos viene de lo agregado. "
           "«Toda la ontología» es un problema general, sin elementos que contar.",
           "head": "| Código | Problema | Importancia | OntoPriv | Base | Cambio |",
           "whole": "toda la ontología"},
    "en": {"title": "Evaluation with OOPS!", "date": "OOPS! checked the original OntoPriv and the "
           "base on {date}. The base has more elements than OntoPriv (the alignment, the bridge "
           "and the concepts of the LOPDP), so part of the new cases comes from what was added. "
           "«Whole ontology» is a general pitfall, with no elements to count.",
           "head": "| Code | Pitfall | Importance | OntoPriv | Base | Change |",
           "whole": "whole ontology"}}


@dataclass(frozen=True)
class Pitfall:
    code: str
    name: str
    level: str                      # critical | important | minor
    cases: int | None               # None: about the whole ontology
    elements: tuple[str, ...] = ()


def request_body(rdfxml: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?><OOPSRequest><OntologyUrl></OntologyUrl>'
            f"<OntologyContent><![CDATA[ {rdfxml} ]]></OntologyContent><Pitfalls></Pitfalls>"
            "<OutputFormat>RDF/XML</OutputFormat></OOPSRequest>")


def post(body: str, timeout: int) -> bytes:
    request = urllib.request.Request(URL, data=body.encode("utf-8"), method="POST",
                                     headers={"Accept": "application/xml",
                                              "Content-Type": "application/xml"})
    with urllib.request.urlopen(request, timeout=timeout) as answer:
        return answer.read()


def parse(answer: bytes) -> list[Pitfall]:
    g = Graph().parse(data=answer, format="xml")
    out = []
    for node in g.subjects(RDF.type, OOPS.pitfall):
        cases = g.value(node, OOPS.hasNumberAffectedElements)
        out.append(Pitfall(
            code=str(g.value(node, OOPS.hasCode)), name=str(g.value(node, OOPS.hasName)),
            level=str(g.value(node, OOPS.hasImportanceLevel)).strip().lower(),
            cases=int(cases) if cases is not None else None,
            elements=tuple(sorted(str(e) for e in g.objects(node, OOPS.hasAffectedElement)))))
    return sorted(out, key=lambda p: (LEVELS.get(p.level, 3), p.code))


def change(before: Pitfall | None, after: Pitfall | None) -> str:
    if after is None:
        return "fixed"
    if before is None:
        return "new"
    if before.cases is None or after.cases is None or before.cases == after.cases:
        return "same"
    return "fewer" if after.cases < before.cases else "more"


def compare(ontopriv: list[Pitfall], base: list[Pitfall]) -> list[dict]:
    a, b = {p.code: p for p in ontopriv}, {p.code: p for p in base}
    rows = []
    for code in sorted(a.keys() | b.keys(),
                       key=lambda c: (LEVELS.get((a.get(c) or b.get(c)).level, 3), c)):
        known = a.get(code) or b.get(code)
        rows.append({"code": code, "name": known.name, "level": known.level,
                     "ontopriv": a.get(code), "base": b.get(code),
                     "change": change(a.get(code), b.get(code))})
    return rows


def _cases(p: Pitfall | None, lang: str) -> str:
    if p is None:
        return "0"
    return TEXT[lang]["whole"] if p.cases is None else str(p.cases)


def table(result: dict, lang: str) -> str:
    """The comparison in Markdown, for the description of the documentation."""
    words = TEXT[lang]
    rows = compare(*(load_pitfalls(result, s) for s in SUBJECTS))
    lines = [f"### {words['title']}", "", words["date"].format(date=result["date"]), "",
             words["head"], "|---|---|---|---|---|---|"]
    for r in rows:
        level = LEVEL_WORDS[lang].get(r["level"], r["level"])
        lines.append(f"| {r['code']} | {r['name']} | {level} | {_cases(r['ontopriv'], lang)} | "
                     f"{_cases(r['base'], lang)} | {CHANGE_WORDS[lang][r['change']]} |")
    return "\n".join(lines)


def load_pitfalls(result: dict, subject: str) -> list[Pitfall]:
    return [Pitfall(**{**p, "elements": tuple(p["elements"])}) for p in result[subject]]


def save(folder: Path, answers: dict[str, bytes], date: str) -> dict:
    """Keeps what OOPS! answered (RDF/XML) and what it means, so the page is rebuilt without
    asking again."""
    folder.mkdir(parents=True, exist_ok=True)
    result = {"date": date}
    for subject in SUBJECTS:
        (folder / f"{subject}-oops.rdf").write_bytes(answers[subject])
        result[subject] = [{**asdict(p), "elements": list(p.elements)}
                           for p in parse(answers[subject])]
    (folder / RESULT_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    return result


def load(folder: Path) -> dict | None:
    path = folder / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
