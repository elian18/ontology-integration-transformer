# The catalog of competency questions (S9-T03): load it, check it and say how each one is judged.
from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATALOG_FILE = Path(__file__).with_name("questions.yaml")
BASE, GENERIC = "base", "generic"
GROUPS = {BASE: "Base del sistema", GENERIC: "Genéricas (base y subidas)"}
MODES = {"base": "Base del sistema", "ontology": "Mi ontología", "law": "Una ley",
         "ontology+law": "Mi ontología y una ley"}
SOURCES = {"paper": "Artículo de OntoPriv", "lopdp": "LOPDP", "dpv": "DPV 2.3", "system": "Sistema"}
ALL, ANY, VALUE, AT_LEAST = "all", "any", "value", "at_least"
KINDS = {BASE: (ALL, ANY, VALUE), GENERIC: (AT_LEAST,)}
ANY_DPV, CITES_ARTICLE = "dpv:*", "dct:source"
LOOKS_FOR = re.compile(r"^(dpv:(\*|[A-Za-z]+)|dct:source)$")
PAPER_FIELDS = ("original", "reported", "note")
ARTICLE = re.compile(r"\bart(?:[íi]culo|\.)?\s*(\d+)", re.IGNORECASE)
COLORS = ("Colores: verde si la ontología responde según el criterio; amarillo si no; gris si el "
          "motor no corrió o la pregunta no aplica a ese modo.")


class CatalogError(ValueError):
    """The catalog file is incomplete (message in Spanish)."""


@dataclass(frozen=True)
class Element:
    name: str
    articles: tuple[int, ...]
    terms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Expected:
    text: str
    articles: tuple[int, ...]
    elements: tuple[Element, ...] = ()
    value: float | None = None
    unit: str = ""
    quote: str = ""


@dataclass(frozen=True)
class Question:
    id: str
    group: str
    question: str
    source: str
    reference: str
    modes: tuple[str, ...]
    kind: str
    min: int = 1
    expected: Expected | None = None
    looks_for: str = ""
    paper: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Catalog:
    version: int
    date: str
    law: str
    law_file: str
    paper: str
    relations: tuple[str, ...]
    hierarchy: tuple[str, ...]
    questions: tuple[Question, ...]
    sha256: str
    file: str


def _text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _articles(value) -> bool:
    return (isinstance(value, list) and bool(value)
            and all(isinstance(a, int) and not isinstance(a, bool) and a > 0 for a in value))


def _check_expected(qid: str, kind: str, expected) -> list[str]:
    if not isinstance(expected, dict):
        return [f"{qid}: le falta la respuesta esperada"]
    problems = [] if _text(expected.get("text")) else [f"{qid}: la respuesta esperada no tiene texto"]
    if kind == VALUE:
        if not _articles(expected.get("articles")):
            problems.append(f"{qid}: la respuesta esperada no dice su artículo")
        if not isinstance(expected.get("value"), (int, float)) or isinstance(expected.get("value"), bool):
            problems.append(f"{qid}: la respuesta esperada no trae su valor")
        for key, what in (("unit", "su unidad"), ("quote", "el texto de la ley que la respalda")):
            if not _text(expected.get(key)):
                problems.append(f"{qid}: la respuesta esperada no trae {what}")
        return problems
    elements = expected.get("elements")
    if not isinstance(elements, list) or not elements:
        return problems + [f"{qid}: la respuesta esperada no trae sus elementos"]
    for n, e in enumerate(elements, 1):
        if not isinstance(e, dict) or not _text(e.get("name")):
            problems.append(f"{qid}: el elemento {n} no tiene nombre")
            continue
        if not _articles(e.get("articles")):
            problems.append(f"{qid}: el elemento «{e['name']}» no dice su artículo")
        terms = e.get("terms", [])
        if not isinstance(terms, list) or not all(_text(t) for t in terms):
            problems.append(f"{qid}: los términos de «{e['name']}» deben ser textos")
    return problems


def check(data) -> list[str]:
    """Everything missing in the catalog, in Spanish (empty when it is complete)."""
    if not isinstance(data, dict):
        return ["el catálogo no es un diccionario YAML"]
    problems = []
    if not isinstance(data.get("version"), int) or data["version"] < 1:
        problems.append("al catálogo le falta su versión")
    for key in ("law", "law_file", "paper"):
        if not _text(data.get(key)):
            problems.append(f"al catálogo le falta «{key}»")
    matching = data.get("matching") or {}
    for key in ("relations", "hierarchy"):
        if not isinstance(matching.get(key), list) or not matching[key]:
            problems.append(f"al catálogo le falta «matching.{key}»")
    questions = data.get("questions")
    if not isinstance(questions, list) or not questions:
        return problems + ["el catálogo no tiene preguntas"]
    seen = set()
    for n, q in enumerate(questions, 1):
        qid = q.get("id") if isinstance(q, dict) else None
        if not _text(qid):
            problems.append(f"la pregunta {n} no tiene identificador")
            continue
        if qid in seen:
            problems.append(f"{qid}: el identificador está repetido")
        seen.add(qid)
        group = q.get("group")
        if group not in GROUPS:
            problems.append(f"{qid}: el grupo debe ser base o generic")
            continue
        text = q.get("question")
        if not _text(text) or not (text.strip().startswith("¿") and text.strip().endswith("?")):
            problems.append(f"{qid}: la pregunta debe ir entre ¿ y ?")
        origin = q.get("origin")
        if not isinstance(origin, dict) or origin.get("source") not in SOURCES \
                or not _text(origin.get("reference")):
            problems.append(f"{qid}: le falta el origen")
        elif origin["source"] == "paper":
            problems += [f"{qid}: al origen del artículo le falta «{k}»"
                         for k in PAPER_FIELDS if not _text(origin.get(k))]
        modes = q.get("modes")
        if not isinstance(modes, list) or not modes or not set(modes) <= set(MODES) \
                or len(set(modes)) != len(modes):
            problems.append(f"{qid}: le faltan los modos (base, ontology, law u ontology+law)")
        elif group == BASE and modes != ["base"]:
            problems.append(f"{qid}: una pregunta de la base solo aplica a la base")
        criterion = q.get("criterion")
        kind = criterion.get("kind") if isinstance(criterion, dict) else None
        if kind not in KINDS[group]:
            problems.append(f"{qid}: le falta el criterio ({' o '.join(KINDS[group])})")
        elif kind == AT_LEAST and (not isinstance(criterion.get("min"), int) or criterion["min"] < 1):
            problems.append(f"{qid}: el criterio at_least necesita un mínimo de 1 o más")
        if group == BASE:
            if kind in KINDS[BASE]:
                problems += _check_expected(qid, kind, q.get("expected"))
        else:
            if "expected" in q:
                problems.append(f"{qid}: una pregunta genérica no lleva respuesta esperada")
            if not isinstance(q.get("looks_for"), str) or not LOOKS_FOR.match(q["looks_for"]):
                problems.append(f"{qid}: le falta qué busca (dpv:Término, dpv:* o dct:source)")
    return problems


def _element(e: dict) -> Element:
    return Element(e["name"].strip(), tuple(e["articles"]), tuple(t.strip() for t in e.get("terms", [])))


def _question(q: dict) -> Question:
    origin, criterion = q["origin"], q["criterion"]
    expected = None
    if q["group"] == BASE:
        e = q["expected"]
        elements = tuple(_element(x) for x in e.get("elements", []))
        articles = tuple(e["articles"]) if "articles" in e else tuple(
            sorted({a for x in elements for a in x.articles}))
        expected = Expected(e["text"].strip(), articles, elements, e.get("value"),
                            e.get("unit", ""), e.get("quote", ""))
    return Question(q["id"], q["group"], q["question"].strip(), origin["source"],
                    origin["reference"].strip(), tuple(q["modes"]), criterion["kind"],
                    criterion.get("min", 1), expected, q.get("looks_for", ""),
                    {k: origin[k] for k in PAPER_FIELDS if k in origin})


def load_catalog(path: str | Path = CATALOG_FILE) -> Catalog:
    path = Path(path)
    raw = path.read_bytes()
    data = yaml.safe_load(raw.decode("utf-8"))
    problems = check(data)
    if problems:
        raise CatalogError("El catálogo de preguntas está incompleto:\n  - " + "\n  - ".join(problems))
    return Catalog(data["version"], str(data.get("date", "")), data["law"], data["law_file"],
                   data["paper"], tuple(data["matching"]["relations"]),
                   tuple(data["matching"]["hierarchy"]),
                   tuple(_question(q) for q in data["questions"]),
                   hashlib.sha256(raw).hexdigest(), path.name)


def for_mode(catalog: Catalog, mode: str) -> list[Question]:
    return [q for q in catalog.questions if mode in q.modes]


# ------------------------------------------------------------------ how an answer is recognized

def normalize(text: str) -> str:
    """Lowercase words without accents: «DataSubject_Right-of-Access» -> «data subject right of access»."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def names(element: Element, text: str) -> bool:
    """The element is named in the text: its name or one of its terms, as whole words."""
    hay = f" {normalize(text)} "
    return any(f" {normalize(n)} " in hay for n in (element.name, *element.terms) if normalize(n))


def cited_articles(text: str) -> set[int]:
    return {int(n) for n in ARTICLE.findall(str(text))}


def own_articles(question: Question, element: Element) -> set[int]:
    """The articles that only this element of the question has (shared ones prove nothing)."""
    others = {a for e in question.expected.elements if e is not element for a in e.articles}
    return set(element.articles) - others


def found(question: Question, element: Element, texts: list[str], sources: list[str]) -> bool:
    """One row of an answer holds the element: it names it or cites an article only it has."""
    cited = {a for s in sources for a in cited_articles(s)}
    return bool(own_articles(question, element) & cited) or any(names(element, t) for t in texts)


def criterion_text(q: Question) -> str:
    if q.kind == VALUE:
        value = f"{q.expected.value:g} {q.expected.unit}".strip()
        return f"Verde si la respuesta trae el valor {value}; amarillo si no lo trae."
    if q.kind == ALL:
        return (f"Verde si la respuesta trae los {len(q.expected.elements)} elementos que da la "
                f"ley; amarillo si falta alguno.")
    if q.kind == ANY:
        return (f"Verde si la respuesta trae al menos uno de los {len(q.expected.elements)} "
                f"elementos que da la ley; amarillo si no trae ninguno.")
    return (f"Verde si la respuesta trae al menos {q.min} concepto{'s' if q.min > 1 else ''}; "
            f"amarillo si no trae ninguno.")


def looks_for_text(q: Question) -> str:
    if q.looks_for == ANY_DPV:
        return "conceptos alineados con cualquier término del DPV"
    if q.looks_for == CITES_ARTICLE:
        return "conceptos que citan el artículo de la ley del que salen (dct:source)"
    return f"conceptos alineados con {q.looks_for} o con un término que esté debajo"


def matching_text(catalog: Catalog) -> list[str]:
    relations = ", ".join(catalog.relations[:-1]) + " o " + catalog.relations[-1] \
        if len(catalog.relations) > 1 else catalog.relations[0]
    hierarchy = " o ".join(catalog.hierarchy)
    return [
        "Un elemento esperado está en la respuesta si una fila lo nombra (su nombre o uno de sus "
        "términos, como palabras completas, sin distinguir mayúsculas, tildes ni guiones) o cita "
        "un artículo que solo tiene ese elemento.",
        f"En las genéricas cuenta un concepto alineado por {relations} con el término del DPV o "
        f"con uno que esté debajo ({hierarchy}).",
    ]


# ------------------------------------------------------------------ console

def render_console(catalog: Catalog) -> str:
    base = [q for q in catalog.questions if q.group == BASE]
    paper = sum(q.source == "paper" for q in base)
    generic = len(catalog.questions) - len(base)
    lines = [f"Catálogo de preguntas de competencia v{catalog.version} "
             f"({catalog.date}, sha256 {catalog.sha256[:12]})",
             f"  Archivo: src/validation/{catalog.file}",
             f"  {len(catalog.questions)} preguntas: {len(base)} de la base ({paper} del artículo "
             f"de OntoPriv y {len(base) - paper} de la {catalog.law}) y {generic} genéricas",
             f"  Respuestas esperadas: tomadas de la {catalog.law} ({catalog.law_file}) antes de "
             f"escribir cualquier consulta",
             f"  Artículo de OntoPriv: {catalog.paper}",
             *(f"  {t}" for t in matching_text(catalog)),
             f"  {COLORS}"]
    for group, title in GROUPS.items():
        lines += ["", title]
        for q in (q for q in catalog.questions if q.group == group):
            lines.append(f"  {q.id} {q.question}")
            origin = f"{SOURCES[q.source]}, {q.reference}"
            if q.paper:
                origin += f"; respondió: {q.paper['reported']}"
            lines.append(f"      Origen: {origin}")
            if q.paper.get("note"):
                lines.append(f"      Nota: {q.paper['note']}")
            if q.expected:
                count = f" ({len(q.expected.elements)} elementos)" if q.expected.elements else ""
                lines.append(f"      Respuesta esperada{count}: {q.expected.text}")
            else:
                lines.append(f"      Modos: {', '.join(MODES[m] for m in q.modes)}")
                lines.append(f"      Busca: {looks_for_text(q)}")
            lines.append(f"      Criterio: {criterion_text(q)}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="py -m src.validation.question_catalog",
        description="Muestra el catálogo de preguntas de competencia (S9-T03).")
    parser.parse_args(argv)
    try:
        catalog = load_catalog(CATALOG_FILE)
    except CatalogError as err:
        print(err)
        return 1
    print(render_console(catalog))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
