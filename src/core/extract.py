"""Extract candidate concepts from the law text and propose them for the profile (AI step).

This is the first place the AI enters the pipeline. It reads the law article by article and
proposes data-protection concepts for the jurisdiction profile, each carrying the article it
came from (provenance). It only PROPOSES: nothing is written into an ontology here. Deciding
which proposals are genuinely new vs. already in OntoPriv-Core, and wiring them in, is the
alignment sprint (S5); this module stops at a reviewed-by-human "por validar" list.

Design forced by the free-tier LLM (Gemini free tier allows ~20 requests/day):
- BATCHING: several articles per call (~8 calls for the whole LOPDP, not 77), to fit the quota.
- JSONL OUTPUT: one compact JSON object per line, so a truncated answer only loses its last
  line instead of failing the whole batch; each line is parsed leniently.
- RESUME: proposals are saved to disk after every batch. If the daily quota runs out (429),
  the run stops cleanly keeping what it had; re-running skips the articles already processed,
  so quota is never spent redoing work.
The LLM is injected (``llm``) so the fast tests use a deterministic fake and never hit the
network; ``_run`` uses the real Sprint 2 ``LLMClient``.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path

_INSTRUCTION = (
    "Te entrego varios articulos de una ley de proteccion de datos, cada uno con su numero. "
    "Extrae los conceptos relevantes (entidades, roles, principios, derechos, obligaciones, "
    "mecanismos). Devuelve UNA LINEA por concepto; cada linea es un objeto JSON compacto con "
    "estas claves exactas: 'name' (identificador corto en ingles, CamelCase, sin espacios), "
    "'label' (el termino en espanol como aparece), 'type' ('class' o 'property'), "
    "'definition' (definicion breve en espanol, en UNA sola linea, basada solo en el articulo), "
    "'article' (el numero del articulo del que proviene). No uses saltos de linea dentro de un "
    "valor. No agregues texto adicional ni marcadores de codigo."
)


@dataclass
class ProposedConcept:
    name: str
    label: str
    ctype: str                       # class | property
    definition: str
    articles: list[int] = field(default_factory=list)   # provenance: source article numbers

    def as_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "type": self.ctype,
                "definition": self.definition, "articles": self.articles}


@dataclass
class ExtractionReport:
    source: str
    concepts: list[ProposedConcept] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)      # {articles, error}
    processed_articles: list[int] = field(default_factory=list)
    quota_exhausted: bool = False

    def counts(self) -> dict:
        return {
            "articles_processed": len(self.processed_articles),
            "concepts_proposed": len(self.concepts),
            "classes": sum(1 for c in self.concepts if c.ctype == "class"),
            "properties": sum(1 for c in self.concepts if c.ctype == "property"),
            "batches_with_errors": len(self.errors),
            "quota_exhausted": self.quota_exhausted,
        }


def _article_fields(article) -> tuple[int | None, str, str]:
    return article.get("number"), (article.get("title") or ""), (article.get("text") or "")


def _is_quota_error(message: str) -> bool:
    m = message.lower()
    return "429" in m or "resource_exhausted" in m or "quota" in m


def _batch_context(batch) -> tuple[str, set[int]]:
    parts, numbers = [], set()
    for article in batch:
        number, title, text = _article_fields(article)
        if not text.strip():
            continue
        if number is not None:
            numbers.add(number)
        header = f"=== ARTICULO {number} ({title}) ===" if number is not None else "=== ARTICULO ==="
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts), numbers


def _parse_jsonl(raw: str) -> tuple[list[dict], int]:
    """Parse one JSON object per line, salvaging what is valid; returns (objects, n_bad)."""
    objects, bad = [], 0
    for line in (raw or "").splitlines():
        line = line.strip().rstrip(",")
        if not line or line in ("[", "]") or line.startswith("```"):
            continue
        if not (line.startswith("{") and line.endswith("}")):
            i, j = line.find("{"), line.rfind("}")
            if i == -1 or j <= i:
                bad += 1
                continue
            line = line[i:j + 1]
        try:
            obj = json.loads(line, strict=False)     # strict=False tolerates stray control chars
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        else:
            bad += 1
    return objects, bad


def _normalize(concept: dict, valid_numbers: set[int]) -> ProposedConcept | None:
    name = (str(concept.get("name") or "")).strip()
    if not name:
        return None
    ctype = str(concept.get("type") or "class").strip().lower()
    if ctype not in ("class", "property"):
        ctype = "class"
    articles: list[int] = []
    raw_article = concept.get("article")
    try:
        num = int(raw_article)
        if num in valid_numbers:
            articles = [num]
    except (TypeError, ValueError):
        pass
    return ProposedConcept(name=name, label=(str(concept.get("label") or "")).strip(),
                           ctype=ctype, definition=(str(concept.get("definition") or "")).strip(),
                           articles=articles)


def _chunk(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _load_state(out_path: Path):
    try:
        data = json.loads(out_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return OrderedDict(), set(), []
    merged: "OrderedDict[str, ProposedConcept]" = OrderedDict()
    for c in data.get("concepts", []):
        merged[c["name"].lower()] = ProposedConcept(
            name=c["name"], label=c.get("label", ""), ctype=c.get("type", "class"),
            definition=c.get("definition", ""), articles=list(c.get("articles", [])))
    return merged, set(data.get("processed_articles", [])), data.get("errors", [])


def _save_state(out_path: Path, report: ExtractionReport) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(extraction_summary(report), indent=2, ensure_ascii=False),
                        encoding="utf-8")


def propose_profile_concepts(articles, llm=None, source: str = "", batch_size: int = 8,
                             max_tokens: int = 4096, retries: int = 1,
                             out_path=None, resume: bool = True) -> ExtractionReport:
    """Extract concepts from the articles in batches and propose them, with provenance."""
    if llm is None:                                 # lazy import so tests need no network/key
        from src.ai.llm_client import LLMClient
        llm = LLMClient()

    out_path = Path(out_path) if out_path else None
    merged: "OrderedDict[str, ProposedConcept]" = OrderedDict()
    processed: set[int] = set()
    errors: list[dict] = []
    if out_path and resume and out_path.exists():
        merged, processed, _old_errors = _load_state(out_path)

    pending = [a for a in articles if a.get("number") not in processed]
    report = ExtractionReport(source=source, concepts=list(merged.values()),
                              errors=errors, processed_articles=sorted(processed))

    for batch in _chunk(pending, batch_size):
        context, numbers = _batch_context(batch)
        if not context.strip():
            continue
        raw, err = None, None
        for attempt in range(retries + 1):
            try:
                raw = llm.ask(prompt=_INSTRUCTION, context=context, max_tokens=max_tokens) or ""
                err = None
                break
            except Exception as exc:                # noqa: BLE001 - report, don't crash
                err = str(exc)
                if _is_quota_error(err):
                    report.quota_exhausted = True
                    break
                if attempt < retries:
                    time.sleep(0.5)
        if report.quota_exhausted:
            errors.append({"articles": sorted(numbers), "error": err})
            break
        if err is not None:
            errors.append({"articles": sorted(numbers), "error": err})
            continue

        parsed, _bad = _parse_jsonl(raw)
        for raw_concept in parsed:
            concept = _normalize(raw_concept, numbers)
            if concept is None:
                continue
            key = concept.name.lower()
            if key in merged:
                for num in concept.articles:
                    if num not in merged[key].articles:
                        merged[key].articles.append(num)
            else:
                merged[key] = concept
        processed.update(numbers)

        report.concepts = list(merged.values())
        report.errors = errors
        report.processed_articles = sorted(processed)
        if out_path:
            _save_state(out_path, report)

    report.concepts = list(merged.values())
    report.errors = errors
    report.processed_articles = sorted(processed)
    if out_path:
        _save_state(out_path, report)
    return report


def extraction_summary(report: ExtractionReport) -> dict:
    """Serializable summary (safe for JSON, the CLI and the web)."""
    return {
        "source": report.source,
        "counts": report.counts(),
        "processed_articles": report.processed_articles,
        "quota_exhausted": report.quota_exhausted,
        "errors": report.errors,
        "concepts": [c.as_dict() for c in report.concepts],
    }


def render_console(report: ExtractionReport) -> str:
    c = report.counts()
    lines = [
        f"Conceptos propuestos para el perfil (por validar): {report.source or 'ley'}",
        (f"  Articulos procesados: {c['articles_processed']} | "
         f"propuestos: {c['concepts_proposed']} "
         f"({c['classes']} clases, {c['properties']} propiedades) | "
         f"lotes con error: {c['batches_with_errors']}"),
        "  Ejemplos:",
    ]
    for concept in report.concepts[:10]:
        arts = ", ".join(str(a) for a in concept.articles) or "s/n"
        lines.append(f"    - {concept.name} ({concept.label}) [{concept.ctype}] <- art. {arts}")
    if report.quota_exhausted:
        lines.append("  CUOTA AGOTADA: se detuvo la corrida. Vuelve a ejecutar mas tarde para "
                     "continuar donde quedo (se saltan los articulos ya procesados).")
    if report.errors:
        lines.append("  Lotes con error:")
        for e in report.errors[:5]:
            lines.append(f"    - art. {e['articles']}: {str(e['error'])[:120]}")
    return "\n".join(lines)


def _run() -> None:
    """Demo: propose profile concepts from the base law named in config.yaml (hits the LLM)."""
    from src.config import load_config
    from src.ingest.text_loader import load_legal_text
    from src.ingest.legal_segmenter import segment_articles

    cfg = load_config()
    law_path = cfg.get("inputs", {}).get("legal_text", "data/input/lopdp.pdf")
    out_dir = cfg.get("outputs", {}).get("dir", "data/output")
    if not Path(law_path).exists():
        print(f"No se encontro el texto de la ley en '{law_path}'.")
        return
    text = load_legal_text(law_path).text
    segmentation = segment_articles(text, source=Path(law_path).stem)
    out_path = Path(out_dir) / "perfil-conceptos-propuestos.json"
    report = propose_profile_concepts(segmentation["articles"], source=segmentation["source"],
                                      out_path=out_path)
    print(render_console(report))
    print(f"  Guardado en: {out_path}")


if __name__ == "__main__":
    _run()