"""The AI proposes the SKOS mapping type of every DPV candidate and justifies it (S4-T08).

Order agreed in the Scrumban row: the candidates are computed first, WITHOUT the AI (S4-T04/05);
the AI only acts afterwards, on the candidates already in ``alignment-candidates.json``
(S4-T07). For each concept it receives:
- the concept: identifier, label, kinds, definition;
- one law article as evidence: the concept's own source article for the AI proposals, or the
  article retrieved with the RAG (ChromaDB, Sprint 2) for the OntoPriv entities;
- its top-k DPV candidates with label, definition and direct DPV parents (the parents help to
  tell a broader match from an exact one).
It answers, per candidate, one SKOS mapping property (the Sprint 5 graph uses them) or
``none``, plus a one-sentence justification in Spanish. Nothing is approved here: every row
stays ``pending``; the human decides in Sprint 5.

Direction of the relation: always FROM the concept TO the DPV candidate. ``skos:broadMatch``
means the DPV candidate is MORE GENERAL than the concept (e.g. a specific right ->
``dpv:DataSubjectRight``); ``skos:narrowMatch`` means the DPV candidate is MORE SPECIFIC.

Design forced by the free-tier LLM (same approach as the Sprint 3 extraction):
- BATCHING: ``justify_batch_size`` concepts per call (~10K input tokens per call).
- CALL LIMIT: at most ``justify_max_calls`` calls per run (daily quota); re-run to continue.
- PAUSE: ``justify_pause_seconds`` between calls, to stay under the per-minute limits
  (gemini-3.1-flash-lite free tier: 15 requests and 250K tokens per minute, 500 per day).
- RESUME: the candidates file is rewritten after every batch; rows already typed are never sent
  again. A quota error (429) waits once and retries; if it persists, the run stops cleanly.
  A temporary server error (503 "high demand", 500, overloaded) also waits once and retries;
  if it persists, only that batch is skipped (it stays pending for the next run).
- JSONL answers parsed leniently; a relation outside the list becomes ``untyped`` (never a
  crash); a candidate the AI did not answer stays pending for the next run.
The LLM, the article lookup and the retriever are injected, so the fast tests use fakes and
never touch the network; ``_run`` uses the real Sprint 2 ``LLMClient`` and ``retrieve_articles``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.core.dpv_proximity import _humanize, _trim
from src.core.extract import _parse_jsonl, _is_quota_error
from src.alignment.export import (CandidateTable, write_candidate_files, load_candidate_file,
                                  CANDIDATES_JSON)

# SKOS mapping properties (stored value -> Spanish label for console and web).
RELATIONS = {
    "skos:exactMatch": "equivalente",
    "skos:closeMatch": "casi equivalente",
    "skos:broadMatch": "el DPV es mas general",
    "skos:narrowMatch": "el DPV es mas especifico",
    "skos:relatedMatch": "relacionado",
    "none": "sin correspondencia",
}
RELATION_UNTYPED = "untyped"        # the AI answered something outside RELATIONS

_ALIASES = {
    "exactmatch": "skos:exactMatch", "closematch": "skos:closeMatch",
    "broadmatch": "skos:broadMatch", "narrowmatch": "skos:narrowMatch",
    "relatedmatch": "skos:relatedMatch", "none": "none", "nomatch": "none",
    "sin_correspondencia": "none", "sincorrespondencia": "none",
}

DEFAULT_BATCH_SIZE = 15
DEFAULT_MAX_CALLS = 60
DEFAULT_PAUSE_SECONDS = 5.0
DEFAULT_ARTICLE_CHARS = 1200
DEFAULT_MAX_TOKENS = 8192
QUOTA_WAIT_SECONDS = 60
TRANSIENT_WAIT_SECONDS = 20

_INSTRUCTION = (
    "Te entrego conceptos de una ontologia de proteccion de datos, cada uno con un id (C1, C2...), "
    "su definicion, un articulo de la ley como evidencia y sus candidatos del Data Privacy "
    "Vocabulary (DPV). Para CADA candidato decide la relacion SKOS del concepto HACIA el "
    "candidato: 'exactMatch' (mismo significado), 'closeMatch' (casi el mismo, con matices), "
    "'broadMatch' (el candidato DPV es MAS GENERAL que el concepto), 'narrowMatch' (el candidato "
    "DPV es MAS ESPECIFICO que el concepto), 'relatedMatch' (relacionados sin jerarquia) o "
    "'none' (no hay correspondencia). Devuelve UNA LINEA por candidato; cada linea es un objeto "
    "JSON compacto con estas claves exactas: 'concept' (el id, p. ej. C1), 'dpv' (el nombre del "
    "candidato tal como aparece antes de la barra), 'relation' (uno de los seis valores) y "
    "'justification' (una sola oracion en espanol que se apoye en las definiciones o en el "
    "articulo). No uses saltos de linea dentro de un valor. No agregues texto adicional ni "
    "marcadores de codigo."
)


@dataclass(frozen=True)
class JustifySettings:
    batch_size: int = DEFAULT_BATCH_SIZE
    max_calls: int = DEFAULT_MAX_CALLS
    article_chars: int = DEFAULT_ARTICLE_CHARS
    max_tokens: int = DEFAULT_MAX_TOKENS
    pause_seconds: float = DEFAULT_PAUSE_SECONDS


def justify_settings(cfg: dict | None = None) -> JustifySettings:
    """Read ``justify_*`` keys of the ``alignment`` block of config.yaml (defaults if missing)."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    block = (cfg or {}).get("alignment", {}) or {}
    s = JustifySettings(
        batch_size=int(block.get("justify_batch_size", DEFAULT_BATCH_SIZE)),
        max_calls=int(block.get("justify_max_calls", DEFAULT_MAX_CALLS)),
        article_chars=int(block.get("justify_article_chars", DEFAULT_ARTICLE_CHARS)),
        max_tokens=int(block.get("justify_max_tokens", DEFAULT_MAX_TOKENS)),
        pause_seconds=float(block.get("justify_pause_seconds", DEFAULT_PAUSE_SECONDS)),
    )
    if (s.batch_size < 1 or s.max_calls < 1 or s.article_chars < 100 or s.max_tokens < 256
            or s.pause_seconds < 0):
        raise ValueError("Valores de justify_* invalidos en el bloque alignment de config.yaml.")
    return s


def _is_transient_error(message: str) -> bool:
    """Server-side, temporary failures (not our quota): worth one retry after a short wait."""
    m = message.lower()
    return any(k in m for k in ("503", "500", "unavailable", "overloaded", "high demand",
                                "deadline", "timeout", "timed out"))


_BATCH_ID = re.compile(r"\s*\(\s*C\d+\s*\)")


def clean_justification(text) -> str | None:
    """Drop the batch ids the AI sometimes copies into its sentence ('... seguridad (C4) son')."""
    if text is None:
        return None
    cleaned = re.sub(r"\s{2,}", " ", _BATCH_ID.sub("", str(text))).strip()
    return cleaned or None


def normalize_relation(value) -> str:
    """'skos:ExactMatch', 'exactMatch', 'EXACTMATCH' -> 'skos:exactMatch'; unknown -> 'untyped'."""
    key = str(value or "").strip().lower().replace("skos:", "").replace(" ", "")
    return _ALIASES.get(key, RELATION_UNTYPED)


@dataclass
class JustifyReport:
    calls: int = 0
    concepts_sent: int = 0
    rows_filled: int = 0
    rows_untyped: int = 0
    rows_unanswered: int = 0
    quota_exhausted: bool = False
    call_limit_reached: bool = False
    errors: list[dict] = field(default_factory=list)       # {concepts, error}
    remaining_concepts: int = 0
    json_path: str | None = None
    csv_path: str | None = None


def _pending_concepts(rows: list[dict]) -> list[str]:
    """Concept keys (in file order) that must be justified and still have untyped rows."""
    order: dict[str, bool] = {}                      # insertion-ordered: key -> has pending row
    for r in rows:
        if not r.get("needs_justification"):
            continue
        k = r["concept_key"]
        order[k] = order.get(k, False) or r.get("proposed_relation") is None
    return [k for k, pending in order.items() if pending]


def _evidence(first: dict, article_lookup, retrieve_fn) -> dict | None:
    """AI proposals: their own source article. OntoPriv entities: the article retrieved by RAG."""
    articles = first.get("concept_articles") or []
    if articles and article_lookup is not None:
        found = article_lookup(int(articles[0]))
        if found:
            return found
    if retrieve_fn is not None:
        query = ". ".join(p for p in (_humanize(first["concept_name"]), first.get("concept_label"),
                                      first.get("concept_definition")) if p)
        hits = retrieve_fn(query)
        if hits:
            return hits[0]
    return None


def _concept_block(cid: str, rows: list[dict], evidence: dict | None, article_chars: int) -> str:
    first = rows[0]
    kinds = ", ".join(first.get("concept_kinds") or [])
    lines = [f"=== {cid} ===",
             f"Concepto: {first['concept_name']} (tipo: {kinds})"]
    if first.get("concept_label"):
        lines.append(f"Etiqueta: {first['concept_label']}")
    if first.get("concept_definition"):
        lines.append(f"Definicion: {_trim(first['concept_definition'], 400)}")
    if evidence:
        lines.append(f"Articulo {evidence.get('number')} de la ley ({evidence.get('title') or ''}): "
                     f"{_trim(evidence.get('text') or '', article_chars)}")
    lines.append("Candidatos DPV:")
    for r in sorted(rows, key=lambda x: x["rank"]):
        parents = ", ".join(r.get("dpv_parents") or []) or "-"
        definition = _trim(r.get("dpv_definition") or "", 300) or "-"
        lines.append(f"- {r['dpv_name']} | {r['dpv_label']} | definicion: {definition} | "
                     f"padres en el DPV: {parents}")
    return "\n".join(lines)


def _save(data: dict, out_dir, report: JustifyReport, model: str | None) -> None:
    meta = dict(data.get("metadata", {}))
    meta["justification"] = {
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "llm_model": model,
        "remaining_concepts": report.remaining_concepts,
    }
    data["metadata"] = meta
    table = CandidateTable(rows=data["rows"], metadata=meta)
    json_path, csv_path = write_candidate_files(table, out_dir)
    report.json_path, report.csv_path = str(json_path), str(csv_path)


def justify_candidates(data: dict, llm=None, article_lookup=None, retrieve_fn=None,
                       settings: JustifySettings | None = None, out_dir=None,
                       sleep_fn=time.sleep, model_name: str | None = None) -> JustifyReport:
    """Ask the AI for the SKOS type + justification of every pending candidate, in batches.

    ``data`` is the candidates JSON already loaded (it is updated in place). When ``out_dir`` is
    given, the JSON and CSV are rewritten after every batch, so an interrupted run loses nothing.
    """
    if llm is None:                                  # lazy import: tests need no network/key
        from src.ai.llm_client import LLMClient
        llm = LLMClient()
        model_name = model_name or getattr(llm, "model", None)
    if settings is None:
        settings = justify_settings()

    rows = data["rows"]
    for r in rows:                                   # files written before this column existed
        r.setdefault("evidence_article", None)
        if r.get("justification"):                    # tidy rows justified by earlier runs
            r["justification"] = clean_justification(r["justification"])
    by_concept: dict[str, list[dict]] = {}
    for r in rows:
        by_concept.setdefault(r["concept_key"], []).append(r)

    report = JustifyReport()
    pending = _pending_concepts(rows)

    for start in range(0, len(pending), settings.batch_size):
        if report.calls >= settings.max_calls:
            report.call_limit_reached = True
            break
        if report.calls > 0 and settings.pause_seconds > 0:
            sleep_fn(settings.pause_seconds)         # stay under the per-minute limits
        keys = pending[start:start + settings.batch_size]
        ids, blocks, evidence_of = {}, [], {}
        for n, key in enumerate(keys, start=1):
            cid = f"C{n}"
            ids[cid] = key
            concept_rows = [r for r in by_concept[key] if r.get("needs_justification")]
            ev = _evidence(concept_rows[0], article_lookup, retrieve_fn)
            evidence_of[key] = ev.get("number") if ev else None
            blocks.append(_concept_block(cid, concept_rows, ev, settings.article_chars))
        context = "\n\n".join(blocks)

        raw, err = None, None
        for attempt in range(2):                     # one retry after a 429 or a 503
            try:
                raw = llm.ask(prompt=_INSTRUCTION, context=context,
                              max_tokens=settings.max_tokens) or ""
                err = None
                break
            except Exception as exc:                 # noqa: BLE001 - report, don't crash
                err = str(exc)
                if attempt == 0 and _is_quota_error(err):
                    sleep_fn(QUOTA_WAIT_SECONDS)
                    continue
                if attempt == 0 and _is_transient_error(err):
                    sleep_fn(TRANSIENT_WAIT_SECONDS)
                    continue
                break
        report.calls += 1
        if err is not None:
            report.errors.append({"concepts": [by_concept[k][0]["concept_name"] for k in keys],
                                  "error": err[:300]})
            if _is_quota_error(err):
                report.quota_exhausted = True
                break
            continue

        report.concepts_sent += len(keys)
        answers, _bad = _parse_jsonl(raw)
        for ans in answers:
            key = ids.get(str(ans.get("concept") or "").strip())
            if key is None:
                continue
            dpv = str(ans.get("dpv") or "").strip()
            row = next((r for r in by_concept[key] if r.get("needs_justification")
                        and r["dpv_name"] == dpv and r.get("proposed_relation") is None), None)
            if row is None:
                continue
            relation = normalize_relation(ans.get("relation"))
            row["proposed_relation"] = relation
            row["justification"] = clean_justification(ans.get("justification"))
            row["evidence_article"] = evidence_of[key]
            report.rows_filled += 1
            if relation == RELATION_UNTYPED:
                report.rows_untyped += 1
        report.rows_unanswered += sum(1 for k in keys for r in by_concept[k]
                                      if r.get("needs_justification")
                                      and r.get("proposed_relation") is None)
        report.remaining_concepts = len(_pending_concepts(rows))
        if out_dir is not None:
            _save(data, out_dir, report, model_name)

    report.remaining_concepts = len(_pending_concepts(rows))
    if out_dir is not None:
        _save(data, out_dir, report, model_name)
    return report


def relation_counts(rows: list[dict]) -> dict:
    """How many justified rows fall in each relation (Spanish labels), for console and web."""
    out = {label: 0 for label in RELATIONS.values()}
    out["sin tipo valido"] = 0
    for r in rows:
        rel = r.get("proposed_relation")
        if rel is None:
            continue
        out[RELATIONS.get(rel, "sin tipo valido")] += 1
    return out


def render_console(report: JustifyReport, rows: list[dict] | None = None) -> str:
    lines = [
        "Tipo SKOS y justificacion propuestos por la IA (todo sigue 'por validar'):",
        (f"  Llamadas al LLM: {report.calls} | conceptos enviados: {report.concepts_sent} | "
         f"filas con tipo: {report.rows_filled} (sin tipo valido: {report.rows_untyped}) | "
         f"sin respuesta: {report.rows_unanswered}"),
        f"  Conceptos pendientes para la proxima corrida: {report.remaining_concepts}",
    ]
    if rows is not None:
        lines.append("  Tipos propuestos hasta ahora:")
        for label, n in relation_counts(rows).items():
            lines.append(f"    - {label}: {n}")
    if report.call_limit_reached:
        lines.append("  Se alcanzo el limite de llamadas de esta corrida (justify_max_calls). "
                     "Vuelve a ejecutar para continuar donde quedo.")
    if report.quota_exhausted:
        lines.append("  CUOTA AGOTADA: se detuvo la corrida. Vuelve a ejecutar mas tarde; los "
                     "conceptos ya justificados no se vuelven a enviar.")
    for e in report.errors[:5]:
        lines.append(f"  Error en lote ({', '.join(e['concepts'][:3])}...): {e['error'][:120]}")
    if report.json_path:
        lines.append(f"  Actualizado: {report.json_path}")
        lines.append(f"  Actualizado: {report.csv_path}")
    return "\n".join(lines)


def _run() -> None:
    """Justify the pending candidates of data/output/alignment-candidates.json (hits the LLM)."""
    from src.config import load_config
    from src.ingest.text_loader import load_legal_text
    from src.ingest.legal_segmenter import segment_articles
    from src.ai.rag.retriever import retrieve_articles

    cfg = load_config()
    out_dir = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    law_path = cfg.get("inputs", {}).get("legal_text", "data/input/lopdp.pdf")
    candidates = out_dir / CANDIDATES_JSON
    if not candidates.exists():
        print(f"No existe {candidates}. Corre primero: py -m src.alignment.export")
        return
    if not Path(law_path).exists():
        print(f"No se encontro el texto de la ley en '{law_path}'.")
        return
    articles = segment_articles(load_legal_text(law_path).text,
                                source=Path(law_path).stem)["articles"]
    by_number = {a.get("number"): a for a in articles}

    data = load_candidate_file(candidates)
    report = justify_candidates(
        data,
        article_lookup=by_number.get,
        retrieve_fn=lambda q: retrieve_articles(q, top_k=1),
        out_dir=out_dir,
    )
    print(render_console(report, data["rows"]))


if __name__ == "__main__":
    _run()