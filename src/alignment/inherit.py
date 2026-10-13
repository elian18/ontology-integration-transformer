"""Inherit the validated alignment of the base for an uploaded ontology (S6-T05).

In Sprint 5 a person decided, once, the DPV alignment of the 552 base concepts (527 OntoPriv
entities + 25 new concepts from the LOPDP): 173 aligned (190 SKOS correspondences) and 379
"sin correspondencia". When a person uploads their own ontology, every concept that is the SAME
as a base concept receives that decision as it is, so the user does not review anything for
it. Only the rest goes to the automatic proposals (S6-T06).

How "the same" is decided, in order (the first rule that holds wins):
1. ``same_iri``: the concept has the IRI of a base concept (e.g. an ontology that reuses
   OntoPriv terms, or OntoPriv itself: data/input/sample.ttl inherits all 527 entities).
2. ``same_name``: its normalized name or label equals the name, label or an ALIAS of a base
   concept ('Data_Controller' = 'DataController' = 'data controller'). The aliases are the
   names of the 66 AI concepts a person confirmed as duplicates of an OntoPriv entity in
   Sprint 5 (e.g. 'BusinessVolume' for 'Turnover'), so they also catch other wordings.
3. ``similar``: the same rule as the Sprint 4 duplicate detector (S4-T06), with its measured
   thresholds from config.yaml: combined score (0.4 lexical + 0.6 embeddings) >=
   ``alignment.duplicate_threshold`` (0.75) or lexical >= ``alignment.duplicate_name_match``
   (0.95). On the LOPDP run, 33 of the 34 concepts it marked were confirmed by a person.
Kinds must be compatible (a class only matches a class, a property a property). When the best
match is a tie between base concepts with DIFFERENT decisions, nothing is inherited: the
concept goes to the automatic proposals instead (``ambiguous``).

What is inherited and how it is labelled: the DPV decision itself was validated by a person
(who and when travel with it); the equivalence between the user concept and the base concept
is automatic and is recorded with its rule and score, so the output can say "heredada de
OntoPriv:Consent (mismo nombre)".

Embeddings are only computed for the concepts that rules 1 and 2 did not match (sample.ttl
needs none). The result is saved in the workspace (``work/inherited.json``).

CLI: ``py -m src.alignment.inherit ARCHIVO [--sin-embeddings]``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from src.core.emit import PROFILE_IRI
from src.alignment.ai_concepts import ai_concepts, concept_decisions, confirmed_duplicates
from src.alignment.assistant_review import PROPOSED_BY
from src.alignment.candidates import alignment_settings
from src.alignment.decisions import DEFAULT_SOURCE_ID, decisions_path, read_log
from src.alignment.duplicates import duplicate_threshold, name_match_threshold
from src.alignment.export import CANDIDATES_JSON, load_candidate_file
from src.alignment.lexical import lexical_matrix, normalize_for_lexical
from src.alignment.mapping_review import (STATUS_ALIGNED, STATUS_NO_MATCH, approved_mappings,
                                          concept_status, mapping_concepts, mapping_decisions)
from src.alignment.sources import SourceConcept, build_alignment_sources

METHOD_SAME_IRI = "same_iri"
METHOD_SAME_NAME = "same_name"
METHOD_SIMILAR = "similar"
METHOD_LABELS = {METHOD_SAME_IRI: "mismo IRI", METHOD_SAME_NAME: "mismo nombre",
                 METHOD_SIMILAR: "muy parecido"}
RESULT_FILE = "inherited.json"
_ALIGNABLE = {"class", "property"}


# --------------------------------------------------------------------------- the base

@dataclass
class BaseConcept:
    """A base concept with its validated decision (Sprint 5)."""
    key: str
    iri: str
    origin: str                       # ontology | ai (new concept of the Ecuador profile)
    name: str
    label: str | None
    definition: str | None
    kinds: tuple[str, ...]
    status: str                       # aligned | no_match
    mappings: list[dict] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)    # names of confirmed AI duplicates

    def semantic_text(self) -> str:
        parts = [self.name.replace("_", " ")]
        if self.label:
            parts.append(self.label)
        if self.definition:
            parts.append(self.definition[:300])
        return ". ".join(parts)

    def decision_signature(self) -> tuple:
        return (self.status, tuple(sorted((m["dpv_iri"], m["relation"])
                                          for m in self.mappings)))


@dataclass
class BaseIndex:
    concepts: list[BaseConcept]
    source_id: str
    pending: int = 0                  # base concepts still undecided (should be 0)

    def by_iri(self) -> dict[str, BaseConcept]:
        return {c.iri: c for c in self.concepts}

    def name_index(self) -> dict[str, list[BaseConcept]]:
        """Normalized name / label / alias -> base concepts that carry it."""
        out: dict[str, list[BaseConcept]] = {}
        for c in self.concepts:
            keys = {normalize_for_lexical(t) for t in [c.name, c.label, *c.aliases] if t}
            for k in keys - {""}:
                out.setdefault(k, []).append(c)
        return out


def base_paths(cfg: dict | None = None) -> dict[str, Path]:
    from src.config import ROOT, load_config
    cfg = cfg if cfg is not None else load_config()
    out_dir = Path(cfg.get("outputs", {}).get("dir", "data/output"))
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    return {"candidates": out_dir / CANDIDATES_JSON, "log": decisions_path(cfg)}


def load_base(candidates_path: str | Path | None = None, log_path: str | Path | None = None,
              source_id: str = DEFAULT_SOURCE_ID, profile_iri: str = PROFILE_IRI,
              cfg: dict | None = None) -> BaseIndex:
    """The validated base: candidates file of Sprint 4 + decisions log of Sprint 5."""
    paths = base_paths(cfg) if candidates_path is None or log_path is None else {}
    cand = Path(candidates_path or paths["candidates"])
    log = Path(log_path or paths["log"])
    if not cand.exists():
        raise FileNotFoundError(f"No está el archivo de candidatos de la base: {cand}")
    data = load_candidate_file(cand)
    records = read_log(log)
    concepts = mapping_concepts(data, records, source_id, profile_iri)
    current = mapping_decisions(records, source_id)

    mappings: dict[str, list[dict]] = {}
    for m in approved_mappings(concepts, current):
        d = current[m["concept_key"]][m["dpv_iri"]]
        mappings.setdefault(m["concept_key"], []).append({
            "dpv_iri": m["dpv_iri"], "dpv_name": m["dpv_name"], "relation": m["relation"],
            "reviewer": m["reviewer"], "decided_at": m["decided_at"],
            "decided_by": ("assistant_confirmed" if d.snapshot.get("proposed_by") == PROPOSED_BY
                           else "person")})
    aliases: dict[str, list[str]] = {}
    ai = ai_concepts(data, profile_iri)
    labels = {c.key: c.label for c in ai}
    for dup in confirmed_duplicates(ai, concept_decisions(records, source_id)):
        if dup["duplicate_of"]:
            aliases.setdefault(dup["duplicate_of"], []).extend(
                t for t in (dup["name"], labels.get(dup["key"])) if t)

    out, pending = [], 0
    for c in concepts:
        status = concept_status(c, current)
        if status not in (STATUS_ALIGNED, STATUS_NO_MATCH):
            pending += 1
            continue
        out.append(BaseConcept(
            key=c.key, iri=c.iri, origin=c.origin, name=c.name, label=c.label,
            definition=c.definition, kinds=tuple(k for k in c.kinds if k in _ALIGNABLE),
            status=status, mappings=mappings.get(c.key, []),
            aliases=sorted(set(aliases.get(c.key, [])))))
    return BaseIndex(concepts=out, source_id=source_id, pending=pending)


# --------------------------------------------------------------------------- matching

@dataclass
class Inherited:
    """A user concept that receives the decision of a base concept."""
    key: str
    name: str
    kinds: list[str]
    method: str
    score: float
    base_key: str
    base_iri: str
    base_name: str
    base_origin: str
    status: str                       # aligned | no_match (validated in Sprint 5)
    mappings: list[dict] = field(default_factory=list)


@dataclass
class InheritReport:
    inherited: list[Inherited] = field(default_factory=list)
    not_inherited: list[dict] = field(default_factory=list)   # go to S6-T06
    total: int = 0
    base_concepts: int = 0
    settings: dict = field(default_factory=dict)

    def summary(self) -> dict:
        by_method = Counter(i.method for i in self.inherited)
        by_status = Counter(i.status for i in self.inherited)
        reasons = Counter(n["reason"] for n in self.not_inherited)
        return {
            "concepts": self.total,
            "inherited": len(self.inherited),
            "inherited_aligned": by_status.get(STATUS_ALIGNED, 0),
            "inherited_no_match": by_status.get(STATUS_NO_MATCH, 0),
            "inherited_mappings": sum(len(i.mappings) for i in self.inherited),
            "by_method": {m: by_method.get(m, 0) for m in METHOD_LABELS},
            "not_inherited": len(self.not_inherited),
            "not_inherited_reasons": dict(reasons),
            "base_concepts": self.base_concepts,
        }

    def to_dict(self) -> dict:
        return {"summary": self.summary(), "settings": self.settings,
                "inherited": [asdict(i) for i in self.inherited],
                "not_inherited": self.not_inherited}


def _compatible(user_kinds, base_kinds) -> bool:
    return bool(set(user_kinds) & set(base_kinds) & _ALIGNABLE)


def _pick(candidates: list[BaseConcept]) -> BaseConcept | None:
    """One base concept, or None when the tied candidates disagree on the decision."""
    if not candidates:
        return None
    signatures = {c.decision_signature() for c in candidates}
    return candidates[0] if len(signatures) == 1 else None


def _inherited(user: SourceConcept, base: BaseConcept, method: str, score: float) -> Inherited:
    return Inherited(key=user.key, name=user.name, kinds=list(user.kinds), method=method,
                     score=round(float(score), 4), base_key=base.key, base_iri=base.iri,
                     base_name=base.name, base_origin=base.origin, status=base.status,
                     mappings=[dict(m) for m in base.mappings])


def _embed(texts: list[str], embed_fn) -> np.ndarray:
    if embed_fn is None:                      # lazy: tests and rules 1-2 never load the model
        from src.ai.rag.embeddings import embed as embed_fn
    return np.asarray(embed_fn(texts), dtype=float)


def inherit(user_concepts: list[SourceConcept], base: BaseIndex, *, embed_fn=None,
            use_embeddings: bool = True, cfg: dict | None = None) -> InheritReport:
    """Apply the three rules (see the module docstring) to every user concept."""
    settings = alignment_settings(cfg)
    threshold, name_match = duplicate_threshold(cfg), name_match_threshold(cfg)
    report = InheritReport(total=len(user_concepts), base_concepts=len(base.concepts),
                           settings={"duplicate_threshold": threshold,
                                     "duplicate_name_match": name_match,
                                     "weight_lexical": settings.weight_lexical,
                                     "weight_semantic": settings.weight_semantic,
                                     "embeddings": use_embeddings,
                                     "base_source_id": base.source_id})
    by_iri, names = base.by_iri(), base.name_index()
    rest: list[SourceConcept] = []
    for user in user_concepts:
        if not set(user.kinds) & _ALIGNABLE:
            report.not_inherited.append({"key": user.key, "name": user.name,
                                         "reason": "kind"})
            continue
        b = by_iri.get(user.iri or user.key)
        if b is not None and _compatible(user.kinds, b.kinds):
            report.inherited.append(_inherited(user, b, METHOD_SAME_IRI, 1.0))
            continue
        texts = {normalize_for_lexical(t) for t in (user.name, user.label) if t} - {""}
        found = [c for t in texts for c in names.get(t, []) if _compatible(user.kinds, c.kinds)]
        unique = list({c.key: c for c in found}.values())
        if unique:
            chosen = _pick(unique)
            if chosen is not None:
                report.inherited.append(_inherited(user, chosen, METHOD_SAME_NAME, 1.0))
            else:
                report.not_inherited.append({"key": user.key, "name": user.name,
                                             "reason": "ambiguous",
                                             "candidates": [c.key for c in unique]})
            continue
        rest.append(user)

    if not rest:
        return report
    pool = [c for c in base.concepts if c.kinds]
    lex = lexical_matrix([u.name_text() for u in rest], [c.name.replace("_", " ") for c in pool])
    if use_embeddings:
        vec_u = _embed([u.semantic_text() for u in rest], embed_fn)
        vec_b = _embed([c.semantic_text() for c in pool], embed_fn)
        sem = np.clip(vec_u @ vec_b.T, 0.0, 1.0)
        combined = settings.weight_lexical * lex + settings.weight_semantic * sem
    else:
        combined = lex * 0.0                 # only the same-name rule can fire
    for i, user in enumerate(rest):
        allowed = [j for j, c in enumerate(pool) if _compatible(user.kinds, c.kinds)]
        if not allowed:
            report.not_inherited.append({"key": user.key, "name": user.name, "reason": "kind"})
            continue
        best_score = max(combined[i, j] for j in allowed)
        best_lex = max(lex[i, j] for j in allowed)
        if use_embeddings and best_score >= threshold:
            tied = [pool[j] for j in allowed if combined[i, j] == best_score]
            score = best_score
        elif best_lex >= name_match:
            tied = [pool[j] for j in allowed if lex[i, j] == best_lex]
            score = best_lex
        else:
            best = best_score if use_embeddings else best_lex
            report.not_inherited.append({"key": user.key, "name": user.name,
                                         "reason": "no_match",
                                         "best_score": round(float(best), 4)})
            continue
        chosen = _pick(tied)
        if chosen is None:
            report.not_inherited.append({"key": user.key, "name": user.name,
                                         "reason": "ambiguous",
                                         "candidates": [c.key for c in tied]})
            continue
        report.inherited.append(_inherited(user, chosen, METHOD_SIMILAR, score))
    return report


# --------------------------------------------------------------------------- workspace

def user_concepts(ontology_path: str | Path) -> list[SourceConcept]:
    """The concepts to align of an uploaded ontology (classes and properties).

    ``proposals={}`` is passed on purpose: with None, build_alignment_sources would add the
    LOPDP concepts proposed by the AI (Sprint 3) to the user's ontology."""
    from src.ingest.ontology_loader import load_ontology
    report = load_ontology(ontology_path)
    return build_alignment_sources(report, proposals={}).concepts


def inherit_for_workspace(ws, *, base: BaseIndex | None = None, embed_fn=None,
                          use_embeddings: bool = True, cfg: dict | None = None
                          ) -> InheritReport:
    """Run the inheritance for a workspace (S6-T04), save it and mark the step."""
    from src.workspace.store import DONE, ERROR, RUNNING
    ws.mark_step("inherit", RUNNING)
    try:
        base = base or load_base(cfg=cfg)
        report = inherit(user_concepts(ws.input_path()), base, embed_fn=embed_fn,
                         use_embeddings=use_embeddings, cfg=cfg)
    except Exception as exc:
        ws.mark_step("inherit", ERROR, str(exc))
        raise
    ws.work_dir.mkdir(parents=True, exist_ok=True)
    (ws.work_dir / RESULT_FILE).write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    s = report.summary()
    ws.mark_step("inherit", DONE, f"{s['inherited']} de {s['concepts']} conceptos heredados",
                 **{k: s[k] for k in ("concepts", "inherited", "inherited_aligned",
                                      "inherited_no_match", "inherited_mappings",
                                      "not_inherited")})
    return report


def load_result(ws) -> dict | None:
    path = ws.work_dir / RESULT_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------------- console

def render_console(report: InheritReport, examples: int = 5) -> str:
    s = report.summary()
    lines = ["Herencia de lo validado (S6-T05)",
             f"  Conceptos de la ontología subida: {s['concepts']}",
             f"  Heredan la decisión de la base: {s['inherited']} "
             f"({s['inherited_aligned']} alineados con {s['inherited_mappings']} "
             f"correspondencias, {s['inherited_no_match']} sin correspondencia)"]
    for method, label in METHOD_LABELS.items():
        lines.append(f"    por {label}: {s['by_method'][method]}")
    reasons = s["not_inherited_reasons"]
    lines.append(f"  Pasan a propuestas automáticas (S6-T06): {s['not_inherited']}"
                 + (f"  ({', '.join(f'{k}: {v}' for k, v in sorted(reasons.items()))})"
                    if reasons else ""))
    shown = [i for i in report.inherited if i.status == STATUS_ALIGNED][:examples]
    if shown:
        lines.append("  Ejemplos:")
    for i in shown:
        maps = ", ".join(f"{m['relation']} dpv:{m['dpv_iri'].rsplit('#', 1)[-1]}"
                         for m in i.mappings)
        lines.append(f"    {i.name} <- {i.base_name} ({METHOD_LABELS[i.method]}): {maps}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.workspace.store import WorkspaceError, open_ontology
    parser = argparse.ArgumentParser(
        prog="py -m src.alignment.inherit",
        description="Hereda las decisiones validadas de la base para una ontología (S6-T05).")
    parser.add_argument("file", help="ontología .rdf, .owl, .ttl o .jsonld")
    parser.add_argument("--sin-embeddings", action="store_true",
                        help="solo reglas de mismo IRI y mismo nombre (sin cargar el modelo)")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.exists():
        print(f"Error: no existe el archivo {path}")
        return 1
    try:
        ws = open_ontology(path.name, path.read_bytes())
        report = inherit_for_workspace(ws, use_embeddings=not args.sin_embeddings)
    except (WorkspaceError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render_console(report))
    print(f"  Guardado en: {ws.work_dir / RESULT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())