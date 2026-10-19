"""The automatic reference to measure the alignment decider, with no hand labelling (S7-T03).

The reference is what a person already decided, so nothing has to be labelled again:

- ``base``: the 552 decided base concepts of Sprint 5 (173 with a DPV term, 190
  correspondences; 379 "sin correspondencia"). It is used as "leave one out": each concept is
  solved as if it were new and compared with the decision.
- ``gdprtext``: the GDPRtEXT concepts that inherit a decision of the base (22 with the real
  embeddings; S6-T09). Their known answer is the decision of the base concept they inherit.
- ``blind``: the 30-concept blind sample of Sprint 4 (labelled BEFORE the assistant proposals),
  the only reference that does not inherit the bias of the Sprint 5 review.

Leak control (what cannot be shown to the decider as a precedent): when a concept is solved as
new, the decisions of the base concepts that are "the same" would give the answer away (e.g.
OntoPriv has ``Corrective_measures`` in its two namespaces, plus ``ai:CorrectiveMeasures``).
"The same" is decided with the inheritance rules of S6-T05, so the evaluation reproduces
production exactly: a user concept reaches the decider only when no base concept is the same.
Every base concept that satisfies ANY rule is excluded (not only the best one), which is
slightly stricter than production:

1. same IRI;
2. same normalized name or label, or an alias of the 66 confirmed duplicates;
3. very similar: combined score (0.4 lexical + 0.6 embeddings) >= ``duplicate_threshold``
   (0.75) or lexical >= ``duplicate_name_match`` (0.95), kinds compatible.

``score`` measures any system with the same rules: precision, coverage over the aligned
concepts of the reference (173 in the base, not only the 92 whose term was rank 1), F1,
"sin correspondencia" claimed right or wrong, and the SKOS type compared with the person's.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from src.alignment.candidates import alignment_settings
from src.alignment.dpv_targets import DPV_NAMESPACE
from src.alignment.duplicates import duplicate_threshold, name_match_threshold
from src.alignment.inherit import METHOD_SAME_IRI, METHOD_SAME_NAME, METHOD_SIMILAR, BaseIndex
from src.alignment.lexical import lexical_matrix, normalize_for_lexical
from src.alignment.mapping_review import STATUS_ALIGNED, STATUS_NO_MATCH
from src.core.dpv_proximity import _humanize

REFERENCE_FILE = "decision-reference.json"
REF_BASE, REF_GDPRTEXT, REF_BLIND = "base", "gdprtext", "blind"
REF_LABELS = {REF_BASE: "tus 552 decisiones", REF_GDPRTEXT: "GDPRtEXT (heredados)",
              REF_BLIND: "muestra ciega del Sprint 4"}
_ALIGNABLE = {"class", "property"}


# --------------------------------------------------------------------------- data

@dataclass
class RefItem:
    """One reference concept: what the person decided and which precedents are forbidden."""
    key: str
    name: str
    status: str                                   # aligned | no_match
    terms: dict[str, str] = field(default_factory=dict)     # dpv_iri -> SKOS relation
    forbidden: dict[str, str] = field(default_factory=dict)  # base key -> rule that matched
    origin: str = ""


@dataclass
class Reference:
    name: str
    items: list[RefItem] = field(default_factory=list)

    def by_key(self) -> dict[str, RefItem]:
        return {i.key: i for i in self.items}

    def aligned(self) -> int:
        return sum(i.status == STATUS_ALIGNED for i in self.items)

    def summary(self) -> dict:
        with_forbidden = [i for i in self.items if i.forbidden]
        rules: dict[str, int] = {}
        for i in self.items:
            for rule in i.forbidden.values():
                rules[rule] = rules.get(rule, 0) + 1
        return {"concepts": len(self.items), "aligned": self.aligned(),
                "no_match": len(self.items) - self.aligned(),
                "correspondences": sum(len(i.terms) for i in self.items),
                "with_forbidden": len(with_forbidden),
                "forbidden_pairs": sum(len(i.forbidden) for i in self.items),
                "forbidden_by_rule": rules}

    def to_dict(self) -> dict:
        return {"name": self.name, "summary": self.summary(),
                "items": [asdict(i) for i in self.items]}


@dataclass(frozen=True)
class Probe:
    """A concept to compare against the base with the inheritance rules."""
    key: str
    iri: str | None
    name: str
    label: str | None
    kinds: tuple[str, ...]


def probe_of(concept) -> Probe:
    """A Probe from a SourceConcept or a BaseConcept."""
    return Probe(key=concept.key, iri=getattr(concept, "iri", None) or None, name=concept.name,
                 label=concept.label, kinds=tuple(k for k in concept.kinds if k in _ALIGNABLE))


# --------------------------------------------------------------------------- "the same"

def _compatible(a, b) -> bool:
    return bool(set(a) & set(b) & _ALIGNABLE)


def same_concepts(probes: list[Probe], base: BaseIndex, probe_vectors, base_vectors,
                  cfg: dict | None = None, *, exclude_self: bool = True) -> list[dict[str, str]]:
    """For every probe, ALL base concepts the inheritance rules consider the same as it.

    Returns one dict per probe: base key -> rule (same_iri | same_name | similar). The vectors
    are the embeddings of the texts the inheritance compares (rows in the order of ``probes``
    and of ``base.concepts``); ``None`` disables rule 3's semantic part (lexical >= 0.95 still
    applies), as ``inherit(..., use_embeddings=False)`` does."""
    settings = alignment_settings(cfg)
    threshold, name_match = duplicate_threshold(cfg), name_match_threshold(cfg)
    pool = base.concepts
    by_iri = base.by_iri()
    names = base.name_index()
    lex = lexical_matrix([_humanize(p.name) for p in probes],
                         [c.name.replace("_", " ") for c in pool])
    if probe_vectors is not None and base_vectors is not None:
        sem = np.clip(np.asarray(probe_vectors, float) @ np.asarray(base_vectors, float).T, 0, 1)
        combined = settings.weight_lexical * lex + settings.weight_semantic * sem
    else:
        combined = None
    out = []
    for i, p in enumerate(probes):
        found: dict[str, str] = {}
        b = by_iri.get(p.iri or p.key)
        if b is not None and _compatible(p.kinds, b.kinds):
            found[b.key] = METHOD_SAME_IRI
        for text in {normalize_for_lexical(t) for t in (p.name, p.label) if t} - {""}:
            for c in names.get(text, []):
                if _compatible(p.kinds, c.kinds):
                    found.setdefault(c.key, METHOD_SAME_NAME)
        for j, c in enumerate(pool):
            if c.key in found or not _compatible(p.kinds, c.kinds):
                continue
            if lex[i, j] >= name_match or (combined is not None and combined[i, j] >= threshold):
                found[c.key] = METHOD_SIMILAR
        if exclude_self:
            found.pop(p.key, None)
        out.append(found)
    return out


# --------------------------------------------------------------------------- the references

def _terms(mappings) -> dict[str, str]:
    return {m["dpv_iri"]: m.get("relation") for m in mappings}


def base_reference(base: BaseIndex, base_vectors=None, cfg: dict | None = None) -> Reference:
    """The 552 decided base concepts, each with the base concepts it cannot see as precedent."""
    probes = [probe_of(c) for c in base.concepts]
    forbidden = same_concepts(probes, base, base_vectors, base_vectors, cfg)
    return Reference(REF_BASE, [RefItem(key=c.key, name=c.name, status=c.status,
                                        terms=_terms(c.mappings), forbidden=f, origin=c.origin)
                                for c, f in zip(base.concepts, forbidden)])


def external_reference(concepts, inherited: dict, base: BaseIndex, concept_vectors=None,
                       base_vectors=None, cfg: dict | None = None,
                       name: str = REF_GDPRTEXT) -> Reference:
    """The concepts of an external ontology that inherited a decision (their known answer).

    ``concepts`` are the SourceConcepts of that ontology, ``inherited`` the result of the
    inheritance (``InheritReport.to_dict()``) and ``concept_vectors`` their embeddings in the
    order of ``concepts``. The base concept each one inherited from is always forbidden."""
    known = {i["key"]: i for i in inherited.get("inherited", [])}
    chosen = [(n, c) for n, c in enumerate(concepts) if c.key in known]
    vectors = (None if concept_vectors is None
               else np.asarray(concept_vectors, float)[[n for n, _ in chosen]])
    forbidden = same_concepts([probe_of(c) for _, c in chosen], base, vectors, base_vectors, cfg,
                              exclude_self=False)
    items = []
    for (_, c), f in zip(chosen, forbidden):
        k = known[c.key]
        f = {**f, k["base_key"]: f.get(k["base_key"], k["method"])}
        items.append(RefItem(key=c.key, name=c.name, status=k["status"],
                             terms=_terms(k.get("mappings", [])), forbidden=f,
                             origin=k["base_key"]))
    return Reference(name, sorted(items, key=lambda i: i.name.lower()))


def blind_reference(sample: list[dict], base_ref: Reference, base: BaseIndex,
                    concepts_by_key: dict, concept_vectors: dict | None = None,
                    base_vectors=None, cfg: dict | None = None) -> Reference:
    """The blind sample of Sprint 4 (gold term + relation, or 'none').

    A concept that is in the base takes its forbidden precedents from ``base_ref``. The ones
    that are not (7 AI concepts later confirmed as duplicates of an OntoPriv entity) get the
    base concepts that are the same as them, so their OntoPriv twin is never a precedent.
    ``concepts_by_key`` holds their SourceConcepts and ``concept_vectors`` their embeddings."""
    in_base = base_ref.by_key()
    items, outside = [], []
    for row in sample:
        gold = (row.get("gold_dpv_name") or "").strip()
        if not gold:
            continue
        rel = (row.get("gold_relation") or "").strip()
        terms = {} if gold.lower() == "none" else {
            DPV_NAMESPACE + gold: rel if rel.startswith("skos:") else f"skos:{rel}"}
        item = RefItem(key=row["concept_key"], name=row.get("concept_name") or row["concept_key"],
                       status=STATUS_ALIGNED if terms else STATUS_NO_MATCH, terms=terms,
                       origin=row.get("origin") or "")
        if item.key in in_base:
            item.forbidden = dict(in_base[item.key].forbidden)
        else:
            outside.append(item)
        items.append(item)
    concepts = [concepts_by_key[i.key] for i in outside if i.key in concepts_by_key]
    if concepts:
        vecs = (None if concept_vectors is None
                else np.asarray([concept_vectors[c.key] for c in concepts], float))
        found = same_concepts([probe_of(c) for c in concepts], base, vecs, base_vectors, cfg)
        by_key = {c.key: f for c, f in zip(concepts, found)}
        for it in outside:
            it.forbidden = by_key.get(it.key, {})
    return Reference(REF_BLIND, items)


# --------------------------------------------------------------------------- scoring

def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 4) if b else None


def score(outputs: dict[str, dict], reference: Reference) -> dict:
    """Measure a system's answers against the reference.

    ``outputs[key]`` = {"term": dpv_iri or None, "relation": SKOS property or None,
    "none": True when the system claims "sin correspondencia"}; a missing key is "sin
    propuesta". A proposal is correct when its term is one the person approved."""
    proposals = correct = on_no_match = none_ok = none_wrong = relation_ok = typed = 0
    for item in reference.items:
        out = outputs.get(item.key) or {}
        term = out.get("term")
        if term:
            proposals += 1
            if item.status != STATUS_ALIGNED:
                on_no_match += 1
            elif term in item.terms:
                correct += 1
                if out.get("relation"):
                    typed += 1
                    relation_ok += out["relation"] == item.terms[term]
        elif out.get("none"):
            if item.status == STATUS_ALIGNED:
                none_wrong += 1
            else:
                none_ok += 1
    aligned = reference.aligned()
    precision = _ratio(correct, proposals)
    coverage = _ratio(correct, aligned)
    f1 = (round(2 * precision * coverage / (precision + coverage), 4)
          if precision and coverage else None)
    return {"concepts": len(reference.items), "aligned": aligned,
            "no_match": len(reference.items) - aligned,
            "proposals": proposals, "correct": correct, "wrong": proposals - correct,
            "on_no_match": on_no_match, "precision": precision, "coverage": coverage, "f1": f1,
            "none_claimed": none_ok + none_wrong, "none_ok": none_ok, "none_wrong": none_wrong,
            "none_recall": _ratio(none_ok, len(reference.items) - aligned),
            "relation_typed": typed, "relation_ok": relation_ok,
            "relation_accuracy": _ratio(relation_ok, typed)}


def agreement(first: Reference, second: Reference) -> dict:
    """How often two references agree on the concepts they share (same term, or both none)."""
    other = second.by_key()
    shared = [i for i in first.items if i.key in other]
    same = sum((not i.terms and not other[i.key].terms) or bool(set(i.terms) &
               set(other[i.key].terms)) for i in shared)
    return {"shared": len(shared), "agree": same, "rate": _ratio(same, len(shared))}