"""What the person chooses after testing an upload (S10-T03): a fixed rule checks each change,
the accepted ones go to an append-only log (work/user-decisions.jsonl) and the alignment is
written again as a new version; the previous one stays in history/v<N>/ with its four zips.

The person's choices never edit the uploaded ontology, the rule of the decider or the base:
they are laid over what the base and the rule decided, correspondence by correspondence."""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.alignment import decisions as dl
from src.alignment.dpv_targets import COMPATIBLE_TARGETS, DPV_NAMESPACE

LOG_FILE = "user-decisions.jsonl"
HISTORY_DIR = "history"
REVIEWER = "usuario"                     # anonymous: whoever uploaded the file
BY_USER = "user"
SET, REMOVE = "set", "remove"
RELATIONS = ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch")
REMOVE_WORDS = ("quitar", "quita", "ninguna", "ninguno")
WHO = {"base": "base", "rule": "regla", "rule_without_ai": "regla sin IA", BY_USER: "usuario"}


@dataclass
class Change:
    """One change the person asks for on a concept of the upload."""
    concept: str                    # its name or IRI
    action: str = SET               # SET: align it with ``dpv`` | REMOVE: no correspondence
    relation: str | None = None     # SKOS mapping property, with or without "skos:"
    dpv: str | None = None          # DPV term: name, dpv:Name or IRI (REMOVE: only that one)
    replaces: str | None = None     # the DPV term it replaces when the concept has several
    note: str | None = None         # what the person read or typed (it goes to the log)

    def text(self) -> str:
        if self.action == REMOVE:
            return f"{self.concept} quitar" + (f" {self.dpv}" if self.dpv else "")
        return (f"{self.concept} {self.relation} {self.dpv}"
                + (f" en vez de {self.replaces}" if self.replaces else ""))


@dataclass
class Checked:
    """A change after the fixed rule: valid or refused with its reason (Spanish)."""
    change: Change
    ok: bool = False
    reason: str = ""
    concept_key: str | None = None
    concept_name: str | None = None
    dpv_iri: str | None = None
    dpv_name: str | None = None
    relation: str | None = None     # "skos:broadMatch"
    replaces: str | None = None     # IRI of the DPV term it takes out
    before: list[dict] = field(default_factory=list)
    after: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------- the console form

def parse_change(text: str) -> Change:
    """'Concepto broadMatch Término', '... en vez de Otro', 'Concepto quitar [Término]'."""
    words = (text or "").split()
    low = [w.lower() for w in words]
    if len(words) in (2, 3) and low[1] in REMOVE_WORDS:
        return Change(words[0], REMOVE, dpv=words[2] if len(words) == 3 else None)
    if len(words) == 3:
        return Change(words[0], SET, relation=words[1], dpv=words[2])
    if len(words) == 7 and low[3:6] == ["en", "vez", "de"]:
        return Change(words[0], SET, relation=words[1], dpv=words[2], replaces=words[6])
    raise ValueError(f"No entiendo el cambio «{text}». Escríbelo como «Concepto broadMatch "
                     "TérminoDelDPV», «Concepto broadMatch Término en vez de Otro» o "
                     "«Concepto quitar».")


# --------------------------------------------------------------------------- the state

def log_path(ws) -> Path:
    return ws.work_dir / LOG_FILE


def read(ws) -> list[dl.Decision]:
    return dl.read_log(log_path(ws))


def current_version(ws) -> int:
    from src.alignment.propose import ALIGNMENT_VERSION
    return int(ws.manifest.get("alignment_version") or ALIGNMENT_VERSION)


def touched(ws) -> set[str]:
    """The concepts the person decided on (the AI reviewer does not review them again)."""
    return {d.concept_key for d in read(ws)}


def concepts(inherited: dict, decisions: dict | None) -> dict[str, dict]:
    """Every concept of the upload by key: its name, kinds and whether it can be aligned."""
    out = {}
    for item in [*inherited.get("inherited", []), *(decisions or {}).get("entries", [])]:
        out[item["key"]] = {"key": item["key"], "name": item["name"],
                            "kinds": list(item.get("kinds") or []), "alignable": True}
    for n in inherited.get("not_inherited", []):
        out.setdefault(n["key"], {"key": n["key"], "name": n["name"], "kinds": [],
                                  "alignable": n.get("reason") != "kind"})
    return out


def system_state(inherited: dict, decisions: dict | None) -> dict[str, dict[str, dict]]:
    """What the base and the rule put in the ontology: concept -> DPV IRI -> correspondence."""
    from src.alignment import propose as dec
    state: dict[str, dict[str, dict]] = {}
    for item in inherited.get("inherited", []):
        for m in item.get("mappings", []):
            state.setdefault(item["key"], {})[m["dpv_iri"]] = {
                "relation": m["relation"], "dpv_name": m.get("dpv_name"), "decided_by": "base",
                "version": dec.ALIGNMENT_VERSION}
    for e in (decisions or {}).get("entries", []):
        if e["status"] == dec.ACCEPTED:
            state.setdefault(e["key"], {})[e["dpv_iri"]] = {
                "relation": e.get("relation") or "rdfs:seeAlso", "dpv_name": e.get("dpv_name"),
                "decided_by": e["decided_by"], "version": e.get("version",
                                                                 dec.ALIGNMENT_VERSION)}
    return state


def replay(state: dict[str, dict[str, dict]], records: list[dl.Decision]) -> dict:
    """The person's decisions laid over ``state``, in the order they were taken."""
    out = {k: dict(v) for k, v in state.items()}
    for d in records:
        maps = out.setdefault(d.concept_key, {})
        if d.decision == dl.DECISION_APPROVED:
            maps[d.dpv_iri] = {"relation": d.relation, "dpv_name": d.snapshot.get("dpv_name"),
                               "decided_by": BY_USER, "version": d.snapshot.get("version"),
                               "decided_at": d.decided_at, "note": d.note,
                               "before": d.snapshot.get("before") or []}
        elif d.decision == dl.DECISION_REJECTED:
            maps.pop(d.dpv_iri, None)
        elif d.decision == dl.DECISION_NO_MATCH:
            maps.clear()
    return out


@dataclass
class Overlay:
    """What the person changed, ready to be written: the final correspondences of each
    concept they touched (with who decided each one)."""
    final: dict[str, dict[str, dict]] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    last: dict[str, dl.Decision] = field(default_factory=dict)    # last decision per concept

    @property
    def keys(self) -> set[str]:
        return set(self.final)

    def keeps(self, key: str, dpv_iri: str, decided_by: str) -> bool:
        """Whether a correspondence the base or the rule wrote stays as they wrote it."""
        if key not in self.final:
            return True
        kept = self.final[key].get(dpv_iri)
        return kept is not None and kept["decided_by"] == decided_by

    def mine(self) -> list[tuple[str, str, dict]]:
        """The correspondences the person decided: (concept, DPV IRI, data)."""
        return sorted((k, iri, m) for k, maps in self.final.items() for iri, m in maps.items()
                      if m["decided_by"] == BY_USER)


def overlay(ws, inherited: dict, decisions: dict | None) -> Overlay | None:
    records = read(ws)
    if not records:
        return None
    state = replay(system_state(inherited, decisions), records)
    index = concepts(inherited, decisions)
    keys = {d.concept_key for d in records}
    return Overlay(final={k: state.get(k, {}) for k in keys},
                   names={k: index.get(k, {}).get("name", _local(k)) for k in keys},
                   last={d.concept_key: d for d in records})


# --------------------------------------------------------------------------- the fixed rule

def _local(value: str) -> str:
    return re.split(r"[#/:]", value)[-1]


def _relation(value: str | None) -> str | None:
    name = _local(value or "")
    match = next((r for r in RELATIONS if r.lower() == name.lower()), None)
    return f"skos:{match}" if match else None


def _term(targets, value: str | None):
    if not value:
        return None
    iri = value if value.startswith("http") else DPV_NAMESPACE + _local(value)
    by_iri = {t.iri: t for t in targets.targets}
    if iri in by_iri:
        return by_iri[iri]
    name = _local(value).lower()
    return next((t for t in targets.targets if t.name.lower() == name), None)


def _mapped(maps: dict[str, dict], value: str | None) -> str | None:
    """The DPV IRI among the concept's correspondences that ``value`` names (IRI or name)."""
    if not value:
        return None
    if value in maps:
        return value
    same = [iri for iri in maps if _local(iri).lower() == _local(value).lower()]
    return same[0] if len(same) == 1 else None


def _concept(index: dict[str, dict], value: str) -> tuple[dict | None, str]:
    if value in index:
        return index[value], ""
    same = [c for c in index.values() if c["name"].lower() == _local(value).lower()]
    if not same:
        return None, f"No encontré el concepto «{value}» en tu ontología."
    if len(same) > 1:
        return None, f"Hay {len(same)} conceptos llamados «{value}»: escribe su IRI."
    return same[0], ""


def _listing(maps: dict[str, dict]) -> list[dict]:
    return [{"dpv_iri": iri, "dpv_name": m.get("dpv_name") or _local(iri),
             "relation": m["relation"], "decided_by": m["decided_by"],
             "version": m.get("version")} for iri, m in sorted(maps.items())]


def check(ws, changes: list[Change], *, targets=None, cfg: dict | None = None
          ) -> list[Checked]:
    """The fixed rule: the concept exists and can be aligned, the DPV term exists and is of its
    kind (a property only with DPV properties), the relation is SKOS, the change changes
    something and says which correspondence it replaces; one change per concept in a list."""
    from src.alignment.inherit import load_result as inherited_result
    from src.alignment.propose import load_dpv_targets, load_result as decisions_result
    inherited, decisions = inherited_result(ws), decisions_result(ws)
    if inherited is None or decisions is None:
        raise ValueError("Primero hay que alinear el archivo (todavía no tiene correspondencias).")
    targets = targets if targets is not None else load_dpv_targets(cfg)
    index = concepts(inherited, decisions)
    state = replay(system_state(inherited, decisions), read(ws))
    seen: set[str] = set()
    out = []
    for change in changes:
        result = Checked(change)
        out.append(result)
        concept, why = _concept(index, change.concept)
        if concept is None:
            result.reason = why
            continue
        key, name = concept["key"], concept["name"]
        result.concept_key, result.concept_name = key, name
        if not concept["alignable"]:
            result.reason = f"«{name}» es un individuo: no se alinea con el DPV."
            continue
        maps = state.get(key, {})
        result.before = _listing(maps)
        term = _term(targets, change.dpv)
        if change.action == REMOVE:
            if not maps:
                result.reason = f"«{name}» no tiene correspondencias que quitar."
                continue
            gone = _mapped(maps, change.dpv)
            if change.dpv and gone is None:
                result.reason = (f"«{change.dpv}» no es un término del DPV 2.3." if term is None
                                 else f"«{name}» no está alineado con «{term.name}».")
                continue
            new = {k: v for k, v in maps.items() if gone is not None and k != gone}
            if gone is not None:
                result.dpv_iri = gone
                result.dpv_name = maps[gone].get("dpv_name") or _local(gone)
        elif change.action == SET:
            relation = _relation(change.relation)
            if relation is None:
                result.reason = (f"«{change.relation}» no es una relación SKOS: usa "
                                 + ", ".join(RELATIONS[:-1]) + f" o {RELATIONS[-1]}.")
                continue
            if term is None:
                result.reason = f"«{change.dpv}» no es un término del DPV 2.3."
                continue
            allowed = set().union(*(COMPATIBLE_TARGETS.get(k, set()) for k in concept["kinds"]))
            if allowed and term.kind not in allowed:
                result.reason = (f"«{name}» es una propiedad y «{term.name}» no: una propiedad "
                                 "solo se alinea con propiedades del DPV."
                                 if term.kind == "class" else
                                 f"«{name}» es una clase y «{term.name}» es una propiedad del "
                                 "DPV: una clase solo se alinea con clases del DPV.")
                continue
            replaced = _mapped(maps, change.replaces)
            if change.replaces and replaced is None:
                result.reason = f"«{name}» no está alineado con «{change.replaces}»."
                continue
            if replaced is None and term.iri not in maps and len(maps) == 1:
                replaced = next(iter(maps))              # the only one: it is replaced
            if replaced is None and term.iri not in maps and len(maps) > 1:
                names = " y ".join(m["dpv_name"] for m in _listing(maps))
                result.reason = (f"«{name}» tiene {len(maps)} correspondencias ({names}): di "
                                 "cuál cambias con «en vez de».")
                continue
            if term.iri in maps and maps[term.iri]["relation"] == relation and (
                    replaced in (None, term.iri)):
                result.reason = (f"«{name}» ya es {relation[5:]} de «{term.name}»: no cambia "
                                 "nada.")
                continue
            new = {k: v for k, v in maps.items() if k != replaced}
            new[term.iri] = {"relation": relation, "dpv_name": term.name, "decided_by": BY_USER}
            result.dpv_iri, result.dpv_name, result.relation = term.iri, term.name, relation
            if replaced not in (None, term.iri):
                result.replaces = replaced
        else:
            result.reason = f"Acción desconocida: {change.action}."
            continue
        if key in seen:                              # its own problems are said first
            result.reason = f"«{name}» ya tiene otro cambio en esta lista: uno por concepto."
            continue
        result.ok, result.after = True, _listing(new)
        seen.add(key)
    return out


# --------------------------------------------------------------------------- applying

def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def local_day(when: str | None) -> str:
    """The day of a UTC time where the person is (the log keeps UTC)."""
    return datetime.fromisoformat(when).astimezone().date().isoformat() if when else ""


def records_for(ws, checked: Checked, version: int, when: str) -> list[dl.Decision]:
    """The lines of the log for one valid change (codes of the base log, reviewer 'usuario')."""
    change = checked.change
    after = [{**m, "version": m.get("version") or version} for m in checked.after]
    snapshot = {"version": version, "before": checked.before, "after": after,
                "change": change.text(), "dpv_name": checked.dpv_name}
    common = dict(source_id=ws.source_id, target=dl.TARGET_MAPPING,
                  concept_key=checked.concept_key, reviewer=REVIEWER,
                  note=change.note or change.text(), snapshot=snapshot, decided_at=when)
    if change.action == REMOVE:
        if checked.dpv_iri:
            return [dl.make_decision(decision=dl.DECISION_REJECTED, dpv_iri=checked.dpv_iri,
                                     **common)]
        return [dl.make_decision(decision=dl.DECISION_NO_MATCH, **common)]
    out = []
    if checked.replaces:
        out.append(dl.make_decision(decision=dl.DECISION_REJECTED, dpv_iri=checked.replaces,
                                    **common))
    out.append(dl.make_decision(decision=dl.DECISION_APPROVED, dpv_iri=checked.dpv_iri,
                                relation=checked.relation, **common))
    return out


def history_dir(ws, version: int) -> Path:
    return ws.path / HISTORY_DIR / f"v{version}"


def archive(ws, version: int) -> Path:
    """Keep what version ``version`` gave: its four zips, manifest, CSV and validation."""
    from src.alignment.user_alignment import (CSV_FILE, MANIFEST_FILE, build_user_package,
                                              package_name)
    from src.transform.convert import OUTPUT_FORMATS
    from src.validation import upload_validation as uv
    folder = history_dir(ws, version)
    folder.mkdir(parents=True, exist_ok=True)
    for spec in OUTPUT_FORMATS.values():
        package = build_user_package(ws, spec.key)
        (folder / package_name(ws, spec.key, version)).write_bytes(package.data)
    for name in (MANIFEST_FILE, CSV_FILE, uv.RESULT_FILE, uv.LOG_FILE):
        if (ws.output_dir / name).exists():
            shutil.copy2(ws.output_dir / name, folder / name)
    for old in ws.output_dir.glob("*.zip"):          # they are in the folder of their version
        old.unlink()
    return folder


def apply(ws, changes: list[Change], *, cfg: dict | None = None, targets=None,
          fmt: str = "turtle", llm=None, use_llm: bool | None = None, sleep_fn=None,
          core=None, when: str | None = None) -> dict:
    """Check the changes; if any is valid, keep the current version in history/, log the valid
    ones and write, validate and package the next version (the steps that call the AI are not
    redone: what the person decided is laid over them)."""
    import time
    from src.alignment.user_alignment import write_package, write_user_alignment
    from src.validation.upload_validation import validate_workspace
    checked = check(ws, changes, targets=targets, cfg=cfg)
    valid = [c for c in checked if c.ok]
    version = current_version(ws)
    if not valid:
        return {"checked": checked, "applied": False, "version": version}
    folder = archive(ws, version)
    when = when or _now()
    for c in valid:
        for record in records_for(ws, c, version + 1, when):
            dl.append_decision(record, log_path(ws))
    ws.manifest["alignment_version"] = version + 1
    ws.manifest.setdefault("versions", []).append(
        {"version": version, "folder": folder.relative_to(ws.path).as_posix(), "until": when})
    ws.save()
    started = time.perf_counter()
    manifest = write_user_alignment(ws, cfg=cfg, core=core,
                                    recomputed={k: False for k in ("inherit", "definitions",
                                                                   "decider", "labels",
                                                                   "evidence", "audit")})
    validate_workspace(ws, cfg, llm=llm, use_llm=use_llm, sleep_fn=sleep_fn or time.sleep)
    path = write_package(ws, fmt)
    return {"checked": checked, "applied": True, "version": version + 1, "previous": folder,
            "manifest": manifest, "zip": path, "seconds": round(time.perf_counter() - started, 1)}


# --------------------------------------------------------------------------- console

def mapping_text(m: dict) -> str:
    relation = (m.get("relation") or "").replace("skos:", "")
    return (f"{relation} {m.get('dpv_name')} ({WHO.get(m.get('decided_by'), m.get('decided_by'))}"
            + (f", versión {m['version']}" if m.get("version") else "") + ")")


def state_text(maps: list[dict]) -> str:
    return "; ".join(mapping_text(m) for m in maps) if maps else "sin correspondencia"


def render_checked(checked: list[Checked]) -> str:
    lines = ["Cambios pedidos (los revisa una regla fija)"]
    for c in checked:
        who = c.concept_name or c.change.concept
        if c.ok:
            after = [m for m in c.after if m["decided_by"] == BY_USER] or c.after
            lines.append(f"  [válido] {who}: {state_text(c.before)} -> "
                         f"{state_text(after) if after else 'sin correspondencia'}")
        else:
            lines.append(f"  [rechazado] {c.change.text()}: {c.reason}")
    return "\n".join(lines)
