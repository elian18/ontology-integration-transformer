# PROV-O of the base (S10-T06): who decided each correspondence of the base with the DPV and
# when, how each concept the AI took from the LOPDP was produced and who decided it (with its
# article), the repairs R1 and R2 of the core and the profile, and the bridge with the GDPR.
# It reads what the pipeline already wrote, in a module of its own (ontopriv-provenance) in four
# formats; it calls no AI and never changes the base.
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

from src.console import counted
from src.provenance.model import (PROV, People, Recorder, describe, digest, records_about,
                                  slug, when)

PROVENANCE_IRI = "http://www.semanticweb.org/ontopriv-provenance"
MODULE = "ontopriv-provenance"
MANIFEST_FILE = "provenance-manifest.json"
FORMATS = ("rdfxml", "owl", "turtle", "jsonld")
LAW = "LOPDP"
DPV_IRI = "https://w3id.org/dpv"
MAPPINGS = (SKOS.exactMatch, SKOS.closeMatch, SKOS.broadMatch, SKOS.narrowMatch,
            SKOS.relatedMatch)
REPAIR_NAMES = {"several_domains": "R1", "several_ranges": "R1", "double_type": "R2"}
KIND_WORDS = {"class": "clase", "object_property": "propiedad de objeto",
              "datatype_property": "propiedad de datos", "property": "propiedad"}


@dataclass(frozen=True)
class Settings:
    people: dict = field(default_factory=dict)
    extraction_model: str | None = None
    extraction_prompt: int = 1


def settings(cfg: dict | None = None) -> Settings:
    block = (cfg or {}).get("provenance") or {}
    extraction = block.get("extraction") or {}
    return Settings(people=block.get("people") or {}, extraction_model=extraction.get("model"),
                    extraction_prompt=int(extraction.get("prompt_version") or 1))


# --------------------------------------------------------------------------- what it reads

@dataclass
class Inputs:
    data: dict                      # the candidates file
    records: list                   # the decisions log
    alignment: Graph                # the alignment module of the base
    source_id: str
    files: dict                     # name -> Path (only the ones that exist)
    alignment_manifest: dict | None = None
    repairs: dict | None = None
    proposed: dict | None = None
    bridge: dict | None = None
    bridge_cache: list = field(default_factory=list)


def _json(path: Path | None):
    return json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else None


def load_inputs(cfg: dict | None = None) -> Inputs:
    from src.alignment import gdpr_bridge as gb
    from src.alignment.assistant_review import proposals_path
    from src.alignment.decisions import decisions_path, read_log, review_settings
    from src.alignment.export import CANDIDATES_JSON
    from src.alignment.inherit import base_paths
    from src.alignment.write import ALIGNMENT_FILE, ALIGNMENT_MANIFEST
    from src.config import ROOT, load_config
    from src.core.emit import CORE_FILE, PROFILE_FILE
    from src.core.extract import PROPOSALS_FILE
    from src.core.repair import REPAIRS_FILE
    from src.transform.convert import load_graph
    cfg = cfg if cfg is not None else load_config()
    out = base_paths(cfg)["candidates"].parent
    inputs = cfg.get("inputs") or {}

    def root(p):
        return Path(p) if Path(p).is_absolute() else ROOT / p
    files = {"candidates": out / CANDIDATES_JSON, "log": decisions_path(cfg),
             "alignment": out / ALIGNMENT_FILE, "core": out / CORE_FILE,
             "profile": out / PROFILE_FILE, "proposed": out / PROPOSALS_FILE,
             "proposals": proposals_path(cfg), "repairs": out / REPAIRS_FILE,
             "bridge": out / f"{gb.MODULE}.rdf", "bridge_result": out / gb.RESULT_FILE,
             "bridge_csv": out / gb.CSV_FILE, "bridge_cache": out / gb.CACHE_FILE,
             **{k: root(v) for k, v in (("ontopriv", inputs.get("ontology")),
                                        ("lopdp", inputs.get("legal_text")),
                                        ("dpv", inputs.get("dpv"))) if v},
             **gb.paths(cfg)}
    for name in ("candidates", "log", "alignment"):
        if not files[name].exists():
            raise FileNotFoundError(f"Falta {files[name]}")
    cache = []
    if files["bridge_cache"].exists():
        with open(files["bridge_cache"], encoding="utf-8") as fh:
            cache = [json.loads(line) for line in fh if line.strip()]
    return Inputs(
        data=_json(files["candidates"]), records=read_log(files["log"]),
        alignment=load_graph(files["alignment"]),
        source_id=review_settings(cfg)["source_id"],
        files={k: p for k, p in files.items() if p.exists()},
        alignment_manifest=_json(out / ALIGNMENT_MANIFEST), repairs=_json(files["repairs"]),
        proposed=_json(files["proposed"]), bridge=_json(files["bridge_result"]),
        bridge_cache=cache)


def _sha256(path: Path | None) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path and path.exists() else None


def _shown(path: Path) -> str:
    from src.config import ROOT
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def local(iri: str) -> str:
    return str(iri).replace("/", "#").rsplit("#", 1)[-1]


def _alignment_iri() -> str:
    from src.alignment.write import alignment_iri
    from src.core.emit import PROFILE_IRI
    return alignment_iri(PROFILE_IRI)


def mapping_record(subject: str, relation, term: str) -> URIRef:
    """The IRI of the record of a correspondence of the base (S10-T07 links to it)."""
    return URIRef(f"{PROVENANCE_IRI}#mapping-"
                  f"{digest(_alignment_iri(), str(subject), str(relation), str(term))}")


def no_match_record(subject: str) -> URIRef:
    """The IRI of the record of a concept of the base without correspondence."""
    return URIRef(f"{PROVENANCE_IRI}#no-match-{digest(_alignment_iri(), str(subject))}")


def _articles(numbers) -> str:
    """'el art. 17 de la LOPDP', 'los arts. 3 y 4 de la LOPDP'; '' without articles."""
    nums = [str(n) for n in sorted({int(n) for n in numbers or []})]
    if not nums:
        return ""
    if len(nums) == 1:
        return f"el art. {nums[0]} de la {LAW}"
    return f"los arts. {', '.join(nums[:-1])} y {nums[-1]} de la {LAW}"


def _day(value) -> str:
    return str(value or "")[:10]


# --------------------------------------------------------------------------- the graph

class _Base:
    """Writes the records of the base into a Recorder, step by step."""

    def __init__(self, inp: Inputs, s: Settings):
        from src.alignment.write import alignment_iri
        from src.core.emit import PROFILE_IRI
        self.inp, self.s = inp, s
        self.rec = Recorder(PROVENANCE_IRI, "provenance", People(s.people))
        self.alignment_iri = alignment_iri(PROFILE_IRI)
        self.counts: dict = {}
        self.mapping_nodes: dict = {}         # (subject, relation, term) -> record
        self.decision_nodes: dict = {}        # AI concept key -> record of its decision

    # ------------------------------------------------------------------ helpers
    def file(self, name: str, label: str, comment: str | None = None):
        path = self.inp.files.get(name)
        if path is None:
            return None
        return self.rec.file(f"file-{slug(name)}", f"{label} ({_shown(path)})", _sha256(path),
                             comment)

    def article(self, n: int) -> URIRef:
        node = self.rec.node(f"{LAW.lower()}-art-{int(n)}")
        if (node, RDF.type, PROV.Entity) not in self.rec.g:
            self.rec.entity(local(node), f"{LAW}, art. {int(n)}")
            self.rec.add(node, DCTERMS.isPartOf, self.files.get("lopdp"))
        return node

    def llm(self, model: str | None):
        if not model:
            return None
        return self.rec.agent(f"agent-llm-{slug(model)}", model,
                              "Modelo de lenguaje que usa el sistema: propone o explica; "
                              "nada de lo que dice entra a la base sin una persona o una regla.")

    def plan(self, name: str, version: int, sha: str | None = None):
        from src.ai.prompts import load_prompt
        if sha is None:
            try:
                sha = load_prompt(name, version).manifest()["sha256"]
            except (FileNotFoundError, KeyError, ValueError):
                sha = None
        return self.rec.plan(f"plan-prompt-{name}-v{version}", f"Prompt {name}.v{version}",
                             f"Instrucción que recibió la IA (src/ai/prompts/{name}.v{version}"
                             ".txt).", "sha256:" + sha if sha else None)

    # ------------------------------------------------------------------ build
    def build(self) -> Graph:
        from src.alignment import gdpr_bridge as gb
        from src.core.emit import CORE_IRI, PROFILE_IRI
        rec = self.rec
        rec.header("Procedencia de la base del sistema (PROV-O)",
                   "Quién decidió y cuándo cada correspondencia de la base con el DPV (y cada "
                   "concepto que quedó sin correspondencia), cómo se produjo y quién decidió "
                   f"cada concepto que la IA tomó de la {LAW} (con su artículo), las "
                   "reparaciones R1 y R2 del núcleo y el perfil y el puente con el GDPR. Usa "
                   "PROV-O (W3C, 2013). Cada correspondencia es un owl:Axiom con nombre que "
                   "anota su tripleta, como escribe SSSOM un mapeo: Protégé lo muestra en la "
                   "tripleta y la base no cambia. PROV-O, la base y el DPV se referencian por "
                   "IRI; no se importan.",
                   (CORE_IRI, PROFILE_IRI, self.alignment_iri, gb.BRIDGE_IRI, DPV_IRI))
        self.decided_by = rec.annotation_property(
            "decidedBy", {"es": "decidido por", "en": "decided by"},
            "Quién decidió la correspondencia, con los valores que usa el sistema en cada "
            "subida. En la base siempre es «base»: la decidió una persona en el Sprint 5 "
            "(quién y cuándo lo dicen prov:wasAttributedTo y prov:generatedAtTime).")
        self.agents()
        self.inputs()
        self.activities()
        self.modules()
        self.repairs()
        self.law_concepts()
        self.mappings()
        self.bridge()
        return rec.g

    def agents(self) -> None:
        from src.alignment.assistant_review import PROPOSED_BY
        rec = self.rec
        self.system = rec.agent("agent-system", "ontology-integration-transformer",
                                "El sistema de este TIC: aplica las reglas y escribe los "
                                "módulos.")
        meta = self.inp.data.get("metadata") or {}
        self.embeddings = None
        if meta.get("embedding_model"):
            self.embeddings = rec.agent(
                f"agent-embeddings-{slug(meta['embedding_model'])}", meta["embedding_model"],
                "Modelo de embeddings (sentence-transformers) que mide el parecido de "
                "significado entre un concepto y los términos del DPV.")
        assistant = sorted({d.snapshot.get("proposal_assistant") for d in self.inp.records
                            if d.snapshot.get("proposed_by") == PROPOSED_BY
                            and d.snapshot.get("proposal_assistant")})
        self.assistant = None
        if assistant:
            self.assistant = rec.agent(
                "agent-assistant", assistant[0],
                "Propuso por lotes la correspondencia de los conceptos aún por validar "
                "(S5-T08); cada propuesta entró al registro solo cuando una persona la "
                "confirmó, y la decisión queda a nombre de esa persona.")
            for name in sorted({d.reviewer for d in self.inp.records
                                if d.snapshot.get("proposed_by") == PROPOSED_BY}):
                rec.g.add((self.assistant, PROV.actedOnBehalfOf, rec.person(name)))

    def inputs(self) -> None:
        names = {"ontopriv": "OntoPriv, la ontología de entrada",
                 "lopdp": f"{LAW}: Ley Orgánica de Protección de Datos Personales (Ecuador)",
                 "dpv": "Data Privacy Vocabulary (DPV 2.3)",
                 "proposed": "Conceptos que la IA propuso a partir de la ley",
                 "candidates": "Candidatos del DPV y tipo SKOS propuesto por la IA",
                 "proposals": "Propuestas del asistente",
                 "log": "Registro de decisiones de la base",
                 "gdprtext": "GDPRtEXT, la ontología del GDPR",
                 "gdprtext_alignment": "Alineación de GDPRtEXT con el DPV (fija)",
                 "bridge_csv": "Explicación de la IA revisora de cada par del puente"}
        self.files = {k: self.file(k, label) for k, label in names.items()}
        if self.files.get("dpv") is not None:
            self.rec.g.add((self.files["dpv"], PROV.specializationOf, URIRef(DPV_IRI)))

    def activities(self) -> None:
        from src.alignment.decisions import TARGET_CONCEPT, TARGET_MAPPING
        rec, f, inp = self.rec, self.files, self.inp
        self.r1 = rec.activity(
            "activity-repair-r1", "Reparación R1: varios dominios o rangos pasan a una unión",
            "Una propiedad con varios rdfs:domain (o rdfs:range) queda con uno: la unión de "
            "esas clases (P19 de OOPS!). Se aplica después de separar el núcleo y el perfil "
            "(S9-T02); el archivo de OntoPriv no se modifica.",
            agents=[self.system], used=[x for x in [f.get("ontopriv")] if x is not None],
            plan=rec.plan("plan-repair-r1", "Regla R1 (P19 de OOPS!)",
                          "Varios rdfs:domain, o varios rdfs:range, pasan a uno: la unión de "
                          "esas clases."))
        self.r2 = rec.activity(
            "activity-repair-r2", "Reparación R2: un solo tipo para cada propiedad",
            "Una propiedad declarada de objeto y de datos a la vez (no se permite en OWL 2 DL) "
            "se queda con el tipo que muestra su uso (S9-T02).",
            agents=[self.system], used=[x for x in [f.get("ontopriv")] if x is not None],
            plan=rec.plan("plan-repair-r2", "Regla R2 (un solo tipo)",
                          "Se queda como propiedad de datos solo si todos sus valores son "
                          "textos (o no tiene valores y sus subpropiedades son de datos); si no, "
                          "como propiedad de objeto."))
        self.core_build = rec.activity(
            "activity-core-build", "Construcción del núcleo y el perfil de Ecuador",
            "Separa OntoPriv en un núcleo sin jurisdicción (OntoPriv-Core) y el perfil de "
            f"Ecuador ({LAW}) por familias y namespace (S3) y escribe los dos módulos con las "
            "reparaciones R1 y R2.",
            agents=[self.system], used=[x for x in [f.get("ontopriv")] if x is not None],
            informed_by=[self.r1, self.r2])
        proposed = (inp.proposed or {}).get("counts") or {}
        model = (inp.proposed or {}).get("model") or self.s.extraction_model
        version = self._prompt_version((inp.proposed or {}).get("prompt")) \
            or self.s.extraction_prompt
        self.extractor = self.llm(model)
        self.extraction = rec.activity(
            "activity-law-extraction", f"Extracción de conceptos de la {LAW} con IA",
            "La IA lee la ley por lotes de artículos y propone conceptos de protección de "
            "datos, cada uno con el artículo del que sale (S3); solo propone: una persona "
            "decide cada uno." + self._extracted(proposed),
            agents=[a for a in (self.extractor, self.system) if a is not None],
            used=[x for x in [f.get("lopdp")] if x is not None],
            plan=self.plan("extract", version) if version else None)
        self._gen(f.get("proposed"), self.extraction)
        meta = inp.data.get("metadata") or {}
        just = meta.get("justification") or {}
        self.justifier = self.llm(just.get("llm_model"))
        sets = meta.get("settings") or {}
        self.candidates = rec.activity(
            "activity-candidates", "Candidatos del DPV y tipo SKOS propuesto por la IA",
            "Para cada concepto, los términos del DPV más parecidos "
            + (f"({sets['top_k']} por concepto; {sets['weight_lexical']} nombre + "
               f"{sets['weight_semantic']} significado) " if sets.get("top_k") else "")
            + "y el tipo SKOS que propone la IA con su justificación (S4). No decide nada.",
            agents=[a for a in (self.embeddings, self.justifier, self.system) if a is not None],
            used=[x for x in (f.get("ontopriv"), f.get("proposed"), f.get("dpv"))
                  if x is not None],
            started=meta.get("generated_at"), ended=just.get("updated_at"),
            plan=self.plan("justify", int(just["prompt_version"]))
            if just.get("prompt_version") else None)
        self._gen(f.get("candidates"), self.candidates)
        self.proposing = None
        if self.assistant is not None:
            self.proposing = rec.activity(
                "activity-assistant-proposals", "Propuestas del asistente",
                "El asistente propuso, por lotes, la correspondencia con el DPV de los conceptos "
                "que quedaban por validar, o que no tienen (S5-T08). Son propuestas, no "
                "decisiones.", agents=[self.assistant],
                used=[x for x in [f.get("candidates")] if x is not None])
            days = sorted({d.snapshot.get("proposal_created_at") for d in inp.records
                           if d.snapshot.get("proposal_created_at")})
            if days:
                rec.g.add((self.proposing, DCTERMS.date, Literal(days[0], datatype=XSD.date)))
            self._gen(f.get("proposals"), self.proposing)
        logged = [d for d in inp.records if d.source_id == inp.source_id]
        self.concept_review = self._review(
            "activity-concept-review", "Revisión de los conceptos de la IA",
            f"Una persona decidió cada concepto que la IA tomó de la {LAW}: concepto nuevo del "
            "perfil, ya existe en OntoPriv o rechazado (S5).",
            [d for d in logged if d.target == TARGET_CONCEPT], [f.get("proposed")])
        self.mapping_review = self._review(
            "activity-mapping-review", "Revisión de las correspondencias con el DPV",
            "Una persona decidió la correspondencia de cada concepto con el DPV, o que no tiene, "
            "en la tarjeta de revisión o confirmando una propuesta del asistente (S5).",
            [d for d in logged if d.target == TARGET_MAPPING],
            [f.get("candidates"), f.get("proposals")])
        self.alignment_write = rec.activity(
            "activity-alignment-write", "Escritura del módulo de alineación",
            "Escribe las correspondencias aprobadas como tripletas SKOS, los conceptos nuevos y "
            "los artículos de los duplicados (S5-T06), solo cuando no queda nada por validar.",
            agents=[self.system], used=[x for x in (f.get("log"), f.get("candidates"))
                                        if x is not None],
            ended=(inp.alignment_manifest or {}).get("created"))

    @staticmethod
    def _extracted(counts: dict) -> str:
        if counts.get("articles_processed") is None:
            return ""
        return (" " + counted(counts["articles_processed"], "artículo procesado",
                              "artículos procesados")
                + ", " + counted(counts.get("concepts_proposed", 0), "concepto propuesto",
                                 "conceptos propuestos") + ".")

    def _review(self, local_name, label, comment, decisions, used):
        people = sorted({self.rec.person(d.reviewer) for d in decisions}, key=str)
        times = sorted(d.decided_at for d in decisions if d.decided_at)
        return self.rec.activity(local_name, label, comment, agents=people,
                                 used=[x for x in used if x is not None],
                                 started=times[0] if times else None,
                                 ended=times[-1] if times else None)

    def _gen(self, entity, activity) -> None:
        if entity is not None:
            self.rec.g.add((entity, PROV.wasGeneratedBy, activity))

    @staticmethod
    def _prompt_version(prompt) -> int | None:
        if isinstance(prompt, dict) and prompt.get("version"):
            return int(prompt["version"])
        return None

    def modules(self) -> None:
        from src.alignment import gdpr_bridge as gb
        from src.core.emit import CORE_IRI, PROFILE_IRI
        rec, f = self.rec, self.files
        self.module = {}
        for name, iri, label, activity in (
                ("core", CORE_IRI, "Núcleo OntoPriv-Core", self.core_build),
                ("profile", PROFILE_IRI, "Perfil de Ecuador", self.core_build),
                ("alignment", self.alignment_iri, "Alineación de la base con el DPV",
                 self.alignment_write),
                ("bridge", gb.BRIDGE_IRI, "Puente de la base con el GDPR", None)):
            path = self.inp.files.get(name)
            if path is None:
                continue
            node = rec.file(f"module-{name}", f"{label} ({_shown(path)})", _sha256(path),
                            "Esta versión del módulo (su SHA-256 es el del archivo).")
            rec.g.add((node, PROV.specializationOf, URIRef(iri)))
            if activity is not None:
                rec.g.add((node, PROV.wasGeneratedBy, activity))
            self.module[name] = node
        for name in ("core", "profile"):
            if name in self.module and f.get("ontopriv") is not None:
                rec.g.add((self.module[name], PROV.wasDerivedFrom, f["ontopriv"]))
        if "alignment" in self.module:
            node = self.module["alignment"]
            rec.add(node, PROV.generatedAtTime,
                    when((self.inp.alignment_manifest or {}).get("created")))
            for x in (f.get("log"), f.get("candidates"), self.module.get("profile")):
                rec.add(node, PROV.wasDerivedFrom, x)

    # ------------------------------------------------------------------ R1 and R2
    def repairs(self) -> None:
        rec = self.rec
        changes = (self.inp.repairs or {}).get("changes") or []
        made = Counter()
        for c in changes:
            name = REPAIR_NAMES.get(c.get("rule"))
            if name is None or not c.get("iri"):
                continue
            node = rec.entity(f"repair-{digest(c['rule'], c['iri'])}",
                              f"{name} · {c.get('property') or local(c['iri'])}",
                              self._repair_text(c))
            rec.generated(node, self.r1 if name == "R1" else self.r2, self.system)
            rec.g.add((node, DCTERMS.subject, URIRef(c["iri"])))
            rec.add(node, PROV.wasDerivedFrom, self.files.get("ontopriv"))
            rec.add(node, DCTERMS.isPartOf, self.module.get(c.get("module")))
            made[name] += 1
        self.counts["repairs"] = {"R1": made["R1"], "R2": made["R2"]}

    @staticmethod
    def _repair_text(c: dict) -> str:
        where = {"core": "el núcleo", "profile": "el perfil"}.get(c.get("module"), "la base")
        if c["rule"] == "double_type":
            ev = c.get("evidence") or {}
            ranges = c.get("removed_ranges") or []
            return (f"R2 en {where}: estaba declarada como propiedad de objeto y de datos a la "
                    f"vez; queda como {c.get('kept')} y deja de ser {c.get('removed')} (es de "
                    f"datos solo si todos sus valores son textos). Uso: "
                    f"{counted(ev.get('resources', 0), 'valor recurso', 'valores recurso')}, "
                    f"{counted(ev.get('texts', 0), 'texto', 'textos')}, "
                    f"{counted(ev.get('sub_properties', 0), 'subpropiedad', 'subpropiedades')}."
                    + (f" Se quitaron los rangos que ya no le sirven: {', '.join(ranges)}."
                       if ranges else ""))
        what = "dominios" if c["rule"] == "several_domains" else "rangos"
        classes = c.get("classes") or []
        return (f"R1 en {where}: tenía {len(classes)} {what} ({', '.join(classes)}); ahora "
                f"tiene uno, la unión de esas clases (owl:unionOf), como pide P19 de OOPS!.")

    # ------------------------------------------------------------------ the concepts of the law
    def law_concepts(self) -> None:
        from src.alignment.ai_concepts import ai_concepts, concept_decisions
        from src.alignment.decisions import DECISION_APPROVED, DECISION_DUPLICATE
        from src.core.emit import PROFILE_IRI
        rec = self.rec
        decided = concept_decisions(self.inp.records, self.inp.source_id)
        outcome = Counter()
        for c in ai_concepts(self.inp.data, PROFILE_IRI):
            kind = KIND_WORDS.get(c.suggested_kind, c.suggested_kind or "concepto")
            arts = [self.article(n) for n in sorted(set(c.articles or []))]
            extracted = rec.entity(
                f"law-concept-{digest(c.key)}", f"{c.label or c.name} ({c.name})",
                f"Propuesto por la IA con base en {_articles(c.articles) or 'la ley'}, como "
                f"{kind}" + (f": {c.definition}" if c.definition else "."))
            rec.generated(extracted, self.extraction, self.extractor)
            for a in arts:
                rec.g.add((extracted, PROV.hadPrimarySource, a))
            rec.add(extracted, DCTERMS.isPartOf, self.files.get("proposed"))
            if c.iri:                                     # the IRI it was proposed with
                rec.g.add((extracted, DCTERMS.subject, URIRef(c.iri)))
            d = decided.get(c.key)
            if d is None:
                continue
            if d.decision == DECISION_APPROVED:
                words = "concepto nuevo del perfil"
                text = (f"Concepto nuevo del perfil de Ecuador "
                        f"({KIND_WORDS.get(d.entity_kind, d.entity_kind)}): entra a la "
                        "alineación con su etiqueta, su definición y el artículo de la ley.")
                subject = c.iri
            elif d.decision == DECISION_DUPLICATE:
                same = d.same_as or c.duplicate_of
                words = f"ya existe en OntoPriv ({local(same)})"
                text = (f"Ya existía en OntoPriv como {local(same)}"
                        + (f" (la comprobación automática había encontrado "
                           f"{local(c.duplicate_of)})" if c.duplicate_of and
                           c.duplicate_of != same else "")
                        + f"; sus artículos de la {LAW} se agregan a esa entidad.")
                subject = same
            else:
                words, text, subject = "rechazado", "Rechazado: no entra a la ontología.", None
                rec.g.add((extracted, PROV.wasInvalidatedBy, self.concept_review))
                rec.add(extracted, PROV.invalidatedAtTime, when(d.decided_at))
            node = rec.entity(f"concept-decision-{digest(c.key)}", f"{c.name}: {words}",
                              f"{text} Decidido por {rec.people.name(d.reviewer)} el "
                              f"{_day(d.decided_at)}" + (f". Nota: {d.note}" if d.note else "")
                              + ".")
            rec.generated(node, self.concept_review, rec.person(d.reviewer), d.decided_at)
            rec.g.add((node, PROV.wasDerivedFrom, extracted))
            for a in arts:
                rec.g.add((node, PROV.hadPrimarySource, a))
            if subject:
                rec.g.add((node, DCTERMS.subject, URIRef(subject)))
            self.decision_nodes[c.key] = node
            outcome[d.decision] += 1
        self.counts["law_concepts"] = {"total": len(ai_concepts(self.inp.data, PROFILE_IRI)),
                                       **{k: outcome[k] for k in
                                          ("approved", "duplicate", "rejected")}}

    # ------------------------------------------------------------------ the correspondences
    def mappings(self) -> None:
        from src.alignment.mapping_review import (
            STATUS_NO_MATCH, concept_status, mapping_concepts, mapping_decisions)
        from src.alignment.write import DECIDED_BY_ASSISTANT, collect
        from src.core.emit import PROFILE_IRI
        rec, inp = self.rec, self.inp
        content = collect(inp.data, inp.records, inp.source_id, PROFILE_IRI)
        logged = {(m["subject_iri"], SKOS[m["relation"].split(":", 1)[1]], m["dpv_iri"])
                  for m in content.mappings}
        written = {(str(s), p, str(o)) for p in MAPPINGS
                   for s, o in inp.alignment.subject_objects(p)}
        if logged != written:
            raise ValueError(
                "La alineación de la base no coincide con el registro de decisiones: "
                f"{len(written - logged)} del módulo no están en el registro y "
                f"{len(logged - written)} del registro no están en el módulo. Vuelve a "
                "escribirla: py -m src.alignment.write")
        current = mapping_decisions(inp.records, inp.source_id)
        attached = {}
        for dup in content.duplicates:
            if dup.get("duplicate_of"):
                attached.setdefault(dup["duplicate_of"], set()).update(dup.get("articles") or [])
        by = Counter()
        for m in content.mappings:
            d = current[m["concept_key"]][m["dpv_iri"]]
            prop = SKOS[m["relation"].split(":", 1)[1]]
            node = rec.entity(local(mapping_record(m["subject_iri"], prop, m["dpv_iri"])),
                              f"{m['concept_name']} {m['relation']} dpv:{local(m['dpv_iri'])}",
                              self._mapping_text(m, d))
            rec.axiom(node, m["subject_iri"], prop, m["dpv_iri"])
            rec.g.add((node, self.decided_by, Literal("base", lang="es")))
            rec.g.add((node, OWL.versionInfo, Literal("1")))
            rec.generated(node, self.mapping_review, rec.person(d.reviewer), d.decided_at)
            self._sources(node, d, m["concept_key"], m["found_by"] == "candidates")
            for n in sorted(set(m.get("concept_articles") or [])
                            | attached.get(m["subject_iri"], set())):
                rec.g.add((node, PROV.hadPrimarySource, self.article(n)))
            rec.add(node, DCTERMS.isPartOf, self.module.get("alignment"))
            self.mapping_nodes[(m["subject_iri"], str(prop), m["dpv_iri"])] = node
            by[m["decided_by"] == DECIDED_BY_ASSISTANT] += 1
        none = 0
        for c in mapping_concepts(inp.data, inp.records, inp.source_id, PROFILE_IRI):
            if concept_status(c, current) != STATUS_NO_MATCH:
                continue
            d = current[c.key][""]
            node = rec.entity(local(no_match_record(c.iri)),
                              f"{c.name}: sin correspondencia en el DPV",
                              self._no_match_text(d))
            rec.generated(node, self.mapping_review, rec.person(d.reviewer), d.decided_at)
            rec.g.add((node, DCTERMS.subject, URIRef(c.iri)))
            self._sources(node, d, c.key, bool(c.candidates))
            for n in sorted(set(c.articles or []) | attached.get(c.iri, set())):
                rec.g.add((node, PROV.hadPrimarySource, self.article(n)))
            none += 1
        self.counts["mappings"] = {"total": len(content.mappings), "assistant": by[True],
                                   "person": by[False]}
        self.counts["no_match"] = none

    def _sources(self, node, d, key: str, from_candidates: bool) -> None:
        from src.alignment.assistant_review import PROPOSED_BY
        rec = self.rec
        if from_candidates:
            rec.add(node, PROV.wasDerivedFrom, self.files.get("candidates"))
        if key in self.decision_nodes:
            rec.g.add((node, PROV.wasDerivedFrom, self.decision_nodes[key]))
        if d.snapshot.get("proposed_by") == PROPOSED_BY and self.proposing is not None:
            rec.g.add((node, PROV.wasDerivedFrom, self.proposal(d, key)))

    def proposal(self, d, key: str) -> URIRef:
        rec = self.rec
        node = rec.node(f"proposal-{digest(key)}")
        if (node, RDF.type, PROV.Entity) in rec.g:
            return node
        p = d.snapshot.get("proposal") or {}
        name = d.snapshot.get("concept_name") or local(key)
        if p.get("decision") == "no_match":
            what = "sin correspondencia"
        else:
            what = "; ".join(f"{m.get('relation')} dpv:{local(m.get('dpv_iri', ''))}"
                             for m in p.get("mappings") or []) or "alinear"
        rec.entity(local(node), f"Propuesta del asistente para {name}",
                   f"Propuso {what}" + (f": «{p['reason']}»" if p.get("reason") else "") + ".")
        rec.generated(node, self.proposing, self.assistant)
        if d.snapshot.get("subject_iri"):
            rec.g.add((node, DCTERMS.subject, URIRef(d.snapshot["subject_iri"])))
        rec.add(node, DCTERMS.isPartOf, self.files.get("proposals"))
        return node

    def _who(self, d, word: str = "Decidida") -> str:
        from src.alignment.assistant_review import PROPOSED_BY
        text = f"{word} por {self.rec.people.name(d.reviewer)} el {_day(d.decided_at)}"
        if d.snapshot.get("proposed_by") == PROPOSED_BY:
            return text + ", confirmando la propuesta del asistente"
        return text + " en la tarjeta de revisión, sin propuesta del asistente"

    def _mapping_text(self, m: dict, d) -> str:
        snap = d.snapshot
        parts = [self._who(d) + "."]
        if m["found_by"] == "candidates":
            rank, score = snap.get("rank"), snap.get("score")
            parts.append("Candidato" + (f" n.º {rank}" if rank else "") + " del DPV"
                         + (f" (puntaje {score}: nombre {snap.get('lexical')}, significado "
                            f"{snap.get('semantic')})" if score is not None else "") + ".")
        else:
            parts.append("Término buscado en el DPV (no estaba entre los candidatos).")
        if m.get("ai_relation"):
            model = snap.get("llm_model")
            parts.append(f"La IA{f' ({model})' if model else ''} había propuesto "
                         + ("ninguno" if m["ai_relation"] == "none" else m["ai_relation"])
                         + (f" con el art. {m['evidence_article']} de la {LAW} como evidencia"
                            if m.get("evidence_article") else "") + ".")
        if d.note and d.note != "propuesta del asistente confirmada":
            parts.append(f"Nota: {d.note}.")
        return " ".join(parts)

    def _no_match_text(self, d) -> str:
        text = self._who(d, "Decidido") + "."
        if d.note and d.note != "propuesta del asistente confirmada":
            text += f" Nota: {d.note}."
        return f"Sin correspondencia en el DPV. {text}"

    # ------------------------------------------------------------------ the bridge
    def bridge(self) -> None:
        from src.alignment import gdpr_bridge as gb
        rec, out = self.rec, self.inp.bridge
        if not out or "bridge" not in self.module:
            self.counts["bridge"] = None
            return
        cache = {}
        for e in self.inp.bridge_cache:
            if e.get("prompt") == out.get("prompt"):
                cache.setdefault(e.get("concept_key"), []).append(e)
        models = sorted({e.get("model") for es in cache.values() for e in es if e.get("model")})
        times = sorted(e["created"] for es in cache.values() for e in es if e.get("created"))
        rule = rec.plan("plan-bridge", "Regla del puente",
                        "Un concepto de la base y uno de GDPRtEXT forman un par cuando se "
                        "corresponden con el mismo término del DPV con exactMatch o closeMatch "
                        "en los dos lados; el par se escribe con skos:closeMatch.")
        building = rec.activity(
            "activity-bridge", "Puente de la base con el GDPR",
            "Encadena la base y GDPRtEXT por el DPV (S9-T09). La alineación de GDPRtEXT está "
            "fija en data/reference, así que el puente se rehace sin IA y da los mismos pares.",
            agents=[self.system], used=[x for x in (self.module.get("alignment"),
                                                    self.files.get("gdprtext_alignment"),
                                                    self.files.get("gdprtext")) if x is not None],
            ended=out.get("created"), plan=rule)
        node = self.module["bridge"]
        rec.g.add((node, PROV.wasGeneratedBy, building))
        rec.add(node, PROV.generatedAtTime, when(out.get("created")))
        for x in (self.module.get("alignment"), self.files.get("gdprtext_alignment")):
            rec.add(node, PROV.wasDerivedFrom, x)
        reviewer = self.llm(models[0]) if models else None
        prompt = out.get("prompt") or {}
        review = rec.activity(
            "activity-bridge-review", "IA revisora del puente",
            "La IA explica cada par: coherente, dudosa o incoherente. Solo explica: no cambia "
            "los pares ni entra a la ontología.",
            agents=[x for x in [reviewer] if x is not None], used=[node],
            started=times[0] if times else None, ended=times[-1] if times else None,
            plan=self.plan(prompt["name"], int(prompt["version"]), prompt.get("sha256"))
            if prompt.get("name") and prompt.get("version") else None)
        self._gen(self.files.get("bridge_csv"), review)
        n = 0
        for r in out.get("rows") or []:
            key = digest(gb.BRIDGE_IRI, r["ontopriv_iri"], r["gdprtext_iri"])
            pair = rec.entity(f"bridge-pair-{key}",
                              f"{r['ontopriv']} ↔ {r['gdprtext']} (por {r['dpv_terms']})",
                              self._pair_text(r))
            rec.axiom(pair, r["ontopriv_iri"], SKOS.closeMatch, r["gdprtext_iri"])
            rec.generated(pair, building, self.system, out.get("created"))
            rec.g.add((pair, DCTERMS.isPartOf, node))
            rec.add(pair, PROV.wasDerivedFrom, self.files.get("gdprtext_alignment"))
            terms = [t.strip().removeprefix("dpv:") for t in (r.get("dpv_terms") or "").split(";")]
            relations = [x.strip() for x in (r.get("ontopriv_relation") or "").split(";")]
            for term, rel in zip(terms, relations):
                base = self.mapping_nodes.get((r["ontopriv_iri"], str(SKOS[rel]),
                                               "https://w3id.org/dpv#" + term))
                rec.add(pair, PROV.wasDerivedFrom, base)
            if r.get("verdict"):
                said = self._said(cache.get(f"{r['ontopriv_iri']} {r['gdprtext_iri']}", []), r)
                note = rec.entity(f"bridge-review-{key}",
                                  f"IA revisora: {r['ontopriv']} ↔ {r['gdprtext']}, "
                                  f"{r['verdict']}", self._review_text(r))
                rec.generated(note, review, reviewer, said.get("created"))
                rec.g.add((note, DCTERMS.subject, pair))
            n += 1
        self.counts["bridge"] = {"pairs": n, "summary": out.get("summary")}

    @staticmethod
    def _pair_text(r: dict) -> str:
        text = (f"Encadenado por {r.get('dpv_terms')}: en la base {r.get('ontopriv_relation')}, "
                f"en GDPRtEXT {r.get('gdprtext_relation')}.")
        if r.get("ontopriv_source"):
            text += f" En la base: {r['ontopriv_source']}."
        if r.get("gdpr_articles"):
            text += f" En el GDPR: {r['gdpr_articles']}."
        return text

    @staticmethod
    def _said(entries: list, r: dict) -> dict:
        """The cached answer of the reviewing AI that gave this row (the latest one)."""
        same = [e for e in entries if (e.get("answer") or {}).get("verdict") == r["verdict"]
                and (e.get("answer") or {}).get("explanation") == r.get("explanation")]
        return max(same, key=lambda e: e.get("created") or "", default={})

    @staticmethod
    def _review_text(r: dict) -> str:
        return (f"Dijo «{r['verdict']}»" + (f" ({r['ai_relation']})" if r.get("ai_relation")
                                            else "")
                + (f": {r['explanation']}" if r.get("explanation") else ".")
                + " Solo explica: no cambia el par ni entra a la ontología.")


def build_graph(inp: Inputs, s: Settings | None = None) -> tuple[Graph, dict]:
    b = _Base(inp, s or Settings())
    graph = b.build()
    return graph, b.counts


# --------------------------------------------------------------------------- checks

def missing(prov: Graph, alignment: Graph) -> list[tuple]:
    """The SKOS triples of the alignment without a record that says who and when."""
    people = set(prov.subjects(RDF.type, PROV.Person))
    out = []
    for p in MAPPINGS:
        for s, o in alignment.subject_objects(p):
            found = [r for r in prov.subjects(OWL.annotatedSource, s)
                     if (r, OWL.annotatedProperty, p) in prov
                     and (r, OWL.annotatedTarget, o) in prov
                     and set(prov.objects(r, PROV.wasAttributedTo)) & people
                     and any(True for _ in prov.objects(r, PROV.generatedAtTime))]
            if not found:
                out.append((str(s), str(p), str(o)))
    return sorted(out)


def write_module(graph: Graph, folder: Path) -> list[Path]:
    from src.transform.convert import output_format, serialize
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in FORMATS:
        path = folder / f"{MODULE}{output_format(key).extension}"
        path.write_bytes(serialize(graph, key))
        paths.append(path)
    return paths


def same_in_all_formats(files: list[Path]) -> bool:
    from src.transform.convert import load_graph
    graphs = [load_graph(f) for f in files]
    return all(isomorphic(graphs[0], g) for g in graphs[1:])


def validate(graph: Graph, path: Path, cfg: dict | None, engine: str | None,
             inp: Inputs) -> dict:
    """Forma and Lógica on the base (and the bridge, when there is one) plus the provenance:
    what the base already had is inherited, what only the provenance brings is new."""
    from src.alignment import gdpr_bridge as gb
    from src.config import ROOT
    from src.validation import shape_inspector as si
    parts, _ = si.base_parts(cfg)
    if "bridge" in inp.files:
        bridge = inp.files["bridge"]
        parts.append(si._part("bridge", bridge, si.BASE, si._shown(bridge, ROOT)))
    new = si.Part("provenance", si._shown(path, ROOT), si.NEW, graph, _sha256(path), path)
    return gb.validate(parts, new, cfg, engine, name="Base del sistema + procedencia")


def build(cfg: dict | None = None, *, engine: str | None = None,
          out_dir: str | Path | None = None, check: bool = True) -> dict:
    """The module in four formats, its checks and a manifest with the counts."""
    from src.config import load_config
    cfg = cfg if cfg is not None else load_config()
    inp = load_inputs(cfg)
    folder = Path(out_dir) if out_dir else inp.files["candidates"].parent
    s = settings(cfg)
    graph, counts = build_graph(inp, s)
    files = write_module(graph, folder)
    turtle = next(f for f in files if f.suffix == ".ttl")
    people = sorted(graph.subjects(RDF.type, PROV.Person), key=str)
    out = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "iri": PROVENANCE_IRI, "files": [_shown(f) for f in files], "triples": len(graph),
           "same_in_all_formats": same_in_all_formats(files),
           "missing": [list(t) for t in missing(graph, inp.alignment)],
           "counts": counts,
           "people": [{"name": str(graph.value(p, RDFS.label)),
                       "also_signed_as": sorted(str(n) for n in graph.objects(
                           p, URIRef("http://xmlns.com/foaf/0.1/nick")))} for p in people],
           "inputs": {k: {"file": _shown(p), "sha256": _sha256(p)}
                      for k, p in sorted(inp.files.items())},
           "validation": validate(graph, turtle, cfg, engine, inp) if check else None}
    (folder / MANIFEST_FILE).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return out


# --------------------------------------------------------------------------- console

def render(out: dict) -> str:
    from src.alignment import gdpr_bridge as gb
    c = out["counts"]
    m, lc, r = c["mappings"], c["law_concepts"], c["repairs"]
    covered = m["total"] - len(out["missing"])
    lines = ["Procedencia de la base (PROV-O, S10-T06)",
             f"  Módulo: {', '.join(out['files'])} ({out['triples']} tripletas)",
             "  " + ("Los cuatro formatos dan las mismas tripletas." if out["same_in_all_formats"]
                     else "Los cuatro formatos NO dan las mismas tripletas: revisa la escritura."),
             f"  Correspondencias con el DPV: {covered} de {m['total']} con quién y cuándo "
             f"({m['assistant']} confirmando una propuesta del asistente, "
             + counted(m["person"], "decidida por la persona sola",
                       "decididas por la persona sola") + ")",
             f"  Conceptos sin correspondencia: {c['no_match']}, con quién y cuándo",
             f"  Conceptos que la IA tomó de la {LAW}: {lc['total']} "
             f"({counted(lc['approved'], 'nuevo', 'nuevos')}, "
             + counted(lc["duplicate"], "ya existía en OntoPriv", "ya existían en OntoPriv")
             + f", {counted(lc['rejected'], 'rechazado', 'rechazados')}), con su artículo "
             "y quién los decidió",
             f"  Reparaciones: {r['R1']} R1 (varios dominios o rangos → una unión), "
             f"{r['R2']} R2 (un solo tipo)"]
    b = c.get("bridge")
    lines.append(f"  Puente con el GDPR: {counted(b['pairs'], 'par', 'pares')}, cada uno con la "
                 "explicación de la IA revisora" if b else
                 "  Puente con el GDPR: no está construido (py -m src.alignment.gdpr_bridge)")
    for p in out["people"]:
        also = p["also_signed_as"]
        lines.append(f"  Persona: {p['name']}" + (
            " (en el registro firmó también como " + ", ".join(f"«{n}»" for n in also) + ")"
            if also else ""))
    if out["missing"]:
        lines.append(f"  Sin quién y cuándo: {len(out['missing'])}, por ejemplo "
                     f"{' '.join(local(x) for x in out['missing'][0])}")
    lines += gb.validation_lines(out["validation"], "la procedencia")
    return "\n".join(lines)


def find_term(g: Graph, name: str) -> list[str]:
    """IRIs with records whose local name (or IRI) is ``name``."""
    terms = {o for p in (OWL.annotatedSource, OWL.annotatedTarget, RDF.subject, RDF.object,
                         DCTERMS.subject) for o in g.objects(None, p)}
    return sorted(str(t) for t in terms if str(t) == name or local(t) == name)


RECORD_ORDER = ("law-concept", "concept-decision", "mapping", "no-match", "proposal", "repair",
                "bridge-pair")


def _order(record) -> tuple:
    name = local(record)
    kind = next((i for i, k in enumerate(RECORD_ORDER) if name.startswith(k + "-")),
                len(RECORD_ORDER))
    return kind, str(record)


def render_records(g: Graph, term: str) -> str:
    lines = [f"Procedencia de {local(term)} ({term})"]
    for r in sorted(records_about(g, term), key=_order):
        d = describe(g, r)
        lines.append(f"  · {d['label']}")
        if d["comment"]:
            lines.append(f"    {d['comment']}")
        who = " · ".join(x for x in (d["activity"], d["by"], d["at"]) if x)
        if who:
            lines.append(f"    {who}")
        if d["sources"]:
            lines.append(f"    Fuente: {'; '.join(d['sources'])}")
        if d["derived_from"]:
            lines.append(f"    Viene de: {'; '.join(d['derived_from'])}")
        if d["led_to"]:
            lines.append(f"    Llevó a: {'; '.join(d['led_to'])}")
        for n in d["notes"]:
            lines.append(f"    {n['label']} ({n['by'] or n['activity']}): {n['comment']}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.config import load_config
    from src.transform.convert import load_graph
    parser = argparse.ArgumentParser(
        prog="py -m src.provenance.base",
        description="Escribe la procedencia (PROV-O) de la base: quién decidió y cuándo cada "
                    "correspondencia, los conceptos de la ley, las reparaciones y el puente.")
    parser.add_argument("--motor", choices=["jena", "owlrl"], help="motor de la lógica")
    parser.add_argument("--ver", metavar="NOMBRE",
                        help="muestra lo que dice la procedencia de un concepto o propiedad")
    args = parser.parse_args(argv)
    cfg = load_config()
    if args.ver:
        from src.alignment.inherit import base_paths
        path = base_paths(cfg)["candidates"].parent / f"{MODULE}.ttl"
        if not path.exists():
            print(f"Error: falta {path}; corre antes py -m src.provenance.base")
            return 1
        g = load_graph(path)
        terms = find_term(g, args.ver)
        if not terms:
            print(f"La procedencia no dice nada de {args.ver}.")
            return 1
        print("\n\n".join(render_records(g, t) for t in terms))
        return 0
    try:
        out = build(cfg, engine=args.motor)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1
    print(render(out))
    from src.alignment.inherit import base_paths
    print(f"  Guardado en: {base_paths(cfg)['candidates'].parent / MANIFEST_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
