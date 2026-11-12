# PROV-O of each upload (S10-T07): the module <name>-provenance says, for every correspondence
# that entered the ontology, who decided it (base, regla, regla sin IA or usuario), how it was
# produced and in which version; it keeps the ones a later version replaced (as revisions), the
# concepts without correspondence and why, what the AI took from a law and the rule accepted,
# the labels and definitions the AI added and what the AI reviewer said. It reads the
# workspace; it calls no AI and goes in every .zip, in the format of the .zip.
from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from src.console import counted
from src.provenance.model import PROV, People, Recorder, digest, slug, when

MODULE_SUFFIX = "-provenance"
QUERY_FILE = Path(__file__).with_name("records.rq")
WHO = {"base": "base", "rule": "regla", "rule_without_ai": "regla sin IA", "user": "usuario"}
RELATION_IRIS = {f"skos:{n}": SKOS[n] for n in
                 ("exactMatch", "closeMatch", "broadMatch", "narrowMatch", "relatedMatch")}
STEP_TEXT = {"suggested": "sugerida (no entra a la ontología)",
             "no_match": "sin correspondencia (decidido por la regla)",
             "no_proposal": "sin propuesta",
             "undecided": "sin decidir (la IA no estuvo disponible)"}


def relation_iri(relation: str | None) -> URIRef:
    return RELATION_IRIS.get(relation or "", RDFS.seeAlso)


def local(iri: str) -> str:
    """The last part of an IRI or of a key of the base ('ai:Name' -> 'Name')."""
    return re.split(r"[#/:]", str(iri))[-1]


def _when(value) -> str:
    """'2026-11-11T04:12:32+00:00' -> '2026-11-11 04:12 UTC' (the files keep UTC)."""
    return f"{str(value)[:16].replace('T', ' ')} UTC" if value else ""


def _day(value) -> str:
    return str(value or "")[:10]


# --------------------------------------------------------------------------- what it reads

@dataclass
class UploadInputs:
    """What one version of an upload holds (read from its workspace)."""
    workspace: str
    display: str                        # the name of the uploaded file
    mode: str | None
    onto_iri: str
    alignment_iri: str
    version: int
    created: str
    inherited: dict
    decisions: dict
    labels: dict | None = None
    labels_iri: str | None = None
    definitions: dict | None = None
    definitions_iri: str | None = None
    audit: dict | None = None
    records: list = field(default_factory=list)          # the person's log
    law_concepts: dict | None = None
    law_profile: dict | None = None
    inputs: dict = field(default_factory=dict)           # ws.manifest["inputs"]
    steps: dict = field(default_factory=dict)            # ws.manifest["steps"]
    versions: list = field(default_factory=list)         # ws.manifest["versions"]
    history: dict = field(default_factory=dict)          # version -> its created time
    embedding_model: str | None = None


def provenance_iri(onto_iri: str) -> str:
    return onto_iri.rstrip("#/") + MODULE_SUFFIX


def collect(ws, *, inherited: dict, decisions: dict, labels: dict | None,
            definitions: dict | None, audit: dict | None, version: int, created: str,
            onto_iri: str, alignment_iri: str, modules: dict) -> UploadInputs:
    """The inputs of ``build_graph`` from a workspace, with what ``write_user_alignment``
    already loaded."""
    import json
    import os
    from src.alignment import user_decisions as ud
    from src.alignment.user_alignment import MANIFEST_FILE
    law_concepts = law_profile = None
    if (ws.manifest.get("inputs", {}).get("ontology") or {}).get("generated_from") == "law":
        from src.core.law_concepts import load_result as concepts_result
        from src.core.law_profile import load_result as profile_result
        law_concepts, law_profile = concepts_result(ws), profile_result(ws)
    history = {}
    for v in ws.manifest.get("versions") or []:
        path = ws.path / v.get("folder", "") / MANIFEST_FILE
        if path.exists():
            history[int(v["version"])] = json.loads(path.read_text(encoding="utf-8"))["created"]
    return UploadInputs(
        workspace=ws.id, display=ws.manifest["inputs"]["ontology"]["original_name"],
        mode=ws.manifest.get("mode"), onto_iri=onto_iri, alignment_iri=alignment_iri,
        version=version, created=created, inherited=inherited, decisions=decisions or {},
        labels=labels, labels_iri=modules.get("labels"), definitions=definitions,
        definitions_iri=modules.get("definitions"), audit=audit, records=ud.read(ws),
        law_concepts=law_concepts, law_profile=law_profile,
        inputs=ws.manifest.get("inputs") or {}, steps=ws.manifest.get("steps") or {},
        versions=ws.manifest.get("versions") or [], history=history,
        embedding_model=os.environ.get("EMBEDDING_MODEL"))


# --------------------------------------------------------------------------- the graph

class _Upload:
    """Writes the records of one upload into a Recorder, step by step."""

    def __init__(self, inp: UploadInputs, people: dict | None = None):
        self.inp = inp
        self.rec = Recorder(provenance_iri(inp.onto_iri), "provenance", People(people))
        self.counts: dict = {}
        self.nodes: dict = {}               # (key, dpv, relation, by, version) -> record
        self.status: dict = {}              # concept key -> its record without correspondence
        self.rule_nodes: dict = {}          # (key, dpv) -> record of the rule
        self.triples: dict = {}             # record -> the triple it is about

    def at(self, step: str):
        return (self.inp.steps.get(step) or {}).get("at")

    # ------------------------------------------------------------------ build
    def build(self) -> Graph:
        from src.provenance import base as pb
        inp, rec = self.inp, self.rec
        rec.header(f"Procedencia de {inp.display} (PROV-O)",
                   "Quién decidió cada correspondencia de esta subida (base, regla, regla "
                   "sin IA o usuario), cómo se produjo y en qué versión; las que reemplazó "
                   "una versión posterior, como revisiones; los conceptos sin correspondencia "
                   "y por qué; lo que la IA tomó de una ley y aceptó la regla; las etiquetas y "
                   "definiciones que agregó la IA y lo que dijo la IA revisora. Usa PROV-O "
                   "(W3C, 2013). Cada correspondencia vigente es un owl:Axiom con nombre que "
                   "anota su tripleta, como escribe SSSOM un mapeo; una reemplazada es un "
                   "rdf:Statement, que la describe sin afirmarla. Las heredadas apuntan a su "
                   "registro en la procedencia de la base. Nada se importa: todo se "
                   "referencia por IRI.",
                   (inp.alignment_iri, inp.onto_iri, pb.PROVENANCE_IRI, pb.DPV_IRI))
        rec.g.remove((URIRef(rec.iri), OWL.versionInfo, None))
        rec.g.add((URIRef(rec.iri), OWL.versionInfo, Literal(str(inp.version))))
        self.decided_by = rec.annotation_property(
            "decidedBy", {"es": "decidido por", "en": "decided by"},
            "Quién decidió la correspondencia: base (una persona, Sprint 5), regla (el decisor "
            "con IA y la regla calibrada), regla sin IA o usuario (quien subió el archivo, "
            "después de probarlo).")
        self.agents()
        self.files()
        self.activities()
        self.versions()
        self.law()
        self.inheritance()
        self.decider()
        self.user()
        self.labels()
        self.definitions()
        self.review()
        self.close()
        return rec.g

    def agents(self) -> None:
        rec = self.rec
        self.system = rec.agent("agent-system", "ontology-integration-transformer",
                                "El sistema de este TIC: hereda, aplica las reglas y escribe "
                                "los módulos.")
        self.user_agent = rec.person("usuario")
        rec.texts(self.user_agent, None, "Quien subió el archivo (anónimo). Lo que decide "
                                          "después de probar se anota con decidedBy «usuario».")
        self.embeddings = None
        model = self.inp.embedding_model
        if model:
            self.embeddings = rec.agent(f"agent-embeddings-{slug(model)}", model,
                                        "Modelo de embeddings (sentence-transformers).")

    def llm(self, model: str | None):
        if not model:
            return self.rec.agent("agent-llm", "Modelo de lenguaje del sistema",
                                  "El archivo de este paso no dice qué modelo respondió.")
        return self.rec.agent(f"agent-llm-{slug(model)}", model,
                              "Modelo de lenguaje que usa el sistema: propone o explica; lo "
                              "que dice entra a la ontología solo si una regla lo acepta.")

    def plan(self, prompt: dict | None):
        if not prompt or not prompt.get("name"):
            return None
        name, version = prompt["name"], int(prompt.get("version") or 1)
        return self.rec.plan(f"plan-prompt-{name}-v{version}", f"Prompt {name}.v{version}",
                             f"Instrucción que recibió la IA (src/ai/prompts/{name}.v{version}"
                             ".txt).",
                             "sha256:" + prompt["sha256"] if prompt.get("sha256") else None)

    def files(self) -> None:
        rec, inputs = self.rec, self.inp.inputs
        onto = inputs.get("ontology") or {}
        law = inputs.get("law") or {}
        sha = onto.get("sha256") or onto.get("content_sha256")
        made = onto.get("generated_from") == "law"
        shown = onto.get("stored_as" if made else "original_name") or self.inp.display
        self.ontology = rec.file("file-ontology", (f"Perfil de la ley ({shown})" if made else
                                                    f"Tu ontología ({shown})"),
                                 sha, "La ontología que se alinea; nunca se edita.")
        rec.g.add((self.ontology, PROV.specializationOf, URIRef(self.inp.onto_iri)))
        self.law_file = None
        if law:
            self.law_file = rec.file("file-law", f"Ley subida ({law.get('original_name')})",
                                     law.get("sha256"))
        self.law_name = law.get("original_name") or law.get("stored_as")

    def article(self, n: int) -> URIRef:
        node = self.rec.node(f"law-art-{int(n)}")
        if (node, RDF.type, PROV.Entity) not in self.rec.g:
            self.rec.entity(local(node), f"{self.law_name or 'ley'}, art. {int(n)}")
            self.rec.add(node, DCTERMS.isPartOf, self.law_file)
        return node

    # ------------------------------------------------------------------ activities
    def activities(self) -> None:
        rec, inp = self.rec, self.inp
        settings = inp.inherited.get("settings") or {}
        self.inheriting = rec.activity(
            "activity-inheritance", "Herencia de la base",
            "Cada concepto que coincide con uno de la base (mismo IRI, mismo nombre o muy "
            f"parecido: puntaje ≥ {settings.get('duplicate_threshold', 0.75)} con "
            f"{settings.get('weight_lexical', 0.4)} nombre + "
            f"{settings.get('weight_semantic', 0.6)} significado) recibe las correspondencias que "
            "una persona decidió en la base (S6-T05).",
            agents=[a for a in (self.system, self.embeddings) if a is not None],
            used=[self.ontology], ended=self.at("inherit"))
        d = inp.decisions
        summary = d.get("summary") or {}
        models = sorted({e.get("model") for e in d.get("entries", []) if e.get("model")})
        self.decider_llm = self.llm(models[0] if models else (summary.get("ai") or {})
                                    .get("model")) if d.get("entries") else None
        rule = rec.plan("plan-decider-rule", "Regla del decisor", summary.get("rule")
                        or "La regla calibrada del decisor (S7-T05).")
        self.deciding = rec.activity(
            "activity-decider", "Decisor: candidatos del DPV, IA y regla",
            "Para cada concepto que no heredó nada, la IA elige entre los 10 términos del DPV "
            "más parecidos, con los precedentes de la base, y la regla fija acepta solo lo que "
            "cumple; lo demás queda sugerido o sin correspondencia (S7-T06).",
            agents=[a for a in (self.system, self.embeddings) if a is not None],
            used=[self.ontology], ended=self.at("propose"), plan=rule)
        if self.decider_llm is not None:
            self._associate(self.deciding, self.decider_llm, self.plan(d.get("prompt")))
        self.writing = rec.activity(
            "activity-alignment", "Alineación automática (versión 1)",
            "Escribe el módulo de alineación con lo heredado y lo que aceptó la regla, y "
            "anota en cada correspondencia quién la decidió y su versión (S6-T07).",
            agents=[self.system], used=[self.ontology],
            ended=inp.history.get(1) or (inp.created if inp.version == 1 else None))

    def _associate(self, activity, agent, plan) -> None:
        rec = self.rec
        rec.g.add((activity, PROV.wasAssociatedWith, agent))
        if plan is not None:
            assoc = rec.node(f"{local(activity)}-association-{local(agent)}")
            rec.g.add((assoc, RDF.type, PROV.Association))
            rec.g.add((assoc, PROV.agent, agent))
            rec.g.add((assoc, PROV.hadPlan, plan))
            rec.g.add((activity, PROV.qualifiedAssociation, assoc))

    def versions(self) -> None:
        rec, inp = self.rec, self.inp
        until = {int(v["version"]): v.get("until") for v in inp.versions}
        self.version_nodes = {}
        self.user_activities = {}
        by_version: dict[int, list] = {}
        for d in inp.records:
            by_version.setdefault(int(d.snapshot.get("version") or inp.version), []).append(d)
        rule = rec.plan("plan-user-rule", "Regla fija de los cambios del usuario",
                        "El concepto existe y se puede alinear, el término es del DPV 2.3 y de "
                        "su tipo, la relación es SKOS, el cambio cambia algo y dice qué "
                        "correspondencia reemplaza; un cambio por concepto (S10-T03).")
        for v in range(2, inp.version + 1):
            done = sorted(d.decided_at for d in by_version.get(v, []) if d.decided_at)
            self.user_activities[v] = rec.activity(
                f"activity-user-v{v}", f"Cambios del usuario (versión {v})",
                "Lo que el usuario decidió después de probar la versión anterior: escrito a "
                "mano o elegido en las preguntas; una regla fija revisa cada cambio y la "
                "alineación se escribe de nuevo, sin volver a llamar a la IA.",
                agents=[self.user_agent], started=done[0] if done else None,
                ended=done[-1] if done else None, plan=rule)
        for v in range(1, inp.version + 1):
            node = rec.entity(f"version-{v}", f"Versión {v} de la alineación",
                              "Lo que dio esta versión: sus cuatro descargas quedan en history/"
                              f"v{v}/ del espacio." if v < inp.version else
                              "La versión vigente de la alineación.")
            rec.g.add((node, PROV.specializationOf, URIRef(inp.alignment_iri)))
            rec.g.add((node, OWL.versionInfo, Literal(str(v))))
            rec.g.add((node, PROV.wasGeneratedBy,
                       self.writing if v == 1 else self.user_activities[v]))
            rec.add(node, PROV.generatedAtTime,
                    when(inp.created if v == inp.version else inp.history.get(v)))
            if v > 1:
                rec.g.add((node, PROV.wasRevisionOf, rec.node(f"version-{v - 1}")))
            if v < inp.version:
                rec.add(node, PROV.invalidatedAtTime, when(until.get(v)))
            self.version_nodes[v] = node
        self.by_version = by_version

    # ------------------------------------------------------------------ the law ("Una ley")
    def law(self) -> None:
        rec, inp = self.rec, self.inp
        concepts, profile = inp.law_concepts, inp.law_profile
        if not concepts:
            self.counts["law_concepts"] = None
            return
        extraction = concepts.get("extraction") or {}
        llm = self.llm(extraction.get("model") or (concepts.get("ai") or {}).get("model"))
        extracting = rec.activity(
            "activity-law-extraction", f"Extracción de conceptos de {self.law_name} con IA",
            "La IA lee la ley por lotes de artículos y propone conceptos, cada uno con los "
            "artículos de los que sale (S7-T09); solo propone." + (
                " " + counted(extraction.get("articles_processed", 0), "artículo procesado",
                              "artículos procesados") + ", "
                + counted(extraction.get("concepts", 0), "concepto propuesto",
                          "conceptos propuestos") + "."
                if extraction.get("articles_processed") is not None else ""),
            agents=[llm, self.system], used=[x for x in [self.law_file] if x is not None],
            ended=self.at("extract"), plan=self.plan(concepts.get("prompt")))
        accepting = rec.activity(
            "activity-law-acceptance", "Aceptación de los conceptos por la regla",
            "Una regla fija acepta un concepto cuando cita un artículo de la ley, está bien "
            "formado y no se repite (S7-T09, decisión D1); nadie los revisa uno por uno.",
            agents=[self.system], ended=self.at("accept"),
            plan=rec.plan("plan-law-rule", "Regla D1 de los conceptos de una ley",
                          concepts.get("rule")))
        building = rec.activity(
            "activity-law-profile", "Perfil de la ley",
            "Escribe el perfil: reutiliza el IRI de OntoPriv-Core cuando el concepto ya está en "
            "el núcleo, lo enlaza cuando es muy parecido y crea uno nuevo si no (D6); cada "
            "concepto lleva su artículo (dct:source).",
            agents=[self.system], informed_by=[accepting], ended=self.at("profile"))
        rec.g.add((self.ontology, PROV.wasGeneratedBy, building))
        placed = {e["name"]: e for e in (profile or {}).get("entries", [])}
        made = Counter()
        for e in concepts.get("entries", []):
            name = e["name"]
            arts = [self.article(n) for n in sorted(set(e.get("articles") or []))]
            kind = {"class": "clase", "property": "propiedad"}.get(e.get("kind"), "concepto")
            extracted = rec.entity(
                f"law-concept-{digest(self.inp.onto_iri, name)}",
                f"{e.get('label') or name} ({name})",
                f"Propuesto por la IA con base en {self._articles(e.get('articles'))}, como "
                f"{kind}" + (f": {e['definition']}" if e.get("definition") else "."))
            rec.generated(extracted, extracting, llm)
            for a in arts:
                rec.g.add((extracted, PROV.hadPrimarySource, a))
            where = placed.get(name) or {}
            decision = rec.entity(
                f"law-acceptance-{digest(self.inp.onto_iri, name)}",
                f"{name}: {self._acceptance_words(e)}", self._acceptance_text(e, where))
            rec.generated(decision, accepting, self.system, self.at("accept"))
            rec.g.add((decision, PROV.wasDerivedFrom, extracted))
            for a in arts:
                rec.g.add((decision, PROV.hadPrimarySource, a))
            if where.get("iri"):
                rec.g.add((extracted, DCTERMS.subject, URIRef(where["iri"])))
                rec.g.add((decision, DCTERMS.subject, URIRef(where["iri"])))
            else:
                rec.g.add((extracted, PROV.wasInvalidatedBy, accepting))
            made[e.get("status")] += 1
        self.counts["law_concepts"] = {"total": sum(made.values()), **dict(sorted(made.items()))}

    def _articles(self, numbers) -> str:
        nums = [str(n) for n in sorted({int(n) for n in numbers or []})]
        law = self.law_name or "la ley"
        if not nums:
            return law
        if len(nums) == 1:
            return f"el art. {nums[0]} de {law}"
        return f"los arts. {', '.join(nums[:-1])} y {nums[-1]} de {law}"

    @staticmethod
    def _acceptance_words(e: dict) -> str:
        return {"accepted": "aceptado por la regla", "repeated": "repetido",
                "rejected": "rechazado por la regla"}.get(e.get("status"), e.get("status") or "")

    @staticmethod
    def _acceptance_text(e: dict, where: dict) -> str:
        status = e.get("status")
        if status == "repeated":
            return (f"Repetido: se une a {e.get('repeated_of')}, que conserva sus artículos; no "
                    "se escribe aparte.")
        if status == "rejected":
            return "Rechazado por la regla: " + ("; ".join(e.get("reasons") or []) or
                                                 "no cumple la regla") + "."
        text = "Aceptado por la regla: cita su artículo, está bien formado y no se repite."
        joined = e.get("joined") or []
        if joined:
            text += f" Se le unieron: {', '.join(joined)}."
        core = local(where.get("core_iri") or "")
        if where.get("status") == "reused":
            text += f" Reutiliza {core} de OntoPriv-Core (el mismo IRI)."
        elif where.get("status") == "linked":
            text += f" Es nuevo en el perfil y se enlaza con {core} de OntoPriv-Core."
        elif where.get("status") == "new":
            text += " Es nuevo en el perfil."
        return text

    # ------------------------------------------------------------------ the correspondences
    def correspondence(self, key: str, name: str, dpv: str, relation: str, by: str,
                       version, comment: str, activity, agent, at) -> URIRef:
        """The record of one correspondence (one decision): stable IRI from the module, the
        triple, who decided it and the version it was decided in."""
        rec, prop = self.rec, relation_iri(relation)
        version = int(version or 1)
        ident = (key, dpv, str(prop), by, version)
        if ident in self.nodes:
            return self.nodes[ident]
        ident_text = digest(self.inp.alignment_iri, key, str(prop), dpv, by, str(version))
        node = rec.entity(
            f"correspondence-{ident_text}",
            f"{name} {relation or 'rdfs:seeAlso'} dpv:{local(dpv)} ({WHO.get(by, by)}, versión "
            f"{version})", comment)
        rec.generated(node, activity, agent, at)
        rec.g.add((node, self.decided_by, Literal(WHO.get(by, by), lang="es")))
        rec.g.add((node, OWL.versionInfo, Literal(str(version))))
        rec.g.add((node, DCTERMS.subject, URIRef(key)))
        self.nodes[ident] = node
        self.triples[node] = (key, prop, dpv)
        return node

    def inheritance(self) -> None:
        from src.alignment.inherit import METHOD_LABELS
        from src.provenance import base as pb
        rec = self.rec
        made = Counter()
        for item in self.inp.inherited.get("inherited", []):
            how = (f"Heredada de {item['base_name']} de la base "
                   f"({METHOD_LABELS.get(item['method'], item['method'])}"
                   + (f", puntaje {item['score']}" if item.get("method") == "similar" else "")
                   + ")")
            if item.get("status") == "aligned" and item.get("mappings"):
                for m in item["mappings"]:
                    who = m.get("reviewer") or "una persona"
                    node = self.correspondence(
                        item["key"], item["name"], m["dpv_iri"], m["relation"], "base", 1,
                        f"{how}: la decidió {rec.people.name(who)} el {_day(m.get('decided_at'))}"
                        " en la base.", self.inheriting,
                        rec.person(who) if m.get("reviewer") else self.system, self.at("inherit"))
                    rec.g.add((node, PROV.wasDerivedFrom, pb.mapping_record(
                        item["base_iri"], relation_iri(m["relation"]), m["dpv_iri"])))
                    made["mappings"] += 1
            else:
                node = rec.entity(
                    f"inherited-no-match-{digest(self.inp.alignment_iri, item['key'])}",
                    f"{item['name']}: sin correspondencia (heredado de la base)",
                    f"{how}: en la base no tiene correspondencia con el DPV.")
                rec.generated(node, self.inheriting, self.system, self.at("inherit"))
                rec.g.add((node, DCTERMS.subject, URIRef(item["key"])))
                rec.g.add((node, PROV.wasDerivedFrom, pb.no_match_record(item["base_iri"])))
                self.status[item["key"]] = node
                made["no_match"] += 1
        self.counts["inherited"] = dict(made)

    def decider(self) -> None:
        from src.alignment import propose as dec
        d = self.inp.decisions
        rule = d.get("rule") or {}
        threshold = d.get("threshold_without_ai")
        made = Counter()
        for e in d.get("entries", []):
            if e["status"] == dec.ACCEPTED:
                node = self.correspondence(
                    e["key"], e["name"], e["dpv_iri"], e.get("relation"), e["decided_by"],
                    e.get("version"), self._accepted_text(e, d, threshold), self.deciding,
                    self.system, self.at("propose"))
                self.rule_nodes[(e["key"], e["dpv_iri"])] = node
                made["accepted"] += 1
                continue
            node = self.rec.entity(
                f"decider-{digest(self.inp.alignment_iri, e['key'])}",
                f"{e['name']}: {STEP_TEXT.get(e['status'], e['status'])}",
                self._not_accepted_text(e, rule, threshold))
            self.rec.generated(node, self.deciding, self.system, self.at("propose"))
            self.rec.g.add((node, DCTERMS.subject, URIRef(e["key"])))
            self.status.setdefault(e["key"], node)
            made[e["status"]] += 1
        self.counts["decider"] = dict(sorted(made.items()))

    def _accepted_text(self, e: dict, d: dict, threshold) -> str:
        if e["decided_by"] == "rule_without_ai":
            return (f"Aceptada sin IA: el mejor candidato tenía puntaje {e.get('score')} (mínimo "
                    f"{threshold}); sin tipo SKOS (rdfs:seeAlso).")
        text = (f"Aceptada automáticamente por la regla "
                f"({(d.get('summary') or {}).get('rule', '')}).")
        return text + self._ai_said(e)

    @staticmethod
    def _ai_said(e: dict) -> str:
        text = ""
        model = f" ({e['model']})" if e.get("model") else ""
        if e.get("confidence"):
            text += f" La IA{model} respondió con confianza {e['confidence']}"
            text += f": «{e['justification']}»." if e.get("justification") else "."
        elif e.get("justification"):
            text += f" La IA{model} dijo: «{e['justification']}»."
        if e.get("rank"):
            text += f" Candidato n.º {e['rank']} del DPV (puntaje {e.get('score')})."
        precedents = [local(p) for p in e.get("precedents") or []]
        if precedents:
            text += f" Precedentes de la base que vio la IA: {', '.join(precedents)}."
        return text

    def _not_accepted_text(self, e: dict, rule: dict, threshold) -> str:
        status = e["status"]
        if status == "suggested":
            why = []
            if "property" in (e.get("kinds") or []) and not rule.get("property_relations"):
                why.append("las propiedades nunca son automáticas")
            if e.get("relation") and rule.get("relations") and \
                    e["relation"] not in rule["relations"] and "property" not in (
                        e.get("kinds") or []):
                why.append(f"no acepta {e['relation']}")
            if e.get("score") is not None and rule.get("min_score") is not None and \
                    e["score"] < rule["min_score"]:
                why.append(f"puntaje {e['score']} < {rule['min_score']}")
            return (f"La IA propuso {e.get('relation')} dpv:{local(e.get('dpv_iri') or '')} con "
                    "confianza alta, pero la regla no la acepta"
                    + (f" ({'; '.join(why)})" if why else "") + ": no entra a la ontología."
                    + self._ai_said({**e, "confidence": None, "model": None,
                                     "justification": e.get("justification")}))
        if status == "no_match":
            return "La IA dijo que ningún candidato del DPV corresponde." + self._ai_said(
                {**e, "rank": None})
        if status == "undecided":
            return (f"La IA no estuvo disponible y la regla sin IA no encontró un candidato con "
                    f"puntaje ≥ {threshold}.")
        return ("La regla no aceptó ni sugirió nada: la IA no respondió con la confianza que "
                "pide." + self._ai_said({**e, "rank": None}))

    # ------------------------------------------------------------------ the person's choices
    def user(self) -> None:
        rec = self.rec
        made = Counter()
        groups: dict = {}
        for d in self.inp.records:
            ver = int(d.snapshot.get("version") or self.inp.version)
            groups.setdefault((d.concept_key, ver, d.decided_at), []).append(d)
        for (key, ver, at), records in sorted(groups.items(), key=lambda kv: (kv[0][2] or "",
                                                                               kv[0][0])):
            snap = records[0].snapshot
            after = {_ident(m) for m in snap.get("after") or []}
            removed = [m for m in snap.get("before") or [] if _ident(m) not in after]
            activity = self.user_activities.get(ver)
            name = self._name(key)
            gone = []
            for m in removed:
                old = self.correspondence(key, name, m["dpv_iri"], m["relation"],
                                          m["decided_by"], m.get("version"),
                                          "Correspondencia que reemplazó el usuario.",
                                          self.writing, self.system, None)
                rec.g.add((old, PROV.wasInvalidatedBy, activity))
                rec.add(old, PROV.invalidatedAtTime, when(at))
                gone.append(old)
            approved = [d for d in records if d.decision == "approved"]
            for d in approved:
                node = self.correspondence(
                    key, name, d.dpv_iri, d.relation, "user", ver,
                    f"Decidida por el usuario (versión {ver}). Antes: "
                    f"{self._state(snap.get('before') or [])}."
                    + (f" Pedido: {d.note}" if d.note else ""),
                    activity, self.user_agent, d.decided_at)
                for old in gone:
                    rec.g.add((node, PROV.wasRevisionOf, old))
                if not gone and key in self.status:
                    rec.g.add((node, PROV.wasRevisionOf, self.status[key]))
                    rec.g.add((self.status[key], PROV.wasInvalidatedBy, activity))
                    rec.add(self.status[key], PROV.invalidatedAtTime, when(at))
                made["approved"] += 1
            if not approved:
                node = rec.entity(
                    f"user-removal-{digest(self.inp.alignment_iri, key, str(ver))}",
                    f"{name}: quitada por el usuario (versión {ver})",
                    f"El usuario quitó {self._state(removed)}."
                    + (f" Pedido: {records[0].note}" if records[0].note else ""))
                rec.generated(node, activity, self.user_agent, at)
                rec.g.add((node, DCTERMS.subject, URIRef(key)))
                rec.g.add((node, OWL.versionInfo, Literal(str(ver))))
                for old in gone:
                    rec.g.add((node, PROV.wasRevisionOf, old))
                made["removed"] += 1
        self.counts["user"] = dict(made)

    def _name(self, key: str) -> str:
        from src.alignment import user_decisions as ud
        if not hasattr(self, "_index"):
            self._index = ud.concepts(self.inp.inherited, self.inp.decisions)
        return (self._index.get(key) or {}).get("name") or local(key)

    @staticmethod
    def _state(maps: list[dict]) -> str:
        from src.alignment import user_decisions as ud
        return ud.state_text(maps)

    # ------------------------------------------------------------------ what the AI added
    def labels(self) -> None:
        result = self.inp.labels
        if not result or not self.inp.labels_iri:
            self.counts["labels"] = 0
            return
        rec = self.rec
        module = rec.entity("module-labels", "Módulo de etiquetas",
                            "Las etiquetas en español e inglés que le faltaban a la ontología.")
        rec.g.add((module, PROV.specializationOf, URIRef(self.inp.labels_iri)))
        llm = self.llm((result.get("ai") or {}).get("model"))
        writing = rec.activity(
            "activity-labels", "Etiquetas es/en con IA",
            "La IA traduce la etiqueta que falta con el glosario de la base; ninguna etiqueta "
            "original cambia (S7-T08).", agents=[llm], ended=self.at("labels"),
            plan=self.plan(result.get("prompt")))
        rec.g.add((module, PROV.wasGeneratedBy, writing))
        n = 0
        for e in result.get("entries", []):
            added = e.get("added") or {}
            if not added:
                continue
            node = rec.entity(
                f"labels-{digest(self.inp.labels_iri, e['key'])}",
                f"{e['name']}: etiquetas agregadas por la IA",
                "La IA agregó " + ", ".join(f"«{text}» ({lang})"
                                            for lang, text in sorted(added.items()))
                + ". Tus etiquetas no cambian.")
            rec.generated(node, writing, llm, self.at("labels"))
            rec.g.add((node, DCTERMS.subject, URIRef(e["key"])))
            rec.g.add((node, DCTERMS.isPartOf, module))
            n += 1
        self.counts["labels"] = n

    def definitions(self) -> None:
        from src.alignment import law_definitions as ld
        result = self.inp.definitions
        entries = ld.added(result)
        if not entries or not self.inp.definitions_iri:
            self.counts["definitions"] = 0
            return
        rec = self.rec
        module = rec.entity("module-definitions", "Módulo de definiciones de la ley",
                            "Las definiciones que le faltaban a la ontología, tomadas de la ley.")
        rec.g.add((module, PROV.specializationOf, URIRef(self.inp.definitions_iri)))
        llm = self.llm((result.get("ai") or {}).get("model"))
        writing = rec.activity(
            "activity-definitions", "Definiciones tomadas de la ley con IA",
            "Para los conceptos sin definición: la de un concepto que la IA extrajo de la ley, "
            "o una que la IA escribe con sus artículos; ninguna definición original cambia "
            "(S8-T03).", agents=[llm], ended=self.at("definitions"),
            plan=self.plan(result.get("prompt")))
        rec.g.add((module, PROV.wasGeneratedBy, writing))
        law = (result.get("law") or {}).get("file") or self.law_name
        for e in entries:
            node = rec.entity(
                f"definition-{digest(self.inp.definitions_iri, e['key'])}",
                f"{e['name']}: definición de la ley",
                f"{ld.source_text(e, law)}: «{e['definition']}».")
            rec.generated(node, writing, llm, self.at("definitions"))
            rec.g.add((node, DCTERMS.subject, URIRef(e["key"])))
            rec.g.add((node, DCTERMS.isPartOf, module))
            if e.get("article") is not None:
                rec.g.add((node, PROV.hadPrimarySource, self.article(e["article"])))
        self.counts["definitions"] = len(entries)

    def review(self) -> None:
        audit = self.inp.audit
        if not audit:
            self.counts["reviews"] = 0
            return
        rec = self.rec
        llm = self.llm((audit.get("ai") or {}).get("model"))
        reviewing = rec.activity(
            "activity-audit", "IA revisora",
            "La IA lee cada correspondencia que aceptó la regla y dice si es coherente, dudosa "
            "o incoherente; solo explica: no cambia lo aceptado (S8-T09).",
            agents=[llm], ended=audit.get("created"), plan=self.plan(audit.get("prompt")))
        n = 0
        for e in audit.get("entries", []):
            record = self.rule_nodes.get((e.get("key"), e.get("dpv_iri")))
            if record is None or not e.get("verdict"):
                continue
            note = rec.entity(
                f"review-{digest(str(record))}",
                f"IA revisora: {e.get('name')} dpv:{local(e.get('dpv_iri') or '')}, "
                f"{e['verdict']}",
                f"Dijo «{e['verdict']}»" + (f" ({e['ai_relation']})" if e.get("ai_relation")
                                             else "")
                + (f": {e['explanation']}" if e.get("explanation") else ".")
                + " Solo explica: no cambia lo aceptado.")
            rec.generated(note, reviewing, llm, audit.get("created"))
            rec.g.add((note, DCTERMS.subject, record))
            n += 1
        self.counts["reviews"] = n

    # ------------------------------------------------------------------ what holds now
    def close(self) -> None:
        """The records whose correspondence is in this version are owl:Axiom of its triple;
        the replaced ones are rdf:Statement (described, not asserted)."""
        from src.alignment import user_decisions as ud
        final = ud.replay(ud.system_state(self.inp.inherited, self.inp.decisions),
                          self.inp.records)
        rec = self.rec
        current = Counter()
        replaced = 0
        for (key, dpv, prop, by, version), node in sorted(self.nodes.items(),
                                                          key=lambda kv: str(kv[1])):
            m = (final.get(key) or {}).get(dpv)
            now = (m is not None and str(relation_iri(m["relation"])) == prop
                   and m["decided_by"] == by and int(m.get("version") or 1) == version)
            subject, p, target = self.triples[node]
            if now:
                rec.axiom(node, subject, p, target)
                rec.g.add((node, DCTERMS.isPartOf, self.version_nodes[self.inp.version]))
                current[WHO.get(by, by)] += 1
            else:
                rec.g.add((node, RDF.type, RDF.Statement))
                rec.g.add((node, RDF.subject, URIRef(subject)))
                rec.g.add((node, RDF.predicate, p))
                rec.g.add((node, RDF.object, URIRef(target)))
                replaced += 1
        self.counts["correspondences"] = {"current": dict(sorted(current.items())),
                                          "total": sum(current.values()),
                                          "replaced": replaced}
        self.counts["versions"] = self.inp.version


def _ident(m: dict) -> tuple:
    """A correspondence as the person's log lists it, before and after a change."""
    return (m["dpv_iri"], relation_iri(m["relation"]), m["decided_by"],
            int(m.get("version") or 1))


def build_graph(inp: UploadInputs, people: dict | None = None) -> tuple[Graph, dict]:
    u = _Upload(inp, people)
    graph = u.build()
    return graph, u.counts


# --------------------------------------------------------------------------- checks

WRITTEN = (*RELATION_IRIS.values(), RDFS.seeAlso)


def missing(prov: Graph, alignment: Graph) -> list[tuple]:
    """The correspondences of the alignment module without a record that says who decided
    them (an owl:Axiom of that triple with decidedBy and prov:wasAttributedTo)."""
    out = []
    for p in WRITTEN:
        for s, o in alignment.subject_objects(p):
            if not isinstance(o, URIRef) or not str(o).startswith("https://w3id.org/dpv"):
                continue
            found = [r for r in prov.subjects(OWL.annotatedSource, s)
                     if (r, OWL.annotatedProperty, p) in prov
                     and (r, OWL.annotatedTarget, o) in prov
                     and any(str(x).endswith("#decidedBy") for x in prov.predicates(r))
                     and (r, PROV.wasAttributedTo, None) in prov]
            if not found:
                out.append((str(s), str(p), str(o)))
    return sorted(out)


# --------------------------------------------------------------------------- writing

def write(ws, *, out: Path, stem: str, alignment: Graph, cfg: dict | None = None,
          **kwargs) -> dict:
    """Write output/<stem>-provenance.ttl (``kwargs``: those of ``collect``); what goes to
    the manifest, with how many correspondences of ``alignment`` say nobody decided them."""
    from src.config import load_config
    from src.provenance.base import settings
    from src.transform.convert import serialize
    inp = collect(ws, **kwargs)
    people = settings(cfg if cfg is not None else load_config()).people
    graph, counts = build_graph(inp, people)
    name = f"{stem}{MODULE_SUFFIX}.ttl"
    (out / name).write_bytes(serialize(graph, "turtle"))
    return {"iri": provenance_iri(inp.onto_iri), "file": f"output/{name}",
            "triples": len(graph), "counts": counts,
            "without_who": len(missing(graph, alignment))}


# --------------------------------------------------------------------------- reading it

def query_text() -> str:
    return QUERY_FILE.read_text(encoding="utf-8")


def records_of(g: Graph, name: str) -> list[dict]:
    """The fixed SPARQL query (records.rq) with the concept's name or IRI bound."""
    rows = g.query(query_text(), initBindings={"name": Literal(name)})
    keys = [str(v) for v in rows.vars]
    return [{k: (str(r[k]) if r[k] is not None else None) for k in keys} for r in rows]


def render_records(rows: list[dict], name: str) -> str:
    if not rows:
        return f"La procedencia no dice nada de {name}."
    lines = [f"Procedencia de {name} ({rows[0]['concept']})"]
    for r in rows:
        head = f"  · {r['label']}"
        if r.get("replaced"):
            head += f" — reemplazada el {_when(r['replaced'])}"
        lines.append(head)
        if r.get("comment"):
            lines.append(f"    {r['comment']}")
        if r.get("decidedBy"):
            lines.append(f"    Decidida por: {r['decidedBy']}"
                         + (f" ({r['who']})" if r.get("who") and r["who"] != r["decidedBy"]
                            else ""))
            who = " · ".join(x for x in (r.get("activity"), _when(r.get("at"))) if x)
        else:
            who = " · ".join(x for x in (r.get("activity"), r.get("who"), _when(r.get("at")))
                             if x)
        if who:
            lines.append(f"    {who}")
        if r.get("revises"):
            lines.append(f"    Revisa: {r['revises']}")
        if r.get("sources"):
            lines.append(f"    Fuente: {r['sources']}")
        if r.get("notes"):
            lines.append(f"    {r['notes']}")
    return "\n".join(lines)


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.user_alignment import load_manifest
    from src.config import load_config
    from src.transform.convert import load_graph
    from src.validation.shape_inspector import open_workspace
    from src.workspace.store import WorkspaceError
    parser = argparse.ArgumentParser(
        prog="py -m src.provenance.upload",
        description="Muestra la procedencia (PROV-O) de un concepto de una subida, con la "
                    "consulta SPARQL fija src/provenance/records.rq.")
    parser.add_argument("--espacio", required=True, help="el id o la carpeta del espacio")
    parser.add_argument("--ver", required=True, metavar="NOMBRE",
                        help="el nombre (o el IRI) de un concepto de esa subida")
    args = parser.parse_args(argv)
    try:
        ws = open_workspace(args.espacio, load_config())
    except (WorkspaceError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1
    entry = (load_manifest(ws) or {}).get("provenance")
    if not entry:
        print("Error: esta subida todavía no tiene procedencia; vuelve a correrla: "
              f"py -m src.alignment.user_alignment --espacio {ws.id}")
        return 1
    rows = records_of(load_graph(ws.path / entry["file"]), args.ver)
    print(render_records(rows, args.ver))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(_run())
