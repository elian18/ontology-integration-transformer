"""The base in one file for its documentation (S11-T05): the four modules under one header with
the authorship and the credit to OntoPriv, and a sentence per concept with its correspondences.
WIDOCO reads this file; the modules of data/output are not changed."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS

from src.console import counted
from src.core.dpv_proximity import _humanize as humanize
from src.transform.merge import merge

DOC_IRI = "https://elian18.github.io/ontology-integration-transformer/"   # its GitHub Pages
DOC_FILE = "widoco/ontopriv-base.ttl"                 # inside data/output, ignored by git
MODULES = ("ontopriv-core", "profile-ecuador-lopdp", "ontopriv-dpv-alignment",
           "ontopriv-gdpr-alignment")
VERSION = "1.1.0"

SCHEMA = Namespace("https://schema.org/")
BIBO = Namespace("http://purl.org/ontology/bibo/")
DPV = "https://w3id.org/dpv#"
GDPRTEXT = "http://purl.org/adaptcentre/openscience/ontologies/GDPRtEXT#"
CHAINED_BY = URIRef("http://www.semanticweb.org/profiles/ontopriv-gdpr-alignment#chainedBy")
LOPDP = "http://www.semanticweb.org/ley-organica-proteccion-datos-personales#"
DESCRIPTION = URIRef(LOPDP + "description")           # where OntoPriv writes its descriptions
BASE_NAMESPACES = (LOPDP, "http://www.semanticweb.org/franc/ontologies/2023/9/OntologiaLOPDP#",
                   "http://www.semanticweb.org/profiles/ecuador-lopdp#")
TERM_TYPES = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.NamedIndividual)
MAPPINGS = (SKOS.exactMatch, SKOS.closeMatch, SKOS.broadMatch, SKOS.narrowMatch,
            SKOS.relatedMatch)

AUTHOR = "Elian Alexander Gavilanes Miranda"
INSTITUTION = "Escuela Politécnica Nacional (EPN)"
INSTITUTION_URL = "https://www.epn.edu.ec"
LICENSE = "https://www.apache.org/licenses/LICENSE-2.0"
REPOSITORY = "https://github.com/elian18/ontology-integration-transformer"
ONTOPRIV = "https://github.com/kelvinojedaepn/OntoPriv"
ONTOPRIV_DOI = "https://doi.org/10.1109/CLEI64178.2024.10700326"
TITLE = {"es": "Base de conocimiento legal de privacidad: OntoPriv-LOPDP alineada con el DPV "
               "y el GDPR",
         "en": "Privacy legal knowledge base: OntoPriv-LOPDP aligned with the DPV and the GDPR"}
ABSTRACT = {
    "es": "Base de conocimiento legal construida a partir de OntoPriv y de la Ley Orgánica de "
          "Protección de Datos Personales del Ecuador (LOPDP), alineada con el Data Privacy "
          "Vocabulary (DPV) y enlazada con el GDPR por medio de GDPRtEXT.",
    "en": "Legal knowledge base built from OntoPriv and Ecuador's Organic Law on Personal Data "
          "Protection (LOPDP), aligned with the Data Privacy Vocabulary (DPV) and linked to the "
          "GDPR through GDPRtEXT."}
DESCRIPTION_TEXT = {
    "es": "Reúne en un solo documento los cuatro módulos de la base: el núcleo reutilizable "
          "(OntoPriv-Core), el perfil de Ecuador (LOPDP), la alineación con el DPV aprobada por "
          "revisión humana y el puente con el GDPR (GDPRtEXT). Basada en OntoPriv, de G. "
          "Suntaxi, K. Ojeda y F. Rodríguez (CLEI 2024), y en su versión basada en el DPV, de "
          "G. Suntaxi, K. Ojeda, F. Rodríguez, P. del Hierro, J. Miño y D. A. Flores (2025). "
          "La frase de cada concepto con sus correspondencias se escribe solo para esta "
          "documentación; los módulos del repositorio no la tienen. Algunos términos de OntoPriv "
          "tienen dos tipos a la vez, por ejemplo clase y propiedad (OWL Full): su ficha no "
          "muestra la frase, pero la tabla de abajo sí.",
    "en": "Brings together in one document the four modules of the base: the reusable core "
          "(OntoPriv-Core), the Ecuador profile (LOPDP), the alignment with the DPV approved by "
          "human review and the bridge with the GDPR (GDPRtEXT). Based on OntoPriv, by G. "
          "Suntaxi, K. Ojeda and F. Rodríguez (CLEI 2024), and on its DPV-based version, by "
          "G. Suntaxi, K. Ojeda, F. Rodríguez, P. del Hierro, J. Miño and D. A. Flores (2025). "
          "The sentence of each concept with its correspondences is written only for this "
          "documentation; the modules of the repository do not have it. Some terms of OntoPriv "
          "have two types at once, for example class and property (OWL Full): their entry does "
          "not show the sentence, but the table below does."}
CITATION = (f"E. A. Gavilanes Miranda, «{TITLE['es']}», versión {VERSION}, Escuela Politécnica "
            "Nacional, 2026.")
VERBS = {"es": {"exactMatch": "es igual a", "closeMatch": "se parece a",
                "broadMatch": "es un tipo de", "narrowMatch": "es más general que",
                "relatedMatch": "está relacionado con"},
         "en": {"exactMatch": "is the same as", "closeMatch": "is close to",
                "broadMatch": "is a kind of", "narrowMatch": "is broader than",
                "relatedMatch": "is related to"}}
TABLE = {"es": ("Correspondencias de la base", "Concepto", "Artículos de la LOPDP"),
         "en": ("Correspondences of the base", "Concept", "Articles of the LOPDP")}
LEADS = {"es": ("Correspondencias con el DPV: {}.", "Puente con el GDPR: {}.", " (por {})"),
         "en": ("Correspondences with the DPV: {}.", "Bridge with the GDPR: {}.", " (through {})")}


@dataclass(frozen=True)
class Summary:
    terms: int
    labelled: int
    described: int
    with_dpv: int
    with_gdpr: int


def curie(iri) -> str:
    for prefix, ns in (("dpv", DPV), ("gdprtext", GDPRTEXT)):
        if str(iri).startswith(ns):
            return f"{prefix}:{str(iri)[len(ns):]}"
    return str(iri)


def base_terms(g: Graph) -> list[URIRef]:
    return sorted({s for t in TERM_TYPES for s in g.subjects(RDF.type, t)
                   if isinstance(s, URIRef) and str(s).startswith(BASE_NAMESPACES)})


def dpv_links(g: Graph) -> dict[URIRef, list[tuple[str, URIRef]]]:
    out = defaultdict(list)
    for relation in MAPPINGS:
        for s, o in g.subject_objects(relation):
            if str(o).startswith(DPV):
                out[s].append((relation.fragment, o))
    order = [r.fragment for r in MAPPINGS]                # the closest correspondence first
    return {s: sorted(v, key=lambda x: (order.index(x[0]), str(x[1]))) for s, v in out.items()}


def gdpr_links(g: Graph) -> dict[URIRef, list[tuple[URIRef, URIRef]]]:
    """The pairs of the bridge: (GDPRtEXT term, DPV term that chains them), from its axioms."""
    out = defaultdict(list)
    for axiom in g.subjects(CHAINED_BY, None):
        source = g.value(axiom, OWL.annotatedSource)
        target = g.value(axiom, OWL.annotatedTarget)
        out[source].append((target, g.value(axiom, CHAINED_BY)))
    return {s: sorted(v) for s, v in out.items()}


def sentence(lang: str, description: str | None, dpv: list, gdpr: list) -> str | None:
    dpv_lead, gdpr_lead, through = LEADS[lang]
    parts = [description] if description else []
    if dpv:
        parts.append(dpv_lead.format("; ".join(f"{VERBS[lang][r]} {curie(t)}" for r, t in dpv)))
    if gdpr:
        parts.append(gdpr_lead.format("; ".join(
            f"{VERBS[lang]['closeMatch']} {curie(t)}" + through.format(curie(d))
            for t, d in gdpr)))
    return " ".join(parts) or None


def table(g: Graph, rows: list[URIRef], dpv: dict, gdpr: dict, lang: str) -> str:
    """The correspondences of every concept in Markdown, which WIDOCO shows in the description:
    also those of the terms whose entry does not show their sentence."""
    title, concept, articles = TABLE[lang]
    lines = [f"### {title}", "", f"| {concept} | DPV | GDPR | {articles} |", "|---|---|---|---|"]
    for term in sorted(rows, key=lambda t: str(g.value(t, RDFS.label)).lower()):
        name = str(g.value(term, RDFS.label)).replace("|", "/")
        laws = sorted({str(o).removeprefix("LOPDP, ") for o in g.objects(term, DCTERMS.source)},
                      key=lambda a: int("0" + "".join(c for c in a if c.isdigit())))
        lines.append("| " + " | ".join([
            f"[{name}](#{term})",                       # WIDOCO's id: the IRI
            "; ".join(f"{VERBS[lang][r]} {curie(t)}" for r, t in dpv.get(term, [])),
            "; ".join(f"{VERBS[lang]['closeMatch']} {curie(t)}" for t, _ in gdpr.get(term, [])),
            ", ".join(laws)]) + " |")
    return "\n".join(lines)


def add_header(g: Graph) -> None:
    doc = URIRef(DOC_IRI)
    affiliation, publisher, creator = BNode(), BNode(), BNode()
    for node in (affiliation, publisher):
        g.add((node, SCHEMA.name, Literal(INSTITUTION)))
        g.add((node, SCHEMA.url, URIRef(INSTITUTION_URL)))
    g.add((creator, SCHEMA.name, Literal(AUTHOR)))
    g.add((creator, SCHEMA.affiliation, affiliation))
    g.add((doc, RDF.type, OWL.Ontology))
    for lang in ("es", "en"):
        g.add((doc, DCTERMS.title, Literal(TITLE[lang], lang=lang)))
        g.add((doc, RDFS.label, Literal(TITLE[lang], lang=lang)))
        g.add((doc, DCTERMS.abstract, Literal(ABSTRACT[lang], lang=lang)))
    for p, o in ((DCTERMS.creator, creator), (DCTERMS.publisher, publisher),
                 (DCTERMS.license, URIRef(LICENSE)), (OWL.versionInfo, Literal(VERSION)),
                 (DCTERMS.source, URIRef(ONTOPRIV)), (RDFS.seeAlso, URIRef(ONTOPRIV_DOI)),
                 (DCTERMS.bibliographicCitation, Literal(CITATION)),
                 (SCHEMA.codeRepository, URIRef(REPOSITORY)),
                 (BIBO.status, Literal(f"v{VERSION}")),
                 (DCTERMS.references, URIRef(DPV.rstrip("#"))),
                 (DCTERMS.references, URIRef(GDPRTEXT.rstrip("#")))):
        g.add((doc, p, o))


def build(graphs: list[Graph], extra: dict[str, str] | None = None,
          document: bool = True) -> tuple[Graph, Summary]:
    """``extra``: more Markdown for the description, per language (the OOPS! table);
    ``document=False``: only the modules and the header, what OOPS! evaluates (S11-T06)."""
    g = merge(graphs, keep=DOC_IRI)
    for prefix, ns in (("lopdp", LOPDP), ("dpv", DPV), ("gdprtext", GDPRTEXT),
                       ("schema", str(SCHEMA)), ("bibo", str(BIBO))):
        g.bind(prefix, ns, override=True, replace=True)
    add_header(g)
    dpv, gdpr = dpv_links(g), gdpr_links(g)
    terms, labelled, described = base_terms(g), 0, 0
    for term in terms if document else ():
        if g.value(term, RDFS.label) is None:
            g.add((term, RDFS.label, Literal(humanize(str(term).rsplit("#", 1)[-1]))))
            labelled += 1
        description = g.value(term, DESCRIPTION)
        for lang in ("es", "en"):
            text = sentence(lang, description and str(description), dpv.get(term, []),
                            gdpr.get(term, []))
            if text:
                g.add((term, RDFS.comment, Literal(text, lang=lang)))
        described += bool(description or dpv.get(term) or gdpr.get(term))
    linked = [t for t in terms if t in dpv or t in gdpr]
    for lang in ("es", "en"):
        parts = [ABSTRACT[lang]]
        if document:
            parts = [DESCRIPTION_TEXT[lang], table(g, linked, dpv, gdpr, lang),
                     *([extra[lang]] if extra else [])]
        g.add((URIRef(DOC_IRI), DCTERMS.description, Literal("\n\n".join(parts), lang=lang)))
    return g, Summary(terms=len(terms), labelled=labelled, described=described,
                      with_dpv=sum(t in dpv for t in terms),
                      with_gdpr=sum(t in gdpr for t in terms))


def load_modules(folder: Path) -> list[Graph]:
    missing = [m for m in MODULES if not (folder / f"{m}.ttl").exists()]
    if missing:
        raise FileNotFoundError(f"faltan módulos de la base en {folder}: {', '.join(missing)}")
    return [Graph().parse(folder / f"{m}.ttl", format="turtle") for m in MODULES]


def write(folder: Path, out: Path) -> Summary:
    g, summary = build(load_modules(folder))
    out.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(out, format="turtle")
    return summary


def render(s: Summary, out: Path) -> str:
    return "\n".join([
        "Documento de la base para WIDOCO (S11-T05)",
        f"  Términos de la base: {s.terms}",
        f"  Con frase para la documentación: {s.described} "
        f"({s.with_dpv} con el DPV, {s.with_gdpr} con el GDPR)",
        f"  Nombres legibles para {counted(s.labelled, 'término', 'términos')} sin "
        "etiqueta",
        f"  Guardado en: {out}",
        "  Los módulos de data/output no cambian."])


def _run(argv: list[str] | None = None) -> int:
    from src.alignment.inherit import base_paths
    folder = base_paths()["candidates"].parent
    parser = argparse.ArgumentParser(
        prog="py -m src.publish.base_doc",
        description="Arma la base en un solo archivo, con la cabecera de autoría y una frase "
                    "por concepto, para que WIDOCO genere su documentación.")
    parser.add_argument("--salida", type=Path, default=folder / DOC_FILE,
                        help="archivo .ttl que se escribe")
    args = parser.parse_args(argv)
    try:
        summary = write(folder, args.salida)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    print(render(summary, args.salida))
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
