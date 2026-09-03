# Changelog

## v0.1.0 — Sprint 1 (Insumos)

Ingesta y caracterización de insumos, con back-end (rdflib) y front-end (Streamlit).

### Back-end (src/ingest)
- `ontology_loader`: carga ontologías en RDF/XML, Turtle y JSON-LD; caracteriza el
  tipo (owl-full / dl-compatible / rdfs) con evidencia; normaliza a RDF/XML.
- `text_loader`: lee el texto de ley en .txt o PDF (extracción con pypdf); cuenta artículos.
- `dpv_loader`: carga el DPV 2.3 en memoria (rdflib) como vocabulario de referencia.
- `cli`: reporte de consola que integra ontología + ley + DPV.

### Front-end (app, Streamlit)
- Vista Insumos: la carga del usuario (ontología y/o texto) es el foco; los insumos
  base del proyecto (OntoPriv, LOPDP, DPV) se muestran como estado del sistema.

### Hallazgos y decisiones
- OntoPriv es OWL Full (34 propiedades objeto+datos, 65 entidades clase+propiedad).
  El proyecto trabaja sobre OWL sin exigir el perfil OWL 2 DL.
- OWL/XML solo se acepta como entrada (se convierte a RDF/XML con Protégé); PDF solo
  como fuente de texto (se extrae, no es formato semántico).
- La validación por razonador semántico (exigida por el plan) se hará con Apache Jena
  (RDFS/OWL-RL), compatible con OWL Full, en la fase de validación.

### Insumos base
- OntoPriv: 175 clases, 122 prop. objeto, 329 prop. datos, 181 individuos, 5480 tripletas.
- LOPDP: 77 artículos, 141.871 caracteres (PDF digital).
- DPV 2.3: 14.909 tripletas, 1.123 conceptos.