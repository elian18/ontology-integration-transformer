# Changelog

Bitácora del proyecto por sprint. Cada sprint cierra con un tag de versión.

---

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

---

## v0.2.0 — Sprint 2 (Segmentación)

**Entregable:** la web muestra la ley subida partida por artículo; ChromaDB queda
poblada con los artículos y es consultable por significado.

### Qué se construyó
- `src/ingest/legal_segmenter.py` — parte el texto de ley en artículos numerados
  (número, título, texto y offsets de carácter). En la LOPDP: 77 artículos.
- `src/ai/rag/indexer.py` — indexa los artículos en ChromaDB (colección `normativa`).
  Idempotente: reindexar la misma ley no duplica.
- `src/ai/rag/retriever.py` — búsqueda semántica; devuelve número y título, con
  `top_k` desde `config.yaml`.
- `src/ai/rag/vector_store.py` — `upsert` / `count` / `reset`; nombre de colección y
  ruta desde config, con override de `CHROMA_PATH`.
- `src/config.py` — lector central de `config/config.yaml`.
- `src/ingest/cli.py` — bandera `--index [--reset]` para segmentar e indexar desde consola.
- `app/services/segmentation.py` + `app/views/articles.py` — puente y vista que
  lista los artículos.

### Decisiones (ancladas al Scrumban y al estado del arte)
- **Artículo = unidad:** un vector por artículo, sin sub-chunking. Habilita la
  trazabilidad concepto → artículo (Sprint 11).
- **Detección de encabezado** alineada a `text_loader._count_articles` (exige
  `Art. N.-`), así la segmentación real reproduce el conteo tentativo (77) y
  descarta referencias cruzadas e índice.
- **Embeddings `all-MiniLM-L6-v2`** (Sentence-BERT, documentado en el corpus
  revisado): local, sin coste y sin que el texto legal salga de la máquina.
  Modelo intercambiable vía `EMBEDDING_MODEL` en `.env`.
- **Colección `normativa`**, persistida en `chroma_db/` (ignorado en git). Los
  tests aíslan ChromaDB en una carpeta temporal vía `CHROMA_PATH`.
- **Idempotencia** por ids con prefijo `<ley>-art-N` + `upsert`.

### Cómo usar
Consola (segmentar + indexar la LOPDP):

    py -m src.ingest.cli --index --reset

Web (demo):

    py -m streamlit run app/app.py

y entrar a la página **Artículos**.

### Pruebas
`tests/test_legal_segmenter.py`, `tests/test_rag.py`, `tests/test_ui_segmentation.py`,
`tests/test_cli.py`. ChromaDB queda aislado en carpeta temporal (`tests/conftest.py`).

### Alcance y pendientes
- Solo artículos numerados; las disposiciones (reformas a otras leyes) quedan
  fuera por ahora.
- `all-MiniLM-L6-v2` es mayormente inglés; si la recuperación en español flojea,
  se puede cambiar a `paraphrase-multilingual-MiniLM-L12-v2` sin tocar código.

---

## v0.3.0 — Sprint 3 (Núcleo)

**Entregable:** a partir de OntoPriv se genera un núcleo modular reutilizable
(OntoPriv-Core) y un perfil de Ecuador (LOPDP), descargables en RDF/XML desde la web;
además, la IA extrae conceptos del texto de la LOPDP y los propone para el perfil, con
procedencia al artículo. Es la primera fase de la arquitectura (etapa "Núcleo ontológico
modular") y el aporte central de la tesis.

### Qué se construyó
- `src/core/inventory.py` — desglosa la ontología concepto por concepto (familia,
  namespace, posición en la jerarquía). Genérico para cualquier ontología cargada.
- `src/core/dpv_proximity.py` — puntúa cada concepto por su cercanía al DPV
  (embeddings `all-MiniLM-L6-v2`), como señal auxiliar para la partición.
- `src/core/split.py` — divide la ontología en núcleo (general, nivel DPV/GDPR) y
  perfil (propio de la jurisdicción); detecta referencias núcleo → perfil.
- `src/core/emit.py` — materializa el núcleo y el perfil en RDF/XML; deja el núcleo
  autónomo (mueve al perfil lo que lo referenciaba, hasta punto fijo); el perfil hace
  `owl:imports` al núcleo.
- `src/core/extract.py` — la IA lee la ley por lotes y propone conceptos para el perfil,
  con procedencia; salida JSONL, reanudable, tolerante a la cuota del LLM.
- `app/services/core_module.py` + `app/views/core_module.py` — estructura del núcleo/
  perfil y descarga de los dos archivos.
- `app/views/ai_proposals.py` — lista los conceptos propuestos por la IA, "por validar".

### Decisiones (ancladas al Scrumban, al plan y al estado del arte)
- **owlready2 descartado** del núcleo: no carga OntoPriv por ser OWL Full (conflicto de
  metaclases). Solo rdflib. (Queda como recomendación / trabajo futuro: corregir OntoPriv
  a OWL 2 DL para poder usar owlready2.)
- **Criterio de partición: rol conceptual (familias) + namespace.** Núcleo:
  Members, Principles, Rights, Processing, Terminology. Perfil: Verification,
  Personal_data_security, Transfer_or_communication, Sanctions, Violations,
  Corrective_measures, Collection; las ramas `*_in_verification` van al perfil.
- **La cercanía al DPV NO decide el corte.** Se midió (T02) pero quedó apiñada
  (0.38–0.59) y no separa núcleo de perfil; sirve como semilla de candidatos para la
  alineación (Sprint 5), no como juez de la división.
- **Núcleo autónomo.** El núcleo no puede referenciar al perfil (o deja de ser
  reutilizable para una ley sin ontología). 9 propiedades se movieron al perfil para
  lograrlo; el perfil las alcanza porque importa el núcleo.
- **La IA entra aquí (primera vez en el pipeline) y solo PROPONE.** Nada entra a la
  ontología sin validación humana. Alinear las propuestas con el núcleo/DPV es el Sprint 5.
- **Extracción por lotes.** El free tier del LLM limita las peticiones diarias, así que
  se agrupan varios artículos por llamada; formato JSONL para tolerar cortes; guardado y
  reanudación para no re-quemar cuota. Modelo: `gemini-3.1-flash-lite`.

### Resultados con OntoPriv + LOPDP
- Núcleo: 90 clases, 175 propiedades (familias generales). Perfil: 85 clases, 177
  propiedades (lo propio de Ecuador).
- Archivos emitidos: `ontopriv-core.rdf` (350 entidades, 1830 tripletas) y
  `profile-ecuador-lopdp.rdf` (354 entidades, 2106 tripletas); 9 propiedades movidas
  para dejar el núcleo autónomo.
- IA: 93 conceptos propuestos (47 clases, 46 propiedades) desde los 77 artículos.

### Cómo usar
Consola (cada etapa del núcleo):

    py -m src.core.inventory        # inventario por concepto
    py -m src.core.dpv_proximity    # cercanía al DPV
    py -m src.core.split            # división núcleo/perfil
    py -m src.core.emit             # genera los dos archivos RDF/XML
    py -m src.core.extract          # la IA propone conceptos para el perfil

Web (demo):

    py -m streamlit run app/app.py

y entrar a las páginas **Núcleo** (estructura + descarga) y **Propuestas IA**.

### Pruebas
`tests/test_core_inventory.py`, `tests/test_dpv_proximity.py`, `tests/test_split.py`,
`tests/test_emit.py`, `tests/test_extract.py`, `tests/test_ui_core_module.py`,
`tests/test_core_pipeline.py` (integración de extremo a extremo). Suite del proyecto: 83
pruebas en verde (80 sin las marcadas `slow`).

### Alcance y pendientes
- **Alineación con DPV/GDPR: Sprint 5.** Ahí se separa lo nuevo de lo que ya está en
  OntoPriv-Core y se resuelven los conceptos duplicados de las propuestas.
- **Recomendaciones / trabajo futuro:** (1) owlready2 condicionado a corregir OntoPriv a
  OWL 2 DL; (2) un modularizador genérico que derive núcleo/perfil de cualquier ontología
  subida (fuera del alcance de este TIC).
- Algún concepto propuesto sale sin artículo (la IA no lo etiquetó); se asigna en la
  validación, no se inventa.