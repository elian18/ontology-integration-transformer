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

# Sprint 2 — Segmentación

**Entregable:** la web muestra la ley subida partida por artículo; ChromaDB queda
poblada con los artículos y es consultable por significado. Tag: `v0.2.0`.

## Qué se construyó

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

## Decisiones (ancladas al Scrumban y al estado del arte)

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

## Cómo usar

Consola (segmentar + indexar la LOPDP):

    py -m src.ingest.cli --index --reset

Web (demo):

    py -m streamlit run app/app.py

y entrar a la página **Artículos**.

## Pruebas

`tests/test_legal_segmenter.py`, `tests/test_rag.py`, `tests/test_ui_segmentation.py`,
`tests/test_cli.py`. ChromaDB queda aislado en carpeta temporal (`tests/conftest.py`).

## Alcance y pendientes

- Solo artículos numerados; las disposiciones (reformas a otras leyes) quedan
  fuera por ahora.
- `all-MiniLM-L6-v2` es mayormente inglés; si la recuperación en español flojea,
  se puede cambiar a `paraphrase-multilingual-MiniLM-L12-v2` sin tocar código.