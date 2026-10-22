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

---

## v0.4.0 — Sprint 4 (Candidatos)

**Entregable:** la web muestra, para cada concepto de OntoPriv y para cada concepto que la IA
propuso desde la LOPDP, sus candidatos de alineación con el DPV (similitud léxica + embeddings)
y el tipo de correspondencia SKOS que propone la IA, con su justificación y el artículo de la ley
usado como evidencia. Todo queda "por validar": aprobar es el Sprint 5. Es la primera fase de la
etapa 2 de la arquitectura ("Alineación semántica": candidatos automáticos; la IA propone y
justifica).

### Qué se construyó
- `src/alignment/sources.py` (S4-T02) — reúne los dos flujos que se alinean: A, las 527
  entidades de OntoPriv (las 65 que son clase y propiedad a la vez se comparan con ambos tipos),
  y B, los 93 conceptos propuestos por la IA.
- `src/alignment/dpv_targets.py` (S4-T03) — los 1116 términos propios del DPV 2.3 (972
  conceptos, 144 propiedades) con definición y padres; excluye 7 términos de otros vocabularios
  que trae el archivo (`dct:`, `dcat:`, `foaf:`).
- `src/alignment/lexical.py` (S4-T04) — similitud por escritura (rapidfuzz, `token_sort_ratio`
  sobre nombres normalizados); solo compara identificadores en inglés.
- `src/alignment/candidates.py` (S4-T05) — similitud por significado (embeddings multilingües),
  puntaje combinado 0,4 × léxico + 0,6 × semántico y los 3 mejores candidatos de tipo compatible
  por concepto, sin umbral.
- `src/alignment/duplicates.py` (S4-T06) — marca los conceptos de la IA que ya existen en
  OntoPriv: puntaje combinado ≥ 0,75 o mismo nombre (léxico ≥ 0,95).
- `src/alignment/export.py` (S4-T07) — `alignment-candidates.json` y `.csv`: una fila por
  (concepto, candidato DPV), autocontenida, en estado `pending`. No se sobrescribe si ya tiene
  justificaciones de la IA (salvo `--force`).
- `src/alignment/justify.py` (S4-T08) — la IA propone el tipo SKOS y lo justifica, con un
  artículo de la LOPDP como evidencia (el de origen para los conceptos de la IA; el recuperado
  por RAG para OntoPriv). Lotes de 15 conceptos, pausa entre llamadas, reintento ante 429/503 y
  reanudación.
- `src/alignment/evaluate.py` (S4-T09) — muestra ciega etiquetada a mano, métricas Hit@1/Hit@3,
  acierto del tipo y prueba de pesos; `find` para buscar términos del DPV al etiquetar.
- `app/services/alignment.py` + `app/views/alignment.py` (S4-T10) — página **Alineación DPV**:
  tabla concepto · candidato DPV · similitud · tipo propuesto, con justificación, filtros y la
  medición.
- S4-T01 — las salidas del Sprint 3 pasan a nombres en inglés (`profile-proposed-concepts.json`,
  `core-manifest.json`), con una prueba que impide volver a los nombres en español.

### Decisiones (ancladas al Scrumban, al plan y al estado del arte)
- **Se alinean los dos flujos.** Los conceptos de la IA se cruzan primero con OntoPriv: un
  posible duplicado conserva sus candidatos pero no gasta cuota de la IA, porque hereda la
  alineación de su entidad de OntoPriv.
- **Correspondencias SKOS** (`exactMatch`, `closeMatch`, `broadMatch`, `narrowMatch`,
  `relatedMatch`) o `none`, siempre del concepto hacia el DPV. No se usan axiomas OWL
  (`owl:equivalentClass`) hacia el DPV, para no introducir consecuencias lógicas en el
  razonador de Jena (Sprint 8).
- **Sin umbral en el ranking.** El estado del arte reporta que los resultados dependen del
  umbral elegido; se conservan los 3 mejores y decide la persona.
- **La IA actúa después de los candidatos** (orden del Scrumban) y no aprueba nada.
- **Individuos fuera.** Los 181 individuos de OntoPriv son registros de ejemplo (`banking001`,
  `consentimiento001`: uno por clase, sin etiqueta); 177 no se alinean (se alinea su clase) y 4
  entran porque también son clase o propiedad.
- **LLM:** `gemini-3.1-flash-lite` (plan gratuito: 500 solicitudes al día, 15 por minuto); la
  corrida completa usa unas 45 llamadas.

### Medición (S4-T09) y correcciones con evidencia
Muestra ciega de 30 conceptos (20 de OntoPriv y 10 de la IA, semilla 42) etiquetada a mano
(`data/reference/alignment-reference-sample.csv`): 19 con término DPV correcto y 11 sin
contraparte en el DPV.

| Métrica | v1 | v2 (corregida) |
|---|---|---|
| Hit@1 (el correcto sale primero) | 47,4 % | 47,4 % |
| Hit@3 (el correcto está entre los 3) | 89,5 % | 84,2 % |
| Hit@3 OntoPriv | 11/11 | 11/11 |
| Acierto del tipo propuesto por la IA | 23,5 % (4/17) | 25,0 % (4/16) |
| Sin correspondencia respetada | 63,6 % (7/11) | 63,6 % (7/11) |

Hallazgos y correcciones:
1. **La extracción del Sprint 3 marcó 46 de los 93 conceptos como propiedad**, y los 46 tienen
   nombre de clase (`RightToErasure`, `PrincipleOfLoyalty`), así que solo podían encontrar
   propiedades del DPV. Corrección: el tipo se deduce del nombre (convención OWL: clases en
   UpperCamelCase, propiedades en lowerCamelCase). Con ello `InternationalTransfer` y
   `RightToErasure` recuperan su término correcto (`CrossBorderTransfer`, `DataErasurePolicy`).
2. **La IA abusaba de `exactMatch`** (271 "equivalente" frente a 8 "casi equivalente"; 5 de sus
   13 errores). El prompt v2 endurece el criterio (como máximo un `exactMatch` por concepto; ante
   la duda, `closeMatch`) con ejemplos que no están en la muestra: 158 `exactMatch` y 104
   `closeMatch`. En la muestra el acierto casi no cambia; los errores se movieron a la frontera
   entre "casi equivalente" y "más general/específico".
3. **Pesos 0,4 / 0,6 confirmados**: son el máximo de Hit@1 y Hit@3 en las dos corridas.

Limitaciones: un solo anotador y 30 conceptos; la corrección se evaluó con la misma muestra que
reveló los problemas. La muestra se etiquetó viendo los candidatos de v1: la baja de Hit@3 en v2
viene de 3 conceptos de la IA (`AdequateProtectionLevel`, `ProportionalityPrinciple`,
`SecurityPrinciple`) cuya etiqueta se eligió entre las propiedades que v1 les mostraba y que v2 ya
no propone. Conclusión: el ranking es confiable (OntoPriv 11/11 en Hit@3), mientras que el tipo
propuesto por la IA es el eslabón débil, lo que respalda la validación humana obligatoria del
Sprint 5.

### Resultados con OntoPriv + LOPDP + DPV 2.3
- 620 conceptos (527 de OntoPriv, 93 de la IA) y 1860 filas (3 candidatos por concepto).
- Conceptos de la IA: 34 posibles duplicados de OntoPriv (28 por puntaje, 6 por mismo nombre) y
  59 nuevos.
- La IA justificó 586 conceptos (1758 filas): 158 equivalentes, 104 casi equivalentes, 188 "el
  DPV es más general", 142 "el DPV es más específico", 525 relacionados y 641 sin
  correspondencia; ninguna respuesta fuera de la lista.

### Cómo usar
Consola (en orden):

    py -m src.alignment.export              # candidatos (JSON + CSV)
    py -m src.alignment.justify             # la IA propone el tipo y lo justifica
    py -m src.alignment.evaluate sample     # muestra ciega para etiquetar (una sola vez)
    py -m src.alignment.evaluate find <t>   # buscar un término del DPV al etiquetar
    py -m src.alignment.evaluate --sweep    # medición + prueba de pesos

Web (demo):

    py -m streamlit run app/app.py

y entrar a la página **Alineación DPV**.

### Pruebas
`tests/test_file_names.py`, `test_alignment_sources.py`, `test_dpv_targets.py`,
`test_lexical.py`, `test_candidates.py`, `test_duplicates.py`, `test_alignment_export.py`,
`test_alignment_justify.py`, `test_alignment_evaluate.py`, `test_ui_alignment.py` y
`test_alignment_pipeline.py` (flujo completo del sprint con embeddings y LLM de prueba). Suite
del proyecto: 175 pruebas en verde sin las marcadas `slow` (181 en total).

### Alcance y pendientes
- **Sprint 5 (Alineación final):** primero aprobar los conceptos propuestos por la IA, luego las
  correspondencias, y escribir el grafo alineado con propiedades de mapeo SKOS.
- **Extracción del Sprint 3:** su prompt no distingue bien clase de propiedad. La alineación ya
  no depende de ese tipo, pero para leyes nuevas conviene corregir el prompt (propiedades en
  lowerCamelCase).
- **Derechos específicos** (portabilidad, suspensión, oposición…) no están en `dpv.ttl`, sino
  en extensiones del DPV; incorporarlas queda como trabajo futuro.
- **Muestra de referencia:** un solo anotador; conviene que la tutora revise una parte.

---

## v0.5.0 — Sprint 5 (Alineación final)

**Entregable:** el grafo alineado con el DPV, construido solo con decisiones humanas
registradas. La web permite aprobar los conceptos que la IA propuso desde la LOPDP, revisar las
correspondencias con el DPV (una por una o confirmando por lotes las propuestas del asistente),
escribir el grafo y descargarlo, incluso como paquete que se abre en Protégé. Cierra la etapa 2
de la arquitectura ("Alineación semántica": la persona valida lo que la IA propuso).

### Qué se construyó
- `src/alignment/decisions.py` (S5-T01) — registro de decisiones humanas
  (`data/review/alignment-decisions.jsonl`): una línea por decisión, solo se agregan líneas,
  vale la última por clave y "deshacer" es otra línea. Cada decisión guarda quién, cuándo y una
  foto de cómo se generó el candidato (modelos, versión del prompt, puntajes, tipo de la IA).
  Va en git. Bloque `review:` en `config.yaml` (`source_id`, archivo, revisor).
- `src/alignment/ai_concepts.py` (S5-T02) + `app/services/concept_review.py` y
  `app/views/concept_review.py` (S5-T03) — página **Aprobar conceptos IA**: cada concepto de la
  IA se aprueba como nuevo (con su tipo: clase, propiedad de objeto o de datos), se marca como
  duplicado eligiendo la entidad de OntoPriv a la que equivale, o se descarta.
- `src/alignment/mapping_review.py` (S5-T04) + `app/services/mapping_review.py` y la pestaña
  **Revisar correspondencias** de **Alineación DPV** (S5-T05) — tarjeta por concepto con sus 3
  candidatos: la persona elige el tipo SKOS, aprueba o descarta, busca otro término del DPV
  (solo del tipo compatible) o marca "sin correspondencia".
- `src/alignment/assistant_review.py` (S5-T08) + pestaña **Confirmar propuestas del asistente**
  — propuestas para los conceptos pendientes (`data/review/assistant-proposals.json`, con una
  razón por concepto) que la persona revisa y confirma por familia en una tabla editable.
- `src/alignment/write.py` (S5-T06) — `py -m src.alignment.write` escribe
  `ontopriv-dpv-alignment.rdf`, `alignment-manifest.json` y `alignment-approved.csv`.
- `src/alignment/bundle.py` + `app/services/alignment_graph.py` (S5-T07) — pestaña **Grafo
  alineado**: escribe el grafo y ofrece el RDF/XML, el manifiesto, el CSV y un .zip para
  Protégé con su propio `catalog-v001.xml`.

### Decisiones (ancladas al Scrumban, al plan y al estado del arte)
- **La revisión humana es parte de la construcción de la base, no del uso.** OntoPriv y la LOPDP
  son entradas fijas: sus 552 conceptos se alinean una sola vez y la alineación validada se
  reutiliza. El usuario final no revisa estos conceptos (sus modos de entrada: Sprints 6 y 7).
- **Las decisiones viven fuera de la ontología**, en un registro auditable separado por entrada
  (`source_id = ontopriv+lopdp`), para que otra ley u ontología tenga sus propias decisiones.
- **El tipo SKOS no viene preseleccionado**: la IA acertó el tipo en el 25 % de la muestra del
  Sprint 4 y preseleccionarlo empujaría a aceptarlo.
- **Duplicados elegidos por la persona.** El detector del Sprint 4 no ve equivalencias entre
  formulaciones distintas (`BusinessVolume` ↔ `Turnover`, `DataProtectionOfficer` ↔
  `Delegate`); por eso cualquier concepto puede marcarse como duplicado, y la persona elige la
  entidad (el selector muestra familia, tipo y módulo, porque OntoPriv repite nombres en sus dos
  espacios de nombres). Un duplicado no se alinea aparte: hereda la alineación de OntoPriv y le
  aporta su artículo de la ley.
- **Grafo como módulo aparte**: `owl:imports` del perfil (que importa el núcleo) y sin importar
  el DPV, que se referencia por IRI. El núcleo y el perfil del Sprint 3 no cambian; los 25
  conceptos nuevos se declaran en este módulo, sin ubicarlos en la jerarquía de OntoPriv.
- **Solo propiedades de mapeo SKOS** hacia el DPV; "sin correspondencia" no escribe nada. El
  artículo de la ley se anota con `dct:source` ("LOPDP, art. N"); PROV-O llega en el Sprint 10.
- **El grafo solo se escribe sin nada pendiente**, y su manifiesto guarda la huella SHA-256 del
  registro de decisiones del que salió.
- **Revisión asistida por lotes (S5-T08).** Tras revisar 20 conceptos uno por uno, el revisor
  notó que estaba siguiendo la sugerencia de la IA (coincidió con su tipo en 25 de 28
  correspondencias, frente a un 25 % de acierto medido) y que revisar 552 así no era viable. Se
  cambió el método: el asistente de programación propuso una decisión con su razón para cada
  concepto pendiente y la persona las confirmó por familia. Las propuestas no son decisiones:
  entran al registro solo al confirmarse, a nombre de quien confirma, con la nota "propuesta del
  asistente confirmada" y la propuesta original en la foto de procedencia.

### Resultados con OntoPriv + LOPDP + DPV 2.3
- **Conceptos de la IA (93):** 25 aprobados como nuevos (todos clases), 66 duplicados de
  OntoPriv y 2 descartados. De los 34 posibles duplicados del detector, 33 se confirmaron (29
  con la misma entidad y 4 con otra) y 1 resultó nuevo; los otros 33 duplicados los encontró la
  persona entre los que el detector marcó como nuevos.
- **Correspondencias (552 conceptos = 527 de OntoPriv + 25 nuevos):** 552 revisados, 173
  alineados y 379 sin correspondencia en el DPV.

| Tipo SKOS | Correspondencias |
|---|---|
| `exactMatch` (equivalente) | 25 |
| `closeMatch` (casi equivalente) | 36 |
| `broadMatch` (el DPV es más general) | 57 |
| `narrowMatch` (el DPV es más específico) | 4 |
| `relatedMatch` (relacionado) | 68 |
| **Total** | **190** |

- 156 correspondencias de entidades de OntoPriv y 34 de conceptos nuevos; 41 con términos
  encontrados fuera de los 3 candidatos (búsqueda).
- **Quién decidió:** 20 conceptos (31 correspondencias) la persona sola; 532 conceptos (159
  correspondencias) confirmados desde propuestas del asistente, sin cambios.
- **Tipo igual al de la IA (Gemini):** 64 de 144 correspondencias con tipo de la IA (44,4 %);
  25 de 28 en la revisión manual y 39 de 116 en las propuestas del asistente.
- **Grafo:** 385 tripletas; 25 conceptos nuevos declarados con etiqueta, definición y artículo;
  58 entidades de OntoPriv reciben artículos de la ley. Verificado en Protégé: el .zip resuelve
  alineación → perfil → núcleo y todas las entidades con correspondencia están declaradas.

### Cómo usar
Consola:

    py -m src.alignment.decisions           # resumen del registro de decisiones
    py -m src.alignment.ai_concepts         # estado de los conceptos de la IA
    py -m src.alignment.mapping_review      # estado de las correspondencias
    py -m src.alignment.assistant_review    # propuestas del asistente abiertas, por familia
    py -m src.alignment.write               # escribe el grafo, el manifiesto y el CSV

Web (demo):

    py -m streamlit run app/app.py

y entrar a **Aprobar conceptos IA** y a **Alineación DPV** (pestañas Revisar
correspondencias, Confirmar propuestas del asistente y Grafo alineado).

### Pruebas
`tests/test_alignment_decisions.py`, `test_alignment_ai_concepts.py`,
`test_ui_concept_review.py`, `test_alignment_mapping_review.py`, `test_ui_mapping_review.py`,
`test_alignment_assistant_review.py`, `test_alignment_write.py`, `test_alignment_bundle.py`,
`test_ui_alignment_graph.py` y `test_review_pipeline.py` (flujo completo del sprint a través de
los mismos servicios de la web, de los conceptos de la IA al .zip para Protégé). Suite del
proyecto: 289 pruebas en verde sin las marcadas `slow` (295 en total).

### Alcance y pendientes
- **Sprints 6 y 7 (modos del usuario):** conectar solo ontología, solo ley y ley + ontología.
  La idea es que el usuario solo suba su material: los conceptos que coincidan con OntoPriv
  heredan la alineación validada aquí, y el resto recibe propuestas automáticas marcadas como
  "no validadas". El alcance de esa automatización lo decide la PO frente al "garantizando la
  validación humana" del plan.
- **Sprint 8 (validación con Jena):** con razonamiento RDFS/OWL-RL las propiedades de mapeo SKOS
  tipan sujeto y objeto como `skos:Concept` (punning, aceptable porque OntoPriv ya es OWL Full).
- **Sprint 10 (trazabilidad):** convertir las fotos de procedencia del registro a PROV-O.
- **Limitaciones para la tesis:** un solo revisor; 532 de 552 conceptos se decidieron
  confirmando propuestas del asistente sin cambios, en una sesión; conviene una segunda revisión
  independiente de una muestra (por ejemplo, la tutora). Los conceptos nuevos no tienen lugar
  en la jerarquía de OntoPriv.
- `data/output/catalog-v001.xml` (lo dejó Protégé) apunta el IRI del núcleo al archivo del
  perfil; el .zip trae un catálogo correcto y ese archivo no se usa.
- Sigue pendiente incorporar las extensiones del DPV (derechos específicos).

---

## v0.6.0 — Sprint 6 (Transformación 1 + modo "solo ontología")

**Entregable:** el usuario sube su ontología de protección de datos en la página **Alinear mi
ontología** y, sin revisar nada, la recibe alineada con el DPV y la descarga en el formato que
elija (RDF/XML, OWL, Turtle o JSON-LD), como un .zip que se abre en Protégé. Lo que ya fue
validado en la base se hereda; lo demás va como propuesta automática, en un módulo aparte y
marcado como no validado. Es la primera parte de la etapa 3 de la arquitectura ("Transformación
de formatos"); la segunda (etiquetas es/en) es del Sprint 7.

### Qué se construyó
- `src/transform/convert.py` (S6-T01) — convierte entre RDF/XML, OWL (RDF/XML con extensión
  .owl), Turtle y JSON-LD sin perder tripletas. Corrige dos pérdidas del serializador de rdflib:
  en Turtle reescribía decimales ("15" pasaba a "15.0") y en JSON-LD perdía las tripletas
  `rdf:type rdf:List` de las listas (OntoPriv bajaba de 5480 a 5468 tripletas). OWL/XML (.owx)
  sigue solo de entrada, con un mensaje que pide convertirlo en Protégé.
- `src/transform/roundtrip.py` (S6-T02) — prueba de ida y vuelta entre formatos con isomorfismo
  de grafos (`rdflib.compare`: incluye nodos en blanco, idioma y tipo de dato) sobre los archivos
  reales; guarda `data/output/roundtrip-report.json`.
- `src/transform/package.py` + `src/alignment/bundle.py` (S6-T03) — un .zip por formato con su
  propio `catalog-v001.xml`, para que Protégé resuelva los `owl:imports` (alineación → perfil →
  núcleo) sin internet. La pestaña **Grafo alineado** tiene selector de formato.
- `src/workspace/store.py` (S6-T04) — un espacio de trabajo por cada subida
  (`data/workspaces/<nombre>-<hash8>`, ignorado en git) con su manifiesto y el estado de cada
  paso. Una subida nunca escribe en la base (`data/output`, `data/review`); subir el mismo
  archivo reutiliza lo ya calculado. Límite de 25 MB (bloque `workspaces:` de `config.yaml` y
  `.streamlit/config.toml`).
- `src/alignment/inherit.py` (S6-T05) — la ontología subida hereda las decisiones validadas de
  la base con tres reglas: mismo IRI; mismo nombre o etiqueta (incluye como alias los nombres de
  los 66 conceptos de la IA confirmados como duplicados en el Sprint 5); o muy parecido, con los
  umbrales del detector de duplicados del Sprint 4 (combinado ≥ 0.75 o léxico ≥ 0.95).
- `src/alignment/calibrate.py` + `src/alignment/propose.py` (S6-T06) — propuestas automáticas
  para lo que no se heredó: el mejor candidato del ranking del Sprint 4, solo si su puntaje
  llega al umbral (bloque `auto:` de `config.yaml`, 0.75); la IA pone el tipo SKOS o la descarta
  ("none"). Sin IA (sin clave o sin cuota), la propuesta queda sin tipo y se escribe como
  `rdfs:seeAlso`.
- `src/alignment/user_alignment.py` (S6-T07) — escribe dos módulos que importan la ontología
  del usuario sin editarla: el **validado** (solo correspondencias heredadas, con `owl:Axiom`,
  `dct:source` al concepto de la base y quién lo decidió) y el de **propuestas** ("Propuesta
  automática, no validada", con puntaje y justificación de la IA); además
  `alignment-manifest.json`, `alignment.csv` y el paquete `<archivo>-alineada-<formato>.zip`.
- `app/services/align_ontology.py` + `app/views/align_ontology.py` (S6-T08) — página **Alinear
  mi ontología**, primera del menú (sección "Tu ontología"): la alineación arranca sola al subir
  el archivo, con el progreso por pasos; muestra cuatro cifras explicadas, el formato de
  descarga y, si se quiere, una tabla por concepto. Las páginas anteriores pasan a la sección
  "Construcción de la base".
- `src/alignment/external_check.py` (S6-T09) — comprobación automática con una ontología externa,
  sin etiquetar nada a mano (ver Resultados).
- `.gitattributes` (`* text=auto eol=lf`): los merges dejaban archivos "modificados" solo por
  los fines de línea de Windows.

### Decisiones (ancladas al Scrumban, al plan y al estado del arte)
- **Opción híbrida, con el usuario haciendo lo mínimo** (subir, elegir formato y descargar). El
  Scrumban decía "el usuario aprueba su alineación": en este sprint no hay aprobar o descartar
  en la web. La validación humana del plan está en la base (Sprint 5); lo que el usuario recibe
  sin validar va en otro módulo y marcado, para que pueda importar solo lo validado.
- **Umbral 0.75**: el menor con precisión ≥ 75 % y al menos 10 propuestas, medido contra las
  552 decisiones de la base con el filtro de la IA (tabla en Resultados). Se prefirió precisión
  a cobertura porque nadie revisa estas propuestas.
- **OWL como salida** = RDF/XML con extensión .owl (mismos bytes que el .rdf). OWL/XML sigue
  solo de entrada.
- **JSON-LD y Protégé**: Protégé 5.6.9 (OWL API 4.5.29) no abre JSON-LD (lo lee como RDF/JSON).
  El archivo es válido (verificado con pyld) y sirve para otras herramientas; el paquete y la web
  lo advierten, y su readme dice "Archivo principal" en lugar de "Abre … en Protégé".
- **Comprobación externa sin hoja manual (S6-T09).** La referencia ya existe: los conceptos de
  la ontología externa que heredaron una decisión de la base tienen la respuesta que dio una
  persona en el Sprint 5. Se resuelven otra vez como si fueran nuevos y se comparan. No hay un
  mapeo oficial GDPRtEXT ↔ DPV publicado que sirviera de referencia externa.
- **GDPRtEXT como ontología externa**: Pandit, H. J., Fatema, K., O'Sullivan, D. y Lewis, D.
  (2018). *GDPRtEXT - GDPR as a Linked Data Resource*. ESWC 2018, LNCS 10843, pp. 481-496. ADAPT
  Centre, Trinity College Dublin; licencia CC-BY-4.0. Es el id12 de la revisión de alcance y se
  incluye en `data/input/gdprtext-ontology.ttl`.

### Resultados
**Conversión (S6-T02).** 6 archivos (OntoPriv, `sample.ttl`, núcleo, perfil, grafo alineado y
DPV 2.3), 24 idas y vueltas, las 24 con el mismo grafo; 30.190 tripletas comprobadas. Con el
serializador de rdflib sin las correcciones fallan 4 de 12: Turtle cambia 2 decimales de
OntoPriv y del núcleo, y JSON-LD pierde 12 tripletas de OntoPriv y de `sample.ttl`.

Hallazgo: OntoPriv trae 1 literal tipado `rdf:langString` sin etiqueta de idioma
(`responsible_for_treatment`, propiedad `activity_within_right_of_access`), algo que RDF 1.1 no
admite. La conversión lo conserva tal cual (no corrige datos); se revisa en la validación.

**Calibración del umbral (S6-T06)**, contra las decisiones de la base (552 conceptos, 173 con
término del DPV; en 92 el mejor candidato es uno que la persona aprobó). Con el filtro de la IA:

| Umbral | Propuestas | Correctas | Precisión | Cobertura (de 92) |
|---|---|---|---|---|
| 0.65 | 93 | 50 | 53,8 % | 54,3 % |
| 0.70 | 69 | 44 | 63,8 % | 47,8 % |
| **0.75** | **34** | **28** | **82,4 %** | **30,4 %** |
| 0.80 | 23 | 21 | 91,3 % | 22,8 % |
| 0.85 | 10 | 9 | 90,0 % | 9,8 % |

- Sin el filtro de la IA, 0.75 da 36 propuestas y 28 correctas (77,8 %).
- Muestra ciega del Sprint 4 (30 conceptos etiquetados a mano): con 0.75 sale 1 sola propuesta,
  y es incorrecta.
- **Ranking:** de los 173 conceptos alineados de la base, el término correcto quedó en el
  puesto 1 en 92 (53 %), en el puesto 2 o 3 en 46 (27 %) y fuera de los 3 candidatos en 35
  (20 %). Un diseño que solo mira el puesto 1 no puede recuperar esos 81.

**GDPRtEXT en la web (S6-T08)**, subido como lo haría un usuario: 1632 tripletas, 223 clases y
13 propiedades de objeto, sin señales de OWL Full; 235 conceptos alineables.
- Heredados: 22 (18 por mismo nombre y 4 muy parecidos); 21 con término (23 correspondencias
  validadas) y 1 sin correspondencia (`Accountability`).
- Propuestas automáticas: 38, todas con tipo de la IA (24 `exactMatch`, 3 `closeMatch`, 8
  `narrowMatch`, 3 `broadMatch`), con 3 llamadas a Gemini; 175 conceptos sin propuesta (mejor
  candidato bajo 0.75).
- Módulo validado: 166 tripletas; módulo de propuestas: 274. El .zip resuelve la importación de
  GDPRtEXT en los cuatro formatos.

**Comprobación automática con GDPRtEXT (S6-T09).** Los 22 conceptos heredados se resolvieron
como si fueran nuevos, con el diseño actual (puesto 1 + umbral 0.75 + IA; 1 llamada a
`gemini-3.1-flash-lite`, embeddings `paraphrase-multilingual-MiniLM-L12-v2`), y se compararon
con la decisión de la persona:

| Medida | Resultado |
|---|---|
| Propuestas correctas | 10 de 10 (precisión 100 %) |
| Cobertura | 10 de 21 (48 %) |
| Tipo SKOS igual al de la persona | 2 de 10 |
| "Sin correspondencia" detectada | 1 de 1 |
| Término correcto en el puesto 1 / top 3 / top 10 | 14 / 19 / 21 de 21 |
| No propuestos con el correcto en el puesto 1 (bajo 0.75) | 4 |
| No propuestos con el correcto fuera del puesto 1 | 7 |

- Las 10 propuestas son de conceptos con el mismo nombre en la base; de los 4 "muy parecidos"
  no se propuso ninguno. Es una muestra chica y de casos fáciles.
- **Tipo SKOS:** en 7 de los 10 aciertos la persona había elegido `exactMatch`; la IA eligió
  `closeMatch` en 6 de ellos, lo que sigue la regla del prompt v2 ("ante la duda, closeMatch").
- **Lo que alcanzaría un paso que elija entre candidatos:** 19 de 21 con los 3 primeros y 21 de
  21 con los 10 primeros, frente a 10 de 21 hoy. Es la línea base del rediseño del Sprint 7.
- Resultado completo: `data/output/gdprtext-ontology-external-check.json`.

### Cómo usar
Consola:

    py -m src.transform.convert ARCHIVO --to turtle     # rdfxml, owl, turtle, jsonld o all
    py -m src.transform.roundtrip                        # ida y vuelta de la base
    py -m src.alignment.bundle --formato turtle          # paquete de la base para Protégé
    py -m src.alignment.calibrate                        # tabla del umbral
    py -m src.alignment.user_alignment ARCHIVO --formato turtle   # alinea una ontología
    py -m src.alignment.external_check data/input/gdprtext-ontology.ttl

Web (demo):

    py -m streamlit run app/app.py

y entrar a **Alinear mi ontología** (sección "Tu ontología").

### Pruebas
`tests/test_transform_convert.py`, `test_transform_roundtrip.py`, `test_transform_package.py`,
`test_alignment_bundle.py`, `test_ui_alignment_graph.py`, `test_workspace.py`,
`test_alignment_inherit.py`, `test_alignment_calibrate.py`, `test_alignment_propose.py`,
`test_user_alignment.py`, `test_ui_align_ontology.py`, `test_alignment_external_check.py` y
`test_sprint6_pipeline.py` (flujo completo con el GDPRtEXT real, embeddings y LLM de prueba:
subida, herencia, propuestas, módulos, .zip en los cuatro formatos, comprobación externa y la
base sin tocar). Suite del proyecto: 432 pruebas en verde sin las marcadas `slow` (439 en total).

### Alcance y pendientes
- **Sprint 7, al inicio: rediseño de la IA de alineación.** Hoy el RAG no se usa en el modo de
  usuario (ChromaDB solo tiene la colección `normativa` de la LOPDP), el LLM solo ve el
  candidato 1 y los prompts viven dentro de `justify.py` y `extract.py` (`src/ai/prompts/` está
  vacía). Plan: colecciones de ChromaDB para el DPV (1116 términos) y para las 552 decisiones
  de la base; el LLM elige entre los 10 primeros candidatos (o ninguno) con precedentes
  recuperados; prompts versionados en `src/ai/prompts/`; evaluación automática antes y después
  (dejando fuera cada decisión de la base, y con GDPRtEXT). Después, los modos con ley y las
  etiquetas es/en del Scrumban.
- **Decisión pendiente para el Sprint 7:** el Scrumban dice "IA extrae conceptos → aprobación"
  en el modo solo ley; con la regla de este sprint, el usuario no aprobaría nada.
- **Limitaciones para la tesis:** la referencia de la calibración no es independiente (532 de
  552 decisiones confirman propuestas del asistente); la comprobación externa usa 22 conceptos,
  casi todos con el mismo nombre que en la base; el umbral es sensible (de 0.70 a 0.80 la
  precisión pasa de 64 % a 91 % y la cobertura de 48 % a 23 %).
- Aprobar o descartar propuestas en la web queda fuera de este sprint.
- **Sprint 8 (validación):** revisar el literal `rdf:langString` sin idioma de OntoPriv.

---

## v0.7.0 — Sprint 7 (Transformación 2 + modos con ley)

**Entregable:** en la página **Alinear con el DPV** el usuario elige qué sube (su ontología, una
ley o las dos) y, sin revisar nada, descarga una ontología alineada con el DPV y con etiquetas en
español e inglés, en RDF/XML, OWL, Turtle o JSON-LD. Con una ley distinta de la LOPDP (probado
con el GDPR en inglés y la Ley 1581 de 2012 de Colombia), la IA extrae sus conceptos, una regla
los acepta y el sistema arma con ellos un perfil que importa OntoPriv-Core. La alineación la
decide ahora un decisor con RAG (candidatos del DPV y precedentes de la base) y una regla
calibrada; ya no hay un módulo de propuestas "no validadas".

### Qué se construyó
- `src/ai/rag/vector_store.py` + `src/alignment/knowledge.py` (S7-T01) — colecciones de ChromaDB
  por nombre, construidas una vez desde la base y luego solo leídas: `dpv` (1116 términos
  alineables: 972 clases y 144 propiedades) y `decisions` (los 552 conceptos decididos en el
  Sprint 5, con su término y relación SKOS o "sin correspondencia"). Cada colección guarda la
  huella de lo que la construyó (modelo, archivos, versión del texto) y se reconstruye sola si
  algo cambió.
- `src/ai/prompts/` + `src/ai/llm_client.py` (S7-T02) — los prompts viven en archivos
  versionados (`decide.v1`, `justify.v2`, `extract.v2`, `labels.v1`) y cada manifiesto guarda su
  versión y su sha256. `LLMClient` usa temperatura 0 y pide JSON cuando el modelo lo admite.
- `src/alignment/reference.py` + `src/alignment/benchmark.py` (S7-T03) — referencia automática
  (tus decisiones, sin etiquetar a mano) con control de fugas: al medir un concepto no se le
  muestra su propia decisión ni la de los conceptos de la base de los que heredaría (mismo
  nombre o muy parecido). Mide la línea base del Sprint 6 y elige,
  con un criterio fijado antes, de dónde salen los candidatos del decisor.
- `src/alignment/decider.py` (S7-T04) — por concepto, el LLM ve los 10 mejores candidatos del
  DPV (ranking híbrido) y los 4 precedentes más parecidos de la base, y elige uno con su
  relación SKOS y su confianza, o "ninguno". Lotes de 10 conceptos, caché por espacio de trabajo
  y un límite de llamadas por corrida (60), con pausa de 5 s por la cuota gratuita.
- `src/alignment/decider_eval.py` (S7-T05) — evaluación antes/después sobre las 552 decisiones
  dejando fuera cada una, con dos grillas de reglas y validación cruzada (ver Resultados).
- `src/alignment/propose.py` + `src/alignment/user_alignment.py` (S7-T06) — el decisor entra al
  flujo de la subida. Un solo módulo `<nombre>-dpv-alignment` con lo heredado de la base y lo
  que la regla aceptó; cada tripleta lleva un `owl:Axiom` con quién decidió ("base", "regla" o
  "regla sin IA"), su versión y su justificación. Las **sugeridas** (confianza alta que la regla
  no acepta) no entran a la ontología: van a `alignment.csv`. Sin IA, el mejor candidato con
  puntaje ≥ 0.75 entra sin tipo, como `rdfs:seeAlso`.
- `src/ingest/law_text.py` + `src/ingest/legal_segmenter.py` (v2) + `src/ingest/law_upload.py`
  (S7-T07) — limpieza del PDF de una ley subida (encabezados y pies de página repetidos,
  marcadores de EUR-Lex, ligaduras, títulos con letras espaciadas), segmentación de leyes en
  español ("Art. N.-", "Artículo N°.") y en inglés ("Article N") con detección del idioma, e
  índice de la ley en una ChromaDB propia del espacio de trabajo. La LOPDP de la base no cambia
  (sigue con 77 artículos y su índice del Sprint 2).
- `src/transform/labels.py` (S7-T08) — completa las etiquetas en español e inglés que le faltan
  a cada clase y propiedad, en lotes de 40, con un glosario de 87 términos de la LOPDP tomados
  de la base del Sprint 5. Van al módulo `<nombre>-labels`; las etiquetas del usuario no
  cambian. Además, las lecturas de ChromaDB reintentan en Windows ("Nothing found on disk").
- `src/core/law_concepts.py` + `src/core/acceptance.py` + `src/core/extract.py` (S7-T09) — la IA
  extrae los conceptos de cualquier ley (prompt `extract.v2`, que pide etiqueta y definición en
  el idioma de la ley) y una regla fija los acepta: cita un artículo de la ley, nombre técnico
  bien formado, etiqueta y definición dentro de límites, y no se repite (clase y propiedad con la
  misma etiqueta no son repetidas). La extracción se retoma donde quedó si se agota la cuota.
- `src/core/law_profile.py` (S7-T10) — modo **solo ley**: los conceptos aceptados forman un
  perfil nuevo que importa OntoPriv-Core. Si el concepto ya está en el núcleo, reutiliza su IRI;
  si solo es muy parecido, se crea y se enlaza con `skos:exactMatch`; si no, es nuevo. Cada
  concepto lleva `dct:source` con su artículo. El perfil pasa por la misma alineación y el .zip
  incluye una copia del núcleo para que Protégé resuelva la importación.
- `src/alignment/law_evidence.py` (S7-T11) — modo **ley + ontología**: la ontología se alinea
  igual que en "solo ontología" y cada concepto recibe los artículos de la ley más cercanos, en la
  columna `law_evidence` de `alignment.csv`, para revisarlos. No entran a la ontología (ver
  Decisiones).
- `app/services/align_ontology.py` + `app/views/align_ontology.py` + `app/views/home.py`
  (S7-T12) — la página pasa a llamarse **Alinear con el DPV** y pregunta "¿Qué vas a subir?" (Mi
  ontología · Una ley · Mi ontología y una ley). Cada modo pide solo sus archivos y arranca solo
  cuando están todos. Muestra lo que se armó con la ley o los artículos para revisar, y avisa
  cuando la IA no terminó de leer la ley.
- `tests/test_sprint7_pipeline.py` + consola de `user_alignment` (S7-T13) — prueba de punta a
  punta de los modos con ley (ver Pruebas). La consola dice cuándo la IA no se volvió a llamar y
  las cifras son de la corrida anterior, y la línea del decisor muestra la cuota agotada, el
  límite de llamadas y el primer error.

### Decisiones (ancladas al Scrumban, al plan y al estado del arte)
- **Alineación automática con supervisión humana después.** Al arrancar el sprint se cambió la
  arquitectura: el usuario sube, alinea y descarga sin aprobar nada; la validación humana pasa
  al ciclo de comentarios del Sprint 10. La arquitectura, el stack y el Scrumban del Proyecto se
  actualizaron el 19 de octubre.
- **Origen de los candidatos medido, no supuesto:** gana el ranking híbrido del Sprint 4 (más
  aciertos en los 10 primeros de la base). La búsqueda solo por significado en `dpv` acierta
  menos (138 frente a 154 de 173).
- **La regla del decisor mantiene el piso de 75 %.** Ninguna regla de la primera grilla (fijada
  antes de correr) llegó; la segunda grilla se declaró después del análisis de errores
  (relación SKOS y clase/propiedad), sin bajar el piso. Regla elegida: confianza "alta", puntaje
  ≥ 0.5, clases con cualquier relación menos `relatedMatch` y propiedades nunca automáticas.
- **La ley no entra al decisor.** El Scrumban decía que la ley da los artículos de evidencia al
  decisor; medido en S7-T05, dárselos baja la precisión de 75,0 % a 64,1 %.
- **Solo ley con "aprobación automática por regla".** La regla de aceptación es estructural:
  revisa la forma del concepto, no su significado.
- **Conceptos de la ley que ya están en OntoPriv-Core reutilizan su IRI** (si solo son muy
  parecidos, se enlazan), con las mismas reglas de herencia del Sprint 6 (mismo nombre o
  etiqueta, o combinado ≥ 0.75 o léxico ≥ 0.95).
- **Etiquetas es/en:** el LLM escribe solo la que falta, en un módulo aparte, guiado por el
  glosario de la LOPDP; ninguna etiqueta original cambia.
- **Artículos de la ley solo para revisar.** Con la grilla y el criterio fijados antes (precisión
  ≥ 75 %, al menos 10 enlaces), ninguna regla llegó (la mejor, 73,9 % con 23 enlaces). Los
  artículos van a `alignment.csv` con la regla de mayor cobertura (3 por concepto, distancia
  ≤ 0,55) y la persona los confirma en el Sprint 10, igual que las sugeridas.
- **LLM:** `gemini-3.1-flash-lite` (cuota gratuita: 500 llamadas por día); embeddings
  `paraphrase-multilingual-MiniLM-L12-v2`.

### Resultados
**Origen de los candidatos (S7-T03).** Término correcto en el puesto 1 / top 3 / top 10, con 10
candidatos:

| Origen | Base (173) | GDPRtEXT (21) | Muestra ciega (19) |
|---|---|---|---|
| **híbrido (Sprint 4)** | **92 / 138 / 154** | **14 / 19 / 21** | **9 / 16 / 16** |
| solo `dpv` (significado) | 70 / 109 / 138 | 12 / 18 / 21 | 5 / 9 / 15 |
| `dpv` + reordenado híbrido | 93 / 138 / 153 | 14 / 19 / 21 | 9 / 16 / 16 |

- Línea base del Sprint 6 con la misma medida: 34 propuestas, 28 correctas (82,4 %), cobertura
  16,2 % de los 173 alineados; en la muestra ciega, 1 propuesta y es incorrecta.
- La referencia es ruidosa: la muestra ciega del Sprint 4 y las decisiones del Sprint 5
  coinciden en 10 de los 23 conceptos que comparten (43,5 %).

**Evaluación del decisor (S7-T05)**, sobre las 552 decisiones dejando fuera cada una, con la regla
elegida:

| Diseño | Propuestas | Correctas | Precisión | Cobertura | F1 |
|---|---|---|---|---|---|
| Línea base (Sprint 6) | 34 | 28 | 82,4 % | 16,2 % | 27,1 % |
| Decisor sin precedentes | 69 | 47 | 68,1 % | 27,2 % | 38,9 % |
| **Decisor con precedentes** | **68** | **51** | **75,0 %** | **29,5 %** | **42,3 %** |
| Decisor con precedentes + ley | 78 | 50 | 64,1 % | 28,9 % | 39,8 % |

- Validación cruzada de la segunda grilla (elegir en una mitad y medir en la otra, 100 veces):
  precisión 72,0 % (p10 62,5 %, p90 86,4 %), cobertura 30,3 %; sin regla en 9 de 200 mitades.
- Confirmación sin reajustar: en GDPRtEXT, 18 propuestas y 17 correctas (94,4 %), cobertura 81,0
  % de 21 (con el diseño del Sprint 6 eran 10 de 21); en la muestra ciega, 6 propuestas y 3
  correctas (50,0 %).
- 119 sugeridas (confianza alta que la regla no acepta): 52 coinciden con tu decisión y 59 caen
  sobre conceptos que dejaste sin correspondencia.

**GDPRtEXT subido en "Mi ontología" (S7-T06, S7-T08)**, 235 conceptos:
- Heredados de la base: 22 (18 por mismo nombre y 4 muy parecidos), con 23 correspondencias
  validadas; `Accountability` sin correspondencia.
- Aceptados por la regla: 80 (54 `exactMatch`, 14 `broadMatch`, 7 `closeMatch` y 5
  `narrowMatch`); sugeridos 21; sin correspondencia 78; sin propuesta 35. En el Sprint 6 eran 38
  propuestas sin validar.
- 22 llamadas al decisor. Módulo de alineación: 940 tripletas.
- Etiquetas: la IA agregó 235 en español y 234 en inglés (una ya tenía su etiqueta en inglés) en
  6 llamadas; módulo de etiquetas: 475 tripletas.

**Leyes subidas (S7-T07, S7-T09, S7-T10):**

| | GDPR (inglés) | Ley 1581 de 2012, Colombia (español) |
|---|---|---|
| PDF | 78 páginas; 151 líneas de encabezado o pie quitadas, 30 marcadores de EUR-Lex, 32 títulos con letras espaciadas | 9 páginas; 18 líneas quitadas, 46 ligaduras |
| Artículos | 99 | 30 |
| Conceptos extraídos / aceptados | 122 / 122 (74 clases, 48 propiedades), 13 llamadas | 28 / 27 (1 repetido), 4 llamadas |
| Perfil sobre OntoPriv-Core | 13 reutilizan su IRI, 1 se enlaza (`Pseudonymisation` → `Pseudonymization`, 0,79), 108 nuevos | 9 reutilizan su IRI, 18 nuevos |
| Heredados de la base | 22 conceptos, 24 correspondencias | 11 conceptos, 12 correspondencias |
| Aceptados por la regla / sugeridos | 24 / 14 (10 llamadas) | 3 / 3 (2 llamadas) |
| Etiquetas agregadas | 122 en español (4 llamadas) | 27 en inglés (1 llamada) |

- Calibración de la regla de aceptación con los 93 conceptos que la IA propuso desde la LOPDP en
  el Sprint 3: coincide con tus decisiones del Sprint 5 en 91 de 93 (97,8 %). Los dos
  desacuerdos: `ExplicitConsent` (la regla lo acepta y tú lo descartaste) y `UnfulfilledRegistry`
  (la regla lo rechaza por no citar artículo y tú lo conservaste).
- En las dos leyes reales la regla no rechazó ningún concepto.

**Artículos de la ley para revisar (S7-T11)**, calibración con GDPRtEXT + GDPR: GDPRtEXT dice en qué
artículo se define cada uno de 80 conceptos (113 pares concepto-artículo en 38 artículos). El
artículo correcto queda primero en 44 de 80 y entre los 3 primeros en 53 de 80.

| Regla | Enlaces | Precisión | Cobertura |
|---|---|---|---|
| 1 artículo, distancia ≤ 0,30 (la de mayor precisión) | 23 | 73,9 % | 21,2 % |
| 1 artículo, sin distancia máxima | 80 | 55,0 % | 55,0 % |
| **3 artículos, distancia ≤ 0,55 (la elegida para revisar)** | **222** | **26,6 %** | **66,2 %** |
| 3 artículos, sin distancia máxima | 240 | 24,6 % | 66,2 % |

- Validación cruzada: precisión media 64,5 % (p10 50,0 %, p90 72,7 %); sin regla en 98 de 200
  mitades.
- La precisión es una cota inferior: GDPRtEXT nombra los artículos donde se define cada
  concepto (los más frecuentes son el 6, el 4, el 9 y el 5) y la búsqueda devuelve a menudo otro
  artículo del mismo tema. Para `Consent` da primero el 7 ("Conditions for consent") y GDPRtEXT
  apunta al 4 y al 6; para `Processor` da el 28 ("Processor") y GDPRtEXT apunta al 4.
- GDPRtEXT + GDPR en "Mi ontología y una ley": la misma alineación que GDPRtEXT solo (80
  aceptadas, 940 tripletas) y 210 de 235 conceptos con artículos para revisar (580 enlaces a 76
  artículos distintos).

**La web (S7-T12).** Los tres modos funcionan con los archivos de las corridas anteriores y
reutilizan lo ya calculado.

### Cómo usar
Consola:

    py -m src.alignment.knowledge --estado               # colecciones dpv y decisions
    py -m src.alignment.benchmark                        # línea base y origen de los candidatos
    py -m src.alignment.decider_eval                     # evaluación antes/después del decisor
    py -m src.alignment.user_alignment ONTOLOGIA --formato turtle     # Mi ontología
    py -m src.ingest.law_upload LEY [--consulta "texto"]             # segmenta e indexa una ley
    py -m src.core.acceptance                            # calibra la regla de aceptación
    py -m src.core.law_concepts LEY                      # extrae y acepta los conceptos
    py -m src.core.law_profile LEY --formato turtle      # Una ley
    py -m src.alignment.law_evidence --calibrar          # calibra los artículos para revisar
    py -m src.alignment.law_evidence ONTOLOGIA LEY --formato turtle  # Mi ontología y una ley

Las que llaman al LLM aceptan `--sin-ia`, salvo `decider_eval`, que limita las llamadas con
`--max-llamadas` y reutiliza su caché. Web (demo):

    py -m streamlit run app/app.py

y entrar a **Alinear con el DPV** (sección "Tu ontología").

### Pruebas
`tests/test_vector_store.py`, `test_alignment_knowledge.py`, `test_ai_prompts.py`,
`test_llm_client.py`, `test_alignment_reference.py`, `test_alignment_benchmark.py`,
`test_alignment_decider.py`, `test_alignment_decider_eval.py`, `test_law_text.py`,
`test_law_upload.py`, `test_labels.py`, `test_extract.py`, `test_acceptance.py`,
`test_law_concepts.py`, `test_law_profile.py`, `test_law_evidence.py` y
`test_sprint7_pipeline.py` (los modos con ley de la web con las leyes reales, la base real,
OntoPriv-Core y el DPV, con embeddings y LLM de prueba: la ley de Colombia se convierte en un
perfil que reutiliza el núcleo y el .zip resuelve sus importaciones; GDPRtEXT con el GDPR da la
misma alineación que GDPRtEXT solo, con sus artículos para revisar; la base no se toca). Las
pruebas lentas (`-m slow`) usan el modelo real de embeddings o el LLM. Suite del proyecto: 757
pruebas en verde sin las marcadas `slow` (770 en total).

### Alcance y pendientes
- **Sprint 8 (validación estructural):** SHACL y el razonador de Jena sobre lo que producen los
  tres modos; sigue pendiente el literal `rdf:langString` sin idioma de OntoPriv.
- **Sprint 10 (comentarios del usuario):** el material ya está en `alignment.csv`: las
  sugeridas (21 en GDPRtEXT), las propiedades (nunca automáticas) y los artículos para revisar.
  Falta decidir cómo se escriben en la ontología los artículos que la persona confirme (la vista
  de trazabilidad muestra "de qué artículo sale" cada concepto) y si entra el botón "Confirmar
  esta versión".
- **Mejoras posibles, por decidir:**
  - **Compartir la caché de la IA entre espacios de trabajo.** Hoy GDPRtEXT subido solo y con el
    GDPR gasta dos veces las mismas 28 llamadas (22 del decisor y 6 de las etiquetas).
  - **Indexar la ley por párrafo o por definición numerada.** GDPRtEXT apunta al párrafo
    (`article4-11`), y un artículo largo como el 4 del GDPR (26 definiciones) queda en un solo
    vector que mezcla todos sus temas.
- **Limitaciones para la tesis:**
  - la referencia es ruidosa (43,5 % de acuerdo entre la muestra ciega y el Sprint 5) y no es
    independiente (la mayoría de las decisiones confirman propuestas del asistente);
  - la segunda grilla de la regla se declaró después de ver los errores de la primera (la
    validación cruzada mide el optimismo: 72,0 % frente a 75,0 %);
  - el LLM se muestra confiado de más: 119 sugeridas con confianza alta que la regla no acepta;
  - la regla de aceptación de conceptos de una ley es estructural y no juzga el significado;
  - el segmentador deja a veces el título del capítulo siguiente al final del texto de un
    artículo;
  - hay correspondencias para revisar en los comentarios: una heredada de la base
    (`NationalDatabaseRegistry` → `ProtectionOfNationalSecurity`) y una aceptada en el GDPR
    (`EmploymentContext` `exactMatch` `ProcessingContext`).