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