# ontology-integration-transformer

Componente de Integración y Transformación de Ontologías de Privacidad para
Sistemas de Cumplimiento Legal (Trabajo de Integración Curricular, EPN).

Toma una ontología de privacidad, un texto de ley o los dos, y los alinea con el DPV
(Data Privacy Vocabulary 2.3) usando como base OntoPriv y la LOPDP, revisadas a mano una
sola vez. Devuelve una ontología nueva en RDF/XML, OWL, Turtle o JSON-LD, validada y con
la procedencia de cada correspondencia. También convierte una ontología de un formato a
otro sin tocar su contenido. El historial de cada sprint está en `docs/CHANGELOG.md`.

## Requisitos
- Git y Python 3.13 (en Windows se invoca con `py`; los comandos de abajo son para Git Bash)
- Java 21 o superior (probado con Java 24) y Apache Jena 6.2.0 para el razonador y las
  consultas SPARQL de la validación; sin ellos se usan owlrl (OWL 2 RL) y rdflib
- Una clave de un LLM compatible con la API de OpenAI (probado con `gemini-3.1-flash-lite`)
- Internet la primera vez: `pip` baja las librerías (torch es la más pesada) y la primera
  subida baja el modelo de embeddings de Hugging Face (`EMBEDDING_MODEL`)

## Instalación
    py -m venv .venv
    source .venv/Scripts/activate    # Git Bash en Windows
    py -m pip install -r requirements.txt
    cp .env.example .env             # y colocar la clave del LLM

En `.env` van `LLM_API_KEY`, `LLM_BASE_URL` y `LLM_MODEL`. El `.env` no se sube a git.
`EDITABLE_BASE=0` deja la base del sistema en solo lectura (ver "La base del sistema").
`BUILD_TOOLS=0` muestra solo el asistente, como en la versión desplegada; con `BUILD_TOOLS=1`
aparece el menú "Herramientas de construcción", donde se arma y revisa la base.

## Razonador (Apache Jena)
Jena no se instala con pip: se descarga una vez dentro de `tools/` (git lo ignora).

    mkdir -p tools && cd tools
    curl -LO https://dlcdn.apache.org/jena/binaries/apache-jena-6.2.0.zip
    curl -LO https://downloads.apache.org/jena/binaries/apache-jena-6.2.0.zip.sha512
    sha512sum apache-jena-6.2.0.zip && cat apache-jena-6.2.0.zip.sha512   # deben coincidir
    py -m zipfile -e apache-jena-6.2.0.zip .
    cd ..

La carpeta y el razonador se configuran en el bloque `validation` de `config/config.yaml`.

## La web
    py -m streamlit run app/app.py

La página de inicio ofrece dos caminos.

**Solo transformar.** Subes una ontología y la descargas en otro formato. Tu archivo no
pasa por la IA y el contenido es el mismo en los cuatro formatos.

**Alinear y transformar.** Un asistente en cinco pasos:
1. **¿Qué tienes?** Eliges el modo:
   - *Mi ontología*: una ontología (.rdf, .owl, .ttl o .jsonld);
   - *Una ley*: un texto legal (.txt o PDF). La IA extrae sus conceptos, una regla fija
     los acepta y se arma un perfil que importa OntoPriv-Core;
   - *Mi ontología y una ley*: la ley completa las definiciones que le faltan a tu ontología
     y da los artículos de evidencia de cada concepto.
2. **Se procesa.** Una barra de progreso dice en palabras simples qué está haciendo:
   - reutiliza lo que ya está revisado en la base;
   - busca los equivalentes en el DPV y la IA elige el mejor (una regla fija decide qué entra);
   - pone los nombres en español e inglés y la IA revisa los enlaces;
   - revisa que esté bien formada, sin contradicciones, y que responda preguntas básicas.

   El botón "Atrás" vuelve al paso anterior sin perder los archivos que subiste.
3. **¡Tu ontología está lista!** Una vista previa en grafo muestra tus conceptos y sus
   equivalentes en el DPV.
4. **Lo que hicimos.** Tres pantallas explican lo que se encontró, cómo se enlazó con el DPV
   y cómo se revisó, con ejemplos de tu propia ontología.
5. **Tú decides.** Las preguntas sobre lo que quedó en duda, una a una (ver "Las preguntas y
   las versiones"). Responder es opcional. "Ajustar otro concepto" permite cambiar cualquier
   otra correspondencia y buscar otro término del DPV.

Al final, **Descarga tu ontología** da un solo archivo por formato,
`<nombre>-dpv-aligned`, con tus conceptos, sus equivalentes en el DPV, las etiquetas es/en,
las definiciones y lo que usa de OntoPriv-Core. Se abre directamente en Protégé; Protégé 5.6
no abre JSON-LD. Si hay una versión anterior, también se puede descargar.

"Ver los detalles técnicos" muestra lo demás:
- la validación en cuatro tarjetas: Forma (SHACL), Lógica (razonador), Preguntas (SPARQL)
  e IA revisora;
- la tabla de conceptos con su estado;
- la ficha de procedencia de cada concepto;
- lo que queda para revisar;
- las preguntas de competencia y los artículos de la ley.

"Herramientas de construcción" reúne las páginas con las que se construyó la base: Núcleo,
Propuestas IA, Aprobar conceptos IA, Alineación DPV y Preguntas de la base. Solo aparecen con
`BUILD_TOOLS=1`; la versión desplegada no las muestra, pero el código sigue en `app/views`.

## Las preguntas y las versiones
**Qué se pregunta.** Lo que queda en duda en una subida:
- las *sugeridas*: la IA eligió un término con confianza alta y la regla no lo aceptó, como
  las propiedades, que nunca se alinean solas;
- las correspondencias que la regla aceptó y la IA revisora marcó como incoherentes o
  dudosas.

Hay una pregunta por duda, hasta 10, con opciones fijas: tomar la sugerida, cambiar a la
relación que propone la IA revisora, quitar la correspondencia o dejarla como está. La
pregunta muestra lo que dijo cada IA, sin recomendar una opción: decides tú. Las preguntas
redactadas por la IA (prompt `ask.v1`) se midieron y no se adoptaron (ver
`docs/CHANGELOG.md`, v1.0.0).

**Cómo se aplica una respuesta.** Cada respuesta y cada ajuste pasa por una regla fija. La
regla revisa que el término exista en el DPV 2.3, que la relación sea SKOS, que una propiedad
vaya con una propiedad y que haya un cambio por concepto. Lo aceptado:
- se guarda en `work/user-decisions.jsonl`;
- crea una versión nueva de la alineación;
- deja la anterior en `history/v<N>/` con sus cuatro descargas.

"Dejar como está" también se recuerda, y lo respondido no se vuelve a preguntar. Lo que
decides no cambia tu archivo, la regla ni la base: queda encima de lo que decidieron la base
y la regla, correspondencia por correspondencia. Tus respuestas no se usan como precedentes
del decisor, y la IA revisora no vuelve a revisar lo que decidiste.

**Procedencia.** Cada versión trae su procedencia (PROV-O) dentro del paquete. Dice quién
decidió cada correspondencia (base, regla o usuario), cuándo y con qué versión, y enlaza
cada corrección con la que reemplazó.

## Consola
Alinear una ontología (igual que la web: cada subida se valida con la forma, la lógica, las
preguntas de competencia y la IA revisora; el detalle va en el paquete, en
`validation-log.txt`). En Git Bash, antes, `export PYTHONIOENCODING=utf-8` para que se vean
las tildes:

    py -m src.alignment.user_alignment ONTOLOGIA [--formato turtle] [--sin-ia]   # Mi ontología
    py -m src.core.law_profile LEY [--formato turtle]                            # Una ley
    py -m src.alignment.law_evidence ONTOLOGIA LEY        # Mi ontología y una ley

La consola muestra las preguntas de la subida y el comando para responderlas. Cada respuesta
o cambio deja una versión nueva:

    py -m src.alignment.user_alignment --espacio ESPACIO --responder "1=a,3=b"
    py -m src.alignment.user_alignment --espacio ESPACIO \
        --cambio "EmploymentContext broadMatch ProcessingContext"
    py -m src.alignment.user_alignment --espacio ESPACIO \
        --cambio "InfoAboutProcessing quitar" --aplicar

Sin `--aplicar`, `--cambio` solo muestra si la regla acepta el cambio. Otras formas:
`"Concepto relación Término en vez de OtroTérmino"`. `ESPACIO` es el id o la carpeta de la
subida en `data/workspaces/`: la consola lo imprime al final (p. ej.
`gdprtext-ontology-d35361ae`) y el mismo archivo cae siempre en el mismo espacio. Ahí está
cada subida, con sus carpetas `input/`, `work/`, `output/` y `history/`. Las respuestas de la
IA se guardan en `data/workspaces/_cache/`, que comparten todas las subidas.

Procedencia de un concepto:

    py -m src.provenance.upload --espacio ESPACIO --ver NOMBRE
    py -m src.provenance.base --ver NOMBRE

Validación de la base o de una subida:

    py -m src.validation.shape_inspector --base | ESPACIO
    py -m src.validation.logic_inspector --base | ESPACIO [--motor jena|owlrl]
    py -m src.validation.questions --base | ESPACIO [--motor jena|rdflib]
    py -m src.validation.ai_auditor ESPACIO

Mediciones (se comparan con las 552 decisiones de la base; las respuestas de la IA salen de
su caché):

    py -m src.alignment.decider_eval          # el decisor y su regla
    py -m src.validation.ai_auditor --medir   # la IA revisora
    py -m src.alignment.question_eval         # las preguntas de la IA frente a las fijas

## La base del sistema
OntoPriv-Core, el perfil de Ecuador (LOPDP) y su alineación con el DPV, revisada a mano
una vez (552 conceptos, 190 correspondencias). Viene en el repositorio, en `data/output`:

| Módulo | Qué es |
|---|---|
| `ontopriv-core` | El núcleo reutilizable, sacado de OntoPriv |
| `profile-ecuador-lopdp` | El perfil de Ecuador, que importa el núcleo |
| `ontopriv-dpv-alignment` | Las correspondencias con el DPV y los conceptos tomados de la LOPDP |
| `ontopriv-gdpr-alignment` | El puente con el GDPR (OntoPriv ↔ GDPRtEXT por el DPV) |
| `ontopriv-provenance` | Quién decidió cada correspondencia y cuándo (PROV-O) |

Cada módulo está en `.rdf`, `.owl`, `.ttl` y `.jsonld`. `catalog-v001.xml` le dice a Protégé
dónde están los módulos que se importan, y `alignment-candidates.json` guarda los candidatos
de los que el programa hereda lo revisado. También van los archivos que muestran las
"Herramientas de construcción" (las propuestas de la IA, la tabla de lo aprobado, el puente y
las respuestas de las preguntas de la base). Lo demás de `data/output` (mediciones, reportes,
paquetes) no se sube a git.

Se rehace desde OntoPriv:

    py -m src.core.emit                    # núcleo y perfil, con las reparaciones
    py -m src.alignment.write              # la alineación, desde el registro de decisiones
    py -m src.transform.convert data/output/ontopriv-core.rdf \
        data/output/profile-ecuador-lopdp.rdf data/output/ontopriv-dpv-alignment.rdf --to all
    py -m src.alignment.bundle             # paquetes para Protégé
    py -m src.alignment.gdpr_bridge        # puente con el GDPR (OntoPriv ↔ GDPRtEXT por el DPV)
    py -m src.provenance.base              # procedencia (PROV-O) en los cuatro formatos

Con `BUILD_TOOLS=1` y `EDITABLE_BASE=1` en `.env`, la página "Alineación DPV" de
"Herramientas de construcción" corrige una decisión de la base. La corrección se agrega al
registro (`data/review/alignment-decisions.jsonl`) y la anterior queda como versión previa en la
procedencia. Después se rehacen la alineación y lo que sigue con los comandos de arriba. Las
subidas que heredaron de la base se actualizan solas la próxima vez que corren.

El puente usa una alineación fija de GDPRtEXT, copiada de una subida. Para tomar su versión
actual (también con `EDITABLE_BASE=1`):

    py -m src.alignment.gdpr_bridge --espacio ESPACIO

## Documentación de la base (WIDOCO)
La página de la base está publicada en https://elian18.github.io/ontology-integration-transformer/
(GitHub Pages: rama `master`, carpeta `/docs`). La web la enlaza en el inicio y en la pantalla
final del asistente. La genera WIDOCO 1.4.26, que necesita Java 17 o mayor. WIDOCO no se
instala con pip: se descarga una vez dentro de `tools/` (git lo ignora):

    mkdir -p tools/widoco
    F=widoco-1.4.26-jar-with-dependencies_JDK-17.jar
    curl -L -o tools/widoco/$F https://github.com/dgarijo/Widoco/releases/download/v1.4.26/$F

    py -m src.publish.widoco --oops        # OOPS! revisa OntoPriv y la base; WIDOCO escribe docs/
    py -m src.publish.widoco               # rehace la página con la evaluación guardada

La página junta los cuatro módulos (núcleo, perfil, alineación con el DPV y puente con el
GDPR) en un solo archivo, `data/output/widoco/ontopriv-base.ttl` (git lo ignora), con la
cabecera de autoría, el crédito a OntoPriv, una frase por concepto con sus correspondencias y
la tabla de todas las correspondencias. Los módulos de `data/output` no cambian.

OOPS! revisa OntoPriv como la publicaron sus autores (`data/input/ontopriv.rdf`) y la base con
su cabecera, sin las frases escritas para la página. Sus respuestas quedan en `docs/oops` y la
comparación va en la descripción de la página. `--oops` necesita internet
(https://oops.linkeddata.es). Solo el archivo de la base, sin WIDOCO:
`py -m src.publish.base_doc`.

## Pruebas
    py -m pytest -q                  # suite completa
    py -m pytest -m "not slow" -q    # rápida: sin el LLM, los embeddings ni Jena reales

Pruebas de punta a punta (S11-T09): cada caso de `data/reference/e2e-s11.yaml` pasa por lo
mismo que la web y sus resultados quedan solos en `data/output/e2e/s11`: `runs.jsonl` (una
línea por corrida) y `resumen.md` (la última corrida de cada caso). Los casos usan sus
propios espacios (`data/workspaces/_e2e`) y las respuestas de la IA que ya tienen las subidas:

    py -m src.e2e.runner data/reference/e2e-s11.yaml [--solo gdprtext pizza] [--limpio]

`--limpio` borra antes el espacio de cada caso, para medir una corrida completa.

Casos límite (S11-T10): su material lo genera `py -m src.e2e.edge` en
`data/workspaces/_e2e/_material` (git lo ignora: un archivo pasa el límite de 25 MB). Los casos
sin clave, sin Java y con la cuota agotada corren aislados, sin la caché de la IA, para que la
condición se pruebe de verdad.

## Insumos base del proyecto
- OntoPriv (RDF/XML, exportada de Protégé): ontología base, perfil OWL Full.
- LOPDP (PDF): texto normativo, 77 artículos.
- DPV 2.3 (Turtle): vocabulario de referencia para la alineación.
- GDPRtEXT (Turtle) y el GDPR en inglés (PDF): ejemplo principal de una subida y del puente.

## Formatos
- Ontología (semántico): RDF/XML, OWL (RDF/XML), Turtle y JSON-LD, como entrada y salida.
  OWL/XML solo como entrada (se convierte a RDF/XML con Protégé).
- Texto de ley (documento de origen): .txt y PDF, solo como entrada.

## Licencia
El código y la base se publican con la licencia Apache 2.0 (`LICENSE`). La base es una obra
derivada de OntoPriv (Gabriela Suntaxi, Kelvin Ojeda y Francisco Rodríguez, EPN), que también
es Apache 2.0; `NOTICE` dice de dónde viene cada material y qué se cambió.
