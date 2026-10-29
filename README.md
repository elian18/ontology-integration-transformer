# ontology-integration-transformer

Componente de Integración y Transformación de Ontologías de Privacidad para
Sistemas de Cumplimiento Legal (Trabajo de Integración Curricular, EPN).

Toma una ontología o un texto de ley, la caracteriza y (en fases siguientes) la alinea
con el DPV usando OntoPriv como base, para generar una ontología nueva descargable en
varios formatos.

## Requisitos
- Python 3.13 (en Windows se invoca con `py`)
- Java 21 o superior (probado con Java 24) y Apache Jena 6.2.0 para el razonador de la
  validación; sin ellos el inspector de lógica usa owlrl (OWL 2 RL en Python)

## Instalación
    py -m venv .venv
    source .venv/Scripts/activate    # Git Bash en Windows
    py -m pip install -r requirements.txt
    cp .env.example .env             # y colocar la clave del LLM (LLM_API_KEY, LLM_MODEL, LLM_BASE_URL)

## Razonador (Apache Jena)
Jena no se instala con pip: se descarga una vez dentro de `tools/` (git lo ignora).

    mkdir -p tools && cd tools
    curl -LO https://dlcdn.apache.org/jena/binaries/apache-jena-6.2.0.zip
    curl -LO https://downloads.apache.org/jena/binaries/apache-jena-6.2.0.zip.sha512
    sha512sum apache-jena-6.2.0.zip && cat apache-jena-6.2.0.zip.sha512   # deben coincidir
    py -m zipfile -e apache-jena-6.2.0.zip .
    cd ..

La carpeta y el razonador se configuran en el bloque `validation` de `config/config.yaml`.

## Uso
Reporte de insumos por consola (ontología + ley + DPV):

    py -m src.ingest.cli

Interfaz web (demostrador):

    py -m streamlit run app/app.py

Validación de la base o de un espacio de trabajo (forma con SHACL, lógica con el razonador):

    py -m src.validation.shape_inspector --base
    py -m src.validation.logic_inspector --base
    py -m src.validation.logic_inspector ESPACIO --motor owlrl

## Pruebas
    py -m pytest -q                  # suite completa
    py -m pytest -m "not slow" -q    # rápida (omite la prueba de red/LLM)

## Insumos base del proyecto
- OntoPriv (RDF/XML, exportada de Protégé): ontología base, perfil OWL Full.
- LOPDP (PDF): texto normativo, 77 artículos.
- DPV 2.3 (Turtle): vocabulario de referencia para la alineación.

## Formatos
- Ontología (semántico): RDF/XML, Turtle, JSON-LD como entrada y salida.
  OWL/XML solo como entrada (se convierte a RDF/XML con Protégé).
- Texto de ley (documento de origen): .txt y PDF, solo como entrada.