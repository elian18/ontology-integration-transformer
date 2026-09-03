# ontology-integration-transformer

Componente de Integración y Transformación de Ontologías de Privacidad para
Sistemas de Cumplimiento Legal (Trabajo de Integración Curricular, EPN).

Toma una ontología o un texto de ley, la caracteriza y (en fases siguientes) la alinea
con el DPV usando OntoPriv como base, para generar una ontología nueva descargable en
varios formatos.

## Requisitos
- Python 3.13 (en Windows se invoca con `py`)
- Java 24 (para el razonador de Apache Jena en la fase de validación)

## Instalación
    py -m venv .venv
    source .venv/Scripts/activate    # Git Bash en Windows
    py -m pip install -r requirements.txt
    cp .env.example .env             # y colocar la clave del LLM (LLM_API_KEY, LLM_MODEL, LLM_BASE_URL)

## Uso
Reporte de insumos por consola (ontología + ley + DPV):

    py -m src.ingest.cli

Interfaz web (demostrador):

    py -m streamlit run app/app.py

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