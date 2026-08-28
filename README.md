# ontology-integration-transformer

Componente de Integración y Transformación de Ontologías de Privacidad para
Sistemas de Cumplimiento Legal (Trabajo de Integración Curricular, EPN).

## Requisitos
- Python 3.11 en adelante
- JDK 17+ para razonadores

## Instalación
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env             # y colocar la clave del LLM

## Prueba de humo
    pytest -v