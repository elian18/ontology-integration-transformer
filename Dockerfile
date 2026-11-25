# The program in one image (S11-T12): Python 3.13, Java 21 with Apache Jena 6.2.0, the libraries
# and the embeddings model. The key of the LLM is not in the image: docker-compose.yml passes the
# .env when the container starts. See "Despliegue con Docker" in the README.

# The JDK, not only the JRE: the reasoner runs JenaCheck.java as a source file (Java compiles it).
FROM eclipse-temurin:21-jdk AS java

FROM python:3.13-slim

ARG JENA_VERSION=6.2.0
ARG JENA_SHA512=fb97634215b4667124bdbe4f42efc25d4dc4e9046b70bfa702d7a71e1a8aed049ec79a38eb469cf0e13b18724cc024adec3cc35113a24e8fa62aa66091d7431e
ARG EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

ENV JAVA_HOME=/opt/java/openjdk \
    PATH=/opt/java/openjdk/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBEDDING_MODEL=$EMBEDDING_MODEL \
    HF_HOME=/home/app/.cache/huggingface \
    CHROMA_PATH=/app/chroma_db

COPY --from=java /opt/java/openjdk /opt/java/openjdk
WORKDIR /app

# torch from its CPU index (the one of PyPI brings the CUDA libraries, several GB), in the version
# that requirements.txt pins; then the rest.
COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==$(sed -n 's/^torch==//p' requirements.txt)" \
 && pip install -r requirements.txt

# Apache Jena from the Apache archive (it keeps every version), checked against its SHA-512.
RUN python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], 'jena.zip')" \
        "https://archive.apache.org/dist/jena/binaries/apache-jena-${JENA_VERSION}.zip" \
 && echo "${JENA_SHA512}  jena.zip" | sha512sum -c - \
 && mkdir -p tools && python -m zipfile -e jena.zip tools && rm jena.zip

# The program runs as a user without privileges; it writes only the workspaces and ChromaDB.
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/chroma_db /app/data/workspaces \
 && chown app:app /app/chroma_db /app/data /app/data/workspaces
USER app

# The embeddings model, downloaded once; from here on it is read without internet.
RUN python -c "import os; from sentence_transformers import SentenceTransformer; \
SentenceTransformer(os.environ['EMBEDDING_MODEL'])"
ENV HF_HUB_OFFLINE=1

COPY --chown=app:app . .

# The collections of the base (DPV and decisions) are built now, and the image is checked: Jena
# has to be the reasoner, not owlrl.
RUN python -m src.alignment.knowledge \
 && python -c "from src.validation import logic_inspector as li; \
engine = li.choose_engine(li.settings()); assert engine['name'] == 'jena', engine; \
print('Razonador: Apache Jena', engine['version'], 'con Java', engine['java'])"

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 CMD ["python", "-c", \
    "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=8)"]
CMD ["python", "-m", "streamlit", "run", "app/app.py", "--server.address=0.0.0.0", \
     "--server.port=8501", "--server.headless=true", "--server.fileWatcherType=none", \
     "--browser.gatherUsageStats=false"]
