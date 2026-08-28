from src.ai.rag.embeddings import embed
from src.ai.rag.vector_store import add
from src.ai.rag.retriever import retrieve
from src.ai.llm_client import LLMClient

def test_rag_circuit():
    docs = ["El titular tiene derecho de acceso a sus datos personales.",
            "El responsable debe garantizar la seguridad del tratamiento."]
    add(ids=["d1", "d2"], docs=docs, embeddings=embed(docs))
    hits = retrieve("derechos del titular", top_k=1)
    assert len(hits) == 1

def test_llm_connection():
    reply = LLMClient().ask("Responde solo con la palabra: OK")
    assert isinstance(reply, str) and len(reply) > 0