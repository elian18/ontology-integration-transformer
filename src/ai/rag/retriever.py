from .embeddings import embed
from .vector_store import collection

def retrieve(query: str, top_k: int = 4) -> list[str]:
    q_emb = embed([query])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=top_k)
    return res["documents"][0] if res["documents"] else []