"""Step 3: embed chunks with BGE and store them in ChromaDB (vectors + text + metadata).

All vector-DB code lives in this file. Swapping Chroma for FAISS/Qdrant only changes this file.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

_MODELS: dict = {}


class Embedder:
    """BGE embeddings. Queries get BGE's retrieval instruction prefix; passages do not."""

    def __init__(self, cfg: dict):
        e = cfg["embedding"]
        if e["model"] not in _MODELS:
            _MODELS[e["model"]] = SentenceTransformer(e["model"], device=e.get("device", "cpu"))
        self.model = _MODELS[e["model"]]
        self.instruction = e.get("query_instruction", "")
        self.batch_size = e.get("batch_size", 64)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 200).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(self.instruction + text, normalize_embeddings=True).tolist()


def collection_name(cfg: dict) -> str:
    """One collection per (chunking strategy, embedding model), so ablation indexes coexist."""
    model = cfg["embedding"]["model"].split("/")[-1].replace(".", "-")
    return f"{cfg['store']['collection']}_{cfg['chunking']['strategy']}_{model}"[:63]


def get_collection(cfg: dict, create: bool = False):
    client = chromadb.PersistentClient(path=cfg["paths"]["chroma_dir"],
                                       settings=Settings(anonymized_telemetry=False))
    name = collection_name(cfg)
    if create:
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
    return client.get_collection(name)


def add_chunks(chunks: list[dict], cfg: dict) -> int:
    """Add (or overwrite, same id) chunks in the existing index: used for API uploads."""
    if not chunks:
        return 0
    col = get_collection(cfg, create=True)
    vectors = Embedder(cfg).embed_passages([c["text"] for c in chunks])
    col.upsert(ids=[c["id"] for c in chunks], documents=[c["text"] for c in chunks],
               metadatas=[c["metadata"] for c in chunks], embeddings=vectors)
    return len(chunks)


def delete_report(cfg: dict, where: dict) -> None:
    """Delete chunks matching a Chroma filter, e.g. retrieve.patient_filter(patient_id, report_id)."""
    get_collection(cfg).delete(where=where)


def build_index(chunks: list[dict], cfg: dict, reset: bool = True) -> int:
    client = chromadb.PersistentClient(path=cfg["paths"]["chroma_dir"],
                                       settings=Settings(anonymized_telemetry=False))
    name = collection_name(cfg)
    if reset and name in [c.name for c in client.list_collections()]:
        client.delete_collection(name)
    col = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})

    embedder = Embedder(cfg)
    vectors = embedder.embed_passages([c["text"] for c in chunks])
    step = 2000                                           # Chroma caps batch size
    for i in range(0, len(chunks), step):
        part = chunks[i:i + step]
        col.add(ids=[c["id"] for c in part], documents=[c["text"] for c in part],
                metadatas=[c["metadata"] for c in part], embeddings=vectors[i:i + step])
    return col.count()
