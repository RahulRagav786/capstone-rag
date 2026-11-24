"""Step 4: semantic retrieval for ONE patient, then cross-encoder reranking.

Privacy rule: the patient filter is applied INSIDE the vector search (Chroma `where`),
never after it, so another patient's chunks can never be returned.
"""

from sentence_transformers import CrossEncoder

from .embed_store import Embedder, get_collection

REFERENCE_ID = "REFERENCE"
_RERANKERS: dict = {}


def patient_filter(patient_id: str, report_id: str | None = None) -> dict:
    """Chroma `where` clause: always the patient, optionally one specific report of theirs."""
    if report_id:
        return {"$and": [{"patient_id": patient_id}, {"report_id": report_id}]}
    return {"patient_id": patient_id}


class Retriever:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.r = cfg["retrieval"]
        self.embedder = Embedder(cfg)
        self.collection = get_collection(cfg)
        self.reranker = None
        if self.r.get("use_reranker"):
            name = cfg["reranker"]["model"]
            if name not in _RERANKERS:
                _RERANKERS[name] = CrossEncoder(name, device=cfg["embedding"].get("device", "cpu"))
            self.reranker = _RERANKERS[name]

    def _dense(self, query_vec, where: dict, k: int) -> list[dict]:
        res = self.collection.query(query_embeddings=[query_vec], n_results=k, where=where,
                                    include=["documents", "metadatas", "distances"])
        return [{"id": i, "text": d, "metadata": m, "dense_score": 1 - dist}
                for i, d, m, dist in zip(res["ids"][0], res["documents"][0],
                                         res["metadatas"][0], res["distances"][0])]

    def _rerank(self, question: str, hits: list[dict]) -> list[dict]:
        if not self.reranker or not hits:
            return hits
        scores = self.reranker.predict([(question, h["text"]) for h in hits])
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        by_rerank = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        if self.cfg["reranker"].get("mode", "replace") == "replace":
            return by_rerank
        # "rrf": reciprocal rank fusion of the dense order and the reranker order, so neither
        # model alone decides (the cross-encoder is weak on lab abbreviations like MCV / TSH).
        k = 60
        for rank, h in enumerate(hits):                   # `hits` arrives in dense order
            h["fused_score"] = 1 / (k + rank + 1)
        for rank, h in enumerate(by_rerank):
            h["fused_score"] += 1 / (k + rank + 1)
        return sorted(hits, key=lambda h: h["fused_score"], reverse=True)

    def retrieve(self, patient_id: str, question: str, final_k: int | None = None,
                 use_reranker: bool | None = None, report_id: str | None = None) -> dict:
        final_k = final_k or self.r["final_k"]
        rerank = self.r.get("use_reranker") if use_reranker is None else use_reranker
        qv = self.embedder.embed_query(question)

        where = patient_filter(patient_id, report_id)
        patient = self._dense(qv, where, self.r["candidate_k"])
        if rerank:
            patient = self._rerank(question, patient)
        reference = []
        if self.r.get("reference_k", 0):
            reference = self._dense(qv, {"patient_id": REFERENCE_ID}, max(self.r["reference_k"] * 3, 5))
            reference = (self._rerank(question, reference) if rerank else reference)[: self.r["reference_k"]]
        return {"patient": patient[:final_k], "reference": reference}
