"""The one entry point teammates need:

    from rag.pipeline import RAGPipeline
    rag = RAGPipeline()                                  # or RAGPipeline(overrides={"generation.backend": "hf"})
    out = rag.answer("P0001", "Explain my HbA1c report")
    out["answer"], out["sources"]                        # text with [n] citations, and what [n] points to
"""

import time

from .config import load_config
from .generate import Generator, build_messages, format_sources
from .retrieve import Retriever, patient_filter

NOT_FOUND = "I could not find a report for this patient ID, so I can't answer from your results."


class RAGPipeline:
    def __init__(self, config_path=None, overrides: dict | None = None):
        self.cfg = load_config(config_path, overrides)
        self.retriever = Retriever(self.cfg)
        self.generator = Generator(self.cfg)

    def _all_patient_chunks(self, patient_id: str, report_id: str | None = None) -> dict:
        """'full_report' baseline: the whole report in the prompt, no search or ranking."""
        res = self.retriever.collection.get(where=patient_filter(patient_id, report_id),
                                            include=["documents", "metadatas"])
        hits = [{"id": i, "text": d, "metadata": m} for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]
        return {"patient": sorted(hits, key=lambda h: h["metadata"]["page"]), "reference": []}

    def answer(self, patient_id: str, question: str, mode: str | None = None,
               report_id: str | None = None) -> dict:
        """mode (or config retrieval.mode): rag | full_report | none (question only, no report).
        report_id: restrict to one report of this patient (default: all their reports)."""
        mode = mode or self.cfg["retrieval"].get("mode", "rag")
        t0 = time.time()
        if mode == "none":
            text = self.generator(build_messages(question, "(no sources provided)"))
            return {"answer": text, "sources": [], "patient_id": patient_id, "question": question,
                    "timing_s": {"retrieval": 0.0, "generation": round(time.time() - t0, 2)}}
        hits = self._all_patient_chunks(patient_id, report_id) if mode == "full_report" else \
            self.retriever.retrieve(patient_id, question, report_id=report_id)
        t1 = time.time()
        if not hits["patient"]:
            return {"answer": NOT_FOUND, "sources": [], "patient_id": patient_id, "question": question,
                    "timing_s": {"retrieval": t1 - t0, "generation": 0.0}}
        context, sources = format_sources(hits["patient"], hits["reference"])
        text = self.generator(build_messages(question, context))
        return {"answer": text, "sources": sources, "patient_id": patient_id, "question": question,
                "timing_s": {"retrieval": round(t1 - t0, 2), "generation": round(time.time() - t1, 2)}}
