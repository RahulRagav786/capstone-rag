"""HTTP API for the blood-report RAG engine.

Run:
    uvicorn rag.api:app --host 0.0.0.0 --port 8000
    open http://localhost:8000/docs                      # interactive Swagger UI

Environment variables (all optional):
    RAG_API_KEY      if set, every request except /health must send header  X-API-Key: <value>
    RAG_BACKEND      override generation.backend (ollama | hf | none)
    RAG_CONFIG       path to an alternative config.yaml
    RAG_CORS_ORIGINS comma-separated allowed browser origins (default "*")

Endpoints:
    GET  /health                          liveness + index size
    POST /ask        {patient_id, question}                 -> answer + cited sources
    POST /reports    multipart: file (PDF), patient_id?     -> report is indexed, returns ids
    POST /explain    multipart: file (PDF), question?, patient_id?   -> upload + answer in one call
    GET  /patients/{patient_id}/reports                     -> reports indexed for that patient
    DELETE /patients/{patient_id}/reports/{report_id}       -> remove one report of that patient
"""

import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .chunk import chunk_corpus
from .embed_store import add_chunks, delete_report
from .ingest import load_pdf
from .pipeline import RAGPipeline
from .retrieve import patient_filter

MAX_UPLOAD_MB = 15
ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
DEFAULT_EXPLAIN_QUESTION = "Explain my blood report. Which values are outside the normal range and what do they mean?"

STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    overrides = {"generation.backend": os.environ["RAG_BACKEND"]} if os.environ.get("RAG_BACKEND") else None
    rag = RAGPipeline(os.environ.get("RAG_CONFIG"), overrides)          # loads BGE + reranker once
    upload_dir = Path(rag.cfg["paths"]["chroma_dir"]).parent / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    STATE.update(rag=rag, upload_dir=upload_dir)
    yield
    STATE.clear()


app = FastAPI(title="Blood Report RAG API", version="1.0",
              description="Explains a patient's blood report, grounded in the report itself, with page citations. "
                          "Not a medical device: explanations are informational and do not replace a doctor.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("RAG_CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.environ.get("RAG_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(401, "missing or invalid X-API-Key header")


# ---------- schemas ----------
class AskRequest(BaseModel):
    patient_id: str = Field(pattern=ID_PATTERN, examples=["P0001"])
    question: str = Field(min_length=3, max_length=1000, examples=["Explain my HbA1c report"])
    mode: Literal["rag", "full_report"] = Field(
        "rag", description="rag = retrieve the most relevant results; full_report = give the model the whole report")


class Source(BaseModel):
    n: int
    label: str
    report_id: str | None = None
    page: int | None = None
    source_file: str | None = None
    doc_type: str | None = None
    text: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]
    patient_id: str
    question: str
    timing_s: dict


class UploadResponse(BaseModel):
    patient_id: str
    report_id: str
    report_date: str | None
    pages: int
    chunks_indexed: int


# ---------- helpers ----------
async def _save_and_index(file: UploadFile, patient_id: str | None) -> dict:
    rag = STATE["rag"]
    data = await file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"file larger than {MAX_UPLOAD_MB} MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "only PDF files are accepted")
    if patient_id and not re.match(ID_PATTERN, patient_id):
        raise HTTPException(422, "patient_id may only contain letters, digits, '_' and '-'")

    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(file.filename or "report.pdf").name)
    path = STATE["upload_dir"] / f"{uuid.uuid4().hex[:12]}_{safe_name}"     # unique: no id collisions
    path.write_bytes(data)
    try:
        doc = load_pdf(path, "report")
    except Exception as e:                                                  # corrupt / encrypted PDF
        path.unlink(missing_ok=True)
        raise HTTPException(422, f"could not read PDF: {e}") from e

    meta = doc["meta"]
    if patient_id:                       # explicit value wins over what the header regex found
        meta["patient_id"] = patient_id
    if not meta.get("patient_id"):
        path.unlink(missing_ok=True)
        raise HTTPException(422, "no patient ID found in the report header; send the patient_id form field")
    meta.setdefault("report_id", "U" + uuid.uuid4().hex[:10])
    if not any(p["full_text"].strip() for p in doc["pages"]):
        path.unlink(missing_ok=True)
        raise HTTPException(422, "the PDF has no extractable text (scanned image?). OCR is not supported yet")

    chunks = chunk_corpus([doc], rag.cfg)
    n = add_chunks(chunks, rag.cfg)
    return {"patient_id": meta["patient_id"], "report_id": meta["report_id"],
            "report_date": meta.get("report_date"), "pages": len(doc["pages"]), "chunks_indexed": n}


# ---------- routes ----------
@app.get("/health")
def health():
    rag = STATE.get("rag")
    return {"status": "ok" if rag else "starting",
            "chunks_indexed": rag.retriever.collection.count() if rag else 0,
            "generation_backend": rag.cfg["generation"]["backend"] if rag else None}


@app.post("/ask", response_model=AnswerResponse, dependencies=[Depends(require_key)])
def ask(req: AskRequest):
    return STATE["rag"].answer(req.patient_id, req.question, mode=req.mode)


@app.post("/reports", response_model=UploadResponse, dependencies=[Depends(require_key)])
async def upload_report(file: UploadFile = File(..., description="Blood report PDF"),
                        patient_id: str | None = Form(None, description="Overrides the ID found in the PDF header")):
    return await _save_and_index(file, patient_id)


@app.post("/explain", dependencies=[Depends(require_key)])
async def explain_report(file: UploadFile = File(..., description="Blood report PDF"),
                         question: str = Form(DEFAULT_EXPLAIN_QUESTION),
                         patient_id: str | None = Form(None)):
    """Upload a report and get an explanation in one call. With the default question the model sees the
    whole report (full_report mode); with a specific question it retrieves the relevant results."""
    info = await _save_and_index(file, patient_id)
    rag = STATE["rag"]
    mode = "full_report" if question == DEFAULT_EXPLAIN_QUESTION else "rag"
    # restricted to the report just uploaded, not the patient's older reports;
    # run in a worker thread so a slow LLM call does not block the server
    out = await run_in_threadpool(rag.answer, info["patient_id"], question, mode, info["report_id"])
    return {"upload": info, **out}


@app.get("/patients/{patient_id}/reports", dependencies=[Depends(require_key)])
def list_reports(patient_id: str):
    if not re.match(ID_PATTERN, patient_id):
        raise HTTPException(422, "invalid patient_id")
    res = STATE["rag"].retriever.collection.get(where={"patient_id": patient_id}, include=["metadatas"])
    reports = {}
    for m in res["metadatas"]:
        r = reports.setdefault(m["report_id"], {"report_id": m["report_id"], "report_date": m.get("report_date"),
                                                "source_file": m.get("source_file"), "pages": set()})
        r["pages"].add(m["page"])
    return [{**r, "pages": sorted(r["pages"])} for r in reports.values()]


@app.delete("/patients/{patient_id}/reports/{report_id}", dependencies=[Depends(require_key)])
def remove_report(patient_id: str, report_id: str):
    """Scoped to the patient: report IDs come from PDF headers and are not unique across patients."""
    if not (re.match(ID_PATTERN, report_id) and re.match(ID_PATTERN, patient_id)):
        raise HTTPException(422, "invalid id")
    rag = STATE["rag"]
    where = patient_filter(patient_id, report_id)
    found = rag.retriever.collection.get(where=where, include=["metadatas"])["metadatas"]
    if not found:
        raise HTTPException(404, "report not found for this patient")
    delete_report(rag.cfg, where)
    for name in {m.get("source_file") for m in found}:          # remove the stored upload too
        f = STATE["upload_dir"] / name
        if name and f.parent == STATE["upload_dir"] and f.exists():
            f.unlink()
    return {"deleted": report_id, "chunks_removed": len(found)}
