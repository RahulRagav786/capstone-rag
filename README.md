# Blood-Report RAG (Capstone Track A — RAG layer)

Explains a patient's blood report (CBC, HbA1c, lipids, kidney, thyroid) in plain language,
grounded only in that patient's own report, with page citations.

```
PDF report ─► PyMuPDF (text + tables + page no.) ─► chunking (one lab test per chunk)
          ─► BGE embeddings ─► ChromaDB (vectors + text + patient metadata)

question + patient_id ─► BGE query ─► Chroma search filtered to that patient (top 20)
                      ─► bge-reranker (top 5) + 2 general-explainer chunks
                      ─► Qwen2.5-7B-Instruct ─► answer with [n] citations → report page
```

## Status / handoff

**Done (RAG role):** ingestion, chunking, embeddings, vector store, patient-filtered retrieval,
reranking, grounded generation with page citations, retrieval evaluation, synthetic corpus + eval set.

**For the rest of the team:**

| Who | What to do |
|---|---|
| Fine-tuning | Point `generation.hf.adapter_path` at your LoRA adapter (`~/checkpoints/job-<ID>`) |
| Evaluation | Run the 3-way comparison on the GPU server (table below) using `RAGPipeline().answer(...)`; answer-quality metrics / LLM-as-judge live with you |
| Serving | Call `RAGPipeline().answer(patient_id, question)` from the FastAPI endpoint; it returns `answer` + `sources` |
| Data | If real/de-identified reports replace the synthetic ones, check the header regexes in `rag/ingest.py` (`HEADER_PATTERNS`) |

**Not yet verified:** the `hf` backend (transformers + LoRA adapter) is written but untested; it needs a CUDA GPU.
Test it in a Slurm job on VM03 before access ends on 2 Dec 2026. Set `embedding.device: cuda` there.

## Quick start (laptop)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
# python scripts/make_synthetic_reports.py --n 200   # optional: PDFs are already in data/reports/
python scripts/build_index.py                        # PDFs -> chunks -> BGE -> chroma_db/
ollama pull qwen2.5:7b-instruct                      # 4-bit local copy of the base model
python scripts/ask.py --patient P0001 "Explain my HbA1c report"
python scripts/eval_retrieval.py                     # Recall@k / MRR, with and without reranker
```

## Using it from other code (serving / evaluation teammates)

```python
from rag.pipeline import RAGPipeline
rag = RAGPipeline()                       # reads config.yaml
out = rag.answer("P0001", "Explain my HbA1c report")
out["answer"]    # text with [1], [2] ... citations
out["sources"]   # [{n, label, report_id, page, source_file, text}, ...]
```

Overrides without editing the file: `RAGPipeline(overrides={"generation.backend": "none"})`.

## On the GPU server (fine-tuned model)

Set in `config.yaml`:

```yaml
generation:
  backend: hf
  hf:
    base_model: Qwen/Qwen2.5-7B-Instruct      # or the VM's local read-only path
    adapter_path: ~/checkpoints/job-<ID>      # the fine-tuning run's LoRA adapter; null = base model
```

Then run `ask.py` (or the evaluation script) inside a Slurm job — GPU use is only allowed via `sbatch`/`srun`.
The required comparison is three runs of the same eval set:

| Run | backend | adapter_path | retrieval.mode |
|---|---|---|---|
| Base + RAG | hf | null | rag |
| Fine-tuned | hf | job-ID | `full_report` (whole report pasted, no search) or `none` (question only) — team to agree which |
| Fine-tuned + RAG | hf | job-ID | rag |

## Design decisions (for the report)

| Decision | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF `find_tables()` + per-cell `get_textbox()` | `table.extract()` corrupted decimals ("12.2" → "122\n."); reading each cell by its bounding box keeps values exact |
| Chunking | one chunk per lab-test row (`row`), remarks as separate chunks | keeps value + unit + reference range + flag together; compared against fixed 500-char windows (`fixed`) |
| Embeddings | `BAAI/bge-small-en-v1.5` | strong retrieval quality for its size; runs on a laptop CPU in ~50 s for 4,200 chunks. Already reaches R@5 = 0.99, so a larger model was not needed (a bge-base run is one command: `--model BAAI/bge-base-en-v1.5`) |
| Vector store | ChromaDB (embedded, folder on disk) | native metadata filtering → patient filter inside the search; no server to run |
| Patient isolation | `where={"patient_id": ...}` inside the vector query | another patient's data can never be retrieved; checked by `cross_patient_leaks` in eval |
| Reranker | `BAAI/bge-reranker-base` over top-20, **fused with the dense ranking (RRF)** | used alone it *hurt* (it ranks "Total Cholesterol" first for "Is my MCV normal?": weak on lab abbreviations); fused, it improves Recall@1 0.835 → 0.89 |
| Grounding | numbered sources, cite-or-abstain prompt, no diagnosis/medication | safety for medical content; report text treated as data (prompt-injection guard) |

## Retrieval results (200 synthetic reports, 600 questions, bge-small)

| Chunking | Ranking | R@1 | R@3 | R@5 | MRR | Cross-patient leaks | ms/query (M2 CPU) |
|---|---|---|---|---|---|---|---|
| row | dense only | 0.835 | 0.970 | 0.993 | 0.898 | 0 | 27 |
| row | reranker only | 0.725 | 0.905 | 0.952 | 0.822 | 0 | 886 |
| **row** | **dense + reranker (RRF)** | **0.890** | 0.968 | 0.992 | **0.931** | 0 | 892 |
| fixed 500 | dense only | 0.750 | 0.982 | 1.000 | 0.862 | 0 | 23 |
| fixed 500 | reranker only | 0.480 | 0.898 | 0.998 | 0.691 | 0 | 659 |

Caveat: fixed chunks hold ~9 tests each and a report has only ~5 of them, so fixed R@5 ≈ "the whole
report" and is not comparable at equal k. Compare R@1 / MRR, or prompt tokens per answer.

End-to-end check (Ollama 4-bit Qwen2.5-7B, 8 GB M2): correct, cited answer for "Explain my HbA1c report";
~8 min generation because the model does not fit in RAM. Run generation evals on the GPU server.

## Layout

```
rag/          ingest.py  chunk.py  embed_store.py  retrieve.py  generate.py  pipeline.py  config.py
scripts/      make_synthetic_reports.py  build_index.py  ask.py  eval_retrieval.py
config.yaml   all settings
data/eval/    retrieval_questions.jsonl   (frozen eval set: question + gold test + gold page)
```

All report data here is **synthetic**. Never commit real or de-identified patient reports or `chroma_db/`.
