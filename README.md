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

## Quick start (laptop)

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
python scripts/make_synthetic_reports.py --n 200     # synthetic PDFs + eval questions
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
| Embeddings | `BAAI/bge-small-en-v1.5` (bge-base as comparison) | strong retrieval quality for its size; runs on CPU |
| Vector store | ChromaDB (embedded, folder on disk) | native metadata filtering → patient filter inside the search; no server to run |
| Patient isolation | `where={"patient_id": ...}` inside the vector query | another patient's data can never be retrieved; checked by `cross_patient_leaks` in eval |
| Reranker | `BAAI/bge-reranker-base` cross-encoder over top-20 | re-scores question–chunk pairs jointly; fixes near-miss ordering of dense search |
| Grounding | numbered sources, cite-or-abstain prompt, no diagnosis/medication | safety for medical content; report text treated as data (prompt-injection guard) |

## Layout

```
rag/          ingest.py  chunk.py  embed_store.py  retrieve.py  generate.py  pipeline.py  config.py
scripts/      make_synthetic_reports.py  build_index.py  ask.py  eval_retrieval.py
config.yaml   all settings
data/eval/    retrieval_questions.jsonl   (frozen eval set: question + gold test + gold page)
```

All report data here is **synthetic**. Never commit real or de-identified patient reports or `chroma_db/`.
