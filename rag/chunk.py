"""Step 2: turn ingested pages into chunks.

Two strategies (set in config.yaml -> chunking.strategy), so the report can compare them:

  row   : one chunk per lab-test row. Value, unit, reference range and flag always stay
          together, which is the failure mode of naive splitting on tabular lab reports.
          Non-table text (e.g. pathologist remarks) becomes its own chunk.
  fixed : fixed-size character windows over the raw page text (the naive baseline).

Reference (non-patient) documents are always split into paragraph-sized windows.

Every chunk carries metadata: patient_id, report_id, page, panel, tests ... so retrieval
can filter by patient and answers can cite pages.
"""

from collections import Counter

FLAG_TEXT = {"HIGH": "HIGH (above reference range)", "LOW": "LOW (below reference range)", "": "within reference range"}


def _base_meta(meta: dict, page: int, strategy: str) -> dict:
    keep = ("patient_id", "report_id", "report_date", "age_sex", "source_file", "doc_type")
    out = {k: meta[k] for k in keep if meta.get(k)}
    out.update(page=page, chunk_strategy=strategy)
    return out


def _window(text: str, size: int, overlap: int) -> list[str]:
    """Character windows that try to break on a newline or space (line breaks are preserved)."""
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n", start + size // 2, end), text.rfind(" ", start + size // 2, end))
            end = cut if cut > 0 else end
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def _repeated_lines(doc: dict) -> set:
    """Lines that appear on every page (letterhead / patient header) — not worth a chunk."""
    n = len(doc["pages"])
    if n < 2:
        return set()
    counts = Counter(line for p in doc["pages"] for line in set(p["text"].splitlines()))
    return {line for line, c in counts.items() if c == n}


def chunk_report_rows(doc: dict) -> list[dict]:
    meta, chunks = doc["meta"], []
    header_lines = _repeated_lines(doc)
    for page in doc["pages"]:
        base = _base_meta(meta, page["page"], "row")
        for table in page["tables"]:
            for row in table["rows"]:
                test, result, unit, ref, flag = (row + [""] * 5)[:5]
                if not test:
                    continue
                text = (f"{table['panel']} — {test}: {result} {unit} "
                        f"(reference range {ref}); {FLAG_TEXT.get(flag, flag)}. "
                        f"Report {meta.get('report_id', '')} dated {meta.get('report_date', '')}, page {page['page']}.")
                chunks.append({"text": text, "metadata": {**base, "panel": table["panel"], "tests": test,
                                                          "flag": flag or "NORMAL"}})
        # free text (remarks etc.), minus repeated headers and panel titles
        panels = {t["panel"] for t in page["tables"]}
        lines = [l for l in page["text"].splitlines() if l.strip() and l not in header_lines and l not in panels]
        free = " ".join(lines).strip()
        if len(free) > 40:
            chunks.append({"text": free + f" (Report {meta.get('report_id', '')}, page {page['page']})",
                           "metadata": {**base, "panel": "remarks", "tests": "", "flag": "NA"}})
    return chunks


def chunk_report_fixed(doc: dict, size: int, overlap: int) -> list[dict]:
    meta, chunks = doc["meta"], []
    for page in doc["pages"]:
        base = _base_meta(meta, page["page"], "fixed")
        tests_on_page = [r[0] for t in page["tables"] for r in t["rows"] if r and r[0]]
        for piece in _window(page["full_text"], size, overlap):
            lines = {l.strip() for l in piece.splitlines()}
            present = [t for t in tests_on_page if t in lines]   # whole-line match: "MCH" != "MCHC"
            chunks.append({"text": piece, "metadata": {**base, "panel": "", "tests": "|".join(present), "flag": "NA"}})
    return chunks


def chunk_reference(doc: dict, size: int = 700, overlap: int = 100) -> list[dict]:
    meta, chunks = doc["meta"], []
    for page in doc["pages"]:
        base = _base_meta(meta, page["page"], "reference")
        for piece in _window(page["full_text"], size, overlap):
            chunks.append({"text": piece, "metadata": {**base, "panel": "reference", "tests": "", "flag": "NA"}})
    return chunks


def chunk_corpus(docs: list[dict], cfg: dict) -> list[dict]:
    strategy = cfg["chunking"]["strategy"]
    size, overlap = cfg["chunking"]["fixed_size"], cfg["chunking"]["fixed_overlap"]
    out = []
    for doc in docs:
        if doc["meta"]["doc_type"] == "reference":
            out += chunk_reference(doc)
        elif strategy == "row":
            out += chunk_report_rows(doc)
        elif strategy == "fixed":
            out += chunk_report_fixed(doc, size, overlap)
        else:
            raise ValueError(f"unknown chunking strategy: {strategy}")
    # stable, unique ids (so re-indexing overwrites instead of duplicating)
    for i, c in enumerate(out):
        m = c["metadata"]
        c["id"] = f"{m.get('source_file')}::p{m['page']}::{m['chunk_strategy']}::{i}"
    return out
