"""Step 1 of the pipeline: read PDFs with PyMuPDF, keeping page numbers.

For each page we return:
  - the lab tables on it (panel name + rows), read cell-by-cell so numbers stay intact
  - the free text that is NOT inside a table (headers, pathologist remarks, explainer prose)
  - document metadata (patient_id, report_id, report_date, ...) parsed from the header

Output is plain dicts, so chunking can be swapped without touching this file.
"""

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf

# Header patterns used by the synthetic lab. Real labs differ: adapt these regexes when
# the team settles on a real/de-identified corpus.
HEADER_PATTERNS = {
    "patient_id": r"Patient ID:\s*(\S+)",
    "report_id": r"Report ID:\s*(\S+)",
    "report_date": r"Report Date:\s*(\d{4}-\d{2}-\d{2})",
    "age_sex": r"Age/Sex:\s*(\d+\s*/\s*[MF])",
}


def _parse_metadata(first_page_text: str, path: Path, doc_type: str) -> dict:
    meta = {"source_file": path.name, "doc_type": doc_type}
    for key, pat in HEADER_PATTERNS.items():
        m = re.search(pat, first_page_text)
        if m:
            meta[key] = m.group(1).replace(" ", "")
    if doc_type == "reference":
        meta["patient_id"] = "REFERENCE"       # shared docs, visible to every patient
    return meta


def _read_table(page, table) -> list[list[str]]:
    """Read each cell by its bounding box (more reliable than table.extract() for numbers)."""
    rows = []
    for row in table.rows:
        rows.append([page.get_textbox(pymupdf.Rect(c)).strip() if c else "" for c in row.cells])
    return rows


def _panel_title(blocks, table_rect) -> str:
    """The panel name is the text block sitting closest above the table."""
    above = [b for b in blocks if b[3] <= table_rect.y0 + 1]
    if not above:
        return ""
    nearest = max(above, key=lambda b: b[3])
    return nearest[4].strip().splitlines()[-1]


def load_pdf(path: Path, doc_type: str = "report") -> dict:
    doc = pymupdf.open(path)
    meta = _parse_metadata(doc[0].get_text(), path, doc_type)
    pages = []
    for page in doc:
        blocks = page.get_text("blocks")                 # (x0, y0, x1, y1, text, block_no, type)
        tables = page.find_tables().tables if doc_type == "report" else []
        table_rects = [pymupdf.Rect(t.bbox) for t in tables]

        parsed_tables = []
        for t, rect in zip(tables, table_rects):
            rows = _read_table(page, t)
            parsed_tables.append({"panel": _panel_title(blocks, rect), "header": rows[0], "rows": rows[1:]})

        # free text = blocks that do not overlap any table
        free = [b[4].strip() for b in blocks
                if b[6] == 0 and not any(pymupdf.Rect(b[:4]).intersects(r) for r in table_rects)]
        pages.append({"page": page.number + 1, "tables": parsed_tables,
                      "text": "\n".join(x for x in free if x), "full_text": page.get_text()})
    doc.close()
    return {"meta": meta, "pages": pages}


def _load_one(args):
    path, doc_type = args
    return str(path), path.stat().st_mtime, load_pdf(path, doc_type)


def load_corpus(reports_dir: str, reference_dir: str | None = None, cache_file: str | None = None,
                workers: int | None = None) -> list[dict]:
    """Parse all PDFs in parallel. Parsed pages are cached (keyed by path + modified time),
    because table detection costs ~1 s/page and is identical for every chunking experiment."""
    jobs = [(p, "report") for p in sorted(Path(reports_dir).glob("*.pdf"))]
    if reference_dir and Path(reference_dir).exists():
        jobs += [(p, "reference") for p in sorted(Path(reference_dir).glob("*.pdf"))]

    cache = {}
    if cache_file and Path(cache_file).exists():
        cache = json.loads(Path(cache_file).read_text())
    todo = [j for j in jobs if cache.get(str(j[0]), {}).get("mtime") != j[0].stat().st_mtime]
    if todo:
        with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
            for path, mtime, doc in pool.map(_load_one, todo, chunksize=4):
                cache[path] = {"mtime": mtime, "doc": doc}
        if cache_file:
            Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_file).write_text(json.dumps(cache))
    return [cache[str(p)]["doc"] for p, _ in jobs]
