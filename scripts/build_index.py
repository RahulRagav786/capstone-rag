"""Read PDFs -> chunk -> embed -> store in ChromaDB.

  python scripts/build_index.py                          # uses config.yaml
  python scripts/build_index.py --strategy fixed         # build the baseline index for the ablation
  python scripts/build_index.py --model BAAI/bge-base-en-v1.5
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.chunk import chunk_corpus  # noqa: E402
from rag.config import load_config  # noqa: E402
from rag.embed_store import build_index, collection_name  # noqa: E402
from rag.ingest import load_corpus  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["row", "fixed"])
    ap.add_argument("--model")
    args = ap.parse_args()
    overrides = {}
    if args.strategy:
        overrides["chunking.strategy"] = args.strategy
    if args.model:
        overrides["embedding.model"] = args.model
    cfg = load_config(overrides=overrides)

    t0 = time.time()
    docs = load_corpus(cfg["paths"]["reports_dir"], cfg["paths"]["reference_dir"],
                       cache_file=str(Path(cfg["paths"]["chroma_dir"]) / "ingest_cache.json"))
    print(f"Loaded {len(docs)} documents ({sum(len(d['pages']) for d in docs)} pages) in {time.time() - t0:.1f}s")

    chunks = chunk_corpus(docs, cfg)
    kinds = Counter(c["metadata"]["doc_type"] for c in chunks)
    print(f"Chunks: {len(chunks)} {dict(kinds)}  [strategy={cfg['chunking']['strategy']}]")

    t1 = time.time()
    n = build_index(chunks, cfg)
    print(f"Indexed {n} chunks into '{collection_name(cfg)}' at {cfg['paths']['chroma_dir']} in {time.time() - t1:.1f}s")


if __name__ == "__main__":
    main()
