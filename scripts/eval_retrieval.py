"""Retrieval-only evaluation (no LLM): Recall@k and MRR against gold (test, page) labels.

A retrieved chunk is a hit if it is from the gold page of the patient's report and contains
the gold test. Retrieval is measured separately from answer quality, so a bad answer can be
blamed on the right component (retriever vs. generator).

  python scripts/eval_retrieval.py                                  # current config, with and without reranker
  python scripts/eval_retrieval.py --strategy fixed                 # needs: build_index.py --strategy fixed
  python scripts/eval_retrieval.py --limit 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.config import load_config  # noqa: E402
from rag.retrieve import Retriever  # noqa: E402

KS = (1, 3, 5, 10)


def is_hit(hit: dict, q: dict) -> bool:
    m = hit["metadata"]
    return (m.get("report_id") == q["report_id"] and m.get("page") == q["gold_page"]
            and q["gold_test"] in m.get("tests", "").split("|"))


def evaluate(retriever: Retriever, questions: list[dict], rerank: bool) -> dict:
    depth = retriever.r["candidate_k"]
    recall = {k: 0 for k in KS}
    rr_total, leaks, t0 = 0.0, 0, time.time()
    for q in questions:
        hits = retriever.retrieve(q["patient_id"], q["question"], final_k=depth, use_reranker=rerank)["patient"]
        leaks += sum(h["metadata"].get("patient_id") != q["patient_id"] for h in hits)
        rank = next((i + 1 for i, h in enumerate(hits) if is_hit(h, q)), None)
        if rank:
            rr_total += 1 / rank
            for k in KS:
                recall[k] += rank <= k
    n = len(questions)
    return {"reranker": rerank, "n": n, **{f"recall@{k}": round(recall[k] / n, 3) for k in KS},
            "mrr": round(rr_total / n, 3), "cross_patient_leaks": leaks,
            "ms_per_query": round(1000 * (time.time() - t0) / n, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["row", "fixed"])
    ap.add_argument("--model")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rerank-mode", choices=["replace", "rrf"])
    args = ap.parse_args()
    overrides = {"retrieval.reference_k": 0}
    if args.rerank_mode:
        overrides["reranker.mode"] = args.rerank_mode
    if args.strategy:
        overrides["chunking.strategy"] = args.strategy
    if args.model:
        overrides["embedding.model"] = args.model
    cfg = load_config(overrides={**overrides, "retrieval.use_reranker": True})

    questions = [json.loads(l) for l in open(cfg["paths"]["eval_file"])]
    if args.limit:
        questions = questions[: args.limit]
    retriever = Retriever(cfg)

    results = [evaluate(retriever, questions, rerank=False), evaluate(retriever, questions, rerank=True)]
    print(f"\nstrategy={cfg['chunking']['strategy']}  embedding={cfg['embedding']['model']}  "
          f"reranker={cfg['reranker']['model']} ({cfg['reranker'].get('mode', 'replace')})")
    cols = ["reranker"] + [f"recall@{k}" for k in KS] + ["mrr", "cross_patient_leaks", "ms_per_query"]
    print(" | ".join(cols))
    for r in results:
        print(" | ".join(str(r[c]) for c in cols))

    out = Path(cfg["paths"]["chroma_dir"]).parent / "results"
    out.mkdir(exist_ok=True)
    tag = (f"{cfg['chunking']['strategy']}_{cfg['embedding']['model'].split('/')[-1]}"
           f"_{cfg['reranker'].get('mode', 'replace')}")
    (out / f"retrieval_{tag}.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
