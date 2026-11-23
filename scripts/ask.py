"""Ask a question about one patient's report.

  python scripts/ask.py --patient P0001 "Explain my HbA1c report"
  python scripts/ask.py --patient P0001 --backend none "Is my hemoglobin low?"   # retrieval only
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.pipeline import RAGPipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--patient", required=True)
    ap.add_argument("--backend", choices=["ollama", "hf", "none"])
    args = ap.parse_args()

    rag = RAGPipeline(overrides={"generation.backend": args.backend} if args.backend else None)
    out = rag.answer(args.patient, args.question)

    print("\n=== ANSWER ===\n" + (out["answer"] or "(generation skipped)"))
    print("\n=== SOURCES ===")
    for s in out["sources"]:
        print(f"[{s['n']}] {s['label']}\n    {s['text'][:160]}")
    print(f"\nTiming: {out['timing_s']}")


if __name__ == "__main__":
    main()
