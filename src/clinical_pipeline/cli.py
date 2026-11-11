"""Interactive terminal chat."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import TONES
from .pipeline import ClinicalQAPipeline
from .safety import DISCLAIMER


def main() -> None:
    parser = argparse.ArgumentParser(description="Local clinical Q&A chat")
    parser.add_argument("--tone", choices=TONES, default=None)
    parser.add_argument("--show-evidence", action="store_true", help="print ClinicalBERT evidence")
    parser.add_argument("question", nargs="*", help="ask one question and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    print("Loading ClinicalBERT and index ...", file=sys.stderr)
    pipeline = ClinicalQAPipeline.from_settings()
    tone = args.tone

    def run(q: str) -> None:
        meta, tokens = pipeline.ask_stream(q, tone)
        for piece in tokens:
            print(piece, end="", flush=True)
        print(f"\n\n_{DISCLAIMER}_")
        if args.show_evidence:
            print(f"\n[relevant={meta.is_relevant}]")
            for e in meta.evidence:
                print(f"  ({e.score:.3f}) [{e.title}] {e.sentence}")

    if args.question:
        run(" ".join(args.question))
        return

    print(f"Ask a medical question. Commands: /tone <{'|'.join(TONES)}>, /quit\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in {"/quit", "/exit"}:
            break
        if q.startswith("/tone"):
            parts = q.split()
            if len(parts) == 2 and parts[1] in TONES:
                tone = parts[1]
                print(f"tone set to {tone}")
            else:
                print(f"usage: /tone <{'|'.join(TONES)}>")
            continue
        print("assistant> ", end="")
        run(q)
        print()


if __name__ == "__main__":
    main()
