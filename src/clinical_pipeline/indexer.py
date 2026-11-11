"""Build the ClinicalBERT passage index from the knowledge base."""

from __future__ import annotations

import argparse
import logging
import time

from .clinical_bert import ClinicalBertEncoder, PassageIndex
from .config import get_settings
from .knowledge_base import load_passages


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kb-dir", default=str(s.knowledge_base_dir))
    parser.add_argument("--index-dir", default=str(s.index_dir))
    args = parser.parse_args()

    from pathlib import Path

    passages = load_passages(Path(args.kb_dir))
    logging.info("Loaded %d passages; encoding with %s ...", len(passages), s.clinical_bert_model)
    t0 = time.time()
    encoder = ClinicalBertEncoder(s.clinical_bert_model, device=s.device)
    index = PassageIndex.build(passages, encoder, s.clinical_bert_model)
    index.save(Path(args.index_dir))
    logging.info("Index saved to %s in %.1fs", args.index_dir, time.time() - t0)


if __name__ == "__main__":
    main()
