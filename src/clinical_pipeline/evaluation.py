"""Retrieval evaluation: BM25-only vs ClinicalBERT-only vs hybrid, plus the deployed pipeline.

Metrics
- Top-1 / Recall@3 / MRR on in-scope questions (is an acceptable topic ranked first / in top 3).
- Relevance AUROC: how well each method's best score separates in-scope from out-of-scope
  questions (0.5 = coin flip, 1.0 = perfect), independent of any threshold.
- End-to-end accuracy of the deployed retriever (gate + topic selection), exactly as served.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .clinical_bert import ClinicalBertRetriever, Encoder, PassageIndex, _minmax
from .knowledge_base import load_passages


@dataclass(frozen=True)
class Item:
    question: str
    expected: list[str]
    type: str

    @property
    def in_scope(self) -> bool:
        return bool(self.expected)


def load_items(path: Path) -> list[Item]:
    with path.open(encoding="utf-8") as fh:
        return [Item(**json.loads(line)) for line in fh if line.strip()]


# ---------- metrics (pure functions, unit-tested) ----------

def first_hit_rank(ranked_ids: list[str], expected: list[str]) -> int | None:
    for rank, pid in enumerate(ranked_ids, start=1):
        if pid in expected:
            return rank
    return None


def ranking_metrics(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks)
    return {
        "top1": sum(r == 1 for r in ranks) / n,
        "recall@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(random positive scores higher than random negative); ties count half."""
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


# ---------- evaluation ----------

METHODS = {"BM25 only": 0.0, "ClinicalBERT only": 1.0}


def evaluate(items: list[Item], retriever: ClinicalBertRetriever, alphas: list[float]) -> dict:
    index = retriever.index
    ids = [p.id for p in index.passages]
    q_vecs = retriever.encoder.encode([it.question for it in items])
    bert = q_vecs @ index.embeddings.T  # (n_questions, n_passages)
    lex = np.array([retriever.bm25.scores(it.question) for it in items], dtype=np.float32)

    def ranked(i: int, alpha: float) -> list[str]:
        score = alpha * _minmax(bert[i]) + (1 - alpha) * _minmax(lex[i])
        return [ids[j] for j in np.argsort(-score, kind="stable")]

    in_idx = [i for i, it in enumerate(items) if it.in_scope]
    out_idx = [i for i, it in enumerate(items) if not it.in_scope]

    methods = dict(METHODS)
    methods[f"Hybrid (alpha={retriever.alpha})"] = retriever.alpha
    ranking = {}
    for name, alpha in methods.items():
        ranks = [first_hit_rank(ranked(i, alpha), items[i].expected) for i in in_idx]
        row = ranking_metrics(ranks)
        for style in ("keyword", "paraphrase"):
            sub = [r for r, i in zip(ranks, in_idx) if items[i].type == style]
            row[f"top1_{style}"] = ranking_metrics(sub)["top1"] if sub else float("nan")
        ranking[name] = row

    sweep = {
        a: ranking_metrics([first_hit_rank(ranked(i, a), items[i].expected) for i in in_idx])["top1"]
        for a in alphas
    }

    gate = {
        "BM25 max score": auroc([lex[i].max() for i in in_idx], [lex[i].max() for i in out_idx]),
        "ClinicalBERT max cosine": auroc([bert[i].max() for i in in_idx], [bert[i].max() for i in out_idx]),
    }

    # End-to-end with the deployed retriever (gate + keep-ratio + topic order)
    e2e_rows = []
    for it in items:
        res = retriever.analyze(it.question)
        top = res.evidence[0].passage_id if res.evidence else None
        ok = (res.is_relevant and top in it.expected) if it.in_scope else (not res.is_relevant)
        e2e_rows.append({"question": it.question, "type": it.type, "expected": it.expected,
                         "got": top if res.is_relevant else "(rejected)", "correct": ok})

    def acc(rows):
        return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")

    e2e = {
        "overall": acc(e2e_rows),
        "in_scope": acc([r for r in e2e_rows if r["expected"]]),
        "non_medical_rejected": acc([r for r in e2e_rows if r["type"] == "non-medical"]),
        "uncovered_medical_rejected": acc([r for r in e2e_rows if r["type"] == "uncovered-medical"]),
    }
    return {
        "n_in_scope": len(in_idx), "n_out_of_scope": len(out_idx),
        "n_keyword": sum(items[i].type == "keyword" for i in in_idx),
        "n_paraphrase": sum(items[i].type == "paraphrase" for i in in_idx),
        "ranking": ranking, "alpha_sweep": sweep, "gate_auroc": gate,
        "end_to_end": e2e, "failures": [r for r in e2e_rows if not r["correct"]],
        "config": {"alpha": retriever.alpha, "min_lexical_score": retriever.min_lexical_score,
                   "passage_keep_ratio": retriever.passage_keep_ratio},
    }


def to_markdown(r: dict, model_name: str) -> str:
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        "# Retrieval evaluation results", "",
        f"Generated {date.today()} by `python scripts/evaluate.py` · encoder `{model_name}` · "
        f"{r['n_in_scope']} in-scope questions ({r['n_keyword']} keyword, {r['n_paraphrase']} paraphrase) "
        f"+ {r['n_out_of_scope']} out-of-scope.", "",
        "## 1. Topic ranking (in-scope questions)", "",
        "| Method | Top-1 | Recall@3 | MRR | Top-1 keyword | Top-1 paraphrase |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in r["ranking"].items():
        lines.append(f"| {name} | {pct(m['top1'])} | {pct(m['recall@3'])} | {m['mrr']:.3f} | "
                     f"{pct(m['top1_keyword'])} | {pct(m['top1_paraphrase'])} |")
    lines += ["", "## 2. Hybrid weight sweep (Top-1)", "",
              "| alpha (ClinicalBERT weight) | " + " | ".join(f"{a:.1f}" for a in r["alpha_sweep"]) + " |",
              "|---|" + "---|" * len(r["alpha_sweep"]),
              "| Top-1 | " + " | ".join(pct(v) for v in r["alpha_sweep"].values()) + " |",
              "", "## 3. Relevance detection (AUROC, in-scope vs out-of-scope)", "",
              "0.5 = no better than chance, 1.0 = perfect separation.", "",
              "| Signal | AUROC |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in r["gate_auroc"].items()]
    e = r["end_to_end"]
    c = r["config"]
    lines += ["", "## 4. Deployed pipeline, end to end", "",
              f"Config: alpha={c['alpha']}, MIN_LEXICAL_SCORE={c['min_lexical_score']}, "
              f"PASSAGE_KEEP_RATIO={c['passage_keep_ratio']}", "",
              "| Metric | Value |", "|---|---|",
              f"| Overall accuracy | {pct(e['overall'])} |",
              f"| In-scope: correct topic | {pct(e['in_scope'])} |",
              f"| Non-medical questions rejected | {pct(e['non_medical_rejected'])} |",
              f"| Uncovered medical questions rejected | {pct(e['uncovered_medical_rejected'])} |",
              "", "### Failures", "", "| Question | Type | Expected | Got |", "|---|---|---|---|"]
    for f in r["failures"]:
        lines.append(f"| {f['question']} | {f['type']} | {', '.join(f['expected']) or '(reject)'} | {f['got']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    from .clinical_bert import ClinicalBertEncoder
    from .config import get_settings

    s = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default="eval/questions.jsonl")
    parser.add_argument("--out", default="eval/results.md")
    args = parser.parse_args()

    items = load_items(Path(args.questions))
    encoder: Encoder = ClinicalBertEncoder(s.clinical_bert_model, device=s.device)
    # Build the index in memory from the current KB so results never come from a stale index.
    index = PassageIndex.build(load_passages(s.knowledge_base_dir), encoder, s.clinical_bert_model)
    retriever = ClinicalBertRetriever(encoder, index, s.top_k_passages, s.top_k_sentences,
                                      s.min_lexical_score, s.hybrid_alpha, s.passage_keep_ratio)
    results = evaluate(items, retriever, alphas=[round(a * 0.1, 1) for a in range(11)])
    md = to_markdown(results, s.clinical_bert_model)
    Path(args.out).write_text(md, encoding="utf-8")
    Path(args.out).with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
