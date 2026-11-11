"""Stage 1 — ClinicalBERT.

Bio_ClinicalBERT is an *encoder*: it understands clinical language but cannot
generate text. We use it to embed the user's question and the knowledge base,
retrieve the most relevant passages, then pick the most relevant sentences as
"clinical evidence" for the tone LLM to phrase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .knowledge_base import Passage, split_sentences
from .lexical import BM25, tokenize


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:  # (n, dim), L2-normalised
        ...


class ClinicalBertEncoder:
    """Mean-pooled, L2-normalised sentence embeddings from a BERT-family model."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 16) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        torch = self._torch
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                chunks.append(pooled.cpu().numpy().astype(np.float32))
        if not chunks:
            return np.zeros((0, 0), dtype=np.float32)
        return np.vstack(chunks)


@dataclass(frozen=True)
class Evidence:
    sentence: str
    score: float
    passage_id: str
    title: str
    source: str


@dataclass(frozen=True)
class ClinicalResult:
    question: str
    evidence: list[Evidence]
    top_lexical_score: float
    is_relevant: bool


class PassageIndex:
    """Tiny on-disk vector index (numpy) — enough for thousands of passages."""

    EMB_FILE = "embeddings.npy"
    META_FILE = "passages.json"

    def __init__(self, passages: list[Passage], embeddings: np.ndarray, model_name: str) -> None:
        if len(passages) != embeddings.shape[0]:
            raise ValueError("passages and embeddings length mismatch")
        self.passages = passages
        self.embeddings = embeddings
        self.model_name = model_name

    @classmethod
    def build(cls, passages: list[Passage], encoder: Encoder, model_name: str) -> "PassageIndex":
        texts = [f"{p.title}. {p.text}" for p in passages]
        return cls(passages, encoder.encode(texts), model_name)

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / self.EMB_FILE, self.embeddings)
        meta = {"model_name": self.model_name, "passages": [p.to_dict() for p in self.passages]}
        (index_dir / self.META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: Path, expected_model: str) -> "PassageIndex":
        meta_path, emb_path = index_dir / cls.META_FILE, index_dir / cls.EMB_FILE
        if not meta_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                f"No index in {index_dir}. Run `python scripts/build_index.py` first."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["model_name"] != expected_model:
            raise ValueError(
                f"Index was built with {meta['model_name']!r} but CLINICAL_BERT_MODEL is "
                f"{expected_model!r}. Rebuild the index."
            )
        passages = [Passage(**p) for p in meta["passages"]]
        return cls(passages, np.load(emb_path), meta["model_name"])

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[Passage, float]]:
        scores = self.embeddings @ query_vec
        order = np.argsort(-scores)[:top_k]
        return [(self.passages[i], float(scores[i])) for i in order]


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    return np.zeros_like(x) if hi - lo < 1e-9 else (x - lo) / (hi - lo)


class ClinicalBertRetriever:
    """Hybrid retrieval: BM25 (relevance gate + recall) + ClinicalBERT (semantic re-rank).

    score = alpha * ClinicalBERT_cosine_norm + (1 - alpha) * BM25_norm
    A question is "relevant" only if some passage has BM25 >= min_lexical_score,
    i.e. the knowledge base genuinely discusses its key terms.
    """

    def __init__(
        self,
        encoder: Encoder,
        index: PassageIndex,
        top_k_passages: int = 3,
        top_k_sentences: int = 5,
        min_lexical_score: float = 1.0,
        alpha: float = 0.3,
        passage_keep_ratio: float = 0.7,
    ) -> None:
        self.encoder = encoder
        self.index = index
        self.top_k_passages = top_k_passages
        self.top_k_sentences = top_k_sentences
        self.min_lexical_score = min_lexical_score
        self.alpha = alpha
        self.passage_keep_ratio = passage_keep_ratio
        self.bm25 = BM25([f"{p.title}. {p.text}" for p in index.passages])

    def _hybrid(self, bert: np.ndarray, lexical: np.ndarray) -> np.ndarray:
        return self.alpha * _minmax(bert) + (1 - self.alpha) * _minmax(lexical)

    def analyze(self, question: str) -> ClinicalResult:
        q_vec = self.encoder.encode([question])[0]
        lexical = np.asarray(self.bm25.scores(question), dtype=np.float32)
        bert = self.index.embeddings @ q_vec
        hybrid = self._hybrid(bert, lexical)

        best_lexical = float(lexical.max()) if lexical.size else 0.0
        is_relevant = best_lexical >= self.min_lexical_score
        if not is_relevant:
            return ClinicalResult(question, [], best_lexical, False)

        # keep only passages that match lexically and are close to the best one
        order = [i for i in np.argsort(-hybrid) if lexical[i] > 0][: self.top_k_passages]
        top = hybrid[order[0]]
        kept = [i for i in order if hybrid[i] >= self.passage_keep_ratio * top]

        q_terms = set(tokenize(question))
        evidence: list[Evidence] = []
        for i in kept:  # passage order first, then best sentences within each passage
            passage = self.index.passages[i]
            sentences = split_sentences(passage.text)
            s_bert = self.encoder.encode(sentences) @ q_vec
            s_lex = np.array([len(q_terms & set(tokenize(s))) for s in sentences], dtype=np.float32)
            s_score = self._hybrid(s_bert, s_lex)
            for j in np.argsort(-s_score):
                evidence.append(
                    Evidence(sentences[j], float(s_score[j]), passage.id, passage.title, passage.source)
                )
        return ClinicalResult(question, evidence[: self.top_k_sentences], best_lexical, True)
