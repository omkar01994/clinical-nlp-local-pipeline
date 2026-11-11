"""Minimal Okapi BM25 — the lexical half of hybrid retrieval.

Raw Bio_ClinicalBERT is a masked-LM, not a sentence-similarity model, so its cosine
scores sit in a narrow band (~0.8-0.9) for related *and* unrelated text. BM25 gives
a reliable "does the knowledge base actually talk about this?" signal.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    """a about after all also am an and any are as at be been before being but by can
    could did do does doing for from get got had has have having he her his how i if in
    into is it its just me more most my no not of on or our out over should so some such
    than that the their them then there these they this to too up very was we were what
    when where which while who why will with would you your yours im ive dont cant
    really much many thing things feel feeling know want need please help tell""".split()
)


def tokenize(text: str) -> list[str]:
    tokens = []
    for tok in _TOKEN.findall(text.lower().replace("'", "")):
        if tok in STOPWORDS or len(tok) < 2:
            continue
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]  # crude plural folding: symptoms -> symptom
        tokens.append(tok)
    return tokens


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens = [Counter(tokenize(d)) for d in docs]
        self.doc_len = [sum(c.values()) for c in self.doc_tokens]
        self.avgdl = (sum(self.doc_len) / len(docs)) if docs else 0.0
        df: Counter[str] = Counter()
        for counts in self.doc_tokens:
            df.update(counts.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q_terms = set(tokenize(query))
        out = []
        for counts, dl in zip(self.doc_tokens, self.doc_len):
            s = 0.0
            for t in q_terms:
                tf = counts.get(t, 0)
                if tf:
                    denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                    s += self.idf[t] * tf * (self.k1 + 1) / denom
            out.append(s)
        return out
