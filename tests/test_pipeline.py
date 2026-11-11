"""Unit tests using a deterministic fake encoder and fake LLM (no model downloads)."""

import json

import numpy as np
import pytest

from clinical_pipeline.clinical_bert import ClinicalBertRetriever, PassageIndex
from clinical_pipeline.knowledge_base import Passage, load_passages, split_sentences
from clinical_pipeline.pipeline import NO_INFO_TEXT, ClinicalQAPipeline
from clinical_pipeline.safety import EMERGENCY_NOTICE, detect_red_flags
from clinical_pipeline.tone_llm import ToneLLMError, build_messages

VOCAB = ["fever", "temperature", "blood", "pressure", "salt", "asthma", "inhaler", "pizza"]


class BagOfWordsEncoder:
    def encode(self, texts):
        out = np.zeros((len(texts), len(VOCAB)), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, w in enumerate(VOCAB):
                out[i, j] = t.lower().count(w)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class FakeLLM:
    def __init__(self, fail=False):
        self.fail, self.messages = fail, None

    def generate(self, messages):
        self.messages = messages
        if self.fail:
            raise ToneLLMError("down")
        return "styled answer"

    def stream(self, messages):
        yield self.generate(messages)


PASSAGES = [
    Passage("bp", "Blood pressure", "High blood pressure is common. Reducing salt lowers blood pressure.", "t"),
    Passage("fever", "Fever", "A fever is a high temperature. Drink fluids when you have a fever.", "t"),
    Passage("asthma", "Asthma", "Asthma narrows airways. An inhaler relieves asthma symptoms.", "t"),
]


def make_pipeline(llm=None):
    enc = BagOfWordsEncoder()
    index = PassageIndex.build(PASSAGES, enc, "fake")
    retriever = ClinicalBertRetriever(enc, index, top_k_passages=2, top_k_sentences=2)
    return ClinicalQAPipeline(retriever, llm or FakeLLM(), "empathetic")


def test_retrieves_relevant_evidence():
    ans = make_pipeline().ask("How can I lower my blood pressure with less salt?")
    assert ans.is_relevant and ans.tone_applied
    assert ans.evidence[0].passage_id == "bp"
    assert ans.answer == "styled answer"


def test_irrelevant_question_skips_llm():
    llm = FakeLLM()
    ans = make_pipeline(llm).ask("best pizza topping?")
    assert not ans.is_relevant and not ans.tone_applied
    assert llm.messages is None
    assert ans.answer == NO_INFO_TEXT


def test_stream_irrelevant_skips_llm():
    llm = FakeLLM()
    meta, tokens = make_pipeline(llm).ask_stream("best pizza topping?")
    assert "".join(tokens) == NO_INFO_TEXT and llm.messages is None


def test_fallback_when_llm_down():
    ans = make_pipeline(FakeLLM(fail=True)).ask("what helps a fever temperature")
    assert not ans.tone_applied and ans.warnings
    assert "fever" in ans.answer.lower()


def test_emergency_notice_prepended():
    ans = make_pipeline().ask("I have chest pain and high blood pressure")
    assert ans.red_flags and ans.answer.startswith(EMERGENCY_NOTICE)


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        make_pipeline().ask("   ")


def test_prompt_contains_evidence_and_tone():
    llm = FakeLLM()
    make_pipeline(llm).ask("asthma inhaler?", tone="simple")
    system, user = llm.messages[0]["content"], llm.messages[1]["content"]
    assert "Plain language" in system
    assert "inhaler" in user.lower() and "EVIDENCE RELEVANT: yes" in user


def test_unknown_tone_rejected():
    with pytest.raises(ValueError):
        make_pipeline().ask("fever?", tone="angry")


def test_red_flags():
    assert detect_red_flags("sudden worst headache ever and slurred speech")
    assert detect_red_flags("I want to kill myself")
    assert detect_red_flags("mild runny nose") == []


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_index_roundtrip_and_model_mismatch(tmp_path):
    idx = PassageIndex.build(PASSAGES, BagOfWordsEncoder(), "fake")
    idx.save(tmp_path)
    assert len(PassageIndex.load(tmp_path, "fake").passages) == 3
    with pytest.raises(ValueError):
        PassageIndex.load(tmp_path, "other-model")


def test_kb_rejects_duplicate_ids(tmp_path):
    row = json.dumps({"id": "a", "title": "t", "text": "x"})
    (tmp_path / "kb.jsonl").write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError):
        load_passages(tmp_path)


def test_seed_kb_loads():
    assert len(load_passages(__import__("pathlib").Path("data/knowledge_base"))) >= 10


def test_evidence_only_from_matching_passages():
    ans = make_pipeline().ask("blood pressure salt")
    assert {e.passage_id for e in ans.evidence} == {"bp"}
    assert not make_pipeline().ask("capital of France").evidence


def test_tokenize_drops_stopwords_and_folds_plurals():
    from clinical_pipeline.lexical import tokenize
    assert tokenize("What are the symptoms of diabetes?") == ["symptom", "diabete"]
