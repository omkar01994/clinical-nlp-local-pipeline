"""Orchestrates: safety check -> ClinicalBERT evidence -> tone LLM -> safety wrap."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from .clinical_bert import ClinicalBertEncoder, ClinicalBertRetriever, ClinicalResult, Evidence, PassageIndex
from .config import Settings, get_settings
from .safety import DISCLAIMER, EMERGENCY_NOTICE, detect_red_flags
from .tone_llm import OllamaToneLLM, ToneLLMError, build_messages

log = logging.getLogger(__name__)


@dataclass
class Answer:
    question: str
    tone: str
    answer: str
    evidence: list[Evidence]
    is_relevant: bool
    red_flags: list[str]
    tone_applied: bool
    disclaimer: str = DISCLAIMER
    warnings: list[str] = field(default_factory=list)


NO_INFO_TEXT = (
    "I'm sorry, I don't have reliable medical information on that topic. "
    "For questions about your health, a doctor or pharmacist is the best person to ask."
)


def _fallback_text(result: ClinicalResult) -> str:
    if not result.is_relevant or not result.evidence:
        return NO_INFO_TEXT
    return "Here is what I found:\n" + "\n".join(f"- {e.sentence}" for e in result.evidence)


class ClinicalQAPipeline:
    def __init__(self, retriever: ClinicalBertRetriever, tone_llm: OllamaToneLLM, default_tone: str) -> None:
        self.retriever = retriever
        self.tone_llm = tone_llm
        self.default_tone = default_tone

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "ClinicalQAPipeline":
        s = settings or get_settings()
        encoder = ClinicalBertEncoder(s.clinical_bert_model, device=s.device)
        index = PassageIndex.load(s.index_dir, expected_model=s.clinical_bert_model)
        retriever = ClinicalBertRetriever(
            encoder, index, s.top_k_passages, s.top_k_sentences, s.min_lexical_score, s.hybrid_alpha,
            s.passage_keep_ratio,
        )
        llm = OllamaToneLLM(s.ollama_host, s.tone_llm_model, s.llm_temperature, s.llm_timeout_seconds)
        return cls(retriever, llm, s.default_tone)

    def _prepare(self, question: str, tone: str | None):
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty")
        tone = tone or self.default_tone
        red_flags = detect_red_flags(question)
        result = self.retriever.analyze(question)
        messages = build_messages(result, tone, red_flags)
        return question, tone, red_flags, result, messages

    def ask(self, question: str, tone: str | None = None) -> Answer:
        question, tone, red_flags, result, messages = self._prepare(question, tone)
        warnings: list[str] = []
        if not result.is_relevant and not red_flags:
            text = NO_INFO_TEXT  # deterministic: never let the LLM improvise without evidence
            return Answer(question, tone, text, [], False, red_flags, False, warnings=warnings)
        try:
            text, tone_applied = self.tone_llm.generate(messages), True
        except ToneLLMError as exc:
            log.warning("Tone LLM unavailable, returning raw evidence: %s", exc)
            text, tone_applied = _fallback_text(result), False
            warnings.append("Tone LLM unavailable; showing un-styled clinical evidence.")
        if red_flags:
            text = f"{EMERGENCY_NOTICE}\n\n{text}"
        return Answer(question, tone, text, result.evidence, result.is_relevant, red_flags, tone_applied, warnings=warnings)

    def ask_stream(self, question: str, tone: str | None = None) -> tuple[Answer, Iterator[str]]:
        """Returns metadata immediately plus a token iterator (used by the CLI)."""
        question, tone, red_flags, result, messages = self._prepare(question, tone)
        meta = Answer(question, tone, "", result.evidence, result.is_relevant, red_flags, True)

        def tokens() -> Iterator[str]:
            if red_flags:
                yield EMERGENCY_NOTICE + "\n\n"
            if not result.is_relevant and not red_flags:
                meta.tone_applied = False
                yield NO_INFO_TEXT
                return
            try:
                yield from self.tone_llm.stream(messages)
            except ToneLLMError as exc:
                log.warning("Tone LLM unavailable, returning raw evidence: %s", exc)
                meta.tone_applied = False
                meta.warnings.append("Tone LLM unavailable; showing un-styled clinical evidence.")
                yield "\n" + _fallback_text(result)

        return meta, tokens()
