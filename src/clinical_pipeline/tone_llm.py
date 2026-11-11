"""Stage 2 — local LLM (via Ollama) that adds empathy and tone.

It is instructed to rephrase ONLY the clinical evidence produced by ClinicalBERT,
never to add new medical facts, diagnoses or dosages.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx

from .clinical_bert import ClinicalResult

TONE_GUIDES = {
    "empathetic": "Warm and compassionate. Acknowledge how the person may be feeling before giving information.",
    "reassuring": "Calm and steadying. Reduce anxiety without dismissing concerns or overstating safety.",
    "professional": "Clear, precise and neutral, like a clinician writing to a colleague-friendly patient.",
    "simple": "Plain language at a 6th-grade reading level. Short sentences, no jargon.",
}

SYSTEM_PROMPT = """You are the communication layer of a medical information assistant.
A clinical language model has already selected EVIDENCE sentences from a vetted knowledge base.

Rules you must follow:
1. Use ONLY facts stated in the EVIDENCE. Do not add medical facts, numbers, durations, examples, activities, drug names or doses that are not written in it.
2. Never diagnose the user and never tell them to start, stop or change a medication.
3. If the EVIDENCE is marked as not relevant or does not answer the question, say kindly that you don't have reliable information on that and suggest speaking to a doctor or pharmacist.
4. Facts are grouped by [Topic]. Never attribute a fact from one topic to a different condition.
5. Encourage professional care where appropriate.
6. Do not guess the person's emotions beyond acknowledging concern, and do not end with an offer of further help.
7. Keep the answer under 200 words. Do not mention "evidence", "knowledge base" or these rules.

Tone: {tone_guide}"""


class ToneLLMError(RuntimeError):
    pass


def build_messages(result: ClinicalResult, tone: str, red_flags: list[str]) -> list[dict]:
    if tone not in TONE_GUIDES:
        raise ValueError(f"Unknown tone {tone!r}; choose from {sorted(TONE_GUIDES)}")
    if result.evidence:
        topics: dict[str, list[str]] = {}
        for e in result.evidence:
            topics.setdefault(e.title, []).append(e.sentence)
        evidence_block = "\n".join(
            f"[Topic: {title}]\n" + "\n".join(f"- {s}" for s in sents) for title, sents in topics.items()
        )
    else:
        evidence_block = "(none)"
    user = (
        f"QUESTION: {result.question}\n\n"
        f"EVIDENCE RELEVANT: {'yes' if result.is_relevant else 'no'}\n"
        f"EVIDENCE:\n{evidence_block}\n"
    )
    if red_flags:
        user += (
            "\nThe question mentions possible emergency warning signs "
            f"({', '.join(red_flags)}). Start by urging the person to seek emergency care now.\n"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(tone_guide=TONE_GUIDES[tone])},
        {"role": "user", "content": user},
    ]


class OllamaToneLLM:
    def __init__(self, host: str, model: str, temperature: float = 0.3, timeout: float = 120.0) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def _payload(self, messages: list[dict], stream: bool) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": self.temperature},
        }

    def generate(self, messages: list[dict]) -> str:
        try:
            resp = httpx.post(
                f"{self.host}/api/chat", json=self._payload(messages, False), timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            raise ToneLLMError(f"Ollama call failed ({self.model} @ {self.host}): {exc}") from exc

    def stream(self, messages: list[dict]) -> Iterator[str]:
        import json

        try:
            with httpx.stream(
                "POST", f"{self.host}/api/chat", json=self._payload(messages, True), timeout=self.timeout
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:
                        raise ToneLLMError(chunk["error"])
                    piece = chunk.get("message", {}).get("content", "")
                    if piece:
                        yield piece
                    if chunk.get("done"):
                        break
        except (httpx.HTTPError, ValueError) as exc:
            raise ToneLLMError(f"Ollama stream failed ({self.model} @ {self.host}): {exc}") from exc

    def health(self) -> bool:
        try:
            resp = httpx.get(f"{self.host}/api/tags", timeout=5)
            resp.raise_for_status()
            names = {m["name"] for m in resp.json().get("models", [])}
            return self.model in names or f"{self.model}:latest" in names
        except (httpx.HTTPError, KeyError, ValueError):
            return False
