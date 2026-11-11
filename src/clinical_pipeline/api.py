"""FastAPI server: `uvicorn clinical_pipeline.api:app --app-dir src`."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from .pipeline import ClinicalQAPipeline

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state["pipeline"] = ClinicalQAPipeline.from_settings()
    yield
    _state.clear()


app = FastAPI(title="Local Clinical Q&A", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    tone: Literal["empathetic", "reassuring", "professional", "simple"] | None = None


class EvidenceOut(BaseModel):
    sentence: str
    score: float
    title: str
    source: str


class AskResponse(BaseModel):
    answer: str
    tone: str
    tone_applied: bool
    is_relevant: bool
    red_flags: list[str]
    evidence: list[EvidenceOut]
    disclaimer: str
    warnings: list[str]


@app.get("/health")
def health() -> dict:
    pipeline: ClinicalQAPipeline = _state["pipeline"]
    return {"clinical_bert": "loaded", "tone_llm_ready": pipeline.tone_llm.health()}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    pipeline: ClinicalQAPipeline = _state["pipeline"]
    try:
        ans = await run_in_threadpool(pipeline.ask, req.question, req.tone)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AskResponse(
        answer=ans.answer,
        tone=ans.tone,
        tone_applied=ans.tone_applied,
        is_relevant=ans.is_relevant,
        red_flags=ans.red_flags,
        evidence=[EvidenceOut(sentence=e.sentence, score=e.score, title=e.title, source=e.source) for e in ans.evidence],
        disclaimer=ans.disclaimer,
        warnings=ans.warnings,
    )
