"""Settings loaded from environment variables / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TONES = ("empathetic", "reassuring", "professional", "simple")


def _env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    # allow inline comments in .env values: "cpu   # cpu | cuda"
    return value.split("#", 1)[0].strip() or default


@dataclass(frozen=True)
class Settings:
    clinical_bert_model: str = _env("CLINICAL_BERT_MODEL", "emilyalsentzer/Bio_ClinicalBERT")
    device: str = _env("DEVICE", "cpu")
    top_k_passages: int = int(_env("TOP_K_PASSAGES", "3"))
    top_k_sentences: int = int(_env("TOP_K_SENTENCES", "5"))
    min_lexical_score: float = float(_env("MIN_LEXICAL_SCORE", "1.0"))
    hybrid_alpha: float = float(_env("HYBRID_ALPHA", "0.3"))
    passage_keep_ratio: float = float(_env("PASSAGE_KEEP_RATIO", "0.7"))

    ollama_host: str = _env("OLLAMA_HOST", "http://localhost:11434")
    tone_llm_model: str = _env("TONE_LLM_MODEL", "llama3.2:3b")
    default_tone: str = _env("DEFAULT_TONE", "empathetic")
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0.3"))
    llm_timeout_seconds: float = float(_env("LLM_TIMEOUT_SECONDS", "120"))

    knowledge_base_dir: Path = Path(_env("KNOWLEDGE_BASE_DIR", "./data/knowledge_base"))
    index_dir: Path = Path(_env("INDEX_DIR", "./data/index"))

    def __post_init__(self) -> None:
        if self.default_tone not in TONES:
            raise ValueError(f"DEFAULT_TONE must be one of {TONES}, got {self.default_tone!r}")


def get_settings() -> Settings:
    return Settings()
