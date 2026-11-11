"""Download all models once so the pipeline runs fully offline afterwards."""

import subprocess
import sys

sys.path.insert(0, "src")
from clinical_pipeline.config import get_settings  # noqa: E402

s = get_settings()
print(f"[1/2] Downloading ClinicalBERT: {s.clinical_bert_model}")
from transformers import AutoModel, AutoTokenizer  # noqa: E402

AutoTokenizer.from_pretrained(s.clinical_bert_model)
AutoModel.from_pretrained(s.clinical_bert_model)

print(f"[2/2] Pulling tone LLM via Ollama: {s.tone_llm_model}")
try:
    subprocess.run(["ollama", "pull", s.tone_llm_model], check=True)
except FileNotFoundError:
    sys.exit("ollama not found — install it from https://ollama.com and re-run.")
print("Done. Set HF_HUB_OFFLINE=1 to guarantee no network access at runtime.")
