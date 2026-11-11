# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models/hf \
    KNOWLEDGE_BASE_DIR=/app/data/knowledge_base \
    INDEX_DIR=/app/data/index

ARG CLINICAL_BERT_MODEL=emilyalsentzer/Bio_ClinicalBERT
ENV CLINICAL_BERT_MODEL=${CLINICAL_BERT_MODEL}

WORKDIR /app

# 1) Dependencies (CPU-only torch keeps the image ~1.5 GB smaller than the CUDA build)
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && grep -v '^pytest' requirements.txt > /tmp/req.txt \
 && pip install -r /tmp/req.txt

# 2) Bake ClinicalBERT into the image so the container runs fully offline
RUN python -c "import os; from transformers import AutoModel, AutoTokenizer; \
m=os.environ['CLINICAL_BERT_MODEL']; AutoTokenizer.from_pretrained(m); AutoModel.from_pretrained(m)"

# 3) Application code
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY data/knowledge_base ./data/knowledge_base
COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/data/index \
 && chown -R app:app /app/data \
 && chmod +x /usr/local/bin/entrypoint.sh
USER app

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=8)" || exit 1

ENTRYPOINT ["entrypoint.sh"]
CMD ["uvicorn", "clinical_pipeline.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
