#!/bin/sh
set -e
# Build the ClinicalBERT index on first start (or when REBUILD_INDEX=1, e.g. after editing the KB).
if [ "${REBUILD_INDEX:-0}" = "1" ] || [ ! -f "${INDEX_DIR}/passages.json" ]; then
  echo "[entrypoint] Building ClinicalBERT index from ${KNOWLEDGE_BASE_DIR} ..."
  python -m clinical_pipeline.indexer
fi
exec "$@"
