#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-"$ROOT/outputs/smoke"}"

rm -rf "$OUT"
mkdir -p "$OUT"

python -m mention2meddra validate dictionary "$ROOT/examples/synthetic_meddra.csv" >/dev/null
python -m mention2meddra validate pairs "$ROOT/examples/synthetic_pairs.jsonl" >/dev/null
python -m mention2meddra retrieve \
  --dictionary "$ROOT/examples/synthetic_meddra.csv" \
  --pairs "$ROOT/examples/synthetic_pairs.jsonl" \
  --output-dir "$OUT/retrieval" \
  --template-name full \
  --top-k 5 >/dev/null

python -m mention2meddra evaluate \
  --predictions "$ROOT/examples/synthetic_predictions.jsonl" \
  --metrics "$OUT/predictions/metrics.json" \
  --threshold 0.3 \
  --thresholds 0.1,0.2,0.3,0.4,0.5 >/dev/null

echo "Smoke workflow complete: $OUT"
