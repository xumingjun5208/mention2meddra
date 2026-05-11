# mention2meddra

Public code release for the Chinese ADR mention-to-MedDRA workflow described in the Journal of Biomedical Informatics submission.

The repository includes reusable code, tests, configs, realistic synthetic examples, and audit scripts. It intentionally excludes real adverse-event records, the expert corpus, licensed dictionary files, and model weights.

## Install

```bash
python -m pip install -e .
```

For training a transformer cross-encoder, install the optional training dependencies:

```bash
python -m pip install -e ".[train]"
```

## Schemas

Dictionary CSV:

```text
pt_code,pt_name,llt_name,hlt_name,hlgt_name,soc_name
```

Pair JSONL fields:

```text
mention_id, text_a/raw_term, candidate_pt_code, candidate_pt_name,
candidate_llt_names, candidate_hlt_name, candidate_hlgt_name,
candidate_soc_name, gold_pt_codes, gold_soc_codes, label
```

Prediction JSONL adds:

```text
prob_1, pred_label
```

Metrics JSON includes pair metrics, exact set match, example/micro metrics, top-1 accuracy, `recall_at_k_all_gold`, and `recall_at_k_any_hit`.

## Data and Access Boundaries

This public release does not contain real adverse-event records, the expert corpus, licensed MedDRA dictionary files, trained model weights, or downstream signal-detection datasets. The files in `examples/` are realistic synthetic fixtures for schema validation, retrieval smoke tests, and metric calculation only. They are not source study data and should not be used as a replacement for licensed dictionary resources or restricted pharmacovigilance datasets.

The code can reproduce the package-level validation, candidate retrieval, prediction evaluation, repository audit, and smoke workflow. Full study reproduction requires the restricted source data, licensed terminology resources, and trained model artifacts described in the manuscript.

## Model Weights

The publicly released fine-tuned MacBERT reranker weights are available on Hugging Face:

https://huggingface.co/xumingjun/mention2meddra-macbert-base

The model repository contains inference artifacts only. It does not include real adverse-event records, expert annotation files, licensed MedDRA dictionary files, or downstream signal-detection datasets.

## Smoke Workflow

```bash
bash scripts/run_smoke_workflow.sh
```

This validates synthetic schemas, runs exact and character-level BM25 retrieval, and evaluates synthetic predictions on CPU.

## Main Commands

```bash
python -m mention2meddra validate dictionary examples/synthetic_meddra.csv
python -m mention2meddra validate pairs examples/synthetic_pairs.jsonl
python -m mention2meddra retrieve --dictionary examples/synthetic_meddra.csv --pairs examples/synthetic_pairs.jsonl --output-dir outputs/retrieval
python -m mention2meddra evaluate --predictions examples/synthetic_predictions.jsonl --metrics outputs/metrics.json --threshold 0.3
python scripts/audit_repository.py
```

Training command template:

```bash
python scripts/train_cross_encoder.py \
  --model-name-or-path hfl/chinese-macbert-base \
  --train-file /path/to/train.jsonl \
  --dev-file /path/to/dev.jsonl \
  --test-file /path/to/test.jsonl \
  --output-dir outputs/formal_v1 \
  --max-length 64 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 64 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --num-train-epochs 5 \
  --warmup-ratio 0.06 \
  --mention-threshold-candidates 0.1,0.2,0.3,0.4,0.5 \
  --threshold-optimize exact_set_match \
  --template-name full \
  --seed 42
```

## Verification

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python scripts/audit_repository.py
bash scripts/run_smoke_workflow.sh
```
