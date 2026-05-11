from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_json, write_jsonl
from .metrics import compute_binary_pair_metrics, compute_mention_metrics, ranking_metrics_for_mentions
from .templates import render_candidate_text
from .thresholds import select_threshold


def parse_thresholds(raw: str) -> list[float]:
    values = [float(token.strip()) for token in raw.split(",") if token.strip()]
    return values or [0.5]


def logits_to_pair_predictions(rows: list[dict[str, Any]], logits: Any) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    logits_array = np.asarray(logits)
    probs = torch.softmax(torch.tensor(logits_array), dim=-1).numpy()
    enriched: list[dict[str, Any]] = []
    for index, (row, prob) in enumerate(zip(rows, probs)):
        updated = dict(row)
        updated["logit_0"] = float(logits_array[index][0])
        updated["logit_1"] = float(logits_array[index][1])
        updated["prob_0"] = float(prob[0])
        updated["prob_1"] = float(prob[1])
        updated["pred_label"] = int(prob[1] >= prob[0])
        enriched.append(updated)
    return enriched


class PairJsonlDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int, template_name: str = "full") -> None:
        import torch
        from torch.utils.data import Dataset

        class _Dataset(Dataset):
            def __len__(self_inner) -> int:
                return len(rows)

            def __getitem__(self_inner, index: int) -> dict[str, Any]:
                row = rows[index]
                text_a = str(row.get("text_a", row.get("raw_term", "")))
                encoded = tokenizer(
                    text_a,
                    render_candidate_text(row, template_name),
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                item = {key: value.squeeze(0) for key, value in encoded.items()}
                item["labels"] = torch.tensor(int(row["label"]), dtype=torch.long)
                return item

        self.dataset = _Dataset()


def compute_transformer_binary_metrics(eval_pred: Any) -> dict[str, float]:
    import numpy as np
    import torch

    logits, labels = eval_pred
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    rows = [{"label": int(label), "prob_1": float(prob)} for label, prob in zip(labels, probs)]
    return compute_binary_pair_metrics(rows, threshold=0.5)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a MacBERT-style cross-encoder on pair JSONL.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--dev-file", type=Path, required=True)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--template-name", default="full", choices=["full", "pt_only", "pt_llt"])
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--per-device-train-batch-size", type=int, default=32)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--metric-for-best-model", default="eval_f1")
    parser.add_argument("--mention-threshold-candidates", default="0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--threshold-optimize", default="example_f1", choices=["example_f1", "exact_set_match"])
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    return parser


def train_cross_encoder(args: argparse.Namespace) -> int:
    try:
        import accelerate  # noqa: F401
        import numpy as np
        import torch  # noqa: F401
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit("Training requires optional dependencies: numpy, torch, transformers, and accelerate.") from exc

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=2)

    train_rows = read_jsonl(args.train_file, args.max_train_samples)
    dev_rows = read_jsonl(args.dev_file, args.max_eval_samples)
    test_rows = read_jsonl(args.test_file, args.max_eval_samples) if args.test_file else []

    train_dataset = PairJsonlDataset(train_rows, tokenizer, args.max_length, args.template_name).dataset
    dev_dataset = PairJsonlDataset(dev_rows, tokenizer, args.max_length, args.template_name).dataset
    test_dataset = PairJsonlDataset(test_rows, tokenizer, args.max_length, args.template_name).dataset if test_rows else None

    try:
        training_args = TrainingArguments(
            output_dir=str(args.output_dir),
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=True,
            bf16=args.bf16,
            fp16=args.fp16,
            report_to=[],
            seed=args.seed,
        )
    except TypeError:
        training_args = TrainingArguments(
            output_dir=str(args.output_dir),
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=args.logging_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            save_total_limit=args.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=args.metric_for_best_model,
            greater_is_better=True,
            bf16=args.bf16,
            fp16=args.fp16,
            report_to=[],
            seed=args.seed,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding="longest"),
        compute_metrics=compute_transformer_binary_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    predictions_dir = args.output_dir / "predictions"
    metrics_dir = args.output_dir / "metrics"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    dev_output = trainer.predict(dev_dataset, metric_key_prefix="dev_pair")
    dev_predictions = logits_to_pair_predictions(dev_rows, np.asarray(dev_output.predictions))
    write_jsonl(dev_predictions, predictions_dir / "dev_pair_predictions.jsonl")
    thresholds = parse_thresholds(args.mention_threshold_candidates)
    selection = select_threshold(dev_predictions, thresholds, optimize=args.threshold_optimize)
    dev_metrics = dict(selection.metrics, selected_threshold=selection.threshold, selection_rule=selection.selection_rule)
    dev_retrieval = ranking_metrics_for_mentions(dev_predictions, ks=(1, 3, 5, 10))
    write_json(dev_metrics, metrics_dir / "dev_mention_metrics.json")
    write_json(dev_retrieval, metrics_dir / "dev_retrieval_metrics.json")

    summary: dict[str, Any] = {"dev_mention": dev_metrics, "dev_retrieval": dev_retrieval}
    if test_dataset is not None:
        test_output = trainer.predict(test_dataset, metric_key_prefix="test_pair")
        test_predictions = logits_to_pair_predictions(test_rows, np.asarray(test_output.predictions))
        write_jsonl(test_predictions, predictions_dir / "test_pair_predictions.jsonl")
        test_metrics = compute_mention_metrics(test_predictions, threshold=selection.threshold)
        test_metrics["selected_threshold"] = selection.threshold
        test_retrieval = ranking_metrics_for_mentions(test_predictions, ks=(1, 3, 5, 10))
        write_json(test_metrics, metrics_dir / "test_mention_metrics.json")
        write_json(test_retrieval, metrics_dir / "test_retrieval_metrics.json")
        summary.update({"test_mention": test_metrics, "test_retrieval": test_retrieval})

    write_json(summary, args.output_dir / "metrics_summary.json")
    return 0


def main(argv: list[str] | None = None) -> int:
    return train_cross_encoder(build_arg_parser().parse_args(argv))
