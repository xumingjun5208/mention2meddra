from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .audit import audit_repository
from .dictionary import candidate_to_pair_row, load_dictionary
from .io import iter_jsonl, read_jsonl, write_json, write_jsonl
from .metrics import compute_mention_metrics, ranking_metrics_for_mentions
from .retrieval import build_bm25_index, exact_match_rank, lexical_bm25_rank_indexed
from .schemas import validate_dictionary_csv, validate_pair_jsonl, validate_prediction_jsonl
from .thresholds import scan_thresholds


def _cmd_validate(args: argparse.Namespace) -> int:
    if args.kind == "dictionary":
        report = validate_dictionary_csv(args.path)
    elif args.kind == "pairs":
        report = validate_pair_jsonl(args.path)
    else:
        report = validate_prediction_jsonl(args.path)
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ok else 1


def _representative_mentions(rows: Iterable[dict], limit: int | None = None) -> list[dict]:
    if limit is not None and limit <= 0:
        return []
    grouped: dict[str, dict] = {}
    for row in rows:
        mention_id = str(row["mention_id"])
        if mention_id in grouped:
            continue
        grouped[mention_id] = row
        if limit is not None and len(grouped) >= limit:
            break
    return list(grouped.values())


def _ranked_rows_for_mode(mentions: list[dict], candidates: list[dict], mode: str, top_k: int) -> list[dict]:
    rows: list[dict] = []
    bm25 = build_bm25_index(candidates) if mode == "bm25" else None
    for mention in mentions:
        query = str(mention.get("text_a", mention.get("raw_term", "")))
        if mode == "exact":
            ranked = exact_match_rank(query, candidates)[:top_k]
        else:
            ranked = lexical_bm25_rank_indexed(query, bm25, candidates, top_k=top_k)  # type: ignore[arg-type]
        rows.extend(candidate_to_pair_row(mention, candidate, float(candidate.get("prob_1", 0.0))) for candidate in ranked)
    return rows


def _cmd_retrieve(args: argparse.Namespace) -> int:
    candidates = load_dictionary(args.dictionary, template_name=args.template_name)
    mentions = _representative_mentions(iter_jsonl(args.pairs), args.max_mentions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for mode in ("exact", "bm25"):
        rows = _ranked_rows_for_mode(mentions, candidates, mode, args.top_k)
        mode_dir = args.output_dir / mode
        write_jsonl(rows, mode_dir / "ranked.jsonl")
        summary[mode] = {"retrieval": ranking_metrics_for_mentions(rows, ks=tuple(args.ks))}
        write_json(summary[mode], mode_dir / "metrics_summary.json")
    write_json(summary, args.output_dir / "lexical_retrieval_summary.json")
    print(f"wrote retrieval outputs to {args.output_dir}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    rows = read_jsonl(args.predictions)
    metrics = compute_mention_metrics(rows, threshold=args.threshold, ks=tuple(args.ks))
    if args.thresholds:
        thresholds = [float(token.strip()) for token in args.thresholds.split(",") if token.strip()]
        metrics["threshold_scan"] = scan_thresholds(rows, thresholds, optimize=args.threshold_optimize)
    write_json(metrics, args.metrics)
    print(f"wrote metrics to {args.metrics}")
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    report = audit_repository(args.root)
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ok else 1


def _cmd_train(args: argparse.Namespace) -> int:
    from .train import train_cross_encoder

    return train_cross_encoder(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mention2meddra", description="Chinese ADR mention-to-MedDRA release utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a public schema file")
    validate.add_argument("kind", choices=["dictionary", "pairs", "predictions"])
    validate.add_argument("path", type=Path)
    validate.set_defaults(func=_cmd_validate)

    retrieve = sub.add_parser("retrieve", help="run exact-match and character-BM25 retrieval")
    retrieve.add_argument("--dictionary", type=Path, required=True)
    retrieve.add_argument("--pairs", type=Path, required=True)
    retrieve.add_argument("--output-dir", type=Path, required=True)
    retrieve.add_argument("--template-name", default="full", choices=["full", "pt_only", "pt_llt"])
    retrieve.add_argument("--top-k", type=int, default=100)
    retrieve.add_argument("--max-mentions", type=int)
    retrieve.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    retrieve.set_defaults(func=_cmd_retrieve)

    evaluate = sub.add_parser("evaluate", help="compute pair and mention-level metrics from prediction JSONL")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--metrics", type=Path, required=True)
    evaluate.add_argument("--threshold", type=float, default=0.3)
    evaluate.add_argument("--thresholds", default="")
    evaluate.add_argument("--threshold-optimize", default="example_f1", choices=["example_f1", "exact_set_match"])
    evaluate.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    evaluate.set_defaults(func=_cmd_evaluate)

    train = sub.add_parser("train", help="train a transformer cross-encoder")
    train.add_argument("--model-name-or-path", required=True)
    train.add_argument("--train-file", type=Path, required=True)
    train.add_argument("--dev-file", type=Path, required=True)
    train.add_argument("--test-file", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--template-name", default="full", choices=["full", "pt_only", "pt_llt"])
    train.add_argument("--max-length", type=int, default=64)
    train.add_argument("--per-device-train-batch-size", type=int, default=32)
    train.add_argument("--per-device-eval-batch-size", type=int, default=64)
    train.add_argument("--gradient-accumulation-steps", type=int, default=1)
    train.add_argument("--learning-rate", type=float, default=2e-5)
    train.add_argument("--weight-decay", type=float, default=0.01)
    train.add_argument("--num-train-epochs", type=float, default=5.0)
    train.add_argument("--warmup-ratio", type=float, default=0.06)
    train.add_argument("--logging-steps", type=int, default=50)
    train.add_argument("--save-total-limit", type=int, default=2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--bf16", action="store_true")
    train.add_argument("--fp16", action="store_true")
    train.add_argument("--early-stopping-patience", type=int, default=2)
    train.add_argument("--metric-for-best-model", default="eval_f1")
    train.add_argument("--mention-threshold-candidates", default="0.1,0.2,0.3,0.4,0.5")
    train.add_argument("--threshold-optimize", default="example_f1", choices=["example_f1", "exact_set_match"])
    train.add_argument("--resume-from-checkpoint")
    train.add_argument("--max-train-samples", type=int)
    train.add_argument("--max-eval-samples", type=int)
    train.set_defaults(func=_cmd_train)

    audit = sub.add_parser("audit", help="audit repository contents for public release")
    audit.add_argument("--root", type=Path, default=Path.cwd())
    audit.set_defaults(func=_cmd_audit)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
