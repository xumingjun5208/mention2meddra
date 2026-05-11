from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dictionary import DICTIONARY_COLUMNS
from .io import read_jsonl


PAIR_REQUIRED = (
    "mention_id",
    "candidate_pt_code",
    "candidate_pt_name",
    "candidate_llt_names",
    "candidate_hlt_name",
    "candidate_hlgt_name",
    "candidate_soc_name",
    "gold_pt_codes",
    "gold_soc_codes",
    "label",
)
PREDICTION_REQUIRED = PAIR_REQUIRED + ("prob_1", "pred_label")
METRICS_REQUIRED = (
    "pair_metrics",
    "exact_set_match",
    "example_f1",
    "micro_f1",
    "top1_accuracy",
    "recall_at_k_all_gold",
    "recall_at_k_any_hit",
)


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    errors: list[str]
    row_count: int = 0


def _missing(row: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    return [field for field in fields if field not in row]


def _validate_binary_field(row: dict[str, Any], field: str, line_no: int, errors: list[str]) -> None:
    if field not in row:
        return
    try:
        value = int(row[field])
    except (TypeError, ValueError):
        errors.append(f"line {line_no}: {field} must be 0 or 1")
        return
    if value not in (0, 1):
        errors.append(f"line {line_no}: {field} must be 0 or 1")


def _validate_probability_field(row: dict[str, Any], field: str, line_no: int, errors: list[str]) -> None:
    if field not in row:
        return
    try:
        value = float(row[field])
    except (TypeError, ValueError):
        errors.append(f"line {line_no}: {field} must be in [0, 1]")
        return
    if not 0.0 <= value <= 1.0:
        errors.append(f"line {line_no}: {field} must be in [0, 1]")


def validate_dictionary_csv(path: str | Path) -> ValidationReport:
    path = Path(path)
    errors: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [field for field in DICTIONARY_COLUMNS if field not in fieldnames]
        if missing:
            errors.append(f"missing dictionary columns: {', '.join(missing)}")
        row_count = 0
        for row_count, row in enumerate(reader, start=1):
            for field in ("pt_code", "pt_name"):
                if not str(row.get(field, "")).strip():
                    errors.append(f"row {row_count}: {field} is required")
    return ValidationReport(ok=not errors, errors=errors, row_count=row_count if "row_count" in locals() else 0)


def _validate_jsonl_rows(path: str | Path, required: tuple[str, ...], label: str) -> ValidationReport:
    rows = read_jsonl(path)
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        missing = _missing(row, required)
        if missing:
            errors.append(f"line {index}: missing {label} fields: {', '.join(missing)}")
        if "text_a" not in row and "raw_term" not in row:
            errors.append(f"line {index}: one of text_a or raw_term is required")
        for list_field in ("candidate_llt_names", "gold_pt_codes", "gold_soc_codes"):
            if list_field in row and not isinstance(row[list_field], list):
                errors.append(f"line {index}: {list_field} must be a list")
        _validate_binary_field(row, "label", index, errors)
        _validate_binary_field(row, "pred_label", index, errors)
        _validate_probability_field(row, "prob_1", index, errors)
    return ValidationReport(ok=not errors, errors=errors, row_count=len(rows))


def validate_pair_jsonl(path: str | Path) -> ValidationReport:
    return _validate_jsonl_rows(path, PAIR_REQUIRED, "pair")


def validate_prediction_jsonl(path: str | Path) -> ValidationReport:
    return _validate_jsonl_rows(path, PREDICTION_REQUIRED, "prediction")


def validate_metrics(payload: dict[str, Any]) -> ValidationReport:
    errors = [f"missing metrics key: {key}" for key in METRICS_REQUIRED if key not in payload]
    if "recall_at_k_all_gold" in payload and not isinstance(payload["recall_at_k_all_gold"], dict):
        errors.append("recall_at_k_all_gold must be an object")
    if "recall_at_k_any_hit" in payload and not isinstance(payload["recall_at_k_any_hit"], dict):
        errors.append("recall_at_k_any_hit must be an object")
    return ValidationReport(ok=not errors, errors=errors, row_count=1 if payload else 0)
