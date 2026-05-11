from __future__ import annotations

import json
from pathlib import Path

import pytest

from mention2meddra.io import read_jsonl, write_jsonl
from mention2meddra.schemas import (
    validate_dictionary_csv,
    validate_metrics,
    validate_pair_jsonl,
    validate_prediction_jsonl,
)


def test_dictionary_schema_requires_public_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "meddra.csv"
    csv_path.write_text(
        "pt_code,pt_name,llt_name,hlt_name,hlgt_name,soc_name\n"
        "1001,皮疹,皮疹,皮肤症状,皮肤疾病,皮肤及皮下组织类疾病\n",
        encoding="utf-8",
    )

    report = validate_dictionary_csv(csv_path)

    assert report.ok
    assert report.row_count == 1


def test_dictionary_schema_reports_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("pt_code,pt_name\n1001,皮疹\n", encoding="utf-8")

    report = validate_dictionary_csv(csv_path)

    assert not report.ok
    assert "llt_name" in report.errors[0]


def test_pair_prediction_and_metrics_schemas_round_trip(tmp_path: Path) -> None:
    pairs = [
        {
            "mention_id": "m1",
            "text_a": "皮疹",
            "raw_term": "皮疹",
            "candidate_pt_code": "1001",
            "candidate_pt_name": "皮疹",
            "candidate_llt_names": ["皮疹", "皮肤发疹"],
            "candidate_hlt_name": "皮肤症状",
            "candidate_hlgt_name": "皮肤疾病",
            "candidate_soc_name": "皮肤及皮下组织类疾病",
            "gold_pt_codes": ["1001"],
            "gold_soc_codes": ["SOC100"],
            "label": 1,
        }
    ]
    pair_path = tmp_path / "pairs.jsonl"
    write_jsonl(pairs, pair_path)

    assert read_jsonl(pair_path) == pairs
    assert validate_pair_jsonl(pair_path).ok

    predictions = [dict(pairs[0], prob_1=0.93, pred_label=1)]
    pred_path = tmp_path / "predictions.jsonl"
    write_jsonl(predictions, pred_path)

    assert validate_prediction_jsonl(pred_path).ok
    assert validate_metrics(
        {
            "mentions": 1,
            "pair_metrics": {"accuracy": 1.0},
            "exact_set_match": 1.0,
            "example_f1": 1.0,
            "micro_f1": 1.0,
            "top1_accuracy": 1.0,
            "recall_at_k_all_gold": {"1": 1.0},
            "recall_at_k_any_hit": {"1": 1.0},
        }
    ).ok


def test_jsonl_loader_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        read_jsonl(path)


def test_pair_schema_reports_malformed_label_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "bad_label.jsonl"
    write_jsonl(
        [
            {
                "mention_id": "m1",
                "text_a": "皮疹",
                "candidate_pt_code": "1001",
                "candidate_pt_name": "皮疹",
                "candidate_llt_names": ["皮疹"],
                "candidate_hlt_name": "皮肤症状",
                "candidate_hlgt_name": "皮肤疾病",
                "candidate_soc_name": "皮肤及皮下组织类疾病",
                "gold_pt_codes": ["1001"],
                "gold_soc_codes": ["SOC100"],
                "label": "positive",
            }
        ],
        path,
    )

    report = validate_pair_jsonl(path)

    assert not report.ok
    assert "line 1: label must be 0 or 1" in report.errors


def test_prediction_schema_reports_malformed_numeric_fields_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "bad_prediction_numbers.jsonl"
    write_jsonl(
        [
            {
                "mention_id": "m1",
                "text_a": "皮疹",
                "candidate_pt_code": "1001",
                "candidate_pt_name": "皮疹",
                "candidate_llt_names": ["皮疹"],
                "candidate_hlt_name": "皮肤症状",
                "candidate_hlgt_name": "皮肤疾病",
                "candidate_soc_name": "皮肤及皮下组织类疾病",
                "gold_pt_codes": ["1001"],
                "gold_soc_codes": ["SOC100"],
                "label": 1,
                "prob_1": "high",
                "pred_label": "yes",
            }
        ],
        path,
    )

    report = validate_prediction_jsonl(path)

    assert not report.ok
    assert "line 1: prob_1 must be in [0, 1]" in report.errors
    assert "line 1: pred_label must be 0 or 1" in report.errors
