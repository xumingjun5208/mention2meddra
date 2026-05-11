from __future__ import annotations

from mention2meddra.metrics import compute_binary_pair_metrics, compute_mention_metrics, ranking_metrics_for_mentions
from mention2meddra.thresholds import scan_thresholds, select_threshold


def _prediction(
    mention_id: str,
    pt_code: str,
    label: int,
    prob_1: float,
    gold_pt_codes: list[str] | None = None,
) -> dict:
    return {
        "mention_id": mention_id,
        "raw_term": mention_id,
        "candidate_pt_code": pt_code,
        "candidate_pt_name": pt_code,
        "candidate_llt_names": [pt_code],
        "candidate_hlt_name": "HLT",
        "candidate_hlgt_name": "HLGT",
        "candidate_soc_name": "SOC",
        "gold_pt_codes": gold_pt_codes or ([pt_code] if label else ["gold"]),
        "gold_soc_codes": ["SOC"],
        "label": label,
        "prob_1": prob_1,
        "pred_label": int(prob_1 >= 0.5),
    }


def test_mention_metrics_report_all_gold_and_any_hit_recall_separately() -> None:
    rows = [
        _prediction("m1", "pt1", 1, 0.90, ["pt1", "pt2"]),
        _prediction("m1", "ptx", 0, 0.80, ["pt1", "pt2"]),
        _prediction("m1", "pt2", 1, 0.70, ["pt1", "pt2"]),
    ]

    metrics = compute_mention_metrics(rows, threshold=0.75, ks=(1, 2, 3))

    assert metrics["exact_set_match"] == 0.0
    assert metrics["example_recall"] == 0.5
    assert metrics["recall_at_k_any_hit"]["1"] == 1.0
    assert metrics["recall_at_k_all_gold"]["2"] == 0.0
    assert metrics["recall_at_k_all_gold"]["3"] == 1.0


def test_ranking_metrics_include_stable_flat_keys_for_reporting() -> None:
    rows = [
        _prediction("m1", "pt1", 1, 0.90, ["pt1", "pt2"]),
        _prediction("m1", "pt2", 1, 0.80, ["pt1", "pt2"]),
        _prediction("m2", "ptx", 0, 0.90, ["pt3"]),
        _prediction("m2", "pt3", 1, 0.70, ["pt3"]),
    ]

    metrics = ranking_metrics_for_mentions(rows, ks=(1, 2))

    assert metrics["recall_at_1_any_hit"] == 0.5
    assert metrics["recall_at_1_all_gold"] == 0.0
    assert metrics["recall_at_2_all_gold"] == 1.0


def test_pair_metrics_use_probability_threshold() -> None:
    rows = [
        _prediction("m1", "pt1", 1, 0.7),
        _prediction("m1", "pt2", 0, 0.6),
        _prediction("m2", "pt3", 0, 0.2),
    ]

    metrics = compute_binary_pair_metrics(rows, threshold=0.5)

    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fn"] == 0


def test_threshold_selection_documents_example_f1_then_exact_set_rule() -> None:
    rows = [
        _prediction("m1", "pt1", 1, 0.80, ["pt1", "pt2"]),
        _prediction("m1", "pt2", 1, 0.45, ["pt1", "pt2"]),
        _prediction("m1", "ptx", 0, 0.40, ["pt1", "pt2"]),
    ]

    selected = select_threshold(rows, [0.3, 0.5], optimize="example_f1")
    scan = scan_thresholds(rows, [0.3, 0.5], optimize="example_f1")

    assert selected.threshold == 0.3
    assert selected.selection_rule == "example_f1_then_exact_set_match_then_lower_threshold"
    assert scan["selected_threshold"] == 0.3
    assert scan["thresholds"]["0.3"]["example_f1"] > scan["thresholds"]["0.5"]["example_f1"]
