from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .metrics import compute_mention_metrics


SELECTION_RULES = {
    "example_f1": "example_f1_then_exact_set_match_then_lower_threshold",
    "exact_set_match": "exact_set_match_then_example_f1_then_lower_threshold",
}


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    metrics: dict[str, Any]
    optimize: str
    selection_rule: str


def _selection_key(metrics: dict[str, Any], threshold: float, optimize: str) -> tuple[float, float, float]:
    if optimize == "example_f1":
        return (float(metrics["example_f1"]), float(metrics["exact_set_match"]), -float(threshold))
    if optimize == "exact_set_match":
        return (float(metrics["exact_set_match"]), float(metrics["example_f1"]), -float(threshold))
    raise ValueError("optimize must be 'example_f1' or 'exact_set_match'")


def select_threshold(
    rows: list[dict[str, Any]],
    thresholds: list[float] | tuple[float, ...],
    *,
    optimize: str = "example_f1",
) -> ThresholdSelection:
    if not thresholds:
        raise ValueError("at least one threshold is required")
    best_threshold = float(thresholds[0])
    best_metrics = compute_mention_metrics(rows, threshold=best_threshold)
    best_key = _selection_key(best_metrics, best_threshold, optimize)
    for threshold in thresholds[1:]:
        threshold = float(threshold)
        metrics = compute_mention_metrics(rows, threshold=threshold)
        key = _selection_key(metrics, threshold, optimize)
        if key > best_key:
            best_threshold = threshold
            best_metrics = metrics
            best_key = key
    return ThresholdSelection(
        threshold=best_threshold,
        metrics=best_metrics,
        optimize=optimize,
        selection_rule=SELECTION_RULES[optimize],
    )


def scan_thresholds(
    rows: list[dict[str, Any]],
    thresholds: list[float] | tuple[float, ...],
    *,
    optimize: str = "example_f1",
) -> dict[str, Any]:
    metrics_by_threshold = {
        f"{float(threshold):g}": compute_mention_metrics(rows, threshold=float(threshold))
        for threshold in thresholds
    }
    selected = select_threshold(rows, list(thresholds), optimize=optimize)
    return {
        "selection_rule": selected.selection_rule,
        "optimize": optimize,
        "selected_threshold": selected.threshold,
        "thresholds": metrics_by_threshold,
    }
