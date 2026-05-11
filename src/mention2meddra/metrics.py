from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any


def ensure_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def group_rows_by_mention(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mention_id"])].append(row)
    return dict(grouped)


def _gold_set(rows: list[dict[str, Any]]) -> set[str]:
    return {str(code) for code in ensure_list(rows[0].get("gold_pt_codes"))}


def _predicted_set(rows: list[dict[str, Any]], threshold: float) -> set[str]:
    return {
        str(row["candidate_pt_code"])
        for row in rows
        if float(row.get("prob_1", 0.0)) >= threshold
    }


def _sorted_candidates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get("prob_1", row.get("retrieval_score", 0.0))), reverse=True)


def compute_binary_pair_metrics(rows: list[dict[str, Any]], threshold: float = 0.5) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for row in rows:
        gold = int(row.get("label", 0))
        pred = int(float(row.get("prob_1", 0.0)) >= threshold)
        if pred == 1 and gold == 1:
            tp += 1
        elif pred == 0 and gold == 0:
            tn += 1
        elif pred == 1 and gold == 0:
            fp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {
        "pairs": float(len(rows)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def _per_mention(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    gold = _gold_set(rows)
    pred = _predicted_set(rows, threshold)
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact": float(pred == gold),
        "cardinality": float(len(pred) == len(gold)),
    }


def compute_mention_metrics(rows: list[dict[str, Any]], threshold: float = 0.5, ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, Any]:
    grouped = group_rows_by_mention(rows)
    if not grouped:
        return {
            "mentions": 0,
            "pair_metrics": compute_binary_pair_metrics([], threshold),
            "exact_set_match": 0.0,
            "example_precision": 0.0,
            "example_recall": 0.0,
            "example_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "cardinality_accuracy": 0.0,
            "top1_accuracy": 0.0,
            "recall_at_k_all_gold": {str(k): 0.0 for k in ks},
            "recall_at_k_any_hit": {str(k): 0.0 for k in ks},
        }

    per = []
    total_tp = total_fp = total_fn = 0.0
    top1_hits = 0
    all_gold = {k: 0 for k in ks}
    any_hit = {k: 0 for k in ks}

    for mention_rows in grouped.values():
        gold = _gold_set(mention_rows)
        item = _per_mention(mention_rows, threshold)
        per.append(item)
        total_tp += item["tp"]
        total_fp += item["fp"]
        total_fn += item["fn"]

        ranked_codes = [str(row["candidate_pt_code"]) for row in _sorted_candidates(mention_rows)]
        if ranked_codes[:1] and ranked_codes[0] in gold:
            top1_hits += 1
        for k in ks:
            topk = ranked_codes[:k]
            topk_set = set(topk)
            if any(code in gold for code in topk):
                any_hit[k] += 1
            if gold.issubset(topk_set):
                all_gold[k] += 1

    mentions = len(grouped)
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0

    return {
        "mentions": mentions,
        "pair_metrics": compute_binary_pair_metrics(rows, threshold),
        "exact_set_match": mean(item["exact"] for item in per),
        "example_precision": mean(item["precision"] for item in per),
        "example_recall": mean(item["recall"] for item in per),
        "example_f1": mean(item["f1"] for item in per),
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "cardinality_accuracy": mean(item["cardinality"] for item in per),
        "top1_accuracy": top1_hits / mentions,
        "recall_at_k_all_gold": {str(k): all_gold[k] / mentions for k in ks},
        "recall_at_k_any_hit": {str(k): any_hit[k] / mentions for k in ks},
    }


def ranking_metrics_for_mentions(rows: list[dict[str, Any]], ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    grouped = group_rows_by_mention(rows)
    if not grouped:
        result = {"mentions": 0.0, "mrr": 0.0}
        for k in ks:
            result[f"recall_at_{k}_any_hit"] = 0.0
            result[f"recall_at_{k}_all_gold"] = 0.0
            result[f"ndcg_at_{k}"] = 0.0
        return result

    any_hits = {k: 0 for k in ks}
    all_gold = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    mrr = 0.0
    for mention_rows in grouped.values():
        gold = _gold_set(mention_rows)
        ranked_codes = [str(row["candidate_pt_code"]) for row in _sorted_candidates(mention_rows)]
        for index, code in enumerate(ranked_codes, start=1):
            if code in gold:
                mrr += 1.0 / index
                break
        for k in ks:
            topk = ranked_codes[:k]
            topk_set = set(topk)
            if any(code in gold for code in topk):
                any_hits[k] += 1
            if gold.issubset(topk_set):
                all_gold[k] += 1
            dcg = sum(1.0 / math.log2(index + 1) for index, code in enumerate(topk, start=1) if code in gold)
            ideal_hits = min(len(gold), k)
            idcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
            ndcg[k] += dcg / idcg if idcg else 0.0

    mentions = len(grouped)
    result = {"mentions": float(mentions), "mrr": mrr / mentions}
    for k in ks:
        result[f"recall_at_{k}_any_hit"] = any_hits[k] / mentions
        result[f"recall_at_{k}_all_gold"] = all_gold[k] / mentions
        result[f"ndcg_at_{k}"] = ndcg[k] / mentions
    return result
