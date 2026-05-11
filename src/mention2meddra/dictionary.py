from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from .templates import render_candidate_text


DICTIONARY_COLUMNS = ("pt_code", "pt_name", "llt_name", "hlt_name", "hlgt_name", "soc_name")


def read_dictionary_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        return [{key: (value or "") for key, value in row.items()} for row in reader]


def build_candidates_from_rows(rows: Iterable[dict[str, Any]], template_name: str = "full") -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for row in rows:
        pt_code = str(row.get("pt_code", "")).strip()
        if not pt_code:
            continue
        if pt_code not in grouped:
            order.append(pt_code)
            grouped[pt_code] = {
                "pt_code": pt_code,
                "pt_name": str(row.get("pt_name", "")).strip(),
                "llt_names": [],
                "hlt_name": str(row.get("hlt_name", "")).strip(),
                "hlgt_name": str(row.get("hlgt_name", "")).strip(),
                "soc_name": str(row.get("soc_name", "")).strip(),
            }
        llt_name = str(row.get("llt_name", "")).strip()
        if llt_name and llt_name not in grouped[pt_code]["llt_names"]:
            grouped[pt_code]["llt_names"].append(llt_name)

    candidates = [grouped[pt_code] for pt_code in order]
    for candidate in candidates:
        candidate["text_b"] = render_candidate_text(candidate, template_name)
    return candidates


def load_dictionary(path: str | Path, template_name: str = "full") -> list[dict[str, Any]]:
    return build_candidates_from_rows(read_dictionary_csv(path), template_name=template_name)


def candidate_to_pair_row(mention_row: dict[str, Any], candidate: dict[str, Any], score: float, *, pred_label: int = 0) -> dict[str, Any]:
    text_a = str(mention_row.get("text_a", mention_row.get("raw_term", ""))).strip()
    return {
        "mention_id": str(mention_row["mention_id"]),
        "text_a": text_a,
        "raw_term": str(mention_row.get("raw_term", text_a)),
        "candidate_pt_code": str(candidate["pt_code"]),
        "candidate_pt_name": str(candidate["pt_name"]),
        "candidate_llt_names": list(candidate.get("llt_names", [])),
        "candidate_hlt_name": str(candidate.get("hlt_name", "")),
        "candidate_hlgt_name": str(candidate.get("hlgt_name", "")),
        "candidate_soc_name": str(candidate.get("soc_name", "")),
        "gold_pt_codes": [str(code) for code in mention_row.get("gold_pt_codes", [])],
        "gold_soc_codes": [str(code) for code in mention_row.get("gold_soc_codes", [])],
        "label": int(str(candidate["pt_code"]) in {str(code) for code in mention_row.get("gold_pt_codes", [])}),
        "prob_1": float(score),
        "pred_label": int(pred_label),
    }
