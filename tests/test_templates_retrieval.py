from __future__ import annotations

from pathlib import Path

from mention2meddra.dictionary import build_candidates_from_rows
from mention2meddra.io import read_jsonl, write_jsonl
from mention2meddra.cli import main
from mention2meddra.retrieval import exact_match_rank, lexical_bm25_rank
from mention2meddra.templates import render_candidate_text


def _row(
    pt_code: str,
    pt_name: str,
    llt_name: str,
    hlt_name: str = "上位语",
    hlgt_name: str = "高位组语",
    soc_name: str = "系统器官分类",
) -> dict[str, str]:
    return {
        "pt_code": pt_code,
        "pt_name": pt_name,
        "llt_name": llt_name,
        "hlt_name": hlt_name,
        "hlgt_name": hlgt_name,
        "soc_name": soc_name,
    }


def test_template_variants_match_public_contract() -> None:
    row = {
        "candidate_pt_name": "呼吸困难",
        "candidate_llt_names": ["气短", "喘不上气"],
        "candidate_hlt_name": "呼吸异常",
        "candidate_hlgt_name": "呼吸系统症状",
        "candidate_soc_name": "呼吸系统疾病",
    }

    assert render_candidate_text(row, "pt_only") == "PT: 呼吸困难"
    assert render_candidate_text(row, "pt_llt") == "PT: 呼吸困难；LLT别名: 气短 | 喘不上气"
    assert render_candidate_text(row, "full") == (
        "PT: 呼吸困难；LLT别名: 气短 | 喘不上气；HLT: 呼吸异常；"
        "HLGT: 呼吸系统症状；SOC: 呼吸系统疾病"
    )


def test_dictionary_rows_group_to_pt_level_candidates() -> None:
    candidates = build_candidates_from_rows(
        [
            _row("1001", "皮疹", "皮疹"),
            _row("1001", "皮疹", "皮肤发疹"),
            _row("1002", "瘙痒", "瘙痒"),
        ]
    )

    assert len(candidates) == 2
    assert candidates[0]["pt_code"] == "1001"
    assert candidates[0]["llt_names"] == ["皮疹", "皮肤发疹"]


def test_exact_match_rank_prefers_matching_pt_or_llt_alias() -> None:
    candidates = build_candidates_from_rows(
        [
            _row("1001", "呼吸困难", "气短"),
            _row("1002", "心悸", "心慌"),
        ]
    )

    ranked = exact_match_rank("气短", candidates)

    assert ranked[0]["pt_code"] == "1001"
    assert ranked[0]["match_type"] == "exact"


def test_character_bm25_ranks_chinese_overlap_first() -> None:
    candidates = build_candidates_from_rows(
        [
            _row("1001", "呼吸困难", "气短", hlt_name="呼吸异常", soc_name="呼吸系统疾病"),
            _row("1002", "心悸", "心慌", hlt_name="心率异常", soc_name="心脏疾病"),
            _row("1003", "恶心", "想吐", hlt_name="胃肠道症状", soc_name="胃肠系统疾病"),
        ]
    )

    ranked = lexical_bm25_rank("患者出现心慌胸闷", candidates, top_k=3)

    assert ranked[0]["pt_code"] == "1002"
    assert len(ranked) == 3
    assert all("bm25_score" in row for row in ranked)


def test_retrieve_max_mentions_limits_unique_mentions_after_grouping(tmp_path: Path) -> None:
    dictionary = tmp_path / "dictionary.csv"
    dictionary.write_text(
        "pt_code,pt_name,llt_name,hlt_name,hlgt_name,soc_name\n"
        "1001,皮疹,皮疹,皮肤症状,皮肤疾病,皮肤及皮下组织类疾病\n"
        "1002,头痛,头痛,神经系统症状,神经系统疾病,神经系统类疾病\n"
        "1003,恶心,恶心,胃肠道症状,胃肠道疾病,胃肠系统类疾病\n",
        encoding="utf-8",
    )
    base = {
        "candidate_pt_name": "皮疹",
        "candidate_llt_names": ["皮疹"],
        "candidate_hlt_name": "皮肤症状",
        "candidate_hlgt_name": "皮肤疾病",
        "candidate_soc_name": "皮肤及皮下组织类疾病",
        "gold_soc_codes": ["SOC"],
    }
    pairs = [
        dict(base, mention_id="m1", text_a="皮疹", raw_term="皮疹", candidate_pt_code="1001", gold_pt_codes=["1001"], label=1),
        dict(base, mention_id="m1", text_a="皮疹", raw_term="皮疹", candidate_pt_code="1002", gold_pt_codes=["1001"], label=0),
        dict(base, mention_id="m2", text_a="头痛", raw_term="头痛", candidate_pt_code="1002", gold_pt_codes=["1002"], label=1),
        dict(base, mention_id="m2", text_a="头痛", raw_term="头痛", candidate_pt_code="1003", gold_pt_codes=["1002"], label=0),
        dict(base, mention_id="m3", text_a="恶心", raw_term="恶心", candidate_pt_code="1003", gold_pt_codes=["1003"], label=1),
    ]
    pairs_path = tmp_path / "pairs.jsonl"
    output_dir = tmp_path / "retrieval"
    write_jsonl(pairs, pairs_path)

    exit_code = main(
        [
            "retrieve",
            "--dictionary",
            str(dictionary),
            "--pairs",
            str(pairs_path),
            "--output-dir",
            str(output_dir),
            "--top-k",
            "1",
            "--max-mentions",
            "2",
        ]
    )

    assert exit_code == 0
    ranked_mentions = {row["mention_id"] for row in read_jsonl(output_dir / "exact" / "ranked.jsonl")}
    assert ranked_mentions == {"m1", "m2"}
