from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from mention2meddra.audit import audit_repository


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_repository_audit_blocks_private_artifact_patterns() -> None:
    report = audit_repository(PROJECT_ROOT)

    assert report.ok, report.errors
    assert report.scanned_files > 0


def test_readme_records_public_data_boundaries() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    for phrase in [
        "realistic synthetic examples",
        "real adverse-event records",
        "expert corpus",
        "licensed MedDRA dictionary files",
        "trained model weights",
        "downstream signal-detection datasets",
        "MedDRA",
    ]:
        assert phrase in readme


def test_smoke_workflow_script_produces_metrics_json(tmp_path: Path) -> None:
    output_dir = tmp_path / "smoke"
    result = subprocess.run(
        ["bash", "scripts/run_smoke_workflow.sh", str(output_dir)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "Smoke workflow complete" in result.stdout
    assert (output_dir / "retrieval" / "lexical_retrieval_summary.json").exists()
    assert (output_dir / "predictions" / "metrics.json").exists()


def test_manuscript_claim_audit_runs_on_fixture_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    metrics_dir = results / "step4_finetuned" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "dev_mention_metrics.json").write_text(
        (
            '{"selected_threshold": 0.3, "exact_set_match": 0.8830645161290323, '
            '"example_precision": 0.9279052936311001, '
            '"example_recall": 0.9430314309346567, '
            '"example_f1": 0.9311429162235614}'
        ),
        encoding="utf-8",
    )
    (metrics_dir / "test_mention_metrics.json").write_text(
        (
            '{"exact_set_match": 0.8957816377171216, '
            '"micro_f1": 0.9586813186813188, '
            '"top1_accuracy": 0.9779776674937966, '
            '"recall_at_3": 0.983560794044665, '
            '"recall_at_5": 0.9996898263027295}'
        ),
        encoding="utf-8",
    )
    (results / "step4_finetuned" / "metrics_summary.json").write_text(
        '{"test": {"test_roc_auc": 0.9966351963325706}}',
        encoding="utf-8",
    )
    bm25_dir = results / "step1_bm25_retrieval"
    bm25_dir.mkdir(parents=True)
    (bm25_dir / "lexical_retrieval_summary.json").write_text(
        '{"bm25": {"test_retrieval": {"mrr": 0.5103}}}',
        encoding="utf-8",
    )
    claims_file = tmp_path / "claims.yml"
    claims_file.write_text(
        "\n".join(
            [
                "claims:",
                "  - id: selected_threshold",
                "    file: step4_finetuned/metrics/dev_mention_metrics.json",
                "    json_path: selected_threshold",
                "    expected: 0.3",
                "  - id: test_top1_accuracy",
                "    file: step4_finetuned/metrics/test_mention_metrics.json",
                "    json_path: top1_accuracy",
                "    expected: 0.9779776674937966",
                "  - id: test_auc",
                "    file: step4_finetuned/metrics_summary.json",
                "    json_path: test.test_roc_auc",
                "    expected: 0.9966351963325706",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/audit_manuscript_claims.py",
            "--claims",
            str(claims_file),
            "--results",
            str(results),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "claims checked" in result.stdout
