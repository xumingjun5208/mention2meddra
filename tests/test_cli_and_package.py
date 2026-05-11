from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_package_imports_version() -> None:
    import mention2meddra

    assert mention2meddra.__version__


def test_train_extra_declares_accelerate_dependency() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"accelerate"' in pyproject


def test_module_cli_help_lists_core_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "mention2meddra", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "retrieve" in result.stdout
    assert "evaluate" in result.stdout
    assert "train" in result.stdout
    assert "audit" in result.stdout


def test_public_script_help_runs_without_optional_training_dependencies() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_cross_encoder.py", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--max-length" in result.stdout
    assert "--mention-threshold-candidates" in result.stdout


def test_train_help_does_not_import_numpy() -> None:
    script = """
import importlib.abc
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

class BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError("blocked optional numpy import")
        return None

sys.meta_path.insert(0, BlockNumpy())

from mention2meddra.train import main

try:
    main(["--help"])
except SystemExit as exc:
    raise SystemExit(exc.code)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--max-length" in result.stdout
