from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


PRIVATE_NAME_PATTERNS = [
    re.compile(r"jiangsu_adr_macbert_(strict|weak|pairs)", re.IGNORECASE),
    re.compile(r"JSADE", re.IGNORECASE),
    re.compile(r"(?:full|licensed)[_-]?meddra|meddra[_-]?(?:full|licensed)", re.IGNORECASE),
]
LARGE_OR_PRIVATE_SUFFIXES = {
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
    ".ckpt",
    ".xlsx",
    ".docx",
    ".zip",
    ".tar",
    ".gz",
}
LOCAL_PREFIX = "/" + "llama" + "/" + "gaofh"
HOME_PREFIX = "/" + "home" + "/"
ABSOLUTE_PATH_RE = re.compile(rf"({re.escape(LOCAL_PREFIX)}|{re.escape(HOME_PREFIX)}[^\\s'\"]+|[A-Za-z]:\\\\)")
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "dist", "build", ".mypy_cache", ".ruff_cache"}


@dataclass(frozen=True)
class AuditReport:
    ok: bool
    errors: list[str]
    warnings: list[str]
    scanned_files: int


def audit_repository(root: str | Path) -> AuditReport:
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    scanned = 0
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_dir():
            continue
        scanned += 1
        rel_text = rel.as_posix()
        if path.suffix.lower() in LARGE_OR_PRIVATE_SUFFIXES:
            errors.append(f"large or private artifact suffix is not allowed: {rel_text}")
        if path.stat().st_size > 5_000_000:
            errors.append(f"file exceeds public release size limit: {rel_text}")
        if any(pattern.search(rel_text) for pattern in PRIVATE_NAME_PATTERNS):
            errors.append(f"private artifact name pattern is not allowed: {rel_text}")
        if path.suffix.lower() in {".py", ".md", ".yml", ".yaml", ".json", ".jsonl", ".csv", ".toml", ".sh", ".cff", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if ABSOLUTE_PATH_RE.search(text):
                errors.append(f"absolute local path found in {rel_text}")
            if any(pattern.search(text) for pattern in PRIVATE_NAME_PATTERNS) and rel_text not in {
                "README.md",
                "src/mention2meddra/audit.py",
            }:
                warnings.append(f"restricted-data term mentioned in {rel_text}")
    return AuditReport(ok=not errors, errors=errors, warnings=warnings, scanned_files=scanned)
