#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_claims(path: Path) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError:
        yaml = None

    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        payload = yaml.safe_load(text)
        return list(payload.get("claims", []))

    claims: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- id:"):
            if current:
                claims.append(current)
            current = {"id": stripped.split(":", 1)[1].strip()}
        elif current is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            value = value.strip().strip("'\"")
            try:
                current[key] = float(value)
            except ValueError:
                current[key] = value
    if current:
        claims.append(current)
    return claims


def _get_nested(payload: Any, path: str) -> Any:
    current = payload
    for token in path.split("."):
        if isinstance(current, dict) and token in current:
            current = current[token]
        else:
            raise KeyError(path)
    return current


def main() -> int:
    parser = argparse.ArgumentParser(description="Check manuscript-facing claims against formal result files.")
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    claims = _load_claims(args.claims)
    failures: list[str] = []
    checked = 0
    skipped = 0
    for claim in claims:
        if str(claim.get("status", "active")) != "active":
            skipped += 1
            continue
        result_file = args.results / str(claim["file"])
        if not result_file.exists():
            failures.append(f"{claim['id']}: missing file {result_file}")
            continue
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        try:
            observed = _get_nested(payload, str(claim["json_path"]))
        except KeyError:
            failures.append(f"{claim['id']}: missing JSON path {claim['json_path']}")
            continue
        expected = claim.get("expected")
        tolerance = float(claim.get("tolerance", args.tolerance))
        if expected is not None:
            if isinstance(expected, float):
                if abs(float(observed) - expected) > tolerance:
                    failures.append(f"{claim['id']}: expected {expected}, observed {observed}")
                    continue
            elif observed != expected:
                failures.append(f"{claim['id']}: expected {expected!r}, observed {observed!r}")
                continue
        checked += 1

    print(f"{checked} claims checked; {skipped} skipped; {len(failures)} failures")
    for failure in failures:
        print(f"FAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
