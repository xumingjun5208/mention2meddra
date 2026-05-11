#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mention2meddra.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *(__import__("sys").argv[1:])]))
