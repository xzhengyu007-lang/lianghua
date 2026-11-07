"""Thin wrapper: block-bootstrap confidence intervals for walk-forward reports.

Real implementation lives in ``quant_system.bootstrap_report`` and is exposed as
``python -m quant_system bootstrap-ci``. This script exists so the research runs
in this repo can be reproduced with one short command.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["bootstrap-ci", *sys.argv[1:]]))
