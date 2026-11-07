"""Sector attribution for the finished walk-forward windows (2022-23, 2024-26).

The question: the low-volatility core beats an equal-weight universe by a wide
margin each month. Is that *sector allocation* (the low-vol pool happens to sit
in sectors that outperformed) or *stock selection inside sectors*? Brinson-
Fachler answers it descriptionally. Nothing here refits the strategy, so it
cannot introduce look-ahead; the inputs are the positions.csv / equity.csv a
walk-forward already wrote.

Three benchmarks are reported per period, because "beat the average stock" and
"beat the calm third" are different claims: a low-volatility portfolio that only
beats the equal-weight pool may be harvesting the low-volatility factor rather
than picking stocks.

Usage::

    python scripts/run_sector_attribution.py
    python scripts/run_sector_attribution.py --periods 2024 --modes equal lowvol
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DRIVERS = {
    "2022": "reports/attrib_2022",
    "2024": "reports/attrib_2024",
    "full": "reports/attrib_full",
    "1921": "reports/attrib_1921",
}


MODES = ["equal", "lowvol", "highvol"]


def build_job(period: str, mode: str) -> tuple[list[str], Path, Path]:
    report_dir = ROOT / DRIVERS[period]
    tag = f"attribution_{period}" if mode == "equal" else f"attribution_{period}_{mode}"
    output = ROOT / "reports" / tag
    log = ROOT / "logs" / f"{tag}.log"
    cmd = [
        sys.executable, "-m", "quant_system", "--universe", "pit", "--adjustment", "rebuilt",
        "attribution", "--report-dir", str(report_dir), "--benchmark-mode", mode,
        "--output", str(output),
    ]
    return cmd, output, log


def run(job: tuple[list[str], Path, Path]) -> tuple[str, int]:
    cmd, output, log = job
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as fh, open(
        log.with_suffix(".err"), "w", encoding="utf-8"
    ) as eh:
        code = subprocess.call(cmd, stdout=fh, stderr=eh)
    return output.name, code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periods", nargs="+", default=list(DRIVERS), choices=sorted(DRIVERS))
    parser.add_argument("--modes", nargs="+", default=MODES, choices=MODES)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)

    jobs = [build_job(period, mode) for period in args.periods for mode in args.modes]
    print(f"{len(jobs)} runs, {args.workers} workers", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for name, code in pool.map(run, jobs):
            print(f"{name}: exit={code}", flush=True)
            failures += int(code != 0)
    print("done" if not failures else f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
