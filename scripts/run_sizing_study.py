"""Volatility-target sizing study: same strategy, three risk budgets, two regimes.

Answers "if I want a smaller drawdown, what does halving the risk budget actually
buy me?" — with the same walk-forward windows, parameters and costs, only
``target_annual_vol`` changes. Runs are independent processes so they can be
parallelised; each writes a normal walk-forward report under
``reports/sizing/p<period>_v<vol>``.

Usage::

    python scripts/run_sizing_study.py --workers 3
    python scripts/run_sizing_study.py --vols 0.09 0.12 0.18 --periods 2024
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRID = '{"top_n":[8],"vol_window":[60],"trend_window":[120],"max_per_sector":[2],"rebalance_days":[20]}'
PERIODS = {
    "2024": ("2024-01-01", ""),
    "2022": ("2022-01-01", "2023-12-31"),
}


def build_job(period: str, vol: float) -> tuple[list[str], Path, Path]:
    start, end = PERIODS[period]
    tag = f"p{period}_v{int(round(vol * 100)):02d}"
    output = ROOT / "reports" / "sizing" / tag
    log = ROOT / "logs" / f"sizing_{tag}.log"
    cmd = [
        sys.executable, "-m", "quant_system", "--universe", "pit", "--adjustment", "rebuilt",
        "walkforward", "--strategy", "low_vol_core", "--rebalance-reset", "year",
        "--phase-sweep", "20", "--target-vol", str(vol), "--grid", GRID,
        "--start", start, "--output", str(output),
    ]
    if end:
        cmd += ["--end", end]
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
    parser.add_argument("--vols", type=float, nargs="+", default=[0.12, 0.18, 0.24])
    parser.add_argument("--periods", nargs="+", default=list(PERIODS), choices=sorted(PERIODS))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)

    jobs = [build_job(period, vol) for period in args.periods for vol in args.vols]
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
