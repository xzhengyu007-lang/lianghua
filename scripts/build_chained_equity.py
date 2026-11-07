"""Stitch two consecutive walk-forward equity curves into one continuous series.

Both walk-forwards must use the same strategy, parameters and risk config, and
the first must end exactly where the second begins (no gap, no overlap). The
result is written as a normal report directory so every downstream tool
(``bootstrap-ci``, ``fingerprint``) can read it unchanged.

This is *not* a substitute for one long walk-forward: each segment still starts
flat and its parameter was chosen by its own training window. It is only a way to
see 2022-2026 as one path when the segments are fixed-parameter controls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.report import write_json  # noqa: E402


def load_equity(report_dir: Path) -> pd.Series:
    frame = pd.read_csv(report_dir / "equity.csv", encoding="utf-8-sig", index_col=0, parse_dates=True)
    series = frame.iloc[:, 0].astype(float)
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def stitch(first: pd.Series, second: pd.Series) -> pd.Series:
    if first.index.max() >= second.index.min():
        raise ValueError(
            f"segments overlap or touch: {first.index.max().date()} >= {second.index.min().date()}"
        )
    scaled = second / second.iloc[0] * first.iloc[-1]
    return pd.concat([first, scaled])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", required=True, help="report dir of the earlier segment")
    parser.add_argument("--second", required=True, help="report dir of the later segment")
    parser.add_argument("--output", required=True, help="new report dir to write")
    parser.add_argument("--label", default="chained")
    args = parser.parse_args(argv)

    first = load_equity(ROOT / args.first)
    second = load_equity(ROOT / args.second)
    equity = stitch(first, second)
    returns = equity.pct_change().dropna()
    metrics = performance_metrics(returns, equity=equity).to_dict()

    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    equity.to_frame("equity").to_csv(out / "equity.csv")
    write_json(
        {
            "label": args.label,
            "segments": [args.first, args.second],
            "first": [str(first.index.min().date()), str(first.index.max().date())],
            "second": [str(second.index.min().date()), str(second.index.max().date())],
            "metrics": metrics,
            "note": "stitched, not a single walk-forward; both segments start flat",
        },
        out / "metrics.json",
    )
    print(json.dumps({"total_return": metrics["total_return"], "sharpe": metrics["sharpe"],
                      "max_drawdown": metrics["max_drawdown"], "cagr": metrics["cagr"]}, indent=2))
    print(f"report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
