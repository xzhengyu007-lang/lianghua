"""Factor study: does anything survive after low volatility is controlled for?

Runs the candidate library from ``quant_system.factors`` over the point-in-time
universe, at the strategy's own rebalancing horizon (20 trading days), and
reports three numbers per factor:

* raw IC  - Spearman rank correlation with the forward 20-day return;
* corr with low-vol - how much of the factor is just the incumbent signal;
* partial IC - the same correlation with the low-vol rank stripped out daily.

Only the third number matters for "add a second, orthogonal signal".

Usage::

    python scripts/run_factor_study.py
    python scripts/run_factor_study.py --horizon 20 --start 2023-01-01
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analytics import forward_returns, ic_summary, spearman_ic  # noqa: E402
from quant_system.data import pit_universe, price_frame  # noqa: E402
from quant_system.factors import factor_library, partial_ic, rank_correlation  # noqa: E402


def rebuilt_panel(cache_dir: Path, symbols: list[str]) -> dict[str, pd.DataFrame]:
    from quant_system.cli import _rebuilt_panel

    return _rebuilt_panel(symbols, cache_dir, "rebuilt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--start", default="2022-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=str(ROOT / "reports" / "factor_study"))
    args = parser.parse_args(argv)

    panel = rebuilt_panel(Path(args.cache_dir), list(pit_universe()))
    close = price_frame(panel, "close")
    if args.end:
        close = close.loc[: args.end]
    close = close.loc[args.start:]
    panel = {k: v.loc[: args.end] if args.end else v for k, v in panel.items()}

    factors = factor_library(panel)
    future = forward_returns(close, args.horizon)
    control = factors["low_vol"]

    rows = []
    for name, signal in factors.items():
        signal = signal.reindex_like(close).loc[args.start:]
        ic = spearman_ic(signal, future)
        summary = ic_summary(ic)
        partial = partial_ic(signal, control.reindex_like(close).loc[args.start:], future)
        partial_summary = ic_summary(partial)
        rows.append(
            {
                "factor": name,
                "mean_ic": summary["mean_ic"],
                "ic_ir": summary["ic_ir"],
                "t_stat": summary["t_stat"],
                "positive_ratio": summary["positive_ratio"],
                "corr_with_lowvol": rank_correlation(signal, control.reindex_like(close).loc[args.start:]),
                "partial_ic": partial_summary["mean_ic"],
                "partial_ir": partial_summary["ic_ir"],
                "partial_t": partial_summary["t_stat"],
                "days": int(ic.notna().sum()),
            }
        )

    frame = pd.DataFrame(rows).set_index("factor").sort_values("partial_ic", ascending=False)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "factor_ic.csv", encoding="utf-8-sig")

    display = frame.copy()
    for column in ("mean_ic", "ic_ir", "t_stat", "positive_ratio", "corr_with_lowvol",
                   "partial_ic", "partial_ir", "partial_t"):
        display[column] = display[column].astype(float).round(4)
    print(display.to_string())
    (output / "factor_ic.json").write_text(
        json.dumps(frame.reset_index().to_dict(orient="records"), ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
