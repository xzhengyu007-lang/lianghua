"""Does hedging the residual market beta pay, after futures carry?

The strategy runs at beta 0.35 with a long-only book, so roughly a third of its
risk is a passive CSI300 exposure it never asked for. Shorting index futures
removes that, at the cost of futures carry. This script answers "what does the
hedge actually buy" with a static and a causal rolling hedge ratio, and with the
carry cost charged explicitly.

Usage::

    python scripts/run_hedge_study.py
    python scripts/run_hedge_study.py --report reports/attrib_full --carry 0.02
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

from quant_system.bootstrap import bootstrap_summary  # noqa: E402
from quant_system.equity_io import daily_returns_from_equity  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402


def load_report_returns(report: Path) -> pd.Series:
    return daily_returns_from_equity(report).rename("portfolio")


def load_index(cache_dir: Path) -> pd.Series:
    frame = pd.read_parquet(cache_dir / "sh000300_raw.parquet")
    frame.index = pd.to_datetime(frame.index)
    return frame["close"].astype(float).rename("index")


def rolling_beta(portfolio: pd.Series, market: pd.Series, window: int = 60) -> pd.Series:
    """Causal beta: the value used on day t only sees data up to t-1."""
    covariance = portfolio.rolling(window).cov(market)
    variance = market.rolling(window).var()
    return (covariance / variance).shift(1)


def summarise(returns: pd.Series, label: str) -> dict:
    metrics = performance_metrics(returns)
    boot = bootstrap_summary(returns, n_boot=1000)
    return {
        "label": label,
        "cagr": metrics.cagr,
        "vol": metrics.annual_volatility,
        "sharpe": metrics.sharpe,
        "mdd": metrics.max_drawdown,
        "calmar": metrics.calmar,
        "sharpe_low": boot["sharpe"]["p2"],
        "sharpe_high": boot["sharpe"]["p97"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "reports" / "attrib_full"))
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--carry", type=float, default=0.02, help="annual futures carry charged to the hedge")
    parser.add_argument("--output", default=str(ROOT / "reports" / "hedge_study"))
    args = parser.parse_args(argv)

    portfolio = load_report_returns(Path(args.report))
    market = load_index(Path(args.cache_dir))
    market_returns = market.pct_change().reindex(portfolio.index).fillna(0.0)

    static_beta = float(np.cov(portfolio, market_returns)[0, 1] / np.var(market_returns))
    dynamic_beta = rolling_beta(portfolio, market_returns).reindex(portfolio.index)

    rows = [summarise(portfolio, "unhedged")]
    daily_carry = args.carry / 252.0
    for ratio in (0.25, 0.35, 0.5, 1.0):
        hedged = portfolio - ratio * market_returns - abs(ratio) * daily_carry
        rows.append(summarise(hedged, f"static hedge {ratio:.0%}"))

    for cap in (0.5, 1.0):
        ratio = dynamic_beta.clip(lower=0.0, upper=cap).fillna(0.0)
        hedged = portfolio - ratio * market_returns - ratio * daily_carry
        rows.append(summarise(hedged, f"rolling 60d beta, cap {cap:.0%}"))

    table = pd.DataFrame(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "hedge_table.csv", encoding="utf-8-sig", index=False)
    (output / "hedge_summary.json").write_text(
        json.dumps({"static_beta": static_beta, "carry": args.carry, "rows": rows},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    display = table.copy()
    for column in ("cagr", "vol", "mdd"):
        display[column] = (display[column] * 100).round(2)
    for column in ("sharpe", "calmar", "sharpe_low", "sharpe_high"):
        display[column] = display[column].astype(float).round(2)
    print(f"static beta over the window: {static_beta:.3f}, carry {args.carry:.1%}/yr")
    print(display.to_string(index=False))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
