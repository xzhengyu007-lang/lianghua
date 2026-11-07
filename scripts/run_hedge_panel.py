"""Beta-hedge panel across the three independent windows.

The single-window hedge study (RESEARCH_LOG 23.43) showed that a static 35%
short CSI300 overlay improved Sharpe and drawdown after a 2%/yr carry. That
could be a coincidence of 2022-2026, so this script repeats the test on
2019-2021, 2022-2023 and 2024-2026 with the same fixed beta, and adds a
12-month half-life EWMA beta instead of a noisy 60-day rolling estimate.

Usage::

    python scripts/run_hedge_panel.py
    python scripts/run_hedge_panel.py --carry 0.02 --n-boot 3000
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

from quant_system.bootstrap import paired_bootstrap_diff  # noqa: E402
from quant_system.equity_io import daily_returns_from_equity  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402


RUNS = {
    "2019-2021": "reports/wf_pit267_2019_2021",
    "2022-2023": "reports/wf_pit267_2022_trend120",
    "2024-2026": "reports/wf_pit_full267_year",
}


def load_returns(report: Path) -> pd.Series:
    return daily_returns_from_equity(report).rename("portfolio")


def load_market(cache_dir: Path) -> pd.Series:
    frame = pd.read_parquet(cache_dir / "sh000300_raw.parquet")
    frame.index = pd.to_datetime(frame.index)
    return frame["close"].astype(float).rename("market")


def ewma_beta(portfolio: pd.Series, market: pd.Series, halflife: int = 252) -> pd.Series:
    cov = portfolio.ewm(halflife=halflife, min_periods=60).cov(market)
    var = market.ewm(halflife=halflife, min_periods=60).var()
    return (cov / var).shift(1)


def summarise(returns: pd.Series, label: str) -> dict:
    m = performance_metrics(returns)
    return {
        "label": label,
        "cagr": float(m.cagr),
        "annual_volatility": float(m.annual_volatility),
        "sharpe": float(m.sharpe),
        "max_drawdown": float(m.max_drawdown),
        "calmar": float(m.calmar),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "hedge_panel"))
    parser.add_argument("--carry", type=float, default=0.02)
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    market = load_market(Path(args.cache_dir))
    rows: list[dict] = []
    pair_rows: list[dict] = []
    daily_carry = args.carry / 252.0

    for window, report in RUNS.items():
        portfolio = load_returns(ROOT / report)
        mr = market.pct_change().reindex(portfolio.index).fillna(0.0)
        beta = ewma_beta(portfolio, mr)
        base = summarise(portfolio, "unhedged")
        base["window"] = window
        rows.append(base)

        variants = {
            "static 25%": 0.25 * np.ones(len(portfolio)),
            "static 35%": 0.35 * np.ones(len(portfolio)),
            "static 50%": 0.50 * np.ones(len(portfolio)),
            "EWMA beta (12m HL), cap 50%": beta.clip(0.0, 0.5).fillna(0.0).to_numpy(),
            "EWMA beta (12m HL), cap 100%": beta.clip(0.0, 1.0).fillna(0.0).to_numpy(),
            "60d rolling beta, cap 50%": (
                portfolio.rolling(60).cov(mr) / mr.rolling(60).var()
            ).shift(1).clip(0.0, 0.5).fillna(0.0).to_numpy(),
        }
        returns_by_label = {"unhedged": portfolio}
        for label, ratio in variants.items():
            ratio = pd.Series(ratio, index=portfolio.index).astype(float)
            hedged = portfolio - ratio * mr - ratio.abs() * daily_carry
            returns_by_label[label] = hedged
            row = summarise(hedged, label)
            row["window"] = window
            rows.append(row)
            diff = paired_bootstrap_diff(
                hedged, portfolio, n_boot=args.n_boot, seed=args.seed
            )
            pair_rows.append({"window": window, "variant": label, **diff.to_dict()})

    frame = pd.DataFrame(rows)
    pairs = pd.DataFrame(pair_rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "hedge_windows.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output / "paired_sharpe.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {"carry": args.carry, "rows": rows, "paired_sharpe": pair_rows},
            ensure_ascii=False, indent=2, default=float,
        ),
        encoding="utf-8",
    )

    display = frame.copy()
    for col in ("cagr", "annual_volatility", "max_drawdown"):
        display[col] = (display[col] * 100).round(2)
    for col in ("sharpe", "calmar"):
        display[col] = display[col].astype(float).round(3)
    print(display[["window", "label", "cagr", "annual_volatility", "sharpe",
                   "max_drawdown", "calmar"]].to_string(index=False))

    lines = [
        "# Beta hedge panel", "",
        f"Carry: {args.carry:.1%}/yr, paired bootstrap B={args.n_boot}.", "",
        "| 窗口 | 方案 | CAGR | 年化波动 | Sharpe | 最大回撤 | 卡玛 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['window']} | {row['label']} | {row['cagr']*100:.2f}% | "
            f"{row['annual_volatility']*100:.2f}% | {row['sharpe']:.3f} | "
            f"{row['max_drawdown']*100:.2f}% | {row['calmar']:.2f} |"
        )
    lines += ["", "## 配对 ΔSharpe（对冲 − 不对冲）", "",
              "| 窗口 | 方案 | ΔSharpe | 95% 区间 | P(Δ>0) |", "|---|---|---:|---|---:|"]
    for row in pair_rows:
        lines.append(
            f"| {row['window']} | {row['variant']} | {row['observed']:+.3f} | "
            f"[{row['low']:+.3f}, {row['high']:+.3f}] | {row['prob_positive']:.3f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
