"""Volatility-target panel: published 18% vs a 12% risk cap.

RESEARCH_LOG 23.35 measured a single 2022-2026 path and concluded that
``--target-vol 0.12`` raised Sharpe after cutting drawdown. One path is not
evidence, so this script repeats the same comparison on the three independent
walk-forward windows and pairs the daily returns with a block bootstrap.

Nothing here changes the signal. ``--target-vol`` only rescales gross exposure,
so any Sharpe change is a leverage decision rather than an alpha claim.

Usage::

    python scripts/run_vol_target_panel.py
    python scripts/run_vol_target_panel.py --n-boot 5000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.bootstrap import paired_bootstrap_diff  # noqa: E402
from quant_system.equity_io import daily_returns_from_equity  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402


RUNS = {
    "2019-2021": ("reports/wf_pit267_2019_2021", "reports/wf_vol12_2019"),
    "2022-2023": ("reports/wf_pit267_2022_trend120", "reports/wf_vol12_2022"),
    "2024-2026": ("reports/wf_pit_full267_year", "reports/wf_vol12_2024"),
}

METRICS = ("cagr", "annual_volatility", "sharpe", "max_drawdown", "calmar")


def load_returns(report: Path) -> pd.Series:
    return daily_returns_from_equity(report)


def summarise(returns: pd.Series) -> dict:
    m = performance_metrics(returns)
    return {name: float(getattr(m, name)) for name in METRICS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "vol_target_panel"))
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    boot_rows: list[dict] = []

    for window, (base_report, low_report) in RUNS.items():
        base = load_returns(ROOT / base_report)
        low = load_returns(ROOT / low_report)
        common = base.index.intersection(low.index)
        paired_base, paired_low = base.loc[common], low.loc[common]

        for label, series in (("18%", base), ("12%", low)):
            row = {
                "window": window,
                "target_vol": label,
                "days": int(series.size),
                "paired_days": int(common.size),
            }
            row.update(summarise(series))
            rows.append(row)

        for metric in ("sharpe", "cagr", "max_drawdown"):
            res = paired_bootstrap_diff(
                paired_low,
                paired_base,
                metric=metric,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            boot_rows.append(
                {
                    "window": window,
                    "metric": metric,
                    "delta": float(res.observed),
                    "lo": float(res.low),
                    "hi": float(res.high),
                    "prob_positive": float(res.prob_positive),
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(out / "bootstrap.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps({"summary": rows, "bootstrap": boot_rows}, indent=2),
        encoding="utf-8",
    )

    lines = ["# 目标波动率对照：18% vs 12%", ""]
    lines.append("| 窗口 | 口径 | 天数 | 配对天数 | CAGR | 波动 | Sharpe | MDD | Calmar |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {window} | {target_vol} | {days} | {paired_days} | {cagr:.2%} | {annual_volatility:.2%} | "
            "{sharpe:.3f} | {max_drawdown:.2%} | {calmar:.3f} |".format(**row)
        )
    lines.append("")
    lines.append("## 配对自助法（12% − 18%）")
    lines.append("")
    lines.append("| 窗口 | 指标 | Δ | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---|---:|---|---:|")
    for row in boot_rows:
        lines.append(
            "| {window} | {metric} | {delta:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {prob_positive:.3f} |".format(
                **row
            )
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(summary.to_string(index=False))
    print()
    print(boot.to_string(index=False))
    print(f"report: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
