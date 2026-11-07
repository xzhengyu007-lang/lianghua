"""Volatility-target calibration: why does a 12% target realise 13.4%?

Section 23.47 showed ``--target-vol 0.12`` cuts drawdown in all three windows
but realised volatility lands above target in 2019-2021 (13.39% vs 12%) and in
2024-2026 (12.26%). A scaler that only de-risks can still overshoot the target
if its volatility forecast is stale: the 40-day realised window is calm right
before a vol spike, so the book stays fully invested into it.

The forecast horizon is the one knob that controls this lag and it is *not* a
signal parameter - it cannot create alpha, only change how much of the existing
alpha you carry. This script runs the same backtest at four lookbacks and two
targets, slices it into the three independent windows, and reports both the
performance and the calibration error (realised / target).

Usage::

    python scripts/run_vol_calibration.py
    python scripts/run_vol_calibration.py --n-boot 5000
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

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.bootstrap import paired_bootstrap_diff  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-05"),
}

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
LOOKBACKS = (20, 40, 80, 120)
TARGETS = (0.12, 0.18)
BASELINE_LOOKBACK = 40

METRICS = ("total_return", "cagr", "annual_volatility", "sharpe", "max_drawdown", "calmar")


def run_variant(panel, target: float, lookback: int):
    strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
    raw = strategy.generate(panel)
    weights = build_portfolio(
        raw,
        panel,
        risk=RiskConfig(target_annual_vol=target, vol_lookback=lookback),
        use_regime=False,
        use_vol_target=True,
    )
    result = Backtester(BacktestConfig()).run(panel, weights)
    return result, raw, weights


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "vol_calibration"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _ = align_panel(panel)
    print(f"panel: {len(aligned)} symbols", flush=True)

    rows: list[dict] = []
    returns_by_key: dict[tuple, pd.Series] = {}

    for target in TARGETS:
        for lookback in LOOKBACKS:
            label = f"target{int(target * 100)}_lb{lookback}"
            print(f"running {label}...", flush=True)
            result, raw, weights = run_variant(aligned, target, lookback)
            gross_raw = raw.sum(axis=1).replace(0.0, float("nan"))
            scale = (weights.sum(axis=1) / gross_raw).dropna()
            for window, (start, end) in WINDOWS.items():
                sliced = slice_result(result, start=start, end=end)
                m = sliced.metrics
                gross = weights.sum(axis=1).loc[start:end]
                invested = gross > 1e-9
                # The continuous run is still in cash for the first ~180 bars of
                # the panel (the trend/vol windows need warm-up), so a full-window
                # volatility mixes zero-return days into the calibration.  Report
                # the invested-day figure as well; that is the one that answers
                # "does the scaler hit its target while it is holding something".
                returns = sliced.returns.reindex(gross.index).fillna(0.0)
                invested_returns = returns[invested]
                invested_vol = (
                    float(invested_returns.std(ddof=0) * np.sqrt(252))
                    if len(invested_returns) > 20
                    else float("nan")
                )
                years = max(len(gross) / 252.0, 1e-9)
                mean_equity = float(sliced.equity.mean())
                trades = sliced.trades
                cost_bps = turnover_bps = float("nan")
                if len(trades) and mean_equity > 0:
                    cost = float((trades["fees"].sum() + trades["slippage"].sum()))
                    cost_bps = cost / mean_equity / years * 1e4
                    turnover_bps = float(trades["notional"].sum()) / mean_equity / years * 1e4
                row = {
                    "target": target,
                    "lookback": lookback,
                    "window": window,
                    "label": label,
                    "annual_cost_bps": cost_bps,
                    "annual_turnover_bps": turnover_bps,
                    "annual_turnover_x": turnover_bps / 1e4,
                    "vol_ratio": float(m.annual_volatility / target),
                    "invested_share": float(invested.mean()),
                    "invested_vol": invested_vol,
                    "vol_ratio_invested": invested_vol / target,
                    "avg_scale": float(scale.loc[start:end].mean()) if len(scale.loc[start:end]) else float("nan"),
                    "days_scaled_down": float((scale.loc[start:end] < 0.999).mean())
                    if len(scale.loc[start:end]) else float("nan"),
                }
                row.update({name: float(getattr(m, name)) for name in METRICS})
                rows.append(row)
                returns_by_key[(target, lookback, window)] = sliced.returns

    panel_df = pd.DataFrame(rows)

    boot_rows: list[dict] = []
    for target in TARGETS:
        for lookback in LOOKBACKS:
            if lookback == BASELINE_LOOKBACK:
                continue
            for window in WINDOWS:
                alt = returns_by_key[(target, lookback, window)]
                base = returns_by_key[(target, BASELINE_LOOKBACK, window)]
                common = alt.index.intersection(base.index)
                boot = paired_bootstrap_diff(
                    alt.loc[common], base.loc[common], n_boot=args.n_boot, seed=args.seed
                )
                boot_rows.append(
                    {
                        "target": target,
                        "lookback": lookback,
                        "vs_lookback": BASELINE_LOOKBACK,
                        "window": window,
                        "delta_sharpe": float(boot.observed),
                        "low": float(boot.low),
                        "high": float(boot.high),
                        "p_gt_zero": float(boot.prob_positive),
                    }
                )
                print(
                    f"  {target:.2f} lb{lookback} vs lb{BASELINE_LOOKBACK} {window}: "
                    f"dSharpe {boot.observed:+.3f} [{boot.low:+.3f}, {boot.high:+.3f}]",
                    flush=True,
                )

    boot_df = pd.DataFrame(boot_rows)
    panel_df.to_csv(out / "panel.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(out / "bootstrap.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "params": PARAMS,
                "lookbacks": list(LOOKBACKS),
                "targets": list(TARGETS),
                "windows": WINDOWS,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "panel": rows,
                "bootstrap": boot_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = ["# 波动目标校准：预测窗口是滞后源吗", ""]
    for target in TARGETS:
        lines.append(f"## 目标波动 {int(target * 100)}%")
        lines.append("")
        lines.append("| 窗口 | vol_lookback | 累计收益 | 实现波动 | 持仓日/目标 | Sharpe | 最大回撤 | 平均缩放 | 年化换手 | 年化成本 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        sub = panel_df[panel_df["target"] == target]
        for _, row in sub.iterrows():
            lines.append(
                "| {window} | {lookback} | {total_return:+.2%} | "
                "{annual_volatility:.2%} | {vol_ratio_invested:.2f} | {sharpe:.3f} | "
                "{max_drawdown:.2%} | {avg_scale:.3f} | {annual_turnover_x:.1f}x | "
                "{annual_cost_bps:.0f}bps |".format(**row.to_dict())
            )
        lines.append("")
    if not boot_df.empty:
        lines.append("## 配对自助：换窗口 vs 默认 40 日")
        lines.append("")
        lines.append("| 目标波动 | 窗口 | vol_lookback | ΔSharpe | 95% 区间 | P(Δ>0) |")
        lines.append("|---:|---|---:|---:|---|---:|")
        for _, row in boot_df.iterrows():
            lines.append(
                "| {target:.0%} | {window} | {lookback} | {delta_sharpe:+.3f} | "
                "[{low:+.3f}, {high:+.3f}] | {p_gt_zero:.2f} |".format(**row.to_dict())
            )
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n" + panel_df.to_string(index=False))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
