"""Does the 12% risk budget survive a harsh cost model?

Section 23.55 showed the published 18% configuration keeps Sharpe 1.075 /
1.032 / 0.922 under a harsh cost scenario, and 23.47 showed 12% trades a little
Sharpe for drawdown. Both were measured separately, so "is 12% better when
costs bite" was answered by comparing two numbers from two runs.

This script runs the four cost scenarios x two risk budgets, slices every run
into the three independent windows and pairs 12% against 18% on the same
resampled calendar inside each (scenario, window) cell: 12 comparisons.

Usage::

    python scripts/run_cost_target_interaction.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.bootstrap import paired_bootstrap_diff  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, CostConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-05"),
}

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)

SCENARIOS = {
    "published": dict(),
    "doubled": dict(commission_rate=0.0005, slippage_bps=10.0),
    "impact40": dict(impact_coefficient_bps=40.0),
    "harsh": dict(commission_rate=0.0005, slippage_bps=20.0, impact_coefficient_bps=60.0),
}

TARGETS = {"low": 0.12, "high": 0.18}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "cost_target_interaction"))
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _ = align_panel(panel)
    print(f"panel: {len(aligned)} symbols", flush=True)

    weights_by_target = {}
    for name, target in TARGETS.items():
        strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
        weights_by_target[name] = build_portfolio(
            strategy.generate(aligned),
            aligned,
            risk=RiskConfig(target_annual_vol=target),
            use_regime=False,
            use_vol_target=True,
        )

    rows: list[dict] = []
    returns: dict[tuple, pd.Series] = {}
    for scenario, kwargs in SCENARIOS.items():
        config = BacktestConfig(costs=CostConfig(**kwargs))
        for name, weights in weights_by_target.items():
            print(f"running {scenario} / {name} ...", flush=True)
            result = Backtester(config).run(aligned, weights)
            for window, (start, end) in WINDOWS.items():
                sliced = slice_result(result, start=start, end=end)
                m = sliced.metrics
                returns[(scenario, name, window)] = sliced.returns
                rows.append(
                    {
                        "scenario": scenario,
                        "target": TARGETS[name],
                        "window": window,
                        "total_return": float(m.total_return),
                        "cagr": float(m.cagr),
                        "annual_volatility": float(m.annual_volatility),
                        "sharpe": float(m.sharpe),
                        "max_drawdown": float(m.max_drawdown),
                        "calmar": float(m.calmar),
                        "turnover": float(m.turnover),
                    }
                )

    boot: list[dict] = []
    for scenario in SCENARIOS:
        for window in WINDOWS:
            low = returns[(scenario, "low", window)]
            high = returns[(scenario, "high", window)]
            common = low.index.intersection(high.index)
            result = paired_bootstrap_diff(
                low.loc[common], high.loc[common], n_boot=args.n_boot, seed=args.seed
            )
            boot.append(
                {
                    "scenario": scenario,
                    "window": window,
                    "delta_sharpe": float(result.observed),
                    "low": float(result.low),
                    "high": float(result.high),
                    "prob_positive": float(result.prob_positive),
                }
            )
            print(
                f"{scenario:9s} {window}: dSharpe {result.observed:+.3f} "
                f"[{result.low:+.3f}, {result.high:+.3f}] P={result.prob_positive:.2f}",
                flush=True,
            )

    panel_df = pd.DataFrame(rows)
    boot_df = pd.DataFrame(boot)
    panel_df.to_csv(out / "panel.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(out / "bootstrap.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "params": PARAMS,
                "scenarios": SCENARIOS,
                "targets": TARGETS,
                "windows": WINDOWS,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "panel": rows,
                "bootstrap": boot,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = ["# 成本情景 × 风险预算：12% 与 18% 的配对比较", ""]
    lines.append("| 情景 | 窗口 | 12% Sharpe | 18% Sharpe | 12% 回撤 | 18% 回撤 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for scenario in SCENARIOS:
        for window in WINDOWS:
            low = panel_df[(panel_df.scenario == scenario) & (panel_df.window == window)
                           & (panel_df.target == TARGETS["low"])].iloc[0]
            high = panel_df[(panel_df.scenario == scenario) & (panel_df.window == window)
                            & (panel_df.target == TARGETS["high"])].iloc[0]
            lines.append(
                f"| {scenario} | {window} | {low.sharpe:.3f} | {high.sharpe:.3f} | "
                f"{low.max_drawdown:.2%} | {high.max_drawdown:.2%} |"
            )
    lines.append("")
    lines.append("## 配对自助（12% − 18%）")
    lines.append("")
    lines.append("| 情景 | 窗口 | ΔSharpe | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---|---:|---|---:|")
    for _, row in boot_df.iterrows():
        lines.append(
            "| {scenario} | {window} | {delta_sharpe:+.3f} | [{low:+.3f}, {high:+.3f}] | "
            "{prob_positive:.2f} |".format(**row.to_dict())
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
