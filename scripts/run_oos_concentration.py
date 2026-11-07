"""Out-of-sample check of portfolio breadth and sector concentration.

The one-at-a-time plateau study (reports/robustness) showed that narrowing the
book helped a lot *in sample*: top_n 8 -> 6 lifted total return from +137% to
+208%, and relaxing the sector cap from 2 to 5 lifted it to +206%. Both knobs
are exactly the kind of parameter that can be curve-fit, so they get the same
treatment as everything else here: choose inside the training window only, then
apply unchanged to the untouched test window.

Grid: top_n in {5, 6, 8, 10} x max_per_sector in {2, 3, 5} = 12 candidates.
Control: the current default (top_n=8, cap=2) under the identical windows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import (  # noqa: E402
    BROAD_UNIVERSE,
    BacktestConfig,
    ResearchConfig,
    RiskConfig,
)
from quant_system.data import load_panel, load_symbol, trim_panel  # noqa: E402
from quant_system.html_report import write_dashboard  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.report import write_json  # noqa: E402

FIXED = {"vol_window": 60, "trend_window": 120, "rebalance_days": 20}


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    bench = load_symbol("sh000300", cache, adjust="")["close"]
    research = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
    out = ROOT / "reports" / "oos_concentration"
    out.mkdir(parents=True, exist_ok=True)

    grid = {
        **{k: [v] for k, v in FIXED.items()},
        "top_n": [5, 6, 8, 10],
        "max_per_sector": [2, 3, 5],
    }
    tuned = walk_forward(
        panel,
        "low_vol_core",
        grid,
        benchmark=bench,
        research=research,
        backtest=BacktestConfig(initial_cash=1_000_000),
        risk=RiskConfig(),
        portfolio_kwargs={"use_regime": False},
        verbose=False,
    )
    tuned.windows.to_csv(out / "windows.csv", index=False, encoding="utf-8-sig")
    tuned.audit.to_csv(out / "audit.csv", index=False, encoding="utf-8-sig")
    tuned.equity.to_frame("equity").to_csv(out / "equity.csv")

    fixed = walk_forward(
        panel,
        "low_vol_core",
        {**{k: [v] for k, v in FIXED.items()}, "top_n": [8], "max_per_sector": [2]},
        benchmark=bench,
        research=research,
        backtest=BacktestConfig(initial_cash=1_000_000),
        risk=RiskConfig(),
        portfolio_kwargs={"use_regime": False},
        verbose=False,
    )
    fixed.windows.to_csv(out / "fixed_windows.csv", index=False, encoding="utf-8-sig")

    # Also report the single-window full-sample result for a concentrated 6-name
    # book, so the OOS number can be compared with the in-sample plateau.
    from quant_system.backtest import Backtester
    from quant_system.portfolio import build_portfolio
    from quant_system.strategies import build_strategy

    start = pd.Timestamp(research.start) if research.start else panel_start(panel)
    full_rows = []
    for top_n in (5, 6, 8, 10):
        for cap in (2, 3, 5):
            params = {**FIXED, "top_n": top_n, "max_per_sector": cap}
            raw = build_strategy("low_vol_core", **params).generate(panel)
            weights = build_portfolio(
                raw, panel, benchmark=bench, risk=RiskConfig(), use_regime=False
            )
            metrics = Backtester(BacktestConfig(initial_cash=1_000_000)).run(
                panel, weights, benchmark=bench
            ).metrics.to_dict()
            full_rows.append({"top_n": top_n, "max_per_sector": cap, **metrics})
    full = pd.DataFrame(full_rows)
    full.to_csv(out / "full_sample_grid.csv", index=False, encoding="utf-8-sig")

    tuned_summary = tuned.summary()
    fixed_summary = fixed.summary()
    write_json(
        {
            "tuned": tuned_summary,
            "fixed": fixed_summary,
            "selections": tuned.selections,
            "grid": grid,
            "overlay": {"use_regime": False, "use_vol_target": True},
            "note": "top_n and sector cap chosen inside each training window only",
        },
        out / "metrics.json",
    )

    def line(name, summary, windows):
        return (
            f"{name:22s} oos={summary['total_return']:+.2%} sharpe={summary['sharpe']:.2f} "
            f"mdd={summary['max_drawdown']:.2%} ir={summary['information_ratio']:.2f} "
            f"hit={int((windows['test_return'] > 0).sum())}/{len(windows)}"
        )

    print(line("tuned breadth+cap", tuned_summary, tuned.windows))
    print(line("fixed top_n=8 cap=2", fixed_summary, fixed.windows))
    print(tuned.windows[["test_start", "test_end", "params", "test_return"]].to_string(index=False))
    print(full[["top_n", "max_per_sector", "total_return", "sharpe", "max_drawdown"]].round(4).to_string(index=False))

    bench_eq = bench.reindex(tuned.equity.index).ffill().dropna()
    bench_eq = bench_eq / bench_eq.iloc[0] * tuned.equity.iloc[0]
    write_dashboard(
        out / "dashboard.html",
        title="Breadth + sector cap · out-of-sample selection",
        equity=tuned.equity,
        metrics=tuned_summary,
        benchmark_equity=bench_eq,
        windows=tuned.windows,
        notes=[
            "top_n 与行业上限在训练窗口内联合选择，测试窗口不调整。",
            "基准趋势开关关闭；波动率目标开启。",
            "对照：固定 top_n=8、行业上限 2。",
        ],
    )
    print(f"report: {out}")
    return 0


def panel_start(panel) -> pd.Timestamp:
    from quant_system.data import usable_window

    start, _ = usable_window(panel, min_active_ratio=0.95)
    return start


if __name__ == "__main__":
    raise SystemExit(main())
