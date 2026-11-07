"""Does a sector cap help out of sample once the harmful regime switch is off?

The cap is chosen inside each training window from {1, 2, 3, 5} and applied
unchanged to the untouched test window.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import BROAD_UNIVERSE, BacktestConfig, ResearchConfig, RiskConfig  # noqa: E402
from quant_system.data import load_panel, load_symbol, trim_panel  # noqa: E402
from quant_system.html_report import write_dashboard  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.report import write_json  # noqa: E402


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    bench = load_symbol("sh000300", cache, adjust="")["close"]
    research = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
    out = ROOT / "reports" / "oos_sector_cap_v2"
    out.mkdir(parents=True, exist_ok=True)

    grid = {
        "max_per_sector": [1, 2, 3, 5],
        "top_n": [8],
        "vol_window": [60],
        "trend_window": [120],
        "rebalance_days": [20],
    }
    result = walk_forward(
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
    result.windows.to_csv(out / "windows.csv", index=False, encoding="utf-8-sig")
    result.audit.to_csv(out / "audit.csv", index=False, encoding="utf-8-sig")
    result.equity.to_frame("equity").to_csv(out / "equity.csv")
    summary = result.summary()
    write_json(
        {
            "metrics": summary,
            "selections": result.selections,
            "params": {k: v for k, v in grid.items()},
            "overlay": {"use_regime": False, "use_vol_target": True},
            "note": "sector cap chosen inside each training window only",
        },
        out / "metrics.json",
    )

    # Fixed-cap control: same walk-forward with the cap pinned at 2.
    fixed = walk_forward(
        panel,
        "low_vol_core",
        {k: [v] for k, v in {
            "max_per_sector": 2, "top_n": 8, "vol_window": 60,
            "trend_window": 120, "rebalance_days": 20,
        }.items()},
        benchmark=bench,
        research=research,
        backtest=BacktestConfig(initial_cash=1_000_000),
        risk=RiskConfig(),
        portfolio_kwargs={"use_regime": False},
        verbose=False,
    )
    fixed_summary = fixed.summary()
    write_json(
        {"metrics": fixed_summary, "cap": 2},
        out / "fixed_cap_2.json",
    )
    fixed.windows.to_csv(out / "fixed_cap_2_windows.csv", index=False, encoding="utf-8-sig")

    def line(name, s, w):
        return (
            f"{name:18s} oos={s['total_return']:+.2%} sharpe={s['sharpe']:.2f} "
            f"mdd={s['max_drawdown']:.2%} ir={s['information_ratio']:.2f} "
            f"hit={int((w['test_return'] > 0).sum())}/{len(w)}"
        )

    print(line("tuned cap", summary, result.windows))
    print(line("fixed cap=2", fixed_summary, fixed.windows))
    print(result.windows[["test_start", "test_end", "test_return"]].to_string(index=False))

    bench_eq = bench.reindex(result.equity.index).ffill().dropna()
    bench_eq = bench_eq / bench_eq.iloc[0] * result.equity.iloc[0]
    write_dashboard(
        out / "dashboard.html",
        title="Sector cap · out-of-sample (regime switch off)",
        equity=result.equity,
        metrics=summary,
        benchmark_equity=bench_eq,
        windows=result.windows,
        notes=[
            "行业上限在训练窗口内从 1/2/3/5 中选择，测试窗口不调整。",
            "基准趋势开关关闭（样本外证伪）。",
            "对照组：上限固定为 2，不参与选择。",
        ],
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())