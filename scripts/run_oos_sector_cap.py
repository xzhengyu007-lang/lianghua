"""Out-of-sample selection experiment for the sector cap.

The sector limit is the one knob that looked strong in full-sample sweeps.
This script chooses it only inside each training window and then reports the
untouched test windows, so the result is evidence rather than a fitted number.
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
from quant_system.research import walk_forward  # noqa: E402
from quant_system.report import write_json  # noqa: E402


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = load_panel(list(BROAD_UNIVERSE), cache)
    panel = trim_panel(panel, min_coverage=0.8)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"]
    print(f"panel: {len(panel)} symbols", flush=True)

    grid = {
        "max_per_sector": [1, 2, 3, 5],
        "top_n": [8],
        "vol_window": [60],
        "trend_window": [120],
        "rebalance_days": [20],
    }
    research = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
    result = walk_forward(
        panel,
        "low_vol_core",
        grid,
        benchmark=benchmark,
        research=research,
        backtest=BacktestConfig(initial_cash=1_000_000),
        risk=RiskConfig(),
    )
    output = ROOT / "reports" / "oos_sector_cap"
    output.mkdir(parents=True, exist_ok=True)
    result.windows.to_csv(output / "windows.csv", index=False, encoding="utf-8-sig")
    result.audit.to_csv(output / "audit.csv", index=False, encoding="utf-8-sig")
    result.equity.to_frame("equity").to_csv(output / "equity.csv")
    write_json(
        {
            "metrics": result.summary(),
            "selections": result.selections,
            "research": research.__dict__,
            "note": "sector cap chosen inside each training window only",
        },
        output / "metrics.json",
    )

    from quant_system.html_report import write_dashboard

    bench_equity = benchmark.reindex(result.equity.index).ffill()
    bench_equity = bench_equity / bench_equity.dropna().iloc[0] * result.equity.iloc[0]
    write_dashboard(
        output / "dashboard.html",
        title="Out-of-sample sector-cap selection",
        equity=result.equity,
        metrics=result.summary(),
        benchmark_equity=bench_equity,
        windows=result.windows,
        notes=[
            "行业上限只在训练窗口内选择，测试窗口不重新选参。",
            "候选：1 / 2 / 3 / 5 只同行业上限。",
            "最后一段测试窗口内沪深 300 始终在 60 日均线下方，风控开关正确切到空仓，"
            "因此该窗口收益为 0 且没有成交（audit.cash_only=True），这不是数据缺失。",
        ],
    )
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
    print(result.windows.to_string(index=False))
    print(f"report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())