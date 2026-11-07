"""Walk-forward the point-in-time pool and a survivor-only control.

Both runs use the same dates, parameters, costs, and walk-forward windows. The
only difference is membership. This answers a narrower question than the
full-sample comparison: does the measured survivorship gap survive when the
evaluation is strictly out of sample?

Outputs go to reports/pit_walkforward/ so the default baseline report is never
overwritten.
"""

from __future__ import annotations

import json
import sys
import time
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
from quant_system.data import (  # noqa: E402
    align_panel,
    load_panel,
    load_symbol,
    pit_universe,
    trim_panel,
    usable_window,
)
from quant_system.report import write_json  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402

PARAMS = {
    "top_n": [8],
    "vol_window": [60],
    "trend_window": [120],
    "max_per_sector": [2],
    "rebalance_days": [20],
}
RESEARCH = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
BACKTEST = BacktestConfig(initial_cash=1_000_000)


def main() -> int:
    cache = ROOT / "data" / "cache"
    survivors = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(survivors, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"]
    universe = pit_universe()
    # Keep the exact same survivor cross-section in both arms. Otherwise the
    # comparison also picks up late listings that fall outside the survivor
    # coverage filter, which is not a survivorship-bias effect. Only names that
    # are absent from BROAD_UNIVERSE are added to form the PIT arm.
    survivors = {symbol: frame.loc[start:end] for symbol, frame in survivors.items()}
    delisted = {symbol: name for symbol, name in universe.items() if symbol not in BROAD_UNIVERSE}
    pit = {
        symbol: frame.loc[start:end]
        for symbol, frame in load_panel(list(survivors) + list(delisted), cache).items()
        if len(frame.loc[start:end])
    }
    print(
        f"window {start.date()}..{end.date()} survivor={len(survivors)} "
        f"pit={len(pit)} extra={len(pit) - len(survivors)}",
        flush=True,
    )

    outputs = {}
    windows = []
    out = ROOT / "reports" / "pit_walkforward"
    out.mkdir(parents=True, exist_ok=True)
    for label, panel in (("survivor-only", survivors), ("point-in-time", pit)):
        print(f"running {label}", flush=True)
        started = time.time()
        result = walk_forward(
            align_panel(panel)[0],
            "low_vol_core",
            PARAMS,
            benchmark=benchmark,
            research=RESEARCH,
            backtest=BACKTEST,
            risk=RiskConfig(),
        )
        table = result.windows.copy()
        table["pool"] = label
        windows.append(table)
        summary = {
            "pool": label,
            "symbols": len(panel),
            "window": [str(start.date()), str(end.date())],
            "params": PARAMS,
            "metrics": result.summary(),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        outputs[label] = summary
        result.equity.to_frame("equity").to_csv(out / f"equity_{label}.csv")
        write_json(summary, out / f"summary_{label}.json")
        print(json.dumps(result.summary(), ensure_ascii=False, indent=2), flush=True)

    combined = pd.concat(windows, ignore_index=True)
    combined.to_csv(out / "windows.csv", index=False, encoding="utf-8-sig")
    keys = {
        "total_return", "cagr", "annual_volatility", "sharpe",
        "max_drawdown", "exposure", "information_ratio",
    }
    comparison = pd.DataFrame(
        {
            label: {key: value for key, value in summary["metrics"].items() if key in keys}
            for label, summary in outputs.items()
        }
    ).T
    comparison.to_csv(out / "comparison.csv", encoding="utf-8-sig")
    with pd.option_context("display.width", 180):
        print(comparison.round(4).to_string())
    print("report: reports/pit_walkforward")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
