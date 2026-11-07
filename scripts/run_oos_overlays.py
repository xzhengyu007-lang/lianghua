"""Out-of-sample comparison of portfolio overlays (fixed parameters)."""

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

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
VARIANTS = {
    "regime_and_vol": dict(use_regime=True, use_vol_target=True),
    "vol_only": dict(use_regime=False, use_vol_target=True),
    "regime_only": dict(use_regime=True, use_vol_target=False),
    "raw_weights": dict(use_regime=False, use_vol_target=False),
}


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    bench = load_symbol("sh000300", cache, adjust="")["close"]
    research = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
    out = ROOT / "reports" / "oos_overlays"
    out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for name, flags in VARIANTS.items():
        result = walk_forward(
            panel,
            "low_vol_core",
            {k: [v] for k, v in PARAMS.items()},
            benchmark=bench,
            research=research,
            backtest=BacktestConfig(initial_cash=1_000_000),
            risk=RiskConfig(),
            portfolio_kwargs=flags,
            verbose=False,
        )
        summary = result.summary()
        rows[name] = summary
        windows = result.windows
        print(
            f"{name:15s} oos_ret={summary['total_return']:+.2%} sharpe={summary['sharpe']:.2f} "
            f"mdd={summary['max_drawdown']:.2%} ir={summary['information_ratio']:.2f} "
            f"hit={int((windows['test_return'] > 0).sum())}/{len(windows)}"
        )
        result.windows.to_csv(out / f"{name}_windows.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "params": PARAMS,
            "research": research.__dict__,
            "variants": rows,
            "note": "fixed parameters, only the overlay design differs; test windows never tuned",
        },
        out / "summary.json",
    )
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
