"""Generate daily signals for every strategy, plus a paper-account snapshot."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import load_panel, load_symbol, trim_panel  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.signals import latest_signal, save_signal  # noqa: E402


CASES = {
    "low_vol_core": dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20),
    "low_volatility": dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20),
    "momentum": dict(top_n=6, lookback=60, trend_window=120, max_per_sector=2, rebalance_days=20),
}


def main() -> int:
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.9)
    benchmark = load_symbol("sh000300", cache, adjust="")["close"]
    output = ROOT / "reports" / "signals_latest"
    output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, params in CASES.items():
        signal = latest_signal(
            panel, name, params=params, benchmark=benchmark, cash=1_000_000,
                )
        target = output / name
        save_signal(signal, target)
        summary[name] = {
            "as_of": str(signal.as_of.date()),
            "regime_on": signal.regime_on,
            "gross_exposure": round(signal.gross_exposure, 4),
            "positions": int(len(signal.rows)),
            "warnings": signal.warnings,
            "symbols": list(signal.target_weights.index),
        }
        print(f"== {name} ==")
        print(signal.rows.to_string(index=False) if not signal.rows.empty else "(all cash)")
        if not signal.watchlist.empty:
            print("watchlist:", signal.watchlist.to_string(index=False))
        print()
    write_json(summary, output / "summary.json")
    print(f"signals: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
