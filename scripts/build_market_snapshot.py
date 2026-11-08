"""Market context for the daily page: index, breadth, sectors, next rebalance.

The 68k-star reference project (ZhuLinsen/daily_stock_analysis) opens its dashboard
with a market recap -- indices, advancing/declining counts, sector leaders -- before
any single-stock detail. That ordering is right: a reader wants to know what kind of
day it is before reading what to do.

This computes the same shape from data this repository already has, and says
explicitly that none of it is a signal: 23.58 found breadth timing does not work,
so breadth here is context, not an input.

Usage::

    python scripts/build_market_snapshot.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import align_panel, pit_universe, price_frame  # noqa: E402
from quant_system.sectors import sector_of  # noqa: E402

BENCHMARK = "sh000300"
CADENCE = 20


def _benchmark(cache: Path):
    path = cache / (BENCHMARK + "_raw.parquet")
    if not path.exists():
        path = cache / (BENCHMARK + "_tr.parquet")
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    close = frame["close"].dropna()
    if len(close) < 65:
        return None
    ma60 = close.rolling(60).mean()

    def chg(days):
        if len(close) <= days:
            return None
        return float(close.iloc[-1] / close.iloc[-1 - days] - 1.0)

    return {
        "symbol": BENCHMARK,
        "close": float(close.iloc[-1]),
        "as_of": str(close.index[-1].date()),
        "chg_1d": chg(1),
        "chg_5d": chg(5),
        "chg_20d": chg(20),
        "ma60": float(ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else None,
        "above_ma60": bool(close.iloc[-1] > ma60.iloc[-1]) if pd.notna(ma60.iloc[-1]) else None,
    }


def _breadth(close: pd.DataFrame):
    rets = close.pct_change(fill_method=None)
    last = rets.iloc[-1].dropna()
    ma60 = close.rolling(60).mean().iloc[-1]
    last_close = close.iloc[-1]
    above = (last_close > ma60).dropna()
    return {
        "names": int(last.shape[0]),
        "up": int((last > 0).sum()),
        "down": int((last < 0).sum()),
        "flat": int((last == 0).sum()),
        "median_ret_1d": float(last.median()),
        "above_ma60_share": float(above.mean()) if len(above) else None,
        "above_ma60_names": int(above.sum()) if len(above) else 0,
    }


def _sectors(close: pd.DataFrame):
    rets = close.pct_change(fill_method=None)
    last = rets.iloc[-1].dropna()
    window = rets.tail(20).mean()
    buckets: dict[str, list] = {}
    for symbol in last.index:
        buckets.setdefault(sector_of(symbol), []).append(symbol)
    rows = []
    for sector, symbols in buckets.items():
        if len(symbols) < 2:
            continue
        rows.append({
            "sector": sector,
            "names": len(symbols),
            "ret_1d": float(last[symbols].mean()),
            "ret_20d": float(window[symbols].mean()),
        })
    rows.sort(key=lambda r: r["ret_1d"], reverse=True)
    return rows


def _cadence(calendar: pd.DatetimeIndex):
    """No single next-rebalance date exists for the published configuration.

    The deployable book is 20 phases offset by one trading day each, so one phase
    rebalances every day. Reporting a "next rebalance" would be wrong; the useful
    fact is the cadence and how the phases interleave.
    """
    year_start = calendar[calendar.year == calendar[-1].year][0]
    phase0 = [d for i, d in enumerate(calendar[calendar >= year_start]) if i % CADENCE == 0]
    upcoming = [d for d in phase0 if d > calendar[-1]]
    return {
        "cadence_days": CADENCE,
        "phases": 20,
        "last_phase0_rebalance": str(phase0[-1].date()) if phase0 else None,
        "next_phase0_rebalance": str(upcoming[0].date()) if upcoming else None,
        "note": "20 个相位各错开 1 个交易日，因此每个交易日都有一个相位在调仓；"
                "phase0 的日期仅作示例。",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "market" / "snapshot.json"))
    args = parser.parse_args(argv)

    cache = Path(args.cache_dir)
    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), cache, "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    close = pd.DataFrame({s: f["close"] for s, f in aligned.items()}).sort_index()
    calendar = close.index

    payload = {
        "as_of": str(calendar[-1].date()),
        "index": _benchmark(cache),
        "breadth": _breadth(close),
        "sectors": _sectors(close),
        "cadence": _cadence(calendar),
        "note": "全部为对照信息，不参与信号计算；23.58 已证明广度择时无效。",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")

    b = payload["breadth"]
    print("as_of %s | up %d / down %d | above MA60 %.0f%%"
          % (payload["as_of"], b["up"], b["down"], 100 * (b["above_ma60_share"] or 0)))
    if payload["index"]:
        print("index %s %.2f (%+.2f%%)"
              % (BENCHMARK, payload["index"]["close"],
                 100 * (payload["index"]["chg_1d"] or 0)))
    print("snapshot -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
