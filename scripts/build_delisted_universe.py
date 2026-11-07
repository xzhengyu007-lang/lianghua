"""Build a point-in-time universe from real delisted names.

Rather than inventing synthetic delistings (see reports/survivorship/), this
recovers actual names that stopped trading inside the backtest window. That is
the only way to measure, rather than bound, survivorship bias.

Pipeline:
1. `data/probe_cache.json` holds the last bar date for every code in the
   relevant ranges (from scripts/probe_delisted.py).
2. Codes whose last bar falls inside the window are the delisted set.
3. Their full qfq history is downloaded like any other symbol.

The result is written to data/cache so the standard loader picks it up, and a
manifest lists exactly which names were added and when they stopped trading.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import BROAD_UNIVERSE, CACHE_DIR  # noqa: E402
from quant_system.data import TencentProvider, load_panel  # noqa: E402

WINDOW_START = pd.Timestamp("2019-06-01")
WINDOW_END = pd.Timestamp("2026-11-06")
# A name counts as delisted when it stopped printing bars well before today;
# otherwise it is simply a currently listed name that the probe saw last week.
DELISTED_BEFORE = pd.Timestamp("2026-09-01")


def main() -> int:
    probe_path = ROOT / "data" / "probe_cache.json"
    if not probe_path.exists():
        print("run scripts/probe_delisted.py first")
        return 1
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    present = set(BROAD_UNIVERSE)

    delisted = {}
    for symbol, row in probe.items():
        bars = int(row.get("bars", 0))
        last = row.get("last")
        if bars <= 0 or last is None:
            continue
        date = pd.Timestamp(last)
        if not (WINDOW_START <= date <= DELISTED_BEFORE):
            continue
        if symbol in present:
            continue
        delisted[symbol] = date

    print(f"probe codes: {len(probe)}")
    print(f"delisted inside window: {len(delisted)}")
    by_year = pd.Series([d.year for d in delisted.values()]).value_counts().sort_index()
    print(by_year.to_string())

    client = TencentProvider(CACHE_DIR)
    targets = sorted(delisted)
    print(f"downloading {len(targets)} histories", flush=True)
    results = client.download_many(targets, count=1800, retry_rounds=3)
    print(f"downloaded {len(results)}", flush=True)

    manifest = {}
    for result in results:
        frame = pd.read_parquet(result.path)
        if frame.empty:
            continue
        manifest[result.symbol] = {
            "first_bar": str(frame.index.min().date()),
            "last_bar": str(frame.index.max().date()),
            "rows": int(len(frame)),
            "probe_last_bar": str(delisted.get(result.symbol, pd.NaT).date())
            if result.symbol in delisted
            else None,
        }
    out = ROOT / "data" / "delisted_manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {out} ({len(manifest)} names)")

    # Sanity check: how many have bars inside the strategy window?
    panel = load_panel(list(manifest), CACHE_DIR)
    inside = 0
    for symbol in manifest:
        frame = panel.get(symbol)
        if frame is None:
            continue
        if frame.index.max() >= WINDOW_START:
            inside += 1
    print(f"names with bars at/after {WINDOW_START.date()}: {inside}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
