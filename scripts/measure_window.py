"""Where would the research window land if the extended history were used?

RESEARCH_LOG 23.121-23.124 established that the 2019-06 start is a download
parameter, not a data limit, and that raw 4000-bar history reaches 2010. What was
never measured is the thing that actually matters: the window is set by
_research_panel -- 95% coverage of the 128 survivors -- so a handful of late
listers can hold the start far later than the data allows.

This replicates that rule on the extended cache and prints the answer next to the
production window.

Usage::

    python scripts/measure_window.py --cache-dir data/cache_ext
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import normalize_symbol, trim_panel, usable_window  # noqa: E402


def load_panel(cache: Path, symbols) -> dict:
    """Oldest-and-longest frame per symbol; the window only needs date coverage."""
    panel = {}
    for symbol in symbols:
        best = None
        # hfq is the only adjustment that survives a 4000-bar range (§23.130);
        # qfq from the provider goes non-positive on long windows.
        for suffix in ("_hfq.parquet", "_raw.parquet", "_tr.parquet", "_qfq.parquet"):
            path = cache / (symbol + suffix)
            if not path.exists():
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            if best is None or len(frame) > len(best):
                best = frame
        if best is not None and "close" in best.columns:
            panel[symbol] = best.loc[:, ["close"]]
    return panel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache_ext"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "window_measure" / "summary.json"))
    args = parser.parse_args(argv)

    cache = Path(args.cache_dir)
    symbols = [normalize_symbol(s) for s in BROAD_UNIVERSE]
    panel = load_panel(cache, symbols)
    print("anchor symbols with data: %d / %d" % (len(panel), len(symbols)), flush=True)
    if len(panel) < 100:
        print("not enough coverage to measure; download more first")
        return 2

    trimmed = trim_panel(panel, min_coverage=0.95)
    start, end = usable_window(trimmed, min_active_ratio=0.95)

    # The 95% rule is what holds the start back: a handful of names listed after
    # 2015, so 95% of the cross-section does not exist before them. Relaxing the
    # ratio buys history at the cost of a narrower early cross-section (23.93:
    # Sharpe falls to ~0.39 at 60 names), so the trade is worth tabulating.
    # Sweep on the UNTRIMMED panel: trim_panel calls usable_window with its own
    # default of 0.9 and clips the index first, so sweeping the trimmed panel is
    # guaranteed to be flat.
    sweep = []
    for ratio in (0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.40):
        try:
            s, e = usable_window(panel, min_active_ratio=ratio)
        except Exception:
            continue
        names = sum(1 for f in panel.values() if f.index.min() <= s)
        sweep.append({
            "ratio": ratio,
            "start": str(s.date()),
            "years": round((e - s).days / 365.25, 2),
            "names_at_start": names,
            "share_names": round(names / len(panel), 3),
        })

    firsts = sorted(str(f.index.min().date()) for f in panel.values())
    active_at_start = sum(1 for f in panel.values() if f.index.min() <= start)

    def span(frame):
        return str(frame.index.min().date()), str(frame.index.max().date())

    payload = {
        "cache": str(cache),
        "anchor_symbols": len(symbols),
        "symbols_with_data": len(panel),
        "kept_after_trim": len(trimmed),
        "window_start": str(start.date()),
        "window_end": str(end.date()),
        "years": round((end - start).days / 365.25, 2),
        "active_at_start": active_at_start,
        "earliest_first_bar": firsts[0] if firsts else None,
        "median_first_bar": firsts[len(firsts) // 2] if firsts else None,
        "late_listers": [s for s, f in
                         ((s, f) for s, f in panel.items() if f.index.min() > start)][:20],
        "sweep": sweep,
        "note": "窗口由 95% 覆盖率规则决定，不是由最长的那只票决定。",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("window: %s .. %s  (%.2f years)" % (payload["window_start"], payload["window_end"],
                                              payload["years"]))
    print("kept %d/%d after trim | active at start %d | median first bar %s"
          % (payload["kept_after_trim"], len(panel), active_at_start,
             payload["median_first_bar"]))
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
