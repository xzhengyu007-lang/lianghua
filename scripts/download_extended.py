"""Resume an interrupted long-history download instead of starting over.

RESEARCH_LOG 23.123 established that extending the sample needs raw 4000-bar
history for 128 survivors, and that the provider throttles hard: the batch died at
26 symbols and a 5-bar probe still returned 501 minutes later. The existing
download command is all-or-nothing per invocation, so every retry restarts from
symbol one and re-burns the throttle budget on names that already have data.

This walks the survivor list, skips anything already deep enough, tolerates 501s by
sleeping and retrying on a later pass, and is safe to stop and restart at any point.

Usage::

    python scripts/download_extended.py --dry-run
    python scripts/download_extended.py --target-rows 4000 --passes 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import TencentProvider, normalize_symbol  # noqa: E402


def depth(cache_dir: Path, symbol: str) -> int:
    """Bars already cached for this symbol, raw or rebuilt, 0 if absent."""
    best = 0
    for suffix in ("_raw.parquet", "_tr.parquet", "_qfq.parquet"):
        path = cache_dir / (symbol + suffix)
        if not path.exists():
            continue
        try:
            import pyarrow.parquet as pq

            best = max(best, pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            continue
    return best


def plan(cache_dir: Path, target_rows: int) -> tuple[list[str], list[str]]:
    symbols = [normalize_symbol(s) for s in BROAD_UNIVERSE]
    have, need = [], []
    for symbol in symbols:
        (have if depth(cache_dir, symbol) >= target_rows else need).append(symbol)
    return have, need


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache_ext"))
    parser.add_argument("--target-rows", type=int, default=4000)
    parser.add_argument("--passes", type=int, default=4, help="resume passes over the remainder")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--backoff", type=float, default=90.0, help="seconds to wait after a 501 burst")
    parser.add_argument("--burst", type=int, default=5, help="consecutive failures before backing off")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "reports" / "download_extended"))
    args = parser.parse_args(argv)

    cache = Path(args.cache_dir)
    have, need = plan(cache, args.target_rows)

    print("target %d rows | already deep enough: %d | remaining: %d"
          % (args.target_rows, len(have), len(need)))
    if args.dry_run:
        print("dry run; would fetch:", " ".join(need[:20]), "..." if len(need) > 20 else "")
        return 0

    cache.mkdir(parents=True, exist_ok=True)
    provider = TencentProvider(cache, pause=args.pause, retries=2, timeout=25)
    log = []
    for attempt in range(1, args.passes + 1):
        pending = [s for s in need if depth(cache, s) < args.target_rows]
        if not pending:
            break
        print("pass %d: %d pending" % (attempt, len(pending)), flush=True)
        streak = 0
        for symbol in pending:
            if depth(cache, symbol) >= args.target_rows:
                continue
            try:
                frame = provider.download(symbol, count=args.target_rows, adjust="", force=True)
                rows = len(frame)
                streak = 0
                print("  %s -> %d rows from %s" % (symbol, rows, frame.index.min().date()), flush=True)
                log.append({"symbol": symbol, "rows": rows, "pass": attempt, "ok": True})
            except Exception as exc:
                streak += 1
                message = str(exc)[:120]
                print("  %s FAILED (%d) %s" % (symbol, streak, message), flush=True)
                log.append({"symbol": symbol, "pass": attempt, "ok": False, "error": message})
                if streak >= args.burst:
                    print("  backing off %.0fs" % args.backoff, flush=True)
                    time.sleep(args.backoff)
                    streak = 0
            time.sleep(args.pause)

    have, need = plan(cache, args.target_rows)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"target_rows": args.target_rows, "have": len(have), "need": len(need),
                    "log": log}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\nfinal: %d of %d at depth >= %d" % (len(have), len(have) + len(need), args.target_rows))
    return 0 if not need else 1


if __name__ == "__main__":
    raise SystemExit(main())
