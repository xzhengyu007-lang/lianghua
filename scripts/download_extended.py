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


SOURCE_LIMIT = "2010-06-30"


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


def first_bar(cache_dir: Path, symbol: str):
    """Date of the oldest cached bar, or None."""
    for suffix in ("_raw.parquet", "_tr.parquet", "_qfq.parquet"):
        path = cache_dir / (symbol + suffix)
        if not path.exists():
            continue
        try:
            import pandas as pd

            frame = pd.read_parquet(path, columns=[])
            return frame.index.min()
        except Exception:
            continue
    return None


def is_complete(cache_dir: Path, symbol: str, target_rows: int, adjust: str = "") -> bool:
    """A cached file means the paging finished; nothing more can be fetched.

    The provider serves a fixed number of bars and then stops, so a name that
    listed in 2018 legitimately ends up with ~2000 rows. Keying completion on the
    row count re-downloads those on every pass and burns the throttle budget for
    data that does not exist.
    """
    suffix = "_%s.parquet" % (adjust or "raw")
    if (cache_dir / (symbol + suffix)).exists():
        return True
    if adjust:
        # A specific variant was requested: depth in another variant does not
        # satisfy it, or the rebuilt step would run on a half-filled cache.
        return False
    return depth(cache_dir, symbol) >= target_rows


def plan(cache_dir: Path, target_rows: int, adjust: str = "") -> tuple[list[str], list[str]]:
    symbols = [normalize_symbol(s) for s in BROAD_UNIVERSE]
    have, need = [], []
    for symbol in symbols:
        (have if is_complete(cache_dir, symbol, target_rows, adjust) else need).append(symbol)
    return have, need


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache_ext"))
    parser.add_argument("--target-rows", type=int, default=4000)
    parser.add_argument("--adjust", default="", help="'' for raw, or qfq/hfq")
    parser.add_argument("--passes", type=int, default=4, help="resume passes over the remainder")
    parser.add_argument("--pause", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--backoff", type=float, default=90.0, help="seconds to wait after a 501 burst")
    parser.add_argument("--burst", type=int, default=5, help="consecutive failures before backing off")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "reports" / "download_extended"))
    args = parser.parse_args(argv)

    cache = Path(args.cache_dir)
    have, need = plan(cache, args.target_rows, args.adjust)

    print("target %d rows | already deep enough: %d | remaining: %d"
          % (args.target_rows, len(have), len(need)))
    if args.dry_run:
        print("dry run; would fetch:", " ".join(need[:20]), "..." if len(need) > 20 else "")
        return 0

    cache.mkdir(parents=True, exist_ok=True)
    provider = TencentProvider(cache, pause=args.pause, retries=2, timeout=25)
    log = []
    for attempt in range(1, args.passes + 1):
        pending = [s for s in need if not is_complete(cache, s, args.target_rows, args.adjust)]
        if not pending:
            break
        print("pass %d: %d pending" % (attempt, len(pending)), flush=True)
        streak = 0
        for symbol in pending:
            # Must use the same predicate as plan(); depth() counts any variant and
            # would skip every symbol whose raw file is already deep, which is how
            # 38 passes once ran without a single request.
            if is_complete(cache, symbol, args.target_rows, args.adjust):
                continue
            try:
                result = provider.download(symbol, count=args.target_rows,
                                           adjust=args.adjust, force=True)
                # download() returns a DownloadResult, not a frame: it already
                # carries the row count and the span, so asking len() fails.
                rows = int(result.rows)
                streak = 0
                print("  %s -> %d rows from %s" % (symbol, rows, result.start), flush=True)
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

    have, need = plan(cache, args.target_rows, args.adjust)
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
