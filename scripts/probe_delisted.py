"""Probe the kline endpoint for many codes and classify by last bar date.

Tencent still serves delisted names, so a code-range sweep can recover a
"point in time" universe: any symbol whose last bar falls inside our window is
a name a survivor-only pool would have silently dropped.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
CACHE = Path("data/probe_cache.json")


def probe(symbol: str, session: requests.Session) -> tuple[str, int, str | None]:
    params = {"param": f"{symbol},day,,,5,qfq"}
    for attempt in range(3):
        try:
            r = session.get(URL, params=params, timeout=10)
            if r.status_code == 501:
                time.sleep(1.5)
                continue
            payload = r.json()
            node = (payload.get("data") or {}).get(symbol) or {}
            rows = node.get("qfqday") or node.get("day") or []
            if not rows:
                return symbol, 0, None
            return symbol, len(rows), str(rows[-1][0])
        except Exception:
            time.sleep(1.0 + attempt)
    return symbol, -1, None


def main() -> int:
    ranges = []
    for lo, hi in (
        (600000, 604000),
        (605000, 606000),
        (688000, 689000),
        (1, 3000),
        (300000, 301000),
    ):
        ranges.append((lo, hi))
    codes = []
    for lo, hi in ranges:
        prefix = "sh" if lo >= 600000 else "sz"
        codes.extend(f"{prefix}{code:06d}" for code in range(lo, hi))

    existing = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = [c for c in codes if c not in existing]
    print(f"probing {len(todo)} of {len(codes)} codes", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(probe, c, session): c for c in todo}
        for future in futures:
            symbol, bars, last = future.result()
            existing[symbol] = {"bars": bars, "last": last}
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(todo)}", flush=True)
                CACHE.parent.mkdir(parents=True, exist_ok=True)
                CACHE.write_text(json.dumps(existing), encoding="utf-8")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(existing), encoding="utf-8")
    print("saved", CACHE, len(existing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
