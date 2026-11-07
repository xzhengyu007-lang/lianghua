"""Retry raw downloads for the PIT pool until every symbol has a cache.

Tencent returns HTTP 501 intermittently for individual symbols. A single pass
loses those symbols to transient failures, so this driver re-scans the missing
set after each pass and stops only when nothing is left or two consecutive
passes make no progress.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
from quant_system.data import TencentProvider, normalize_symbol, pit_universe

CACHE = Path('data/cache')
MAX_PASSES = 6


def missing() -> list[str]:
    names = [normalize_symbol(s) for s in pit_universe()]
    return [s for s in names if not (CACHE / f'{s}_raw.parquet').exists()]


def main() -> int:
    failures: dict[str, str] = {}
    stalled = 0
    for pass_no in range(1, MAX_PASSES + 1):
        todo = missing()
        print(f'=== pass {pass_no}: {len(todo)} missing', flush=True)
        if not todo:
            break
        provider = TencentProvider(CACHE, pause=1.5, retries=3, timeout=25)
        before = len(todo)
        for i, sym in enumerate(todo, 1):
            try:
                result = provider.download(sym, count=1800, adjust='raw')
                print(f'[{i:03d}/{len(todo)}] {sym}: {result.rows} rows {result.start}..{result.end}', flush=True)
                failures.pop(sym, None)
            except Exception as exc:  # noqa: BLE001 - provider errors are opaque
                failures[sym] = str(exc)[:200]
                print(f'[{i:03d}/{len(todo)}] {sym}: FAILED {str(exc)[:90]}', flush=True)
            time.sleep(0.8)
        after = len(missing())
        print(f'=== pass {pass_no} done: {before} -> {after} missing', flush=True)
        stalled = stalled + 1 if after >= before else 0
        Path('data/raw_pit_failures.json').write_text(
            json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8'
        )
        if stalled >= 2:
            print('no progress in two passes; stopping', flush=True)
            break
        time.sleep(20)
    print(f'finished. missing={len(missing())} failures={len(failures)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
