"""Persist rebuilt total-return series next to the provider cache.

Rebuilding 128 symbols takes ~10s and is deterministic.  Writing the result
also makes the correction auditable: the sidecar records every corporate
action the fit found, so a later reader can see exactly why a factor changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from quant_system.adjustment import rebuild_frame
from quant_system.config import BROAD_UNIVERSE
from quant_system.data import normalize_symbol

CACHE = ROOT / "data" / "cache"
# Raw downloads live in two places: the broad universe was verified into
# ``cache_rawcheck`` (a side-by-side copy), while ``download --adjust raw``
# writes straight into ``data/cache``.  Search both so one builder covers the
# live and the delisted pools instead of silently reporting "missing raw".
RAW_DIRS = (ROOT / "data" / "cache", ROOT / "data" / "cache_rawcheck")


def _raw_path(sym: str) -> Path | None:
    for directory in RAW_DIRS:
        candidate = directory / f"{sym}_raw.parquet"
        if candidate.exists():
            return candidate
    return None


def build(symbols: list[str], *, force: bool = False) -> dict:
    written = 0
    skipped = 0
    failed: list[tuple[str, str]] = []
    warnings: list[tuple[str, list[str]]] = []
    for symbol in symbols:
        sym = normalize_symbol(symbol)
        target = CACHE / f"{sym}_tr.parquet"
        raw_path = _raw_path(sym)
        qfq_path = CACHE / f"{sym}_qfq.parquet"
        if target.exists() and not force:
            skipped += 1
            continue
        if raw_path is None or not qfq_path.exists():
            failed.append((sym, "missing raw or qfq cache"))
            continue
        raw = pd.read_parquet(raw_path)
        qfq = pd.read_parquet(qfq_path)
        raw.index = pd.to_datetime(raw.index)
        qfq.index = pd.to_datetime(qfq.index)
        try:
            adjusted, fit = rebuild_frame(raw, qfq, symbol=sym)
        except (ValueError, KeyError) as exc:
            failed.append((sym, str(exc)[:120]))
            continue
        adjusted.to_parquet(target)
        sidecar = {
            "symbol": sym,
            "source": "raw+qfq piecewise-affine rebuild",
            "rows": int(len(adjusted)),
            "start": str(adjusted.index.min().date()),
            "end": str(adjusted.index.max().date()),
            "segments": len(fit.segments),
            "events": [
                {
                    "date": str(event.date.date()),
                    "split": float(event.split),
                    "cash": float(event.cash),
                    "raw_price": float(event.raw_price),
                    "factor_ratio": float(event.factor_ratio),
                }
                for event in fit.events
            ],
            "max_relative_residual": float(fit.max_relative_residual),
            "max_abs_residual_edge": float(
                fit.diagnostics.get("max_abs_residual_edge", float("nan"))
            ),
            "problems": fit.problems,
        }
        (CACHE / f"{sym}_tr.meta.json").write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if fit.problems:
            warnings.append((sym, fit.problems))
        written += 1
        print(f"{sym}: {len(adjusted)} rows, {len(fit.events)} events", flush=True)
    return {
        "written": written,
        "skipped": skipped,
        "failed": failed,
        "warnings": warnings,
    }


def main() -> int:
    if "--pit" in sys.argv:
        from quant_system.data import pit_universe

        symbols = sorted(normalize_symbol(s) for s in pit_universe())
    else:
        symbols = sorted(normalize_symbol(s) for s in BROAD_UNIVERSE)
    force = "--force" in sys.argv
    payload = build(symbols, force=force)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not payload["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
