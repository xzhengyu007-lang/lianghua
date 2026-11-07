"""Rebuild a symbol's qfq cache from hfq rescaled to the latest raw close.

Mixing qfq and hfq segments is unsafe: their anchors differ by a constant that
is not always uniform across corporate actions, which creates artificial jumps
at the splice. The whole series must come from one adjustment convention.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.data import (  # noqa: E402
    OHLCV_COLUMNS,
    SHARES_PER_VOLUME_UNIT,
    TencentProvider,
    normalize_symbol,
)


def main(symbols: list[str], count: int = 1800) -> int:
    cache = ROOT / "data" / "cache"
    provider = TencentProvider(cache, pause=2.0)
    for symbol in symbols:
        sym = normalize_symbol(symbol)
        hfq_path = cache / f"{sym}_hfq.parquet"
        if not hfq_path.exists():
            provider.download(sym, count=count, adjust="hfq", force=True)
        hfq = pd.read_parquet(hfq_path)
        raw_path = cache / f"{sym}_raw.parquet"
        if raw_path.exists():
            raw = pd.read_parquet(raw_path)
            anchor_date = raw["close"].dropna().index[-1]
            anchor_price = float(raw["close"].dropna().iloc[-1])
        else:
            qfq_old = pd.read_parquet(cache / f"{sym}_qfq.parquet")
            anchor_date = qfq_old["close"].dropna().index[-1]
            anchor_price = float(qfq_old["close"].dropna().iloc[-1])
        if anchor_date not in hfq.index:
            raise SystemExit(f"{sym}: anchor {anchor_date} missing from hfq")
        factor = anchor_price / float(hfq.loc[anchor_date, "close"])
        rebuilt = hfq.copy()
        for column in OHLCV_COLUMNS[:-1]:
            rebuilt[column] = rebuilt[column] * factor
        rebuilt["volume"] = rebuilt["volume"] * 1.0  # already in lots
        rebuilt = rebuilt.loc[:, OHLCV_COLUMNS].sort_index()
        negative = int((rebuilt[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
        worst = float(rebuilt["close"].pct_change(fill_method=None).abs().max())
        rebuilt.to_parquet(cache / f"{sym}_qfq.parquet")
        meta = {
            "symbol": sym,
            "adjust": "qfq_rebuilt_from_hfq",
            "source": "hfq",
            "anchor_date": str(anchor_date.date()),
            "anchor_price": anchor_price,
            "rows": int(len(rebuilt)),
            "start": str(rebuilt.index.min().date()),
            "end": str(rebuilt.index.max().date()),
            "pages": 0,
            "warnings": [{"type": "rebuilt_from_hfq", "factor": factor}],
        }
        (cache / f"{sym}_qfq.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"[rebuilt] {sym}: rows={len(rebuilt)} factor={factor:.6f} "
            f"negatives={negative} max_daily_move={worst:.2%}"
        )
    return 0


if __name__ == "__main__":
    targets = sys.argv[1:] or ["sh601919", "sh600188", "sh601088", "sh601225"]
    raise SystemExit(main(targets))