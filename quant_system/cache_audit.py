"""Offline audit of cached price files and their download metadata.

The downloader writes a sidecar metadata file per symbol.  Those sidecars are
the only record of how many pages were stitched and whether an adjustment
rescale happened.  Without a cross-check it is easy to run research on a cache
that was silently truncated by a different ``--count``, or to quote a result
after a parquet was edited while its metadata stayed stale.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import OHLCV_COLUMNS


CACHE_FILE_RE = re.compile(
    r"^(?P<symbol>(?:sh|sz|bj)\d{6})_(?P<adjust>qfq|hfq|raw|tr)\.parquet$"
)


@dataclass
class CacheFileAudit:
    symbol: str
    adjust: str
    path: str
    rows_file: int
    rows_meta: int | None
    start_file: str | None
    end_file: str | None
    start_meta: str | None
    end_meta: str | None
    pages: int | None
    non_positive_prices: int
    duplicate_dates: int
    warnings_in_meta: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    adjustment_constant: float | None = None
    adjustment_constant_share: float | None = None
    adjustment_events: int | None = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict:
        return {**self.__dict__, "ok": self.ok}


@dataclass
class CacheAudit:
    files: list[CacheFileAudit]
    manifest_present: bool
    manifest_symbols: int | None

    @property
    def problems(self) -> list[str]:
        result: list[str] = []
        for item in self.files:
            for problem in item.problems:
                result.append(f"{item.symbol} [{item.adjust}]: {problem}")
        return result

    @property
    def ok(self) -> bool:
        return not self.problems

    def to_dict(self) -> dict:
        return {
            "files": len(self.files),
            "ok": self.ok,
            "manifest_present": self.manifest_present,
            "manifest_symbols": self.manifest_symbols,
            "problems": self.problems,
            "details": [item.to_dict() for item in self.files],
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([item.to_dict() for item in self.files])


ADJUSTMENT_PROBE_SYMBOLS = 6


def _probe_adjustment(cache_dir: Path, symbol: str, qfq: pd.DataFrame) -> tuple[float | None, float | None, int | None]:
    """Detect a purely additive (non-multiplicative) adjustment series.

    Tencent's qfq series subtracts a piecewise-constant amount from the raw
    price.  On a proportional (multiplicative) adjustment the ratio
    ``qfq / raw`` is constant between corporate actions and jumps only on an
    ex-date; on an additive series the *difference* is constant and the ratio
    drifts every day with the price.  The test is therefore the share of bars
    on which ``qfq - raw`` is unchanged from the previous bar, which is ~100%
    inside a dividend segment for additive data.
    """
    raw_path = cache_dir / f"{symbol}_raw.parquet"
    if not raw_path.exists():
        return None, None, None
    try:
        raw = pd.read_parquet(raw_path)
    except (OSError, ValueError):
        return None, None, None
    raw.index = pd.to_datetime(raw.index)
    common = qfq.index.intersection(raw.index)
    if len(common) < 20:
        return None, None, None
    q = qfq.loc[common, "close"].to_numpy(dtype=float)
    r = raw.loc[common, "close"].to_numpy(dtype=float)
    difference = q - r
    unchanged = float(np.mean(np.abs(np.diff(difference)) <= 1e-4))
    steps = int(np.sum(np.abs(np.diff(difference)) > 1e-4))
    return float(np.median(difference)), unchanged, steps


def _read_meta(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _summarize_meta_warnings(warnings: object) -> list[str]:
    if not isinstance(warnings, list):
        return []
    text: list[str] = []
    for warning in warnings:
        if isinstance(warning, dict):
            kind = str(warning.get("type", "unknown"))
            factor = warning.get("factor")
            if factor is not None:
                text.append(f"{kind} x{factor:.6f}")
            else:
                text.append(kind)
        else:
            text.append(str(warning))
    return text


def audit_cache(cache_dir: Path) -> CacheAudit:
    """Check every parquet cache file against its sidecar metadata.

    A missing parquet for a sidecar is reported; a legacy parquet without a
    sidecar is a warning only, because older caches predate metadata.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        raise FileNotFoundError(f"cache directory does not exist: {cache_dir}")

    files: list[CacheFileAudit] = []
    seen_stems: set[str] = set()
    for path in sorted(cache_dir.glob("*.parquet")):
        match = CACHE_FILE_RE.match(path.name)
        if not match:
            continue
        symbol = match.group("symbol")
        adjust = match.group("adjust")
        stem = f"{symbol}_{adjust}"
        seen_stems.add(stem)
        meta_path = cache_dir / f"{stem}.meta.json"
        meta = _read_meta(meta_path)
        problems: list[str] = []
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError) as exc:
            files.append(
                CacheFileAudit(
                    symbol=symbol,
                    adjust=adjust,
                    path=str(path),
                    rows_file=0,
                    rows_meta=None,
                    start_file=None,
                    end_file=None,
                    start_meta=None,
                    end_meta=None,
                    pages=None,
                    non_positive_prices=0,
                    duplicate_dates=0,
                    problems=[f"unreadable parquet: {exc}"],
                )
            )
            continue

        if frame.empty:
            problems.append("empty parquet")
        missing_columns = [c for c in OHLCV_COLUMNS if c not in frame.columns]
        if missing_columns:
            problems.append(f"missing columns: {missing_columns}")
        if not isinstance(frame.index, pd.DatetimeIndex):
            problems.append("index is not DatetimeIndex")
            index = pd.DatetimeIndex([])
        else:
            index = frame.index
        duplicate_dates = int(index.duplicated().sum()) if len(index) else 0
        if duplicate_dates:
            problems.append(f"{duplicate_dates} duplicate dates")
        if not index.is_monotonic_increasing:
            problems.append("index is not sorted")

        non_positive = 0
        price_columns = [c for c in OHLCV_COLUMNS[:4] if c in frame.columns]
        if price_columns:
            non_positive = int((frame[price_columns] <= 0).any(axis=1).sum())
            if non_positive:
                problems.append(f"{non_positive} non-positive price rows")

        start_file = str(index.min().date()) if len(index) else None
        end_file = str(index.max().date()) if len(index) else None
        rows_meta: int | None = None
        pages: int | None = None
        start_meta: str | None = None
        end_meta: str | None = None
        warnings_in_meta: list[str] = []

        if meta is None:
            problems.append("missing or invalid meta.json")
        else:
            try:
                rows_meta = int(meta.get("rows"))
            except (TypeError, ValueError):
                problems.append("meta.rows is missing or invalid")
            raw_pages = meta.get("pages")
            try:
                pages = int(raw_pages) if raw_pages is not None else None
            except (TypeError, ValueError):
                problems.append("meta.pages is invalid")
            start_meta = str(meta["start"]) if meta.get("start") else None
            end_meta = str(meta["end"]) if meta.get("end") else None
            warnings_in_meta = _summarize_meta_warnings(meta.get("warnings"))
            meta_symbol = meta.get("symbol")
            if meta_symbol and meta_symbol != symbol:
                problems.append(f"meta.symbol {meta_symbol!r} != {symbol!r}")
            meta_adjust = meta.get("adjust")
            expected_adjust = "" if adjust == "raw" else adjust
            rebuilt_from_hfq = (
                adjust == "qfq" and str(meta_adjust) == "qfq_rebuilt_from_hfq"
            )
            # A total-return rebuild has no provider ``adjust`` value at all;
            # its sidecar identifies itself with a ``source`` string instead.
            rebuilt_tr = adjust == "tr"
            if rebuilt_tr and not meta.get("source"):
                problems.append("total-return cache records no source")
            if (
                meta_adjust is not None
                and not rebuilt_from_hfq
                and not rebuilt_tr
                and str(meta_adjust) != expected_adjust
            ):
                problems.append(f"meta.adjust {meta_adjust!r} != {expected_adjust!r}")
            if rebuilt_from_hfq and not meta.get("source"):
                problems.append("rebuilt cache records no source adjustment")
            if rows_meta is not None and rows_meta != len(frame):
                problems.append(
                    f"row count mismatch: file={len(frame)} meta={rows_meta}"
                )
            if start_meta and start_file and start_meta != start_file:
                problems.append(
                    f"start mismatch: file={start_file} meta={start_meta}"
                )
            if end_meta and end_file and end_meta != end_file:
                problems.append(f"end mismatch: file={end_file} meta={end_meta}")
            if pages is not None and pages < 1 and not (rebuilt_from_hfq or rebuilt_tr):
                problems.append("meta.pages < 1")
            if (
                pages is not None
                and rows_meta
                and pages <= 1
                and rows_meta > 640
                and not (rebuilt_from_hfq or rebuilt_tr)
            ):
                # A single provider page holds at most 640 bars; more rows in
                # one "page" means the sidecar does not describe this download.
                problems.append("more than 640 rows but meta.pages <= 1")
            if any("mismatch" in w for w in warnings_in_meta):
                problems.append("meta records a page overlap mismatch")

        constant = share = events = None
        if adjust == "qfq" and not frame.empty and "close" in frame.columns:
            constant, share, events = _probe_adjustment(cache_dir, symbol, frame)
            if share is not None and share >= 0.95 and events is not None:
                problems.append(
                    "qfq is an additive adjustment (qfq - raw is constant on "
                    f"{share:.1%} of bars, {events} jumps); returns are distorted "
                    "and must be rebuilt with quant_system.adjustment"
                )

        files.append(
            CacheFileAudit(
                symbol=symbol,
                adjust=adjust,
                path=str(path),
                rows_file=int(len(frame)),
                rows_meta=rows_meta,
                start_file=start_file,
                end_file=end_file,
                start_meta=start_meta,
                end_meta=end_meta,
                pages=pages,
                non_positive_prices=non_positive,
                duplicate_dates=duplicate_dates,
                warnings_in_meta=warnings_in_meta,
                problems=problems,
                adjustment_constant=constant,
                adjustment_constant_share=share,
                adjustment_events=events,
            )
        )

    for meta_path in sorted(cache_dir.glob("*.meta.json")):
        match = re.match(
            r"^(?P<stem>(?:sh|sz|bj)\d{6}_(?:qfq|hfq|raw|tr))\.meta\.json$",
            meta_path.name,
        )
        if not match:
            continue
        stem = match.group("stem")
        if stem not in seen_stems:
            symbol, adjust = stem.rsplit("_", 1)
            files.append(
                CacheFileAudit(
                    symbol=symbol,
                    adjust=adjust,
                    path=str(cache_dir / f"{stem}.parquet"),
                    rows_file=0,
                    rows_meta=None,
                    start_file=None,
                    end_file=None,
                    start_meta=None,
                    end_meta=None,
                    pages=None,
                    non_positive_prices=0,
                    duplicate_dates=0,
                    problems=["meta.json has no matching parquet"],
                )
            )

    manifest_path = cache_dir / "download_manifest.json"
    manifest_present = manifest_path.exists()
    manifest_symbols: int | None = None
    if manifest_present:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            symbols = manifest.get("symbols")
            if isinstance(symbols, list):
                manifest_symbols = len(symbols)
        except (OSError, ValueError):
            manifest_symbols = None

    return CacheAudit(
        files=files,
        manifest_present=manifest_present,
        manifest_symbols=manifest_symbols,
    )
