"""Market data ingestion, caching, and integrity checks.

The default provider is Tencent's public kline endpoint.  It is deliberately
isolated behind a small interface so production data can be substituted later.
Adjusted prices are used for research; raw traded prices are approximated by
the adjusted series because corporate-action data is not exposed separately.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .config import BROAD_UNIVERSE, CACHE_DIR, DATA_DIR, DEFAULT_UNIVERSE


TENCENT_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
# Tencent reports daily volume in lots (1 lot = 100 shares).
SHARES_PER_VOLUME_UNIT = 100.0


def normalize_symbol(symbol: str, *, index: bool = False) -> str:
    """Return a Tencent-style symbol such as sh600000 or sz000001."""
    raw = symbol.strip().lower()
    if raw.startswith(("sh", "sz", "bj")):
        return raw
    digits = "".join(ch for ch in raw if ch.isdigit())
    if len(digits) != 6:
        raise ValueError(f"invalid A-share symbol: {symbol!r}")
    if index and digits.startswith("000"):
        return "sh" + digits
    if digits.startswith(("6", "9")):
        return "sh" + digits
    if digits.startswith(("0", "2", "3")):
        return "sz" + digits
    if digits.startswith(("4", "8")):
        return "bj" + digits
    raise ValueError(f"cannot infer exchange for {symbol!r}")


def bare_symbol(symbol: str) -> str:
    return normalize_symbol(symbol)[2:]


@dataclass
class DownloadResult:
    symbol: str
    rows: int
    start: str
    end: str
    path: Path


class TencentProvider:
    """Download forward-adjusted daily OHLCV from Tencent Finance."""

    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        *,
        timeout: float = 25.0,
        retries: int = 6,
        pause: float = 0.8,
        session: requests.Session | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.retries = retries
        self.pause = pause
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/124 Safari/537.36"
                ),
                "Referer": "https://gu.qq.com/",
            }
        )

    def _request(
        self,
        symbol: str,
        count: int,
        adjust: str,
        end: str | None = None,
    ) -> dict:
        """Fetch up to ``count`` bars ending at ``end`` (default: latest)."""
        tail = end or ""
        params = {"param": f"{symbol},day,,{tail},{count},{adjust}"}
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(
                    TENCENT_URL, params=params, timeout=self.timeout
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("code") != 0:
                    raise RuntimeError(f"provider error: {payload.get('msg')}")
                return payload
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in (429, 501, 502, 503, 504):
                    time.sleep(3.0 + 2.0 * attempt)
                time.sleep(self.pause * (2**attempt) + 1.0 * attempt)
        raise RuntimeError(f"failed to download {symbol}: {last_error}") from last_error

    def _page(
        self,
        symbol: str,
        count: int,
        adjust: str,
        end: str | None,
    ) -> pd.DataFrame:
        payload = self._request(symbol, count, adjust, end)
        rows = self._extract_rows(payload, symbol, adjust)
        return self.parse_rows(rows) if rows else pd.DataFrame(columns=OHLCV_COLUMNS)

    @staticmethod
    def _merge_pages(pages: list[pd.DataFrame]) -> tuple[pd.DataFrame, list[dict]]:
        """Stitch pages fetched backwards in time.

        Forward-adjusted prices are anchored to the most recent bar, so a page
        downloaded in a different session can be a uniform rescale of an older
        page. Overlapping dates are therefore used to measure a scalar ratio,
        and the older block is rescaled when the ratio is consistent.
        """
        warnings: list[dict] = []
        if not pages:
            return pd.DataFrame(columns=OHLCV_COLUMNS), warnings
        merged = pages[-1].copy()
        for page in reversed(pages[:-1]):
            overlap = merged.index.intersection(page.index)
            if len(overlap) >= 3:
                ratios = (
                    merged.loc[overlap, "close"].to_numpy()
                    / page.loc[overlap, "close"].to_numpy()
                )
                median = float(np.median(ratios))
                spread = float(np.max(np.abs(ratios / median - 1.0))) if median else 0.0
                if abs(median - 1.0) > 1e-9 and spread <= 0.02:
                    # A uniform ratio across the overlap is the signature of a
                    # different forward-adjustment anchor, not bad data.
                    rescaled = page.copy()
                    for column in OHLCV_COLUMNS:
                        rescaled[column] = rescaled[column] * median
                    page = rescaled
                    warnings.append(
                        {
                            "type": "adjusted_rescale",
                            "overlap": len(overlap),
                            "factor": median,
                        }
                    )
                elif spread > 1e-6:
                    warnings.append(
                        {
                            "type": "overlap_mismatch",
                            "overlap": len(overlap),
                            "max_relative_error": spread,
                        }
                    )
            merged = pd.concat([page, merged])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        return merged.loc[:, OHLCV_COLUMNS], warnings

    @staticmethod
    def _extract_rows(payload: dict, symbol: str, adjust: str) -> list[list]:
        item = (payload.get("data") or {}).get(symbol) or {}
        keys = [f"{adjust}day", "day"] if adjust else ["day"]
        for key in keys:
            rows = item.get(key)
            if rows:
                return rows
        return []

    def download(
        self,
        symbol: str,
        *,
        count: int = 1600,
        adjust: str = "qfq",
        force: bool = False,
    ) -> DownloadResult:
        if count < 60:
            raise ValueError("count must be >= 60")
        if adjust == "raw":
            adjust = ""
        if adjust not in {"qfq", "hfq", ""}:
            raise ValueError("adjust must be qfq, hfq, raw, or empty")
        sym = normalize_symbol(symbol)
        path = self.cache_dir / f"{sym}_{adjust or 'raw'}.parquet"
        meta_path = self.cache_dir / f"{sym}_{adjust or 'raw'}.meta.json"
        if meta_path.exists() and path.exists() and not force:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = {}
            if int(meta.get("rows", 0)) >= count:
                return DownloadResult(
                    sym,
                    int(meta["rows"]),
                    meta["start"],
                    meta["end"],
                    path,
                )
        if path.exists() and not force and not meta_path.exists():
            # Legacy cache without metadata: honour it rather than silently
            # re-downloading everything on the first upgrade.
            frame = pd.read_parquet(path)
            if len(frame):
                return DownloadResult(
                    sym,
                    len(frame),
                    str(frame.index.min().date()),
                    str(frame.index.max().date()),
                    path,
                )

        pages: list[pd.DataFrame] = []
        cursor: str | None = (date.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        while True:
            page = self._page(sym, min(640, count), adjust, cursor)
            if page.empty:
                break
            pages.append(page)
            got = sum(len(p) for p in pages)
            if page.index.min() <= pd.Timestamp("2005-01-01") or got >= count:
                break
            cursor = (page.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            time.sleep(self.pause)
        if not pages:
            raise ValueError(f"no daily data returned for {sym}")
        frame, warnings = self._merge_pages(pages)
        if frame.empty:
            raise ValueError(f"no daily data returned for {sym}")
        mismatches = [w for w in warnings if w.get("type") == "overlap_mismatch"]
        if mismatches:
            # A non-uniform overlap means the pages do not describe the same
            # price series. Storing it would splice two incompatible series and
            # fabricate a jump at the seam, so refuse the download instead.
            worst = max(float(w["max_relative_error"]) for w in mismatches)
            raise ValueError(
                f"{sym}: page overlap mismatch up to {worst:.2%}; "
                "adjusted history is not internally consistent"
            )
        frame = frame.tail(count)
        # Reject a damaged adjustment series instead of caching it. Long qfq
        # pages can come back with negative prices when the anchor does not
        # span the requested history; silently storing that would poison every
        # later backtest.
        prices = frame[["open", "high", "low", "close"]]
        invalid = int((prices <= 0).any(axis=1).sum())
        if invalid:
            raise ValueError(
                f"{sym}: {invalid} non-positive price rows in adjusted history; "
                "provider adjustment is invalid for this range"
            )
        frame.to_parquet(path)
        meta = {
            "symbol": sym,
            "adjust": adjust,
            "pages": len(pages),
            "rows": int(len(frame)),
            "start": str(frame.index.min().date()),
            "end": str(frame.index.max().date()),
            "warnings": warnings,
        }
        (self.cache_dir / f"{sym}_{adjust or 'raw'}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return DownloadResult(
            sym,
            len(frame),
            str(frame.index.min().date()),
            str(frame.index.max().date()),
            path,
        )

    @staticmethod
    def parse_rows(rows: Iterable[Iterable]) -> pd.DataFrame:
        clean: list[list] = []
        for row in rows:
            if len(row) < 6:
                continue
            clean.append(list(row[:6]))
        if not clean:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        # Tencent returns date, open, close, high, low, volume.
        frame = pd.DataFrame(clean, columns=["date", "open", "close", "high", "low", "volume"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        for col in OHLCV_COLUMNS:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = (
            frame.dropna(subset=["date", "open", "high", "low", "close"])
            .drop_duplicates("date", keep="last")
            .set_index("date")
            .sort_index()
        )
        frame["volume"] = frame["volume"].fillna(0.0).clip(lower=0.0)
        return frame.loc[:, OHLCV_COLUMNS]

    def download_many(
        self,
        symbols: Iterable[str] = DEFAULT_UNIVERSE,
        *,
        count: int = 1600,
        adjust: str = "qfq",
        force: bool = False,
        retry_rounds: int = 3,
    ) -> list[DownloadResult]:
        """Download a universe, retrying transient provider failures.

        The public endpoint intermittently answers HTTP 501 under load. Failed
        symbols are parked and retried in later rounds instead of aborting the
        whole run, so a partial download makes forward progress.
        """
        pending = list(symbols)
        results: list[DownloadResult] = []
        failures: dict[str, str] = {}
        for round_index in range(max(1, retry_rounds)):
            if not pending:
                break
            failed_now: list[str] = []
            for position, symbol in enumerate(pending, start=1):
                try:
                    result = self.download(symbol, count=count, adjust=adjust, force=force)
                except (RuntimeError, ValueError, OSError) as exc:
                    failures[str(symbol)] = str(exc)
                    failed_now.append(symbol)
                    print(f"[!] {symbol}: {exc} (will retry)", flush=True)
                    continue
                results.append(result)
                failures.pop(str(symbol), None)
                print(
                    f"[r{round_index + 1} {position:03d}] {result.symbol}: {result.rows} rows "
                    f"{result.start}..{result.end}",
                    flush=True,
                )
                if position < len(pending):
                    time.sleep(self.pause)
            pending = failed_now
            if pending:
                time.sleep(5.0 * (round_index + 1))
        for symbol in pending:
            failures.setdefault(
                str(symbol),
                (cache_dir_hint := "unresolved after retries"),
            )
        if failures:
            print(f"[!] {len(failures)} symbols still failed: {sorted(failures)}", flush=True)
        return results


def infer_price_limit(symbol: str) -> float:
    """Return the normal daily price limit for a mainland-listed stock."""
    code = normalize_symbol(symbol)[2:]
    if code.startswith(("300", "301", "688", "689")):
        return 0.20
    if code.startswith(("4", "8")):
        return 0.30
    return 0.10


def tradeability(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Compute conservative buy/sell permissions at each bar's open.

    A limit-up/open gap blocks buying and a limit-down/open gap blocks selling.
    Zero volume is treated as a suspension and blocks both directions.
    """
    limit = infer_price_limit(symbol)
    prev_close = frame["close"].shift(1)
    up = frame["open"] >= prev_close * (1 + limit) - 1e-6
    down = frame["open"] <= prev_close * (1 - limit) + 1e-6
    suspended = (frame["volume"] <= 0) | prev_close.isna()
    return pd.DataFrame(
        {"can_buy": ~(suspended | up), "can_sell": ~(suspended | down)},
        index=frame.index,
    )


def load_symbol(symbol: str, cache_dir: Path = CACHE_DIR, *, adjust: str = "qfq") -> pd.DataFrame:
    """Load a cached OHLCV frame with invalid price rows removed.

    Long forward-adjusted histories from the provider can contain non-positive
    prices when the adjustment anchor does not cover the full range. Such rows
    are not tradeable prices; keeping them silently corrupts every return,
    volatility and drawdown calculation. They are dropped here and the count is
    reported by ``validate_panel`` through the coverage figures, so a damaged
    cache is visible instead of being treated as a real price.
    """
    sym = normalize_symbol(symbol)
    path = Path(cache_dir) / f"{sym}_{adjust or 'raw'}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing cached data for {sym}: run `aq download` first")
    frame = pd.read_parquet(path)
    if not isinstance(frame.index, pd.DatetimeIndex):
        frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index().loc[:, OHLCV_COLUMNS]
    prices = frame[["open", "high", "low", "close"]]
    invalid = (prices <= 0).any(axis=1) | prices.isna().all(axis=1)
    if invalid.any():
        frame = frame.loc[~invalid]
    return frame


def pit_universe() -> dict[str, str]:
    """Current broad names plus every real symbol that stopped trading in-window.

    Membership is derived from ``data/delisted_manifest.json`` (built by
    ``scripts/build_delisted_universe.py``). Readable names come from
    ``data/delisted_names_raw.json`` so the pool is reproducible offline.
    A delisted name is only eligible while it has a real bar; the availability
    mask, not this static list, enforces the point-in-time condition.
    """
    manifest_path = DATA_DIR / "delisted_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "missing data/delisted_manifest.json; run "
            "scripts/build_delisted_universe.py first"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    names: dict[str, str] = {}
    names_path = DATA_DIR / "delisted_names_raw.json"
    if names_path.exists():
        payload = json.loads(names_path.read_text(encoding="utf-8"))
        names = {symbol: info.get("name", "") for symbol, info in payload.items()}
    return {
        **BROAD_UNIVERSE,
        **{symbol: names.get(symbol) or symbol for symbol in manifest},
    }


def load_panel(
    symbols: Iterable[str] = DEFAULT_UNIVERSE,
    cache_dir: Path = CACHE_DIR,
    *,
    adjust: str = "qfq",
) -> dict[str, pd.DataFrame]:
    return {normalize_symbol(s): load_symbol(s, cache_dir, adjust=adjust) for s in symbols}


def panel_to_frame(panel: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    """Convert a symbol panel to a date x symbol matrix for one field."""
    series = []
    for symbol, frame in panel.items():
        if field not in frame:
            raise KeyError(f"{field} missing for {symbol}")
        series.append(frame[field].rename(symbol))
    if not series:
        return pd.DataFrame()
    return pd.concat(series, axis=1, sort=True).sort_index()


@dataclass
class DataQualityReport:
    rows: int
    symbols: int
    start: str
    end: str
    missing_ratio: float
    internal_missing_ratio: float
    duplicate_dates: int
    non_positive_prices: int
    ohlc_violations: int
    stale_ratio: float
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.warnings

    def to_dict(self) -> dict:
        return {
            **self.__dict__,
            "ok": self.ok,
        }


def validate_panel(panel: dict[str, pd.DataFrame]) -> DataQualityReport:
    if not panel:
        raise ValueError("panel is empty")
    total_rows = sum(len(frame) for frame in panel.values())
    all_dates = pd.concat([frame.index.to_series() for frame in panel.values()])
    unique_dates = pd.DatetimeIndex(all_dates.unique()).sort_values()
    close = panel_to_frame(panel, "close")
    # Missing cells before a symbol's first bar are "not listed yet", not data
    # gaps. Only count holes inside each symbol's own range.
    missing_ratio = float(close.isna().mean().mean()) if close.size else 0.0
    holes = 0
    observed = 0
    for symbol in close.columns:
        series = close[symbol].dropna()
        if series.empty:
            continue
        window = close.loc[series.index.min(): series.index.max(), symbol]
        holes += int(window.isna().sum())
        observed += int(len(window))
    duplicate_dates = int(sum(frame.index.duplicated().sum() for frame in panel.values()))
    non_positive = int(sum(((frame[OHLCV_COLUMNS[:4]] <= 0).any(axis=1)).sum() for frame in panel.values()))
    violations = 0
    stale = 0
    for frame in panel.values():
        violations += int(
            (
                (frame["high"] < frame[["open", "close"]].max(axis=1))
                | (frame["low"] > frame[["open", "close"]].min(axis=1))
                | (frame["high"] < frame["low"])
            ).sum()
        )
        stale += int((frame["close"].diff() == 0).sum())
    warnings: list[str] = []
    if duplicate_dates:
        warnings.append(f"{duplicate_dates} duplicate dates")
    if non_positive:
        warnings.append(f"{non_positive} non-positive price rows")
    if violations:
        warnings.append(f"{violations} OHLC violations")
    internal_missing = holes / observed if observed else 0.0
    if internal_missing > 0.05:
        warnings.append(f"high in-range missing ratio: {internal_missing:.1%}")
    if len(unique_dates) < 120:
        warnings.append("fewer than 120 trading dates")
    stale_ratio = stale / max(total_rows, 1)
    if stale_ratio > 0.15:
        warnings.append(f"high unchanged-close ratio: {stale_ratio:.1%}")
    return DataQualityReport(
        rows=total_rows,
        symbols=len(panel),
        start=str(unique_dates.min().date()),
        end=str(unique_dates.max().date()),
        missing_ratio=missing_ratio,
        internal_missing_ratio=internal_missing,
        duplicate_dates=duplicate_dates,
        non_positive_prices=non_positive,
        ohlc_violations=violations,
        stale_ratio=stale_ratio,
        warnings=warnings,
    )


def align_panel(
    panel: dict[str, pd.DataFrame],
    *,
    require_all: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DatetimeIndex]:
    """Align a panel on the union of trading dates.

    The returned frames contain NaN where a symbol has no bar, so strategy and
    backtest code apply point-in-time availability instead of dropping every
    date on which one delisted or suspended name is missing. Set
    ``require_all=True`` for the old common-date intersection.
    """
    if not panel:
        raise ValueError("panel is empty")
    close = panel_to_frame(panel, "close")
    if require_all:
        dates = close.dropna(how="any").index.sort_values()
    else:
        dates = close.dropna(how="all").index.sort_values()
    aligned = {symbol: frame.reindex(dates) for symbol, frame in panel.items()}
    return aligned, dates


def price_frame(panel: dict[str, pd.DataFrame], field: str = "close") -> pd.DataFrame:
    """Continuous indicator price series, forward-filled only inside a range.

    Carrying the last close across a suspension avoids poisoning every rolling
    window with NaN, while the fill stops at the symbol's last real bar so a
    delisted name cannot keep receiving fresh scores.
    """
    frame = panel_to_frame(panel, field)
    return frame.ffill(limit_area="inside")


def usable_window(
    panel: dict[str, pd.DataFrame],
    *,
    min_active_ratio: float = 0.9,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Longest span in which at least ``min_active_ratio`` names have bars.

    Cached symbols rarely share an exact history. Intersecting globally would
    discard years of history because of one delisted name, while using the raw
    union would dilute statistics over long stretches where little trades.
    """
    close = panel_to_frame(panel, "close")
    if close.empty:
        raise ValueError("panel is empty")
    active = close.notna().mean(axis=1)
    ok = active >= min_active_ratio
    if not ok.any():
        raise ValueError("no window with sufficient cross-sectional coverage")
    valid = close.index[ok]
    return valid.min(), valid.max()


def trim_panel(
    panel: dict[str, pd.DataFrame],
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    *,
    min_coverage: float = 0.85,
) -> dict[str, pd.DataFrame]:
    """Clip to a window and drop names that do not cover it."""
    if start is None or end is None:
        auto_start, auto_end = usable_window(panel)
        start = auto_start if start is None else start
        end = auto_end if end is None else end
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    sliced = {s: f.loc[(f.index >= lo) & (f.index <= hi)] for s, f in panel.items()}
    sliced = {s: f for s, f in sliced.items() if len(f)}
    if not sliced:
        raise ValueError("no bars in the requested window")
    close = panel_to_frame(sliced, "close")
    coverage = close.notna().sum() / max(len(close), 1)
    return {s: f for s, f in sliced.items() if float(coverage.get(s, 0.0)) >= min_coverage}


def availability(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Boolean date x symbol matrix: a real OHLC bar exists on that date."""
    close = panel_to_frame(panel, "close")
    opened = panel_to_frame(panel, "open")
    return close.notna() & opened.notna()


def last_bar_dates(panel: dict[str, pd.DataFrame]) -> pd.Series:
    """Last date with real data per symbol, used to force delisting exits."""
    close = panel_to_frame(panel, "close")
    return pd.Series(
        {symbol: close[symbol].last_valid_index() for symbol in close.columns},
        dtype="datetime64[ns]",
    )


def save_universe(names: dict[str, str], path: Path | None = None) -> Path:
    target = path or (CACHE_DIR / "universe.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def download_universe(
    *,
    count: int = 1600,
    force: bool = False,
    cache_dir: Path = CACHE_DIR,
    include_benchmark: bool = True,
    universe: dict[str, str] | None = None,
    pause: float = 2.5,
    adjust: str = "qfq",
) -> dict:
    # The CLI and the reports call the unadjusted series "raw"; the provider
    # uses an empty string in its query parameter.  Accept both spellings here
    # instead of making every caller remember the provider's internal encoding.
    if adjust == "raw":
        adjust = ""
    if adjust not in {"qfq", "hfq", ""}:
        raise ValueError("adjust must be qfq, hfq, raw, or empty")
    provider = TencentProvider(cache_dir, pause=pause)
    names = dict(universe or DEFAULT_UNIVERSE)
    results = provider.download_many(names, count=count, force=force, adjust=adjust)
    suffix = adjust or "raw"
    payload = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "provider": f"Tencent {suffix} daily kline",
        "adjust": suffix,
        "symbols": [r.symbol for r in results],
        "rows": {r.symbol: r.rows for r in results},
        "pages": {
            r.symbol: (
                json.loads(
                    (cache_dir / f"{r.symbol}_{suffix}.meta.json").read_text(
                        encoding="utf-8"
                    )
                ).get("pages", 1)
                if (cache_dir / f"{r.symbol}_{suffix}.meta.json").exists()
                else None
            )
            for r in results
        },
    }
    if include_benchmark:
        try:
            bench = provider.download("sh000300", count=count, adjust="", force=force)
            payload["benchmark"] = {"symbol": bench.symbol, "rows": bench.rows}
            print(f"[benchmark] {bench.symbol}: {bench.rows} rows {bench.start}..{bench.end}")
        except (RuntimeError, ValueError, OSError) as exc:
            payload["benchmark"] = {"error": str(exc)}
            print(f"[!] benchmark unavailable this run: {exc}", flush=True)
    save_universe(names)
    (cache_dir / "download_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload

