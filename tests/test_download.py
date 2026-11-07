from __future__ import annotations

import pandas as pd
import pytest

from quant_system.data import TencentProvider


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeSession:
    """Serves a fixed price history in 640-bar pages anchored on ``end``."""

    headers: dict = {}

    def __init__(self, dates, start_price=10.0):
        self.dates = list(dates)
        self.start_price = start_price
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        param = params["param"]
        parts = param.split(",")
        symbol, end = parts[0], parts[3]
        count = int(parts[4])
        if end:
            cutoff = pd.Timestamp(end)
            usable = [d for d in self.dates if d <= cutoff]
        else:
            usable = self.dates
        page = usable[-count:]
        rows = []
        for i, d in enumerate(page):
            index = self.dates.index(d)
            close = self.start_price * (1.0 + 0.001 * index)
            rows.append([
                d.strftime("%Y-%m-%d"),
                f"{close:.4f}",
                f"{close:.4f}",
                f"{close * 1.01:.4f}",
                f"{close * 0.99:.4f}",
                "1000",
            ])
        return _FakeResponse({"code": 0, "data": {symbol: {"qfqday": rows}}})


def _write_cache(cache, symbol, adjust, frame, rows=None, pages=1, warnings=None):
    import json

    frame.to_parquet(cache / f"{symbol}_{adjust}.parquet")
    meta = {
        "symbol": symbol,
        "adjust": adjust,
        "pages": pages,
        "rows": len(frame) if rows is None else rows,
        "start": str(frame.index.min().date()),
        "end": str(frame.index.max().date()),
        "warnings": warnings or [],
    }
    (cache / f"{symbol}_{adjust}.meta.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )


def _cache_frame(periods=300, price=10.0):
    index = pd.bdate_range("2024-01-02", periods=periods)
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price,
            "volume": 1000.0,
        },
        index=index,
    )


def test_cache_audit_accepts_consistent_files(tmp_path) -> None:
    from quant_system.cache_audit import audit_cache

    _write_cache(tmp_path, "sh600519", "qfq", _cache_frame(), pages=1)
    audit = audit_cache(tmp_path)
    assert audit.ok, audit.problems
    assert audit.files[0].rows_file == 300


def test_cache_audit_detects_metadata_row_mismatch(tmp_path) -> None:
    from quant_system.cache_audit import audit_cache

    _write_cache(tmp_path, "sh600519", "qfq", _cache_frame(), rows=999)
    audit = audit_cache(tmp_path)
    assert not audit.ok
    assert any("row count mismatch" in problem for problem in audit.problems)


def test_cache_audit_detects_missing_metadata(tmp_path) -> None:
    from quant_system.cache_audit import audit_cache

    _cache_frame().to_parquet(tmp_path / "sh600519_qfq.parquet")
    audit = audit_cache(tmp_path)
    assert not audit.ok
    assert any("missing or invalid meta.json" in problem for problem in audit.problems)


def test_cache_audit_detects_non_positive_prices(tmp_path) -> None:
    from quant_system.cache_audit import audit_cache

    frame = _cache_frame()
    frame.loc[frame.index[10], "close"] = -1.0
    _write_cache(tmp_path, "sh600519", "qfq", frame)
    audit = audit_cache(tmp_path)
    assert not audit.ok
    assert any("non-positive price rows" in problem for problem in audit.problems)


def test_cache_audit_accepts_rebuilt_from_hfq_metadata(tmp_path) -> None:
    import json

    from quant_system.cache_audit import audit_cache

    frame = _cache_frame(periods=700)
    frame.to_parquet(tmp_path / "sh601088_qfq.parquet")
    meta = {
        "symbol": "sh601088",
        "adjust": "qfq_rebuilt_from_hfq",
        "source": "hfq",
        "rows": 700,
        "start": str(frame.index.min().date()),
        "end": str(frame.index.max().date()),
        "pages": 0,
        "warnings": [{"type": "rebuilt_from_hfq", "factor": 0.5}],
    }
    (tmp_path / "sh601088_qfq.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    audit = audit_cache(tmp_path)
    assert audit.ok, audit.problems


def test_segmented_download_stitches_full_history(tmp_path) -> None:
    dates = pd.bdate_range("2019-01-02", periods=1800)
    session = _FakeSession(dates)
    provider = TencentProvider(tmp_path, session=session, pause=0.0)
    result = provider.download("600519", count=1800, force=True)
    assert result.rows == 1800
    frame = pd.read_parquet(tmp_path / "sh600519_qfq.parquet")
    assert len(frame) == 1800
    assert frame.index.min() == dates[0]
    assert frame.index.max() == dates[-1]
    assert frame.index.is_unique
    assert session.calls >= 3


def test_download_is_idempotent_after_metadata_written(tmp_path) -> None:
    dates = pd.bdate_range("2020-01-02", periods=1500)
    session = _FakeSession(dates)
    provider = TencentProvider(tmp_path, session=session, pause=0.0)
    provider.download("600519", count=1500, force=True)
    calls_after_first = session.calls
    provider.download("600519", count=1500, force=False)
    assert session.calls == calls_after_first


def test_download_short_cache_is_refreshed(tmp_path) -> None:
    dates = pd.bdate_range("2020-01-02", periods=1500)
    session = _FakeSession(dates)
    provider = TencentProvider(tmp_path, session=session, pause=0.0)
    provider.download("600519", count=700, force=True)
    result = provider.download("600519", count=1500, force=False)
    assert result.rows == 1500


def test_overlap_rescale_records_warning() -> None:
    index_a = pd.bdate_range("2020-01-02", periods=5)
    index_b = pd.bdate_range("2020-01-01", periods=6)
    def frame(index, scale):
        return pd.DataFrame(
            {
                "open": 10 * scale,
                "high": 10.5 * scale,
                "low": 9.5 * scale,
                "close": [10 * scale] * len(index),
                "volume": 1000.0,
            },
            index=index,
        )
    new = frame(index_a, 1.0)
    old = frame(index_b, 2.0)
    merged, warnings = TencentProvider._merge_pages([old, new])
    assert len(merged) == 6
    assert merged["close"].nunique() == 1
    assert any(w["type"] == "adjusted_rescale" for w in warnings)


def test_non_uniform_overlap_is_rejected(tmp_path, monkeypatch) -> None:
    """A non-uniform page overlap must not be silently stitched."""
    index = pd.bdate_range("2019-01-02", periods=12)
    clean = pd.DataFrame(
        {
            "open": 10.0, "high": 10.5, "low": 9.5, "close": 10.0, "volume": 1000.0
        },
        index=index,
    )
    provider = TencentProvider(tmp_path, session=_FakeSession(index), pause=0.0)

    def bad_merge(pages):
        return clean, [
            {"type": "overlap_mismatch", "overlap": 5, "max_relative_error": 0.35}
        ]

    monkeypatch.setattr(provider, "_merge_pages", bad_merge)
    with pytest.raises(ValueError, match="overlap mismatch"):
        provider.download("600519", count=60, force=True)


def test_download_many_retries_failures(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2021-01-04", periods=800)
    provider = TencentProvider(tmp_path, session=_FakeSession(dates), pause=0.0)
    attempts = {"n": 0}
    original = provider.download

    def flaky(symbol, **kwargs):
        attempts["n"] += 1
        if symbol == "sz000001" and attempts["n"] < 3:
            raise RuntimeError("transient 501")
        return original(symbol, **kwargs)

    monkeypatch.setattr(provider, "download", flaky)
    results = provider.download_many(["sh600519", "sz000001"], count=800, retry_rounds=3)
    assert {r.symbol for r in results} == {"sh600519", "sz000001"}