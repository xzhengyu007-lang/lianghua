from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.attribution import (
    load_attribution_inputs,
    low_vol_benchmark,
    save_attribution,
    sector_attribution,
)


def _frame(monthly_growth: float, n: int = 400, start: str = "2023-01-03") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=n)
    steps = np.arange(n) / 21.0
    close = 10.0 * (1.0 + monthly_growth) ** steps
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1e5},
        index=index,
    )


def _positions_and_equity(panel: dict[str, pd.DataFrame], held: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for date in panel[held[0]].index:
        for symbol in held:
            price = float(panel[symbol].loc[date, "close"])
            rows.append({"date": date, "symbol": symbol, "shares": 1000, "close": price,
                         "market_value": 1000 * price})
    positions = pd.DataFrame(rows)
    equity = positions.groupby("date")["market_value"].sum()
    return positions, equity


def test_decomposition_identity_holds() -> None:
    panel = {
        "600036": _frame(0.02),
        "600000": _frame(0.02),
        "600519": _frame(0.0),
        "000858": _frame(0.0),
    }
    positions, equity = _positions_and_equity(panel, ["600036", "600000"])
    result = sector_attribution(positions, panel, equity)
    summary = result.summary
    total = summary["allocation"] + summary["selection"] + summary["interaction"]
    assert total == pytest.approx(summary["excess"], abs=1e-9)
    assert summary["months"] >= 12


def test_allocation_dominates_when_picking_the_growing_sector() -> None:
    panel = {
        "600036": _frame(0.02),
        "600000": _frame(0.02),
        "600519": _frame(0.0),
        "000858": _frame(0.0),
    }
    positions, equity = _positions_and_equity(panel, ["600036", "600000"])
    result = sector_attribution(positions, panel, equity)
    summary = result.summary
    ratios = summary["allocation"] / summary["excess"]
    assert 0.9 < ratios < 1.1
    assert abs(summary["selection"]) < 1e-6
    financials = result.sectors.set_index("sector").loc["金融"]
    assert financials["weight_p"] > financials["weight_u"]


def test_selection_shows_up_when_same_sector_splits() -> None:
    panel = {
        "600036": _frame(0.04),
        "600000": _frame(0.0),
        "600519": _frame(0.0),
        "000858": _frame(0.0),
    }
    positions, equity = _positions_and_equity(panel, ["600036"])
    result = sector_attribution(positions, panel, equity)
    summary = result.summary
    assert summary["selection"] > 0
    assert summary["selection"] + summary["allocation"] == pytest.approx(
        summary["excess"] - summary["interaction"], abs=1e-9
    )


def test_empty_positions_raise() -> None:
    panel = {"600036": _frame(0.01)}
    with pytest.raises(ValueError):
        sector_attribution(pd.DataFrame(columns=["date", "symbol", "shares", "market_value"]),
                           panel, pd.Series(dtype=float))


def test_exited_names_do_not_linger_in_month_end_weights() -> None:
    """A closed position must not be forward-filled into later month-ends.

    Regression: dropping the zero-share rows and ffill-ing afterwards carried
    every sold name to the end of the sample, so month-end weights summed to
    6x equity and the "monthly returns" reached +65%.
    """
    panel = {
        "600036": _frame(0.01, n=250),
        "600000": _frame(0.01, n=250),
        "600519": _frame(0.005, n=250),
    }
    index = panel["600036"].index

    rows = []
    for date in index:
        # 600519 is held only for the first 60 bars, then sold (explicit zeros).
        for symbol in ("600036", "600000", "600519"):
            price = float(panel[symbol].loc[date, "close"])
            shares = 0 if (symbol == "600519" and date > index[60]) else 1000
            rows.append({"date": date, "symbol": symbol, "shares": shares, "close": price,
                         "market_value": shares * price})
    positions = pd.DataFrame(rows)
    equity = positions.groupby("date")["market_value"].sum()

    result = sector_attribution(positions, panel, equity)
    monthly = result.monthly
    assert monthly["r_p"].abs().max() < 0.2
    assert "cash_weight" in monthly.columns
    assert monthly["residual"].abs().max() < 1e-6


def test_save_attribution_writes_markdown_and_frames(tmp_path) -> None:
    panel = {"600036": _frame(0.02), "600519": _frame(0.0)}
    positions, equity = _positions_and_equity(panel, ["600036"])
    result = sector_attribution(positions, panel, equity)

    output = save_attribution(result, tmp_path / "attrib", title="归因测试", meta={"k": "v"})
    assert (output / "sectors.csv").exists()
    assert (output / "monthly.csv").exists()
    assert (output / "summary.json").exists()
    text = (output / "report.md").read_text(encoding="utf-8")
    assert "归因测试" in text
    assert "行业内选股" in text


def test_load_attribution_inputs_reads_report_pair(tmp_path) -> None:
    report = tmp_path / "run"
    report.mkdir()
    pd.DataFrame(
        [{"date": "2023-01-03", "symbol": "600036", "shares": 100, "close": 10.0,
          "market_value": 1000.0, "weight": 1.0}]
    ).to_csv(report / "positions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"equity": [1000.0]}, index=pd.to_datetime(["2023-01-03"])).to_csv(
        report / "equity.csv", encoding="utf-8-sig"
    )
    positions, equity = load_attribution_inputs(report)
    assert len(positions) == 1
    assert float(equity.iloc[0]) == 1000.0
    with pytest.raises(FileNotFoundError):
        load_attribution_inputs(tmp_path / "missing")


def _noisy_frame(daily_vol: float, n: int = 220, seed: int = 0) -> pd.DataFrame:
    """Random walk whose *measured* volatility really is ``daily_vol``.

    A deterministic exponential drift has a volatility of ~1e-16, so vol-rank
    tests built on it would rank numerical noise instead of risk.
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-03", periods=n)
    steps = rng.normal(0.0, daily_vol, n)
    close = 10.0 * np.exp(np.cumsum(steps))
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1e5},
        index=index,
    )


def test_low_vol_benchmark_selects_the_calm_names() -> None:
    panel = {
        "600036": _noisy_frame(0.008, seed=1),
        "600000": _noisy_frame(0.009, seed=2),
        "600519": _noisy_frame(0.050, seed=3),
        "000858": _noisy_frame(0.045, seed=4),
    }
    month_end = pd.DatetimeIndex([panel["600036"].index[-1]])
    weights = low_vol_benchmark(panel, month_end, fraction=0.5)
    row = weights.iloc[0]
    assert row.sum() == pytest.approx(1.0)
    assert set(row[row > 0].index) == {"600036", "600000"}


def test_low_vol_benchmark_does_not_look_ahead() -> None:
    """Truncating the panel at the decision date must not change the basket."""
    panel = {"600036": _noisy_frame(0.008, seed=5), "600519": _noisy_frame(0.040, seed=6)}
    timestamp = panel["600036"].index[150]
    month_end = pd.DatetimeIndex([timestamp])
    truncated = {k: v.loc[:timestamp] for k, v in panel.items()}

    full_weights = low_vol_benchmark(panel, month_end, fraction=0.5)
    cut_weights = low_vol_benchmark(truncated, month_end, fraction=0.5)
    pd.testing.assert_frame_equal(full_weights[["600036", "600519"]],
                                  cut_weights[["600036", "600519"]])


def test_factor_benchmark_changes_the_excess_but_not_the_portfolio() -> None:
    panel = {
        "600036": _noisy_frame(0.008, seed=7),   # calm, held
        "600000": _noisy_frame(0.009, seed=8),   # calm, not held
        "600519": _noisy_frame(0.050, seed=9),   # wild, not held
        "000858": _noisy_frame(0.045, seed=10),  # wild, not held
    }
    positions, equity = _positions_and_equity(panel, ["600036"])
    equal = sector_attribution(positions, panel, equity)
    month_ends = equal.monthly["month"].pipe(pd.to_datetime)
    benchmark = low_vol_benchmark(panel, pd.DatetimeIndex(month_ends), fraction=0.5)
    factor = sector_attribution(positions, panel, equity, benchmark=benchmark,
                                benchmark_label="calm half")

    assert factor.summary["portfolio_return"] == pytest.approx(equal.summary["portfolio_return"])
    assert factor.summary["benchmark"] == "calm half"
    assert factor.summary["universe_return"] != pytest.approx(equal.summary["universe_return"])
    assert factor.summary["allocation"] + factor.summary["selection"] + factor.summary["interaction"] \
        == pytest.approx(factor.summary["excess"], abs=1e-9)


def test_suspended_holding_is_not_relabelled_as_cash() -> None:
    """A held name without a month-end price is a zero-return position, not cash.

    Regression: dropping names with a missing next-month return made a 16%
    suspended holding look like cash, inflating the reported average cash
    weight and distorting the Brinson sector weights.
    """
    panel = {
        "600036": _frame(0.02, n=300),
        "600000": _frame(0.0, n=300),
        "600519": _frame(0.0, n=300),
    }
    index = panel["600036"].index
    cutoff = index[220]
    # 600000 stops printing after ``cutoff`` but stays in the portfolio at its
    # last mark. 600036 keeps trading, so the book still has a live return.
    suspended = panel["600000"].loc[:cutoff].copy()
    panel["600000"] = suspended.reindex(index)

    rows = []
    for date in index:
        for symbol in ("600036", "600000"):
            shares = 1000
            price = float(panel[symbol].loc[:date, "close"].dropna().iloc[-1])
            rows.append({"date": date, "symbol": symbol, "shares": shares,
                         "close": price, "market_value": shares * price})
    positions = pd.DataFrame(rows)
    equity = positions.groupby("date")["market_value"].sum()

    result = sector_attribution(positions, panel, equity)
    monthly = result.monthly
    assert monthly["cash_weight"].max() < 1e-9
    assert monthly["suspended_weight"].max() > 0.3
    assert result.summary["avg_cash_weight"] == pytest.approx(0.0, abs=1e-9)
    assert result.summary["avg_suspended_weight"] > 0.05
