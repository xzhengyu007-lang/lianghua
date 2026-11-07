"""Execution-level tests: delisting, capacity, T+1, and no look-ahead."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_system.backtest import Backtester
from quant_system.config import BacktestConfig, CostConfig


def _frame(index: pd.DatetimeIndex, price: float = 10.0, volume: float = 1e6) -> pd.DataFrame:
    close = pd.Series(price, index=index, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(index), volume, dtype=float),
        },
        index=index,
    )


def _panel(n: int = 100, symbols=("sh600000", "sh600036", "sz000001")) -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2024-01-02", periods=n)
    return {s: _frame(index, price=10.0 + i) for i, s in enumerate(symbols)}


def test_delisted_symbol_is_liquidated_at_last_bar() -> None:
    panel = _panel(120)
    index = next(iter(panel.values())).index
    delisted = "sh600036"
    panel[delisted] = panel[delisted].iloc[:70]
    last_bar = panel[delisted].index[-1]

    targets = pd.DataFrame(1 / 3, index=index, columns=sorted(panel))
    targets.iloc[:2] = 0.0
    result = Backtester(BacktestConfig(initial_cash=300_000)).run(panel, targets)

    sells = result.trades[(result.trades["symbol"] == delisted) & (result.trades["side"] == "sell")]
    assert not sells.empty
    assert sells["reason"].eq("delisting").any()
    exit_row = sells[sells["reason"] == "delisting"].iloc[0]
    assert exit_row["date"] == last_bar
    held_after = result.positions[
        (result.positions["symbol"] == delisted) & (result.positions["date"] > last_bar)
    ]
    assert held_after["shares"].eq(0).all()
    assert result.metadata["delisted_exits"] == 1


def test_delisting_does_not_freeze_portfolio_value() -> None:
    panel = _panel(120)
    surviving = "sh600000"
    for offset in range(len(panel[surviving])):
        panel[surviving].iloc[offset, panel[surviving].columns.get_loc("close")] = 10.0 + 0.1 * offset
    for column in ("open", "high", "low"):
        panel[surviving][column] = panel[surviving]["close"]
    panel["sz000001"] = panel["sz000001"].iloc[:70]
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=sorted(panel))
    targets[surviving] = 0.5
    targets.iloc[:2] = 0.0
    result = Backtester(BacktestConfig(initial_cash=100_000)).run(panel, targets)
    after = result.equity.loc[panel["sz000001"].index[-1] :]
    assert after.iloc[-1] != pytest.approx(after.iloc[0])


def test_adv_participation_caps_daily_order_size() -> None:
    panel = _panel(40, symbols=("sh600000",))
    for frame in panel.values():
        frame["volume"] = 10_000.0
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=["sh600000"])
    targets.iloc[2:, 0] = 0.9
    cfg = BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0, max_adv_participation=0.1)
    result = Backtester(cfg).run(panel, targets)
    buys = result.trades[result.trades["side"] == "buy"]
    assert not buys.empty
    # Volume is reported in lots (100 shares); 10,000 lots x 10% = 1,000 lots.
    assert buys["shares"].max() <= 100_000
    assert buys.groupby("date")["shares"].sum().max() <= 100_000


def test_no_buy_and_sell_on_same_day() -> None:
    panel = _panel(60)
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=sorted(panel))
    targets.loc[index[5:20], "sh600000"] = 0.5
    targets.loc[index[20:40], "sh600036"] = 0.5
    result = Backtester(BacktestConfig(initial_cash=200_000)).run(panel, targets)
    trades = result.trades
    if len(trades):
        both = trades.groupby(["date", "symbol"])["side"].nunique()
        assert (both <= 1).all()


def test_rebalance_acts_on_the_bar_after_signal_change() -> None:
    """A signal change must execute at the very next tradable open."""
    panel = _panel(40, symbols=("sh600000", "sh600036"))
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=sorted(panel))
    targets.loc[index[10]:, "sh600000"] = 0.5
    result = Backtester(
        BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0)
    ).run(panel, targets)
    first_buy = result.trades[result.trades["symbol"] == "sh600000"]["date"].min()
    assert first_buy == index[11]
    switch = targets.copy()
    switch.loc[index[20]:, "sh600000"] = 0.0
    switch.loc[index[20]:, "sh600036"] = 0.5
    result = Backtester(
        BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0)
    ).run(panel, switch)
    sells = result.trades[
        (result.trades["symbol"] == "sh600000") & (result.trades["side"] == "sell")
    ]
    assert not sells.empty
    assert sells["date"].min() == index[21]


def test_target_weight_change_without_basket_change_is_executed() -> None:
    """A sector cap or risk factor can change weights without changing names."""
    panel = _panel(40, symbols=("sh600000", "sh600036"))
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=sorted(panel))
    targets.loc[index[5]:index[19], "sh600000"] = 0.20
    targets.loc[index[5]:index[19], "sh600036"] = 0.20
    targets.loc[index[20]:, "sh600000"] = 0.40
    targets.loc[index[20]:, "sh600036"] = 0.05
    result = Backtester(
        BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0)
    ).run(panel, targets)
    first = result.trades[result.trades["symbol"] == "sh600000"]
    assert not first.empty
    initial = first[first["side"] == "buy"].iloc[0]
    later = first[first["date"] == index[21]]
    assert not later.empty
    assert later["shares"].sum() > 0


def test_limit_down_sell_is_retried_on_the_next_tradable_bar() -> None:
    """A sell blocked by a limit-down open must execute once trading resumes."""
    panel = _panel(30, symbols=("sh600000", "sh600036"))
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=sorted(panel))
    targets.loc[index[2]:index[9], "sh600000"] = 0.5
    targets.loc[index[10]:, "sh600036"] = 0.5

    # The sell is due at index[10]. Block that open, then let it trade on the
    # following bar. A blocked order must remain pending rather than vanish.
    blocked_day = index[10]
    panel["sh600000"].loc[blocked_day, ["open", "high", "low", "close"]] = (
        float(panel["sh600000"].loc[index[6], "close"]) * 0.90
    )
    result = Backtester(
        BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0)
    ).run(panel, targets)
    assert result.trades[result.trades["date"] == blocked_day].empty
    sells = result.trades[
        (result.trades["symbol"] == "sh600000") & (result.trades["side"] == "sell")
    ]
    assert not sells.empty
    assert sells["date"].min() == index[11]


def test_delisting_board_first_day_is_tradable_despite_a_huge_gap() -> None:
    """The first delisting-consolidation session has no price limit.

    Real tape: ``sh600193`` opened at 1.09 after a 2.17 close (-49.8%) and
    traded 987,964 lots.  A fixed +/-10% rule calls that a limit-down halt and
    keeps the position, manufacturing a loss the account could have avoided.
    """
    panel = _panel(30, symbols=("sh600000", "sh600036"))
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=sorted(panel))
    targets.loc[index[2]:index[9], "sh600000"] = 0.5
    targets.loc[index[10]:, "sh600036"] = 0.5

    # The signal flips on index[10] and therefore executes at the *next* open,
    # which is the session that must be tradeable despite the huge gap.
    exit_day = index[11]
    previous = float(panel["sh600000"].loc[index[10], "close"])
    panel["sh600000"].loc[exit_day, ["open", "high", "low", "close"]] = previous * 0.5
    panel["sh600000"].loc[exit_day, "volume"] = 900_000.0
    result = Backtester(
        BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0)
    ).run(panel, targets)
    sells = result.trades[
        (result.trades["symbol"] == "sh600000") & (result.trades["side"] == "sell")
    ]
    assert not sells.empty
    assert sells["date"].min() == exit_day


def test_target_change_only_affects_next_day() -> None:
    panel = _panel(50)
    index = next(iter(panel.values())).index
    base = pd.DataFrame(0.0, index=index, columns=sorted(panel))
    base.loc[index[10]:, "sh600000"] = 0.5
    shocked = base.copy()
    shocked.loc[index[30]:, "sh600000"] = 0.0
    shocked.loc[index[30]:, "sh600036"] = 0.5
    cfg = BacktestConfig(initial_cash=200_000)
    a = Backtester(cfg).run(panel, base)
    b = Backtester(cfg).run(panel, shocked)
    pd.testing.assert_series_equal(a.equity.loc[: index[30]], b.equity.loc[: index[30]])


def test_limit_up_open_blocks_entries() -> None:
    panel = _panel(30, symbols=("sh600000",))
    frame = panel["sh600000"]
    index = frame.index
    frame.loc[index[6], ["open", "high", "low", "close"]] = [11.0, 11.0, 11.0, 11.0]
    frame.loc[index[5], "close"] = 10.0
    targets = pd.DataFrame(0.0, index=index, columns=["sh600000"])
    targets.iloc[6:, 0] = 0.9
    result = Backtester(BacktestConfig(initial_cash=100_000, max_position_weight=1.0)).run(panel, targets)
    first_buy = result.trades[result.trades["side"] == "buy"].iloc[0]
    assert first_buy["date"] > index[6]

def test_partial_fills_never_overshoot_the_frozen_target() -> None:
    """A capped order must be completed over following days, never exceeded."""
    panel = _panel(80, symbols=("sh600000",))
    for frame in panel.values():
        frame["volume"] = 20_000.0
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=["sh600000"])
    targets.iloc[5:, 0] = 0.5
    cfg = BacktestConfig(
        initial_cash=1_000_000, max_position_weight=1.0, max_adv_participation=0.01
    )
    result = Backtester(cfg).run(panel, targets)
    buys = result.trades[result.trades["side"] == "buy"]
    assert len(buys) > 1, "a 1% participation cap must split the order over several days"
    # Frozen target: 50% of the rebalance open equity, rounded down to lots.
    rebalance_open = float(panel["sh600000"]["open"].iloc[5])
    target_shares = int((0.5 * 1_000_000) // rebalance_open // 100 * 100)
    assert buys["shares"].sum() == target_shares
    assert result.positions[result.positions["symbol"] == "sh600000"]["shares"].max() <= target_shares


def test_split_fills_stay_inside_the_participation_cap_each_day() -> None:
    panel = _panel(80, symbols=("sh600000",))
    for frame in panel.values():
        frame["volume"] = 20_000.0
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=["sh600000"])
    targets.iloc[5:, 0] = 0.5
    cfg = BacktestConfig(
        initial_cash=1_000_000, max_position_weight=1.0, max_adv_participation=0.01
    )
    result = Backtester(cfg).run(panel, targets)
    daily = result.trades.groupby("date")["shares"].sum()
    # 20,000 lots * 1% = 200 lots = 20,000 shares per day.
    assert daily.max() <= 20_000

def test_drawdown_guard_reduces_exposure_after_a_loss() -> None:
    index = pd.bdate_range("2024-01-02", periods=120)
    symbols = ("sh600000",)
    # A gradual decline, not a one-bar gap: a guard cannot avoid a gap-down,
    # but it can de-risk while the decline is still unfolding.
    decline = np.linspace(10.0, 6.0, 60)
    close = np.concatenate([np.full(30, 10.0), decline, np.full(30, 9.0)])
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(len(index), 1e6),
        },
        index=index,
    )
    panel = {"sh600000": frame}
    targets = pd.DataFrame(0.9, index=index, columns=["sh600000"])
    targets.iloc[:2] = 0.0
    plain = Backtester(
        BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0)
    ).run(panel, targets)
    guarded = Backtester(
        BacktestConfig(
            initial_cash=1_000_000,
            max_position_weight=1.0,
            drawdown_stop=0.10,
            drawdown_de_risk=0.4,
        )
    ).run(panel, targets)
    # The guard must cut the loss and leave the book partially in cash at the
    # bottom, then re-risk after the recovery.
    assert guarded.equity.min() > plain.equity.min()
    weights = (
        guarded.positions.pivot(index="date", columns="symbol", values="weight")
        .fillna(0.0)["sh600000"]
    )
    assert float(weights.loc[index[80]]) < 0.85
    assert guarded.metadata["trades"] > plain.metadata["trades"]


def test_drawdown_guard_uses_only_realised_equity() -> None:
    """Shocking the final bar cannot change earlier allocations."""
    index = pd.bdate_range("2024-01-02", periods=90)
    close = np.concatenate([np.full(45, 10.0), np.full(45, 7.0)])
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.full(len(index), 1e6),
        },
        index=index,
    )
    panel = {"sh600000": frame}
    targets = pd.DataFrame(0.9, index=index, columns=["sh600000"])
    cfg = BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0, drawdown_stop=0.05)
    base = Backtester(cfg).run(panel, targets)
    shocked = {k: v.copy() for k, v in panel.items()}
    shocked["sh600000"].loc[index[-1], ["open", "high", "low", "close"]] = 0.01
    other = Backtester(cfg).run(shocked, targets)
    pd.testing.assert_series_equal(
        base.equity.iloc[:-1], other.equity.iloc[:-1]
    )


def test_market_impact_is_off_by_default_and_monotone() -> None:
    from quant_system.config import CostConfig
    from quant_system.costs import execution_price, impact_bps

    base = CostConfig()
    assert impact_bps(100_000, 1_000_000, base) == 0.0
    assert execution_price("buy", 10.0, base) == pytest.approx(10.0 * 1.0005)

    impacted = CostConfig(impact_coefficient_bps=50.0)
    small = impact_bps(10_000, 1_000_000, impacted)
    large = impact_bps(40_000, 1_000_000, impacted)
    # 10k / 1M = 1% participation, 40k / 1M = 4%.
    assert small == pytest.approx(50.0 * 0.01 ** 0.5)
    assert large == pytest.approx(50.0 * 0.04 ** 0.5)
    assert large > small
    assert large / small == pytest.approx(2.0)
    cheap = execution_price("buy", 10.0, impacted, shares=10_000, adv_shares=1_000_000)
    dear = execution_price("buy", 10.0, impacted, shares=40_000, adv_shares=1_000_000)
    assert dear > cheap > 10.0
    assert execution_price("sell", 10.0, impacted, shares=40_000, adv_shares=1_000_000) < 10.0


def test_impact_only_changes_results_when_enabled() -> None:
    from quant_system.config import CostConfig

    panel = _panel(80, symbols=("sh600000",))
    for frame in panel.values():
        frame["volume"] = 20_000.0
    index = next(iter(panel.values())).index
    targets = pd.DataFrame(0.0, index=index, columns=["sh600000"])
    targets.iloc[5:, 0] = 0.5
    plain = Backtester(BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0)).run(
        panel, targets
    )
    costly = Backtester(
        BacktestConfig(
            initial_cash=1_000_000,
            max_position_weight=1.0,
            costs=CostConfig(impact_coefficient_bps=60.0),
        )
    ).run(panel, targets)
    assert costly.equity.iloc[-1] < plain.equity.iloc[-1]


def test_slippage_is_charged_once_not_twice() -> None:
    """Regression: slippage lives in the executed price, not also in cash fees."""
    from quant_system.costs import transaction_costs

    cfg = CostConfig(slippage_bps=50.0)
    open_price = 10.0
    from quant_system.costs import execution_price
    price = execution_price("buy", open_price, cfg)
    breakdown = transaction_costs(
        "buy", 10_000, price, open_price=open_price, config=cfg
    )
    assert breakdown.slippage > 0
    assert breakdown.total == pytest.approx(breakdown.fees + breakdown.slippage)
    assert breakdown.fees < breakdown.total

    # The cash actually spent must equal notional at the executed price plus
    # explicit fees only.
    notional = 10_000 * price
    expected_fees = (
        max(5.0, notional * cfg.commission_rate)
        + notional * cfg.transfer_fee_rate
    )
    assert breakdown.fees == pytest.approx(expected_fees)


def test_ledger_replay_matches_backtest_cash() -> None:
    """A paper ledger replaying the backtest's fills must end at the same equity."""
    from quant_system.config import RiskConfig
    from quant_system.paper import Ledger, apply_fills, summarize
    from quant_system.portfolio import build_portfolio
    from quant_system.strategies import build_strategy
    from tests.conftest import make_frame

    panel = {f"sh60000{i}": make_frame(seed=i + 1) for i in range(4)}
    raw = build_strategy(
        "low_vol_core", top_n=2, vol_window=30, trend_window=60, rebalance_days=40
    ).generate(panel)
    weights = build_portfolio(raw, panel, risk=RiskConfig(), use_regime=False)
    costs = CostConfig()
    result = Backtester(
        BacktestConfig(initial_cash=200_000, costs=costs, max_adv_participation=1.0)
    ).run(panel, weights)
    ledger = Ledger(cash=200_000.0)
    for date, group in result.trades.sort_values(["date", "side"], ascending=[True, False]).groupby("date"):
        ledger = apply_fills(
            ledger,
            group[["symbol", "side", "shares", "price"]].copy(),
            date=date,
            costs=costs,
        )
    last = result.equity.index[-1]
    prices = pd.Series({s: float(panel[s]["close"].loc[:last].ffill().iloc[-1]) for s in panel})
    assert summarize(ledger, prices)["equity"] == pytest.approx(
        float(result.equity.iloc[-1]), abs=1.0
    )


def test_adv_cap_uses_only_volume_known_before_the_open() -> None:
    """Liquidity capacity must not include the bar being traded.

    Orders execute at today's open. If the ADV window includes today's volume,
    a backtest can size an order against the very volume that order will
    consume. This test inflates the *last* bar's volume enormously: a correct
    implementation must ignore it and keep the cap set by prior bars.
    """
    from quant_system.backtest import Backtester
    from quant_system.config import BacktestConfig

    index = pd.bdate_range("2024-01-02", periods=40)
    frame = pd.DataFrame(
        {
            "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0,
            "volume": 10_000.0,
        },
        index=index,
    )
    frame.loc[index[-1], "volume"] = 1_000_000_000.0
    panel = {"sh600000": frame}
    targets = pd.DataFrame(0.0, index=index, columns=["sh600000"])
    targets.loc[index[20]:, "sh600000"] = 0.9
    cfg = BacktestConfig(
        initial_cash=1_000_000, max_position_weight=1.0, max_adv_participation=0.10
    )
    result = Backtester(cfg).run(panel, targets)
    last_trade = result.trades.iloc[-1]
    assert last_trade["date"] < index[-1]
    # 10,000 lots * 100 shares * 10% = 100,000 shares, not 10^10 shares.
    assert float(result.trades["shares"].max()) <= 100_000


def test_delisting_liquidates_at_the_open_not_the_close() -> None:
    """The exit decision happens before the final bar trades.

    Using the final bar's close would let the backtest sell at a price it only
    learns after the fact. A +100% final day must therefore realise the open,
    not the close.
    """
    from quant_system.backtest import Backtester
    from quant_system.config import BacktestConfig

    index = pd.bdate_range("2024-01-02", periods=30)
    dying = _frame(index[:20], price=10.0)
    dying.loc[index[19], "close"] = 20.0
    dying.loc[index[19], "high"] = 20.0
    panel = {"sh600000": dying, "sh600036": _frame(index, price=10.0)}
    targets = pd.DataFrame(0.0, index=index, columns=sorted(panel))
    targets[["sh600000", "sh600036"]] = 0.5
    targets.iloc[:2] = 0.0
    result = Backtester(
        BacktestConfig(initial_cash=1_000_000, max_position_weight=1.0)
    ).run(panel, targets)
    exit_row = result.trades[
        (result.trades["symbol"] == "sh600000") & (result.trades["reason"] == "delisting")
    ].iloc[0]
    assert exit_row["date"] == index[19]
    # 10.0 open minus 5bp slippage, not the 20.0 close.
    assert float(exit_row["price"]) == pytest.approx(10.0 * (1 - 5 / 10_000))


def test_position_frame_matches_the_traded_book(sample_panel) -> None:
    """The dense position matrix must describe exactly what was traded.

    The vectorised kernel builds ``positions`` from a shares x price grid
    instead of appending one row per symbol per bar.  If the column order of
    that grid ever drifts from ``sorted(panel)``, every symbol's weight would
    be silently attributed to its neighbour.  Recompute market value from the
    reported shares and compare it with the reported frame and equity.
    """
    from quant_system.data import align_panel, price_frame
    from quant_system.strategies import build_strategy

    aligned, dates = align_panel(sample_panel)
    weights = build_strategy(
        "low_vol_core", top_n=3, vol_window=30, trend_window=60, rebalance_days=10
    ).generate(aligned)
    result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(aligned, weights)

    frame = result.positions
    assert len(frame) == len(dates) * len(aligned)
    assert list(frame["symbol"].drop_duplicates()) == sorted(aligned)

    held = frame[frame["shares"] > 0]
    assert not held.empty
    close = price_frame(aligned, "close").ffill()
    expected = close.stack().reindex(
        pd.MultiIndex.from_arrays([held["date"], held["symbol"]])
    ).to_numpy() * held["shares"].to_numpy()
    assert np.allclose(held["market_value"].to_numpy(), expected)
    total = frame.groupby("date")["market_value"].sum()
    cash = result.equity - total
    assert (cash >= -1e-6).all()
    assert np.allclose((total + cash).to_numpy(), result.equity.to_numpy())
