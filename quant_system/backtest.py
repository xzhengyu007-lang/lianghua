"""Daily event-driven backtester with conservative A-share mechanics.

Design rules, all of which exist to remove look-ahead and unrealistic fills:

* Signals are produced at the close and executed at the *next* open.
* The target share basket is frozen at the rebalance open; daily price drift
  never re-derives share counts (which would create momentum-chasing trades).
* A symbol is tradable only on dates where a real bar exists, and a position
  in a delisted name is liquidated at the last available bar.
* Orders are capped by a share of trailing average daily volume.
* T+1 is enforced: shares bought today cannot be sold on the same day.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .costs import execution_price, transaction_costs
from .data import SHARES_PER_VOLUME_UNIT, align_panel, infer_price_limit, panel_to_frame
from .metrics import PerformanceMetrics, performance_metrics
from .portfolio import normalize_targets


@dataclass
class Position:
    shares: int = 0
    avg_cost: float = 0.0
    last_trade_day: int = -1


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    weights: pd.DataFrame
    metrics: PerformanceMetrics
    benchmark_returns: pd.Series | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "metrics": self.metrics.to_dict(),
            "metadata": self.metadata,
            "start": str(self.equity.index.min().date()),
            "end": str(self.equity.index.max().date()),
        }


def _price_limits(symbol: str) -> float:
    return infer_price_limit(symbol)


class Backtester:
    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()
        self.config.validate()

    @staticmethod
    def _normalize_targets(
        targets: pd.DataFrame,
        symbols: list[str],
        dates: pd.DatetimeIndex,
        config: BacktestConfig,
    ) -> pd.DataFrame:
        """Delegate to the shared helper so the live signal cannot drift."""
        return normalize_targets(targets, symbols, dates, config)

    def run(
        self,
        panel: dict[str, pd.DataFrame],
        targets: pd.DataFrame,
        *,
        benchmark: pd.Series | None = None,
    ) -> BacktestResult:
        if not panel:
            raise ValueError("panel is empty")
        aligned, dates = align_panel(panel)
        symbols = sorted(aligned)
        if len(dates) < 3:
            raise ValueError("need at least three aligned bars")
        cfg = self.config

        open_frame = panel_to_frame(aligned, "open").reindex(columns=symbols)
        close_frame = panel_to_frame(aligned, "close").reindex(columns=symbols)
        volume_frame = panel_to_frame(aligned, "volume").reindex(columns=symbols)
        has_bar = close_frame.notna() & open_frame.notna() & (volume_frame > 0)
        # Valuation price: forward fill only inside a symbol's real bar range so
        # a delisted name is not held at a frozen price forever.
        mark_price = close_frame.ffill(limit_area="inside")
        # Liquidity capacity must be known *before* the order is sent. Orders
        # execute at today's open, so today's volume is future information.
        # Shifting by one bar is the difference between a tradeable capacity
        # limit and a look-ahead that lets the backtest size against the very
        # volume it is about to consume.
        adv_basis = volume_frame if cfg.adv_includes_today else volume_frame.shift(1)
        adv = (
            adv_basis.rolling(cfg.adv_window, min_periods=max(3, cfg.adv_window // 4))
            .mean()
            * SHARES_PER_VOLUME_UNIT
        )

        prev_open = open_frame.shift(1)
        prev_close = close_frame.shift(1)
        limit = pd.Series({s: _price_limits(s) for s in symbols})
        limit_up_open = open_frame.ge(prev_close.mul(1 + limit, axis=1) - 1e-6)
        limit_down_open = open_frame.le(prev_close.mul(1 - limit, axis=1) + 1e-6)

        # A stock entering the delisting-consolidation board trades its first
        # session without a price limit, so an open 50% below the previous
        # close is *executable*, not a limit-down halt.  Treating it as a halt
        # would make the backtest hold a position it could have sold, which
        # silently manufactures a loss that no real account would suffer.
        # The signal is unambiguous: the open is far beyond the symbol's own
        # limit while the bar still traded (volume > 0).
        unbounded_open = (
            open_frame.lt(prev_close.mul(1 - 2 * limit, axis=1))
            | open_frame.gt(prev_close.mul(1 + 2 * limit, axis=1))
        ) & has_bar
        limit_up_open = limit_up_open & ~unbounded_open
        limit_down_open = limit_down_open & ~unbounded_open

        last_bar = {s: close_frame[s].last_valid_index() for s in symbols}
        sample_end = dates[-1]
        retiring = {s: d for s, d in last_bar.items() if d is not None and d < sample_end}
        retired: set[str] = set()

        weights = self._normalize_targets(targets, symbols, dates, cfg)

        cash = float(cfg.initial_cash)
        n_symbols = len(symbols)
        position_index = {symbol: j for j, symbol in enumerate(symbols)}
        held_shares = np.zeros(n_symbols, dtype=np.int64)
        avg_cost = np.zeros(n_symbols, dtype=float)
        last_trade_day = np.full(n_symbols, -1, dtype=np.int64)

        # Dense arrays for the hot loop.  Per-bar ``DataFrame.loc[date, symbol]``
        # lookups cost about eight million pandas calls per backtest and were the
        # dominant runtime term.  These arrays hold exactly the same values in
        # exactly the same order, so the equity path is bit-identical.
        open_values = open_frame.to_numpy(dtype=float)
        close_values = mark_price.to_numpy(dtype=float)
        close_filled = mark_price.ffill().to_numpy(dtype=float)
        bar_values = has_bar.to_numpy(dtype=bool)
        limit_up_values = limit_up_open.to_numpy(dtype=bool)
        limit_down_values = limit_down_open.to_numpy(dtype=bool)
        adv_values = adv.to_numpy(dtype=float)
        weight_values = weights.to_numpy(dtype=float)

        exits_by_day: dict[int, list[int]] = {}
        for symbol, last_day in retiring.items():
            exits_by_day.setdefault(dates.get_loc(last_day), []).append(position_index[symbol])

        trade_rows: list[dict] = []
        equity_rows: list[dict] = []
        shares_history = np.zeros((len(dates), n_symbols), dtype=np.int64)
        turnover_value = 0.0
        exposure_sum = 0.0
        frozen_lots = np.zeros(n_symbols, dtype=np.int64)
        frozen_weights = np.zeros(n_symbols, dtype=float)
        # Causal drawdown guard: the scale is decided from equity already
        # realised through the previous close, never from the current bar.
        guard_active = False
        guard_scale = 1.0
        peak_equity = float(cfg.initial_cash)

        for i, date in enumerate(dates):
            row_open = open_values[i]
            row_close = close_values[i]
            row_bar = bar_values[i]
            row_limit_up = limit_up_values[i]
            row_limit_down = limit_down_values[i]
            row_adv = adv_values[i]

            open_equity = cash
            for j in np.flatnonzero(held_shares):
                price = row_open[j]
                if pd.notna(price):
                    open_equity += int(held_shares[j]) * float(price)

            # ---- delisting exits happen before any rebalance -----------------
            for j in exits_by_day.get(i, ()):
                quantity = int(held_shares[j])
                if quantity <= 0:
                    continue
                # Liquidate at the *open* of the final bar, not its close.
                # The decision to exit is made before that day trades, so
                # using the close would be a same-day look-ahead: the
                # backtest would "know" the final day's return before
                # selling. A real liquidation is also filled at the first
                # available price, which is the open.
                reference = float(row_open[j])
                if not np.isfinite(reference) or reference <= 0:
                    reference = float(row_close[j])
                price = reference * (1 - cfg.costs.slippage_bps / 10_000)
                cost = transaction_costs(
                    "sell", quantity, price, open_price=price, config=cfg.costs
                )
                # ``price`` is already the executed (slipped) price, so only the
                # explicit cash fees are debited; adding ``cost.slippage`` again
                # would charge the same slippage twice.
                cash += quantity * price - cost.fees
                turnover_value += quantity * price
                realized = quantity * (price - avg_cost[j]) - cost.fees
                trade_rows.append({
                    "date": date, "symbol": symbols[j], "side": "sell", "shares": quantity,
                    "price": price, "notional": quantity * price, "reason": "delisting",
                    "commission": cost.commission, "stamp_duty": cost.stamp_duty,
                    "transfer_fee": cost.transfer_fee, "slippage": cost.slippage,
                    "fees": cost.commission + cost.stamp_duty + cost.transfer_fee,
                    "realized_pnl": realized,
                })
                held_shares[j] = 0
                avg_cost[j] = 0.0
                retired.add(j)

            # ---- scheduled rebalance from the previous close's target --------
            desired = weight_values[i - 1] if i > 0 else np.zeros(n_symbols, dtype=float)
            if i == 0:
                rebalance = True
            else:
                # Compare the target with the one frozen at the last
                # rebalance. A pure weight change also requires an order even
                # when the set of held names is unchanged.
                rebalance = bool(np.any(desired != frozen_weights))

            guard_changed = False
            if cfg.drawdown_stop is not None and i > 0:
                realised = float(equity_rows[-1]["equity"])
                peak_equity = max(peak_equity, realised)
                drawdown = realised / peak_equity - 1.0 if peak_equity > 0 else 0.0
                was_active = guard_active
                if not guard_active and drawdown <= -cfg.drawdown_stop:
                    guard_active = True
                elif guard_active and drawdown > -cfg.drawdown_stop * cfg.drawdown_recover:
                    guard_active = False
                guard_scale = cfg.drawdown_de_risk if guard_active else 1.0
                guard_changed = guard_active != was_active
            if rebalance or guard_changed:
                target_lots = np.zeros(n_symbols, dtype=np.int64)
                for j in range(n_symbols):
                    if j in retired:
                        continue
                    price = row_open[j]
                    if pd.isna(price) or price <= 0 or not row_bar[j]:
                        # Keep whatever is currently held; do not invent a price.
                        target_lots[j] = held_shares[j]
                        continue
                    amount = float(desired[j]) * open_equity * guard_scale
                    target_lots[j] = int(
                        np.floor(max(amount, 0.0) / float(price) / cfg.lot_size) * cfg.lot_size
                    )
                frozen_lots = target_lots
                frozen_weights = desired.copy()
            else:
                target_lots = frozen_lots

            # ---- sells first, so the proceeds fund the buys -------------------
            for j in np.flatnonzero(held_shares):
                difference = int(held_shares[j]) - int(target_lots[j])
                if difference <= 0 or last_trade_day[j] == i:
                    continue
                if not row_bar[j] or row_limit_down[j]:
                    continue
                capacity_value = row_adv[j]
                if pd.isna(capacity_value) or capacity_value <= 0:
                    cap = 0
                else:
                    cap = int(
                        np.floor(
                            float(capacity_value) * cfg.max_adv_participation / cfg.lot_size
                        ) * cfg.lot_size
                    )
                quantity = min(difference, cap) if cap > 0 else 0
                if quantity <= 0:
                    continue
                open_price = float(row_open[j])
                if quantity * open_price < cfg.min_order_value:
                    continue
                adv_shares = float(row_adv[j])
                price = execution_price(
                    "sell", open_price, cfg.costs, shares=quantity, adv_shares=adv_shares
                )
                cost = transaction_costs("sell", quantity, price, open_price=open_price, config=cfg.costs)
                cash += quantity * price - cost.fees
                turnover_value += quantity * price
                realized = quantity * (price - avg_cost[j]) - cost.fees
                held_shares[j] -= quantity
                if held_shares[j] == 0:
                    avg_cost[j] = 0.0
                trade_rows.append({
                    "date": date, "symbol": symbols[j], "side": "sell", "shares": quantity,
                    "price": price, "notional": quantity * price, "reason": "rebalance",
                    "commission": cost.commission, "stamp_duty": cost.stamp_duty,
                    "transfer_fee": cost.transfer_fee, "slippage": cost.slippage,
                    "fees": cost.commission + cost.stamp_duty + cost.transfer_fee,
                    "realized_pnl": realized,
                })
                last_trade_day[j] = i

            buy_orders = []
            for j in np.flatnonzero(target_lots):
                if j in retired:
                    continue
                difference = int(target_lots[j]) - int(held_shares[j])
                if difference <= 0 or last_trade_day[j] == i:
                    continue
                if not row_bar[j] or row_limit_up[j]:
                    continue
                open_price = float(row_open[j])
                if open_price <= 0:
                    continue
                capacity_value = row_adv[j]
                if pd.isna(capacity_value) or capacity_value <= 0:
                    cap = 0
                else:
                    cap = int(
                        np.floor(
                            float(capacity_value) * cfg.max_adv_participation / cfg.lot_size
                        ) * cfg.lot_size
                    )
                if cap <= 0:
                    continue
                quantity = min(difference, cap)
                quantity = int(np.floor(quantity / cfg.lot_size) * cfg.lot_size)
                adv_shares = float(row_adv[j])
                price = execution_price(
                    "buy", open_price, cfg.costs, shares=quantity, adv_shares=adv_shares
                )
                if quantity <= 0 or quantity * price < cfg.min_order_value:
                    continue
                buy_orders.append((quantity, j, open_price, price))
            required_cash = sum(
                quantity * price * (1 + cfg.costs.commission_rate + cfg.costs.transfer_fee_rate)
                for quantity, _, _, price in buy_orders
            )
            cash_scale = min(1.0, cash / required_cash) if required_cash > 0 else 1.0
            for quantity, j, open_price, price in buy_orders:
                if cash_scale < 1.0:
                    quantity = int(np.floor(quantity * cash_scale / cfg.lot_size) * cfg.lot_size)
                if quantity <= 0:
                    continue
                cost = transaction_costs("buy", quantity, price, open_price=open_price, config=cfg.costs)
                total_cash = quantity * price + cost.fees
                if total_cash > cash:
                    affordable = int(
                        np.floor(cash / (price * (1 + cfg.costs.commission_rate)) / cfg.lot_size)
                        * cfg.lot_size
                    )
                    quantity = min(quantity, affordable)
                    if quantity <= 0:
                        continue
                    cost = transaction_costs("buy", quantity, price, open_price=open_price, config=cfg.costs)
                    total_cash = quantity * price + cost.fees
                    if total_cash > cash:
                        continue
                cash -= total_cash
                previous_cost = avg_cost[j] * held_shares[j]
                held_shares[j] += quantity
                avg_cost[j] = (previous_cost + total_cash) / held_shares[j]
                last_trade_day[j] = i
                turnover_value += quantity * price
                trade_rows.append({
                    "date": date, "symbol": symbols[j], "side": "buy", "shares": quantity,
                    "price": price, "notional": quantity * price, "reason": "rebalance",
                    "commission": cost.commission, "stamp_duty": cost.stamp_duty,
                    "transfer_fee": cost.transfer_fee, "slippage": cost.slippage,
                    "fees": cost.commission + cost.stamp_duty + cost.transfer_fee,
                    "realized_pnl": 0.0,
                })

            market_value = 0.0
            for j in np.flatnonzero(held_shares):
                price = row_close[j]
                if pd.isna(price):
                    price = close_filled[i, j]
                market_value += int(held_shares[j]) * float(price)
            equity = cash + market_value
            if open_equity > 0:
                exposure_sum += market_value / equity if equity else 0.0
            equity_rows.append({"date": date, "equity": equity, "cash": cash, "market_value": market_value})
            shares_history[i] = held_shares

        equity = pd.DataFrame(equity_rows).set_index("date")["equity"].astype(float)
        # Position rows were previously appended one dict per symbol per bar
        # (about 290k rows).  The same frame is built from the dense history in
        # one pass; row order is still date-major then symbol order.
        equity_values = equity.to_numpy(dtype=float)
        close_grid = close_filled.reshape(-1)
        shares_grid = shares_history.reshape(-1)
        market_grid = np.where(
            np.isfinite(close_grid), shares_grid.astype(float) * close_grid, 0.0
        )
        weight_denom = np.repeat(equity_values, n_symbols)
        weight_grid = np.divide(
            market_grid,
            weight_denom,
            out=np.zeros_like(market_grid),
            where=weight_denom != 0,
        )
        position_frame = pd.DataFrame({
            "date": np.repeat(dates.to_numpy(), n_symbols),
            "symbol": np.tile(np.array(symbols, dtype=object), len(dates)),
            "shares": shares_grid,
            "close": close_grid,
            "market_value": market_grid,
            "weight": weight_grid,
        })

        returns = equity.pct_change(fill_method=None).fillna(0.0)
        trades = pd.DataFrame(trade_rows)
        traded_notional = turnover_value
        # One-way turnover: buys and sells count as the same capital moving, so
        # the round-trip notional is halved before dividing by average equity.
        turnover = traded_notional / 2.0 / max(float(equity.mean()), 1.0)
        bench_returns = (
            benchmark.pct_change(fill_method=None).reindex(dates).fillna(0.0)
            if benchmark is not None
            else None
        )
        metrics = performance_metrics(
            returns,
            equity=equity,
            benchmark_returns=bench_returns,
            turnover=turnover,
            exposure=exposure_sum / len(dates),
        )
        active_symbols = int((position_frame.groupby("symbol")["shares"].max() > 0).sum()) if len(position_frame) else 0
        # ``retiring`` counts names whose final bar is inside the window. A name
        # can stop trading without ever being held to its last bar, so that is
        # *not* the number of forced liquidations. Count the actual exit trades.
        forced_exits = int((trades["reason"] == "delisting").sum()) if len(trades) else 0
        metadata = {
            "initial_cash": cfg.initial_cash,
            "execution_price": "next_open",
            "lot_size": cfg.lot_size,
            "max_position_weight": cfg.max_position_weight,
            "max_gross_exposure": cfg.max_gross_exposure,
            "max_adv_participation": cfg.max_adv_participation,
            "adv_includes_today": cfg.adv_includes_today,
            "symbols_in_panel": len(symbols),
            "symbols_ever_held": active_symbols,
            "delisted_candidates": len(retiring),
            "delisted_exits": forced_exits,
            "trades": len(trades),
            "traded_notional": traded_notional,
            "annual_turnover": turnover * 252.0 / max(len(dates), 1),
            "turnover_definition": (
                "turnover = one-way traded notional / average equity over the full "
                "backtest; annual_turnover = turnover * 252 / trading days"
            ),
            "bars": int(has_bar.to_numpy().sum()),
            "costs": cfg.costs.__dict__,
        }
        weight_frame = (
            position_frame.pivot(index="date", columns="symbol", values="weight").fillna(0.0)
            if len(position_frame)
            else pd.DataFrame()
        )
        return BacktestResult(
            equity=equity,
            returns=returns,
            positions=position_frame,
            trades=trades,
            weights=weight_frame,
            metrics=metrics,
            benchmark_returns=bench_returns,
            metadata=metadata,
        )


def slice_result(
    result: BacktestResult,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> BacktestResult:
    """Report a sub-period of an already-run backtest.

    The backtest must be executed on the full history so that positions and
    holdings exist on the first reported day; slicing the *input* panel instead
    would silently start the portfolio from cash.
    """
    equity = result.equity
    if start is not None:
        equity = equity.loc[equity.index >= pd.Timestamp(start)]
    if end is not None:
        equity = equity.loc[equity.index <= pd.Timestamp(end)]
    if len(equity) < 2:
        raise ValueError("the requested window has fewer than two bars")
    lo, hi = equity.index[0], equity.index[-1]
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    positions = result.positions
    trades = result.trades
    if len(positions):
        positions = positions[
            (positions["date"] >= lo) & (positions["date"] <= hi)
        ].reset_index(drop=True)
    if len(trades):
        trades = trades[(trades["date"] >= lo) & (trades["date"] <= hi)].reset_index(drop=True)
    weights = (
        positions.pivot(index="date", columns="symbol", values="weight").fillna(0.0)
        if len(positions)
        else pd.DataFrame()
    )
    exposure = float(weights.sum(axis=1).mean()) if len(weights) else float("nan")
    traded = float(trades["notional"].sum()) if len(trades) else 0.0
    metrics = performance_metrics(
        returns,
        equity=equity,
        benchmark_returns=(
            result.benchmark_returns.reindex(equity.index).fillna(0.0)
            if result.benchmark_returns is not None
            else None
        ),
        turnover=traded / 2.0 / max(float(equity.mean()), 1.0),
        exposure=exposure,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "window_start": str(lo.date()),
            "window_end": str(hi.date()),
            "window_trades": int(len(trades)),
            "window_note": "slice of a full-history run",
        }
    )
    return BacktestResult(
        equity=equity,
        returns=returns,
        positions=positions,
        trades=trades,
        weights=weights,
        metrics=metrics,
        benchmark_returns=(
            result.benchmark_returns.reindex(equity.index).fillna(0.0)
            if result.benchmark_returns is not None
            else None
        ),
        metadata=metadata,
    )


def simulate_buy_and_hold(
    panel: dict[str, pd.DataFrame],
    *,
    config: BacktestConfig | None = None,
    benchmark: pd.Series | None = None,
) -> BacktestResult:
    aligned, dates = align_panel(panel)
    symbols = sorted(aligned)
    targets = pd.DataFrame(1.0 / len(symbols), index=dates, columns=symbols)
    targets.iloc[:2] = 0.0
    return Backtester(config).run(aligned, targets, benchmark=benchmark)