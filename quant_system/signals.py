"""Daily signal generation for paper trading and manual execution.

The signal is produced from data available at the last close and is meant to
be executed at the next open, matching the backtester's execution model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BacktestConfig, RiskConfig
from .data import availability, infer_price_limit, price_frame
from .portfolio import build_portfolio, normalize_targets
from .sectors import sector_of
from .strategies import build_strategy


@dataclass
class DailySignal:
    as_of: pd.Timestamp
    strategy: str
    params: dict
    regime_on: bool | None
    target_weights: pd.Series
    rows: pd.DataFrame
    watchlist: pd.DataFrame
    warnings: list[str] = field(default_factory=list)
    sizing_equity: float | None = None
    lot_size: int = 100

    @property
    def gross_exposure(self) -> float:
        return float(self.target_weights.sum())

    def to_dict(self) -> dict:
        return {
            "as_of": str(self.as_of.date()),
            "execution": "next trading day open",
            "strategy": self.strategy,
            "params": self.params,
            "regime_on": self.regime_on,
            "regime_applied": self.regime_on is not None,
            "gross_exposure": round(self.gross_exposure, 6),
            "sizing_equity": self.sizing_equity,
            "lot_size": self.lot_size,
            "planned_notional": round(
                float(self.rows["amount"].fillna(0.0).sum())
                if len(self.rows) and "amount" in self.rows
                else 0.0,
                2,
            ),
            "warnings": self.warnings,
            "orders": self.rows.to_dict(orient="records"),
            "watchlist": self.watchlist.to_dict(orient="records"),
        }


def _regime_on(benchmark: pd.Series | None, as_of: pd.Timestamp) -> bool:
    if benchmark is None:
        return True
    history = benchmark.loc[:as_of]
    if len(history) < 60:
        return True
    slow = history.rolling(60).mean()
    return bool(history.iloc[-1] > slow.iloc[-1] and history.rolling(20).mean().iloc[-1] > slow.iloc[-1])


def latest_signal(
    panel: dict[str, pd.DataFrame],
    strategy_name: str,
    *,
    params: dict | None = None,
    benchmark: pd.Series | None = None,
    risk: RiskConfig | None = None,
    cash: float = 1_000_000.0,
    lot_size: int = 100,
    use_regime: bool = False,
    use_vol_target: bool = True,
    vol_forecast: str = "realised",
    equity: float | None = None,
    max_position_weight: float = 0.25,
    max_gross_exposure: float = 0.95,
    today: pd.Timestamp | None = None,
    max_age_bdays: int = 5,
) -> DailySignal:
    """Target book from the last close, sized for the next open.

    The raw strategy output is a *relative* score, not an executable book: a
    2-name basket is 50% per name and 100% gross. The backtester clamps the
    single-name weight and then scales the book to the gross limit, so the live
    signal applies the exact same transformation here (``normalize_targets``).
    Without this the CLI would have recommended positions the backtest never
    held.

    ``cash`` is the account's buying power used when ``equity`` is not given.
    ``equity`` should be total account equity (cash + holdings at the signal
    close); the backtester sizes target lots on equity at the next open, not on
    leftover cash, and sizing on cash would systematically under-invest.
    """
    if not panel:
        raise ValueError("panel is empty")
    params = dict(params or {})
    strategy = build_strategy(strategy_name, **params)
    raw = strategy.generate(panel)
    weights = build_portfolio(
        raw,
        panel,
        benchmark=benchmark,
        risk=risk,
        use_regime=use_regime,
        use_vol_target=use_vol_target,
        vol_forecast=vol_forecast,
    )
    if weights.empty:
        raise ValueError("strategy produced no weights")
    as_of = weights.index[-1]
    limits = BacktestConfig(
        max_position_weight=max_position_weight,
        max_gross_exposure=max_gross_exposure,
    )
    capped = normalize_targets(
        weights, sorted(panel), weights.index, limits
    ).loc[as_of]
    target = capped.fillna(0.0)
    target = target[target > 1e-9].sort_values(ascending=False)
    sizing_equity = float(cash if equity is None else equity)
    if sizing_equity <= 0:
        raise ValueError("equity must be positive")

    close = price_frame(panel, "close")
    traded = availability(panel)
    warnings: list[str] = []
    if today is not None:
        age = int(np.busday_count(as_of.date(), pd.Timestamp(today).normalize().date()))
        if age > max_age_bdays:
            warnings.append(
                f"面板最新交易日 {as_of.date()} 已是 {age} 个交易日之前，"
                "请确认数据源已更新再执行这批指令"
            )
    rows: list[dict] = []
    # The strategy score explains *why* each name is in the book, and the
    # watchlist needs the same numbers.  Computing it once here also halves the
    # cost of a daily signal, because ``score`` rebuilds the full volatility
    # panel and was previously called a second time for the watchlist only.
    score_row = pd.Series(dtype=float)
    eligible_row: pd.Series | None = None
    try:
        scores, eligible = strategy.score(panel)
        score_row = scores.loc[as_of].dropna()
        if eligible is not None:
            eligible_row = (
                eligible.loc[as_of]
                .reindex(score_row.index)
                .fillna(False)
                .astype(bool)
            )
    except (AttributeError, KeyError, NotImplementedError) as exc:  # pragma: no cover
        warnings.append(f"strategy score unavailable: {exc}")

    def _score_of(symbol: str) -> float | None:
        if symbol not in score_row.index:
            return None
        value = float(score_row.loc[symbol])
        return round(value, 4) if np.isfinite(value) else None

    for symbol, weight in target.items():
        price = float(close.loc[as_of, symbol]) if symbol in close else np.nan
        shares = (
            int(np.floor(weight * sizing_equity / price / lot_size) * lot_size)
            if price > 0
            else 0
        )
        bars = traded[symbol].loc[:as_of] if symbol in traded else pd.Series(dtype=bool)
        last_real = bars[bars].index.max() if len(bars) and bars.any() else None
        stale = last_real is None or last_real < as_of
        limit = infer_price_limit(symbol)
        last_return = float(close[symbol].pct_change(fill_method=None).loc[as_of]) if len(close[symbol].dropna()) > 1 else 0.0
        flags = []
        if stale:
            flags.append("no_bar_today")
            warnings.append(f"{symbol}: no bar on {as_of.date()}; order may not fill")
        if abs(last_return) >= limit * 0.98:
            flags.append("limit_move")
            warnings.append(f"{symbol}: moved {last_return:+.2%} today, near the {limit:.0%} limit")
        rows.append(
            {
                "symbol": symbol,
                "sector": sector_of(symbol),
                "weight": round(float(weight), 6),
                "price": round(price, 4) if np.isfinite(price) else None,
                "shares": shares,
                "amount": round(shares * price, 2) if np.isfinite(price) else None,
                "score": _score_of(symbol),
                "flags": ",".join(flags),
            }
        )

    # Watchlist: strongest eligible names that did not make the basket, so the
    # user can see which holdings are close to being replaced.
    latest = score_row
    if eligible_row is not None:
        latest = latest[eligible_row.reindex(latest.index).fillna(False).astype(bool)]
    latest = latest.sort_values(ascending=False)
    outsiders = [(s, float(latest[s])) for s in latest.index if s not in target][:5]
    watch = pd.DataFrame(
        [
            {"symbol": s, "sector": sector_of(s), "score": round(v, 4)}
            for s, v in outsiders
        ]
    )
    planned = float(sum(row["amount"] or 0.0 for row in rows))
    if planned > sizing_equity + 1e-6:
        warnings.append(
            f"planned notional {planned:,.0f} exceeds sizing equity "
            f"{sizing_equity:,.0f}; the paper ledger will scale buys down to "
            "available cash after sells"
        )
    if target.empty:
        if use_regime and not _regime_on(benchmark, as_of):
            warnings.append(
                "regime filter is off (benchmark below its 60-day average): "
                "the signal is intentionally 100% cash"
            )
        else:
            warnings.append("no eligible names: the signal is 100% cash")
    return DailySignal(
        as_of=as_of,
        strategy=strategy_name,
        params=params,
        regime_on=_regime_on(benchmark, as_of) if use_regime else None,
        target_weights=target,
        rows=pd.DataFrame(rows),
        watchlist=watch,
        warnings=warnings,
        sizing_equity=sizing_equity,
        lot_size=lot_size,
    )


def save_signal(signal: DailySignal, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    payload = signal.to_dict()
    (output / "signal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not signal.rows.empty:
        signal.rows.to_csv(output / "orders.csv", index=False, encoding="utf-8-sig")
    if not signal.watchlist.empty:
        signal.watchlist.to_csv(output / "watchlist.csv", index=False, encoding="utf-8-sig")
    return output