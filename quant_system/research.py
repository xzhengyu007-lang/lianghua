"""Walk-forward evaluation and parameter selection without test leakage.

Every test window is preceded by a training window that ends at least
``embargo_days`` bars before the first test bar, and each window's parameters
are chosen using only data available inside its own training window.  The
backtest that produces the out-of-sample returns is run on the full history up
to the last test bar, which mirrors a strategy that has been trading live
before the test period begins.

``phase_offsets`` turns on a rebalance-calendar sweep.  A fixed ``N``-bar
cadence is anchored to the first bar of the history, which is an arbitrary
choice: shifting that anchor by one day can move a full-sample result by more
than 50pp (see RESEARCH_LOG 23.20/23.21).  With a sweep, each window reports
the median, worst and best phase in addition to the phase the headline path
actually traded.  ``phase_blend`` replaces the headline path with the average
of all phase books, which removes the phase choice entirely instead of picking
a lucky one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .backtest import Backtester, BacktestResult
from .config import BacktestConfig, ResearchConfig, RiskConfig
from .data import align_panel, usable_window
from .metrics import performance_metrics
from .portfolio import build_portfolio
from .strategies import build_strategy


@dataclass
class WindowResult:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict
    train_sharpe: float
    test_return: float
    test_sharpe: float
    test_max_drawdown: float
    test_trades: int
    positions_at_start: int = 0
    rebalances: int = 0
    exposure: float = float("nan")
    risk_variant: str | None = None
    # Rebalance-phase diagnostics.  A 20-day cadence has an arbitrary phase;
    # reporting one number hides a 100pp+ spread.  These are filled only when
    # ``walk_forward(phase_offsets=...)`` is used.
    phase_count: int = 0
    phase_return_median: float = float("nan")
    phase_return_worst: float = float("nan")
    phase_return_best: float = float("nan")
    phase_sharpe_median: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "params": self.params,
            "train_sharpe": round(float(self.train_sharpe), 4),
            "test_return": round(float(self.test_return), 6),
            "test_sharpe": round(float(self.test_sharpe), 4),
            "test_max_drawdown": round(float(self.test_max_drawdown), 6),
            "test_trades": int(self.test_trades),
            "positions_at_start": int(self.positions_at_start),
            "rebalances": int(self.rebalances),
            "exposure": round(float(self.exposure), 4),
            "risk_variant": self.risk_variant,
            "phase_count": int(self.phase_count),
            "phase_return_median": round(float(self.phase_return_median), 6),
            "phase_return_worst": round(float(self.phase_return_worst), 6),
            "phase_return_best": round(float(self.phase_return_best), 6),
            "phase_sharpe_median": round(float(self.phase_sharpe_median), 4),
        }


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame
    equity: pd.Series
    returns: pd.Series
    benchmark_returns: pd.Series | None
    selections: list[dict] = field(default_factory=list)
    turnover: float = float("nan")
    # Average *test-window* target gross.  Before this field was fixed the
    # headline exposure was the mean of ``full.metrics.exposure`` over each
    # window's whole backtest history, which includes the pre-OOS warm-up and
    # therefore mixes 2019-2023 (where the volatility target binds) with the
    # test period.  See RESEARCH_LOG 23.36.
    exposure: float = float("nan")
    exposure_full_history: float = float("nan")
    total_trades: int = 0
    audit: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> dict:
        if self.returns.empty:
            return {}
        metrics = performance_metrics(
            self.returns,
            equity=self.equity,
            benchmark_returns=self.benchmark_returns,
            turnover=self.turnover,
            exposure=self.exposure,
        )
        payload = metrics.to_dict()
        payload["exposure_full_history"] = float(self.exposure_full_history)
        return payload


def parameter_grid(grid: dict[str, Iterable]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def _window_turnover(
    result: BacktestResult,
    segment: pd.Series,
    test_slice: pd.DatetimeIndex | None = None,
) -> float:
    """One-way turnover for one evaluation window.

    ``BacktestResult.metrics.turnover`` divides the *whole-history* traded
    notional by the *whole-history* average equity.  In a walk-forward the
    history starts at ``initial_cash`` every window, so once earlier windows
    have compounded, that denominator no longer matches the capital this
    window actually traded.  Slice the trades and use the window's own mean
    equity instead.
    """
    if len(segment) < 2 or result.trades.empty:
        return 0.0
    if test_slice is not None and len(test_slice):
        # ``segment`` includes the bar *before* the window so the first day has
        # a return.  Trades, however, belong to the window itself; including
        # the base bar's rebalance would double count it in two windows.
        lo, hi = pd.Timestamp(test_slice[0]), pd.Timestamp(test_slice[-1])
    else:
        lo, hi = segment.index[0], segment.index[-1]
    trades = result.trades
    in_window = trades[(trades["date"] >= lo) & (trades["date"] <= hi)]
    traded = float(in_window["notional"].sum()) if len(in_window) else 0.0
    return traded / 2.0 / float(segment.mean())


def _window_metrics(
    result: BacktestResult,
    base_pos: int,
    start_pos: int,
    end_pos: int,
) -> tuple[pd.Series, pd.Series, float]:
    """Slice a full-history equity curve into one evaluation window.

    ``base_pos`` is the bar before the window so the first window day has a
    well-defined return.  Returns (returns, equity_segment, segment_return).
    """
    equity = result.equity
    low = max(base_pos, 0)
    segment = equity.iloc[low : end_pos + 1]
    returns = segment.pct_change(fill_method=None).dropna()
    total = float(segment.iloc[-1] / segment.iloc[0] - 1.0) if len(segment) > 1 else 0.0
    return returns, segment, total


def walk_forward(
    panel: dict[str, pd.DataFrame],
    strategy_name: str,
    grid: dict[str, Iterable],
    *,
    benchmark: pd.Series | None = None,
    research: ResearchConfig | None = None,
    backtest: BacktestConfig | None = None,
    risk: RiskConfig | None = None,
    selection_metric: str = "sharpe",
    portfolio_kwargs: dict | None = None,
    risk_variants: dict[str, RiskConfig] | None = None,
    weight_transform: Callable[[pd.DataFrame, dict[str, pd.DataFrame]], pd.DataFrame] | None = None,
    phase_offsets: Iterable[int] | None = None,
    phase_blend: bool = False,
    verbose: bool = True,
) -> WalkForwardResult:
    cfg = research or ResearchConfig()
    cfg.validate()
    bt_config = backtest or BacktestConfig()
    extra = dict(portfolio_kwargs or {})
    # Hoist the two overlay switches out of ``extra`` so the per-window
    # covariance forecast can be computed once and reused across phases.
    use_vol_target = bool(extra.pop("use_vol_target", True))
    vol_forecast = str(extra.pop("vol_forecast", "realised"))
    aligned, dates = align_panel(panel)
    # The raw union of dates can reach far back when a handful of names (or the
    # benchmark) have a longer history. Windows there have a thin cross-section
    # and produce cash-only, meaningless "0.00" scores, so the default start is
    # one full training window inside the region with >=90% coverage.
    usable_start, _ = usable_window(aligned, min_active_ratio=0.9)
    usable_pos = int(dates.searchsorted(usable_start, side="left"))
    if cfg.start:
        start_pos = int(dates.searchsorted(pd.Timestamp(cfg.start), side="left"))
        if start_pos < usable_pos and verbose:
            print(
                f"warning: start {cfg.start} precedes the 90%-coverage date "
                f"{usable_start.date()}; early windows may be thin",
                flush=True,
            )
    else:
        start_pos = usable_pos + cfg.train_days + cfg.embargo_days
    end_pos = len(dates) if not cfg.end else int(
        dates.searchsorted(pd.Timestamp(cfg.end), side="right")
    )
    if end_pos <= start_pos:
        raise ValueError("research window is empty after applying start/end")
    candidates = parameter_grid(grid)
    if not candidates:
        raise ValueError("parameter grid is empty")
    # Model-selection grid can also include *risk* configurations.  They are
    # chosen inside the training window exactly like strategy parameters, which
    # keeps the out-of-sample evaluation honest: a fixed target-vol chosen on
    # the full sample would otherwise look better simply because the test period
    # was used to pick it.
    risk_options = dict(risk_variants or {})
    if risk_options:
        candidates = [
            (params, risk_name, risk_options[risk_name])
            for params in candidates
            for risk_name in risk_options
        ]
    else:
        candidates = [(params, None, risk) for params in candidates]

    windows: list[WindowResult] = []
    selections: list[dict] = []
    oos_returns: list[pd.Series] = []
    audit_rows: list[dict] = []
    window_turnover: list[float] = []
    window_exposure: list[float] = []
    window_full_exposure: list[float] = []
    window_trades: list[int] = []

    cursor = max(cfg.train_days + cfg.embargo_days, start_pos)
    while cursor < end_pos:
        train_end_pos = cursor - cfg.embargo_days  # exclusive, so embargo is unused
        train_start_pos = train_end_pos - cfg.train_days
        if train_start_pos < 0 or train_end_pos <= train_start_pos:
            break
        test_end_pos = min(cursor + cfg.test_days, end_pos)
        if test_end_pos - cursor < 5:
            break
        train_slice = dates[train_start_pos:train_end_pos]
        test_slice = dates[cursor:test_end_pos]
        # Hard guarantee: training bars and test bars never touch.
        assert train_slice[-1] < test_slice[0]
        assert train_end_pos + cfg.embargo_days == cursor

        history_train = {s: f.loc[: train_slice[-1]] for s, f in aligned.items()}
        best_value, best_params, best_result = -np.inf, None, None
        best_risk_name: str | None = None
        best_risk: RiskConfig | None = None
        for params, risk_name, risk_cfg in candidates:
            strategy = build_strategy(strategy_name, **params)
            raw = strategy.generate(history_train)
            if weight_transform is not None:
                raw = weight_transform(raw, history_train)
            weights = build_portfolio(
                raw,
                history_train,
                benchmark=benchmark,
                risk=risk_cfg,
                use_vol_target=use_vol_target,
                vol_forecast=vol_forecast,
                **extra,
            )
            result = Backtester(bt_config).run(history_train, weights, benchmark=benchmark)
            _, segment, _ = _window_metrics(
                result, train_start_pos - 1, train_start_pos, train_end_pos - 1
            )
            if len(segment) < 3:
                continue
            train_metrics = performance_metrics(
                segment.pct_change(fill_method=None).dropna(),
                equity=segment / segment.iloc[0],
            )
            value = float(getattr(train_metrics, selection_metric))
            if not np.isfinite(value):
                value = -np.inf
            if value > best_value:
                best_value, best_params, best_result = value, params, result
                best_risk_name, best_risk = risk_name, risk_cfg
        if best_params is None:
            raise RuntimeError("no valid parameter combination in training window")
        params = best_params
        selected_risk = best_risk or risk

        # Out-of-sample: generate weights with information available at each
        # test close, execute at the following open, evaluate only test bars.
        history_test = {s: f.loc[: test_slice[-1]] for s, f in aligned.items()}
        strategy = build_strategy(strategy_name, **params)
        scores, eligible, available = strategy.prepare(history_test)
        # The covariance forecast is a function of prices and the column set,
        # not of the weights.  Compute it once per window and hand the same
        # series to all 20 phases: otherwise a sweep pays for 20 identical
        # 128x128 rolling covariance blocks, and each phase could in principle
        # see a slightly different forecast.
        ex_ante = None
        if use_vol_target and vol_forecast == "ex_ante":
            from .risk import ex_ante_volatility

            ex_ante = ex_ante_volatility(
                scores.index, sorted(history_test), history_test, selected_risk
            )

        def book_for(offset: int | None) -> pd.DataFrame:
            raw = strategy.select(scores, eligible, available, offset=offset)
            if weight_transform is not None:
                raw = weight_transform(raw, history_test)
            return build_portfolio(
                raw,
                history_test,
                benchmark=benchmark,
                risk=selected_risk,
                ex_ante_vol=ex_ante,
                use_vol_target=use_vol_target,
                vol_forecast=vol_forecast,
                **extra,
            )

        phase_books: dict[int, pd.DataFrame] = {}
        if phase_offsets:
            for offset in phase_offsets:
                phase_books[int(offset)] = book_for(int(offset))
            if phase_blend and phase_books:
                frames = list(phase_books.values())
                weights = sum(frames[1:], frames[0].copy()) / len(frames)
            elif 0 in phase_books:
                weights = phase_books[0]
            else:
                weights = book_for(None)
        else:
            weights = book_for(None)

        full = Backtester(bt_config).run(history_test, weights, benchmark=benchmark)
        test_returns, test_segment, test_total = _window_metrics(
            full, cursor - 1, cursor, test_end_pos - 1
        )
        if test_returns.empty:
            cursor += cfg.test_days
            continue

        phase_stats: dict[str, float] = {}
        if phase_books:
            phase_totals: list[float] = []
            phase_sharpes: list[float] = []
            for book in phase_books.values():
                phase_full = Backtester(bt_config).run(
                    history_test, book, benchmark=benchmark
                )
                phase_ret, phase_seg, phase_total = _window_metrics(
                    phase_full, cursor - 1, cursor, test_end_pos - 1
                )
                if phase_ret.empty:
                    continue
                phase_totals.append(float(phase_total))
                phase_sharpes.append(
                    float(
                        performance_metrics(
                            phase_ret, equity=phase_seg / phase_seg.iloc[0]
                        ).sharpe
                    )
                )
            if phase_totals:
                phase_stats = {
                    "phase_count": len(phase_totals),
                    "phase_return_median": float(np.median(phase_totals)),
                    "phase_return_worst": float(np.min(phase_totals)),
                    "phase_return_best": float(np.max(phase_totals)),
                    "phase_sharpe_median": float(np.nanmedian(phase_sharpes)),
                }

        test_weights = weights.reindex(test_slice).fillna(0.0)
        test_metrics = performance_metrics(
            test_returns,
            equity=test_segment / test_segment.iloc[0],
            exposure=float(test_weights.sum(axis=1).mean()) if len(test_weights) else float("nan"),
        )
        in_window = full.trades[full.trades["date"].isin(test_slice)] if len(full.trades) else full.trades
        positions_at_start = int(
            full.positions[full.positions["date"] == test_slice[0]]["shares"].gt(0).sum()
        ) if len(full.positions) else 0
        rebalances = int(in_window["date"].nunique()) if len(in_window) else 0

        windows.append(
            WindowResult(
                train_start=train_slice[0],
                train_end=train_slice[-1],
                test_start=test_slice[0],
                test_end=test_slice[-1],
                params=params,
                train_sharpe=float(best_value),
                test_return=float(test_total),
                test_sharpe=float(test_metrics.sharpe),
                test_max_drawdown=float(test_metrics.max_drawdown),
                test_trades=int(len(in_window)),
                positions_at_start=positions_at_start,
                rebalances=rebalances,
                exposure=float(test_metrics.exposure),
                risk_variant=best_risk_name,
                **phase_stats,
            )
        )
        selections.append(
            {
                "test_start": str(test_slice[0].date()),
                "test_end": str(test_slice[-1].date()),
                **({"risk_variant": best_risk_name} if best_risk_name else {}),
                **params,
            }
        )
        oos_returns.append(test_returns.rename("return"))
        # The window's turnover must use the *same* average equity the window
        # actually traded.  ``full.metrics.turnover`` divides by the mean equity
        # of the whole-history backtest, which starts from ``initial_cash`` and
        # therefore differs from the live window equity once earlier windows
        # have compounded.  That made turnover incomparable across runs
        # (41 vs 57 for a 2% change in return).
        window_turnover.append(_window_turnover(full, test_segment, test_slice))
        window_exposure.append(float(test_metrics.exposure))
        window_full_exposure.append(float(full.metrics.exposure))
        window_trades.append(int(len(in_window)))
        audit_rows.append(
            {
                "test_start": test_slice[0],
                "bars": len(test_slice),
                "train_bars": len(train_slice),
                "params": params,
                "risk_variant": best_risk_name,
                "trades": int(len(in_window)),
                "positions_at_start": positions_at_start,
                "cash_only": bool(positions_at_start == 0 and len(in_window) == 0),
                "exposure": float(test_metrics.exposure),
                "test_return": float(test_total),
                **phase_stats,
                "equity_start": float(test_segment.iloc[0]),
                "equity_end": float(test_segment.iloc[-1]),
            }
        )
        if verbose:
            print(
                f"window {len(windows):02d} test={test_slice[0].date()}..{test_slice[-1].date()} "
                f"params={params} risk={best_risk_name or 'default'} oos_ret={test_total:+.2%} "
                f"oos_sharpe={test_metrics.sharpe:+.2f} trades={len(in_window)}"
                + (
                    " phase_ret(med/worst/best)="
                    f"{phase_stats['phase_return_median']:+.2%}/"
                    f"{phase_stats['phase_return_worst']:+.2%}/"
                    f"{phase_stats['phase_return_best']:+.2%}"
                    if phase_stats
                    else ""
                ),
                flush=True,
            )
        cursor += cfg.test_days

    if not oos_returns:
        raise ValueError("not enough history for a walk-forward run")
    returns = pd.concat(oos_returns).sort_index()
    returns = returns[~returns.index.duplicated(keep="last")]
    equity = (1 + returns).cumprod() * bt_config.initial_cash
    bench_returns = (
        benchmark.pct_change(fill_method=None).reindex(returns.index).fillna(0.0)
        if benchmark is not None
        else None
    )
    return WalkForwardResult(
        windows=pd.DataFrame([w.to_dict() for w in windows]),
        equity=equity,
        returns=returns,
        benchmark_returns=bench_returns,
        selections=selections,
        turnover=float(np.nanmean(window_turnover)) if window_turnover else float("nan"),
        exposure=float(np.nanmean(window_exposure)) if window_exposure else float("nan"),
        exposure_full_history=(
            float(np.nanmean(window_full_exposure)) if window_full_exposure else float("nan")
        ),
        total_trades=int(sum(window_trades)),
        audit=pd.DataFrame(audit_rows),
    )


def compare_strategies(
    panel: dict[str, pd.DataFrame],
    strategy_params: dict[str, dict],
    *,
    benchmark: pd.Series | None = None,
    research: ResearchConfig | None = None,
    backtest: BacktestConfig | None = None,
    risk: RiskConfig | None = None,
    phase_offsets: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Run several strategies through the same walk-forward.

    ``phase_offsets`` is forwarded so a cross-strategy comparison is not made
    on one arbitrary calendar phase.  When it is supplied the table also
    carries the per-window median phase return, which is the number a
    comparison should actually rank on.
    """
    rows = []
    for name, grid in strategy_params.items():
        try:
            result = walk_forward(
                panel,
                name,
                grid,
                benchmark=benchmark,
                research=research,
                backtest=backtest,
                risk=risk,
                phase_offsets=phase_offsets,
                verbose=False,
            )
            row = {"strategy": name, **result.summary()}
            windows = result.windows
            if phase_offsets and not windows.empty and "phase_return_median" in windows:
                row["phase_return_median_mean"] = float(
                    windows["phase_return_median"].mean()
                )
                row["phase_return_worst_min"] = float(
                    windows["phase_return_worst"].min()
                )
        except Exception as exc:  # keep the comparison running
            row = {"strategy": name, "error": str(exc)}
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")
