"""Can anything predict when the low-vol book will lose to the equal-weight pool?

RESEARCH_LOG 23.44-23.52 established two facts: the strategy is defensive, and
its excess return is concentrated in months when the equal-weight pool falls.
The obvious follow-up is "then cut exposure when the pool is about to fall".
This script tests that directly instead of assuming it.

Every predictor is built from information available at the month-end signal
close:

    breadth         share of traded names above their 120-day moving average
    vol_dispersion  interquartile range of 60-day realised volatility
    index_mom12     trailing 12-month CSI300 return
    index_vol60     60-day realised CSI300 volatility
    factor_mom12    trailing 12-month low-vol-basket minus equal-weight return

Target: next calendar month's return difference (strategy - equal weight).

Three layers of evidence:

1. univariate OLS with Newey-West standard errors (lag 3);
2. the same regression on the *excess* return and on the strategy return
   separately, so a predictor that only times the market is visible;
3. a causal timing rule: within expanding history, scale the book to 0.25 when
   the predictor is in its bottom tercile, otherwise hold 1.0. Thresholds use
   only past months, and the position applies to the following month.

Usage::

    python scripts/run_timing_study.py
    python scripts/run_timing_study.py --lags 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe, price_frame  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def newey_west(y: np.ndarray, x: np.ndarray, lags: int) -> dict:
    """Univariate OLS with HAC standard errors."""
    n = len(y)
    design = np.column_stack([np.ones(n), x])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    omega = (design * resid[:, None]).T @ (design * resid[:, None]) / n
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = (design[lag:] * resid[lag:, None]).T @ (design[:-lag] * resid[:-lag, None]) / n
        omega += weight * (gamma + gamma.T)
    # omega is the *mean* outer product; the sandwich needs the sum, i.e. n*omega.
    cov = xtx_inv @ omega @ xtx_inv * n
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return {
        "slope": float(beta[1]),
        "t": float(beta[1] / se) if se > 0 else float("nan"),
        "r2": float(1 - resid.var() / y.var()) if y.var() > 0 else float("nan"),
        "n": int(n),
    }


def build_predictors(aligned: dict[str, pd.DataFrame], benchmark: pd.Series, strategy_returns: pd.Series) -> pd.DataFrame:
    close = price_frame(aligned, "close")
    traded = close.notna() & (price_frame(aligned, "volume").reindex_like(close) > 0)
    sma120 = close.rolling(120, min_periods=60).mean()
    breadth = ((close > sma120) & traded).sum(axis=1) / traded.sum(axis=1).replace(0, np.nan)
    vol = close.pct_change(fill_method=None).rolling(60, min_periods=30).std() * np.sqrt(252)
    dispersion = vol.quantile(0.75, axis=1) - vol.quantile(0.25, axis=1)

    bench = benchmark.reindex(close.index).ffill()
    bench_ret = bench.pct_change()
    bench_mom12 = bench / bench.shift(252) - 1.0
    bench_vol60 = bench_ret.rolling(60, min_periods=30).std() * np.sqrt(252)

    equal_weight = close.pct_change().mean(axis=1, skipna=True)
    factor_cum = (1 + strategy_returns).cumprod() / (1 + equal_weight.reindex(strategy_returns.index)).cumprod()
    factor_mom12 = factor_cum / factor_cum.shift(252) - 1.0
    factor_mom12 = factor_mom12.reindex(close.index).ffill()

    frame = pd.DataFrame(
        {
            "breadth": breadth,
            "vol_dispersion": dispersion,
            "index_mom12": bench_mom12,
            "index_vol60": bench_vol60,
            "factor_mom12": factor_mom12,
        }
    )
    return frame


def monthly(series: pd.Series) -> pd.Series:
    return (1 + series).groupby([series.index.year, series.index.month]).prod() - 1.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "timing_study"))
    parser.add_argument("--lags", type=int, default=3)
    args = parser.parse_args(argv)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print(f"panel: {len(aligned)} symbols, {len(next(iter(aligned.values())))} bars", flush=True)

    strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
    weights = build_portfolio(
        strategy.generate(aligned),
        aligned,
        risk=RiskConfig(),
        use_regime=False,
        use_vol_target=True,
    )
    result = Backtester(BacktestConfig()).run(aligned, weights)
    close = price_frame(aligned, "close")
    equal_weight = close.pct_change().mean(axis=1, skipna=True).dropna()

    bench_raw = pd.read_parquet(Path(args.cache_dir) / "sh000300_raw.parquet")
    bench_raw.index = pd.to_datetime(bench_raw.index)
    benchmark = bench_raw["close"].astype(float)

    predictors = build_predictors(aligned, benchmark, result.returns)

    strat_m = monthly(result.returns)
    bench_m = monthly(equal_weight.reindex(result.returns.index))
    # Month-end predictor value known before the next month starts.
    pred_m = predictors.groupby([predictors.index.year, predictors.index.month]).last()
    pred_m.index = strat_m.index

    target = (strat_m - bench_m).rename("excess")
    frame = pd.concat([target, strat_m.rename("strategy"), bench_m.rename("equal_weight"), pred_m], axis=1)
    frame = frame.iloc[12:].dropna(subset=["excess"])  # need a year for momentum
    print(f"months used: {len(frame)}", flush=True)

    rows = []
    for column in predictors.columns:
        x = frame[column].to_numpy(dtype=float)
        for metric in ("excess", "strategy", "equal_weight"):
            y = frame[metric].to_numpy(dtype=float)
            stats = newey_west(y, x, args.lags)
            rows.append({"predictor": column, "target": metric, **stats})
    regressions = pd.DataFrame(rows)

    # Causal timing rule: expanding terciles, position applied to the next month.
    scale_rows = []
    for column in predictors.columns:
        values = frame[column]
        positions = []
        for i in range(len(frame)):
            history = values.iloc[:i]
            if len(history) < 24:
                positions.append(1.0)
                continue
            low, high = history.quantile(1 / 3), history.quantile(2 / 3)
            current = values.iloc[i]
            # Bottom tercile = de-risk; the sign of "bad" is checked in-sample
            # only, so this is a test of the rule, not of a fitted sign.
            positions.append(0.25 if current <= low else 1.0)
        pos = pd.Series(positions, index=frame.index)
        strat = frame["strategy"] * pos
        bench = frame["equal_weight"]
        scale_rows.append(
            {
                "rule": f"{column} bottom tercile -> 0.25",
                "strategy_total": float((1 + strat).prod() - 1),
                "benchmark_total": float((1 + bench).prod() - 1),
                "strategy_sharpe": float(strat.mean() / strat.std() * np.sqrt(12)) if strat.std() > 0 else float("nan"),
                "benchmark_sharpe": float(bench.mean() / bench.std() * np.sqrt(12)) if bench.std() > 0 else float("nan"),
                "months_in_cash": int((pos < 1).sum()),
            }
        )
    base_strat = frame["strategy"]
    base_bench = frame["equal_weight"]
    scale_rows.insert(
        0,
        {
            "rule": "no timing (always 1.0)",
            "strategy_total": float((1 + base_strat).prod() - 1),
            "benchmark_total": float((1 + base_bench).prod() - 1),
            "strategy_sharpe": float(base_strat.mean() / base_strat.std() * np.sqrt(12)),
            "benchmark_sharpe": float(base_bench.mean() / base_bench.std() * np.sqrt(12)),
            "months_in_cash": 0,
        },
    )
    rules = pd.DataFrame(scale_rows)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "monthly.csv", encoding="utf-8-sig")
    regressions.to_csv(output / "regressions.csv", index=False, encoding="utf-8-sig")
    rules.to_csv(output / "timing_rules.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "params": PARAMS,
                "months": int(len(frame)),
                "start": str(frame.index[0]),
                "end": str(frame.index[-1]),
                "regressions": rows,
                "timing_rules": scale_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = ["# 择时检验：能不能预判低波组合跑输的月份", ""]
    lines.append(f"月度样本 {frame.index[0]} .. {frame.index[-1]}，共 {len(frame)} 个月；"
                 "Newey-West 滞后 3 期。")
    lines.append("")
    lines.append("## 单变量回归")
    lines.append("")
    lines.append("| 预测变量 | 目标 | 斜率 | t (NW) | R² |")
    lines.append("|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {predictor} | {target} | {slope:+.4f} | {t:+.2f} | {r2:.3f} |".format(**row)
        )
    lines.append("")
    lines.append("## 因果择时规则（扩张窗口三分位，次月生效）")
    lines.append("")
    lines.append("| 规则 | 策略累计 | 等权累计 | 策略 Sharpe | 等权 Sharpe | 减仓月数 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in scale_rows:
        lines.append(
            "| {rule} | {strategy_total:+.1%} | {benchmark_total:+.1%} | {strategy_sharpe:.2f} | "
            "{benchmark_sharpe:.2f} | {months_in_cash} |".format(**row)
        )
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(regressions.to_string(index=False))
    print()
    print(rules.to_string(index=False))
    try:
        shown = output.resolve().relative_to(ROOT)
    except ValueError:
        shown = output
    print(f"report: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
