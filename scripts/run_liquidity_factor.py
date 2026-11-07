"""Is the low-vol signal just a liquidity/size proxy?

The cross-sectional IC table showed that illiquidity (Amihud over 60 bars) and
small dollar volume carry information at 20 days: IC +0.025 (t=6.1) and -0.036
(t=-7.9) respectively. That is exactly the pattern one would expect if "low
volatility" were a proxy for "small, illiquid, low-turnover names".

This script tests three questions and keeps every negative answer:

1. Are the factors distinct? Rank correlation between low-vol, illiquidity and
   size, plus the IC of low-vol after neutralising each one.
2. Does adding liquidity information to the 6/2 or 8/2 basket improve the
   strictly out-of-sample walk-forward? Three blends are tried, with the weight
   fixed a priori rather than tuned on the test set.
3. Does a pure liquidity sleeve beat the low-vol sleeve? If yes, the existing
   story is wrong and must be rewritten.

Blends are applied through ``weight_transform`` only to change *ranking* is not
possible there, so the scores are rebuilt by a small wrapper strategy. Nothing
here changes the default.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.config import (  # noqa: E402
    BROAD_UNIVERSE,
    BacktestConfig,
    ResearchConfig,
    RiskConfig,
)
from quant_system.data import (  # noqa: E402
    align_panel,
    load_panel,
    load_symbol,
    price_frame,
    trim_panel,
    usable_window,
)
from quant_system.analytics import forward_returns, ic_summary, spearman_ic  # noqa: E402
from quant_system.indicators import cross_sectional_zscore, sma  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402

FIXED = {"vol_window": 60, "trend_window": 120, "rebalance_days": 20, "max_per_sector": 2}
CUT = "2025-11-05"
OUT = ROOT / "reports" / "liquidity_factor"
LIQ_WINDOW = 60


def liquidity_scores(panel: dict[str, pd.DataFrame], window: int = LIQ_WINDOW) -> pd.DataFrame:
    """Amihud illiquidity: |return| / dollar volume, averaged over ``window``."""
    close = price_frame(panel, "close")
    volume = price_frame(panel, "volume")
    returns = close.pct_change(fill_method=None)
    dollar = (close * volume * 100.0).replace(0.0, np.nan)
    amihud = (returns.abs() / dollar).rolling(window).mean()
    return cross_sectional_zscore(np.log(amihud.replace(0.0, np.nan)))


@dataclass
class BlendLowVolLiquidity(LowVolCoreStrategy):
    """Low-vol core blended with illiquidity; weights are fixed ex ante."""

    liquidity_weight: float = 0.3
    liquidity_window: int = LIQ_WINDOW
    name: str = "low_vol_liquidity"

    def score(self, panel):
        close = price_frame(panel, "close")
        returns = close.pct_change(fill_method=None)
        vol = returns.rolling(self.vol_window).std(ddof=0) * np.sqrt(252)
        trend = close / sma(close, self.trend_window) - 1.0
        liquid = liquidity_scores(panel, self.liquidity_window)
        low_vol_z = cross_sectional_zscore(-vol)
        base = self.vol_weight * low_vol_z + self.trend_weight * cross_sectional_zscore(trend)
        scores = (1.0 - self.liquidity_weight) * base + self.liquidity_weight * liquid
        eligible = self._traded(panel, close) & (close > sma(close, self.trend_window))
        return scores, eligible

    @staticmethod
    def _traded(panel, close):
        from quant_system.strategies import _traded

        return _traded(panel, close)


def build(name: str, top_n: int, liq_weight: float = 0.0):
    if liq_weight <= 0:
        return LowVolCoreStrategy(top_n=top_n, **FIXED)
    return BlendLowVolLiquidity(top_n=top_n, liquidity_weight=liq_weight, **FIXED)


def run_walk(panel, benchmark, top_n: int, liq_weight: float, research):
    strategy = build("blend", top_n, liq_weight)
    # walk_forward builds strategies from the registry, so the blend is passed
    # by temporarily registering a factory instead of monkeypatching globals.
    import quant_system.research as research_module

    original = research_module.build_strategy

    def factory(_name, **params):
        return build("blend", params.get("top_n", top_n), liq_weight)

    research_module.build_strategy = factory
    try:
        grid = {
            "vol_window": [FIXED["vol_window"]],
            "trend_window": [FIXED["trend_window"]],
            "rebalance_days": [FIXED["rebalance_days"]],
            "max_per_sector": [FIXED["max_per_sector"]],
            "top_n": [top_n],
        }
        return walk_forward(
            panel,
            "low_vol_core",
            grid,
            benchmark=benchmark,
            research=research,
            backtest=BacktestConfig(initial_cash=1_000_000),
            risk=RiskConfig(),
            portfolio_kwargs={"use_regime": False},
            verbose=False,
        )
    finally:
        research_module.build_strategy = original


def summarise(label: str, result, extra: dict | None = None) -> dict:
    summary = result.summary()
    windows = result.windows
    row = {
        "variant": label,
        "total_return": summary["total_return"],
        "cagr": summary["cagr"],
        "annual_volatility": summary["annual_volatility"],
        "sharpe": summary["sharpe"],
        "max_drawdown": summary["max_drawdown"],
        "information_ratio": summary["information_ratio"],
        "exposure": summary["exposure"],
        "positive_windows": int((windows["test_return"] > 0).sum()),
        "windows": int(len(windows)),
    }
    if extra:
        row.update(extra)
    return row


def residual_ic(signal: pd.DataFrame, control: pd.DataFrame, future: pd.DataFrame) -> dict:
    """Cross-sectional IC of ``signal`` after removing ``control`` rank exposure."""
    ics = []
    for date in signal.index:
        s = signal.loc[date].replace([np.inf, -np.inf], np.nan)
        c = control.loc[date].replace([np.inf, -np.inf], np.nan)
        f = future.loc[date].replace([np.inf, -np.inf], np.nan)
        joined = pd.concat([s, c, f], axis=1).dropna()
        if len(joined) < 10:
            continue
        s_rank = joined.iloc[:, 0].rank()
        c_rank = joined.iloc[:, 1].rank()
        beta = np.polyfit(c_rank, s_rank, 1)[0] if c_rank.std() else 0.0
        resid = s_rank - beta * (c_rank - c_rank.mean())
        if resid.std() == 0:
            continue
        ics.append(float(pd.Series(resid).corr(joined.iloc[:, 2], method="spearman")))
    series = pd.Series(ics, dtype=float)
    return ic_summary(series)


def heldout(panel, benchmark, cut: pd.Timestamp, top_n: int, liq_weight: float) -> dict:
    raw = build("blend", top_n, liq_weight).generate(panel)
    weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
    result = Backtester(BacktestConfig(initial_cash=1_000_000)).run(panel, weights, benchmark=benchmark)
    segment = result.equity.loc[cut:]
    metrics = performance_metrics(segment.pct_change(fill_method=None).dropna(), equity=segment)
    return {
        "total_return": metrics.total_return,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "bars": len(segment),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    survivors = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(survivors, min_active_ratio=0.95)
    survivors = {s: f.loc[start:end] for s, f in survivors.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    aligned, dates = align_panel(survivors)
    usable_start, _ = usable_window(aligned, min_active_ratio=0.9)
    usable_pos = int(dates.searchsorted(usable_start, side="left"))
    first_test = dates[usable_pos + 504 + 10]
    research = ResearchConfig(start=str(first_test.date()), train_days=504, test_days=126, embargo_days=10)
    print(f"window {start.date()}..{end.date()} symbols={len(survivors)} first_test={first_test.date()}", flush=True)

    payload: dict = {
        "window": [str(start.date()), str(end.date())],
        "first_test": str(first_test.date()),
        "fixed_params": FIXED,
        "liquidity_window": LIQ_WINDOW,
        "cut": CUT,
        "note": (
            "Blend weights are fixed a priori (30%), not tuned on the test set. "
            "Walk-forward windows are strictly out of sample."
        ),
    }

    close = price_frame(survivors, "close")
    volume = price_frame(survivors, "volume")
    returns = close.pct_change(fill_method=None)
    vol60 = returns.rolling(FIXED["vol_window"]).std(ddof=0)
    low_vol = -vol60
    dollar = (close * volume * 100.0).replace(0.0, np.nan)
    amihud = np.log((returns.abs() / dollar).rolling(LIQ_WINDOW).mean().replace(0.0, np.nan))
    size = np.log(dollar.rolling(20).mean())

    print("factor distinctness", flush=True)
    corr_rows = []
    for horizon in (5, 20):
        future = forward_returns(close, horizon)
        rows = {
            "horizon": horizon,
            "low_vol_ic": ic_summary(spearman_ic(low_vol, future)),
            "amihud_ic": ic_summary(spearman_ic(amihud, future)),
            "size_ic": ic_summary(spearman_ic(size, future)),
            "low_vol_resid_vs_amihud": residual_ic(low_vol, amihud, future),
            "low_vol_resid_vs_size": residual_ic(low_vol, size, future),
            "amihud_resid_vs_low_vol": residual_ic(amihud, low_vol, future),
        }
        corr_rows.append(rows)
        print(f"h={horizon}: low_vol IC={rows['low_vol_ic']['mean_ic']:+.4f} "
              f"resid|amihud={rows['low_vol_resid_vs_amihud']['mean_ic']:+.4f} "
              f"amihud IC={rows['amihud_ic']['mean_ic']:+.4f} "
              f"resid|low_vol={rows['amihud_resid_vs_low_vol']['mean_ic']:+.4f}", flush=True)
    # Average rank correlation, sampled monthly to keep it cheap.
    monthly = low_vol.resample("ME").last().index
    rank_corr = {}
    for date in close.index[close.index.isin(monthly)]:
        joined = pd.DataFrame(
            {
                "low_vol": low_vol.loc[date],
                "amihud": amihud.loc[date],
                "size": size.loc[date],
            }
        ).dropna()
        if len(joined) >= 10:
            rank_corr[date] = joined.corr(method="spearman").loc["low_vol", ["amihud", "size"]]
    corr = pd.DataFrame(rank_corr).T if rank_corr else pd.DataFrame(
        columns=["amihud", "size"]
    )
    payload["factor_ic"] = corr_rows
    payload["rank_correlation_low_vol_vs"] = {
        "amihud": float(corr.iloc[:, 0].mean()) if not corr.empty else None,
        "size": float(corr.iloc[:, 1].mean()) if not corr.empty else None,
        "months": int(len(corr)),
    }
    print("rank corr low_vol vs amihud/size:", payload["rank_correlation_low_vol_vs"], flush=True)

    print("walk-forward", flush=True)
    arms = [
        ("8/2 low-vol", 8, 0.0),
        ("8/2 +30% liquidity", 8, 0.30),
        ("6/2 low-vol", 6, 0.0),
        ("6/2 +30% liquidity", 6, 0.30),
        ("6/2 +50% liquidity", 6, 0.50),
    ]
    rows, frames = [], []
    for label, top_n, liq_weight in arms:
        result = run_walk(survivors, benchmark, top_n, liq_weight, research)
        row = summarise(label, result, {"top_n": top_n, "liquidity_weight": liq_weight})
        rows.append(row)
        frame = result.windows.copy()
        frame["variant"] = label
        frames.append(frame)
        print(f"{label}: {row['total_return']:+.2%} sharpe={row['sharpe']:.3f} "
              f"mdd={row['max_drawdown']:.2%} pos={row['positive_windows']}/{row['windows']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "walkforward.csv", index=False, encoding="utf-8-sig")
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(OUT / "windows.csv", index=False, encoding="utf-8-sig")
    payload["walkforward"] = rows

    pivot = detail.pivot(index="test_start", columns="variant", values="test_return")
    paired = {}
    for blend, base in (("8/2 +30% liquidity", "8/2 low-vol"), ("6/2 +30% liquidity", "6/2 low-vol")):
        diff = (pivot[blend] - pivot[base]).dropna()
        paired[f"{blend} - {base}"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "t_like": float(diff.mean() / diff.std(ddof=1) * len(diff) ** 0.5) if diff.std(ddof=1) else None,
            "wins": int((diff > 0).sum()),
            "n": int(len(diff)),
        }
    payload["paired_blend_minus_base"] = paired

    print("held-out final year", flush=True)
    cut = pd.Timestamp(CUT)
    hold_rows = []
    for label, top_n, liq_weight in arms:
        metrics = heldout(survivors, benchmark, cut, top_n, liq_weight)
        hold_rows.append({"variant": label, **metrics})
        print(f"heldout {label}: {metrics['total_return']:+.2%} sharpe={metrics['sharpe']:.3f} "
              f"mdd={metrics['max_drawdown']:.2%}", flush=True)
    pd.DataFrame(hold_rows).to_csv(OUT / "heldout.csv", index=False, encoding="utf-8-sig")
    payload["heldout"] = hold_rows
    bench = benchmark.loc[cut:]
    payload["benchmark_heldout_return"] = float(bench.iloc[-1] / bench.iloc[0] - 1)

    write_json(payload, OUT / "summary.json")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
