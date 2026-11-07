"""Causal stress de-leveraging: cut exposure when the market is in a drawdown.

The diversification audit found the book's beta rises from 0.48 to 0.70 on the
worst benchmark days. A natural fix is to reduce exposure after the benchmark
enters a drawdown. That idea has failed before in this project (the 20/60 trend
switch), so this script is deliberately narrow and causal:

* the signal uses the benchmark's close and its trailing peak **up to the
  previous bar**, never the current bar's close,
* the scale is locked at the signal bar and held until the next signal, exactly
  like the existing volatility target,
* every de-risked arm is paired with a "matched exposure" control that keeps
  the baseline composition but scales the whole book by the same factor.

A note on the control: a *uniform* multiplier applied to every name cannot be
distinguished from scaling the whole book, so for uniform rules the control is
mathematically identical. That is not a bug, it is a proof that a uniform
de-risking rule contains no information beyond exposure. The control only
becomes informative for a rule that changes the *composition* (below).

Four rules are tested: benchmark drawdown > 5/10/15% and a 20/60 trend switch,
all on the default 8/2 configuration and on the 6/2 candidate. Negative results
are kept. Nothing changes a default.
"""

from __future__ import annotations

import sys
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
    trim_panel,
    usable_window,
)
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

FIXED = {"vol_window": 60, "trend_window": 120, "rebalance_days": 20, "max_per_sector": 2}
CUT = "2025-11-05"
OUT = ROOT / "reports" / "stress_deleveraging"


def drawdown_scale(benchmark: pd.Series, threshold: float, floor: float = 0.5) -> pd.Series:
    """Scale in [floor, 1] from the benchmark drawdown known at the prior close."""
    level = benchmark.shift(1)
    peak = level.cummax()
    drawdown = (level / peak - 1.0).fillna(0.0)
    severity = (drawdown.abs() / threshold).clip(0.0, 1.0)
    return (1.0 - (1.0 - floor) * severity).clip(floor, 1.0)


def trend_scale(benchmark: pd.Series, fast: int = 20, slow: int = 60) -> pd.Series:
    """Classic 20/60 switch, shifted one bar so only prior closes are used."""
    level = benchmark.shift(1)
    fast_ma = level.rolling(fast).mean()
    slow_ma = level.rolling(slow).mean()
    risk_on = (level > slow_ma) & (fast_ma > slow_ma)
    return risk_on.astype(float).reindex(benchmark.index).fillna(0.0)


def scale_weights(raw: pd.DataFrame, scale: pd.Series) -> pd.DataFrame:
    return raw.mul(scale.reindex(raw.index).fillna(1.0), axis=0)


def matched(raw: pd.DataFrame, scaled: pd.DataFrame) -> pd.DataFrame:
    """Baseline composition at the scaled book's total exposure, same date."""
    base_total = raw.sum(axis=1).replace(0, np.nan)
    target_total = scaled.sum(axis=1)
    factor = (target_total / base_total).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    return raw.mul(factor, axis=0).fillna(0.0)


def beta_tilt_scale(
    raw: pd.DataFrame, benchmark: pd.Series, panel: dict[str, pd.DataFrame], threshold: float
) -> pd.DataFrame:
    """Cut only the names with the highest benchmark beta, keep the rest whole.

    A uniform multiplier has no information beyond exposure. This rule instead
    removes a fraction of the *high-beta* names (relative to their own trailing
    beta to the benchmark) and redistributes their weight to the remaining
    names. If low-vol names are the defensive ones, this should help; if they
    are not (section 23.14 says the stress beta rises), it will not.
    """
    from quant_system.data import price_frame

    close = price_frame(panel, "close").reindex(columns=raw.columns)
    returns = close.pct_change(fill_method=None)
    bench_returns = benchmark.pct_change(fill_method=None).reindex(returns.index)
    betas = {}
    for column in raw.columns:
        pair = pd.concat([returns[column], bench_returns], axis=1).dropna().tail(120)
        betas[column] = (
            float(np.polyfit(pair.iloc[:, 1], pair.iloc[:, 0], 1)[0])
            if len(pair) > 20 and pair.iloc[:, 1].std() > 0
            else 1.0
        )
    severity = drawdown_scale(benchmark, threshold)
    result = raw.copy()
    for date in raw.index:
        row = raw.loc[date]
        held = row[row > 0]
        if held.empty:
            continue
        cut = 1.0 - float(severity.loc[date])
        if cut <= 1e-9:
            continue
        ordered = held.sort_values(ascending=False)
        high_beta = [c for c in ordered.index if betas.get(c, 1.0) > 0]
        drop = set(high_beta[: max(1, int(round(len(high_beta) * cut)))])
        keep = [c for c in held.index if c not in drop]
        if not keep:
            continue
        freed = float(held[list(drop)].sum())
        base = float(held[keep].sum())
        for column in keep:
            result.loc[date, column] = float(row[column]) + freed * float(row[column]) / base
        for column in drop:
            result.loc[date, column] = 0.0
    return result


def transform(kind: str | None, benchmark: pd.Series, panel: dict[str, pd.DataFrame], *, matched_exposure: bool = False):
    if kind is None:
        return None

    def apply(raw: pd.DataFrame, _panel) -> pd.DataFrame:
        if kind.startswith("dd"):
            threshold = float(kind.split(":")[1])
            scale = drawdown_scale(benchmark, threshold)
        elif kind.startswith("tilt"):
            threshold = float(kind.split(":")[1])
            tilted = beta_tilt_scale(raw, benchmark, panel, threshold)
            return matched(raw, tilted) if matched_exposure else tilted
        else:
            scale = trend_scale(benchmark)
        scaled = scale_weights(raw, scale)
        return matched(raw, scaled) if matched_exposure else scaled

    return apply


def run_walk(panel, benchmark, top_n: int, kind, research, *, matched_exposure=False):
    grid = {
        "vol_window": [FIXED["vol_window"]],
        "trend_window": [FIXED["trend_window"]],
        "rebalance_days": [FIXED["rebalance_days"]],
        "max_per_sector": [FIXED["max_per_sector"]],
        "top_n": [top_n],
    }
    return walk_forward(
        panel, "low_vol_core", grid, benchmark=benchmark, research=research,
        backtest=BacktestConfig(initial_cash=1_000_000), risk=RiskConfig(),
        portfolio_kwargs={"use_regime": False},
        weight_transform=transform(kind, benchmark, panel, matched_exposure=matched_exposure),
        verbose=False,
    )


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


def heldout(panel, benchmark, cut: pd.Timestamp, top_n: int, kind, matched_exposure: bool) -> dict:
    raw = build_strategy("low_vol_core", **{**FIXED, "top_n": top_n}).generate(panel)
    apply = transform(kind, benchmark, panel, matched_exposure=matched_exposure)
    if apply is not None:
        raw = apply(raw, panel)
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
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {s: f.loc[start:end] for s, f in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    aligned, dates = align_panel(panel)
    usable_start, _ = usable_window(aligned, min_active_ratio=0.9)
    usable_pos = int(dates.searchsorted(usable_start, side="left"))
    first_test = dates[usable_pos + 504 + 10]
    research = ResearchConfig(start=str(first_test.date()), train_days=504, test_days=126, embargo_days=10)
    print(f"window {start.date()}..{end.date()} symbols={len(panel)} first_test={first_test.date()}", flush=True)

    payload: dict = {
        "window": [str(start.date()), str(end.date())],
        "first_test": str(first_test.date()),
        "fixed_params": FIXED,
        "cut": CUT,
        "note": (
            "All scales use benchmark closes up to the previous bar only. "
            "Matched-exposure arms keep the baseline composition and scale the "
            "whole book to the same exposure, isolating composition effects."
        ),
    }

    arms = [
        ("8/2 no stress rule", 8, None, False),
        ("8/2 dd>5%", 8, "dd:0.05", False),
        ("8/2 dd>5% matched", 8, "dd:0.05", True),
        ("8/2 dd>10%", 8, "dd:0.10", False),
        ("8/2 dd>10% matched", 8, "dd:0.10", True),
        ("8/2 dd>15%", 8, "dd:0.15", False),
        ("8/2 20/60 trend", 8, "trend", False),
        ("8/2 20/60 matched", 8, "trend", True),
        ("8/2 dd>10% beta-tilt", 8, "tilt:0.10", False),
        ("8/2 dd>10% beta-tilt matched", 8, "tilt:0.10", True),
        ("6/2 dd>10%", 6, "dd:0.10", False),
        ("6/2 dd>10% matched", 6, "dd:0.10", True),
        ("6/2 dd>10% beta-tilt", 6, "tilt:0.10", False),
        ("6/2 dd>10% beta-tilt matched", 6, "tilt:0.10", True),
    ]

    print("walk-forward", flush=True)
    rows, frames = [], []
    for label, top_n, kind, is_matched in arms:
        result = run_walk(panel, benchmark, top_n, kind, research, matched_exposure=is_matched)
        row = summarise(label, result, {"top_n": top_n, "rule": kind, "matched": is_matched})
        rows.append(row)
        frame = result.windows.copy()
        frame["variant"] = label
        frames.append(frame)
        print(f"{label}: {row['total_return']:+.2%} sharpe={row['sharpe']:.3f} "
              f"mdd={row['max_drawdown']:.2%} exposure={row['exposure']:.3f} "
              f"pos={row['positive_windows']}/{row['windows']}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "walkforward.csv", index=False, encoding="utf-8-sig")
    detail = pd.concat(frames, ignore_index=True)
    detail.to_csv(OUT / "windows.csv", index=False, encoding="utf-8-sig")
    payload["walkforward"] = rows

    pivot = detail.pivot(index="test_start", columns="variant", values="test_return")
    paired = {}
    for rule, control in (
        ("8/2 dd>5%", "8/2 dd>5% matched"),
        ("8/2 dd>10%", "8/2 dd>10% matched"),
        ("8/2 20/60 trend", "8/2 20/60 matched"),
        ("8/2 dd>10% beta-tilt", "8/2 dd>10% beta-tilt matched"),
        ("6/2 dd>10%", "6/2 dd>10% matched"),
        ("6/2 dd>10% beta-tilt", "6/2 dd>10% beta-tilt matched"),
    ):
        diff = (pivot[rule] - pivot[control]).dropna()
        paired[f"{rule} - matched"] = {
            "mean_diff": float(diff.mean()),
            "std_diff": float(diff.std(ddof=1)),
            "t_like": float(diff.mean() / diff.std(ddof=1) * len(diff) ** 0.5) if diff.std(ddof=1) else None,
            "wins": int((diff > 0).sum()),
            "n": int(len(diff)),
        }
    payload["paired_rule_minus_matched"] = paired

    print("held-out final year", flush=True)
    cut = pd.Timestamp(CUT)
    hold_rows = []
    for label, top_n, kind, is_matched in arms:
        metrics = heldout(panel, benchmark, cut, top_n, kind, is_matched)
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
