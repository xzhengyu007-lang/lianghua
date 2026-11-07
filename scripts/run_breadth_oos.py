"""Out-of-sample test of the one timing signal that survived a t=-3.9.

RESEARCH_LOG 23.58: market breadth (share of names above their 120-day average)
predicts next-month *excess* return with t=-3.9 over 78 months, and a
bottom-tercile de-risk rule beat hold in-sample (+116% vs +196% total, Sharpe
1.34 vs 1.19). In-sample slope and an in-sample-chosen rule are exactly where
this repo has been fooled before, so the rule is re-tested three ways:

1. expanding-window terciles, no hindsight threshold (already causal);
2. the three independent walk-forward windows as holdout periods;
3. lag and placebo checks: shift breadth by 1/2/3 extra months, and use a
   random walk with the same autocorrelation as a null.

Usage::

    python scripts/run_breadth_oos.py
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
WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-05"),
}


def monthly(series: pd.Series) -> pd.Series:
    return (1 + series).groupby([series.index.year, series.index.month]).prod() - 1.0


def expanding_low_flag(values: pd.Series, min_history: int = 24) -> pd.Series:
    flags = []
    for i in range(len(values)):
        history = values.iloc[:i]
        if len(history) < min_history:
            flags.append(False)
        else:
            flags.append(bool(values.iloc[i] <= history.quantile(1 / 3)))
    return pd.Series(flags, index=values.index)


def summarise(monthly_returns: pd.Series) -> dict:
    total = float((1 + monthly_returns).prod() - 1)
    sharpe = float(monthly_returns.mean() / monthly_returns.std() * np.sqrt(12)) if monthly_returns.std() > 0 else float("nan")
    return {"total": total, "sharpe": sharpe, "months": int(len(monthly_returns))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "breadth_timing_oos"))
    parser.add_argument("--placebo", type=int, default=200)
    args = parser.parse_args(argv)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print(f"panel: {len(aligned)} symbols", flush=True)

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
    traded = close.notna() & (price_frame(aligned, "volume").reindex_like(close) > 0)
    sma120 = close.rolling(120, min_periods=60).mean()
    breadth = ((close > sma120) & traded).sum(axis=1) / traded.sum(axis=1).replace(0, np.nan)
    breadth_m = breadth.groupby([breadth.index.year, breadth.index.month]).last()

    strat_m = monthly(result.returns)
    bench_m = monthly(close.pct_change().mean(axis=1, skipna=True).dropna())
    frame = pd.concat(
        [strat_m.rename("strategy"), bench_m.rename("equal_weight"), breadth_m.rename("breadth")],
        axis=1,
    ).dropna()
    frame = frame.iloc[12:]
    flag = expanding_low_flag(frame["breadth"])
    position = pd.Series(np.where(flag, 0.25, 1.0), index=frame.index)

    timed = frame["strategy"] * position
    rows = []
    for window, (start, end) in WINDOWS.items():
        mask = (frame.index.get_level_values(0) >= int(start[:4]))
        if window == "2022-2023":
            mask = (frame.index.get_level_values(0) >= 2022) & (frame.index.get_level_values(0) <= 2023)
        elif window == "2019-2021":
            mask = (frame.index.get_level_values(0) <= 2021)
        elif window == "2024-2026":
            mask = (frame.index.get_level_values(0) >= 2024)
        hold = summarise(frame.loc[mask, "strategy"])
        rule = summarise(timed.loc[mask])
        bench = summarise(frame.loc[mask, "equal_weight"])
        rows.append(
            {
                "window": window,
                "hold_total": hold["total"],
                "rule_total": rule["total"],
                "hold_sharpe": hold["sharpe"],
                "rule_sharpe": rule["sharpe"],
                "bench_total": bench["total"],
                "months": hold["months"],
                "months_in_cash": int(position.loc[mask].lt(1).sum()),
            }
        )
    whole = summarise(frame["strategy"])
    whole_rule = summarise(timed)

    # Control: apply the identical rule to the equal-weight pool and to the
    # index. If breadth helps everything equally, it is a market-timing signal,
    # not an enhancement of this strategy.
    bench_timed = frame["equal_weight"] * position
    controls = {
        "strategy": {"hold": whole, "rule": whole_rule},
        "equal_weight": {"hold": summarise(frame["equal_weight"]), "rule": summarise(bench_timed)},
    }
    bench_raw = pd.read_parquet(Path(args.cache_dir) / "sh000300_raw.parquet")
    bench_raw.index = pd.to_datetime(bench_raw.index)
    index_ret = bench_raw["close"].pct_change().dropna()
    index_m = monthly(index_ret).reindex(frame.index).dropna()
    index_pos = position.reindex(index_m.index)
    controls["index"] = {
        "hold": summarise(index_m),
        "rule": summarise(index_m * index_pos),
    }

    # Lag placebo: shifting breadth only forward should kill a real signal.
    lag_rows = []
    for extra in (0, 1, 2, 3):
        shifted = frame["breadth"].shift(extra)
        shifted_flag = expanding_low_flag(shifted.dropna())
        shifted_pos = pd.Series(np.where(shifted_flag, 0.25, 1.0), index=shifted_flag.index)
        series = frame["strategy"].reindex(shifted_pos.index) * shifted_pos
        stats = summarise(series)
        lag_rows.append({"extra_lag_months": extra, "total": stats["total"], "sharpe": stats["sharpe"]})

    # Random placebo: bootstrap the same de-risk frequency from iid flags.
    rng = np.random.default_rng(20261107)
    frequency = float(position.lt(1).mean())
    random_sharpes = []
    for _ in range(args.placebo):
        random_pos = pd.Series(np.where(rng.random(len(frame)) < frequency, 0.25, 1.0), index=frame.index)
        series = frame["strategy"] * random_pos
        random_sharpes.append(float(series.mean() / series.std() * np.sqrt(12)))
    random_sharpes = np.array(random_sharpes)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "params": PARAMS,
        "months": int(len(frame)),
        "start": str(frame.index[0]),
        "end": str(frame.index[-1]),
        "de_risk_frequency": frequency,
        "hold": whole,
        "rule": whole_rule,
        "benchmark": summarise(frame["equal_weight"]),
        "windows": rows,
        "lag_placebo": lag_rows,
        "controls": controls,
        "random_placebo": {
            "draws": args.placebo,
            "sharpe_mean": float(random_sharpes.mean()),
            "sharpe_p95": float(np.quantile(random_sharpes, 0.95)),
            "share_ge_rule": float((random_sharpes >= whole_rule["sharpe"]).mean()),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "windows.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(lag_rows).to_csv(output / "lag_placebo.csv", index=False, encoding="utf-8-sig")

    lines = ["# 广度择时的样本外与安慰剂检验", ""]
    lines.append(f"{frame.index[0]} .. {frame.index[-1]}，{len(frame)} 个月；"
                 f"减仓频率 {frequency:.1%}。")
    lines.append("")
    lines.append("| 口径 | 持有 | 择时 | 等权 | 持有 Sharpe | 择时 Sharpe | 减仓月数 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {window} | {hold_total:+.1%} | {rule_total:+.1%} | {bench_total:+.1%} | "
            "{hold_sharpe:.2f} | {rule_sharpe:.2f} | {months_in_cash} |".format(**row)
        )
    lines.append("")
    lines.append(f"全样本：持有 {whole['total']:+.1%} / {whole['sharpe']:.2f}；"
                 f"择时 {whole_rule['total']:+.1%} / {whole_rule['sharpe']:.2f}。")
    lines.append("")
    lines.append("## 对照：同一条规则套在其他组合上")
    lines.append("")
    lines.append("| 组合 | 持有累计 | 规则累计 | 持有 Sharpe | 规则 Sharpe |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, control in controls.items():
        lines.append(
            "| {name} | {hold_total:+.1%} | {rule_total:+.1%} | {hold_sharpe:.2f} | {rule_sharpe:.2f} |".format(
                name=name,
                hold_total=control["hold"]["total"],
                rule_total=control["rule"]["total"],
                hold_sharpe=control["hold"]["sharpe"],
                rule_sharpe=control["rule"]["sharpe"],
            )
        )
    lines.append("")
    lines.append("## 滞后安慰剂")
    lines.append("")
    lines.append("| 额外滞后（月） | 累计 | Sharpe |")
    lines.append("|---:|---:|---:|")
    for row in lag_rows:
        lines.append("| {extra_lag_months} | {total:+.1%} | {sharpe:.2f} |".format(**row))
    lines.append("")
    lines.append("## 随机安慰剂")
    lines.append("")
    lines.append(f"{args.placebo} 次随机同频率减仓，Sharpe 均值 "
                 f"{random_sharpes.mean():.2f}，95% 分位 {np.quantile(random_sharpes, 0.95):.2f}；"
                 f"达到或超过规则 Sharpe 的比例 {float((random_sharpes >= whole_rule['sharpe']).mean()):.1%}。")
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print(pd.DataFrame(lag_rows).to_string(index=False))
    print(json.dumps(summary["random_placebo"], indent=2))
    try:
        shown = output.resolve().relative_to(ROOT)
    except ValueError:
        shown = output
    print(f"report: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
