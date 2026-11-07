"""How much of a 2-3 year result is window luck?

Every headline in this repo is a point estimate over one 2-3 year window, and
RESEARCH_LOG 23.44-23.51 keeps showing the sign flip when the window changes.
This script quantifies the lottery directly: it runs the published strategy once
over the whole research history, then slides a 12-month window across it in
monthly steps and reports the distribution of Sharpe, return and drawdown.

It also builds the equal-weight benchmark over the same panel, so "did this
12-month slice beat the naive basket" is answered on every slice rather than on
the three windows we happened to publish.

Usage::

    python scripts/run_window_scan.py
    python scripts/run_window_scan.py --months 12 --step 21
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
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.metrics import performance_metrics  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


PARAMS = {
    "top_n": 8,
    "vol_window": 60,
    "trend_window": 120,
    "max_per_sector": 2,
    "rebalance_days": 20,
}


def equal_weight_benchmark(aligned: dict[str, pd.DataFrame]) -> pd.Series:
    closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in aligned.items()})
    returns = closes.pct_change()
    return returns.mean(axis=1, skipna=True).dropna().rename("equal_weight")


def scan(returns: pd.Series, benchmark: pd.Series, days: int, step: int) -> pd.DataFrame:
    rows = []
    index = returns.index
    position = 0
    while position + days <= len(index):
        window = returns.iloc[position : position + days]
        bench = benchmark.reindex(window.index).dropna()
        m = performance_metrics(window)
        rows.append(
            {
                "start": window.index[0].date().isoformat(),
                "end": window.index[-1].date().isoformat(),
                "total_return": float(m.total_return),
                "sharpe": float(m.sharpe),
                "max_drawdown": float(m.max_drawdown),
                "annual_volatility": float(m.annual_volatility),
                "benchmark_return": float((1.0 + bench).prod() - 1.0) if len(bench) else float("nan"),
            }
        )
        position += step
    frame = pd.DataFrame(rows)
    frame["excess"] = frame["total_return"] - frame["benchmark_return"]
    return frame


def blocks(returns: pd.Series, benchmark: pd.Series, days: int) -> pd.DataFrame:
    rows = []
    position = 0
    while position + days <= len(returns):
        window = returns.iloc[position : position + days]
        bench = benchmark.reindex(window.index).dropna()
        m = performance_metrics(window)
        rows.append(
            {
                "start": window.index[0].date().isoformat(),
                "end": window.index[-1].date().isoformat(),
                "total_return": float(m.total_return),
                "sharpe": float(m.sharpe),
                "max_drawdown": float(m.max_drawdown),
                "benchmark_return": float((1.0 + bench).prod() - 1.0) if len(bench) else float("nan"),
            }
        )
        position += days
    frame = pd.DataFrame(rows)
    frame["excess"] = frame["total_return"] - frame["benchmark_return"]
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "window_scan"))
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--step", type=int, default=21)
    parser.add_argument("--block-months", type=int, default=24)
    args = parser.parse_args(argv)

    days = int(round(args.months * 21))
    block_days = int(round(args.block_months * 21))

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
    returns = result.returns
    benchmark = equal_weight_benchmark(aligned)

    rolling = scan(returns, benchmark, days, args.step)
    chunks = blocks(returns, benchmark, block_days)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rolling.to_csv(output / "rolling.csv", index=False, encoding="utf-8-sig")
    chunks.to_csv(output / "blocks.csv", index=False, encoding="utf-8-sig")

    def describe(frame: pd.DataFrame) -> dict:
        return {
            "n_windows": int(len(frame)),
            "sharpe_p05": float(frame["sharpe"].quantile(0.05)),
            "sharpe_p25": float(frame["sharpe"].quantile(0.25)),
            "sharpe_median": float(frame["sharpe"].median()),
            "sharpe_p75": float(frame["sharpe"].quantile(0.75)),
            "sharpe_p95": float(frame["sharpe"].quantile(0.95)),
            "sharpe_min": float(frame["sharpe"].min()),
            "sharpe_max": float(frame["sharpe"].max()),
            "return_median": float(frame["total_return"].median()),
            "return_p05": float(frame["total_return"].quantile(0.05)),
            "return_p95": float(frame["total_return"].quantile(0.95)),
            "mdd_median": float(frame["max_drawdown"].median()),
            "mdd_worst": float(frame["max_drawdown"].min()),
            "share_positive": float((frame["total_return"] > 0).mean()),
            "share_beat_benchmark": float((frame["excess"] > 0).mean()),
            "median_excess": float(frame["excess"].median()),
        }

    def conditional(frame: pd.DataFrame) -> dict:
        up = frame[frame["benchmark_return"] > 0]
        down = frame[frame["benchmark_return"] <= 0]
        return {
            "benchmark_up_windows": int(len(up)),
            "benchmark_up_beat_share": float((up["excess"] > 0).mean()) if len(up) else float("nan"),
            "benchmark_up_median_excess": float(up["excess"].median()) if len(up) else float("nan"),
            "benchmark_down_windows": int(len(down)),
            "benchmark_down_beat_share": float((down["excess"] > 0).mean()) if len(down) else float("nan"),
            "benchmark_down_median_excess": float(down["excess"].median()) if len(down) else float("nan"),
        }

    summary = {
        "rolling": describe(rolling),
        "blocks": describe(chunks),
        "rolling_conditional": conditional(rolling),
    }
    (output / "summary.json").write_text(
        json.dumps(
            {
                "params": PARAMS,
                "rolling_days": days,
                "step_days": args.step,
                "block_days": block_days,
                "history": [returns.index[0].date().isoformat(), returns.index[-1].date().isoformat()],
                **summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [f"# 窗口扫描：{args.months} 个月滚动窗口的运气", ""]
    lines.append(f"全历史 {returns.index[0].date()} .. {returns.index[-1].date()}，"
                 f"固定参数，连续回测一次后滚动切片（步长 {args.step} 个交易日）。")
    lines.append("")
    lines.append("## 滚动窗口分布")
    lines.append("")
    lines.append("| 指标 | P5 | P25 | 中位 | P75 | P95 | 最好 | 最差 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    roll = summary["rolling"]
    lines.append(
        f"| Sharpe | {roll['sharpe_p05']:.2f} | {roll['sharpe_p25']:.2f} | {roll['sharpe_median']:.2f} | "
        f"{roll['sharpe_p75']:.2f} | {roll['sharpe_p95']:.2f} | {roll['sharpe_max']:.2f} | {roll['sharpe_min']:.2f} |"
    )
    lines.append(
        f"| 累计收益 | {roll['return_p05']:.1%} | - | {roll['return_median']:.1%} | - | {roll['return_p95']:.1%} | - | - |"
    )
    lines.append(
        f"| 最大回撤 | - | - | {roll['mdd_median']:.1%} | - | - | - | {roll['mdd_worst']:.1%} |"
    )
    lines.append("")
    lines.append(f"- 窗口数 {roll['n_windows']}；收益为正占 {roll['share_positive']:.0%}，"
                 f"跑赢等权全池占 {roll['share_beat_benchmark']:.0%}，中位超额 {roll['median_excess']:+.1%}。")
    cond = summary["rolling_conditional"]
    lines.append("")
    lines.append("按同期等权基准涨跌分组：")
    lines.append("")
    lines.append("| 基准状态 | 窗口数 | 跑赢比例 | 中位超额 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| 上涨 | {cond['benchmark_up_windows']} | {cond['benchmark_up_beat_share']:.0%} | "
                 f"{cond['benchmark_up_median_excess']:+.1%} |")
    lines.append(f"| 下跌/走平 | {cond['benchmark_down_windows']} | {cond['benchmark_down_beat_share']:.0%} | "
                 f"{cond['benchmark_down_median_excess']:+.1%} |")
    lines.append("")
    lines.append(f"## 不重叠 {args.block_months} 个月分块")
    lines.append("")
    lines.append("| 区间 | 累计收益 | Sharpe | 最大回撤 | 等权基准 | 超额 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for _i, row in chunks.iterrows():
        lines.append(
            "| {start}..{end} | {total_return:.1%} | {sharpe:.2f} | {max_drawdown:.1%} | "
            "{benchmark_return:.1%} | {excess:+.1%} |".format(**row)
        )
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(chunks.to_string(index=False))
    try:
        shown = output.resolve().relative_to(ROOT)
    except ValueError:
        shown = output
    print(f"report: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
