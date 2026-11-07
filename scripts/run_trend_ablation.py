"""Trend-filter ablation around the low-vol core.

The 2019-2021 attribution showed a large month-end cash weight (19.2%) because
the stock-level 120-day trend gate empties the book when the whole PIT
cross-section is below its own trend line. This script asks whether that gate
is worth its cash drag. It is deliberately a *research* switch:

    baseline      published low_vol_core (strict above-trend eligibility)
    fallback      prefer above-trend names, fill vacant slots with calm names
                  below trend (a fixed rank penalty keeps the ordering)
    no_filter     all traded names are eligible; trend still gets 20% of score
    pure_lowvol   all traded names are eligible; score is 100% low volatility

All variants use the same aligned panel, risk overlay and execution engine. A
full-history backtest is run once per variant and then sliced into the three
independent windows, so each window inherits a warm book rather than starting
from cash at an arbitrary calendar boundary.

Usage::

    python scripts/run_trend_ablation.py
    python scripts/run_trend_ablation.py --windows 2024 --variants baseline fallback
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

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.bootstrap import paired_bootstrap_diff  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


WINDOWS = {
    "2019": ("2019-07-01", "2021-12-31"),
    "2022": ("2022-01-01", "2023-12-31"),
    "2024": ("2024-01-01", "2026-11-05"),
}

VARIANTS = {
    "baseline": {},
    "fallback": {"trend_fallback": True, "fallback_penalty": 5.0},
    "no_filter": {"trend_filter": False},
    "pure_lowvol": {"trend_filter": False, "trend_weight": 0.0},
}


def build_variant(panel: dict[str, pd.DataFrame], params: dict) -> tuple:
    strategy = LowVolCoreStrategy(rebalance_reset="year", **params)
    raw = strategy.generate(panel)
    weights = build_portfolio(
        raw,
        panel,
        risk=RiskConfig(),
        use_regime=False,
        use_vol_target=True,
    )
    result = Backtester(BacktestConfig()).run(panel, weights)
    return result, weights


def summarize(result, weights, label: str) -> dict:
    m = result.metrics
    active = weights.gt(0).sum(axis=1) if len(weights) else pd.Series(dtype=float)
    gross = weights.sum(axis=1) if len(weights) else pd.Series(dtype=float)
    return {
        "label": label,
        "total_return": float(m.total_return),
        "cagr": float(m.cagr),
        "annual_volatility": float(m.annual_volatility),
        "sharpe": float(m.sharpe),
        "max_drawdown": float(m.max_drawdown),
        "calmar": float(m.calmar),
        "exposure": float(m.exposure),
        "avg_names": float(active.mean()) if len(active) else float("nan"),
        "avg_gross": float(gross.mean()) if len(gross) else float("nan"),
        "n_days": int(len(result.returns)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "trend_ablation"))
    parser.add_argument("--windows", nargs="+", default=list(WINDOWS), choices=sorted(WINDOWS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=sorted(VARIANTS))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print(f"panel: {len(aligned)} symbols, {len(next(iter(aligned.values())))} bars", flush=True)

    runs = {}
    for name in args.variants:
        print(f"running {name} ...", flush=True)
        result, weights = build_variant(aligned, VARIANTS[name])
        runs[name] = (result, weights)

    rows: list[dict] = []
    pair_rows: list[dict] = []
    returns_by_window: dict[str, dict[str, pd.Series]] = {}
    for window in args.windows:
        start, end = WINDOWS[window]
        returns_by_window[window] = {}
        for name in args.variants:
            sliced = slice_result(runs[name][0], start=start, end=end)
            sliced_weights = runs[name][1].reindex(sliced.equity.index).fillna(0.0)
            row = summarize(sliced, sliced_weights, name)
            row["window"] = window
            rows.append(row)
            returns_by_window[window][name] = sliced.returns
        if "baseline" in returns_by_window[window]:
            for name in args.variants:
                if name == "baseline":
                    continue
                diff = paired_bootstrap_diff(
                    returns_by_window[window][name],
                    returns_by_window[window]["baseline"],
                    n_boot=args.n_boot,
                    seed=args.seed,
                )
                pair_rows.append({"window": window, "variant": name, **diff.to_dict()})

    frame = pd.DataFrame(rows)
    pairs = pd.DataFrame(pair_rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "windows.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output / "paired_sharpe.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "windows": {k: list(v) for k, v in WINDOWS.items()},
                "variants": VARIANTS,
                "rows": rows,
                "paired_sharpe_vs_baseline": pair_rows,
                "panel_symbols": len(aligned),
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    display = frame.copy()
    for column in ("total_return", "cagr", "annual_volatility", "max_drawdown",
                   "exposure", "avg_gross"):
        display[column] = (display[column] * 100).round(2)
    for column in ("sharpe", "calmar", "avg_names"):
        display[column] = display[column].astype(float).round(2)
    display = display[["window", "label", "total_return", "cagr", "annual_volatility",
                       "sharpe", "max_drawdown", "calmar", "exposure", "avg_names"]]
    print(display.to_string(index=False), flush=True)

    header = ["# Trend-filter ablation", "", "同一 PIT 面板、同一风险层与执行引擎；窗口为全历史回测的切片。", "",
              "| 窗口 | 方案 | 累计收益 | CAGR | 年化波动 | Sharpe | 最大回撤 | 卡玛 | 平均仓位 | 平均持仓数 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    lines = header + [
        "| {window} | {label} | {total_return:+.2f}% | {cagr:.2f}% | {annual_volatility:.2f}% | "
        "{sharpe:.3f} | {max_drawdown:.2f}% | {calmar:.2f} | {exposure:.1f}% | {avg_names:.2f} |".format(**row)
        for row in rows
    ]
    if len(pairs):
        lines += ["", "## 配对自助法：ΔSharpe（变体 − baseline）", "",
                  "| 窗口 | 变体 | ΔSharpe | 95% 区间 | P(Δ>0) |", "|---|---|---:|---|---:|"]
        for row in pair_rows:
            lines.append(
                f"| {row['window']} | {row['variant']} | {row['observed']:+.3f} | "
                f"[{row['low']:+.3f}, {row['high']:+.3f}] | {row['prob_positive']:.3f} |"
            )
    lines += ["", "## 口径", "",
              "- `baseline`：严格 120 日均线过滤，申报口径。",
              "- `fallback`：优先均线上方，均线下方仅用于补足 top_n，固定排名惩罚 5 个 z 单位。",
              "- `no_filter`：全池可选，趋势仍占评分 20%。",
              "- `pure_lowvol`：全池可选，评分 100% 低波。",
              "- 窗口切片继承全历史回测的持仓，不从现金重新开始；因此与独立 walk-forward 会有小幅路径差。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
