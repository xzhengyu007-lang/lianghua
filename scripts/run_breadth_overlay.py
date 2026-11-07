"""Backtest the breadth-timing overlay through the real engine.

RESEARCH_LOG 23.58 measured the rule on monthly arithmetic returns. That ignored
the actual execution: rebalancing 8 names, lot rounding, ADV caps, and the
20-day signal cadence. This script runs the rule through ``build_portfolio`` +
``Backtester`` so the published number and the production path agree.

The implementation is causal by construction: the breadth reading at a month-end
close shifts forward one month and forward-fills onto the daily calendar, so the
scale used on day t was formed at least one month earlier.

Usage::

    python scripts/run_breadth_overlay.py
    python scripts/run_breadth_overlay.py --de-risk 0.0 0.25 0.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-05"),
}

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def build(panel, de_risk: float, use_overlay: bool):
    strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
    risk = RiskConfig(use_breadth_timing=use_overlay, breadth_de_risk=de_risk)
    weights = build_portfolio(
        strategy.generate(panel),
        panel,
        risk=risk,
        use_regime=False,
        use_vol_target=True,
    )
    return Backtester(BacktestConfig()).run(panel, weights), weights


def summarise(result, weights, label: str, window: str) -> dict:
    m = result.metrics
    gross = weights.sum(axis=1) if len(weights) else pd.Series(dtype=float)
    return {
        "window": window,
        "label": label,
        "total_return": float(m.total_return),
        "cagr": float(m.cagr),
        "annual_volatility": float(m.annual_volatility),
        "sharpe": float(m.sharpe),
        "max_drawdown": float(m.max_drawdown),
        "calmar": float(m.calmar),
        "avg_gross": float(gross.mean()) if len(gross) else float("nan"),
        "turnover": float(m.turnover),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "breadth_overlay"))
    parser.add_argument("--de-risk", nargs="+", type=float, default=[0.0, 0.25, 0.5])
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print(f"panel: {len(aligned)} symbols", flush=True)

    runs = {"baseline": build(aligned, 1.0, False)}
    for value in args.de_risk:
        if value >= 1.0:
            continue
        label = f"de_risk={value:g}"
        print(f"running {label} ...", flush=True)
        runs[label] = build(aligned, value, True)

    rows: list[dict] = []
    paired: list[dict] = []
    for window, (start, end) in WINDOWS.items():
        sliced = {label: slice_result(res, start=start, end=end) for label, (res, _w) in runs.items()}
        for label, res in sliced.items():
            weights = runs[label][1].reindex(res.equity.index).fillna(0.0)
            rows.append(summarise(res, weights, label, window))
        base = sliced["baseline"].returns
        for label in runs:
            if label == "baseline":
                continue
            diff = paired_bootstrap_diff(sliced[label].returns, base, n_boot=args.n_boot, seed=args.seed)
            paired.append({"window": window, "variant": label, **diff.to_dict()})

    frame = pd.DataFrame(rows)
    pairs = pd.DataFrame(paired)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output / "paired_sharpe.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "windows": {k: list(v) for k, v in WINDOWS.items()},
                "params": PARAMS,
                "de_risk": args.de_risk,
                "rows": rows,
                "paired_sharpe_vs_baseline": paired,
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# 广度择时叠加：真实撮合口径", ""]
    lines.append("规则在 `quant_system/risk.py:breadth_timing_scale`，默认关闭；"
                 "这里打开并走完整 `Backtester`。")
    lines.append("")
    lines.append("| 窗口 | 配置 | 累计收益 | CAGR | 波动 | Sharpe | 最大回撤 | 卡玛 | 平均仓位 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {window} | {label} | {total_return:.2%} | {cagr:.2%} | {annual_volatility:.2%} | "
            "{sharpe:.3f} | {max_drawdown:.2%} | {calmar:.2f} | {avg_gross:.2%} |".format(**row)
        )
    lines.append("")
    lines.append("## 配对 ΔSharpe（叠加 − 基线）")
    lines.append("")
    lines.append("| 窗口 | 变体 | ΔSharpe | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---|---:|---|---:|")
    for row in paired:
        lines.append(
            "| {window} | {variant} | {observed:+.3f} | [{low:+.3f}, {high:+.3f}] | "
            "{prob_positive:.3f} |".format(**row)
        )
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print(frame.pivot_table(index="label", columns="window", values="sharpe").round(3).to_string())
    print()
    print(frame.pivot_table(index="label", columns="window", values="total_return").mul(100).round(1).to_string())
    print()
    print(pairs.to_string(index=False))
    try:
        shown = output.resolve().relative_to(ROOT)
    except ValueError:
        shown = output
    print(f"report: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
