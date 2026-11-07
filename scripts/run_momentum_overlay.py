"""Does 12-1 momentum add anything on top of low volatility, in all 3 windows?

RESEARCH_LOG 23.41 concluded "no usable second signal": Amihud's IC does not
survive the portfolio layer, and 12-1 momentum diluted a 2022-2026 path
(+75% vs +95%). The three-window factor study re-opened that: 12-1 momentum's
*partial* IC (low-vol rank stripped out) is positive and significant in all
three windows (+0.064/+0.033/+0.041, t=6.7/4.5/4.5), while illiquidity's is not
stable. Factor IC is not portfolio return, so this script re-tests the
portfolio question window by window.

Members are the same engine and risk overlay as the published baseline; only
``momentum_weight`` changes. All runs are sliced from one continuous backtest
so every variant starts invested at the slice boundary.

Usage::

    python scripts/run_momentum_overlay.py
    python scripts/run_momentum_overlay.py --weights 0.1 0.2 0.3
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
from quant_system.strategies import LowVolPlusStrategy  # noqa: E402


WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-05"),
}

BASE = dict(
    top_n=8,
    vol_window=60,
    trend_window=120,
    illiquidity_window=20,
    vol_weight=0.8,
    trend_weight=0.2,
    illiquidity_weight=0.0,
    momentum_window=252,
    momentum_skip=21,
    blend="zscore",
)


def build(panel: dict[str, pd.DataFrame], momentum_weight: float):
    strategy = LowVolPlusStrategy(rebalance_reset="year", momentum_weight=momentum_weight, **BASE)
    weights = build_portfolio(
        strategy.generate(panel),
        panel,
        risk=RiskConfig(),
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
        "n_days": int(len(result.returns)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "momentum_overlay"))
    parser.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print(f"panel: {len(aligned)} symbols, {len(next(iter(aligned.values())))} bars", flush=True)

    runs = {"baseline": build(aligned, 0.0)}
    for weight in args.weights:
        label = f"momentum={weight:g}"
        print(f"running {label} ...", flush=True)
        runs[label] = build(aligned, weight)

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
                "base": BASE,
                "weights": args.weights,
                "rows": rows,
                "paired_sharpe_vs_baseline": paired,
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# 12-1 动量叠加：三窗口组合层复检", ""]
    lines.append("基准 = `low_vol_core`（0.8 低波 / 0.2 趋势，无动量项）。")
    lines.append("")
    lines.append("| 窗口 | 变体 | 累计收益 | CAGR | 波动 | Sharpe | 最大回撤 | 卡玛 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {window} | {label} | {total_return:.2%} | {cagr:.2%} | {annual_volatility:.2%} | "
            "{sharpe:.3f} | {max_drawdown:.2%} | {calmar:.2f} |".format(**row)
        )
    lines.append("")
    lines.append("## 配对 ΔSharpe（变体 − 基准）")
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
