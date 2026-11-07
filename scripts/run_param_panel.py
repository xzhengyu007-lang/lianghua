"""Parameter panel: does the published setting survive all three windows?

Every default in this repo (``top_n=8``, ``vol_window=60``) was chosen on a
2022-2026 path, then re-checked once on 2019-2021. That leaves the usual
question: is 8 names / 60 days the good region, or just the point that happened
to win in 2022?

This script backtests a small grid once over the whole history and then slices
the same run into the three independent windows, so every variant sees the same
panel, the same execution engine and the same calendar. Slicing a continuous run
is not the same as a walk-forward refit (the book is already invested at the
slice boundary), but it is the right tool for "is the parameter region stable";
walk-forward numbers for the chosen default live in ``reports/wf_pit267_*``.

Usage::

    python scripts/run_param_panel.py
    python scripts/run_param_panel.py --axis top_n --values 6 8 10 12
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

DEFAULTS = {"top_n": 8, "vol_window": 60, "rebalance_days": 20, "max_per_sector": 2, "trend_window": 120}

AXES = {
    "top_n": [6, 8, 10, 12],
    "vol_window": [40, 60, 90],
    "rebalance_days": [10, 20, 40],
    "max_per_sector": [1, 2, 3],
    "trend_window": [100, 120, 140, 160, 180],
}


def build_variant(panel: dict[str, pd.DataFrame], params: dict):
    strategy = LowVolCoreStrategy(rebalance_reset="year", **params)
    raw = strategy.generate(panel)
    weights = build_portfolio(
        raw,
        panel,
        risk=RiskConfig(),
        use_regime=False,
        use_vol_target=True,
    )
    return Backtester(BacktestConfig()).run(panel, weights), weights


def summarize(result, weights, label: str, window: str) -> dict:
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
    parser.add_argument("--output", default=str(ROOT / "reports" / "param_panel"))
    parser.add_argument("--axis", default="top_n,vol_window",
                        help="comma separated axes from: " + ", ".join(AXES))
    parser.add_argument("--values", nargs="+", type=int, default=None,
                        help="override the value list, only valid with a single axis")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    axes = [a.strip() for a in args.axis.split(",") if a.strip()]
    for axis in axes:
        if axis not in AXES:
            raise SystemExit(f"unknown axis {axis!r}; choose from {sorted(AXES)}")
    if args.values is not None and len(axes) != 1:
        raise SystemExit("--values only works with a single axis")

    variants: list[tuple[str, dict]] = [("default(8/60)", dict(DEFAULTS))]
    seen: set[tuple] = {tuple(sorted(DEFAULTS.items()))}
    axis_values = (
        [(axes[0], args.values)] if len(axes) == 1 and args.values is not None
        else [(axis, AXES[axis]) for axis in axes]
    )
    for axis, values in axis_values:
        if len(axes) > 1 and args.values is not None:
            raise SystemExit("--values only works with a single axis")
        for value in values:
            if value == DEFAULTS.get(axis):
                continue
            params = {**DEFAULTS, axis: value}
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            variants.append((f"{axis}={value}", params))

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print(f"panel: {len(aligned)} symbols, {len(next(iter(aligned.values())))} bars", flush=True)

    runs: dict[str, tuple] = {}
    for label, params in variants:
        started = time.perf_counter()
        print(f"running {label} {params} ...", flush=True)
        runs[label] = build_variant(aligned, params)
        print(f"  done in {time.perf_counter() - started:.1f}s", flush=True)

    rows: list[dict] = []
    paired: list[dict] = []
    for window, (start, end) in WINDOWS.items():
        baseline_returns = None
        for label, _params in variants:
            result, weights = runs[label]
            sliced = slice_result(result, start=start, end=end)
            sliced_weights = weights.reindex(sliced.equity.index).fillna(0.0)
            rows.append(summarize(sliced, sliced_weights, label, window))
            if label == "default(8/60)":
                baseline_returns = sliced.returns
        for label, _params in variants:
            if label == "default(8/60)":
                continue
            sliced = slice_result(runs[label][0], start=start, end=end)
            diff = paired_bootstrap_diff(
                sliced.returns,
                baseline_returns,
                n_boot=args.n_boot,
                seed=args.seed,
            )
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
                "defaults": DEFAULTS,
                "variants": {label: params for label, params in variants},
                "rows": rows,
                "paired_sharpe_vs_default": paired,
                "panel_symbols": len(aligned),
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# Parameter stability across the three windows", ""]
    lines.append("| 窗口 | 变体 | 累计收益 | CAGR | 波动 | Sharpe | 最大回撤 | 卡玛 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {window} | {label} | {total_return:.2%} | {cagr:.2%} | {annual_volatility:.2%} | "
            "{sharpe:.3f} | {max_drawdown:.2%} | {calmar:.2f} |".format(**row)
        )
    lines.append("")
    lines.append("## 配对 ΔSharpe（变体 − 默认）")
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
    print(frame.to_string(index=False))
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
