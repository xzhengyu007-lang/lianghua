"""Parameter averaging: can we buy robustness without giving up return?

RESEARCH_LOG 23.49 showed a steep parameter surface: one grid step in ``top_n``
or ``max_per_sector`` moves a window from Sharpe 1.2 to 0.2. The classic remedy
is not to pick a better point but to stop picking: hold the *average* of the
neighbourhood, which trades a little peak performance for a lot of stability.

This script builds the weight matrices of several neighbourhood settings (all
else equal), averages them name by name, and runs the same engine on the
average. It is a portfolio of portfolios, so no extra look-ahead is introduced:
every component weight is already causal.

Usage::

    python scripts/run_param_ensemble.py
    python scripts/run_param_ensemble.py --members top_n:6 top_n:8 top_n:10 top_n:12

Each member is written ``axis:value`` (``axis:value`` maps to
``LowVolCoreStrategy(**{axis: value})`` on top of the published defaults).
"""

from __future__ import annotations

import argparse
import json
import re
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

DEFAULTS = {"top_n": 8, "vol_window": 60, "trend_window": 120, "max_per_sector": 2, "rebalance_days": 20}

DEFAULT_MEMBERS = ["top_n:6", "top_n:8", "top_n:10", "top_n:12"]


def parse_member(text: str) -> dict:
    match = re.fullmatch(r"([a-z_]+):(\d+)", text.strip())
    if not match:
        raise SystemExit(f"bad member {text!r}; expected axis:value like top_n:10")
    axis, value = match.group(1), int(match.group(2))
    if axis not in DEFAULTS:
        raise SystemExit(f"unknown axis {axis!r}; choose from {sorted(DEFAULTS)}")
    return {**DEFAULTS, axis: value}


def build_weights(panel: dict[str, pd.DataFrame], params: dict) -> pd.DataFrame:
    strategy = LowVolCoreStrategy(rebalance_reset="year", **params)
    raw = strategy.generate(panel)
    return build_portfolio(raw, panel, risk=RiskConfig(), use_regime=False, use_vol_target=True)


def summarise(result, label: str, window: str) -> dict:
    m = result.metrics
    return {
        "window": window,
        "label": label,
        "total_return": float(m.total_return),
        "cagr": float(m.cagr),
        "annual_volatility": float(m.annual_volatility),
        "sharpe": float(m.sharpe),
        "max_drawdown": float(m.max_drawdown),
        "calmar": float(m.calmar),
        "n_days": int(len(result.returns)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "param_ensemble"))
    parser.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    member_params = [(m, parse_member(m)) for m in args.members]
    if len(member_params) < 2:
        raise SystemExit("need at least two members to average")

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print(f"panel: {len(aligned)} symbols, {len(next(iter(aligned.values())))} bars", flush=True)

    weight_frames: dict[str, pd.DataFrame] = {}
    for label, params in member_params:
        started = time.perf_counter()
        print(f"weights for {label} {params} ...", flush=True)
        weight_frames[label] = build_weights(aligned, params)
        print(f"  {time.perf_counter() - started:.1f}s", flush=True)

    members = list(weight_frames)
    union_index = weight_frames[members[0]].index
    union_columns = sorted(set().union(*(set(frame.columns) for frame in weight_frames.values())))
    aligned_frames = [
        weight_frames[m].reindex(index=union_index, columns=union_columns).fillna(0.0)
        for m in members
    ]
    average = sum(aligned_frames) / float(len(aligned_frames))

    default_weights = build_weights(aligned, dict(DEFAULTS))
    runs = {
        **{m: Backtester(BacktestConfig()).run(aligned, weight_frames[m]) for m in members},
        "default": Backtester(BacktestConfig()).run(aligned, default_weights),
        "ensemble": Backtester(BacktestConfig()).run(aligned, average),
    }

    rows: list[dict] = []
    paired: list[dict] = []
    default_label = "default"
    for window, (start, end) in WINDOWS.items():
        sliced = {label: slice_result(res, start=start, end=end) for label, res in runs.items()}
        for label, res in sliced.items():
            rows.append(summarise(res, label, window))
        base = sliced[default_label].returns
        for label in ("ensemble",):
            diff = paired_bootstrap_diff(sliced[label].returns, base, n_boot=args.n_boot, seed=args.seed)
            paired.append({"window": window, "variant": label, "vs": default_label, **diff.to_dict()})

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
                "members": {label: params for label, params in member_params},
                "rows": rows,
                "paired_sharpe": paired,
                "panel_symbols": len(aligned),
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# 参数平均（邻域等权组合）", ""]
    lines.append("成员：" + "、".join(f"`{m}`" for m in members))
    lines.append("")
    lines.append("| 窗口 | 配置 | 累计收益 | CAGR | 波动 | Sharpe | 最大回撤 | 卡玛 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {window} | {label} | {total_return:.2%} | {cagr:.2%} | {annual_volatility:.2%} | "
            "{sharpe:.3f} | {max_drawdown:.2%} | {calmar:.2f} |".format(**row)
        )
    lines.append("")
    lines.append("## 配对 ΔSharpe（ensemble − 默认）")
    lines.append("")
    lines.append("| 窗口 | ΔSharpe | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---|---|---:|")
    for row in paired:
        lines.append(
            "| {window} | {observed:+.3f} | [{low:+.3f}, {high:+.3f}] | {prob_positive:.3f} |".format(**row)
        )
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    pivot = frame.pivot_table(index="label", columns="window", values="sharpe")
    print(pivot.round(3).to_string())
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
