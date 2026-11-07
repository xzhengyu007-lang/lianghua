"""Is the 12% volatility target still worth switching on, on the new baseline?

RESEARCH_LOG 23.47 and 23.63 judged the volatility target against the single-phase
book, whose published return was the lucky phase. 23.83 replaced that baseline
with the phase blend (+27.72% / Sharpe 0.780 for 2024-2026, down from the +43.68%
/ 1.08 that phase 0 showed). A risk overlay is a trade of return for drawdown, so
its verdict has to be re-priced whenever the return it gives up gets smaller.

Every target rebuilds the whole blend -- the overlay lives inside the portfolio
construction, so the weights differ -- and then runs the same paired bootstrap
the earlier sections used.

Usage::

    python scripts/run_voltarget_blend.py
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
from quant_system.strategies import build_strategy  # noqa: E402

WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-06"),
}
PHASES = tuple(range(20))
TARGETS = (0.10, 0.12, 0.15, 0.18)
REFERENCE = 0.18


def blend_for_target(aligned, target: float) -> pd.DataFrame:
    frames = []
    for offset in PHASES:
        strategy = build_strategy(
            "low_vol_core",
            top_n=8,
            vol_window=60,
            trend_window=120,
            max_per_sector=2,
            rebalance_days=20,
            rebalance_reset="year",
            rebalance_offset=offset,
        )
        frames.append(
            build_portfolio(
                strategy.generate(aligned),
                aligned,
                risk=RiskConfig(target_annual_vol=target),
                use_regime=False,
                use_vol_target=True,
            )
        )
    return sum(frames) / len(frames)


def summarise(result, window: str, target: float) -> dict:
    m = result.metrics
    return {
        "window": window,
        "target": target,
        "total_return": float(m.total_return),
        "cagr": float(m.cagr),
        "annual_volatility": float(m.annual_volatility),
        "sharpe": float(m.sharpe),
        "max_drawdown": float(m.max_drawdown),
        "calmar": float(m.calmar),
        "vol_ratio": float(m.annual_volatility) / target,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "voltarget_blend"))
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    sliced_by_target = {}
    rows = []
    for target in TARGETS:
        print("building blend for target=%.0f%% ..." % (target * 100), flush=True)
        blended = blend_for_target(aligned, target)
        result = Backtester(BacktestConfig()).run(aligned, blended)
        sliced_by_target[target] = {}
        for window, (start, end) in WINDOWS.items():
            sliced = slice_result(result, start=start, end=end)
            sliced_by_target[target][window] = sliced
            rows.append(summarise(sliced, window, target))

    data = pd.DataFrame(rows)
    data.to_csv(out / "panel.csv", index=False, encoding="utf-8-sig")

    paired = []
    for window in WINDOWS:
        base = sliced_by_target[REFERENCE][window].returns
        for target in TARGETS:
            if target == REFERENCE:
                continue
            diff = paired_bootstrap_diff(
                sliced_by_target[target][window].returns, base, n_boot=args.n_boot, seed=args.seed
            )
            row = {"window": window, "target": target, "reference": REFERENCE, **diff.to_dict()}
            reference_row = data[(data["window"] == window) & (data["target"] == REFERENCE)].iloc[0]
            row["d_mdd_pp"] = (
                data[(data["window"] == window) & (data["target"] == target)].iloc[0]["max_drawdown"]
                - reference_row["max_drawdown"]
            ) * 100
            row["d_calmar"] = (
                data[(data["window"] == window) & (data["target"] == target)].iloc[0]["calmar"]
                - reference_row["calmar"]
            )
            paired.append(row)
    pairs = pd.DataFrame(paired)
    pairs.to_csv(out / "paired.csv", index=False, encoding="utf-8-sig")

    (out / "summary.json").write_text(
        json.dumps(
            {"windows": {k: list(v) for k, v in WINDOWS.items()}, "targets": list(TARGETS),
             "reference": REFERENCE, "rows": rows, "paired": paired},
            ensure_ascii=False, indent=2, default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# 目标波动率在混合口径下是否还值得开", ""]
    lines.append('基准 = 混合（20 相位等权），三段窗口。参照档 18% 视为「基本不设限」。')
    lines.append("")
    lines.append("| 窗口 | 目标 | 累计收益 | 波动 | 实现/目标 | Sharpe | 最大回撤 | 卡玛 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in data.itertuples(index=False):
        lines.append(
            "| %s | %.0f%% | %+.2f%% | %.2f%% | %.2f | %.3f | %.2f%% | %.2f |"
            % (row.window, row.target * 100, row.total_return * 100, row.annual_volatility * 100,
               row.vol_ratio, row.sharpe, row.max_drawdown * 100, row.calmar)
        )
    lines.append("")
    lines.append("## 相对 18% 的配对 ΔSharpe")
    lines.append("")
    lines.append("| 窗口 | 目标 | ΔSharpe | 95% 区间 | P(Δ>0) | Δ回撤 (pp) | Δ卡玛 |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for row in pairs.itertuples(index=False):
        lines.append(
            "| %s | %.0f%% | %+.3f | [%+.3f, %+.3f] | %.2f | %+.2f | %+.2f |"
            % (row.window, row.target * 100, row.observed, row.low, row.high,
               row.prob_positive, row.d_mdd_pp, row.d_calmar)
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + data.pivot_table(index="window", columns="target", values="sharpe").round(3).to_string())
    print("\n" + pairs[["window", "target", "observed", "low", "high", "d_mdd_pp"]].to_string(index=False))
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
