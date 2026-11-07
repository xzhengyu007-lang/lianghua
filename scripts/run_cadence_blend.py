"""Rebalance cadence, measured on the blend instead of a single phase.

RESEARCH_LOG 23.74 found the median holding period is one cadence (20 trading
days) while 23.66 found the score's IC keeps rising out to 250 days. Holding much
shorter than the signal's validity wastes turnover. 23.49 tested 40 days and got
a split verdict, but it tested it on a single phase -- and 23.82 showed a single
phase carries 32-47pp of calendar noise, which is the same order as the effect
being measured.

The blend removes that noise, so the cadence question can finally be asked
cleanly: does holding longer raise the Sharpe, and what does it save in turnover?
This matters because 23.89 established that the binding constraint on this system
is evidence, not return -- and fewer, better-spaced bets is one of the only
levers that changes signal-to-noise without new data.

Usage::

    python scripts/run_cadence_blend.py
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
CADENCES = (20, 40, 60, 90, 120)
REFERENCE = 20
TARGET_VOL = 0.12


def blend_for(aligned, cadence: int) -> pd.DataFrame:
    frames = []
    for offset in PHASES:
        strategy = build_strategy(
            "low_vol_core",
            top_n=8,
            vol_window=60,
            trend_window=120,
            max_per_sector=2,
            rebalance_days=cadence,
            rebalance_reset="year",
            rebalance_offset=offset,
        )
        frames.append(
            build_portfolio(
                strategy.generate(aligned),
                aligned,
                risk=RiskConfig(target_annual_vol=TARGET_VOL),
                use_regime=False,
                use_vol_target=True,
            )
        )
    return sum(frames) / len(frames)


def median_holding(weights: pd.DataFrame) -> float:
    """Median consecutive-day spell a name stays in the book."""
    held = (weights.fillna(0.0) > 0).to_numpy()
    spells = []
    for column in range(held.shape[1]):
        run = 0
        for flag in held[:, column]:
            if flag:
                run += 1
            elif run:
                spells.append(run)
                run = 0
        if run:
            spells.append(run)
    return float(pd.Series(spells).median()) if spells else float("nan")


def summarise(result, window: str, cadence: int, holding: float) -> dict:
    m = result.metrics
    return {
        "window": window,
        "cadence": cadence,
        "median_holding_days": holding,
        "total_return": float(m.total_return),
        "cagr": float(m.cagr),
        "annual_volatility": float(m.annual_volatility),
        "sharpe": float(m.sharpe),
        "max_drawdown": float(m.max_drawdown),
        "calmar": float(m.calmar),
        "turnover": float(m.turnover),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "cadence_blend"))
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    sliced_by: dict[int, dict] = {}
    rows = []
    for cadence in CADENCES:
        print("building blend at cadence=%d ..." % cadence, flush=True)
        weights = blend_for(aligned, cadence)
        holding = median_holding(weights)
        result = Backtester(BacktestConfig()).run(aligned, weights)
        sliced_by[cadence] = {}
        for window, (start, end) in WINDOWS.items():
            sliced = slice_result(result, start=start, end=end)
            sliced_by[cadence][window] = sliced
            rows.append(summarise(sliced, window, cadence, holding))

    data = pd.DataFrame(rows)
    data.to_csv(out / "panel.csv", index=False, encoding="utf-8-sig")

    paired = []
    for window in WINDOWS:
        base = sliced_by[REFERENCE][window].returns
        for cadence in CADENCES:
            if cadence == REFERENCE:
                continue
            diff = paired_bootstrap_diff(
                sliced_by[cadence][window].returns, base, n_boot=args.n_boot, seed=args.seed
            )
            paired.append({"window": window, "cadence": cadence, "reference": REFERENCE, **diff.to_dict()})
    pairs = pd.DataFrame(paired)
    pairs.to_csv(out / "paired.csv", index=False, encoding="utf-8-sig")

    (out / "summary.json").write_text(
        json.dumps(
            {"windows": {k: list(v) for k, v in WINDOWS.items()}, "cadences": list(CADENCES),
             "phases": len(PHASES), "target_vol": TARGET_VOL, "rows": rows, "paired": paired},
            ensure_ascii=False, indent=2, default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# 调仓周期在混合口径下的复核", ""]
    lines.append("部署口径（混合 %d 相位 + %.0f%% 目标波动），只改调仓间隔。" % (len(PHASES), TARGET_VOL * 100))
    lines.append("")
    lines.append("| 窗口 | 间隔 | 中位持有 | 累计收益 | 波动 | Sharpe | 最大回撤 | 换手 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in data.itertuples(index=False):
        lines.append(
            "| %s | %d 日 | %.0f | %+.2f%% | %.2f%% | %.3f | %.2f%% | %.1f |"
            % (row.window, row.cadence, row.median_holding_days, row.total_return * 100,
               row.annual_volatility * 100, row.sharpe, row.max_drawdown * 100, row.turnover)
        )
    lines.append("")
    lines.append("## 相对 %d 日的配对 ΔSharpe" % REFERENCE)
    lines.append("")
    lines.append("| 窗口 | 间隔 | ΔSharpe | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---:|---:|---|---:|")
    for row in pairs.itertuples(index=False):
        lines.append(
            "| %s | %d 日 | %+.3f | [%+.3f, %+.3f] | %.2f |"
            % (row.window, row.cadence, row.observed, row.low, row.high, row.prob_positive)
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    pivot = data.pivot_table(index="window", columns="cadence", values="sharpe")
    turns = data.pivot_table(index="window", columns="cadence", values="turnover")
    print("\nSharpe:\n" + pivot.round(3).to_string())
    print("\nturnover:\n" + turns.round(1).to_string())
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
