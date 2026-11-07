"""Calendar-year record of the deployable configuration, decomposed.

RESEARCH_LOG 23.71 published a year table for the single-phase book at the 18%
target and concluded that all eight calendar years were positive. 23.83 replaced
the phase with the blend and 23.85 lowered the target to 12%, so that conclusion
no longer describes what would be deployed.

Two changes happened at once, and "some year went negative" is not useful unless
it is attributed. This rebuilds the year table for the 2x2 -- {blend, phase 0} x
{12%, 18%} -- from continuous backtests, so the sign flips can be pinned on one
of the two changes rather than both.

Usage::

    python scripts/run_year_table_blend.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402
from run_year_table import year_stats  # noqa: E402

PHASES = tuple(range(20))
TARGETS = (0.12, 0.18)
KINDS = ("phase0", "blend")


def phase_target(aligned, offset: int, target_vol: float) -> pd.DataFrame:
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
    return build_portfolio(
        strategy.generate(aligned),
        aligned,
        risk=RiskConfig(target_annual_vol=target_vol),
        use_regime=False,
        use_vol_target=True,
    )


def build(aligned, kind: str, target_vol: float) -> pd.DataFrame:
    frames = [phase_target(aligned, k, target_vol) for k in PHASES]
    if kind == "phase0":
        return frames[0]
    return sum(frames) / len(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "year_table_blend"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    rows = []
    overall = {}
    for target_vol in TARGETS:
        for kind in KINDS:
            label = "%s@%.0f%%" % (kind, target_vol * 100)
            print("running %s ..." % label, flush=True)
            weights = build(aligned, kind, target_vol)
            result = Backtester(BacktestConfig()).run(aligned, weights)
            returns = result.returns
            exposure = weights.sum(axis=1).reindex(returns.index)
            for year, block in returns.groupby(returns.index.year):
                stats = year_stats(block)
                stats["year"] = int(year)
                stats["config"] = label
                stats["kind"] = kind
                stats["target"] = target_vol
                stats["avg_exposure"] = float(exposure[exposure.index.year == year].mean())
                rows.append(stats)
            summary_stats = year_stats(returns)
            summary_stats["avg_exposure"] = float(exposure.mean())
            overall[label] = summary_stats

    data = pd.DataFrame(rows)
    data.to_csv(out / "year_stats.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(
            {"phases": list(PHASES), "targets": list(TARGETS), "overall": overall, "years": rows},
            ensure_ascii=False, indent=2, default=float,
        ),
        encoding="utf-8",
    )

    returns_table = data.pivot_table(index="year", columns="config", values="return")
    lines = ["# 发布配置的年度表：2×2 分解", ""]
    lines.append("单次连续回测，无窗口/年账本重启。四种组合：{单相位, 混合} × {12%, 18%}。")
    lines.append("")
    lines.append("| 年份 | " + " | ".join(returns_table.columns) + " |")
    lines.append("|---:|" + "---:|" * len(returns_table.columns))
    for year, row in returns_table.iterrows():
        cells = " | ".join(("%+.2f%%" % (row[c] * 100)) for c in returns_table.columns)
        lines.append("| %d | %s |" % (year, cells))
    negative = " | ".join(
        "%d" % int((returns_table[c] < 0).sum()) for c in returns_table.columns
    )
    lines.append("| 负年数 | %s |" % negative)
    lines.append("")
    lines.append("## 全期")
    lines.append("")
    lines.append("| 配置 | 收益 | 波动 | Sharpe | 最大回撤 | 平均仓位 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for label, stats in overall.items():
        lines.append(
            "| %s | %+.2f%% | %.2f%% | %.2f | %.2f%% | %.3f |"
            % (label, stats["return"] * 100, stats["volatility"] * 100, stats["sharpe"],
               stats["max_drawdown"] * 100, stats["avg_exposure"])
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nreturn by year:\n" + (returns_table * 100).round(2).to_string())
    print("\nnegative years:\n" + (returns_table < 0).sum().to_string())
    print("\noverall:")
    for label, stats in overall.items():
        print("  %-12s ret %+7.2f%%  sharpe %.3f  mdd %7.2f%%  exposure %.3f"
              % (label, stats["return"] * 100, stats["sharpe"], stats["max_drawdown"] * 100, stats["avg_exposure"]))
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
