"""Capacity under the configuration we would actually deploy: the phase blend.

RESEARCH_LOG 23.72 measured how much money the book can carry, but it did so with
a single phase. 23.83 then showed the blend is the only configuration whose
result does not depend on the calendar, and that it trades roughly 10% more. More
turnover against the same square-root impact means capacity has to be re-measured
rather than inherited: the number that matters is the blend's, not phase 0's.

Both configurations are rebuilt here through the same code path so the comparison
is internally consistent -- same cash levels, same participation cap, same cost
model, only the target weights differ.

Usage::

    python scripts/run_capacity_blend.py
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

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, CostConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402
from run_capacity_curve import CASH_LEVELS, WINDOW, weight_drift  # noqa: E402

PHASES = tuple(range(20))
IMPACTS = (0.0, 40.0)


def phase_target(aligned, offset: int) -> pd.DataFrame:
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
        risk=RiskConfig(),
        use_regime=False,
        use_vol_target=True,
    )


def run_level(aligned, target, cash: float, impact: float, participation: float) -> dict:
    config = BacktestConfig(
        initial_cash=cash,
        max_adv_participation=participation,
        costs=CostConfig(impact_coefficient_bps=impact),
    )
    result = Backtester(config).run(aligned, target)
    sliced = slice_result(result, start=WINDOW[0], end=WINDOW[1])
    realized = sliced.weights.reindex(sliced.equity.index).fillna(0.0)
    wanted = target.reindex(realized.index).fillna(0.0)
    drift = weight_drift(realized, wanted)
    m = sliced.metrics
    return {
        "cash": cash,
        "impact_bps": impact,
        "cagr": float(m.cagr),
        "sharpe": float(m.sharpe),
        "max_drawdown": float(m.max_drawdown),
        "annual_volatility": float(m.annual_volatility),
        "turnover": float(m.turnover),
        "avg_gross": float(realized.sum(axis=1).mean()),
        "weight_drift": float(drift.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "capacity_blend"))
    parser.add_argument("--participation", type=float, default=0.20)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    print("building 20 phase targets...", flush=True)
    weights = [phase_target(aligned, k) for k in PHASES]
    blended = sum(weights) / len(weights)
    single = weights[0]

    rows = []
    for config_name, target in (("phase0", single), ("blend", blended)):
        for impact in IMPACTS:
            for cash in CASH_LEVELS:
                row = run_level(aligned, target, cash, impact, args.participation)
                row["config"] = config_name
                rows.append(row)
                print("  %-6s impact=%2.0f cash=%15.0f sharpe=%.3f" % (
                    config_name, impact, cash, row["sharpe"]), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "capacity.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(
            {"window": list(WINDOW), "participation": args.participation, "rows": rows},
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# 容量曲线：混合口径", ""]
    lines.append("窗口 %s .. %s，ADV 参与上限 %.0f%%。同一代码路径下重建两种配置。" % (
        WINDOW[0], WINDOW[1], args.participation * 100))
    lines.append("")
    for impact in IMPACTS:
        sub = frame[frame["impact_bps"] == impact]
        lines.append("## 冲击系数 %.0fbp" % impact)
        lines.append("")
        lines.append("| 资金 | phase0 Sharpe | blend Sharpe | 差 | phase0 CAGR | blend CAGR | blend 换手 | blend 漂移 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cash in CASH_LEVELS:
            a = sub[(sub["cash"] == cash) & (sub["config"] == "phase0")].iloc[0]
            b = sub[(sub["cash"] == cash) & (sub["config"] == "blend")].iloc[0]
            lines.append(
                "| %.0f | %.3f | %.3f | %+.3f | %.2f%% | %.2f%% | %.1f | %.3f |"
                % (cash, a["sharpe"], b["sharpe"], b["sharpe"] - a["sharpe"],
                   a["cagr"] * 100, b["cagr"] * 100, b["turnover"], b["weight_drift"])
            )
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    pivot = frame[frame["impact_bps"] == 40.0].pivot(index="cash", columns="config", values="sharpe")
    print("\n40bp Sharpe by cash:\n" + pivot.round(3).to_string())
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
