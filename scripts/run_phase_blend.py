"""Does holding all 20 phases at once still remove the calendar lottery?

RESEARCH_LOG 23.22 found that equal-weighting the 20 phase sub-portfolios landed
almost exactly on the phase median while cutting volatility below any single
phase. That was measured before the adjustment rebuild. 23.82 then showed the
published per-window numbers sit in the lucky tail of the phase distribution, so
the blend is no longer just a curiosity -- it is the only configuration whose
result does not depend on which day you started counting.

This rebuilds the blend on current data and answers three things per window:
does it land on the median, does it cut volatility, and what does it cost in
turnover.

Usage::

    python scripts/run_phase_blend.py
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


def phase_weights(aligned, offset: int) -> pd.DataFrame:
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
        "turnover": float(m.turnover),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "phase_blend"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    weights = {}
    for offset in PHASES:
        weights[offset] = phase_weights(aligned, offset)
        print("  phase %d weights built" % offset, flush=True)

    blended = sum(weights.values()) / len(weights)
    runs = {("phase %d" % k): v for k, v in weights.items()}
    runs["blended"] = blended

    rows = []
    for label, frame in runs.items():
        result = Backtester(BacktestConfig()).run(aligned, frame)
        for window, (start, end) in WINDOWS.items():
            rows.append(summarise(slice_result(result, start=start, end=end), label, window))

    data = pd.DataFrame(rows)
    data.to_csv(out / "blend_vs_phases.csv", index=False, encoding="utf-8-sig")

    singles = data[data["label"] != "blended"]
    blend = data[data["label"] == "blended"]

    comparison = []
    for window in WINDOWS:
        sub = singles[singles["window"] == window]
        row = blend[blend["window"] == window].iloc[0]
        percentile = float((sub["total_return"] < row["total_return"]).mean())
        comparison.append(
            {
                "window": window,
                "blend_return": float(row["total_return"]),
                "blend_sharpe": float(row["sharpe"]),
                "blend_vol": float(row["annual_volatility"]),
                "blend_mdd": float(row["max_drawdown"]),
                "blend_turnover": float(row["turnover"]),
                "phase_median_return": float(sub["total_return"].median()),
                "phase_beats_blend": percentile,
                "phase_vol_median": float(sub["annual_volatility"].median()),
                "phase_vol_min": float(sub["annual_volatility"].min()),
                "phase_turnover_median": float(sub["turnover"].median()),
            }
        )
    compare_frame = pd.DataFrame(comparison)
    compare_frame.to_csv(out / "comparison.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps({"rows": rows, "comparison": comparison}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    lines = ["# 相位平均复核（重建口径，reset=year）", ""]
    lines.append("同时持有 20 个相位的子组合，各 1/20 权重；对照是同一批单相位。")
    lines.append("")
    lines.append("| 窗口 | 混合收益 | 相位中位 | 混合 Sharpe | 比混合更好的相位占比 | 混合波动 | 相位波动中位 | 混合换手 | 相位换手中位 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in comparison:
        lines.append(
            "| %s | %+.2f%% | %+.2f%% | %.3f | %.0f%% | %.2f%% | %.2f%% | %.1f | %.1f |"
            % (
                row["window"],
                row["blend_return"] * 100,
                row["phase_median_return"] * 100,
                row["blend_sharpe"],
                100 * row["phase_beats_blend"],
                100 * row["blend_vol"],
                100 * row["phase_vol_median"],
                row["blend_turnover"],
                row["phase_turnover_median"],
            )
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + compare_frame.to_string(index=False))
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
