"""Does "longer is better" ever stop? If it never does, that is a warning, not a finding.

RESEARCH_LOG 23.116/23.117 found the trend window improving monotonically from 60 to
250, and 23.117 made the point that a rule which keeps improving past every boundary
is more likely a property of this sample than of the strategy. This extends the
sweep to 300/400/500 under the deployable blend, with phase spans, so the shape of
the curve is visible instead of extrapolated.

Usage::

    python scripts/run_window_extent.py --windows 60 120 250 300 400 500
"""

from __future__ import annotations

import argparse
import importlib.util
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

WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-06"),
}
TARGET_VOL = 0.12
PHASES = 20
BASE = dict(top_n=8, vol_window=60, max_per_sector=2, rebalance_days=20,
            rebalance_reset="year")


def _filtered_class():
    spec = importlib.util.spec_from_file_location(
        "run_filter_form", ROOT / "scripts" / "run_filter_form.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Filtered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "window_extent"))
    parser.add_argument("--windows", nargs="+", type=int, default=[60, 120, 250, 300, 400, 500])
    parser.add_argument("--render-only", action="store_true",
                        help="rebuild report.md from the saved panel.csv")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        frame = pd.read_csv(out / "panel.csv")
        _write_report(frame, out)
        print(frame[["window", "full_sharpe", "phase_min", "turnover"]].round(3).to_string(index=False))
        return 0

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    Filtered = _filtered_class()

    rows = []
    for window in args.windows:
        frames = []
        for offset in range(PHASES):
            strategy = Filtered(filter_kind="sma", filter_window=window,
                                rebalance_offset=offset, trend_window=window, **BASE)
            frames.append(build_portfolio(
                strategy.generate(aligned), aligned,
                risk=RiskConfig(target_annual_vol=TARGET_VOL),
                use_regime=False, use_vol_target=True,
            ))
        blend = sum(frames) / len(frames)

        phase_sharpes = [Backtester(BacktestConfig()).run(aligned, f).metrics.sharpe for f in frames]
        blended = Backtester(BacktestConfig()).run(aligned, blend)
        held = blend.fillna(0.0) > 0
        first = held.index[held.any(axis=1)][0]

        row = {
            "window": window,
            "full_sharpe": float(blended.metrics.sharpe),
            "full_return": float(blended.metrics.total_return),
            "full_max_drawdown": float(blended.metrics.max_drawdown),
            "turnover": float(blended.metrics.turnover),
            "phase_min": float(min(phase_sharpes)),
            "phase_median": float(pd.Series(phase_sharpes).median()),
            "phase_max": float(max(phase_sharpes)),
            "first_position": str(first.date()),
        }
        for name, (start, end) in WINDOWS.items():
            row[name] = float(slice_result(blended, start=start, end=end).metrics.sharpe)
        rows.append(row)
        print("  SMA%d done: full %.3f, first position %s" % (window, row["full_sharpe"], first.date()),
              flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "panel.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps({"windows": args.windows, "rows": rows}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    _write_report(frame, out)
    print("\n" + frame[["window", "full_sharpe", "phase_min", "phase_median", "turnover", "first_position"]]
          .round(3).to_string(index=False))
    print("report: %s" % out)
    return 0


def _write_report(frame: pd.DataFrame, out: Path) -> None:
    lines = ["# 趋势窗口的外延：越长越好会停在哪里", ""]
    lines.append("部署口径（20 相位等权 + 12% 目标波动）。窗口越大，热身越长，2019 段越吃亏。")
    lines.append("")
    lines.append("| 窗口 | " + " | ".join(WINDOWS) + " | 全期 Sharpe | 相位 min–max | 换手 | 首次建仓 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---|")
    for record in frame.to_dict("records"):
        lines.append("| %d | %.3f | %.3f | %.3f | **%.3f** | %.3f–%.3f | %.1f | %s |"
                     % (record["window"], record["2019-2021"], record["2022-2023"],
                        record["2024-2026"], record["full_sharpe"],
                        record["phase_min"], record["phase_max"],
                        record["turnover"], record["first_position"]))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
