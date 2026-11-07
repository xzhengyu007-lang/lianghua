"""Does trend_window=250 survive the gates a re-release has to pass?

RESEARCH_LOG 23.116 found SMA250 beating the published SMA120 in all three windows
after phase blending (+0.51 / +0.19 / +0.34). That is one gate. A re-release has to
pass the rest, and this repository already knows which ones they are:

  * phase span -- 23.82 found the deployable config's 20 phases span 0.73-0.99
    Sharpe; a candidate whose advantage is smaller than its own span is not a
    finding, it is a draw.
  * year by year -- a config that wins on the total but loses most individual
    years is a different product.
  * turnover and cost -- 23.91 / 23.100: a longer window should trade less, which
    matters more than Sharpe once costs bind.

Capacity is checked separately by run_capacity_blend and is not repeated here.

Usage::

    python scripts/run_window_gate.py
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


def phase_sets(aligned, Filtered, window: int):
    """One weight frame per phase, plus the equal-weighted blend."""
    frames = []
    for offset in range(PHASES):
        strategy = Filtered(filter_kind="sma", filter_window=window,
                            rebalance_offset=offset, trend_window=window, **BASE)
        frames.append(build_portfolio(
            strategy.generate(aligned), aligned,
            risk=RiskConfig(target_annual_vol=TARGET_VOL),
            use_regime=False, use_vol_target=True,
        ))
    return frames, sum(frames) / len(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "window_gate"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    Filtered = _filtered_class()

    results = {}
    for window in (120, 250):
        label = "SMA%d" % window
        print("  building %s (20 phases)..." % label, flush=True)
        frames, blend = phase_sets(aligned, Filtered, window)

        phase_sharpes = []
        for frame in frames:
            phase_sharpes.append(Backtester(BacktestConfig()).run(aligned, frame).metrics.sharpe)

        blended = Backtester(BacktestConfig()).run(aligned, blend)
        entry = {
            "full_return": float(blended.metrics.total_return),
            "full_sharpe": float(blended.metrics.sharpe),
            "full_max_drawdown": float(blended.metrics.max_drawdown),
            "full_turnover": float(blended.metrics.turnover),
            "by_window": {},
            "by_year": {},
            "phase_span": {
                "min": float(min(phase_sharpes)),
                "median": float(pd.Series(phase_sharpes).median()),
                "max": float(max(phase_sharpes)),
            },
        }
        for name, (start, end) in WINDOWS.items():
            entry["by_window"][name] = float(
                slice_result(blended, start=start, end=end).metrics.sharpe
            )
        for year in range(2019, 2027):
            sliced = slice_result(blended, start="%d-01-01" % year, end="%d-12-31" % year)
            entry["by_year"][str(year)] = {
                "return": float(sliced.metrics.total_return),
                "sharpe": float(sliced.metrics.sharpe),
            }
        results[label] = entry
        print("    blend sharpe %.3f, turnover %.1f" % (entry["full_sharpe"], entry["full_turnover"]),
              flush=True)

    (out / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    a, b = results["SMA120"], results["SMA250"]
    lines = ["# trend_window=250 的重发布关卡", ""]
    lines.append("部署口径（20 相位等权 + 12% 目标波动）下的两支配置对比。")
    lines.append("")
    lines.append("| 指标 | SMA120（发布） | SMA250 | 差 |")
    lines.append("|---|---:|---:|---:|")
    lines.append("| 全期累计收益 | %+.2f%% | %+.2f%% | %+.2f pp |"
                 % (a["full_return"] * 100, b["full_return"] * 100,
                    (b["full_return"] - a["full_return"]) * 100))
    lines.append("| 全期 Sharpe | %.3f | %.3f | %+.3f |"
                 % (a["full_sharpe"], b["full_sharpe"], b["full_sharpe"] - a["full_sharpe"]))
    lines.append("| 全期最大回撤 | %.2f%% | %.2f%% | %+.2f pp |"
                 % (a["full_max_drawdown"] * 100, b["full_max_drawdown"] * 100,
                    (b["full_max_drawdown"] - a["full_max_drawdown"]) * 100))
    lines.append("| 年换手 | %.1f | %.1f | %+.1f |"
                 % (a["full_turnover"], b["full_turnover"], b["full_turnover"] - a["full_turnover"]))
    lines.append("| 相位跨度 min–max | %.3f–%.3f | %.3f–%.3f | — |"
                 % (a["phase_span"]["min"], a["phase_span"]["max"],
                    b["phase_span"]["min"], b["phase_span"]["max"]))
    lines.append("")
    lines.append("## 逐年（收益 / Sharpe）")
    lines.append("")
    lines.append("| 年 | SMA120 | SMA250 |")
    lines.append("|---|---|---|")
    for year in sorted(a["by_year"]):
        ra, rb = a["by_year"][year], b["by_year"][year]
        lines.append("| %s | %+.1f%% / %.2f | %+.1f%% / %.2f |"
                     % (year, ra["return"] * 100, ra["sharpe"], rb["return"] * 100, rb["sharpe"]))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nfull: SMA120 %.3f vs SMA250 %.3f | turnover %.1f vs %.1f"
          % (a["full_sharpe"], b["full_sharpe"], a["full_turnover"], b["full_turnover"]))
    print("phase span: SMA120 %.3f-%.3f | SMA250 %.3f-%.3f"
          % (a["phase_span"]["min"], a["phase_span"]["max"],
             b["phase_span"]["min"], b["phase_span"]["max"]))
    wins = sum(1 for y in a["by_year"] if b["by_year"][y]["sharpe"] > a["by_year"][y]["sharpe"])
    print("years won by SMA250: %d of %d" % (wins, len(a["by_year"])))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
