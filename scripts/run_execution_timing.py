"""How much of the edge survives being late?

The engine executes every rebalance at the next day's open. 23.100 pointed out that
this is an unverified assumption: if the open print is not actually available to
this book, the fills happen later and at worse prices. That is testable without
auction data, because it is a question about *timing*, not about spreads.

The target weights are built once and then shifted forward by 0, 1, 2 and 3
trading days. Same decisions, same basket, only the execution date moves -- so
whatever the difference is, it belongs to timing and nothing else.

Usage::

    python scripts/run_execution_timing.py
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
TARGET_VOL = 0.12
DELAYS = (0, 1, 2, 3)


def blended_weights(aligned) -> pd.DataFrame:
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
                risk=RiskConfig(target_annual_vol=TARGET_VOL),
                use_regime=False,
                use_vol_target=True,
            )
        )
    return sum(frames) / len(frames)


def summarise(result, window: str, delay: int) -> dict:
    m = result.metrics
    return {
        "window": window,
        "delay_days": delay,
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
    parser.add_argument("--output", default=str(ROOT / "reports" / "execution_timing"))
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    print("building the deployable blend...", flush=True)
    base_weights = blended_weights(aligned)

    rows = []
    sliced_by: dict[int, dict] = {}
    for delay in DELAYS:
        weights = base_weights.shift(delay) if delay else base_weights
        result = Backtester(BacktestConfig()).run(aligned, weights.fillna(0.0))
        sliced_by[delay] = {}
        for window, (start, end) in WINDOWS.items():
            sliced = slice_result(result, start=start, end=end)
            sliced_by[delay][window] = sliced
            rows.append(summarise(sliced, window, delay))
        print("  delay=%d done" % delay, flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "panel.csv", index=False, encoding="utf-8-sig")

    paired = []
    for window in WINDOWS:
        base = sliced_by[0][window].returns
        for delay in DELAYS:
            if delay == 0:
                continue
            diff = paired_bootstrap_diff(
                sliced_by[delay][window].returns, base, n_boot=args.n_boot, seed=args.seed
            )
            paired.append({"window": window, "delay_days": delay, **diff.to_dict()})
    pairs = pd.DataFrame(paired)
    pairs.to_csv(out / "paired.csv", index=False, encoding="utf-8-sig")

    (out / "summary.json").write_text(
        json.dumps({"windows": {k: list(v) for k, v in WINDOWS.items()},
                    "phases": len(PHASES), "target_vol": TARGET_VOL,
                    "delays": list(DELAYS), "rows": rows, "paired": paired},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    lines = ["# 执行时点：晚一天会怎样", ""]
    lines.append("同一批目标权重，整体向前平移 N 个交易日再执行。决策不变，只有成交日变。")
    lines.append("")
    lines.append("| 窗口 | 延迟 | 累计收益 | 波动 | Sharpe | 最大回撤 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in frame.itertuples(index=False):
        lines.append("| %s | %d 日 | %+.2f%% | %.2f%% | %.3f | %.2f%% |"
                     % (row.window, row.delay_days, row.total_return * 100,
                        row.annual_volatility * 100, row.sharpe, row.max_drawdown * 100))
    lines.append("")
    lines.append("## 相对不延迟的配对 ΔSharpe")
    lines.append("")
    lines.append("| 窗口 | 延迟 | ΔSharpe | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---:|---:|---|---:|")
    for row in pairs.itertuples(index=False):
        lines.append("| %s | %d 日 | %+.3f | [%+.3f, %+.3f] | %.2f |"
                     % (row.window, row.delay_days, row.observed, row.low, row.high, row.prob_positive))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + frame.pivot_table(index="window", columns="delay_days", values="sharpe").round(3).to_string())
    print("\n" + pairs[["window", "delay_days", "observed", "low", "high"]].round(3).to_string(index=False))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
