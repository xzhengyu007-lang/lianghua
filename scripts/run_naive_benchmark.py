"""Head to head against the obvious alternative, not against randomness.

RESEARCH_LOG 23.102 showed the ranking beats random selection by a wide margin. But
random selection is not what anyone would replace this with. The natural
alternative is the thing its name says: the lowest-volatility names among today's
large caps, equal weighted, same cadence, same risk budget -- no PIT pool, no trend
filter, no sector cap, no score blend.

Three variants separate the pieces:

  A  naive      -- 128 currently listed names, -vol only, no filter, no cap
  B  naive+PIT  -- A but on the full PIT pool (adds the delisted cohort)
  C  published  -- PIT pool, trend filter, sector cap, 0.8/0.2 blend

A -> B isolates the pool; B -> C isolates the machinery.

Usage::

    python scripts/run_naive_benchmark.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.bootstrap import paired_bootstrap_diff  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe, price_frame  # noqa: E402
from quant_system.indicators import realised_volatility  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy, _traded  # noqa: E402

WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-06"),
}
TARGET_VOL = 0.12


@dataclass
class NaiveLowVol(LowVolCoreStrategy):
    """Lowest volatility, full stop: no trend filter, no blend."""

    def score(self, panel):
        close = price_frame(panel, "close")
        traded = _traded(panel, close)
        vol = realised_volatility(panel, self.vol_window).reindex_like(close)
        return -vol, traded


def run(aligned, strategy) -> dict:
    weights = build_portfolio(
        strategy.generate(aligned),
        aligned,
        risk=RiskConfig(target_annual_vol=TARGET_VOL),
        use_regime=False,
        use_vol_target=True,
    )
    result = Backtester(BacktestConfig()).run(aligned, weights)
    return {"result": result, "weights": weights}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "naive_benchmark"))
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    listed = {s: f for s, f in aligned.items() if s[-6:] in set(BROAD_UNIVERSE)}
    print("panel %d symbols, listed subset %d" % (len(aligned), len(listed)), flush=True)

    base = dict(top_n=8, vol_window=60, trend_window=120, rebalance_days=20, rebalance_reset="year")
    variants = {
        "A 朴素（128 在册，无过滤无上限）": (listed, NaiveLowVol(max_per_sector=None, **base)),
        "B 朴素+PIT 池": (aligned, NaiveLowVol(max_per_sector=None, **base)),
        "C 发布策略（PIT+过滤+上限+混合打分）": (aligned, LowVolCoreStrategy(max_per_sector=2, **base)),
    }

    rows = []
    sliced_by: dict[str, dict] = {}
    for label, (sub, strategy) in variants.items():
        info = run(sub, strategy)
        sliced_by[label] = {}
        for window, (start, end) in WINDOWS.items():
            sliced = slice_result(info["result"], start=start, end=end)
            sliced_by[label][window] = sliced
            m = sliced.metrics
            rows.append({
                "variant": label,
                "window": window,
                "total_return": float(m.total_return),
                "sharpe": float(m.sharpe),
                "max_drawdown": float(m.max_drawdown),
                "turnover": float(m.turnover),
            })
        print("  %s done" % label, flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "panel.csv", index=False, encoding="utf-8-sig")

    reference = "C 发布策略（PIT+过滤+上限+混合打分）"
    paired = []
    for window in WINDOWS:
        base_returns = sliced_by[reference][window].returns
        for label in variants:
            if label == reference:
                continue
            diff = paired_bootstrap_diff(
                sliced_by[label][window].returns, base_returns, n_boot=args.n_boot, seed=args.seed
            )
            paired.append({"window": window, "variant": label, "vs": reference, **diff.to_dict()})
    pairs = pd.DataFrame(paired)
    pairs.to_csv(out / "paired.csv", index=False, encoding="utf-8-sig")

    (out / "summary.json").write_text(
        json.dumps({"target_vol": TARGET_VOL, "windows": {k: list(v) for k, v in WINDOWS.items()},
                    "rows": rows, "paired": paired}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    lines = ["# 与「最自然的大盘低波替代品」对打", ""]
    lines.append("同权重、同 20 日调仓、同年度重置、同 12% 目标波动、同成本，只改选股规则。")
    lines.append("")
    lines.append("| 变体 | 窗口 | 累计收益 | Sharpe | 最大回撤 | 换手 |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for row in frame.itertuples(index=False):
        lines.append("| %s | %s | %+.2f%% | %.3f | %.2f%% | %.1f |"
                     % (row.variant, row.window, row.total_return * 100, row.sharpe,
                        row.max_drawdown * 100, row.turnover))
    lines.append("")
    lines.append("## 配对 ΔSharpe（变体 − 发布策略）")
    lines.append("")
    lines.append("| 窗口 | 变体 | ΔSharpe | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---|---:|---|---:|")
    for row in pairs.itertuples(index=False):
        lines.append("| %s | %s | %+.3f | [%+.3f, %+.3f] | %.2f |"
                     % (row.window, row.variant, row.observed, row.low, row.high, row.prob_positive))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + frame.pivot_table(index="window", columns="variant", values="sharpe").round(3).to_string())
    print("\n" + pairs[["window", "variant", "observed", "low", "high"]].round(3).to_string(index=False))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
