"""Split the edge between the trend filter and the score blend.

RESEARCH_LOG 23.109 showed the published strategy beats the obvious alternative by
about one Sharpe, and attributed the gap to the trend filter rather than the
ranking. That attribution was inferred from a variant that changed several things
at once. This isolates the two parts on a 2x2:

    { trend filter on, off } x { 0.8/0.2 blend on, off }

All four run on the same PIT pool with the same sector cap, cadence, risk target
and costs, so the only thing that differs is the eligibility rule and the score.

Usage::

    python scripts/run_filter_blend_ablation.py
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
from quant_system.indicators import cross_sectional_zscore, realised_volatility, sma  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy, _traded  # noqa: E402

WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-06"),
}
TARGET_VOL = 0.12


@dataclass
class Ablation(LowVolCoreStrategy):
    """The published score with the filter and/or the blend switched off."""

    use_filter: bool = True
    use_blend: bool = True

    def score(self, panel):
        close = price_frame(panel, "close")
        traded = _traded(panel, close)
        vol = realised_volatility(panel, self.vol_window).reindex_like(close)
        if self.use_blend:
            trend = close / sma(close, self.trend_window) - 1.0
            scores = (
                self.vol_weight * cross_sectional_zscore(-vol)
                + self.trend_weight * cross_sectional_zscore(trend)
            )
        else:
            scores = -vol
        eligible = traded & (close > sma(close, self.trend_window)) if self.use_filter else traded
        return scores, eligible


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "filter_blend_ablation"))
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    base = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2,
                rebalance_days=20, rebalance_reset="year")
    variants = {
        "过滤+混合（发布）": Ablation(use_filter=True, use_blend=True, **base),
        "只过滤（纯低波打分）": Ablation(use_filter=True, use_blend=False, **base),
        "只混合（不过滤）": Ablation(use_filter=False, use_blend=True, **base),
        "都关（朴素）": Ablation(use_filter=False, use_blend=False, **base),
    }

    rows = []
    sliced_by: dict[str, dict] = {}
    for label, strategy in variants.items():
        weights = build_portfolio(
            strategy.generate(aligned),
            aligned,
            risk=RiskConfig(target_annual_vol=TARGET_VOL),
            use_regime=False,
            use_vol_target=True,
        )
        result = Backtester(BacktestConfig()).run(aligned, weights)
        sliced_by[label] = {}
        for window, (start, end) in WINDOWS.items():
            sliced = slice_result(result, start=start, end=end)
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

    reference = "过滤+混合（发布）"
    paired = []
    for window in WINDOWS:
        base_returns = sliced_by[reference][window].returns
        for label in variants:
            if label == reference:
                continue
            diff = paired_bootstrap_diff(
                sliced_by[label][window].returns, base_returns, n_boot=args.n_boot, seed=args.seed
            )
            paired.append({"window": window, "variant": label, **diff.to_dict()})
    pairs = pd.DataFrame(paired)
    pairs.to_csv(out / "paired.csv", index=False, encoding="utf-8-sig")

    (out / "summary.json").write_text(
        json.dumps({"target_vol": TARGET_VOL, "rows": rows, "paired": paired},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    lines = ["# 消融：趋势过滤 vs 打分混合（同一 PIT 池、同一行业上限）", ""]
    lines.append("2×2：过滤开/关 × 混合开/关。其余（池子、8 只、行业上限 2、20 日、年度重置、12% 目标、成本）全部相同。")
    lines.append("")
    pivot = frame.pivot_table(index="window", columns="variant", values="sharpe")
    order = ["过滤+混合（发布）", "只过滤（纯低波打分）", "只混合（不过滤）", "都关（朴素）"]
    lines.append("| 窗口 | " + " | ".join(order) + " |")
    lines.append("|---|" + "---:|" * len(order))
    for window in WINDOWS:
        lines.append("| %s | " % window + " | ".join("%.3f" % pivot.loc[window, v] for v in order) + " |")
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

    print("\n" + pivot.round(3).to_string())
    print("\n" + pairs[["window", "variant", "observed", "low", "high"]].round(3).to_string(index=False))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
