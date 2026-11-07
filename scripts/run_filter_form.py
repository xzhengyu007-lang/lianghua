"""Is the trend filter robust, or did it just happen to be set to SMA(120)?

RESEARCH_LOG 23.110 showed the filter carries essentially the whole edge (+0.60 /
-0.09 / +0.18 Sharpe) while the score blend contributes nothing. But the filter has
never been examined on its own: "close above the 120-day average" has been the rule
since day one, and nobody asked whether 120 is special, whether the form matters,
or whether any of it is just the phase luck that 23.82 already found everywhere.

This sweeps the window (60/90/120/150/250) and three alternative forms -- SMA
slope, a drawdown threshold, and no filter at all -- with everything else fixed.

Usage::

    python scripts/run_filter_form.py
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
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe, price_frame  # noqa: E402
from quant_system.indicators import realised_volatility, sma  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy, _traded  # noqa: E402

WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-06"),
}
TARGET_VOL = 0.12
BASE = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2,
            rebalance_days=20, rebalance_reset="year")


@dataclass
class Filtered(LowVolCoreStrategy):
    """Pure -vol score; only the eligibility rule varies."""

    filter_kind: str = "sma"
    filter_window: int = 120
    filter_threshold: float = 0.10

    def score(self, panel):
        close = price_frame(panel, "close")
        traded = _traded(panel, close)
        vol = realised_volatility(panel, self.vol_window).reindex_like(close)
        scores = -vol

        if self.filter_kind == "none":
            eligible = traded
        elif self.filter_kind == "sma":
            eligible = traded & (close > sma(close, self.filter_window))
        elif self.filter_kind == "slope":
            line = sma(close, self.filter_window)
            eligible = traded & (line > line.shift(20))
        elif self.filter_kind == "drawdown":
            peak = close.rolling(self.filter_window, min_periods=self.filter_window).max()
            eligible = traded & (close > peak * (1.0 - self.filter_threshold))
        else:
            raise ValueError(self.filter_kind)
        return scores, eligible


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "filter_form"))
    parser.add_argument("--blend", action="store_true",
                        help="also run 20 phases for SMA120 vs SMA250")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    variants = {
        "无过滤": Filtered(filter_kind="none", **BASE),
        "SMA60": Filtered(filter_kind="sma", filter_window=60, **BASE),
        "SMA90": Filtered(filter_kind="sma", filter_window=90, **BASE),
        "SMA120（发布）": Filtered(filter_kind="sma", filter_window=120, **BASE),
        "SMA150": Filtered(filter_kind="sma", filter_window=150, **BASE),
        "SMA250": Filtered(filter_kind="sma", filter_window=250, **BASE),
        "斜率(120)": Filtered(filter_kind="slope", filter_window=120, **BASE),
        "回撤10%(250)": Filtered(filter_kind="drawdown", filter_window=250, filter_threshold=0.10, **BASE),
    }

    rows = []
    for label, strategy in variants.items():
        weights = build_portfolio(
            strategy.generate(aligned),
            aligned,
            risk=RiskConfig(target_annual_vol=TARGET_VOL),
            use_regime=False,
            use_vol_target=True,
        )
        held = weights.fillna(0.0) > 0
        first = held.index[held.any(axis=1)][0]
        for window, (start, end) in WINDOWS.items():
            sliced = slice_result(
                Backtester(BacktestConfig()).run(aligned, weights), start=start, end=end
            )
            m = sliced.metrics
            # Exposure matters here: a longer window simply refuses to hold during
            # more of the sample, so part of its higher Sharpe may be time out of
            # the market rather than better decisions.
            gross = sliced.weights.fillna(0.0).sum(axis=1)
            rows.append({
                "variant": label,
                "window": window,
                "total_return": float(m.total_return),
                "sharpe": float(m.sharpe),
                "max_drawdown": float(m.max_drawdown),
                "turnover": float(m.turnover),
                "mean_gross": float(gross.mean()),
                "days_invested": float((gross > 0).mean()),
                "first_position": str(first.date()),
            })
        print("  %s done (first position %s)" % (label, first.date()), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "panel.csv", index=False, encoding="utf-8-sig")

    order = list(variants)
    pivot = frame.pivot_table(index="window", columns="variant", values="sharpe")[order]
    full = frame.groupby("variant")["sharpe"].mean().reindex(order)

    summary = {
        "target_vol": TARGET_VOL,
        "windows": {k: list(v) for k, v in WINDOWS.items()},
        "rows": rows,
        "mean_sharpe_by_variant": {k: float(v) for k, v in full.items()},
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 趋势过滤的形式与窗口：它是稳健的，还是恰好设在 120", ""]
    lines.append("打分固定为纯 -波动（§23.110 已证明混合项不贡献），只改可交易资格里的过滤规则。")
    lines.append("同池、同 8 只、同行业上限 2、同 20 日调仓、同 12% 目标、同成本。")
    lines.append("")
    lines.append("| 变体 | " + " | ".join(WINDOWS) + " | 三段均值 | 首次建仓 | 在场天数占比(三段) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    firsts = frame.groupby("variant")["first_position"].first()
    invested = frame.pivot_table(index="window", columns="variant", values="days_invested")[order]
    for label in order:
        lines.append("| %s | " % label
                     + " | ".join("%.3f" % pivot.loc[w, label] for w in WINDOWS)
                     + " | %.3f | %s | %s |" % (full[label], firsts[label],
                                                " / ".join("%.0f%%" % (100 * invested.loc[w, label]) for w in WINDOWS)))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    if args.blend:
        # 23.82: a single phase is a date-luck draw. If SMA250's edge is real it must
        # survive being averaged over the 20 phases the deployable config uses.
        blend_rows = []
        for label, window in (("SMA120（发布）", 120), ("SMA250", 250)):
            frames = []
            for offset in range(20):
                strat = Filtered(filter_kind="sma", filter_window=window,
                                 rebalance_offset=offset, **BASE)
                frames.append(build_portfolio(
                    strat.generate(aligned), aligned,
                    risk=RiskConfig(target_annual_vol=TARGET_VOL),
                    use_regime=False, use_vol_target=True,
                ))
            weights = sum(frames) / len(frames)
            blended = Backtester(BacktestConfig()).run(aligned, weights)
            for win, (start, end) in WINDOWS.items():
                m = slice_result(blended, start=start, end=end).metrics
                blend_rows.append({"variant": label, "window": win,
                                   "sharpe": float(m.sharpe),
                                   "total_return": float(m.total_return)})
            print("  blend %s done" % label, flush=True)
        blend = pd.DataFrame(blend_rows)
        blend.to_csv(out / "blend.csv", index=False, encoding="utf-8-sig")
        bp = blend.pivot_table(index="window", columns="variant", values="sharpe")
        lines.append("## 相位混合后（20 相位等权，部署口径）")
        lines.append("")
        lines.append("| 变体 | " + " | ".join(WINDOWS) + " |")
        lines.append("|---|" + "---:|" * len(WINDOWS))
        for label in bp.columns:
            lines.append("| %s | " % label
                         + " | ".join("%.3f" % bp.loc[w, label] for w in WINDOWS) + " |")
        lines.append("")
        Path(out / "report.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n" + bp.round(3).to_string())

    print("\n" + pivot.round(3).to_string())
    print("\nmean sharpe:\n" + full.round(3).to_string())
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
