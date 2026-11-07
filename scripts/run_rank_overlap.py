"""How often does the 0.8/0.2 blend actually change the basket, and does it help?

RESEARCH_LOG 23.110 found the blend contributed exactly 0.000 Sharpe in 2019-2021,
which means the two scores picked the same eight names there. That is a strange
result worth explaining rather than filing: if the trend term almost never
reorders the eligible names, then "0.8 / 0.2" is decoration, and 23.49's "the
parameter surface is steep" needs restating in terms of the knobs that bite.

This measures the overlap directly -- per rebalance, |A cap B| / top_n for the
pure -vol basket A and the blended basket B -- and then asks the sharper question:
on the dates where they differ, whose extra names did better over the following
20 sessions?

Usage::

    python scripts/run_rank_overlap.py
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

from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import align_panel, pit_universe, price_frame  # noqa: E402
from quant_system.indicators import cross_sectional_zscore, realised_volatility, sma  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy, _traded  # noqa: E402

BASE = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2,
            rebalance_days=20, rebalance_reset="year")
HORIZON = 20


@dataclass
class Ratio(LowVolCoreStrategy):
    """Trend weight on top of the published score; 0 reproduces pure -vol."""

    trend_weight_override: float = 0.2

    def score(self, panel):
        close = price_frame(panel, "close")
        traded = _traded(panel, close)
        vol = realised_volatility(panel, self.vol_window).reindex_like(close)
        trend = close / sma(close, self.trend_window) - 1.0
        scores = (
            (1.0 - self.trend_weight_override) * cross_sectional_zscore(-vol)
            + self.trend_weight_override * cross_sectional_zscore(trend)
        )
        eligible = traded & (close > sma(close, self.trend_window))
        return scores, eligible


def held_sets(weights: pd.DataFrame) -> dict:
    held = weights.fillna(0.0) > 0
    changed = held.ne(held.shift(1)).any(axis=1)
    first = held.index[held.any(axis=1)][0]
    out = {}
    for date in [d for d in held.index[changed] if d >= first]:
        out[date] = set(held.columns[held.loc[date]])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "rank_overlap"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    pure = held_sets(Ratio(trend_weight_override=0.0, **BASE).generate(aligned))
    blend = held_sets(Ratio(trend_weight_override=0.2, **BASE).generate(aligned))

    dates = sorted(set(pure) | set(blend))
    close = pd.DataFrame({s: f["close"] for s, f in aligned.items()}).sort_index()
    forward = close.shift(-HORIZON) / close - 1.0

    rows = []
    for date in dates:
        a = pure.get(date, set())
        b = blend.get(date, set())
        if not a or not b:
            continue
        only_b = sorted(b - a)
        only_a = sorted(a - b)
        fwd = forward.loc[date] if date in forward.index else None
        edge = None
        if fwd is not None and only_b and only_a:
            edge = float(fwd[only_b].mean() - fwd[only_a].mean())
        rows.append(
            {
                "date": date,
                "year": date.year,
                "overlap": len(a & b),
                "identical": len(a & b) == len(a) == len(b),
                "only_blend": len(only_b),
                "only_pure": len(only_a),
                "forward_edge_20d": edge,
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "rebalances.csv", index=False, encoding="utf-8-sig")

    yearly = frame.groupby("year").agg(
        rebalances=("date", "count"),
        mean_overlap=("overlap", "mean"),
        share_identical=("identical", "mean"),
        mean_edge=("forward_edge_20d", "mean"),
    ).reset_index()
    yearly.to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")

    differing = frame[~frame["identical"]]
    summary = {
        "top_n": BASE["top_n"],
        "horizon": HORIZON,
        "rebalances": int(len(frame)),
        "mean_overlap": float(frame["overlap"].mean()),
        "share_identical": float(frame["identical"].mean()),
        "differing": int(len(differing)),
        "edge_mean": float(differing["forward_edge_20d"].mean()) if len(differing) else None,
        "edge_positive_share": float((differing["forward_edge_20d"] > 0).mean()) if len(differing) else None,
        "by_year": yearly.to_dict("records"),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 排序一致性：0.8/0.2 混合到底改不改变篮子", ""]
    lines.append("每次调仓比较「纯 -波动前 8」与「0.8/0.2 混合前 8」的重合数量（满分 %d）。"
                 % BASE["top_n"])
    lines.append("")
    lines.append("| 年份 | 调仓次数 | 平均重合 | 完全相同占比 | 差异日的 20 日边际 |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in yearly.itertuples(index=False):
        edge = "—" if pd.isna(row.mean_edge) else "%+.2f%%" % (row.mean_edge * 100)
        lines.append("| %d | %d | %.1f / %d | %.0f%% | %s |"
                     % (row.year, row.rebalances, row.mean_overlap, BASE["top_n"],
                        100 * row.share_identical, edge))
    lines.append("")
    lines.append("全期平均重合 **%.1f / %d**，完全相同占 **%.0f%%**，有差异的调仓 %d 次。"
                 % (summary["mean_overlap"], BASE["top_n"], 100 * summary["share_identical"],
                    summary["differing"]))
    if summary["edge_mean"] is not None:
        lines.append("在那些有差异的调仓日，混合多买的名字相对少买的名字，接下来 %d 个交易日的"
                     "平均差是 **%+.2f%%**（为正的比例 %.0f%%）。"
                     % (HORIZON, summary["edge_mean"] * 100, 100 * summary["edge_positive_share"]))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + yearly.round(3).to_string(index=False))
    print("\nmean overlap %.2f/%d | identical %.0f%% | differing %d | edge %s"
          % (summary["mean_overlap"], BASE["top_n"], 100 * summary["share_identical"],
             summary["differing"],
             "n/a" if summary["edge_mean"] is None else "%+.3f%%" % (summary["edge_mean"] * 100)))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
