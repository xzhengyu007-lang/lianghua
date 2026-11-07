"""Selection slack: how many names actually compete for the eight slots?

RESEARCH_LOG 23.75 measured the tradable cross-section shrinking from a median of
251 names to 128, and 23.79 showed the profit that came with the wider pool has
disappeared with it. But "top 8 of 267" was never the real choice: the low-vol
score only ranks names that already have a full volatility window and are above
their trend line. That investable set is much smaller than the tradable set and
it can shrink for a second, independent reason -- a bear phase pushes names below
their moving average.

This separates the two:

* tradable  -- has a bar and positive volume today (23.75);
* eligible  -- passes the trend filter and has a score, i.e. can actually be bought;
* attainable -- the most names the sector cap allows given today's eligible mix.

If attainable ever falls below top_n the published rule cannot be executed as
advertised, and that is a fact about the current pool rather than about returns.

Usage::

    python scripts/run_selection_slack.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.sectors import sector_of  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402


def rebalance_dates(weights: pd.DataFrame) -> list:
    """Dates where the target set changes -- the strategy's own cadence."""
    held = weights.fillna(0.0) > 0
    changed = held.ne(held.shift(1)).any(axis=1)
    first = held.index[held.any(axis=1)][0]
    return [d for d in held.index[changed] if d >= first]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "selection_slack"))
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--max-per-sector", type=int, default=2)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    close = pd.DataFrame({s: f["close"] for s, f in aligned.items()})
    volume = pd.DataFrame({s: f["volume"] for s, f in aligned.items()})
    tradable = close.notna() & volume.notna() & (volume > 0)

    strategy = build_strategy(
        "low_vol_core",
        top_n=args.top_n,
        max_per_sector=args.max_per_sector,
        rebalance_reset="year",
    )
    scores, eligible = strategy.score(aligned)
    investable = scores.notna() & eligible
    weights = strategy.generate(aligned)
    sector = {c: sector_of(c) for c in close.columns}

    rows = []
    for date in rebalance_dates(weights):
        names = [c for c in close.columns if investable.at[date, c]]
        per_sector: dict[str, int] = {}
        for name in names:
            per_sector[sector[name]] = per_sector.get(sector[name], 0) + 1
        attainable = sum(min(args.max_per_sector, n) for n in per_sector.values())
        held = int((weights.loc[date].fillna(0.0) > 0).sum())
        rows.append(
            {
                "date": date,
                "n_tradable": int(tradable.loc[date].sum()),
                "n_eligible": len(names),
                "n_held": held,
                "attainable_under_cap": int(attainable),
                "sectors_with_eligible": len(per_sector),
                "cap_binds": bool(attainable < args.top_n),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "rebalances.csv", index=False, encoding="utf-8-sig")

    frame["year"] = pd.to_datetime(frame["date"]).dt.year
    yearly = frame.groupby("year").agg(
        rebalances=("date", "count"),
        median_tradable=("n_tradable", "median"),
        median_eligible=("n_eligible", "median"),
        min_eligible=("n_eligible", "min"),
        median_attainable=("attainable_under_cap", "median"),
        median_held=("n_held", "median"),
        cap_bind_share=("cap_binds", "mean"),
    ).reset_index()
    yearly["eligible_to_tradable"] = yearly["median_eligible"] / yearly["median_tradable"]
    yearly.to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")

    summary = {
        "top_n": args.top_n,
        "max_per_sector": args.max_per_sector,
        "rebalances": int(len(frame)),
        "start": str(frame["date"].min().date()),
        "end": str(frame["date"].max().date()),
        "yearly": yearly.to_dict("records"),
        "latest": {k: (str(v) if k == "date" else v) for k, v in frame.iloc[-1].to_dict().items()},
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 可选性：到底有多少名字在竞争这 8 个仓位", ""]
    lines.append(
        "策略 `low_vol_core`（top_n=%d，每行业上限 %d，reset=year）。%d 次调仓，%s .. %s。"
        % (args.top_n, args.max_per_sector, len(frame), summary["start"], summary["end"])
    )
    lines.append("")
    lines.append("| 年份 | 调仓次数 | 中位可交易 | 中位可选 | 最少可选 | 可选/可交易 | 中位可容纳 | 上限咬住的调仓占比 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in yearly.itertuples(index=False):
        lines.append(
            "| %d | %d | %.0f | %.0f | %d | %.0f%% | %.0f | %.0f%% |"
            % (
                row.year,
                row.rebalances,
                row.median_tradable,
                row.median_eligible,
                row.min_eligible,
                100 * row.eligible_to_tradable,
                row.median_attainable,
                100 * row.cap_bind_share,
            )
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(yearly.to_string(index=False))
    print("\nlatest rebalance: %s" % summary["latest"])
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
