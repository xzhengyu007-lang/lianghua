"""Decompose every factor's IC into "still listed" versus "eventually delisted".

RESEARCH_LOG 23.76 showed the low-vol score earns most of its IC from names that
later leave the pool. If that is a general property of this panel rather than a
quirk of one factor, it explains a whole class of results in FINDINGS: factors
whose IC is statistically strong but whose portfolio overlay loses money
(momentum, Amihud, MAX). A factor whose signal lives entirely in the delisting
cohort cannot be traded on its own -- you do not know who leaves in advance --
and an overlay built on it dilutes the distress protection the base book already
enjoys.

This measures all seven library factors on the same three cuts and the same
panel, so the comparison is internally consistent.

Usage::

    python scripts/run_factor_cohorts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.analytics import forward_returns, ic_summary, spearman_ic  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.factors import factor_library  # noqa: E402

HORIZONS = (5, 20, 60)


def load(cache_dir: Path, vol_window: int):
    panel = _rebuilt_panel(list(pit_universe()), cache_dir, "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _ = align_panel(panel)
    close = pd.DataFrame({s: f["close"] for s, f in aligned.items()})
    volume = pd.DataFrame({s: f["volume"] for s, f in aligned.items()})
    eligible = close.notna() & volume.notna() & (volume > 0)
    return aligned, close, eligible, factor_library(aligned, vol_window=vol_window)


def split_columns(columns) -> tuple[list[str], list[str]]:
    codes = set(BROAD_UNIVERSE)
    listed = [c for c in columns if c[-6:] in codes]
    gone = [c for c in columns if c[-6:] not in codes]
    return listed, gone


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "factor_cohorts"))
    parser.add_argument("--vol-window", type=int, default=60)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    aligned, close, eligible, library = load(Path(args.cache_dir), args.vol_window)
    listed, gone = split_columns(close.columns)
    print("cohorts: listed=%d delisted=%d" % (len(listed), len(gone)), flush=True)

    rows = []
    for name, raw in library.items():
        signal = raw.reindex_like(close).where(eligible)
        for horizon in HORIZONS:
            future = forward_returns(close, horizon)
            for cohort, frame in (
                ("all", signal),
                ("listed", signal[listed]),
                ("delisted", signal[gone]),
            ):
                series = spearman_ic(frame, future)
                stats = ic_summary(series)
                rows.append(
                    {
                        "factor": name,
                        "horizon": horizon,
                        "cohort": cohort,
                        "mean_ic": stats["mean_ic"],
                        "t_stat": stats["t_stat"],
                        "positive_ratio": stats["positive_ratio"],
                        "days": int(len(series)),
                    }
                )
        print("  %s done" % name, flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "ic_by_cohort.csv", index=False, encoding="utf-8-sig")

    lines = ["# 因子的成分分解（PIT 267）", ""]
    lines.append(
        "面板 %s .. %s，%d 只（仍在册 %d / 最终退市 %d）。三个口径同一天、同一分位、"
        "同一前瞻收益，只有成员不同。"
        % (close.index.min().date(), close.index.max().date(), close.shape[1], len(listed), len(gone))
    )
    lines.append("")
    for horizon in HORIZONS:
        block = frame[frame["horizon"] == horizon].pivot_table(
            index="factor", columns="cohort", values="mean_ic"
        )
        lines.append("## %d 日前瞻" % horizon)
        lines.append("")
        lines.append("| 因子 | 全池 | 仍在册 | 最终退市 | 退市/在册 |")
        lines.append("|---|---:|---:|---:|---:|")
        for factor, row in block.sort_values("all", ascending=False).iterrows():
            ratio = row["delisted"] / row["listed"] if row["listed"] else float("nan")
            lines.append(
                "| %s | %+.4f | %+.4f | %+.4f | %.1fx |"
                % (factor, row["all"], row["listed"], row["delisted"], ratio)
            )
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "symbols": int(close.shape[1]),
        "listed": len(listed),
        "delisted": len(gone),
        "start": str(close.index.min().date()),
        "end": str(close.index.max().date()),
        "horizons": list(HORIZONS),
        "factors": sorted(library),
        "rows": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    wide = frame[frame["horizon"] == 20].pivot_table(index="factor", columns="cohort", values="mean_ic")
    print("\n20-day IC by cohort:\n" + wide.round(4).to_string())
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
