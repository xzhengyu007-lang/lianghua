"""Where the money actually came from: realised P&L split by eventual fate.

RESEARCH_LOG 23.76/23.77 decomposed factor IC by whether a name is still listed
today. That invited an over-reading: "the delisting cohort's IC cannot be traded,
you do not know who dies". This script checks that against the book's own fills
instead of reasoning about it.

Every exit in the three archived windows is tagged rebalance -- the forced
liquidation path never fired -- which means the strategy traded these names while
they were alive and sold them normally. So the question is not whether the cohort
was avoidable, but how much of the realised P&L it produced and whether that
source still exists.

Usage::

    python scripts/run_pnl_cohorts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import BROAD_UNIVERSE  # noqa: E402

WINDOWS = {
    "2019-2021": "reports/attrib_1921",
    "2022-2023": "reports/attrib_2022",
    "2024-2026": "reports/attrib_2024",
}
COHORTS = ("listed", "delisted")


def cohort_of(symbol: str, codes: set[str]) -> str:
    return "listed" if symbol[-6:] in codes else "delisted"


def summarise(directory: Path, label: str, codes: set[str]) -> list[dict]:
    positions = pd.read_csv(directory / "positions.csv", parse_dates=["date"])
    trades = pd.read_csv(directory / "trades.csv", parse_dates=["date"])
    held = positions[positions["shares"] > 0].copy()
    held["cohort"] = [cohort_of(s, codes) for s in held["symbol"]]
    trades["cohort"] = [cohort_of(s, codes) for s in trades["symbol"]]

    exposure = held.groupby("cohort")["weight"].sum()
    total_exposure = float(exposure.sum())
    rows = []
    for cohort in COHORTS:
        book = trades[trades["cohort"] == cohort]
        per_name = book.groupby("symbol")["realized_pnl"].sum().sort_values(ascending=False)
        realised = float(per_name.sum()) if len(per_name) else 0.0
        gross_wins = float(per_name[per_name > 0].sum()) if len(per_name) else 0.0
        top3 = float(per_name.head(3).sum()) if len(per_name) else 0.0
        rows.append(
            {
                "window": label,
                "cohort": cohort,
                "names_held": int(held[held["cohort"] == cohort]["symbol"].nunique()),
                "name_days": int(len(held[held["cohort"] == cohort])),
                "exposure": float(exposure.get(cohort, 0.0)),
                "exposure_share": float(exposure.get(cohort, 0.0) / total_exposure) if total_exposure else float("nan"),
                "realised_pnl": realised,
                "trades": int(len(book)),
                "winners": int((per_name > 0).sum()),
                "win_rate": float((per_name > 0).mean()) if len(per_name) else float("nan"),
                "top3_pnl": top3,
                "top3_share_of_cohort": float(top3 / realised) if realised else float("nan"),
                "top3_share_of_wins": float(top3 / gross_wins) if gross_wins else float("nan"),
            }
        )
    total = sum(r["realised_pnl"] for r in rows)
    for row in rows:
        row["pnl_share"] = float(row["realised_pnl"] / total) if total else float("nan")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "pnl_cohorts"))
    args = parser.parse_args(argv)

    codes = set(BROAD_UNIVERSE)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for label, relative in WINDOWS.items():
        directory = ROOT / relative
        if not (directory / "trades.csv").exists():
            print("skip missing %s" % directory)
            continue
        rows.extend(summarise(directory, label, codes))

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps({"windows": WINDOWS, "rows": rows}, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    lines = ["# 已实现盈亏的成分分解", ""]
    lines.append("按「今天仍在册 / 最终退市」两个口径拆开三个归档窗口的已平仓盈亏。")
    lines.append("全部卖出标签都是 `rebalance`，退市强制清算没有触发过——也就是说这本书")
    lines.append("**在它们活着的时候正常交易并卖出**，不是躲开了会死的那些。")
    lines.append("")
    lines.append("| 窗口 | 口径 | 持有名数 | 敞口占比 | 已实现盈亏 | 占该段盈亏 | 胜率 | 前三名占本口径 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| %s | %s | %d | %.1f%% | %+.0f | %.1f%% | %.0f%% | %.0f%% |"
            % (
                row["window"],
                row["cohort"],
                row["names_held"],
                100 * row["exposure_share"],
                row["realised_pnl"],
                100 * row["pnl_share"],
                100 * row["win_rate"],
                100 * row["top3_share_of_cohort"],
            )
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(frame[["window", "cohort", "names_held", "exposure_share", "realised_pnl", "pnl_share", "win_rate", "top3_share_of_cohort"]].to_string(index=False))
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
