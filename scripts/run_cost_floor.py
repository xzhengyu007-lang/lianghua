"""A structural floor on per-trade cost that no parameter choice can undercut.

RESEARCH_LOG 23.70 decomposed the published cost of 10.2bp per side into explicit
fees plus a 5bp slippage assumption. That 5bp is a *parameter*. A-share equities
have a minimum price increment of one fen (0.01 CNY), which puts a floor on the
quoted spread: a stock trading at 5 CNY has a one-tick spread of 20bp, whatever
the model says. If the book holds cheap names, no slippage parameter below that
floor is realistic.

23.81 already found the book's median holding price moved from 5.76 to 18.54
between the first and last window, so the relevance of this floor is itself
time-varying. This measures it per window from the actual held basket.

Usage::

    python scripts/run_cost_floor.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

WINDOWS = {
    "2019-2021": "reports/attrib_1921",
    "2022-2023": "reports/attrib_2022",
    "2024-2026": "reports/attrib_2024",
}
TICK = 0.01          # A-share minimum price increment, CNY
COMMISSION_BP = 2.5  # per side
TRANSFER_BP = 0.1    # per side
STAMP_BP = 5.0       # sell side only
PUBLISHED_SLIPPAGE_BP = 5.0


def window_cost(directory: Path) -> dict:
    positions = pd.read_csv(directory / "positions.csv", parse_dates=["date"])
    held = positions[positions["shares"] > 0]
    if held.empty:
        return {}

    # Per side, crossing half the quoted spread (one tick) costs tick/2 / price.
    held = held.assign(tick_bp=TICK / held["close"] * 1e4)
    weighted = held.groupby("date").apply(
        lambda block: pd.Series(
            {
                "price_median": block["close"].median(),
                "tick_bp_median": block["tick_bp"].median(),
                "tick_bp_weighted": float(
                    (block["tick_bp"] * block["weight"]).sum() / max(block["weight"].sum(), 1e-12)
                ),
            }
        ),
        include_groups=False,
    )
    price_median = float(weighted["price_median"].median())
    tick_median = float(weighted["tick_bp_median"].median())
    tick_weighted = float(weighted["tick_bp_weighted"].median())

    half_spread = 0.5 * tick_weighted
    full_spread = tick_weighted
    buy_published = PUBLISHED_SLIPPAGE_BP + COMMISSION_BP + TRANSFER_BP
    sell_published = PUBLISHED_SLIPPAGE_BP + COMMISSION_BP + TRANSFER_BP + STAMP_BP
    buy_floor = max(half_spread, PUBLISHED_SLIPPAGE_BP) + COMMISSION_BP + TRANSFER_BP
    sell_floor = max(half_spread, PUBLISHED_SLIPPAGE_BP) + COMMISSION_BP + TRANSFER_BP + STAMP_BP
    return {
        "price_median": price_median,
        "tick_bp_median": tick_median,
        "tick_bp_weighted": tick_weighted,
        "half_spread_bp": half_spread,
        "full_spread_bp": full_spread,
        "buy_published_bp": buy_published,
        "sell_published_bp": sell_published,
        "buy_floor_bp": buy_floor,
        "sell_floor_bp": sell_floor,
        "excess_per_side_bp": 0.5 * ((buy_floor - buy_published) + (sell_floor - sell_published)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "cost_floor"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, relative in WINDOWS.items():
        directory = ROOT / relative
        if not (directory / "positions.csv").exists():
            print("skip missing %s" % directory)
            continue
        info = window_cost(directory)
        info["window"] = label
        rows.append(info)
        print("  %-11s price median %.2f  tick %.1fbp  half-spread %.1fbp  excess/side %+.1fbp"
              % (label, info["price_median"], info["tick_bp_weighted"],
                 info["half_spread_bp"], info["excess_per_side_bp"]), flush=True)

    frame = pd.DataFrame(rows)

    # Translate the per-side excess into an annual drag and a Sharpe impact,
    # using the deployable configuration's own turnover and volatility.
    cadence = ROOT / "reports" / "cadence_blend" / "panel.csv"
    if cadence.exists():
        panel = pd.read_csv(cadence)
        panel = panel[panel["cadence"] == 20].set_index("window")
        frame["turnover"] = frame["window"].map(panel["turnover"])
        frame["annual_volatility"] = frame["window"].map(panel["annual_volatility"])
        frame["annual_drag_bp"] = frame["turnover"] * frame["excess_per_side_bp"]
        frame["sharpe_impact"] = (frame["annual_drag_bp"] / 1e4) / frame["annual_volatility"]

    frame.to_csv(out / "floor.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps({"tick": TICK, "slippage_assumption_bp": PUBLISHED_SLIPPAGE_BP, "rows": rows},
                   ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    lines = ["# 成本的微观结构下界", ""]
    lines.append("A 股最小变动价位 %.2f 元。若按交叉半个价差（一个 tick）估算，单边价差成本 = 0.5 × tick / 价格，" % TICK)
    lines.append("这是任何滑点参数都不能低于的下界（除非假设自己是挂单方，那是另一回事）。")
    lines.append("")
    lines.append("| 窗口 | 持仓价格中位 | 加权 tick | 单边半价差 | 单边全价差 | published 买/卖 | 下界买/卖 | 超出 published（单边） |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| %s | %.2f | %.1f bp | %.1f bp | %.1f bp | %.1f / %.1f | %.1f / %.1f | **%+.1f bp** |"
            % (row["window"], row["price_median"], row["tick_bp_weighted"], row["half_spread_bp"],
               row["full_spread_bp"], row["buy_published_bp"], row["sell_published_bp"],
               row["buy_floor_bp"], row["sell_floor_bp"], row["excess_per_side_bp"])
        )
    lines.append("")
    if "annual_drag_bp" in frame.columns:
        lines.append("## 换成年度拖累（用部署口径的换手与波动）")
        lines.append("")
        lines.append("| 窗口 | 年换手 | 年化波动 | 年度额外拖累 | Sharpe 影响 |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in frame.itertuples(index=False):
            lines.append("| %s | %.1fx | %.2f%% | **%.0f bp** | %.3f |"
                         % (row.window, row.turnover, row.annual_volatility * 100,
                            row.annual_drag_bp, row.sharpe_impact))
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + frame[["window", "price_median", "half_spread_bp", "excess_per_side_bp"]].round(2).to_string(index=False))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
