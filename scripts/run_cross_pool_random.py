"""Does the edge survive a narrower cross-section? Random sub-pool test.

RESEARCH_LOG 23.92 killed the sector-split idea: sub-pools were only 0.33
correlated yet each one lost money, because a single-sector cross-section is one
beta wearing eight names. That test conflated two things -- narrowness and sector
concentration -- so it cannot answer the question that actually matters for
"should we buy a second market's data":

    if the same strategy were run on a smaller but *representative* universe,
    would it still have an edge?

Random sub-pools answer exactly that. Draw N names without replacement, run the
book, repeat. If the median Sharpe at N=60 is comfortably positive, the factor
travels to narrower cross-sections and a second market with ~60+ names is worth
paying for. If it collapses toward zero, the edge lives in the breadth of this
particular universe and no amount of extra markets will reproduce it.

Usage::

    python scripts/run_cross_pool_random.py --draws 60 --sizes 60 120
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

TARGET_VOL = 0.12


def run_pool(sub_panel: dict) -> dict:
    strategy = build_strategy(
        "low_vol_core",
        top_n=8,
        vol_window=60,
        trend_window=120,
        max_per_sector=None,
        rebalance_days=20,
        rebalance_reset="year",
    )
    weights = build_portfolio(
        strategy.generate(sub_panel),
        sub_panel,
        risk=RiskConfig(target_annual_vol=TARGET_VOL),
        use_regime=False,
        use_vol_target=True,
    )
    metrics = Backtester(BacktestConfig()).run(sub_panel, weights).metrics
    return {
        "sharpe": float(metrics.sharpe),
        "cagr": float(metrics.cagr),
        "max_drawdown": float(metrics.max_drawdown),
        "turnover": float(metrics.turnover),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "cross_pool_random"))
    parser.add_argument("--draws", type=int, default=60)
    parser.add_argument("--sizes", nargs="+", type=int, default=[60, 120])
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    symbols = sorted(aligned)
    print("universe: %d symbols" % len(symbols), flush=True)

    rng = np.random.default_rng(args.seed)
    rows = []
    started = time.time()
    for size in args.sizes:
        if size >= len(symbols):
            continue
        for draw in range(args.draws):
            picks = rng.choice(len(symbols), size=size, replace=False)
            sub = {symbols[i]: aligned[symbols[i]] for i in picks}
            info = run_pool(sub)
            rows.append({"size": size, "draw": draw, **info})
            if draw % 10 == 0:
                print("  size=%d draw=%d sharpe=%.3f (%.0fs)" % (size, draw, info["sharpe"], time.time() - started), flush=True)

    full = run_pool(aligned)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "draws.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for size, block in frame.groupby("size"):
        summary_rows.append(
            {
                "size": int(size),
                "draws": int(len(block)),
                "sharpe_median": float(block["sharpe"].median()),
                "sharpe_p10": float(block["sharpe"].quantile(0.10)),
                "sharpe_p90": float(block["sharpe"].quantile(0.90)),
                "sharpe_min": float(block["sharpe"].min()),
                "sharpe_max": float(block["sharpe"].max()),
                "share_positive": float((block["sharpe"] > 0).mean()),
                "turnover_median": float(block["turnover"].median()),
                "max_drawdown_median": float(block["max_drawdown"].median()),
            }
        )
    summary_rows.append(
        {
            "size": len(symbols),
            "draws": 1,
            "sharpe_median": full["sharpe"],
            "sharpe_p10": float("nan"),
            "sharpe_p90": float("nan"),
            "sharpe_min": full["sharpe"],
            "sharpe_max": full["sharpe"],
            "share_positive": 1.0 if full["sharpe"] > 0 else 0.0,
            "turnover_median": full["turnover"],
            "max_drawdown_median": full["max_drawdown"],
        }
    )
    summary = pd.DataFrame(summary_rows).sort_values("size")
    summary.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(
            {"seed": args.seed, "draws": args.draws, "sizes": args.sizes,
             "target_vol": TARGET_VOL, "full_pool": full, "rows": summary_rows},
            ensure_ascii=False, indent=2, default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# 随机子池检验：边缘要不要靠宽度", ""]
    lines.append("每次从 %d 只里无放回抽 N 只、跑同一策略（池内 max_per_sector=None），重复若干次。" % len(symbols))
    lines.append("")
    lines.append("| 池子大小 | 抽样次数 | Sharpe 中位 | p10 | p90 | 正的比例 | 中位换手 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary.itertuples(index=False):
        lines.append(
            "| %d | %d | %.3f | %.3f | %.3f | %.0f%% | %.1f |"
            % (row.size, row.draws, row.sharpe_median, row.sharpe_p10, row.sharpe_p90,
               100 * row.share_positive, row.turnover_median)
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + summary.to_string(index=False))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
