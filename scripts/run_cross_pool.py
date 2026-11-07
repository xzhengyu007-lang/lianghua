"""Would a second market actually buy us independent evidence?

RESEARCH_LOG 23.89 concluded the binding constraint is evidence: at a true Sharpe
near 0.90 the current 7.1 years carries only 56% power, and 80% needs 13.6 years.
The one lever that changes that without new signal is more *independent* return
streams, not more years of the same one.

Before paying for another market's data, this asks the question locally: if the
universe is split into disjoint sector pools and the same strategy is run on each
independently, how correlated are the resulting return streams? Low correlation
means running several pools multiplies the effective sample; high correlation
means a second market would mostly repackage the same factor.

Two design choices that matter:

* the sector cap is switched OFF inside each sub-pool, because a single-sector
  universe with a 2-name cap could hold two stocks and measure nothing;
* a single phase is used, because 12 pools x 20 phases would be 240 backtests.
  The phase only shifts the calendar; it does not change how two pools co-move.

Usage::

    python scripts/run_cross_pool.py
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

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.sectors import sector_of  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

TARGET_VOL = 0.12
MIN_NAMES = 20


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
    result = Backtester(BacktestConfig()).run(sub_panel, weights)
    metrics = result.metrics
    return {
        "returns": result.returns,
        "sharpe": float(metrics.sharpe),
        "cagr": float(metrics.cagr),
        "max_drawdown": float(metrics.max_drawdown),
        "turnover": float(metrics.turnover),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "cross_pool"))
    parser.add_argument("--min-names", type=int, default=MIN_NAMES)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    groups: dict[str, list[str]] = {}
    for symbol in sorted(aligned):
        groups.setdefault(sector_of(symbol), []).append(symbol)
    usable = {s: members for s, members in groups.items() if len(members) >= args.min_names}
    print("sectors with >=%d names: %s" % (args.min_names, {k: len(v) for k, v in usable.items()}), flush=True)

    rows = []
    streams = {}
    for sector, members in sorted(usable.items()):
        sub = {symbol: aligned[symbol] for symbol in members}
        info = run_pool(sub)
        streams[sector] = info["returns"]
        rows.append(
            {
                "pool": sector,
                "names": len(members),
                "sharpe": info["sharpe"],
                "cagr": info["cagr"],
                "max_drawdown": info["max_drawdown"],
                "turnover": info["turnover"],
            }
        )
        print("  %-14s n=%3d sharpe=%.3f" % (sector, len(members), info["sharpe"]), flush=True)

    whole = run_pool(aligned)
    rows.append({"pool": "ALL", "names": len(aligned), "sharpe": whole["sharpe"],
                 "cagr": whole["cagr"], "max_drawdown": whole["max_drawdown"],
                 "turnover": whole["turnover"]})
    streams["ALL"] = whole["returns"]

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "pools.csv", index=False, encoding="utf-8-sig")

    parts = [s for s in streams if s != "ALL"]
    matrix = pd.concat({s: streams[s] for s in parts}, axis=1).dropna()
    corr = matrix.corr()
    corr.to_csv(out / "correlations.csv", encoding="utf-8-sig")

    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
    mean_corr = float(upper.mean()) if len(upper) else float("nan")
    n = len(parts)
    effective = n / (1.0 + (n - 1) * mean_corr) if n > 1 else float("nan")

    combined = matrix.mean(axis=1)
    combined_sharpe = float(combined.mean() / combined.std(ddof=0) * np.sqrt(252)) if combined.std(ddof=0) else float("nan")

    summary = {
        "target_vol": TARGET_VOL,
        "pools": rows,
        "mean_pairwise_correlation": mean_corr,
        "n_pools": n,
        "effective_independent_pools": effective,
        "combined_sharpe": combined_sharpe,
        "whole_pool_sharpe": whole["sharpe"],
        "best_single_pool_sharpe": float(frame[frame["pool"] != "ALL"]["sharpe"].max()),
        "worst_single_pool_sharpe": float(frame[frame["pool"] != "ALL"]["sharpe"].min()),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 切池检验：多池并行能买到多少独立证据", ""]
    lines.append("把 %d 只按行业切成互不重叠的池子（每个 ≥%d 只），各自独立跑同一策略。" % (len(aligned), args.min_names))
    lines.append("池内 `max_per_sector=None`（否则单行业池只能持两只），单相位。")
    lines.append("")
    lines.append("| 池 | 只数 | Sharpe | CAGR | 最大回撤 | 换手 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| %s | %d | %.3f | %+.2f%% | %.2f%% | %.1f |"
            % (row["pool"], row["names"], row["sharpe"], row["cagr"] * 100,
               row["max_drawdown"] * 100, row["turnover"])
        )
    lines.append("")
    lines.append("平均两两相关：**%.3f**；等效独立池数 **%.1f / %d**；等权合并 Sharpe **%.3f**（全池 %.3f）。"
                 % (mean_corr, effective, n, combined_sharpe, whole["sharpe"]))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + frame.to_string(index=False))
    print("\nmean pairwise corr: %.3f | effective pools: %.1f/%d | combined Sharpe: %.3f | whole: %.3f"
          % (mean_corr, effective, n, combined_sharpe, whole["sharpe"]))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
