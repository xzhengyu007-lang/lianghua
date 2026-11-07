"""Detection power for the Sharpe we can actually claim.

RESEARCH_LOG 23.64 answered "how long before a Sharpe of 1.0 is provable" and got
about six years for a 95% interval and 9.3 years for 80% power. That calculation
inherited the *single-phase* Sharpe of 1.16. 23.82 and 23.86 then showed the
deployable configuration delivers a repeatable 0.90, not 1.16 -- and power falls
faster than Sharpe does, because it scales roughly with the square.

So the question the roadmap actually needs is: given a true Sharpe near 0.90, how
much record is required before "this works" is a finding rather than a hope, and
how much power does the record we already have?

Usage::

    python scripts/run_power_recheck.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.bootstrap import bootstrap_summary  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402
from run_power_study import Z95, Z80, analytic_years, simulate  # noqa: E402

PHASES = tuple(range(20))
TARGET_VOL = 0.12
SHARPES = (0.80, 0.85, 0.90, 0.95, 1.00, 1.16)
YEARS = (3, 5, 6, 7.3, 8, 10, 12, 14, 16)


def blended_weights(aligned) -> pd.DataFrame:
    frames = []
    for offset in PHASES:
        strategy = build_strategy(
            "low_vol_core",
            top_n=8,
            vol_window=60,
            trend_window=120,
            max_per_sector=2,
            rebalance_days=20,
            rebalance_reset="year",
            rebalance_offset=offset,
        )
        frames.append(
            build_portfolio(
                strategy.generate(aligned),
                aligned,
                risk=RiskConfig(target_annual_vol=TARGET_VOL),
                use_regime=False,
                use_vol_target=True,
            )
        )
    return sum(frames) / len(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "power_recheck"))
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    print("building deployable blend...", flush=True)
    weights = blended_weights(aligned)
    returns = Backtester(BacktestConfig()).run(aligned, weights).returns
    years_of_record = float(len(returns)) / 252.0
    print("record: %.2f years (%d days)" % (years_of_record, len(returns)), flush=True)

    print("bootstrapping the recorded Sharpe...", flush=True)
    summary = bootstrap_summary(returns, n_boot=args.n_boot, seed=args.seed)
    block = summary["sharpe"]
    # "point" also starts with a p; only p<number> keys are interval endpoints.
    bounds = sorted(
        (key, value) for key, value in block.items() if re.fullmatch(r"p\d+", key)
    )
    recorded = {
        "sharpe_point": float(block["point"]),
        "sharpe_low": float(bounds[0][1]),
        "sharpe_high": float(bounds[-1][1]),
        "sharpe_median": float(block["median"]),
        "n_boot": int(args.n_boot),
        "years": years_of_record,
        "observed_sharpe": float(returns.mean() / returns.std(ddof=0) * np.sqrt(252)),
    }

    print("simulating power grid...", flush=True)
    rows = []
    for i, sharpe in enumerate(SHARPES):
        for years in YEARS:
            rows.append(simulate(sharpe, years, draws=args.draws, seed=args.seed + i * 100 + int(years * 10)))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "power.csv", index=False, encoding="utf-8-sig")

    analytic = {
        "%.2f" % s: {
            "years_for_95ci": analytic_years(s, power_z=Z95),
            "years_for_80pct_power": analytic_years(s, power_z=Z95 + Z80),
        }
        for s in SHARPES
    }

    pivot = frame.pivot(index="sharpe", columns="years", values="power")
    payload = {
        "target_vol": TARGET_VOL,
        "phases": len(PHASES),
        "recorded": recorded,
        "sharpes": list(SHARPES),
        "years": list(YEARS),
        "analytic": analytic,
        "power": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 按可复现 Sharpe 重算检出力", ""]
    lines.append(
        "部署口径（混合 + %.0f%%）实测记录 %.2f 年。自助法（%d 次）Sharpe 点估计 %.3f，"
        "95%% 区间 [%.3f, %.3f]。" % (
            TARGET_VOL * 100, years_of_record, args.n_boot,
            recorded["sharpe_point"], recorded["sharpe_low"], recorded["sharpe_high"],
        )
    )
    lines.append("")
    lines.append("## 达到 95% 显著所需的年数（解析）")
    lines.append("")
    lines.append("| 真 Sharpe | 95% 区间排除 0 | 80% 检出力 |")
    lines.append("|---:|---:|---:|")
    for s in SHARPES:
        a = analytic["%.2f" % s]
        lines.append("| %.2f | %.1f 年 | %.1f 年 |" % (s, a["years_for_95ci"], a["years_for_80pct_power"]))
    lines.append("")
    lines.append("## 模拟检出力")
    lines.append("")
    header = "| 真 Sharpe | " + " | ".join("%s 年" % y for y in YEARS) + " |"
    lines.append(header)
    lines.append("|---|" + "---:|" * len(YEARS))
    for sharpe in SHARPES:
        cells = " | ".join("%.0f%%" % (pivot.loc[sharpe, y] * 100) for y in YEARS)
        lines.append("| %.2f | %s |" % (sharpe, cells))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nrecorded Sharpe: %.3f [%.3f, %.3f] over %.2f years" % (
        recorded["sharpe_point"], recorded["sharpe_low"], recorded["sharpe_high"], years_of_record))
    print("\n" + (pivot * 100).round(0).to_string())
    print("\nreport: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
