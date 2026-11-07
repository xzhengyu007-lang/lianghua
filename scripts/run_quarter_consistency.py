"""A sign test on quarters: far more powerful than a Sharpe interval, and weaker.

RESEARCH_LOG 23.89 found the current record carries only 56% power for a Sharpe of
0.90, and 80% needs 13.6 years. That is a statement about *magnitude*. A different
question can be answered with the same data and much more power: does the strategy
make money in more than half of its quarters?

For a true Sharpe S at annual volatility sigma, the quarterly win probability is
roughly Phi(S/2), so S = 0.9 implies about 82%. With ~28 quarters, observing 23
wins rejects p = 0.5 at p < 1e-5 -- which converts "suggestive" into "the median
quarter is positive, repeatedly". It does NOT convert it into "the Sharpe is 0.9":
a sign test ignores magnitude, and a strategy that wins small and loses big passes
it. Both statements belong in the answer.

Benchmark-relative is the decision-relevant variant: how often does this beat the
equal-weight pool it selects from?

Usage::

    python scripts/run_quarter_consistency.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:  # the Windows console is GBK; keep the script printable there
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PHASES = tuple(range(20))
TARGET_VOL = 0.12


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


def quarterly(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).groupby(returns.index.to_period("Q")).prod() - 1.0


def sign_test(series: pd.Series, label: str) -> dict:
    clean = series.dropna()
    wins = int((clean > 0).sum())
    total = int(len(clean))
    test = binomtest(wins, total, 0.5, alternative="greater")
    return {
        "series": label,
        "n": total,
        "wins": wins,
        "win_rate": wins / total if total else float("nan"),
        "p_value": float(test.pvalue),
        "median": float(clean.median()),
        "worst": float(clean.min()),
        "best": float(clean.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "quarter_consistency"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    print("building the deployable blend...", flush=True)
    weights = blended_weights(aligned)
    result = Backtester(BacktestConfig()).run(aligned, weights)
    strategy_returns = result.returns

    close = pd.DataFrame({s: f["close"] for s, f in aligned.items()})
    pool_returns = close.pct_change(fill_method=None).mean(axis=1).reindex(strategy_returns.index)
    excess = strategy_returns - pool_returns.fillna(0.0)

    strategy_q = quarterly(strategy_returns)
    pool_q = quarterly(pool_returns.dropna())
    excess_q = quarterly(excess)

    rows = [sign_test(strategy_q, "策略绝对收益 > 0"),
            sign_test(excess_q, "策略 - 等权全池 > 0"),
            sign_test(pool_q, "等权全池 > 0（参照）")]

    sharpe = float(result.metrics.sharpe)
    implied = float(norm.cdf(sharpe / 2.0))

    # Power of the sign test at this sample size, against the alternative that
    # the true quarter win rate is what the Sharpe implies.
    from scipy.stats import binom

    n_quarters = rows[0]["n"]
    critical = next(
        (k for k in range(n_quarters + 1) if binom.sf(k - 1, n_quarters, 0.5) <= 0.05),
        n_quarters + 1,
    )
    power = float(binom.sf(critical - 1, n_quarters, implied))
    quarters_for_80 = next(
        (n for n in range(4, 400)
         if binom.sf(
             next((k for k in range(n + 1) if binom.sf(k - 1, n, 0.5) <= 0.05), n + 1) - 1,
             n, implied) >= 0.80),
        None,
    )

    summary = {
        "phases": len(PHASES),
        "target_vol": TARGET_VOL,
        "start": str(strategy_returns.index.min().date()),
        "end": str(strategy_returns.index.max().date()),
        "sharpe": sharpe,
        "implied_quarter_win_rate": implied,
        "sign_test_critical_wins": int(critical),
        "sign_test_power": power,
        "quarters_for_80pct_power": quarters_for_80,
        "tests": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    pd.DataFrame({"strategy": strategy_q, "pool": pool_q, "excess": excess_q}).to_csv(
        out / "quarters.csv", encoding="utf-8-sig"
    )

    lines = ["# 季度一致性：符号检验", ""]
    lines.append("部署口径（混合 + %.0f%%），%s .. %s。" % (TARGET_VOL * 100, summary["start"], summary["end"]))
    lines.append("")
    lines.append("| 序列 | 季度数 | 胜 | 胜率 | 二项检验 p | 中位 | 最差 | 最好 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append("| %s | %d | %d | %.0f%% | %.2g | %+.2f%% | %+.2f%% | %+.2f%% |"
                     % (row["series"], row["n"], row["wins"], 100 * row["win_rate"], row["p_value"],
                        row["median"] * 100, row["worst"] * 100, row["best"] * 100))
    lines.append("")
    lines.append("按 Sharpe %.3f 推算的季度胜率理论值：**%.0f%%**（Φ(S/2)）。"
                 % (sharpe, implied * 100))
    lines.append("")
    lines.append("符号检验在这个样本量下的检出力：需要 **%d 胜**才达到 5%% 显著，而真胜率 %.0f%% 时"
                 % (critical, implied * 100))
    lines.append("达标概率只有 **%.0f%%**；要做到 80%% 检出力需要约 **%s 个季度**。"
                 % (power * 100, quarters_for_80))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + pd.DataFrame(rows)[["series", "n", "wins", "win_rate", "p_value"]].to_string(index=False))
    print("implied quarter win rate from Sharpe %.3f: %.0f%%" % (sharpe, implied * 100))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
