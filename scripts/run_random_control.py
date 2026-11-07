"""Does the ranking add anything over random selection with the same constraints?

Every downstream conclusion rests on "the low-vol score picks better names". 23.77
already showed the score's in-pool IC is only +0.031 once the delisting channel is
removed, so the question is fair: if you drew eight names at random from the same
eligible set, honouring the same sector cap, rebalance cadence and volatility
target, how much of the result would survive?

The control keeps everything the strategy has except the ranking: the random
score replaces the low-vol score, so eligibility (traded, above the trend line),
the sector cap, the sizing and the costs are byte-identical.

Usage::

    python scripts/run_random_control.py --draws 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe, price_frame  # noqa: E402
from quant_system.indicators import sma  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy, _traded  # noqa: E402

TARGET_VOL = 0.12
BASE = dict(
    top_n=8,
    vol_window=60,
    trend_window=120,
    max_per_sector=2,
    rebalance_days=20,
    rebalance_reset="year",
)


@dataclass
class RandomSelection(LowVolCoreStrategy):
    """Identical eligibility and constraints, random ranking."""

    seed: int = 0

    def score(self, panel):
        close = price_frame(panel, "close")
        traded = _traded(panel, close)
        trend_ok = traded & (close > sma(close, self.trend_window))
        rng = np.random.default_rng(self.seed)
        scores = pd.DataFrame(
            rng.random(close.shape), index=close.index, columns=close.columns
        )
        return scores, trend_ok


def run(strategy, aligned):
    weights = build_portfolio(
        strategy.generate(aligned),
        aligned,
        risk=RiskConfig(target_annual_vol=TARGET_VOL),
        use_regime=False,
        use_vol_target=True,
    )
    metrics = Backtester(BacktestConfig()).run(aligned, weights).metrics
    return {
        "sharpe": float(metrics.sharpe),
        "cagr": float(metrics.cagr),
        "total_return": float(metrics.total_return),
        "max_drawdown": float(metrics.max_drawdown),
        "turnover": float(metrics.turnover),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "random_control"))
    parser.add_argument("--draws", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    print("running the published strategy (single phase, %.0f%% target)..." % (TARGET_VOL * 100), flush=True)
    real = run(LowVolCoreStrategy(**BASE), aligned)
    print("  real: %+.2f%%  sharpe %.3f" % (real["total_return"] * 100, real["sharpe"]), flush=True)

    rows = []
    started = time.time()
    for draw in range(args.draws):
        info = run(RandomSelection(seed=args.seed + draw, **BASE), aligned)
        rows.append({"draw": draw, **info})
        if draw % 10 == 0:
            print("  draw=%d sharpe=%.3f (%.0fs)" % (draw, info["sharpe"], time.time() - started), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "draws.csv", index=False, encoding="utf-8-sig")

    sharpes = frame["sharpe"]
    percentile = float((sharpes < real["sharpe"]).mean())
    summary = {
        "target_vol": TARGET_VOL,
        "draws": args.draws,
        "seed": args.seed,
        "real": real,
        "random": {
            "sharpe_median": float(sharpes.median()),
            "sharpe_p5": float(sharpes.quantile(0.05)),
            "sharpe_p95": float(sharpes.quantile(0.95)),
            "sharpe_min": float(sharpes.min()),
            "sharpe_max": float(sharpes.max()),
            "cagr_median": float(frame["cagr"].median()),
            "max_drawdown_median": float(frame["max_drawdown"].median()),
            "turnover_median": float(frame["turnover"].median()),
        },
        "real_percentile_among_random": percentile,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 随机选股对照（同池、同约束、同风控）", ""]
    lines.append("把低波分换成随机分，其余（可交易、趋势过滤、每行业 2 只、8 只、20 日、12% 目标、成本）全部不变。")
    lines.append("%d 次抽样，单相位（offset 0），全期。" % args.draws)
    lines.append("")
    lines.append("| 口径 | 累计收益 | CAGR | Sharpe | 最大回撤 | 换手 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append("| **低波分（发布策略）** | %+.2f%% | %+.2f%% | **%.3f** | %.2f%% | %.1f |"
                 % (real["total_return"] * 100, real["cagr"] * 100, real["sharpe"],
                    real["max_drawdown"] * 100, real["turnover"]))
    rnd = summary["random"]
    lines.append("| 随机选中位 | — | %+.2f%% | %.3f | %.2f%% | %.1f |"
                 % (rnd["cagr_median"] * 100, rnd["sharpe_median"],
                    rnd["max_drawdown_median"] * 100, rnd["turnover_median"]))
    lines.append("| 随机 p5 / p95 | — | — | %.3f / %.3f | — | — |"
                 % (rnd["sharpe_p5"], rnd["sharpe_p95"]))
    lines.append("")
    lines.append("发布策略在 %d 次随机抽样中排在 **%.0f 分位**。" % (args.draws, percentile * 100))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\nreal sharpe %.3f | random median %.3f | p5 %.3f p95 %.3f | real percentile %.0f%%"
          % (real["sharpe"], rnd["sharpe_median"], rnd["sharpe_p5"], rnd["sharpe_p95"], percentile * 100))
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
