"""Proximity to the 52-week high as a portfolio overlay (RESEARCH_LOG 23.77).

The cohort decomposition left exactly one candidate standing: `high_52w`
is the only library factor whose signal does not live in the delisting cohort (at
60 days its delisted/listed IC ratio is 0.4, the only ratio below 1). Its IC is
small -- +0.0150 in the listed cohort at 20 days -- so the honest question is not
"is it significant" but "does adding it to the book help after costs".

Same engine, same risk overlay, same three independent windows as the momentum
and illiquidity overlays, so the result is comparable to those.

Usage::

    python scripts/run_52w_overlay.py
    python scripts/run_52w_overlay.py --weights 0.1 0.2
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

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.bootstrap import paired_bootstrap_diff  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.factors import price_frame  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolPlusStrategy  # noqa: E402

WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-05"),
}

BASE = dict(
    top_n=8,
    vol_window=60,
    trend_window=120,
    illiquidity_window=20,
    vol_weight=0.8,
    trend_weight=0.2,
    illiquidity_weight=0.0,
    momentum_weight=0.0,
    blend="zscore",
)


@dataclass
class High52wOverlay(LowVolPlusStrategy):
    """The published score plus a term for distance below the 52-week high."""

    high_52w_window: int = 252
    high_52w_weight: float = 0.0

    def score(self, panel):
        scores, eligible = super().score(panel)
        if not self.high_52w_weight:
            return scores, eligible
        close = price_frame(panel, "close")
        distance = close / close.rolling(self.high_52w_window).max() - 1.0
        return scores + self.high_52w_weight * self._standardise(distance), eligible


def build(panel, weight: float):
    strategy = High52wOverlay(rebalance_reset="year", high_52w_weight=weight, **BASE)
    weights = build_portfolio(
        strategy.generate(panel),
        panel,
        risk=RiskConfig(),
        use_regime=False,
        use_vol_target=True,
    )
    return Backtester(BacktestConfig()).run(panel, weights), weights


def summarise(result, weights, label: str, window: str) -> dict:
    m = result.metrics
    gross = weights.sum(axis=1) if len(weights) else pd.Series(dtype=float)
    return {
        "window": window,
        "label": label,
        "total_return": float(m.total_return),
        "cagr": float(m.cagr),
        "annual_volatility": float(m.annual_volatility),
        "sharpe": float(m.sharpe),
        "max_drawdown": float(m.max_drawdown),
        "calmar": float(m.calmar),
        "avg_gross": float(gross.mean()) if len(gross) else float("nan"),
        "n_days": int(len(result.returns)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "fiftytwo_overlay"))
    parser.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args(argv)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print("panel: %d symbols" % len(aligned), flush=True)

    runs = {"baseline": build(aligned, 0.0)}
    for weight in args.weights:
        label = "high52w=%g" % weight
        print("running %s ..." % label, flush=True)
        runs[label] = build(aligned, weight)

    rows: list[dict] = []
    paired: list[dict] = []
    for window, (start, end) in WINDOWS.items():
        sliced = {label: slice_result(res, start=start, end=end) for label, (res, _w) in runs.items()}
        for label, res in sliced.items():
            day_weights = runs[label][1].reindex(res.equity.index).fillna(0.0)
            rows.append(summarise(res, day_weights, label, window))
        base = sliced["baseline"].returns
        for label in runs:
            if label == "baseline":
                continue
            diff = paired_bootstrap_diff(
                sliced[label].returns, base, n_boot=args.n_boot, seed=args.seed
            )
            paired.append({"window": window, "variant": label, **diff.to_dict()})

    frame = pd.DataFrame(rows)
    pairs = pd.DataFrame(paired)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "summary.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(output / "paired_sharpe.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "windows": {k: list(v) for k, v in WINDOWS.items()},
                "base": BASE,
                "weights": args.weights,
                "rows": rows,
                "paired_sharpe_vs_baseline": paired,
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# 52 周高点距离叠加：三窗口组合层复检", ""]
    lines.append("基准 = `low_vol_core`（0.8 低波 / 0.2 趋势）。§23.77 的分解里 high_52w 是")
    lines.append("唯一不被退市通道解释掉的候选，所以值得一次真实回测而不是再看 IC。")
    lines.append("")
    lines.append("| 窗口 | 变体 | 累计收益 | CAGR | 波动 | Sharpe | 最大回撤 | 卡玛 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {window} | {label} | {total_return:.2%} | {cagr:.2%} | {annual_volatility:.2%} | "
            "{sharpe:.3f} | {max_drawdown:.2%} | {calmar:.2f} |".format(**row)
        )
    lines.append("")
    lines.append("## 配对 ΔSharpe（变体 − 基准）")
    lines.append("")
    lines.append("| 窗口 | 变体 | ΔSharpe | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---|---:|---|---:|")
    for row in paired:
        low = row.get("low")
        high = row.get("high")
        lines.append(
            "| %s | %s | %+.3f | [%+.3f, %+.3f] | %.2f |"
            % (
                row["window"],
                row["variant"],
                row.get("diff", float("nan")),
                low if low is not None else float("nan"),
                high if high is not None else float("nan"),
                row.get("prob_greater", float("nan")),
            )
        )
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + frame[["window", "label", "total_return", "sharpe", "max_drawdown"]].to_string(index=False))
    print("\n" + pairs.to_string(index=False))
    print("\nreport: %s" % output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
