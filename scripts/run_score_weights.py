"""Score weights: is the 20% trend term buying anything?

The horizon study (23.66) found the trend *component* has a negative IC at
every horizon (-0.03 to -0.09) while the low-vol component rises to +0.23:
among names already above their 120-day average, the ones furthest above it
tend to give some back. The published score is ``0.8 * z(-vol) + 0.2 * z(trend)``,
so that term is an IC drag.

But the trend term does two jobs: it scores, and it also selects (the filter
keeps only names above the trend line). This script separates the two by
re-running the same strategy with four weight splits and the filter left on,
sliced into the three windows, with a paired bootstrap against the default.

Usage::

    python scripts/run_score_weights.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester, slice_result  # noqa: E402
from quant_system.bootstrap import paired_bootstrap_diff  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-05"),
}

BASE = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)
VARIANTS = {
    "0.8/0.2 (default)": (0.8, 0.2),
    "1.0/0.0 (pure low-vol)": (1.0, 0.0),
    "0.6/0.4": (0.6, 0.4),
    "0.5/0.5": (0.5, 0.5),
}
DEFAULT_LABEL = "0.8/0.2 (default)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "score_weights"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _ = align_panel(panel)
    print(f"panel: {len(aligned)} symbols", flush=True)

    rows: list[dict] = []
    returns: dict[tuple, pd.Series] = {}
    for label, (vol_w, trend_w) in VARIANTS.items():
        print(f"running {label} ...", flush=True)
        strategy = LowVolCoreStrategy(
            rebalance_reset="year", vol_weight=vol_w, trend_weight=trend_w, **BASE
        )
        weights = build_portfolio(
            strategy.generate(aligned),
            aligned,
            risk=RiskConfig(),
            use_regime=False,
            use_vol_target=True,
        )
        result = Backtester(BacktestConfig()).run(aligned, weights)
        for window, (start, end) in WINDOWS.items():
            sliced = slice_result(result, start=start, end=end)
            m = sliced.metrics
            returns[(label, window)] = sliced.returns
            rows.append(
                {
                    "variant": label,
                    "vol_weight": vol_w,
                    "trend_weight": trend_w,
                    "window": window,
                    "total_return": float(m.total_return),
                    "cagr": float(m.cagr),
                    "annual_volatility": float(m.annual_volatility),
                    "sharpe": float(m.sharpe),
                    "max_drawdown": float(m.max_drawdown),
                    "calmar": float(m.calmar),
                    "turnover": float(m.turnover),
                    "n_trades": int(len(sliced.trades)),
                }
            )

    boot: list[dict] = []
    for label in VARIANTS:
        if label == DEFAULT_LABEL:
            continue
        for window in WINDOWS:
            alt = returns[(label, window)]
            base = returns[(DEFAULT_LABEL, window)]
            common = alt.index.intersection(base.index)
            res = paired_bootstrap_diff(
                alt.loc[common], base.loc[common], n_boot=args.n_boot, seed=args.seed
            )
            boot.append(
                {
                    "variant": label,
                    "window": window,
                    "delta_sharpe": float(res.observed),
                    "low": float(res.low),
                    "high": float(res.high),
                    "prob_positive": float(res.prob_positive),
                }
            )
            print(
                f"  {label} {window}: dSharpe {res.observed:+.3f} "
                f"[{res.low:+.3f}, {res.high:+.3f}]",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    boot_df = pd.DataFrame(boot)
    frame.to_csv(out / "panel.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(out / "bootstrap.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps(
            {"windows": WINDOWS, "variants": {k: list(v) for k, v in VARIANTS.items()},
             "n_boot": args.n_boot, "seed": args.seed,
             "panel": rows, "bootstrap": boot},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    lines = ["# 打分权重：趋势项到底有没有用", ""]
    lines.append("| 权重 | 窗口 | 累计收益 | 实现波动 | Sharpe | 最大回撤 | 换手 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, row in frame.iterrows():
        lines.append(
            f"| {row.variant} | {row.window} | {row.total_return:+.2%} | "
            f"{row.annual_volatility:.2%} | {row.sharpe:.3f} | {row.max_drawdown:.2%} | "
            f"{row.turnover:.1f}x |"
        )
    lines.append("")
    lines.append("## 配对自助（变体 − 默认）")
    lines.append("")
    lines.append("| 变体 | 窗口 | ΔSharpe | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---|---:|---|---:|")
    for _, row in boot_df.iterrows():
        lines.append(
            f"| {row.variant} | {row.window} | {row.delta_sharpe:+.3f} | "
            f"[{row.low:+.3f}, {row.high:+.3f}] | {row.prob_positive:.2f} |"
        )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
