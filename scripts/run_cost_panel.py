"""Cost sensitivity across the three windows: does the edge survive 30bp/side?

Published numbers use 25bp commission/side, 50bp stamp duty on sells, 5bp
slippage and zero market impact. A-share retail costs are usually quoted higher
(commission 2.5-3bp is institutional; retail is nearer 2.5bp *with* a 5 yuan
floor, but slippage on a 8-name basket can be 10-30bp). This script re-runs the
same signal with four cost scenarios and slices every run into the three
windows, so "the edge is a cost artifact" can be rejected or confirmed instead
of assumed.

Scenarios:

    published   25bp commission, 5bp slippage, no impact (current default)
    doubled     50bp commission, 10bp slippage
    impact40    25bp commission, 5bp slippage, 40bp sqrt-impact coefficient
    harsh       50bp commission, 20bp slippage, 60bp sqrt-impact coefficient

Usage::

    python scripts/run_cost_panel.py
    python scripts/run_cost_panel.py --scenarios published doubled impact40
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
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, CostConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


WINDOWS = {
    "2019-2021": ("2019-07-01", "2021-12-31"),
    "2022-2023": ("2022-01-01", "2023-12-31"),
    "2024-2026": ("2024-01-01", "2026-11-05"),
}

PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)

SCENARIOS = {
    "published": dict(),
    "doubled": dict(commission_rate=0.0005, slippage_bps=10.0),
    "impact40": dict(impact_coefficient_bps=40.0),
    "harsh": dict(commission_rate=0.0005, slippage_bps=20.0, impact_coefficient_bps=60.0),
}


def summarise(result, label: str, window: str) -> dict:
    m = result.metrics
    return {
        "window": window,
        "scenario": label,
        "total_return": float(m.total_return),
        "cagr": float(m.cagr),
        "annual_volatility": float(m.annual_volatility),
        "sharpe": float(m.sharpe),
        "max_drawdown": float(m.max_drawdown),
        "calmar": float(m.calmar),
        "turnover": float(m.turnover),
        "n_days": int(len(result.returns)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "cost_panel"))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=sorted(SCENARIOS))
    parser.add_argument("--target-vol", type=float, default=None,
                        help="override RiskConfig.target_annual_vol (e.g. 0.12); "
                             "used to re-check the cost ranking at a lower risk budget")
    parser.add_argument("--n-boot", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)
    print(f"panel: {len(aligned)} symbols, {len(next(iter(aligned.values())))} bars", flush=True)

    strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
    risk = RiskConfig() if args.target_vol is None else RiskConfig(target_annual_vol=args.target_vol)
    weights = build_portfolio(
        strategy.generate(aligned),
        aligned,
        risk=risk,
        use_regime=False,
        use_vol_target=True,
    )

    runs = {}
    for name in args.scenarios:
        config = BacktestConfig(costs=CostConfig(**SCENARIOS[name]))
        print(f"running {name} ...", flush=True)
        runs[name] = Backtester(config).run(aligned, weights)

    rows: list[dict] = []
    paired: list[dict] = []
    for window, (start, end) in WINDOWS.items():
        sliced = {name: slice_result(res, start=start, end=end) for name, res in runs.items()}
        for name, res in sliced.items():
            rows.append(summarise(res, name, window))
        if "published" in sliced:
            base = sliced["published"].returns
            for name in sliced:
                if name == "published":
                    continue
                diff = paired_bootstrap_diff(sliced[name].returns, base, n_boot=args.n_boot, seed=args.seed)
                paired.append({"window": window, "scenario": name, **diff.to_dict()})

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
                "params": PARAMS,
                "scenarios": {k: SCENARIOS[k] for k in args.scenarios},
                "rows": rows,
                "paired_sharpe_vs_published": paired,
            },
            ensure_ascii=False,
            indent=2,
            default=float,
        ),
        encoding="utf-8",
    )

    lines = ["# 成本敏感度：三段窗口", ""]
    lines.append("同一组信号与权重，只改 `CostConfig`；每段都是切片而非独立回测。")
    lines.append("")
    lines.append("| 窗口 | 情景 | 累计收益 | CAGR | Sharpe | 最大回撤 | 卡玛 | 换手 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {window} | {scenario} | {total_return:.2%} | {cagr:.2%} | {sharpe:.3f} | "
            "{max_drawdown:.2%} | {calmar:.2f} | {turnover:.2f} |".format(**row)
        )
    lines.append("")
    lines.append("## 配对 ΔSharpe（情景 − published）")
    lines.append("")
    lines.append("| 窗口 | 情景 | ΔSharpe | 95% 区间 | P(Δ>0) |")
    lines.append("|---|---|---:|---|---:|")
    for row in paired:
        lines.append(
            "| {window} | {scenario} | {observed:+.3f} | [{low:+.3f}, {high:+.3f}] | "
            "{prob_positive:.3f} |".format(**row)
        )
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print(frame.pivot_table(index="scenario", columns="window", values="sharpe").round(3).to_string())
    print()
    print(frame.pivot_table(index="scenario", columns="window", values="total_return").mul(100).round(1).to_string())
    print()
    print(pairs.to_string(index=False))
    try:
        shown = output.resolve().relative_to(ROOT)
    except ValueError:
        shown = output
    print(f"report: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
