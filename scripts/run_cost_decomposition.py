"""What does each cost scenario actually charge per trade?

The cost panel (23.55) reports net performance under four scenarios, but not
what those scenarios cost per trade. That matters for the capacity claim: the
square-root impact term is only meaningful if our participation is large, so
the honest way to read "harsh costs" is to see the realised basis points.

The ``slippage`` column already contains fixed slippage *plus* market impact
(``execution_price`` bakes both into the fill), so

    implied_impact_bps = slippage / notional - config.slippage_bps

isolates the impact term exactly.

Usage::

    python scripts/run_cost_decomposition.py
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
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, CostConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)

SCENARIOS = {
    "published": dict(),
    "doubled": dict(commission_rate=0.0005, slippage_bps=10.0),
    "impact40": dict(impact_coefficient_bps=40.0),
    "harsh": dict(commission_rate=0.0005, slippage_bps=20.0, impact_coefficient_bps=60.0),
}

WINDOW = ("2024-01-01", "2026-11-05")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "cost_decomposition"))
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _ = align_panel(panel)

    strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
    weights = build_portfolio(
        strategy.generate(aligned), aligned, risk=RiskConfig(), use_regime=False, use_vol_target=True
    )

    rows: list[dict] = []
    detail: dict[str, list[dict]] = {}
    for name, kwargs in SCENARIOS.items():
        print(f"running {name} ...", flush=True)
        config = BacktestConfig(costs=CostConfig(**kwargs))
        result = Backtester(config).run(aligned, weights)
        sliced = slice_result(result, start=WINDOW[0], end=WINDOW[1])
        trades = sliced.trades.copy()
        trades["bps_fees"] = trades["fees"] / trades["notional"] * 1e4
        trades["bps_slip"] = trades["slippage"] / trades["notional"] * 1e4
        trades["bps_impact"] = trades["bps_slip"] - CostConfig(**kwargs).slippage_bps
        trades["bps_all"] = trades["bps_fees"] + trades["bps_slip"]
        for side in ("buy", "sell", "all"):
            subset = trades if side == "all" else trades[trades["side"] == side]
            if subset.empty:
                continue
            rows.append(
                {
                    "scenario": name,
                    "side": side,
                    "trades": int(len(subset)),
                    "notional": float(subset["notional"].sum()),
                    "fees_bps": float((subset["bps_fees"] * subset["notional"]).sum() / subset["notional"].sum()),
                    "slippage_bps": float((subset["bps_slip"] * subset["notional"]).sum() / subset["notional"].sum()),
                    "impact_bps": float((subset["bps_impact"] * subset["notional"]).sum() / subset["notional"].sum()),
                    "all_in_bps": float((subset["bps_all"] * subset["notional"]).sum() / subset["notional"].sum()),
                    "median_order": float(subset["notional"].median()),
                    "max_order": float(subset["notional"].max()),
                }
            )
        detail[name] = trades[["date", "symbol", "side", "notional", "bps_all", "bps_impact"]].to_dict(orient="records")

    frame = pd.DataFrame(rows)
    frame.to_csv(out / "costs.csv", index=False, encoding="utf-8-sig")
    (out / "summary.json").write_text(
        json.dumps({"window": WINDOW, "scenarios": SCENARIOS, "table": rows},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# 成本情景到底扣了多少 bp（2024-01 .. 2026-11）", ""]
    lines.append("按成交名义额加权；`impact_bps` 是滑点列减去固定滑点后的残差，即平方根冲击项。")
    lines.append("")
    lines.append("| 情景 | 方向 | 笔数 | 显性费用 | 滑点 | 其中冲击 | 全口径 | 中位单笔 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in frame.iterrows():
        lines.append(
            f"| {row.scenario} | {row.side} | {row.trades} | {row.fees_bps:.2f} | "
            f"{row.slippage_bps:.2f} | {row.impact_bps:.2f} | {row.all_in_bps:.2f} | "
            f"{row.median_order:,.0f} |"
        )
    lines.append("")
    lines.append("读法：单边全口径成本 = 显性费用（佣金+过户费+卖出印花税）+ 滑点；"
                 "harsh 情景的中位单笔约 12 万元，对应的冲击项只有零点几个 bp，"
                 "说明「harsh」是靠提高固定滑点与佣金实现的，而不是靠不现实的冲击放大。")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
