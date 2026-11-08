"""Does the edge exist before 2019? A first look on the names that reach 2010.

RESEARCH_LOG 23.129 found the window is held back by the 95% coverage rule, and
23.130 found that only hfq survives a 16-year range. Full coverage is still being
back-filled under provider throttling, but the question can be asked now on the
names that already reach back: run the published specification on 2010-2018 and on
2019-2026 and compare. A narrow universe penalises both halves equally, so the
comparison is about *when* the edge existed, not about absolute Sharpe.

hfq prices are inflated by a cumulative factor, which would make 100-share lots
unaffordable and break the fill model. Each series is therefore rescaled so that
its last close matches the raw close on the same date; returns are unchanged and
price levels become realistic.

Usage::

    python scripts/run_pre2019_check.py
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
from quant_system.config import BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, normalize_symbol  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402

CACHE = ROOT / "data" / "cache_ext"
TARGET_VOL = 0.12
FIELDS = ("open", "high", "low", "close", "volume")
HALVES = {"pre-2019": ("2010-01-01", "2018-12-31"),
          "post-2019": ("2019-01-01", "2026-11-06")}


def load_scaled(cache: Path) -> dict:
    """hfq frames rescaled so the last close matches raw; returns unchanged."""
    panel = {}
    for path in sorted(cache.glob("*_hfq.parquet")):
        symbol = normalize_symbol(path.name.replace("_hfq.parquet", ""))
        try:
            hfq = pd.read_parquet(path)
        except Exception:
            continue
        if not set(FIELDS).issubset(hfq.columns) or len(hfq) < 300:
            continue
        raw_path = cache / (symbol + "_raw.parquet")
        factor = None
        if raw_path.exists():
            try:
                raw = pd.read_parquet(raw_path, columns=["close"])
                common = hfq.index.intersection(raw.index)
                if len(common):
                    last = common[-1]
                    if float(hfq.loc[last, "close"]) > 0:
                        factor = float(raw.loc[last, "close"]) / float(hfq.loc[last, "close"])
            except Exception:
                factor = None
        if factor is None:
            factor = 1.0 / float(hfq["close"].iloc[-1]) * 10.0
        frame = hfq.loc[:, list(FIELDS)].copy()
        for column in ("open", "high", "low", "close"):
            frame[column] = frame[column] * factor
        panel[symbol] = frame
    return panel


@dataclass
class Spec(LowVolCoreStrategy):
    pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "pre2019_check" / "summary.json"))
    parser.add_argument("--phases", type=int, default=20,
                        help="phases to average; 1 = single phase (phase luck not removed)")
    args = parser.parse_args(argv)

    panel = load_scaled(CACHE)
    print("symbols with long hfq history: %d" % len(panel), flush=True)
    if len(panel) < 20:
        print("not enough names yet; download more hfq first")
        return 2

    aligned, _dates = align_panel(panel)

    frames = []
    for offset in range(max(1, args.phases)):
        spec = Spec(top_n=8, vol_window=60, trend_window=120, max_per_sector=2,
                    rebalance_days=20, rebalance_reset="year", rebalance_offset=offset)
        frames.append(build_portfolio(spec.generate(aligned), aligned,
                                      risk=RiskConfig(target_annual_vol=TARGET_VOL),
                                      use_regime=False, use_vol_target=True))
    weights = sum(frames) / len(frames)
    result = Backtester(BacktestConfig()).run(aligned, weights)
    print("phases averaged: %d" % len(frames), flush=True)

    rows = []
    for label, (start, end) in HALVES.items():
        sliced = slice_result(result, start=start, end=end)
        m = sliced.metrics
        rows.append({
            "half": label,
            "start": start,
            "end": end,
            "total_return": float(m.total_return),
            "cagr": float(m.cagr),
            "volatility": float(m.annual_volatility),
            "sharpe": float(m.sharpe),
            "max_drawdown": float(m.max_drawdown),
            "turnover": float(m.turnover),
            "days": int(len(sliced.returns)),
        })

    # Year by year: an "edge" that lives in one bubble year is not an edge.
    by_year = []
    for year in range(2010, 2027):
        sliced = slice_result(result, start="%d-01-01" % year, end="%d-12-31" % year)
        if len(sliced.returns) < 30:
            continue
        m = sliced.metrics
        by_year.append({
            "year": year,
            "total_return": float(m.total_return),
            "sharpe": float(m.sharpe),
            "max_drawdown": float(m.max_drawdown),
            "days": int(len(sliced.returns)),
        })

    # Leave-one-year-out: if the edge dies when one year is removed, it was one year.
    loo = []
    pre = [row["year"] for row in by_year if row["year"] <= 2018]
    for drop in pre:
        # Scope to the pre-2019 window: dropping a year from the whole series would
        # mix in the post-2019 years and cannot answer "is 2010-2018 one year?".
        keep = [r for r in result.returns.index if r.year <= 2018 and r.year != drop]
        subset = result.returns.loc[keep]
        if len(subset) < 500:
            continue
        mean = float(subset.mean())
        std = float(subset.std())
        loo.append({
            "dropped": drop,
            "sharpe": (mean / std * (252 ** 0.5)) if std else None,
            "total_return": float((1.0 + subset).prod() - 1.0),
        })

    # The two years that dominate: what is left without both of them?
    keep = [r for r in result.returns.index if r.year <= 2018 and r.year not in (2014, 2017)]
    combined = None
    if len(keep) > 500:
        subset = result.returns.loc[keep]
        std = float(subset.std())
        combined = {
            "dropped": [2014, 2017],
            "sharpe": (float(subset.mean()) / std * (252 ** 0.5)) if std else None,
            "total_return": float((1.0 + subset).prod() - 1.0),
            "days": int(len(subset)),
        }

    payload = {"symbols": len(panel), "phases": len(frames), "target_vol": TARGET_VOL,
               "by_year": by_year, "leave_one_out_pre2019": loo,
               "drop_2014_2017": combined,
               "spec": {"top_n": 8, "vol_window": 60, "trend_window": 120,
                        "max_per_sector": 2, "rebalance_days": 20},
               "rows": rows,
               "note": "hfq 按最后一日与 raw 的比例还原价格水平；收益率不变。"
                       "子集口径（非 128 只），用于比较两个时期，不与生产数字对照。"}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Daily returns are needed downstream (style regression), so persist them
    # rather than making the next script re-run the whole blend.
    (out.parent / "daily_returns.csv").write_text(
        "date,return\n" + "\n".join(
            "%s,%.10f" % (d.date(), v) for d, v in result.returns.items()
        ) + "\n",
        encoding="utf-8",
    )
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")

    for row in by_year:
        print("    %d  %+7.1f%%  Sharpe %5.2f" % (row["year"], row["total_return"] * 100,
                                                  row["sharpe"]))
    for row in loo:
        print("    drop %d -> Sharpe %.3f  cumulative %+.1f%%"
              % (row["dropped"], row["sharpe"] or 0, row["total_return"] * 100))
    if combined:
        print("    drop 2014+2017 -> Sharpe %.3f  cumulative %+.1f%%  (%d days)"
              % (combined["sharpe"] or 0, combined["total_return"] * 100, combined["days"]))
    for row in rows:
        print("  %-10s %s..%s  %+.1f%%  Sharpe %.3f  MDD %.1f%%  %d days"
              % (row["half"], row["start"], row["end"], row["total_return"] * 100,
                 row["sharpe"], row["max_drawdown"] * 100, row["days"]))
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
