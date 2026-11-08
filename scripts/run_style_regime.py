"""Is the strategy's return just the low-volatility style factor?

RESEARCH_LOG 23.134 found the pre-2019 edge collapses to Sharpe 0.397 without 2014
and 2017, and noted both are large-cap/low-vol leadership years. That was an
observation. This measures it: build the style factor directly (lowest-volatility
tercile minus highest-volatility tercile, equal weighted, monthly rebalanced) and
ask whether the strategy's good years are the years the style paid.

If the correlation is high, then "how much will this earn" is a question about the
style, not about the stock picker -- which changes what evidence would settle it.

Usage::

    python scripts/run_style_regime.py
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

from quant_system.data import align_panel, normalize_symbol  # noqa: E402

CACHE = ROOT / "data" / "cache_ext"
FIELDS = ("open", "high", "low", "close", "volume")


def load_scaled(cache: Path) -> dict:
    panel = {}
    for path in sorted(cache.glob("*_hfq.parquet")):
        symbol = normalize_symbol(path.name.replace("_hfq.parquet", ""))
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if not set(FIELDS).issubset(frame.columns) or len(frame) < 300:
            continue
        raw_path = cache / (symbol + "_raw.parquet")
        factor = None
        if raw_path.exists():
            try:
                raw = pd.read_parquet(raw_path, columns=["close"])
                common = hfq_index = frame.index.intersection(raw.index)
                if len(common):
                    last = common[-1]
                    if float(frame.loc[last, "close"]) > 0:
                        factor = float(raw.loc[last, "close"]) / float(frame.loc[last, "close"])
            except Exception:
                factor = None
        if factor is None:
            factor = 10.0 / float(frame["close"].iloc[-1])
        scaled = frame.loc[:, list(FIELDS)].copy()
        for column in ("open", "high", "low", "close"):
            scaled[column] = scaled[column] * factor
        panel[symbol] = scaled
    return panel


def style_spread(close: pd.DataFrame, vol_window: int = 60) -> pd.Series:
    """Monthly return of (low-vol tercile) minus (high-vol tercile), equal weighted."""
    returns = close.pct_change(fill_method=None)
    vol = returns.rolling(vol_window).std()
    month_ends = close.groupby([close.index.year, close.index.month]).tail(1).index
    rows = []
    for i in range(len(month_ends) - 1):
        start, end = month_ends[i], month_ends[i + 1]
        ranking = vol.loc[start].dropna()
        if len(ranking) < 15:
            continue
        low = ranking.nsmallest(max(3, len(ranking) // 3)).index
        high = ranking.nlargest(max(3, len(ranking) // 3)).index
        window = returns.loc[(returns.index > start) & (returns.index <= end)]
        if window.empty:
            continue
        rows.append({
            "date": end,
            "style": float(window[low].mean().mean() - window[high].mean().mean()),
        })
    if not rows:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(rows).set_index("date")["style"]
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "style_regime" / "summary.json"))
    args = parser.parse_args(argv)

    panel = load_scaled(CACHE)
    print("symbols: %d" % len(panel), flush=True)
    if len(panel) < 30:
        print("not enough names")
        return 2
    aligned, _dates = align_panel(panel)
    close = pd.DataFrame({s: f["close"] for s, f in aligned.items()}).sort_index()

    style = style_spread(close)
    if style.empty:
        print("could not build the style factor")
        return 2

    check_path = ROOT / "reports" / "pre2019_check" / "summary.json"
    strategy_years = {}
    if check_path.exists():
        payload = json.loads(check_path.read_text(encoding="utf-8"))
        for row in payload.get("by_year", []):
            strategy_years[int(row["year"])] = row["total_return"]

    yearly = style.groupby(style.index.year).apply(lambda s: float((1.0 + s).prod() - 1.0))
    rows = []
    for year, spread in yearly.items():
        rows.append({
            "year": int(year),
            "style_return": float(spread),
            "strategy_return": strategy_years.get(int(year)),
        })

    paired = [(r["style_return"], r["strategy_return"]) for r in rows
              if r["strategy_return"] is not None]
    correlation = None
    if len(paired) >= 6:
        correlation = float(np.corrcoef([p[0] for p in paired], [p[1] for p in paired])[0, 1])

    ordered = sorted(paired, key=lambda p: p[0])
    half = len(ordered) // 2
    bottom = ordered[:half]
    top = ordered[half:]

    def stats(bucket):
        if not bucket:
            return None
        strat = [p[1] for p in bucket]
        return {
            "years": len(bucket),
            "mean_strategy_return": float(np.mean(strat)),
            "median_strategy_return": float(np.median(strat)),
            "share_positive": float(np.mean([1.0 if s > 0 else 0.0 for s in strat])),
        }

    # The vol-tercile factor above cannot see size: every name in this panel is a
    # large cap. Size/style rotation needs indices, so fetch them from the cache.
    indices = {}
    for name in ("sh000300", "sh000016", "sh000905", "sz399006"):
        for directory in (CACHE, ROOT / "data" / "cache"):
            path = directory / (name + "_raw.parquet")
            if not path.exists():
                continue
            try:
                frame = pd.read_parquet(path, columns=["close"])
            except Exception:
                continue
            indices[name] = frame["close"].dropna()
            break

    # Prefer SSE50 as the large leg: the production cache's CSI300 only starts in
    # 2019, which silently cut the size regression to 90 months.
    large_key = "sh000016" if "sh000016" in indices else ("sh000300" if "sh000300" in indices else None)

    size_rows = []
    if "sh000905" in indices and large_key:
        small = indices["sh000905"]
        large = indices[large_key]
        small_y = small.groupby(small.index.year).apply(lambda s: float(s.iloc[-1] / s.iloc[0] - 1.0))
        large_y = large.groupby(large.index.year).apply(lambda s: float(s.iloc[-1] / s.iloc[0] - 1.0))
        for year in sorted(set(small_y.index) & set(large_y.index)):
            size_rows.append({
                "year": int(year),
                "small": float(small_y.loc[year]),
                "large": float(large_y.loc[year]),
                "size_spread": float(small_y.loc[year] - large_y.loc[year]),
                "strategy_return": strategy_years.get(int(year)),
            })

    size_paired = [(r["size_spread"], r["strategy_return"]) for r in size_rows
                   if r["strategy_return"] is not None]
    size_corr = None
    if len(size_paired) >= 6:
        size_corr = float(np.corrcoef([p[0] for p in size_paired],
                                      [p[1] for p in size_paired])[0, 1])

    size_ordered = sorted(size_paired)
    size_half = len(size_ordered) // 2

    # Annual correlations (n=17) cannot resolve anything: |r| must exceed ~0.48 to
    # reach 5% significance. Regress monthly instead (n ~ 190).
    monthly = {}
    strat_path = ROOT / "reports" / "pre2019_check" / "daily_returns.csv"
    if strat_path.exists():
        strat = pd.read_csv(strat_path, parse_dates=["date"]).set_index("date")["return"]
        strat_m = (1.0 + strat).groupby([strat.index.year, strat.index.month]).prod() - 1.0
        style_m = (1.0 + style).groupby([style.index.year, style.index.month]).prod() - 1.0
        shared = strat_m.index.intersection(style_m.index)
        if len(shared) >= 24:
            a = style_m.loc[shared].to_numpy()
            b = strat_m.loc[shared].to_numpy()
            r = float(np.corrcoef(a, b)[0, 1])
            n = len(shared)
            t = r * np.sqrt((n - 2) / max(1e-12, 1 - r * r))
            monthly["style_vs_strategy"] = {"n": n, "r": r, "t": float(t)}

        if "sh000905" in indices and large_key:
            small = indices["sh000905"].pct_change(fill_method=None).dropna()
            large = indices[large_key].pct_change(fill_method=None).dropna()
            common = small.index.intersection(large.index)
            spread_d = (small.loc[common] - large.loc[common])
            spread_m = (1.0 + spread_d).groupby(
                [spread_d.index.year, spread_d.index.month]).prod() - 1.0
            shared2 = strat_m.index.intersection(spread_m.index)
            if len(shared2) >= 24:
                a2 = spread_m.loc[shared2].to_numpy()
                b2 = strat_m.loc[shared2].to_numpy()
                r2 = float(np.corrcoef(a2, b2)[0, 1])
                n2 = len(shared2)
                t2 = r2 * np.sqrt((n2 - 2) / max(1e-12, 1 - r2 * r2))
                monthly["size_vs_strategy"] = {"n": n2, "r": r2, "t": float(t2)}

    payload = {
        "symbols": len(panel),
        "months": int(len(style)),
        "monthly": monthly,
        "size_large_leg": large_key,
        "size_yearly": size_rows,
        "size_correlation": size_corr,
        "small_cap_leads": stats(size_ordered[size_half:]),
        "large_cap_leads": stats(size_ordered[:size_half]),
        "yearly": rows,
        "correlation": correlation,
        "style_top_half": stats(top),
        "style_bottom_half": stats(bottom),
        "note": "风格因子 = 低波三分位 − 高波三分位（等权、月度调仓）；策略收益取自 "
                "reports/pre2019_check 的逐年表。",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")

    print("years: %d | monthly obs: %d | correlation(style, strategy): %s"
          % (len(rows), len(style),
             "n/a" if correlation is None else "%.2f" % correlation))
    if payload["style_top_half"] and payload["style_bottom_half"]:
        print("style-favourable years : mean %+.1f%%, positive %.0f%%"
              % (payload["style_top_half"]["mean_strategy_return"] * 100,
                 100 * payload["style_top_half"]["share_positive"]))
        print("style-unfavourable  : mean %+.1f%%, positive %.0f%%"
              % (payload["style_bottom_half"]["mean_strategy_return"] * 100,
                 100 * payload["style_bottom_half"]["share_positive"]))
    if size_corr is not None:
        print("size spread (small-large) vs strategy: correlation %.2f" % size_corr)
        print("small-cap-leads years : mean %+.1f%%"
              % (payload["small_cap_leads"]["mean_strategy_return"] * 100))
        print("large-cap-leads years : mean %+.1f%%"
              % (payload["large_cap_leads"]["mean_strategy_return"] * 100))
    for key, value in monthly.items():
        print("%-18s n=%d  r=%+.3f  t=%+.2f" % (key, value["n"], value["r"], value["t"]))
    print("-> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
