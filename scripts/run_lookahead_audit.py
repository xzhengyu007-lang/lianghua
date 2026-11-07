"""Look-ahead detector: scramble the future, the past must not move.

Every claim in this repo rests on "the backtest only used information available
at the time". Unit tests check individual functions; this checks the *whole
pipeline* end to end with a blunt instrument: replace every bar after a cut date
with a different but still realistic-looking series, rerun the identical
strategy, and require the equity curve up to the cut to be bit-identical.

A pipeline that peeks at the future - a centered rolling window, a full-sample
z-score, a covariance fitted on the whole history, a sector map built from later
data - will move the past, because the scrambled future changes some statistic
it used.

Two modes:

    truncate  drop everything after the cut, rerun, compare the prefix;
    scramble  keep the same length but permute the *returns* after the cut
              (same marginal distribution, no look-ahead), compare the prefix.

``truncate`` is the stricter test and also catches "needs the full panel length"
bugs. ``scramble`` catches code that silently reads a future row that truncation
would have removed outright.

Usage::

    python scripts/run_lookahead_audit.py
    python scripts/run_lookahead_audit.py --cut 2023-06-30 --mode both
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
from quant_system.strategies import LowVolCoreStrategy  # noqa: E402


PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def run(panel: dict[str, pd.DataFrame]) -> pd.Series:
    strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
    weights = build_portfolio(
        strategy.generate(panel),
        panel,
        risk=RiskConfig(),
        use_regime=False,
        use_vol_target=True,
    )
    return Backtester(BacktestConfig()).run(panel, weights).equity


def truncate(panel: dict[str, pd.DataFrame], cut: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {symbol: frame.loc[:cut] for symbol, frame in panel.items() if len(frame.loc[:cut]) > 0}


def scramble_future(
    panel: dict[str, pd.DataFrame], cut: pd.Timestamp, seed: int
) -> dict[str, pd.DataFrame]:
    """Replace post-cut **returns**, keeping the pre-cut series untouched.

    Prices after the cut are rebuilt as ``last_pre_cut_close * cumprod(scrambled
    returns)``. The scrambled returns are a permutation of the real post-cut
    returns for that symbol, so marginal distribution, volatility and limit
    properties are preserved while every path-dependent statistic is destroyed.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for symbol, frame in panel.items():
        after = frame.loc[frame.index > cut]
        before = frame.loc[:cut]
        if before.empty or after.empty:
            out[symbol] = frame
            continue
        anchor = float(before["close"].iloc[-1])
        # Suspension gaps are NaN rows.  Forward fill (seeded with the pre-cut
        # close) so a halted day earns 0% and the resumption jump is kept; a raw
        # ratio would emit NaN and ``cumprod`` would poison the whole path.
        prices = after["close"].ffill().fillna(anchor).to_numpy(dtype=float)
        previous = np.concatenate([[anchor], prices[:-1]])
        returns = prices / previous - 1.0
        permutation = rng.permutation(len(returns))
        scrambled = returns[permutation]
        close = anchor * np.cumprod(1.0 + scrambled)
        rebuilt = after.copy()
        scale = close / rebuilt["close"].to_numpy(dtype=float)
        for column in ("open", "high", "low", "close"):
            if column in rebuilt:
                rebuilt[column] = rebuilt[column].to_numpy(dtype=float) * scale
        out[symbol] = pd.concat([before, rebuilt])
    return out


def compare(expected: pd.Series, actual: pd.Series, cut: pd.Timestamp) -> dict:
    left = expected.loc[:cut]
    right = actual.reindex(left.index)
    if right.isna().any():
        return {"ok": False, "reason": "missing overlapping dates", "n": int(len(left))}
    diff = (left - right).abs()
    relative = (diff / left.abs().clip(lower=1e-9)).max()
    return {
        "ok": bool(relative <= 1e-10),
        "n": int(len(left)),
        "max_abs_diff": float(diff.max()),
        "max_rel_diff": float(relative),
        "first_mismatch": None if diff.empty else str(diff.idxmax().date()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "lookahead_audit"))
    parser.add_argument("--cut", nargs="+", default=["2021-06-30", "2023-06-30"])
    parser.add_argument("--mode", choices=["truncate", "scramble", "both"], default="both")
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args(argv)

    print("loading rebuilt PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), Path(args.cache_dir), "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, dates = align_panel(panel)
    print(f"panel: {len(aligned)} symbols, {len(dates)} bars", flush=True)

    full_equity = run(aligned)
    rows = []
    for cut_text in args.cut:
        cut = pd.Timestamp(cut_text)
        if cut not in dates:
            cut = dates[dates.searchsorted(cut) - 1]
        if args.mode in {"truncate", "both"}:
            prefix = run(truncate(aligned, cut))
            result = compare(full_equity, prefix, cut)
            result.update({"cut": str(cut.date()), "mode": "truncate"})
            rows.append(result)
            print(f"truncate {cut.date()}: ok={result['ok']} "
                  f"max_rel={result['max_rel_diff']:.2e}", flush=True)
        if args.mode in {"scramble", "both"}:
            scrambled = scramble_future(aligned, cut, args.seed)
            prefix = run(scrambled)
            result = compare(full_equity, prefix, cut)
            result.update({"cut": str(cut.date()), "mode": "scramble"})
            rows.append(result)
            print(f"scramble {cut.date()}: ok={result['ok']} "
                  f"max_rel={result['max_rel_diff']:.2e}", flush=True)

    # Also check the *signal* path alone, which is where most peeking happens.
    strategy = LowVolCoreStrategy(rebalance_reset="year", **PARAMS)
    full_weights = strategy.generate(aligned)
    for cut_text in args.cut:
        cut = pd.Timestamp(cut_text)
        if cut not in dates:
            cut = dates[dates.searchsorted(cut) - 1]
        probe = truncate(aligned, cut)
        partial = strategy.generate(probe)
        left = full_weights.loc[:cut]
        right = partial.reindex(left.index)
        # Columns may legitimately differ (a name with no pre-cut bars), so
        # compare the intersection of symbols and require identical values.
        columns = [c for c in left.columns if c in right.columns]
        same_shape = left[columns].shape == right[columns].shape
        diff = (left[columns] - right[columns]).abs()
        rows.append(
            {
                "cut": str(cut.date()),
                "mode": "signal-truncate",
                "ok": bool(same_shape and float(diff.to_numpy().max()) <= 1e-12),
                "n": int(left.shape[0]),
                "max_abs_diff": float(diff.to_numpy().max()),
                "max_rel_diff": float(diff.to_numpy().max()),
                "first_mismatch": None,
            }
        )
        print(f"signal {cut.date()}: ok={rows[-1]['ok']}", flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "checks.csv", index=False, encoding="utf-8-sig")
    payload = {
        "params": PARAMS,
        "cuts": args.cut,
        "modes": args.mode,
        "checks": rows,
        "all_ok": all(row["ok"] for row in rows),
    }
    (output / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 前视偏差审计：打乱未来，过去必须不动", ""]
    lines.append("| 切点 | 模式 | 结果 | 天数 | 最大绝对差 | 最大相对差 |")
    lines.append("|---|---|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {cut} | {mode} | {verdict} | {n} | {max_abs_diff:.2e} | {max_rel_diff:.2e} |".format(
                verdict="**通过**" if row["ok"] else "**失败**", **row
            )
        )
    lines.append("")
    lines.append("- `truncate`：删掉切点之后全部数据重跑，切点前净值必须逐位一致。")
    lines.append("- `scramble`：保留长度、置换切点后收益，切点前净值必须逐位一致。")
    lines.append("- `signal-truncate`：只比较信号权重矩阵，不经过撮合。")
    lines.append(
        "- 负对照（`scripts/run_lookahead_negative_control.py`）往同一条管线里注入居中滚动窗口、"
        "全样本 z-score、全样本截面排名三种泄漏，三者都必须被 `compare` 标记为失败；"
        "永久通过而未做负对照的审计没有信息量。"
    )
    lines.append("")
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print("all_ok:", payload["all_ok"])
    try:
        shown = output.resolve().relative_to(ROOT)
    except ValueError:
        shown = output
    print(f"report: {shown}")
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
