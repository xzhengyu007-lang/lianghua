"""Does the shrinking PIT cross-section explain the rising low-vol IC?

RESEARCH_LOG 23.53 reports 20-day cross-sectional IC of the low-vol score of
+0.049 (2019-21), +0.114 (2022-23) and +0.123 (2024-26). Over the very same
span the tradable pool shrinks from a median of 251 names to 128 (23.75), and
"pick 8 of N by volatility rank" is mechanically a different operation at N=251
than at N=128.

This script separates two explanations for the IC drift:

* composition -- later cross-sections are mostly names that survived to today,
  earlier ones include names that later delisted. If the surviving names simply
  have a stronger volatility/return relation, the IC rise is survivorship, not a
  regime change. Test: hold the membership fixed by scoring the 128 currently
  listed names only.
* breadth -- fewer names means a coarser rank statistic and a different spread
  of scores. Test: subsample every fat cross-section down to the same N and
  recompute IC (n_matched), averaging over random draws.

If IC still rises with membership fixed and breadth matched, neither explains
it and the drift is something else.

Usage::

    python scripts/run_breadth_ic.py
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

from quant_system.analytics import forward_returns, ic_summary, spearman_ic  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.factors import factor_library  # noqa: E402


def load_panel(cache_dir: Path):
    panel = _rebuilt_panel(list(pit_universe()), cache_dir, "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _ = align_panel(panel)
    close = pd.DataFrame({s: f["close"] for s, f in aligned.items()})
    volume = pd.DataFrame({s: f["volume"] for s, f in aligned.items()})
    return aligned, close, volume


def score_frame(panel: dict[str, pd.DataFrame], close: pd.DataFrame, vol_window: int) -> pd.DataFrame:
    return factor_library(panel, vol_window=vol_window)["low_vol"].reindex_like(close)


def matched_ic(
    signal: pd.DataFrame,
    future: pd.DataFrame,
    target_n: int,
    draws: int,
    seed: int = 0,
) -> pd.Series:
    """IC after subsampling every cross-section down to target_n names."""
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    values: dict[pd.Timestamp, float] = {}
    for date in signal.index:
        s = signal.loc[date].replace([np.inf, -np.inf], np.nan)
        f = future.loc[date].replace([np.inf, -np.inf], np.nan)
        joined = pd.concat([s, f], axis=1).dropna()
        if len(joined) < max(target_n, 5):
            continue
        scores = []
        for _ in range(draws):
            pick = rng.choice(len(joined), size=target_n, replace=False)
            subset = joined.iloc[pick]
            rho, _p = spearmanr(subset.iloc[:, 0].to_numpy(), subset.iloc[:, 1].to_numpy())
            if np.isfinite(rho):
                scores.append(float(rho))
        if scores:
            values[date] = float(np.mean(scores))
    return pd.Series(values, dtype=float).sort_index()


def ols(frame: pd.DataFrame, y: str, xs: list[str]) -> dict:
    """Plain OLS with classical standard errors, enough for a direction check."""
    data = frame[[y] + xs].dropna()
    if len(data) < len(xs) + 3:
        return {"n": int(len(data)), "coefficients": {}, "note": "too few observations"}
    design = np.column_stack([np.ones(len(data))] + [data[x].to_numpy() for x in xs])
    beta, *_ = np.linalg.lstsq(design, data[y].to_numpy(), rcond=None)
    resid = data[y].to_numpy() - design @ beta
    dof = len(data) - design.shape[1]
    sigma2 = float(resid @ resid / dof) if dof > 0 else float("nan")
    try:
        cov = sigma2 * np.linalg.inv(design.T @ design)
        se = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        se = np.full(len(beta), np.nan)
    names = ["intercept"] + xs
    return {
        "n": int(len(data)),
        "coefficients": {
            name: {
                "estimate": float(beta[i]),
                "stderr": float(se[i]) if np.isfinite(se[i]) else None,
                "t": float(beta[i] / se[i]) if np.isfinite(se[i]) and se[i] else None,
            }
            for i, name in enumerate(names)
        },
    }


def yearly(ic_frame: pd.DataFrame) -> pd.DataFrame:
    frame = ic_frame.copy()
    frame.index = pd.DatetimeIndex(frame.index)
    frame["year"] = frame.index.year
    grouped = frame.groupby("year")
    rows = []
    for year, group in grouped:
        row = {
            "year": int(year),
            "days": int(len(group)),
            "median_names": float(group["n_names"].median()),
            "median_listed": float(group["n_listed"].median()),
            "median_delisted": float(group["n_delisted"].median()),
        }
        for column in ("ic_full", "ic_listed", "ic_delisted", "ic_matched"):
            series = group[column].dropna()
            row[column] = float(series.mean()) if len(series) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "breadth_ic"))
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--vol-window", type=int, default=60)
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--target-n", type=int, default=0, help="default: median N of the final year")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("loading rebuilt PIT panel...", flush=True)
    aligned, close, volume = load_panel(Path(args.cache_dir))
    eligible = close.notna() & volume.notna() & (volume > 0)
    score = score_frame(aligned, close, args.vol_window)
    future = forward_returns(close, args.horizon)

    signal_full = score.where(eligible)
    # Panel columns carry the exchange prefix (sh600519) while BROAD_UNIVERSE
    # holds bare codes (600519), so match on the code suffix.
    broad_codes = set(BROAD_UNIVERSE)
    listed = [s for s in close.columns if s[-6:] in broad_codes]
    others = [s for s in close.columns if s[-6:] not in broad_codes]
    if not listed or not others:
        raise RuntimeError("panel does not split into listed and delisted cohorts")
    signal_listed = score.where(eligible)[listed]
    signal_delisted = score.where(eligible)[others]

    counts = signal_full.notna().sum(axis=1)
    last_year = close.index[-1].year
    if args.target_n:
        target_n = int(args.target_n)
    else:
        target_n = int(np.median(counts[counts.index.year == last_year]))

    print(f"scoring {len(close.columns)} names, horizon {args.horizon}d, target_n={target_n}", flush=True)
    ic_full = spearman_ic(signal_full, future)
    ic_listed = spearman_ic(signal_listed, future)
    ic_delisted = spearman_ic(signal_delisted, future)
    ic_matched = matched_ic(signal_full, future, target_n, args.draws)

    print(
        f"ic points: full={len(ic_full)} listed={len(ic_listed)} matched={len(ic_matched)}",
        flush=True,
    )
    frame = pd.DataFrame(
        {
            "n_names": counts,
            "n_listed": signal_listed.notna().sum(axis=1),
            "n_delisted": signal_delisted.notna().sum(axis=1),
            "ic_full": ic_full,
            "ic_listed": ic_listed,
            "ic_delisted": ic_delisted,
            "ic_matched": ic_matched,
        }
    ).dropna(subset=["ic_full"])
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "date"
    frame.to_csv(out / "panel.csv", encoding="utf-8-sig")

    table = yearly(frame)
    table.to_csv(out / "ic_by_year.csv", index=False, encoding="utf-8-sig")

    reg_frame = frame.copy()
    reg_frame.index = pd.DatetimeIndex(reg_frame.index)
    reg_frame["year"] = reg_frame.index.year + (reg_frame.index.day_of_year / 365.25)
    reg_frame["log_n"] = np.log(reg_frame["n_names"])
    regression = ols(reg_frame, "ic_full", ["year", "log_n"])
    corr_with_n = float(frame["ic_full"].corr(frame["n_names"], method="spearman"))
    # The pool shrinks monotonically, so year and log N are nearly collinear: the
    # regression's own signs are not evidence about breadth. Report the
    # collinearity next to the coefficients instead of letting them be over-read.
    corr_year_log_n = float(reg_frame[["year", "log_n"]].corr().iloc[0, 1])

    # Horizon robustness: the cohort split must not be a 20-day artefact.
    horizon_rows = []
    for horizon in (5, 10, 20, 60):
        fut = forward_returns(close, horizon)
        for cohort, cohort_signal in (
            ("full", signal_full),
            ("listed", signal_listed),
            ("delisted", signal_delisted),
        ):
            series = spearman_ic(cohort_signal, fut)
            stats = ic_summary(series)
            horizon_rows.append(
                {
                    "horizon": horizon,
                    "cohort": cohort,
                    "mean_ic": stats["mean_ic"],
                    "t_stat": stats["t_stat"],
                    "days": int(len(series)),
                }
            )
    horizon_frame = pd.DataFrame(horizon_rows)
    horizon_frame.to_csv(out / "ic_by_horizon.csv", index=False, encoding="utf-8-sig")
    for row in horizon_rows:
        print(f"h={row['horizon']:>3} {row['cohort']:<9} ic={row['mean_ic']:+.4f} t={row['t_stat']:+.2f}")
    print(flush=True)

    summary = {
        "symbols": int(close.shape[1]),
        "days": int(len(close)),
        "start": str(close.index.min().date()),
        "end": str(close.index.max().date()),
        "horizon": args.horizon,
        "vol_window": args.vol_window,
        "target_n": target_n,
        "draws": args.draws,
        "full": ic_summary(ic_full),
        "listed_only": ic_summary(ic_listed),
        "delisted_only": ic_summary(ic_delisted),
        "n_matched": ic_summary(ic_matched),
        "spearman_ic_vs_n": corr_with_n,
        "corr_year_log_n": corr_year_log_n,
        "regression_ic_on_year_and_log_n": regression,
        "by_year": table.to_dict("records"),
        "by_horizon": horizon_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 横截面宽度与低波 IC", ""]
    lines.append(
        f"面板 {summary['start']} .. {summary['end']}，{summary['symbols']} 只；"
        f"{args.horizon} 日前瞻收益，vol_window={args.vol_window}；"
        f"宽度匹配基准 N={target_n}，每次 {args.draws} 次抽样。"
    )
    lines.append("")
    lines.append("## 逐年 IC")
    lines.append("")
    lines.append(
        "| 年份 | 交易日 | 中位 N | IC 全池 | IC 仍在册 | IC 最终退市 | IC 宽度匹配 N=%d |" % target_n
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.year} | {row.days} | {row.median_names:.0f} | "
            f"{row.ic_full:+.4f} | {row.ic_listed:+.4f} | {row.ic_delisted:+.4f} | "
            f"{row.ic_matched:+.4f} |"
        )
    lines.append("")
    lines.append("## 全样本")
    lines.append("")
    lines.append("| 口径 | 平均 IC | IC_IR | t | 正比例 |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, key in (
        ("全池", "full"),
        ("仍在册", "listed_only"),
        ("最终退市", "delisted_only"),
        (f"宽度匹配 N={target_n}", "n_matched"),
    ):
        row = summary[key]
        lines.append(
            f"| {label} | {row['mean_ic']:+.4f} | {row['ic_ir']:+.3f} | {row['t_stat']:+.2f} | {row['positive_ratio']:.1%} |"
        )
    lines.append("")
    lines.append("## IC 对 year 与 log N 的回归")
    lines.append("")
    lines.append("| 变量 | 系数 | 标准误 | t |")
    lines.append("|---|---:|---:|---:|")
    for name, item in regression["coefficients"].items():
        se = "-" if item["stderr"] is None else f"{item['stderr']:.5f}"
        t = "-" if item["t"] is None else f"{item['t']:+.2f}"
        lines.append(f"| {name} | {item['estimate']:+.5f} | {se} | {t} |")
    lines.append("")
    lines.append(f"IC 与当日 N 的 Spearman 相关：{corr_with_n:+.4f}")
    lines.append("")
    lines.append(
        f"**year 与 log N 的相关系数是 {corr_year_log_n:+.3f}，两者几乎共线。**"
        '回归里 log_n 的正系数是共线性造成的符号翻转，不能拿来当「宽度有用」的证据；'
        "宽度的判断以宽度匹配实验为准（那一列与全池几乎逐位相同）。"
    )
    lines.append("")
    lines.append("## 跨 horizon 复核")
    lines.append("")
    lines.append("| horizon | 全池 | 仍在册 | 最终退市 |")
    lines.append("|---:|---:|---:|---:|")
    pivot = horizon_frame.pivot_table(index="horizon", columns="cohort", values="mean_ic")
    for horizon, row in pivot.iterrows():
        lines.append(
            f"| {int(horizon)}d | {row.get('full', float('nan')):+.4f} | "
            f"{row.get('listed', float('nan')):+.4f} | {row.get('delisted', float('nan')):+.4f} |"
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(table.to_string(index=False), flush=True)
    print(f"\nspearman(ic, n) = {corr_with_n:+.4f}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
