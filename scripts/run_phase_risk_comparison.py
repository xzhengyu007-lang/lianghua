"""Does a lower risk budget also shrink the calendar (rebalance-phase) risk?

Section 23.21 found a >100pp spread between the best and worst rebalance phase
over a long window: the same strategy, same names, same costs, different
20-day anchor. If ``--target-vol 0.12`` scaled every phase by the same factor,
the *absolute* spread would shrink proportionally and the *relative* spread
would not move at all.

The script reads walk-forward runs that swept 20 phases and compares two risk
budgets per window group. Run labels are ``target:group=dir`` (for example
``low:2019-2021=reports/wf_phase_vol12_2019``); without a colon the whole label
is treated as the target and the group is empty, which keeps the original
single-group usage working.

    spread    = best phase return - worst phase return
    relative  = spread / |median phase return|   (calendar risk per unit P&L)

A risk budget that only scales exposure should leave ``relative`` roughly
unchanged; a real reduction would show up there.

Usage::

    python scripts/run_phase_risk_comparison.py
    python scripts/run_phase_risk_comparison.py --run low:2019=reports/wf_phase_vol12_2019 --run high:2019=reports/wf_phase_vol18_2019
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_RUNS = [
    f"{target}:{group}=reports/wf_phase_vol{target}{group}"
    for target, group in (
        ("12", "_2019"), ("18", "_2019"),
        ("12", "_2022"), ("18", "_2022"),
        ("12", "_2024"), ("18", "_2024"),
    )
]

GROUP_NAMES = {
    "_2019": "2019-2021",
    "_2022": "2022-2023",
    "_2024": "2024-2026",
}

TARGET_NAMES = {"12": "12%", "18": "18%", "low": "12%", "high": "18%"}


def parse_run(item: str) -> tuple[str, str, Path]:
    label, _, path = item.partition("=")
    if not path:
        raise SystemExit(f"--run expects label=dir, got {item!r}")
    target, _, group = label.partition(":")
    return TARGET_NAMES.get(target, target), GROUP_NAMES.get(group, group), Path(path)


def summarise(path: Path, target: str, group: str) -> pd.DataFrame:
    frame = pd.read_csv(path / "windows.csv")
    frame = frame[frame["phase_count"].fillna(0) > 0].copy()
    frame["spread"] = frame["phase_return_best"] - frame["phase_return_worst"]
    frame["relative_spread"] = frame["spread"] / frame["phase_return_median"].abs().clip(lower=0.01)
    frame["target"] = target
    frame["group"] = group
    return frame[
        ["target", "group", "test_start", "test_end", "phase_count", "phase_return_median",
         "phase_return_worst", "phase_return_best", "phase_sharpe_median", "spread", "relative_spread"]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=None, help="target[:group]=report_dir")
    parser.add_argument("--output", default=str(ROOT / "reports" / "phase_risk_comparison"))
    args = parser.parse_args(argv)

    runs = args.run if args.run else DEFAULT_RUNS
    frames = []
    for item in runs:
        target, group, path = parse_run(item)
        if not (ROOT / path / "windows.csv").exists():
            print(f"skip missing {path}")
            continue
        frames.append(summarise(ROOT / path, target, group))
    if not frames:
        print("no runs found")
        return 1
    data = pd.concat(frames, ignore_index=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    data.to_csv(out / "windows.csv", index=False, encoding="utf-8-sig")

    by_group = (
        data.groupby(["target", "group"])
        .agg(
            windows=("spread", "size"),
            mean_spread=("spread", "mean"),
            mean_abs_median=("phase_return_median", lambda s: s.abs().mean()),
            mean_relative_spread=("relative_spread", "mean"),
            worst_spread=("spread", "max"),
        )
        .reset_index()
    )
    by_group["ratio"] = by_group["mean_spread"] / by_group["mean_abs_median"]
    by_group.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")

    pooled = (
        data.groupby("target")
        .agg(
            windows=("spread", "size"),
            mean_spread=("spread", "mean"),
            mean_abs_median=("phase_return_median", lambda s: s.abs().mean()),
            mean_relative_spread=("relative_spread", "mean"),
            worst_spread=("spread", "max"),
        )
        .reset_index()
    )
    pooled["ratio"] = pooled["mean_spread"] / pooled["mean_abs_median"]

    pairs = []
    for group in by_group["group"].unique():
        low = by_group[(by_group.group == group) & (by_group.target == "12%")]
        high = by_group[(by_group.group == group) & (by_group.target == "18%")]
        if low.empty or high.empty:
            continue
        pairs.append(
            {
                "group": group,
                "spread_12": float(low.mean_spread.iloc[0]),
                "spread_18": float(high.mean_spread.iloc[0]),
                "shrink": float(1 - low.mean_spread.iloc[0] / high.mean_spread.iloc[0]),
                "ratio_12": float(low.ratio.iloc[0]),
                "ratio_18": float(high.ratio.iloc[0]),
            }
        )

    (out / "summary.json").write_text(
        json.dumps(
            {
                "runs": runs,
                "by_group": by_group.to_dict(orient="records"),
                "pooled": pooled.to_dict(orient="records"),
                "pairs": pairs,
                "windows": data.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = ["# 相位风险：降低风险预算会缩小日历风险吗", ""]
    lines.append("| 风险预算 | 窗口组 | 窗口数 | 平均相位离散 | 平均绝对中位收益 | 离散/中位 | 最大离散 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for _, row in by_group.iterrows():
        lines.append(
            f"| {row.target} | {row.group or '全部'} | {row.windows} | {row.mean_spread:.2%} | "
            f"{row.mean_abs_median:.2%} | {row.ratio:.2f} | {row.worst_spread:.2%} |"
        )
    for _, row in pooled.iterrows():
        lines.append(
            f"| {row.target} | 合计 | {row.windows} | {row.mean_spread:.2%} | "
            f"{row.mean_abs_median:.2%} | {row.ratio:.2f} | {row.worst_spread:.2%} |"
        )
    if pairs:
        lines.append("")
        lines.append("## 同一窗口组内 12% vs 18%")
        lines.append("")
        lines.append("| 窗口组 | 12% 离散 | 18% 离散 | 缩小 | 12% 离散/中位 | 18% 离散/中位 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in pairs:
            lines.append(
                f"| {row['group']} | {row['spread_12']:.2%} | {row['spread_18']:.2%} | "
                f"{row['shrink']:.0%} | {row['ratio_12']:.2f} | {row['ratio_18']:.2f} |"
            )
    lines.append("")
    lines.append("## 逐窗口")
    lines.append("")
    lines.append("| 风险预算 | 窗口组 | 测试窗口 | 中位 | 最差 | 最好 | 离散 | 相对离散 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for _, row in data.iterrows():
        lines.append(
            f"| {row.target} | {row.group or '-'} | {row.test_start}..{row.test_end} | "
            f"{row.phase_return_median:+.2%} | {row.phase_return_worst:+.2%} | "
            f"{row.phase_return_best:+.2%} | {row.spread:.2%} | {row.relative_spread:.2f} |"
        )
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(by_group.round(4).to_string(index=False))
    print(pooled.round(4).to_string(index=False))
    if pairs:
        print(pd.DataFrame(pairs).round(4).to_string(index=False))
    print(f"\nreport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
