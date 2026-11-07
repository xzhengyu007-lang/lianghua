"""Build a bootstrap confidence-interval report from walk-forward equity curves.

The report is a pure post-processing step: it never refits a strategy, so it
cannot introduce look-ahead. It answers a different question from the
walk-forward table: given ~2-3 years of daily returns, how wide is the sampling
interval around the headline Sharpe / drawdown?
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .bootstrap import bootstrap_summary, paired_bootstrap_diff
from .equity_io import daily_returns_from_equity
from .report import write_json

DEFAULT_SUITE = [
    ("t120_2022", "reports/wf_pit267_2022_trend120"),
    ("t160_2022", "reports/wf_pit267_2022_trend160"),
    ("sel_2022", "reports/wf_pit267_sel_2022"),
    ("t120_2024", "reports/wf_pit_full267_year"),
    ("t160_2024", "reports/wf_pit267_trend160"),
    ("t180_2024", "reports/wf_pit267_trend180"),
    ("chain120", "reports/chained_trend120"),
    ("chain160", "reports/chained_trend160"),
]
DEFAULT_PAIRS = [
    ("t160_2022", "t120_2022"),
    ("t160_2024", "t120_2024"),
    ("t180_2024", "t120_2024"),
    ("chain160", "chain120"),
]


def load_returns(report_dir: Path) -> pd.Series:
    """Daily returns for a saved run, including the first session.

    Uses ``equity_io`` because ``equity.csv`` stores the level *after* day one
    while ``metrics.json`` was computed with that day included; a plain
    ``pct_change()`` would silently drop it.
    """
    return daily_returns_from_equity(report_dir)


def _fmt(value: float, kind: str) -> str:
    if value is None or pd.isna(value):
        return "-"
    if kind in {"cagr", "max_drawdown", "annual_volatility"}:
        return f"{value * 100:+.2f}%"
    return f"{value:+.3f}"


def build_bootstrap_report(
    runs: list[tuple[str, str]],
    pairs: list[tuple[str, str]] | None = None,
    *,
    n_boot: int = 5000,
    seed: int = 20261107,
    root: Path | None = None,
    log=None,
) -> tuple[dict, str]:
    """Return ``(summary, markdown)`` for the given ``(label, report_dir)`` runs."""
    root = Path(root) if root is not None else Path.cwd()
    say = log or (lambda message: None)
    series: dict[str, pd.Series] = {}
    summaries: dict[str, dict] = {}
    skipped: list[str] = []
    for label, path in runs:
        report_dir = root / path
        try:
            returns = load_returns(report_dir)
        except (FileNotFoundError, ValueError) as exc:
            say(f"[skip] {label}: {exc}")
            skipped.append(f"{label}: {exc}")
            continue
        series[label] = returns
        summary = bootstrap_summary(returns, n_boot=n_boot, seed=seed)
        summary["_meta"]["start"] = str(returns.index.min().date())
        summary["_meta"]["end"] = str(returns.index.max().date())
        summary["_meta"]["report"] = path
        summaries[label] = summary
        say(
            f"{label:12s} {summary['_meta']['start']}..{summary['_meta']['end']} "
            f"n={summary['_meta']['n_obs']} "
            f"sharpe {summary['sharpe']['point']:.3f} "
            f"[{summary['sharpe']['p2']:.3f}, {summary['sharpe']['p97']:.3f}]"
        )

    pair_results: dict[str, dict] = {}
    for a, b in pairs or []:
        if a not in series or b not in series:
            say(f"[skip pair] {a} vs {b}: missing series")
            continue
        joined = pd.concat([series[a].rename("a"), series[b].rename("b")], axis=1).dropna()
        if joined.empty:
            say(f"[skip pair] {a} vs {b}: no overlapping dates")
            continue
        result = paired_bootstrap_diff(
            joined["a"].to_numpy(),
            joined["b"].to_numpy(),
            metric="sharpe",
            n_boot=n_boot,
            seed=seed,
        )
        payload = result.to_dict()
        payload["overlap_start"] = str(joined.index.min().date())
        payload["overlap_end"] = str(joined.index.max().date())
        payload["n_obs"] = int(len(joined))
        pair_results[f"{a}-{b}"] = payload
        say(
            f"paired {a} - {b}: dSharpe {payload['observed']:+.3f} "
            f"[{payload['low']:+.3f}, {payload['high']:+.3f}] "
            f"P(>0)={payload['prob_positive']:.2f}"
        )

    summary = {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "runs": summaries,
        "pairs": pair_results,
        "skipped": skipped,
        "note": "stationary block bootstrap on daily OOS returns; paired mode shares blocks",
    }
    return summary, render_markdown(summary)


def render_markdown(summary: dict) -> str:
    lines = [
        "# 样本外收益的块自助法置信区间",
        "",
        f"B={summary['n_boot']}，seed={summary['seed']}，块长按 n^(1/3)，95% 百分位区间。",
        "点估计来自原始净值；区间来自对日收益的平稳块重采样。",
        "",
        "| 组合 | 区间 | 交易日 | CAGR | Sharpe [2.5%, 97.5%] | MDD [2.5%, 97.5%] | 年化波动 [2.5%, 97.5%] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for label, item in summary["runs"].items():
        meta = item["_meta"]
        lines.append(
            f"| {label} | {meta['start']}..{meta['end']} | {meta['n_obs']} | "
            f"{_fmt(item['cagr']['point'], 'cagr')} | "
            f"{_fmt(item['sharpe']['point'], 'sharpe')} "
            f"[{_fmt(item['sharpe']['p2'], 'sharpe')}, {_fmt(item['sharpe']['p97'], 'sharpe')}] | "
            f"{_fmt(item['max_drawdown']['point'], 'max_drawdown')} "
            f"[{_fmt(item['max_drawdown']['p2'], 'max_drawdown')}, {_fmt(item['max_drawdown']['p97'], 'max_drawdown')}] | "
            f"{_fmt(item['annual_volatility']['point'], 'annual_volatility')} "
            f"[{_fmt(item['annual_volatility']['p2'], 'annual_volatility')}, "
            f"{_fmt(item['annual_volatility']['p97'], 'annual_volatility')}] |"
        )
    if summary["pairs"]:
        lines += [
            "",
            "## 配对差值（同一重采样日历）",
            "",
            "两条净值共享同一天历，块索引只抽一次，因此差值里不含市场择时运气。",
            "",
            "| 对比 | 重叠区间 | 交易日 | ΔSharpe 点估计 | 95% 区间 | P(Δ>0) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for key, item in summary["pairs"].items():
            lines.append(
                f"| {key} | {item['overlap_start']}..{item['overlap_end']} | {item['n_obs']} | "
                f"{item['observed']:+.3f} | [{item['low']:+.3f}, {item['high']:+.3f}] | "
                f"{item['prob_positive']:.2f} |"
            )
    if summary.get("skipped"):
        lines += ["", "## 跳过", ""]
        lines += [f"- {reason}" for reason in summary["skipped"]]
    lines += [
        "",
        "## 读法",
        "",
        "- 区间覆盖 0 的 Sharpe 不能当作「已经验证的优势」；它只说明方向为正。",
        "- 配对差值比两个单独区间更适合回答「A 是否优于 B」，因为两者在同一段行情里比较。",
        "- 块长默认 n^(1/3)（约 9 天），保留 20 日调仓周期带来的自相关；更长块会放宽区间。",
    ]
    return "\n".join(lines) + "\n"


def write_bootstrap_report(output_dir: Path, summary: dict, markdown: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(summary, output_dir / "summary.json")
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")
