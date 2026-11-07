"""Reproducible report artifacts: CSV, JSON, Markdown, and PNG charts."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# CJK labels in the PNG charts need an installed family; fall back silently to
# English labels when none is available.
_CJK_CANDIDATES = ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC")
_available = {f.name for f in font_manager.fontManager.ttflist}
CJK_FONT = next((name for name in _CJK_CANDIDATES if name in _available), None)
if CJK_FONT:
    plt.rcParams["font.sans-serif"] = [CJK_FONT, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .config import REPORT_DIR, config_dict
from .html_report import write_dashboard
from .metrics import performance_metrics
from .metrics import drawdown_series


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def create_run_dir(prefix: str = "run", root: Path = REPORT_DIR / "runs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = root / f"{prefix}_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def plot_equity(
    result: BacktestResult,
    path: Path,
    *,
    title: str = "Strategy equity curve",
    benchmark: pd.Series | None = None,
) -> Path:
    """Plot strategy and benchmark growth of 1.

    ``benchmark`` is a **return** series aligned to the equity dates, not a
    price level. Compounding returns and then normalising gives the correct
    benchmark curve; normalising a price series would only be right when it
    starts at the same date with the same base.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    normalized = result.equity / result.equity.iloc[0]
    ax1.plot(normalized.index, normalized.values, label="Strategy", linewidth=1.8, color="#0B6E4F")
    if benchmark is not None:
        bench_returns = benchmark.reindex(result.equity.index).fillna(0.0)
        bench = (1.0 + bench_returns).cumprod()
        bench = bench / bench.iloc[0]
        ax1.plot(bench.index, bench.values, label="Benchmark", linewidth=1.3, color="#6C757D")
    ax1.set_title(title)
    ax1.set_ylabel("Growth of 1")
    ax1.grid(alpha=0.25)
    ax1.legend(frameon=False)
    dd = drawdown_series(normalized)
    ax2.fill_between(dd.index, dd.values * 100, 0, color="#C0392B", alpha=0.55)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_monthly_returns(equity: pd.Series, path: Path) -> Path:
    from .metrics import monthly_returns

    table = monthly_returns(equity)
    pivot = table.pivot(index="year", columns="month", values="return").fillna(0.0) * 100
    fig, ax = plt.subplots(figsize=(12, 4.8))
    image = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-10, vmax=10)
    month_labels = [f"{m}月" for m in pivot.columns] if CJK_FONT else [
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]
        for m in pivot.columns
    ]
    ax.set_xticks(range(len(pivot.columns)), month_labels)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            ax.text(col, row, f"{pivot.iloc[row, col]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("月度收益 (%)" if CJK_FONT else "Monthly returns (%)")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def write_json(payload: dict, path: Path) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def write_markdown_report(
    path: Path,
    *,
    title: str,
    metrics: dict,
    metadata: dict,
    windows: pd.DataFrame | None = None,
    notes: list[str] | None = None,
    equity_plot: str | None = None,
    monthly_plot: str | None = None,
) -> Path:
    def pct(value: Any) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "—"
        return "—" if not np.isfinite(value) else f"{value:.2%}"

    def num(value: Any, digits: int = 2) -> str:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "—"
        return "—" if not np.isfinite(value) else f"{value:.{digits}f}"

    lines = [f"# {title}", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ""]
    lines += ["## 核心指标", "", "| 指标 | 数值 |", "|---|---:|"]
    labels = {
        "total_return": "累计收益",
        "cagr": "年化收益",
        "annual_volatility": "年化波动",
        "sharpe": "夏普比率",
        "sortino": "索提诺比率",
        "max_drawdown": "最大回撤",
        "calmar": "卡玛比率",
        "win_rate": "日胜率",
        "profit_factor": "盈亏比",
        "turnover": "区间单边换手",
        "exposure": "平均仓位",
        "alpha": "年化 Alpha",
        "beta": "Beta",
        "information_ratio": "信息比率",
        "tracking_error": "跟踪误差",
    }
    for key, label in labels.items():
        if key not in metrics:
            continue
        value = metrics[key]
        rendered = pct(value) if key in {"total_return", "cagr", "annual_volatility", "max_drawdown", "win_rate", "turnover", "exposure", "alpha", "tracking_error"} else num(value)
        lines.append(f"| {label} | {rendered} |")
    if "annual_turnover" in metadata:
        lines.append(f"| 年化单边换手 | {pct(metadata['annual_turnover'])} |")
    if equity_plot:
        lines += ["", "## 净值与回撤", "", f"![equity]({equity_plot})"]
    if monthly_plot:
        lines += ["", "## 月度收益", "", f"![monthly]({monthly_plot})"]
    if windows is not None and not windows.empty:
        if "phase_count" in windows.columns and (windows["phase_count"] > 0).any():
            lines += [
                "",
                "## 调仓相位分布",
                "",
                "同一策略、同一数据、同一参数，只把 20 日调仓的相位平移若干根 bar。",
                "`headline` 是报告实际交易的那条路径（通常是 offset=0）；"
                "`median/worst/best` 才是稳健区间。",
                "",
                "| 测试窗口 | 相位数 | headline | 中位 | 最差 | 最好 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
            for _, row in windows.iterrows():
                lines.append(
                    f"| {row['test_start']} .. {row['test_end']} | "
                    f"{int(row['phase_count'])} | {pct(row['test_return'])} | "
                    f"{pct(row['phase_return_median'])} | "
                    f"{pct(row['phase_return_worst'])} | "
                    f"{pct(row['phase_return_best'])} |"
                )
            counts = windows.loc[windows["phase_count"] > 0, "phase_count"]
            medians = windows.loc[windows["phase_count"] > 0, "phase_return_median"]
            worsts = windows.loc[windows["phase_count"] > 0, "phase_return_worst"]
            bests = windows.loc[windows["phase_count"] > 0, "phase_return_best"]
            lines += [
                "",
                f"聚合：{int(counts.max())} 个相位 × {len(counts)} 个窗口；"
                f"逐窗中位收益合计 {pct(float(medians.mean()))}（算术平均）、"
                f"最差 {pct(float(worsts.min()))}、最好 {pct(float(bests.max()))}。",
                "逐窗平均不是组合收益，只用于看相位敏感度。",
            ]
        lines += ["", "## 滚动样本外窗口", "", "```text", windows.to_string(index=False), "```"]
    lines += ["", "## 配置", "", "```json", json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default), "```"]
    if notes:
        lines += ["", "## 说明", ""] + [f"- {note}" for note in notes]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_backtest(
    result: BacktestResult,
    output_dir: Path,
    *,
    title: str = "Backtest",
    benchmark: pd.Series | None = None,
    notes: list[str] | None = None,
    caveats: list[str] | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.equity.to_frame("equity").to_csv(output_dir / "equity.csv")
    result.trades.to_csv(output_dir / "trades.csv", index=False, encoding="utf-8-sig")
    result.positions.to_csv(output_dir / "positions.csv", index=False, encoding="utf-8-sig")
    write_json(result.to_dict(), output_dir / "metrics.json")
    equity_plot = plot_equity(result, output_dir / "equity.png", title=title, benchmark=result.benchmark_returns)
    monthly_plot = plot_monthly_returns(result.equity, output_dir / "monthly_returns.png")
    write_markdown_report(
        output_dir / "report.md",
        title=title,
        metrics=result.metrics.to_dict(),
        metadata=result.metadata,
        equity_plot=equity_plot.name,
        monthly_plot=monthly_plot.name,
        notes=notes,
    )
    bench_equity = None
    if benchmark is not None:
        bench = benchmark.reindex(result.equity.index).ffill().dropna()
        if len(bench) > 1:
            bench_equity = bench
    write_dashboard(
        output_dir / "dashboard.html",
        title=title,
        equity=result.equity,
        metrics=result.metrics.to_dict(),
        benchmark_equity=bench_equity,
        trades=result.trades,
        positions=result.positions,
        notes=notes,
        caveats=caveats,
    )
    return {"directory": str(output_dir), "metrics": result.metrics.to_dict()}

