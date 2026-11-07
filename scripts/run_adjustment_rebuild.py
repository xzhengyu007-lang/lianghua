"""Rebuild multiplicative total-return series from raw + qfq, then re-run the baseline (23.19)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from quant_system.adjustment import rebuild_frame
from quant_system.backtest import Backtester
from quant_system.config import BacktestConfig, RiskConfig
from quant_system.data import load_panel, load_symbol, trim_panel, usable_window
from quant_system.portfolio import build_portfolio
from quant_system.report import write_json
from quant_system.strategies import build_strategy

OUT = ROOT / "reports" / "adjustment_rebuild"
RAW_DIR = ROOT / "data" / "cache_rawcheck"
CACHE = ROOT / "data" / "cache"
PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def load_raw(symbols, start, end):
    out = {}
    for symbol in symbols:
        path = RAW_DIR / f"{symbol}_raw.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index)
        frame = frame.loc[start:end]
        if len(frame):
            out[symbol] = frame
    return out


def main() -> int:
    from quant_system.cli import _rebuilt_panel
    from quant_system.config import BROAD_UNIVERSE
    from quant_system.data import normalize_symbol

    OUT.mkdir(parents=True, exist_ok=True)
    symbols = sorted(normalize_symbol(symbol) for symbol in BROAD_UNIVERSE)
    print(f"symbols={len(symbols)}", flush=True)

    # Use exactly the CLI's panel construction so the report and the command
    # cannot disagree: trim_panel(95% coverage) -> usable_window(95% active) ->
    # clip.  An earlier version clipped with the qfq window and silently
    # dropped the newest raw session, which moved the 20-day rebalance phase
    # and changed the headline by 55pp.
    qfq_panel = trim_panel(load_panel(symbols, CACHE), min_coverage=0.95)
    start, end = usable_window(qfq_panel, min_active_ratio=0.95)
    benchmark = load_symbol("sh000300", CACHE, adjust="")["close"]

    raw_panel = load_raw(symbols, start, end)
    rebuilt_panel = {}
    fits = {}
    for symbol, raw in raw_panel.items():
        qfq_path = CACHE / f"{symbol}_qfq.parquet"
        if not qfq_path.exists():
            continue
        qfq = pd.read_parquet(qfq_path)
        qfq.index = pd.to_datetime(qfq.index)
        adjusted, fit = rebuild_frame(raw, qfq, symbol=symbol)
        rebuilt_panel[symbol] = adjusted
        fits[symbol] = fit

    common = sorted(set(raw_panel) & set(rebuilt_panel) & set(qfq_panel))
    # The raw download is one session ahead of qfq, so the three conventions
    # cannot share an identical last bar.  Use the common span for all three;
    # otherwise the 20-day rebalance phase shifts and the comparison stops
    # being a pure price-convention experiment (see RESEARCH_LOG §23.20).
    last_common = min(
        min(
            raw_panel[symbol].index.max(),
            rebuilt_panel[symbol].index.max(),
            qfq_panel[symbol].index.max(),
        )
        for symbol in common
    )
    panels = {
        "qfq_additive": {s: qfq_panel[s].loc[start:last_common] for s in common},
        "raw": {s: raw_panel[s].loc[start:last_common] for s in common},
        "rebuilt": {s: rebuilt_panel[s].loc[start:last_common] for s in common},
    }
    print(f"common symbols={len(common)} window={start.date()}..{last_common.date()}", flush=True)

    rows = []
    results = {}
    for label, panel in panels.items():
        strategy = build_strategy("low_vol_core", **PARAMS).generate(panel)
        weights = build_portfolio(
            strategy, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
        )
        result = Backtester(BacktestConfig(initial_cash=1_000_000.0)).run(
            panel, weights, benchmark=benchmark
        )
        results[label] = result
        metrics = result.metrics.to_dict()
        rows.append({"basis": label, **{k: float(v) for k, v in metrics.items()}})
        print(
            label,
            {
                k: round(float(metrics[k]), 4)
                for k in (
                    "total_return",
                    "cagr",
                    "sharpe",
                    "max_drawdown",
                    "annual_volatility",
                    "exposure",
                )
            },
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(OUT / "bases.csv", index=False, encoding="utf-8-sig")

    qfq_returns = results["qfq_additive"].returns
    raw_returns = results["raw"].returns
    rebuilt_returns = results["rebuilt"].returns
    index = qfq_returns.index.intersection(rebuilt_returns.index)
    correlations = {
        "qfq_vs_raw": float(qfq_returns.loc[index].corr(raw_returns.loc[index])),
        "qfq_vs_rebuilt": float(qfq_returns.loc[index].corr(rebuilt_returns.loc[index])),
        "raw_vs_rebuilt": float(raw_returns.loc[index].corr(rebuilt_returns.loc[index])),
    }

    per_symbol = []
    for symbol in common:
        fit = fits[symbol]
        per_symbol.append(
            {
                "symbol": symbol,
                "segments": len(fit.segments),
                "events": len(fit.events),
                "max_relative_residual": fit.max_relative_residual,
                "problems": "; ".join(fit.problems),
                "dividends": sum(1 for e in fit.events if abs(e.cash) > 1e-9),
                "splits": sum(1 for e in fit.events if abs(e.split - 1.0) > 1e-3),
            }
        )
    per_symbol_frame = pd.DataFrame(per_symbol).sort_values(
        "max_relative_residual", ascending=False
    )
    per_symbol_frame.to_csv(
        OUT / "symbol_fits.csv", index=False, encoding="utf-8-sig"
    )

    deltas = {}
    base = results["qfq_additive"].metrics.to_dict()
    for label in ("raw", "rebuilt"):
        other = results[label].metrics.to_dict()
        deltas[label] = {
            key: float(other[key] - base[key])
            for key in (
                "total_return",
                "cagr",
                "sharpe",
                "max_drawdown",
                "annual_volatility",
                "exposure",
            )
        }

    payload = {
        "symbols": len(common),
        "window": [str(start.date()), str(last_common.date())],
        "parameters": PARAMS,
        "initial_cash": 1_000_000.0,
        "metrics": {row["basis"]: row for row in rows},
        "deltas_vs_qfq": deltas,
        "return_correlations": correlations,
        "fit_problems": int((per_symbol_frame["problems"] != "").sum()),
        "note": (
            "qfq_additive is the cached Tencent series and is NOT a valid "
            "total-return series. rebuilt is the corrected multiplicative "
            "series. raw ignores dividends entirely. All three are in-sample "
            "on the same window; none is out-of-sample evidence."
        ),
    }
    write_json(payload, OUT / "summary.json")

    lines = [
        "# 复权口径对默认基线的影响",
        "",
        f"- 股票数：{len(common)}（128 只宽基中同时具备 qfq 与 raw 缓存的样本）",
        f"- 窗口：{start.date()} .. {last_common.date()}",
        f"- 策略：`low_vol_core` {PARAMS}",
        "- 初始资金：100 万；成本、整手、涨跌停、行业约束完全一致",
        "",
        "| 口径 | 总收益 | CAGR | Sharpe | 最大回撤 | 年化波动 | 平均仓位 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['basis']}` | {row['total_return']:.2%} | {row['cagr']:.2%} | "
            f"{row['sharpe']:.3f} | {row['max_drawdown']:.2%} | "
            f"{row['annual_volatility']:.2%} | {row['exposure']:.2%} |"
        )
    lines += [
        "",
        "## 日收益相关性",
        "",
        f"- qfq vs 原始价：{correlations['qfq_vs_raw']:.4f}",
        f"- qfq vs 重建：{correlations['qfq_vs_rebuilt']:.4f}",
        f"- 原始价 vs 重建：{correlations['raw_vs_rebuilt']:.4f}",
        "",
        "## 结论",
        "",
        "1. 加法前复权把日收益波动系统性放大，且放大倍数随股价水平变化——"
        "低价股被放大得最多。这会让低波动因子选中**被复权噪声污染得最少**的"
        "股票，而不是真正低波的股票。",
        "2. 原始价口径完全忽略分红，长期收益被系统性低估；它也不能作为正确答案。",
        "3. 重建口径同时保留分红与送转，是三者中唯一自洽的总收益序列。",
        f"4. 逐股拟合：{len(per_symbol_frame)} 只中 "
        f"{int((per_symbol_frame['problems'] != '').sum())} 只仍有残差告警，"
        "详见 `symbol_fits.csv`。",
        "5. 重建口径的最大回撤可能比 qfq 更深：噪声在部分时段掩盖了真实回撤。"
        "修正不是“指标全面改善”。",
        "",
        "**重要：本目录之前发布的所有 qfq 口径数字（含 +149.48%）都建立在"
        "加法复权序列上，不能作为修正后的结论。**",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(table[["basis", "total_return", "cagr", "sharpe", "max_drawdown"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
