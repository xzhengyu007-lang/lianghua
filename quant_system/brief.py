"""One-page daily brief: data health, next-open book, risk snapshot, ledger.

The brief is a presentation layer only. Every number it prints comes from
``latest_signal`` / ``portfolio`` / ``paper``; it never invents a target or a
statistic that the backtest would not produce, so a brief and a backtest cannot
disagree about the book.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RiskConfig
from .paper import Ledger, positions_frame, summarize
from .sectors import sector_of
from .signals import DailySignal, latest_signal

TRADING_DAYS = 252


@dataclass
class Brief:
    as_of: pd.Timestamp
    signal: DailySignal
    data: dict
    risk: dict
    ledger: dict | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "as_of": str(self.as_of.date()),
            "data": self.data,
            "signal": self.signal.to_dict(),
            "risk": self.risk,
            "ledger": self.ledger,
            "notes": self.notes,
        }


def data_health(
    panel: dict[str, pd.DataFrame],
    lookback: int = 5,
    retired_after_days: int = 90,
    today: pd.Timestamp | None = None,
) -> dict:
    last = {}
    for symbol, frame in panel.items():
        close = frame["close"].dropna() if "close" in frame else pd.Series(dtype=float)
        if len(close):
            last[symbol] = close.index.max()
    if not last:
        raise ValueError("panel has no usable close series")
    as_of = max(last.values())
    # A point-in-time universe contains names that genuinely stopped trading.
    # Counting those as "stale data" would report 47% coverage on a healthy pool
    # (126 live + 141 delisted).  Split them: only names whose last bar is within
    # ``retired_after_days`` are expected to be trading today.
    retired_cutoff = as_of - pd.Timedelta(days=retired_after_days)
    live = {s: d for s, d in last.items() if d >= retired_cutoff}
    retired = {s: d for s, d in last.items() if d < retired_cutoff}
    fresh = [s for s, d in live.items() if d == as_of]
    stale = sorted(((s, d) for s, d in live.items() if d < as_of), key=lambda item: item[1])
    recent = as_of - pd.Timedelta(days=lookback * 2)
    lagging = [s for s, d in live.items() if d < recent]
    age_days = age_bdays = None
    if today is not None:
        today = pd.Timestamp(today).normalize()
        age_days = int((today - as_of).days)
        age_bdays = int(np.busday_count(as_of.date(), today.date()))
    return {
        "as_of": str(as_of.date()),
        "as_of_age_days": age_days,
        "as_of_age_bdays": age_bdays,
        "symbols": len(last),
        "live_symbols": len(live),
        "retired_symbols": len(retired),
        "on_latest_bar": len(fresh),
        "coverage": round(len(fresh) / len(live), 4) if live else 0.0,
        "stale_count": len(stale),
        "stale_examples": [{"symbol": s, "last_bar": str(d.date())} for s, d in stale[:5]],
        "lagging_count": len(lagging),
    }


def risk_snapshot(
    weights: pd.Series,
    panel: dict[str, pd.DataFrame],
    benchmark: pd.Series | None = None,
    *,
    window: int = 60,
) -> dict:
    """Ex-ante volatility, beta and concentration of a single target book.

    Volatility and beta use only bars at or before the signal close; weights are
    held fixed, so this is a risk *of the book as it would be held*, not a
    forecast of P&L.
    """
    held = weights[weights > 0].sort_values(ascending=False)
    if held.empty:
        return {"positions": 0, "note": "empty book"}

    frames = {}
    for symbol in held.index:
        frame = panel.get(symbol)
        if frame is None or frame.empty:
            continue
        close = frame["close"].dropna()
        if len(close) < 20:
            continue
        frames[symbol] = close
    if not frames:
        return {"positions": int(len(held)), "note": "no usable price history"}

    prices = pd.DataFrame(frames).sort_index().ffill().tail(window + 1)
    returns = prices.pct_change().dropna()
    if returns.empty:
        return {"positions": int(len(held)), "note": "no overlapping returns"}

    cols = [c for c in held.index if c in returns.columns]
    w = held.reindex(cols).fillna(0.0).to_numpy(dtype=float)
    cov = returns[cols].cov().to_numpy(dtype=float)
    variance = float(w @ cov @ w)
    vol = float(np.sqrt(max(variance, 0.0)) * np.sqrt(TRADING_DAYS))

    beta = None
    beta_se = None
    beta_days = 0
    if benchmark is not None and len(benchmark) > window:
        bench = benchmark.loc[: prices.index.max()].pct_change().dropna()
        bench = bench.reindex(returns.index)
        proxy = (returns[cols] @ w).reindex(bench.index)
        joined = pd.concat([proxy.rename("proxy"), bench.rename("bench")], axis=1).dropna()
        if len(joined) > 20 and joined["bench"].var() > 0:
            beta = float(joined["proxy"].cov(joined["bench"]) / joined["bench"].var())
            # A point beta from 59 sessions is extremely noisy: a book whose
            # true beta is +0.3 can print -0.26. Publish the standard error so
            # the daily brief cannot imply more precision than the window has.
            residual = joined["proxy"] - beta * joined["bench"]
            dof = max(len(joined) - 2, 1)
            sxx = float(((joined["bench"] - joined["bench"].mean()) ** 2).sum())
            beta_se = (
                float(np.sqrt(residual.pow(2).sum() / dof / sxx)) if sxx > 0 else None
            )
            beta_days = int(len(joined))

    sectors: dict[str, float] = {}
    for symbol, weight in held.items():
        name = sector_of(symbol)
        sectors[name] = sectors.get(name, 0.0) + float(weight)
    sector_rows = sorted(sectors.items(), key=lambda item: -item[1])
    return {
        "positions": int(len(held)),
        "gross": round(float(held.sum()), 4),
        "ex_ante_vol": round(vol, 4),
        "beta": None if beta is None else round(beta, 3),
        "beta_se": None if beta_se is None else round(beta_se, 3),
        "beta_days": int(beta_days),
        "window": int(window),
        "top3_weight": round(float(held.head(3).sum()), 4),
        "sectors": [{"sector": s, "weight": round(v, 4)} for s, v in sector_rows],
    }


def _ledger_block(ledger_path: Path, target: pd.Series, prices: pd.Series) -> dict | None:
    if not ledger_path.exists():
        return None
    ledger = Ledger.load(ledger_path)
    info = summarize(ledger, prices)
    frame = positions_frame(ledger, prices)
    current: dict[str, float] = {}
    if not frame.empty and "symbol" in frame and "market_value" in frame:
        total = float(frame["market_value"].fillna(0.0).sum()) or 1.0
        for _, row in frame.iterrows():
            current[str(row["symbol"])] = float(row["market_value"]) / total
    drift = []
    for symbol in sorted(set(current) | set(target[target > 0].index)):
        diff = float(target.get(symbol, 0.0)) - float(current.get(symbol, 0.0))
        if abs(diff) >= 0.005:
            drift.append(
                {
                    "symbol": symbol,
                    "target": round(float(target.get(symbol, 0.0)), 4),
                    "current": round(float(current.get(symbol, 0.0)), 4),
                    "diff": round(diff, 4),
                }
            )
    drift.sort(key=lambda item: -abs(item["diff"]))
    return {"summary": info, "drift": drift[:10]}


def build_brief(
    panel: dict[str, pd.DataFrame],
    strategy_name: str,
    *,
    params: dict | None = None,
    benchmark: pd.Series | None = None,
    risk: RiskConfig | None = None,
    equity: float | None = None,
    ledger_path: str | Path | None = None,
    use_regime: bool = False,
    use_vol_target: bool = True,
    today: pd.Timestamp | None = None,
    max_age_bdays: int = 5,
) -> Brief:
    data = data_health(panel, today=today)
    signal = latest_signal(
        panel,
        strategy_name,
        params=params,
        benchmark=benchmark,
        risk=risk or RiskConfig(),
        equity=equity,
        use_regime=use_regime,
        use_vol_target=use_vol_target,
    )
    risk_block = risk_snapshot(signal.target_weights, panel, benchmark)
    prices = {}
    for symbol in signal.target_weights[signal.target_weights > 0].index:
        frame = panel.get(symbol)
        if frame is not None and not frame.empty:
            prices[symbol] = float(frame["close"].dropna().iloc[-1])
    ledger_block = None
    if ledger_path is not None:
        ledger_block = _ledger_block(Path(ledger_path), signal.target_weights, pd.Series(prices))
    notes = []
    if data["coverage"] < 0.95:
        notes.append(
            f"只有 {data['coverage']:.1%} 的在册标的最新交易日有数据，横截面偏薄，"
            "信号按现有横截面计算。"
        )
    if data["stale_count"]:
        notes.append(
            f"{data['stale_count']} 只在册标的最新 bar 早于 {data['as_of']}"
            "（停牌或行情未更新），这些标的不会入选。"
        )
    age = data.get("as_of_age_bdays")
    if age is not None and age > max_age_bdays:
        notes.append(
            f"最新 bar 是 {age} 个交易日之前（as_of {data['as_of']}），"
            "数据源可能未更新或已停更；这份信号不要直接照做。"
        )
    if signal.warnings:
        notes.extend(signal.warnings)
    return Brief(
        as_of=signal.as_of,
        signal=signal,
        data=data,
        risk=risk_block,
        ledger=ledger_block,
        notes=notes,
    )


def _beta_line(risk: dict) -> str:
    """Render beta with its standard error; a bare point estimate lies."""
    beta = risk.get("beta")
    if beta is None:
        return "-"
    se = risk.get("beta_se")
    days = risk.get("beta_days")
    if se is None:
        return f"{beta:+.2f}"
    return f"{beta:+.2f} ± {se:.2f}（{days} 日，标准误）"


def _pct(value: float | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not np.isfinite(value):
        return "-"
    return f"{value * 100:.2f}%"


def render_markdown(brief: Brief) -> str:
    signal = brief.signal
    lines = [
        f"# 每日简报 {brief.as_of.date()}",
        "",
        "## 数据",
        "",
        f"- 最新 bar：{brief.data['as_of']}，{brief.data['on_latest_bar']}/"
        f"{brief.data.get('live_symbols', brief.data['symbols'])} 只在册标的有数据"
        f"（覆盖率 {brief.data['coverage']:.1%}）",
        f"- 在册但落后：{brief.data['stale_count']} 只（其中超过 10 天未更新 "
        f"{brief.data['lagging_count']} 只）；已退市/长期停牌："
        f"{brief.data.get('retired_symbols', 0)} 只",
        "",
        "## 信号（下一交易日开盘执行）",
        "",
        f"- 策略：`{signal.strategy}`，参数 `{signal.params}`",
        f"- 目标毛仓位：{_pct(signal.gross_exposure)}，持仓 {len(signal.rows)} 只"
        + (f"，按账户权益 {signal.sizing_equity:,.0f} 定量" if signal.sizing_equity else ""),
    ]
    if len(signal.rows):
        lines += ["", "| 代码 | 权重 | 股数 | 金额 | 行业 |", "|---|---:|---:|---:|---|"]
        for _, row in signal.rows.iterrows():
            lines.append(
                f"| {row['symbol']} | {_pct(float(row.get('weight', 0.0)))} | "
                f"{int(row.get('shares', 0) or 0)} | "
                f"{float(row.get('amount', 0.0) or 0.0):,.0f} | {row.get('sector', '')} |"
            )
    lines += ["", "## 风险", ""]
    if brief.risk.get("positions"):
        lines += [
            f"- 事前波动（{brief.risk.get('window', 60)} 日协方差）："
            f"{_pct(brief.risk.get('ex_ante_vol'))}",
            f"- 对基准 beta：{_beta_line(brief.risk)}",
            f"- 前三大权重合计：{_pct(brief.risk.get('top3_weight'))}",
            "- 行业分布："
            + "、".join(
                f"{item['sector']} {_pct(item['weight'])}"
                for item in brief.risk.get("sectors", [])[:5]
            ),
        ]
    else:
        lines.append(f"- {brief.risk.get('note', 'no risk estimate available')}")
    if brief.ledger is not None:
        summary = brief.ledger["summary"]
        lines += ["", "## 模拟盘账本", ""]
        lines.append(
            f"- 权益 {summary.get('equity', 0):,.0f}，现金 {summary.get('cash', 0):,.0f}，"
            f"持仓 {summary.get('positions', 0)} 只"
        )
        if brief.ledger["drift"]:
            lines += ["", "| 代码 | 目标 | 当前 | 差额 |", "|---|---:|---:|---:|"]
            for row in brief.ledger["drift"]:
                lines.append(
                    f"| {row['symbol']} | {_pct(row['target'])} | {_pct(row['current'])} | "
                    f"{_pct(row['diff'])} |"
                )
        else:
            lines.append("- 账本与目标一致（差额 < 0.5%）。")
    if brief.notes:
        lines += ["", "## 提示", ""]
        lines += [f"- {note}" for note in brief.notes]
    lines += ["", "> 信号在收盘后生成，订单在下一交易日开盘成交；本简报不做任何额外择时。"]
    return "\n".join(lines) + "\n"


def render_html(brief: Brief) -> str:
    body = render_markdown(brief)
    escaped = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>每日简报 {brief.as_of.date()}</title>
<style>
body {{ font-family: -apple-system, "Segoe UI", sans-serif; margin: 32px auto;
       max-width: 900px; color: #1c1d22; line-height: 1.6; }}
pre {{ background: #f6f7f9; padding: 20px; border-radius: 10px; white-space: pre-wrap;
      font-family: ui-monospace, Consolas, monospace; font-size: 13px; }}
</style></head>
<body><pre>{escaped}</pre></body></html>
"""


def save_brief(brief: Brief, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "brief.md").write_text(render_markdown(brief), encoding="utf-8")
    (out / "brief.html").write_text(render_html(brief), encoding="utf-8")
    (out / "brief.json").write_text(
        json.dumps(brief.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out
