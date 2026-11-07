"""Self-contained HTML dashboard for backtest and walk-forward results."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .backtest import BacktestResult
from .metrics import drawdown_series, monthly_returns

CSS = """
:root { --ink:#111827; --muted:#6b7280; --line:#e5e7eb; --pos:#0b6e4f; --neg:#c0392b;
        --bg:#f8fafc; --card:#ffffff; }
* { box-sizing: border-box; }
body { margin:0; padding:32px; font-family:"Segoe UI","Microsoft YaHei",sans-serif;
       color:var(--ink); background:var(--bg); }
h1 { margin:0 0 4px; font-size:26px; }
h2 { margin:32px 0 12px; font-size:18px; border-bottom:1px solid var(--line); padding-bottom:6px; }
.sub { color:var(--muted); font-size:13px; margin-bottom:24px; }
.cards { display:flex; flex-wrap:wrap; gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:14px 18px; min-width:150px; }
.card .k { color:var(--muted); font-size:12px; letter-spacing:.04em; text-transform:uppercase; }
.card .v { font-size:22px; font-weight:600; margin-top:4px; font-variant-numeric:tabular-nums; }
.pos { color:var(--pos); } .neg { color:var(--neg); }
.panel { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }
table { width:100%; border-collapse:collapse; font-size:13px;
        font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:7px 10px; border-bottom:1px solid var(--line); }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase; }
tr:hover td { background:#f3f7fb; }
.heat td { text-align:center; border-radius:4px; color:#111; font-weight:500; }
.note { background:#fffbeb; border:1px solid #fde68a; border-radius:8px;
        padding:12px 16px; font-size:13px; color:#78350f; }
.warn { background:#fef2f2; border:1px solid #fecaca; border-radius:8px;
        padding:12px 16px; font-size:13px; color:#7f1d1d; margin:16px 0; }
.warn ul { margin:6px 0 0 18px; }
footer { margin-top:40px; color:var(--muted); font-size:12px; }
"""


def _fmt_pct(value: float, digits: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value * 100:.{digits}f}%"


def _fmt_num(value: float, digits: int = 2) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _tone(value: float) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""
    return "pos" if value >= 0 else "neg"


def _polyline(values: Iterable[float], width: int, height: int, pad: int = 8) -> str:
    values = [float(v) for v in values]
    if not values:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo or 1.0
    step = (width - 2 * pad) / max(len(values) - 1, 1)
    points = []
    for i, value in enumerate(values):
        x = pad + i * step
        y = pad + (hi - value) / span * (height - 2 * pad)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _equity_svg(series: dict[str, pd.Series], width: int = 980, height: int = 340) -> str:
    normalized = {}
    for name, value in series.items():
        value = value.dropna()
        if len(value) < 2 or value.iloc[0] == 0:
            continue
        normalized[name] = value / value.iloc[0]
    if not normalized:
        return "<p>No equity data.</p>"
    all_values = pd.concat(normalized.values())
    lo, hi = float(all_values.min()), float(all_values.max())
    pad = 8
    span = hi - lo or 1.0
    segments = []
    colors = {"strategy": "#0b6e4f", "benchmark": "#6b7280"}
    for name, value in normalized.items():
        step = (width - 2 * pad) / max(len(value) - 1, 1)
        points = " ".join(
            f"{pad + i * step:.2f},{pad + (hi - float(v)) / span * (height - 2 * pad):.2f}"
            for i, v in enumerate(value.to_numpy())
        )
        color = colors.get(name, "#2563eb")
        segments.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round"/>'
        )
    labels = "".join(
        f'<text x="{pad}" y="{pad + (hi - level) / span * (height - 2 * pad):.0f}" '
        f'font-size="10" fill="#9ca3af">{level:.2f}</text>'
        for level in (lo, (lo + hi) / 2, hi)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Equity curve">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" '
        f'stroke="#e5e7eb"/>'
        f"{segments}{labels}</svg>"
    )


def _drawdown_svg(equity: pd.Series, width: int = 980, height: int = 180) -> str:
    dd = drawdown_series(equity).fillna(0.0)
    if dd.empty:
        return ""
    lowest = float(dd.min())
    span = abs(lowest) or 1.0
    pad = 8
    step = (width - 2 * pad) / max(len(dd) - 1, 1)
    points = [f"{pad},{pad}"]
    for i, value in enumerate(dd.to_numpy()):
        points.append(f"{pad + i * step:.2f},{pad + abs(float(value)) / span * (height - 2 * pad):.2f}")
    points.append(f"{width - pad},{pad}")
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">'
        f'<rect width="100%" height="100%" fill="white"/>'
        f'<polygon points="{" ".join(points)}" fill="#f3b6ad" stroke="#c0392b" stroke-width="1"/>'
        f'<text x="{pad}" y="14" font-size="10" fill="#9ca3af">{lowest * 100:.1f}%</text>'
        f"</svg>"
    )


def _heat_table(equity: pd.Series) -> str:
    table = monthly_returns(equity)
    if table.empty:
        return "<p>No monthly data.</p>"
    pivot = table.pivot(index="year", columns="month", values="return") * 100
    months = list(range(1, 13))
    head = "<th>Year</th>" + "".join(f"<th>{m}</th>" for m in months) + "<th>Year</th>"
    rows = []
    for year, row in pivot.iterrows():
        cells = []
        for month in months:
            value = row.get(month, np.nan)
            if pd.isna(value):
                cells.append('<td class="heat" style="background:#f9fafb;color:#d1d5db">·</td>')
                continue
            intensity = min(abs(value) / 12.0, 1.0)
            if value >= 0:
                color = f"rgba(11,110,79,{0.10 + 0.55 * intensity:.2f})"
            else:
                color = f"rgba(192,57,43,{0.10 + 0.55 * intensity:.2f})"
            cells.append(f'<td class="heat" style="background:{color}">{value:.1f}</td>')
        total = float(np.nansum(row.to_numpy()))
        rows.append(
            f"<tr><td>{int(year)}</td>{''.join(cells)}"
            f'<td class="{_tone(total)}">{total:.1f}</td></tr>'
        )
    return f'<table class="heat"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def _frame_table(frame: pd.DataFrame, *, pct_columns: Iterable[str] = (), max_rows: int = 20) -> str:
    if frame is None or frame.empty:
        return "<p>No rows.</p>"
    view = frame.head(max_rows).copy()
    pct_columns = set(pct_columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    rows = []
    for _, row in view.iterrows():
        cells = []
        for column in view.columns:
            value = row[column]
            if column in pct_columns:
                text = _fmt_pct(value)
                tone = _tone(value)
            elif isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                text = _fmt_num(value)
                tone = ""
            else:
                text = str(value)
                tone = ""
            cells.append(f'<td class="{tone}">{html.escape(text)}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_dashboard(
    path: Path,
    *,
    title: str,
    equity: pd.Series,
    metrics: dict,
    benchmark_equity: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    positions: pd.DataFrame | None = None,
    windows: pd.DataFrame | None = None,
    analytics: dict | None = None,
    notes: list[str] | None = None,
    caveats: list[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    trade_count = int(len(trades)) if trades is not None else int(metrics.get("trades", 0) or 0)
    cards = [
        ("Total return", _fmt_pct(metrics.get("total_return")), metrics.get("total_return")),
        ("CAGR", _fmt_pct(metrics.get("cagr")), metrics.get("cagr")),
        ("Volatility", _fmt_pct(metrics.get("annual_volatility")), None),
        ("Sharpe", _fmt_num(metrics.get("sharpe")), metrics.get("sharpe")),
        ("Max drawdown", _fmt_pct(metrics.get("max_drawdown")), metrics.get("max_drawdown")),
        ("Turnover (interval, one-way)", f"{_fmt_num(metrics.get('turnover'))}x", None),
        ("Exposure", _fmt_pct(metrics.get("exposure")), None),
        ("Trades", _fmt_num(trade_count, 0), None),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{html.escape(k)}</div>'
        f'<div class="v {_tone(v) if v is not None else ""}">{html.escape(text)}</div></div>'
        for k, text, v in cards
    )
    series = {"strategy": equity}
    if benchmark_equity is not None:
        series["benchmark"] = benchmark_equity
    sections = [
        f"<h1>{html.escape(title)}</h1>",
        f'<div class="sub">{html.escape(str(equity.index.min().date()))} → '
        f'{html.escape(str(equity.index.max().date()))} · generated '
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}</div>",
        f'<div class="cards">{card_html}</div>',
    ]
    if caveats:
        sections.append(
            '<div class="warn"><strong>Caveats — read before using these numbers</strong><ul>'
            + "".join(f"<li>{html.escape(str(c))}</li>" for c in caveats)
            + "</ul></div>"
        )
    sections += [
        "<h2>Equity curve</h2>",
        f'<div class="panel">{_equity_svg(series)}</div>',
        "<h2>Drawdown</h2>",
        f'<div class="panel">{_drawdown_svg(equity)}</div>',
        "<h2>Monthly returns (%)</h2>",
        f'<div class="panel">{_heat_table(equity)}</div>',
    ]
    if windows is not None and not windows.empty:
        display = windows.copy()
        display["params"] = display["params"].astype(str)
        sections += ["<h2>Walk-forward windows</h2>",
                     f'<div class="panel">{_frame_table(display, pct_columns=["test_return", "test_max_drawdown"])}</div>']
    if analytics:
        rows = [
            {"factor": name, "mean_ic": v.get("mean_ic"), "ic_ir": v.get("ic_ir"),
             "positive_ratio": v.get("positive_ratio"), "t_stat": v.get("t_stat")}
            for name, v in analytics.items()
        ]
        sections += ["<h2>Factor diagnostics</h2>",
                     f'<div class="panel">{_frame_table(pd.DataFrame(rows), pct_columns=["positive_ratio"])}</div>']
    if trades is not None and not trades.empty:
        recent = trades.sort_values("date", ascending=False).head(25)
        sections += ["<h2>Recent trades</h2>",
                     f'<div class="panel">{_frame_table(recent, pct_columns=[])}</div>']
    if positions is not None and not positions.empty:
        latest_date = positions["date"].max()
        latest = positions[(positions["date"] == latest_date) & (positions["shares"] > 0)]
        if latest.empty:
            latest = positions[positions["date"] == latest_date]
        sections += ["<h2>Holdings at last bar</h2>",
                     f'<div class="panel">{_frame_table(latest, pct_columns=["weight"])}</div>']
    if notes:
        sections += ["<h2>Notes</h2>", '<div class="note"><ul>' +
                     "".join(f"<li>{html.escape(n)}</li>" for n in notes) + "</ul></div>"]
    sections.append(
        "<footer>Generated by quant_system · "
        f"{html.escape(str(equity.index.max().date()))} · "
        "signals at close, execution at next open</footer>"
    )
    document = (
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )
    path.write_text(document, encoding="utf-8")
    return path