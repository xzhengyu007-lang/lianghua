"""Brinson-style sector attribution on realised holdings.

Answers one question: of the portfolio's excess return over an equal-weight
universe, how much came from *sector weights* (allocation) and how much from
*stock picking inside a sector* (selection)?

Method (monthly, no look-ahead):

1. Take holdings at the last bar of month ``m`` (weights over equity).
2. Compute each name's return during month ``m+1`` from close-to-close prices.
3. Aggregate names into sectors; the benchmark defaults to the equal-weight
   pool that is actually tradable in that month, but any per-name weight frame
   can be passed in -- ``low_vol_benchmark`` builds the equal-weight book of the
   calmest third, which is the honest comparison for a low-volatility strategy.
4. Brinson-Fachler decomposition per sector::

       allocation  = (w_p,s - w_u,s) * (r_u,s - r_u)
       selection   =  w_u,s           * (r_p,s - r_u,s)
       interaction = (w_p,s - w_u,s)  * (r_p,s - r_u,s)

   Cash is treated as a sector with zero return and zero benchmark weight, so
   the three terms sum exactly to ``r_p - r_u`` (the cash term is the drag).

The decomposition is descriptive: it says where the money came from, not that a
sector tilt is repeatable. RESEARCH_LOG 23.38 pairs it with the walk-forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .sectors import sector_of


@dataclass
class AttributionResult:
    summary: dict
    sectors: pd.DataFrame
    monthly: pd.DataFrame
    warnings: list[str] = field(default_factory=list)

    def to_frames(self) -> dict[str, pd.DataFrame]:
        return {"sectors": self.sectors, "monthly": self.monthly}


def _monthly_returns(panel: dict[str, pd.DataFrame], start, end) -> pd.DataFrame:
    frames = {}
    for symbol, frame in panel.items():
        close = frame["close"].dropna()
        if len(close) < 2:
            continue
        frames[symbol] = close
    if not frames:
        raise ValueError("panel has no usable close series")
    prices = pd.DataFrame(frames).sort_index()
    month_end = prices.resample("ME").last()
    return month_end.pct_change(fill_method=None).loc[str(start):str(end)]


def low_vol_benchmark(
    panel: dict[str, pd.DataFrame],
    month_end: pd.DatetimeIndex,
    *,
    window: int = 60,
    fraction: float = 1.0 / 3.0,
    high: bool = False,
) -> pd.DataFrame:
    """Equal-weight the lowest (or highest) trailing-volatility names each month.

    Volatility uses only bars up to the month-end that decides the book, so the
    benchmark cannot peek at the month it is being scored on. This is the
    comparison that matters for a low-volatility strategy: beating the average
    stock is easy if the average stock is high-vol; beating the calm third is
    the actual claim.
    """
    closes = {symbol: frame["close"].dropna() for symbol, frame in panel.items()}
    prices = pd.DataFrame(closes).sort_index()
    if prices.empty:
        raise ValueError("panel has no usable close series")
    vol = prices.pct_change(fill_method=None).rolling(window).std()

    rows: dict[pd.Timestamp, pd.Series] = {}
    for ts in month_end:
        history = vol.loc[:ts]
        if history.empty:
            continue
        snapshot = history.iloc[-1].dropna()
        if len(snapshot) < 2:
            continue
        k = max(1, int(round(len(snapshot) * fraction)))
        picked = snapshot.nlargest(k).index if high else snapshot.nsmallest(k).index
        rows[pd.Timestamp(ts)] = pd.Series(1.0 / k, index=list(picked))
    if not rows:
        raise ValueError("no month-end has enough volatility history")
    return pd.DataFrame(rows).T.reindex(columns=prices.columns).fillna(0.0)


def _holdings_weights(
    positions: pd.DataFrame,
    month_end: pd.DatetimeIndex,
    equity: pd.Series,
) -> pd.DataFrame:
    """End-of-month portfolio weights over equity (cash is implied)."""
    if positions.empty:
        return pd.DataFrame(index=month_end)
    frame = positions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    # Keep zero-share rows: they are the explicit "I sold this" signal. Dropping
    # them and forward-filling afterwards carries every exited name to the end of
    # the sample, which silently multiplied the position book (weights summed to
    # 6x equity by 2026) and produced impossible +65% months.
    frame = frame.sort_values("date")
    pivot = frame.pivot_table(index="date", columns="symbol", values="market_value", aggfunc="sum")
    last_seen = {symbol: pivot[symbol].last_valid_index() for symbol in pivot.columns}
    snap = pivot.reindex(index=pivot.index.union(month_end)).ffill().reindex(month_end)
    for symbol, last in last_seen.items():
        if last is None:
            continue
        snap.loc[snap.index > last, symbol] = 0.0
    snap = snap.fillna(0.0)
    # ``reindex(...).ffill()`` cannot carry a value into a row that does not
    # exist yet, so the equity series must be forward-filled on its own index
    # first. Getting this wrong silently inflated weights on month-ends that
    # fell on a weekend (weight sums of 1.02) and broke the decomposition
    # identity by ~0.5% of NAV per year.
    eq_series = equity.sort_index().astype(float)
    eq = pd.Series(
        [eq_series.asof(ts) for ts in snap.index], index=snap.index, dtype=float
    )
    weights = snap.div(eq, axis=0)
    return weights.fillna(0.0)


def sector_attribution(
    positions: pd.DataFrame,
    panel: dict[str, pd.DataFrame],
    equity: pd.Series,
    *,
    start: str | None = None,
    end: str | None = None,
    benchmark: pd.DataFrame | None = None,
    benchmark_label: str = "等权全池",
) -> AttributionResult:
    if positions.empty:
        raise ValueError("positions are empty; run a backtest first")
    start = start or str(pd.to_datetime(positions["date"]).min().date())
    end = end or str(pd.to_datetime(positions["date"]).max().date())
    monthly = _monthly_returns(panel, start, end)
    if monthly.empty:
        raise ValueError("no monthly returns in the requested window")

    weights = _holdings_weights(positions, monthly.index, equity)
    over = weights.sum(axis=1)
    over = over[over > 1.0 + 1e-6]
    weight_warning = None
    if not over.empty:
        weight_warning = (
            f"{len(over)} month-end snapshot(s) hold more than 100% of equity "
            f"(max {over.max():.3f}); positions and equity disagree"
        )

    warnings: list[str] = [weight_warning] if weight_warning else []
    rows = []
    sector_totals: dict[str, dict[str, float]] = {}

    for month in monthly.index:
        position = list(monthly.index).index(month)
        month_alloc = month_sel = month_inter = 0.0
        if position == len(monthly.index) - 1:
            continue  # the last month has no follow-up return
        nxt = monthly.index[position + 1]
        w = weights.loc[month].dropna()
        w = w[w > 0]
        ret = monthly.loc[nxt]

        # A held name with no month-end price is suspended, not cash. Dropping
        # it made the position look like uninvested cash (and understated the
        # sector exposure); its return must be zero until it trades again.
        held = [s for s in w.index if s in ret.index]
        if not held:
            continue
        w = w.reindex(held).fillna(0.0)
        r_held_raw = ret.reindex(held)
        suspended_weight = float(w[r_held_raw.isna()].sum())
        r_held = r_held_raw.fillna(0.0)

        # universe: names with a return in this month
        universe = ret.dropna()
        if universe.empty:
            continue
        u_sector = pd.Series({s: sector_of(s) for s in universe.index})
        p_sector = pd.Series({s: sector_of(s) for s in held})

        if benchmark is None:
            bench = pd.Series(1.0 / len(universe), index=universe.index)
        else:
            if month in benchmark.index:
                bench = benchmark.loc[month].reindex(universe.index).fillna(0.0)
            else:
                bench = pd.Series(0.0, index=universe.index)
            if float(bench.sum()) <= 0:
                bench = pd.Series(1.0 / len(universe), index=universe.index)
            else:
                bench = bench / float(bench.sum())

        r_u_total = float((bench * universe).sum())
        r_p_total = float((w * r_held).sum())
        cash_weight = max(0.0, 1.0 - float(w.sum()))

        for sector in sorted(set(p_sector.unique()) | set(u_sector.unique())):
            p_names = p_sector[p_sector == sector].index
            u_names = u_sector[u_sector == sector].index
            w_p = float(w.reindex(p_names).sum())
            w_u = float(bench.reindex(u_names).sum())
            if w_u > 0:
                r_u = float(
                    (bench.reindex(u_names) * universe.reindex(u_names)).sum() / w_u
                )
            else:
                r_u = 0.0
            if w_p > 0:
                inner = w.reindex(p_names).fillna(0.0)
                r_p = float((inner * r_held.reindex(p_names).fillna(0.0)).sum() / w_p)
            else:
                r_p = r_u
            allocation = (w_p - w_u) * (r_u - r_u_total)
            selection = w_u * (r_p - r_u)
            interaction = (w_p - w_u) * (r_p - r_u)
            bucket = sector_totals.setdefault(
                sector, {"allocation": 0.0, "selection": 0.0, "interaction": 0.0,
                         "weight_p": 0.0, "weight_u": 0.0}
            )
            bucket["allocation"] += allocation
            bucket["selection"] += selection
            bucket["interaction"] += interaction
            month_alloc += allocation
            month_sel += selection
            month_inter += interaction
            bucket["weight_p"] += w_p
            bucket["weight_u"] += w_u

        cash_drag = -cash_weight * r_u_total
        rows.append(
            {
                "month": nxt.date(),
                "r_p": r_p_total,
                "r_u": r_u_total,
                "excess": r_p_total - r_u_total,
                "cash_weight": cash_weight,
                "suspended_weight": suspended_weight,
                "cash_drag": cash_drag,
                "allocation": month_alloc,
                "selection": month_sel,
                "interaction": month_inter,
                "residual": (r_p_total - r_u_total)
                - (month_alloc + month_sel + month_inter + cash_drag),
            }
        )

    monthly_frame = pd.DataFrame(rows)
    if monthly_frame.empty:
        raise ValueError("no complete month pairs; window too short")

    # A non-zero monthly residual means the identity is broken for that month,
    # which is a bug, not noise: the three terms plus cash drag must reconstruct
    # the excess exactly. Surface the worst offenders instead of hiding them in
    # a single annual number.
    offenders = monthly_frame[monthly_frame["residual"].abs() > 1e-6]
    if not offenders.empty:
        worst = offenders.reindex(offenders["residual"].abs().sort_values(ascending=False).index)
        detail = ", ".join(
            f"{pd.Timestamp(row.month).strftime('%Y-%m')} {row.residual:+.5f}"
            for row in worst.head(3).itertuples(index=False)
        )
        warnings.append(
            f"{len(offenders)} month(s) break the decomposition identity: {detail}"
        )

    sector_frame = pd.DataFrame(
        [
            {
                "sector": sector,
                "allocation": values["allocation"],
                "selection": values["selection"],
                "interaction": values["interaction"],
                "weight_p": values["weight_p"] / len(monthly_frame),
                "weight_u": values["weight_u"] / len(monthly_frame),
            }
            for sector, values in sorted(sector_totals.items())
        ]
    ).sort_values("allocation", ascending=False)

    cash_drag = float(monthly_frame["cash_drag"].sum())
    allocated = float(sector_frame["allocation"].sum()) + cash_drag
    selected = float(sector_frame["selection"].sum())
    interacted = float(sector_frame["interaction"].sum())
    excess = float(monthly_frame["excess"].sum())
    residual = excess - (allocated + selected + interacted)
    if abs(residual) > 1e-6:
        warnings.append(f"decomposition residual {residual:+.6f} (rounding only)")

    # Month-end weights held for a full month ignore the mid-month rebalances and
    # all trading costs, so the compounded monthly series is a gross upper bound
    # on realised NAV growth. Report the gap instead of letting a reader assume
    # the two are the same number.
    compounded = float((1.0 + monthly_frame["r_p"]).prod() - 1.0)
    snapshot_start = monthly.index[0]
    snapshot_end = pd.Timestamp(monthly_frame["month"].max())
    eq_series = equity.sort_index().astype(float)
    equity_return = float(eq_series.asof(snapshot_end) / eq_series.asof(snapshot_start) - 1.0)

    summary = {
        "benchmark": benchmark_label,
        "months": int(len(monthly_frame)),
        "portfolio_return": float(monthly_frame["r_p"].sum()),
        "portfolio_compounded": compounded,
        "equity_return": equity_return,
        "compounding_gap": compounded - equity_return,
        "universe_return": float(monthly_frame["r_u"].sum()),
        "excess": excess,
        "allocation": allocated,
        "selection": selected,
        "interaction": interacted,
        "cash_drag": cash_drag,
        "residual": residual,
        "avg_cash_weight": float(monthly_frame["cash_weight"].mean()),
        "avg_suspended_weight": float(monthly_frame["suspended_weight"].mean()),
    }
    if summary["avg_suspended_weight"] > 0.05:
        warnings.append(
            "held names with no month-end return treated as zero-return positions; "
            f"average {summary['avg_suspended_weight']:.1%} of equity"
        )
    return AttributionResult(summary=summary, sectors=sector_frame, monthly=monthly_frame, warnings=warnings)


def load_attribution_inputs(report_dir: Path | str) -> tuple[pd.DataFrame, pd.Series]:
    """Read the positions/equity pair written by a backtest or walk-forward run."""
    report_dir = Path(report_dir)
    positions_path = report_dir / "positions.csv"
    equity_path = report_dir / "equity.csv"
    if not positions_path.exists():
        raise FileNotFoundError(positions_path)
    if not equity_path.exists():
        raise FileNotFoundError(equity_path)
    positions = pd.read_csv(positions_path, encoding="utf-8-sig")
    positions["date"] = pd.to_datetime(positions["date"])
    equity_frame = pd.read_csv(equity_path, encoding="utf-8-sig", index_col=0, parse_dates=True)
    if equity_frame.empty:
        raise ValueError(f"{equity_path} is empty")
    equity = equity_frame.iloc[:, 0].astype(float)
    equity.index = pd.to_datetime(equity.index)
    return positions, equity


def _pct(value: float, digits: int = 2) -> str:
    return f"{value * 100:+.{digits}f}%"


def _share(value: float, total: float) -> str:
    if abs(total) < 1e-12:
        return "-"
    return f"{value / total * 100:.1f}%"


def render_markdown(
    result: AttributionResult,
    *,
    title: str = "行业归因",
    meta: dict | None = None,
) -> str:
    summary = result.summary
    excess = summary["excess"]
    lines = [f"# {title}", ""]
    if meta:
        for key, value in meta.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines += [
        "## 超额来源（逐月累加）",
        "",
        "组合收益对等权全池的超额被拆成配置（行业权重偏离）、选股（行业内选股）",
        "与交互三项；现金按零收益行业处理，因此其拖累计入配置。三项之和等于超额。",
        "",
        "| 项目 | 累计 | 占超额 |",
        "|---|---:|---:|",
        f"| 组合收益（逐月算术累加） | {_pct(summary['portfolio_return'])} | - |",
        f"| 等权基准 | {_pct(summary['universe_return'])} | - |",
        f"| 超额 | {_pct(excess)} | 100.0% |",
        f"| 行业配置 | {_pct(summary['allocation'])} | {_share(summary['allocation'], excess)} |",
        f"| 行业内选股 | {_pct(summary['selection'])} | {_share(summary['selection'], excess)} |",
        f"| 交互 | {_pct(summary['interaction'])} | {_share(summary['interaction'], excess)} |",
        f"| 其中现金拖累 | {_pct(summary['cash_drag'])} | {_share(summary['cash_drag'], excess)} |",
        f"| 残差 | {summary['residual']:+.2e} | - |",
        "",
        f"覆盖 {summary['months']} 个完整月份，平均现金权重 {summary['avg_cash_weight'] * 100:.1f}%，"
        f"平均停牌/缺失收益权重 {summary.get('avg_suspended_weight', 0.0) * 100:.1f}%。",
        "",
        f"月度收益复利 {_pct(summary['portfolio_compounded'])}，同期净值 {_pct(summary['equity_return'])}，"
        f"差 {_pct(summary['compounding_gap'])}。差额来自月中调仓（月末权重再持有一整个月）"
        "与全部未计的成本，因此上表是**毛口径上界**。",
        "",
        "## 分行业",
        "",
        "| 行业 | 平均组合权重 | 平均基准权重 | 配置 | 选股 | 交互 | 合计 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result.sectors.itertuples(index=False):
        total = row.allocation + row.selection + row.interaction
        lines.append(
            f"| {row.sector} | {row.weight_p * 100:.1f}% | {row.weight_u * 100:.1f}% | "
            f"{_pct(row.allocation)} | {_pct(row.selection)} | {_pct(row.interaction)} | {_pct(total)} |"
        )

    monthly = result.monthly
    if not monthly.empty:
        best = monthly.nlargest(3, "excess")
        worst = monthly.nsmallest(3, "excess")
        lines += [
            "",
            "## 极值月份",
            "",
            "| 月份 | 组合 | 基准 | 超额 | 现金权重 |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in pd.concat([best, worst]).itertuples(index=False):
            lines.append(
                f"| {pd.Timestamp(row.month).strftime('%Y-%m')} | {_pct(row.r_p)} | {_pct(row.r_u)} | "
                f"{_pct(row.excess)} | {row.cash_weight * 100:.0f}% |"
            )

    if result.warnings:
        lines += ["", "## 提示", ""]
        lines += [f"- {item}" for item in result.warnings]
    lines.append("")
    return "\n".join(lines)


def save_attribution(
    result: AttributionResult,
    output: Path | str,
    *,
    title: str = "行业归因",
    meta: dict | None = None,
) -> Path:
    from .report import write_json

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result.sectors.to_csv(output / "sectors.csv", encoding="utf-8-sig", index=False)
    result.monthly.to_csv(output / "monthly.csv", encoding="utf-8-sig", index=False)
    write_json(
        {
            "summary": result.summary,
            "warnings": result.warnings,
            "meta": meta or {},
            "sectors": result.sectors.to_dict(orient="records"),
        },
        output / "summary.json",
    )
    (output / "report.md").write_text(
        render_markdown(result, title=title, meta=meta), encoding="utf-8"
    )
    return output
