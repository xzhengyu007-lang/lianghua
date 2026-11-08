"""Reconciliation and the daily statement -- the "never silently fix the books" rules.

Red line #8: a difference between our ledger and the broker statement must raise an
**alert**, and the ledger must not be adjusted to make the numbers agree.  That
rule exists because the tempting fix (silently writing the broker's number into the
ledger) erases the evidence of whatever caused the difference -- a missed fill, a
duplicated order, a fee that changed.

So this module is read-only with respect to the ledger:

* :func:`reconcile` compares and classifies; it never writes.
* :func:`daily_statement` produces the end-of-day numbers (positions, fills, fees,
  realised PnL, equity) from the ledger and a price vector.
* :func:`write_alerts` persists the differences as a file **and** returns a list of
  human lines, so an unattended run cannot drop them on the floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..paper import Ledger, Position, positions_frame, reconcile as paper_reconcile, summarize


@dataclass
class ReconciliationReport:
    trade_date: str
    ok: bool = True
    differences: list[dict] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    broker_cash: float | None = None
    ledger_cash: float | None = None

    def to_dict(self) -> dict:
        return {
            "trade_date": self.trade_date,
            "ok": bool(self.ok),
            "ledger_cash": self.ledger_cash,
            "broker_cash": self.broker_cash,
            "differences": self.differences,
            "alerts": self.alerts,
            "rule": "差异只报警，绝不自动改账本（红线 8）",
        }

    def markdown(self) -> str:
        lines = ["## 对账结果（%s）" % self.trade_date, ""]
        if self.ok:
            lines.append("- ✅ 账本与券商流水一致（持仓、现金）")
            return "\n".join(lines) + "\n"
        lines.append("- ❌ **存在差异 %d 项**：必须人工处理，账本不会被自动修改" % len(self.differences))
        lines.append("")
        lines.append("| 项目 | 账本 | 券商 | 差异 |")
        lines.append("|---|---:|---:|---:|")
        for row in self.differences:
            lines.append(
                "| %s | %s | %s | %s |"
                % (row["item"], row["ledger"], row["broker"], row["difference"])
            )
        lines.append("")
        for alert in self.alerts:
            lines.append("- ⚠ %s" % alert)
        return "\n".join(lines) + "\n"


def reconcile(
    ledger: Ledger,
    broker_positions: dict[str, int],
    *,
    broker_cash: float | None = None,
    tolerance: float = 0.0,
    trade_date: str | None = None,
) -> ReconciliationReport:
    """Compare the ledger with a broker statement and classify every difference."""
    day = trade_date or str(ledger.as_of or "")
    frame: pd.DataFrame = paper_reconcile(
        ledger, broker_positions, broker_cash=broker_cash, tolerance=tolerance
    )
    report = ReconciliationReport(
        trade_date=day,
        ledger_cash=round(float(ledger.cash), 2),
        broker_cash=None if broker_cash is None else round(float(broker_cash), 2),
    )
    if frame.empty:
        return report

    report.ok = False
    for row in frame.itertuples(index=False):
        item = str(row.item)
        difference = float(row.difference)
        entry = {
            "item": item,
            "ledger": row.ledger if isinstance(row.ledger, (int, float)) else str(row.ledger),
            "broker": row.broker if isinstance(row.broker, (int, float)) else str(row.broker),
            "difference": difference if item == "cash" else int(difference),
        }
        report.differences.append(entry)
        if item == "cash":
            report.alerts.append(
                "现金差 %.2f 元（账本 %.2f / 券商 %.2f）：先查费用与未登记的成交，再谈调账"
                % (difference, float(row.ledger), float(row.broker))
            )
        else:
            direction = "账本多" if difference > 0 else "账本少"
            report.alerts.append(
                "%s %s %d 股：券商端可能漏登记成交，或有一笔未回报的委托"
                % (item, direction, abs(int(difference)))
            )
    return report


def write_alerts(report: ReconciliationReport, output_dir: str | Path) -> Path:
    """Persist the report next to the artifacts it belongs to."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / "reconciliation.md"
    path.write_text(report.markdown(), encoding="utf-8")
    (target / "reconciliation.json").write_text(
        __import__("json").dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def daily_statement(
    ledger: Ledger,
    prices: pd.Series,
    *,
    trade_date: str | None = None,
) -> dict:
    """End-of-day statement: positions at the closing mark, today's fills, fees, equity.

    ``prices`` must be the *close* vector for ``trade_date``; using a live quote
    would make the statement disagree with the next day's opening cash.
    """
    day = trade_date or str(ledger.as_of or "")
    frames = positions_frame(ledger, prices)
    summary = summarize(ledger, prices)
    trades = [trade for trade in ledger.trades if str(trade.get("date")) == day]
    fees = {
        "commission": round(sum(float(t.get("commission", 0.0)) for t in trades), 2),
        "stamp_duty": round(sum(float(t.get("stamp_duty", 0.0)) for t in trades), 2),
        "transfer_fee": round(sum(float(t.get("transfer_fee", 0.0)) for t in trades), 2),
    }
    return {
        "trade_date": day,
        "cash": summary.get("cash"),
        "market_value": summary.get("market_value"),
        "equity": summary.get("equity"),
        "positions": int(summary.get("positions") or 0),
        "realized_pnl_total": summary.get("realized_pnl"),
        "fills": len(trades),
        "fees_today": fees,
        "fees_today_total": round(sum(fees.values()), 2),
        "trades": trades,
        "marks": frames.to_dict(orient="records") if not frames.empty else [],
        "note": "收盘价估值；不含未成交委托的资金冻结",
    }


def append_equity_point(path: str | Path, statement: dict) -> dict:
    """Append one point to the daily equity curve (JSONL, idempotent per date)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    if target.exists():
        for line in target.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(__import__("json").loads(line))
            except ValueError:  # pragma: no cover
                continue
    point = {
        "trade_date": statement["trade_date"],
        "equity": statement["equity"],
        "cash": statement["cash"],
        "market_value": statement["market_value"],
        "positions": statement["positions"],
        "fees_today_total": statement["fees_today_total"],
    }
    rows = [row for row in rows if row.get("trade_date") != point["trade_date"]]
    rows.append(point)
    rows.sort(key=lambda row: str(row.get("trade_date")))
    target.write_text(
        "".join(__import__("json").dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return point


def equity_curve(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if not target.exists():
        return pd.DataFrame()
    rows = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(__import__("json").loads(line))
            except ValueError:  # pragma: no cover
                continue
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    frame["equity"] = frame["equity"].astype(float)
    frame["return"] = frame["equity"].pct_change()
    frame["drawdown"] = frame["equity"] / frame["equity"].cummax() - 1.0
    return frame
