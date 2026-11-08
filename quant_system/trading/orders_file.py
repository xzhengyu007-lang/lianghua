"""L2 output: broker-importable order files plus a checklist a human can read.

Semi-automatic means the machine prepares and a person confirms.  That only works
if the artifact is (a) in a format a broker's "import orders" dialog accepts,
(b) small enough to eyeball, and (c) accompanied by the facts needed to decide.

This module writes three files and nothing else:

* ``orders.csv``      -- the import file (UTF-8 with BOM so Excel does not mangle
                          Chinese broker names; A-share brokers accept it)
* ``orders.json``     -- the same content for machines/tests
* ``checklist.md``    -- the human gate: freshness, band, T+1, total cash, fingerprint

It refuses to write a file that violates the two invariants a broker will reject
anyway (whole lots, positive limit prices) -- emitting a bad file is worse than
failing loudly, because a bad file gets imported.

No network, no submission: this module's entire output is text on disk.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

COLUMNS = ["证券代码", "证券名称", "买卖方向", "委托数量", "委托价格", "委托类型", "备注"]
SIDE_LABEL = {"buy": "买入", "sell": "卖出"}


@dataclass
class BrokerFileResult:
    csv_path: Path
    json_path: Path
    checklist_path: Path
    rows: int
    total_buy: float
    total_sell: float

    def to_dict(self) -> dict:
        return {
            "csv": str(self.csv_path), "json": str(self.json_path),
            "checklist": str(self.checklist_path), "rows": self.rows,
            "total_buy": round(self.total_buy, 2), "total_sell": round(self.total_sell, 2),
        }


def _code_of(symbol: str) -> str:
    """``sh600519`` -> ``600519``: brokers want the six digits, market by prefix."""
    digits = "".join(ch for ch in str(symbol) if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else digits


def write_broker_files(
    orders: Iterable,
    output_dir: str | Path,
    *,
    trade_date: str,
    names: dict | None = None,
    lot_size: int = 100,
    signal_fingerprint: str = "",
    snapshot_time: str = "",
    data_as_of: str = "",
    level_label: str = "L2 半自动",
    extra_notes: Iterable[str] = (),
) -> BrokerFileResult:
    """Write the import file + checklist for one trading session."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    names = names or {}

    rows: list[dict] = []
    total_buy = total_sell = 0.0
    for order in orders:
        symbol = str(getattr(order, "symbol"))
        side = str(getattr(order, "side"))
        shares = int(getattr(order, "shares"))
        price = float(getattr(order, "limit_price", getattr(order, "price", 0.0)))
        if side not in SIDE_LABEL:
            raise ValueError(f"{symbol}: 未知方向 {side!r}，拒绝生成导入文件")
        if shares <= 0 or shares % lot_size:
            raise ValueError(f"{symbol}: {shares} 股不是 {lot_size} 的整数倍，券商一定会拒单")
        if price <= 0:
            raise ValueError(f"{symbol}: 限价必须为正")
        notional = shares * price
        if side == "buy":
            total_buy += notional
        else:
            total_sell += notional
        rows.append(
            {
                "symbol": symbol,
                "code": _code_of(symbol),
                "name": str(names.get(symbol, "")),
                "side": side,
                "shares": shares,
                "limit_price": round(price, 2),
                "notional": round(notional, 2),
                "order_id": str(getattr(order, "order_id", "")),
                "reason": str(getattr(order, "reason", "")),
            }
        )

    csv_path = target / "orders.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row["code"], row["name"], SIDE_LABEL[row["side"]], row["shares"],
                    "%.2f" % row["limit_price"], "限价", row["reason"],
                ]
            )

    json_path = target / "orders.json"
    json_path.write_text(
        json.dumps(
            {
                "trade_date": trade_date,
                "level": level_label,
                "signal_fingerprint": signal_fingerprint,
                "snapshot_time": snapshot_time,
                "data_as_of": data_as_of,
                "orders": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    checklist_path = target / "checklist.md"
    checklist_path.write_text(
        _checklist(
            rows=rows, trade_date=trade_date, total_buy=total_buy, total_sell=total_sell,
            signal_fingerprint=signal_fingerprint, snapshot_time=snapshot_time,
            data_as_of=data_as_of, level_label=level_label, extra=extra_notes,
        ),
        encoding="utf-8",
    )
    return BrokerFileResult(
        csv_path=csv_path, json_path=json_path, checklist_path=checklist_path,
        rows=len(rows), total_buy=total_buy, total_sell=total_sell,
    )


def _checklist(*, rows, trade_date, total_buy, total_sell, signal_fingerprint, snapshot_time,
               data_as_of, level_label, extra) -> str:
    lines = [
        "# 下单核对清单（%s）" % trade_date,
        "",
        "级别：**%s**。本文件**不会**自动提交到券商：请在券商端逐条导入并人工确认。" % level_label,
        "",
        "## 一、这份文件是什么",
        "",
        "- 信号基准日：%s ｜ 行情快照：%s" % (data_as_of or "—", snapshot_time or "—"),
        "- 信号指纹：`%s`" % (signal_fingerprint or "—"),
        "- 订单 %d 张：买入合计 **%.2f** 元，卖出合计 **%.2f** 元" % (len(rows), total_buy, total_sell),
        "",
        "## 二、导入前逐条确认（券商端）",
        "",
        "| 标的 | 名称 | 方向 | 股数 | 限价 | 金额 |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| %s | %s | %s | %d | %.2f | %.2f |"
            % (row["symbol"], row["name"] or "—", SIDE_LABEL[row["side"]], row["shares"],
               row["limit_price"], row["notional"])
        )
    lines += [
        "",
        "## 三、必须人工回答的六个问题",
        "",
        "1. **行情是不是今天的？** 快照时间晚于今天 09:30 才算当日；否则按清单里的收盘价挂单会成交在错的价格上。",
        "2. **有没有涨停/跌停/停牌？** 涨停买不到、跌停卖不掉、停牌根本不动；这三类今天应当跳过而不是硬挂。",
        "3. **可卖数量够不够？** A 股 T+1：今天买的明天才能卖。卖出前核对「可用」而不是「持仓」。",
        "4. **总金额与现金是否匹配？** 买入会先冻结资金（含佣金约 %.3f%%、过户费）。" % (0.025),
        "5. **限价区间是否合理？** 本文件按现价 ±50bp 给限价；若要更快成交，向上放宽会直接改变成交价。",
        "6. **指纹是否与今天生成的信号一致？** 不一致说明这份文件是旧的，不要导入。",
        "",
        "## 四、导入后",
        "",
        "- 把券商回执（成交价、成交数量、手续费）记进账本：`aq paper --fills <csv> --date %s`" % trade_date,
        "- 与账本对账：`aq trade --action reconcile --ledger <ledger.json> --broker <券商流水.json>`",
        "- 差异**必须报警处理**，不允许手工改账本把差异抹平",
        "",
    ]
    for note in extra:
        lines.append("- %s" % note)
    if extra:
        lines.append("")
    lines += [
        "> 免责：本文件由日频信号生成，历史回测结果不构成投资建议；订单最终由你在券商端确认。",
        "",
    ]
    return "\n".join(lines)
