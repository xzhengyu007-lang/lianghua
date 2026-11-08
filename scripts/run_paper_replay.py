"""Replay the paper loop over the last N sessions (the missing evidence artifact).

Why this exists: the desk's acceptance needs a *20-session* paper equity curve, and
the ledger in `reports/paper/` only has today's fills.  Re-running the same files
by hand for three weeks is not reproducible, so the loop is scripted:

    for each session d in the last N:
        signal  = latest_signal(panel truncated at d)      # no look-ahead by construction
        orders  = plan_orders(ledger, weights, close(d))   # sized on the signal close
        fills   = next session's open, with slippage      # the desk's execution model
        ledger  = apply_fills(ledger, fills)              # fees, T+1, cash
        append_equity_point(close(d+1) marks)

The truncation is the point: `latest_signal` never sees a bar after `d`, exactly as
in live use.  This is **not** a backtest of the strategy (the backtester already
exists and owns that number) -- it is a replay of the *paper bookkeeping path*:
same `plan_orders` / `apply_fills` / `statement` / `equity_curve` code the desk uses.

Usage::

    python scripts/run_paper_replay.py --sessions 20 --equity 1000000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import CACHE_DIR, CostConfig  # noqa: E402
from quant_system.data import (  # noqa: E402
    load_symbol,
    normalize_symbol,
    panel_to_frame,
    pit_universe,
)
from quant_system.paper import Ledger, Position, apply_fills, plan_orders  # noqa: E402
from quant_system.signals import latest_signal  # noqa: E402
from quant_system.trading.recon import append_equity_point, daily_statement, equity_curve  # noqa: E402

SLIPPAGE_BPS = 5.0


def build_panel(symbols: list[str], cache_dir: Path) -> dict[str, pd.DataFrame]:
    """Rebuilt total-return panel: the same convention the deployment numbers use.

    Keys must be **normalized** (`sz000333`, not `000333`): ``pit_universe`` hands out
    bare codes, and mixing the two spellings puts fills into the ledger under a name
    that no price vector can value -- which is exactly how the first run of this
    script died with "missing price for held symbol 000333".
    """
    panel: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        normalized = normalize_symbol(str(symbol))
        try:
            raw = load_symbol(normalized, cache_dir, adjust="tr")
        except FileNotFoundError:
            continue
        if len(raw) > 200:
            panel[normalized] = raw
    return panel


def replay(
    panel: dict[str, pd.DataFrame],
    *,
    sessions: int = 20,
    equity: float = 1_000_000.0,
    strategy: str = "low_vol_core",
    output: Path,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    close = panel_to_frame(panel, "close")
    # A held name can miss a bar (suspension).  Marking it at the last traded close
    # is the honest accounting -- using NaN would make the statement unbuildable and
    # dropping the position would silently liquidate a suspended stock.
    mark_close = close.ffill()
    dates = close.index
    if len(dates) < sessions + 2:
        raise ValueError("缓存里的交易日不够：需要至少 %d 根" % (sessions + 2))
    start_index = len(dates) - sessions - 1
    ledger_path = output / "ledger.json"
    equity_path = output / "equity.jsonl"
    ledger = Ledger(cash=float(equity), positions={}, as_of=str(dates[start_index].date()))
    ledger.save(ledger_path)
    equity_path.unlink(missing_ok=True)
    costs = CostConfig(slippage_bps=SLIPPAGE_BPS)

    rows: list[dict] = []
    for index in range(start_index, len(dates) - 1):
        signal_day = dates[index]
        exec_day = dates[index + 1]
        truncated = {
            symbol: frame.loc[:signal_day] for symbol, frame in panel.items()
        }
        signal = latest_signal(
            truncated, strategy, equity=ledger.equity(
                pd.Series({s: float(mark_close.loc[signal_day, s]) for s in ledger.positions})
            ),
            today=signal_day,
        )
        # plan_orders values the *whole* book to size orders, so the price vector must
        # cover held names too -- a name that dropped out of the target basket still
        # has to be pricable, otherwise sizing raises "missing price for held symbol".
        wanted = set(signal.target_weights.index) | set(ledger.positions)
        signal_prices = pd.Series(
            {symbol: float(mark_close.loc[signal_day, symbol]) for symbol in wanted}
        )
        orders = plan_orders(ledger, signal.target_weights, signal_prices, costs=costs)

        fills = []
        for row in orders.itertuples(index=False):
            try:
                exec_price = float(load_symbol(row.symbol, cache_dir, adjust="tr").loc[exec_day, "open"])
            except (FileNotFoundError, KeyError):
                continue
            slip = exec_price * (SLIPPAGE_BPS / 10000.0) * (1 if row.side == "buy" else -1)
            fills.append(
                {"symbol": row.symbol, "side": row.side, "shares": int(row.shares),
                 "price": round(exec_price + slip, 4)}
            )
        if fills:
            ledger = apply_fills(ledger, pd.DataFrame(fills), date=str(exec_day.date()), costs=costs)
            ledger.save(ledger_path)

        marks = pd.Series(
            {symbol: float(mark_close.loc[exec_day, symbol]) for symbol in ledger.positions}, dtype=float
        )
        statement = daily_statement(ledger, marks, trade_date=str(exec_day.date()))
        append_equity_point(equity_path, statement)
        rows.append(
            {
                "signal_day": str(signal_day.date()),
                "exec_day": str(exec_day.date()),
                "orders": int(len(fills)),
                "names": int(len(signal.target_weights)),
                "equity": statement["equity"],
                "fees": statement["fees_today_total"],
            }
        )

    curve = equity_curve(equity_path)
    summary = {
        "sessions": int(len(curve)),
        "start": rows[0]["exec_day"] if rows else None,
        "end": rows[-1]["exec_day"] if rows else None,
        "start_equity": float(curve["equity"].iloc[0]) if len(curve) else None,
        "end_equity": float(curve["equity"].iloc[-1]) if len(curve) else None,
        "total_return": float(curve["equity"].iloc[-1] / curve["equity"].iloc[0] - 1) if len(curve) else None,
        "max_drawdown": float(curve["drawdown"].min()) if len(curve) else None,
        "fees_paid": round(float(curve["fees_today_total"].sum()), 2),
        "orders_replayed": int(sum(row["orders"] for row in rows)),
        "note": "模拟盘回放：逐日截断面板重算信号，按次日开盘价加 5bp 滑点成交；非策略回测",
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回放模拟盘账本若干交易日（生成每日权益曲线）")
    parser.add_argument("--sessions", type=int, default=20)
    parser.add_argument("--equity", type=float, default=1_000_000.0)
    parser.add_argument("--strategy", default="low_vol_core")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--output", default=str(ROOT / "reports" / "paper_replay"))
    parser.add_argument("--symbols", default=None, help="逗号分隔；默认用 PIT 池里数据完整的标的")
    args = parser.parse_args(argv)

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = list(pit_universe())
    panel = build_panel(symbols, Path(args.cache_dir))
    print("panel: %d 只（重建总收益）" % len(panel), flush=True)
    summary = replay(
        panel, sessions=args.sessions, equity=args.equity, strategy=args.strategy,
        output=Path(args.output), cache_dir=Path(args.cache_dir),
    )
    print(
        "回放 %d 个交易日（%s → %s）：%.2f → %.2f，累计 %+.2f%%，最大回撤 %.2f%%，费用 %.2f"
        % (
            summary["sessions"], summary["start"], summary["end"],
            summary["start_equity"], summary["end_equity"],
            (summary["total_return"] or 0) * 100, (summary["max_drawdown"] or 0) * 100,
            summary["fees_paid"],
        ),
        flush=True,
    )
    print("-> %s" % Path(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
