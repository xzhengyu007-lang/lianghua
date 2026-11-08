"""Local trading desk: one process, one page, no build step.

Run it with ``python scripts/desk_server.py`` and open http://127.0.0.1:8770.
It binds loopback only, holds no credentials, and sends no orders (there is no
gateway in this repository -- see ``docs/实时行情与交易架构.md`` §11).

Architecture, and why it is not FastAPI:

* the repository already ships a stdlib ``ThreadingHTTPServer`` for the research
  dashboard, and the desk needs four JSON endpoints plus static files;
* the chart library is a single ``<script>`` file, so there is nothing to bundle
  and no reason to introduce Node;
* one background :class:`quant_system.realtime.QuotePoller` owns the 3 s cadence,
  so the browser polls ``/api/state`` (cheap, reads memory) instead of hitting the
  quote endpoint itself -- that keeps one throttle for the whole machine.

Endpoints
---------
    GET  /                  the desk (three columns)
    GET  /ui/<file>         static: desk.js, desk.css, vendor/*.js
    GET  /api/state         gate + quotes + advice + ledger + equity + health
    GET  /api/bars?symbol=  OHLCV for the chart (gzip, sliced server-side)
    GET  /api/card          the advice card as JSON (written by `aq advice`)
    GET  /card              the same card as a standalone HTML page
    POST /api/gate          {"action": "kill"|"reset-kill"|"level", "level": 2}
    POST /api/plan          run the session pipeline, return the plan

Every POST that changes state is confirmed in the UI first (the page asks; the
server also refuses L3 without an explicit date).
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import threading
import time
import webbrowser
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

UI_DIR = ROOT / "ui"
REPORTS = ROOT / "reports"

from quant_system.paper import Ledger, summarize  # noqa: E402
from quant_system.realtime import (  # noqa: E402
    DEFAULT_STORE,
    QuotePoller,
    SnapshotStore,
    default_source,
    previous_trading_day,
)
from quant_system.trading import AuditLog, IdempotencyBook, TradingGate, TradingLevel  # noqa: E402
from quant_system.trading.pipeline import default_paths, plan_session  # noqa: E402
from quant_system.trading.recon import equity_curve  # noqa: E402

def json_safe(value):
    """Make a payload strict JSON: NaN/inf become null.

    The page broke on ``"return": NaN`` -- Python's json module writes bare
    ``NaN``, which is *not* JSON, so ``JSON.parse`` threw and the whole desk went
    blank.  Converting non-finite floats here (and passing ``allow_nan=False`` so a
    new leak fails loudly) is the fix; a chart of "nothing" is worse than an error.
    """
    import math

    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):  # numpy scalars
        try:
            return json_safe(value.item())
        except Exception:  # pragma: no cover
            return value
    return value


STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


class DeskState:
    """Everything the page reads, assembled from files the repo already writes."""

    def __init__(
        self, *, trading_dir: Path, ledger_path: Path | None, store: Path,
        reports_dir: Path | None = None,
    ) -> None:
        self.reports_dir = Path(reports_dir) if reports_dir is not None else REPORTS
        self.paths = default_paths(trading_dir)
        self.ledger_path = ledger_path
        self.store = SnapshotStore(store)
        self.gate = TradingGate(self.paths["gate"])
        self.audit = AuditLog(self.paths["audit"])
        self.book = IdempotencyBook(self.paths["idempotency"])
        self.poller: QuotePoller | None = None
        self._symbols: list[str] = []
        self._mtimes: dict[str, float] = {}

    # ---------- live artifacts --------------------------------------------
    def reload_if_changed(self) -> None:
        """Re-read gate.json / orders.jsonl when a CLI command rewrote them.

        The desk runs for hours; the operator runs ``aq trade --action kill`` in a
        terminal.  Caching the gate in memory meant the page kept showing "正常"
        after a stop -- the worst possible stale state, so the mtimes are checked
        on every state read (a 1 KB JSON file, at a 3 s cadence).
        """
        for key in ("gate", "idempotency"):
            path = self.paths[key]
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                mtime = 0.0
            if self._mtimes.get(key) != mtime:
                self._mtimes[key] = mtime
                if key == "gate":
                    self.gate = TradingGate(path)
                else:
                    self.book = IdempotencyBook(path)

    # ---------- quotes -----------------------------------------------------
    def symbols(self) -> list[str]:
        if self._symbols:
            return self._symbols
        symbols: list[str] = []
        card = self.advice()
        if card and card.get("items"):
            symbols = [str(item["symbol"]) for item in card["items"]]
        elif self.ledger_path and self.ledger_path.exists():
            symbols = list(Ledger.load(self.ledger_path).positions)
        self._symbols = symbols
        return symbols

    def start_poller(self) -> None:
        symbols = self.symbols()
        if not symbols:
            return
        self.poller = QuotePoller(
            symbols, source=default_source(cross_check=True, attempts=1),
            store=self.store, interval=3.0,
        ).start()

    def quote_block(self) -> dict:
        if self.poller is None:
            return {"state": "yellow", "message": "未启动行情轮询（没有可查询的标的）", "board": None}
        status = self.poller.status()
        return status.to_dict()

    # ---------- artifacts --------------------------------------------------
    def advice(self) -> dict | None:
        for path in (
            self.reports_dir / "advice_today" / "card.json",
            self.reports_dir / "advice" / "card.json",
        ):
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except ValueError:  # pragma: no cover - half-written file
                    continue
        return None

    def ledger_block(self) -> dict | None:
        if not self.ledger_path or not self.ledger_path.exists():
            return None
        ledger = Ledger.load(self.ledger_path)
        prices = self._marks(ledger)
        summary = summarize(ledger, prices if len(prices) else None)
        return {
            "path": str(self.ledger_path),
            "as_of": ledger.as_of,
            "cash": summary.get("cash"),
            "positions": summary.get("positions"),
            "equity": summary.get("equity"),
            "market_value": summary.get("market_value"),
            "realized_pnl": summary.get("realized_pnl"),
            "last_signal_date": ledger.last_signal_date,
        }

    def _marks(self, ledger: Ledger):
        import pandas as pd

        prices: dict[str, float] = {}
        board = self.poller.status().board if self.poller is not None else None
        for symbol in ledger.positions:
            quote = board.quotes.get(symbol) if board is not None else None
            if quote is not None and quote.price > 0:
                prices[symbol] = float(quote.price)
        return pd.Series(prices, dtype=float)

    def equity_block(self) -> dict:
        curve = equity_curve(self.paths["equity"])
        if curve.empty:
            return {"points": [], "last": None}
        # The first point has no return and no drawdown: emit null, not NaN.
        points = [
            {
                "trade_date": str(row["trade_date"]),
                "equity": float(row["equity"]),
                "return": None if row["return"] != row["return"] else float(row["return"]),
                "drawdown": None if row["drawdown"] != row["drawdown"] else float(row["drawdown"]),
            }
            for _, row in curve.iterrows()
        ]
        return {"points": points, "last": points[-1] if points else None}

    def state(self) -> dict:
        self.reload_if_changed()
        today = date.today()
        last_trading = str(previous_trading_day(today))
        return {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "today": str(today),
            "last_trading_day": last_trading,
            "gate": self.gate.describe(today),
            "quotes": self.quote_block(),
            "advice": self.advice(),
            "ledger": self.ledger_block(),
            "equity": self.equity_block(),
            "audit": {"events": self.audit.count(), "recent": self.audit.events()[-8:]},
            "orders": {"active": len(self.book.active())},
            "mode": "模拟盘（本仓无券商网关：不存在发单代码）",
        }


class Handler(BaseHTTPRequestHandler):
    state: DeskState | None = None
    server_version = "aq-desk/1.0"

    # ---------- helpers ----------------------------------------------------
    def log_message(self, fmt: str, *args) -> None:  # quieter console
        if self.path.startswith("/api/"):
            return
        super().log_message(fmt, *args)

    def _send(self, status: int, body: bytes, content_type: str, *, gzip_ok: bool = True) -> None:
        if gzip_ok and "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024:
            body = gzip.compress(body, 6)
            self.send_response(status)
            self.send_header("Content-Encoding", "gzip")
        else:
            self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(
            json_safe(payload), ensure_ascii=False, default=str, allow_nan=False
        ).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _fail(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _static(self, relative: str) -> None:
        candidate = (UI_DIR / relative.lstrip("/")).resolve()
        if UI_DIR.resolve() not in candidate.parents and candidate != UI_DIR.resolve():
            self._fail(403, "路径越界")
            return
        if not candidate.is_file():
            self._fail(404, "找不到 %s" % relative)
            return
        content_type = STATIC_TYPES.get(candidate.suffix, "application/octet-stream")
        self._send(200, candidate.read_bytes(), content_type)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except ValueError as exc:
            raise ValueError("请求体不是 JSON：%s" % exc) from exc

    # ---------- routes -----------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        try:
            if path == "/favicon.ico":
                self._send(204, b"", "image/x-icon")
                return
            if path in ("/", "/index.html"):
                self._static("desk.html")
                return
            if path.startswith("/ui/"):
                self._static(path[len("/ui/"):])
                return
            if path == "/api/state":
                self._json(self.state.state())
                return
            if path == "/api/card":
                card = self.state.advice()
                if card is None:
                    self._json({"blocked": True, "block_reasons": ["尚未生成建议卡：先跑 aq advice"]}, 200)
                else:
                    self._json(card)
                return
            if path == "/card":
                page = self.state.reports_dir / "advice_today" / "card.html"
                if not page.exists():
                    self._fail(404, "尚未生成 card.html：先跑 aq advice --output reports/advice_today")
                    return
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
                return
            if path == "/api/bars":
                symbol = (query.get("symbol") or [""])[0]
                limit = int((query.get("limit") or ["600"])[0])
                self._json(bars_payload(symbol, limit=max(50, min(limit, 20000))))
                return
            self._fail(404, "未知路径 %s" % path)
        except Exception as exc:  # pragma: no cover - defensive
            self._fail(500, "%s: %s" % (type(exc).__name__, exc))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        try:
            if path == "/api/gate":
                body = self._read_json()
                self._json(self._gate_action(body))
                return
            if path == "/api/plan":
                body = self._read_json()
                self._json(self._plan(body))
                return
            self._fail(404, "未知路径 %s" % path)
        except ValueError as exc:
            self._fail(400, str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self._fail(500, "%s: %s" % (type(exc).__name__, exc))

    # ---------- actions ----------------------------------------------------
    def _gate_action(self, body: dict) -> dict:
        state = self.state
        state.reload_if_changed()
        action = str(body.get("action", ""))
        today = date.today()
        if action == "kill":
            state.gate.kill(str(body.get("reason") or "界面停机"))
            state.audit.record("gate.kill", trade_date=str(today), source="operator",
                               detail={"reason": body.get("reason") or "界面停机"})
        elif action == "reset-kill":
            state.gate.reset_kill(reset_level=True)
            state.audit.record("gate.reset", trade_date=str(today), source="operator")
        elif action == "level":
            level = int(body.get("level", 1))
            if level >= int(TradingLevel.L3_FULL_AUTO):
                raise ValueError("L3 只能在命令行显式授权（--level 3 --date），界面不允许一键开到全自动")
            state.gate.set_level(TradingLevel(level))
            state.audit.record("gate.level", trade_date=str(today), source="operator",
                               level=level)
        else:
            raise ValueError("未知 action：%s" % action)
        return {"gate": state.gate.describe(today)}

    def _plan(self, body: dict) -> dict:
        state = self.state
        state.reload_if_changed()
        ledger_path = state.ledger_path
        if ledger_path is None or not ledger_path.exists():
            raise ValueError("没有账本：先 aq paper --init")
        card = state.advice()
        if card is None:
            raise ValueError("没有建议卡：先 aq advice --output reports/advice_today")
        if card.get("blocked"):
            return {"blocked": True, "reasons": card.get("block_reasons", []),
                    "plan": None, "note": "数据闸门已关闭：本日不出建议，也不排单"}
        import pandas as pd

        ledger = Ledger.load(ledger_path)
        rows = card.get("items") or []
        weights = pd.Series({row["symbol"]: float(row["weight"]) for row in rows}, dtype=float)
        prices = pd.Series({row["symbol"]: float(row["price"]) for row in rows}, dtype=float)
        board = state.poller.status().board if state.poller is not None else None
        if board is not None:
            for item in rows:
                quote = board.quotes.get(item["symbol"])
                if quote is not None and quote.price > 0:
                    prices[item["symbol"]] = float(quote.price)
        equity = float(body.get("equity") or card.get("equity") or 1_000_000.0)
        outcome = plan_session(
            ledger=ledger, target_weights=weights, prices=prices, gate=state.gate,
            trade_date=str(body.get("trade_date") or date.today()), equity=equity,
            book=state.book, audit=state.audit, quotes=board,
            strategy=str(card.get("strategy", "")), signal_as_of=str(card.get("as_of", "")),
            snapshot_time=(board.fetched_at.isoformat() if board is not None and board.fetched_at is not None else ""),
            output_dir=state.paths["orders"],
            names={row["symbol"]: row.get("name", "") for row in rows},
        )
        return {"plan": outcome.to_dict(), "markdown": outcome.markdown()}


def bars_payload(symbol: str, *, limit: int = 600) -> dict:
    """OHLCV for the chart, sliced server-side.

    The cache is the rebuilt total-return series, i.e. exactly what the backtest
    used -- plotting a raw quote series next to it would compare two different
    price conventions on the same axis.
    """
    from quant_system.data import load_symbol, normalize_symbol

    if not symbol:
        return {"symbol": "", "bars": [], "count": 0}
    sym = normalize_symbol(symbol)
    try:
        frame = load_symbol(sym, adjust="tr")
    except FileNotFoundError:
        try:
            frame = load_symbol(sym, adjust="raw")
        except FileNotFoundError:
            return {"symbol": sym, "bars": [], "count": 0, "error": "缓存里没有 %s" % sym}
    frame = frame.tail(max(50, min(int(limit), 20000)))
    bars = [
        {
            "time": str(index.date()),
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": float(row["volume"]),
        }
        for index, row in frame.iterrows()
    ]
    return {"symbol": sym, "bars": bars, "count": len(bars),
            "convention": "rebuilt total-return (same series the backtest used)"}


def build_server(host: str, port: int, state: DeskState) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地交易台（只监听 127.0.0.1）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--ledger", default=str(REPORTS / "paper" / "ledger.json"))
    parser.add_argument("--trading-dir", default=str(REPORTS / "trading"))
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--reports", default=str(REPORTS), help="产物目录（card.json / gate 等）")
    parser.add_argument("--no-quotes", action="store_true", help="不启动行情轮询（离线演示）")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args(argv)

    state = DeskState(
        trading_dir=Path(args.trading_dir),
        ledger_path=Path(args.ledger) if args.ledger else None,
        store=Path(args.store),
        reports_dir=Path(args.reports),
    )
    if not args.no_quotes:
        state.start_poller()
    server = build_server(args.host, args.port, state)
    url = "http://%s:%d" % (args.host, args.port)
    print("交易台: %s  （Ctrl-C 退出）" % url, flush=True)
    if not args.no_open:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，正在关闭…", flush=True)
    finally:
        if state.poller is not None:
            state.poller.stop()
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
