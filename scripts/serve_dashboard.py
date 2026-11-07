"""Local server that makes the dashboard interactive.

The static page (build_dashboard.py) is a build-time snapshot. This serves the
same page plus two endpoints the page cannot get on its own:

* /api/meta  -- data window, available knobs and the published baseline;
* /api/run   -- actually re-runs the backtest with the parameters you send and
                returns the metrics and the equity curve.

It binds to 127.0.0.1 only and uses the standard library; nothing is installed
and nothing leaves the machine.

Usage::

    python scripts/serve_dashboard.py            # http://127.0.0.1:8765
    python scripts/serve_dashboard.py --port 9000 --no-open
"""

from __future__ import annotations

import argparse
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_dashboard as dashboard  # noqa: E402

BASE_DATE = date(2019, 1, 1)

INT_PARAMS = {
    "top_n": (1, 40, 8),
    "vol_window": (5, 250, 60),
    "trend_window": (20, 250, 120),
    "max_per_sector": (0, 20, 2),
    "rebalance_days": (5, 120, 20),
}
FLOAT_PARAMS = {
    "vol_weight": (0.0, 1.0, 0.8),
    "trend_weight": (0.0, 1.0, 0.2),
    "target_vol": (0.05, 0.60, 0.18),
    # 5.0 is the CLI default; passing 0 here silently beat the published baseline.
    "slippage_bps": (0.0, 50.0, 5.0),
}
CHOICE_PARAMS = {"rebalance_reset": (("year", "start"), "year")}

DEFAULT_RUN = {
    **{k: v[2] for k, v in INT_PARAMS.items()},
    **{k: v[2] for k, v in FLOAT_PARAMS.items()},
    **{k: v[1] for k, v in CHOICE_PARAMS.items()},
}


class Engine:
    """Loads the panel once, then runs backtests against it on demand."""

    def __init__(self, cache_dir: Path, initial_cash: float = 1_000_000.0):
        self.cache_dir = cache_dir
        self.initial_cash = initial_cash
        self._aligned = None
        self._benchmark = None
        self._load_seconds = None

    def _load(self):
        if self._aligned is not None:
            return
        started = time.time()
        from types import SimpleNamespace

        from quant_system.cli import _benchmark, _rebuilt_panel, _research_panel
        from quant_system.config import BROAD_UNIVERSE
        from quant_system.data import align_panel, pit_universe

        panel = _rebuilt_panel(list(pit_universe()), self.cache_dir, "rebuilt")
        panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
        self._aligned, self._dates = align_panel(panel)
        self._benchmark = _benchmark(
            SimpleNamespace(
                benchmark="sh000300",
                no_benchmark=False,
                cache_dir=str(self.cache_dir),
            )
        )
        self._load_seconds = time.time() - started

    @property
    def window(self) -> dict:
        self._load()
        return {
            "start": str(self._dates[0].date()),
            "end": str(self._dates[-1].date()),
            "symbols": len(self._aligned),
            "days": len(self._dates),
            "load_seconds": round(self._load_seconds or 0.0, 2),
        }

    def run(self, raw: dict) -> dict:
        self._load()
        _, applied = _clean(raw)

        from quant_system.backtest import Backtester, BacktestConfig
        from quant_system.config import CostConfig
        from quant_system.portfolio import build_portfolio
        from quant_system.risk import RiskConfig
        from quant_system.strategies import build_strategy

        target_vol = applied.pop("target_vol")
        slippage = applied.pop("slippage_bps")
        strategy_params = dict(applied)
        if strategy_params.get("max_per_sector") == 0:
            strategy_params["max_per_sector"] = None

        started = time.time()
        strategy = build_strategy("low_vol_core", **strategy_params)
        raw_signals = strategy.generate(self._aligned)
        weights = build_portfolio(
            raw_signals,
            self._aligned,
            benchmark=self._benchmark,
            risk=RiskConfig(target_annual_vol=target_vol),
            use_regime=False,
            use_vol_target=True,
            use_turnover_cap=False,
            vol_forecast="realised",
        )
        result = Backtester(
            BacktestConfig(
                initial_cash=self.initial_cash,
                benchmark="sh000300",
                costs=CostConfig(slippage_bps=slippage),
                drawdown_stop=None,
            )
        ).run(self._aligned, weights, benchmark=self._benchmark)
        elapsed = time.time() - started

        metrics = result.metrics.to_dict()
        equity = result.equity
        base = float(equity.iloc[0])
        pct = ((equity / base) - 1.0) * 100.0
        offsets = [(d.date() - BASE_DATE).days for d in equity.index]
        keep = dashboard.downsample(list(range(len(pct))), 400)
        points = [[int(offsets[i]), round(float(pct.iloc[i]), 3)] for i in keep]

        return {
            "params": applied,
            "metrics": metrics,
            "eq": points,
            "elapsed_seconds": round(elapsed, 2),
            "start": str(equity.index[0].date()),
            "end": str(equity.index[-1].date()),
        }


def _clean(raw: dict) -> tuple[dict, dict]:
    """Coerce and clamp the incoming knobs; never trust the client."""
    applied: dict = {}
    for name, (low, high, default) in INT_PARAMS.items():
        value = raw.get(name, default)
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError):
            number = default
        applied[name] = max(low, min(high, number))
    for name, (low, high, default) in FLOAT_PARAMS.items():
        value = raw.get(name, default)
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        applied[name] = max(low, min(high, number))
    for name, (choices, default) in CHOICE_PARAMS.items():
        value = str(raw.get(name, default))
        applied[name] = value if value in choices else default
    return raw, applied


ENGINE: Engine | None = None
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "QuantDash/1.0"

    def log_message(self, fmt, *args):  # keep the console readable
        sys.stderr.write("  %s\n" % (fmt % args))

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def do_GET(self):  # noqa: N802 - stdlib naming
        parsed = urlparse(self.path)
        route = parsed.path
        query = parse_qs(parsed.query)
        try:
            if route in ("/", "/index.html"):
                page = dashboard.render_page(dashboard.build_data(320)).encode("utf-8")
                self._send(page, "text/html; charset=utf-8")
                return
            if route == "/favicon.ico":
                self._send(b"", "image/x-icon", 204)
                return
            if route == "/api/meta":
                self._json(
                    {
                        "window": ENGINE.window,
                        "defaults": DEFAULT_RUN,
                        "baseline": {
                            "name": "pit_full267_year",
                            "note": "默认参数（top_n=8 / vol60 / trend120 / 每行业 2 只 / 20 日 / reset=year）在仓里的归档值",
                        },
                        "int_params": {k: {"min": v[0], "max": v[1]} for k, v in INT_PARAMS.items()},
                        "float_params": {k: {"min": v[0], "max": v[1]} for k, v in FLOAT_PARAMS.items()},
                    }
                )
                return
            if route == "/api/run":
                payload = {}
                if "params" in query:
                    try:
                        payload = json.loads(query["params"][0])
                    except json.JSONDecodeError as exc:
                        self._json({"error": "params is not valid JSON: %s" % exc}, 400)
                        return
                with LOCK:
                    result = ENGINE.run(payload)
                self._json(result)
                return
            if route == "/api/rebuild":
                self._json(dashboard.build_data(320))
                return
            self._json({"error": "not found", "path": route}, 404)
        except Exception as exc:  # surface the real cause, not a blank 500
            self._json({"error": "%s: %s" % (type(exc).__name__, exc)}, 500)


def main(argv: list[str] | None = None) -> int:
    global ENGINE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    ENGINE = Engine(Path(args.cache_dir), initial_cash=args.cash)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = "http://%s:%d/" % (args.host, args.port)
    print("quant dashboard serving on %s" % url)
    print("ctrl-c to stop")
    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
