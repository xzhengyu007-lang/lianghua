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

# Backtest-level knobs: not strategy fields, but they change the answer.
COST_SPEC = {
    "target_vol": (0.05, 0.60, 0.18),
    # 5.0 is the CLI default; passing 0 here silently beat the published baseline.
    "slippage_bps": (0.0, 50.0, 5.0),
}

# Explicit bounds for the fields where a heuristic would be silly.
RANGES = {
    "top_n": (1, 30),
    "max_per_sector": (0, 20),
    "rebalance_days": (5, 120),
    "rebalance_offset": (0, 19),
    "buffer_rank": (1.0, 3.0),
    "vol_window": (5, 250),
    "trend_window": (20, 250),
    "illiquidity_window": (5, 120),
    "momentum_window": (20, 300),
    "momentum_skip": (0, 60),
    "calm_fraction": (0.0, 1.0),
    "fallback_penalty": (0.0, 20.0),
    "per_sector": (1, 10),
    "lookback": (10, 250),
    "skip": (0, 40),
    "entry_window": (10, 150),
    "exit_window": (5, 120),
    "rsi_window": (3, 40),
    "entry_rsi": (10, 50),
    "exit_rsi": (40, 90),
}
CHOICES = {
    "rebalance_reset": ("start", "year"),
    "rebalance_on": ("interval", "monthly"),
    "blend": ("zscore", "rank"),
}
# The published baseline re-anchors the cadence every January (RESEARCH_LOG 23.23),
# so the lab must default to "year" even though the dataclass defaults to "start".
DEFAULT_RESET = "year"


def strategy_spec(name: str) -> dict:
    """Knobs the UI may set for one strategy: {field: (kind, low, high, default)}."""
    import dataclasses

    from quant_system.strategies import STRATEGY_REGISTRY

    cls = STRATEGY_REGISTRY[name]
    spec: dict = {}
    for field in dataclasses.fields(cls):
        if field.name in ("name", "weights"):
            continue
        default = None if field.default is dataclasses.MISSING else field.default
        kind = field.type if isinstance(field.type, str) else str(field.type)
        if field.name in CHOICES:
            spec[field.name] = ("choice", list(CHOICES[field.name]), None, default)
        elif kind.startswith("bool"):
            spec[field.name] = ("bool", None, None, bool(default))
        elif kind.startswith("int"):
            low, high = RANGES.get(field.name, (0, max(int(default or 0) * 4, 60)))
            spec[field.name] = ("int", low, high, default)
        elif kind.startswith("float"):
            low, high = RANGES.get(field.name, (0.0, 1.0 if (default or 0) <= 1.0 else 10.0))
            spec[field.name] = ("float", low, high, default)
    if "rebalance_reset" in spec:
        # One source of truth for the calendar anchor: the dataclass says
        # "start", but the published baseline re-anchors yearly (23.23), and a
        # garbage value must not silently switch the lab to a different calendar.
        spec["rebalance_reset"] = (
            spec["rebalance_reset"][0],
            spec["rebalance_reset"][1],
            spec["rebalance_reset"][2],
            DEFAULT_RESET,
        )
    return spec


def default_params(name: str) -> dict:
    params = {k: v[3] for k, v in strategy_spec(name).items()}
    params["rebalance_reset"] = DEFAULT_RESET
    return params


DEFAULT_RUN = {**default_params("low_vol_core"), **{k: v[2] for k, v in COST_SPEC.items()}}


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
        from quant_system.backtest import Backtester, BacktestConfig
        from quant_system.config import CostConfig
        from quant_system.portfolio import build_portfolio
        from quant_system.risk import RiskConfig
        from quant_system.strategies import STRATEGY_REGISTRY, build_strategy

        name = raw.get("strategy") if isinstance(raw, dict) else None
        if name not in STRATEGY_REGISTRY:
            name = "low_vol_core"
        payload = raw.get("params") if isinstance(raw.get("params"), dict) else raw
        if not isinstance(payload, dict):
            payload = {}
        applied = _coerce(payload, strategy_spec(name))
        costs = _coerce(payload, cost_spec())
        target_vol = costs["target_vol"]
        slippage = costs["slippage_bps"]
        strategy_params = dict(applied)
        if strategy_params.get("max_per_sector") == 0:
            strategy_params["max_per_sector"] = None

        started = time.time()
        strategy = build_strategy(name, **strategy_params)
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
            "strategy": name,
            "params": applied,
            "metrics": metrics,
            "eq": points,
            "elapsed_seconds": round(elapsed, 2),
            "start": str(equity.index[0].date()),
            "end": str(equity.index[-1].date()),
        }


def _coerce(raw: dict, spec: dict) -> dict:
    """Coerce and clamp one spec; never trust the client."""
    applied: dict = {}
    for name, (kind, low, high, default) in spec.items():
        value = raw.get(name, default)
        if kind == "choice":
            text = str(value)
            applied[name] = text if text in low else default
        elif kind == "bool":
            if isinstance(value, bool):
                applied[name] = value
            elif isinstance(value, str):
                applied[name] = value.strip().lower() in ("1", "true", "yes", "on")
            else:
                applied[name] = bool(value)
        elif kind == "int":
            try:
                number = int(round(float(value)))
            except (TypeError, ValueError):
                number = int(default)
            applied[name] = max(low, min(high, number))
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float(default)
            applied[name] = max(low, min(high, number))
    return applied


def cost_spec() -> dict:
    return {k: ("float", v[0], v[1], v[2]) for k, v in COST_SPEC.items()}


def _clean(raw: dict) -> tuple[dict, dict]:
    """Default entry point: low_vol_core knobs plus the cost knobs."""
    applied = _coerce(raw, strategy_spec("low_vol_core"))
    applied.update(_coerce(raw, cost_spec()))
    return raw, applied


def _spec_payload_from(spec: dict) -> list:
    out = []
    for key, (kind, low, high, default) in spec.items():
        out.append({"key": key, "kind": kind, "low": low, "high": high, "default": default})
    return out


def _spec_payload(name: str) -> list:
    return _spec_payload_from(strategy_spec(name))


ENGINE: Engine | None = None
LOCK = threading.Lock()

from quant_system.strategies import STRATEGY_REGISTRY  # noqa: E402


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
                        "default_strategy": "low_vol_core",
                        "strategies": {
                            name: _spec_payload(name) for name in sorted(STRATEGY_REGISTRY)
                        },
                        "defaults": {
                            name: default_params(name) for name in sorted(STRATEGY_REGISTRY)
                        },
                        "cost_params": _spec_payload_from(cost_spec()),
                        "cost_defaults": {k: v[2] for k, v in COST_SPEC.items()},
                        "baseline": {
                            "name": "pit_full267_year",
                            "note": "low_vol_core 默认参数（top_n=8 / vol60 / trend120 / 每行业 2 只 / 20 日 / reset=year）在仓里的归档值",
                        },
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
