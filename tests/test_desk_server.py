"""Tests for the local desk server (stage 4).

Integration-flavoured but local: the server binds an ephemeral port on loopback and
the test speaks HTTP to it.  That is the only way to catch the two bugs that
actually happened while building it -- a NaN in the JSON blanking the whole page,
and the gate staying "正常" after a CLI kill because the server cached it.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from scripts.desk_server import DeskState, bars_payload, build_server, json_safe


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def state(tmp_path) -> DeskState:
    return DeskState(
        trading_dir=tmp_path / "trading",
        ledger_path=tmp_path / "ledger.json",
        store=tmp_path / "quotes.sqlite",
        reports_dir=tmp_path / "reports",
    )


@pytest.fixture()
def server(state):
    httpd = build_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, response.read()


def _post(url: str, payload: dict):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# --------------------------------------------------------------------------- #
# static assets
# --------------------------------------------------------------------------- #
def test_the_desk_page_is_served(server) -> None:
    status, body = _get(server + "/")
    assert status == 200
    text = body.decode("utf-8")
    assert "交易台" in text
    assert "/ui/desk.js" in text
    assert "不构成投资建议" in text


def test_the_js_and_css_are_served(server) -> None:
    for path in ("/ui/desk.js", "/ui/desk.css"):
        status, body = _get(server + path)
        assert status == 200 and len(body) > 500


def test_the_vendor_bundle_is_served_locally(server) -> None:
    """No CDN: the chart library must come from this repo, not the internet."""
    status, body = _get(server + "/ui/vendor/lightweight-charts.standalone.production.js")
    assert status == 200
    assert b"Apache License 2.0" in body[:400]


def test_path_traversal_is_refused(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server + "/ui/../quant_system/cli.py")
    assert excinfo.value.code in (403, 404)


def test_unknown_path_is_a_404(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server + "/nope")
    assert excinfo.value.code == 404


# --------------------------------------------------------------------------- #
# state: strict JSON, live reload, never a blank page
# --------------------------------------------------------------------------- #
def test_state_is_strict_json_with_no_nan(server) -> None:
    """`NaN` is not JSON: it blanked the desk once, so the raw bytes are checked."""
    status, body = _get(server + "/api/state")
    assert status == 200
    text = body.decode("utf-8")
    assert "NaN" not in text and "Infinity" not in text
    payload = json.loads(text)
    for key in ("generated_at", "gate", "quotes", "advice", "ledger", "equity", "audit", "mode"):
        assert key in payload, key


def test_state_reports_the_gate_and_the_no_gateway_mode(server) -> None:
    _, body = _get(server + "/api/state")
    payload = json.loads(body)
    assert payload["gate"]["level_label"].startswith("L1")
    assert "无券商网关" in payload["mode"]


def test_a_cli_kill_shows_up_without_restarting_the_server(server, state) -> None:
    """The operator runs `aq trade --action kill` in a terminal; the page must see it."""
    from quant_system.trading import TradingGate

    TradingGate(state.paths["gate"]).kill("终端停机")
    _, body = _get(server + "/api/state")
    payload = json.loads(body)
    assert payload["gate"]["kill_switch"] is True
    assert payload["gate"]["halt_reason"] == "终端停机"


def test_advice_block_is_reported_when_missing(server) -> None:
    _, body = _get(server + "/api/card")
    payload = json.loads(body)
    assert payload["blocked"] is True
    assert any("aq advice" in reason for reason in payload["block_reasons"])


def test_the_advice_card_page_is_404_until_generated(server) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server + "/card")
    assert excinfo.value.code == 404


# --------------------------------------------------------------------------- #
# bars
# --------------------------------------------------------------------------- #
def test_bars_payload_uses_the_rebuilt_total_return_series() -> None:
    payload = bars_payload("sh600036", limit=120)
    if not payload["bars"]:
        pytest.skip("cache not present in this checkout")
    assert payload["count"] == len(payload["bars"]) <= 120
    assert "rebuilt total-return" in payload["convention"]
    bar = payload["bars"][-1]
    assert {"time", "open", "high", "low", "close", "volume"} <= set(bar)
    assert bar["high"] >= bar["low"]


def test_bars_for_an_unknown_symbol_is_an_empty_answer_not_an_error(server) -> None:
    status, body = _get(server + "/api/bars?symbol=sh999999&limit=100")
    payload = json.loads(body)
    assert status == 200
    assert payload["bars"] == []


# --------------------------------------------------------------------------- #
# gate actions
# --------------------------------------------------------------------------- #
def test_the_ui_cannot_switch_to_full_auto(server) -> None:
    status, payload = _post(server + "/api/gate", {"action": "level", "level": 3})
    assert status == 400
    assert "L3" in payload["error"]


def test_level_two_is_allowed_from_the_ui(server) -> None:
    status, payload = _post(server + "/api/gate", {"action": "level", "level": 2})
    assert status == 200
    assert payload["gate"]["level"] == 2


def test_kill_and_reset_round_trip(server, state) -> None:
    _, killed = _post(server + "/api/gate", {"action": "kill", "reason": "测试"})
    assert killed["gate"]["kill_switch"] is True
    assert killed["gate"]["level"] == 1  # kill drops back to paper
    _, reset = _post(server + "/api/gate", {"action": "reset-kill"})
    assert reset["gate"]["kill_switch"] is False


def test_an_unknown_gate_action_is_a_400(server) -> None:
    status, payload = _post(server + "/api/gate", {"action": "launch_missiles"})
    assert status == 400


def test_plan_without_a_ledger_explains_what_is_missing(server) -> None:
    status, payload = _post(server + "/api/plan", {})
    assert status == 400
    assert "账本" in payload["error"] or "建议卡" in payload["error"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_json_safe_converts_non_finite_floats() -> None:
    payload = json_safe({"a": float("nan"), "b": [float("inf"), 1.5], "c": {"d": float("-inf")}})
    assert payload == {"a": None, "b": [None, 1.5], "c": {"d": None}}
    json.loads(json.dumps(payload, allow_nan=False))  # must be strict-JSON serialisable


def test_the_server_binds_loopback_only() -> None:
    source = (Path(__file__).resolve().parent.parent / "scripts" / "desk_server.py").read_text(
        encoding="utf-8"
    )
    assert 'default="127.0.0.1"' in source
    assert 'default="0.0.0.0"' not in source


# --------------------------------------------------------------------------- #
# the "生成计划" button, end to end
# --------------------------------------------------------------------------- #
def _calendar_through(last: str):
    """A bar calendar ending on ``last`` -- what the real cache looks like."""
    import pandas as pd

    from quant_system.realtime import TradingCalendar

    days = [
        day.date()
        for day in pd.bdate_range("2026-09-21", last)
        if not (pd.Timestamp("2026-10-01") <= day <= pd.Timestamp("2026-10-08"))
    ]
    return TradingCalendar(days)


def _seed_plan_fixture(state, tmp_path, *, clock: str = "2026-11-06 16:30") -> None:
    """A ledger, a gate at L2 and a non-blocked card: enough for /api/plan to trade.

    The clock is pinned right after the card's session closed: without that the
    desk (correctly) refuses to plan from a card older than the last close, and the
    test would be asserting yesterday's behaviour instead of today's.
    """
    import pandas as pd

    from quant_system.paper import Ledger
    from quant_system.trading import TradingGate, TradingLevel

    state.clock = lambda: pd.Timestamp(clock)
    state.calendar = _calendar_through("2026-11-06")
    state.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    Ledger(cash=200_000.0, positions={}, as_of="2026-11-06").save(state.ledger_path)
    TradingGate(state.paths["gate"]).set_level(TradingLevel.L2_SEMI_AUTO)
    card = {
        "as_of": "2026-11-06",
        "generated_at": "2026-11-06 16:30:00",
        "strategy": "low_vol_core",
        "equity": 200_000.0,
        "blocked": False,
        "block_reasons": [],
        "items": [
            {"symbol": "sh600036", "name": "招商银行", "weight": 0.2, "price": 42.0,
             "limit_low": 41.79, "limit_high": 42.21, "shares": 900, "lot_size": 100},
        ],
        "uncertainty": {"complete": True},
        "disclaimer": "历史回测结果，不构成投资建议。",
    }
    target = state.reports_dir / "advice_today"
    target.mkdir(parents=True, exist_ok=True)
    (target / "card.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")


def test_the_plan_button_plans_an_order_and_writes_the_broker_file(server, state, tmp_path) -> None:
    _seed_plan_fixture(state, tmp_path)
    status, payload = _post(server + "/api/plan", {})
    assert status == 200, payload
    plan = payload["plan"]
    assert plan["halted"] is False
    assert len(plan["orders"]) == 1
    assert plan["orders"][0]["symbol"] == "sh600036"
    assert plan["written"] is not None, "L2 must produce the broker file"
    assert "招商银行" in (tmp_path / "trading" / "orders_today" / "orders.csv").read_text(encoding="utf-8-sig")


def test_the_plan_button_is_idempotent(server, state, tmp_path) -> None:
    """Clicking twice is the most likely operator error in the whole UI."""
    _seed_plan_fixture(state, tmp_path)
    _post(server + "/api/plan", {})
    status, second = _post(server + "/api/plan", {})
    assert status == 200
    plan = second["plan"]
    assert plan["orders"] == []
    assert plan["rejected"] and "已有一张有效订单" in " ".join(plan["rejected"][0]["reasons"])


def test_the_plan_button_refuses_while_the_gate_is_killed(server, state, tmp_path) -> None:
    from quant_system.trading import TradingGate

    _seed_plan_fixture(state, tmp_path)
    TradingGate(state.paths["gate"]).kill("界面停机")
    status, payload = _post(server + "/api/plan", {})
    assert status == 200
    assert payload["plan"]["halted"] is True
    assert "kill switch" in payload["plan"]["halt_reason"]


def test_the_plan_button_reports_a_blocked_card_instead_of_trading(server, state, tmp_path) -> None:
    _seed_plan_fixture(state, tmp_path)
    card_path = state.reports_dir / "advice_today" / "card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["blocked"] = True
    card["block_reasons"] = ["缓存最新交易日 2026-11-05 ≠ 上一交易日 2026-11-06"]
    card_path.write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")

    status, payload = _post(server + "/api/plan", {})
    assert status == 200
    assert payload["blocked"] is True
    assert payload["plan"] is None
    assert "不排单" in payload["note"]


def test_plan_refuses_a_card_whose_session_has_since_closed(server, state, tmp_path) -> None:
    """The card's own ``blocked`` flag is a frozen snapshot: Monday night it lies.

    Generating on Friday is fine; acting on it Tuesday is not. The server must
    re-derive the verdict from the clock, not trust the file.
    """
    _seed_plan_fixture(state, tmp_path, clock="2026-11-09 23:00")
    status, payload = _post(server + "/api/plan", {})
    assert status == 200
    assert payload["blocked"] is True
    assert payload["plan"] is None
    assert "2026-11-06" in payload["reasons"][0] and "2026-11-09" in payload["reasons"][0]
    assert not (tmp_path / "trading" / "orders_today" / "orders.csv").exists()


def test_state_reports_the_world_clock_and_the_cache(server, state, tmp_path) -> None:
    """Two dates, side by side: the desk's clock (market) and ``as_of`` (our data)."""
    _seed_plan_fixture(state, tmp_path, clock="2026-11-09 23:00")
    status, body = _get(server + "/api/state")
    assert status == 200
    world = json.loads(body)["world"]
    assert world["last_completed_session"] == "2026-11-09"
    assert world["cache_as_of"] == "2026-11-06"
    assert world["behind_sessions"] == 1
    assert world["stale"] is True
    assert world["session"] == "after_close"


def test_state_is_not_stale_on_the_session_the_card_belongs_to(server, state, tmp_path) -> None:
    _seed_plan_fixture(state, tmp_path)  # clock = Friday 16:30, card as_of = Friday
    _, body = _get(server + "/api/state")
    world = json.loads(body)["world"]
    assert world["behind_sessions"] == 0
    assert world["stale"] is False


def test_bars_limit_is_clamped(server) -> None:
    """A browser asking for 10 million bars gets a bounded answer, not an OOM."""
    status, body = _get(server + "/api/bars?symbol=sh600036&limit=10000000")
    payload = json.loads(body)
    assert status == 200
    assert payload["count"] <= 20000
