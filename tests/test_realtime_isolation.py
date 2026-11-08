"""The iron rule, with teeth: live quotes may never reach the research path.

A comment saying "don't import this from the backtest" is not a guard -- a stray
import during a refactor would silently mix a live price into a total-return
series and nothing would fail.  Three independent probes:

1. **Static**: walk the real import graph of the package from every research
   module and assert ``realtime`` is unreachable (catches edits, not just today's
   text).
2. **Runtime, clean interpreter**: run a backtest and a daily signal in a fresh
   subprocess and assert ``quant_system.realtime`` never lands in ``sys.modules``.
3. **Poisoned feed**: monkeypatch the quote transport to raise, then re-run the
   same backtest and require a bit-identical equity curve -- so "the backtest does
   not need the live feed" is measured, not asserted.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent

# Modules that produce or evaluate the research number.  A live quote inside any
# of these is either a look-ahead or a price-convention break.
RESEARCH_MODULES = (
    "backtest",
    "factors",
    "signals",
    "strategies",
    "portfolio",
    "risk",
    "research",
    "metrics",
    "costs",
    "bootstrap",
    "bootstrap_report",
    "attribution",
    "sectors",
    "analytics",
    "indicators",
    "adjustment",
    "report",
    "html_report",
    "equity_io",
    "cache_audit",
    "fingerprint",
)


def _package_imports() -> dict[str, set[str]]:
    """Map ``module -> {package-relative modules it imports}``."""
    graph: dict[str, set[str]] = {}
    for path in sorted((ROOT / "quant_system").glob("*.py")):
        module = path.stem
        if module == "__init__":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level and node.module:
                    imported.add(node.module.split(".")[0])
                elif node.module and node.module.startswith("quant_system."):
                    imported.add(node.module.split(".")[1])
                elif node.level and not node.module:
                    for alias in node.names:
                        imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("quant_system."):
                        imported.add(alias.name.split(".")[1])
        graph[module] = {name for name in imported if name != module}
    return graph


def test_realtime_is_unreachable_from_every_research_module() -> None:
    graph = _package_imports()
    assert "realtime" in graph, "the module itself must be visible to the walker"
    assert any("realtime" in deps for deps in graph.values()), (
        "sanity check failed: nothing imports realtime, so the walker proves nothing"
    )

    seen: set[str] = set()
    queue = [m for m in RESEARCH_MODULES if m in graph]
    assert len(queue) >= 15, queue
    offenders: list[str] = []
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        for dependency in graph.get(module, ()):
            if dependency == "realtime":
                offenders.append(module)
            elif dependency not in seen:
                queue.append(dependency)
    assert offenders == [], f"research modules import the live feed: {offenders}"


def test_realtime_source_is_never_mentioned_in_research_files() -> None:
    for module in RESEARCH_MODULES:
        text = (ROOT / "quant_system" / f"{module}.py").read_text(encoding="utf-8")
        assert "realtime" not in text, module


_BACKTEST_SCRIPT = """
import sys
import numpy as np
import pandas as pd
from quant_system.backtest import Backtester
from quant_system.config import BacktestConfig

index = pd.bdate_range("2024-01-02", periods=60)
panel = {}
for i, symbol in enumerate(["sh600000", "sh600036", "sz000001"]):
    close = pd.Series(10 + np.arange(60) * 0.05 + i, index=index)
    panel[symbol] = pd.DataFrame(
        {"open": close * 1.001, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1_000_000.0},
        index=index,
    )
targets = pd.DataFrame(1 / 3, index=index, columns=sorted(panel))
targets.iloc[:2] = 0.0
result = Backtester(BacktestConfig(initial_cash=100_000, max_position_weight=0.5)).run(panel, targets)
assert len(result.equity) == len(index)
leaked = sorted(m for m in sys.modules if m == "quant_system.realtime")
print("LEAKED=%s" % leaked)
"""

_SIGNAL_SCRIPT = """
import sys
import numpy as np
import pandas as pd
from quant_system.signals import latest_signal

rng = np.random.default_rng(11)
index = pd.bdate_range("2023-01-02", periods=300)
panel = {}
for i in range(6):
    close = pd.Series(10.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 300))), index=index)
    panel["sh60000%d" % i] = pd.DataFrame(
        {"open": close * 1.001, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 500_000.0 + i},
        index=index,
    )
signal = latest_signal(panel, "low_vol_core", params={"top_n": 2}, cash=1_000_000.0)
assert signal.gross_exposure > 0
leaked = sorted(m for m in sys.modules if m == "quant_system.realtime")
print("LEAKED=%s" % leaked)
"""


@pytest.mark.parametrize("script", [_BACKTEST_SCRIPT, _SIGNAL_SCRIPT], ids=["backtest", "signal"])
def test_a_clean_interpreter_never_loads_the_quote_module(script: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert "LEAKED=[]" in completed.stdout, completed.stdout[-2000:]


def _panel() -> dict[str, pd.DataFrame]:
    index = pd.bdate_range("2024-01-02", periods=80)
    panel = {}
    for i, symbol in enumerate(["sh600000", "sh600036", "sz000001"]):
        close = pd.Series(10 + np.arange(80) * 0.05 + i, index=index)
        panel[symbol] = pd.DataFrame(
            {
                "open": close * 1.001,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
            },
            index=index,
        )
    return panel


def _run_backtest(panel: dict[str, pd.DataFrame]):
    from quant_system.backtest import Backtester
    from quant_system.config import BacktestConfig

    dates = next(iter(panel.values())).index
    targets = pd.DataFrame(1 / 3, index=dates, columns=sorted(panel))
    targets.iloc[:2] = 0.0
    return Backtester(BacktestConfig(initial_cash=100_000, max_position_weight=0.5)).run(
        panel, targets
    )


def test_a_dead_quote_feed_cannot_change_a_backtest(monkeypatch) -> None:
    """Poison the transport, re-run, demand a bit-identical equity curve."""
    from quant_system import realtime

    panel = _panel()
    baseline = _run_backtest(panel).equity

    def dead_transport(url: str, *, timeout=None) -> bytes:  # pragma: no cover - must not run
        raise AssertionError("the research path opened a live quote connection")

    monkeypatch.setattr(realtime, "default_transport", dead_transport)
    perturbed = _run_backtest(panel).equity
    pd.testing.assert_series_equal(baseline, perturbed)
