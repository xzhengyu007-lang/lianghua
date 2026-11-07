"""Guard: the script inventory must match the tree, and nothing stays opaque.

RESEARCH_LOG 23.98 found nine scripts with no docstring at all -- runnable, with
report directories behind them, and no one-line statement of what they do. The
inventory makes that visible; these tests keep it from coming back.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
INVENTORY = ROOT / "SCRIPTS.md"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_script_inventory", ROOT / "scripts" / "build_script_inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inventory = _load_module()


def test_checked_in_inventory_matches_a_fresh_render():
    if not INVENTORY.exists():
        pytest.skip("run scripts/build_script_inventory.py first")
    assert inventory.render(inventory.collect()) == INVENTORY.read_text(encoding="utf-8")


def test_every_script_appears_in_the_inventory():
    if not INVENTORY.exists():
        pytest.skip("run scripts/build_script_inventory.py first")
    text = INVENTORY.read_text(encoding="utf-8")
    missing = [p.name for p in ROOT.glob("scripts/*.py") if p.name not in text]
    assert missing == [], missing


def test_every_script_explains_itself():
    """The gap 23.98 found: a script nobody can describe in one line."""
    missing = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        if path.name == "__init__.py":
            continue
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        if not doc.strip():
            missing.append(path.name)
    assert missing == [], missing


def test_counts_in_the_inventory_are_self_consistent():
    payload = inventory.collect()
    rows = payload["scripts"]
    text = INVENTORY.read_text(encoding="utf-8") if INVENTORY.exists() else ""
    orphans = sum(1 for r in rows if not r["documented"] and not r["tested"])
    assert ("| **两者都没有（孤儿）** | **%d** |" % orphans) in text
