"""Regression tests for CLI argument handling.

Two defects motivated these tests:

* ``--symbols`` used ``nargs="*"``, so ``--symbols a b backtest`` consumed
  ``backtest`` as a symbol and the command never ran.
* An explicit ``--symbols`` list under ``--universe pit`` skipped the survivor
  anchor, so ``_research_panel`` fell back to the 95%-coverage rule.  That rule
  drops names which stop trading early and silently shortened the window from
  2019-06-06..2026-11-06 to ..2024-06-13, changing every reported number.
"""

from __future__ import annotations

import argparse

import pandas as pd
import pytest

from quant_system.cli import _research_anchor, build_parser


def _args(**kwargs) -> argparse.Namespace:
    base = {"universe": "pit", "symbols": None}
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_comma_separated_symbols_do_not_swallow_the_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        ["--symbols=600000,000001", "backtest", "--strategy", "low_vol_core"]
    )
    assert args.command == "backtest"
    assert args.symbols == ["600000,000001"]


def test_space_separated_symbols_still_swallow_the_subcommand():
    # argparse cannot know where a greedy nargs="+" list ends, so the
    # space-separated form remains unusable ahead of the subcommand.  The
    # comma form above is the documented one; this test pins the limitation so
    # nobody "fixes" it by silently truncating a user's universe.
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--symbols", "600000", "000001", "backtest"])


def test_pit_anchor_without_symbols_uses_the_full_survivor_pool():
    anchor = _research_anchor(_args())
    assert anchor is not None
    assert len(anchor) > 100


def test_pit_anchor_with_explicit_symbols_keeps_only_survivors():
    anchor = _research_anchor(_args(symbols=["600000,000001"]))
    assert anchor == ["sh600000", "sz000001"]


def test_pit_anchor_returns_none_when_no_survivor_is_selected():
    # A pool made only of delisted names has no survivor anchor; the caller
    # then keeps the plain 95%-coverage behaviour instead of raising.
    assert _research_anchor(_args(symbols=["sh600074"])) is None


def test_non_pit_universes_are_never_anchored():
    assert _research_anchor(_args(universe="broad", symbols=["600000"])) is None
