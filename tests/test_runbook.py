"""Guard: the runbook must reference things that exist.

A runbook rots faster than code: it names scripts, subcommands and expected exit
codes, and nothing would notice if one of them were renamed. These checks are
cheap and catch exactly that.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNBOOK = ROOT / "RUNBOOK.md"


def _text() -> str:
    if not RUNBOOK.exists():
        pytest.skip("RUNBOOK.md is missing")
    return RUNBOOK.read_text(encoding="utf-8")


def _subcommands() -> set[str]:
    from quant_system.cli import build_parser

    parser = build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return set(action.choices)


def test_every_referenced_script_exists():
    missing = [
        name for name in re.findall(r"scripts[\\/]([A-Za-z0-9_]+\.py)", _text())
        if not (ROOT / "scripts" / name).exists()
    ]
    assert missing == [], missing


def test_every_referenced_cli_subcommand_exists():
    known = _subcommands()
    problems = []
    for line in _text().splitlines():
        if "-m quant_system" not in line:
            continue
        tokens = re.findall(r"[a-z][a-z\-]{2,}", line)
        if not (set(tokens) & known):
            problems.append(line.strip()[:80])
    assert problems == [], problems


def test_runbook_documents_the_known_nonzero_exit():
    """cache-audit exits 1 by design; forgetting that is the first trap."""
    text = _text()
    assert "非零退出但属正常" in text
    assert "cache-audit" in text
    assert "§23.19" in text


def test_no_temporary_directories_are_tracked():
    """Repo hygiene: pytest's basetemp once leaked 246 files into the index."""
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "ls-files"], cwd=str(ROOT), capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git is unavailable")
    if completed.returncode != 0:
        pytest.skip("not a git checkout")
    junk = [
        line for line in completed.stdout.splitlines()
        if "pytest-of-" in line or "dsh-spill" in line or line.startswith("pytest-")
    ]
    assert junk == [], junk[:5]


def test_runbook_points_at_the_authority_documents():
    text = _text()
    for name in ("DECISION.md", "FINDINGS.md", "RESEARCH_LOG.md", "README.md"):
        assert name in text, name


def test_runbook_carries_the_six_conclusion_checks():
    """The self-check list is the operational form of 23.107 and 23.118."""
    text = _text()
    assert "写结论前的自检清单" in text
    for marker in ("适用范围", "一次改了几个变量", "能不能生成", "扫描的边界", "单相位", "换表达"):
        assert marker in text, marker
    assert "§23.118" in text
    assert "修订必须写回原处" in text


def test_runbook_lists_the_three_self_checks():
    text = _text()
    for script in ("run_verification.py", "run_publish_check.py", "audit_staleness.py"):
        assert script in text, script
