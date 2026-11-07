"""Guard for the generated table of contents in RESEARCH_LOG.md.

The TOC is generated, so the thing worth testing is the generator plus the
fact that the checked-in file was actually regenerated after the last section
was appended (a stale TOC is worse than no TOC).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "build_research_toc", ROOT / "scripts" / "build_research_toc.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


toc = _load_module()

SAMPLE = """
# 日志

## 1. 起因

### 23.1 第一个结论

### 23.2 第二个结论
"""


def test_build_toc_lists_every_section():
    block = toc.build_toc(SAMPLE)
    assert "- 23.1 第一个结论" in block
    assert "- 23.2 第二个结论" in block


def test_apply_toc_inserts_before_the_first_section():
    updated = toc.apply_toc(SAMPLE)
    assert updated.index(toc.START) < updated.index("## 1. 起因")


def test_apply_toc_is_idempotent():
    once = toc.apply_toc(SAMPLE)
    twice = toc.apply_toc(once)
    assert once == twice


def test_check_mode_flags_a_stale_file(tmp_path: Path):
    path = tmp_path / "log.md"
    path.write_text(SAMPLE, encoding="utf-8")
    script = ROOT / "scripts" / "build_research_toc.py"
    stale = subprocess.run(
        [sys.executable, str(script), "--path", str(path), "--check"],
        capture_output=True,
        text=True,
    )
    assert stale.returncode == 1

    subprocess.run([sys.executable, str(script), "--path", str(path)], check=True)
    fresh = subprocess.run(
        [sys.executable, str(script), "--path", str(path), "--check"],
        capture_output=True,
        text=True,
    )
    assert fresh.returncode == 0


def test_repository_log_toc_is_current():
    path = ROOT / "RESEARCH_LOG.md"
    if not path.exists():
        import pytest

        pytest.skip("no research log")
    text = path.read_text(encoding="utf-8")
    assert toc.apply_toc(text) == text, "run scripts/build_research_toc.py"
