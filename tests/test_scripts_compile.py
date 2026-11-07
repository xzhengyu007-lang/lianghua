"""Guard: every research script still compiles.

A published table is only reproducible if the script that produced it still runs.
One capacity script silently stopped compiling -- a report field referenced a
local variable from another function -- and nothing noticed, because no test ever
executed the scripts. A compile check is the cheapest guard that would have
caught the syntax-level version of that rot; runtime errors still need a real
run, which is why the report field itself is covered by tests/test_capacity_curve.
"""

from __future__ import annotations

import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_every_script_compiles(tmp_path):
    broken = []
    for path in sorted((ROOT / "scripts").glob("*.py")):
        try:
            py_compile.compile(
                str(path), cfile=str(tmp_path / (path.stem + ".pyc")), doraise=True
            )
        except py_compile.PyCompileError as exc:
            broken.append("%s: %s" % (path.name, exc.msg))
    assert not broken, "scripts that no longer compile:\n" + "\n".join(broken)


def test_the_repository_has_scripts_to_check():
    scripts = list((ROOT / "scripts").glob("*.py"))
    assert len(scripts) > 30
