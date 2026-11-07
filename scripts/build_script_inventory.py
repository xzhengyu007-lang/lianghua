"""Inventory of every script: what it does, whether it is documented, whether it is tested.

The forward audit (RUNBOOK) checks that what the docs describe actually runs. This
is the reverse: what exists that nobody wrote down. A script that ran once, left a
report directory behind, and was never described is the part of a research repo
that quietly becomes debt -- nobody knows whether it is a result or an accident.

The inventory is generated (SCRIPTS.md) and compared against a fresh render by a
test, so it cannot drift.

Usage::

    python scripts/build_script_inventory.py
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"


def first_docstring_line(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return "(does not parse)"
    doc = ast.get_docstring(tree) or ""
    for line in doc.splitlines():
        line = line.strip()
        if line:
            return line
    return "(no docstring)"


def category(name: str) -> str:
    for prefix in ("run_", "build_", "audit_", "serve_", "download_", "check_"):
        if name.startswith(prefix):
            return prefix.rstrip("_")
    return "other"


def collect() -> dict:
    log = (ROOT / "RESEARCH_LOG.md").read_text(encoding="utf-8") if (ROOT / "RESEARCH_LOG.md").exists() else ""
    tests = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py")}

    rows = []
    for path in sorted(SCRIPTS.glob("*.py")):
        name = path.name
        documented = name in log
        tested = any(name in text for text in tests.values())
        rows.append(
            {
                "script": name,
                "category": category(name),
                "purpose": first_docstring_line(path),
                "documented": documented,
                "tested": tested,
                "lines": len(path.read_text(encoding="utf-8").splitlines()),
            }
        )
    return {"scripts": rows}


def render(payload: dict) -> str:
    rows = payload["scripts"]
    orphans = [r for r in rows if not r["documented"] and not r["tested"]]
    documented = [r for r in rows if r["documented"]]
    tested = [r for r in rows if r["tested"]]

    lines = ["# 脚本清单", ""]
    lines.append("> 由 `scripts/build_script_inventory.py` 生成，**不要手改**；"
                 "`tests/test_script_inventory.py` 会比对重新生成的结果。")
    lines.append("")
    lines.append("| 项 | 数量 |")
    lines.append("|---|---:|")
    lines.append("| 脚本总数 | %d |" % len(rows))
    lines.append("| 在研究日志里出现过 | %d |" % len(documented))
    lines.append("| 有对应测试 | %d |" % len(tested))
    lines.append("| **两者都没有（孤儿）** | **%d** |" % len(orphans))
    lines.append("")
    if orphans:
        lines.append("## 孤儿脚本（未记录、未测试）")
        lines.append("")
        lines.append("| 脚本 | 类别 | 行数 | 用途（取自 docstring） |")
        lines.append("|---|---|---:|---|")
        for row in orphans:
            lines.append("| `%s` | %s | %d | %s |"
                         % (row["script"], row["category"], row["lines"], row["purpose"]))
        lines.append("")
    lines.append("## 全部脚本")
    lines.append("")
    lines.append("| 脚本 | 类别 | 记录 | 测试 | 用途 |")
    lines.append("|---|---|:--:|:--:|---|")
    for row in rows:
        lines.append("| `%s` | %s | %s | %s | %s |"
                     % (row["script"], row["category"],
                        "✓" if row["documented"] else "—",
                        "✓" if row["tested"] else "—",
                        row["purpose"]))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "SCRIPTS.md"))
    parser.add_argument("--json-output", default=str(ROOT / "reports" / "script_inventory.json"))
    args = parser.parse_args(argv)

    payload = collect()
    text = render(payload)
    Path(args.output).write_text(text, encoding="utf-8")
    out = Path(args.json_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = payload["scripts"]
    orphans = [r for r in rows if not r["documented"] and not r["tested"]]
    print("scripts=%d documented=%d tested=%d orphans=%d"
          % (len(rows), sum(r["documented"] for r in rows), sum(r["tested"] for r in rows), len(orphans)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
