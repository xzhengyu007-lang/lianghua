"""Build a self-contained visual dashboard over the research artifacts.

Reads reports/*/metrics.json + equity.csv, FINDINGS.md and RESEARCH_LOG.md and
emits one offline HTML file with every number embedded. No server, no CDN, no
extra dependency -- open the output and it works.

Usage::

    python scripts/build_dashboard.py
"""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TICK = chr(96)

METRIC_KEYS = [
    "total_return",
    "cagr",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "win_rate",
    "turnover",
    "exposure",
    "alpha",
    "beta",
]

BASE_DATE = date(2019, 1, 1)


def downsample(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return values
    step = (len(values) - 1) / (limit - 1)
    return [values[round(i * step)] for i in range(limit)]


def load_reports(reports_dir: Path, max_points: int) -> list[dict]:
    rows: list[dict] = []
    for directory in sorted(reports_dir.iterdir()):
        if not directory.is_dir():
            continue
        metrics_path = directory / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = payload.get("metrics") or {}
        row: dict = {
            "name": directory.name,
            "mtime": datetime.fromtimestamp(metrics_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            "start": payload.get("start"),
            "end": payload.get("end"),
            "metrics": {key: metrics.get(key) for key in METRIC_KEYS if metrics.get(key) is not None},
        }
        equity_path = directory / "equity.csv"
        if equity_path.exists():
            frame = None
            try:
                frame = pd.read_csv(equity_path, usecols=["date", "equity"])
            except (ValueError, FileNotFoundError):
                frame = None
            if frame is not None and len(frame) > 1:
                frame["date"] = pd.to_datetime(frame["date"])
                frame = frame.dropna(subset=["equity"])
                base = float(frame["equity"].iloc[0])
                if base and len(frame) > 1:
                    pct = ((frame["equity"] / base) - 1.0) * 100.0
                    offsets = [(d.date() - BASE_DATE).days for d in frame["date"]]
                    kept = downsample(list(range(len(pct))), max_points)
                    row["eq"] = [[int(offsets[i]), round(float(pct.iloc[i]), 3)] for i in kept]
                    row["eq_start"] = str(frame["date"].iloc[0].date())
                    row["eq_end"] = str(frame["date"].iloc[-1].date())
        rows.append(row)
    return rows


def parse_findings(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 5 or not cells[0].isdigit():
            continue
        rows.append(
            {
                "id": int(cells[0]),
                "claim": cells[1],
                "numbers": cells[2],
                "confidence": cells[3],
                "ref": cells[4],
            }
        )
    return rows


def inline(text: str) -> str:
    text = html.escape(text)
    code = TICK + r"([^" + TICK + r"]+)" + TICK
    text = re.sub(code, r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)
    return text


def render_table(block: list[str]) -> str:
    rows = []
    for raw in block:
        cells = [c.strip() for c in raw.strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    if not rows:
        return ""
    head, body = rows[0], rows[1:]
    parts = ["<table><thead><tr>"]
    parts.extend("<th>" + inline(c) + "</th>" for c in head)
    parts.append("</tr></thead><tbody>")
    for row in body:
        parts.append("<tr>")
        parts.extend("<td>" + inline(c) + "</td>" for c in row)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def render_markdown(lines: list[str]) -> str:
    out: list[str] = []
    index, total = 0, len(lines)
    while index < total:
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("|"):
            block = []
            while index < total and lines[index].strip().startswith("|"):
                block.append(lines[index].strip())
                index += 1
            out.append(render_table(block))
            continue
        if stripped.startswith("#"):
            level = min(len(stripped) - len(stripped.lstrip("#")), 6)
            out.append("<h%d>%s</h%d>" % (level, inline(stripped[level:].strip()), level))
            index += 1
            continue
        if stripped.startswith(("- ", "* ")):
            items: list[str] = []
            while index < total and lines[index].strip().startswith(("- ", "* ")):
                items.append(lines[index].strip()[2:])
                index += 1
                while index < total and lines[index].startswith("  ") and lines[index].strip():
                    items[-1] += " " + lines[index].strip()
                    index += 1
            out.append("<ul>" + "".join("<li>" + inline(i) + "</li>" for i in items) + "</ul>")
            continue
        if re.match(r"^\d+\.\s", stripped):
            numbered: list[str] = []
            while index < total and re.match(r"^\d+\.\s", lines[index].strip()):
                numbered.append(re.sub(r"^\d+\.\s+", "", lines[index].strip()))
                index += 1
                while index < total and lines[index].startswith("   ") and lines[index].strip():
                    numbered[-1] += " " + lines[index].strip()
                    index += 1
            out.append("<ol>" + "".join("<li>" + inline(i) + "</li>" for i in numbered) + "</ol>")
            continue
        paragraph = []
        while index < total:
            current = lines[index].strip()
            if not current or current.startswith(("|", "#", "- ", "* ")) or re.match(r"^\d+\.\s", current):
                break
            paragraph.append(current)
            index += 1
        out.append("<p>" + inline(" ".join(paragraph)) + "</p>")
    return "\n".join(out)


def parse_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"<!-- TOC -->.*?<!-- /TOC -->", "", text, flags=re.S)
    sections: list[dict] = []
    current: dict | None = None
    buffer: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^(#{2,3})\s+(\d+(?:\.\d+)?)\.?\s+(.+)$", line)
        if match:
            if current is not None:
                current["html"] = render_markdown(buffer)
                sections.append(current)
            current = {"id": match.group(2), "title": match.group(3).strip()}
            buffer = []
            continue
        if current is not None:
            buffer.append(line)
    if current is not None:
        current["html"] = render_markdown(buffer)
        sections.append(current)
    return sections


def build_data(max_points: int = 320) -> dict:
    """Collect every artifact the page shows. Reused by the live server."""
    reports = load_reports(ROOT / "reports", max_points)
    findings = parse_findings(ROOT / "FINDINGS.md")
    sections = parse_log(ROOT / "RESEARCH_LOG.md")
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "counts": {
            "reports": len(reports),
            "with_equity": sum(1 for r in reports if r.get("eq")),
            "findings": len(findings),
            "sections": len(sections),
        },
        "reports": reports,
        "findings": findings,
        "sections": sections,
    }


def render_page(data: dict) -> str:
    """Inject the payload into the template. The placeholder is a JS literal."""
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    template = (ROOT / "scripts" / "dashboard_template.html").read_text(encoding="utf-8")
    return template.replace("/*__DATA__*/null", payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "dashboard"))
    parser.add_argument("--max-points", type=int, default=320)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("reading reports...", flush=True)
    data = build_data(args.max_points)
    page = render_page(data)
    target = out / "index.html"
    target.write_text(page, encoding="utf-8")
    counts = data["counts"]
    print("reports=%d findings=%d sections=%d" % (counts["reports"], counts["findings"], counts["sections"]))
    print("size=%.0f KiB" % (target.stat().st_size / 1024))
    print("dashboard: %s" % target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
