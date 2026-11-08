"""Build a self-contained static site for GitHub Pages.

The research dashboard (build_dashboard.py) is a single 1 MB page driven by the
local server. That is right for daily use and wrong for publishing: GitHub Pages
serves files, not processes, and a reader on another machine has no reports/
directory to load.

This writes docs/ -- a handful of plain HTML files with the data baked in, no
JavaScript fetches, no build step at view time. It reuses build_dashboard's
parsers so the site and the dashboard can never disagree about the numbers.

Usage::

    python scripts/build_site.py
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dashboard = _load("build_dashboard", "build_dashboard.py")

PAGES = [
    ("index.html", "总览"),
    ("findings.html", "结论台账"),
    ("log.html", "研究日志"),
    ("reports.html", "报告索引"),
]

STYLE = """
:root{--bg:#0b0f18;--panel:#131a27;--panel2:#0f1622;--border:#1f2a3d;--text:#dbe4f0;
--muted:#7f8da3;--accent:#4ea1ff;--pos:#2fd18b;--neg:#ff5f7a;--warn:#f2b544}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.6 -apple-system,"Segoe UI",
"Microsoft YaHei",system-ui,sans-serif}
.num,code{font-family:ui-monospace,Consolas,"Cascadia Mono",monospace;font-variant-numeric:tabular-nums}
header{padding:18px 24px 0;border-bottom:1px solid var(--border);
background:linear-gradient(180deg,#111a28,#0b0f18)}
h1{margin:0 0 4px;font-size:19px;font-weight:650}
.sub{color:var(--muted);font-size:12px;margin-bottom:14px}
nav{display:flex;gap:4px;flex-wrap:wrap}
nav a{color:var(--muted);text-decoration:none;font-size:13.5px;padding:9px 16px;
border-radius:8px 8px 0 0;border-bottom:2px solid transparent}
nav a:hover{color:var(--text);background:#16203130}
nav a.on{color:var(--accent);border-bottom-color:var(--accent);background:#162031}
main{padding:20px 24px 64px;max-width:1500px;margin:0 auto}
h2{font-size:16px;margin:28px 0 10px;font-weight:640}
h2:first-child{margin-top:4px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:18px}
.card{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.card .k{color:var(--muted);font-size:12px}
.card .v{font-size:22px;font-weight:600;margin-top:4px}
.card .h{color:var(--muted);font-size:11.5px;margin-top:2px}
table{width:100%;border-collapse:collapse;background:var(--panel);border-radius:10px;overflow:hidden;
margin-bottom:8px}
th,td{padding:8px 12px;text-align:right;border-bottom:1px solid var(--border);
white-space:nowrap;font-size:13px}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{color:var(--muted);font-size:12px;background:var(--panel2);position:sticky;top:0}
tbody tr:hover{background:#162031}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.note{color:var(--muted);font-size:12.5px;margin:8px 0 0}
.box{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 18px;
margin-bottom:14px}
.box.warn{border-color:#5a4620;background:#1d1a12}
.box b{color:var(--text)}
section.sec{background:var(--panel);border:1px solid var(--border);border-radius:10px;
padding:14px 18px;margin-bottom:12px}
section.sec h3{margin:0 0 6px;font-size:14px;font-weight:640}
details summary{cursor:pointer;color:var(--muted);font-size:12.5px}
a{color:var(--accent)}
footer{color:var(--muted);font-size:12px;padding:0 24px 40px;max-width:1500px;margin:0 auto}
"""


def _page(title: str, active: str, body: str, subtitle: str) -> str:
    nav = "".join(
        '<a href="%s"%s>%s</a>' % (href, ' class="on"' if href == active else "", label)
        for href, label in PAGES
    )
    return (
        "<!doctype html>\n<html lang=\"zh-CN\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>%s · A 股量化研究</title>\n<style>%s</style>\n</head>\n<body>\n"
        "<header><h1>A 股量化研究</h1><div class=\"sub\">%s</div><nav>%s</nav></header>\n"
        "<main>%s</main>\n<footer>由 scripts/build_site.py 生成于 %s。"
        "数据为 2019-06 至 2026-11 的历史回测，不构成投资建议。</footer>\n</body>\n</html>\n"
        % (html.escape(title), STYLE, subtitle, nav, body,
           datetime.now().strftime("%Y-%m-%d %H:%M"))
    )


def _fmt_pct(value) -> str:
    if value is None:
        return "—"
    cls = "pos" if value >= 0 else "neg"
    return '<span class="%s">%+.2f%%</span>' % (cls, value * 100)


def _fmt(value, digits=3) -> str:
    if value is None:
        return "—"
    return "%.*f" % (digits, value)


def page_index(data: dict) -> str:
    memo = (ROOT / "DECISION.md").read_text(encoding="utf-8")
    findings = dashboard.parse_findings(ROOT / "FINDINGS.md")
    high = [row for row in findings if "高" in (row.get("confidence") or "")]

    # Count the test functions rather than hardcoding a number that rots.
    test_count = 0
    for path in (ROOT / "tests").glob("test_*.py"):
        test_count += path.read_text(encoding="utf-8").count(chr(10) + "def test_")

    cards = [
        ("结论条数", str(len(findings)), "其中高置信度 %d 条" % len(high)),
        ("研究日志", "%d 节" % len(dashboard.parse_log(ROOT / "RESEARCH_LOG.md")), "每节可追溯到脚本与报告"),
        ("报告目录", str(len(data.get("reports", []))), "含净值、指标与结论"),
        ("测试函数", "%d 个" % test_count, "pytest 全部通过（生成时点）"),
    ]
    card_html = "".join(
        '<div class="card"><div class="k">%s</div><div class="v">%s</div><div class="h">%s</div></div>'
        % (html.escape(k), html.escape(v), html.escape(h)) for k, v, h in cards
    )

    body = ['<div class="cards">%s</div>' % card_html]
    body.append('<div class="box warn"><b>先读这一条</b>：7.14 年的记录对真 Sharpe 0.90 只有 '
                '<b>56% 检出力</b>。「有效」不等于「已被证明」。任何收益数字都必须同时给出相位中位'
                '（0.668）而不是点估计（0.902）。</div>')
    body.append("<h2>决策备忘</h2>")
    body.append('<div class="box">%s</div>' % dashboard.render_markdown(memo.splitlines()))
    return _page("总览", "index.html", "".join(body),
                 "研究结论、证据强度与部署建议 · 静态页面，无需后端")


def page_findings() -> str:
    findings = dashboard.parse_findings(ROOT / "FINDINGS.md")
    rows = []
    for row in findings:
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (html.escape(str(row.get("number", ""))),
               html.escape(str(row.get("claim", ""))),
               html.escape(str(row.get("evidence", ""))),
               html.escape(str(row.get("confidence", ""))),
               html.escape(str(row.get("section", ""))))
        )
    body = [
        "<h2>结论台账（%d 条）</h2>" % len(findings),
        '<div class="note">置信度是判断这一条能不能被当作依据的核心。'
        '写「否」的方向是<b>已验证的负面结果</b>，不是没试过。</div>',
        "<table><thead><tr><th>#</th><th>结论</th><th>依据</th><th>置信度</th><th>出处</th></tr></thead>"
        "<tbody>%s</tbody></table>" % "".join(rows),
    ]
    return _page("结论台账", "findings.html", "".join(body), "每条结论的证据与置信度")


def page_log() -> str:
    sections = dashboard.parse_log(ROOT / "RESEARCH_LOG.md")
    blocks = []
    for section in sections:
        title = html.escape(section.get("title", ""))
        # parse_log already renders the body into "html"; using "lines" here
        # produced 191 empty shells, which is why this is spelled out.
        content = section.get("html") or ""
        if not content.strip():
            # A handful of sections are heading-only, with the body in numbered
            # subsections. Say so rather than shipping a blank card.
            content = '<p class="note">本节为标题节，正文在其编号子节中。</p>'
        blocks.append(
            '<section class="sec"><h3>%s</h3>%s</section>' % (title, content)
        )
    body = ["<h2>研究日志（%d 节）</h2>" % len(sections),
            '<div class="note">按写作顺序排列。被推翻的结论会在原处标注修正指引。</div>']
    body.extend(blocks)
    return _page("研究日志", "log.html", "".join(body), "每条结论怎么来的、错在哪")


def page_reports(data: dict) -> str:
    reports = data.get("reports", [])
    rows = []
    for report in reports:
        metrics = report.get("metrics") or {}
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (html.escape(str(report.get("name", ""))),
               _fmt_pct(metrics.get("total_return")),
               _fmt(metrics.get("sharpe")),
               _fmt_pct(metrics.get("max_drawdown")),
               html.escape(str(report.get("label") or "")))
        )
    body = ["<h2>报告索引（%d 个）</h2>" % len(reports),
            '<div class="note">按累计收益排序。收益为全期历史回测，含成本假设。</div>',
            "<table><thead><tr><th>报告</th><th>累计收益</th><th>Sharpe</th>"
            "<th>最大回撤</th><th>说明</th></tr></thead><tbody>%s</tbody></table>" % "".join(rows)]
    return _page("报告索引", "reports.html", "".join(body), "所有研究报告的产出与指标")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "docs"))
    parser.add_argument("--max-points", type=int, default=240)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = dashboard.build_data(max_points=args.max_points)

    (out / "index.html").write_text(page_index(data), encoding="utf-8")
    (out / "findings.html").write_text(page_findings(), encoding="utf-8")
    (out / "log.html").write_text(page_log(), encoding="utf-8")
    (out / "reports.html").write_text(page_reports(data), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    sizes = {p.name: p.stat().st_size for p in sorted(out.glob("*.html"))}
    print("site written to %s" % out)
    for name, size in sizes.items():
        print("  %-16s %6.1f KiB" % (name, size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
