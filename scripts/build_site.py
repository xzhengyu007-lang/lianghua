"""Build the published site: application-oriented, one job per page.

The second pass crammed the memo, the ledger summary, the analysis and the charts
onto one page. This pass splits by *purpose* instead of by data source, because a
reader arrives with one of three questions:

    应用  今天该做什么            总览 / 每日
    研究  为什么是它、有多可信    策略 / 证据 / 风险 / 解读
    档案  依据在哪                台账 / 日志 / 报告

Design constraints still come from the host, not from taste: GitHub Pages serves
files only, so everything is inline, no CDN, no build step at view time.

Usage::

    python scripts/build_site.py
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
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

NAV = [
    ("应用", [("index.html", "总览"), ("daily.html", "每日"), ("market.html", "市场")]),
    ("研究", [("strategy.html", "策略"), ("evidence.html", "证据"),
              ("risk.html", "风险"), ("analysis.html", "解读")]),
    ("档案", [("findings.html", "台账"), ("log.html", "日志"), ("reports.html", "报告")]),
]
PAGES = [(href, label) for _g, items in NAV for href, label in items]

STYLE = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#070a10; --s1:#0e1420; --s2:#131b2a;
  --line:#1c2636; --line2:#2a3a52;
  --tx:#e9effa; --d1:#8798b0; --d2:#5b6a80;
  --a1:#5b9dff; --a2:#8b5cf6; --ok:#22d39a; --bad:#ff5c7c; --warn:#f5b942;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 12px 32px -18px rgba(0,0,0,.9);
}
html{-webkit-text-size-adjust:100%}
body.light{
  --bg:#f6f8fc; --s1:#ffffff; --s2:#f1f4f9;
  --line:#e3e8f0; --line2:#cfd8e6;
  --tx:#121a28; --d1:#4a5568; --d2:#7b8794;
  --shadow:0 1px 2px rgba(16,24,40,.06), 0 8px 24px -16px rgba(16,24,40,.28);
}
body.light{background:radial-gradient(1000px 480px at 10% -10%, rgba(91,157,255,.10), transparent 60%),var(--bg)}
body.light header{background:rgba(255,255,255,.86)}
body.light nav a.on{color:var(--a1);background:rgba(91,157,255,.10)}
body.light code{background:rgba(91,157,255,.12);color:#245ac2}
body.light thead th{background:var(--s2);color:#5b6a80}
body.light .q{background:var(--s2)}
#theme{background:var(--s2);border:1px solid var(--line);color:var(--d1);cursor:pointer;
  border-radius:8px;padding:5px 10px;font-size:12.5px;margin-left:8px}
#theme:hover{color:var(--tx);border-color:var(--line2)}
body{
  background:
    radial-gradient(1100px 520px at 12% -8%, rgba(91,157,255,.09), transparent 62%),
    radial-gradient(900px 460px at 92% -6%, rgba(139,92,246,.08), transparent 58%),
    var(--bg);
  color:var(--tx); min-height:100vh;
  font:14px/1.62 ui-sans-serif,-apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.num,code{font-family:ui-monospace,"SF Mono",Consolas,monospace;font-variant-numeric:tabular-nums}
code{font-size:.9em;background:rgba(91,157,255,.10);color:#a9c8ff;padding:1px 6px;border-radius:5px}
.wrap{max-width:1180px;margin:0 auto;padding:0 26px}

header{position:sticky;top:0;z-index:60;backdrop-filter:blur(16px) saturate(1.4);
  background:rgba(7,10,16,.82);border-bottom:1px solid var(--line)}
.hd{display:flex;align-items:center;gap:18px;min-height:60px;flex-wrap:wrap;padding:8px 0}
.brand{display:flex;align-items:center;gap:10px;font-weight:650;font-size:15px}
.logo{width:25px;height:25px;border-radius:7px;flex:none;position:relative;
  background:linear-gradient(135deg,var(--a1),var(--a2));
  box-shadow:0 0 20px -6px rgba(91,157,255,.9)}
.logo::after{content:"";position:absolute;inset:5px;border-radius:3px;
  border:1.5px solid rgba(255,255,255,.9);border-bottom-color:transparent;border-right-color:transparent}
nav{display:flex;align-items:center;gap:2px;margin-left:auto;flex-wrap:wrap}
nav .grp{color:var(--d2);font-size:10.5px;letter-spacing:1.4px;text-transform:uppercase;
  font-weight:700;padding:0 6px 0 14px}
nav a{color:var(--d1);font-size:13px;padding:6px 12px;border-radius:7px;transition:.15s;white-space:nowrap}
nav a:hover{color:var(--tx);background:rgba(255,255,255,.05)}
nav a.on{color:#fff;background:linear-gradient(135deg,rgba(91,157,255,.24),rgba(139,92,246,.18));
  box-shadow:inset 0 0 0 1px rgba(91,157,255,.36)}

.hero{padding:40px 0 22px;max-width:820px}
.eyebrow{color:var(--a1);font-size:11.5px;letter-spacing:2px;text-transform:uppercase;
  font-weight:700;margin-bottom:10px}
h1{font-size:30px;line-height:1.22;font-weight:680;letter-spacing:-.5px;margin-bottom:12px}
h1 em{font-style:normal;background:linear-gradient(100deg,var(--a1),var(--a2));
  -webkit-background-clip:text;background-clip:text;color:transparent}
.lede{color:var(--d1);font-size:14.5px}
.lede b{color:var(--tx);font-weight:600}

.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));gap:12px;margin:22px 0}
.card{background:linear-gradient(180deg,var(--s1),rgba(14,20,32,.55));border:1px solid var(--line);
  border-radius:13px;padding:15px 17px;box-shadow:var(--shadow)}
.card .k{color:var(--d2);font-size:11px;letter-spacing:.9px;text-transform:uppercase;
  font-weight:700;margin-bottom:6px}
.card .v{font-size:24px;font-weight:660;letter-spacing:-.4px;line-height:1.16}
.card .h{color:var(--d2);font-size:11.5px;margin-top:4px}
.card.hl{border-color:rgba(91,157,255,.30);
  background:linear-gradient(180deg,rgba(91,157,255,.10),rgba(14,20,32,.7))}
.card.link{display:block;color:inherit;transition:.16s}
.card.link:hover{transform:translateY(-2px);border-color:var(--line2)}
.card .t{font-size:14.5px;font-weight:640;margin-bottom:5px}
.card .d{color:var(--d2);font-size:12.5px;line-height:1.6}

.panel{background:var(--s1);border:1px solid var(--line);border-radius:15px;padding:19px 21px;
  margin-bottom:14px;box-shadow:var(--shadow)}
.panel>h2{font-size:15px;font-weight:650;margin-bottom:3px;display:flex;align-items:center;gap:9px}
.panel>h2::before{content:"";width:3px;height:14px;border-radius:2px;flex:none;
  background:linear-gradient(180deg,var(--a1),var(--a2))}
.panel .sub{color:var(--d2);font-size:12.5px;margin-bottom:13px}
.panel.warn{border-color:rgba(245,185,66,.26);
  background:linear-gradient(180deg,rgba(245,185,66,.06),var(--s1))}

.scroll{overflow-x:auto;border:1px solid var(--line);border-radius:11px}
table{width:100%;border-collapse:collapse;font-size:12.8px}
th,td{padding:8px 12px;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}
thead th{color:var(--d2);font-size:10.8px;letter-spacing:.8px;text-transform:uppercase;
  font-weight:700;background:var(--s2);position:sticky;top:0}
th.r,td.r{text-align:right}
td.wrap{white-space:normal;min-width:220px}
tbody tr:hover{background:rgba(91,157,255,.055)}
tbody tr:last-child td{border-bottom:0}
.pos{color:var(--ok);font-weight:560}
.neg{color:var(--bad);font-weight:560}
.dim{color:var(--d2)}
.chart{width:100%;height:auto;display:block}

.b{display:inline-block;font-size:11px;font-weight:700;padding:2px 8px;border-radius:999px}
.b.hi{color:var(--ok);background:rgba(34,211,154,.13)}
.b.mid{color:var(--warn);background:rgba(245,185,66,.13)}
.b.no{color:var(--bad);background:rgba(255,92,124,.13)}
.b.na{color:var(--d2);background:rgba(255,255,255,.06)}

.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
input[type=search]{background:var(--s2);border:1px solid var(--line);color:var(--tx);
  padding:7px 12px;border-radius:8px;font:inherit;font-size:13px;min-width:210px;outline:none}
input[type=search]:focus{border-color:var(--a1)}
.chip{background:var(--s2);border:1px solid var(--line);color:var(--d1);font:inherit;font-size:12.3px;
  padding:5px 13px;border-radius:999px;cursor:pointer}
.chip.on{color:#08101c;background:linear-gradient(135deg,var(--a1),var(--a2));
  border-color:transparent;font-weight:700}

.live-top{display:flex;align-items:center;gap:9px;font-size:12.3px;color:var(--d1);
  margin-bottom:13px;flex-wrap:wrap}
.dot{width:7px;height:7px;border-radius:50%;background:var(--d2);flex:none}
.dot.on{background:var(--ok);box-shadow:0 0 0 3px rgba(34,211,154,.16)}
.dot.err{background:var(--bad);box-shadow:0 0 0 3px rgba(255,92,124,.16)}
.qgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
.q{background:var(--s2);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.q .sym{font-size:11.5px;color:var(--d1)}
.q .px{font-size:18px;font-weight:640;margin:2px 0}
.q .chg{font-size:12px;font-weight:620}
.q.up{border-color:rgba(34,211,154,.3)}
.q.down{border-color:rgba(255,92,124,.26)}

.note{color:var(--d2);font-size:12.3px;line-height:1.72}
ul.tight{margin:6px 0 0 17px;color:var(--d1);font-size:13px}
ul.tight li{margin:5px 0}
ul.tight b{color:var(--tx)}
details{border:1px solid var(--line);border-radius:11px;background:var(--s1);
  margin-bottom:9px;overflow:hidden}
details>summary{cursor:pointer;padding:12px 16px;font-size:13.5px;font-weight:600;list-style:none;
  display:flex;align-items:center;gap:10px}
details>summary::-webkit-details-marker{display:none}
details>summary::before{content:">";color:var(--a1);font-size:11px}
details>summary:hover{background:rgba(91,157,255,.05)}
details .body{padding:2px 16px 16px;border-top:1px solid var(--line)}
.sid{color:var(--d2);font-size:11.5px;font-weight:600}
footer{border-top:1px solid var(--line);margin-top:34px;padding:20px 0 38px;
  color:var(--d2);font-size:11.8px}
@media(max-width:760px){
  .wrap{padding:0 15px}
  nav{width:100%;margin-left:0;overflow-x:auto;flex-wrap:nowrap}
  h1{font-size:24px}
}
"""


THEME_SCRIPT = """
<script>
(function(){
  var KEY='aq-theme';
  function apply(mode){document.body.classList.toggle('light', mode==='light');
    try{localStorage.setItem(KEY,mode);}catch(e){}}
  var saved=null; try{saved=localStorage.getItem(KEY);}catch(e){}
  if(saved==='light'){document.body.classList.add('light');}
  var btn=document.getElementById('theme');
  if(btn) btn.onclick=function(){
    apply(document.body.classList.contains('light')?'dark':'light');};
})();
</script>
"""


def esc(value) -> str:
    return html.escape(str(value))


def badge(confidence: str) -> str:
    c = (confidence or "").strip()
    if c.startswith("高"):
        return '<span class="b hi">%s</span>' % esc(c)
    if c.startswith("否"):
        return '<span class="b no">%s</span>' % esc(c)
    if c.startswith("中"):
        return '<span class="b mid">%s</span>' % esc(c)
    return '<span class="b na">%s</span>' % esc(c or "-")


def pct(value, digits=2) -> str:
    if not isinstance(value, (int, float)):
        return '<span class="dim">-</span>'
    cls = "pos" if value >= 0 else "neg"
    return '<span class="%s">%+.*f%%</span>' % (cls, digits, value * 100)


def num(value, digits=3) -> str:
    if not isinstance(value, (int, float)):
        return '<span class="dim">-</span>'
    return "%.*f" % (digits, value)


def card(k, v, h="", hl=False) -> str:
    return ('<div class="card%s"><div class="k">%s</div><div class="v">%s</div>'
            '<div class="h">%s</div></div>' % (" hl" if hl else "", esc(k), v, esc(h)))


def link_card(href, title, desc) -> str:
    return ('<a class="card link" href="%s"><div class="t">%s</div>'
            '<div class="d">%s</div></a>' % (href, esc(title), esc(desc)))


def panel(title, body, sub="", cls="") -> str:
    head = "<h2>%s</h2>" % esc(title)
    if sub:
        head += '<div class="sub">%s</div>' % esc(sub)
    return '<div class="panel%s">%s%s</div>' % (" " + cls if cls else "", head, body)


def table(headers, rows, numeric=()) -> str:
    head = "".join(
        '<th class="r">%s</th>' % esc(h) if i in numeric else "<th>%s</th>" % esc(h)
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            cells.append('<td class="r">%s</td>' % cell if i in numeric else "<td>%s</td>" % cell)
        body.append("<tr>%s</tr>" % "".join(cells))
    return ('<div class="scroll"><table><thead><tr>%s</tr></thead>'
            "<tbody>%s</tbody></table></div>" % (head, "".join(body)))


def svg_curve(points, width=980, height=200, colour="#5b9dff") -> str:
    if not points or len(points) < 2:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys + [0.0]), max(ys + [0.0])
    pad_l, pad_r, pad_t, pad_b = 48, 12, 12, 20
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    sx = (x1 - x0) or 1
    sy = (y1 - y0) or 1

    def px(x):
        return pad_l + (x - x0) / sx * w

    def py(y):
        return pad_t + h - (y - y0) / sy * h

    line = " ".join("%.1f,%.1f" % (px(x), py(y)) for x, y in points)
    zero = py(0.0)
    area = "%.1f,%.1f " % (px(x0), zero) + line + " %.1f,%.1f" % (px(x1), zero)
    ticks = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = y0 + (y1 - y0) * frac
        ty = py(val)
        ticks.append(
            '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#1a2331" stroke-width="1"/>'
            '<text x="%d" y="%.1f" fill="#5b6a80" font-size="10" text-anchor="end" '
            'font-family="ui-monospace,monospace">%+.0f%%</text>'
            % (pad_l, ty, width - pad_r, ty, pad_l - 7, ty + 3, val)
        )
    gid = "g%d" % abs(hash((colour, len(points))))
    return (
        '<svg class="chart" viewBox="0 0 %d %d" role="img" aria-label="净值曲线">'
        '<defs><linearGradient id="%s" x1="0" y1="0" x2="0" y2="1">'
        '<stop offset="0%%" stop-color="%s" stop-opacity=".26"/>'
        '<stop offset="100%%" stop-color="%s" stop-opacity="0"/></linearGradient></defs>%s'
        '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#2a3a52" stroke-width="1" '
        'stroke-dasharray="3 3"/>'
        '<polygon points="%s" fill="url(#%s)"/>'
        '<polyline points="%s" fill="none" stroke="%s" stroke-width="1.7" '
        'stroke-linejoin="round"/></svg>'
        % (width, height, gid, colour, colour, "".join(ticks),
           pad_l, zero, width - pad_r, zero, area, gid, line, colour)
    )


def shell(title: str, active: str, body: str) -> str:
    parts = []
    for group, items in NAV:
        parts.append('<span class="grp">%s</span>' % esc(group))
        for href, label in items:
            parts.append('<a href="%s"%s>%s</a>'
                         % (href, ' class="on"' if href == active else "", esc(label)))
    return (
        '<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<meta name="color-scheme" content="dark">\n'
        "<title>%s · A 股量化研究</title>\n<style>%s</style>\n</head>\n<body>\n"
        '<header><div class="wrap hd"><div class="brand"><span class="logo"></span>'
        "A 股量化研究</div><nav>%s</nav>"
        '<button id="theme" title="切换主题">主题</button></div></header>\n'
        '<div class="wrap">%s</div>\n'
        '<footer><div class="wrap">由 <code>scripts/build_site.py</code> 生成于 %s · '
        "历史回测含成本假设，不构成投资建议。</div></footer>\n</body>\n</html>\n"
        % (esc(title), STYLE, "".join(parts), body,
           datetime.now().strftime("%Y-%m-%d %H:%M"))
        + THEME_SCRIPT
    )


def load(relative: str):
    path = ROOT / "reports" / relative
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def holdings():
    for name in ("orders.csv", "positions.csv"):
        path = ROOT / "reports" / "signal_today" / name
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig") as handle:
            rows = [r for r in csv.DictReader(handle) if r.get("symbol")]
        if rows:
            return rows
    return []


DEPLOYABLE_KEY = "blend@12%"


def deployable():
    """The published configuration's own numbers.

    These are not in any report's metrics.json -- they live in the year table's
    overall block, keyed by config. Reading a "blend" report by name instead picks
    an arbitrary study, which is how the first version of this page showed the
    wrong headline number.
    """
    payload = load("year_table_blend/summary.json")
    if not payload:
        return None
    return (payload.get("overall") or {}).get(DEPLOYABLE_KEY)


def deployable_curve(max_points: int = 240):
    path = ROOT / "reports" / "year_table_blend" / "daily_returns.csv"
    if not path.exists():
        return []
    import csv as _csv

    with path.open(encoding="utf-8-sig") as handle:
        rows = list(_csv.DictReader(handle))
    if not rows:
        return []
    equity = 1.0
    series = []
    for index, row in enumerate(rows):
        try:
            equity *= 1.0 + float(row["return"])
        except (TypeError, ValueError):
            continue
        series.append((index, (equity - 1.0) * 100.0))
    if len(series) <= max_points:
        return series
    step = len(series) / float(max_points)
    return [series[min(len(series) - 1, int(i * step))] for i in range(max_points)]


def find_report(reports, needle: str):
    for report in reports:
        if needle in report.get("name", ""):
            return report
    return None


# ---------------------------------------------------------------- pages

def page_index(ctx) -> str:
    findings = ctx["findings"]
    high = [f for f in findings if str(f.get("confidence", "")).startswith("高")]
    published = deployable() or {}
    market = load("market/snapshot.json") or {}

    cards = "".join([
        card("部署口径累计收益", pct(published.get("return")), "混合 20 相位 + 12% 目标波动", True),
        card("Sharpe", num(published.get("sharpe")), "相位中位仅 0.668", True),
        card("最大回撤", pct(published.get("max_drawdown")), "含成本假设", False),
        card("结论", "%d 条" % len(findings), "高置信度 %d 条" % len(high), False),
    ])
    chart = svg_curve(deployable_curve()) if published else ""

    strip = ""
    if market:
        breadth = market.get("breadth") or {}
        index_info = market.get("index") or {}
        cadence = market.get("cadence") or {}
        bits = ['<span class="dim">数据截至</span> <b>%s</b>' % esc(market.get("as_of", ""))]
        if index_info:
            bits.append('<span class="dim">沪深300</span> %s' % pct(index_info.get("chg_1d")))
        if breadth:
            bits.append('<span class="dim">涨/跌</span> <b>%d / %d</b>'
                        % (breadth.get("up", 0), breadth.get("down", 0)))
            bits.append('<span class="dim">MA60 上方</span> <b>%.0f%%</b>'
                        % (100 * (breadth.get("above_ma60_share") or 0)))
        if cadence:
            bits.append('<span class="dim">调仓</span> <b>每 %d 日 / %d 相位错开</b>'
                        % (cadence.get("cadence_days", 20), cadence.get("phases", 20)))
        strip = ('<div class="panel tight"><div class="bar" style="margin:0">%s</div></div>'
                 % "".join('<span>%s</span>' % b for b in bits))

    body = [
        '<div class="hero"><div class="eyebrow">A-share Low-Volatility Research</div>'
        "<h1>先看结论，再看<em>证据</em></h1>"
        '<p class="lede">一个 A 股日频量化研究仓库的公开站点：'
        "<b>策略有效，但远未到「已被证明」</b>。下面按你来的目的分流。</p></div>",
        strip,
        '<div class="grid">%s</div>' % cards,
        panel("净值曲线", chart or '<p class="note">未找到净值曲线。</p>',
              "部署口径（20 相位等权 + 12% 目标波动），虚线为零轴"),
        '<div class="grid">%s</div>' % "".join([
            link_card("daily.html", "每日应用", "今天的目标持仓、实时行情与执行注意"),
            link_card("strategy.html", "策略解剖", "收益是哪个零件赚的：趋势过滤 vs 打分混合"),
            link_card("evidence.html", "证据强度", "56% 检出力意味着什么，随机对照有多强"),
            link_card("risk.html", "风险与成本", "价差下界、执行时点、容量、已知限制"),
        ]),
        panel("三件事", '<ul class="tight">'
              "<li><b>它是「在最平静的 8 只里剔除跌破趋势的那些」</b>，不是「低波打分 + 趋势确认」。"
              "零件账显示趋势过滤承担全部贡献，打分混合平均为零。</li>"
              "<li><b>把 0.902 当期望值是错的。</b>同窗口相位中位 Sharpe 只有 0.668，跨度 0.681-1.327。</li>"
              "<li><b>相对随机选股 +1.03 Sharpe，相对最自然的替代品六格全不显著。</b>"
              "「有效」不等于「比显而易见的做法更好」。</li></ul>", cls="warn"),
    ]
    return shell("总览", "index.html", "".join(body))


def page_daily(ctx) -> str:
    rows = holdings()
    total = 0.0
    body_rows = []
    for row in rows:
        try:
            total += float(row.get("weight") or 0)
        except ValueError:
            pass
        body_rows.append([
            esc(row.get("symbol", "")),
            '<span class="dim">%s</span>' % esc(row.get("sector", "")),
            '<span class="num">%s</span>' % esc(row.get("weight", "")),
            '<span class="num">%s</span>' % esc(row.get("price", "")),
            '<span class="dim">%s</span>' % esc(row.get("flags", "") or "-"),
        ])

    body = [
        '<div class="hero"><div class="eyebrow">Daily</div>'
        "<h1>今天该<em>做什么</em></h1>"
        '<p class="lede">目标持仓来自前一交易日收盘信号，按次日开盘执行；'
        "实时价只用于对照，<b>不参与任何信号计算</b>。</p></div>",
        '<div class="grid">%s</div>' % "".join([
            card("持仓数量", "%d 只" % len(rows), "每行业上限 2 只"),
            card("合计目标权重", "%.1f%%" % (total * 100), "其余为现金缓冲"),
            card("调仓节奏", "每日有相位调仓", "20 相位各错开 1 日，按自然年重置"),
            card("目标波动", "12%", "波动目标，不是收益目标"),
        ]),
        panel("实时行情", '<div class="live-top"><span class="dot" id="dot"></span>'
              '<span id="status">正在连接…</span><span class="dim">·</span>'
              '<span id="stamp" class="num dim"></span></div>'
              '<div class="qgrid" id="quotes"></div>'
              '<div class="note" id="err" style="display:none;color:#ff5c7c"></div>',
              "浏览器直连公开行情接口（腾讯财经，CORS 开放）；取不到数据会明确报错，不显示旧价"),
        panel("目标持仓",
              table(["标的", "行业", "目标权重", "信号价", "备注"], body_rows, numeric=(2, 3))
              if body_rows else '<p class="note">未找到 reports/signal_today/orders.csv。</p>',
              "权重为次日开盘的目标，不是当前市值权重"),
        panel("执行检查清单", '<ul class="tight" id="ck">'
              "<li><label><input type=\"checkbox\"> 数据新鲜度：信号用的是前一交易日收盘，"
              "检查该日数据是否完整</label></li>"
              "<li><label><input type=\"checkbox\"> 涨跌停：开盘即封板的名字当日无法买入，"
              "按次日再试</label></li>"
              "<li><label><input type=\"checkbox\"> 整手取整：A 股 100 股一手，"
              "金额不足一手会留下残差</label></li>"
              "<li><label><input type=\"checkbox\"> 仓位上限：单票不超 25%，合计不超 95%</label></li>"
              "<li><label><input type=\"checkbox\"> 成交记录：把实际成交价记下来，"
              "与信号价的差就是真实滑点</label></li></ul>",
              "勾选状态只保存在本地浏览器，不上传"),
        panel("执行注意", '<ul class="tight">'
              "<li><b>晚一天执行代价很小</b>：实测晚 1 日约掉 0.04 Sharpe，九格区间全含 0；"
              "但斜率约 0.03/日，<b>晚一周会掉 0.15</b>。</li>"
              "<li><b>开盘集合竞价是否拿得到价，本仓没有证据。</b>若在连续竞价吃单，"
              "低价窗口的单边价差下界可达 10.5bp。</li>"
              "<li><b>涨跌停/停牌当日无法成交</b>，整手取整会留下残差仓位。</li></ul>", cls="warn"),
    ]

    script = r"""
<script>
const HOLDINGS = __HOLDINGS__;
const SYMS = HOLDINGS.map(h => h.symbol);
const dot=document.getElementById('dot'), status=document.getElementById('status');
const stamp=document.getElementById('stamp'), errBox=document.getElementById('err');
function fail(m){dot.className='dot err';status.textContent='取不到实时数据';
  errBox.style.display='block';errBox.textContent=m;}
function parse(text){const out={};text.split(';').forEach(c=>{
  const m=c.match(/v_(\w+)="([^"]*)"/);if(!m)return;const f=m[2].split('~');
  if(f.length<33)return;out[m[1]]={price:parseFloat(f[3]),chg:parseFloat(f[32]),time:f[30]};});
  return out;}
async function refresh(){
  if(!SYMS.length){fail('没有可查询的持仓标的。');return;}
  try{
    const res=await fetch('https://qt.gtimg.cn/q='+SYMS.join(','),{cache:'no-store'});
    if(!res.ok)throw new Error('HTTP '+res.status);
    const text=new TextDecoder('gbk').decode(await res.arrayBuffer());
    const q=parse(text);
    if(!Object.keys(q).length)throw new Error('返回内容为空');
    dot.className='dot on';status.textContent='已连接';errBox.style.display='none';
    const box=document.getElementById('quotes');box.innerHTML='';let newest='';
    SYMS.forEach(s=>{const d=q[s];const el=document.createElement('div');
      if(!d){el.className='q';el.innerHTML='<div class="sym">'+s+'</div>'+
        '<div class="px dim">无报价</div>';box.appendChild(el);return;}
      const up=d.chg>=0;el.className='q '+(up?'up':'down');
      el.innerHTML='<div class="sym">'+s+'</div><div class="px num">'+d.price.toFixed(2)+
        '</div><div class="chg num '+(up?'pos':'neg')+'">'+(up?'+':'')+d.chg.toFixed(2)+'%</div>';
      box.appendChild(el);if(d.time>newest)newest=d.time;});
    if(newest)stamp.textContent=newest.replace(
      /(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})/,'$1-$2-$3 $4:$5:$6');
  }catch(e){fail('请求失败：'+e.message+'。可能是网络限制或接口变更。');}
}
refresh();setInterval(refresh,15000);
</script>
""".replace("__HOLDINGS__", json.dumps(
        [{"symbol": r.get("symbol", ""), "sector": r.get("sector", "")} for r in rows],
        ensure_ascii=False))

    return shell("每日", "daily.html", "".join(body) + script)


def page_market(ctx) -> str:
    """Market recap: indices, breadth, sectors -- context, never a signal."""
    market = load("market/snapshot.json")
    if not market:
        return shell("市场", "market.html",
                     '<div class="hero"><h1>市场</h1><p class="lede">'
                     "未找到 reports/market/snapshot.json，请先运行 "
                     "<code>scripts/build_market_snapshot.py</code>。</p></div>")

    index_info = market.get("index") or {}
    breadth = market.get("breadth") or {}
    sectors = market.get("sectors") or []

    def plain(value, digits=2):
        return "-" if not isinstance(value, (int, float)) else "%+.*f%%" % (digits, value * 100)

    cards = "".join([
        card("沪深 300", num(index_info.get("close"), 2),
             "日 %s · 20 日 %s" % (plain(index_info.get("chg_1d")),
                                   plain(index_info.get("chg_20d"))), True),
        card("涨 / 跌", "%d / %d" % (breadth.get("up", 0), breadth.get("down", 0)),
             "全池 %d 只，平盘 %d" % (breadth.get("names", 0), breadth.get("flat", 0))),
        card("MA60 上方", "%.0f%%" % (100 * (breadth.get("above_ma60_share") or 0)),
             "%d 只站上 60 日均线" % breadth.get("above_ma60_names", 0)),
        card("中位日涨跌", pct(breadth.get("median_ret_1d")), "横截面中位数，不是市值加权"),
    ])

    sector_rows = [[esc(s["sector"]), "%d" % s["names"], pct(s["ret_1d"]), pct(s["ret_20d"])]
                   for s in sectors]
    leaders = ", ".join(esc(s["sector"]) for s in sectors[:3]) or "-"
    laggards = ", ".join(esc(s["sector"]) for s in sectors[-3:]) or "-"

    body = [
        '<div class="hero"><div class="eyebrow">Market</div>'
        "<h1>今天是什么<em>行情</em></h1>"
        '<p class="lede">读单只之前先看整体。数据截至 <b>%s</b>。'
        "这一页是<b>对照信息</b>，不参与信号计算——广度择时在本仓已被证伪。</p></div>"
        % esc(market.get("as_of", "")),
        '<div class="grid">%s</div>' % cards,
        panel("板块（等权，按当日涨跌排序）",
              table(["板块", "只数", "当日", "近 20 日"], sector_rows, numeric=(1, 2, 3)),
              "领涨：%s ｜ 领跌：%s" % (leaders, laggards)),
        panel("为什么它不作为信号",
              '<ul class="tight">'
              "<li><b>广度择时已证伪</b>：78 个月 t = -3.9，且不是低波策略专属。</li>"
              "<li><b>上了真实撮合后失效</b>：月度算术口径的 +0.15 Sharpe 是成本幻觉。</li>"
              "<li><b>它的用途是判断当天该不该动手</b>（比如全池普跌时更容易遇到跌停无法成交），"
              "而不是预测方向。</li></ul>", cls="warn"),
    ]
    return shell("市场", "market.html", "".join(body))


def page_strategy(ctx) -> str:
    parts = load("filter_blend_ablation/summary.json")
    overlap = load("rank_overlap/summary.json")
    gate = load("window_gate/summary.json")
    extent = load("window_extent/summary.json")

    body = [
        '<div class="hero"><div class="eyebrow">Strategy</div>'
        "<h1>收益是<em>哪个零件</em>赚的</h1>"
        '<p class="lede">同一 PIT 池、同行业上限、同节奏、同风控下，把策略拆成'
        "「趋势过滤」与「打分混合」两个开关做 2x2。结论：<b>只有一个零件在干活</b>。</p></div>",
    ]

    order = ["过滤+混合（发布）", "只过滤（纯低波打分）", "只混合（不过滤）", "都关（朴素）"]
    windows = ["2019-2021", "2022-2023", "2024-2026"]

    if parts:
        lookup = {}
        for row in parts["rows"]:
            lookup[(row["variant"], row["window"])] = row
        rows = []
        for name in order:
            cells = ["<b>%s</b>" % esc(name) if name == order[0] else esc(name)]
            for w in windows:
                cells.append(num(lookup.get((name, w), {}).get("sharpe")))
            cells.append('<span class="num">%s</span>'
                         % num(lookup.get((name, windows[0]), {}).get("turnover"), 1))
            rows.append(cells)
        body.append(panel("2x2 消融（Sharpe）",
                          table(["变体"] + windows + ["换手"], rows, numeric=(1, 2, 3, 4)),
                          "同池同约束，只改资格规则与打分方式"))

        def delta(a, b, w):
            ra, rb = lookup.get((a, w)), lookup.get((b, w))
            if not ra or not rb:
                return 0.0
            return ra["sharpe"] - rb["sharpe"]

        effect = [[esc(w),
                   "<span class=\"num\">%+.2f</span>" % delta(order[0], order[2], w),
                   "<span class=\"num\">%+.2f</span>" % delta(order[0], order[1], w)]
                  for w in windows]
        body.append(panel("边际效应（相对发布策略）",
                          table(["窗口", "加趋势过滤", "加 0.8/0.2 混合"], effect, numeric=(1, 2)),
                          "过滤承担全部贡献；混合在三段里时正时负、平均为零"))

    if overlap:
        body.append(panel(
            "混合到底改不改变选股",
            '<div class="grid">%s</div>' % "".join([
                card("平均重合", "%.2f / %d" % (overlap["mean_overlap"], overlap["top_n"]),
                     "每期换掉约 %.1f 只" % (overlap["top_n"] - overlap["mean_overlap"])),
                card("完全相同占比", "%.0f%%" % (100 * overlap["share_identical"]),
                     "其余调仓都会换名"),
                card("换名的 20 日边际", "%+.2f%%" % (100 * (overlap["edge_mean"] or 0)),
                     "均值由右尾拉起，胜率 49%"),
            ]),
            "它确实在动，但换掉的那只没有信息"))

    if gate and "SMA250" in gate:
        g120, g250 = gate["SMA120"], gate["SMA250"]
        rows = [
            ["全期 Sharpe", num(g120["full_sharpe"]), num(g250["full_sharpe"])],
            ["全期最大回撤", pct(g120["full_max_drawdown"]), pct(g250["full_max_drawdown"])],
            ["年换手", num(g120["full_turnover"], 1), num(g250["full_turnover"], 1)],
            ["相位跨度",
             "%.3f-%.3f" % (g120["phase_span"]["min"], g120["phase_span"]["max"]),
             "%.3f-%.3f" % (g250["phase_span"]["min"], g250["phase_span"]["max"])],
        ]
        body.append(panel("趋势窗口：测过但没采纳",
                          table(["指标", "SMA120（发布）", "SMA250"], rows, numeric=(1, 2)),
                          "250 在样本内全面更好，但推到 500 后曲线见顶回落，"
                          "且 250 与 300 的差小于单配置相位跨度——内点峰，不改"))

    if extent and extent.get("rows"):
        rows = [[str(r["window"]), num(r["full_sharpe"]),
                 "%.3f" % r["phase_min"], num(r["turnover"], 1)]
                for r in extent["rows"]]
        body.append(panel("窗口外延：越长越好会停在哪",
                          table(["窗口", "全期 Sharpe", "最差相位", "换手"], rows, numeric=(1, 2, 3)),
                          "60 太短、500 太长，120-300 是一段分不开的平台"))

    return shell("策略", "strategy.html", "".join(body))


def page_evidence(ctx) -> str:
    power = load("power_recheck/summary.json")
    rnd = load("random_control/summary.json")
    naive = load("naive_benchmark/summary.json")
    quart = load("quarter_consistency/summary.json")

    body = [
        '<div class="hero"><div class="eyebrow">Evidence</div>'
        "<h1>这条记录<em>有多可信</em></h1>"
        '<p class="lede">三个独立问题：统计上够不够强（检出力）、'
        "去掉选股还剩多少（对照）、方向是否稳定（胜率）。</p></div>",
    ]

    if power:
        analytic = power.get("analytic") or {}
        years = analytic.get("years_for_80pct_power") or 0
        body.append(panel(
            "检出力：7.14 年能证明什么",
            '<div class="grid">%s</div>' % "".join([
                card("当前检出力", "%.0f%%" % ((power.get("power_at_7_3y") or 0) * 100),
                     "对真 Sharpe 0.90，5% 显著性", True),
                card("80% 检出力需要", "%.1f 年" % years,
                     "还差约 %.1f 年" % max(0.0, years - 7.14)),
                card("已记录", "7.14 年", "2019-06 至 2026-11"),
            ]),
            "这条决定了另外两段怎么读：不是「无效」，是「还不够长」"))

    if rnd:
        r = rnd["random"]
        body.append(panel(
            "对照一：随机选股",
            '<div class="grid">%s</div>' % "".join([
                card("发布策略", num(rnd["real"]["sharpe"]), "单相位、12% 目标", True),
                card("随机中位", num(r["sharpe_median"]), "p95 %s" % num(r["sharpe_p95"])),
                card("随机最好的一次", num(r["sharpe_max"]), "100 次抽样"),
                card("发布策略百分位",
                     "第 %.0f 位" % (rnd["real_percentile_among_random"] * 100), "无一追平"),
            ]),
            "同池、同约束、同风控，只把打分换成随机"))

    if naive and naive.get("rows"):
        rows = []
        for variant in ("A 朴素（128 在册，无过滤无上限）", "B 朴素+PIT 池",
                        "C 发布策略（PIT+过滤+上限+混合打分）"):
            cells = [esc(variant)]
            for w in ("2019-2021", "2022-2023", "2024-2026"):
                hit = next((x for x in naive["rows"]
                            if x["variant"] == variant and x["window"] == w), None)
                cells.append(num((hit or {}).get("sharpe")))
            rows.append(cells)
        body.append(panel(
            "对照二：最自然的替代品",
            table(["变体", "2019-2021", "2022-2023", "2024-2026"], rows, numeric=(1, 2, 3)),
            "六格配对区间全部含 0——「相对随机很强」不等于「比显而易见的做法更好」", cls="warn"))

    if quart:
        rows = []
        for t in quart["tests"]:
            rows.append([esc(t["series"]), "%d" % t["n"], "%d" % t["wins"],
                         "%.0f%%" % (100 * t["win_rate"]), "%.2g" % t["p_value"]])
        body.append(panel(
            "方向稳定性：季度胜率",
            table(["序列", "季度数", "胜", "胜率", "二项 p"], rows, numeric=(1, 2, 3, 4)),
            "符号检验在本样本量的检出力是 %.0f%%（Sharpe 区间 56%%）——换检验不产生新信息"
            % (quart["sign_test_power"] * 100)))

    return shell("证据", "evidence.html", "".join(body))


def page_risk(ctx) -> str:
    floor = load("cost_floor/summary.json")
    timing = load("execution_timing/summary.json")
    capacity = load("capacity_blend/summary.json")

    body = [
        '<div class="hero"><div class="eyebrow">Risk</div>'
        "<h1>成本、执行与<em>已知的坑</em></h1>"
        '<p class="lede">这一页只讲会让实盘结果低于回测的东西，以及本仓明确没解决的事。</p></div>',
    ]

    if floor:
        rows = [[esc(r["window"]), "%.2f" % r["price_median"],
                 "%.1f bp" % r["tick_bp_weighted"], "%.1f bp" % r["half_spread_bp"],
                 "%+.1f bp" % r["excess_per_side_bp"]] for r in floor["rows"]]
        body.append(panel(
            "微观结构下界：5bp 滑点在低价窗口不可能",
            table(["窗口", "持仓价中位", "加权 tick", "单边半价差", "超出 published"],
                  rows, numeric=(1, 2, 3, 4)),
            "A 股最小变动价位 0.01 元；单边价差 = 0.5 x tick / 价格，是滑点不能低于的下界"))

    if timing and timing.get("rows"):
        rows = []
        for delay in timing.get("delays", []):
            cells = ["晚 %d 日" % delay if delay else "不延迟"]
            for w in ("2019-2021", "2022-2023", "2024-2026"):
                hit = next((x for x in timing["rows"]
                            if x["delay_days"] == delay and x["window"] == w), None)
                cells.append(num((hit or {}).get("sharpe")))
            rows.append(cells)
        body.append(panel(
            "执行时点：晚一天代价多大",
            table(["延迟", "2019-2021", "2022-2023", "2024-2026"], rows, numeric=(1, 2, 3)),
            "九格配对区间全含 0；但斜率约 0.03 Sharpe/日，晚一周会掉约 0.15"))

    if capacity and capacity.get("rows"):
        rows = []
        for row in capacity["rows"][:12]:
            rows.append([esc(str(row.get("window", ""))), esc(str(row.get("participation", ""))),
                         num(row.get("sharpe")), pct(row.get("total_return"))])
        body.append(panel("容量",
                          table(["窗口", "参与率", "Sharpe", "累计收益"], rows, numeric=(2, 3)),
                          "资金规模上升时冲击成本如何侵蚀收益"))

    body.append(panel("本仓明确没解决的事", '<ul class="tight">'
                      "<li><b>数据缓存不入库。</b>别人 clone 后必须自己下载，"
                      "数据清单只能核对是否同一份，不能保证一致。</li>"
                      "<li><b>2019-06 起的历史是下载截断</b>，不是数据源限制：调大下载参数可回溯到 2010，"
                      "但受 provider 限流，目前只完成一部分。</li>"
                      "<li><b>开盘集合竞价的成交假设未被验证</b>，需要竞价明细，本仓没有。</li>"
                      "<li><b>200+ 个非关键报告没做删除重跑实测</b>，只在关键集上验证过逐字节复现。</li>"
                      "</ul>", cls="warn"))

    return shell("风险", "risk.html", "".join(body))


def page_analysis(ctx) -> str:
    payload = load("analysis/commentary.json") or {"sections": [], "note": ""}
    blocks = []
    for section in payload.get("sections", []):
        blocks.append(
            '<div style="padding:16px 0;border-bottom:1px solid var(--line)">'
            '<div style="font-size:11px;color:var(--a1);letter-spacing:1.4px;'
            'text-transform:uppercase;font-weight:700;margin-bottom:7px">%s</div>'
            '<div style="font-size:15px;font-weight:640;margin-bottom:7px">%s</div>'
            '<div class="note">%s</div></div>'
            % (esc(section["heading"]), esc(section["verdict"]), esc(section["evidence"]))
        )
    body = [
        '<div class="hero"><div class="eyebrow">Analysis</div>'
        "<h1>机器<em>解读</em></h1>"
        '<p class="lede">下面每一句都由 reports/ 下的产物计算，没有引入新的模型或观点。</p></div>',
        panel("四段结论", "".join(blocks) or '<p class="note">未找到解读产物。</p>',
              payload.get("note", "")),
    ]
    return shell("解读", "analysis.html", "".join(body))


def page_findings(ctx) -> str:
    findings = ctx["findings"]
    rows = []
    for row in findings:
        conf = str(row.get("confidence", ""))
        tag = ("high" if conf.startswith("高") else
               "no" if conf.startswith("否") else
               "mid" if conf.startswith("中") else "na")
        rows.append(
            '<tr data-tag="%s"><td class="dim">%s</td><td class="wrap">%s</td>'
            '<td class="wrap dim">%s</td><td>%s</td><td class="dim">%s</td></tr>'
            % (tag, esc(row.get("number", "")), esc(row.get("claim", "")),
               esc(row.get("evidence", "")), badge(conf), esc(row.get("section", "")))
        )
    body = [
        '<div class="hero"><div class="eyebrow">Findings</div>'
        "<h1>结论台账 <em>%d 条</em></h1>"
        '<p class="lede">置信度决定这一条能不能当作依据。标「否」的是<b>已验证的负面结果</b>。</p></div>'
        % len(findings),
        '<div class="bar"><input type="search" id="q" placeholder="搜索结论或依据…">'
        '<button class="chip on" data-f="all">全部</button>'
        '<button class="chip" data-f="high">高</button>'
        '<button class="chip" data-f="mid">中</button>'
        '<button class="chip" data-f="no">否</button>'
        '<span class="note" id="cnt"></span></div>',
        '<div class="scroll"><table><thead><tr><th>#</th><th>结论</th><th>依据</th>'
        '<th>置信度</th><th>出处</th></tr></thead><tbody id="tb">%s</tbody></table></div>'
        % "".join(rows),
    ]
    script = """
<script>
const tb=document.getElementById('tb'),q=document.getElementById('q');
const cnt=document.getElementById('cnt'),chips=[...document.querySelectorAll('.chip')];
let tag='all';
function apply(){const term=q.value.trim().toLowerCase();let n=0;
  [...tb.rows].forEach(tr=>{const okTag=tag==='all'||tr.dataset.tag===tag;
    const okTxt=!term||tr.textContent.toLowerCase().includes(term);
    const show=okTag&&okTxt;tr.style.display=show?'':'none';if(show)n++;});
  cnt.textContent='显示 '+n+' / '+tb.rows.length+' 条';}
chips.forEach(c=>c.onclick=()=>{chips.forEach(x=>x.classList.remove('on'));
  c.classList.add('on');tag=c.dataset.f;apply();});
q.oninput=apply;apply();
</script>
"""
    return shell("台账", "findings.html", "".join(body) + script)


def page_log(ctx) -> str:
    sections = ctx["sections"]
    blocks = []
    for section in sections:
        content = section.get("html") or ""
        if not content.strip():
            content = '<p class="note">本节为标题节，正文在其编号子节中。</p>'
        text = (str(section.get("title", "")) + " " + str(section.get("id", ""))).lower()
        blocks.append(
            '<details data-text="%s"><summary><span class="sid">%s</span> %s</summary>'
            '<div class="body">%s</div></details>'
            % (esc(text), esc(section.get("id", "")), esc(section.get("title", "")), content)
        )
    body = [
        '<div class="hero"><div class="eyebrow">Log</div>'
        "<h1>研究日志 <em>%d 节</em></h1>"
        '<p class="lede">按写作顺序。被推翻的结论会在原处标注修正指引。</p></div>' % len(sections),
        '<div class="bar"><input type="search" id="q" placeholder="搜索标题或编号…">'
        '<button class="chip" id="expand">全部展开</button>'
        '<button class="chip" id="collapse">全部收起</button>'
        '<span class="note" id="cnt"></span></div>',
        '<div id="list">%s</div>' % "".join(blocks),
    ]
    script = """
<script>
const list=document.getElementById('list'),q=document.getElementById('q');
const cnt=document.getElementById('cnt'),items=[...list.querySelectorAll('details')];
function apply(){const t=q.value.trim().toLowerCase();let n=0;
  items.forEach(d=>{const show=!t||d.dataset.text.includes(t);
    d.style.display=show?'':'none';if(show)n++;});
  cnt.textContent='显示 '+n+' / '+items.length+' 节';}
q.oninput=apply;
document.getElementById('expand').onclick=()=>items.forEach(d=>d.open=true);
document.getElementById('collapse').onclick=()=>items.forEach(d=>d.open=false);
apply();
</script>
"""
    return shell("日志", "log.html", "".join(body) + script)


def page_reports(ctx) -> str:
    reports = ctx["reports"]
    rows_html = []
    for report in reports:
        m = report.get("metrics") or {}
        rows_html.append(
            '<tr><td class="wrap">%s</td><td class="r">%s</td><td class="r">%s</td>'
            '<td class="r">%s</td><td class="r">%s</td><td class="r">%s</td></tr>'
            % (esc(report.get("name", "")), pct(m.get("total_return")), num(m.get("sharpe")),
               pct(m.get("max_drawdown")), num(m.get("annual_volatility")),
               num(m.get("turnover"), 1))
        )
    top = sorted(reports, key=lambda r: (r.get("metrics") or {}).get("sharpe", -9),
                 reverse=True)[:4]

    body = [
        '<div class="hero"><div class="eyebrow">Reports</div>'
        "<h1>报告索引 <em>%d 个</em></h1>"
        '<p class="lede">全部为历史回测，含成本假设。高收益不等于可部署——'
        "相位与检出力见「证据」。</p></div>" % len(reports),
        '<div class="grid">%s</div>' % "".join(
            card(r.get("name", ""), num((r.get("metrics") or {}).get("sharpe")),
                 "Sharpe · %+.1f%%" % (100 * ((r.get("metrics") or {}).get("total_return") or 0)))
            for r in top),
        '<div class="bar"><input type="search" id="q" placeholder="按名称筛选…">'
        '<span class="note" id="cnt"></span></div>',
        '<div class="scroll"><table><thead><tr><th>报告</th><th class="r">累计收益</th>'
        '<th class="r">Sharpe</th><th class="r">最大回撤</th><th class="r">年化波动</th>'
        '<th class="r">换手</th></tr></thead><tbody id="tb">%s</tbody></table></div>'
        % "".join(rows_html),
    ]
    script = """
<script>
const tb=document.getElementById('tb'),q=document.getElementById('q'),cnt=document.getElementById('cnt');
function apply(){const t=q.value.trim().toLowerCase();let n=0;
  [...tb.rows].forEach(tr=>{const show=!t||tr.cells[0].textContent.toLowerCase().includes(t);
    tr.style.display=show?'':'none';if(show)n++;});
  cnt.textContent='显示 '+n+' / '+tb.rows.length+' 个';}
q.oninput=apply;apply();
</script>
"""
    return shell("报告", "reports.html", "".join(body) + script)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "docs"))
    parser.add_argument("--max-points", type=int, default=240)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = dashboard.build_data(max_points=args.max_points)
    ctx = {
        "reports": data.get("reports", []),
        "findings": dashboard.parse_findings(ROOT / "FINDINGS.md"),
        "sections": dashboard.parse_log(ROOT / "RESEARCH_LOG.md"),
    }

    pages = {
        "index.html": page_index(ctx),
        "daily.html": page_daily(ctx),
        "market.html": page_market(ctx),
        "strategy.html": page_strategy(ctx),
        "evidence.html": page_evidence(ctx),
        "risk.html": page_risk(ctx),
        "analysis.html": page_analysis(ctx),
        "findings.html": page_findings(ctx),
        "log.html": page_log(ctx),
        "reports.html": page_reports(ctx),
    }
    for name, content in pages.items():
        (out / name).write_text(content, encoding="utf-8")

    # The second pass published live.html; keep the URL alive after the split into
    # 每日 (targets + quotes). A stale page with an old nav is worse than a redirect.
    (out / "live.html").write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0; url=daily.html">'
        "<title>已移动到「每日」</title></head><body>"
        '<p>页面已移动到 <a href="daily.html">每日</a>。</p>'
        "</body></html>",
        encoding="utf-8",
    )
    (out / ".nojekyll").write_text("", encoding="utf-8")

    for name in pages:
        print("  %-16s %6.1f KiB" % (name, (out / name).stat().st_size / 1024))
    print("site written to %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
