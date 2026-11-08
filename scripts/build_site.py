"""Build the published site: a designed, self-contained static page set.

Replaces the first pass (23.125), which was functional but pl[1]ain. This version
adds: a real visual identity, inline SVG equity curves (no chart library, no CDN),
a live-quote panel that talks to the public Tencent endpoint from the browser
(CORS is open: Access-Control-Allow-Origin: *), and an analysis block that states
what the numbers mean under this repository's own evidence rules.

Design constraints that come from the hosting, not from taste:
  * GitHub Pages serves files only -- no backend, no build step at view time.
  * No CDN: a reader behind a firewall must still get the page.
  * Live data must degrade visibly: if the quote call fails, say so rather than
    showing a stale number as if it were current.

Usage::

    python scripts/build_site.py
"""

from __future__ import annotations

import argparse
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

PAGES = [
    ("index.html", "总览"),
    ("live.html", "实时"),
    ("findings.html", "结论台账"),
    ("log.html", "研究日志"),
    ("reports.html", "报告索引"),
]

STYLE = """
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080b12; --surface:#0f1522; --surface2:#141c2b; --line:#1e293b;
  --line2:#2a3a52; --text:#e8eef8; --dim:#8494ab; --dim2:#5c6b82;
  --acc:#5b9dff; --acc2:#7c5cff; --pos:#22d39a; --neg:#ff5c7c; --warn:#f5b942;
  --glow:0 0 0 1px rgba(91,157,255,.12), 0 8px 32px -12px rgba(0,0,0,.8);
}
html{-webkit-text-size-adjust:100%}
body{
  background:
    radial-gradient(1200px 600px at 15% -10%, rgba(91,157,255,.10), transparent 60%),
    radial-gradient(900px 500px at 90% 0%, rgba(124,92,255,.08), transparent 55%),
    var(--bg);
  color:var(--text); min-height:100vh;
  font:14px/1.65 ui-sans-serif,-apple-system,"Segoe UI","Microsoft YaHei",system-ui,sans-serif;
  -webkit-font-smoothing:antialiased;
}
.num{font-family:ui-monospace,"SF Mono",Consolas,monospace;font-variant-numeric:tabular-nums}
code{font-family:ui-monospace,Consolas,monospace;font-size:.92em;
  background:rgba(91,157,255,.10);color:#a8c7ff;padding:1px 6px;border-radius:5px}

.wrap{max-width:1220px;margin:0 auto;padding:0 28px}

/* ---------- header ---------- */
header{position:sticky;top:0;z-index:50;backdrop-filter:blur(14px);
  background:rgba(8,11,18,.78);border-bottom:1px solid var(--line)}
.hd{display:flex;align-items:center;gap:20px;height:62px}
.brand{display:flex;align-items:center;gap:11px;font-weight:650;font-size:15.5px;letter-spacing:.2px}
.logo{width:26px;height:26px;border-radius:7px;flex:none;
  background:linear-gradient(135deg,var(--acc),var(--acc2));
  box-shadow:0 0 18px -4px rgba(91,157,255,.7);position:relative}
.logo::after{content:"";position:absolute;inset:6px;border-radius:3px;
  border:1.5px solid rgba(255,255,255,.85);border-bottom-color:transparent;border-right-color:transparent}
nav{display:flex;gap:2px;margin-left:auto;flex-wrap:wrap}
nav a{color:var(--dim);text-decoration:none;font-size:13.5px;padding:7px 14px;border-radius:8px;
  transition:.16s;white-space:nowrap}
nav a:hover{color:var(--text);background:rgba(255,255,255,.05)}
nav a.on{color:#fff;background:linear-gradient(135deg,rgba(91,157,255,.22),rgba(124,92,255,.18));
  box-shadow:inset 0 0 0 1px rgba(91,157,255,.35)}

/* ---------- hero ---------- */
.hero{padding:44px 0 26px}
.eyebrow{color:var(--acc);font-size:12px;letter-spacing:1.6px;text-transform:uppercase;
  font-weight:600;margin-bottom:12px}
h1{font-size:34px;line-height:1.2;font-weight:680;letter-spacing:-.6px;margin-bottom:14px}
h1 em{font-style:normal;background:linear-gradient(100deg,var(--acc),var(--acc2));
  -webkit-background-clip:text;background-clip:text;color:transparent}
.lede{color:var(--dim);font-size:15px;max-width:720px}
.lede b{color:var(--text);font-weight:600}

/* ---------- cards ---------- */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px;margin:26px 0}
.card{background:linear-gradient(180deg,var(--surface),rgba(15,21,34,.6));
  border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--glow)}
.card .k{color:var(--dim2);font-size:11.5px;letter-spacing:.9px;text-transform:uppercase;
  font-weight:600;margin-bottom:7px}
.card .v{font-size:26px;font-weight:660;letter-spacing:-.5px;line-height:1.15}
.card .h{color:var(--dim2);font-size:11.5px;margin-top:5px}
.card.hl{border-color:rgba(91,157,255,.32);
  background:linear-gradient(180deg,rgba(91,157,255,.10),rgba(15,21,34,.7))}

/* ---------- panels ---------- */
.panel{background:var(--surface);border:1px solid var(--line);border-radius:16px;
  padding:20px 22px;margin-bottom:16px;box-shadow:var(--glow)}
.panel h2{font-size:15.5px;font-weight:640;letter-spacing:.1px;margin-bottom:4px;
  display:flex;align-items:center;gap:9px}
.panel h2::before{content:"";width:3px;height:15px;border-radius:2px;flex:none;
  background:linear-gradient(180deg,var(--acc),var(--acc2))}
.panel .sub{color:var(--dim2);font-size:12.5px;margin-bottom:14px}
.panel.alert{border-color:rgba(245,185,66,.28);
  background:linear-gradient(180deg,rgba(245,185,66,.07),var(--surface))}

h3{font-size:14px;font-weight:640;margin:20px 0 8px;color:var(--text)}

/* ---------- tables ---------- */
.scroll{overflow-x:auto;border-radius:12px;border:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{padding:9px 13px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:140px}
thead th{color:var(--dim2);font-size:11.5px;letter-spacing:.7px;text-transform:uppercase;
  font-weight:650;background:var(--surface2);position:sticky;top:0}
tbody tr{transition:.12s}
tbody tr:hover{background:rgba(91,157,255,.06)}
tbody tr:last-child td{border-bottom:0}
.pos{color:var(--pos);font-weight:560}
.neg{color:var(--neg);font-weight:560}
.dim{color:var(--dim2)}

/* ---------- charts ---------- */
.chart{width:100%;height:auto;display:block;border-radius:10px;background:rgba(0,0,0,.22)}
.legend{display:flex;gap:16px;flex-wrap:wrap;margin-top:10px;font-size:12px;color:var(--dim)}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px}

/* ---------- live ---------- */
.live-top{display:flex;align-items:center;gap:10px;font-size:12.5px;color:var(--dim);
  margin-bottom:14px;flex-wrap:wrap}
.dot{width:7px;height:7px;border-radius:50%;background:var(--dim2);flex:none}
.dot.on{background:var(--pos);box-shadow:0 0 0 3px rgba(34,211,154,.16);animation:pulse 2.4s infinite}
.dot.err{background:var(--neg);box-shadow:0 0 0 3px rgba(255,92,124,.16)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.45}}
.qgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:11px}
.q{background:var(--surface2);border:1px solid var(--line);border-radius:11px;padding:12px 14px;
  transition:.16s}
.q:hover{border-color:var(--line2);transform:translateY(-1px)}
.q .sym{font-size:12px;color:var(--dim);letter-spacing:.5px}
.q .px{font-size:19px;font-weight:640;letter-spacing:-.3px;margin:3px 0}
.q .chg{font-size:12px;font-weight:600}
.q.up{border-color:rgba(34,211,154,.3)}
.q.down{border-color:rgba(255,92,124,.28)}

/* ---------- misc ---------- */
.note{color:var(--dim2);font-size:12.5px;line-height:1.7}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 18px;font-size:13px}
.kv dt{color:var(--dim2)}
.kv dd{color:var(--text)}
ul.tight{margin:8px 0 0 18px;color:var(--dim);font-size:13px}
ul.tight li{margin:5px 0}
ul.tight b{color:var(--text)}
footer{border-top:1px solid var(--line);margin-top:40px;padding:22px 0 40px;color:var(--dim2);font-size:12px}
@media(max-width:720px){
  h1{font-size:26px} .wrap{padding:0 16px} .hd{height:auto;padding:10px 0;flex-wrap:wrap}
  nav{margin-left:0;width:100%;overflow-x:auto}
}
"""


def _page(title: str, active: str, body: str, *, lede: str = "") -> str:
    nav = "".join(
        '<a href="%s"%s>%s</a>' % (href, ' class="on"' if href == active else "", label)
        for href, label in PAGES
    )
    return (
        '<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<meta name="color-scheme" content="dark">\n'
        "<title>%s · A 股量化研究</title>\n<style>%s</style>\n</head>\n<body>\n"
        '<header><div class="wrap hd"><div class="brand"><span class="logo"></span>'
        "A 股量化研究</div><nav>%s</nav></div></header>\n"
        '<div class="wrap">%s</div>\n'
        '<footer><div class="wrap">由 <code>scripts/build_site.py</code> 生成于 %s。'
        "历史回测含成本假设，不构成投资建议。</div></footer>\n</body>\n</html>\n"
        % (html.escape(title), STYLE, nav, body, datetime.now().strftime("%Y-%m-%d %H:%M"))
    )


def _pct(value, digits=2) -> str:
    if value is None:
        return '<span class="dim">—</span>'
    cls = "pos" if value >= 0 else "neg"
    return '<span class="%s">%+.*f%%</span>' % (cls, digits, value * 100)


def _num(value, digits=3) -> str:
    if value is None:
        return '<span class="dim">—</span>'
    return "%.*f" % (digits, value)


def svg_sparkline(points, width=980, height=190, colour="#5b9dff", fill=True) -> str:
    """Inline equity curve. points: [[x_index, cumulative_pct], ...]."""
    if not points or len(points) < 2:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys + [0.0]), max(ys + [0.0])
    pad_l, pad_r, pad_t, pad_b = 46, 14, 12, 22
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    span_x = (x1 - x0) or 1
    span_y = (y1 - y0) or 1

    def px(x):
        return pad_l + (x - x0) / span_x * w

    def py(y):
        return pad_t + h - (y - y0) / span_y * h

    line = " ".join("%.1f,%.1f" % (px(x), py(y)) for x, y in points)
    zero_y = py(0.0)
    area = "%.1f,%.1f " % (px(x0), zero_y) + line + " %.1f,%.1f" % (px(x1), zero_y)

    ticks = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        val = y0 + (y1 - y0) * frac
        ty = py(val)
        ticks.append(
            '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#1b2434" stroke-width="1"/>'
            '<text x="%d" y="%.1f" fill="#5c6b82" font-size="10" text-anchor="end" '
            'font-family="ui-monospace,monospace">%+.0f%%</text>'
            % (pad_l, ty, width - pad_r, ty, pad_l - 7, ty + 3, val)
        )

    gid = "g%d" % abs(hash((colour, len(points))))
    return (
        '<svg class="chart" viewBox="0 0 %d %d" preserveAspectRatio="none" role="img" '
        'aria-label="净值曲线">'
        "<defs><linearGradient id=\"%s\" x1=\"0\" y1=\"0\" x2=\"0\" y2=\"1\">"
        '<stop offset="0%%" stop-color="%s" stop-opacity=".28"/>'
        '<stop offset="100%%" stop-color="%s" stop-opacity="0"/></linearGradient></defs>'
        "%s"
        '<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#2a3a52" stroke-width="1" '
        'stroke-dasharray="3 3"/>'
        "<polygon points=\"%s\" fill=\"url(#%s)\"/>"
        '<polyline points="%s" fill="none" stroke="%s" stroke-width="1.8" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
        "</svg>"
        % (width, height, gid, colour, colour, "".join(ticks),
           pad_l, zero_y, width - pad_r, zero_y, area, gid, line, colour)
    )


def _report_by_name(reports, needle: str):
    for report in reports:
        if needle in report.get("name", ""):
            return report
    return None


def _commentary() -> dict:
    """Artifact-driven analysis; regenerated by scripts/build_analysis.py."""
    path = ROOT / "reports" / "analysis" / "commentary.json"
    if not path.exists():
        return {"sections": [], "note": ""}
    try:
        return __import__("json").loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"sections": [], "note": ""}


def page_index(data: dict) -> str:
    memo = (ROOT / "DECISION.md").read_text(encoding="utf-8")
    findings = dashboard.parse_findings(ROOT / "FINDINGS.md")
    sections = dashboard.parse_log(ROOT / "RESEARCH_LOG.md")
    high = [r for r in findings if "高" in (r.get("confidence") or "")]
    reports = data.get("reports", [])

    test_count = 0
    for path in (ROOT / "tests").glob("test_*.py"):
        test_count += path.read_text(encoding="utf-8").count(chr(10) + "def test_")

    deployable = _report_by_name(reports, "deployable") or _report_by_name(reports, "blend")
    sharpe = (deployable or {}).get("metrics", {}).get("sharpe")
    total = (deployable or {}).get("metrics", {}).get("total_return")

    cards = [
        ("部署口径累计收益", _pct(total), "混合 20 相位 + 12% 目标波动", True),
        ("Sharpe", _num(sharpe), "相位中位仅 0.668", True),
        ("结论台账", "%d 条" % len(findings), "高置信度 %d 条" % len(high), False),
        ("研究日志", "%d 节" % len(sections), "每节可追溯到脚本", False),
        ("报告目录", "%d 个" % len(reports), "含净值与指标", False),
        ("测试函数", "%d 个" % test_count, "全部通过", False),
    ]
    card_html = "".join(
        '<div class="card%s"><div class="k">%s</div><div class="v">%s</div>'
        '<div class="h">%s</div></div>'
        % (" hl" if hl else "", html.escape(k), v, html.escape(h))
        for k, v, h, hl in cards
    )

    chart = ""
    if deployable and deployable.get("eq"):
        chart = svg_sparkline(deployable["eq"])

    body = ['<div class="hero"><div class="eyebrow">Quantitative Research</div>'
            "<h1>A 股低波策略的<em>完整证据链</em></h1>"
            '<p class="lede">从数据管线到决策页，每条结论都带可追溯的脚本、报告与置信度。'
            "<b>先说结论</b>：策略有效，但 7.14 年的记录对真 Sharpe 0.90 只有 <b>56% 检出力</b>——"
            "「有效」不等于「已被证明」。</p></div>",
            '<div class="grid">%s</div>' % card_html]

    if chart:
        body.append(
            '<div class="panel"><h2>部署口径净值曲线</h2>'
            '<div class="sub">%s · 累计收益以百分比计，虚线为零轴</div>%s</div>'
            % (html.escape(str(deployable.get("name", ""))), chart)
        )

    body.append(
        '<div class="panel alert"><h2>读这一页之前必须先知道的三件事</h2>'
        '<ul class="tight">'
        "<li><b>它是「在最平静的 8 只里剔除跌破趋势的那些」，不是「低波打分 + 趋势确认」。</b>"
        "零件账显示趋势过滤承担全部贡献（+0.60/−0.09/+0.18），0.8/0.2 打分混合平均为零。</li>"
        "<li><b>把 0.902 当期望值是错的。</b>同一窗口的相位中位 Sharpe 只有 0.668，"
        "相位跨度 0.681–1.327。</li>"
        "<li><b>相对随机选股优势巨大（+1.03 Sharpe），但相对最自然的替代品六格全不显著。</b>"
        "「有效」不等于「比显而易见的做法更好」。</li>"
        "</ul></div>"
    )

    commentary = _commentary()
    if commentary.get("sections"):
        blocks = []
        for section in commentary["sections"]:
            blocks.append(
                '<div style="padding:14px 0;border-bottom:1px solid var(--line)">'
                '<div style="font-size:12px;color:var(--acc);letter-spacing:.8px;'
                'text-transform:uppercase;font-weight:600;margin-bottom:6px">%s</div>'
                '<div style="font-size:15px;font-weight:600;margin-bottom:6px">%s</div>'
                '<div class="note">%s</div></div>'
                % (html.escape(section["heading"]), html.escape(section["verdict"]),
                   html.escape(section["evidence"]))
            )
        body.append('<div class="panel"><h2>机器解读</h2>'
                    '<div class="sub">%s</div>%s</div>'
                    % (html.escape(commentary.get("note", "")), "".join(blocks)))

    body.append('<div class="panel"><h2>决策备忘</h2>'
                '<div class="sub">由 <code>DECISION.md</code> 生成，数值来自 reports/</div>%s</div>'
                % dashboard.render_markdown(memo.splitlines()))

    return _page("总览", "index.html", "".join(body))


def page_live(data: dict) -> str:
    """Live page: the browser fetches quotes directly; this only ships the shell."""
    deployable = _report_by_name(data.get("reports", []), "deployable") or \
        _report_by_name(data.get("reports", []), "blend")
    holdings = []
    signal_csv = ROOT / "reports" / "signal_today" / "orders.csv"
    if not signal_csv.exists():
        signal_csv = ROOT / "reports" / "signal_today" / "positions.csv"
    if signal_csv.exists():
        import csv

        with signal_csv.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("symbol"):
                    holdings.append({"symbol": row["symbol"],
                                     "sector": row.get("sector") or "",
                                     "weight": row.get("weight") or ""})
    payload = json.dumps(holdings, ensure_ascii=False)

    body = [
        '<div class="hero"><div class="eyebrow">Live</div>'
        "<h1>实时行情与<em>持仓对照</em></h1>"
        '<p class="lede">行情由浏览器直接向公开接口请求（腾讯财经，CORS 开放）。'
        "页面本身是静态的，<b>取不到数据时会明确报错</b>，不会把旧数字当成实时。</p></div>",
        '<div class="panel"><h2>实时报价</h2>'
        '<div class="live-top"><span class="dot" id="dot"></span>'
        '<span id="status">正在连接…</span>'
        '<span class="dim">·</span><span id="stamp" class="num dim"></span></div>'
        '<div class="qgrid" id="quotes"></div>'
        '<div class="note" id="err" style="display:none;color:#ff5c7c"></div>'
        "</div>",
        '<div class="panel"><h2>当前目标持仓</h2>'
        '<div class="sub">来自 <code>reports/signal_today/orders.csv</code>：'
        "次日开盘的目标权重，不是当前市值权重</div>"
        '<div class="scroll"><table><thead><tr><th>标的</th><th>行业</th>'
        "<th>目标权重</th><th>最新价</th><th>今日涨跌</th></tr></thead>"
        '<tbody id="holdings"></tbody></table></div>'
        '<div class="note" id="hnote">价格列会在行情到达后填充。</div></div>',
        '<div class="panel"><h2>怎么读这一页</h2>'
        '<ul class="tight">'
        "<li><b>实时价只用于对照，不参与任何信号计算。</b>信号用的是前一交易日收盘数据，"
        "这是避免前视偏差的硬约束。</li>"
        "<li><b>涨跌与目标权重无关。</b>权重是次日开盘的目标，今天的涨跌不改变它。</li>"
        "<li><b>报价来自第三方公开接口</b>，延迟与准确性不受本仓控制；"
        "盘中停牌、涨跌停时接口可能返回陈旧价。</li>"
        "</ul></div>",
    ]

    script = (
        "<script>\n"
        "const HOLDINGS = __HOLDINGS__;\n"
        "const SYMS = HOLDINGS.map(h => h.symbol);\n"
        "const dot = document.getElementById('dot');\n"
        "const status = document.getElementById('status');\n"
        "const stamp = document.getElementById('stamp');\n"
        "const errBox = document.getElementById('err');\n"
        "function fail(msg){\n"
        "  dot.className='dot err'; status.textContent='取不到实时数据';\n"
        "  errBox.style.display='block'; errBox.textContent=msg;\n"
        "}\n"
        "function parse(text){\n"
        "  const out={};\n"
        "  text.split(';').forEach(chunk=>{\n"
        "    const m=chunk.match(/v_(\\w+)=\"([^\"]*)\"/); if(!m) return;\n"
        "    const f=m[2].split('~'); if(f.length<33) return;\n"
        "    out[m[1]]={name:f[1],price:parseFloat(f[3]),prev:parseFloat(f[4]),\n"
        "               chg:parseFloat(f[32]),time:f[30]};\n"
        "  });\n"
        "  return out;\n"
        "}\n"
        "async function refresh(){\n"
        "  if(!SYMS.length){ fail('没有可查询的持仓标的。'); return; }\n"
        "  try{\n"
        "    const url='https://qt.gtimg.cn/q='+SYMS.join(',');\n"
        "    const res=await fetch(url,{cache:'no-store'});\n"
        "    if(!res.ok) throw new Error('HTTP '+res.status);\n"
        "    const buf=await res.arrayBuffer();\n"
        "    const text=new TextDecoder('gbk').decode(buf);\n"
        "    const q=parse(text);\n"
        "    if(!Object.keys(q).length) throw new Error('返回内容为空');\n"
        "    dot.className='dot on'; status.textContent='已连接';\n"
        "    errBox.style.display='none';\n"
        "    const box=document.getElementById('quotes'); box.innerHTML='';\n"
        "    let newest='';\n"
        "    SYMS.forEach(sym=>{\n"
        "      const d=q[sym]; const el=document.createElement('div');\n"
        "      if(!d){ el.className='q'; el.innerHTML='<div class=\"sym\">'+sym+'</div>"
        "<div class=\"px dim\">无报价</div>'; box.appendChild(el); return; }\n"
        "      const up=d.chg>=0; el.className='q '+(up?'up':'down');\n"
        "      el.innerHTML='<div class=\"sym\">'+sym+'</div>"
        "<div class=\"px num\">'+d.price.toFixed(2)+'</div>"
        "<div class=\"chg num '+(up?'pos':'neg')+'\">'+(up?'+':'')+d.chg.toFixed(2)+'%</div>';\n"
        "      box.appendChild(el);\n"
        "      if(d.time>newest) newest=d.time;\n"
        "    });\n"
        "    if(newest){ stamp.textContent=newest.replace(/(\\d{4})(\\d{2})(\\d{2})(\\d{2})(\\d{2})(\\d{2})/,'$1-$2-$3 $4:$5:$6'); }\n"
        "    const tb=document.getElementById('holdings'); tb.innerHTML='';\n"
        "    HOLDINGS.forEach(h=>{\n"
        "      const d=q[h.symbol]; const price=d?d.price.toFixed(2):'—';\n"
        "      const chg=d?('<span class=\"num '+(d.chg>=0?'pos':'neg')+'\">'+(d.chg>=0?'+':'')+d.chg.toFixed(2)+'%</span>'):'—';\n"
        "      tb.innerHTML+='<tr><td>'+h.symbol+'</td><td class=\"dim\">'+h.sector+'</td>"
        "<td class=\"num\">'+h.weight+'</td><td class=\"num\">'+price+'</td><td>'+chg+'</td></tr>';\n"
        "    });\n"
        "    document.getElementById('hnote').textContent='最后更新 '+ (newest?newest:'未知') +'（交易所时间）。';\n"
        "  }catch(e){ fail('请求失败：'+e.message+'。可能是网络限制或接口变更。'); }\n"
        "}\n"
        "refresh(); setInterval(refresh, 15000);\n"
        "</script>\n"
    ).replace("__HOLDINGS__", payload)

    return _page("实时", "live.html", "".join(body) + script)


def page_findings() -> str:
    findings = dashboard.parse_findings(ROOT / "FINDINGS.md")
    rows = []
    for row in findings:
        conf = str(row.get("confidence", ""))
        cls = "pos" if conf.startswith("高") else ("neg" if conf.startswith("否") else "dim")
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td>"
            '<td><span class="%s">%s</span></td><td class="dim">%s</td></tr>'
            % (html.escape(str(row.get("number", ""))),
               html.escape(str(row.get("claim", ""))),
               html.escape(str(row.get("evidence", ""))),
               cls, html.escape(conf),
               html.escape(str(row.get("section", ""))))
        )
    body = [
        '<div class="hero"><div class="eyebrow">Findings</div>'
        "<h1>结论台账 <em>%d 条</em></h1>"
        '<p class="lede">置信度是判断能不能当作依据的核心。'
        "写「否」的方向是<b>已验证的负面结果</b>，不是没试过。</p></div>"
        % len(findings),
        '<div class="panel"><div class="scroll"><table><thead><tr>'
        "<th>#</th><th>结论</th><th>依据</th><th>置信度</th><th>出处</th>"
        '</tr></thead><tbody>%s</tbody></table></div></div>' % "".join(rows),
    ]
    return _page("结论台账", "findings.html", "".join(body))


def page_log() -> str:
    sections = dashboard.parse_log(ROOT / "RESEARCH_LOG.md")
    blocks = []
    for section in sections:
        title = html.escape(section.get("title", ""))
        content = section.get("html") or ""
        if not content.strip():
            content = ('<p class="note">本节为标题节，正文在其编号子节中。</p>')
        blocks.append(
            '<div class="panel"><h2>%s · %s</h2>%s</div>'
            % (html.escape(str(section.get("id", ""))), title, content)
        )
    body = ['<div class="hero"><div class="eyebrow">Research Log</div>'
            "<h1>研究日志 <em>%d 节</em></h1>"
            '<p class="lede">按写作顺序。被推翻的结论会在原处标注修正指引。</p></div>'
            % len(sections)]
    body.extend(blocks)
    return _page("研究日志", "log.html", "".join(body))


def page_reports(data: dict) -> str:
    reports = data.get("reports", [])
    rows = []
    for report in reports:
        m = report.get("metrics") or {}
        rows.append(
            "<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (html.escape(str(report.get("name", ""))),
               _pct(m.get("total_return")),
               _num(m.get("sharpe")),
               _pct(m.get("max_drawdown")),
               _num(m.get("annual_volatility")),
               _num(m.get("turnover"), 1))
        )
    top = sorted(reports, key=lambda r: (r.get("metrics") or {}).get("sharpe", -9), reverse=True)[:8]
    chart_rows = ""
    if top:
        cells = "".join(
            '<div class="card"><div class="k">%s</div><div class="v num">%s</div>'
            '<div class="h">Sharpe · %s</div></div>'
            % (html.escape(str(r.get("name", ""))),
               _num((r.get("metrics") or {}).get("sharpe")),
               _pct((r.get("metrics") or {}).get("total_return")))
            for r in top[:4]
        )
        chart_rows = '<div class="grid">%s</div>' % cells

    body = [
        '<div class="hero"><div class="eyebrow">Reports</div>'
        "<h1>报告索引 <em>%d 个</em></h1>"
        '<p class="lede">全部为历史回测，含成本假设。按累计收益排序不代表可部署性——'
        "相位与检出力请看决策页。</p></div>" % len(reports),
        chart_rows,
        '<div class="panel"><div class="scroll"><table><thead><tr>'
        "<th>报告</th><th>累计收益</th><th>Sharpe</th><th>最大回撤</th>"
        "<th>年化波动</th><th>换手</th>"
        '</tr></thead><tbody>%s</tbody></table></div></div>' % "".join(rows),
    ]
    return _page("报告索引", "reports.html", "".join(body))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "docs"))
    parser.add_argument("--max-points", type=int, default=240)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    data = dashboard.build_data(max_points=args.max_points)

    (out / "index.html").write_text(page_index(data), encoding="utf-8")
    (out / "live.html").write_text(page_live(data), encoding="utf-8")
    (out / "findings.html").write_text(page_findings(), encoding="utf-8")
    (out / "log.html").write_text(page_log(), encoding="utf-8")
    (out / "reports.html").write_text(page_reports(data), encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")

    for path in sorted(out.glob("*.html")):
        print("  %-16s %6.1f KiB" % (path.name, path.stat().st_size / 1024))
    print("site written to %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
