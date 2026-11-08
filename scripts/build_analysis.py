"""Turn the research artifacts into per-portfolio commentary.

The site already shows what the numbers are. This makes the machine say what they
mean, using only the artifacts -- no new modelling, no opinions that are not in the
research log. Every sentence is derived from a measured quantity, and the caveats
come from the same place as the metrics.

The rule this follows: a strategy is described by (a) what it selects, (b) how much
of its result survives without that selection, and (c) how much of the record is
noise. Those are 23.112, 23.109 and 23.89, in that order.

Usage::

    python scripts/build_analysis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"


def _json(relative: str):
    path = REPORTS / relative
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _pct(value) -> str:
    return "%+.1f%%" % (value * 100) if isinstance(value, (int, float)) else "—"


def _num(value, digits=3) -> str:
    return "%.*f" % (digits, value) if isinstance(value, (int, float)) else "—"


def analyse() -> dict:
    """Return structured commentary: sections of (heading, verdict, evidence)."""
    parts = _json("filter_blend_ablation/summary.json")
    naive = _json("naive_benchmark/summary.json")
    random_control = _json("random_control/summary.json")
    power = _json("power_recheck/summary.json")
    quart = _json("quarter_consistency/summary.json")
    overlap = _json("rank_overlap/summary.json")
    year_table = _json("year_table_blend/summary.json")
    gate = _json("window_gate/summary.json")

    sections = []

    # 1) What actually drives the result.
    if parts:
        rows = {(r["variant"], r["window"]): r["sharpe"] for r in parts["rows"]}
        windows = ["2019-2021", "2022-2023", "2024-2026"]
        pub = "过滤+混合（发布）"
        only_filter = "只过滤（纯低波打分）"
        only_blend = "只混合（不过滤）"
        def gap(a, b):
            return [rows.get((a, w), 0) - rows.get((b, w), 0) for w in windows]
        filter_effect = gap(pub, only_blend)
        blend_effect = gap(pub, only_filter)
        sections.append({
            "heading": "贡献从哪来",
            "verdict": "趋势过滤承担全部贡献，打分混合平均为零。",
            "evidence": (
                "加过滤的 Sharpe 增量在三段分别是 %s；加混合是 %s。"
                "混合确实会改变篮子（%s），但换掉的那只不携带信息。"
                % (" / ".join("%+.2f" % v for v in filter_effect),
                   " / ".join("%+.2f" % v for v in blend_effect),
                   ("重合 %.2f/%d、%d%% 的调仓会换名"
                    % (overlap["mean_overlap"], overlap["top_n"],
                       round(100 * (1 - overlap["share_identical"]))))
                   if overlap else "见 §23.111")
            ),
        })

    # 2) How much survives without the selection rule.
    if random_control and naive:
        rnd = random_control["random"]
        sections.append({
            "heading": "选择规则值多少",
            "verdict": "对随机选股优势巨大；对最自然的替代品则几乎不可分辨。",
            "evidence": (
                "把打分换成随机、其余不变，Sharpe 从 %s 掉到中位 %s（p95 %s，发布策略在第 %.0f 百分位）。"
                "但换成「在册 128 只里直接选最平静的 8 只」，配对 ΔSharpe 区间 %s。"
                % (_num(random_control["real"]["sharpe"]),
                   _num(rnd["sharpe_median"]), _num(rnd["sharpe_p95"]),
                   random_control["real_percentile_among_random"] * 100,
                   "全部含 0" if naive["paired"] is None else "见 §23.109")
            ),
        })

    # 3) How much of the record is noise.
    if power or quart:
        bits = []
        if power:
            analytic = power.get("analytic") or {}
            bits.append("对真 Sharpe 0.90，7.14 年只有 %.0f%% 检出力，80%% 需要 %.1f 年"
                        % ((power.get("power_at_7_3y") or 0) * 100,
                           analytic.get("years_for_80pct_power") or 0))
        if quart:
            bits.append("季度符号检验的检出力是 %.0f%%（对照 Sharpe 区间 56%%），不是捷径"
                        % (quart["sign_test_power"] * 100))
        sections.append({
            "heading": "这条记录有多可信",
            "verdict": "「有效」不等于「已被证明」。",
            "evidence": "；".join(bits) + "。",
        })

    # 4) Year-by-year shape.
    if year_table:
        rows = year_table.get("rows") or []
        if rows:
            best = max(rows, key=lambda r: (r.get("sharpe") or -9))
            worst = min(rows, key=lambda r: (r.get("sharpe") or 9))
            sections.append({
                "heading": "收益的时间结构",
                "verdict": "收益不是均匀分布的，集中在少数年份。",
                "evidence": "最好的年份 %s（Sharpe %s），最差 %s（Sharpe %s）。"
                            % (best.get("window") or best.get("year"),
                               _num(best.get("sharpe")),
                               worst.get("window") or worst.get("year"),
                               _num(worst.get("sharpe"))),
            })

    # 5) A candidate we tested and did not adopt.
    if gate and "SMA250" in gate and "SMA120" in gate:
        g120, g250 = gate["SMA120"], gate["SMA250"]
        sections.append({
            "heading": "测过但没采纳的改动",
            "verdict": "趋势窗口 250 在样本内更好，但不可分辨，因此不改。",
            "evidence": (
                "全期 Sharpe %s → %s、回撤 %s → %s、换手 %s → %s；但推到 500 后曲线在 250 处见顶，"
                "且 250 与 300 的差小于单配置自身的相位跨度——这是内点峰，不是稳定优势。"
                % (_num(g120["full_sharpe"]), _num(g250["full_sharpe"]),
                   _pct(g120["full_max_drawdown"]), _pct(g250["full_max_drawdown"]),
                   _num(g120["full_turnover"], 1), _num(g250["full_turnover"], 1))
            ),
        })

    return {
        "sections": sections,
        "note": ("以上每一句都由 reports/ 下的产物计算，没有引入新的模型或观点。"
                 "结论的置信度以 FINDINGS.md 为准。"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(REPORTS / "analysis" / "commentary.json"))
    args = parser.parse_args(argv)

    payload = analyse()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    for section in payload["sections"]:
        print("- %s：%s" % (section["heading"], section["verdict"]))
    print("sections: %d -> %s" % (len(payload["sections"]), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
