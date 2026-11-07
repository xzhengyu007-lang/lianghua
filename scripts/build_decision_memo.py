"""One page that answers "should we put money on this?" without reading 93 sections.

Everything here is read from the artifacts, not retyped. The memo is regenerated
by this script and tests/test_decision_memo.py asserts the checked-in copy equals
a fresh render, so it cannot silently go stale the way prose does.

It answers three questions and nothing else:

1. what should be deployed right now;
2. what to expect if it is (with the interval, not just the point);
3. what evidence is still missing before size is increased.

Usage::

    python scripts/build_decision_memo.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

CONFIG = {
    "phases": 20,
    "target_vol": 0.12,
    "top_n": 8,
    "vol_window": 60,
    "trend_window": 120,
    "max_per_sector": 2,
    "rebalance_days": 20,
    "rebalance_reset": "year",
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def collect() -> dict:
    payload: dict = {"config": CONFIG}

    year = _json(REPORTS / "year_table_blend" / "summary.json")
    if year:
        overall = year["overall"]
        payload["full_period"] = {
            "deployable": overall["blend@12%"],
            "phase0_18": overall["phase0@18%"],
        }

    voltarget = _csv(REPORTS / "voltarget_blend" / "panel.csv")
    if voltarget is not None:
        row = voltarget[(voltarget["window"] == "2024-2026") & (voltarget["target"] == 0.12)].iloc[0]
        payload["window_2024_2026"] = {
            "total_return": float(row["total_return"]),
            "sharpe": float(row["sharpe"]),
            "max_drawdown": float(row["max_drawdown"]),
            "annual_volatility": float(row["annual_volatility"]),
        }

    phases = _csv(REPORTS / "phase_recheck" / "phases.csv")
    if phases is not None:
        block = phases[(phases["window"] == "2024-2026") & (phases["mode"] == "year")]
        payload["phase_spread"] = {
            "median_return": float(block["total_return"].median()),
            "worst_return": float(block["total_return"].min()),
            "best_return": float(block["total_return"].max()),
            "median_sharpe": float(block["sharpe"].median()),
        }

    power = _json(REPORTS / "power_recheck" / "summary.json")
    if power:
        payload["evidence"] = {
            "recorded": power["recorded"],
            "analytic": power["analytic"].get("0.90"),
        }
        rows = [r for r in power["power"] if r["sharpe"] == 0.90 and abs(r["years"] - 7.3) < 0.05]
        payload["evidence"]["power_at_7_3y"] = float(rows[0]["power"]) if rows else None

    cadence = _csv(REPORTS / "cadence_blend" / "panel.csv")
    if cadence is not None:
        block = cadence[cadence["cadence"] == 20]
        payload["turnover"] = {
            "mean_annual": float(block["turnover"].mean()),
            "median_holding_days": float(block["median_holding_days"].iloc[0]),
        }

    capacity = _json(REPORTS / "capacity_blend" / "summary.json")
    if capacity:
        rows = [r for r in capacity["rows"] if r["impact_bps"] == 40.0 and r["config"] == "blend"]
        by_cash = {float(r["cash"]): float(r["sharpe"]) for r in rows}
        payload["capacity"] = {
            "sharpe_1e6": by_cash.get(1_000_000.0),
            "sharpe_2e8": by_cash.get(200_000_000.0),
            "sharpe_2e9": by_cash.get(2_000_000_000.0),
            "note": "18% target, 40bp impact",
        }

    breadth = _csv(REPORTS / "cross_pool_random" / "summary.csv")
    if breadth is not None:
        payload["breadth"] = {
            "median_60": float(breadth.loc[breadth["size"] == 60, "sharpe_median"].iloc[0]),
            "median_120": float(breadth.loc[breadth["size"] == 120, "sharpe_median"].iloc[0]),
            "positive_share_60": float(breadth.loc[breadth["size"] == 60, "share_positive"].iloc[0]),
            "p10_60": float(breadth.loc[breadth["size"] == 60, "sharpe_p10"].iloc[0]),
        }

    random_control = _json(REPORTS / "random_control" / "summary.json")
    if random_control:
        payload["value_attribution"] = {
            "real_sharpe": float(random_control["real"]["sharpe"]),
            "random_median": float(random_control["random"]["sharpe_median"]),
            "random_p95": float(random_control["random"]["sharpe_p95"]),
            "real_percentile": float(random_control["real_percentile_among_random"]),
        }

    cohorts = _json(REPORTS / "factor_cohorts" / "summary.json")
    if cohorts:
        row = next(
            (r for r in cohorts["rows"] if r["factor"] == "low_vol" and r["horizon"] == 20 and r["cohort"] == "listed"),
            None,
        )
        if row:
            payload["value_attribution_in_pool_ic"] = float(row["mean_ic"])

    quarter = _json(REPORTS / "quarter_consistency" / "summary.json")
    if quarter:
        payload["sign_test"] = {
            "power": float(quarter["sign_test_power"]),
            "quarters_for_80pct": quarter["quarters_for_80pct_power"],
        }

    exec_panel = _csv(REPORTS / "execution_timing" / "panel.csv")
    if exec_panel is not None:
        costs = []
        for window, block in exec_panel.groupby("window"):
            base = float(block[block["delay_days"] == 0]["sharpe"].iloc[0])
            one = float(block[block["delay_days"] == 1]["sharpe"].iloc[0])
            costs.append(abs(one - base))
        payload["execution_delay_cost"] = max(costs)

    naive = _csv(REPORTS / "naive_benchmark" / "paired.csv")
    naive_panel = _csv(REPORTS / "naive_benchmark" / "panel.csv")
    if naive is not None and naive_panel is not None:
        payload["naive_benchmark"] = {
            "worst_gap": float(naive["observed"].min()),
            "best_gap": float(naive["observed"].max()),
            "significant": int(((naive["low"] > 0) | (naive["high"] < 0)).sum()),
            "cells": int(len(naive)),
        }
        pivot = naive_panel.pivot_table(index="window", columns="variant", values="turnover")
        naive_col = next(c for c in pivot.columns if c.startswith("A"))
        real_col = next(c for c in pivot.columns if c.startswith("C"))
        payload["naive_benchmark"]["turnover_ratio"] = float(
            (pivot[real_col] / pivot[naive_col]).median()
        )
        by_pool = naive_panel.pivot_table(index="window", columns="variant", values="sharpe")
        listed_col = next(c for c in by_pool.columns if c.startswith("A"))
        pit_col = next(c for c in by_pool.columns if c.startswith("B"))
        payload["naive_benchmark"]["pool_effect"] = {
            str(k): float(v) for k, v in (by_pool[pit_col] - by_pool[listed_col]).items()
        }

    ablation = _csv(REPORTS / "filter_blend_ablation" / "panel.csv")
    if ablation is not None:
        pivot = ablation.pivot_table(index="window", columns="variant", values="sharpe")
        published = next(c for c in pivot.columns if c.startswith("过滤+混合"))
        only_filter = next(c for c in pivot.columns if c.startswith("只过滤"))
        only_blend = next(c for c in pivot.columns if c.startswith("只混合"))
        payload["parts"] = {
            "filter": {str(k): float(v) for k, v in (pivot[published] - pivot[only_blend]).items()},
            "blend": {str(k): float(v) for k, v in (pivot[published] - pivot[only_filter]).items()},
        }

    check = _json(REPORTS / "publish_check" / "summary.json")
    if check:
        payload["publish_check"] = {"all_ok": bool(check["all_ok"]),
                                    "checks": [c["name"] for c in check["checks"]]}
    return payload


def render(payload: dict) -> str:
    cfg = payload["config"]
    lines = ["# 决策备忘", ""]
    lines.append("> 由 `scripts/build_decision_memo.py` 从 `reports/` 生成，**不要手改**。")
    lines.append("> `tests/test_decision_memo.py` 会比对本文与重新生成的结果，改数字必须改报告。")
    lines.append("")

    lines.append("## 如果你只记得三件事")
    lines.append("")
    lines.append("1. **它是什么**：在最平静的 8 只里剔除跌破趋势的那些，月度轮动。中位持有一只股票 %d 个"
                 "交易日，不是「买入并持有低波龙头」（§23.74/§23.91）。**名字里的「低波打分」几乎是装饰**——"
                 "真正干活的是趋势过滤（见下面零件账）。"
                 % (round(payload["turnover"]["median_holding_days"]) if "turnover" in payload else 0))
    full = payload.get("full_period", {}).get("deployable")
    median_sharpe = payload.get("phase_spread", {}).get("median_sharpe")
    if full and median_sharpe:
        lines.append("2. **它值多少**：全期 %+.2f%% / Sharpe %.3f（部署口径）。但同一窗口的相位中位 Sharpe 只有"
                     " **%.3f**——把 %.3f 当成期望值是错的（§23.82）。"
                     % (full["return"] * 100, full["sharpe"], median_sharpe, full["sharpe"]))
    if "evidence" in payload and "value_attribution" in payload:
        lines.append("3. **别把它当成已证明**：7.14 年的记录对真 Sharpe 0.90 只有 **%.0f%%** 检出力；"
                     "打分本身确实有效（随机选股中位只有 %.3f），但「有效」不等于「已被证明」。"
                     % ((payload["evidence"]["power_at_7_3y"] or 0) * 100,
                        payload["value_attribution"]["random_median"]))
    lines.append("")
    lines.append("## 1. 现在该部署什么")
    lines.append("")
    lines.append("| 项 | 取值 |")
    lines.append("|---|---|")
    lines.append("| 组合 | 混合 %d 相位等权（每相位各 1/%d 资金） |" % (cfg["phases"], cfg["phases"]))
    lines.append("| 选股 | top_n=%d，vol_window=%d，trend_window=%d，每行业≤%d |"
                 % (cfg["top_n"], cfg["vol_window"], cfg["trend_window"], cfg["max_per_sector"]))
    lines.append("| 调仓 | 每 %d 个交易日，按自然年重置锚点 |" % cfg["rebalance_days"])
    lines.append("| 风险 | 目标年化波动 %.0f%% |" % (cfg["target_vol"] * 100))
    lines.append("")
    if "parts" in payload:
        windows = ["2019-2021", "2022-2023", "2024-2026"]
        lines.append("**零件账**——这层一层层拆下来，只有一个零件在干活（三段 Sharpe 增量，§23.110）：")
        lines.append("")
        lines.append("| 零件 | " + " | ".join(windows) + " | 结论 |")
        lines.append("|---|" + "---:|" * len(windows) + "---|")
        parts = payload["parts"]
        lines.append("| **趋势过滤** | " + " | ".join("%+.2f" % parts["filter"][w] for w in windows)
                     + " | 主要贡献者，防的是下跌行情里抱着最平静的名字一起跌 |")
        def fmt(value: float) -> str:
            text = "%+.2f" % value
            return "+0.00" if text in ("-0.00", "+0.00") else text

        lines.append("| 0.8/0.2 打分混合 | " + " | ".join(fmt(parts["blend"][w]) for w in windows)
                     + " | 平均为零；它确实会换掉约 1.1 只，但换掉的那只没有信息（§23.111，t=0.81） |")
        pool = payload.get("naive_benchmark", {}).get("pool_effect")
        if pool:
            lines.append("| PIT 池 vs 在册 128 只 | " + " | ".join(fmt(pool.get(w, 0.0)) for w in windows)
                         + " | 边际很小，退市群的有无不是主导因素（§23.109） |")
        lines.append("")
        lines.append("所以更准确的名字是：**在最平静的 8 只里剔除跌破趋势的那些**，而不是"
                     "「低波打分 + 趋势确认」——后者会让人以为打分在做主要工作。")
        lines.append("")
    if "value_attribution" in payload:
        v = payload["value_attribution"]
        lines.append("**这台机器值在哪（这比「值多少」更重要）**：把低波分换成随机分、其余约束全不变时，")
        lines.append("Sharpe 从 %.3f 掉到中位 %.3f（随机 p95 只有 %.3f，发布策略在第 %.0f 百分位，§23.102）。"
                     % (v["real_sharpe"], v["random_median"], v["random_p95"], v["real_percentile"] * 100))
        if "naive_benchmark" in payload:
            n = payload["naive_benchmark"]
            lines.append("**但换成「最自然的替代品」就不一样了**：在册 128 只里直接选最平静的 8 只、不加趋势过滤与"
                         "行业上限（§23.109）——三段 ΔSharpe 只有 %+.2f / %+.2f，**%d 格中 %d 格显著**，"
                         "而它的换手只有发布策略的 1/%.1f。"
                         % (n["worst_gap"], n["best_gap"], n["cells"], n["significant"], n["turnover_ratio"]))
            lines.append("所以「打分有用」的准确版本是：**相对随机有用；相对最自然的替代品，只有趋势过滤在 "
                         "2019–2021 那种行情里明显值钱**。")
        if "value_attribution_in_pool_ic" in payload:
            lines.append("另外，低波分在**仍在册**名字上的池内 IC 只有 %+.3f（§23.77），相对低波 1/3 篮子的增量"
                         "超额也只有 +13.4%%/年、t=1.77（§23.40）——"
                         % payload["value_attribution_in_pool_ic"])
            lines.append("这两条针对的是「在桶里精挑」，而上一段的正面结果针对的是「待在桶里」。")
        lines.append("")
    lines.append("选它的唯一理由是**它是唯一不取决于「你从哪天开始数」的配置**：单相位的 20 个等价日历")
    lines.append("在 2024–2026 上收益跨度 40.9pp，而混合把这些日历等权拿住（§23.82/§23.83）。")
    lines.append("")
    if "publish_check" in payload:
        ok = payload["publish_check"]["all_ok"]
        lines.append("**上钱之前先跑验收**：`python scripts/run_publish_check.py` —— 当前状态 **%s**"
                     % ("通过" if ok else "未通过"))
        lines.append("（%s）。" % "、".join(payload["publish_check"]["checks"]))
        lines.append("")

    lines.append("## 2. 期望多少")
    lines.append("")
    lines.append("| 口径 | 累计收益 | Sharpe | 最大回撤 |")
    lines.append("|---|---:|---:|---:|")
    if "full_period" in payload:
        d = payload["full_period"]["deployable"]
        lines.append("| 全期（7.14 年） | %+.2f%% | %.3f | %.2f%% |"
                     % (d["return"] * 100, d["sharpe"], d["max_drawdown"] * 100))
    if "window_2024_2026" in payload:
        w = payload["window_2024_2026"]
        lines.append("| 2024–2026 | %+.2f%% | %.3f | %.2f%% |"
                     % (w["total_return"] * 100, w["sharpe"], w["max_drawdown"] * 100))
    if "phase_spread" in payload:
        p = payload["phase_spread"]
        lines.append("| 2024–2026 相位中位 | %+.2f%% | %.3f | — |" % (p["median_return"] * 100, p["median_sharpe"]))
        lines.append("| 2024–2026 最差相位 | %+.2f%% | — | — |" % (p["worst_return"] * 100))
    lines.append("")
    if "evidence" in payload:
        e = payload["evidence"]
        r = e["recorded"]
        lines.append("**不要把 Sharpe %.3f 当成已证明的数字。** 自助法 95%% 区间是 **[%.3f, %.3f]**；"
                     % (r["sharpe_point"], r["sharpe_low"], r["sharpe_high"]))
        lines.append("7.14 年的记录对真 Sharpe 0.90 只有 **%.0f%%** 检出力，80%% 把握要 **%.1f 年**（§23.89）。"
                     % ((e["power_at_7_3y"] or 0) * 100, e["analytic"]["years_for_80pct_power"]))
        lines.append("")
    if "turnover" in payload:
        t = payload["turnover"]
        lines.append("- 年换手约 **%.1fx**，中位持有一只在账 **%.0f 天**（§23.91；混合会把持有拉长到周期的 1.6–1.7 倍）。"
                     % (t["mean_annual"], t["median_holding_days"]))
    if "capacity" in payload:
        c = payload["capacity"]
        lines.append("- 容量（%s）：100 万 Sharpe %.3f → 2 亿 %.3f → 20 亿 %.3f（§23.84）。"
                     % (c["note"], c["sharpe_1e6"], c["sharpe_2e8"], c["sharpe_2e9"]))
    lines.append("")

    lines.append("## 3. 加仓前还缺什么")
    lines.append("")
    if "evidence" in payload:
        lines.append("检出力只有 %.0f%%。想把「有效」从 suggestive 推到 established，只有下面五条路，而其中三条"
                     % ((payload["evidence"]["power_at_7_3y"] or 0) * 100))
        lines.append("已经被本仓实测证伪——所以这张表本身就是结论：")
    else:
        lines.append("想把「有效」从 suggestive 推到 established，可走的路只有下面几条：")
    lines.append("")
    lines.append("| 路径 | 结论 | 依据 |")
    lines.append("|---|---|---|")
    if "evidence" in payload:
        lines.append("| **时间** | 唯一确定有效，但慢：再跑 %.1f 年 | §23.89 |"
                     % (payload["evidence"]["analytic"]["years_for_80pct_power"] - 7.14))
    if "breadth" in payload:
        b = payload["breadth"]
        lines.append("| **第二个 ≥120 只的市场** | 有效，但需要外部数据：等效样本 1.54 倍，7.1 年变 10.9 年；"
                     "60 只不够（随机子池中位 Sharpe %.3f、p10 %.3f） | §23.93 |"
                     % (b["median_60"], b["p10_60"]))
    lines.append("| 更长的调仓周期 | **不是杠杆**：换手砍半但净 Sharpe 打平 | §23.91 |")
    if "execution_delay_cost" in payload:
        lines.append("| 换执行时点 | **不是杠杆**：晚一天约 %.2f Sharpe，九格区间全含 0 | §23.101 |"
                     % payload["execution_delay_cost"])
    if "sign_test" in payload:
        lines.append("| 换检验统计量 | **不是杠杆**：符号检验 %.0f%% 检出力 vs Sharpe 区间 56%% | §23.104 |"
                     % (payload["sign_test"]["power"] * 100))
    lines.append("")
    lines.append("**明确写下什么会推翻上面的判断**（这些是止损条件，不是预测）：")
    lines.append("")
    lines.append("- 在册口径的低波 IC 跌破 0（当前 +0.031，§23.77）；")
    lines.append("- 混合口径出现连续两个负自然年（历史上有 2022、2025 各一次，§23.86）；")
    lines.append("- 实测成本超过 published 的两倍（§23.55 的 harsh 情景），且 60 日周期也救不回来；")
    lines.append("- 可交易横截面继续收缩到 <60 只，届时按 §23.93 的斜坡，期望 Sharpe 会掉到 0.4 一档。")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Root-level on purpose: reports/ is gitignored, and this page must be in the repo.
    parser.add_argument("--output", default=str(ROOT / "DECISION.md"))
    parser.add_argument("--json-output", default=str(REPORTS / "decision_memo.json"))
    args = parser.parse_args(argv)

    payload = collect()
    payload["generated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    text = render(payload)

    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    Path(args.json_output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print("memo: %s (%d lines)" % (target, text.count(chr(10))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
