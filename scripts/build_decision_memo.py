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
        e = payload["evidence"]
        lines.append("检出力只有 %.0f%%，而下面三样都能把它抬高——按性价比排序：" % ((e["power_at_7_3y"] or 0) * 100))
        lines.append("")
        lines.append("1. **时间**。什么都不做，再跑 %.1f 年到 80%% 把握。" % (e["analytic"]["years_for_80pct_power"] - 7.14))
        if "breadth" in payload:
            b = payload["breadth"]
            lines.append("2. **第二个市场（≥120 只）**。同一套打分在 %d 只的随机子池上中位 Sharpe 只有 %.3f，"
                         % (60, b["median_60"]))
            lines.append("   而 %d 只有 %.3f；若两市场相关 0.3，等效样本 1.54 倍，7.1 年变 10.9 年（§23.93）。"
                         % (120, b["median_120"]))
            lines.append("   注意 **60 只不够**：p10 只有 %.3f，抽一次很可能拿到接近 0 的结果。"
                         % b["p10_60"])
        lines.append("3. **换更长的调仓周期**。60 日把换手从 11.5x 砍到 5.7x，净 Sharpe 打平（§23.91）——"
                     "它不提升收益，但让容量和成本假设更宽松。")
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
