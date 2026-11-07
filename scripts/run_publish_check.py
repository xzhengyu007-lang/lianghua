"""Self-check for the configuration we would actually deploy.

RESEARCH_LOG 23.85/23.86 fixed the deployable book: the 20-phase blend at a 12%
volatility target. Everything the one-command self-check (23.68) verifies --
data sanity, determinism, causality -- it verifies on the *old* default, so
"the pipeline has been audited" and "what we are about to run has been audited"
were statements about two different objects.

This closes that gap. It is deliberately a separate, slower script rather than an
extra step in the everyday check: a single blend costs about twenty single-phase
builds, and the five-second check has to stay five seconds.

Three things are verified:

* reproducibility -- two independent blends must be identical, not just two
  backtests of the same weights;
* causality -- rebuild the blend on a panel truncated at the cut, and the equity
  before the cut must not move;
* fingerprint -- the published metrics must still equal the recorded canon, so an
  accidental change to the configuration cannot slip through unnoticed.

Usage::

    python scripts/run_publish_check.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.strategies import build_strategy  # noqa: E402

PHASES = tuple(range(20))
TARGET_VOL = 0.12

# Recorded from the engine's own metrics for the deployable configuration.
# These are NOT byte-identical to the numbers published in 23.86: that year table
# is computed from daily returns with ddof=0, while the engine uses ddof=1, so
# Sharpe differs in the fourth decimal by sqrt(1799/1800). Both are correct for
# their own purpose; this fingerprint guards the engine path, which is what a
# deployment comparison would use. Update it (and the section) only when the
# configuration changes on purpose -- that is what a fingerprint is for.
CANONICAL = {
    "total_return": 0.9587019322603088,
    "sharpe": 0.901821607911435,
    "max_drawdown": -0.15702821433502623,
    "annual_volatility": 0.11124949780115752,
}
TOLERANCE = 1e-9


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "run_lookahead_audit", ROOT / "scripts" / "run_lookahead_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def blend_weights(aligned, target_vol: float = TARGET_VOL) -> pd.DataFrame:
    frames = []
    for offset in PHASES:
        strategy = build_strategy(
            "low_vol_core",
            top_n=8,
            vol_window=60,
            trend_window=120,
            max_per_sector=2,
            rebalance_days=20,
            rebalance_reset="year",
            rebalance_offset=offset,
        )
        frames.append(
            build_portfolio(
                strategy.generate(aligned),
                aligned,
                risk=RiskConfig(target_annual_vol=target_vol),
                use_regime=False,
                use_vol_target=True,
            )
        )
    return sum(frames) / len(frames)


def check_weights(panel, window_note: str = "") -> dict:
    print("  building blend twice...", flush=True)
    first = blend_weights(panel)
    second = blend_weights(panel)
    diff = float((first.fillna(0.0) - second.fillna(0.0)).abs().to_numpy().max())
    return {
        "name": "blend_reproducibility",
        "ok": diff < TOLERANCE,
        "comparison": "two independent 20-phase blends",
        "max_abs_diff": diff,
        "note": window_note,
    }


def check_causality(panel, aligned, weights, cut, audit) -> dict:
    print("  rebuilding blend on the truncated panel...", flush=True)
    full_equity = Backtester(BacktestConfig()).run(aligned, weights).equity
    partial_panel = audit.truncate(panel, cut)
    partial_aligned, _dates = align_panel(partial_panel)
    partial_weights = blend_weights(partial_aligned)
    partial_equity = Backtester(BacktestConfig()).run(partial_aligned, partial_weights).equity
    result = audit.compare(full_equity, partial_equity, cut)
    return {
        "name": "blend_causality",
        "ok": bool(result["ok"]),
        "cut": str(cut.date()),
        "n_days": int(result["n"]),
        "max_abs_diff": float(result.get("max_abs_diff", float("nan"))),
        "max_rel_diff": float(result.get("max_rel_diff", float("nan"))),
    }


def check_fingerprint(metrics) -> dict:
    actual = {
        "total_return": float(metrics.total_return),
        "sharpe": float(metrics.sharpe),
        "max_drawdown": float(metrics.max_drawdown),
        "annual_volatility": float(metrics.annual_volatility),
    }
    drift = {k: float(actual[k] - CANONICAL[k]) for k in CANONICAL}
    ok = all(abs(v) < TOLERANCE for v in drift.values())
    return {
        "name": "fingerprint",
        "ok": ok,
        "actual": actual,
        "canonical": CANONICAL,
        "drift": drift,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "publish_check"))
    parser.add_argument("--cut", default="2023-06-30")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    print("loading PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), cache_dir, "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, _dates = align_panel(panel)

    audit = _load_audit()
    cut = pd.Timestamp(args.cut)

    print("checking blend reproducibility...", flush=True)
    checks = [check_weights(aligned)]

    print("building reference blend...", flush=True)
    weights = blend_weights(aligned)
    result = Backtester(BacktestConfig()).run(aligned, weights)

    print("checking fingerprint...", flush=True)
    checks.append(check_fingerprint(result.metrics))

    print("checking causality at %s..." % cut.date(), flush=True)
    checks.append(check_causality(panel, aligned, weights, cut, audit))

    all_ok = all(c["ok"] for c in checks)
    summary = {
        "config": {"phases": len(PHASES), "target_vol": TARGET_VOL},
        "cut": str(cut.date()),
        "all_ok": bool(all_ok),
        "checks": checks,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 发布配置自检（混合 + 12% 目标波动）", ""]
    lines.append("| 检查 | 结果 | 细节 |")
    lines.append("|---|---|---|")
    for check in checks:
        if check["name"] == "blend_reproducibility":
            detail = "两次独立 20 相位构建，最大差 %.3g" % check["max_abs_diff"]
        elif check["name"] == "fingerprint":
            detail = "收益/Sharpe/回撤/波动 与 §23.86 记录值的最大漂移 %.3g" % max(
                abs(v) for v in check["drift"].values()
            )
        else:
            detail = "切点 %s 之前 %d 天，最大相对差 %.3g" % (
                check["cut"], check["n_days"], check["max_rel_diff"]
            )
        lines.append("| %s | %s | %s |" % (check["name"], "通过" if check["ok"] else "**失败**", detail))
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    for check in checks:
        print("  %-24s %s" % (check["name"], "ok" if check["ok"] else "FAILED"), flush=True)
    print("all_ok: %s" % all_ok)
    print("report: %s" % out)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
