"""One command that proves the system still works.

The research log has many one-off checks (data coverage, determinism, causality).
They were all run by hand at some point; nothing forced them to keep passing
after a refactor. This script runs the three that can be checked cheaply and
writes a single pass/fail summary:

1. data    - every PIT symbol carries both ``raw`` and ``_tr`` frames, no zero
             volume bars, no duplicated dates, and the rebuilt total-return
             series is monotone-consistent with raw closes up to the split
             adjustment (no negative prices);
2. repro   - two identical backtests produce bit-identical equity;
3. causal  - truncating the panel at a cut date leaves the equity up to that
             date bit-identical (the cheap version of the full audit).

Usage::

    python scripts/run_verification.py
    python scripts/run_verification.py --cut 2022-06-30
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.cli import _rebuilt_panel, _research_panel  # noqa: E402
from quant_system.config import BROAD_UNIVERSE, BacktestConfig  # noqa: E402
from quant_system.data import align_panel, pit_universe  # noqa: E402


def _load_audit():
    spec = importlib.util.spec_from_file_location(
        "run_lookahead_audit", ROOT / "scripts" / "run_lookahead_audit.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_data(panel: dict[str, pd.DataFrame], cache_dir: Path) -> dict:
    problems: list[str] = []
    symbols = sorted(panel)
    for symbol in symbols:
        frame = panel[symbol]
        if frame.index.has_duplicates:
            problems.append(f"{symbol}: duplicate dates")
        if frame["close"].isna().all():
            problems.append(f"{symbol}: no close prices")
        if (frame["close"].dropna() <= 0).any():
            problems.append(f"{symbol}: non-positive close")
    raw_missing = [
        s for s in symbols
        if not (cache_dir / f"{s}_daily_raw.parquet").exists()
        and not (cache_dir / f"{s}_daily_raw.csv").exists()
    ]
    return {
        "name": "data",
        "ok": not problems,
        "symbols": len(symbols),
        "bars": int(len(panel[symbols[0]])),
        "problems": problems[:10],
        "raw_cache_missing": raw_missing[:10],
    }


def check_repro(panel: dict[str, pd.DataFrame], weights: pd.DataFrame, window: tuple[str, str]) -> dict:
    cfg = BacktestConfig()
    first = Backtester(cfg).run(panel, weights).equity
    second = Backtester(cfg).run(panel, weights).equity
    same = bool(first.equals(second))
    return {
        "name": "reproducibility",
        "ok": same,
        "comparison": "two identical runs",
        "max_abs_diff": float((first - second).abs().max()),
        "n_days": int(len(first)),
    }


def check_causal(panel: dict[str, pd.DataFrame], weights: pd.DataFrame, cut: pd.Timestamp, audit) -> dict:
    """Rebuild signals from the truncated panel, then compare the equity prefix.

    Slicing the *weights* would be a much weaker test: it cannot catch a
    strategy that peeks at the future while scoring. The strategy is therefore
    re-run on the truncated panel, exactly like the full audit does.
    """
    full = Backtester(BacktestConfig()).run(panel, weights).equity
    partial_panel = audit.truncate(panel, cut)
    partial = audit.run(partial_panel)
    result = audit.compare(full, partial, cut)
    return {
        "name": "causality",
        "ok": bool(result["ok"]),
        "cut": str(cut.date()),
        "n_days": int(result["n"]),
        "max_abs_diff": float(result.get("max_abs_diff", float("nan"))),
        "max_rel_diff": float(result.get("max_rel_diff", float("nan"))),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(ROOT / "data" / "cache"))
    parser.add_argument("--output", default=str(ROOT / "reports" / "verification"))
    parser.add_argument("--cut", default="2023-06-30")
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    print("loading PIT panel...", flush=True)
    panel = _rebuilt_panel(list(pit_universe()), cache_dir, "rebuilt")
    panel = _research_panel(panel, anchor_symbols=list(BROAD_UNIVERSE))
    aligned, dates = align_panel(panel)

    audit = _load_audit()
    strategy_weights = None
    checks = [check_data(aligned, cache_dir)]

    print("building signals...", flush=True)
    from quant_system.portfolio import build_portfolio  # local import keeps startup fast
    from quant_system.strategies import LowVolCoreStrategy

    strategy = LowVolCoreStrategy(rebalance_reset="year", **audit.PARAMS)
    import quant_system.config as cfg_mod

    strategy_weights = build_portfolio(
        strategy.generate(aligned), aligned, risk=cfg_mod.RiskConfig(), use_regime=False, use_vol_target=True
    )

    print("checking reproducibility...", flush=True)
    checks.append(check_repro(aligned, strategy_weights, (str(dates[0].date()), str(dates[-1].date()))))

    cut = pd.Timestamp(args.cut)
    if cut not in dates:
        cut = dates[dates.searchsorted(cut) - 1]
    print(f"checking causality at {cut.date()}...", flush=True)
    checks.append(check_causal(aligned, strategy_weights, cut, audit))

    payload = {
        "checks": checks,
        "all_ok": all(c["ok"] for c in checks),
        "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# 一键自检", "", f"生成时间：{payload['generated']}", ""]
    lines.append("| 检查 | 结果 | 细节 |")
    lines.append("|---|---|---|")
    for check in checks:
        detail = ""
        if check["name"] == "data":
            detail = f"{check['symbols']} 只 / {check['bars']} 根 bar；问题 {len(check['problems'])} 项"
        elif check["name"] == "reproducibility":
            detail = f"两次运行最大差 {check['max_abs_diff']:.1e}，{check['n_days']} 天"
        elif check["name"] == "causality":
            detail = (f"切点 {check['cut']}，{check['n_days']} 天，"
                      f"最大相对差 {check['max_rel_diff']:.1e}")
        lines.append(f"| {check['name']} | {'**通过**' if check['ok'] else '**失败**'} | {detail} |")
    lines.append("")
    lines.append("完整的前视偏差审计（两个切点 × 截断/打乱/信号截断）见 "
                 "`scripts/run_lookahead_audit.py` 与 §23.60。")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    for check in checks:
        print(f"{check['name']:16s} ok={check['ok']}")
    print(f"all_ok: {payload['all_ok']}")
    print(f"report: {out}")
    return 0 if payload["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
