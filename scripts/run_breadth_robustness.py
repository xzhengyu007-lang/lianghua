"""Breadth robustness: fixed top_n=6 vs the default top_n=8.

The one-at-a-time plateau suggested a narrower book, but a single full-sample
sweep is exactly the kind of evidence that can be curve-fit. This script keeps
every comparison paired and writes the raw artifacts so a later reviewer can
re-derive the conclusion:

* fixed 6/2 and 8/2 under 504/126/10 and four other window lengths,
* six start-date shifts on the same 504/126/10 schedule,
* full-sample cost sweep from 0 to 50 bps,
* a local parameter neighbourhood around 6/2,
* concentration: realised-PnL share and drop-the-best reruns,
* random-selection null with the same size, cadence and sector cap.

Nothing here changes the default. It produces reports/breadth_robustness/ and
leaves the decision to a separate point-in-time comparison.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester  # noqa: E402
from quant_system.config import (  # noqa: E402
    BROAD_UNIVERSE,
    BacktestConfig,
    CostConfig,
    ResearchConfig,
    RiskConfig,
)
from quant_system.data import (  # noqa: E402
    availability,
    load_panel,
    load_symbol,
    trim_panel,
    usable_window,
)
from quant_system.html_report import write_dashboard  # noqa: E402
from quant_system.portfolio import build_portfolio  # noqa: E402
from quant_system.report import write_json  # noqa: E402
from quant_system.research import walk_forward  # noqa: E402
from quant_system.strategies import _select_weights, build_strategy  # noqa: E402

FIXED = {"vol_window": 60, "trend_window": 120, "rebalance_days": 20}
WIDTHS = {"top_n=6,cap=2": (6, 2), "top_n=8,cap=2": (8, 2)}
WINDOWS = [(252, 63, 5), (378, 126, 10), (504, 126, 10), (630, 126, 10), (756, 126, 10)]
SHIFTS = [0, 20, 40, 60, 80, 100]
SLIPPAGE = [0, 5, 10, 20, 30, 50]
DRAWS = 100
SEED = 20261107
OUT = ROOT / "reports" / "breadth_robustness"


def params(top_n: int, cap: int) -> dict:
    return {**FIXED, "top_n": top_n, "max_per_sector": cap}


def run_full(panel, benchmark, param: dict, *, slippage: float = 5.0) -> dict:
    raw = build_strategy("low_vol_core", **param).generate(panel)
    weights = build_portfolio(
        raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False
    )
    cfg = BacktestConfig(
        initial_cash=1_000_000,
        costs=CostConfig(slippage_bps=slippage),
    )
    result = Backtester(cfg).run(panel, weights, benchmark=benchmark)
    return {"result": result, "metrics": result.metrics.to_dict()}


def run_walk(panel, benchmark, param: dict, research: ResearchConfig):
    grid = {key: [value] for key, value in param.items()}
    return walk_forward(
        panel,
        "low_vol_core",
        grid,
        benchmark=benchmark,
        research=research,
        backtest=BacktestConfig(initial_cash=1_000_000),
        risk=RiskConfig(),
        portfolio_kwargs={"use_regime": False},
        verbose=False,
    )


def shift_start(research: ResearchConfig, dates: pd.DatetimeIndex, days: int) -> ResearchConfig:
    """Move the first test date forward by ``days`` trading bars.

    ``walk_forward`` starts its first test window at ``start + train + embargo``
    only when ``start`` is unset. When ``start`` is set, it is the first test
    date itself. We therefore anchor the shift on the canonical first test date
    rather than on the beginning of the panel.
    """
    if days == 0:
        return research
    canonical_first = dates[research.train_days + research.embargo_days]
    target_pos = int(dates.searchsorted(canonical_first, side="left")) + days
    if target_pos >= len(dates):
        return research
    requested = dates[target_pos]
    return ResearchConfig(
        start=str(requested.date()),
        end=research.end,
        train_days=research.train_days,
        test_days=research.test_days,
        embargo_days=research.embargo_days,
    )


def summarise(label: str, result, extra: dict | None = None) -> dict:
    summary = result.summary()
    windows = result.windows
    row = {
        "variant": label,
        "total_return": summary["total_return"],
        "cagr": summary["cagr"],
        "annual_volatility": summary["annual_volatility"],
        "sharpe": summary["sharpe"],
        "max_drawdown": summary["max_drawdown"],
        "information_ratio": summary["information_ratio"],
        "exposure": summary["exposure"],
        "positive_windows": int((windows["test_return"] > 0).sum()),
        "windows": int(len(windows)),
    }
    if extra:
        row.update(extra)
    return row


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {symbol: frame.loc[start:end] for symbol, frame in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    dates = pd.DatetimeIndex(sorted({d for frame in panel.values() for d in frame.index}))
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    payload: dict = {
        "window": [str(start.date()), str(end.date())],
        "symbols": len(panel),
        "fixed_params": FIXED,
        "seed": SEED,
        "note": (
            "All comparisons are paired on the same dates, costs and overlays. "
            "Walk-forward test windows are strictly out of sample; the full-sample "
            "cost sweep and concentration reruns are descriptive, not OOS."
        ),
    }

    # 1. Window-length robustness, fixed parameters.
    print("window lengths", flush=True)
    length_rows = []
    length_frames = []
    for train, test, embargo in WINDOWS:
        research = ResearchConfig(train_days=train, test_days=test, embargo_days=embargo)
        for label, (top_n, cap) in WIDTHS.items():
            result = run_walk(panel, benchmark, params(top_n, cap), research)
            row = summarise(label, result, {"train_days": train, "test_days": test, "embargo_days": embargo})
            length_rows.append(row)
            frame = result.windows.copy()
            frame["variant"] = label
            frame["train_days"] = train
            frame["test_days"] = test
            frame["embargo_days"] = embargo
            length_frames.append(frame)
            print(
                f"{train}/{test}/{embargo} {label}: {row['total_return']:+.2%} "
                f"sharpe={row['sharpe']:.3f} mdd={row['max_drawdown']:.2%}",
                flush=True,
            )
    length_table = pd.DataFrame(length_rows)
    length_table.to_csv(OUT / "window_lengths.csv", index=False, encoding="utf-8-sig")
    pd.concat(length_frames, ignore_index=True).to_csv(
        OUT / "window_lengths_detail.csv", index=False, encoding="utf-8-sig"
    )
    payload["window_lengths"] = length_rows

    # 2. Start-date shifts on the canonical schedule. The walk-forward start is
    #    shifted by trading bars, keeping every window length identical.
    print("start shifts", flush=True)
    shift_rows = []
    shift_frames = []
    base_research = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
    for days in SHIFTS:
        research = shift_start(base_research, dates, days)
        for label, (top_n, cap) in WIDTHS.items():
            result = run_walk(panel, benchmark, params(top_n, cap), research)
            row = summarise(label, result, {"shift_days": days, "requested_start": research.start})
            shift_rows.append(row)
            frame = result.windows.copy()
            frame["variant"] = label
            frame["shift_days"] = days
            shift_frames.append(frame)
            print(
                f"shift {days:3d} {label}: {row['total_return']:+.2%} "
                f"sharpe={row['sharpe']:.3f}",
                flush=True,
            )
    shift_table = pd.DataFrame(shift_rows)
    shift_table.to_csv(OUT / "start_shifts.csv", index=False, encoding="utf-8-sig")
    pd.concat(shift_frames, ignore_index=True).to_csv(
        OUT / "start_shifts_detail.csv", index=False, encoding="utf-8-sig"
    )
    payload["start_shifts"] = shift_rows

    # 3. Full-sample cost sweep. This is descriptive: it holds parameters fixed
    #    on the whole sample, so it cannot be read as an OOS result.
    print("cost sweep", flush=True)
    cost_rows = []
    for slippage in SLIPPAGE:
        for label, (top_n, cap) in WIDTHS.items():
            metrics = run_full(panel, benchmark, params(top_n, cap), slippage=slippage)["metrics"]
            cost_rows.append(
                {
                    "variant": label,
                    "slippage_bps": slippage,
                    "total_return": metrics["total_return"],
                    "cagr": metrics["cagr"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "turnover": metrics["turnover"],
                }
            )
            print(
                f"{slippage:2d}bp {label}: {metrics['total_return']:+.2%} "
                f"sharpe={metrics['sharpe']:.3f}",
                flush=True,
            )
    cost_table = pd.DataFrame(cost_rows)
    cost_table.to_csv(OUT / "cost_sweep.csv", index=False, encoding="utf-8-sig")
    payload["cost_sweep"] = cost_rows

    # 4. Local parameter neighbourhood around 6/2 (full sample, descriptive).
    print("neighbourhood", flush=True)
    neighbourhood = []
    for vol_window in (40, 50, 60, 80, 100):
        for trend_window in (80, 100, 120, 150, 180):
            if (vol_window, trend_window) not in {(60, 120)}:
                continue
            for rebalance_days in (10, 20, 25, 30):
                param = {
                    "top_n": 6,
                    "max_per_sector": 2,
                    "vol_window": vol_window,
                    "trend_window": trend_window,
                    "rebalance_days": rebalance_days,
                }
                metrics = run_full(panel, benchmark, param)["metrics"]
                neighbourhood.append({"axis": "rebalance_days", "value": rebalance_days, **param, **metrics})
    for vol_window in (40, 50, 60, 80, 100):
        param = {"top_n": 6, "max_per_sector": 2, "vol_window": vol_window, "trend_window": 120, "rebalance_days": 20}
        metrics = run_full(panel, benchmark, param)["metrics"]
        neighbourhood.append({"axis": "vol_window", "value": vol_window, **param, **metrics})
    for trend_window in (80, 100, 120, 150, 180):
        param = {"top_n": 6, "max_per_sector": 2, "vol_window": 60, "trend_window": trend_window, "rebalance_days": 20}
        metrics = run_full(panel, benchmark, param)["metrics"]
        neighbourhood.append({"axis": "trend_window", "value": trend_window, **param, **metrics})
    for cap in (1, 2, 3, 5):
        param = {"top_n": 6, "max_per_sector": cap, "vol_window": 60, "trend_window": 120, "rebalance_days": 20}
        metrics = run_full(panel, benchmark, param)["metrics"]
        neighbourhood.append({"axis": "max_per_sector", "value": cap, **param, **metrics})
    for buffer_rank in (1.25, 1.5, 2.0):
        param = {"top_n": 6, "max_per_sector": 2, "vol_window": 60, "trend_window": 120, "rebalance_days": 20}
        raw = build_strategy("low_vol_core", buffer_rank=buffer_rank, **param).generate(panel)
        weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
        metrics = Backtester(BacktestConfig(initial_cash=1_000_000)).run(
            panel, weights, benchmark=benchmark
        ).metrics.to_dict()
        neighbourhood.append({"axis": "buffer_rank", "value": buffer_rank, **param, "buffer_rank": buffer_rank, **metrics})
    neighbourhood_table = pd.DataFrame(neighbourhood)
    neighbourhood_table.to_csv(OUT / "neighbourhood.csv", index=False, encoding="utf-8-sig")
    payload["neighbourhood"] = neighbourhood

    # 5. Concentration and drop-the-best reruns for both widths.
    print("concentration", flush=True)
    concentration_rows = []
    concentration_summary = {}
    for label, (top_n, cap) in WIDTHS.items():
        base = run_full(panel, benchmark, params(top_n, cap))["result"]
        trades = base.trades
        realised = (
            trades[trades["side"] == "sell"].groupby("symbol")["realized_pnl"].sum().sort_values(ascending=False)
            if len(trades)
            else pd.Series(dtype=float)
        )
        total = float(realised.sum())
        row = {
            "variant": label,
            "case": "baseline",
            "total_return": base.metrics.total_return,
            "sharpe": base.metrics.sharpe,
            "max_drawdown": base.metrics.max_drawdown,
            "top10_share": float(realised.head(10).sum() / total) if total else float("nan"),
        }
        concentration_rows.append(row)
        for n in (1, 3, 5, 10):
            dropped = set(realised.head(n).index)
            subset = {s: f for s, f in panel.items() if s not in dropped}
            result = run_full(subset, benchmark, params(top_n, cap))["result"]
            concentration_rows.append(
                {
                    "variant": label,
                    "case": f"drop_top_{n}",
                    "total_return": result.metrics.total_return,
                    "sharpe": result.metrics.sharpe,
                    "max_drawdown": result.metrics.max_drawdown,
                    "top10_share": float(realised.head(10).sum() / total) if total else float("nan"),
                }
            )
        concentration_summary[label] = {
            "top10_share": row["top10_share"],
            "realised_pnl": total,
            "top_symbols": realised.head(10).to_dict(),
        }
        print(
            f"{label}: top10 share={row['top10_share']:.1%}",
            flush=True,
        )
    concentration_table = pd.DataFrame(concentration_rows)
    concentration_table.to_csv(OUT / "concentration.csv", index=False, encoding="utf-8-sig")
    payload["concentration"] = concentration_summary

    # 6. Random-selection null for each width. Random scores reuse the same
    #    selection engine, so sector cap, buffer, cadence and costs are paired.
    print("random control", flush=True)
    avail = availability(panel)
    rng = np.random.default_rng(SEED)
    random_rows = []
    for label, (top_n, cap) in WIDTHS.items():
        base_raw = build_strategy("low_vol_core", **params(top_n, cap)).generate(panel)
        base = run_full(panel, benchmark, params(top_n, cap))["metrics"]
        draws = []
        for draw in range(DRAWS):
            scores = pd.DataFrame(rng.random(base_raw.shape), index=base_raw.index, columns=base_raw.columns)
            raw = _select_weights(
                scores,
                top_n,
                available=avail,
                max_per_sector=cap,
                rebalance_days=FIXED["rebalance_days"],
            )
            weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
            metrics = Backtester(BacktestConfig(initial_cash=1_000_000)).run(
                panel, weights, benchmark=benchmark
            ).metrics.to_dict()
            draws.append(
                {
                    "variant": label,
                    "draw": draw,
                    "total_return": metrics["total_return"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                }
            )
            if draw % 20 == 0:
                print(f"{label} draw {draw:03d}", flush=True)
        frame = pd.DataFrame(draws)
        random_rows.extend(draws)
        returns = frame["total_return"]
        summary = {
            "baseline_return": base["total_return"],
            "baseline_sharpe": base["sharpe"],
            "mean_return": float(returns.mean()),
            "median_return": float(returns.median()),
            "p05_return": float(returns.quantile(0.05)),
            "p95_return": float(returns.quantile(0.95)),
            "max_return": float(returns.max()),
            "one_sided_p": float((returns >= base["total_return"]).mean()),
            "mean_sharpe": float(frame["sharpe"].mean()),
            "median_sharpe": float(frame["sharpe"].median()),
            "p95_sharpe": float(frame["sharpe"].quantile(0.95)),
        }
        payload.setdefault("random_control", {})[label] = summary
        print(f"{label} random: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    pd.DataFrame(random_rows).to_csv(OUT / "random_draws.csv", index=False, encoding="utf-8-sig")

    # 7. Equity paths for the canonical 6/2 and 8/2 walk-forward, plus a
    #    dashboard for the candidate. Written last so a partial run still leaves
    #    the tables above intact.
    canonical = ResearchConfig(train_days=504, test_days=126, embargo_days=10)
    for label, (top_n, cap) in WIDTHS.items():
        result = run_walk(panel, benchmark, params(top_n, cap), canonical)
        result.equity.to_frame("equity").to_csv(OUT / f"equity_{label.replace(',', '_').replace('=', '')}.csv")
        if top_n == 6:
            bench_eq = benchmark.reindex(result.equity.index).ffill().dropna()
            bench_eq = bench_eq / bench_eq.iloc[0] * result.equity.iloc[0]
            write_dashboard(
                OUT / "dashboard.html",
                title="Breadth robustness · fixed top_n=6 cap=2",
                equity=result.equity,
                metrics=result.summary(),
                benchmark_equity=bench_eq,
                windows=result.windows,
                notes=[
                    "固定 top_n=6、行业上限 2；504/126/10 严格滚动样本外。",
                    "候选配置，不改变默认 top_n=8；决策见 pit_breadth_6_vs_8。",
                    "随机对照、成本与集中度均为描述性补充，不能与 OOS 混读。",
                ],
            )

    write_json(payload, OUT / "summary.json")
    print(f"report: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
