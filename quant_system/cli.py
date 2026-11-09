"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .adjustment import rebuild_frame
from .backtest import Backtester, slice_result
from .cache_audit import audit_cache
from .config import (  # noqa: F401
    CostConfig,
    BROAD_UNIVERSE,
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_UNIVERSE,
    BacktestConfig,
    ResearchConfig,
    RiskConfig,
    config_dict,
)
from .data import (
    align_panel,
    pit_universe,
    trim_panel,
    usable_window,
    download_universe,
    load_panel,
    load_symbol,
    panel_to_frame,
    validate_panel,
)
from .paper import (
    Ledger,
    apply_fills,
    plan_orders,
    positions_frame,
    reconcile,
    summarize,
)
from .portfolio import build_portfolio
from .report import create_run_dir, save_backtest, write_json, write_markdown_report
from .research import compare_strategies, walk_forward
from .signals import latest_signal, save_signal
from .strategies import STRATEGY_REGISTRY, build_strategy


UNIVERSES = {"default": DEFAULT_UNIVERSE, "broad": BROAD_UNIVERSE}
UNIVERSE_CHOICES = sorted({*UNIVERSES, "pit"})


def _universe_symbols(universe: str) -> dict[str, str]:
    if universe == "pit":
        return pit_universe()
    return UNIVERSES[universe]


def _panel_from_args(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    universe = getattr(args, "universe", "default") or "default"
    raw_symbols = args.symbols or []
    symbols = [
        part
        for item in raw_symbols
        for part in str(item).replace(" ", ",").split(",")
        if part
    ] or list(_universe_symbols(universe))
    cache_dir = Path(args.cache_dir)
    adjustment = getattr(args, "adjustment", "cached") or "cached"
    if adjustment == "cached":
        panel = load_panel(symbols, cache_dir)
    else:
        panel = _rebuilt_panel(symbols, cache_dir, adjustment)
    min_bars = int(getattr(args, "min_bars", 120) or 0)
    kept = {symbol: frame for symbol, frame in panel.items() if len(frame) >= min_bars}
    dropped = sorted(set(panel) - set(kept))
    if dropped:
        print(f"dropped {len(dropped)} symbols with < {min_bars} bars: {dropped}", flush=True)
    if not kept:
        raise ValueError("no symbol has enough history")
    print(f"universe={universe} symbols={len(kept)}", flush=True)
    return kept


def _rebuilt_panel(symbols, cache_dir: Path, adjustment: str) -> dict[str, pd.DataFrame]:
    """Load prices under an explicit corporate-action convention.

    ``cached`` is the provider series (kept as the default until every symbol
    has a raw cache).  ``rebuilt`` multiplies the raw traded price by the
    total-return factor implied by qfq; ``raw`` uses unadjusted prices.  Both
    alternatives need ``<symbol>_raw.parquet`` next to the qfq cache.
    """
    from .data import normalize_symbol

    panel: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for symbol in symbols:
        sym = normalize_symbol(symbol)
        qfq_path = cache_dir / f"{sym}_qfq.parquet"
        raw_path = cache_dir / f"{sym}_raw.parquet"
        if not qfq_path.exists():
            skipped.append(sym)
            continue
        qfq = pd.read_parquet(qfq_path)
        qfq.index = pd.to_datetime(qfq.index)
        if adjustment == "raw":
            if not raw_path.exists():
                skipped.append(sym)
                continue
            raw = pd.read_parquet(raw_path)
            raw.index = pd.to_datetime(raw.index)
            panel[sym] = raw
            continue
        tr_path = cache_dir / f"{sym}_tr.parquet"
        if tr_path.exists():
            # Prefer the persisted rebuild: it is deterministic and carries an
            # event sidecar, so a run does not silently recompute a different
            # segmentation after a code change.
            frame = pd.read_parquet(tr_path)
            frame.index = pd.to_datetime(frame.index)
            panel[sym] = frame
            continue
        if not raw_path.exists():
            skipped.append(sym)
            continue
        raw = pd.read_parquet(raw_path)
        raw.index = pd.to_datetime(raw.index)
        try:
            adjusted, fit = rebuild_frame(raw, qfq, symbol=sym)
        except ValueError as exc:
            # One unusable history must not abort a 189-name research run.
            # The symbol is reported below so the exclusion stays visible.
            skipped.append(sym)
            print(f"[!] {sym}: rebuild failed, symbol skipped: {exc}", flush=True)
            continue
        if fit.problems:
            print(f"[!] {sym}: adjustment rebuild warnings: {fit.problems}", flush=True)
        panel[sym] = adjusted
    if skipped:
        print(
            f"[!] {len(skipped)} symbols skipped for adjustment={adjustment} "
            f"(missing raw cache or rebuild failure): {skipped[:8]}",
            flush=True,
        )
    if not panel:
        raise ValueError(
            f"no symbol has a raw cache for adjustment={adjustment}; "
            "run `download --adjust raw` first"
        )
    return panel


def _benchmark(args: argparse.Namespace) -> pd.Series | None:
    if args.no_benchmark:
        return None
    try:
        return load_symbol(args.benchmark, Path(args.cache_dir), adjust="")["close"]
    except FileNotFoundError:
        return None


def cmd_cache_audit(args: argparse.Namespace) -> int:
    """Check cache files against their download metadata."""
    audit = audit_cache(Path(args.cache_dir))
    output = Path(args.output) if getattr(args, "output", None) else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "cache_audit.json").write_text(
            json.dumps(audit.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        frame = audit.to_frame()
        if not frame.empty:
            columns = [
                "symbol",
                "adjust",
                "rows_file",
                "rows_meta",
                "start_file",
                "end_file",
                "pages",
                "ok",
            ]
            frame.loc[:, columns].to_csv(
                output / "cache_files.csv", index=False, encoding="utf-8-sig"
            )
    summary = {
        "cache_dir": str(Path(args.cache_dir)),
        "files": len(audit.files),
        "manifest_present": audit.manifest_present,
        "manifest_symbols": audit.manifest_symbols,
        "problems": audit.problems,
        "ok": audit.ok,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0 if audit.ok else 1


def cmd_fingerprint(args: argparse.Namespace) -> int:
    """Fingerprint one backtest report, or compare two reports.

    A single ``metrics.json`` cannot prove two runs are identical: different
    paths can share a total return. The fingerprint covers the equity curve,
    the trade log and the metadata, so a real change in one trade is visible.
    """
    from .fingerprint import compare, fingerprint
    from .report import _json_default

    def load(run_dir: Path) -> tuple[pd.Series, pd.DataFrame, dict]:
        equity_path = run_dir / "equity.csv"
        metrics_path = run_dir / "metrics.json"
        if not equity_path.exists() or not metrics_path.exists():
            raise FileNotFoundError(
                f"{run_dir} is not a backtest report (equity.csv + metrics.json required)"
            )
        equity = pd.read_csv(equity_path, parse_dates=["date"], index_col="date")["equity"]
        trades_path = run_dir / "trades.csv"
        trades = (
            pd.read_csv(trades_path, encoding="utf-8-sig")
            if trades_path.exists()
            else pd.DataFrame()
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return equity, trades, metrics.get("metadata", {})

    first_dir = Path(args.run)
    equity, trades, metadata = load(first_dir)
    current = fingerprint(equity, trades, metadata)
    payload: dict = {
        "run": str(first_dir),
        "fingerprint": current,
        "trades": int(len(trades)),
    }
    status = 0
    if args.compare:
        other_dir = Path(args.compare)
        other = fingerprint(*load(other_dir))
        result = compare(current, other)
        payload["compare"] = {
            "run": str(other_dir),
            "fingerprint": other,
            "equal": result["equal"],
            "differences": result["differences"],
        }
        status = 0 if result["equal"] else 1
    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "fingerprint.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return status


def cmd_download(args: argparse.Namespace) -> int:
    payload = download_universe(
        count=args.count,
        force=args.force,
        cache_dir=Path(args.cache_dir),
        include_benchmark=not args.no_benchmark,
        universe=_universe_symbols(args.universe),
        pause=args.pause,
        adjust=getattr(args, "adjust", "qfq"),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    panel = _panel_from_args(args)
    report = validate_panel(panel)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


def cmd_audit(args: argparse.Namespace) -> int:
    """One-shot health check: data, window, sectors, delisting, latest signal.

    This exists because the reports quote headline numbers; a user needs a
    single command that shows the assumptions behind them and any hard failures.
    """
    from .sectors import sector_of

    panel = _panel_from_args(args)
    quality = validate_panel(panel)
    anchor = list(BROAD_UNIVERSE) if args.universe == "pit" else None
    research = _research_panel(panel, anchor_symbols=anchor)
    # ``_research_panel`` already clips to the coverage window; recomputing a
    # 95% cross-section on the mixed PIT pool would fail because delisted names
    # are absent after their last bar by construction.
    dates = pd.concat([frame.index.to_series() for frame in research.values()])
    start, end = dates.min(), dates.max()
    sectors = pd.Series({s: sector_of(s) for s in research})
    counts = sectors.value_counts().to_dict()

    delisted_candidates = 0
    if args.universe == "pit":
        from .data import normalize_symbol

        broad = {normalize_symbol(s) for s in BROAD_UNIVERSE}
        delisted_candidates = len([s for s in research if s not in broad])

    problems: list[str] = []
    if not quality.ok:
        problems.extend(quality.warnings)
    if counts.get("其他"):
        problems.append(f"{counts['其他']} symbols have no sector classification")

    latest = None
    try:
        latest = latest_signal(
            research,
            args.strategy,
            params=json.loads(args.params) if args.params else {},
            benchmark=_benchmark(args),
            risk=RiskConfig(),
        )
    except Exception as exc:  # noqa: BLE001 - audit must report, not crash
        problems.append(f"signal failed: {exc}")

    payload = {
        "universe": args.universe,
        "symbols_loaded": len(panel),
        "symbols_in_window": len(research),
        "window": [str(start.date()), str(end.date())],
        "data_quality": quality.to_dict(),
        "sector_counts": counts,
        "delisted_candidates": delisted_candidates,
        "latest_signal": (
            {
                "as_of": str(latest.as_of.date()),
                "positions": int(len(latest.rows)),
                "gross_exposure": round(latest.gross_exposure, 6),
                "warnings": latest.warnings,
            }
            if latest is not None
            else None
        ),
        "problems": problems,
        "ok": not problems,
    }
    output = Path(args.output) if getattr(args, "output", None) else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output / "audit.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if payload["ok"] else 1


def _research_anchor(args: argparse.Namespace) -> list[str] | None:
    """Anchor the research window on survivors, not on an explicit subset.

    ``--symbols`` narrows the *universe*; it must not change the date window
    used by the archived studies.  Without this guard an explicit PIT subset
    silently fell back to the 95%-coverage rule, which drops every name that
    stops trading early and shortens the window (151 names -> 2024-06-13).
    """
    if getattr(args, "universe", None) != "pit":
        return None
    explicit = getattr(args, "symbols", None)
    if not explicit:
        return list(BROAD_UNIVERSE)
    from .data import normalize_symbol

    wanted = {normalize_symbol(symbol) for symbol in BROAD_UNIVERSE}
    selected = {
        normalize_symbol(part)
        for item in explicit
        for part in str(item).replace(" ", ",").split(",")
        if part
    }
    anchor = sorted(selected & wanted)
    return anchor or None


def _research_panel(
    panel: dict[str, pd.DataFrame],
    *,
    anchor_symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Clip to the span where the cross-section is actually populated.

    A single name with an unusually long history (or a lone stub bar) otherwise
    drags the whole backtest back to a date where one or two stocks trade. The
    published studies use the 95%-coverage window; the CLI must match them or
    the same command reproduces different numbers.

    For the point-in-time pool the anchor is the survivor cross-section. Only
    those names define the date range; delisted names are then clipped to the
    same range instead of being dropped by the 95% coverage filter (they are
    deliberately absent after their last bar).
    """
    anchor = panel
    if anchor_symbols is not None:
        from .data import normalize_symbol

        wanted = {normalize_symbol(symbol) for symbol in anchor_symbols}
        anchor = {symbol: frame for symbol, frame in panel.items() if symbol in wanted}
        if not anchor:
            raise ValueError("none of the anchor symbols are in the panel")
    trimmed = trim_panel(anchor, min_coverage=0.95)
    if not trimmed:
        raise ValueError("no symbol covers 95% of the available history")
    start, end = usable_window(trimmed, min_active_ratio=0.95)
    if anchor_symbols is None:
        # Same rule as every published study: names that list late or stop
        # trading inside the window are excluded, otherwise the CLI and the
        # research scripts disagree on the same data.
        clipped = {
            symbol: frame.loc[start:end]
            for symbol, frame in trimmed.items()
            if len(frame.loc[start:end])
        }
    else:
        # Point-in-time pool: survivors define the window, delisted names are
        # clipped to it instead of being dropped by the coverage filter.
        clipped = {
            symbol: frame.loc[start:end]
            for symbol, frame in panel.items()
            if len(frame.loc[start:end])
        }
    print(
        f"research window {start.date()}..{end.date()} symbols={len(clipped)}",
        flush=True,
    )
    return clipped


def _risk_from_args(args: argparse.Namespace) -> RiskConfig:
    """Risk config with an optional ``--target-vol`` override.

    Accepts either a fraction (0.12) or a percentage (12). The override is
    applied everywhere a strategy is evaluated so a backtest, a walk-forward and
    a live signal cannot silently disagree about position sizing.
    """
    target = getattr(args, "target_vol", None)
    if target is None:
        return RiskConfig()
    value = float(target)
    if value > 1.0:
        value /= 100.0
    return RiskConfig(target_annual_vol=value)


def cmd_backtest(args: argparse.Namespace) -> int:
    panel = _panel_from_args(args)
    if not (args.start or args.end):
        anchor = _research_anchor(args)
        panel = _research_panel(panel, anchor_symbols=anchor)
    aligned, dates = align_panel(panel)
    # Signals are always computed on full available history so indicators are
    # warm when a sub-period is requested; only the reported window is sliced.
    benchmark = _benchmark(args)
    params = json.loads(args.params) if getattr(args, "params", None) else {}
    reset = getattr(args, "rebalance_reset", None)
    if reset:
        # The cadence anchor is a strategy property, not a portfolio overlay.
        # Keeping it in the params dict means a JSON ``--params`` cannot
        # silently disagree with the flag.
        params = {**params, "rebalance_reset": reset}
    strategy = build_strategy(args.strategy, **params)
    raw = strategy.generate(aligned)
    risk = _risk_from_args(args)
    weights = build_portfolio(
        raw,
        aligned,
        benchmark=benchmark,
        risk=risk,
        use_regime=args.use_regime,
        use_vol_target=not args.no_vol_target,
        use_turnover_cap=args.turnover_cap,
        vol_forecast=args.vol_forecast,
    )
    costs = CostConfig(
        slippage_bps=args.slippage_bps,
        impact_coefficient_bps=args.impact_coefficient_bps,
    )
    result = Backtester(
        BacktestConfig(
            initial_cash=args.cash,
            benchmark=args.benchmark,
            costs=costs,
            drawdown_stop=args.drawdown_stop,
        )
    ).run(aligned, weights, benchmark=benchmark)
    if args.start or args.end:
        # Slice the *result*, not the input panel: otherwise every sub-period
        # backtest would start flat and understate exposure.
        result = slice_result(result, start=args.start, end=args.end)
    window_benchmark = (
        benchmark.reindex(result.equity.index).ffill() if benchmark is not None else None
    )
    output = Path(args.output) if args.output else create_run_dir(f"bt_{args.strategy}")
    save_backtest(
        result,
        output,
        title=f"A-share {args.strategy} backtest",
        benchmark=window_benchmark,
        caveats=[
            "默认股票池仍是“当前仍上市”样本，存在幸存者偏差；"
            "归档口径用 140 只真实退市股实测，总收益由 +145.9% 降至 +133.2%；当前行业分类尚未重跑该对照。",
            "行业分类为静态研究映射，不是历史时点分类。",
            "冲击成本默认关闭（系数 0）；50bp 平方根模型下 100 万约 -1.7pp、5000 万约 -10.3pp。",
        ],
        notes=[
            "信号在收盘生成，成交发生在下一交易日开盘。",
            "已计入佣金、印花税、过户费、滑点和成交量参与率上限。",
            "退市标的按最后一根真实 K 线强制清算，不会以冻结价格长期持有。",
        ],
    )
    print(json.dumps(result.metrics.to_dict(), ensure_ascii=False, indent=2))
    print(f"report: {output}")
    return 0


def cmd_signal(args: argparse.Namespace) -> int:
    panel = _panel_from_args(args)
    benchmark = _benchmark(args)
    params = json.loads(args.params) if getattr(args, "params", None) else {}
    reset = getattr(args, "rebalance_reset", None)
    if reset:
        # Must match the backtest that produced the reported numbers, otherwise
        # live signals would trade a different calendar than the one studied.
        params = {**params, "rebalance_reset": reset}
    signal = latest_signal(
        panel,
        args.strategy,
        params=params,
        benchmark=benchmark,
        risk=_risk_from_args(args),
        cash=args.cash,
        lot_size=args.lot_size,
        use_regime=args.use_regime,
        use_vol_target=not args.no_vol_target,
        vol_forecast=args.vol_forecast,
        today=pd.Timestamp.today().normalize(),
        equity=getattr(args, "equity", None),
    )
    output = Path(args.output) if args.output else create_run_dir(f"signal_{args.strategy}")
    save_signal(signal, output)
    print(json.dumps(
        {
            "as_of": str(signal.as_of.date()),
            "execution": "next trading day open",
            "regime_overlay": "on" if signal.regime_on is not None else "off",
            "regime_on": signal.regime_on,
            "gross_exposure": round(signal.gross_exposure, 4),
            "positions": len(signal.rows),
            "warnings": signal.warnings,
        },
        ensure_ascii=False,
        indent=2,
    ))
    if not signal.rows.empty:
        print(signal.rows.to_string(index=False))
    print(f"signal: {output}")
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    panel = _panel_from_args(args)
    if not (args.start or args.end):
        anchor = _research_anchor(args)
        panel = _research_panel(panel, anchor_symbols=anchor)
    aligned, _ = align_panel(panel)
    benchmark = _benchmark(args)
    grid = json.loads(args.grid)
    reset = getattr(args, "rebalance_reset", None)
    if reset:
        # Applied to every candidate so parameter selection and the OOS path
        # share one cadence definition.
        grid = {**grid, "rebalance_reset": [reset]}
    research = ResearchConfig(start=args.start, end=args.end)
    sweep = int(getattr(args, "phase_sweep", 0) or 0)
    if sweep < 0:
        raise ValueError("--phase-sweep cannot be negative")
    result = walk_forward(
        aligned,
        args.strategy,
        grid,
        benchmark=benchmark,
        research=research,
        backtest=BacktestConfig(initial_cash=args.cash, benchmark=args.benchmark),
        risk=_risk_from_args(args),
        phase_offsets=range(sweep) if sweep else None,
        phase_blend=bool(getattr(args, "phase_blend", False)),
    )
    output = Path(args.output) if args.output else create_run_dir(f"wf_{args.strategy}")
    output.mkdir(parents=True, exist_ok=True)
    result.windows.to_csv(output / "windows.csv", index=False, encoding="utf-8-sig")
    result.equity.to_frame("equity").to_csv(output / "equity.csv")
    if not result.audit.empty:
        result.audit.to_csv(output / "audit.csv", index=False, encoding="utf-8-sig")
    write_json({"metrics": result.summary(), "selections": result.selections}, output / "metrics.json")
    write_markdown_report(
        output / "report.md",
        title=f"Walk-forward {args.strategy}",
        metrics=result.summary(),
        metadata={
            "strategy": args.strategy,
            "grid": grid,
            "research": config_dict(research),
            "risk": config_dict(_risk_from_args(args)),
            "phase_sweep": sweep,
            "phase_blend": bool(getattr(args, "phase_blend", False)),
        },
        windows=result.windows,
        notes=[
            "每个测试窗口的参数只由之前的训练窗口选择；测试期不参与参数更新。",
            (
                f"每个窗口还评估了 {sweep} 个调仓相位，"
                "phase_return_median/worst/best 才是稳健估计；"
                "headline 净值只对应其中一个任意相位。"
            )
            if sweep
            else "未做调仓相位扫描；单点收益可能包含显著的日历运气，见 RESEARCH_LOG 23.21。",
        ],
    )
    print(json.dumps(result.summary(), ensure_ascii=False, indent=2))
    print(f"report: {output}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    panel = _panel_from_args(args)
    if not (args.start or args.end):
        anchor = _research_anchor(args)
        panel = _research_panel(panel, anchor_symbols=anchor)
    aligned, _ = align_panel(panel)
    benchmark = _benchmark(args)
    grids = {
        "momentum": {"lookback": [40, 60, 90], "top_n": [5, 8], "trend_window": [100, 120]},
        "low_volatility": {"vol_window": [40, 60], "top_n": [5, 8], "trend_window": [100, 120]},
        "multi_factor": {"momentum_window": [40, 60, 90], "top_n": [5, 8], "trend_window": [100, 120]},
        "breakout": {"entry_window": [40, 55], "top_n": [5, 8]},
    }
    reset = getattr(args, "rebalance_reset", None)
    if reset:
        grids = {name: {**grid, "rebalance_reset": [reset]} for name, grid in grids.items()}
    sweep = int(getattr(args, "phase_sweep", 0) or 0)
    table = compare_strategies(
        aligned,
        grids,
        benchmark=benchmark,
        research=ResearchConfig(start=args.start, end=args.end),
        backtest=BacktestConfig(initial_cash=args.cash, benchmark=args.benchmark),
        risk=_risk_from_args(args),
        phase_offsets=range(sweep) if sweep else None,
    )
    output = Path(args.output) if args.output else create_run_dir("compare")
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "comparison.csv", encoding="utf-8-sig")
    print(table.to_string())
    print(f"report: {output}")
    return 0


def cmd_bootstrap_ci(args: argparse.Namespace) -> int:
    from .bootstrap_report import (
        DEFAULT_PAIRS,
        DEFAULT_SUITE,
        build_bootstrap_report,
        write_bootstrap_report,
    )

    runs: list[tuple[str, str]] = []
    pairs: list[tuple[str, str]] = []
    if args.suite:
        runs.extend(DEFAULT_SUITE)
        pairs.extend(DEFAULT_PAIRS)
    for item in args.run or []:
        label, sep, path = item.partition("=")
        if not sep or not label.strip() or not path.strip():
            raise ValueError(f"--run expects LABEL=REPORT_DIR, got {item!r}")
        runs.append((label.strip(), path.strip()))
    for item in args.pair or []:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"--pair expects A,B, got {item!r}")
        pairs.append((parts[0], parts[1]))
    if not runs:
        raise ValueError("nothing to do: pass --suite or at least one --run LABEL=REPORT_DIR")

    summary, markdown = build_bootstrap_report(
        runs,
        pairs,
        n_boot=args.n_boot,
        seed=args.seed,
        root=Path.cwd(),
        log=print,
    )
    output = Path(args.output) if args.output else create_run_dir("bootstrap_ci")
    write_bootstrap_report(output, summary, markdown)
    print(markdown)
    print(f"report: {output}")
    return 0


def cmd_brief(args: argparse.Namespace) -> int:
    from .brief import build_brief, render_markdown, save_brief

    panel = _panel_from_args(args)
    benchmark = _benchmark(args)
    params = json.loads(args.params) if getattr(args, "params", None) else {}
    reset = getattr(args, "rebalance_reset", None)
    if reset:
        params = {**params, "rebalance_reset": reset}
    brief = build_brief(
        panel,
        args.strategy,
        params=params,
        benchmark=benchmark,
        risk=_risk_from_args(args),
        equity=getattr(args, "equity", None),
        ledger_path=getattr(args, "ledger", None),
        use_regime=args.use_regime,
        use_vol_target=not args.no_vol_target,
        today=pd.Timestamp.today().normalize(),
    )
    output = Path(args.output) if args.output else create_run_dir(f"brief_{args.strategy}")
    save_brief(brief, output)
    print(render_markdown(brief))
    print(f"brief: {output}")
    return 0


def cmd_attribution(args: argparse.Namespace) -> int:
    from .attribution import (
        load_attribution_inputs,
        low_vol_benchmark,
        render_markdown,
        save_attribution,
        sector_attribution,
    )

    report_dir = Path(args.report_dir)
    positions, equity = load_attribution_inputs(report_dir)
    panel = _panel_from_args(args)

    mode = getattr(args, "benchmark_mode", "equal") or "equal"
    benchmark = None
    label = "等权全池"
    if mode != "equal":
        fraction = float(getattr(args, "benchmark_fraction", 1.0 / 3.0))
        window = int(getattr(args, "benchmark_window", 60))
        month_ends = pd.date_range(
            pd.to_datetime(positions["date"]).min(),
            pd.to_datetime(positions["date"]).max(),
            freq="ME",
        )
        benchmark = low_vol_benchmark(
            panel, month_ends, window=window, fraction=fraction, high=(mode == "highvol")
        )
        label = f"{'高' if mode == 'highvol' else '低'}波动率 {fraction:.0%} 等权（{window} 日）"

    result = sector_attribution(
        positions,
        panel,
        equity,
        start=getattr(args, "start", None),
        end=getattr(args, "end", None),
        benchmark=benchmark,
        benchmark_label=label,
    )
    meta = {
        "输入报告": str(report_dir),
        "股票池": f"{args.universe} ({len(panel)} 只)",
        "价格口径": getattr(args, "adjustment", "cached") or "cached",
        "基准": result.summary.get("benchmark", "等权全池"),
        "区间": f"{result.monthly['month'].min() if not result.monthly.empty else '-'} .. "
                f"{result.monthly['month'].max() if not result.monthly.empty else '-'}",
    }
    output = Path(args.output) if args.output else create_run_dir("attribution")
    save_attribution(result, output, title=f"行业归因 {report_dir.name}", meta=meta)
    print(render_markdown(result, title=f"行业归因 {report_dir.name}", meta=meta))
    print(f"attribution: {output}")
    return 0


def _prices_from_panel(panel: dict[str, pd.DataFrame]) -> pd.Series:
    last = {}
    for symbol, frame in panel.items():
        if frame.empty:
            continue
        close = frame["close"].dropna()
        if len(close):
            last[symbol] = float(close.iloc[-1])
    return pd.Series(last, dtype=float)


def _stale_symbols(panel: dict[str, pd.DataFrame], as_of: pd.Timestamp) -> set[str]:
    """Symbols whose last real bar is older than the signal date.

    A suspended name still has a cached price, but that price is not executable
    today. Planning orders against it would size the trade from a stale mark.
    """
    stale: set[str] = set()
    for symbol, frame in panel.items():
        if frame.empty:
            stale.add(symbol)
            continue
        last = frame["close"].dropna().index.max()
        if pd.isna(last) or last < as_of:
            stale.add(symbol)
    return stale


def cmd_paper(args: argparse.Namespace) -> int:
    panel = _panel_from_args(args)
    ledger_path = Path(args.ledger)
    if args.init:
        if ledger_path.exists():
            raise ValueError(f"ledger already exists: {ledger_path}")
        ledger = Ledger(cash=args.cash)
        ledger.save(ledger_path)
    else:
        ledger = Ledger.load(ledger_path)

    prices = _prices_from_panel(panel)
    benchmark = _benchmark(args)

    if args.plan:
        params = json.loads(args.params) if getattr(args, "params", None) else {}
        signal = latest_signal(
            panel,
            args.strategy,
            params=params,
            benchmark=benchmark,
            risk=RiskConfig(),
            cash=ledger.equity(prices) if ledger.positions else ledger.cash,
            lot_size=args.lot_size,
            use_regime=args.use_regime,
            use_vol_target=not args.no_vol_target,
        )
        stale = _stale_symbols(panel, signal.as_of)
        if stale:
            signal.warnings.append(
                f"{len(stale)} symbols have no bar on {signal.as_of.date()}; "
                "they are excluded from order planning"
            )
        orders = plan_orders(
            ledger,
            signal.target_weights,
            prices,
            lot_size=args.lot_size,
            cash_buffer=RiskConfig().cash_buffer,
            stale_symbols=stale,
        )
        output = Path(args.output) if args.output else ledger_path.parent
        output.mkdir(parents=True, exist_ok=True)
        if not orders.empty:
            orders.to_csv(output / "paper_orders.csv", index=False, encoding="utf-8-sig")
        positions = positions_frame(ledger, prices)
        positions.to_csv(output / "paper_positions.csv", index=False, encoding="utf-8-sig")
        print(json.dumps(
            {
                "as_of": str(signal.as_of.date()),
                "equity": round(ledger.equity(prices), 2),
                "cash": round(ledger.cash, 2),
                "positions": len(ledger.positions),
                "orders": len(orders),
                "warnings": signal.warnings,
            },
            ensure_ascii=False,
            indent=2,
        ))
        if not orders.empty:
            print(orders.to_string(index=False))
        print(f"orders: {output}")
        return 0

    if args.fills:
        fills = pd.read_csv(args.fills)
        ledger = apply_fills(
            ledger,
            fills,
            date=args.date or pd.Timestamp.today(),
            costs=CostConfig(slippage_bps=args.slippage_bps),
        )
        ledger.save(ledger_path)
        print(json.dumps(summarize(ledger, prices), ensure_ascii=False, indent=2))
        print(f"ledger: {ledger_path}")
        return 0

    if args.reconcile:
        broker = json.loads(Path(args.reconcile).read_text(encoding="utf-8"))
        diff = reconcile(
            ledger,
            broker.get("positions", {}),
            broker_cash=broker.get("cash"),
            tolerance=float(broker.get("tolerance", 0.0)),
        )
        output = Path(args.output) if args.output else ledger_path.parent
        output.mkdir(parents=True, exist_ok=True)
        diff.to_csv(output / "reconciliation.csv", index=False, encoding="utf-8-sig")
        print(diff.to_string(index=False) if len(diff) else "ledger matches broker")
        print(json.dumps(summarize(ledger, prices), ensure_ascii=False, indent=2))
        return 0

    print(json.dumps(summarize(ledger, prices), ensure_ascii=False, indent=2))
    positions = positions_frame(ledger, prices)
    if len(positions):
        print(positions.to_string(index=False))
    return 0


def _quote_symbols(args: argparse.Namespace) -> list[str]:
    """Symbols for the quote round: explicit list first, today's signal second.

    ``--symbols`` exists on both parsers (global: ``nargs="+"``, quote: one
    string), so the value can arrive as a list or as a single comma string.
    Treating a string as an iterable of characters is how ``'600519'`` turns into
    the symbol ``'6'`` -- the bug this branch exists to prevent.
    """
    raw = getattr(args, "symbols", None) or []
    if isinstance(raw, str):
        raw = [raw]
    symbols: list[str] = []
    for item in raw:
        symbols.extend(part.strip() for part in str(item).split(",") if part.strip())
    if symbols:
        return symbols
    today = Path("reports") / "signal_today" / "orders.csv"
    if today.exists():
        frame = pd.read_csv(today, encoding="utf-8-sig")
        if "symbol" in frame:
            return [str(s) for s in frame["symbol"].dropna()]
    raise ValueError(
        "no symbols: pass --symbols=600519,000001 or generate a signal first "
        "(python -m quant_system ... signal)"
    )


def cmd_quote(args: argparse.Namespace) -> int:
    """L1 snapshot quotes: execution and monitoring only, never research."""
    from .realtime import (
        DEFAULT_STORE,
        GREEN,
        QuotePoller,
        SnapshotStore,
        default_source,
        load_board,
        refresh,
        throttle_remainder,
    )

    symbols = _quote_symbols(args)
    store = SnapshotStore(args.store or DEFAULT_STORE)

    if args.watch:
        poller = QuotePoller(
            symbols,
            source=default_source(cross_check=args.cross_check, attempts=args.attempts),
            store=store,
            interval=args.interval,
            stale_after=args.stale_after,
        )
        try:
            while True:
                status = poller.tick()
                print(f"[{status.state}] {status.message}", flush=True)
                time.sleep(args.interval if status.state == GREEN else 1.0)
        except KeyboardInterrupt:
            print("\n已停止轮询（快照保留在 %s）" % store.path)
        return 0

    if args.offline:
        board = load_board(symbols, store=store, stale_after=args.stale_after)
    else:
        owed = throttle_remainder(store)
        if owed > 0:
            print(f"限频：等待 {owed:.1f}s（上一轮请求未满 3s）", file=sys.stderr)
            time.sleep(owed)
        board = refresh(
            symbols,
            source=default_source(cross_check=args.cross_check, attempts=args.attempts),
            store=store,
            stale_after=args.stale_after,
        )
    payload = board.to_dict()
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(board.markdown())
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(f"-> {output}")
    # A degraded round is not a command failure: outside trading hours every
    # quote is stale by construction, and the desk must keep working offline.
    return 0


def cmd_advice(args: argparse.Namespace) -> int:
    """Advice card: six elements, uncertainty on the same screen, hard freshness gate."""
    from .advice import build_card, write_card
    from .realtime import DEFAULT_STORE, SnapshotStore, last_completed_session, now_cn, refresh

    panel = _panel_from_args(args)
    # Without ``--today`` the card decides for the *next* session, so the cache
    # must contain the last session that has closed.  With ``--today D`` the
    # caller is replaying D's open, which needs the bars up to the close before D.
    if args.today:
        moment = pd.Timestamp(args.today) + pd.Timedelta(hours=9, minutes=15)
    else:
        moment = now_cn()
    # The gate wants the *aware* moment (it compares against exchange prints);
    # everything downstream is naive wall-clock, as the research side expects.
    today = pd.Timestamp(moment)
    today = today.tz_localize(None) if today.tzinfo is not None else today
    today = today.normalize()
    benchmark = _benchmark(args)

    board = None
    if args.quotes:
        store = SnapshotStore(args.store or DEFAULT_STORE)
        symbols = [row for row in (args.symbols or [])]
        if not symbols:
            try:
                signal_path = Path("reports") / "signal_today" / "orders.csv"
                if signal_path.exists():
                    symbols = list(pd.read_csv(signal_path, encoding="utf-8-sig")["symbol"])
            except (KeyError, FileNotFoundError):
                symbols = []
        if symbols:
            board = refresh(list(symbols), store=store)

    ledger = None
    if args.ledger:
        ledger = Ledger.load(args.ledger)

    last_trading = last_completed_session(moment, witness=board.print_date if board else None)

    card = build_card(
        panel,
        strategy=args.strategy,
        params=json.loads(args.params) if args.params else None,
        equity=args.equity,
        quotes=board,
        ledger=ledger,
        today=today,
        benchmark=benchmark,
        band_bps=args.band_bps,
        reports_dir=args.reports_dir,
        last_trading_day=str(last_trading),
    )
    if args.json:
        print(json.dumps(card.to_dict(), ensure_ascii=False, indent=2, default=str))
    else:
        print(card.markdown())
    if args.output:
        write_card(card, args.output)
        print("-> %s (card.md / card.json / card.html)" % args.output)
    return 0


def cmd_trade(args: argparse.Namespace) -> int:
    """Tiered trading: status / level / kill / plan / fills / reconcile / statement."""
    from .trading import (
        AuditLog,
        IdempotencyBook,
        OrderState,
        RiskLimits,
        TradingGate,
        TradingLevel,
        daily_statement,
        reconcile,
        write_alerts,
    )
    from .trading.pipeline import apply_session_fills, default_paths, plan_session
    from .trading.recon import append_equity_point

    paths = default_paths(args.trading_dir)
    gate = TradingGate(paths["gate"])
    action = args.action

    if action == "status":
        payload = {
            "gate": gate.describe(args.date),
            "paths": {key: str(value) for key, value in paths.items()},
            "audit_events": AuditLog(paths["audit"]).count(),
            "active_orders": len(IdempotencyBook(paths["idempotency"])),
            "note": "本仓没有券商网关：L3 也只写提交记录，不存在发单代码",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        return 0

    if action == "level":
        if args.level is None:
            raise ValueError("--action level 需要 --level 1|2|3")
        day = pd.Timestamp(args.date).date() if args.date else pd.Timestamp.now().date()
        gate.set_level(TradingLevel(int(args.level)), trade_date=day if int(args.level) >= 3 else None)
        if int(args.level) >= 3:
            gate.authorize_today(day)
        print(json.dumps(gate.describe(day), ensure_ascii=False, indent=2, default=str))
        return 0

    if action == "kill":
        gate.kill(args.reason or "人工停机")
        print(json.dumps(gate.describe(args.date), ensure_ascii=False, indent=2, default=str))
        return 0

    if action == "reset-kill":
        gate.reset_kill(reset_level=True)
        print(json.dumps(gate.describe(args.date), ensure_ascii=False, indent=2, default=str))
        return 0

    if action == "release":
        # A human decided not to trade a ticket (or the broker refused it).  Without
        # this, the key stays reserved all day and the plan can never be regenerated
        # -- the safe default (refuse duplicates) needs an explicit way out.
        if not (args.symbol and args.side and args.date):
            raise ValueError("--action release 需要 --symbol、--side 与 --date")
        book = IdempotencyBook(paths["idempotency"])
        key = (str(args.date), str(args.symbol), str(args.side))
        if not book.is_reserved(trade_date=key[0], symbol=key[1], side=key[2]):
            raise ValueError("%s 没有有效订单可释放" % " ".join(key))
        book.release(key, state=OrderState.CANCELLED)
        AuditLog(paths["audit"]).record(
            "order.released", trade_date=key[0], symbol=key[1], side=key[2],
            source="operator", detail={"reason": args.reason or "人工释放"},
        )
        print(json.dumps(
            {"released": " ".join(key), "active_orders": len(IdempotencyBook(paths["idempotency"]))},
            ensure_ascii=False, indent=2))
        return 0

    ledger = Ledger.load(args.ledger) if args.ledger else None
    if ledger is None and action in {"plan", "fills", "reconcile", "statement"}:
        raise ValueError("需要 --ledger（先用 paper --init 建账）")

    if action == "plan":
        payload = _load_signal_payload(args)
        trade_date = args.date or payload["as_of"]
        weights = pd.Series(
            {row["symbol"]: float(row["weight"]) for row in payload.get("orders", [])}, dtype=float
        )
        prices = pd.Series(
            {row["symbol"]: float(row["price"]) for row in payload.get("orders", []) if row.get("price")},
            dtype=float,
        )
        board = None
        snapshot_time = ""
        if args.quotes and not prices.empty:
            from .realtime import DEFAULT_STORE, SnapshotStore, refresh

            board = refresh(list(prices.index), store=SnapshotStore(args.store or DEFAULT_STORE))
            snapshot_time = board.fetched_at.isoformat() if board.fetched_at is not None else ""
            for symbol, quote in board.quotes.items():
                if quote.price > 0:
                    prices[symbol] = quote.price
        # plan_orders values the whole book, so held names that dropped out of the
        # target basket still need a price; otherwise sizing fails loudly.
        missing_marks = [s for s in ledger.positions if s not in prices.index]
        for symbol in missing_marks:
            try:
                prices[symbol] = float(load_symbol(symbol, Path(args.cache_dir))["close"].iloc[-1])
            except (FileNotFoundError, IndexError):
                continue
        equity = args.equity if args.equity else (
            ledger.equity(prices.reindex(ledger.positions.keys()).fillna(0.0)) if len(prices) else ledger.cash
        )
        outcome = plan_session(
            ledger=ledger,
            target_weights=weights,
            prices=prices,
            gate=gate,
            trade_date=trade_date,
            equity=float(equity),
            book=IdempotencyBook(paths["idempotency"]),
            audit=AuditLog(paths["audit"]),
            quotes=board,
            strategy=str(payload.get("strategy", "")),
            signal_as_of=str(payload.get("as_of", "")),
            snapshot_time=snapshot_time,
            output_dir=paths["orders"],
            equity_prev_close=args.equity_prev_close,
            names=_symbol_names(),
            limits=RiskLimits(
                max_order_notional=args.max_order_notional,
                max_daily_new_exposure=args.max_daily_new_exposure,
                max_position_weight=args.max_position_weight,
            ),
        )
        print(outcome.markdown())
        if args.output:
            target = Path(args.output)
            target.mkdir(parents=True, exist_ok=True)
            (target / "plan.json").write_text(
                json.dumps(outcome.to_dict(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            (target / "plan.md").write_text(outcome.markdown(), encoding="utf-8")
            print("-> %s" % target)
        return 0 if not outcome.halted else 1

    if action == "fills":
        fills = pd.read_csv(args.fills, encoding="utf-8-sig")
        updated = apply_session_fills(ledger, fills, trade_date=args.date or str(ledger.as_of))
        updated.save(args.ledger)
        prices = _prices_from_ledger(args, updated)
        statement = daily_statement(updated, prices, trade_date=args.date or str(updated.as_of))
        append_equity_point(paths["equity"], statement)
        print(json.dumps(statement, ensure_ascii=False, indent=2, default=str))
        return 0

    if action == "reconcile":
        broker = json.loads(Path(args.broker).read_text(encoding="utf-8"))
        report = reconcile(
            ledger,
            {str(k): int(v) for k, v in (broker.get("positions") or {}).items()},
            broker_cash=broker.get("cash"),
            tolerance=float(broker.get("tolerance", 0.0)),
            trade_date=args.date or str(ledger.as_of or ""),
        )
        write_alerts(report, paths["recon"])
        print(report.markdown())
        return 0 if report.ok else 1

    if action == "statement":
        prices = _prices_from_ledger(args, ledger)
        statement = daily_statement(ledger, prices, trade_date=args.date or str(ledger.as_of))
        append_equity_point(paths["equity"], statement)
        print(json.dumps(statement, ensure_ascii=False, indent=2, default=str))
        return 0

    raise ValueError(f"未知 action: {action}")


def _symbol_names() -> dict[str, str]:
    """Readable names keyed by normalized symbol, e.g. ``sh600519`` -> 贵州茅台.

    ``pit_universe`` keys are bare codes ("600519"), so they must be normalized:
    looking up "sh600519" against a bare-key dict silently yields an empty name
    column in the broker file (which is how this bug was noticed).
    """
    try:
        from .data import normalize_symbol, pit_universe

        return {normalize_symbol(str(key)): str(value) for key, value in pit_universe().items()}
    except Exception:  # pragma: no cover - a checkout without the manifest
        return {}


def _load_signal_payload(args: argparse.Namespace) -> dict:
    """Read the archived signal, or compute one from the panel."""
    if args.signal:
        return json.loads(Path(args.signal).read_text(encoding="utf-8"))
    default = Path("reports") / "signal_today" / "signal.json"
    if default.exists():
        return json.loads(default.read_text(encoding="utf-8"))
    panel = _panel_from_args(args)
    signal = latest_signal(
        panel,
        args.strategy,
        params=json.loads(args.params) if args.params else None,
        equity=args.equity,
        today=pd.Timestamp(args.date) if args.date else None,
    )
    return signal.to_dict()


def _prices_from_ledger(args: argparse.Namespace, ledger: Ledger) -> pd.Series:
    """Close prices for the ledger's holdings (needed to value the statement)."""
    prices: dict[str, float] = {}
    try:
        for symbol in ledger.positions:
            frame = load_symbol(symbol, Path(args.cache_dir))
            prices[symbol] = float(frame["close"].iloc[-1])
    except (FileNotFoundError, IndexError):
        pass
    return pd.Series(prices, dtype=float)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aq", description="A-share quantitative system")
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="explicit symbol list; comma-separated (``--symbols=a,b``) avoids "
        "swallowing the subcommand name",
    )
    parser.add_argument("--benchmark", default="sh000300")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument("--universe", choices=UNIVERSE_CHOICES, default="default")
    parser.add_argument("--min-bars", type=int, default=120)
    parser.add_argument(
        "--adjustment",
        choices=["cached", "rebuilt", "raw"],
        default="cached",
        help=(
            "price convention: cached provider qfq (default, additive and "
            "known-distorted), rebuilt multiplicative total return, or raw"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="data, window, sector and signal health check")
    audit.add_argument("--strategy", default="low_vol_core")
    audit.add_argument("--params", default="{}")
    audit.add_argument("--output", default=None)
    audit.set_defaults(func=cmd_audit)

    fingerprint = sub.add_parser(
        "fingerprint", help="hash a backtest report and optionally compare two runs"
    )
    fingerprint.add_argument("--run", required=True, help="report directory to fingerprint")
    fingerprint.add_argument("--compare", default=None, help="second report directory")
    fingerprint.add_argument("--output", default=None)
    fingerprint.set_defaults(func=cmd_fingerprint)

    cache_audit = sub.add_parser(
        "cache-audit", help="check cache files against download metadata"
    )
    cache_audit.add_argument("--output", default=None)
    cache_audit.set_defaults(func=cmd_cache_audit)

    download = sub.add_parser("download", help="download adjusted daily bars")
    download.add_argument("--count", type=int, default=1600)
    download.add_argument("--pause", type=float, default=2.5)
    download.add_argument("--force", action="store_true")
    download.add_argument(
        "--adjust",
        choices=["qfq", "hfq", "raw"],
        default="qfq",
        help="provider adjustment to request; raw is required for --adjustment rebuilt",
    )
    download.set_defaults(func=cmd_download)

    validate = sub.add_parser("validate", help="validate cached data")
    validate.set_defaults(func=cmd_validate)

    backtest = sub.add_parser("backtest", help="run one strategy")
    backtest.add_argument("--strategy", choices=sorted(STRATEGY_REGISTRY), default="multi_factor")
    backtest.add_argument("--start")
    backtest.add_argument("--end")
    backtest.add_argument("--cash", type=float, default=1_000_000.0)
    backtest.add_argument("--output")
    backtest.add_argument("--params", help="JSON dict of strategy parameters")
    backtest.add_argument("--use-regime", action="store_true")
    backtest.add_argument("--no-vol-target", action="store_true")
    backtest.add_argument(
        "--target-vol", type=float, default=None,
        help="annualised volatility target for position sizing (0.12 or 12)",
    )
    backtest.add_argument(
        "--rebalance-reset",
        choices=["start", "year"],
        default=None,
        help=(
            "interval cadence anchor: 'start' counts bars from the sample start "
            "(historical, makes the result depend on the window), 'year' "
            "re-anchors every January so the calendar is window-independent"
        ),
    )
    backtest.add_argument("--vol-forecast", choices=["realised", "ex_ante"], default="realised",
                          help="volatility forecast: lagged realised P&L (default) or current-basket covariance")
    backtest.add_argument("--turnover-cap", action="store_true")
    backtest.add_argument("--slippage-bps", type=float, default=5.0)
    backtest.add_argument("--impact-coefficient-bps", type=float, default=0.0,
                          help="square-root market impact; 0 disables (default)")
    backtest.add_argument("--drawdown-stop", type=float, default=None,
                          help="optional causal drawdown guard threshold, off by default")
    backtest.set_defaults(func=cmd_backtest)

    walk = sub.add_parser("walkforward", help="walk-forward out-of-sample evaluation")
    walk.add_argument("--strategy", choices=sorted(STRATEGY_REGISTRY), default="momentum")
    walk.add_argument("--grid", required=True, help='JSON e.g. {"lookback":[40,60],"top_n":[5]}')
    walk.add_argument("--start")
    walk.add_argument("--end")
    walk.add_argument("--cash", type=float, default=1_000_000.0)
    walk.add_argument(
        "--target-vol", type=float, default=None,
        help="annualised volatility target for position sizing (0.12 or 12)",
    )
    walk.add_argument(
        "--phase-sweep",
        type=int,
        default=0,
        help=(
            "also evaluate N rebalance phases per window and report median/worst/"
            "best; 20 covers a full 20-day cycle. A single point hides a >100pp "
            "calendar spread (RESEARCH_LOG 23.21)"
        ),
    )
    walk.add_argument(
        "--phase-blend",
        action="store_true",
        help="trade the average of all swept phases instead of phase 0",
    )
    walk.add_argument(
        "--rebalance-reset",
        choices=["start", "year"],
        default=None,
        help="interval cadence anchor; 'year' is window-independent",
    )
    walk.add_argument("--output")
    walk.set_defaults(func=cmd_walkforward)

    brief = sub.add_parser("brief", help="one-page daily brief: data, signal, risk, ledger")
    brief.add_argument("--strategy", choices=sorted(STRATEGY_REGISTRY), default="low_vol_core")
    brief.add_argument("--params", default="{}")
    brief.add_argument("--equity", type=float, default=None,
                       help="account equity used to size share counts")
    brief.add_argument("--ledger", default=None, help="optional paper ledger to reconcile against")
    brief.add_argument("--use-regime", action="store_true")
    brief.add_argument("--no-vol-target", action="store_true")
    brief.add_argument("--target-vol", type=float, default=None,
                       help="annualised volatility target (0.12 or 12)")
    brief.add_argument("--rebalance-reset", choices=["start", "year"], default=None)
    brief.add_argument("--output")
    brief.set_defaults(func=cmd_brief)

    attrib = sub.add_parser(
        "attribution", help="Brinson sector attribution of a finished run's holdings"
    )
    attrib.add_argument("--report-dir", required=True,
                        help="backtest/walk-forward directory holding positions.csv and equity.csv")
    attrib.add_argument("--universe", choices=UNIVERSE_CHOICES, default="pit")
    # SUPPRESS keeps the top-level --cache-dir/--adjustment/--min-bars visible:
    # a subparser default would silently override ``aq --adjustment rebuilt`` and
    # attribute the wrong price series.
    attrib.add_argument("--cache-dir", default=argparse.SUPPRESS)
    attrib.add_argument("--adjustment", choices=["cached", "rebuilt", "raw"], default=argparse.SUPPRESS)
    attrib.add_argument("--min-bars", type=int, default=argparse.SUPPRESS)
    attrib.add_argument("--benchmark-mode", choices=["equal", "lowvol", "highvol"],
                        default="equal",
                        help="equal-weight the whole pool, or the calmest / wildest third")
    attrib.add_argument("--benchmark-window", type=int, default=60,
                        help="trailing volatility window used to build the factor benchmark")
    attrib.add_argument("--benchmark-fraction", type=float, default=1.0 / 3.0,
                        help="share of names in the factor benchmark (default 1/3)")
    attrib.add_argument("--start", default=None, help="first month to attribute (default: from positions)")
    attrib.add_argument("--end", default=None, help="last month to attribute (default: to positions end)")
    attrib.add_argument("--output")
    attrib.set_defaults(func=cmd_attribution)

    signal = sub.add_parser("signal", help="generate the next-day target portfolio")
    signal.add_argument("--strategy", choices=sorted(STRATEGY_REGISTRY), default="low_vol_core")
    signal.add_argument("--params", help="JSON dict of strategy parameters")
    signal.add_argument("--cash", type=float, default=1_000_000.0)
    signal.add_argument(
        "--equity",
        type=float,
        default=None,
        help="total account equity for sizing (default: --cash); use this when "
        "the account already holds positions",
    )
    signal.add_argument("--lot-size", type=int, default=100)
    signal.add_argument("--output")
    signal.add_argument("--slippage-bps", type=float, default=5.0)
    signal.add_argument("--use-regime", action="store_true")
    signal.add_argument("--no-vol-target", action="store_true")
    signal.add_argument(
        "--target-vol", type=float, default=None,
        help="annualised volatility target; must match the backtest you trust",
    )
    signal.add_argument(
        "--rebalance-reset", choices=["start", "year"], default=None,
        help="interval cadence anchor; 'year' matches a year-reset backtest",
    )
    signal.add_argument("--vol-forecast", choices=["realised", "ex_ante"], default="realised")
    signal.set_defaults(func=cmd_signal)

    paper = sub.add_parser("paper", help="paper-trading ledger: plan orders, book fills, reconcile")
    paper.add_argument("--ledger", required=True, help="path to the JSON ledger")
    paper.add_argument("--init", action="store_true", help="create the ledger with --cash")
    paper.add_argument("--strategy", choices=sorted(STRATEGY_REGISTRY), default="low_vol_core")
    paper.add_argument("--params")
    paper.add_argument("--cash", type=float, default=1_000_000.0)
    paper.add_argument("--lot-size", type=int, default=100)
    paper.add_argument("--plan", action="store_true", help="generate orders from the latest signal")
    paper.add_argument("--fills", help="CSV of executed fills: symbol,side,shares,price")
    paper.add_argument("--date", help="fill date (defaults to today)")
    paper.add_argument("--reconcile", help="JSON broker statement: {cash, positions}")
    paper.add_argument("--output")
    paper.add_argument("--slippage-bps", type=float, default=5.0)
    paper.add_argument("--use-regime", action="store_true")
    paper.add_argument("--no-vol-target", action="store_true")
    paper.set_defaults(func=cmd_paper)

    compare = sub.add_parser("compare", help="compare strategies out of sample")
    compare.add_argument("--start")
    compare.add_argument("--end")
    compare.add_argument("--cash", type=float, default=1_000_000.0)
    compare.add_argument(
        "--target-vol", type=float, default=None,
        help="annualised volatility target applied to every strategy",
    )
    compare.add_argument(
        "--phase-sweep", type=int, default=0,
        help="also evaluate N rebalance phases per window for every strategy",
    )
    compare.add_argument(
        "--rebalance-reset", choices=["start", "year"], default=None,
        help="interval cadence anchor applied to every strategy's grid",
    )
    compare.add_argument("--output")
    compare.set_defaults(func=cmd_compare)
    boot = sub.add_parser(
        "bootstrap-ci",
        help="block-bootstrap confidence intervals for walk-forward equity curves",
    )
    boot.add_argument("--suite", action="store_true", help="use the built-in reports/ suite")
    boot.add_argument("--run", action="append", metavar="LABEL=REPORT_DIR")
    boot.add_argument("--pair", action="append", metavar="A,B")
    boot.add_argument("--n-boot", type=int, default=5000)
    boot.add_argument("--seed", type=int, default=20261107)
    boot.add_argument("--output")
    boot.set_defaults(func=cmd_bootstrap_ci)
    quote = sub.add_parser(
        "quote",
        help="L1 snapshot quotes for execution and monitoring (never used by research)",
    )
    quote.add_argument(
        "--symbols",
        default=None,
        help="comma-separated symbols; falls back to reports/signal_today/orders.csv",
    )
    quote.add_argument(
        "--offline",
        action="store_true",
        help="read the local snapshot store only (no network); every price is then stale",
    )
    quote.add_argument("--json", action="store_true", help="print the JSON payload instead of a table")
    quote.add_argument("--store", default=None, help="sqlite snapshot store (default reports/realtime/quotes.sqlite)")
    quote.add_argument("--attempts", type=int, default=2, help="retries per batch before degrading")
    quote.add_argument(
        "--cross-check",
        dest="cross_check",
        action="store_true",
        default=True,
        help="also query Sina and flag a >0.5%% disagreement as suspect (default: on)",
    )
    quote.add_argument("--no-cross-check", dest="cross_check", action="store_false",
                       help="single source only (halves the request volume)")
    quote.add_argument("--watch", action="store_true",
                       help="poll continuously and print the freshness state (Ctrl-C to stop)")
    quote.add_argument("--interval", type=float, default=3.0, help="seconds between rounds in --watch")
    quote.add_argument("--stale-after", type=float, default=60.0,
                       help="seconds after which an in-session print counts as lagging")
    quote.add_argument("--output", default=None, help="also write the JSON payload here")
    quote.set_defaults(func=cmd_quote)
    advice = sub.add_parser(
        "advice",
        help="executable advice card: six elements, uncertainty on screen, hard data gate",
    )
    advice.add_argument("--strategy", choices=sorted(STRATEGY_REGISTRY), default="low_vol_core")
    advice.add_argument("--params")
    advice.add_argument("--equity", type=float, default=1_000_000.0,
                        help="account equity used for sizing (the card labels it as simulation)")
    advice.add_argument("--today", default=None, help="override 'today' (YYYY-MM-DD) for replay/tests")
    advice.add_argument("--quotes", action="store_true",
                        help="attach live L1 quotes (price + timestamp + real limit band)")
    advice.add_argument("--store", default=None, help="snapshot store used with --quotes")
    advice.add_argument("--ledger", default=None, help="paper ledger JSON for T+1 sellable shares")
    advice.add_argument("--band-bps", type=float, default=50.0,
                        help="limit-price band around the executable price, in basis points")
    advice.add_argument("--reports-dir", default=str(Path("reports")),
                        help="where the uncertainty artifacts live")
    advice.add_argument("--json", action="store_true")
    advice.add_argument("--output", default=None, help="write card.md / card.json / card.html here")
    advice.set_defaults(func=cmd_advice)
    trade = sub.add_parser(
        "trade",
        help="tiered trading: status/level/kill/reset-kill/plan/fills/reconcile/statement",
    )
    trade.add_argument(
        "--action",
        required=True,
        choices=["status", "level", "kill", "reset-kill", "release", "plan", "fills",
                 "reconcile", "statement"],
    )
    trade.add_argument("--ledger", default=None, help="paper ledger JSON")
    trade.add_argument("--signal", default=None, help="signal.json to plan from (default reports/signal_today)")
    trade.add_argument("--broker", default=None, help="broker statement JSON: {cash, positions}")
    trade.add_argument("--fills", default=None, help="CSV of executed fills")
    trade.add_argument("--date", default=None, help="trade date (YYYY-MM-DD)")
    trade.add_argument("--level", type=int, choices=[0, 1, 2, 3], default=None)
    trade.add_argument("--reason", default=None, help="kill-switch / release reason")
    trade.add_argument("--symbol", default=None, help="--action release: symbol to release")
    trade.add_argument("--side", default=None, choices=["buy", "sell"], help="--action release")
    trade.add_argument("--equity", type=float, default=None)
    trade.add_argument("--max-order-notional", type=float, default=200_000.0)
    trade.add_argument("--max-daily-new-exposure", type=float, default=None,
                       help="absolute daily new-exposure cap; omit for 0.95 x equity")
    trade.add_argument("--max-position-weight", type=float, default=0.25)
    trade.add_argument("--equity-prev-close", type=float, default=None,
                       help="yesterday's equity; enables the 3%% daily-loss circuit breaker")
    trade.add_argument("--quotes", action="store_true", help="price the plan from live L1 quotes")
    trade.add_argument("--store", default=None, help="snapshot store used with --quotes")
    trade.add_argument("--trading-dir", default=str(Path("reports") / "trading"),
                       help="where gate.json / orders.jsonl / audit.jsonl live")
    trade.add_argument("--output", default=None)
    trade.set_defaults(func=cmd_trade)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

