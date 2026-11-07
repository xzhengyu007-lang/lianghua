from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))

from quant_system.backtest import Backtester
from quant_system.config import BROAD_UNIVERSE, BacktestConfig, RiskConfig
from quant_system.data import (
    SHARES_PER_VOLUME_UNIT, availability, infer_price_limit, load_panel, load_symbol,
    panel_to_frame, trim_panel, usable_window,
)
from quant_system.portfolio import build_portfolio, normalize_targets
from quant_system.report import write_json
from quant_system.signals import latest_signal
from quant_system.strategies import build_strategy

OUT = ROOT / "reports" / "signal_consistency"
PARAMS = dict(top_n=8, vol_window=60, trend_window=120, max_per_sector=2, rebalance_days=20)


def drift_weights(target: pd.Series, returns: pd.Series) -> pd.Series:
    filled = target.fillna(0.0) * (1.0 + returns.reindex(target.index).fillna(0.0))
    gross = float(filled.sum())
    return filled / gross * float(target.sum()) if gross > 0 else filled


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "data" / "cache"
    panel = trim_panel(load_panel(list(BROAD_UNIVERSE), cache), min_coverage=0.95)
    start, end = usable_window(panel, min_active_ratio=0.95)
    panel = {s: f.loc[start:end] for s, f in panel.items()}
    benchmark = load_symbol("sh000300", cache, adjust="")["close"].loc[start:end]
    print(f"window {start.date()}..{end.date()} symbols={len(panel)}", flush=True)

    cfg = BacktestConfig(initial_cash=1_000_000.0)
    open_frame = panel_to_frame(panel, "open")
    close_frame = panel_to_frame(panel, "close")
    volume_frame = panel_to_frame(panel, "volume")
    symbols = sorted(panel)
    has_bar = availability(panel).reindex(columns=symbols).fillna(False)
    adv = (
        volume_frame.rolling(cfg.adv_window, min_periods=max(3, cfg.adv_window // 4)).mean()
        * SHARES_PER_VOLUME_UNIT
    )
    prev_close = close_frame.shift(1)
    limit = pd.Series({s: infer_price_limit(s) for s in symbols})
    limit_down = open_frame.le(prev_close.mul(1 - limit, axis=1) + 1e-6)
    limit_up = open_frame.ge(prev_close.mul(1 + limit, axis=1) - 1e-6)

    cases = [
        ("baseline_8", dict(PARAMS)),
        ("narrow_3", dict(PARAMS, top_n=3)),
        ("narrow_2", dict(PARAMS, top_n=2)),
        ("wide_12", dict(PARAMS, top_n=12)),
    ]
    case_rows, detail_rows, cause_rows = [], [], []
    for name, params in cases:
        raw = build_strategy("low_vol_core", **params).generate(panel)
        weights = build_portfolio(raw, panel, benchmark=benchmark, risk=RiskConfig(), use_regime=False)
        normalised = normalize_targets(weights, symbols, weights.index, cfg)
        result = Backtester(cfg).run(panel, weights, benchmark=benchmark)
        signal = latest_signal(
            panel, "low_vol_core", params=params, benchmark=benchmark,
            risk=RiskConfig(), cash=1_000_000.0, use_regime=False,
        )

        dates = normalised.index
        changed = normalised.ne(normalised.shift(1)).any(axis=1)
        signal_dates = list(dates[changed])
        book = result.positions.pivot(
            index="date", columns="symbol", values="weight"
        ).fillna(0.0).reindex(columns=symbols).fillna(0.0)
        equity_by_date = result.equity

        for sig_date in signal_dates:
            loc = dates.get_loc(sig_date)
            if loc + 1 >= len(dates):
                continue
            fill = dates[loc + 1]
            if fill not in book.index:
                continue
            target = normalised.loc[sig_date].reindex(symbols).fillna(0.0)
            actual = book.loc[fill]
            day_return = (close_frame.loc[fill] / open_frame.loc[fill] - 1.0).reindex(symbols).fillna(0.0)
            ideal = drift_weights(target, day_return)
            equity = float(equity_by_date.loc[fill])

            extras = [s for s in symbols if actual[s] > 1e-6 and target[s] <= 1e-9]
            missing = [s for s in symbols if target[s] > 1e-9 and actual[s] <= 1e-6]

            causes: list[str] = []
            unplanned_names = 0
            for s in extras:
                if not bool(has_bar.loc[fill, s]):
                    causes.append("suspension_blocks_sell")
                elif bool(limit_down.loc[fill, s]):
                    causes.append("limit_down_blocks_sell")
                else:
                    causes.append("unexplained_extra")
                    unplanned_names += 1
            for s in missing:
                price = open_frame.loc[fill, s]
                one_lot = float(price) * cfg.lot_size if pd.notna(price) else np.nan
                if not bool(has_bar.loc[fill, s]):
                    causes.append("suspension_blocks_buy")
                elif bool(limit_up.loc[fill, s]):
                    causes.append("limit_up_blocks_buy")
                elif np.isfinite(one_lot) and one_lot > target[s] * equity:
                    causes.append("one_lot_exceeds_target")
                else:
                    causes.append("unexplained_missing")
                    unplanned_names += 1

            detail_rows.append({
                "case": name,
                "signal_date": str(sig_date.date()),
                "fill_date": str(fill.date()),
                "target_names": int((target > 1e-9).sum()),
                "actual_names": int((actual > 1e-6).sum()),
                "extra_names": len(extras),
                "missing_names": len(missing),
                "unexplained_names": unplanned_names,
                "target_gross": round(float(target.sum()), 6),
                "actual_gross": round(float(actual.sum()), 6),
                "l1_vs_drift_ideal": round(float((actual - ideal).abs().sum()), 6),
                "gross_gap": round(float(actual.sum() - ideal.sum()), 6),
                "causes": "|".join(sorted(causes)),
            })
            for cause in causes:
                cause_rows.append({"case": name, "cause": cause})

        frame = pd.DataFrame([r for r in detail_rows if r["case"] == name])
        case_rows.append({
            "case": name,
            "top_n": params["top_n"],
            "rebalances_checked": int(len(frame)),
            "rebalances_with_name_mismatch": int((frame["unexplained_names"] > 0).sum()),
            "signal_vs_backtest_target_l1": round(
                float((signal.target_weights.reindex(symbols).fillna(0.0) - normalised.loc[dates[-1]]).abs().sum()), 6
            ),
            "l1_vs_drift_ideal_mean": round(float(frame["l1_vs_drift_ideal"].mean()), 6),
            "l1_vs_drift_ideal_p95": round(float(frame["l1_vs_drift_ideal"].quantile(0.95)), 6),
            "l1_vs_drift_ideal_max": round(float(frame["l1_vs_drift_ideal"].max()), 6),
            "gross_gap_mean": round(float(frame["gross_gap"].mean()), 6),
        })
        print(case_rows[-1], flush=True)

    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUT / "rebalance_detail.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame(case_rows)
    summary.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    causes = pd.DataFrame(cause_rows).value_counts().rename("count").reset_index()
    causes.to_csv(OUT / "causes.csv", index=False, encoding="utf-8-sig")
    write_json(
        {
            "cases": case_rows,
            "causes": causes.to_dict(orient="records"),
            "definition": {
                "signal_vs_backtest_target_l1": "L1 distance between latest_signal target weights and the backtester's normalised target on the same close; must be 0",
                "l1_vs_drift_ideal": "L1 distance between the end-of-day book and the target drifted by open->close returns",
                "unexplained_names": "name mismatches not explained by suspension, price limit or one-lot quantization",
            },
        },
        OUT / "consistency.json",
    )
    print()
    print(summary.to_string(index=False))
    print()
    print(causes.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
