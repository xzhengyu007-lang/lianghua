# 脚本清单

> 由 `scripts/build_script_inventory.py` 生成，**不要手改**；`tests/test_script_inventory.py` 会比对重新生成的结果。

| 项 | 数量 |
|---|---:|
| 脚本总数 | 116 |
| 在研究日志里出现过 | 74 |
| 有对应测试 | 45 |
| **两者都没有（孤儿）** | **40** |

## 孤儿脚本（未记录、未测试）

| 脚本 | 类别 | 行数 | 用途（取自 docstring） |
|---|---|---:|---|
| `build_report_index.py` | build | 132 | Build reports/INDEX.md: one line per report directory with its headline metrics. |
| `download_pit_raw.py` | download | 22 | Download the raw (unadjusted) bars needed to rebuild PIT series (RESEARCH_LOG 23.19). |
| `download_pit_raw_rest.py` | download | 21 | Resume the raw-bar download from data/tmp_raw_todo.json after a partial run (RESEARCH_LOG 23.19). |
| `run_adjustment_audit.py` | run | 71 | Compare qfq / raw / rebuilt series on one window: the additive-adjustment problem (23.19). |
| `run_adv_lookahead.py` | run | 63 | Does the ADV participation cap read volume the order is about to consume? (23.17) |
| `run_baseline.py` | run | 153 | Full-period baseline: report, dashboard, and factor diagnostics. |
| `run_bootstrap_ci.py` | run | 19 | Thin wrapper: block-bootstrap confidence intervals for walk-forward reports. |
| `run_breadth_robustness.py` | run | 418 | Breadth robustness: fixed top_n=6 vs the default top_n=8. |
| `run_buffer_rank.py` | run | 200 | Out-of-sample test of the rank buffer (the "hold longer" knob). |
| `run_capacity.py` | run | 132 | Cost and capacity sensitivity on the full-history broad universe. |
| `run_concentration.py` | run | 107 | Concentration study: how much of the headline comes from a few names? |
| `run_delisting_exit_price.py` | run | 76 | Where a delisted holding gets liquidated: that day close or that day open (23.18). |
| `run_diversification_audit.py` | run | 200 | Does the book actually diversify, or only look diversified? |
| `run_final.py` | run | 42 | Archived full-period baseline: report, dashboard and factor diagnostics (superseded, kept for history). |
| `run_heldout_width.py` | run | 98 | Fully held-out final-year check for portfolio width. |
| `run_impact.py` | run | 96 | How much does a nonlinear impact model cost the strategy? |
| `run_liquidity_factor.py` | run | 329 | Is the low-vol signal just a liquidity/size proxy? |
| `run_lot_quantization.py` | run | 50 | How much of the capacity conclusion is only 100-share lot rounding? (23.17) |
| `run_oos_concentration.py` | run | 155 | Out-of-sample check of portfolio breadth and sector concentration. |
| `run_oos_overlays.py` | run | 71 | Out-of-sample comparison of portfolio overlays (fixed parameters). |
| `run_oos_risk_cost.py` | run | 218 | Out-of-sample sensitivity of volatility targeting and trading costs. |
| `run_oos_sector_cap.py` | run | 89 | Out-of-sample selection experiment for the sector cap. |
| `run_oos_sector_cap_v2.py` | run | 119 | Does a sector cap help out of sample once the harmful regime switch is off? |
| `run_overlay_ablation.py` | run | 80 | Benchmark-relative robustness: is the edge real after removing the |
| `run_pit_breadth.py` | run | 249 | Point-in-time comparison of fixed top_n=6 vs top_n=8. |
| `run_pit_walkforward.py` | run | 127 | Walk-forward the point-in-time pool and a survivor-only control. |
| `run_random_control_pit.py` | run | 134 | Random-selection control on the point-in-time pool. |
| `run_robustness.py` | run | 207 | Robustness study for the default low-vol core configuration. |
| `run_sector_cap_negative.py` | run | 83 | Sector-cap alternatives, kept as a negative-result record. |
| `run_sector_neutral.py` | run | 126 | Sector-neutral control for the low-volatility core strategy. |
| `run_sector_relative_momentum.py` | run | 258 | Sector-relative low-vol: rank names *inside* their own sector. |
| `run_sector_weight_budget.py` | run | 251 | Sector *weight* budget instead of a sector *count* cap. |
| `run_sensitivity.py` | run | 116 | Portfolio-construction sensitivity: risk budget, sizing, and turnover caps. |
| `run_signal_consistency.py` | run | 182 | The live signal and the backtest must size the same basket identically (23.16). |
| `run_signals.py` | run | 57 | Generate daily signals for every strategy, plus a paper-account snapshot. |
| `run_stress.py` | run | 151 | Stress tests that attack the two most likely sources of fake alpha. |
| `run_stress_deleveraging.py` | run | 314 | Causal stress de-leveraging: cut exposure when the market is in a drawdown. |
| `run_survivorship.py` | run | 189 | How much does survivorship bias actually buy us? |
| `run_weighting.py` | run | 236 | Does the construction rule add anything after the basket is chosen? |
| `run_weighting_robustness.py` | run | 231 | Robustness of inverse-vol sizing for the 6/2 low-vol book. |

## 全部脚本

| 脚本 | 类别 | 记录 | 测试 | 用途 |
|---|---|:--:|:--:|---|
| `__init__.py` | other | ✓ | ✓ | (no docstring) |
| `audit_decision_claims.py` | audit | ✓ | ✓ | Every claim on the decision page must trace back to an artifact. |
| `audit_staleness.py` | audit | ✓ | ✓ | Which published results are now stale, and which citations point at nothing? |
| `build_chained_equity.py` | build | ✓ | — | Stitch two consecutive walk-forward equity curves into one continuous series. |
| `build_dashboard.py` | build | — | ✓ | Build a self-contained visual dashboard over the research artifacts. |
| `build_data_manifest.py` | build | ✓ | ✓ | A manifest of the cached data, so a fresh checkout can prove it has the same data. |
| `build_decision_memo.py` | build | ✓ | ✓ | One page that answers "should we put money on this?" without reading 93 sections. |
| `build_delisted_universe.py` | build | ✓ | — | Build a point-in-time universe from real delisted names. |
| `build_report_index.py` | build | — | — | Build reports/INDEX.md: one line per report directory with its headline metrics. |
| `build_research_toc.py` | build | — | ✓ | Keep a table of contents at the top of RESEARCH_LOG.md. |
| `build_script_inventory.py` | build | ✓ | ✓ | Inventory of every script: what it does, whether it is documented, whether it is tested. |
| `build_total_return_cache.py` | build | ✓ | — | Persist rebuilt total-return series next to the provider cache. |
| `check_artifact_fingerprints.py` | check | ✓ | ✓ | Fingerprint the upstream research artifacts so a silent change cannot hide. |
| `check_ledger_replay.py` | check | ✓ | — | Verify the paper ledger reconciles with the backtester. |
| `download_pit_raw.py` | download | — | — | Download the raw (unadjusted) bars needed to rebuild PIT series (RESEARCH_LOG 23.19). |
| `download_pit_raw_loop.py` | download | ✓ | — | Retry raw downloads for the PIT pool until every symbol has a cache. |
| `download_pit_raw_rest.py` | download | — | — | Resume the raw-bar download from data/tmp_raw_todo.json after a partial run (RESEARCH_LOG 23.19). |
| `probe_delisted.py` | other | ✓ | — | Probe the kline endpoint for many codes and classify by last bar date. |
| `rebuild_from_hfq.py` | other | ✓ | — | Rebuild a symbol's qfq cache from hfq rescaled to the latest raw close. |
| `refresh_critical.py` | other | ✓ | ✓ | Rebuild the reports the decision rests on, in one command. |
| `run_52w_overlay.py` | run | ✓ | ✓ | Proximity to the 52-week high as a portfolio overlay (RESEARCH_LOG 23.77). |
| `run_adjustment_audit.py` | run | — | — | Compare qfq / raw / rebuilt series on one window: the additive-adjustment problem (23.19). |
| `run_adjustment_rebuild.py` | run | ✓ | — | Rebuild multiplicative total-return series from raw + qfq, then re-run the baseline (23.19). |
| `run_adv_lookahead.py` | run | — | — | Does the ADV participation cap read volume the order is about to consume? (23.17) |
| `run_baseline.py` | run | — | — | Full-period baseline: report, dashboard, and factor diagnostics. |
| `run_bootstrap_ci.py` | run | — | — | Thin wrapper: block-bootstrap confidence intervals for walk-forward reports. |
| `run_breadth_ic.py` | run | ✓ | ✓ | Does the shrinking PIT cross-section explain the rising low-vol IC? |
| `run_breadth_oos.py` | run | ✓ | — | Out-of-sample test of the one timing signal that survived a t=-3.9. |
| `run_breadth_overlay.py` | run | ✓ | — | Backtest the breadth-timing overlay through the real engine. |
| `run_breadth_robustness.py` | run | — | — | Breadth robustness: fixed top_n=6 vs the default top_n=8. |
| `run_buffer_rank.py` | run | — | — | Out-of-sample test of the rank buffer (the "hold longer" knob). |
| `run_cadence_blend.py` | run | ✓ | ✓ | Rebalance cadence, measured on the blend instead of a single phase. |
| `run_capacity.py` | run | — | — | Cost and capacity sensitivity on the full-history broad universe. |
| `run_capacity_blend.py` | run | ✓ | — | Capacity under the configuration we would actually deploy: the phase blend. |
| `run_capacity_curve.py` | run | ✓ | ✓ | Capacity curve: how much capital before the book stops being buildable? |
| `run_concentration.py` | run | — | — | Concentration study: how much of the headline comes from a few names? |
| `run_cost_decomposition.py` | run | ✓ | — | What does each cost scenario actually charge per trade? |
| `run_cost_floor.py` | run | ✓ | ✓ | A structural floor on per-trade cost that no parameter choice can undercut. |
| `run_cost_panel.py` | run | ✓ | — | Cost sensitivity across the three windows: does the edge survive 30bp/side? |
| `run_cost_target_interaction.py` | run | ✓ | ✓ | Does the 12% risk budget survive a harsh cost model? |
| `run_cross_pool.py` | run | ✓ | ✓ | Would a second market actually buy us independent evidence? |
| `run_cross_pool_random.py` | run | ✓ | ✓ | Does the edge survive a narrower cross-section? Random sub-pool test. |
| `run_delisting_exit_price.py` | run | — | — | Where a delisted holding gets liquidated: that day close or that day open (23.18). |
| `run_diversification_audit.py` | run | — | — | Does the book actually diversify, or only look diversified? |
| `run_execution_timing.py` | run | ✓ | ✓ | How much of the edge survives being late? |
| `run_factor_cohorts.py` | run | ✓ | ✓ | Decompose every factor's IC into "still listed" versus "eventually delisted". |
| `run_factor_study.py` | run | ✓ | — | Factor study: does anything survive after low volatility is controlled for? |
| `run_filter_blend_ablation.py` | run | ✓ | ✓ | Split the edge between the trend filter and the score blend. |
| `run_filter_form.py` | run | ✓ | ✓ | Is the trend filter robust, or did it just happen to be set to SMA(120)? |
| `run_final.py` | run | — | — | Archived full-period baseline: report, dashboard and factor diagnostics (superseded, kept for history). |
| `run_hedge_panel.py` | run | ✓ | — | Beta-hedge panel across the three independent windows. |
| `run_hedge_study.py` | run | ✓ | — | Does hedging the residual market beta pay, after futures carry? |
| `run_heldout_width.py` | run | — | — | Fully held-out final-year check for portfolio width. |
| `run_holding_persistence.py` | run | ✓ | ✓ | Holding persistence: is the low-vol book a "core" or a monthly rotation? |
| `run_impact.py` | run | — | — | How much does a nonlinear impact model cost the strategy? |
| `run_liquidity_factor.py` | run | — | — | Is the low-vol signal just a liquidity/size proxy? |
| `run_lookahead_audit.py` | run | ✓ | ✓ | Look-ahead detector: scramble the future, the past must not move. |
| `run_lookahead_negative_control.py` | run | ✓ | — | Negative control: prove the look-ahead audit can actually fail. |
| `run_lot_quantization.py` | run | — | — | How much of the capacity conclusion is only 100-share lot rounding? (23.17) |
| `run_momentum_overlay.py` | run | ✓ | — | Does 12-1 momentum add anything on top of low volatility, in all 3 windows? |
| `run_naive_benchmark.py` | run | ✓ | ✓ | Head to head against the obvious alternative, not against randomness. |
| `run_oos_concentration.py` | run | — | — | Out-of-sample check of portfolio breadth and sector concentration. |
| `run_oos_overlays.py` | run | — | — | Out-of-sample comparison of portfolio overlays (fixed parameters). |
| `run_oos_risk_cost.py` | run | — | — | Out-of-sample sensitivity of volatility targeting and trading costs. |
| `run_oos_sector_cap.py` | run | — | — | Out-of-sample selection experiment for the sector cap. |
| `run_oos_sector_cap_v2.py` | run | — | — | Does a sector cap help out of sample once the harmful regime switch is off? |
| `run_overlay_ablation.py` | run | — | — | Benchmark-relative robustness: is the edge real after removing the |
| `run_param_ensemble.py` | run | ✓ | — | Parameter averaging: can we buy robustness without giving up return? |
| `run_param_panel.py` | run | ✓ | — | Parameter panel: does the published setting survive all three windows? |
| `run_phase_average.py` | run | ✓ | — | Does averaging rebalance phases remove calendar luck, or dilute the edge? |
| `run_phase_blend.py` | run | ✓ | ✓ | Does holding all 20 phases at once still remove the calendar lottery? |
| `run_phase_recheck.py` | run | ✓ | ✓ | Re-check the calendar-anchor claim under the current (rebuilt) data. |
| `run_phase_risk_comparison.py` | run | ✓ | ✓ | Does a lower risk budget also shrink the calendar (rebalance-phase) risk? |
| `run_pit_breadth.py` | run | — | — | Point-in-time comparison of fixed top_n=6 vs top_n=8. |
| `run_pit_walkforward.py` | run | — | — | Walk-forward the point-in-time pool and a survivor-only control. |
| `run_pnl_cohorts.py` | run | ✓ | ✓ | Where the money actually came from: realised P&L split by eventual fate. |
| `run_point_in_time.py` | run | ✓ | — | Extend the universe with real delisted names and compare against survivor-only. |
| `run_power_recheck.py` | run | ✓ | ✓ | Detection power for the Sharpe we can actually claim. |
| `run_power_study.py` | run | ✓ | ✓ | How much out-of-sample data does a Sharpe of 1.0 need to be provable? |
| `run_publish_check.py` | run | ✓ | ✓ | Self-check for the configuration we would actually deploy. |
| `run_quarter_consistency.py` | run | ✓ | ✓ | A sign test on quarters: far more powerful than a Sharpe interval, and weaker. |
| `run_random_control.py` | run | ✓ | ✓ | Does the ranking add anything over random selection with the same constraints? |
| `run_random_control_pit.py` | run | — | — | Random-selection control on the point-in-time pool. |
| `run_rank_overlap.py` | run | ✓ | ✓ | How often does the 0.8/0.2 blend actually change the basket, and does it help? |
| `run_rebalance_anchor.py` | run | ✓ | — | Rebalance-anchor sensitivity: how much of the headline is calendar luck? |
| `run_rebalance_calendar.py` | run | ✓ | — | Rebalance calendar: fixed 20-bar cadence vs calendar month-end. |
| `run_robustness.py` | run | — | — | Robustness study for the default low-vol core configuration. |
| `run_score_weights.py` | run | ✓ | — | Score weights: is the 20% trend term buying anything? |
| `run_sector_attribution.py` | run | ✓ | — | Sector attribution for the finished walk-forward windows (2022-23, 2024-26). |
| `run_sector_cap_negative.py` | run | — | — | Sector-cap alternatives, kept as a negative-result record. |
| `run_sector_neutral.py` | run | — | — | Sector-neutral control for the low-volatility core strategy. |
| `run_sector_relative_momentum.py` | run | — | — | Sector-relative low-vol: rank names *inside* their own sector. |
| `run_sector_weight_budget.py` | run | — | — | Sector *weight* budget instead of a sector *count* cap. |
| `run_selection_character.py` | run | ✓ | ✓ | What are the eight names, really? Selection character across periods. |
| `run_selection_slack.py` | run | ✓ | ✓ | Selection slack: how many names actually compete for the eight slots? |
| `run_sensitivity.py` | run | — | — | Portfolio-construction sensitivity: risk budget, sizing, and turnover caps. |
| `run_signal_consistency.py` | run | — | — | The live signal and the backtest must size the same basket identically (23.16). |
| `run_signal_horizon.py` | run | ✓ | ✓ | How long does the low-vol + trend score stay informative? |
| `run_signals.py` | run | — | — | Generate daily signals for every strategy, plus a paper-account snapshot. |
| `run_sizing_study.py` | run | ✓ | — | Volatility-target sizing study: same strategy, three risk budgets, two regimes. |
| `run_stress.py` | run | — | — | Stress tests that attack the two most likely sources of fake alpha. |
| `run_stress_deleveraging.py` | run | — | — | Causal stress de-leveraging: cut exposure when the market is in a drawdown. |
| `run_survivorship.py` | run | — | — | How much does survivorship bias actually buy us? |
| `run_timing_study.py` | run | ✓ | — | Can anything predict when the low-vol book will lose to the equal-weight pool? |
| `run_tradability.py` | run | ✓ | ✓ | Tradability: how often is a name we want (or hold) actually tradeable? |
| `run_trend_ablation.py` | run | ✓ | — | Trend-filter ablation around the low-vol core. |
| `run_verification.py` | run | ✓ | ✓ | One command that proves the system still works. |
| `run_vol_calibration.py` | run | ✓ | ✓ | Volatility-target calibration: why does a 12% target realise 13.4%? |
| `run_vol_target_panel.py` | run | ✓ | — | Volatility-target panel: published 18% vs a 12% risk cap. |
| `run_voltarget_blend.py` | run | ✓ | ✓ | Is the 12% volatility target still worth switching on, on the new baseline? |
| `run_weighting.py` | run | — | — | Does the construction rule add anything after the basket is chosen? |
| `run_weighting_robustness.py` | run | — | — | Robustness of inverse-vol sizing for the 6/2 low-vol book. |
| `run_window_scan.py` | run | ✓ | — | How much of a 2-3 year result is window luck? |
| `run_year_table.py` | run | ✓ | ✓ | Calendar-year breakdown of the published out-of-sample record. |
| `run_year_table_blend.py` | run | ✓ | ✓ | Calendar-year record of the deployable configuration, decomposed. |
| `serve_dashboard.py` | serve | ✓ | ✓ | Local server that makes the dashboard interactive. |
