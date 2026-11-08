"""Executable advice cards: a target book explained, never a stock tip.

What this module refuses to be
------------------------------
It is **not** a recommendation engine.  There is no "看涨", no target price, no
hit rate, no probability of going up -- not because they are hard to compute but
because the backtest cannot support them and the repo's own rule is that no
number may appear without a command that generates it.  What it does produce is
the only thing a daily signal can honestly produce:

    a target basket + the difference from what you hold + concrete orders
    + why each name is in the basket + what can go wrong

Six elements, all mandatory (a card missing one is not a card):

1. instrument, current price *with its timestamp*, target weight, whole-lot
   share count, limit-price band;
2. why it is in the basket: low-vol percentile, trend state against
   ``trend_window``, sector rank, trading days since the last rebalance;
3. **uncertainty on the same screen as the return** -- phase-median Sharpe, worst
   phase, detection power, bootstrap interval: never +95.87% alone;
4. risk: book weight at a 12% vol target, single-name cap, sector cap, T+1
   unsellable shares;
5. data health: latest cached session, coverage, names suspended or missing;
6. a disclaimer that cannot be collapsed.

Hard gate
---------
If the data behind the signal is not the previous trading session, or coverage is
below 95%, or the uncertainty artifacts are missing, the card carries **no orders
at all** and only the alerts.  Advice built on stale data is worse than no advice,
and a return quoted without its uncertainty is the one failure mode that would
make this whole system dishonest.

The gate is deliberately paranoid in one direction only: it may block a usable
day, it may never pass a stale one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from .brief import data_health, risk_snapshot
from .config import BacktestConfig
from .data import infer_price_limit, panel_to_frame
from .paper import Ledger, Position
from .signals import DailySignal, latest_signal

DISCLAIMER = (
    "历史回测结果，不构成投资建议。本卡由日频信号生成，不是推荐、不是看涨判断，"
    "也不预测涨跌；所有数字都能在 reports/ 下追溯到生成命令。"
    "上钱之前请先读 DECISION.md 与 docs/选型报告.md 里的不确定性一节。"
)

MIN_COVERAGE = 0.95
DEFAULT_BAND_BPS = 50.0


# --------------------------------------------------------------------------- #
# uncertainty: read from artifacts, never typed by hand
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UncertaintyBlock:
    """The numbers that must sit next to the return, and where they came from.

    Every field is read from a generated artifact.  A hand-copied +95.87% would go
    stale within a week (RUNBOOK §7 rule 3), so the card refuses to render at all
    if the artifacts are missing.
    """

    total_return: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    phase_median_sharpe: float | None = None
    phase_worst_return: float | None = None
    power: float | None = None
    power_years: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    sources: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing

    def headline(self) -> str:
        """The deployment line.  Never rendered without :meth:`caveats`."""
        if self.total_return is None or self.sharpe is None:
            return "部署口径收益/Sharpe：**未生成**"
        return "部署口径（12%% 目标波动，20 相位混合）：**%+.2f%% / Sharpe %.3f / 最大回撤 %.2f%%**" % (
            self.total_return * 100,
            self.sharpe,
            (self.max_drawdown or 0.0) * 100,
        )

    def caveats(self) -> list[str]:
        """What stops that headline from meaning what it looks like."""
        lines: list[str] = []
        if self.phase_median_sharpe is not None:
            lines.append(
                "同一窗口的**相位中位 Sharpe 只有 %.3f**、最差相位 %+.2f%%——把 %.3f 当期望值是错的"
                % (
                    self.phase_median_sharpe,
                    (self.phase_worst_return or 0.0) * 100,
                    self.sharpe or 0.0,
                )
            )
        if self.power is not None and self.power_years is not None:
            lines.append(
                "%.1f 年的记录对真 Sharpe 0.90 只有 **%.0f%% 检出力**——「有效」不等于「已被证明」"
                % (self.power_years, self.power * 100)
            )
        if self.ci_low is not None and self.ci_high is not None:
            lines.append(
                "自助法 95%% 区间 **[%.3f, %.3f]**：区间跨过 0 的下界说明这个 Sharpe 可能只是运气"
                % (self.ci_low, self.ci_high)
            )
        return lines

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "phase_median_sharpe": self.phase_median_sharpe,
            "phase_worst_return": self.phase_worst_return,
            "power": self.power,
            "power_years": self.power_years,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "complete": self.complete,
            "missing": list(self.missing),
            "sources": dict(self.sources),
        }


def load_uncertainty(reports_dir: str | Path = Path("reports")) -> UncertaintyBlock:
    """Collect the four uncertainty numbers from the reports they live in."""
    reports = Path(reports_dir)
    missing: list[str] = []
    sources: dict[str, str] = {}
    values: dict[str, float] = {}

    blend = reports / "year_table_blend" / "summary.json"
    if blend.exists():
        try:
            payload = json.loads(blend.read_text(encoding="utf-8"))
            overall = payload["overall"]["blend@12%"]
            values["total_return"] = float(overall["return"])
            values["sharpe"] = float(overall["sharpe"])
            values["max_drawdown"] = float(overall["max_drawdown"])
            sources["deployment"] = str(blend)
        except (KeyError, ValueError, TypeError):
            missing.append("year_table_blend/summary.json 缺 overall.blend@12%")
    else:
        missing.append(str(blend))

    phases = reports / "phase_recheck" / "phases.csv"
    if phases.exists():
        try:
            frame = pd.read_csv(phases)
            window = frame[frame["window"] == "2024-2026"]
            if "mode" in window and (window["mode"] == "year").any():
                window = window[window["mode"] == "year"]
            values["phase_median_sharpe"] = float(window["sharpe"].median())
            values["phase_worst_return"] = float(window["total_return"].min())
            sources["phase_spread"] = str(phases)
        except (KeyError, ValueError, pd.errors.ParserError):
            missing.append("phase_recheck/phases.csv 无法解析")
    else:
        missing.append(str(phases))

    power = reports / "power_recheck" / "summary.json"
    if power.exists():
        try:
            payload = json.loads(power.read_text(encoding="utf-8"))
            recorded = payload["recorded"]
            values["power_years"] = float(recorded["years"])
            values["ci_low"] = float(recorded["sharpe_low"])
            values["ci_high"] = float(recorded["sharpe_high"])
            # Power for the true Sharpe equal to the recorded one, at the recorded
            # horizon: pick the nearest grid cell instead of interpolating, so the
            # number quoted is one the study actually computed.
            grid = [row for row in payload.get("power", []) if abs(float(row["sharpe"]) - 0.90) < 1e-9]
            if grid:
                nearest = min(grid, key=lambda row: abs(float(row["years"]) - values["power_years"]))
                values["power"] = float(nearest["power"])
                sources["power"] = f"{power} (sharpe=0.90, years={nearest['years']})"
            else:
                missing.append("power_recheck/summary.json 没有 Sharpe=0.90 的格子")
            sources["bootstrap"] = str(power)
        except (KeyError, ValueError, TypeError):
            missing.append("power_recheck/summary.json 缺 recorded/power")
    else:
        missing.append(str(power))

    return UncertaintyBlock(
        total_return=values.get("total_return"),
        sharpe=values.get("sharpe"),
        max_drawdown=values.get("max_drawdown"),
        phase_median_sharpe=values.get("phase_median_sharpe"),
        phase_worst_return=values.get("phase_worst_return"),
        power=values.get("power"),
        power_years=values.get("power_years"),
        ci_low=values.get("ci_low"),
        ci_high=values.get("ci_high"),
        sources=sources,
        missing=missing,
    )


# --------------------------------------------------------------------------- #
# one line of advice
# --------------------------------------------------------------------------- #
@dataclass
class AdviceItem:
    symbol: str
    name: str
    sector: str
    price: float
    price_time: str
    price_kind: str            # "live" | "close" | "unavailable"
    weight: float
    shares: int
    notional: float
    lot_size: int
    limit_low: float
    limit_high: float
    limit_basis: str
    score: float | None
    score_pct: float | None
    trend_gap: float | None
    trend_window: int | None
    sector_rank: str | None
    reasons: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "name": self.name, "sector": self.sector,
            "price": self.price, "price_time": self.price_time, "price_kind": self.price_kind,
            "weight": self.weight, "shares": self.shares, "notional": self.notional,
            "lot_size": self.lot_size, "limit_low": self.limit_low, "limit_high": self.limit_high,
            "limit_basis": self.limit_basis,
            "score": self.score, "score_pct": self.score_pct, "trend_gap": self.trend_gap,
            "trend_window": self.trend_window, "sector_rank": self.sector_rank,
            "reasons": list(self.reasons), "risks": list(self.risks), "flags": list(self.flags),
        }

    def summary_line(self) -> str:
        price = "—" if self.price <= 0 else "%.2f" % self.price
        limits = "%.2f - %.2f" % (self.limit_low, self.limit_high) if self.limit_high > 0 else "—"
        return "| %s | %s | %s %s | %d 股 | %.1f%% | %s | %s |" % (
            self.symbol, self.name, price, "（%s）" % self.price_time, self.shares,
            self.weight * 100, limits, self.sector,
        )


# --------------------------------------------------------------------------- #
# the card
# --------------------------------------------------------------------------- #
@dataclass
class AdviceCard:
    as_of: str
    generated_at: str
    strategy: str
    equity: float
    data: dict
    uncertainty: UncertaintyBlock
    risk: dict
    items: list[AdviceItem] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked: bool = False
    block_reasons: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER
    last_trading_day: str | None = None
    bars_since_rebalance: int | None = None
    last_rebalance: str | None = None

    # ---------- reporting --------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "generated_at": self.generated_at,
            "strategy": self.strategy,
            "equity": self.equity,
            "last_trading_day": self.last_trading_day,
            "last_rebalance": self.last_rebalance,
            "bars_since_rebalance": self.bars_since_rebalance,
            "blocked": bool(self.blocked),
            "block_reasons": list(self.block_reasons),
            "data": self.data,
            "uncertainty": self.uncertainty.to_dict(),
            "risk": self.risk,
            "items": [item.to_dict() for item in self.items],
            "warnings": list(self.warnings),
            "disclaimer": self.disclaimer,
        }

    def markdown(self) -> str:
        lines: list[str] = []
        head = "## 今日建议卡（%s 收盘信号 → 下一交易日开盘执行）" % self.as_of
        lines.append(head)
        lines.append("")
        if self.blocked:
            lines.append("### ⛔ 本日不出建议")
            lines.append("")
            for reason in self.block_reasons:
                lines.append("- %s" % reason)
            lines.append("")
            lines.append("数据不新鲜时给出「建议」比不给更危险：本卡只报警，不排单。")
            lines.append("")
            lines.append("**%s**" % self.disclaimer)
            return "\n".join(lines)

        # 1. the numbers, return and uncertainty on the same screen
        lines.append("### 1. 这台机器值多少（收益与不确定性同屏）")
        lines.append("")
        lines.append("- " + self.uncertainty.headline())
        for caveat in self.uncertainty.caveats():
            lines.append("- " + caveat)
        if self.uncertainty.sources:
            lines.append("- 数据来源：" + "；".join(
                "%s=`%s`" % (key, value) for key, value in sorted(self.uncertainty.sources.items())
            ))
        lines.append("")

        # 2. data health
        data = self.data
        lines.append("### 2. 数据健康")
        lines.append("")
        lines.append(
            "- 缓存最新交易日 **%s**（上一交易日 %s）；在册 %s 只，当日有 bar %s 只，覆盖率 **%.1f%%**"
            % (
                data.get("as_of"), self.last_trading_day, data.get("live_symbols"),
                data.get("on_latest_bar"), float(data.get("coverage") or 0.0) * 100,
            )
        )
        if data.get("stale_examples"):
            lines.append("- 落后标的：" + "，".join(
                "%s(%s)" % (row["symbol"], row["last_bar"]) for row in data["stale_examples"][:5]
            ))
        if data.get("retired_symbols"):
            lines.append("- 另有 %s 只已退市/停牌超过 90 天，不计入覆盖率分母" % data["retired_symbols"])
        lines.append("")

        # 3. orders
        lines.append("### 3. 目标篮与订单（现价带时间戳、整手、限价区间）")
        lines.append("")
        if not self.items:
            lines.append("本日目标为空（100% 现金）：策略没有合格标的，或趋势过滤把整段横截面关掉了。")
        else:
            lines.append("| 标的 | 名称 | 现价 | 建议股数 | 目标权重 | 限价区间 | 行业 |")
            lines.append("|---|---|---|---:|---:|---|---|")
            for item in self.items:
                lines.append(item.summary_line())
        lines.append("")
        for warning in self.warnings:
            lines.append("- ⚠ %s" % warning)
        if self.warnings:
            lines.append("")

        # 4. why + risk per name
        lines.append("### 4. 为什么在篮子里 / 风险是什么（逐只）")
        lines.append("")
        for item in self.items:
            lines.append("**%s %s**（%s）" % (item.symbol, item.name or "—", item.sector))
            for reason in item.reasons:
                lines.append("- 为什么：" + reason)
            for risk in item.risks:
                lines.append("- 风险：" + risk)
            if item.flags:
                lines.append("- 标记：%s" % "，".join(item.flags))
            lines.append("")

        # 5. book-level risk
        risk = self.risk
        lines.append("### 5. 组合层风险")
        lines.append("")
        lines.append(
            "- 目标波动 12%% 下的建议总仓位 **%.1f%%**；单票上限 %.0f%%、行业上限 2 只"
            % (
                float(risk.get("gross_exposure") or 0.0) * 100,
                float(risk.get("max_position_weight") or 0.0) * 100,
            )
        )
        if risk.get("ex_ante_vol") is not None:
            lines.append(
                "- 事前波动 **%.1f%%**（窗口 %s 日）、beta %.2f、有效持仓 %s 只"
                % (
                    float(risk["ex_ante_vol"]) * 100, risk.get("window", "-"),
                    float(risk.get("beta") or 0.0), risk.get("positions", "-"),
                )
            )
        for line in risk.get("notes", []):
            lines.append("- " + line)
        lines.append("")

        # 6. disclaimer, never collapsed
        lines.append("### 6. 免责")
        lines.append("")
        lines.append("**%s**" % self.disclaimer)
        return "\n".join(lines)

    def html(self, *, title: str = "今日建议卡") -> str:
        """Self-contained page: the disclaimer cannot be folded away."""
        def esc(value) -> str:
            return (
                str(value)
                .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            )

        parts: list[str] = [
            "<h1>%s · %s</h1>" % (esc(title), esc(self.as_of)),
            '<p class="meta">生成于 %s ｜ 策略 %s ｜ 账户权益 %s ｜ 执行时点：下一交易日开盘</p>'
            % (esc(self.generated_at), esc(self.strategy), "{:,.0f}".format(self.equity)),
        ]
        if self.blocked:
            parts.append('<section class="blocked"><h2>⛔ 本日不出建议</h2><ul>')
            for reason in self.block_reasons:
                parts.append("<li>%s</li>" % esc(reason))
            parts.append("</ul><p>数据不新鲜时给出「建议」比不给更危险：本卡只报警，不排单。</p></section>")
            parts.append('<section class="disclaimer"><strong>%s</strong></section>' % esc(self.disclaimer))
            return self._page("".join(parts))

        parts.append('<section class="numbers"><h2>这台机器值多少</h2>')
        parts.append("<p class='headline'>%s</p>" % esc(self.uncertainty.headline()))
        parts.append("<ul class='caveats'>")
        for caveat in self.uncertainty.caveats():
            parts.append("<li>%s</li>" % esc(caveat))
        parts.append("</ul></section>")

        data = self.data
        parts.append('<section class="health"><h2>数据健康</h2><p>')
        parts.append(
            "缓存最新交易日 <b>%s</b>（上一交易日 %s）｜覆盖率 <b>%.1f%%</b>｜当日有 bar %s/%s 只"
            % (
                esc(data.get("as_of")), esc(self.last_trading_day),
                float(data.get("coverage") or 0.0) * 100,
                esc(data.get("on_latest_bar")), esc(data.get("live_symbols")),
            )
        )
        parts.append("</p></section>")

        parts.append('<section class="orders"><h2>目标篮与订单</h2>')
        if self.items:
            parts.append("<table><thead><tr><th>标的</th><th>名称</th><th>现价</th>"
                         "<th>股数</th><th>权重</th><th>限价区间</th><th>行业</th></tr></thead><tbody>")
            for item in self.items:
                parts.append(
                    "<tr><td>%s</td><td>%s</td><td>%s<span class='dim'> %s</span></td>"
                    "<td class='num'>%d</td><td class='num'>%.2f%%</td><td class='num'>%.2f–%.2f</td>"
                    "<td>%s</td></tr>"
                    % (
                        esc(item.symbol), esc(item.name),
                        "—" if item.price <= 0 else "%.2f" % item.price, esc(item.price_time),
                        item.shares, item.weight * 100, item.limit_low, item.limit_high,
                        esc(item.sector),
                    )
                )
            parts.append("</tbody></table>")
        else:
            parts.append("<p>本日目标为空（100% 现金）。</p>")
        parts.append("</section>")

        parts.append('<section class="why"><h2>为什么在篮子里 / 风险</h2>')
        for item in self.items:
            parts.append("<details open><summary>%s %s</summary><ul>" % (esc(item.symbol), esc(item.name)))
            for reason in item.reasons:
                parts.append("<li>为什么：%s</li>" % esc(reason))
            for risk in item.risks:
                parts.append("<li>风险：%s</li>" % esc(risk))
            parts.append("</ul></details>")
        parts.append("</section>")

        risk = self.risk
        parts.append('<section class="risk"><h2>组合层风险</h2><p>')
        parts.append(
            "建议总仓位 %.1f%%｜单票上限 %.0f%%｜行业上限 2 只"
            % (float(risk.get("gross_exposure") or 0.0) * 100,
               float(risk.get("max_position_weight") or 0.0) * 100)
        )
        if risk.get("ex_ante_vol") is not None:
            parts.append("｜事前波动 %.1f%%" % (float(risk["ex_ante_vol"]) * 100))
        parts.append("</p></section>")

        parts.append('<section class="disclaimer"><strong>%s</strong></section>' % esc(self.disclaimer))
        return self._page("".join(parts))

    @staticmethod
    def _page(body: str) -> str:
        return """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>今日建议卡</title>
<style>
body{font:15px/1.65 -apple-system,"Segoe UI",system-ui,sans-serif;color:#1b1f24;background:#f6f7f9;margin:0;padding:24px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:0 0 10px}
.meta{color:#5b6672;margin:0 0 18px}
section{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:16px 18px;margin-bottom:14px}
.headline{font-size:17px;margin:0 0 8px}
.caveats{margin:0;padding-left:20px}.caveats li{margin:4px 0;color:#8a4b00}
table{width:100%;border-collapse:collapse}th,td{padding:6px 8px;border-bottom:1px solid #eef0f3;text-align:left}
th{color:#5b6672;font-weight:600;font-size:13px}td.num{text-align:right;font-variant-numeric:tabular-nums}
.dim{color:#8b949e;font-size:12px}
.blocked{border-color:#f0b4b4;background:#fff6f6}
.disclaimer{border-color:#d9c27a;background:#fffdf2;font-size:13px}
details{margin:6px 0}
</style></head><body>""" + body + "</body></html>"


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def _name_map() -> dict[str, str]:
    """Readable names from the offline universe list (no network, no quotes needed)."""
    try:
        from .data import pit_universe

        return {str(key): str(value) for key, value in pit_universe().items()}
    except Exception:  # pragma: no cover - a checkout without the manifest
        return {}


def _limit_band(price: float, symbol: str, band_bps: float, quote=None) -> tuple[float, float, str]:
    """Limit-price range around the executable price, clamped to the daily band."""
    pct = infer_price_limit(symbol)
    up = round(price * (1 + pct), 2)
    down = round(price * (1 - pct), 2)
    basis = "按昨收推算的涨跌停（%.0f%%）" % (pct * 100)
    if quote is not None and quote.limit.up > 0:
        up, down = quote.limit.up, quote.limit.down
        basis = "交易所公布的涨跌停价"
    band = band_bps / 10000.0
    high = min(up, round(price * (1 + band), 2))
    low = max(down, round(price * (1 - band), 2))
    if high >= up:
        basis += "；上限已贴住涨停"
    return round(low, 2), round(high, 2), basis


def _t_plus_one_note(ledger: Ledger | None, symbol: str, as_of: pd.Timestamp) -> str:
    if ledger is None:
        return "T+1：未接账本，今日买入数量未知（卖出前请自行确认可卖）"
    position: Position = ledger.positions.get(symbol, Position())
    if position.shares <= 0:
        return "T+1：当前未持有，今日买入后明日才可卖"
    bought_today = 0
    for trade in ledger.trades:
        if str(trade.get("symbol")) == symbol and str(trade.get("side")) == "buy" and str(trade.get("date")) == str(as_of.date()):
            bought_today += int(trade.get("shares", 0))
    sellable = max(0, position.shares - bought_today)
    if bought_today:
        return "T+1：持有 %d 股，今日买入 %d 股不可卖，**可卖 %d 股**" % (
            position.shares, bought_today, sellable
        )
    return "T+1：持有 %d 股，全部可卖" % position.shares


def build_card(
    panel: dict[str, pd.DataFrame],
    *,
    strategy: str = "low_vol_core",
    params: dict | None = None,
    equity: float = 1_000_000.0,
    quotes=None,
    ledger: Ledger | None = None,
    today: pd.Timestamp | str | None = None,
    benchmark: pd.Series | None = None,
    band_bps: float = DEFAULT_BAND_BPS,
    reports_dir: str | Path = Path("reports"),
    min_coverage: float = MIN_COVERAGE,
    last_trading_day: str | None = None,
) -> AdviceCard:
    """Assemble one advice card, or refuse to assemble one.

    ``quotes`` is an optional :class:`quant_system.realtime.Board`; passing it
    turns the price column into a live price with a timestamp and lets the card
    clamp limits to the real band.  Without it the card uses the signal close and
    says so on every row -- it never silently presents a close as a live price.
    """
    moment = pd.Timestamp(today) if today is not None else pd.Timestamp.now().normalize()
    health = data_health(panel, today=moment)
    uncertainty = load_uncertainty(reports_dir)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    card = AdviceCard(
        as_of=health["as_of"],
        generated_at=generated_at,
        strategy=strategy,
        equity=float(equity),
        data=health,
        uncertainty=uncertainty,
        risk={},
        last_trading_day=last_trading_day,
    )

    # ---- the gate, before any order is computed -------------------------
    if last_trading_day is not None and health["as_of"] != last_trading_day:
        card.blocked = True
        card.block_reasons.append(
            "缓存最新交易日 %s ≠ 上一交易日 %s：现在给出的目标篮会晚一个交易日执行"
            % (health["as_of"], last_trading_day)
        )
    if float(health.get("coverage") or 0.0) < min_coverage:
        card.blocked = True
        card.block_reasons.append(
            "覆盖率 %.1f%% < %.0f%%：横截面不完整，选取会偏向「有数据的那些名字」"
            % (float(health["coverage"]) * 100, min_coverage * 100)
        )
    if not uncertainty.complete:
        card.blocked = True
        card.block_reasons.append(
            "不确定性数字未生成（%s）：只报收益不报不确定性的卡片一律不出"
            % "，".join(uncertainty.missing[:3])
        )
    if card.blocked:
        return card

    # ---- the book -------------------------------------------------------
    signal: DailySignal = latest_signal(
        panel,
        strategy,
        params=params,
        benchmark=benchmark,
        equity=float(equity),
        today=moment,
    )
    card.as_of = str(signal.as_of.date())
    card.last_rebalance = str(signal.last_rebalance.date()) if signal.last_rebalance is not None else None
    card.bars_since_rebalance = signal.bars_since_rebalance
    if last_trading_day is not None and signal.as_of != pd.Timestamp(last_trading_day):
        card.blocked = True
        card.block_reasons.append(
            "信号基准日 %s ≠ 上一交易日 %s：不出建议" % (signal.as_of.date(), last_trading_day)
        )
        return card
    if signal.rows.empty:
        card.warnings.extend(signal.warnings or ["信号给出 100% 现金"])
    else:
        card.warnings.extend(signal.warnings or [])

    weights = signal.target_weights
    risk = risk_snapshot(weights, panel, benchmark)
    gross = float(weights.sum())
    cfg = BacktestConfig()
    card.risk = {
        "gross_exposure": gross,
        "max_position_weight": cfg.max_position_weight,
        "max_per_sector": 2,
        "ex_ante_vol": risk.get("ex_ante_vol"),
        "beta": risk.get("beta"),
        "beta_se": risk.get("beta_se"),
        "positions": risk.get("positions"),
        "window": risk.get("window"),
        "top3_weight": risk.get("top3_weight"),
        "notes": [],
    }

    names = _name_map()
    missing_quotes: list[str] = []
    sectors_seen: dict[str, int] = {}
    for row in signal.rows.itertuples(index=False):
        symbol = row.symbol
        sector = row.sector
        sectors_seen[sector] = sectors_seen.get(sector, 0) + 1
        quote = None
        if quotes is not None:
            quote = quotes.quotes.get(symbol)
        price = float(row.price or 0.0)
        price_kind = "close"
        price_time = "%s 收盘" % signal.as_of.date()
        if quote is not None and quote.price > 0:
            price = quote.price
            price_kind = "live" if not quote.stale else "close"
            price_time = "%s" % quote.quote_time.strftime("%m-%d %H:%M:%S")
        elif quote is None:
            missing_quotes.append(symbol)

        low, high, basis = _limit_band(price, symbol, band_bps, quote)
        pct = float(row.score_pct) if row.score_pct is not None and pd.notna(row.score_pct) else None
        gap = float(row.trend_gap) if row.trend_gap is not None and pd.notna(row.trend_gap) else None
        window = int(row.trend_window) if row.trend_window is not None and pd.notna(row.trend_window) else None

        reasons: list[str] = []
        if pct is not None:
            reasons.append(
                "低波分位 %.0f%%（%s 只合格横截面里越靠前越平静）"
                % (pct * 100, signal.pool_size if signal.pool_size else "—")
            )
        if gap is not None and window:
            reasons.append(
                "站上 %d 日均线 %+.1f%%（跌破这条线会被趋势过滤剔除）" % (window, gap * 100)
            )
        if row.sector_rank:
            reasons.append("行业内第 %s（每行业最多 %d 只）" % (row.sector_rank, 2))
        if card.bars_since_rebalance is not None and card.last_rebalance:
            reasons.append(
                "距上次调仓 %d 个交易日（上次 %s；节奏为每 %s 个交易日）"
                % (card.bars_since_rebalance, card.last_rebalance, signal.params.get("rebalance_days", 20))
            )

        risks: list[str] = [
            "目标波动 12%% 下这只占 %.1f%%（%s 元）" % (float(row.weight) * 100, "{:,.0f}".format(float(row.amount or 0))),
            "单票上限 %.0f%%：目标权重已按上限截断" % (cfg.max_position_weight * 100),
            "行业「%s」本次持有 %d 只（上限 2 只）" % (sector, sectors_seen[sector]),
            _t_plus_one_note(ledger, symbol, signal.as_of),
        ]
        flags: list[str] = []
        if row.flags:
            flags.extend(str(row.flags).split(","))
        if quote is not None and quote.stale:
            flags.append("行情滞后")
            risks.append("实时价滞后（%s），下单前请重新取价" % quote.message)
        if quote is not None and quote.suspect:
            flags.append("双源不一致")
            risks.append("两个数据源的价差超过 0.5%%，今日不要按这个价挂单")
        if quote is not None and quote.status != "normal":
            flags.append(quote.status)

        card.items.append(
            AdviceItem(
                symbol=symbol,
                name=(quote.name if quote is not None and quote.name else names.get(symbol, "")),
                sector=sector,
                price=price,
                price_time=price_time,
                price_kind=price_kind,
                weight=float(row.weight),
                shares=int(row.shares),
                notional=float(row.amount or 0.0),
                lot_size=int(signal.lot_size),
                limit_low=low,
                limit_high=high,
                limit_basis=basis,
                score=float(row.score) if row.score is not None and pd.notna(row.score) else None,
                score_pct=pct,
                trend_gap=gap,
                trend_window=window,
                sector_rank=row.sector_rank,
                reasons=reasons,
                risks=risks,
                flags=flags,
            )
        )

    if missing_quotes and quotes is not None:
        card.warnings.append(
            "%d 只标的本轮没有实时报价（%s），限价按收盘价计算：收盘价不是现价，挂单前请重新取价"
            % (len(missing_quotes), "、".join(missing_quotes[:4]))
        )

    if len(card.items) and gross > 0:
        card.risk["notes"].append(
            "混合口径每天有 1/%d 的仓位按相位轮动；单相位口径下才是「每 20 个交易日一次」"
            % 20
        )
    if quotes is not None and quotes.degraded:
        card.warnings.append("行情已降级：%s" % "；".join(quotes.errors or ["数据源不可用"]))
    return card


def card_from_signal_json(
    signal_json: str | Path,
    *,
    panel: dict[str, pd.DataFrame] | None = None,
    **kwargs,
) -> AdviceCard:
    """Rebuild a card from an archived ``signal.json`` (the UI's fast path).

    The desk regenerates the *signal* once a day (~11 s on the PIT panel) and then
    renders the card from the artifact on every refresh (~10 ms).  Anything the
    artifact lacks is reported rather than invented.
    """
    payload = json.loads(Path(signal_json).read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload.get("orders") or [])
    if rows.empty:
        raise ValueError("signal.json 里没有 orders")
    uncertainty = load_uncertainty(kwargs.pop("reports_dir", Path("reports")))
    health = kwargs.pop("data_health", None) or {}
    card = AdviceCard(
        as_of=payload.get("as_of", ""),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        strategy=payload.get("strategy", ""),
        equity=float(payload.get("sizing_equity") or 0.0),
        data=health,
        uncertainty=uncertainty,
        risk={},
        last_trading_day=kwargs.pop("last_trading_day", None),
        last_rebalance=payload.get("last_rebalance"),
        bars_since_rebalance=payload.get("bars_since_rebalance"),
    )
    if card.last_trading_day and card.as_of != card.last_trading_day:
        card.blocked = True
        card.block_reasons.append(
            "归档信号基准日 %s ≠ 上一交易日 %s" % (card.as_of, card.last_trading_day)
        )
    if not uncertainty.complete:
        card.blocked = True
        card.block_reasons.append("不确定性数字未生成：" + "，".join(uncertainty.missing[:3]))
    if card.blocked:
        return card
    for row in rows.itertuples(index=False):
        price = float(getattr(row, "price", 0.0) or 0.0)
        low, high, basis = _limit_band(price, row.symbol, kwargs.pop("band_bps", DEFAULT_BAND_BPS))
        card.items.append(
            AdviceItem(
                symbol=row.symbol,
                name="",
                sector=row.sector,
                price=price,
                price_time="%s 收盘" % card.as_of,
                price_kind="close",
                weight=float(row.weight),
                shares=int(row.shares),
                notional=float(getattr(row, "amount", 0.0) or 0.0),
                lot_size=int(payload.get("lot_size") or 100),
                limit_low=low,
                limit_high=high,
                limit_basis=basis,
                score=getattr(row, "score", None),
                score_pct=getattr(row, "score_pct", None),
                trend_gap=getattr(row, "trend_gap", None),
                trend_window=getattr(row, "trend_window", None),
                sector_rank=getattr(row, "sector_rank", None),
                reasons=["见 signal.json 的 score/trend_gap 字段（本卡片走归档快路径，未重算解释）"],
                risks=[],
            )
        )
    return card


def write_card(card: AdviceCard, output: str | Path) -> Path:
    """Write ``card.md`` / ``card.json`` / ``card.html`` next to each other."""
    target = Path(output)
    target.mkdir(parents=True, exist_ok=True)
    (target / "card.md").write_text(card.markdown(), encoding="utf-8")
    (target / "card.json").write_text(
        json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (target / "card.html").write_text(card.html(), encoding="utf-8")
    return target
