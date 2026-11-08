"""Tiered release: L0 回测 / L1 模拟盘 / L2 半自动 / L3 全自动.

The gate is the only object allowed to answer "may a real order leave this
machine?".  Three independent conditions, all persisted to disk so a restart
cannot launder an authorisation:

1. **level** -- four tiers, default L1 (paper).  L3 requires a *daily* explicit
   authorisation: it expires at midnight, so a forgotten switch cannot trade on
   tomorrow's open.
2. **kill switch** -- one flag, set by a human (CLI/UI) or by the daily loss
   guard; nothing clears it except an explicit reset, and every set/reset is
   appended to the state file's history.
3. **halt reason** -- a free-text witness ("日亏 3.1% 触发熔断") so the operator
   sees *why* the system stopped, not just that it did.

This repository contains **no broker gateway**: there is no code here that can
send an order to an exchange.  That is deliberate (`docs/选型报告.md` §6 D3) and it
is also the strongest form of red line #1 -- you cannot accidentally send a real
order from a codebase that cannot send one.  ``can_submit`` therefore means
"may write a submit record", and the tests assert that level < L3 never even
constructs the submit path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import IntEnum
from pathlib import Path


class TradingLevel(IntEnum):
    L0_BACKTEST = 0
    L1_PAPER = 1
    L2_SEMI_AUTO = 2
    L3_FULL_AUTO = 3

    @property
    def label(self) -> str:
        return {
            TradingLevel.L0_BACKTEST: "L0 回测",
            TradingLevel.L1_PAPER: "L1 模拟盘",
            TradingLevel.L2_SEMI_AUTO: "L2 半自动（只生成券商可导入文件）",
            TradingLevel.L3_FULL_AUTO: "L3 全自动（默认关闭）",
        }[self]


@dataclass
class GateState:
    level: int = int(TradingLevel.L1_PAPER)
    enabled_until: str | None = None      # 逐日授权：只有这一天有效
    kill_switch: bool = False
    halt_reason: str = ""
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": int(self.level),
            "enabled_until": self.enabled_until,
            "kill_switch": bool(self.kill_switch),
            "halt_reason": self.halt_reason,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "GateState":
        return cls(
            level=int(payload.get("level", int(TradingLevel.L1_PAPER))),
            enabled_until=payload.get("enabled_until"),
            kill_switch=bool(payload.get("kill_switch", False)),
            halt_reason=payload.get("halt_reason", ""),
            history=list(payload.get("history") or []),
        )


class TradingGate:
    """Persisted trading permission.  Fail-closed on every ambiguous state.

    A truncated or unreadable state file means L1-with-kill-switch rather than
    "assume the operator meant L3" -- the only safe reading of a missing file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    # ---------- persistence ------------------------------------------------
    def _load(self) -> GateState:
        if not self.path.exists():
            # Fresh install: L1 paper, kill switch off.  There is nothing to protect
            # yet -- L1 writes no order file and this package has no gateway -- and a
            # halt on first run would teach the operator to ignore the badge.
            return GateState(level=int(TradingLevel.L1_PAPER), kill_switch=False)
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A state file that exists but cannot be read is ambiguous: fail closed.
            return GateState(kill_switch=True, halt_reason="状态文件损坏：默认停机，需人工 reset-kill")
        return GateState.from_dict(payload)

    def _save(self, note: str) -> None:
        self.state.history.append(
            {"at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "note": note,
             "level": int(self.state.level), "kill": self.state.kill_switch}
        )
        self.path.write_text(
            json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---------- level ------------------------------------------------------
    def set_level(self, level: TradingLevel | int, *, trade_date: date | None = None) -> None:
        target = TradingLevel(int(level))
        if target is TradingLevel.L3_FULL_AUTO:
            if trade_date is None:
                raise ValueError("L3 需要显式传入当日日期并逐日授权，拒绝无日期的开启")
            self.state.enabled_until = str(trade_date)
        elif target < TradingLevel.L3_FULL_AUTO:
            self.state.enabled_until = None
        self.state.level = int(target)
        if not self.state.kill_switch:
            self.state.halt_reason = ""
        self._save("set_level=%s" % target.name)

    def authorize_today(self, trade_date: date) -> None:
        """Light L3 for one session only."""
        if int(self.state.level) != int(TradingLevel.L3_FULL_AUTO):
            raise ValueError("先 set_level(L3) 再做逐日授权")
        self.state.enabled_until = str(trade_date)
        self._save("authorize_today=%s" % trade_date)

    # ---------- kill switch ------------------------------------------------
    def kill(self, reason: str) -> None:
        self.state.kill_switch = True
        self.state.halt_reason = reason or "人工停机"
        self._save("kill")
        # A kill switch also drops the level back to paper: restarting must be a
        # deliberate act, not a leftover L3.
        self.state.level = int(TradingLevel.L1_PAPER)
        self.state.enabled_until = None
        self._save("kill→L1")

    def reset_kill(self, *, reset_level: bool = False) -> None:
        self.state.kill_switch = False
        self.state.halt_reason = ""
        self._save("reset_kill")
        if reset_level:
            self.state.level = int(TradingLevel.L1_PAPER)
            self.state.enabled_until = None
            self._save("reset_level=L1")

    # ---------- questions --------------------------------------------------
    @property
    def level(self) -> TradingLevel:
        try:
            return TradingLevel(int(self.state.level))
        except ValueError:
            return TradingLevel.L1_PAPER

    def l3_authorized_for(self, trade_date: date | str) -> bool:
        return (
            self.level is TradingLevel.L3_FULL_AUTO
            and not self.state.kill_switch
            and self.state.enabled_until == str(trade_date)
        )

    def can_submit(self, trade_date: date | str) -> bool:
        """May this process write anything that could reach a broker today?

        L2 writes a file a human then confirms at the broker; L3 writes the
        submit record.  L0/L1 never do either, and a kill switch stops all of it.
        """
        if self.state.kill_switch:
            return False
        if self.level is TradingLevel.L3_FULL_AUTO:
            return self.state.enabled_until == str(trade_date)
        return self.level is TradingLevel.L2_SEMI_AUTO

    def describe(self, trade_date: date | str | None = None) -> dict:
        day = str(trade_date) if trade_date is not None else None
        return {
            "level": int(self.level),
            "level_label": self.level.label,
            "kill_switch": bool(self.state.kill_switch),
            "halt_reason": self.state.halt_reason,
            "enabled_until": self.state.enabled_until,
            "l3_authorized": bool(day and self.l3_authorized_for(day)),
            "can_submit": bool(day and self.can_submit(day)),
            "note": "本仓没有券商网关：L3 只写提交记录，不存在发单代码",
        }
