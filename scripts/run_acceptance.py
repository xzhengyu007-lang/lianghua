"""One command that answers "is this checkout still acceptable?".

The four-stage delivery has six promises that must hold *together* every time the
repo changes:

    pytest 全绿 / 发布自检三项 ok / 上游产物指纹没漂 / 引用审计 0 问题
    回测锚点逐位不变 / 交易台能起来并给出严格 JSON

Before this script they were six commands a human had to remember, and "I ran the
tests" was the only one that actually got run.  Now:

    python scripts/run_acceptance.py              # 6 checks, ~4 min
    python scripts/run_acceptance.py --with-sla   # + the network SLA probe (~20 s)

Exit code: 0 all green, 1 something failed.  Each check prints its own verdict, so
a failure names the check instead of a wall of pytest output.

The backtest anchor is compared against the previous accepted run recorded in
``reports/acceptance/last.json``; the *first* run just records it.  That is
deliberately not a hard-coded constant -- a constant would go stale the day the
strategy is retuned on purpose, and the point here is "did this change move the
number", not "is the number 0.9587".
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RECORD = ROOT / "reports" / "acceptance" / "last.json"


# --------------------------------------------------------------------------- #
# individual checks
# --------------------------------------------------------------------------- #
def _run(command: list[str], *, timeout: float = 900.0) -> tuple[int, str]:
    """Run a child process and keep the last few lines of its output.

    ``encoding="utf-8", errors="replace"`` is not cosmetic: with ``text=True`` alone
    Python decodes the child's output with the *locale* codec (GBK on this machine),
    and a single UTF-8 byte in pytest's output makes the reader thread raise
    UnicodeDecodeError -- which surfaced as "pytest ❌" for a run where every test
    passed.  Decode as UTF-8 and replace anything unreadable.
    """
    completed = subprocess.run(
        command, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )
    lines = [line for line in (completed.stdout or "").strip().splitlines() if line.strip()]
    return completed.returncode, " | ".join(lines[-3:])


def check_tests() -> tuple[bool, str]:
    code, tail = _run([sys.executable, "-m", "pytest", "-q"])
    return code == 0, tail or "pytest"


def check_publish() -> tuple[bool, str]:
    code, tail = _run([sys.executable, str(ROOT / "scripts" / "run_publish_check.py")])
    summary = ROOT / "reports" / "publish_check" / "summary.json"
    if code != 0 or not summary.exists():
        return False, tail or "run_publish_check failed"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return bool(payload.get("all_ok")), "all_ok=%s" % payload.get("all_ok")


def check_artifact_fingerprints() -> tuple[bool, str]:
    code, tail = _run([sys.executable, str(ROOT / "scripts" / "check_artifact_fingerprints.py")])
    return code == 0, tail


def check_staleness() -> tuple[bool, str]:
    """Citations must resolve; freshness is reported, not graded.

    ``citation_problems > 0`` means a published claim points at an artifact that is
    gone -- that is a correctness bug.  A stale *critical set* only means the engine
    changed after the reports were rendered; the fix is `refresh_critical.py` (25 min),
    not a red build, so it is surfaced in the detail instead of failing the run.
    """
    code, tail = _run([sys.executable, str(ROOT / "scripts" / "audit_staleness.py")])
    summary = ROOT / "reports" / "audit_staleness" / "summary.json"
    if code != 0 or not summary.exists():
        return False, tail or "audit_staleness failed"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    problems = payload.get("citation_problems") or []   # a list of offending citations
    critical = payload.get("critical") or []
    fresh = sum(1 for row in critical if row.get("fresh"))
    detail = "引用问题 %d｜关键集 %d/%d 新鲜" % (len(problems), fresh, len(critical))
    if critical and fresh < len(critical):
        detail += "（引擎改过 → 需要时跑 refresh_critical.py）"
    return len(problems) == 0, detail


def current_anchor() -> dict | None:
    summary = ROOT / "reports" / "publish_check" / "summary.json"
    if not summary.exists():
        return None
    payload = json.loads(summary.read_text(encoding="utf-8"))
    for check in payload.get("checks", []):
        if check.get("name") == "fingerprint":
            return check.get("actual")
    return None


def check_anchor(previous: dict | None) -> tuple[bool, str]:
    """The backtest number must not move unless somebody says it should."""
    anchor = current_anchor()
    if anchor is None:
        return False, "没有指纹（先跑 run_publish_check）"
    if previous is None:
        return True, "首次验收，记录锚点 total_return=%.10f" % anchor["total_return"]
    same = (
        abs(float(previous["total_return"]) - float(anchor["total_return"])) < 1e-12
        and abs(float(previous["sharpe"]) - float(anchor["sharpe"])) < 1e-12
    )
    detail = "total_return %.10f / sharpe %.10f" % (anchor["total_return"], anchor["sharpe"])
    return same, ("锚点未变｜" + detail) if same else ("锚点已变｜" + detail)


ANCHOR_HISTORY = ROOT / "reports" / "acceptance" / "anchor_history.jsonl"


def _git_state() -> dict:
    """Where the code was when a human said the number may move."""
    state = {"head": None, "dirty": None}
    for key, argv in (("head", ["git", "rev-parse", "--short", "HEAD"]),
                      ("dirty", ["git", "status", "--porcelain"])):
        try:
            out = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=10)
            state[key] = out.stdout.strip() if key == "dirty" else (out.stdout.strip() or None)
        except (OSError, subprocess.SubprocessError):  # no git, no history detail
            state[key] = None
    if state["dirty"] is not None:
        state["dirty"] = bool(state["dirty"])
    return state


def record_anchor_change(previous: dict, current: dict, reason: str) -> dict:
    """Append one line to the anchor history.  Append-only: never rewritten."""
    import os

    entry = {
        "accepted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "previous": previous,
        "current": current,
    }
    entry.update(_git_state())
    ANCHOR_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with ANCHOR_HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return entry


def check_desk() -> tuple[bool, str]:
    """Start the desk in-process, read /api/state, demand strict JSON."""
    import threading

    from scripts.desk_server import DeskState, build_server

    state = DeskState(
        trading_dir=ROOT / "reports" / "trading",
        ledger_path=ROOT / "reports" / "paper" / "ledger.json",
        store=ROOT / "reports" / "realtime" / "quotes.sqlite",
        reports_dir=ROOT / "reports",
    )
    httpd = build_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:%d/api/state" % httpd.server_address[1]
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8")
        assert "NaN" not in body, "state 里出现 NaN，页面会白屏"
        payload = json.loads(body)
        assert "gate" in payload and "quotes" in payload
        return True, "state %d 字节｜gate=%s" % (len(body), payload["gate"]["level_label"])
    except Exception as exc:  # noqa: BLE001 - the verdict is the message
        return False, "%s: %s" % (type(exc).__name__, exc)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=3)


def check_sla() -> tuple[bool, str]:
    code, tail = _run([sys.executable, str(ROOT / "scripts" / "measure_realtime_sla.py"), "--skip-block"])
    return code == 0, tail


CHECKS = [
    ("pytest", check_tests),
    ("publish_check", check_publish),
    ("artifact_fingerprints", check_artifact_fingerprints),
    ("staleness", check_staleness),
    ("desk_smoke", check_desk),
]


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run(*, with_sla: bool = False, record: bool = True, accept_anchor: str | None = None) -> dict:
    previous = None
    if RECORD.exists():
        try:
            previous = json.loads(RECORD.read_text(encoding="utf-8")).get("anchor")
        except ValueError:  # pragma: no cover - a corrupt record is just "no record"
            previous = None

    results: list[dict] = []
    for name, fn in CHECKS:
        started = time.perf_counter()
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001 - a broken check is a failed check
            ok, detail = False, "%s: %s" % (type(exc).__name__, exc)
        results.append({"name": name, "ok": bool(ok), "detail": detail,
                        "seconds": round(time.perf_counter() - started, 1)})
        print("%-24s %s  %5.1fs  %s" % (name, "✅" if ok else "❌", results[-1]["seconds"], detail), flush=True)

    # the anchor check must run *after* publish_check produced its summary
    if previous is not None or current_anchor() is not None:
        started = time.perf_counter()
        ok, detail = check_anchor(previous)
        reason = (accept_anchor or "").strip()
        if not ok and reason:
            # An override is allowed, but only when everything else is green: the
            # anchor is the *last* thing to trust, never the thing that excuses a
            # broken check.  And it leaves a line in the history, always.
            failures = [row["name"] for row in results if not row["ok"]]
            current = current_anchor()
            if failures:
                detail += "｜未确认锚点：%s 还没通过" % "、".join(failures)
            elif previous is None or current is None:
                detail += "｜未确认锚点：没有可比的旧锚点"
            else:
                record_anchor_change(previous, current, reason)
                ok = True
                detail = "锚点已按人工确认更新（理由：%s）｜total_return %.10f / sharpe %.10f" % (
                    reason, current["total_return"], current["sharpe"])
        results.append({"name": "backtest_anchor", "ok": bool(ok), "detail": detail,
                        "seconds": round(time.perf_counter() - started, 1)})
        print("%-24s %s  %5.1fs  %s" % ("backtest_anchor", "✅" if ok else "❌",
                                        results[-1]["seconds"], detail), flush=True)

    if with_sla:
        started = time.perf_counter()
        ok, detail = check_sla()
        results.append({"name": "sla_probe", "ok": bool(ok), "detail": detail,
                        "seconds": round(time.perf_counter() - started, 1)})
        print("%-24s %s  %5.1fs  %s" % ("sla_probe", "✅" if ok else "❌",
                                        results[-1]["seconds"], detail), flush=True)

    ok = all(row["ok"] for row in results)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ok": ok,
        "anchor": current_anchor(),
        "checks": results,
    }
    if record and ok:
        RECORD.parent.mkdir(parents=True, exist_ok=True)
        RECORD.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="一条命令验收：测试 / 自检 / 指纹 / 引用审计 / 交易台")
    parser.add_argument("--with-sla", action="store_true", help="额外跑行情 SLA 实测（需要网络）")
    parser.add_argument(
        "--accept-anchor",
        metavar="REASON",
        help="锚点确实该动时（例如补了新的交易日 bar）写一句理由来确认它；"
             "只在其它检查全绿时生效，并追加一行到 reports/acceptance/anchor_history.jsonl",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    payload = run(with_sla=args.with_sla, accept_anchor=args.accept_anchor)
    print()
    failed = [row["name"] for row in payload["checks"] if not row["ok"]]
    if failed:
        print("❌ 未通过：%s" % "、".join(failed))
    else:
        print("✅ 全部通过（%d 项）" % len(payload["checks"]))
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
