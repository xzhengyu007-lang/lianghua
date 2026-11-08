"""How far back could the panel actually go, if the download were extended?

RESEARCH_LOG 23.121 established that the 2019-06 start is a download parameter
(--count 1800), not a data limit: the provider pages back to 2005 and a live probe
returned 4000 bars from 2010 for two large caps.

But extending the download does not by itself move the panel window. The window is
set by _research_panel: 95% coverage of the 128 survivors. If enough of those names
listed after 2010, the window cannot go back that far no matter how much history is
downloaded. This measures coverage at a few candidate dates with one request per
symbol (count=1 ending at the candidate date), which is cheap and does not touch the
production cache.

Usage::

    python scripts/run_sample_extent.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant_system.config import BROAD_UNIVERSE  # noqa: E402
from quant_system.data import TencentProvider, normalize_symbol  # noqa: E402

CANDIDATES = ("2010-06-01", "2013-06-01", "2015-06-01", "2017-06-01", "2018-06-01")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "reports" / "sample_extent"))
    parser.add_argument("--pause", type=float, default=0.25)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    provider = TencentProvider(ROOT / "data" / "cache_probe", pause=args.pause, retries=2, timeout=15)
    symbols = [normalize_symbol(s) for s in BROAD_UNIVERSE]
    print("probing %d survivor symbols at %d dates" % (len(symbols), len(CANDIDATES)), flush=True)

    coverage = {date: 0 for date in CANDIDATES}
    failures = 0
    for index, symbol in enumerate(symbols, 1):
        for date in CANDIDATES:
            try:
                frame = provider._request(symbol, 1, "qfq", date)
                if frame is not None and len(frame):
                    coverage[date] += 1
            except Exception:
                failures += 1
            time.sleep(args.pause)
        if index % 20 == 0:
            print("  %d/%d done" % (index, len(symbols)), flush=True)

    total = len(symbols)
    share = {date: coverage[date] / total for date in CANDIDATES}
    qualified = [d for d in CANDIDATES if share[d] >= 0.95]

    summary = {
        "symbols": total,
        "probe_failures": failures,
        "candidates": list(CANDIDATES),
        "coverage": {d: coverage[d] for d in CANDIDATES},
        "share": share,
        "dates_at_or_above_95pct": qualified,
        "earliest_qualified": qualified[0] if qualified else None,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    lines = ["# 数据源能给到的样本长度", ""]
    lines.append("用一条只取 1 根 bar、截止在候选日的请求，检查该日是否已有数据。")
    lines.append("面板窗口的规则是 **128 只在册标的的 95% 覆盖率**，所以这里直接按 95% 判。")
    lines.append("")
    lines.append("| 候选日 | 有数据的标的数 | 占比 | 达到 95% |")
    lines.append("|---|---:|---:|:--:|")
    for date in CANDIDATES:
        ok = share[date] >= 0.95
        lines.append("| %s | %d / %d | %.1f%% | %s |"
                     % (date, coverage[date], total, 100 * share[date], "是" if ok else "—"))
    lines.append("")
    if qualified:
        lines.append("**最早可达窗口起点：%s**（该日覆盖率已达 95%%，延长下载后窗口可以走到这里）。"
                     % qualified[0])
    else:
        lines.append("**没有任何候选日达到 95%**：延长下载也不会把窗口拉回这么久，"
                     "瓶颈是在册标的的上市时间，不是数据量。")
    lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\ncoverage:", {d: "%d/%d" % (coverage[d], total) for d in CANDIDATES})
    print("earliest qualified:", summary["earliest_qualified"], "| request failures:", failures)
    print("report: %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
