"""A manifest of the cached data, so a fresh checkout can prove it has the same data.

Nothing in this repository can currently answer "do I have the author's data?".
The cache is gitignored (48 MB of parquet), cache-audit only checks files against
their own download metadata, and the freshness audit compares timestamps. A reader
who runs download/ themselves gets whatever the vendor serves today, with no way
to tell whether the numbers that follow are comparable.

This writes data/cache_manifest.json (tracked): one line per cached file with
symbol, row count, first and last bar, and a truncated SHA-256. Re-running the
research is only meaningful against the same data, and now that is checkable.

Usage::

    python scripts/build_data_manifest.py            # write the manifest
    python scripts/build_data_manifest.py --check    # verify the cache matches
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
MANIFEST = ROOT / "data" / "cache_manifest.json"


def short_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def describe(path: Path, read_dates: bool = True) -> dict:
    entry = {"file": path.name, "sha256_16": short_hash(path)}
    if read_dates:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            entry["error"] = "unreadable"
            return entry
        entry["rows"] = int(len(frame))
        if len(frame):
            index = pd.to_datetime(frame.index)
            entry["start"] = str(index.min().date())
            entry["end"] = str(index.max().date())
    return entry


def build(read_dates: bool = True) -> dict:
    files = sorted(p for p in CACHE.glob("*.parquet"))
    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "files": [describe(p, read_dates=read_dates) for p in files],
    }


def compare(current: dict, manifest: dict) -> dict:
    stored = {row["file"]: row for row in manifest.get("files", [])}
    now = {row["file"]: row for row in current["files"]}

    missing = sorted(set(stored) - set(now))
    extra = sorted(set(now) - set(stored))
    changed = []
    for name in sorted(set(stored) & set(now)):
        if stored[name]["sha256_16"] != now[name]["sha256_16"]:
            changed.append(
                {
                    "file": name,
                    "manifest_rows": stored[name].get("rows"),
                    "current_rows": now[name].get("rows"),
                    "manifest_span": [stored[name].get("start"), stored[name].get("end")],
                    "current_span": [now[name].get("start"), now[name].get("end")],
                }
            )
    return {"missing": missing, "extra": extra, "changed": changed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the cache against the manifest")
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)

    if not args.check:
        payload = build(read_dates=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        size_kb = manifest_path.stat().st_size / 1024
        print("manifest written: %s files, %.0f KiB" % (len(payload["files"]), size_kb))
        return 0

    if not manifest_path.exists():
        print("no manifest at %s; run without --check first" % manifest_path)
        return 2

    # Hashing alone is enough to answer "same data?"; skip the parquet reads.
    current = build(read_dates=False)
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = compare(current, stored)

    for name in result["missing"][:10]:
        print("  MISSING  %s" % name)
    for name in result["extra"][:10]:
        print("  EXTRA    %s" % name)
    for row in result["changed"][:10]:
        print("  CHANGED  %s  行数 %s → %s  区间 %s → %s"
              % (row["file"], row["manifest_rows"], row["current_rows"],
                 row["manifest_span"], row["current_span"]))
    problems = len(result["missing"]) + len(result["extra"]) + len(result["changed"])
    print("files checked: %d, problems: %d (missing %d, extra %d, changed %d)"
          % (len(current["files"]), problems,
             len(result["missing"]), len(result["extra"]), len(result["changed"])))
    return 0 if problems == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
