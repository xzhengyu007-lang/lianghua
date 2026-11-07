import sys, json, time
from pathlib import Path
sys.path.insert(0, '.')
from quant_system.data import pit_universe, normalize_symbol, TencentProvider

cache = Path('data/cache')
names = {normalize_symbol(s): n for s, n in pit_universe().items()}
missing = [s for s in names if not (cache / f'{s}_raw.parquet').exists()]
print(f'missing raw: {len(missing)}', flush=True)
provider = TencentProvider(cache, pause=2.0)
failures = {}
for i, sym in enumerate(missing, 1):
    try:
        r = provider.download(sym, count=1800, adjust='raw')
        print(f'[{i:03d}/{len(missing)}] {sym}: {r.rows} rows {r.start}..{r.end}', flush=True)
    except Exception as exc:
        failures[sym] = str(exc)
        print(f'[{i:03d}/{len(missing)}] {sym}: FAILED {exc}', flush=True)
    time.sleep(1.0)
Path('data/raw_pit_failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'done. failures={len(failures)}', flush=True)
