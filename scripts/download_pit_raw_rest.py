import sys, json, time
from pathlib import Path
sys.path.insert(0, '.')
from quant_system.data import TencentProvider

cache = Path('data/cache')
todo = json.loads(Path('data/tmp_raw_todo.json').read_text(encoding='utf-8'))
print(f'todo: {len(todo)}', flush=True)
provider = TencentProvider(cache, pause=1.2, retries=2, timeout=20)
failures = {}
for i, sym in enumerate(todo, 1):
    try:
        r = provider.download(sym, count=1800, adjust='raw')
        print(f'[{i:03d}/{len(todo)}] {sym}: {r.rows} rows {r.start}..{r.end}', flush=True)
    except Exception as exc:
        failures[sym] = str(exc)[:200]
        print(f'[{i:03d}/{len(todo)}] {sym}: FAILED', flush=True)
    time.sleep(0.6)
Path('data/raw_pit_failures.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
print(f'done. failures={len(failures)}', flush=True)
