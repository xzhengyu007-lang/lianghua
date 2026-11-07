# 运行手册

按顺序照做即可。每一步都标注了**期望结果**；「非零退出但正常」的情况单独列在 §3，
因为那是照文档走最容易踩的坑。

## 0. 环境

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

所有命令在仓库根目录执行。

## 1. 日常（收盘后 / 每次改动后）

| # | 命令 | 期望 | 用时 |
|---|---|---|---:|
| 1 | `python -m quant_system --universe pit audit --output reports\audit_pit` | `"ok": true`、`"problems": []`、exit 0 | 5 s |
| 2 | `python scripts\run_verification.py` | data / reproducibility / causality 三项 `ok=True` | 6 s |
| 3 | `python scripts\audit_staleness.py` | `citation_problems: 0`，关键集 12/12 新鲜 | 3 s |
| 4 | `python -m quant_system --universe pit --adjustment rebuilt brief ...` | exit 0，生成 `brief.md/json/html` | 5 s |
| 5 | `python -m quant_system --universe pit --adjustment rebuilt signal ...` | exit 0，8 只目标权重 | 5 s |

第 4、5 步的完整参数见 `README.md` 的「快速开始」。

## 2. 改动引擎或股票池之后（多跑四步）

| # | 命令 | 期望 | 用时 |
|---|---|---|---:|
| 6 | `python scripts\run_publish_check.py` | 三项全 ok（可复现 / 指纹 / 因果） | ~2 min |
| 7 | `python scripts\refresh_critical.py` | 14/14 步 exit 0，关键集全新鲜 | ~25 min |
| 8 | `python scripts\build_decision_memo.py` | 重新生成 `DECISION.md` | 1 s |
| 9 | `python -m pytest -q` | 全部通过（当前 364 项） | ~2 min |

第 6 步不过**不要**继续：指纹红了说明配置或引擎变了，要么同步改常数（并在
`RESEARCH_LOG` 里说明为什么），要么回滚。

**新增或改名任何脚本后**，还要多跑一步（否则 `SCRIPTS.md` 过期、测试会红）：

```powershell
.\\.venv\\Scripts\\python.exe scripts\\build_script_inventory.py
```

同理，新增了一个报告目录，要跑 `build_report_index.py`；改了发布配置，要跑
`check_artifact_fingerprints.py --update` 并在日志里写明为什么。

## 3. 非零退出但属正常

| 命令 | 退出码 | 为什么正常 | 真正的判断标准 |
|---|---:|---|---|
| `python -m quant_system cache-audit` | **1** | 109 个 qfq 文件被标为 problem 是**预期**：腾讯的"前复权"是加法复权（§23.19） | 看 `*_tr.parquet` 是否 0 problem，而不是退出码 |

除此之外，任何非零退出都应该当作真失败处理。

## 4. 演练记录（2026-11-08）

从当前仓库状态起，按上面顺序实跑了一遍：

| 步骤 | 退出码 | 结果 |
|---|---:|---|
| PIT audit | 0 | `ok: true`，`problems: []`，在册/退市 128/139 |
| cache-audit | 1 | **预期**；qfq 文件报加法复权，`_tr` 正常 |
| run_verification | 0 | 三项全 True |
| brief | 0 | 生成简报；新鲜度守卫报「1 只更旧 bar」 |
| signal | 0 | 8 只等权 11.875%，合计 95% 仓位 |
| run_publish_check | 0 | 三项全 ok |
| build_decision_memo | 0 | 51 行 |
| audit_staleness | 0 | `citation_problems: 0`，40 份新鲜 / 198 份待重跑 |

## 5. 出问题时先看哪里

| 症状 | 先看 |
|---|---|
| 数据类失败（缺 bar、负价、重复索引） | `reports/audit_pit/` 与 `reports/cache_audit/` 的具体条目 |
| 因果性红 | 是否引入了居中窗口或全样本标准化；`RESEARCH_LOG` §23.60 的六项审计是完整版 |
| 指纹红 | `reports/publish_check/summary.json` 的 drift 字段 |
| 结论和 README 不一致 | 先跑 `audit_staleness.py`，再看引用报告的生成时间 |
| 不知道该信哪个数字 | `DECISION.md`（唯一权威）与 `FINDINGS.md`（置信度） |

## 6. 写测试时的三个约定

这三条都是踩过的，不是理论：

1. **用 importlib 加载含 `@dataclass` 的脚本时，先注册到 `sys.modules`。**
   否则 `@dataclass` 解析注解会抛 `AttributeError: 'NoneType' object has no attribute '__dict__'`
   （在 §23.78 与 §23.102 各踩一次）。
2. **构造合成面板要给全 OHLCV。** 只给 `close` 与 `volume` 会在 `panel_to_frame` 里
   `KeyError: open missing`。最小可用列：`open / high / low / close / volume`。
3. **依赖产物的测试用 `pytest.skip` 而不是失败。** 报告目录被 gitignore，干净检出时不存在，
   应当跳过而不是红。

## 7. 相关文档

- `DECISION.md` —— 要不要上钱、期望多少、缺什么证据（生成物，勿手改）
- `FINDINGS.md` —— 69 条结论的置信度与依据
- `RESEARCH_LOG.md` —— 97 节，每条结论怎么来的、错在哪
- `README.md` —— 用法、口径权威表、已知限制
