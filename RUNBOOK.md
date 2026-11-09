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
| 6 | `python scripts\run_acceptance.py` | 五项全 ✅（含回测锚点与交易台冒烟） | ~4 min |

第 4、5 步的完整参数见 `README.md` 的「快速开始」。

## 1.4 交付前的验收（一条命令）

```powershell
.\.venv\Scripts\python.exe scripts\run_acceptance.py
```

五项：`pytest` 全绿、`run_publish_check` 三项 ok、上游产物指纹未漂、引用审计 0 问题、
交易台 `/api/state` 是严格 JSON。**第六项是回测锚点**：与 `reports/acceptance/last.json`
里上次通过的锚点逐位比对——首次运行只记录。失败的那次不会覆盖记录。

锚点变了只有两种解释：要么你确实改了口径（那就同步改 `DECISION.md`/README 并说明），
要么你无意中动了回测路径（那就回滚）。这个脚本不判断哪种，它只拒绝装作没发生。

## 1.5 每个交易日的两条命令（交易层）

上表是"研究口径健康"，这一节是"今天该做什么"。两者互不替代：上面的 1–5 步证明数字可信，
下面这一步决定要不要动钱。

### 收盘后（15:00 之后，一条命令）

```powershell
.\.venv\Scripts\python.exe scripts\run_daily_ops.py --broker reports\paper\broker.json
```

它按顺序做五件事并写 `reports\daily_ops\<日期>\report.md`：信号 → 建议卡 → 下单计划 →
对账单（含每日权益点）→ 券商对账。**退出码就是结论**：

| 退出码 | 含义 | 你要做什么 |
|---:|---|---|
| 0 | 五步全绿 | 明天同一时间再跑 |
| 1 | 需要人看：数据过期 / 停机 / 对账差异 / **真实拒单** | 打开 `report.md` 的「需要人做的事」 |
| 2 | 作业本身失败（异常） | 看 stderr，修完重跑（重跑是安全的） |

报告目录按**刚收盘的那个交易日**命名，不按你按下回车的日期：周一 00:45 跑的是周一的收盘，
写进 `reports\daily_ops\2026-11-09\`，不会是 `2026-11-10\`。同一次运行的"决策日"（`trade_date`，
订单在哪个交易日执行、幂等键和审计日志认的日期）若与它不同，标题会写明
「收盘作业 2026-11-09（决策日 2026-11-10，下一交易日开盘执行）」——两个日期都必须出现，
因为它们是两件事：一个是刚结束的会话，一个是这张单子要去的会话。

两句必须记住的话：

- **`T+1 当日买入不可卖`、`重复生成已被拒绝`、`低于最小下单额` 不是故障**——报告会写"今天不需要动作"。
  只有 `现金不足` / `停牌` / `限价越界` 才升级为 `warn` 并让退出码变 1。
- **`plan` 不落成交**：它只写订单与审计，账本一分钱不动；`--fills` 才移动资金。所以同一天重跑两遍不会重复下单。

### 交易台说"落后 N 个交易日"怎么办

横幅上那两个日期必须分开放：右边是**市场**（`last_completed_session`，交易所今天到底开没开），
左边是**我们的数据**（`cache_as_of`）。它们不一致时，正确动作只有一条——**先补数据，再重算**：

```powershell
# 1) 补当天的 bar（由研究侧的数据更新流程负责，产物进 data/cache）
# 2) 重算建议卡（数据补上后闸门自动放行；没有 --force 开关，也不该有）
.\.venv\Scripts\python.exe -m quant_system advice --output reports\advice_today
# 3) 收盘作业（第 2 步本来就在它内部，单独跑只是更快看到卡片）
.\.venv\Scripts\python.exe scripts\run_daily_ops.py --broker reports\paper\broker.json
```

在补上之前：交易台的「生成计划」是禁用的，`POST /api/plan` 也会在服务端再拒一次（理由里带差几个交易日），
`run_daily_ops` 退出码 1。**这不是故障，是规则在生效**——用上一交易日的篮子在今天开盘下单，
正是这套闸门存在的原因。

> 反过来的情况也测过：日历还没收录今天、但行情已经打印了今天的收盘（新交易日），
> 闸门认的是**交易所的打印日**，不会因为"日历不知道"就误判数据过期。

### 盘中（09:30–11:30 / 13:00–15:00）

```powershell
.\.venv\Scripts\python.exe scripts\desk_server.py      # http://127.0.0.1:8770
```

页面顶栏的徽标就是 `QuotePoller` 的状态机：绿=行情正常、黄=重连中（首轮失败立刻变黄）、红=连续 5 轮失败。
注意"徽标绿"只说明**接口活着**，不说明**数据够新**——后者看数据健康面板里的"落后 N 个交易日"。
命令行等价物是 `python -m quant_system quote --watch --interval=3`。

停机永远优先于其它动作：页面的「全部停机」或 `python -m quant_system trade --action kill --reason "..."`，
两者都会把级别打回 L1 并写审计。恢复要显式 `trade --action reset-kill`。

> 交易层的产物（账本、建议卡、审计、权益曲线）都在 `reports/` 下，已被 gitignore：
> 它们是操作记录，不是研究结论，**不要为了"留个证据"把账本提交进仓库**。

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

### 2.1 如果你是刚 clone 下来的

先下载数据，再核对**拿到的是不是作者那一份**：

```powershell
.\\.venv\\Scripts\\python.exe -m quant_system --universe broad download --count 1800
.\\.venv\\Scripts\\python.exe -m quant_system --universe pit audit --output reports\audit_pit
.\\.venv\\Scripts\\python.exe scripts\\build_data_manifest.py --check
```

`0 problems` 才说明后续复现是有意义的。出现 `CHANGED` 就说明数据源已经变了（下载日期、复权
基准、退市名单都可能不同），**此后的数字不能与仓库里的结论直接对照**。

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

## 7. 写结论前的自检清单

这六条全部来自本仓实际犯过的错（§23.107、§23.118），每条都配一个具体动作，不是"要谨慎"。

| # | 自问 | 如果答"是"，做什么 | 出处 |
|---|---|---|---|
| 1 | 我把这句话的**适用范围**写出来，它还成立吗？ | 用自己的产物反查措辞。§23.79 只读了 trades.csv 的 reason 列，就拆穿了"不可交易"的说法 | §23.76 → §23.79 |
| 2 | 我一次改了几个变量？ | 做 2×2，或给"另一个变量"也做随机对照。§23.93 用随机子池才分离出"集中"才是杀手 | §23.92 → §23.93 |
| 3 | 这个数字**能不能生成**？ | 能生成的就不手抄：生成物 + 逐字比对 + 声明审计。手抄的数字会在一轮内过期 | §23.98、§23.88、§23.106 |
| 4 | 这个"趋势"是不是**正好停在我扫描的边界上**？ | 把参数推到它自己掉头为止。§23.118：「越长越好」只是扫到 250 就停了 | §23.116 → §23.118 |
| 5 | 这是**单相位**的结论吗？ | 用 20 相位混合复核。相位跨度 0.58，比大多数配置间差距都大 | §23.82、§23.116 |
| 6 | 我是在**加证据**，还是在**换表达**？ | 换检验、换执行时点、换调仓周期都不产生新信息——三次都已证伪 | §23.91、§23.101、§23.104 |

第 4 和第 6 条最容易漏，因为它们感觉不到错：扫描边界外的那一半你根本没看，而"换个更聪明的
检验"永远是诱人的。

**修订必须写回原处。** 结论被推翻时，在**那一节的开头**加修正指引（形如「§23.xx 修正了本节
的说法」），而不是只在新一节里顺便说说——读者是按节读的，不按时间线读。

## 8. 相关文档

- `DECISION.md` —— 要不要上钱、期望多少、缺什么证据（生成物，勿手改）
- `FINDINGS.md` —— 69 条结论的置信度与依据
- `RESEARCH_LOG.md` —— 97 节，每条结论怎么来的、错在哪
- `README.md` —— 用法、口径权威表、已知限制
