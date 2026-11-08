# 第三方资产说明（ui/vendor）

本目录只放**一个**运行时第三方文件，因为"零 Node、零构建"是这个交易台的硬约束
（见 `docs/UI方案.md` §2）。许可证与来源如下，均为 Apache-2.0（与本仓的许可策略一致，
且 GPL/AGPL 一律不进仓）。

## lightweight-charts 5.2.1

- 文件：`ui/vendor/lightweight-charts.standalone.production.js`（197,922 字节）
- 来源：`https://unpkg.com/lightweight-charts@5.2.1/dist/lightweight-charts.standalone.production.js`
- 上游：https://github.com/tradingview/lightweight-charts （实测 17,815 stars，Apache-2.0，2026-11-06 有提交）
- 许可：Apache License 2.0（文件头保留了原始 `@license` 注释，已一并入库）
- 为什么是它：唯一同时满足「Apache-2.0」「单个 `<script>` 即用、无需打包器」「canvas 只画可视区间
  （10 万根实测 3.9 ms/帧）」三条的候选；替代方案对比见 `docs/选型报告.md` §2.2
- 商标：Lightweight Charts™ 是 TradingView 的商标；本项目只在"内部自用工具"意义上使用该库，
  不使用其商标做任何品牌宣传
- 升级方式：替换该文件即可（API 有 breaking change 时要同步改 `ui/desk.js` 里
  `chart.addSeries(LightweightCharts.CandlestickSeries, …)` 的调用形式——5.x 起与 4.x 不同）

## 明确不引入的

| 候选 | 为什么不在仓里 |
|---|---|
| KLineChart | Apache-2.0 可用，但 v10 删了 v9 的 `applyNewData`，静默空白且零报错，踩坑成本高于收益 |
| echarts | Apache-2.0 且产物入库，但 1.1 MB、缩放帧 9.2 ms，且无 K 线交易语义；只在右栏多序列图时再考虑 |
| ghostfolio | **AGPL-3.0**：一行代码都不能拷 |
| Highcharts | 非开源许可（`license.txt` 指向付费商店） |
| OpenBB / streamlit / gradio / FastAPI | 与「零 Node + 依赖克制」冲突，且本台只需要 4 个 JSON 端点 |
