/* 交易台前端：无构建、无框架、无 CDN。
 * 唯一的数据来源是本机 /api/*（同源、127.0.0.1）。
 *
 * 三条刻意的设计：
 *  1. 行情只在服务端轮询一次（QuotePoller），浏览器读 /api/state，避免每个标签页各打一遍免费接口；
 *  2. 图表的 K 线来自"重建总收益"缓存（与回测同一口径），不用实时价拼历史；
 *  3. 任何会改变状态的按钮都要先确认，且 L3 不提供界面入口（只能命令行逐日授权）。
 */
'use strict';

const $ = (id) => document.getElementById(id);
const fmt = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(Number(value))
    ? '—' : Number(value).toFixed(digits);
const pct = (value) => (value === null || value === undefined ? '—' : (Number(value) * 100).toFixed(2) + '%');
const cls = (value) => (Number(value) > 0 ? 'up' : Number(value) < 0 ? 'down' : 'na');

let chart = null;
let candleSeries = null;
let volumeSeries = null;
let markersPrimitive = null;   /* v5: markers 是 createSeriesMarkers 返回的原语 */
let currentSymbol = null;
let limit = 600;
let lastState = null;

/* ------------------------------------------------------------------ chart */
function ensureChart() {
  if (chart) return chart;
  const box = $('chart');
  chart = LightweightCharts.createChart(box, {
    layout: { background: { color: 'transparent' }, textColor: '#8b949e' },
    grid: { vertLines: { color: '#20262f' }, horzLines: { color: '#20262f' } },
    rightPriceScale: { borderColor: '#262c36' },
    timeScale: { borderColor: '#262c36', timeVisible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    autoSize: true,
  });
  candleSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
    upColor: '#f6465d', downColor: '#0ecb81',
    wickUpColor: '#f6465d', wickDownColor: '#0ecb81',
    borderVisible: false,
  });
  volumeSeries = chart.addSeries(LightweightCharts.HistogramSeries, {
    priceFormat: { type: 'volume' }, priceScaleId: 'vol', lastValueVisible: false,
  });
  chart.priceScale('vol').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
  return chart;
}

async function loadBars(symbol, nextLimit) {
  if (!symbol) return;
  if (nextLimit) limit = nextLimit;
  currentSymbol = symbol;
  $('chart-symbol').textContent = symbol + ' · ' + limit + ' 根请求';
  const response = await fetch(`/api/bars?symbol=${encodeURIComponent(symbol)}&limit=${limit}`);
  const payload = await response.json();
  if (payload.error) {
    $('chart-note').textContent = payload.error;
    return;
  }
  const bars = payload.bars || [];
  ensureChart();
  candleSeries.setData(bars.map((bar) => ({
    time: bar.time, open: bar.open, high: bar.high, low: bar.low, close: bar.close,
  })));
  volumeSeries.setData(bars.map((bar) => ({
    time: bar.time, value: bar.volume,
    color: bar.close >= bar.open ? 'rgba(246,70,93,.45)' : 'rgba(14,203,129,.45)',
  })));
  chart.timeScale().fitContent();
  $('chart-note').textContent = `数据：${payload.convention} ｜ ${payload.count} 根`;
  applyMarkers();
}

/* 调仓/成交标注：只标本仓真实发生过的日期，不编造信号点。
 * v5 的 API 变了：series 上没有 setMarkers，必须先 createSeriesMarkers 拿原语。
 * 写成"拿不到就安静放弃"而不是让整页挂掉——图表标注是装饰，行情和风控不是。 */
function applyMarkers() {
  if (!candleSeries || !lastState || !window.LightweightCharts) return;
  const trades = (lastState.audit && lastState.audit.recent) || [];
  const marks = trades
    .filter((event) => event.event === 'order.accepted' && event.symbol === currentSymbol && event.trade_date)
    .map((event) => ({
      time: event.trade_date,
      position: event.side === 'buy' ? 'belowBar' : 'aboveBar',
      color: event.side === 'buy' ? '#f6465d' : '#0ecb81',
      shape: event.side === 'buy' ? 'arrowUp' : 'arrowDown',
      text: event.side === 'buy' ? '买' : '卖',
    }));
  try {
    if (!markersPrimitive) {
      if (typeof LightweightCharts.createSeriesMarkers !== 'function') return;
      markersPrimitive = LightweightCharts.createSeriesMarkers(candleSeries, []);
    }
    markersPrimitive.setMarkers(marks);
  } catch (error) {
    console.warn('标注渲染失败（不影响行情与风控）:', error.message);
  }
}

/* ------------------------------------------------------------------ state */
function renderFreshness(quotes) {
  const dot = $('fresh-dot');
  const text = $('fresh-text');
  const state = quotes.state || 'yellow';
  dot.className = 'dot ' + (state === 'green' ? 'ok' : state === 'red' ? 'bad' : 'warn');
  text.textContent = quotes.message || '—';
}

function renderGate(gate) {
  $('gate-badge').textContent = gate.level_label || '—';
  const kill = $('gate-kill');
  kill.textContent = gate.kill_switch ? '已停机' : '正常';
  kill.className = 'badge ' + (gate.kill_switch ? 'bad' : '');
  $('btn-kill').disabled = !!gate.kill_switch;
  $('btn-reset').disabled = !gate.kill_switch;
  $('mode').textContent = gate.level >= 3 ? '实盘级（无网关）' : gate.level === 2 ? '半自动' : '模拟盘';
}

function renderBasket(card, quotes) {
  const body = $('basket').querySelector('tbody');
  body.innerHTML = '';
  const rows = (card && card.items) || [];
  if (!rows.length) {
    $('basket-hint').textContent = card && card.blocked
      ? '建议卡被闸门拦住：' + (card.block_reasons || []).join('；')
      : '没有目标篮（100% 现金或尚未生成建议卡）';
    return;
  }
  const board = (quotes.board && quotes.board.quotes) || {};
  rows.forEach((item) => {
    const quote = board[item.symbol] || {};
    const price = quote.price || item.price;
    const change = quote.change_pct;
    const status = quote.status_label || '—';
    const suspect = quote.suspect ? ' ⚠' : '';
    const stale = quote.stale ? '（滞后）' : '';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${item.symbol}</td>
      <td>${item.name || '—'}</td>
      <td class="r ${cls(change)}">${fmt(price)}</td>
      <td class="r ${cls(change)}">${change === undefined ? '—' : fmt(change) + '%'}</td>
      <td><span class="st">${status}${suspect}</span><span class="dim">${stale}</span></td>`;
    row.addEventListener('click', () => loadBars(item.symbol));
    body.appendChild(row);
  });
  const watch = (card && card.watchlist) || [];
  const hint = $('basket-hint');
  hint.textContent = `${rows.length} 只｜目标权重合计 ${pct(rows.reduce((sum, r) => sum + r.weight, 0))}` +
    (watch.length ? `｜候补 ${watch.length} 只（未在篮）` : '');
  const select = $('symbol-select');
  select.innerHTML = rows.map((item) =>
    `<option value="${item.symbol}">${item.symbol} ${item.name || ''}</option>`).join('');
  const watchBox = $('watchlist');
  if (watchBox) {
    watchBox.innerHTML = watch.length
      ? '候补（最强但未入篮）：' + watch.slice(0, 5).map((row) =>
          `<span class="chipwatch" data-symbol="${row.symbol}">${row.symbol} ${fmt(row.score, 2)}</span>`).join('')
      : '';
    watchBox.querySelectorAll('.chipwatch').forEach((chip) => {
      chip.addEventListener('click', () => loadBars(chip.dataset.symbol));
    });
  }
  if (!currentSymbol) loadBars(rows[0].symbol);
}

function renderLedger(ledger, equity) {
  const kpis = $('ledger-kpis');
  if (!ledger) {
    kpis.innerHTML = '<div class="kpi"><div class="k">账本</div><div class="v">未接入</div></div>';
    $('ledger-meta').textContent = '';
    return;
  }
  $('ledger-meta').textContent = `as_of ${ledger.as_of || '—'}`;
  kpis.innerHTML = `
    <div class="kpi"><div class="k">总权益</div><div class="v">${fmt(ledger.equity, 0)}</div></div>
    <div class="kpi"><div class="k">现金</div><div class="v">${fmt(ledger.cash, 0)}</div></div>
    <div class="kpi"><div class="k">持仓市值</div><div class="v">${fmt(ledger.market_value, 0)}</div></div>
    <div class="kpi"><div class="k">持仓数</div><div class="v">${ledger.positions ?? '—'}</div></div>`;
  const body = $('equity').querySelector('tbody');
  const points = (equity && equity.points) || [];
  body.innerHTML = points.slice(-8).reverse().map((point) => `
    <tr><td>${point.trade_date}</td>
        <td class="r">${fmt(point.equity, 0)}</td>
        <td class="r ${cls(point.return)}">${point.return === null ? '—' : pct(point.return)}</td>
        <td class="r ${cls(point.drawdown)}">${point.drawdown === null ? '—' : pct(point.drawdown)}</td></tr>`).join('')
    || '<tr><td colspan="4" class="na">还没有每日权益点：成交登记后自动累加</td></tr>';
}

function renderHealth(card) {
  const data = (card && card.data) || {};
  const coverage = Number(data.coverage || 0);
  const width = Math.max(0, Math.min(100, coverage * 100)).toFixed(1);
  $('health').innerHTML = `
    <div>缓存最新交易日 <b>${data.as_of || '—'}</b>（上一交易日 ${(lastState && lastState.last_trading_day) || '—'}）</div>
    <div class="bar"><span style="width:${width}%;background:${coverage >= 0.95 ? 'var(--ok)' : 'var(--bad)'}"></span></div>
    <div>覆盖率 ${pct(coverage)}｜在册 ${data.live_symbols ?? '—'} 只｜当日有 bar ${data.on_latest_bar ?? '—'} 只</div>
    ${data.stale_examples && data.stale_examples.length
      ? `<div>落后：${data.stale_examples.map((row) => row.symbol + '(' + row.last_bar + ')').join('，')}</div>` : ''}`;
}

function renderAdvice(card) {
  const box = $('advice');
  if (!card) {
    box.innerHTML = '<p class="hint">尚未生成建议卡。命令行执行：<code>aq advice --output reports/advice_today</code></p>';
    return;
  }
  if (card.blocked) {
    box.innerHTML = `<div class="blocked"><b>⛔ 本日不出建议</b><ul>${
      (card.block_reasons || []).map((reason) => `<li>${reason}</li>`).join('')}</ul>
      <div class="dim">数据不新鲜时给出建议比不给更危险：本卡只报警，不排单。</div></div>`;
    return;
  }
  const risk = card.risk || {};
  const unc = card.uncertainty || {};
  const caveats = [];
  if (unc.phase_median_sharpe !== null && unc.phase_median_sharpe !== undefined) {
    caveats.push(`相位中位 Sharpe 只有 <b>${fmt(unc.phase_median_sharpe, 3)}</b>、最差相位 ${pct(unc.phase_worst_return)}`);
  }
  if (unc.power !== null && unc.power !== undefined) {
    caveats.push(`${fmt(unc.power_years, 1)} 年记录对真 Sharpe 0.90 只有 <b>${Math.round(unc.power * 100)}%</b> 检出力`);
  }
  if (unc.ci_low !== null && unc.ci_low !== undefined) {
    caveats.push(`自助法 95% 区间 [${fmt(unc.ci_low, 3)}, ${fmt(unc.ci_high, 3)}]`);
  }
  box.innerHTML = `
    <div class="headline">部署口径：<b>${unc.total_return === null ? '—' : (unc.total_return * 100).toFixed(2) + '%'} / Sharpe ${fmt(unc.sharpe, 3)} / 回撤 ${pct(unc.max_drawdown)}</b></div>
    <ul class="caveat">${caveats.map((line) => `<li>${line}</li>`).join('')}</ul>
    <h3>目标篮（${(card.items || []).length} 只）</h3>
    <table><thead><tr><th>标的</th><th>股数</th><th>权重</th><th>限价</th></tr></thead><tbody>${
      (card.items || []).map((item) => `<tr>
        <td>${item.symbol}</td>
        <td class="r">${item.shares}</td>
        <td class="r">${pct(item.weight)}</td>
        <td class="r">${fmt(item.limit_low)}–${fmt(item.limit_high)}</td></tr>`).join('')}</tbody></table>
    <h3>为什么在篮子里 / 风险</h3>
    ${(card.items || []).map((item) => `<details class="item"><summary>${item.symbol} ${item.name || ''}</summary>
        <ul class="why">${(item.reasons || []).map((reason) => `<li>${reason}</li>`).join('')}</ul>
        <ul class="risk">${(item.risks || []).map((risk) => `<li>${risk}</li>`).join('')}</ul></details>`).join('')}
    <h3>组合层风险</h3>
    <div>建议总仓位 ${pct(risk.gross_exposure)}｜事前波动 ${risk.ex_ante_vol === null ? '—' : pct(risk.ex_ante_vol)}｜单票上限 ${pct(risk.max_position_weight)}</div>
    <div class="dim" style="margin-top:8px">${card.disclaimer || ''}</div>`;
}

function renderAudit(audit) {
  const box = $('audit');
  const recent = (audit && audit.recent) || [];
  box.innerHTML = recent.length
    ? recent.slice().reverse().map((event) => `<li><b>${event.event}</b> ${event.symbol || ''} ${event.side || ''} ${event.shares || ''} <span class="dim">${event.at}</span></li>`).join('')
    : '<li class="na">还没有审计事件</li>';
}

function guard(name, fn) {
  try { fn(); } catch (error) { console.error(name + ' 渲染失败:', error); }
}

async function refresh() {
  let state;
  try {
    const response = await fetch('/api/state');
    if (!response.ok) throw new Error('HTTP ' + response.status);
    state = await response.json();
  } catch (error) {
    /* 真正的"本地服务不可用"只在这里发生：拿不到状态。渲染错误单独处理，不许混为一谈。 */
    $('fresh-text').textContent = '本地服务不可用：' + error.message;
    $('fresh-dot').className = 'dot bad';
    return;
  }
  lastState = state;
  $('session').textContent = `${state.today} ｜ ${state.mode}`;
  guard('新鲜度', () => renderFreshness(state.quotes));
  guard('闸门', () => renderGate(state.gate));
  guard('目标篮', () => renderBasket(state.advice, state.quotes));
  guard('账本', () => renderLedger(state.ledger, state.equity));
  guard('数据健康', () => renderHealth(state.advice));
  guard('建议卡', () => renderAdvice(state.advice));
  guard('审计', () => renderAudit(state.audit));
  guard('标注', () => applyMarkers());
}

/* ------------------------------------------------------------------ actions */
function toast(message) {
  const box = $('toast');
  box.textContent = message;
  box.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { box.hidden = true; }, 4200);
}

async function post(path, body) {
  const response = await fetch(path, {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}),
  });
  return response.json();
}

$('btn-kill').addEventListener('click', async () => {
  if (!confirm('确认停机？\n\n会立刻：停止生成新单 + 把级别打回 L1 模拟盘。\n已存在的订单不会自动撤销（本仓无网关）。')) return;
  const result = await post('/api/gate', { action: 'kill', reason: '界面人工停机' });
  toast(result.error ? '失败：' + result.error : '已停机');
  refresh();
});

$('btn-reset').addEventListener('click', async () => {
  if (!confirm('确认恢复？\n\n会解除 kill switch 并保持 L1 模拟盘。')) return;
  const result = await post('/api/gate', { action: 'reset-kill' });
  toast(result.error ? '失败：' + result.error : '已恢复（L1）');
  refresh();
});

$('btn-plan').addEventListener('click', async () => {
  if (!confirm('生成今日交易计划？\n\n会依次做：校验 → 限额 → 幂等 → 审计。\n已有有效订单的标会被跳过（幂等）。')) return;
  const result = await post('/api/plan', {});
  if (result.error) { toast('失败：' + result.error); return; }
  if (result.blocked) { toast('数据闸门已关闭：本日不出建议，也不排单'); return; }
  const plan = result.plan || {};
  toast(`通过 ${(plan.orders || []).length} 张，被拒 ${(plan.rejected || []).length} 张` +
        (plan.written ? '，已生成券商导入文件' : ''));
  refresh();
});

$('symbol-select').addEventListener('change', (event) => loadBars(event.target.value));
document.querySelectorAll('.chip[data-limit]').forEach((button) => {
  button.addEventListener('click', () => loadBars(currentSymbol, Number(button.dataset.limit)));
});

refresh();
setInterval(refresh, 3000);
