/**
 * 个股页「正在准备数据」不会卡死的回归测试。
 *
 * 背景（2026-11-11）：服务重启/重载的窗口里，/api/stock/<code> 会回 503，
 * 前端显示「正在准备数据…稍候自动重试」。但重试链有三处断点，导致它常常永久卡住：
 *   1) state.stockCode 只在**成功**后才赋值 → 失败后切走再回来，initStockPage 什么都不做；
 *   2) 重试要求 activePage === 'stock' → 切页后重试链静默断掉；
 *   3) 连接失败（服务正在重启，fetch 抛 TypeError）被当成硬错误，直接甩红框。
 *
 * 本测试不做重新实现：它从 web/static/app.js 里**原文抽取**这几个函数，
 * 在 vm 沙箱里用桩对象执行，验证上面三条都已修好。
 *
 * 运行：node tests/js/test_stock_retry.mjs
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import vm from 'node:vm';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SRC = path.resolve(HERE, '../../web/static/app.js');
const src = readFileSync(SRC, 'utf8');

/** 从源码里按函数名抽取一段 `function name(...) { ... }`（花括号配平） */
function extractFn(name) {
  const start = src.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`找不到函数 ${name}`);
  const open = src.indexOf('{', start);
  let depth = 0;
  for (let i = open; i < src.length; i++) {
    const ch = src[i];
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  throw new Error(`函数 ${name} 花括号不配平`);
}

function extractVar(name) {
  const m = src.match(new RegExp(`var ${name}\\s*=\\s*[^;]+;`));
  if (!m) throw new Error(`找不到变量 ${name}`);
  return m[0];
}

const PIECES = [
  extractVar('STOCK_MAX_RETRY'),
  extractFn('isServiceLoading'),
  extractFn('isTransientStockError'),
  extractFn('loadStock'),
  extractFn('renderStockWaiting'),
  extractFn('timeoutRetryStock'),
  extractFn('initStockPage'),
].join('\n');

/** 造一个沙箱：apiFetch 由外部脚本控制，setTimeout 立刻执行（否则 20 次重试要 2 分钟） */
function makeSandbox(responder) {
  const html = { stockDetail: '', input: '' };
  const calls = { renderStock: [], toast: [], markOk: 0, markDegraded: 0, fetches: [] };

  const sandbox = {
    console,
    state: {
      stockLoadSeq: 0,
      stockCode: null,
      stockRetry: 0,
      stock: null,
      activePage: 'today',            // 故意不在个股页：验证重试不再依赖 current page
      analysisModel: 'wf_h10',
      defaultModel: 'wf_h10',
      loaded: {},
    },
    // ---- DOM 桩 ----
    byId: (id) => (id === 'stockDetail' || id === 'stockSearchInput' ? { } : null),
    setHtml: (id, h) => { if (id === 'stockDetail') html.stockDetail = h; },
    esc: (s) => String(s == null ? '' : s),
    attr: (s) => String(s == null ? '' : s),
    loadingBlock: (t) => `<loading>${t}</loading>`,
    encodeURIComponent,
    // ---- 业务桩 ----
    renderStock: (d) => calls.renderStock.push(d),
    markOk: () => { calls.markOk++; },
    markDegraded: () => { calls.markDegraded++; },
    toast: (m) => calls.toast.push(m),
    apiFetch: (url) => {
      calls.fetches.push(url);
      return responder(url, calls.fetches.length);
    },
    // 立即执行，避免测试等 2 分钟
    setTimeout: (fn) => setImmediate(fn),
    // 提供 querySelector：initStockPage 用它判断"是否已经画过"
    __hasCard: () => /class="card"/.test(html.stockDetail),
  };
  sandbox.byId = (id) => {
    if (id === 'stockSearchInput') return { value: html.input };
    if (id === 'stockDetail') return { querySelector: () => (sandbox.__hasCard() ? {} : null) };
    return null;
  };
  // initStockPage 里 setHtml 写入的是提示文案，需要 let 化
  vm.createContext(sandbox);
  new vm.Script(PIECES, { filename: 'app.js#excerpt' }).runInContext(sandbox);
  return { sandbox, html, calls };
}

/** 让 setImmediate 链跑完（重试是串行的，等几轮 event loop 即可） */
const tick = () => new Promise((r) => setImmediate(r));
async function drain(rounds = 60) {
  for (let i = 0; i < rounds; i++) await tick();
}

let pass = 0, fail = 0;
function check(name, cond, extra) {
  if (cond) { pass++; console.log(`  PASS  ${name}`); }
  else { fail++; console.log(`  FAIL  ${name}${extra ? '  → ' + extra : ''}`); }
}

// ---------------------------------------------------------------- 场景 1
console.log('\n[1] 503 三次后恢复：应自动重试成功，且不依赖当前页');
{
  const { sandbox, html, calls } = makeSandbox((url, n) => {
    if (n <= 3) {
      const e = new Error('打分数据加载中，请稍候重试（2026-11-11）');
      e.status = 503;
      return Promise.reject(e);
    }
    return Promise.resolve({ code: '600519', name: '贵州茅台', score: 1.23 });
  });
  sandbox.loadStock('600519');
  await drain();

  check('失败时就记住了代码（切页回来能重拉）', sandbox.state.stockCode === '600519',
        `stockCode=${sandbox.state.stockCode}`);
  check('重试 3 次后拿到数据并渲染', calls.renderStock.length === 1,
        `renderStock=${calls.renderStock.length}`);
  check('请求次数 = 1 + 3 次重试', calls.fetches.length === 4, `fetches=${calls.fetches.length}`);
  check('成功后重试计数归零', sandbox.state.stockRetry === 0, `stockRetry=${sandbox.state.stockRetry}`);
  check('最终页面上不是"正在准备数据"', !/正在准备/.test(html.stockDetail));
}

// ---------------------------------------------------------------- 场景 2
console.log('\n[2] 失败后切到个股页：initStockPage 应自动重新拉取');
{
  const { sandbox, calls } = makeSandbox(() => {
    const e = new Error('数据加载中，请稍后');
    e.status = 503;
    return Promise.reject(e);
  });
  // 模拟"上次失败、已切走"，且重试已放弃（避免后台重试干扰本场景）
  sandbox.state.stockCode = '000001';
  sandbox.state.stock = null;
  sandbox.state.stockRetry = sandbox.STOCK_MAX_RETRY;
  const before = calls.fetches.length;
  sandbox.initStockPage();
  await drain(5);
  check('进入页面立刻发起了新请求', calls.fetches.length > before,
        `fetches ${before} → ${calls.fetches.length}`);
}

// ---------------------------------------------------------------- 场景 3
console.log('\n[3] 服务连不上（重启中）：应重试而不是直接报错');
{
  const { sandbox, html, calls } = makeSandbox(() =>
    Promise.reject(new Error('无法连接后端服务（127.0.0.1:8765）')));
  sandbox.loadStock('601727');
  await drain(5);
  check('没有走红框报错分支', calls.markDegraded === 0, `markDegraded=${calls.markDegraded}`);
  check('页面提示"正在等它恢复"', /正在等它恢复|正在准备/.test(html.stockDetail));
  check('已经发起了重试', calls.fetches.length >= 2, `fetches=${calls.fetches.length}`);
}

// ---------------------------------------------------------------- 场景 4
console.log('\n[4] 永久 503：重试有上限，最后给出可点的重试按钮');
{
  const { sandbox, html, calls } = makeSandbox(() => {
    const e = new Error('数据加载中，请稍后');
    e.status = 503;
    return Promise.reject(e);
  });
  sandbox.loadStock('600036');
  await drain(120);
  check('请求次数被限制在 1 + 上限', calls.fetches.length === sandbox.STOCK_MAX_RETRY + 1,
        `fetches=${calls.fetches.length} 上限=${sandbox.STOCK_MAX_RETRY}`);
  check('最终显示"连续 N 次没取到"', /连续 \d+ 次没取到/.test(html.stockDetail));
  check('给了 data-retry 按钮', /data-retry="stock"/.test(html.stockDetail));
}

console.log(`\n===== ${pass} passed, ${fail} failed =====`);
process.exit(fail ? 1 : 0);
