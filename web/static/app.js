/* =========================================================================
 * A股量化 · Pangolin — 前端逻辑（原生 ES2020，IIFE，无构建 / 无 CDN / 无外网请求）
 * 数据来源：同源后端 http://127.0.0.1:8765（契约见 docs/WEB_API.md）
 * ========================================================================= */
(function () {
  'use strict';

  /* ============================== 常量 ============================== */
  var PAGE_TITLES = {
    research: '研究',
    tracking: '跟踪',
    today: '今日推荐', monthly: '本月推荐', market: '看盘', backtest: '回测',
    stock: '个股', data: '数据', factors: '因子', reports: '报告', etf: 'ETF趋势', sector: '板块轮动',
    cb: '可转债',
    portfolio: '配置对比'
  };
  var INDEX_NAMES = { csi_all: '中证全指', hs300: '沪深300', csi500: '中证500', csi1000: '中证1000' };
  var BOARD_NAMES = { MAIN: '主板', GEM: '创业板', STAR: '科创板', BSE: '北交所', CHINEXT: '创业板', KCB: '科创板' };
  var PERIOD_NAMES = { full: '全区间', val: '验证期', oos: '样本外' };
  var BENCH_NAMES = {
    hs300: '沪深300', csi500: '中证500', csi1000: '中证1000',
    csi_all: '中证全指', all_eq: '全A等权'
  };
  var FUND_NAMES = {
    ep_ttm: 'EP（TTM）', bp: 'BP', sp_ttm: 'SP（TTM）', roe_ann: '年化 ROE',
    gross_margin: '毛利率', earnings_quality: '盈利现金含量',
    rev_growth: '营收增速', profit_growth: '利润增速',
    profit_accel: '利润增速加速度', rev_accel: '营收增速加速度',
    roe_change: 'ROE 变化', gm_change: '毛利率变化',
    accruals: '应计项目', margin_expansion: '利润率扩张', profit_qoq: '单季利润环比'
  };
  var SOURCE_NAMES = { ext: '扩展行情库（ext）', tencent: '腾讯行情（tencent）', baostock: 'BaoStock', local: '本地库' };
  var TOAST_MS = 3000;

  /* ============================== 状态 ============================== */
  var state = {
    activePage: 'market',
    meta: null,
    metaLoadedFlag: null,
    metaRetryTimer: null,
    metaFailCount: 0,
    market: null,
    recCache: {},
    recModelLoaded: null,
    models: [],
    defaultModel: null,
    analysisModel: null,
    recModel: null,
    btModelUsed: null,
    factors: null,
    reports: null,
    dataStatus: null,
    stock: null,
    stockCode: null,
    stockLoadSeq: 0,
    stockRetry: 0,
    monthly: null,
    suggestSeq: 0,
    suggestTimer: null,
    suggest: [],
    loaded: {
      market: false, backtest: false, monthly: false,
      stock: false, data: false, factors: false, reports: false
    },
    indexCode: 'csi_all',
    bt: { polling: false, jobId: null, startedAt: 0, lastProgress: 0, result: null, lastJob: null, payload: null },
    activeReqs: 0,
    degraded: false,
    lastRefreshAt: 0
  };

  /* ============================== DOM 小工具 ============================== */
  function byId(id) { return document.getElementById(id); }
  function $all(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function setHtml(id, html) { var el = byId(id); if (el) el.innerHTML = html; }
  function setText(id, text) { var el = byId(id); if (el) el.textContent = text; }
  function show(el, on) { if (el) el.hidden = !on; }

  function esc(s) {
    if (s === null || s === undefined) return '';
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function attr(s) { return esc(s).replace(/"/g, '&quot;'); }

  /* ============================== 格式化 ============================== */
  function isNil(v) {
    return v === null || v === undefined || v === '' || (typeof v === 'number' && !isFinite(v));
  }
  function toNum(v) {
    if (v === null || v === undefined || v === '') return 0;
    var n = Number(v);
    return isFinite(n) ? n : 0;
  }
  function fmtNum(v, digits) {
    if (isNil(v)) return '—';
    var n = Number(v);
    if (!isFinite(n)) return '—';
    digits = digits == null ? 2 : digits;
    try {
      return n.toLocaleString('zh-CN', { minimumFractionDigits: digits, maximumFractionDigits: digits });
    } catch (e) {
      return n.toFixed(digits);
    }
  }
  function fmtInt(v) {
    if (isNil(v)) return '—';
    var n = Number(v);
    if (!isFinite(n)) return '—';
    return fmtNum(Math.round(n), 0);
  }
  function fmtPct(v, d) {
    if (isNil(v)) return '—';
    var n = Number(v);
    if (!isFinite(n)) return '—';
    d = d == null ? 2 : d;
    var s = (n * 100).toFixed(d);
    if (n > 0) s = '+' + s;
    return s + '%';
  }
  function fmtPctU(v, d) {
    if (isNil(v)) return '—';
    var n = Number(v);
    if (!isFinite(n)) return '—';
    return (n * 100).toFixed(d == null ? 2 : d) + '%';
  }
  function fmtSigned(v, d) {
    if (isNil(v)) return '—';
    var n = Number(v);
    if (!isFinite(n)) return '—';
    d = d == null ? 4 : d;
    var s = n.toFixed(d);
    if (n > 0) s = '+' + s;
    return s;
  }
  function fmtMoney(v) {
    if (isNil(v)) return '—';
    var n = Number(v);
    if (!isFinite(n)) return '—';
    var abs = Math.abs(n);
    if (abs >= 1e12) return (n / 1e12).toFixed(2) + '万亿';
    if (abs >= 1e8) return (n / 1e8).toFixed(2) + '亿';
    if (abs >= 1e4) return (n / 1e4).toFixed(1) + '万';
    return n.toFixed(0);
  }
  function fmtPrice(v) {
    if (isNil(v)) return '—';
    var n = Number(v);
    if (!isFinite(n)) return '—';
    return fmtNum(n, 3);
  }
  function pctClass(v) {
    if (isNil(v)) return 'flat';
    var n = Number(v);
    if (!isFinite(n) || n === 0) return 'flat';
    return n > 0 ? 'up' : 'down';
  }
  function pctSpan(v, d) {
    return '<span class="mono ' + pctClass(v) + '">' + fmtPct(v, d) + '</span>';
  }
  function pctUSpan(v, d) {
    return '<span class="mono">' + fmtPctU(v, d) + '</span>';
  }
  function fmtDate(v) {
    if (isNil(v)) return '—';
    return String(v).slice(0, 10);
  }
  function fmtDateTime(v) {
    if (isNil(v)) return '—';
    var s = String(v).replace('T', ' ');
    return s.length >= 16 ? s.slice(0, 16) : s;
  }
  function fmtSize(bytes) {
    if (isNil(bytes)) return '—';
    var n = Number(bytes);
    if (!isFinite(n)) return '—';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1024 / 1024).toFixed(2) + ' MB';
  }
  function fmtDur(sec) {
    if (isNil(sec)) return '—';
    var n = Number(sec);
    if (!isFinite(n) || n < 0) return '—';
    if (n < 60) return n.toFixed(1) + 's';
    var m = Math.floor(n / 60), s = Math.round(n % 60);
    return m + 'm ' + s + 's';
  }
  function clamp(n, lo, hi) { return Math.max(lo, Math.min(hi, n)); }
  function withTs(url) {
    if (!url) return '';
    var ts = Date.now();
    return url + (url.indexOf('?') >= 0 ? '&' : '?') + '_=' + ts;
  }
  function boardName(b) { return b ? (BOARD_NAMES[b] || b) : '—'; }

  /* ============================== 通用 HTML 片段 ============================== */
  function badge(label, cls) {
    return '<span class="badge ' + (cls || '') + '">' + esc(label) + '</span>';
  }
  function dotMini(ok, title) {
    return '<span class="dot-mini ' + (ok ? 'ok' : 'bad') + '" title="' + attr(title || '') + '"></span>';
  }
  function kpi(label, valueHtml, subHtml, valueCls) {
    return '<div class="kpi">' +
      '<div class="kpi-label">' + esc(label) + '</div>' +
      '<div class="kpi-value ' + (valueCls || '') + '">' + (valueHtml == null ? '—' : valueHtml) + '</div>' +
      (subHtml ? '<div class="kpi-sub">' + subHtml + '</div>' : '') +
      '</div>';
  }
  function miniStat(label, valueHtml, subHtml, valueCls) {
    return '<div class="mini-stat">' +
      '<div class="ms-label">' + esc(label) + '</div>' +
      '<div class="ms-value ' + (valueCls || '') + '">' + (valueHtml == null ? '—' : valueHtml) + '</div>' +
      (subHtml ? '<div class="ms-sub">' + subHtml + '</div>' : '') +
      '</div>';
  }
  function pctBar(ratio) {
    var w = clamp(toNum(ratio) * 100, 0, 100);
    return '<div class="pct-bar"><div style="width:' + w.toFixed(1) + '%"></div></div>';
  }
  function loadingBlock(text) {
    return '<div class="loading-block"><span class="spinner"></span>' + esc(text || '加载中…') + '</div>';
  }
  function skeletonRows(n) {
    var s = '';
    for (var i = 0; i < (n || 4); i++) s += '<div class="skeleton" style="width:' + (90 - i * 8) + '%"></div>';
    return '<div style="padding:8px 2px">' + s + '</div>';
  }

  /**
   * 通用表格构建器。cols: [{label, num?, cls?, mono?}]；rows: 二维数组（单元格为 HTML 字符串）
   */
  function tableHtml(cols, rows, opts) {
    opts = opts || {};
    var h = '<table class="tbl"><thead><tr>';
    cols.forEach(function (c) {
      var cls = [];
      if (c.num) cls.push('num');
      if (c.mono) cls.push('mono');
      if (c.cls) cls.push(c.cls);
      h += '<th class="' + cls.join(' ') + '">' + esc(c.label) + '</th>';
    });
    h += '</tr></thead><tbody>';
    if (!rows || !rows.length) {
      h += '<tr><td colspan="' + cols.length + '"><div class="empty">' + esc(opts.empty || '暂无数据') + '</div></td></tr>';
    } else {
      rows.forEach(function (cells) {
        h += '<tr>';
        for (var i = 0; i < cols.length; i++) {
          var c = cols[i] || {};
          var cls = [];
          if (c.num) cls.push('num');
          if (c.mono) cls.push('mono');
          if (c.cls) cls.push(c.cls);
          var cell = cells[i];
          h += '<td class="' + cls.join(' ') + '">' + (cell === null || cell === undefined || cell === '' ? '—' : cell) + '</td>';
        }
        h += '</tr>';
      });
    }
    h += '</tbody></table>';
    return h;
  }

  function setChart(el, url, alt) {
    if (!el) return;
    el.classList.remove('img-failed');
    el.innerHTML = '<img alt="' + attr(alt || '图表') + '" loading="lazy" src="' + attr(withTs(url)) +
      '" onerror="this.parentNode.classList.add(\'img-failed\')">';
  }

  /* ============================== fetch 统一封装 ============================== */
  function reqStart() { state.activeReqs++; updateLoadState(); }
  function reqEnd() { state.activeReqs = Math.max(0, state.activeReqs - 1); updateLoadState(); }

  function updateLoadState() {
    var dot = byId('loadDot');
    var pill = byId('globalLoading');
    if (dot) {
      var cls = 'status-dot';
      if (state.activeReqs > 0) cls += ' busy';
      else if (state.degraded) cls += ' err';
      else cls += ' ok';
      dot.className = cls;
      dot.title = state.activeReqs > 0 ? '正在请求…' : (state.degraded ? '存在请求失败' : '正常');
    }
    if (pill) {
      if (state.activeReqs > 0) pill.classList.add('show');
      else pill.classList.remove('show');
    }
  }
  function markOk() { state.degraded = false; updateLoadState(); }
  function markDegraded() { state.degraded = true; updateLoadState(); }

  function apiFetch(path, opts) {
    opts = opts || {};
    var timeoutMs = opts.timeoutMs || 60000;
    var ctrl = new AbortController();
    var timedOut = false;
    var timer = setTimeout(function () { timedOut = true; ctrl.abort(); }, timeoutMs);
    var headers = Object.assign({ 'Accept': 'application/json' }, opts.headers || {});
    var init = {
      method: opts.method || 'GET',
      headers: headers,
      signal: ctrl.signal,
      cache: 'no-store'
    };
    if (opts.body !== undefined) {
      init.body = typeof opts.body === 'string' ? opts.body : JSON.stringify(opts.body);
      init.headers['Content-Type'] = 'application/json; charset=utf-8';
    }
    reqStart();
    return fetch(path, init).then(function (resp) {
      return resp.text().then(function (text) {
        var data = null;
        if (text) {
          try { data = JSON.parse(text); } catch (e) { data = null; }
        }
        if (!resp.ok) {
          var msg = (data && data.error) ? data.error : ('HTTP ' + resp.status + ' ' + (resp.statusText || ''));
          // 后端进程比前端旧时，新增的策略类型会被回成"未知策略类型"。
          // 这时该做的是重启服务，而不是去核对策略名——直接给出可操作提示。
          if (/未知策略类型/.test(String(msg))) {
            msg += '（服务端是旧版本，重启服务后生效）';
          }
          var err = new Error(String(msg));
          err.status = resp.status;
          throw err;
        }
        if (data === null && text) return text;
        return data;
      });
    }).catch(function (err) {
      if (err && (err.name === 'AbortError' || (err.name === 'DOMException' && timedOut))) {
        throw new Error('请求超时（' + Math.round(timeoutMs / 1000) + ' 秒）');
      }
      if (err && err.name === 'TypeError') {
        throw new Error('无法连接后端服务（127.0.0.1:8765）');
      }
      throw err instanceof Error ? err : new Error(String(err));
    }).finally(function () {
      clearTimeout(timer);
      reqEnd();
    });
  }
  function apiPost(path, body) {
    return apiFetch(path, { method: 'POST', body: body || {} });
  }

  /** 后端仍在后台加载数据时返回 503 —— 视为"服务加载中"而非错误 */
  function isServiceLoading(err) {
    if (!err) return false;
    if (err.status === 502 || err.status === 503 || err.status === 504) return true;
    return /加载中|loading|未就绪|not ready/i.test(String(err.message || ''));
  }

  /** 服务加载中的占位提示（页面顶部蓝条），待 /api/meta loaded=true 后自动重试 */
  function serviceLoadingPlaceholder(page, label) {
    var sec = byId('page-' + page);
    if (sec) {
      var old = sec.querySelector('.page-error');
      if (old && old.parentNode) old.parentNode.removeChild(old);
      var box = document.createElement('div');
      box.className = 'page-error notice loading-notice';
      box.innerHTML = '<span class="spinner"></span> 服务正在加载数据（通常 15–20 秒），完成后本页会自动重试。';
      sec.insertBefore(box, sec.children[1] || null);
    }
    if (page === 'market') {
      setHtml('marketIndices', '<div class="empty-block">服务加载中，稍候自动重试…</div>');
      setHtml('marketBreadth', '<div class="empty-block">—</div>');
      setHtml('industryTable', '<div class="empty">服务加载中…</div>');
      setHtml('marketBoards', '');
      setHtml('tickerTrack', '<span class="ticker-item muted">服务正在加载数据，稍候自动重试…</span>');
    } else if (false) {
      setHtml('recCards', '<div class="empty-block">服务加载中，稍候自动重试…</div>');
    } else if (page === 'factors') {
      setHtml('factorCounts', '<div class="empty-block">服务加载中，稍候自动重试…</div>');
    } else if (page === 'reports') {
      setHtml('reportList', '<div class="empty-block">服务加载中，稍候自动重试…</div>');
    } else if (page === 'data') {
      setHtml('dataStatus', '<div class="empty-block">服务加载中，稍候自动重试…</div>');
    }
    var srv = byId('srvText');
    if (srv) { srv.textContent = '服务加载中…'; srv.className = 'srv-text mono'; }
    toast((label || '服务') + '正在加载，稍候自动重试…', 'info');
  }

  /* ============================== toast ============================== */
  var lastToasts = {};
  function toast(msg, type, ms) {
    type = type || 'info';
    ms = ms == null ? TOAST_MS : ms;
    var key = type + '|' + msg;
    var now = Date.now();
    if (lastToasts[key] && now - lastToasts[key] < 4000) return;
    lastToasts[key] = now;
    var wrap = byId('toastWrap');
    if (!wrap) return;
    var el = document.createElement('div');
    el.className = 'toast ' + type;
    var ico = type === 'error' ? '!' : (type === 'ok' ? '✓' : 'i');
    el.innerHTML = '<span class="t-ico">' + ico + '</span><span>' + esc(msg) + '</span>';
    wrap.appendChild(el);
    setTimeout(function () {
      el.classList.add('hide');
      setTimeout(function () { if (el.parentNode) el.parentNode.removeChild(el); }, 320);
    }, ms);
  }

  /* ============================== 页面错误 / 重试 ============================== */
  function pageError(page, err) {
    var sec = byId('page-' + page);
    if (!sec) return;
    var old = sec.querySelector('.page-error');
    if (old && old.parentNode) old.parentNode.removeChild(old);
    var box = document.createElement('div');
    box.className = 'page-error error-box';
    box.innerHTML = '<b>加载失败：</b>' + esc((err && err.message) || String(err)) +
      ' <button class="btn btn-sm" data-retry="' + attr(page) + '">重试</button>';
    sec.insertBefore(box, sec.children[1] || null);
  }
  function clearPageError(page) {
    var sec = byId('page-' + page);
    if (!sec) return;
    var old = sec.querySelector('.page-error');
    if (old && old.parentNode) old.parentNode.removeChild(old);
  }

  /* ============================== 轮询任务 ============================== */
  /**
   * 轮询 /api/jobs/{id}，每 interval ms。
   * handlers: {onUpdate(job), onDone(job), onError(err, job?)}
   * 返回 {stop()}；网络抖动最多重试 5 次。
   */
  function pollJob(jobId, handlers, interval) {
    interval = interval || 1500;
    var alive = true;
    var failCount = 0;
    function tick() {
      if (!alive) return;
      apiFetch('/api/jobs/' + encodeURIComponent(jobId)).then(function (j) {
        if (!alive) return;
        failCount = 0;
        if (j && j.status === 'done') { alive = false; if (handlers.onDone) handlers.onDone(j); }
        else if (j && j.status === 'error') { alive = false; if (handlers.onError) handlers.onError((j.error || j.message || '任务失败'), j); }
        else {
          if (handlers.onUpdate) handlers.onUpdate(j || {});
          setTimeout(tick, interval);
        }
      }).catch(function (err) {
        if (!alive) return;
        failCount++;
        if (failCount >= 5) {
          alive = false;
          if (handlers.onError) handlers.onError(err);
        } else {
          if (handlers.onUpdate) handlers.onUpdate({ status: 'running', message: '任务状态查询失败，重试中（' + failCount + '/5）' });
          setTimeout(tick, 2000);
        }
      });
    }
    tick();
    return { stop: function () { alive = false; } };
  }

  /* ============================== 顶栏 / 元信息 ============================== */
  function loadMeta(force) {
    if (state.metaRetryTimer) { clearTimeout(state.metaRetryTimer); state.metaRetryTimer = null; }
    return apiFetch('/api/meta').then(function (d) {
      state.meta = d || {};
      state.metaFailCount = 0;
      renderMeta(state.meta);
      markOk();
    }).catch(function (err) {
      state.metaFailCount++;
      if (!isServiceLoading(err)) markDegraded();
      renderMetaFail(err);
      if (state.metaFailCount === 1) {
        if (isServiceLoading(err)) toast('服务正在加载数据（约 15–20 秒），页面会自动重试…', 'info');
        else toast('元信息加载失败：' + err.message, 'error');
      }
      scheduleMetaPoll(3000);
    });
  }
  function scheduleMetaPoll(delay) {
    if (state.metaRetryTimer) return;
    state.metaRetryTimer = setTimeout(function () {
      state.metaRetryTimer = null;
      loadMeta();
    }, delay || 2500);
  }
  function renderMetaFail(err) {
    var bar = byId('srvBar');
    var txt = byId('srvText');
    if (bar) { bar.classList.remove('indeterminate'); bar.style.width = '0%'; }
    if (txt) {
      txt.textContent = isServiceLoading(err) ? '服务加载中，稍候自动重试…' : '连接失败，自动重试…';
      txt.className = 'srv-text mono up';
      txt.title = (err && err.message) || '';
    }
    setText('metaDate', '—');
    setText('metaRefit', '—');
  }

  /* ---------- 多模型支持（/api/meta model.models；旧服务缺失时回退单模型） ---------- */
  function modelLabel(name) {
    if (!name) return '—';
    for (var i = 0; i < state.models.length; i++) {
      if (state.models[i].name === name) return state.models[i].label || name;
    }
    return name;
  }
  function modelDesc(name) {
    for (var i = 0; i < state.models.length; i++) {
      if (state.models[i].name === name) return state.models[i].desc || '';
    }
    return '';
  }
  function modelOptionText(m) {
    var label = m.label || m.name;
    return label + (m.name && label !== m.name ? '（' + m.name + '）' : '');
  }
  function modelOptionsHtml() {
    return state.models.map(function (m) {
      return '<option value="' + attr(m.name) + '" title="' + attr(m.desc || '') + '">' +
        esc(modelOptionText(m)) + '</option>';
    }).join('');
  }
  function syncModelSelects() {
    ['recModelSelect', 'stockModelSelect'].forEach(function (id) {
      var el = byId(id);
      if (!el) return;
      el.value = state.analysisModel || state.defaultModel || '';
      el.title = modelDesc(el.value) || '选择模型';
      el.style.display = state.models.length > 1 ? '' : 'none';
    });
    var bt = byId('btModel');
    if (bt) {
      var hint = byId('btModelHint');
      var cur = bt.value || state.defaultModel;
      if (hint) hint.textContent = modelDesc(cur) || '模型来自 /api/meta model.models';
      bt.title = modelDesc(cur) || '';
    }
  }
  function populateModels(model) {
    model = model || {};
    var list = Array.isArray(model.models) ? model.models : [];
    if (!list.length) {
      var fallbackName = model.name || 'wf_h10';
      list = [{ name: fallbackName, label: fallbackName, desc: '', pool: model.n_pool, horizon: model.horizon }];
    }
    state.models = list;
    state.defaultModel = model.default || model.name || list[0].name;
    var names = list.map(function (x) { return x.name; });
    if (!state.analysisModel || names.indexOf(state.analysisModel) < 0) state.analysisModel = state.defaultModel;
    state.recModel = state.analysisModel;
    var opts = modelOptionsHtml();
    ['recModelSelect', 'stockModelSelect'].forEach(function (id) {
      var el = byId(id);
      if (el) el.innerHTML = opts;
    });
    var bt = byId('btModel');
    if (bt) {
      var prefer = bt.value;
      if (names.indexOf(prefer) < 0) prefer = state.defaultModel;
      bt.innerHTML = opts;
      bt.value = prefer;
    }
    syncModelSelects();
  }
  function setAnalysisModel(name, reload) {
    if (!name) return;
    state.analysisModel = name;
    state.recModel = name;
    syncModelSelects();
    var desc = modelDesc(name);
    toast('已切换模型：' + modelLabel(name) + (desc ? ' — ' + desc : ''), 'info', 2800);
    if (!reload) return;
    
    else if (state.activePage === 'stock' && state.stockCode) loadStock(state.stockCode, true);
  }

  function renderMeta(d) {
    var data = d.data || {};
    var model = d.model || {};
    var server = d.server || {};
    setText('metaDate', fmtDate(data.last_date));
    setText('metaRefit', fmtDate(model.refit));
    setText('metaModel', model.name ? (model.name + (model.horizon != null ? ' · H' + model.horizon : '')) : '—');
    setText('metaUniverse', data.n_universe != null ? fmtInt(data.n_universe) : '—');

    var loaded = server.loaded === true;
    var bar = byId('srvBar');
    var txt = byId('srvText');
    if (bar && txt) {
      if (loaded) {
        bar.classList.remove('indeterminate');
        bar.style.width = '100%';
        txt.className = 'srv-text mono';
        txt.textContent = '已就绪' + (server.loaded_seconds != null ? ' · ' + fmtNum(server.loaded_seconds, 1) + 's' : '');
        txt.title = '数据已加载，服务可用';
      } else {
        bar.classList.add('indeterminate');
        bar.style.width = '40%';
        txt.className = 'srv-text mono';
        txt.textContent = '数据加载中…';
        txt.title = '服务器正在加载数据，部分接口可能暂时不可用';
      }
    }

    var wasLoaded = state.metaLoadedFlag === true;
    state.metaLoadedFlag = loaded;

    var side = [];
    if (data.n_dates != null) side.push('日线 ' + fmtInt(data.n_dates) + ' 日');
    if (data.n_codes != null) side.push('股票 ' + fmtInt(data.n_codes) + ' 只');
    if (data.n_universe != null) side.push('选股域 ' + fmtInt(data.n_universe) + ' 只');
    if (data.source) side.push('源 ' + (SOURCE_NAMES[data.source] || data.source));
    side.push('刷新 ' + new Date().toTimeString().slice(0, 8));
    setHtml('sideInfo', side.join('<br>'));
    state.lastRefreshAt = Date.now();

    // 回测结束日期默认值 / 上限
    var endInput = byId('btEnd');
    if (endInput) {
      if (data.last_date) {
        if (!endInput.value) endInput.value = data.last_date;
        endInput.max = data.last_date;
        endInput.title = '默认数据最后交易日 ' + data.last_date;
      }
    }
    var startInput = byId('btStart');
    if (startInput && data.last_date) startInput.max = data.last_date;

    populateModels(model);

    // 数据更新中提示
    var chip = byId('metaDate');
    if (chip && data.update_running) chip.title = '数据更新进行中，完成后请刷新页面';

    // 服务端数据加载完成后，自动重试此前因 503 失败的当前页 / 看盘
    if (!wasLoaded && loaded) {
      if (!state.loaded.market) loadMarket(true);
      if (state.activePage !== 'market' && !state.loaded[state.activePage]) ensurePage(state.activePage, true);
    }
  }

  /* ============================== 看盘 ============================== */
  function loadMarket(force) {
    if (state.loaded.market && !force) return Promise.resolve();
    if (!state.loaded.market) {
      setHtml('marketIndices', skeletonRows(3));
      setHtml('marketBreadth', skeletonRows(4));
      setHtml('industryTable', skeletonRows(5));
      setHtml('marketBoards', '');
    }
    return apiFetch('/api/market/overview').then(function (d) {
      state.market = d || {};
      state.loaded.market = true;
      clearPageError('market');
      renderMarket(state.market);
      markOk();
    }).catch(function (err) {
      if (isServiceLoading(err)) { serviceLoadingPlaceholder('market', '市场数据'); return; }
      markDegraded();
      pageError('market', err);
      setHtml('marketIndices', '<div class="empty-block">市场数据加载失败，请稍后点“刷新”或“重试”。</div>');
      setHtml('marketBreadth', '<div class="empty-block">—</div>');
      setHtml('industryTable', '<div class="empty">暂无数据</div>');
      setHtml('marketBoards', '');
      setHtml('tickerTrack', '<span class="ticker-item muted">市场数据加载失败：' + esc(err.message) + '</span>');
      toast('市场数据加载失败：' + err.message, 'error');
    });
  }

  function renderMarket(d) {
    setText('marketAsOf', d.asof ? '数据截至 ' + d.asof : '');
    setHtml('marketIndices', renderIndices(d.indices));
    setHtml('marketBreadth', renderBreadth(d.breadth, d.regime));
    setHtml('marketBoards', renderBoards(d));
    setHtml('industryTable', renderIndustries(d.industries));
    renderTicker(d);
    drawIndexChart();
  }

  function renderIndices(list) {
    var codes = ['csi_all', 'hs300', 'csi500', 'csi1000'];
    var map = {};
    (list || []).forEach(function (it) { if (it && it.code) map[it.code] = it; });
    var html = '';
    codes.forEach(function (code) {
      var it = map[code];
      if (!it) {
        html += '<div class="card idx-card"><div class="idx-top"><span class="idx-name">' +
          esc(INDEX_NAMES[code] || code) + '</span></div><div class="idx-close mono">—</div>' +
          '<div class="idx-foot"><span>暂无数据</span></div></div>';
        return;
      }
      var above = it.above_ma120;
      var name = it.name || INDEX_NAMES[code] || code;
      html += '<div class="card idx-card">' +
        '<div class="idx-top"><span class="idx-name">' + esc(name) +
        (it.code && it.code !== code ? ' <span class="idx-code">' + esc(code) + '</span>' : '') +
        '</span><span class="trend-dot ' + (above ? 'ok' : 'bad') + '" title="' +
        attr(above ? '收盘站上 MA120（强势）' : '收盘跌破 MA120（弱势）') + '"></span></div>' +
        '<div class="idx-close mono">' + fmtNum(it.close, 2) + '</div>' +
        '<div class="idx-chg">' +
        '<span class="chg ' + pctClass(it.chg_1d) + '">' + fmtPct(it.chg_1d) + '<i>1D</i></span>' +
        '<span class="chg ' + pctClass(it.chg_20d) + '">' + fmtPct(it.chg_20d) + '<i>20D</i></span>' +
        (it.chg_120d != null ? '<span class="chg ' + pctClass(it.chg_120d) + '">' + fmtPct(it.chg_120d) + '<i>120D</i></span>' : '') +
        '</div>' +
        '<div class="idx-foot"><span>MA120 <b>' + fmtNum(it.ma120, 2) + '</b></span>' +
        '<span class="trend-txt ' + (above ? 'up' : 'down') + '">' + (above ? '站上 MA120' : '跌破 MA120') + '</span></div>' +
        '</div>';
    });
    return html;
  }

  function renderBreadth(b, regime) {
    if (!b) return '<div class="empty">暂无市场宽度数据</div>';
    var up = toNum(b.up), down = toNum(b.down), flat = toNum(b.flat);
    var total = up + down + flat || 1;
    var html = '<div class="breadth-bar">' +
      '<div class="bb-up" style="width:' + (up / total * 100).toFixed(1) + '%"></div>' +
      '<div class="bb-flat" style="width:' + (flat / total * 100).toFixed(1) + '%"></div>' +
      '<div class="bb-down" style="width:' + (down / total * 100).toFixed(1) + '%"></div>' +
      '</div>' +
      '<div class="breadth-nums">' +
      '<span class="up">上涨 ' + fmtInt(up) + '</span>' +
      '<span class="flat">平 ' + fmtInt(flat) + '</span>' +
      '<span class="down">下跌 ' + fmtInt(down) + '</span>' +
      '</div>';

    var ratio = b.amount_ratio;
    var ratioHtml = ratio == null ? '—' : ('环比 ' + pctSpan(toNum(ratio) - 1, 1));
    var coeff = regime && regime.coeff != null ? regime.coeff : (regime && regime.regime_coeff);
    var condA = regime ? regime.csi_all_above_MA120 : null;
    var condB = regime ? regime.hs300_above_MA60 : null;
    var regimeSub = '';
    if (regime) {
      regimeSub = dotMini(!!condA, '中证全指 vs MA120') + ' 全指 · ' + dotMini(!!condB, '沪深300 vs MA60') + ' 300';
    }

    html += '<div class="mini-grid">' +
      miniStat('涨停', fmtInt(b.limit_up), '跌停 <span class="mono down">' + fmtInt(b.limit_down) + '</span>', 'up') +
      miniStat('中位涨幅', fmtPct(b.median_ret), '个股涨跌幅中位数', pctClass(b.median_ret)) +
      miniStat('总成交额', fmtMoney(b.total_amount), ratioHtml) +
      miniStat('MA20 上方占比', fmtPctU(b.pct_above_ma20, 1),
        (b.n_above_ma20 != null ? fmtInt(b.n_above_ma20) + ' / ' + fmtInt(b.n_universe) + ' 只' : '') + pctBar(b.pct_above_ma20)) +
      miniStat('Regime 系数', fmtNum(coeff, 2), regimeSub || '趋势仓位系数') +
      miniStat('样本', fmtInt(b.n_universe) + ' 只', '成交额环比 ' + (ratio == null ? '—' : fmtNum(ratio, 2) + 'x')) +
      '</div>';
    return html;
  }

  function renderBoards(d) {
    function board(title, rows, lastCol) {
      var cols = [
        { label: '代码', mono: true }, { label: '名称' }, { label: '行业' }, { label: '涨跌幅', num: true },
        lastCol === 'amount' ? { label: '成交额', num: true } : { label: '价格', num: true }
      ];
      return '<div class="card"><div class="card-hd"><h4>' + esc(title) + '</h4>' +
        '<span class="muted small">' + (rows && rows.length ? rows.length + ' 只' : '—') + '</span></div>' +
        '<div class="card-bd table-wrap" style="max-height:340px">' +
        tableHtml(cols, (rows || []).slice(0, 12).map(function (r) {
          return [
            esc(r.code), '<span class="tbl-name">' + esc(r.name) + '</span>', esc(r.industry),
            pctSpan(r.chg_1d),
            lastCol === 'amount' ? fmtMoney(r.amount) : fmtPrice(r.price)
          ];
        }), { empty: '暂无数据' }) + '</div></div>';
    }
    return board('涨幅榜', d.top_gainers, 'price') +
      board('跌幅榜', d.top_losers, 'price') +
      board('成交额榜', d.top_amount, 'amount');
  }

  function heatCell(v, scale) {
    if (isNil(v)) return '—';
    var n = toNum(v);
    var alpha = clamp(0.10 + Math.abs(n) * (scale || 2.6), 0.10, 0.62);
    var color = n >= 0 ? '239,68,68' : '34,197,94';
    return '<span class="heat-cell" style="background:rgba(' + color + ',' + alpha.toFixed(2) + ');color:' +
      (n >= 0 ? 'var(--up)' : 'var(--down)') + '">' + fmtPct(v) + '</span>';
  }

  function renderIndustries(list) {
    var rows = (list || []).slice().sort(function (a, b) { return toNum(b.ret_20d) - toNum(a.ret_20d); });
    return tableHtml([
      { label: '#', num: true }, { label: '行业' }, { label: '家数', num: true },
      { label: '1D', num: true }, { label: '20D', num: true }, { label: '60D', num: true },
      { label: '成交额', num: true }
    ], rows.map(function (r, i) {
      return [
        String(i + 1), '<span class="tbl-name">' + esc(r.industry) + '</span>', fmtInt(r.n),
        pctSpan(r.ret_1d), heatCell(r.ret_20d), pctSpan(r.ret_60d), fmtMoney(r.amount)
      ];
    }), { empty: '暂无行业数据' });
  }

  function renderTicker(d) {
    var items = [];
    (d.indices || []).forEach(function (it) {
      items.push('<span class="ticker-item"><span class="tk-name">' + esc(it.name || INDEX_NAMES[it.code] || it.code) + '</span>' +
        '<span class="tk-val mono">' + fmtNum(it.close, 2) + '</span>' +
        '<span class="mono ' + pctClass(it.chg_1d) + '">' + fmtPct(it.chg_1d) + '</span></span>');
    });
    var b = d.breadth || {};
    items.push('<span class="ticker-item"><span class="tk-name">上涨</span><span class="tk-val mono up">' + fmtInt(b.up) + '</span>' +
      '<span class="tk-name">下跌</span><span class="tk-val mono down">' + fmtInt(b.down) + '</span></span>');
    items.push('<span class="ticker-item"><span class="tk-name">涨停</span><span class="tk-val mono up">' + fmtInt(b.limit_up) + '</span>' +
      '<span class="tk-name">跌停</span><span class="tk-val mono down">' + fmtInt(b.limit_down) + '</span></span>');
    items.push('<span class="ticker-item"><span class="tk-name">中位涨幅</span><span class="mono ' + pctClass(b.median_ret) + '">' + fmtPct(b.median_ret) + '</span></span>');
    items.push('<span class="ticker-item"><span class="tk-name">成交额</span><span class="tk-val mono">' + fmtMoney(b.total_amount) + '</span></span>');
    items.push('<span class="ticker-item"><span class="tk-name">MA20上方</span><span class="mono">' + fmtPctU(b.pct_above_ma20, 1) + '</span></span>');
    if (d.asof) items.push('<span class="ticker-item"><span class="tk-name">数据日</span><span class="mono">' + esc(d.asof) + '</span></span>');
    setHtml('tickerTrack', items.join('<span class="ticker-sep">|</span>'));
  }

  function drawIndexChart() {
    var box = byId('indexChart');
    if (!box) return;
    setChart(box, '/api/chart/index_line?code=' + encodeURIComponent(state.indexCode) + '&days=250',
      '指数走势 ' + (INDEX_NAMES[state.indexCode] || state.indexCode));
  }

  /* ============================== 推荐 ============================== */
  function loadRecommend(force, model) {
    var m = model || state.analysisModel || state.defaultModel || 'wf_h10';
    state.recModel = m;
    var cached = state.recCache[m];
    if (cached && !force) {
      state.recommend = cached;
      state.loaded.recommend = true;
      state.recModelLoaded = m;
      clearPageError('recommend');
      renderRecommend(cached);
      syncModelSelects();
      markOk();
      return Promise.resolve();
    }
    if (!cached) {
      setHtml('recCards', skeletonRows(3));
      setHtml('recHoldings', skeletonRows(4));
      setHtml('recOrders', skeletonRows(4));
      setHtml('recCandidates', skeletonRows(5));
      setHtml('recNotes', '');
      if (state.models.length > 1) {
        toast('正在计算「' + modelLabel(m) + '」推荐（首次约 20–60 秒）…', 'info', 4500);
      }
    }
    var url = '/api/recommend?top=60' + (m ? '&model=' + encodeURIComponent(m) : '');
    return apiFetch(url, { timeoutMs: 120000 }).then(function (d) {
      d = d || {};
      if (!d.model) d.model = m;
      state.recCache[m] = d;
      state.recommend = d;
      state.loaded.recommend = true;
      state.recModelLoaded = m;
      clearPageError('recommend');
      renderRecommend(d);
      syncModelSelects();
      markOk();
    }).catch(function (err) {
      if (isServiceLoading(err)) {
        // 后端在算打分（首次 20–40 秒）时会回 503，而不是让请求挂到超时。
        // 这里 3 秒重试一次，直到拿到数据 —— 用户看到的是"正在计算"，不是"失败"。
        serviceLoadingPlaceholder('recommend', '推荐数据');
        setTimeout(function () {
          if (state.activePage === 'recommend') loadRecommend(force, m);
        }, 3000);
        return;
      }
      markDegraded();
      pageError('recommend', err);
      toast('推荐数据加载失败：' + err.message, 'error');
    });
  }

  function renderRecommend(d) {
    var regime = d.regime || {};
    var coeff = regime.coeff != null ? regime.coeff : regime.regime_coeff;
    var curModel = d.model || state.recModel || state.analysisModel;
    setText('recAsOf', '模型 ' + modelLabel(curModel) + ' · 信号日 ' + fmtDate(d.asof) +
      ' · refit ' + fmtDate(d.refit) + (d.capital != null ? ' · 资金 ' + fmtMoney(d.capital) : ''));

    var cashPct = d.cash_pct != null ? d.cash_pct : (d.invested_pct != null ? 1 - toNum(d.invested_pct) : null);
    var cards = [];

    cards.push(kpi('目标仓位 / 现金',
      '<span class="up">' + fmtPctU(d.invested_pct, 2) + '</span>' +
      '<span class="muted" style="font-size:13px"> / </span>' +
      '<span class="flat">' + fmtPctU(cashPct, 2) + '</span>',
      '投入 <span class="mono">' + fmtMoney(d.invested_value) + '</span> · 资金 <span class="mono">' + fmtMoney(d.capital) + '</span>',
      ''));

    cards.push(kpi('Regime 系数',
      fmtNum(coeff, 2) + (coeff != null ? '<span class="muted" style="font-size:11px"> / 1.00</span>' : ''),
      '<div class="kv-row"><span class="k">中证全指 &gt; MA120</span><span class="v">' +
      dotMini(!!regime.csi_all_above_MA120, '趋势条件 A') + ' ' + (regime.csi_all_above_MA120 ? '<span class="up">是</span>' : '<span class="down">否</span>') +
      '</span></div>' +
      '<div class="kv-row"><span class="k">沪深300 &gt; MA60</span><span class="v">' +
      dotMini(!!regime.hs300_above_MA60, '趋势条件 B') + ' ' + (regime.hs300_above_MA60 ? '<span class="up">是</span>' : '<span class="down">否</span>') +
      '</span></div>', ''));

    var sleeve = d.cash_sleeve || {};
    cards.push(kpi('现金替代（80 / 20）',
      '<span class="mono">' + fmtPctU(sleeve.bond_weight, 0) + '</span>' +
      '<span class="muted" style="font-size:13px"> / </span>' +
      '<span class="mono">' + fmtPctU(sleeve.gold_weight, 0) + '</span>',
      '<div class="kv-row"><span class="k">国债</span><span class="v">' + fmtMoney(sleeve.bond_value) + '</span></div>' +
      '<div class="kv-row"><span class="k">黄金</span><span class="v">' + fmtMoney(sleeve.gold_value) + '</span></div>' +
      '<div class="kv-row"><span class="k">现金合计</span><span class="v">' + fmtMoney(sleeve.cash_value != null ? sleeve.cash_value : d.invested_value == null ? null : toNum(d.capital) - toNum(d.invested_value)) + '</span></div>',
      ''));

    cards.push(kpi('信号日 / refit',
      '<span class="mono">' + fmtDate(d.asof) + '</span>',
      'refit <span class="mono">' + fmtDate(d.refit) + '</span> · 持仓 <span class="mono">' + fmtInt(d.n_holdings) + '</span> 只' +
      (sleeve.instruments && sleeve.instruments.length
        ? '<div class="kv-row"><span class="k">现金工具</span><span class="v">' + sleeve.instruments.map(function (s) { return esc(s); }).join(' / ') + '</span></div>'
        : ''),
      ''));

    setHtml('recCards', cards.join(''));
    renderHoldings(d);
    renderOrders(d);
    setHtml('recNotes', (d.notes && d.notes.length)
      ? '<ul class="note-list">' + d.notes.map(function (n) { return '<li>' + esc(n) + '</li>'; }).join('') + '</ul>'
      : '<div class="empty">暂无说明</div>');
    renderCandidates();
  }

  function renderHoldings(d) {
    var list = (d.holdings || []).slice().sort(function (a, b) { return toNum(a.rank) - toNum(b.rank); });
    function hAmount(r) { return r.amount != null ? r.amount : r.target_value; }
    setText('recHoldingsCount', list.length + ' 只 · 合计 ' + fmtMoney(list.reduce(function (s, r) { return s + toNum(hAmount(r)); }, 0)));
    var cols = [
      { label: '#', num: true }, { label: '代码', mono: true }, { label: '名称' }, { label: '行业' },
      { label: '板块' }, { label: '权重', num: true }, { label: '价格', num: true },
      { label: '股数', num: true }, { label: '金额', num: true }
    ];
    var rows = list.map(function (r) {
      return [
        fmtInt(r.rank), esc(r.code), '<span class="tbl-name">' + esc(r.name) + '</span>', esc(r.industry),
        boardName(r.board), fmtPctU(r.weight, 2), fmtPrice(r.price), fmtInt(r.shares), fmtMoney(hAmount(r))
      ];
    });
    var weightSum = list.reduce(function (s, r) { return s + toNum(r.weight); }, 0);
    var amountSum = list.reduce(function (s, r) { return s + toNum(hAmount(r)); }, 0);
    var html = tableHtml(cols, rows, { empty: '当前无目标持仓' });
    if (rows.length) {
      html = html.replace('</tbody>', '</tbody><tfoot><tr><td>合计</td><td>—</td><td>' + esc(list.length + ' 只') +
        '</td><td>—</td><td>—</td><td class="num">' + fmtPctU(weightSum, 2) + '</td><td>—</td><td>—</td>' +
        '<td class="num">' + fmtMoney(amountSum) + '</td></tr></tfoot>');
    }
    setHtml('recHoldings', html);
  }

  function renderOrders(d) {
    var list = d.orders || [];
    setText('recOrdersCount', list.length + ' 笔');
    var reasonMap = {
      initial_build: '建仓', rebalance: '调仓', stop_loss: '止损', overlay: '风控减仓',
      take_profit: '止盈', band: '带宽内跳过'
    };
    var cols = [
      { label: '代码', mono: true }, { label: '名称' }, { label: '方向' },
      { label: '股数', num: true }, { label: '手数', num: true }, { label: '参考价', num: true },
      { label: '金额', num: true }, { label: '原因' }
    ];
    var rows = list.map(function (r) {
      var side = String(r.side || '').toLowerCase();
      var sideHtml = side === 'buy' ? '<span class="badge up">买入</span>'
        : (side === 'sell' ? '<span class="badge down">卖出</span>' : esc(r.side));
      return [
        esc(r.code), '<span class="tbl-name">' + esc(r.name) + '</span>', sideHtml,
        fmtInt(r.shares), fmtInt(r.lots), fmtPrice(r.price_ref), fmtMoney(r.amount),
        esc(reasonMap[r.reason] || r.reason || '—')
      ];
    });
    setHtml('recOrders', tableHtml(cols, rows, { empty: '当前无待执行订单' }));
  }

  function renderCandidates() {
    var d = state.recommend || null;
    var box = byId('recCandidates');
    if (!box) return;
    if (!d) return;
    var list = d.candidates || [];
    var onlyTarget = byId('candTargetOnly') && byId('candTargetOnly').checked;
    var q = (byId('candSearch') && byId('candSearch').value || '').trim().toLowerCase();
    var filtered = list.filter(function (r) {
      if (onlyTarget && !r.in_target) return false;
      if (!q) return true;
      return String(r.code || '').toLowerCase().indexOf(q) >= 0 ||
        String(r.name || '').toLowerCase().indexOf(q) >= 0 ||
        String(r.industry || '').toLowerCase().indexOf(q) >= 0;
    });
    var shown = filtered.slice(0, 60);
    setText('recCandCount', '显示 ' + shown.length + ' / 共 ' + list.length);
    var cols = [
      { label: '#', num: true }, { label: '代码', mono: true }, { label: '名称' }, { label: '行业' },
      { label: '板块' }, { label: '复合分', num: true }, { label: '价格', num: true },
      { label: '目标内' }, { label: '说明' }
    ];
    var rows = shown.map(function (r, i) {
      return [
        fmtInt(r.rank != null ? r.rank : i + 1), esc(r.code), '<span class="tbl-name">' + esc(r.name) + '</span>',
        esc(r.industry), boardName(r.board), fmtNum(r.score, 3), fmtPrice(r.price),
        r.in_target ? '<span class="badge accent">目标内</span>' : '<span class="muted">—</span>',
        r.in_target ? '<span class="muted">已入选</span>' : esc(r.excluded_reason || '—')
      ];
    });
    setHtml('recCandidates', tableHtml(cols, rows, { empty: onlyTarget || q ? '无匹配候选（检查过滤条件）' : '暂无候选数据' }));
  }

  /* ============================== 回测 ============================== */
  function applyBtDefaults() {
    var end = byId('btEnd');
    var meta = state.meta;
    if (end && !end.value && meta && meta.data && meta.data.last_date) end.value = meta.data.last_date;
  }

  function initBacktestPage() {
    applyBtDefaults();
    state.loaded.backtest = true;
    if (state.bt.lastJob && state.bt.polling) {
      show(byId('btProgressCard'), true);
      renderBtProgress(state.bt.lastJob);
    }
  }

  function formNum(id, fallback) {
    var el = byId(id);
    if (!el) return fallback;
    var v = parseFloat(el.value);
    return isFinite(v) ? v : fallback;
  }
  function formInt(id, fallback) {
    var v = formNum(id, fallback);
    return Math.round(v);
  }

  function readBtForm() {
    var modelEl = byId('btModel');
    var startEl = byId('btStart');
    var endEl = byId('btEnd');
    var start = startEl ? startEl.value : '';
    var end = endEl ? endEl.value : '';
    if (!start) start = '2020-01-01';
    if (!end) end = (state.meta && state.meta.data && state.meta.data.last_date) || '';
    if (!end) {
      toast('缺少结束日期：请等待元信息加载完成后重试，或手动填写结束日期', 'error');
      return null;
    }
    if (start > end) {
      toast('开始日期不能晚于结束日期', 'error');
      return null;
    }
    var nHoldings = formInt('btNHoldings', 40);
    if (nHoldings < 1 || nHoldings > 200) {
      toast('持仓数需在 1–200 之间', 'error');
      return null;
    }
    var bond = formNum('btCashBond', 0.8);
    var gold = formNum('btCashGold', 0.2);
    if (bond < 0 || gold < 0 || bond + gold > 1.0001) {
      toast('现金替代权重需 ≥ 0 且合计 ≤ 1', 'error');
      return null;
    }
    var bandRel = formNum('btBandRel', 0.3);
    if (bandRel < 0 || bandRel > 1) {
      toast('band_rel 需在 0–1 之间', 'error');
      return null;
    }
    return {
      model: modelEl ? modelEl.value : 'wf_h10',
      n_holdings: nHoldings,
      rebalance_days: Math.max(1, formInt('btRebalance', 10)),
      cash_bond: bond,
      cash_gold: gold,
      overlay: !!(byId('btOverlay') && byId('btOverlay').checked),
      stop_loss: !!(byId('btStopLoss') && byId('btStopLoss').checked),
      start: start,
      end: end,
      max_new: Math.max(1, formInt('btMaxNew', 6)),
      band_rel: bandRel,
      band_abs: Math.max(0, formNum('btBandAbs', 6000))
    };
  }

  function setBtProgress(progress, message, elapsed, running) {
    var bar = byId('btBar');
    var p = clamp(toNum(progress), 0, 1);
    if (bar) {
      bar.style.width = (p * 100).toFixed(1) + '%';
      if (running) bar.classList.add('running');
      else bar.classList.remove('running');
    }
    setText('btStatus', message || '—');
    var sec = elapsed != null ? elapsed : (state.bt.startedAt ? (Date.now() - state.bt.startedAt) / 1000 : null);
    setText('btElapsed', sec != null ? '已用 ' + fmtDur(sec) : '');
  }

  function renderBtProgress(j) {
    state.bt.lastJob = j;
    var p = j && j.progress != null ? j.progress : state.bt.lastProgress;
    state.bt.lastProgress = clamp(toNum(p), 0, 1);
    var msg = (j && j.message) || (j && j.status === 'done' ? '回测完成' : '回测运行中…');
    setBtProgress(state.bt.lastProgress, msg, j && j.elapsed, !!(j && j.status === 'running'));
  }

  function runBacktest() {
    if (state.bt.polling) {
      toast('回测正在运行中，请稍候…', 'info');
      return;
    }
    var payload = readBtForm();
    if (!payload) return;
    state.bt.result = null;
    state.bt.lastProgress = 0;
    state.bt.payload = payload;
    show(byId('btResult'), false);
    show(byId('btError'), false);
    show(byId('btProgressCard'), true);
    setBtProgress(0, '正在提交回测任务…', 0, true);
    var btn = byId('btnRunBacktest');
    if (btn) { btn.disabled = true; btn.textContent = '回测运行中…'; }
    apiPost('/api/backtest', payload).then(function (d) {
      var id = d && (d.job_id || d.id);
      if (!id) throw new Error('后端未返回 job_id');
      state.bt.polling = true;
      state.bt.jobId = id;
      state.bt.startedAt = Date.now();
      toast('回测任务已提交，运行中（约 2–8 秒）', 'info');
      pollJob(id, {
        onUpdate: renderBtProgress,
        onDone: btDone,
        onError: function (err, job) { btFail(err, job); }
      }, 1500);
    }).catch(function (err) {
      if (isServiceLoading(err)) { btFail(new Error('服务正在加载数据（约 15–20 秒），请稍候重试')); return; }
      btFail(err);
    });
  }

  function btDone(j) {
    state.bt.polling = false;
    var btn = byId('btnRunBacktest');
    if (btn) { btn.disabled = false; btn.textContent = '运行回测'; }
    state.bt.jobId = (j && j.id) || state.bt.jobId;
    setBtProgress(1, '回测完成', j && j.elapsed, false);
    var res = j && j.result;
    if (!res) {
      btFail(new Error('任务完成但未返回回测结果（result 为空）'));
      return;
    }
    state.bt.result = res;
    state.btModelUsed = (res.params && res.params.model) ||
      (state.bt.payload && state.bt.payload.model) || state.btModelUsed;
    try {
      renderBtResult(res, state.bt.jobId);
    } catch (e) {
      btFail(e);
      return;
    }
    toast('回测完成', 'ok');
    markOk();
    setTimeout(function () {
      if (!state.bt.polling) show(byId('btProgressCard'), false);
    }, 2600);
  }

  function btFail(err, j) {
    state.bt.polling = false;
    var btn = byId('btnRunBacktest');
    if (btn) { btn.disabled = false; btn.textContent = '运行回测'; }
    var bar = byId('btBar');
    if (bar) bar.classList.remove('running');
    setText('btStatus', '回测失败');
    var msg = (j && (j.error || j.message)) || (err && err.message) || String(err) || '未知错误';
    var box = byId('btError');
    if (box) {
      box.innerHTML = '<b>回测失败：</b>' + esc(msg) +
        '<div class="muted small" style="margin-top:4px">请检查起止日期 / 持仓数 / 现金权重等参数；修改后可直接重试。</div>';
      box.hidden = false;
    }
    show(byId('btResult'), false);
    toast('回测失败：' + msg, 'error', 5000);
    markDegraded();
  }

  function renderBtResult(res, jobId) {
    var m = res.metrics || {};
    setText('btRange', (m.start || '—') + ' ~ ' + (m.end || '—'));

    var defs = [
      { label: 'CAGR 年化收益', v: fmtPct(m.cagr), cls: pctClass(m.cagr) },
      { label: '年化波动', v: fmtPctU(m.ann_vol), cls: 'flat' },
      { label: 'Sharpe', v: fmtNum(m.sharpe, 3), cls: '' },
      { label: 'Sortino', v: fmtNum(m.sortino, 3), cls: '' },
      { label: '最大回撤', v: fmtPct(m.max_dd), cls: 'down' },
      { label: 'Calmar', v: fmtNum(m.calmar, 3), cls: '' },
      { label: '月胜率', v: fmtPctU(m.win_rate_monthly), cls: toNum(m.win_rate_monthly) >= 0.5 ? 'up' : 'flat' },
      { label: '年化换手', v: fmtNum(m.turnover_ann, 2) + 'x', cls: '' },
      { label: '成本占比', v: fmtPctU(m.cost_ratio), cls: 'flat' },
      { label: '交易笔数', v: fmtInt(m.n_trades), cls: '' }
    ];
    setHtml('btMetrics', defs.map(function (x) { return kpi(x.label, x.v, null, x.cls); }).join(''));

    var p = res.params || state.bt.payload || {};
    var sub = [];
    var btModel = p.model || state.btModelUsed || (byId('btModel') && byId('btModel').value);
    if (btModel) sub.push('模型 ' + modelLabel(btModel));
    if (m.avg_holdings != null) sub.push('平均持仓 ' + fmtNum(m.avg_holdings, 1) + ' 只');
    if (m.years != null) sub.push('回测年数 ' + fmtNum(m.years, 2));
    if (p.n_holdings != null) sub.push('持仓数 ' + fmtInt(p.n_holdings));
    if (p.rebalance_days != null) sub.push('调仓 ' + fmtInt(p.rebalance_days) + ' 日');
    if (p.cash_bond != null) sub.push('现金替代 ' + fmtPctU(p.cash_bond, 0) + ' 国债 / ' + fmtPctU(p.cash_gold, 0) + ' 黄金');
    if (p.overlay != null) sub.push('overlay ' + (p.overlay ? '开' : '关'));
    if (p.stop_loss != null) sub.push('止损 ' + (p.stop_loss ? '开' : '关'));
    if (p.band_rel != null) sub.push('band ' + fmtNum(p.band_rel, 2) + ' / ' + fmtCur(p.band_abs));
    setText('btSubInfo', sub.join('  ·  '));

    var chartUrl = res.chart_url || (jobId ? '/api/chart/backtest_nav/' + encodeURIComponent(jobId) : '');
    if (chartUrl) setChart(byId('btChart'), chartUrl, '回测净值与回撤');
    else setHtml('btChart', '<div class="empty">无净值图（任务未返回 chart_url）</div>');

    var yearly = res.yearly || [];
    setText('btYearlyNote', yearly.length ? yearly.length + ' 个年度' : '');
    setHtml('btYearly', tableHtml(
      [{ label: '年份', num: true }, { label: '收益率', num: true }],
      yearly.map(function (y) { return [String(y.year), pctSpan(y.ret, 2)]; }),
      { empty: '暂无分年度数据' }
    ));

    var periods = res.periods || { full: m };
    var order = ['full', 'val', 'oos'].filter(function (k) { return periods[k]; });
    setHtml('btPeriods', tableHtml([
      { label: '区间' }, { label: 'CAGR', num: true }, { label: '年化波动', num: true },
      { label: 'Sharpe', num: true }, { label: 'MaxDD', num: true }, { label: 'Calmar', num: true },
      { label: '月胜率', num: true }, { label: '年换手', num: true }, { label: '成本', num: true },
      { label: '交易数', num: true }
    ], order.map(function (k) {
      var x = periods[k] || {};
      return [
        PERIOD_NAMES[k] || esc(k), pctSpan(x.cagr), fmtPctU(x.ann_vol), fmtNum(x.sharpe, 3),
        '<span class="mono down">' + fmtPct(x.max_dd) + '</span>', fmtNum(x.calmar, 3),
        fmtPctU(x.win_rate_monthly), fmtNum(x.turnover_ann, 2) + 'x', fmtPctU(x.cost_ratio), fmtInt(x.n_trades)
      ];
    }), { empty: '暂无分区间数据' }));

    var bench = res.benchmarks || {};
    var benchRows = [['<span class="tbl-name">策略</span>', pctSpan(m.cagr), '<span class="mono down">' + fmtPct(m.max_dd) + '</span>', fmtNum(m.sharpe, 2)]];
    Object.keys(bench).forEach(function (k) {
      var x = bench[k] || {};
      benchRows.push([esc(BENCH_NAMES[k] || k), pctSpan(x.cagr), '<span class="mono down">' + fmtPct(x.max_dd) + '</span>', fmtNum(x.sharpe, 2)]);
    });
    setHtml('btBench', tableHtml(
      [{ label: '组合 / 基准' }, { label: 'CAGR', num: true }, { label: 'MaxDD', num: true }, { label: 'Sharpe', num: true }],
      benchRows, { empty: '暂无基准数据' }
    ));

    var t = res.trades_summary || {};
    setText('btTradeNote', '成本含佣金 / 印花税 / 过户费 / 滑点');
    var tradeCards = [
      kpi('买入笔数', fmtInt(t.n_buy != null ? t.n_buy : m.n_buy_trades)),
      kpi('卖出笔数', fmtInt(t.n_sell != null ? t.n_sell : m.n_sell_trades)),
      kpi('总成本', fmtMoney(t.cost_total != null ? t.cost_total : m.cost_total)),
      kpi('总交易笔数', fmtInt(m.n_trades))
    ];
    var byYear = t.by_year;
    var tradeHtml = '<div class="grid grid-kpi" style="margin-bottom:10px">' + tradeCards.join('') + '</div>';
    if (byYear && byYear.length) {
      tradeHtml += '<div class="table-wrap" style="max-height:300px">' + tableHtml(
        [{ label: '年份', num: true }, { label: '交易笔数', num: true }, { label: '成本', num: true }],
        byYear.map(function (y) { return [String(y.year), fmtInt(y.n), fmtMoney(y.cost)]; }),
        { empty: '暂无数据' }
      ) + '</div>';
    }
    setHtml('btTrades', tradeHtml);

    show(byId('btResult'), true);
  }

  function fmtCur(v) {
    if (isNil(v)) return '—';
    var n = Number(v);
    if (!isFinite(n)) return '—';
    return fmtMoney(n) + '元';
  }

  /* ============================== 个股 ============================== */
  function bindStockSearch() {
    var input = byId('stockSearchInput');
    var btn = byId('btnStockSearch');
    var box = byId('stockSuggest');
    if (!input || !btn) return;
    input.addEventListener('input', function () { scheduleSuggest(input.value.trim()); });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'Enter') {
        e.preventDefault();
        var q = input.value.trim();
        if (!q) return;
        doStockSearch(q);
      } else if (e.key === 'Escape') {
        hideSuggest();
      }
    });
    input.addEventListener('focus', function () {
      if (state.suggest.length && input.value.trim()) renderSuggest(state.suggest, input.value.trim());
    });
    btn.addEventListener('click', function () {
      var q = input.value.trim();
      if (!q) { toast('请输入股票代码或名称', 'info'); return; }
      doStockSearch(q);
    });
    if (box) {
      box.addEventListener('click', function (e) {
        var item = e.target.closest ? e.target.closest('.suggest-item') : null;
        if (!item) return;
        var code = item.getAttribute('data-code');
        hideSuggest();
        if (code) loadStock(code);
      });
    }
    document.addEventListener('click', function (e) {
      if (!e.target.closest || !e.target.closest('.search-wrap')) hideSuggest();
    });
  }

  function hideSuggest() {
    var box = byId('stockSuggest');
    if (box) box.hidden = true;
  }

  function scheduleSuggest(q) {
    if (state.suggestTimer) { clearTimeout(state.suggestTimer); state.suggestTimer = null; }
    if (!q) { hideSuggest(); return; }
    state.suggestTimer = setTimeout(function () { fetchSuggest(q); }, 260);
  }

  function fetchSuggest(q) {
    var seq = ++state.suggestSeq;
    apiFetch('/api/stock/search?q=' + encodeURIComponent(q) + '&limit=20').then(function (list) {
      if (seq !== state.suggestSeq) return;
      state.suggest = Array.isArray(list) ? list : [];
      renderSuggest(state.suggest, q);
    }).catch(function (err) {
      if (seq !== state.suggestSeq) return;
      if (isServiceLoading(err)) { toast('服务正在加载数据，稍候再试搜索…', 'info'); return; }
      state.suggest = [];
      renderSuggest([], q, err.message);
      markDegraded();
    });
  }

  function renderSuggest(list, q, errMsg) {
    var box = byId('stockSuggest');
    if (!box) return;
    if (errMsg) {
      box.innerHTML = '<div class="suggest-empty">搜索失败：' + esc(errMsg) + '</div>';
      box.hidden = false;
      return;
    }
    if (!list || !list.length) {
      box.innerHTML = '<div class="suggest-empty">没有匹配 “' + esc(q) + '” 的股票</div>';
      box.hidden = false;
      return;
    }
    box.innerHTML = list.map(function (it) {
      return '<div class="suggest-item" data-code="' + attr(it.code) + '">' +
        '<span><span class="s-code">' + esc(it.code) + '</span> <span class="s-name">' + esc(it.name) + '</span></span>' +
        '<span class="s-ind">' + esc(it.industry || '') + '</span></div>';
    }).join('');
    box.hidden = false;
  }

  function doStockSearch(q) {
    hideSuggest();
    apiFetch('/api/stock/search?q=' + encodeURIComponent(q) + '&limit=20').then(function (list) {
      var arr = Array.isArray(list) ? list : [];
      if (!arr.length) {
        toast('未找到匹配 “' + q + '” 的股票', 'info');
        return;
      }
      loadStock(arr[0].code);
    }).catch(function (err) {
      if (isServiceLoading(err)) { toast('服务正在加载数据，稍候再试搜索…', 'info'); return; }
      toast('搜索失败：' + err.message, 'error');
      markDegraded();
    });
  }

  // 自动重试上限：3s 起、逐步拉长到 8s，合计约 2 分钟。
  // 超过就停下来给一个可点的「重试」，而不是无限转圈（无限转圈会让人以为卡死）。
  var STOCK_MAX_RETRY = 20;

  /**
   * 个股页的可重试错误 = 服务还没就绪（503）+ 连不上/超时（服务正在重启）。
   * 后者以前会直接甩一个红框「无法连接后端服务」，用户看到的是一次重启留下的
   * 永久报错；实际上再等十几秒就恢复了，所以这里一并当作"稍后重试"。
   */
  function isTransientStockError(err) {
    if (isServiceLoading(err)) return true;
    return /无法连接后端服务|请求超时|Failed to fetch|NetworkError/i
      .test(String((err && err.message) || ''));
  }

  function loadStock(code, force) {
    if (!code) return;
    var seq = ++state.stockLoadSeq;
    // 立刻记住要加载的代码。以前只在成功后才写 state.stockCode，
    // 于是「第一次就失败」的情况下：切走再切回个股页 → initStockPage 什么都不做
    // → 「正在准备数据」这句永久留在页面上。这是它反复出现的真正原因。
    state.stockCode = code;
    if (!force) state.stockRetry = 0;
    setHtml('stockDetail', '<div class="card"><div class="card-bd">' +
      loadingBlock('正在加载 ' + code + ' 详情…') + '</div></div>');
    var curModel = state.analysisModel || state.defaultModel || 'wf_h10';
    apiFetch('/api/stock/' + encodeURIComponent(code) + '?model=' + encodeURIComponent(curModel)).then(function (d) {
      if (seq !== state.stockLoadSeq) return;
      if (!d || !d.code) throw new Error('接口未返回个股数据');
      state.stock = d;
      state.stockCode = d.code;
      state.stockRetry = 0;
      state.loaded.stock = true;
      renderStock(d);
      var input = byId('stockSearchInput');
      if (input) input.value = d.code + ' ' + (d.name || '');
      markOk();
    }).catch(function (err) {
      if (seq !== state.stockLoadSeq) return;
      state.stock = null;
      if (isTransientStockError(err)) {
        renderStockWaiting(code, err);
        timeoutRetryStock(code, seq);
        return;
      }
      setHtml('stockDetail', '<div class="card"><div class="card-bd"><div class="error-box" style="margin:0">' +
        '<b>加载失败：</b>' + esc(err.message) + '</div>' +
        '<div style="margin-top:10px">' +
        '<button class="btn btn-sm" data-retry="stock">重试</button></div></div></div>');
      toast('个股数据加载失败：' + err.message, 'error');
      markDegraded();
    });
  }

  /** 服务还没就绪时的等待态：说清在等什么、第几次重试、以及后端原话 */
  function renderStockWaiting(code, err) {
    var n = state.stockRetry || 0;
    var why = String((err && err.message) || '');
    var offline = /无法连接后端服务|请求超时|Failed to fetch|NetworkError/i.test(why);
    var head = offline
      ? '后端服务暂时连不上（多半正在重启，一般 1 分钟内起来），正在等它恢复'
      : ('正在准备 ' + esc(code) + ' 的数据（服务刚启动时要先把全市场打分算完，' +
         '约 1~2 分钟；之后秒开）');
    setHtml('stockDetail', '<div class="card"><div class="card-bd">' +
      '<div class="empty-block"><span class="spinner"></span> ' + head +
      (n ? '，第 ' + n + ' 次自动重试' : '') + '…</div>' +
      '<div class="muted" style="font-size:12px;margin-top:8px">' +
      '超过 2 分钟还停在这句，就点下面的按钮重试；后端原话：' + esc(why) + '</div>' +
      '<div style="margin-top:10px">' +
      '<button class="btn btn-sm" data-retry="stock">立即重试</button></div>' +
      '</div></div>');
  }

  function timeoutRetryStock(code, seq) {
    state.stockRetry = (state.stockRetry || 0) + 1;
    if (state.stockRetry > STOCK_MAX_RETRY) {
      setHtml('stockDetail', '<div class="card"><div class="card-bd"><div class="error-box" style="margin:0">' +
        '<b>' + esc(code) + ' 连续 ' + STOCK_MAX_RETRY + ' 次没取到数据。</b><br>' +
        '通常是后端刚重启（约 1 分钟）或这个代码当前不可用。点「重试」再拉一次。</div>' +
        '<div style="margin-top:10px">' +
        '<button class="btn btn-sm" data-retry="stock">重试</button></div></div></div>');
      return;
    }
    var delay = Math.min(3000 + 500 * (state.stockRetry - 1), 8000);
    setTimeout(function () {
      // 不再要求 activePage === 'stock'：用户切到别的页也该把数据拉好。
      // seq 变了说明有更新的请求，本次静默作废即可。
      if (seq !== state.stockLoadSeq) return;
      loadStock(code, true);
    }, delay);
  }

  function renderStock(d) {
    var chg = d.chg_1d;
    var modelName = modelLabel(d.model || state.analysisModel);

    /* ---------- 顶部：价格 + 多周期 + 评分 ---------- */
    var perfs = [['1日', d.chg_1d], ['5日', d.chg_5d], ['20日', d.chg_20d],
                 ['60日', d.chg_60d], ['120日', d.chg_120d]].map(function (p) {
      return '<div class="p"><div class="k">' + p[0] + '</div>' +
        '<div class="v ' + pctClass(p[1]) + '">' + fmtPct(p[1]) + '</div></div>';
    }).join('');

    var tags = [];
    if (d.in_universe) tags.push('<span class="badge badge-accent">选股域内</span>');
    else tags.push('<span class="badge">选股域外</span>');
    if (d.in_model_universe === false) tags.push('<span class="badge">不满足模型域</span>');
    if (d.in_target) {
      tags.push('<span class="badge badge-up">目标持仓' +
        (d.target_weight != null ? ' ' + fmtPctU(d.target_weight, 2) : '') + '</span>');
    }
    tags.push('<span class="badge">' + esc(d.industry || '—') + '</span>');
    tags.push('<span class="badge">' + esc(boardName(d.board)) + '</span>');

    var metas = [
      ['总市值', fmtMoney(d.mcap)],
      ['20日均额', fmtMoney(d.amount_20d_avg)],
      ['上市天数', d.list_days != null ? fmtInt(d.list_days) + ' 天' : '—'],
      ['数据日期', fmtDate(d.asof)]
    ].map(function (m) {
      return '<div class="k">' + m[0] + '</div><div class="v">' + m[1] + '</div>';
    }).join('');

    // 评分（域内分 / 域外参考分）
    var sc = d.score != null ? d.score : d.score_ref;
    var isRef = (d.score == null && d.score_ref != null);
    var pctW = clamp(toNum(d.score_pct) * 100, 0, 100);
    var scoreBlock;
    if (sc != null) {
      var excl = d.exclusion || {};
      var warnHtml = '';
      if (isRef) {
        warnHtml = '<div class="stk-warn">该股<b>不在选股域内</b>，以下为模型参考打分' +
          (excl.reasons && excl.reasons.length
            ? '。<br>原因：' + excl.reasons.map(esc).join('；') : '') + '</div>';
      }
      scoreBlock = '<div class="stk-score-big mono">' + fmtNum(sc, 3) + '</div>' +
        '<div class="muted small">' + esc(modelName) + ' 复合分</div>' +
        '<div class="stk-score-bar"><i style="width:' + pctW.toFixed(1) + '%"></i></div>' +
        '<div class="stk-meta-grid" style="grid-template-columns:repeat(2,1fr)">' +
        '<div class="k">全市场排名</div><div class="v">' +
        (d.score_rank != null ? '第 ' + fmtInt(d.score_rank) + ' 名' : '—') + '</div>' +
        '<div class="k">分位</div><div class="v">' + fmtPctU(d.score_pct, 2) + '</div>' +
        '</div>' + warnHtml;
    } else {
      var ex2 = d.exclusion || {};
      var why = (ex2.reasons && ex2.reasons.length)
        ? ex2.reasons.map(esc).join('；')
        : '该标的不满足选股域条件';
      scoreBlock = '<div class="stk-score-big mono muted">—</div>' +
        '<div class="muted small">不参与打分</div>' +
        '<div class="stk-warn"><b>为什么没有复合分：</b><br>' + why +
        '<br><br><b>说明：</b>模型只对"选股域内"的标的计算因子与排名。' +
        '价格超出 2~200 元或流动性不足的标的会被排除——' +
        '以 30 万本金计，1 手茅台就要 13 万，分散度无法保证。</div>';
    }

    var hero = '<div class="stk-hero">' +
      // 价格卡
      '<div class="stk-box">' +
      '<div class="stk-title"><h3>' + esc(d.name || d.code) + '</h3>' +
      '<span class="stk-code">' + esc(d.code) + '</span></div>' +
      '<div><span class="stk-price ' + pctClass(chg) + '">' + fmtPrice(d.price) + '</span>' +
      '<span class="stk-chg ' + pctClass(chg) + '">' + fmtPct(chg) + '</span></div>' +
      '<div class="stk-tags">' + tags.join('') + '</div>' +
      '<div class="stk-meta-grid">' + metas + '</div>' +
      '</div>' +
      // 多周期
      '<div class="stk-box"><div class="muted small" style="margin-bottom:8px">多周期涨跌</div>' +
      '<div class="stk-perf">' + perfs + '</div>' +
      '<div class="muted smaller" style="margin-top:9px">' +
      '推荐关注 20 日与 60 日：主策略为 10 日反转，短期超跌但中期趋势不坏的标的更易入选。</div>' +
      '</div>' +
      // 评分
      '<div class="stk-box">' + scoreBlock + '</div>' +
      '</div>';

    /* ---------- K 线 ---------- */
    var kline = '<div class="card chart-card"><div class="card-hd">' +
      '<h4>K 线（后复权 · 250 日）</h4>' +
      '<span class="muted small mono">' + esc(d.code) + '</span></div>' +
      '<div class="card-bd"><div class="chart-box" id="stockKline"></div></div></div>';

    /* ---------- 因子贡献 ---------- */
    var factors = (d.factors || []).slice().sort(function (a, b) {
      return Math.abs(toNum(b.contribution)) - Math.abs(toNum(a.contribution));
    });
    var fRows = factors.map(function (f) {
      var c = toNum(f.contribution);
      return [
        '<span class="tbl-code">' + esc(f.key) + '</span>',
        esc(f.name_cn || '—'),
        fmtNum(f.feature, 3),
        fmtNum(f.weight, 4),
        '<span class="mono ' + (c >= 0 ? 'contrib-pos' : 'contrib-neg') + '">' +
        fmtSigned(f.contribution, 4) + '</span>'
      ];
    });
    var factorCard = '<div class="card"><div class="card-hd"><h4>因子贡献</h4>' +
      '<span class="muted small">按 |贡献| 排序 · ' + factors.length + ' 个</span></div>' +
      '<div class="card-bd table-wrap">' +
      (factors.length
        ? tableHtml([
            { label: '因子', mono: true }, { label: '名称' }, { label: '特征值', num: true },
            { label: '权重', num: true }, { label: '贡献', num: true }
          ], fRows)
        : '<div class="empty-block">该股不在选股域内，模型不为其计算因子，' +
          '因此无贡献数据。可参考上方"多周期涨跌"与"基本面"了解其状态。</div>') +
      '</div></div>';

    /* ---------- 基本面（分组） ---------- */
    var fund = d.fundamentals;
    var GROUPS = [
      ['估值', ['ep_ttm', 'bp', 'sp_ttm']],
      ['盈利', ['roe_ann', 'gross_margin', 'earnings_quality', 'margin_expansion']],
      ['成长', ['rev_growth', 'profit_growth', 'profit_accel', 'rev_accel',
                'profit_qoq', 'roe_change', 'gm_change']],
      ['质量', ['accruals']]
    ];
    var fundHtml = '';
    if (fund && typeof fund === 'object' && Object.keys(fund).length) {
      var used = {};
      GROUPS.forEach(function (g) {
        var cells = g[1].filter(function (k) {
          return Object.prototype.hasOwnProperty.call(fund, k);
        }).map(function (k) {
          used[k] = 1;
          var v = toNum(fund[k]);
          return '<div class="fund-cell"><div class="k">' +
            esc(FUND_NAMES[k] || k) + '</div><div class="v">' +
            fmtNum(v, Math.abs(v) >= 10 ? 2 : 4) + '</div></div>';
        }).join('');
        if (cells) {
          fundHtml += '<div class="fund-group"><h5>' + g[0] + '</h5>' +
            '<div class="fund-grid">' + cells + '</div></div>';
        }
      });
      var rest = Object.keys(fund).filter(function (k) { return !used[k]; });
      if (rest.length) {
        fundHtml += '<div class="fund-group"><h5>其他</h5><div class="fund-grid">' +
          rest.map(function (k) {
            var v = toNum(fund[k]);
            return '<div class="fund-cell"><div class="k">' + esc(FUND_NAMES[k] || k) +
              '</div><div class="v">' + fmtNum(v, Math.abs(v) >= 10 ? 2 : 4) + '</div></div>';
          }).join('') + '</div></div>';
      }
    } else {
      fundHtml = '<div class="empty-block">基本面数据缺失（该标的不在基本面覆盖范围内）。</div>';
    }
    var fundCard = '<div class="card"><div class="card-hd"><h4>基本面</h4>' +
      '<span class="muted small">财务指标（最近报告期）</span></div>' +
      '<div class="card-bd">' + fundHtml + '</div></div>';

    var html = hero + kline +
      '<div class="grid grid-2">' + factorCard + fundCard + '</div>';

    setHtml('stockDetail', html);
    setChart(byId('stockKline'),
      '/api/chart/stock_kline?code=' + encodeURIComponent(d.code) + '&days=250',
      (d.name || d.code) + ' K线');
  }



  function renderDataStatus(ds) {
    var host = document.getElementById('dataStatus');
    if (!host) return;
    if (!ds) { host.textContent = '无数据'; return; }
    var rows = [
      ['当前数据源', ds.source || ds.data_source || '—'],
      ['最后交易日', ds.last_date || ds.asof || '—'],
      ['原始数据包', ds.orig_last_date || '—'],
      ['交易日数', ds.n_days != null ? ds.n_days + ' 天' : '—'],
      ['股票数', ds.n_stocks != null ? ds.n_stocks : '—'],
      ['选股域', ds.n_eligible != null ? ds.n_eligible + ' 只' : '—'],
    ];
    host.innerHTML = '<table class="table kv-table">' + rows.map(function (r) {
      return '<tr><td class="muted">' + r[0] + '</td><td><b>' + esc(String(r[1])) + '</b></td></tr>';
    }).join('') + '</table>';
  }




  function setDataLog(lines) {
    var pre = byId('dataLog');
    if (!pre) return;
    var atBottom = pre.scrollHeight - pre.scrollTop - pre.clientHeight < 48;
    var text = (lines || []).map(function (l) { return String(l); }).join('\n');
    if (pre.textContent !== text) {
      pre.textContent = text || '（暂无日志）';
      if (atBottom || pre.scrollTop === 0) pre.scrollTop = pre.scrollHeight;
    }
  }



  /* ============================== 因子 ============================== */
  function loadFactors(force) {
    if (state.loaded.factors && !force) return Promise.resolve();
    if (!state.loaded.factors) {
      setHtml('factorCounts', skeletonRows(3));
      setHtml('factorFamilies', skeletonRows(4));
      setHtml('factorComposite', skeletonRows(4));
      setHtml('factorPool', skeletonRows(5));
      setHtml('factorFundamental', skeletonRows(4));
    }
    return apiFetch('/api/factors/summary').then(function (d) {
      state.factors = d || {};
      state.loaded.factors = true;
      clearPageError('factors');
      renderFactors(state.factors);
      markOk();
    }).catch(function (err) {
      if (isServiceLoading(err)) { serviceLoadingPlaceholder('factors', '因子数据'); return; }
      markDegraded();
      pageError('factors', err);
      toast('因子数据加载失败：' + err.message, 'error');
    });
  }

  function renderFactors(d) {
    var c = d.counts || {};
    var countCards = [
      kpi('因子总数', fmtInt(c.total), '三库 + 趋势族合计'),
      kpi('量化库', fmtInt(c.quant), 'price/volume 等'),
      kpi('动量库', fmtInt(c.momentum), 'momentum-extra'),
      kpi('基本面库', fmtInt(c.fundamental), 'fundamental-extra'),
      kpi('趋势库', fmtInt(c.trend), 'trend-extra · 右侧研究分支'),
      kpi('二阶段', fmtInt(d.stage2), '细分筛选后'),
      kpi('入模池', fmtInt(d.pool), '最终入选因子')
    ];
    setHtml('factorCounts', countCards.join(''));

    var stats = d.statistics || {};
    var note = [];
    if (stats.stage2_ic_median != null) note.push('二阶段 IC 中位数 ' + fmtNum(stats.stage2_ic_median, 4));
    if (stats.stage2_icir_median != null) note.push('ICIR 中位数 ' + fmtNum(stats.stage2_icir_median, 4));
    note.push('因子族 ' + ((d.families || []).length) + ' 个');
    setText('factorStatsNote', note.join(' · '));

    var fam = (d.families || []).slice().sort(function (a, b) { return toNum(b.dev_abs_icir) - toNum(a.dev_abs_icir); });
    setHtml('factorFamilies', tableHtml([
      { label: '因子族' }, { label: '因子数', num: true }, { label: '|dev ICIR|', num: true }, { label: '覆盖率', num: true }
    ], fam.map(function (f) {
      return [esc(f.family), fmtInt(f.n), fmtNum(f.dev_abs_icir, 3), fmtPctU(f.coverage, 1)];
    }), { empty: '暂无因子族数据' }));

    var ci = d.composite_ic || {};
    var comp = '<div class="grid grid-kpi" style="margin-bottom:10px">' +
      kpi('复合分 IC 均值', fmtNum(ci.mean, 4), 'Rank IC') +
      kpi('ICIR', fmtNum(ci.icir, 3), 'IC 信息比') +
      kpi('样本周数', fmtInt(ci.n_weeks), 'walk-forward') +
      '</div>';
    comp += '<div class="table-wrap" style="max-height:300px">' + tableHtml([
      { label: '年份', num: true }, { label: 'IC', num: true }, { label: '周数', num: true }
    ], (ci.by_year || []).map(function (y) {
      return [String(y.year), '<span class="mono ' + pctClass(y.ic) + '">' + fmtNum(y.ic, 4) + '</span>', fmtInt(y.n)];
    }), { empty: '暂无 IC 年度数据' }) + '</div>';
    setHtml('factorComposite', comp);

    var pool = (d.pool_detail || []).slice().sort(function (a, b) {
      return Math.abs(toNum(b.weight)) - Math.abs(toNum(a.weight));
    });
    var warnAt = toNum(d.family_share_warn) || 0.33;
    function shareCell(share, over) {
      if (share == null) return '<span class="muted">—</span>';
      var txt = fmtPctU(share, 1);
      return over ? '<span class="badge warn">' + txt + '</span>'
                  : '<span class="mono">' + txt + '</span>';
    }
    setText('factorPoolCount', pool.length + ' 个因子 · 按 |权重| 排序 · 族占比 > ' +
      fmtPctU(warnAt, 0) + ' 标黄');

    // 族占比监控（2026-11-03）：族上限机制干预三档全否（损伤随干预强度单调递增），
    // 结论是「集中度是结果不是病因」→ 只做展示层监控，不做权重干预。
    var famWList = (d.family_weights || []);
    var monHtml = '';
    if (famWList.length) {
      monHtml = '<div class="table-wrap" style="max-height:240px;margin-bottom:10px">' +
        tableHtml([
          { label: '族（最新一期 |权重| 占比）' }, { label: '入池因子数', num: true },
          { label: '|权重| 合计', num: true }, { label: '占比', num: true }, { label: '监控' }
        ], famWList.map(function (x) {
          var over = toNum(x.share) > warnAt;
          return [esc(x.family), fmtInt(x.n), fmtNum(x.weight_abs, 3),
                  shareCell(toNum(x.share), over),
                  over ? '<span class="badge warn">超 ' + fmtPctU(warnAt, 0) + '</span>'
                       : '<span class="muted">正常</span>'];
        })) + '</div>';
    }

    var maxW = pool.reduce(function (mx, r) { return Math.max(mx, Math.abs(toNum(r.weight))); }, 0) || 1;
    setHtml('factorPool', monHtml + tableHtml([
      { label: '因子', mono: true }, { label: '名称' }, { label: '族' },
      { label: '族占比', num: true },
      { label: '权重', num: true }, { label: 'DEV IC', num: true }, { label: 'DEV ICIR', num: true }, { label: '权重条' }
    ], pool.map(function (r) {
      var abs = Math.abs(toNum(r.weight));
      var barW = Math.round(abs / maxW * 64);
      var family = r.family || r.category_en || String(r.factor_key || '').split(':')[0];
      var sh = r.family_share == null ? null : toNum(r.family_share);
      return [
        '<span class="tbl-code">' + esc(r.factor_key) + '</span>',
        esc(r.name_cn || '—'),
        esc(family || '—'),
        shareCell(sh, sh != null && sh > warnAt),
        '<span class="mono ' + (toNum(r.weight) >= 0 ? 'up' : 'down') + '">' + fmtSigned(r.weight, 4) + '</span>',
        '<span class="mono ' + pctClass(r.ic) + '">' + fmtNum(r.ic, 4) + '</span>',
        fmtNum(r.icir, 3),
        '<span class="cell-bar" style="width:' + barW + 'px"></span>'
      ];
    }), { empty: '暂无入模池数据' }));

    var fund = d.fundamental_stats || [];
    setHtml('factorFundamental', tableHtml([
      { label: '因子', mono: true }, { label: '名称' }, { label: 'DEV IC', num: true },
      { label: 'DEV ICIR', num: true }, { label: '覆盖率', num: true }
    ], fund.map(function (f) {
      return [
        '<span class="tbl-code">' + esc(f.factor_key) + '</span>',
        esc(f.name_cn || '—'),
        '<span class="mono ' + pctClass(f.dev_ic) + '">' + fmtNum(f.dev_ic, 4) + '</span>',
        fmtNum(f.dev_icir, 3),
        fmtPctU(f.coverage, 1)
      ];
    }), { empty: '暂无基本面因子数据' }));
  }

  /* ============================== 报告 ============================== */
  function loadReports(force) {
    if (state.loaded.reports && !force) return Promise.resolve();
    if (!state.loaded.reports) setHtml('reportList', skeletonRows(5));
    return apiFetch('/api/reports').then(function (d) {
      var files = (d && d.files) || [];
      state.reports = files;
      state.loaded.reports = true;
      clearPageError('reports');
      renderReports(files);
      markOk();
    }).catch(function (err) {
      if (isServiceLoading(err)) { serviceLoadingPlaceholder('reports', '报告数据'); return; }
      markDegraded();
      pageError('reports', err);
      setHtml('reportList', '<div class="empty-block">报告列表加载失败：' + esc(err.message) + '</div>');
      toast('报告列表加载失败：' + err.message, 'error');
    });
  }

  function renderReports(files) {
    setText('reportCount', files.length + ' 个文件');
    if (!files.length) {
      setHtml('reportList', '<div class="empty-block">报告目录为空（暂无文件）</div>');
      return;
    }
    var kindMap = { report: '报告', md: 'Markdown', csv: 'CSV', xlsx: 'Excel', json: 'JSON', log: '日志' };
    var sorted = files.slice().sort(function (a, b) {
      return String(b.mtime || '').localeCompare(String(a.mtime || ''));
    });
    setHtml('reportList', tableHtml([
      { label: '文件名' }, { label: '类型' }, { label: '大小', num: true }, { label: '修改时间', mono: true }
    ], sorted.map(function (f) {
      var href = f.path || ('/reports/' + encodeURIComponent(f.name));
      var kind = f.kind || (String(f.name || '').split('.').pop() || '');
      return [
        '<a class="link" href="' + attr(href) + '" target="_blank" rel="noopener">' + esc(f.name || '—') + '</a>',
        '<span class="badge accent">' + esc(kindMap[kind] || kind) + '</span>',
        fmtSize(f.size),
        fmtDateTime(f.mtime)
      ];
    }), { empty: '暂无报告文件' }));
  }

  /* ============================== 页面切换 / 懒加载 ============================== */
  function switchPage(name) {
    if (!PAGE_TITLES[name]) name = 'market';
    if (state.activePage === name && state.loaded[name]) {
      // 同一页签重复点击：仅重新聚焦
      return;
    }
    state.activePage = name;
    $all('.nav-item').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-page') === name);
    });
    $all('.page').forEach(function (p) {
      p.classList.toggle('active', p.getAttribute('data-page') === name);
    });
    setText('pageTitle', PAGE_TITLES[name]);
    document.title = PAGE_TITLES[name] + ' · A股量化 · Pangolin';
    if (('#' + name) !== location.hash) {
      try { history.replaceState(null, '', '#' + name); } catch (e) { /* ignore */ }
    }
    var content = byId('content');
    if (content) content.scrollTop = 0;
    ensurePage(name, false);
  }

  function ensurePage(name, force) {
    if (name === 'backtest') applyBtDefaults();
    if (state.loaded[name] && !force) {
      var activeModel = state.analysisModel || state.defaultModel;
          }
    switch (name) {
      case 'today': loadToday(force); break;
      case 'monthly': loadMonthly(force); break;
      case 'market': loadMarket(force); break;
      case 'factors': loadFactors(force); break;
      case 'reports': loadReports(force); break;
      case 'etf': loadEtf(force); break;
      case 'sector': loadSectorRotation(force); break;
      case 'cb': loadCbSignal(force); break;
      case 'portfolio': loadPortfolio(); break;
      case 'research': loadResearch(); break;
      case 'tracking': loadTracking(); break;
      case 'data': loadDataInfo(); loadPipelineStatus(); loadConsistency(); bindPipeline(); break;
      case 'backtest': initBacktestPage(); break;
      case 'stock': initStockPage(); break;
      default: break;
    }
  }

  function initStockPage() {
    var input = byId('stockSearchInput');
    if (input && state.stock) input.value = state.stock.code + ' ' + (state.stock.name || '');
    var box = byId('stockDetail');
    var hasCard = !!(box && box.querySelector('.card'));

    // 已有数据 → 只在被清空时补画（不重复请求）
    if (state.stock) {
      if (!hasCard) renderStock(state.stock);
      return;
    }
    // 没有数据但记得代码（上次失败/被打断）→ 重新拉，别让占位提示留在页面上
    if (state.stockCode) { loadStock(state.stockCode); return; }
    // 全新进入且还没搜过 → 给明确指引，而不是空白
    if (!hasCard) {
      setHtml('stockDetail', '<div class="card"><div class="card-bd">' +
        '<div class="empty-block">在右上角搜索框输入股票代码或名称（如 600519），' +
        '或从「今日推荐」点任意一只股票进入。</div></div></div>');
    }
  }

  function reloadPage(name) {
    clearPageError(name);
    ensurePage(name, true);
  }

  function refreshCurrent() {
    var name = state.activePage;
    toast('正在刷新' + PAGE_TITLES[name] + ' 数据…', 'info', 1200);
    loadMeta(true);
    if (name === 'backtest') {
      initBacktestPage();
      if (state.bt.result) renderBtResult(state.bt.result, state.bt.jobId);
    } else if (name === 'stock') {
      if (state.stockCode) loadStock(state.stockCode);   // 不带 force → 重置重试计数
      else initStockPage();
    } else {
      ensurePage(name, true);
    }
  }

  /* ============================== 事件绑定 / 启动 ============================== */
  function bindGlobal() {
    $all('.nav-item').forEach(function (btn) {
      btn.addEventListener('click', function () { switchPage(btn.getAttribute('data-page')); });
    });
    var refresh = byId('btnRefresh');
    if (refresh) refresh.addEventListener('click', refreshCurrent);
    document.addEventListener('click', function (e) {
      if (!e.target.closest) return;
      var retry = e.target.closest('[data-retry]');
      if (retry) {
        var page = retry.getAttribute('data-retry');
        if (page) reloadPage(page);
        return;
      }
      var jump = e.target.closest('[data-jump]');
      if (jump) {
        e.preventDefault();
        e.stopPropagation();
        switchPage(jump.getAttribute('data-jump'));
        return;
      }
      var f = e.target.closest('[data-st-filter]');
      if (f) {
        e.preventDefault();
        setTodayFilter(f.getAttribute('data-st-filter'));
        return;
      }
    });
    window.addEventListener('hashchange', function () {
      var name = (location.hash || '').replace(/^#/, '');
      if (PAGE_TITLES[name] && name !== state.activePage) switchPage(name);
    });
    window.addEventListener('offline', function () { toast('网络已断开：本页面需要连接本地后端', 'error'); });
  }

  function bindMarketPage() {
    var sel = byId('indexSelect');
    if (sel) sel.addEventListener('change', function () {
      state.indexCode = sel.value;
      drawIndexChart();
    });
  }

  function bindRecommendPage() {
    var cb = byId('candTargetOnly');
    if (cb) cb.addEventListener('change', renderCandidates);
    var s = byId('candSearch');
    if (s) s.addEventListener('input', renderCandidates);
    var sel = byId('recModelSelect');
    if (sel) sel.addEventListener('change', function () { setAnalysisModel(sel.value, true); });
  }

  function bindBacktestPage() {
    /* 回测按钮统一由 btRunUnified 处理（见 boot 中的绑定）。
       旧的 runBacktest 依赖已删除的 DOM 字段，故不再绑定。 */
  }

  function bindStockModelSelect() {
    var sel = byId('stockModelSelect');
    if (sel) sel.addEventListener('change', function () { setAnalysisModel(sel.value, true); });
  }


  function boot() {
    bindGlobal();
    bindTodayExport();
    bindHoldingsPanel();
    bindStaleUpdate();
    bindMonthlyPage();
    initLoadingStages();
    btBindKindPicker();
    var _runBtn = byId('btnRunBacktest');
    if (_runBtn) _runBtn.addEventListener('click', btRunUnified);
    bindMarketPage();
    bindRecommendPage();
    bindBacktestPage();
    bindStockSearch();
    bindStockModelSelect();
    loadMeta();
    var initial = (location.hash || '').replace(/^#/, '');
    switchPage(PAGE_TITLES[initial] ? initial : 'today');
    updateLoadState();
    autoRunBacktest();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

/* ============================== ETF 趋势 ============================== */
    /* ============================== ETF 趋势 ============================== */
  function loadEtf(force) {
    if (state.loaded.etf && !force) return;
    setText('etfAsOf', '加载中…');
    apiFetch('/api/etf_trend').then(function (d) {
      state.etf = d;
      state.loaded.etf = true;
      clearPageError('etf');
      renderEtf(d);
      loadComboPlan();
      markOk();
    }).catch(function (err) {
      if (isServiceLoading(err)) { serviceLoadingPlaceholder('etf', 'ETF 趋势策略'); return; }
      markDegraded();
      pageError('etf', err);
    });
  }

  function _etfPick(d) {
    var uni = d.universe || [];
    for (var i = 0; i < uni.length; i++) if (uni[i].entry_signal) return uni[i];
    return null;
  }

  function renderEtf(d) {
    setText('etfAsOf', '数据日期 ' + (d.asof || '—'));
    var bt = d.backtest || {};
    var combo = (d.combo && d.combo.results) || {};
    var c50 = (combo['50/50组合'] || {}).full || {};
    var st = byId('etfStatusChip');
    var pick = _etfPick(d);
    if (st) {
      st.innerHTML = pick
        ? '<span class="on">● 持仓信号</span>　' + esc(pick.code + ' ' + (pick.name || ''))
        : '<span class="off">○ 空仓观望</span>　三只 ETF 均未突破 55 日新高';
    }

    /* ---- KPI ---- */
    var kpi = [
      ['ETF趋势 年化', fmtPct(bt.cagr), 'Sharpe ' + fmtNum(bt.sharpe, 2), 'accent'],
      ['ETF趋势 最大回撤', fmtPct(bt.max_dd), 'Calmar ' + fmtNum(bt.calmar, 2), 'down'],
      ['50/50 组合 年化', fmtPct(c50.cagr), 'Sharpe ' + fmtNum(c50.sharpe, 2), 'up'],
      ['50/50 组合 回撤', fmtPct(c50.max_dd), '相关性 ' + fmtNum(d.combo && d.combo.correlation, 2), 'muted']
    ];
    var ke = byId('etfKpis');
    if (ke) {
      ke.innerHTML = kpi.map(function (k) {
        return '<div class="card kpi"><div class="kpi-label">' + k[0] + '</div>' +
          '<div class="kpi-value">' + k[1] + '</div>' +
          '<div class="kpi-sub muted small">' + k[2] + '</div></div>';
      }).join('');
    }

    /* ---- 当前信号表 ---- */
    var uni = d.universe || [];
    var uh = uni.map(function (u) {
      var m = u.mom60 || 0;
      var w = Math.min(50, Math.abs(m) * 100 * 2.2);
      var bar = '<span class="mom-bar"><i class="' + (m >= 0 ? 'pos' : 'neg') +
        '" style="width:' + w.toFixed(1) + '%;' +
        (m >= 0 ? 'left:50%' : 'left:' + (50 - w).toFixed(1) + '%') + '"></i></span>';
      var pill = u.entry_signal ? '<span class="pill pill-on">入场</span>'
                                : '<span class="pill pill-off">观察</span>';
      return '<tr class="' + (u.entry_signal ? 'is-pick' : '') + '">' +
        '<td class="mono">' + esc(u.code) + '</td>' +
        '<td>' + esc(u.name || '') + '</td>' +
        '<td class="num mono">' + fmtNum(u.price, 3) + '</td>' +
        '<td class="num ' + ((u.day_chg || 0) >= 0 ? 'up' : 'down') + '">' + fmtPct(u.day_chg) + '</td>' +
        '<td class="num">' + fmtPct(u.mom60) + ' ' + bar + '</td>' +
        '<td class="num">' + fmtPct(u.dist_to_high) + '</td>' +
        '<td class="num">' + fmtMoney(u.amount_20d) + '</td>' +
        '<td>' + pill + '</td></tr>';
    }).join('');
    var ue = byId('etfUniverse');
    if (ue) {
      ue.innerHTML = '<table><colgroup><col style="width:11%"><col style="width:19%">' +
        '<col style="width:11%"><col style="width:9%"><col style="width:24%">' +
        '<col style="width:10%"><col style="width:9%"><col style="width:7%"></colgroup>' +
        '<thead><tr><th>代码</th><th>名称</th><th class="num">最新价</th>' +
        '<th class="num">当日</th><th class="num">60日动量</th><th class="num">距55日高</th>' +
        '<th class="num">20日均额</th><th>信号</th></tr></thead><tbody>' +
        (uh || '<tr><td colspan="8" class="muted">无数据</td></tr>') + '</tbody></table>';
    }

    /* ---- 组合对比卡 ---- */
    var ce = byId('etfCombo');
    if (ce) {
      var names = [['主策略', '个股 wf_h10'], ['ETF趋势', '宽基 ETF'], ['50/50组合', '每季度再平衡']];
      var c3r = (d.combo3 && d.combo3.results && d.combo3.results['50/25/25']) || null;
      if (c3r) names.push(['三腿组合', '主50+ETF25+红利低波25']);
      ce.innerHTML = names.map(function (nm) {
        var src = nm[0] === '三腿组合' ? (c3r || {}) : (combo[nm[0]] || {});
        var r = src, f = r.full || {}, o = r.oos || {};
        var isCombo = nm[0] === '50/50组合' || nm[0] === '三腿组合';
        return '<div class="cmp-card' + (isCombo ? ' is-combo' : '') + '">' +
          '<div class="cmp-top"><span class="cmp-name">' + nm[0] + '</span>' +
          '<span class="cmp-tag">' + nm[1] + '</span></div>' +
          '<div class="cmp-metrics">' +
          '<div><div class="k">年化</div><div class="v">' + fmtPct(f.cagr) + '</div></div>' +
          '<div><div class="k">Sharpe</div><div class="v">' + fmtNum(f.sharpe, 2) + '</div></div>' +
          '<div><div class="k">回撤</div><div class="v">' + fmtPct(f.max_dd) + '</div></div>' +
          '<div><div class="k">Calmar</div><div class="v">' + fmtNum(f.calmar, 2) + '</div></div>' +
          '</div>' +
          '<div class="cmp-foot">样本外年化 ' + fmtPct(o.cagr) +
          ' · Sharpe ' + fmtNum(o.sharpe, 2) + '</div></div>';
      }).join('');
    }

    /* ---- 参数稳健性（条形图） ---- */
    var sw = d.sweep || {};
    var keys = Object.keys(sw);
    var maxCagr = 0;
    keys.forEach(function (k) { maxCagr = Math.max(maxCagr, sw[k].cagr || 0); });
    var sh = keys.map(function (k) {
      var v = sw[k] || {};
      var cur = (k === 'entry55_exit0');
      var w = maxCagr > 0 ? (v.cagr / maxCagr * 100) : 0;
      return '<tr class="' + (cur ? 'is-cur' : '') + '">' +
        '<td class="mono">' + esc(k) + '</td>' +
        '<td class="num">' + fmtPct(v.cagr) + '</td>' +
        '<td><span class="sweep-bar"><i style="width:' + w.toFixed(1) + '%"></i></span></td>' +
        '<td class="num">' + fmtNum(v.sharpe, 2) + '</td>' +
        '<td class="num">' + fmtPct(v.max_dd) + '</td></tr>';
    }).join('');
    var se = byId('etfSweep');
    if (se) {
      se.innerHTML = sh ? '<table><colgroup><col class="c-param"><col class="c-num">' +
        '<col class="c-bar"><col class="c-num"><col class="c-num"></colgroup>' +
        '<thead><tr><th>参数</th><th class="num">年化</th>' +
        '<th>相对强度</th><th class="num">Sharpe</th><th class="num">最大回撤</th></tr></thead>' +
        '<tbody>' + sh + '</tbody></table>' : '<span class="muted">无</span>';
    }

    /* ---- 净值对比曲线（主策略 / ETF / 组合） ---- */
    renderEtfNav(d);
  }

  function renderEtfNav(d) {
    var el = byId('etfNav');
    if (!el) return;
    var series = d.nav_series || [];
    if (!series.length || series[0].combo == null) {
      // 后端只给了组合净值时的退化显示
      var nav = d.nav || [];
      if (nav.length < 2) { el.innerHTML = '<span class="muted">净值数据不足</span>'; return; }
      series = nav.map(function (x) { return { date: x.date, combo: x.nav }; });
    }
    var W = 1000, H = 320, P = { l: 56, r: 18, t: 14, b: 34 };
    var n = series.length;
    var keysDef = [['main', 'var(--text-2)'], ['etf', 'var(--accent)'], ['combo', 'var(--warn)']];
    if (series[0].combo3 != null) keysDef.push(['combo3', '#8a63d2']);
    var lo = Infinity, hi = -Infinity;
    series.forEach(function (p) {
      keysDef.forEach(function (k) {
        var v = p[k[0]];
        if (v != null && isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
      });
    });
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) { el.innerHTML = '<span class="muted">净值数据异常</span>'; return; }
    var pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    var X = function (i) { return P.l + i * (W - P.l - P.r) / Math.max(n - 1, 1); };
    var Y = function (v) { return P.t + (hi - v) / (hi - lo) * (H - P.t - P.b); };

    function pathOf(key) {
      var dstr = '', started = false;
      series.forEach(function (p, i) {
        var v = p[key];
        if (v == null || !isFinite(v)) { started = false; return; }
        dstr += (started ? ' L' : ' M') + X(i).toFixed(1) + ' ' + Y(v).toFixed(1);
        started = true;
      });
      return dstr;
    }
    // y 轴刻度
    var ticks = [], nt = 4;
    for (var i = 0; i <= nt; i++) {
      var v = lo + (hi - lo) * i / nt;
      ticks.push({ v: v, y: Y(v) });
    }
    var grid = ticks.map(function (t) {
      return '<line class="gridline" x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' + t.y.toFixed(1) +
        '" y2="' + t.y.toFixed(1) + '"/>' +
        '<text x="' + (P.l - 8) + '" y="' + (t.y + 3).toFixed(1) + '" text-anchor="end">' +
        t.v.toFixed(2) + '</text>';
    }).join('');
    // x 轴日期
    var xlab = '';
    [0, Math.floor(n / 3), Math.floor(2 * n / 3), n - 1].forEach(function (i) {
      xlab += '<text x="' + X(i).toFixed(1) + '" y="' + (H - P.b + 16) +
        '" text-anchor="middle">' + (series[i].date || '').slice(2) + '</text>';
    });
    // 组合曲线填充
    var comboPath = pathOf('combo');
    var fill = '';
    if (comboPath) {
      var lastX = X(n - 1), firstX = X(0);
      fill = '<path d="' + comboPath + ' L' + lastX.toFixed(1) + ' ' + (H - P.b) +
        ' L' + firstX.toFixed(1) + ' ' + (H - P.b) + ' Z" fill="url(#etfGrad)" opacity="0.10"/>';
    }
    var lines = keysDef.map(function (k) {
      var pth = pathOf(k[0]);
      if (!pth) return '';
      return '<path d="' + pth + '" fill="none" stroke="' + k[1] +
        '" stroke-width="' + (k[0] === 'combo' || k[0] === 'combo3' ? 2.0 : 1.3) + '" stroke-linejoin="round"' +
        (k[0] === 'main' ? ' opacity="0.85"' : '') + '/>';
    }).join('');

    el.innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" class="axis">' +
      '<defs><linearGradient id="etfGrad" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" stop-color="var(--warn)"/><stop offset="100%" stop-color="var(--warn)" stop-opacity="0"/>' +
      '</linearGradient></defs>' +
      grid + lines + fill +
      '<line class="base" x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' + (H - P.b) + '" y2="' + (H - P.b) + '"/>' +
      xlab +
      '<line class="hover-line" id="etfHover" x1="0" x2="0" y1="' + P.t + '" y2="' + (H - P.b) + '"/>' +
      '<rect x="' + P.l + '" y="' + P.t + '" width="' + (W - P.l - P.r) + '" height="' + (H - P.t - P.b) +
      '" fill="transparent" id="etfHit"/>' +
      '</svg>' +
      '<div class="tip" id="etfTip"></div>' +
      '<div class="chart-meta"><span>' + (series[0].date || '') + ' ~ ' +
      (series[n - 1].date || '') + '</span>' +
      '<span>区间起点归一化为 1.00</span></div>';

    // 悬停交互
    var svg = el.querySelector('svg');
    var hit = byId('etfHit');
    var tip = byId('etfTip');
    var hov = byId('etfHover');
    if (!svg || !hit || !tip) return;
    hit.addEventListener('mousemove', function (ev) {
      var r = svg.getBoundingClientRect();
      var sx = (ev.clientX - r.left) / r.width * W;
      var i = Math.round((sx - P.l) / ((W - P.l - P.r) / Math.max(n - 1, 1)));
      i = Math.max(0, Math.min(n - 1, i));
      var p = series[i];
      if (hov) { hov.setAttribute('x1', X(i).toFixed(1)); hov.setAttribute('x2', X(i).toFixed(1)); hov.style.opacity = 0.8; }
      var rows = [['主策略', p.main, 'var(--text-2)'], ['ETF趋势', p.etf, 'var(--accent)'], ['50/50', p.combo, 'var(--warn)']]
        .filter(function (x) { return x[1] != null && isFinite(x[1]); })
        .map(function (x) {
          return '<div><span style="color:' + x[2] + '">■</span> ' + x[0] +
            ' <b>' + x[1].toFixed(3) + '</b></div>';
        }).join('');
      tip.innerHTML = '<div class="muted">' + (p.date || '') + '</div>' + rows;
      tip.style.left = (X(i) / W * 100) + '%';
      tip.style.top = (Y(p.combo != null ? p.combo : lo) / H * 100) + '%';
      tip.style.opacity = 1;
    });
    hit.addEventListener('mouseleave', function () {
      tip.style.opacity = 0;
      if (hov) hov.style.opacity = 0;
    });
  }

  /* ============================== 板块 ETF 轮动 ============================== */
  function loadSectorRotation(force) {
    if (state.loaded.sector && !force) return;
    setText('secAsOf', '加载中…');
    apiFetch('/api/sector_rotation').then(function (d) {
      state.sector = d;
      state.loaded.sector = true;
      clearPageError('sector');
      renderSectorRotation(d);
      markOk();
    }).catch(function (err) {
      if (isServiceLoading(err)) { serviceLoadingPlaceholder('sector', '板块轮动'); return; }
      markDegraded();
      pageError('sector', err);
    });
  }

  function renderSectorRotation(d) {
    var bt = d.backtest || {}, val = d.val || {}, oos = d.oos || {};
    var sig = d.latest_signal || {};
    var picks = sig.picks || [];
    setText('secAsOf', '数据日期 ' + (d.asof || '—'));

    var chip = byId('secStatusChip');
    if (chip) {
      chip.innerHTML = picks.length
        ? '<span class="on">● 持仓 ' + picks.length + ' 个板块</span>　' +
          picks.map(function (p) { return esc(p.sector); }).join(' / ')
        : '<span class="off">○ 空仓观望</span>　无板块满足突破条件';
    }

    var kpi = [
      ['全期 年化', fmtPct(bt.cagr), 'Sharpe ' + fmtNum(bt.sharpe, 2)],
      ['全期 最大回撤', fmtPct(bt.max_dd), 'Calmar ' + fmtNum(bt.calmar, 2)],
      ['样本外(2022+) 年化', fmtPct(oos.cagr), 'Sharpe ' + fmtNum(oos.sharpe, 2)],
      ['平均仓位', fmtPct(d.avg_exposure), '交易 ' + fmtNum(d.n_trades, 0) + ' 笔']
    ];
    var ke = byId('secKpis');
    if (ke) {
      ke.innerHTML = kpi.map(function (k) {
        return '<div class="card kpi"><div class="kpi-label">' + k[0] + '</div>' +
          '<div class="kpi-value">' + k[1] + '</div>' +
          '<div class="kpi-sub muted small">' + k[2] + '</div></div>';
      }).join('');
    }

    // 最新信号卡
    var se = byId('secSignal');
    if (se) {
      if (picks.length) {
        se.innerHTML = picks.map(function (p, i) {
          return '<div class="cmp-card is-combo"><div class="cmp-top">' +
            '<span class="cmp-name">' + (i + 1) + '. ' + esc(p.sector) + '</span>' +
            '<span class="cmp-tag mono">' + esc(p.code) + '</span></div>' +
            '<div class="cmp-metrics">' +
            '<div><div class="k">价格</div><div class="v">' + fmtNum(p.price, 3) + '</div></div>' +
            '<div><div class="k">60日动量</div><div class="v">' + fmtPct(p.mom60) + '</div></div>' +
            '</div></div>';
        }).join('') +
          '<div class="muted small">突破板块数 ' + (sig.n_breakout || 0) +
          ' · 可交易 ETF ' + (sig.n_avail_etf || d.universe.length) + ' 只</div>';
      } else {
        se.innerHTML = '<div class="cmp-card"><div class="cmp-top">' +
          '<span class="cmp-name">空仓</span><span class="cmp-tag">现金 100% 国债</span></div>' +
          '<div class="cmp-foot">当前 ' + (d.universe || []).length +
          ' 个板块中 0 个满足「突破 55 日新高」→ 全部持现金</div></div>';
      }
    }

    // 分区间
    var pe = byId('secPeriods');
    if (pe) {
      var rows = [['全期', bt], ['验证期 2020-21', val], ['样本外 2022+', oos]];
      pe.innerHTML = rows.map(function (r) {
        var p = r[1] || {};
        return '<div class="cmp-card"><div class="cmp-top"><span class="cmp-name">' + r[0] +
          '</span></div><div class="cmp-metrics">' +
          '<div><div class="k">年化</div><div class="v">' + fmtPct(p.cagr) + '</div></div>' +
          '<div><div class="k">Sharpe</div><div class="v">' + fmtNum(p.sharpe, 2) + '</div></div>' +
          '<div><div class="k">回撤</div><div class="v">' + fmtPct(p.max_dd) + '</div></div>' +
          '<div><div class="k">Calmar</div><div class="v">' + fmtNum(p.calmar, 2) + '</div></div>' +
          '</div></div>';
      }).join('');
    }

    // 板块池表
    var uni = d.universe || [];
    var target = {};
    picks.forEach(function (p) { target[p.code] = true; });
    var info = byId('secPoolInfo');
    if (info) info.textContent = '共 ' + uni.length + ' 个板块（按 60 日动量排序）'
      + ' · 159949 已剔除（实为创业板50）';
    var uh = uni.map(function (u) {
      var pill = u.in_target ? '<span class="pill pill-on">持仓</span>'
               : (u.breakout ? '<span class="pill pill-off">突破</span>'
               : '<span class="pill pill-off">未突破</span>');
      return '<tr class="' + (u.in_target ? 'is-pick' : '') + '">' +
        '<td class="mono">' + esc(u.code) + '</td>' +
        '<td>' + esc(u.sector) + '</td>' +
        '<td class="num mono">' + fmtNum(u.price, 3) + '</td>' +
        '<td class="num">' + fmtPct(u.mom60) + '</td>' +
        '<td class="num">' + fmtPct(u.dist_to_high) + '</td>' +
        '<td class="num">' + fmtMoney(u.amount_20d) + '</td>' +
        '<td>' + pill + '</td></tr>';
    }).join('');
    var ue = byId('secUniverse');
    if (ue) {
      ue.innerHTML = '<table><colgroup><col style="width:10%"><col style="width:16%">' +
        '<col style="width:11%"><col style="width:13%"><col style="width:13%">' +
        '<col style="width:13%"><col style="width:12%"></colgroup>' +
        '<thead><tr><th>代码</th><th>板块</th><th class="num">最新价</th>' +
        '<th class="num">60日动量</th><th class="num">距55日高</th>' +
        '<th class="num">20日均额</th><th>状态</th></tr></thead><tbody>' + uh + '</tbody></table>';
    }

    // 净值曲线
    renderSectorNav(d);
  }

  function renderSectorNav(d) {
    var el = byId('secNav');
    if (!el) return;
    var series = d.nav_series || [];
    if (series.length < 2) { el.innerHTML = '<span class="muted">净值数据不足</span>'; return; }
    var W = 1000, H = 300, P = { l: 56, r: 18, t: 14, b: 32 };
    var n = series.length;
    var lo = Infinity, hi = -Infinity;
    series.forEach(function (p) { lo = Math.min(lo, p.nav); hi = Math.max(hi, p.nav); });
    var pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    function X(i) { return P.l + i * (W - P.l - P.r) / Math.max(n - 1, 1); }
    function Y(v) { return P.t + (hi - v) / (hi - lo) * (H - P.t - P.b); }
    var dstr = series.map(function (p, i) {
      return (i ? ' L' : ' M') + X(i).toFixed(1) + ' ' + Y(p.nav).toFixed(1);
    }).join('');
    var ticks = '', nt = 4;
    for (var i = 0; i <= nt; i++) {
      var v = lo + (hi - lo) * i / nt, y = Y(v);
      ticks += '<line class="gridline" x1="' + P.l + '" x2="' + (W - P.r) +
        '" y1="' + y.toFixed(1) + '" y2="' + y.toFixed(1) + '"/>' +
        '<text x="' + (P.l - 8) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end">' +
        v.toFixed(2) + '</text>';
    }
    var xlab = '';
    [0, Math.floor(n / 2), n - 1].forEach(function (i) {
      xlab += '<text x="' + X(i).toFixed(1) + '" y="' + (H - P.b + 16) +
        '" text-anchor="middle">' + (series[i].date || '').slice(2) + '</text>';
    });
    var fill = '<path d="' + dstr + ' L' + X(n - 1).toFixed(1) + ' ' + (H - P.b) +
      ' L' + X(0).toFixed(1) + ' ' + (H - P.b) + ' Z" fill="url(#secGrad)" opacity="0.12"/>';
    el.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="axis">' +
      '<defs><linearGradient id="secGrad" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" stop-color="var(--accent)"/><stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>' +
      '</linearGradient></defs>' + ticks +
      fill + '<path d="' + dstr + '" fill="none" stroke="var(--accent)" stroke-width="1.8"/>' +
      '<line class="base" x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' + (H - P.b) +
      '" y2="' + (H - P.b) + '"/>' + xlab + '</svg>' +
      '<div class="chart-meta"><span>' + (series[0].date || '') + ' ~ ' +
      (series[n - 1].date || '') + '</span><span>起点归一化为 1.00</span></div>';
  }

  /* ============================== 可转债 ============================== */
  function loadCbSignal(force) {
    if (state.loaded.cb && !force) return;
    setText('cbNavMeta', '加载中…');
    setText('cbHoldInfo', '加载中…');
    apiFetch('/api/cb_signal').then(function (d) {
      state.cb = d;
      state.loaded.cb = true;
      clearPageError('cb');
      renderCbSignal(d);
      markOk();
    }).catch(function (err) {
      if (isServiceLoading(err)) { serviceLoadingPlaceholder('cb', '可转债'); return; }
      markDegraded();
      pageError('cb', err);
    });
  }

  function renderCbSignal(d) {
    var env = d.environment || {}, adv = d.advice || {}, bt = d.backtest || {};
    var full = bt.full || {}, val = bt.val || {}, oos = bt.oos || {};
    var holds = d.holdings || [];
    setText('cbNavMeta', '数据日期 ' + (d.asof || '—'));

    var chip = byId('cbStatusChip');
    if (chip) {
      chip.innerHTML = '<span class="on">● 持仓 ' + holds.length + ' 只</span>　' +
        'YTM 中位 ' + fmtPct(env.ytm_median) +
        '　·　价格中位 ' + fmtNum(env.price_median, 1) + ' 元';
    }

    var sugg = (adv.suggest_pct || []);
    var kpi = [
      ['全期年化', fmtPct(full.cagr), 'Sharpe ' + fmtNum(full.sharpe, 2)],
      ['最大回撤', fmtPctU(full.max_dd), 'Calmar ' + fmtNum(full.calmar, 2)],
      ['样本外(2022+)', fmtPct(oos.cagr), 'Sharpe ' + fmtNum(oos.sharpe, 2)],
      ['建议仓位', (adv.level || '—'),
        sugg.length ? fmtPctU(sugg[0]) + ' ~ ' + fmtPctU(sugg[1]) : '—']
    ];
    var ke = byId('cbKpis');
    if (ke) {
      ke.innerHTML = kpi.map(function (k) {
        return '<div class="card kpi"><div class="kpi-label">' + k[0] + '</div>' +
          '<div class="kpi-value">' + k[1] + '</div>' +
          '<div class="kpi-sub muted small">' + k[2] + '</div></div>';
      }).join('');
    }

    // ---- 仓位建议
    var ae = byId('cbAdvice');
    if (ae) {
      var fp = adv.fwd_proxy;
      var html = '<div class="cmp-card is-combo"><div class="cmp-top">' +
        '<span class="cmp-name">' + esc(adv.level || '—') + '</span>' +
        '<span class="cmp-tag">YTM 历史分位 ' + fmtPctU(adv.ytm_percentile) + '</span></div>' +
        '<div class="cmp-metrics">' +
        '<div><div class="k">YTM 中位</div><div class="v">' + fmtPct(env.ytm_median) + '</div></div>' +
        '<div><div class="k">25/75 分位</div><div class="v">' + fmtPct(env.ytm_p25) +
        '/' + fmtPct(env.ytm_p75) + '</div></div>' +
        '<div><div class="k">&lt;100 元占比</div><div class="v">' +
        fmtPctU(env.price_below_100) + '</div></div>' +
        '<div><div class="k">在市只数</div><div class="v">' + (env.n_live || 0) + '</div></div>' +
        '</div>' +
        '<div class="cmp-foot">' + esc(adv.reason || '') + '</div>';
      if (fp) {
        html += '<div class="cmp-foot">历史同类环境 ' + (fp.n || 0) + ' 个样本，其后一年收益：中位 ' +
          fmtPct(fp.median) + '、p10 ' + fmtPct(fp.p10) + '、最差 ' + fmtPct(fp.min) +
          '<br><span class="muted small">口径：' + esc(fp.source || '—') + '</span></div>';
      }
      ae.innerHTML = html + '</div>';
    }
    setText('cbAdviceTag', '方向：YTM 高 = 便宜 = 该配');

    // ---- 净值曲线
    renderCbNav(d);
    // ---- 环境曲线
    renderCbEnv(d);

    // ---- 持仓表
    var info = byId('cbHoldInfo');
    if (info) {
      info.textContent = '本金 ' + fmtMoney(d.capital) + ' · 转债配置 ' +
        fmtPct(d.cb_weight) + ' · 已投 ' + fmtMoney(d.invested) +
        '（' + fmtPct(d.invested_pct) + '）';
    }
    var rows = holds.map(function (h) {
      var ytmCls = h.ytm > 0.02 ? 'pos' : (h.ytm < 0 ? 'neg' : '');
      return '<tr>' +
        '<td class="mono">' + esc(h.code) + '</td>' +
        '<td>' + esc(h.name) + '</td>' +
        '<td class="num mono">' + fmtNum(h.price, 2) + '</td>' +
        '<td class="num ' + ytmCls + '">' + fmtPct(h.ytm) + '</td>' +
        '<td class="num">' + fmtNum(h.years_left, 2) + '</td>' +
        '<td class="num">' + fmtNum(h.issue_size_yi, 1) + '</td>' +
        '<td class="num">' + fmtNum(h.amount_20d_yi, 2) + '</td>' +
        '<td>' + esc(h.rating || '—') + '</td>' +
        '<td class="num mono">' + (h.sheets || 0) + '</td>' +
        '<td class="num mono">' + fmtMoney(h.amount) + '</td></tr>';
    }).join('');
    var he = byId('cbHoldings');
    if (he) {
      he.innerHTML = rows
        ? '<table><colgroup><col style="width:9%"><col style="width:14%">' +
          '<col style="width:9%"><col style="width:9%"><col style="width:9%">' +
          '<col style="width:9%"><col style="width:9%"><col style="width:8%">' +
          '<col style="width:9%"><col style="width:11%"></colgroup>' +
          '<thead><tr><th>代码</th><th>名称</th><th class="num">价格</th>' +
          '<th class="num">YTM</th><th class="num">剩余年</th><th class="num">规模(亿)</th>' +
          '<th class="num">日额(亿)</th><th>评级</th><th class="num">张数</th>' +
          '<th class="num">金额</th></tr></thead><tbody>' + rows + '</tbody></table>'
        : '<span class="muted">无符合条件的转债</span>';
    }

    // ---- 组合对比
    renderCbCombos(d);
  }

  function renderCbNav(d) {
    var el = byId('cbNav');
    if (!el) return;
    var s = d.nav || [];
    if (s.length < 2) { el.innerHTML = '<span class="muted">净值数据不足</span>'; return; }
    // 用容器实际像素宽度做 viewBox，文字才按真实字号渲染（原来固定 1000 宽，半宽卡里被缩到 4~5px）
    var W = Math.max(320, Math.round(el.clientWidth || 1000));
    var H = 300, P = { l: 56, r: 18, t: 14, b: 32 };
    var n = s.length, lo = Infinity, hi = -Infinity;
    s.forEach(function (p) { lo = Math.min(lo, p.nav); hi = Math.max(hi, p.nav); });
    var pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
    function X(i) { return P.l + i * (W - P.l - P.r) / Math.max(n - 1, 1); }
    function Y(v) { return P.t + (hi - v) / (hi - lo) * (H - P.t - P.b); }
    var dstr = s.map(function (p, i) {
      return (i ? ' L' : ' M') + X(i).toFixed(1) + ' ' + Y(p.nav).toFixed(1);
    }).join('');
    var ticks = '';
    for (var i = 0; i <= 4; i++) {
      var v = lo + (hi - lo) * i / 4, y = Y(v);
      ticks += '<line class="gridline" x1="' + P.l + '" x2="' + (W - P.r) +
        '" y1="' + y.toFixed(1) + '" y2="' + y.toFixed(1) + '"/>' +
        '<text x="' + (P.l - 8) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end">' +
        v.toFixed(2) + '</text>';
    }
    var xlab = '';
    [0, Math.floor(n / 2), n - 1].forEach(function (i) {
      xlab += '<text x="' + X(i).toFixed(1) + '" y="' + (H - P.b + 16) +
        '" text-anchor="middle">' + (s[i].date || '').slice(2) + '</text>';
    });
    el.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="axis">' + ticks +
      '<path d="' + dstr + ' L' + X(n - 1).toFixed(1) + ' ' + (H - P.b) +
      ' L' + X(0).toFixed(1) + ' ' + (H - P.b) + ' Z" fill="var(--accent)" opacity="0.12"/>' +
      '<path d="' + dstr + '" fill="none" stroke="var(--accent)" stroke-width="1.8"/>' +
      '<line class="base" x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' + (H - P.b) +
      '" y2="' + (H - P.b) + '"/>' + xlab + '</svg>' +
      '<div class="chart-meta"><span>' + (s[0].date || '') + ' ~ ' +
      (s[n - 1].date || '') + '</span><span>起点归一化为 1.00</span></div>';
  }

  function renderCbEnv(d) {
    var el = byId('cbEnv');
    if (!el) return;
    var s = d.ytm_env || [];
    if (s.length < 2) { el.innerHTML = '<span class="muted">环境数据不足</span>'; return; }
    var W = Math.max(300, Math.round(el.clientWidth || 1000));
    var H = 240, P = { l: 48, r: 18, t: 14, b: 30 };
    var n = s.length, lo = Infinity, hi = -Infinity;
    s.forEach(function (p) { if (p.ytm != null) { lo = Math.min(lo, p.ytm); hi = Math.max(hi, p.ytm); } });
    if (!isFinite(lo)) { el.innerHTML = '<span class="muted">无数据</span>'; return; }
    var pad = (hi - lo) * 0.1 || 0.01; lo -= pad; hi += pad;
    function X(i) { return P.l + i * (W - P.l - P.r) / Math.max(n - 1, 1); }
    function Y(v) { return P.t + (hi - v) / (hi - lo) * (H - P.t - P.b); }
    var dstr = '', started = false;
    s.forEach(function (p, i) {
      if (p.ytm == null) { started = false; return; }
      dstr += (started ? ' L' : ' M') + X(i).toFixed(1) + ' ' + Y(p.ytm).toFixed(1);
      started = true;
    });
    var ticks = '';
    for (var i = 0; i <= 4; i++) {
      var v = lo + (hi - lo) * i / 4, y = Y(v);
      ticks += '<line class="gridline" x1="' + P.l + '" x2="' + (W - P.r) +
        '" y1="' + y.toFixed(1) + '" y2="' + y.toFixed(1) + '"/>' +
        '<text x="' + (P.l - 8) + '" y="' + (y + 3).toFixed(1) + '" text-anchor="end">' +
        (v * 100).toFixed(0) + '%</text>';
    }
    var zero = (lo < 0 && hi > 0) ? Y(0) : null;
    var xlab = '';
    [0, Math.floor(n / 2), n - 1].forEach(function (i) {
      xlab += '<text x="' + X(i).toFixed(1) + '" y="' + (H - P.b + 16) +
        '" text-anchor="middle">' + (s[i].date || '').slice(2) + '</text>';
    });
    // 末端当前值：一眼看出「我现在在哪一档」
    var last = null;
    for (var k = n - 1; k >= 0; k--) { if (s[k].ytm != null) { last = { i: k, v: s[k].ytm }; break; } }
    var endMark = '';
    if (last) {
      endMark = '<circle cx="' + X(last.i).toFixed(1) + '" cy="' + Y(last.v).toFixed(1) +
        '" r="2.8" fill="var(--warn)"/>' +
        '<text x="' + (X(last.i) - 7).toFixed(1) + '" y="' + (Y(last.v) - 8).toFixed(1) +
        '" text-anchor="end" fill="var(--warn)">' + (last.v * 100).toFixed(2) + '%</text>';
    }
    el.innerHTML = '<svg viewBox="0 0 ' + W + ' ' + H + '" class="axis">' + ticks +
      (zero != null ? '<line x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' + zero.toFixed(1) +
        '" y2="' + zero.toFixed(1) + '" stroke="var(--text-3)" stroke-dasharray="4 4" opacity=".65"/>' : '') +
      '<path d="' + dstr + '" fill="none" stroke="var(--warn)" stroke-width="1.6"/>' +
      endMark + xlab + '</svg>' +
      '<div class="chart-meta"><span>全市场 YTM 中位数 · 虚线 = YTM 0（高 = 便宜）</span>' +
      '<span>当前 ' + fmtPct((s[n - 1] || {}).ytm) + '</span></div>';
  }

  function renderCbCombos(d) {
    var el = byId('cbCombos');
    if (!el) return;
    var schemes = (d.schemes || []).slice().sort(function (a, b) {
      return ((b.stats || {}).sharpe || 0) - ((a.stats || {}).sharpe || 0);
    });
    var single = d.single || {};
    var nm = { stock: '股票', cb: '转债', etf: 'ETF', sector: '板块' };
    var head = '<div class="cmp-card"><div class="cmp-top"><span class="cmp-name">单资产基准</span>' +
      '<span class="cmp-tag">' + ((d.period || []).join(' ~ ')) + '</span></div>' +
      '<div class="cb-single">' + Object.keys(single).map(function (k) {
        var st = single[k] || {};
        return '<div class="cb-single-row"><span>' + (nm[k] || k) + '</span>' +
          '<span class="mono">' + fmtPct(st.cagr) + '</span>' +
          '<span class="mono muted">Sh ' + fmtNum(st.sharpe, 2) + '</span>' +
          '<span class="mono muted">DD ' + fmtPctU(st.max_dd) + '</span></div>';
      }).join('') + '</div></div>';
    var body = schemes.slice(0, 8).map(function (s) {
      var st = s.stats || {}, w = s.weights || {};
      var ws = Object.keys(w).filter(function (k) { return w[k] > 0.001; })
        .map(function (k) { return (nm[k] || k) + ' ' + fmtPctU(w[k], 0); }).join(' + ');
      return '<div class="cmp-card"><div class="cmp-top">' +
        '<span class="cmp-name">' + esc(s.name) + '</span>' +
        '<span class="cmp-tag mono">' + esc(ws) + '</span></div>' +
        '<div class="cmp-metrics">' +
        '<div><div class="k">年化</div><div class="v">' + fmtPct(st.cagr) + '</div></div>' +
        '<div><div class="k">Sharpe</div><div class="v">' + fmtNum(st.sharpe, 2) + '</div></div>' +
        '<div><div class="k">回撤</div><div class="v">' + fmtPctU(st.max_dd) + '</div></div>' +
        '<div><div class="k">Calmar</div><div class="v">' + fmtNum(st.calmar, 2) + '</div></div>' +
        '</div></div>';
    }).join('');
    el.innerHTML = head + body;
    var info = byId('cbComboInfo');
    if (info && (d.period || []).length) {
      info.textContent = '统一区间 ' + d.period.join(' ~ ') + ' · 月度再平衡 · 已扣成本';
    }
  }

  /* ============================== 三策略组合 ============================== */
  function loadComboPlan() {
    if (state.loaded.comboPlan) return;
    apiFetch('/api/combo_plan').then(function (d) {
      state.comboPlan = d;
      state.loaded.comboPlan = true;
      renderComboPlan(d);
    }).catch(function () { /* 组合方案缺失不影响主页面 */ });
  }

  function renderComboPlan(d) {
    setText('comboRecTag', '推荐：' + (d.recommended_label || d.recommended || '—'));
    var el = byId('comboPlan');
    if (!el) return;
    var rec = d.recommended;
    var schemes = d.schemes || [];
    var stTxt = { recommended: '推荐', control: '对照', live: '实盘现行', rejected: '已否' };

    function legText(legs) {
      var o = legs || {};
      return Object.keys(o).map(function (k) {
        return k + ' ' + fmtPctU(o[k], 0);
      }).join(' + ');
    }
    function cell(k, v) {
      return '<div><div class="k">' + k + '</div><div class="v">' + v + '</div></div>';
    }

    var cur = null;
    schemes.forEach(function (s) { if (s.name === rec) cur = s; });

    var html = '';
    if (cur) {
      var cf = cur.full || {}, co = cur.oos || {};
      html += '<div class="cmp-card is-combo"><div class="cmp-top">' +
        '<span class="cmp-name">' + esc(legText(cur.legs)) + '</span>' +
        '<span class="cmp-tag">每季度再平衡 · 已扣成本</span></div>' +
        '<div class="cmp-metrics">' +
        cell('年化', fmtPct(cf.cagr)) + cell('Sharpe', fmtNum(cf.sharpe, 2)) +
        cell('回撤', fmtPctU(cf.max_dd)) + cell('样本外', fmtPct(co.cagr)) +
        '</div></div>';
    }

    if (schemes.length) {
      html += '<table class="table"><thead><tr><th>方案</th>' +
        '<th class="num">年化</th><th class="num">Sharpe</th>' +
        '<th class="num">回撤</th><th class="num">Calmar</th>' +
        '<th class="num">样本外</th><th>说明</th></tr></thead><tbody>';
      schemes.forEach(function (s) {
        var f = s.full || {}, o = s.oos || {};
        html += '<tr' + (s.name === rec ? ' class="is-cur"' : '') + '><td>' +
          esc(s.name) + ' <span class="muted small">' + esc(legText(s.legs)) + '</span></td>' +
          '<td class="num">' + fmtPct(f.cagr) + '</td>' +
          '<td class="num">' + fmtNum(f.sharpe, 2) + '</td>' +
          '<td class="num">' + fmtPctU(f.max_dd) + '</td>' +
          '<td class="num">' + fmtNum(f.calmar, 2) + '</td>' +
          '<td class="num">' + fmtPct(o.cagr) + '</td>' +
          '<td class="small">' + esc(stTxt[s.status] || '') +
          (s.note ? ' · ' + esc(s.note) : '') + '</td></tr>';
      });
      html += '</tbody></table>';
    }

    var lc = d.legs_corr || {};
    var lk = Object.keys(lc);
    if (lk.length >= 2) {
      var pairs = [];
      for (var i = 0; i < lk.length; i++) {
        for (var j = i + 1; j < lk.length; j++) {
          var row = lc[lk[i]] || {};
          pairs.push(esc(lk[i]) + '-' + esc(lk[j]) + ' ' + fmtNum(row[lk[j]], 2));
        }
      }
      html += '<div class="muted small" style="margin-top:10px">腿间相关性：' +
        pairs.join(' · ') + '</div>';
    }

    var notes = d.notes || [];
    if (notes.length) {
      html += '<ul class="plain-list small" style="margin-top:8px">' +
        notes.map(function (t) { return '<li>' + esc(t) + '</li>'; }).join('') +
        '</ul>';
    }

    var ctr = d.controls || [];
    var rejected = ctr.filter(function (c) { return c.status === 'rejected'; });
    if (ctr.length) {
      html += '<div class="muted small" style="margin-top:8px">旧三策略口径（主 + 宽基 + 板块）：' +
        ctr.map(function (c) {
          return esc(c.name) + (c.status === 'rejected' ? '（已否）' : '');
        }).join(' · ');
      if (rejected.length) {
        html += '。板块腿原因：换 mom250 后 OOS 3.02% &lt; 基线 3.58%；' +
          '纯动量 + 仓位管理风险改善真、收益改善假，最多当 ≤10~15% 卫星。';
      }
      html += '</div>';
    }

    el.innerHTML = html;
  }

  /* ============================== 今日推荐 ============================== */
  var TODAY_FILTER = 'all';   // all | main | lowturn | etf | sector | cb
  var TODAY_KIND_OF = { main: ['stock'], lowturn: ['lowturn'],
                        etf: ['etf'], sector: ['sector_etf'], cb: ['cb'] };

  function loadToday(force) {
    if (state.loaded.today && !force) return;
    setText('todayStatus', '加载中…');
    apiFetch('/api/today_recommend').then(function (d) {
      state.today = d;
      state.loaded.today = true;
      clearPageError('today');
      renderToday(d);
      loadTodayCandidates();
      checkDataFreshness();
      markOk();
    }).catch(function (err) {
      if (isServiceLoading(err)) { serviceLoadingPlaceholder('today', '今日推荐'); return; }
      markDegraded();
      pageError('today', err);
    });
  }

  function todayFilteredRows(d) {
    var rows = (d && d.rows) || [];
    if (TODAY_FILTER === 'all') return rows;
    var kinds = TODAY_KIND_OF[TODAY_FILTER] || [];
    return rows.filter(function (r) { return kinds.indexOf(r.kind) >= 0; });
  }

  function setTodayFilter(k) {
    TODAY_FILTER = k || 'all';
    if (state.today) renderToday(state.today);
  }

  function renderTodayStrategies(d) {
    var host = byId('todayStrategies');
    if (!host) return;
    var ss = d.strategies || [];
    if (!ss.length) { host.innerHTML = '<span class="muted small">无策略分解数据</span>'; return; }
    host.innerHTML = ss.map(function (s) {
      var over = (s.amount || 0) > (s.budget || 0) * 1.005;
      var act = TODAY_FILTER === s.key;
      // 主策略的「年化」只是生产相位单点；有 phase_band 时必须把 8 相位区间一并给出，
      // 否则用户会把调仓日历的运气当成策略能力（详见 build/eval/phase_band.json）。
      var pb = s.phase_band;
      var pbLine = pb
        ? '<div class="st-foot muted small">8 个调仓相位 ' + fmtPctU(pb.cagr_min, 1) + '~' +
          fmtPctU(pb.cagr_max, 1) + '，中位 ' + fmtPctU(pb.cagr_median, 1) +
          '（上方年化为生产相位单点）</div>'
        : '';
      return '<button type="button" class="st-card' + (act ? ' is-active' : '') +
        '" data-st-filter="' + attr(s.key) + '">' +
        '<div class="st-top"><span class="st-name">' + esc(s.name) +
        (act ? ' <span class="st-on">筛选中</span>' : '') + '</span>' +
        '<span class="st-tag">' + esc(s.kind_cn) + ' · 权重 ' + fmtPctU(s.weight, 0) + '</span></div>' +
        '<div class="st-desc muted small">' + esc(s.desc || '') + '</div>' +
        '<div class="st-metrics">' +
        '<div><div class="k">标的</div><div class="v">' + (s.n || 0) + '</div></div>' +
        '<div><div class="k">预算</div><div class="v">' + fmtMoney(s.budget) + '</div></div>' +
        '<div><div class="k">已投</div><div class="v' + (over ? ' warn' : '') + '">' +
        fmtMoney(s.amount) + '</div></div>' +
        '<div><div class="k">年化 / Sharpe</div><div class="v">' +
        fmtPctU(s.cagr, 1) + ' / ' + fmtNum(s.sharpe, 2) + '</div></div>' +
        '</div>' +
        '<div class="st-foot">' + (s.jump === 'monthly'
          ? '<span class="link" data-jump="monthly">看本月整月名单 →</span>'
          : '<span class="muted">回测 ' + fmtPctU(s.cagr, 1) + ' · 回撤 ' + fmtPctU(s.max_dd, 1) + '</span>') +
        (over ? '<span class="warn small">整手取整后略超预算</span>' : '') +
        '</div>' + pbLine + '</button>';
    }).join('');
  }

  function renderToday(d) {
    var rowsAll = d.rows || [];
    var rows = todayFilteredRows(d);
    var groups = d.groups || {};
    var cash = d.cash || {};
    setText('todaySub', (d.parts || '') + ' · 本金 ' + fmtMoney(d.capital) +
      ' · 信号日 ' + (d.signal_date || '—'));
    var st = byId('todayStatus');
    if (st) {
      st.innerHTML = rowsAll.length
        ? '<span class="on">● 今日 ' + rowsAll.length + ' 个标的</span>　个股 ' + (d.n_stock || 0) +
          ' · 基金 ' + (d.n_etf || 0)
        : '<span class="off">○ 今日无标的</span>　全部持现金';
    }

    var kpi = [
      ['今日推荐标的', (d.n_stock || 0) + ' 股 + ' + (d.n_etf || 0) + ' 基', '见下方清单'],
      ['建议投入', fmtMoney(d.total_invested || 0), fmtPct((d.total_invested || 0) / (d.capital || 1)) + ' 仓位'],
      ['现金', fmtMoney(cash.value || 0), fmtPct(cash.pct || 0) + ' · 100% 国债'],
      ['再平衡', (d.rebalance_hint || '').indexOf('季度') >= 0 ? '本季度窗口' : '非窗口期',
       d.rebalance_hint || '']
    ];
    var ke = byId('todayKpis');
    if (ke) {
      ke.innerHTML = kpi.map(function (k) {
        return '<div class="card kpi"><div class="kpi-label">' + k[0] + '</div>' +
          '<div class="kpi-value">' + k[1] + '</div>' +
          '<div class="kpi-sub muted small">' + esc(k[2]) + '</div></div>';
      }).join('');
    }

    renderTodayStrategies(d);

    // 清单筛选标签
    var fe = byId('todayFilter');
    if (fe) {
      var chips = [{ k: 'all', t: '全部 ' + rowsAll.length }];
      (d.strategies || []).forEach(function (s) {
        chips.push({ k: s.key, t: s.name + ' ' + (s.n || 0) });
      });
      fe.innerHTML = chips.map(function (c) {
        return '<button type="button" class="fchip' + (TODAY_FILTER === c.k ? ' is-on' : '') +
          '" data-st-filter="' + attr(c.k) + '">' + esc(c.t) + '</button>';
      }).join('');
    }

    // 清单
    var le = byId('todayList');
    if (le) {
      if (!rows.length) {
        le.innerHTML = '<div class="empty-block">' +
          (TODAY_FILTER === 'all' ? '今日四策略均无入场信号 → 全部持现金'
                                  : '这一侧今日无标的（预算进入现金，或本月无变动）') + '</div>';
      } else {
        var kindColor = { stock: 'var(--text-2)', lowturn: 'var(--accent-2)', etf: 'var(--accent)',
                          sector_etf: 'var(--warn)', cb: 'var(--warn)' };
        var html = '<table class="table"><colgroup>' +
          '<col style="width:8%"><col style="width:13%"><col style="width:9%">' +
          '<col style="width:11%"><col style="width:8%"><col style="width:8%">' +
          '<col style="width:11%"><col style="width:32%"></colgroup>' +
          '<thead><tr><th>类别</th><th>名称</th><th>代码</th><th>板块/行业</th>' +
          '<th class="num">参考价</th><th class="num">数量</th><th class="num">金额</th>' +
          '<th>理由</th></tr></thead><tbody>';
        rows.forEach(function (r) {
          var color = kindColor[r.kind] || 'var(--text-2)';
          var hh = state.holdings || {};
          var act = '';
          if (hh[r.code] != null) {
            var cur = hh[r.code], want = r.shares || 0;
            var diff = want - cur;
            // 整手不同：股票/ETF 100 股(份)，转债 10 张
            var step = (r.unit === '张') ? 10 : 100;
            if (Math.abs(diff) < step) act = '<span class="pill pill-off">持有</span>';
            else if (diff > 0) act = '<span class="pill pill-on">买入 ' + Math.round(diff / step) * step + '</span>';
            else act = '<span class="pill pill-down">卖出 ' + Math.abs(Math.round(diff / step) * step) + '</span>';
          } else if (Object.keys(hh).length) {
            act = '<span class="pill pill-on">新买 ' + (r.shares || 0) + '</span>';
          }
          html += '<tr><td><span style="color:' + color + '">■</span> ' + esc(r.kind_cn) +
            (act ? ' ' + act : '') + '</td>' +
            '<td>' + esc(r.name || '') + '</td>' +
            '<td class="mono">' + esc(r.code) + '</td>' +
            '<td>' + esc(r.sector || '') + '</td>' +
            '<td class="num mono">' + fmtNum(r.price, 3) + '</td>' +
            '<td class="num">' + fmtNum(r.shares, 0) + ' ' + esc(r.unit || '') + '</td>' +
            '<td class="num">' + fmtMoney(r.amount) + '</td>' +
            '<td class="muted small">' + esc(r.reason || '') + '</td></tr>';
        });
        html += '</tbody></table>';
        le.innerHTML = html;
      }
    }

    // 分组
    var ge = byId('todayGroups');
    if (ge) {
      var label = { main: '主策略（荐股）', lowturn: '低换手月度（荐股）',
                    etf: '宽基 ETF（荐基）', sector: '板块 ETF（荐基）',
                    cb: '可转债（荐债）' };
      ge.innerHTML = Object.keys(label).filter(function (k) {
        return !!groups[k];
      }).map(function (k) {
        var g = groups[k] || { n: 0, amount: 0 };
        var bp = g.budget_pct != null ? g.budget_pct : 0;
        return '<div class="cmp-card' + (k === 'main' ? '' : ' is-combo') + '">' +
          '<div class="cmp-top"><span class="cmp-name">' + label[k] + '</span>' +
          '<span class="cmp-tag">预算 ' + fmtPct(bp) + '</span></div>' +
          '<div class="cmp-metrics">' +
          '<div><div class="k">标的数</div><div class="v">' + (g.n || 0) + '</div></div>' +
          '<div><div class="k">金额</div><div class="v">' + fmtMoney(g.amount || 0) + '</div></div>' +
          '<div><div class="k">占本金</div><div class="v">' +
          fmtPct((g.amount || 0) / (d.capital || 1)) + '</div></div>' +
          '<div><div class="k">剩余</div><div class="v">' +
          fmtMoney(Math.max(0, (d.capital || 0) * bp - (g.amount || 0))) + '</div></div>' +
          '</div></div>';
      }).join('');
    }

    // 空仓说明
    var ce = byId('todayCash');
    if (ce) {
      var notes = d.idle_notes || [];
      var html2 = '<div class="cmp-card"><div class="cmp-top"><span class="cmp-name">现金</span>' +
        '<span class="cmp-tag">100% 国债</span></div>' +
        '<div class="cmp-metrics"><div><div class="k">金额</div><div class="v">' +
        fmtMoney(cash.value || 0) + '</div></div>' +
        '<div><div class="k">占比</div><div class="v">' + fmtPct(cash.pct || 0) + '</div></div>' +
        '</div></div>';
      if (notes.length) {
        html2 += '<div class="muted small" style="margin-top:8px">' +
          notes.map(function (n) { return '· ' + esc(n); }).join('<br>') + '</div>';
      }
      ce.innerHTML = html2;
    }
  }

  /* ============================== 本月推荐（低换手月度） ============================== */
  function bindMonthlyPage() {
    var b = byId('btnMoToToday');
    if (b) b.addEventListener('click', function () { switchPage('today'); });
  }

  function loadMonthly(force) {
    if (state.loaded.monthly && !force) return;
    setText('moStatus', '加载中…');
    apiFetch('/api/monthly_recommend').then(function (d) {
      state.monthly = d;
      state.loaded.monthly = true;
      clearPageError('monthly');
      renderMonthly(d);
      markOk();
    }).catch(function (err) {
      if (isServiceLoading(err)) { serviceLoadingPlaceholder('monthly', '本月推荐'); return; }
      markDegraded();
      pageError('monthly', err);
    });
  }

  function renderMonthly(d) {
    var perf = d.perf || {};
    setText('moSub', (d.rule || '') + ' · 仅供研究');
    var st = byId('moStatus');
    if (st) {
      st.innerHTML = '<span class="on">● ' + esc(d.target_month || '—') + ' 月名单</span>　' +
        (d.n_pick || 0) + ' 只 · ' + esc(d.status_cn || '');
    }

    var kpi = [
      ['本月标的', (d.n_pick || 0) + ' 只', '目标月 ' + (d.target_month || '—')],
      ['本侧预算', fmtMoney(d.budget || 0), '权重 ' + fmtPctU(d.weight, 0) + ' · 本金 ' + fmtMoney(d.capital)],
      ['已投金额', fmtMoney(d.invested || 0),
       '占预算 ' + fmtPctU(d.invested_pct, 1) + ' · 现金 ' + fmtNum(d.cash, 0) + ' 元'],
      ['回测年化', fmtPctU(perf.cagr, 2), 'Sharpe ' + fmtNum(perf.sharpe, 2) +
       ' · 回撤 ' + fmtPctU(perf.max_dd, 1)]
    ];
    var ke = byId('moKpis');
    if (ke) {
      ke.innerHTML = kpi.map(function (k) {
        return '<div class="card kpi"><div class="kpi-label">' + k[0] + '</div>' +
          '<div class="kpi-value">' + k[1] + '</div>' +
          '<div class="kpi-sub muted small">' + esc(k[2]) + '</div></div>';
      }).join('');
    }

    // ---- 名单
    var rows = d.rows || [];
    setText('moListMeta', (d.sized_by === 'combo' ? '按组合 30% 预算切分' : '单策略满额口径') +
      ' · 信号 ' + (d.asof || '—') + ' · ' + (d.next_action || '—'));
    var le = byId('moList');
    if (le) {
      if (!rows.length) {
        le.innerHTML = '<div class="empty-block">本月无可选标的</div>';
      } else {
        var html = '<table class="table"><colgroup>' +
          '<col style="width:6%"><col style="width:16%"><col style="width:10%">' +
          '<col style="width:11%"><col style="width:10%"><col style="width:9%">' +
          '<col style="width:11%"><col style="width:9%"><col style="width:9%"></colgroup>' +
          '<thead><tr><th>#</th><th>名称</th><th>代码</th><th>行业</th>' +
          '<th class="num">20日换手</th><th class="num">市值(亿)</th><th class="num">金额</th>' +
          '<th class="num">股数</th><th>状态</th></tr></thead><tbody>';
        rows.forEach(function (r, i) {
          html += '<tr><td class="mono muted">' + (i + 1) + '</td>' +
            '<td>' + esc(r.name || '') + '</td>' +
            '<td class="mono">' + esc(r.code) + '</td>' +
            '<td>' + esc(r.sector || '') + '</td>' +
            '<td class="num mono">' + fmtPctU(r.turn20_pct, 2) + '</td>' +
            '<td class="num mono">' + (r.mcap_yi == null ? '—' : fmtNum(r.mcap_yi, 0)) + '</td>' +
            '<td class="num">' + fmtMoney(r.amount) + '</td>' +
            '<td class="num">' + fmtNum(r.shares, 0) + '</td>' +
            '<td>' + (r.is_new ? '<span class="pill pill-on">新进</span>'
                               : '<span class="pill pill-off">保留</span>') + '</td></tr>';
        });
        html += '</tbody></table>';
        le.innerHTML = html;
      }
    }

    // ---- 本期换仓
    var rot = d.rotation || {};
    setText('moRotMeta', '对比 ' + (rot.prev_month || '—') + ' 月');
    var re = byId('moRotation');
    if (re) {
      var ent = rot.entered || [], exi = rot.exited || [];
      re.innerHTML = '<div class="cmp-card"><div class="cmp-top">' +
        '<span class="cmp-name">换入 ' + ent.length + ' 只</span>' +
        '<span class="cmp-tag">买入</span></div>' +
        (ent.length ? ent.map(function (x) {
          return '<div class="mo-line"><span class="mono">' + esc(x.code) + '</span>' +
            '<span>' + esc(x.name || '') + '</span></div>';
        }).join('') : '<div class="muted small">无新进</div>') + '</div>' +
        '<div class="cmp-card"><div class="cmp-top">' +
        '<span class="cmp-name">换出 ' + exi.length + ' 只</span>' +
        '<span class="cmp-tag">卖出</span></div>' +
        (exi.length ? exi.map(function (x) {
          return '<div class="mo-line"><span class="mono">' + esc(x.code) + '</span>' +
            '<span>' + esc(x.name || '') + '</span></div>';
        }).join('') : '<div class="muted small">无移出</div>') + '</div>' +
        '<div class="muted small">保留 ' + (rot.kept || 0) + ' 只不动。' +
        '名单纯滚动时<b>不要清仓重建</b>——年换手 7.1x vs 24x，成本差 13pp。</div>';
    }

    // ---- 行业分布
    var se = byId('moSector');
    if (se) {
      var dist = d.sector_dist || [];
      se.innerHTML = dist.map(function (x) {
        var w = Math.max(2, (x.pct || 0) * 100);
        return '<div class="mo-dist-row"><span class="mo-dist-name">' + esc(x.sector) + '</span>' +
          '<span class="mo-dist-bar"><i style="width:' + w.toFixed(1) + '%"></i></span>' +
          '<span class="mono">' + fmtPctU(x.pct, 0) + ' · ' + x.n + ' 只</span></div>';
      }).join('');
    }

    // ---- 净值曲线 + 近 12 个月
    setText('moNavMeta', '净口径（只换变动，含 49.3bp 单边成本）· 最近 500 个交易日归一');
    renderMonthlyNav(d.nav_series || []);
    var me2 = byId('moMonths');
    if (me2) {
      var mh = d.monthly_returns || [];
      me2.innerHTML = mh.length ? '<div class="mo-months-title muted small">近 12 个月</div>' +
        mh.map(function (m) {
          var cls = m.ret >= 0 ? 'up' : 'down';
          return '<div class="mo-m"><div class="mo-m-k muted">' + esc(m.month.slice(2)) + '</div>' +
            '<div class="mo-m-v ' + cls + '">' + fmtPct(m.ret, 2) + '</div></div>';
        }).join('') : '';
    }

    // ---- 三种交易方式
    var ge = byId('moLedger');
    if (ge) {
      var cards = [
        { n: '只换变动（执行）', c: perf.cagr, t: '净年化', d: '年换手 ' + fmtNum(perf.turnover_ann, 1) + 'x',
          on: true },
        { n: '零成本（毛）', c: perf.gross_cagr, t: '毛年化', d: '同路径不扣费', on: false },
        { n: '每月全清全建', c: perf.strict_cagr, t: '净年化', d: '年换手 ' + fmtNum(perf.turnover_full_switch, 0) + 'x', on: false }
      ];
      ge.innerHTML = cards.map(function (x) {
        return '<div class="cmp-card' + (x.on ? ' is-combo' : '') + '">' +
          '<div class="cmp-top"><span class="cmp-name">' + x.n + '</span>' +
          '<span class="cmp-tag">' + x.t + '</span></div>' +
          '<div class="cmp-metrics"><div><div class="k">年化</div><div class="v">' +
          fmtPctU(x.c, 2) + '</div></div>' +
          '<div><div class="k">差</div><div class="v">' +
          (perf.cagr != null && x.c != null ? (x.c - perf.cagr >= 0 ? '+' : '') +
            ((x.c - perf.cagr) * 100).toFixed(2) + 'pp' : '—') + '</div></div></div>' +
          '<div class="cmp-foot">' + x.d + '</div></div>';
      }).join('') +
        '<div class="muted small">成本拖累 ' + fmtNum(perf.cost_drag_pp, 2) + 'pp/年' +
        '（' + fmtNum(perf.cost_bp, 2) + 'bp 单边 × 年换手 ' + fmtNum(perf.turnover_ann, 1) + 'x）；' +
        '名单月度重合 ' + fmtPctU(perf.avg_overlap, 0) + '，月度胜率 ' +
        fmtPctU(perf.win_rate, 0) + '（' + (perf.n_months || '—') + ' 个月）</div>';
    }

    // ---- 分年度 + 基准
    var ye = byId('moYearly');
    if (ye) {
      var yearly = d.yearly || {};
      var ks = Object.keys(yearly).sort();
      var body = ks.map(function (y) {
        var v = yearly[y];
        return '<tr><td class="mono">' + esc(y) + '</td>' +
          '<td class="num ' + (v >= 0 ? 'pos' : 'neg') + '">' + fmtPct(v, 2) + '</td></tr>';
      }).join('');
      var bench = (d.bench || []).map(function (b) {
        return '<tr><td>' + esc(b.name) + '</td>' +
          '<td class="num">' + fmtPctU(b.cagr, 2) + '</td>' +
          '<td class="num">' + fmtNum(b.sharpe, 2) + '</td>' +
          '<td class="num neg">' + fmtPctU(b.max_dd, 1) + '</td></tr>';
      }).join('');
      ye.innerHTML = '<table class="table"><colgroup><col style="width:50%"><col style="width:50%">' +
        '</colgroup><thead><tr><th>年份</th><th class="num">净收益</th></tr></thead><tbody>' +
        body + '</tbody></table>' +
        (bench ? '<table class="table" style="margin-top:10px"><colgroup><col style="width:34%">' +
          '<col style="width:22%"><col style="width:22%"><col style="width:22%"></colgroup>' +
          '<thead><tr><th>基准</th><th class="num">年化</th><th class="num">Sharpe</th>' +
          '<th class="num">回撤</th></tr></thead><tbody>' + bench + '</tbody></table>' : '');
    }
  }

  function renderMonthlyNav(series) {
    var el = byId('moNav');
    if (!el) return;
    if (!series || series.length < 2) { el.innerHTML = '<span class="muted">净值数据不足</span>'; return; }
    var W = 1000, H = 240, P = { l: 52, r: 16, t: 12, b: 26 };
    var n = series.length;
    var lo = Infinity, hi = -Infinity;
    series.forEach(function (p) {
      if (p.nav != null && isFinite(p.nav)) { lo = Math.min(lo, p.nav); hi = Math.max(hi, p.nav); }
    });
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) { el.innerHTML = '<span class="muted">净值数据异常</span>'; return; }
    var pad = (hi - lo) * 0.1; lo -= pad; hi += pad;
    var X = function (i) { return P.l + i * (W - P.l - P.r) / Math.max(n - 1, 1); };
    var Y = function (v) { return P.t + (hi - v) / (hi - lo) * (H - P.t - P.b); };
    var dstr = '', area = '';
    series.forEach(function (p, i) {
      dstr += (i ? ' L' : 'M') + X(i).toFixed(1) + ' ' + Y(p.nav).toFixed(1);
    });
    area = dstr + ' L' + X(n - 1).toFixed(1) + ' ' + (H - P.b) + ' L' + X(0).toFixed(1) + ' ' +
      (H - P.b) + ' Z';
    var ticks = [], nt = 4;
    for (var i = 0; i <= nt; i++) {
      var v = lo + (hi - lo) * i / nt;
      ticks.push({ v: v, y: Y(v) });
    }
    var grid = ticks.map(function (t) {
      return '<line class="gridline" x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' + t.y.toFixed(1) +
        '" y2="' + t.y.toFixed(1) + '"/>' +
        '<text x="' + (P.l - 8) + '" y="' + (t.y + 3).toFixed(1) + '" text-anchor="end">' +
        t.v.toFixed(2) + '</text>';
    }).join('');
    var xlab = '';
    [0, Math.floor(n / 3), Math.floor(2 * n / 3), n - 1].forEach(function (i) {
      var anchor = i === 0 ? 'start' : (i === n - 1 ? 'end' : 'middle');
      xlab += '<text x="' + X(i).toFixed(1) + '" y="' + (H - P.b + 15) +
        '" text-anchor="' + anchor + '">' + String(series[i].date || '').slice(2) + '</text>';
    });
    el.innerHTML = '<svg class="mo-nav-svg" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none">' +
      '<g class="axis">' + grid + xlab + '</g>' +
      '<path class="mo-nav-area" d="' + area + '"/>' +
      '<path d="' + dstr + '" fill="none" stroke="var(--accent-2)" stroke-width="1.8"/>' +
      '</svg>';
  }

  /* ============================== 全策略回测 ============================== */
  var BT_KIND = 'stock';
  var BT_KIND_FIELDS = {
    stock: [
      { id: 'btkModel', label: '模型', type: 'select', value: 'wf_h10',
        options: [{ v: 'wf_h10', t: '左侧主模型 wf_h10' },
                  { v: 'wf_h10_fund', t: '基本面增强 wf_h10_fund' },
                  { v: 'wf_v1', t: '旧版 wf_v1' }] },
      { id: 'btkNHold', label: '持仓数', type: 'number', value: 40, min: 1, max: 200 },
      { id: 'btkRebal', label: '调仓日（生产口径 40）', type: 'number', value: 40, min: 1, max: 60 },
      { id: 'btkMaxNew', label: '单日最大新开仓（-1=不限）', type: 'number', value: 6, min: -1, max: 50 },
      { id: 'btkCashBond', label: '现金替代·国债权重', type: 'number', value: 1.0, step: 0.05, min: 0, max: 1 },
      { id: 'btkCashGold', label: '现金替代·黄金权重（默认 0）', type: 'number', value: 0.0, step: 0.05, min: 0, max: 1 },
      { id: 'btkOverlay', label: '趋势/回撤 overlay', type: 'check', value: true },
      { id: 'btkStopLoss', label: '个股止损', type: 'check', value: true }
    ],
    etf: [
      { id: 'btkEntry', label: '突破窗口 entry_n', type: 'number', value: 55, min: 5, max: 250 },
      { id: 'btkExit', label: '出场窗口 exit_n（0=紧出场）', type: 'number', value: 0, min: 0, max: 120 },
      { id: 'btkMinAmt', label: '20日均额门槛（亿）', type: 'number', value: 0.5, step: 0.1, min: 0.1 }
    ],
    sector: [
      { id: 'btkEntry', label: '突破窗口 entry_n', type: 'number', value: 55, min: 5, max: 250 },
      { id: 'btkMom', label: '排序动量 mom_n', type: 'number', value: 60, min: 5, max: 250 },
      { id: 'btkTop', label: '持有板块数 top_n', type: 'number', value: 3, min: 1, max: 8 },
      { id: 'btkRebal', label: '调仓日（交易日）', type: 'number', value: 10, min: 1, max: 60 },
      { id: 'btkMinAmt', label: '20日均额门槛（亿）', type: 'number', value: 0.25, step: 0.05, min: 0.05 }
    ],
    cb: [
      { id: 'btkNHold', label: '持仓只数', type: 'number', value: 20, min: 5, max: 60 },
      { id: 'btkRebal', label: '调仓日（交易日）', type: 'number', value: 10, min: 1, max: 60 },
      { id: 'btkPoolMult', label: 'YTM 建池倍数', type: 'number', value: 3, step: 0.5, min: 1, max: 8 },
      { id: 'btkMinPx', label: '价格下限（元）', type: 'number', value: 90, step: 5, min: 50, max: 150 },
      { id: 'btkRank', label: '选券口径', type: 'select', value: 'ytm_duration',
        options: [{ v: 'ytm_duration', t: '高 YTM 建池 → 短久期优先（生产）' },
                  { v: 'ytm', t: '纯高 YTM' }] }
    ],
    lowturn: [
      { id: 'btkNHold', label: '持仓只数', type: 'number', value: 10, min: 3, max: 40 },
      { id: 'btkLookback', label: '换手率窗口（交易日）', type: 'number', value: 20, min: 5, max: 120 },
      { id: 'btkMinList', label: '上市最少交易日（一年≈250）', type: 'number', value: 250, min: 20, max: 1000 },
      { id: 'btkMinAdv', label: '20日均额下限（亿，0=不限）', type: 'number', value: 0, step: 0.1, min: 0 },
      { id: 'btkRebW', label: '每月把权重校准回等权', type: 'check', value: true }
    ],
    combo5050: [
      { id: 'btkWMain', label: '主策略权重', type: 'number', value: 0.5, step: 0.05, min: 0, max: 1 },
      { id: 'btkWETF', label: '宽基 ETF 权重', type: 'number', value: 0.5, step: 0.05, min: 0, max: 1 }
    ],
    combo_eq: [
      { id: 'btkIncCb', label: '纳入可转债（4 类资产等权）', type: 'check', value: false }
    ],
    combo_ivp: [
      { id: 'btkCovWin', label: '协方差窗口（交易日）', type: 'number', value: 252, min: 60, max: 1000 },
      { id: 'btkRebalDyn', label: '再平衡间隔（交易日）', type: 'number', value: 60, min: 5, max: 250 },
      { id: 'btkIncCb', label: '纳入可转债', type: 'check', value: false }
    ],
    combo_erc: [
      { id: 'btkCovWin', label: '协方差窗口（交易日）', type: 'number', value: 252, min: 60, max: 1000 },
      { id: 'btkRebalDyn', label: '再平衡间隔（交易日）', type: 'number', value: 60, min: 5, max: 250 },
      { id: 'btkIncCb', label: '纳入可转债', type: 'check', value: false }
    ],
    combo_mdp: [
      { id: 'btkCovWin', label: '协方差窗口（交易日）', type: 'number', value: 252, min: 60, max: 1000 },
      { id: 'btkRebalDyn', label: '再平衡间隔（交易日）', type: 'number', value: 60, min: 5, max: 250 },
      { id: 'btkIncCb', label: '纳入可转债', type: 'check', value: false }
    ],
    combo_fixed: [
      { id: 'btkWMain', label: '主策略权重', type: 'number', value: 0.4, step: 0.05, min: 0, max: 1 },
      { id: 'btkWETF', label: '宽基 ETF 权重', type: 'number', value: 0.3, step: 0.05, min: 0, max: 1 },
      { id: 'btkWSector', label: '板块轮动权重', type: 'number', value: 0.2, step: 0.05, min: 0, max: 1 },
      { id: 'btkWCb', label: '可转债权重', type: 'number', value: 0.1, step: 0.05, min: 0, max: 1 }
    ],
    combo_5050: [],
    combo: [
      { id: 'btkWMain', label: '主策略权重', type: 'number', value: 0.4, step: 0.05, min: 0, max: 1 },
      { id: 'btkWETF', label: '宽基 ETF 权重', type: 'number', value: 0.3, step: 0.05, min: 0, max: 1 },
      { id: 'btkWSector', label: '板块轮动权重', type: 'number', value: 0.2, step: 0.05, min: 0, max: 1 },
      { id: 'btkWCb', label: '可转债权重', type: 'number', value: 0.1, step: 0.05, min: 0, max: 1 }
    ]
  };

  function btKindParams(kind) {
    var fields = BT_KIND_FIELDS[kind] || [];
    var box = byId('btKindParams');
    if (!box) return;
    box.innerHTML = fields.map(function (f) {
      if (f.type === 'select') {
        return '<label class="field"><span class="field-label">' + f.label + '</span>' +
          '<select class="select" id="' + f.id + '">' +
          f.options.map(function (o) {
            return '<option value="' + o.v + '"' + (o.v === f.value ? ' selected' : '') + '>' +
              o.t + '</option>';
          }).join('') + '</select></label>';
      }
      if (f.type === 'check') {
        return '<label class="field field-check"><input type="checkbox" id="' + f.id + '"' +
          (f.value ? ' checked' : '') + '><span>' + f.label + '</span></label>';
      }
      var attrs = 'type="number" step="' + (f.step || 1) + '" value="' + f.value + '"';
      if (f.min != null) attrs += ' min="' + f.min + '"';
      if (f.max != null) attrs += ' max="' + f.max + '"';
      return '<label class="field"><span class="field-label">' + f.label + '</span>' +
        '<input class="input" id="' + f.id + '" ' + attrs + '></label>';
    }).join('');
  }

  function btBindKindPicker() {
    var picker = byId('btKindPicker');
    if (!picker) return;
    picker.addEventListener('click', function (ev) {
      var btn = ev.target.closest('.strat-card');
      if (!btn) return;
      var kind = btn.getAttribute('data-kind');
      if (!kind || kind === BT_KIND) return;
      BT_KIND = kind;
      picker.querySelectorAll('.strat-card').forEach(function (b) {
        b.classList.toggle('active', b.getAttribute('data-kind') === kind);
      });
      btKindParams(kind);
    });
    btKindParams(BT_KIND);
  }

  function btRunUnified() {
    var body = { kind: BT_KIND, start_date: valById('btStart') || '2020-01-02',
                 end_date: valById('btEnd') || '',
                 cost_model: checkedById('btkCostNet', false) ? 'net' : 'gross' };
    if (BT_KIND === 'stock') {
      body.model = valById('btkModel') || 'wf_h10';
      body.n_holdings = numById('btkNHold', 40);
      body.rebalance_days = numById('btkRebal', 40);
      body.max_new = numById('btkMaxNew', 6);
      body.cash_bond = numById('btkCashBond', 1.0);
      body.cash_gold = numById('btkCashGold', 0.0);
      body.overlay = checkedById('btkOverlay', true);
      body.stop_loss = checkedById('btkStopLoss', true);
    } else if (BT_KIND === 'etf') {
      body.entry_n = numById('btkEntry', 55);
      body.exit_n = numById('btkExit', 0);
      body.min_amount = numById('btkMinAmt', 0.5) * 1e8;
    } else if (BT_KIND === 'sector') {
      body.entry_n = numById('btkEntry', 55);
      body.mom_n = numById('btkMom', 60);
      body.top_n = numById('btkTop', 3);
      body.rebalance_days = numById('btkRebal', 10);
      body.min_amount = numById('btkMinAmt', 0.25) * 1e8;
    } else if (BT_KIND === 'cb') {
      body.n_holdings = numById('btkNHold', 20);
      body.rebalance_days = numById('btkRebal', 10);
      body.pool_mult = numById('btkPoolMult', 3);
      body.min_price = numById('btkMinPx', 90);
      body.rank = valById('btkRank') || 'ytm_duration';
    } else if (BT_KIND === 'lowturn') {
      body.n_holdings = numById('btkNHold', 10);
      body.lookback = numById('btkLookback', 20);
      body.min_list_days = numById('btkMinList', 250);
      body.min_adv20 = numById('btkMinAdv', 0);
      body.rebalance_weight = checkedById('btkRebW', true);
    } else if (BT_KIND === 'combo_5050') {
      body.kind = 'combo';
      body.method = 'fixed';
      body.weights = { main: 0.5, etf: 0.5, sector: 0, cb: 0 };
      body.label = '50/50 主策略+宽基ETF';
    } else if (BT_KIND === 'combo_fixed') {
      body.kind = 'combo';
      body.method = 'fixed';
      body.weights = { main: numById('btkWMain', 0.4), etf: numById('btkWETF', 0.3),
                       sector: numById('btkWSector', 0.2), cb: numById('btkWCb', 0.1) };
    } else if (BT_KIND === 'combo_eq' || BT_KIND === 'combo_ivp' ||
               BT_KIND === 'combo_erc' || BT_KIND === 'combo_mdp') {
      body.kind = 'combo';
      body.method = BT_KIND.replace('combo_', '');
      body.cov_window = numById('btkCovWin', 252);
      body.rebal = numById('btkRebalDyn', 60);
      body.include_cb = checkedById('btkIncCb', false);
    } else if (BT_KIND === 'combo5050') {
      body.kind = 'combo';
      body.method = 'fixed';
      body.weights = { main: 0.5, etf: 0.5, sector: 0, cb: 0 };
      body.label = '50/50 主策略+宽基ETF';
    } else if (BT_KIND === 'combo') {
      body.weights = { main: numById('btkWMain', 0.4), etf: numById('btkWETF', 0.3),
                       sector: numById('btkWSector', 0.2), cb: numById('btkWCb', 0.1) };
    }
    var btn = byId('btnRunBacktest');
    if (btn) { btn.disabled = true; btn.textContent = '回测中…'; }
    apiPost('/api/backtest_strategy', body).then(function (d) {
      renderUnifiedBacktest(d);
      if (btn) { btn.disabled = false; btn.textContent = '运行回测'; }
      var _h = byId('btResult');
      if (_h && /[?&]autobt=/.test(location.search || '')) {
        _h.scrollIntoView({ block: 'start' });
        window.scrollTo(0, 0);
      }
      toast('回测完成：' + d.kind_cn, 'ok');
    }).catch(function (err) {
      pageError('backtest', err);
      if (btn) { btn.disabled = false; btn.textContent = '运行回测'; }
    });
  }

  function checkedById(id, dflt) {
    var e = byId(id);
    return e ? !!e.checked : dflt;
  }

  function renderCostNote(d) {
    var c = d.cost || {};
    if (c.model !== 'net') {
      return '<div class="muted small" style="margin:-2px 0 12px">' +
        '<b>扣费前</b>口径：各策略内嵌成本不一致（个股已含 40bp 全成本，ETF/板块/转债仅含 5bp 滑点，' +
        '低换手按 49.3bp 逐笔扣），横向比较请勾选“扣费后”。</div>';
    }
    var line;
    if (d.kind === 'combo') {
      var by = c.by_asset || {}, lbl = {};
      (d.parts || []).forEach(function (p) { lbl[p.key] = p.label; });
      var seg = (d.parts || []).filter(function (p) { return by[p.key]; }).map(function (p) {
        return esc(lbl[p.key] || p.key) + ' ' + fmtNum((by[p.key].ann_cost || 0) * 100, 2) + '%';
      }).join(' · ');
      line = '组合加权年化成本 <b>' + fmtNum((c.ann_cost || 0) * 100, 2) + '%</b>' +
        (seg ? '（' + seg + '）' : '');
    } else if (c.basis === 'double_book') {
      line = '单边 <b>' + fmtNum(c.target_bp, 1) + 'bp</b>（双账本：毛/净各自逐笔扣费，不是按差额补扣）' +
        ' · 年换手 ' + fmtNum(c.turnover_ann, 1) + 'x（双边成交额口径）' +
        ' · 理论年化成本 <b>' + fmtNum((c.ann_cost || 0) * 100, 2) + '%</b>';
    } else {
      line = '单边 <b>' + fmtNum(c.target_bp, 1) + 'bp</b>（已含 ' + fmtNum(c.embedded_bp, 1) +
        'bp，补扣 ' + fmtNum(c.delta_bp, 1) + 'bp）· 年换手 ' + fmtNum(c.turnover_ann, 1) +
        'x · 年化成本 <b>' + fmtNum((c.ann_cost || 0) * 100, 2) + '%</b>';
    }
    var cmp = '';
    if (c.cagr_gross != null && c.cagr_net != null) {
      cmp = ' ｜ 扣费前 ' + fmtPct(c.cagr_gross) + ' → 扣费后 <b>' + fmtPct(c.cagr_net) + '</b>' +
        '（' + (c.drag_pp >= 0 ? '−' : '+') + fmtNum(Math.abs(c.drag_pp) * 100, 2) + 'pp）';
    } else if (!(c.delta_bp > 0.01)) {
      cmp = ' ｜ 该策略回测本身已按全成本扣费，扣费前后一致';
    }
    return '<div class="muted small" style="margin:-2px 0 12px"><b>扣费后</b>口径：' + line + cmp + '</div>';
  }

  function renderUnifiedBacktest(d) {
    var m = d.metrics || {}, p = d.periods || {};
    var host = byId('btResult');
    if (!host) return;
    host.hidden = false;
    host.setAttribute('data-kind', d.kind || '');
    state.btNav = (d.nav || []);
    var cards = [
      ['年化收益', fmtPct(m.cagr), d.kind_cn + (d.cost && d.cost.model === 'net' ? ' · 扣费后' : '')],
      ['Sharpe', fmtNum(m.sharpe, 2), '波动 ' + fmtPct(m.ann_vol)],
      ['最大回撤', fmtPct(m.max_dd), 'Calmar ' + fmtNum(m.calmar, 2)],
      ['样本外年化', fmtPct((p.oos || {}).cagr), '2022 起']
    ];
    var parts = '';
    if (d.parts && d.parts.length) {
      parts = '<div class="muted small" style="margin-bottom:8px">构成：' +
        d.parts.map(function (x) {
          return esc(x.label) + ' <b>' + fmtPct(x.weight) + '</b>';
        }).join(' + ') + '</div>';
    }
    var ex = d.excess || {};
    var exKeys = ['csi_all', 'hs300', 'csi500', 'csi1000'];
    var exNames = { csi_all: '中证全指', hs300: '沪深300',
                    csi500: '中证500', csi1000: '中证1000' };
    var exHtml = '';
    if (Object.keys(ex).length) {
      var cells = exKeys.filter(function (k) { return ex[k]; }).map(function (k) {
        var e = ex[k];
        var cls = e.excess_ann >= 0 ? 'pos' : 'neg';
        var ir = (e.info_ratio === null || e.info_ratio === undefined) ? '—'
                 : e.info_ratio.toFixed(2);
        return '<div class="kpi"><div class="kpi-label">超额 vs ' + exNames[k] + '</div>' +
               '<div class="kpi-value ' + cls + '">' + fmtPct(e.excess_ann) + '</div>' +
               '<div class="kpi-sub">基准 ' + fmtPct(e.bench_ann) +
               ' · 信息比率 ' + ir + '</div></div>';
      }).join('');
      exHtml = '<div class="grid grid-4" style="margin-bottom:12px">' + cells + '</div>';
    }
    var html = parts + '<div class="grid grid-4" style="margin-bottom:12px">' +
      cards.map(function (c) {
        return '<div class="card kpi"><div class="kpi-label">' + c[0] + '</div>' +
          '<div class="kpi-value">' + c[1] + '</div>' +
          '<div class="kpi-sub muted small">' + esc(c[2]) + '</div></div>';
      }).join('') + '</div>';

    // ---------- 成本口径说明
    html += renderCostNote(d);

    // ---------- 超额收益（相对各基准）
    if (exHtml) { html += exHtml; }

    // ---------- 分区间：三卡对比 + 衰减判定
    html += renderPeriodCards(p);

    // ---------- 净值曲线（坐标轴 + 网格 + 渐变填充 + 回撤阴影 + 悬停）
    var nav = d.nav || [];
    if (nav.length > 2) {
      html += renderNavChart(nav, m);
    }
    host.innerHTML = html;
    btBindChartHover();
  }

  function renderPeriodCards(p) {
    var full = p.full || {}, val = p.val || {}, oos = p.oos || {};
    var rows = [
      { key: 'full', label: '全期', note: '含验证期与样本外', tag: 'full', d: full },
      { key: 'val', label: '验证期', note: '2020-2021 · 牛市', tag: 'val', d: val },
      { key: 'oos', label: '样本外', note: '2022 至今 · 真实水平', tag: 'oos', d: oos }
    ];

    // 衰减判定
    var dec = null;
    if (val.cagr != null && oos.cagr != null) {
      dec = (val.cagr !== 0) ? (1 - oos.cagr / val.cagr) : null;
    }
    var verdict = '';
    if (dec != null) {
      var lvl, cls, msg;
      if (dec >= 0.6) { lvl = '严重衰减'; cls = 'bad';
        msg = '样本外年化只剩验证期的 ' + Math.round((1 - dec) * 100) + '%，策略大概率依赖特定行情，实盘不可按回测预期'; }
      else if (dec >= 0.3) { lvl = '明显衰减'; cls = 'warn';
        msg = '样本外年化衰减 ' + Math.round(dec * 100) + '%，需按样本外水平下调预期'; }
      else if (dec >= 0) { lvl = '轻度衰减'; cls = 'ok';
        msg = '样本外年化衰减 ' + Math.round(dec * 100) + '%，属于正常范围'; }
      else { lvl = '样本外更好'; cls = 'good';
        msg = '样本外年化高于验证期 ' + Math.round(-dec * 100) + '%，稳健性较好'; }
      verdict = '<div class="pd-verdict ' + cls + '">' +
        '<span class="pd-badge">' + lvl + '</span>' +
        '<span>' + msg + '。</span>' +
        (dec >= 0.6 ? '<span class="pd-strong">建议不要用这份回测作为实盘预期。</span>' : '') +
        '</div>';
    }

    // 卡片
    var maxCagr = Math.max(Math.abs(full.cagr || 0), Math.abs(val.cagr || 0),
                           Math.abs(oos.cagr || 0), 1e-9);
    var cards = rows.map(function (r) {
      var d = r.d || {};
      var yrs = d.n_days ? (d.n_days / 252) : null;
      var w = Math.min(100, Math.abs(d.cagr || 0) / maxCagr * 100);
      var isOos = r.key === 'oos';
      var sharpeCls = (d.sharpe == null) ? '' : (d.sharpe >= 1 ? 'up' : (d.sharpe >= 0.5 ? '' : 'down'));
      return '<div class="pd-card' + (isOos ? ' pd-oos' : '') + '">' +
        '<div class="pd-head"><span class="pd-name">' + r.label + '</span>' +
        '<span class="pd-note">' + esc(r.note) + '</span></div>' +
        '<div class="pd-cagr ' + (isOos ? ((d.cagr || 0) >= 0 ? 'up' : 'down') : 'neutral') + '">' +
        fmtPct(d.cagr) + '</div>' +
        '<div class="pd-bar"><i style="width:' + w.toFixed(1) + '%"></i></div>' +
        '<div class="pd-metrics">' +
        '<div><span class="k">年数</span><span class="v">' +
        (yrs ? yrs.toFixed(1) + ' 年' : '—') + '</span></div>' +
        '<div><span class="k">Sharpe</span><span class="v ' + sharpeCls + '">' +
        fmtNum(d.sharpe, 2) + '</span></div>' +
        '<div><span class="k">最大回撤</span><span class="v down">' +
        fmtPct(d.max_dd) + '</span></div>' +
        '<div><span class="k">Calmar</span><span class="v">' +
        fmtNum(d.calmar, 2) + '</span></div>' +
        '</div>' +
        (isOos && val.cagr != null && d.cagr != null
          ? '<div class="pd-delta">较验证期 <b class="' +
            (d.cagr - val.cagr >= 0 ? 'up' : 'down') + '">' +
            (d.cagr - val.cagr >= 0 ? '+' : '') +
            ((d.cagr - val.cagr) * 100).toFixed(2) + 'pp</b></div>'
          : '') +
        '</div>';
    }).join('');

    return '<div class="pd-wrap"><div class="pd-grid">' + cards + '</div>' +
      verdict + '</div>';
  }

  function renderNavChart(nav, metrics) {
    var W = 980, H = 320, P = { l: 58, r: 18, t: 16, b: 34 };
    var n = nav.length;
    var lo = Infinity, hi = -Infinity;
    for (var i = 0; i < n; i++) { lo = Math.min(lo, nav[i].nav); hi = Math.max(hi, nav[i].nav); }
    var pad = (hi - lo) * 0.08 || 0.05;
    lo -= pad; hi += pad;
    var X = function (i) { return P.l + i * (W - P.l - P.r) / Math.max(n - 1, 1); };
    var Y = function (v) { return P.t + (hi - v) / (hi - lo) * (H - P.t - P.b); };

    var pts = [], peak = -Infinity, dd = new Array(n);
    for (var i2 = 0; i2 < n; i2++) {
      pts.push(X(i2).toFixed(1) + ',' + Y(nav[i2].nav).toFixed(1));
      peak = Math.max(peak, nav[i2].nav);
      dd[i2] = nav[i2].nav / peak - 1;
    }
    var line = 'M' + pts.join(' L');
    var fill = line + ' L' + X(n - 1).toFixed(1) + ' ' + (H - P.b) +
      ' L' + X(0).toFixed(1) + ' ' + (H - P.b) + ' Z';

    // y 轴刻度
    var grid = '', nt = 4;
    for (var g = 0; g <= nt; g++) {
      var v = lo + (hi - lo) * g / nt, y = Y(v);
      grid += '<line class="gridline" x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' +
        y.toFixed(1) + '" y2="' + y.toFixed(1) + '"/>' +
        '<text x="' + (P.l - 9) + '" y="' + (y + 3.5).toFixed(1) +
        '" text-anchor="end">' + v.toFixed(2) + '</text>';
    }
    // x 轴日期
    var xlab = '';
    var tickIdx = [0, Math.floor(n * 0.25), Math.floor(n * 0.5), Math.floor(n * 0.75), n - 1];
    tickIdx.forEach(function (i3) {
      xlab += '<text x="' + X(i3).toFixed(1) + '" y="' + (H - P.b + 17) +
        '" text-anchor="middle">' + (nav[i3].date || '').slice(2) + '</text>';
    });
    // 最大回撤区间阴影
    var mdd = 0, pk = 0, mstart = 0, mend = 0, curPeakIdx = 0;
    for (var i4 = 0; i4 < n; i4++) {
      if (nav[i4].nav >= nav[curPeakIdx].nav) curPeakIdx = i4;
      var d4 = nav[i4].nav / nav[curPeakIdx].nav - 1;
      if (d4 < mdd) { mdd = d4; mstart = curPeakIdx; mend = i4; }
    }
    var ddBand = '';
    if (mend > mstart) {
      ddBand = '<rect x="' + X(mstart).toFixed(1) + '" y="' + P.t + '" width="' +
        (X(mend) - X(mstart)).toFixed(1) + '" height="' + (H - P.t - P.b) +
        '" fill="var(--up)" opacity="0.07"/>' +
        '<text x="' + ((X(mstart) + X(mend)) / 2).toFixed(1) + '" y="' + (P.t + 14) +
        '" text-anchor="middle" font-size="10" fill="var(--up)">最大回撤 ' +
        (mdd * 100).toFixed(1) + '%</text>';
    }

    return '<div class="bt-chart" style="margin-top:16px;position:relative">' +
      '<svg viewBox="0 0 ' + W + ' ' + H + '" class="taxis" preserveAspectRatio="none">' +
      '<defs><linearGradient id="btNavGrad" x1="0" y1="0" x2="0" y2="1">' +
      '<stop offset="0%" stop-color="var(--accent)" stop-opacity="0.42"/>' +
      '<stop offset="100%" stop-color="var(--accent)" stop-opacity="0"/>' +
      '</linearGradient></defs>' +
      grid + ddBand +
      '<path d="' + fill + '" fill="url(#btNavGrad)"/>' +
      '<path d="' + line + '" fill="none" stroke="var(--accent)" stroke-width="1.9" ' +
      'stroke-linejoin="round" stroke-linecap="round"/>' +
      '<line class="btbase" x1="' + P.l + '" x2="' + (W - P.r) + '" y1="' + (H - P.b) +
      '" y2="' + (H - P.b) + '"/>' +
      xlab +
      '<line class="bthover" id="btHoverLine" x1="0" x2="0" y1="' + P.t + '" y2="' +
      (H - P.b) + '" style="opacity:0"/>' +
      '<circle id="btHoverDot" r="3.5" fill="var(--accent)" style="opacity:0"/>' +
      '<rect x="' + P.l + '" y="' + P.t + '" width="' + (W - P.l - P.r) + '" height="' +
      (H - P.t - P.b) + '" fill="transparent" id="btHit"/>' +
      '</svg>' +
      '<div class="bttip" id="btTip"></div>' +
      '<div class="chart-meta"><span>' + nav[0].date + ' ~ ' + nav[n - 1].date + '</span>' +
      '<span>起点归一化 1.00 · 末端 ' + nav[n - 1].nav.toFixed(3) + '</span></div>' +
      '</div>';
  }

  function btBindChartHover() {
    var hit = byId('btHit');
    if (!hit) return;
    var svg = hit.ownerSVGElement;
    var tip = byId('btTip'), hov = byId('btHoverLine'), dot = byId('btHoverDot');
    var nav = (state.btNav || []);
    if (!nav.length) return;
    var W = 980, H = 320, P = { l: 58, r: 18, t: 16, b: 34 };
    var n = nav.length;
    var lo = Infinity, hi = -Infinity;
    nav.forEach(function (x) { lo = Math.min(lo, x.nav); hi = Math.max(hi, x.nav); });
    var pad = (hi - lo) * 0.08 || 0.05; lo -= pad; hi += pad;
    var X = function (i) { return P.l + i * (W - P.l - P.r) / Math.max(n - 1, 1); };
    var Y = function (v) { return P.t + (hi - v) / (hi - lo) * (H - P.t - P.b); };
    hit.addEventListener('mousemove', function (ev) {
      var r = svg.getBoundingClientRect();
      var sx = (ev.clientX - r.left) / r.width * W;
      var i = Math.round((sx - P.l) / ((W - P.l - P.r) / Math.max(n - 1, 1)));
      i = Math.max(0, Math.min(n - 1, i));
      var p = nav[i];
      if (hov) { hov.setAttribute('x1', X(i)); hov.setAttribute('x2', X(i)); hov.style.opacity = 0.7; }
      if (dot) { dot.setAttribute('cx', X(i)); dot.setAttribute('cy', Y(p.nav)); dot.style.opacity = 1; }
      tip.innerHTML = '<div class="muted">' + p.date + '</div><div>净值 <b>' +
        p.nav.toFixed(4) + '</b></div>';
      tip.style.left = (X(i) / W * 100) + '%';
      tip.style.top = (Y(p.nav) / H * 100) + '%';
      tip.style.opacity = 1;
    });
    hit.addEventListener('mouseleave', function () {
      tip.style.opacity = 0;
      if (hov) hov.style.opacity = 0;
      if (dot) dot.style.opacity = 0;
    });
  }

  function valById(id) { var e = byId(id); return e ? e.value : ''; }
  function numById(id, dflt) { var v = parseFloat(valById(id)); return isFinite(v) ? v : dflt; }

  function apiPost(path, body) {
    return fetch(path, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(function (r) {
      if (!r.ok) {
        return r.text().then(function (t) {
          var msg = t;
          try { msg = (JSON.parse(t) || {}).error || t; } catch (e) { /* ignore */ }
          throw new Error(msg || ('HTTP ' + r.status));
        });
      }
      return r.json();
    });
  }

  /* ============ 今日推荐：候选榜 + 信号说明 ============ */
  function loadTodayCandidates() {
    var model = (state.defaultModel || 'wf_h10');
    apiFetch('/api/recommend?top=60&model=' + encodeURIComponent(model)).then(function (d) {
      renderTodayCandidates(d || {});
    }).catch(function () {
      setHtml('todayCandidates', '<div class="muted small">候选榜加载失败</div>');
    });
  }

  function renderTodayCandidates(d) {
    var rows = d.rows || d.items || d.list || d.candidates || d.holdings || [];
    var host = byId('todayCandidates');
    if (!host) return;
    if (!rows.length) {
      host.innerHTML = '<div class="muted small">暂无候选数据</div>';
    } else {
      var html = '<table class="table"><colgroup><col style="width:6%"><col style="width:10%">' +
        '<col style="width:15%"><col style="width:13%"><col style="width:9%">' +
        '<col style="width:9%"><col style="width:12%"><col style="width:26%"></colgroup>' +
        '<thead><tr><th>#</th><th>代码</th><th>名称</th><th>行业</th>' +
        '<th class="num">价格</th><th class="num">涨跌</th><th class="num">复合分</th>' +
        '<th>入选理由</th></tr></thead><tbody>';
      rows.slice(0, 60).forEach(function (r, i) {
        html += '<tr><td class="muted mono">' + (i + 1) + '</td>' +
          '<td class="mono">' + esc(r.code || '') + '</td>' +
          '<td>' + esc(r.name || '') + '</td>' +
          '<td>' + esc(r.industry || '') + '</td>' +
          '<td class="num mono">' + fmtNum(r.price, 2) + '</td>' +
          '<td class="num ' + ((r.chg || 0) >= 0 ? 'up' : 'down') + '">' + fmtPct(r.chg) + '</td>' +
          '<td class="num mono">' + fmtNum(r.score, 3) + '</td>' +
          '<td class="muted small">' + esc(r.reason || '低波+超跌反转，复合分靠前') + '</td></tr>';
      });
      html += '</tbody></table>';
      host.innerHTML = html;
    }

    var note = byId('todaySignalNote');
    if (note) {
      var items = [
        '荐股来自主策略 wf_h10：34 个量价因子、10 日标签、walk-forward 每季度 refit、40 只等权；',
        '荐基来自宽基 ETF 趋势（突破 55 日新高）与板块 ETF 轮动（突破 + 动量前 3）；',
        '复合分 = ICIR 加权 + 岭回归的混合打分，经行业中性与市值中性处理；',
        '单票上限 8%、行业上限 25%、双创合计上限 50%；',
        '每 10 个交易日调仓；开盘涨停买不进、跌停卖不出可顺延；',
        '候选榜仅用于解释荐股来源，不代表建议买入（可能因仓位/上限未入选）。'
      ];
      if (d && d.note) items.unshift(d.note);
      note.innerHTML = items.map(function (t) { return '<div>· ' + esc(t) + '</div>'; }).join('');
    }
  }

  /* ============ 自动回测钩子（供无头截图/自测） ============ */
  function autoRunBacktest() {
    var m = /[?&]autobt=(stock|etf|sector|combo)/.exec(location.search || '');
    if (!m) return;
    var kind = m[1];
    var go = function () {
      var card = document.querySelector('.strat-card[data-kind="' + kind + '"]');
      if (!card) { setTimeout(go, 300); return; }
      card.click();
      setTimeout(function () {
        var b = byId('btnRunBacktest');
        if (b) b.click();
        setTimeout(function () {
          var h = byId('btResult');
          if (h) h.scrollIntoView({ block: 'start' });
        }, 2500);
      }, 800);
    };
    var start = function () {
      try { switchPage('backtest'); } catch (e) { /* ignore */ }
      setTimeout(go, 600);
    };
    setTimeout(start, 1800);
  }

  /* ============================== 交互相应（导出/持仓/过期警示） ============================== */

  // ---- 导出今日清单 CSV
  function bindTodayExport() {
    var btn = byId('btnTodayExport');
    if (!btn) return;
    btn.addEventListener('click', function () {
      btn.disabled = true;
      var old = btn.textContent;
      btn.textContent = '导出中…';
      fetch('/api/today_export').then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.blob();
      }).then(function (blob) {
        var d = state.today || {};
        var name = '今日推荐_' + (d.signal_date || '') + '.csv';
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = name;
        document.body.appendChild(a);
        a.click();
        setTimeout(function () {
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }, 200);
        toast('已导出 ' + name, 'ok');
      }).catch(function (err) {
        toast('导出失败：' + err.message, 'error');
      }).then(function () {
        btn.disabled = false;
        btn.textContent = old;
      });
    });
  }

  // ---- 持仓录入
  var HOLDINGS_KEY = 'qf_holdings_v1';

  function parseHoldings(text) {
    var out = {};
    (text || '').split(/\r?\n/).forEach(function (line) {
      line = line.trim();
      if (!line || line.charAt(0) === '#') return;
      var parts = line.split(/[,，\s]+/).filter(Boolean);
      if (parts.length < 2) return;
      var code = parts[0].replace(/\D/g, '').padStart(6, '0');
      var sh = parseInt(parts[1], 10);
      if (code.length === 6 && isFinite(sh) && sh > 0) out[code] = sh;
    });
    return out;
  }

  function bindHoldingsPanel() {
    var openBtn = byId('btnHoldingsInput');
    var panel = byId('holdingsPanel');
    var ta = byId('holdingsText');
    var applyBtn = byId('btnHoldingsApply');
    var clearBtn = byId('btnHoldingsClear');
    if (!openBtn || !panel) return;

    // 恢复上次输入
    try {
      var saved = localStorage.getItem(HOLDINGS_KEY);
      if (saved && ta) ta.value = saved;
    } catch (e) { /* ignore */ }

    openBtn.addEventListener('click', function () {
      panel.hidden = !panel.hidden;
      if (!panel.hidden && ta) ta.focus();
    });

    if (applyBtn) {
      applyBtn.addEventListener('click', function () {
        var txt = ta ? ta.value : '';
        try { localStorage.setItem(HOLDINGS_KEY, txt); } catch (e) { /* ignore */ }
        var h = parseHoldings(txt);
        state.holdings = h;
        var n = Object.keys(h).length;
        setText('holdingsHint', n
          ? '已录入 ' + n + ' 只持仓，清单已标注买/卖/持有'
          : '未录入持仓（或格式不对），仅显示目标持仓');
        renderToday(state.today || {});
        if (n) toast('已录入 ' + n + ' 只持仓', 'ok');
      });
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', function () {
        if (ta) ta.value = '';
        state.holdings = {};
        try { localStorage.removeItem(HOLDINGS_KEY); } catch (e) { /* ignore */ }
        setText('holdingsHint', '已清空');
        renderToday(state.today || {});
      });
    }
  }

  // ---- 数据过期警示
  window.__qfCheckStale = function (a) { checkDataFreshness(a); };

  // 后端检测到「磁盘已更新、服务内存还旧」时会自动重载（约 30 秒）。
  // 这段时间前端每 6 秒重拉一次今日推荐，用户不用手动刷新，也不会再看到
  // 「请点开始更新」这种已经做过一次还让他再做的提示。
  var _reloadPoll = { timer: null, tries: 0 };
  function scheduleReloadPoll() {
    if (_reloadPoll.timer) return;
    _reloadPoll.tries = 0;
    _reloadPoll.timer = setInterval(function () {
      _reloadPoll.tries++;
      if (_reloadPoll.tries > 12) {          // 72 秒还没好就放弃，避免无限轮询
        clearInterval(_reloadPoll.timer);
        _reloadPoll.timer = null;
        return;
      }
      loadToday(true);
    }, 6000);
  }
  function stopReloadPoll() {
    if (_reloadPoll.timer) {
      clearInterval(_reloadPoll.timer);
      _reloadPoll.timer = null;
    }
  }

  function checkDataFreshness(forceAsof) {
    // 优先用后端诊断（区分「数据过期」与「模型滞后」）
    var d = state.today || {};
    var st = d.stale || null;
    var bar = byId('dataStaleBar');
    if (!bar) return;
    if (!st || st.action !== 'reload') stopReloadPoll();

    // 后端已有诊断 → 直接用
    if (st && st.level) {
      if (st.level === 'ok') { bar.hidden = true; return; }
      bar.hidden = false;
      bar.className = 'stale-bar' + (st.level === 'bad' ? ' stale-bad' : ' stale-warn');
      var prefix = st.level === 'bad' ? '⚠️ ' : '';
      var action = st.action === 'retrain' ? '请点「开始更新」重训模型'
        : (st.action === 'reload' ? ''
        : (st.action === 'update_data' ? '请点「开始更新」拉取数据' : ''));
      setText('dataStaleMsg', prefix + (st.message || '') +
        (action ? '，' + action : ''));
      // 按钮文案随动作变化
      var btn = byId('btnStaleUpdate');
      if (st.action === 'reload') scheduleReloadPoll();
      if (btn) {
        // 自动重载期间按钮置灰：这时再点更新只会让等待变长
        btn.disabled = (st.action === 'reload');
        if (!btn.disabled) {
          btn.textContent = st.action === 'retrain' ? '重训模型'
            : (st.action === 'update_data' ? '更新数据' : '去更新');
        } else {
          btn.textContent = '正在重载…';
        }
      }
      return;
    }

    // 兜底：用 signal_date 粗略判断（后端旧版兼容）
    var asof = forceAsof || d.signal_date || (state.meta && state.meta.data_last_date) || '';
    if (!asof) return;
    var cur = new Date(asof + 'T00:00:00');
    var tradingDays = 0;
    var now = new Date();
    while (cur < now) {
      cur.setDate(cur.getDate() + 1);
      var wd = cur.getDay();
      if (wd !== 0 && wd !== 6) tradingDays++;
    }
    if (tradingDays <= 1) { bar.hidden = true; return; }
    bar.hidden = false;
    bar.className = 'stale-bar' + (tradingDays >= 3 ? ' stale-bad' : ' stale-warn');
    setText('dataStaleMsg',
      (tradingDays >= 3 ? '⚠️ 信号已 ' : '信号已 ') + tradingDays +
      ' 个交易日未更新（最后 ' + asof + '）');
  }

  function bindStaleUpdate() {
    var btn = byId('btnStaleUpdate');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var d = state.today || {};
      var st = d.stale || {};
      var act = st.action || 'update_data';
      // 服务端正在自动重载 → 别再启动一次 20 分钟的更新（这正是以前死循环的地方）
      if (act === 'reload') {
        toast('服务正在重载行情（约 30 秒），完成后会自动刷新，不用再点更新', 'info');
        scheduleReloadPoll();
        return;
      }
      // 模型滞后 → 只重跑下游（快）；数据过期 → 全量更新
      var payload = (act === 'retrain')
        ? { skip_data: true, skip_factors: true }
        : {};
      var msg = (act === 'retrain')
        ? '将重训模型并重算回测（约 5 分钟，跳过数据与因子重建）。继续？'
        : '将拉取最新行情并重建因子（约 15~30 分钟）。继续？';
      if (!confirm(msg)) return;
      btn.disabled = true;
      btn.textContent = '启动中…';
      fetch('/api/pipeline/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      }).then(function (r) { return r.json(); }).then(function (res) {
        btn.disabled = false;
        if (res.started) {
          toast('已启动，完成后刷新页面即可（pid ' + res.pid + '）', 'info');
          btn.textContent = '运行中…';
        } else {
          toast('未能启动：' + (res.reason || '未知'), 'error');
          btn.textContent = act === 'retrain' ? '重训模型' : '更新数据';
        }
      }).catch(function (e) {
        btn.disabled = false;
        toast('启动失败：' + e, 'error');
        btn.textContent = '重试';
      });
    });
  }

  // ---- 分阶段加载提示（冷启动 35s 内提示当前阶段）
  var LOAD_STAGES = [
    { until: 18, text: '正在加载行情数据（约 15 秒）…' },
    { until: 40, text: '正在预热模型打分（约 20 秒）…' },
    { until: 999, text: '即将就绪…' }
  ];

  function initLoadingStages() {
    var t0 = Date.now();
    var timer = setInterval(function () {
      var m = state.meta || {};
      var loaded = !!(m.server && m.server.loaded);
      if (loaded) { clearInterval(timer); return; }
      var sec = (Date.now() - t0) / 1000;
      var st = LOAD_STAGES.find(function (x) { return sec < x.until; }) || LOAD_STAGES[LOAD_STAGES.length - 1];
      var el = byId('loadStage');
      if (el) el.textContent = st.text + ' ' + Math.round(sec) + 's';
      var status = byId('sideInfo');
      if (status && /加载/.test(status.textContent || '')) {
        status.textContent = st.text;
      }
    }, 1000);
    setTimeout(function () { clearInterval(timer); }, 120000);
  }


  // ========================================================= 配置对比页
  function fmtPct2(x) {
    if (x == null || !isFinite(x)) return '—';
    return (x * 100).toFixed(2) + '%';
  }

  // ========================================================= 配置对比页
  var PF_META = {
    'EQ 等权': { key: 'eq', desc: '各 1/3，无参数' },
    '5050 主+宽基': { key: 'fixed', desc: '主策略与宽基各半' },
    '50/30/20': { key: 'fixed', desc: '现行默认配置' },
    'IVP 逆波动': { key: 'ivp', desc: '按波动率倒数分配' },
    'ERC 真等风险': { key: 'erc', desc: '各资产风险贡献相等' },
    'MDP 最大分散': { key: 'mdp', desc: '最大化分散化比率' }
  };

  function fmtPct2(x) {
    if (x == null || !isFinite(x)) return '—';
    return (x * 100).toFixed(2) + '%';
  }

  function renderPfMethods(d) {
    var host = document.getElementById('pfMethods');
    if (!host) return;
    var ms = d.methods || {};
    var names = Object.keys(ms);
    if (!names.length) { host.textContent = '无数据'; return; }

    // 找 Sharpe 最高的
    var bestName = names.reduce(function (a, b) {
      var sa = (ms[a].full || {}).sharpe || 0, sb = (ms[b].full || {}).sharpe || 0;
      return sa >= sb ? a : b;
    });

    host.innerHTML = names.map(function (n) {
      var v = ms[n], f = v.full || {}, va = v.val || {}, o = v.oos || {};
      var w = v.weights || [0, 0, 0];
      var isBest = (n === bestName);
      var meta = PF_META[n] || {};
      var method = meta.key || 'fixed';

      var bar = '<div class="pf-bar">' +
        (w[0] > 0 ? '<div class="pf-bar-m" style="flex:' + w[0] + '">' +
          (w[0] > 0.12 ? Math.round(w[0] * 100) + '%' : '') + '</div>' : '') +
        (w[1] > 0 ? '<div class="pf-bar-w" style="flex:' + w[1] + '">' +
          (w[1] > 0.12 ? Math.round(w[1] * 100) + '%' : '') + '</div>' : '') +
        (w[2] > 0 ? '<div class="pf-bar-s" style="flex:' + w[2] + '">' +
          (w[2] > 0.12 ? Math.round(w[2] * 100) + '%' : '') + '</div>' : '') +
        '</div>';

      return '<div class="pf-m' + (isBest ? ' best' : '') + '" ' +
        'data-method="' + method + '" data-name="' + esc(n) + '">' +
        '<div class="pf-m-hd"><span class="pf-m-name">' + esc(n) + '</span>' +
        (isBest ? '<span class="pf-m-badge">Sharpe 最优</span>' : '') + '</div>' +
        '<div class="muted small" style="margin-bottom:9px">' + esc(meta.desc || '') + '</div>' +
        '<div class="pf-m-rows">' +
          '<div class="pf-m-kv"><span>年化</span><span>' + fmtPct2(f.cagr) + '</span></div>' +
          '<div class="pf-m-kv"><span>Sharpe</span><span>' + (f.sharpe || 0).toFixed(3) + '</span></div>' +
          '<div class="pf-m-kv"><span>最大回撤</span><span>' + fmtPct2(f.max_dd) + '</span></div>' +
          '<div class="pf-m-kv"><span>Calmar</span><span>' + (f.calmar || 0).toFixed(2) + '</span></div>' +
          '<div class="pf-m-kv"><span>验证期</span><span>' + fmtPct2(va.cagr) + '</span></div>' +
          '<div class="pf-m-kv"><span>样本外</span><span>' + fmtPct2(o.cagr) + '</span></div>' +
        '</div>' +
        '<div class="pf-m-w">' + bar + '</div>' +
        '</div>';
    }).join('');

    // 点击跳到回测
    host.addEventListener('click', function (ev) {
      var card = ev.target.closest('.pf-m');
      if (!card) return;
      var method = card.getAttribute('data-method');
      var kindMap = { eq: 'combo_eq', ivp: 'combo_ivp', erc: 'combo_erc', mdp: 'combo_mdp' };
      var kind = kindMap[method] || 'combo_fixed';
      try {
        BT_KIND = kind;
        switchPage('backtest');
        var picker = document.getElementById('btKindPicker');
        if (picker) {
          picker.querySelectorAll('.strat-card').forEach(function (b) {
            b.classList.toggle('active', b.getAttribute('data-kind') === kind);
          });
        }
        setTimeout(function () { btKindParams(kind); }, 60);
      } catch (e) { /* ignore */ }
    });
  }

  function renderPfWeights(d) {
    var host = document.getElementById('pfWeightsBar');
    if (!host) return;
    var ms = d.methods || {};
    var names = Object.keys(ms);
    var LAB = ['主策略 M', '宽基 W', '板块 S'];
    var CLS = ['pf-bar-m', 'pf-bar-w', 'pf-bar-s'];
    var html = '<table class="table"><thead><tr><th style="width:150px">方法</th>' +
      '<th>权重分布</th></tr></thead><tbody>';
    names.forEach(function (n) {
      var w = (ms[n].weights || [0, 0, 0]);
      var bar = '<div class="pf-bar" style="margin:0">';
      for (var i = 0; i < 3; i++) {
        if (w[i] > 0.001) {
          bar += '<div class="' + CLS[i] + '" style="flex:' + w[i] + '">' +
            (w[i] > 0.09 ? Math.round(w[i] * 100) + '%' : '') + '</div>';
        }
      }
      bar += '</div>';
      html += '<tr><td>' + esc(n) + '</td><td>' + bar +
        '<div class="muted small" style="margin-top:3px">' +
        w.map(function (x, i) { return LAB[i] + ' ' + (x * 100).toFixed(1) + '%'; }).join(' · ') +
        '</div></td></tr>';
    });
    html += '</tbody></table>';
    html += '<div class="pf-legend" style="margin-top:10px">' +
      '<span><i class="pf-bar-m" style="background:var(--text-3)"></i>主策略</span>' +
      '<span><i class="pf-bar-w" style="background:var(--accent)"></i>宽基 ETF</span>' +
      '<span><i class="pf-bar-s" style="background:var(--warn)"></i>板块轮动</span></div>';
    host.innerHTML = html;
  }

  function renderPortfolioCorr(d) {
    var host = document.getElementById('portfolioCorr');
    if (!host) return;
    var c = d.corr || {};
    var keys = Object.keys(c);
    if (!keys.length) { host.textContent = '无数据'; return; }
    var NAMES = { M: '主策略', W: '宽基ETF', S: '板块' };
    var html = '<table class="table"><thead><tr><th></th>' +
      keys.map(function (k) { return '<th>' + esc(NAMES[k] || k) + '</th>'; }).join('') +
      '</tr></thead><tbody>';
    keys.forEach(function (r) {
      html += '<tr><td><b>' + esc(NAMES[r] || r) + '</b></td>';
      keys.forEach(function (col) {
        var v = (c[r] || {})[col];
        var style = '';
        if (v != null && r !== col) {
          if (v < 0.3) style = 'color:var(--down);font-weight:600';
          else if (v > 0.5) style = 'color:var(--warn)';
        }
        html += '<td style="' + style + '">' + (v == null ? '—' : v.toFixed(3)) + '</td>';
      });
      html += '</tr>';
    });
    html += '</tbody></table>' +
      '<p class="muted small" style="margin-top:8px">' +
      '<span style="color:var(--down)">绿色</span> = &lt; 0.3（分散好）　' +
      '<span style="color:var(--warn)">黄色</span> = &gt; 0.5（相关性高）</p>';
    host.innerHTML = html;
  }

  function renderPfGrid(d) {
    var host = document.getElementById('pfGrid');
    if (!host) return;
    var g = d.grid || {};
    var keys = Object.keys(g);
    if (!keys.length) { host.textContent = '无数据'; return; }
    var rows = keys.map(function (k) {
      var m = k.match(/cov(\d+)_rb(\d+)/);
      return { cov: m ? m[1] : '?', rb: m ? m[2] : '?', d: g[k] };
    }).sort(function (a, b) { return (+a.cov - +b.cov) || (+a.rb - +b.rb); });
    var best = rows.reduce(function (a, b) {
      return (a.d.sharpe || 0) >= (b.d.sharpe || 0) ? a : b;
    });
    var html = '<table class="table"><thead><tr>' +
      '<th>协方差窗口</th><th>再平衡</th><th>年化</th><th>Sharpe</th><th>最大回撤</th><th>Calmar</th>' +
      '</tr></thead><tbody>';
    rows.forEach(function (r) {
      var isBest = (r === best);
      html += '<tr' + (isBest ? ' style="background:var(--down-soft)"' : '') + '>' +
        '<td>' + r.cov + ' 日</td><td>' + r.rb + ' 日</td>' +
        '<td>' + fmtPct2(r.d.cagr) + '</td>' +
        '<td><b>' + (r.d.sharpe || 0).toFixed(3) + '</b></td>' +
        '<td>' + fmtPct2(r.d.max_dd) + '</td>' +
        '<td>' + (r.d.calmar || 0).toFixed(2) + '</td></tr>';
    });
    html += '</tbody></table>';
    var sh = rows.map(function (r) { return r.d.sharpe || 0; });
    html += '<p class="muted small" style="margin-top:8px">' +
      'Sharpe 中位 ' + (sh.sort(function (a, b) { return a - b; })[Math.floor(sh.length / 2)] || 0).toFixed(3) +
      ' ｜ 范围 ' + Math.min.apply(null, sh).toFixed(3) + ' ~ ' +
      Math.max.apply(null, sh).toFixed(3) + '（参数高原 = 稳健）</p>';
    host.innerHTML = html;
  }

  function renderCandidates(d) {
    var host = document.getElementById('candidateTable');
    if (!host) return;
    var rs = d.rolling_start || {}, starts = d.starts || [];
    var latest = d.latest || {};
    var tags = Object.keys(rs);
    if (!tags.length) { host.textContent = '无数据'; return; }

    var baseTag = tags.indexOf('wf_h10') >= 0 ? 'wf_h10' : tags[0];
    var html = '<h4 style="margin:0 0 10px;font-size:13px;color:var(--txt2)">' +
      '① 若在某个时点上线，到最新日期的年化</h4>';
    html += '<table class="table"><thead><tr><th>上线时点</th>' +
      tags.map(function (t) {
        return '<th>' + esc(t.replace('wf_h10', '').replace(/_/g, '') || 'base') + '</th>';
      }).join('') + '</tr></thead><tbody>';
    starts.forEach(function (st) {
      html += '<tr><td>' + esc(st) + '</td>';
      var baseV = (rs[baseTag] || {})[st];
      tags.forEach(function (t) {
        var v = (rs[t] || {})[st];
        var style = '';
        if (v != null && baseV != null && t !== baseTag) {
          style = v > baseV ? 'color:var(--down)' : (v < baseV ? 'color:var(--up)' : '');
        }
        html += '<td style="' + style + '">' + (v == null ? '—' : fmtPct2(v)) + '</td>';
      });
      html += '</tr>';
    });
    html += '</tbody></table>';

    html += '<h4 style="margin:20px 0 10px;font-size:13px;color:var(--txt2)">' +
      '② 最新落盘指标</h4>';
    html += '<table class="table"><thead><tr><th>模型</th><th>说明</th><th>年化</th>' +
      '<th>Sharpe</th><th>最大回撤</th><th>Calmar</th></tr></thead><tbody>';
    Object.keys(latest).forEach(function (t) {
      var v = latest[t] || {};
      var isBase = (t === 'wf_h10');
      html += '<tr' + (isBase ? ' style="background:var(--surface-2)"' : '') + '>' +
        '<td><b>' + esc(t) + '</b>' + (isBase ? ' <span class="muted small">基线</span>' : '') +
        '</td><td class="muted small">' + esc(v.desc || '') + '</td>' +
        '<td>' + fmtPct2(v.cagr) + '</td><td>' + (v.sharpe || 0).toFixed(3) + '</td>' +
        '<td>' + fmtPct2(v.max_dd) + '</td><td>' + (v.calmar || 0).toFixed(2) + '</td></tr>';
    });
    html += '</tbody></table>';
    html += '<p class="muted small" style="margin-top:10px">' + esc(d.note || '') +
      '　判断标准：连续 6 个月跑赢 → 提升为默认；连续 6 个月跑输 → 淘汰。</p>';
    host.innerHTML = html;
  }

  function loadPortfolio() {
    function get(url) {
      return fetch(url).then(function (r) { return r.json(); });
    }
    get('/api/combo_methods').then(function (d) {
      renderPfMethods(d);
      renderPfWeights(d);
      renderPortfolioCorr(d);
      renderPfGrid(d);
    }).catch(function (e) {
      var h = document.getElementById('pfMethods');
      if (h) h.textContent = '加载失败: ' + e;
    });
    get('/api/candidates').then(renderCandidates).catch(function (e) {
      var h = document.getElementById('candidateTable');
      if (h) h.textContent = '加载失败: ' + e;
    });
  }

  // ========================================================= 一键更新管道
  var PIPE_TIMER = null;
  var PIPE_BOUND = false;
  var PIPE_LABELS = {
    data: '数据增量更新', factors: '因子面板重建', model: '模型重训',
    backtest: '主策略回测', etf: 'ETF 趋势回测', sector: '板块轮动回测',
    combo: '50/50 组合', three: '三策略组合', erc: 'ERC 配置对比',
    tracking: '候选跟踪', numbers: '刷新基准数字',
    report_s1: '同步报告摘要', signal: '今日信号',
    verify: '数据源一致性检测'
  };

  function fmtDur(s) {
    if (s == null || !isFinite(s)) return '';
    s = Math.max(0, Math.round(s));
    if (s >= 3600) return Math.floor(s / 3600) + ' 小时 ' + Math.round((s % 3600) / 60) + ' 分';
    if (s >= 60) return Math.floor(s / 60) + ' 分 ' + (s % 60) + ' 秒';
    return s + ' 秒';
  }

  // 按钮联动：运行中禁用，避免重复触发
  function setPipeBusy(busy) {
    ['btnPipeRun', 'btnPipeFast'].forEach(function (id) {
      var b = document.getElementById(id);
      if (!b) return;
      b.disabled = !!busy;
      b.classList.toggle('is-busy', !!busy);
    });
  }

  // 总进度条：百分比 + 第几步 + 本阶段/整体剩余时间
  function renderPipeProgress(d) {
    var wrap = document.getElementById('pipeProg');
    var fill = document.getElementById('pipeProgFill');
    if (!wrap || !fill) return;
    // 启动窗口期：已提交但 status.json 还没被新进程改写，此时渲染的是上一轮的
    // 旧状态（ok:true），会把「正在启动」覆盖成「上次已完成」，必须跳过。
    if (state.pipeLaunching && !d.running) return;
    var p = d.progress || {};
    var running = !!d.running;
    var pct = (p.pct == null ? 0 : Number(p.pct));
    if (!isFinite(pct)) pct = 0;

    if (!running && !p.current_key && !d.ok && !d.finished_at) { wrap.hidden = true; return; }
    if (!running && d.never_run) { wrap.hidden = true; return; }
    wrap.hidden = false;
    wrap.classList.toggle('done', !running && d.ok === true);
    wrap.classList.toggle('fail', !running && d.ok === false);

    var showPct = Math.round(pct * 1000) / 10;
    fill.style.width = Math.max(running ? 2 : 0, Math.min(100, showPct)) + '%';
    setText('pipeProgPct', (showPct >= 100 ? 100 : showPct.toFixed(1)) + '%');

    var stepEl = document.getElementById('pipeProgStep');
    var subEl = document.getElementById('pipeProgSub');
    if (running) {
      var nm = p.current_name || (PIPE_LABELS[p.current_key] || '');
      var step = p.step || 0;
      var tot = p.total || 15;
      if (stepEl) {
        stepEl.textContent = '第 ' + step + '/' + tot + ' 步' + (nm ? ' · ' + nm : '') +
          ((p.done && p.done > 0) ? '（已完成 ' + p.done + ' 步）' : '');
      }
      if (subEl) {
        var bits = [];
        if (p.elapsed_current != null) bits.push('本阶段已用 ' + fmtDur(p.elapsed_current));
        if (p.eta_current != null && p.eta_current > 1) bits.push('本阶段约剩 ' + fmtDur(p.eta_current));
        if (p.eta_remaining != null && p.eta_remaining > 1) bits.push('整体约剩 ' + fmtDur(p.eta_remaining));
        subEl.textContent = bits.join(' · ');
      }
    } else {
      if (stepEl) {
        stepEl.textContent = d.ok === true ? '上次更新已完成（' + (p.done || 15) + ' 步）'
          : (d.ok === false ? '上次更新中断' : '待运行');
      }
      if (subEl) {
        var t = (d.finished_at || '').replace('T', ' ');
        if (d.stale && d.stale_reason) {
          subEl.textContent = d.stale_reason;
        } else {
          subEl.textContent = t + (d.total_elapsed ? ' · 总耗时 ' + fmtDur(d.total_elapsed) : '');
        }
      }
    }
  }

  // 点击后立即给出反馈，不等第一次轮询
  function showPipeStarting(payload) {
    var wrap = document.getElementById('pipeProg');
    var fill = document.getElementById('pipeProgFill');
    if (!wrap || !fill) return;
    var fast = !!(payload && payload.skip_data && payload.skip_factors);
    wrap.hidden = false;
    wrap.classList.remove('done', 'fail');
    fill.style.width = '2%';
    setText('pipeProgPct', '0.0%');
    setText('pipeProgStep', '正在启动' + (fast ? '（只重算策略）' : '（全量更新）') + '…');
    setText('pipeProgSub', fast
      ? '模型重训 → 回测 → 信号，约 5 分钟'
      : '拉行情 → 重建 800 因子 → 重训模型 → 回测 → 信号，约 20~40 分钟');
    setPipeBusy(true);
  }

  function renderPipeline(d) {
    var box = document.getElementById('pipelineStages');
    var txt = document.getElementById('pipeStatusText');
    var tail = document.getElementById('pipelineTail');
    if (!box) return;
    renderPipeProgress(d);
    setPipeBusy(!!d.running);

    var running = !!d.running;
    var lastEl = document.getElementById('pipeLastRun');
    if (lastEl) {
      if (running) {
        lastEl.textContent = '运行中…';
      } else if (d.finished_at) {
        lastEl.textContent = '上次：' + (d.finished_at || '').replace('T', ' ') +
          (d.total_elapsed ? '（' + Math.round(d.total_elapsed / 60) + ' 分）' : '');
      } else {
        lastEl.textContent = '尚未运行';
      }
    }
    var etaEl = document.getElementById('pipeEta');
    if (etaEl) {
      if (running && d.current) {
        var est = d.current.est_sec || 0;
        etaEl.textContent = est >= 60 ? '预计 ' + Math.round(est / 60) + ' 分钟' : '';
      } else {
        etaEl.textContent = '';
      }
    }
    if (txt) {
      txt.className = 'pl-status' + (running ? ' running'
        : (d.ok === true ? ' ok' : (d.ok === false ? ' bad' : '')));
      if (running) {
        var cur = d.current || {};
        txt.textContent = '运行中 · ' + (cur.name || '');
      } else if (d.never_run) {
        txt.textContent = '未运行';
      } else if (d.ok === true) {
        txt.textContent = '上次成功 · ' + (d.finished_at || '').replace('T', ' ') +
          (d.total_elapsed ? '（' + Math.round(d.total_elapsed) + 's）' : '');
      } else if (d.ok === false) {
        txt.textContent = (d.stale ? '上次中断 · ' : '上次有失败 · ') +
          (d.finished_at || '').replace('T', ' ');
      } else {
        txt.textContent = '未运行';
      }
    }

    var all = d.stages || [];
    var curKey = (d.current || {}).key;
    // 只显示"真正跑过"的阶段 + 当前阶段；跳过的折叠
    var shown = all.filter(function (s) {
      return s.status === 'ok' || s.status === 'failed' || s.key === curKey;
    });
    var nSkip = all.length - shown.length;

    if (!all.length) {
      box.innerHTML = d.never_run
        ? '<div class="muted small">尚未运行。点「开始更新」启动全流程。</div>'
        : '';
    } else {
      var html = '<div class="pl-stages">';
      if (nSkip > 0) {
        html += '<div class="pl-skip-note muted small">已跳过 ' + nSkip +
          ' 个阶段（' + (running ? '本次只跑部分阶段' : '未参与本次运行') + '）</div>';
      }
      html += shown.map(function (s) {
        var cls = 'pl-stage';
        var dot = 'pl-dot';
        if (s.status === 'ok') { cls += ' done'; dot += ' done'; }
        else if (s.status === 'failed') { cls += ' fail'; dot += ' fail'; }
        else if (s.key === curKey) { cls += ' cur'; dot += ' cur'; }
        var icon = s.status === 'ok' ? '✓' : (s.status === 'failed' ? '✗' : '◐');
        var t = s.elapsed_sec != null
          ? (s.elapsed_sec >= 60 ? (s.elapsed_sec / 60).toFixed(1) + ' 分'
            : Math.round(s.elapsed_sec) + ' 秒') : '';
        return '<div class="' + cls + '"><span class="' + dot + '"></span>' +
          '<span class="pl-name">' + icon + ' ' +
          esc(s.name || PIPE_LABELS[s.key] || s.key) + '</span>' +
          '<span class="pl-time">' + t + '</span></div>';
      }).join('') + '</div>';
      box.innerHTML = html;
    }

    // 日志尾部：默认折叠
    if (tail) {
      var tboxes = (all || []).filter(function (s) {
        return s.tail && s.tail.length;
      });
      if (tboxes.length) {
        var last = tboxes[tboxes.length - 1];
        var isBad = (last.status === 'failed');
        tail.hidden = false;
        tail.innerHTML = '<details' + (isBad ? ' open' : '') + '>' +
          '<summary style="cursor:pointer;user-select:none">' +
          (isBad ? '⚠️ 失败日志' : '日志') + '（' + esc(last.name || last.key) + '）' +
          '</summary><pre class="pl-log-pre">' +
          esc((last.tail || []).join('\n')) + '</pre></details>';
      } else {
        tail.hidden = true;
      }
    }
  }

  function loadPipelineStatus() {
    return fetch('/api/pipeline/status').then(function (r) { return r.json(); })
      .then(function (d) {
        state.pipeStatus = d;
        if (d.running) state.pipeLaunching = false;
        renderPipeline(d);
        if (d.running) {
          if (!PIPE_TIMER) {
            PIPE_TIMER = setInterval(loadPipelineStatus, 2500);
          }
        } else if (PIPE_TIMER) {
          clearInterval(PIPE_TIMER);
          PIPE_TIMER = null;
          loadConsistency();   // 管道结束后自动刷新一致性
        }
        return d;
      }).catch(function () { return null; });
  }

  /**
   * 启动失败 / 假启动时的错误面板。
   * 以前这里只弹一个 toast，用户既不知道失败原因，也看不到脚本的报错——现在把
   * 后端返回的日志尾部直接铺在进度条下面。
   */
  function showPipeError(reason, logTail) {
    state.pipeLaunching = false;
    var wrap = document.getElementById('pipeProg');
    var fill = document.getElementById('pipeProgFill');
    if (wrap) {
      wrap.hidden = false;
      wrap.classList.remove('done');
      wrap.classList.add('fail');
    }
    if (fill) fill.style.width = '100%';
    setText('pipeProgPct', '失败');
    setText('pipeProgStep', '未能启动');
    setText('pipeProgSub', reason || '未知原因');
    var box = document.getElementById('pipelineStages');
    if (box && logTail && logTail.length) {
      box.innerHTML = '<details open><summary style="cursor:pointer;user-select:none">' +
        '⚠️ 启动日志（最后 ' + logTail.length + ' 行）</summary>' +
        '<pre class="pl-log-pre">' + esc(logTail.join('\n')) + '</pre></details>';
    }
    var tail = document.getElementById('pipelineTail');
    if (tail) tail.hidden = true;
  }

  function bindPipeline() {
    if (PIPE_BOUND) return;          // 每次进数据页都会调用，避免重复绑定导致重复提交
    PIPE_BOUND = true;
    var bRun = document.getElementById('btnPipeRun');
    var bFast = document.getElementById('btnPipeFast');

    function start(payload) {
      var p = payload || {};
      var beforeStartedAt = (state.pipeStatus || {}).started_at || null;
      showPipeStarting(p);
      fetch('/api/pipeline/run', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(p)
      }).then(function (r) { return r.json(); }).then(function (d) {
        if (!d.started) {
          setPipeBusy(false);
          showPipeError(d.reason || '未知原因', d.log_tail || []);
          toast('未能启动：' + (d.reason || '未知原因'), 'error');
        } else {
          state.pipeLaunching = true;
          toast('已启动更新（pid ' + d.pid + '），进度见上方进度条');
          if (PIPE_TIMER) { clearInterval(PIPE_TIMER); PIPE_TIMER = null; }
          setTimeout(loadPipelineStatus, 600);
          PIPE_TIMER = setInterval(loadPipelineStatus, 2500);
          verifyLaunch(beforeStartedAt);
        }
      }).catch(function (e) {
        setPipeBusy(false);
        showPipeError('请求失败：' + e, []);
        toast('启动失败：' + e, 'error');
      });
    }

    /**
     * 假启动检测：脚本进程活着不代表它真的开始跑了（可能启动后立刻抛异常退出，
     * 只是退出得比 2 秒探活更晚）。6 秒后如果 status.json 的 started_at 没变、
     * running 还是 false，就判定这次启动没落地，把结论明确告诉用户。
     */
    function verifyLaunch(beforeStartedAt) {
      setTimeout(function () {
        fetch('/api/pipeline/status').then(function (r) { return r.json(); })
          .then(function (d) {
            state.pipeStatus = d;
            if (d.running) { state.pipeLaunching = false; return; }   // 正常在跑
            if ((d.started_at || null) !== beforeStartedAt) {          // 已换了一轮运行
              state.pipeLaunching = false;
              renderPipeline(d);
              return;
            }
            state.pipeLaunching = false;
            if (PIPE_TIMER) { clearInterval(PIPE_TIMER); PIPE_TIMER = null; }
            setPipeBusy(false);
            showPipeError(
              '进程已启动但没有写入运行状态，脚本可能启动后立刻退出了。' +
              '请查看 build/pipeline/runner.log，或到「日志」一栏确认报错。',
              []);
          })
          .catch(function () { /* 网络抖动忽略，交给正常轮询 */ });
      }, 6000);
    }

    if (bRun) bRun.addEventListener('click', function () {
      if (!confirm('将拉取最新行情并重建 800 因子面板，约需 15~30 分钟。继续？')) return;
      start({});
    });
    if (bFast) bFast.addEventListener('click', function () {
      start({ skip_data: true, skip_factors: true });
    });
    var bCons = document.getElementById('btnConsCheck');
    var bTk = document.getElementById('btnTrackRefresh');
    if (bTk) bTk.addEventListener('click', function () {
      bTk.disabled = true;
      loadTracking().then(function () { bTk.disabled = false; })
        .catch(function () { bTk.disabled = false; });
    });
    var bRs = document.getElementById('btnResearchRefresh');
    if (bRs) bRs.addEventListener('click', function () {
      bRs.disabled = true;
      loadResearch().then(function () { bRs.disabled = false; })
        .catch(function () { bRs.disabled = false; });
    });
if (bCons) bCons.addEventListener('click', function () {
      bCons.disabled = true;
      loadConsistency().then(function () { bCons.disabled = false; })
        .catch(function () { bCons.disabled = false; });
    });
  }

  // ---------------------------------------------------------- 一致性状态
  function renderConsistency(r) {
    var summ = document.getElementById('consSummary');
    if (summ) {
      summ.textContent = r.ok ? '全部一致' : '存在滞后';
    }
    var box = document.getElementById('consistencyBox');
    if (!box) return;
    var items = r.items || [];
    if (!items.length) {
      box.innerHTML = '<span class="muted small">' +
        (r.error ? '检测失败: ' + esc(r.error) : '尚无检测结果') + '</span>';
      return;
    }
    var allOk = !!r.ok;
    var html = '<div class="cons-head ' + (allOk ? 'ok' : 'bad') + '">' +
      (allOk ? '✅ 全链路一致' : '❌ 存在滞后') + '</div>';
    html += '<div class="cons-list">' + items.map(function (it) {
      return '<div class="cons-item ' + (it.ok ? 'ok' : 'bad') + '">' +
        '<span>' + (it.ok ? '✓' : '✗') + '</span>' +
        '<span class="cons-name">' + esc(it.name) + '</span>' +
        '<span class="cons-detail muted small">' + esc(it.detail || '') + '</span></div>';
    }).join('') + '</div>';
    if (!allOk && (r.fix_hint || []).length) {
      html += '<div class="cons-fix"><b>修复建议：</b><br>' +
        r.fix_hint.map(function (h) { return '　' + esc(h); }).join('<br>') + '</div>';
    }
    html += '<div class="muted small" style="margin-top:6px">数据目录：' +
      esc(r.data_raw || '') + '（' + esc(r.source || '') + '）</div>';
    box.innerHTML = html;
  }

  function loadConsistency() {
    return fetch('/api/consistency').then(function (r) { return r.json(); })
      .then(renderConsistency).catch(function (e) {
        var b = document.getElementById('consistencyBox');
        if (b) b.textContent = '检测失败: ' + e;
      });
  }

  function loadDataInfo() {
    var host = document.getElementById('dataStatus');
    if (!host) return;
    fetch('/api/data/status').then(function (r) { return r.json(); })
      .then(renderDataStatus)
      .catch(function (e) {
        if (host) host.textContent = '加载失败: ' + e;
      });
  }

  // ========================================================= 研究页
  var RS = { data: null };

  function rsPct(v, d) {
    if (v == null || isNaN(v)) return '—';
    return (v * 100).toFixed(d == null ? 2 : d) + '%';
  }
  function rsNum(v, d) {
    if (v == null || isNaN(v)) return '—';
    return Number(v).toFixed(d == null ? 3 : d);
  }

  function rsKpi(label, val, sub, cls) {
    return '<div class="rs-kpi ' + (cls || '') + '">' +
      '<div class="rs-kpi-l">' + esc(label) + '</div>' +
      '<div class="rs-kpi-v">' + val + '</div>' +
      (sub ? '<div class="rs-kpi-s">' + esc(sub) + '</div>' : '') +
      '</div>';
  }

  function renderRsKpis(d) {
    var box = document.getElementById('rsKpis');
    if (!box) return;
    var cost = d.cost || {}, reb = d.rebalance || {}, pool = d.pool || {};
    var sw = cost.sweep || [];
    var base = null, real = null;
    for (var i = 0; i < sw.length; i++) {
      if (sw[i].slip === 15) base = sw[i];
      if (sw[i].slip === 40) real = sw[i];
    }
    var n = (pool.liquidity || {}).bad;
    var q = (reb.conservative || {}).recommend;
    var html = '';
    // 主策略当前实际口径：40 日调仓 + 40bp 滑点
    var cur = null;
    for (var j = 0; j < sw.length; j++) {
      if (sw[j].slip === 40) cur = sw[j];
    }
    html += rsKpi('主策略 CAGR（现行口径）',
      cur ? rsPct(cur.cagr) : '—',
      '40 日调仓 + 40bp 滑点', cur && cur.cagr > 0.05 ? 'good' : 'warn');
    html += rsKpi('年换手率',
      (d.cost && d.cost.turnover) ? rsNum(d.cost.turnover, 2) + 'x' : '—',
      '40 日调仓（10 日时为 4.76x）', 'good');
    html += rsKpi('已采用调仓周期',
      '40 日',
      '成本 ↓86%，收益 ↑0.73pp', 'good');
    html += rsKpi('流动性差的标的',
      n == null ? '—' : n + ' 只',
      '日均成交额 <1000 万', n > 0 ? 'warn' : 'good');
    box.innerHTML = html;
  }

  function renderRsCost(d) {
    var box = document.getElementById('rsCost');
    if (!box) return;
    var c = d.cost || {};
    var sw = c.sweep || [];
    if (!sw.length) {
      box.innerHTML = '<span class="muted small">无成本扫描数据（运行 scripts/cost_sensitivity.py）</span>';
      return;
    }
    var maxC = 0, minC = 1;
    sw.forEach(function (r) {
      maxC = Math.max(maxC, r.cagr);
      minC = Math.min(minC, r.cagr);
    });
    var span = Math.max(maxC - Math.min(0, minC), 0.02);

    var html = '<div class="rs-h5">滑点 → 收益（策略逻辑不变，只改成本）</div>';
    sw.forEach(function (r) {
      var w = Math.max(2, (r.cagr - Math.min(0, minC)) / span * 100);
      var col = r.cagr >= 0.05 ? 'var(--up)'
        : (r.cagr >= 0.03 ? 'var(--warn)' : 'var(--down)');
      var mark = (r.slip === 15 || r.slip === 40) ? ' mark' : '';
      var note = r.slip === 15 ? '旧假设' : (r.slip === 40 ? '真实估算' : '');
      html += '<div class="rs-bar-row' + mark + '">' +
        '<span class="rs-bar-l">' + r.slip + ' bp</span>' +
        '<div class="rs-bar-track"><div class="rs-bar-fill" style="width:' +
        w.toFixed(1) + '%;background:var(--accent)"></div></div>' +
        '<span class="rs-bar-v" style="width:130px;flex:0 0 130px;color:' + col + '">' +
        rsPct(r.cagr, 2) + ' / ' + rsNum(r.sharpe, 2) + '</span>' +
        '<span class="rs-bar-note">' + note + '</span>' +
        '</div>';
    });
    // 标题行
    html = html.replace('</div><div class="rs-bar-row',
      '</div><div class="rs-bar-row');
    html += '<div class="rs-bar-row" style="opacity:.55;font-size:10.5px">' +
      '<span class="rs-bar-l">滑点</span>' +
      '<div class="rs-bar-track" style="background:none"></div>' +
      '<span class="rs-bar-v" style="width:130px;flex:0 0 130px">CAGR / Sharpe</span>' +
      '<span class="rs-bar-note"></span></div>';

    var h = c.holding;
    if (h) {
      html += '<div class="rs-h5" style="margin-top:16px">持仓市值分布（决定滑点高低）</div>';
      var dist = h.distribution || [];
      dist.forEach(function (x) {
        var lbl = (x.hi ? (x.lo + '~' + x.hi) : ('>' + x.lo));
        var w = Math.max(1, x.pct * 100);
        html += '<div class="rs-bar-row">' +
          '<span class="rs-bar-l">' + lbl + '</span>' +
          '<div class="rs-bar-track"><div class="rs-bar-fill" style="width:' +
          w.toFixed(1) + '%;background:var(--text-3)"></div></div>' +
          '<span class="rs-bar-v" style="width:130px;flex:0 0 130px">' +
          (x.pct * 100).toFixed(1) + '%　' + x.n + ' 笔</span>' +
          '<span class="rs-bar-note"></span></div>';
      });
      html += '<div class="rs-note">中位市值 ' + (h.median_mcap_yi || 0).toFixed(0) +
        ' 亿，' + (h.p10 || 0).toFixed(0) + '~' + (h.p90 || 0).toFixed(0) +
        ' 亿为中心区间。' +
        '按市值映射估算单边滑点 <b>' + (h.est_slip_bp_weighted || 0).toFixed(1) +
        ' bp</b>（当前假设 ' + (h.current_assumption_bp || 15).toFixed(0) + ' bp）。</div>';
    }
    box.innerHTML = html;
  }

  function renderRsRebal(d) {
    var box = document.getElementById('rsRebal');
    if (!box) return;
    var r = d.rebalance || {};
    var cv = r.conservative;
    if (!cv || !(cv.rows || []).length) {
      box.innerHTML = '<span class="muted small">无数据（运行 scripts/rebalance_conservative.py）</span>';
      return;
    }
    var rec = cv.recommend;
    var html = '<div class="rs-h5">主策略：不同调仓周期（DEV 2020-21 / OOS 2022-26）</div>';
    html += '<table class="table"><thead><tr>' +
      '<th>周期</th><th>DEV</th><th>OOS</th><th>全期</th>' +
      '<th>换手</th><th>年化成本</th><th></th></tr></thead><tbody>';
    var maxO = 0;
    cv.rows.forEach(function (x) { maxO = Math.max(maxO, Math.abs(x.oos_cagr)); });
    cv.rows.forEach(function (x) {
      var isRec = (x.period === rec);
      var col = x.oos_cagr >= 0.03 ? 'var(--up)'
        : (x.oos_cagr >= 0 ? 'var(--warn)' : 'var(--down)');
      html += '<tr' + (isRec ? ' style="background:var(--accent-soft)"' : '') + '>' +
        '<td><b>' + x.period + ' 日</b>' +
        (isRec ? ' <span class="tag-ok">推荐</span>' : '') + '</td>' +
        '<td>' + rsPct(x.dev_cagr) + '</td>' +
        '<td style="color:' + col + '"><b>' + rsPct(x.oos_cagr) + '</b></td>' +
        '<td>' + rsPct(x.full_cagr) + '</td>' +
        '<td>' + rsNum(x.turnover, 2) + 'x</td>' +
        '<td>' + rsPct(x.ann_cost) + '</td>' +
        '<td><span class="rs-mini" style="display:inline-block;height:8px;border-radius:2px;' +
        'background:' + col + ';width:' +
        Math.max(2, Math.abs(x.oos_cagr) / Math.max(maxO, 0.01) * 48).toFixed(0) +
        'px"></span></td></tr>';
    });
    html += '</tbody></table>';
    html += '<div class="rs-note good">判定标准（预注册，不看结果调整）：' +
      '① OOS CAGR 不比 10 日差 0.3pp 以上 ② OOS Sharpe 不劣化 ' +
      '③ 换手降 ≥30% ④ 全期 CAGR 不劣化。<br>' +
      '通过：' + ((cv.qualified || []).join(', ') || '—') +
      '　→ 推荐 <b>' + (rec || '—') + ' 日</b>。' +
      '</div>';
    box.innerHTML = html;
  }

  function renderRsEtfFlex(d) {
    var box = document.getElementById('rsEtfFlex');
    if (!box) return;
    var rows = (d.rebalance || {}).etf_flex || [];
    if (!rows.length) {
      box.innerHTML = '<span class="muted small">无数据</span>';
      return;
    }
    // 只取 OOS，按 CAGR 排序
    var oos = rows.filter(function (x) { return x.period === 'OOS'; })
      .sort(function (a, b) { return b.cagr - a.cagr; });
    var html = '<div class="rs-h5">OOS（2022-2026）表现</div>';
    html += '<table class="table"><thead><tr>' +
      '<th>配置</th><th>CAGR</th><th>Sharpe</th><th>MaxDD</th>' +
      '<th>交易笔数</th></tr></thead><tbody>';
    oos.forEach(function (x) {
      var col = x.cagr >= 0.05 ? 'var(--up)'
        : (x.cagr >= 0 ? 'var(--warn)' : 'var(--down)');
      html += '<tr><td>' + esc(x.config) + '</td>' +
        '<td style="color:' + col + '"><b>' + rsPct(x.cagr) + '</b></td>' +
        '<td>' + rsNum(x.sharpe, 3) + '</td>' +
        '<td>' + rsPct(x.max_dd) + '</td>' +
        '<td>' + x.trades + '</td></tr>';
    });
    html += '</tbody></table>';
    html += '<div class="rs-note bad">' +
      '<b>重要警告</b>：固定 20 日 top2 的 OOS +13.46% 是<b>参数尖峰</b>，不是高原——' +
      '15日 -0.69%、25日 +6.62%、40日 +10.50%，跨度 14pp。' +
      '按文章标准属于过拟合形态，<b>不建议采用该参数</b>。</div>';
    box.innerHTML = html;
  }

  function renderRsPool(d) {
    var box = document.getElementById('rsPool');
    if (!box) return;
    var p = d.pool || {};
    var lq = p.liquidity, cr = p.corr;
    var html = '';
    if (lq && lq.counts) {
      html += '<div class="rs-split"><div>';
      html += '<div class="rs-h5">流动性（日均成交额）</div>';
      var order = ['🟢 良好', '🟡 一般', '🔴 差'];
      order.forEach(function (k) {
        var v = lq.counts[k];
        if (v == null) return;
        var col = k.indexOf('良好') >= 0 ? 'var(--down)'
          : (k.indexOf('一般') >= 0 ? 'var(--warn)' : 'var(--up)');
        html += '<div class="rs-bar-row">' +
          '<span class="rs-bar-l" style="width:78px;flex:0 0 78px">' +
          esc(k) + '</span>' +
          '<div class="rs-bar-track"><div class="rs-bar-fill" style="width:' +
          Math.min(100, v / 41 * 100).toFixed(0) + '%;background:' + col +
          '"></div></div>' +
          '<span class="rs-bar-v">' + v + ' 只</span></div>';
      });
      var bad = (lq.detail || []).filter(function (x) {
        return x.level && x.level.indexOf('差') >= 0;
      });
      if (bad.length) {
        html += '<div class="rs-note bad">需移出：' + bad.map(function (x) {
          return esc(x.code) + ' ' + esc(x.name);
        }).join('、') + '（日均成交额 256 万，单笔占 5.86%）</div>';
      }
      html += '</div><div>';
    }
    if (cr) {
      html += '<div class="rs-h5">相关性（≥0.85 为高度同向）</div>';
      html += '<div class="rs-bar-row"><span class="rs-bar-l" style="width:96px">' +
        '池内平均</span><div class="rs-bar-track"></div>' +
        '<span class="rs-bar-v">' + rsNum(cr.mean_corr, 3) + '</span></div>';
      html += '<div class="rs-bar-row"><span class="rs-bar-l" style="width:96px">' +
        '高相关组合</span><div class="rs-bar-track"></div>' +
        '<span class="rs-bar-v">' + (cr.n_pairs_high || 0) + ' 组</span></div>';
      html += '<div class="rs-note">' +
        '全池平均相关 ' + rsNum(cr.mean_corr, 3) + '（分散尚可）。' +
        '但<b>子集严重同质</b>：ETF 趋势实际用的三只宽基平均 0.885 ' +
        '（中证500 vs 中证1000 = 0.967）→ 所谓 top2 轮动实际近似持有 1 只。</div>';
      html += '</div></div>';
    }
    box.innerHTML = html || '<span class="muted small">无数据</span>';
  }

  function renderResearch(d) {
    RS.data = d;
    renderRsKpis(d);
    renderRsCost(d);
    renderRsRebal(d);
    renderRsEtfFlex(d);
    renderRsPool(d);
    renderRsLedger(d);
  }

  function loadResearch() {
    var box = document.getElementById('rsKpis');
    if (box) box.innerHTML = '<div class="rs-kpi muted">加载中…</div>';
    return fetch('/api/research/overview')
      .then(function (r) { return r.json(); })
      .then(function (d) {
        renderResearch(d);
        return d;
      })
      .catch(function (e) {
        var b = document.getElementById('rsKpis');
        if (b) b.innerHTML = '<div class="rs-kpi bad">加载失败: ' + esc(String(e)) + '</div>';
      });
  }

  function renderRsLedger(d) {
    var box = document.getElementById('rsLedger');
    if (!box) return;
    var lg = (d.ledger || {});
    var dead = lg.experiments || [];
    var kept = lg.kept || [];
    if (!dead.length && !kept.length) {
      box.innerHTML = '<span class="muted small">无台账数据</span>';
      return;
    }
    var html = '';
    if (dead.length) {
      html += '<div class="rs-h5">❌ 已弃用（' + dead.length + ' 项）</div>';
      dead.forEach(function (x) {
        html += '<div class="rs-dead">' +
          '<div class="rs-dead-h"><b>' + esc(x.name) + '</b>' +
          '<span class="rs-dead-tag">弃用</span>' +
          '<span class="muted small" style="margin-left:auto">' +
          esc(x.date || '') + '</span></div>';
        if (x.oos_cagr != null) {
          html += '<div class="rs-dead-m">OOS: ' +
            rsPct(x.oos_cagr) + ' / Sharpe ' + rsNum(x.oos_sharpe, 2) +
            ' / MaxDD ' + rsPct(x.oos_mdd) + '</div>';
        }
        html += '<ul class="rs-dead-list">';
        (x.reasons || []).forEach(function (r) {
          html += '<li>' + esc(r) + '</li>';
        });
        html += '</ul>';
        if (x.note) {
          html += '<div class="muted small" style="margin-top:4px">' + esc(x.note) + '</div>';
        }
        html += '</div>';
      });
    }
    if (kept.length) {
      html += '<div class="rs-h5" style="margin-top:16px">✅ 保留在用（' +
        kept.length + ' 项）</div>';
      html += '<table class="table"><thead><tr><th>策略</th><th>CAGR</th>' +
        '<th>Sharpe</th><th>MaxDD</th><th>备注</th></tr></thead><tbody>';
      kept.forEach(function (x) {
        var col = x.cagr >= 0.08 ? 'var(--up)'
          : (x.cagr >= 0.05 ? 'var(--warn)' : 'var(--down)');
        html += '<tr><td><b>' + esc(x.name) + '</b></td>' +
          '<td style="color:' + col + '">' + rsPct(x.cagr) + '</td>' +
          '<td>' + rsNum(x.sharpe, 3) + '</td>' +
          '<td>' + rsPct(x.mdd) + '</td>' +
          '<td class="muted small">' + esc(x.note || '') + '</td></tr>';
      });
      html += '</tbody></table>';
    }
    box.innerHTML = html;
  }

  function pct_text(v) { return rsPct(v); }

  // ========================================================= 实盘跟踪
  function tkKpi(label, val, sub, cls) {
    return '<div class="rs-kpi ' + (cls || '') + '">' +
      '<div class="rs-kpi-l">' + esc(label) + '</div>' +
      '<div class="rs-kpi-v">' + val + '</div>' +
      (sub ? '<div class="rs-kpi-s">' + esc(sub) + '</div>' : '') + '</div>';
  }

  function renderTracking(d) {
    var cur = d.current, st = d.state || {}, ref = d.reference;

    // ---- KPI
    var kb = document.getElementById('tkKpis');
    if (kb) {
      if (!cur) {
        kb.innerHTML = tkKpi('尚未开始跟踪', '—',
          '运行 track_record.py signal 开始', 'warn');
      } else {
        var cls = cur.pnl_pct > 0 ? 'good' : (cur.pnl_pct < -0.02 ? 'bad' : '');
        kb.innerHTML =
          tkKpi('当前净值', (cur.nav || 0).toLocaleString('zh-CN', {maximumFractionDigits: 0}),
            '起始 ' + (st.capital || 0).toLocaleString('zh-CN'), cls) +
          tkKpi('累计收益', rsPct(cur.pnl_pct), '截至 ' + esc(cur.date || ''), cls) +
          tkKpi('现金 / 持仓', Math.round((cur.cash || 0) / 1000) + 'k / ' +
            Math.round((cur.market_value || 0) / 1000) + 'k',
            (cur.n_holdings || 0) + ' 只持仓', '') +
          tkKpi('跟踪天数', String((d.navs || []).length) + ' 天',
            (d.n_trades || 0) + ' 笔成交', '');
      }
    }

    // ---- 净值曲线
    var nb = document.getElementById('tkNav');
    var nm = document.getElementById('tkNavMeta');
    if (nb) {
      var nv = d.navs || [];
      if (nv.length < 2) {
        nb.innerHTML = '<span class="muted small">净值记录不足（当前 ' +
          nv.length + ' 条），至少 2 条才能画曲线</span>';
        if (nm) nm.textContent = '';
      } else {
        var cap = st.capital || 1;
        var vals = nv.map(function (x) { return x.nav / cap - 1; });
        var maxV = Math.max.apply(null, vals.concat([0.005]));
        var minV = Math.min.apply(null, vals.concat([-0.005]));
        var sp = Math.max(maxV - minV, 0.02);
        // 零轴对齐：正值从零轴向上、负值向下（原来全部向上生长，−1% 也像正收益）
        var zeroPct = (maxV - 0) / sp * 100;
        var upSpan = Math.max(maxV, 1e-9);
        var dnSpan = Math.max(-minV, 1e-9);
        var h = '';
        h += '<div class="tk-chart">' +
          '<div class="tk-zero" style="bottom:' + zeroPct.toFixed(1) + '%"></div>';
        nv.forEach(function (x, i) {
          var v = vals[i];
          var isUp = v >= 0;
          var hPct = isUp
            ? Math.max(0.8, v / upSpan * (100 - zeroPct))
            : Math.max(0.8, (-v) / dnSpan * zeroPct);
          var col = isUp ? 'var(--up)' : 'var(--down)';
          h += '<div class="tk-bar" title="' + esc(x.date) + ' ' + rsPct(v) + '">' +
            '<span style="' + (isUp ? 'bottom:' : 'top:') + zeroPct.toFixed(1) +
            '%;height:' + hPct.toFixed(1) + '%;background:' + col + '"></span></div>';
        });
        h += '</div>';
        h += '<div class="tk-axis"><span>' + esc(nv[0].date) + '</span>' +
          '<span>' + esc(nv[nv.length - 1].date) + '</span></div>';
        // 统计
        var last = vals[vals.length - 1];
        h += '<div class="rs-note ' + (last >= 0 ? 'good' : 'bad') + '">' +
          '累计 ' + rsPct(last) + '，最高 ' + rsPct(maxV) +
          '，最低 ' + rsPct(minV) + '</div>';
        if (ref) {
          h += '<div class="rs-note">回测参考（同期同口径）：CAGR ' +
            rsPct(ref.cagr) + ' / Sharpe ' + rsNum(ref.sharpe, 3) +
            ' / MaxDD ' + rsPct(ref.max_dd) + '</div>';
        }
        nb.innerHTML = h;
        if (nm) nm.textContent = nv.length + ' 个交易日';
      }
    }

    // ---- 持仓
    var hb = document.getElementById('tkHold');
    var hm = document.getElementById('tkHoldMeta');
    if (hb) {
      var holds = st.holdings || {};
      var keys = Object.keys(holds);
      if (!keys.length) {
        hb.innerHTML = '<span class="muted small">当前空仓（现金 ' +
          Math.round(st.cash || 0).toLocaleString('zh-CN') + '）</span>';
      } else {
        var lastNav = (d.navs || [])[((d.navs || []).length - 1)] || {};
        var det = lastNav.detail || [];
        var h = '<table class="table"><thead><tr>' +
          '<th>代码</th><th>股数</th><th>成本</th><th>现价</th>' +
          '<th>市值</th><th>盈亏</th></tr></thead><tbody>';
        keys.forEach(function (c) {
          var p = holds[c];
          var dt = null;
          for (var i = 0; i < det.length; i++) {
            if (det[i].code === c) { dt = det[i]; break; }
          }
          var pnl = dt ? dt.pnl_pct : null;
          var col = pnl == null ? '' : (pnl >= 0 ? 'var(--up)' : 'var(--down)');
          h += '<tr><td><b>' + esc(c) + '</b></td>' +
            '<td>' + p.shares + '</td>' +
            '<td>' + rsNum(p.cost, 3) + '</td>' +
            '<td>' + (dt ? rsNum(dt.price, 3) : '—') + '</td>' +
            '<td>' + (dt ? Math.round(dt.value).toLocaleString('zh-CN') : '—') + '</td>' +
            '<td style="color:' + col + '">' +
            (pnl == null ? '—' : rsPct(pnl)) + '</td></tr>';
        });
        h += '</tbody></table>';
        hb.innerHTML = h;
      }
      if (hm) hm.textContent = keys.length + ' 只';
    }

    // ---- 成交
    var tb = document.getElementById('tkTrades');
    var tm = document.getElementById('tkTradeMeta');
    if (tb) {
      var tr = (d.trades || []).slice(-30).reverse();
      if (!tr.length) {
        tb.innerHTML = '<span class="muted small">暂无成交记录</span>';
      } else {
        var h = '<table class="table"><thead><tr>' +
          '<th>日期</th><th>方向</th><th>代码</th><th>价格</th>' +
          '<th>股数</th><th>金额</th><th>费用</th><th>备注</th><th></th>' +
          '</tr></thead><tbody>';
        tr.forEach(function (x) {
          var isBuy = x.side === 'buy';
          h += '<tr><td>' + esc(x.date) + '</td>' +
            '<td style="color:' + (isBuy ? 'var(--accent)' : 'var(--warn)') + '">' +
            (isBuy ? '买入' : '卖出') + '</td>' +
            '<td>' + esc(x.code) + '</td>' +
            '<td>' + rsNum(x.price, 3) + '</td>' +
            '<td>' + x.shares + '</td>' +
            '<td>' + Math.round(x.amount).toLocaleString('zh-CN') + '</td>' +
            '<td>' + rsNum(x.fee, 0) + '</td>' +
            '<td class="muted small">' + esc(x.reason || '') + '</td>' +
            '<td><button class="btn btn-xs tk-del" data-date="' +
            esc(x.date) + '" data-code="' + esc(x.code) +
            '" data-side="' + esc(x.side) + '">删</button></td></tr>';
        });
        h += '</tbody></table>';
        tb.innerHTML = h;
      }
      if (tm) tm.textContent = (d.n_trades || 0) + ' 笔';
    }

    // ---- 一致性
    var cb = document.getElementById('tkConsistency');
    if (cb) {
      var cs = d.consistency;
      if (!cs) {
        cb.innerHTML = '<span class="muted small">' +
          '需要「信号记录」+「当前持仓」才能对比。' +
          '先运行 track_record.py signal，再补录成交。</span>';
      } else {
        var rate = cs.n_overlap / Math.max(cs.n_target, 1);
        var cls = rate >= 0.8 ? 'good' : (rate >= 0.5 ? '' : 'bad');
        var h = '<div class="rs-h5">信号日 ' + esc(cs.signal_date) + '</div>';
        h += '<div class="rs-bar-row"><span class="rs-bar-l" ' +
          'style="width:110px;flex:0 0 110px">信号目标</span>' +
          '<span class="rs-bar-v">' + cs.n_target + ' 只</span></div>';
        h += '<div class="rs-bar-row"><span class="rs-bar-l" ' +
          'style="width:110px;flex:0 0 110px">实际持仓</span>' +
          '<span class="rs-bar-v">' + cs.n_held + ' 只</span></div>';
        h += '<div class="rs-bar-row"><span class="rs-bar-l" ' +
          'style="width:110px;flex:0 0 110px">重合</span>' +
          '<span class="rs-bar-v">' + cs.n_overlap + ' 只（' +
          (rate * 100).toFixed(0) + '%）</span></div>';
        if (cs.missing && cs.missing.length) {
          h += '<div class="rs-note"><b>未买入（信号有、持仓无）：</b>' +
            cs.missing.join(', ') + '</div>';
        }
        if (cs.extra && cs.extra.length) {
          h += '<div class="rs-note"><b>超出信号（持仓有、信号无）：</b>' +
            cs.extra.join(', ') + '</div>';
        }
        if (rate >= 0.8) {
          h += '<div class="rs-note good">执行一致性良好（≥80%）</div>';
        }
        cb.innerHTML = h;
      }
    }
  }

  var TK_BOUND = false;
  function loadTracking() {
    if (!TK_BOUND) { TK_BOUND = true; bindTrackingOps(); }
    return fetch('/api/tracking').then(function (r) { return r.json(); })
      .then(renderTracking)
      .catch(function (e) {
        var b = document.getElementById('tkKpis');
        if (b) b.innerHTML = '<div class="rs-kpi bad">加载失败: ' +
          esc(String(e)) + '</div>';
      });
  }

  // ------------------------------------------------ 跟踪页操作
  function tkPost(path, payload) {
    return fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload || {})
    }).then(function (r) {
      return r.json().then(function (d) {
        if (!r.ok) throw new Error(d.error || d.msg || ('HTTP ' + r.status));
        return d;
      });
    });
  }

  function tkSetHint(msg, ok) {
    var h = document.getElementById('tkOpHint');
    if (h) {
      h.textContent = msg;
      h.style.color = ok === false ? 'var(--up)'
        : (ok === true ? 'var(--down)' : '');
    }
  }

  function bindTrackingOps() {
    var bSig = document.getElementById('tkBtnSignal');
    var bNav = document.getElementById('tkBtnNav');
    var bFill = document.getElementById('tkBtnFill');
    var bReset = document.getElementById('tkBtnReset');

    var di = document.getElementById('tkDate');
    if (di && !di.value) {
      var t = new Date();
      di.value = t.getFullYear() + '-' +
        String(t.getMonth() + 1).padStart(2, '0') + '-' +
        String(t.getDate()).padStart(2, '0');
    }

    if (bSig) bSig.addEventListener('click', function () {
      bSig.disabled = true;
      tkPost('/api/tracking/signal').then(function (d) {
        bSig.disabled = false;
        tkSetHint(d.skipped ? (d.msg || '信号已存在')
          : ('已记录信号 ' + d.signal_date + '（' + (d.n_targets || 0) + ' 只）'),
          true);
        loadTracking();
      }).catch(function (e) {
        bSig.disabled = false;
        tkSetHint('记录失败: ' + e.message, false);
      });
    });

    if (bNav) bNav.addEventListener('click', function () {
      bNav.disabled = true;
      tkPost('/api/tracking/nav', {}).then(function (d) {
        bNav.disabled = false;
        var p = d.pnl_pct == null ? '' :
          '（' + (d.pnl_pct >= 0 ? '+' : '') +
          (d.pnl_pct * 100).toFixed(2) + '%）';
        tkSetHint('已更新 ' + d.date + ' 净值 ' +
          Number(d.nav).toLocaleString('zh-CN', {maximumFractionDigits: 0}) + p,
          true);
        loadTracking();
      }).catch(function (e) {
        bNav.disabled = false;
        tkSetHint('更新失败: ' + e.message, false);
      });
    });

    if (bFill) bFill.addEventListener('click', function () {
      var code = (document.getElementById('tkCode') || {}).value || '';
      var side = (document.getElementById('tkSide') || {}).value || 'buy';
      var price = parseFloat((document.getElementById('tkPrice') || {}).value);
      var shares = parseInt((document.getElementById('tkShares') || {}).value);
      var fee = parseFloat((document.getElementById('tkFee') || {}).value) || 0;
      var date = (document.getElementById('tkDate') || {}).value || '';
      var reason = (document.getElementById('tkReason') || {}).value || '';

      if (!/^\d{6}$/.test(code)) { tkSetHint('代码必须是 6 位数字', false); return; }
      if (!(price > 0)) { tkSetHint('价格必须大于 0', false); return; }
      if (!(shares > 0)) { tkSetHint('股数必须大于 0', false); return; }

      bFill.disabled = true;
      tkPost('/api/tracking/fill', {
        code: code, side: side, price: price, shares: shares,
        fee: fee, date: date, reason: reason
      }).then(function (d) {
        bFill.disabled = false;
        tkSetHint('已记录 ' + (side === 'buy' ? '买入' : '卖出') + ' ' +
          code + ' ' + shares + ' 股，现金 ' +
          Number(d.cash).toLocaleString('zh-CN', {maximumFractionDigits: 0}) +
          '，持仓 ' + d.n_holdings + ' 只', true);
        ['tkCode', 'tkPrice', 'tkShares', 'tkReason'].forEach(function (id) {
          var el = document.getElementById(id);
          if (el) el.value = '';
        });
        loadTracking();
      }).catch(function (e) {
        bFill.disabled = false;
        tkSetHint('记录失败: ' + e.message, false);
      });
    });

    if (bReset) bReset.addEventListener('click', function () {
      if (!confirm('确定清空台账？（成交和净值记录都会删除，不可恢复）')) return;
      if (!confirm('再次确认：真的清空？')) return;
      bReset.disabled = true;
      tkPost('/api/tracking/reset', { confirm: 'RESET', capital: 300000 })
        .then(function () {
          bReset.disabled = false;
          tkSetHint('台账已清空', true);
          loadTracking();
        }).catch(function (e) {
          bReset.disabled = false;
          tkSetHint('清空失败: ' + e.message, false);
        });
    });

    // 删除成交（事件委托）
    document.addEventListener('click', function (ev) {
      var b = ev.target;
      if (!b || !b.classList || !b.classList.contains('tk-del')) return;
      var dt = b.getAttribute('data-date');
      var cd = b.getAttribute('data-code');
      var sd = b.getAttribute('data-side');
      if (!confirm('删除 ' + dt + ' ' + sd + ' ' + cd + ' 这笔成交？')) return;
      b.disabled = true;
      tkPost('/api/tracking/delete',
        { date: dt, code: cd, side: sd }).then(function () {
          tkSetHint('已删除该笔成交', true);
          loadTracking();
        }).catch(function (e) {
          b.disabled = false;
          tkSetHint('删除失败: ' + e.message, false);
        });
    });
  }
})();

  