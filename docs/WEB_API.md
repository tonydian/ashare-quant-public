# Web UI 接口契约（v1）

> 后端：`ashare_quant/web/server.py`（Python 标准库 `ThreadingHTTPServer`，无第三方 web 依赖），
> 默认 `http://127.0.0.1:8765`，只绑定本机。
> 前端：`ashare_quant/web/static/{index.html,app.js,style.css}`，纯原生 JS + CSS，**不依赖任何 CDN/外网**。
> 图表由后端 matplotlib 生成 PNG（`/api/chart/*`），前端用 `<img src>` 展示即可（带时间戳参数防缓存）。

## 通用约定

- 所有接口返回 JSON（`Content-Type: application/json; charset=utf-8`），失败返回 `{"error": "..."} ` + HTTP 4xx/5xx。
- 时间/日期均为字符串：`"2026-10-20"`。
- 百分比字段：小数形式（0.0936 = 9.36%），前端自行 ×100。
- 金额单位：元。

### 503 的两种含义（2026-11-07 明确）

接口不会让请求线程长时间挂住。拿不到数据就立刻回 503，前端按 3 秒重试：

| `error` 文案 | 含义 | 典型时长 |
|---|---|---|
| `数据加载中，请稍后` | 行情/模型还没读进内存 | 启动后约 15–20 秒 |
| `打分数据加载中，请稍候重试…` | 当日全量特征 + 打分还在算（首次） | 首次约 70 秒；之后命中 `build/cache/live/` 落盘缓存，约 0.01 秒 |

**不要**把 503 当失败。前端 `isServiceLoading()` 认 503 并自动重试，各页面都会显示"服务加载中"而不是报错。
`/api/meta`、`/api/stock/search` 不依赖上述缓存，加载期间也照常返回。

## GET /api/health
```json
{"status":"ok","ts":"2026-10-21T23:05:00","data_last_date":"2026-10-20"}
```

## GET /api/meta
```json
{
  "app":"A股量化系统",
  "data":{"last_date":"2026-10-20","orig_last_date":"2026-10-14","n_dates":2432,
          "n_codes":5281,"n_universe":4370,"source":"ext","update_running":false},
  "model":{"name":"wf_h10","refit":"2026-10-13","n_pool":34,"horizon":10,
           "method":"blend","capital":300000},
  "server":{"loaded":true,"loaded_seconds":38.2}
}
```

## GET /api/market/overview
```json
{
 "asof":"2026-10-20",
 "regime":{"coeff":0.45,"csi_all_above_MA120":false,"hs300_above_MA60":false},
 "indices":[
   {"code":"csi_all","name":"中证全指","close":5306.57,"chg_1d":0.0027,"chg_20d":-0.018,
    "chg_120d":-0.05,"ma120":5400.0,"above_ma120":false,"above_ma60":false},
   {"code":"hs300",  "...": "同上"},
   {"code":"csi500", "...": "同上"},
   {"code":"csi1000","...": "同上"}
 ],
 "breadth":{
   "n_universe":4370,"up":2100,"down":2000,"flat":270,
   "limit_up":60,"limit_down":10,"median_ret":0.001,
   "total_amount":1.2e12,"amount_20d_avg":1.1e12,"amount_ratio":1.05,
   "n_above_ma20":2600,"pct_above_ma20":0.60
 },
 "industries":[{"industry":"电子","n":300,"ret_1d":0.01,"ret_20d":0.05,"ret_60d":0.1,"amount":1.0e11}],
 "top_gainers":[{"code":"300123","name":"xx","industry":"电子","chg_1d":0.099,"price":12.3,"amount":5e8}],
 "top_losers":  [{"...": "同上"}],
 "top_amount":[{"...": "同上"}],
 "chart_urls":{"index_line":"/api/chart/index_line?code=csi_all&days=250",
               "industry_bar":"/api/chart/industry_bar?days=20"}
}
```

## GET /api/recommend?top=60（top=候选榜返回条数）
```json
{
 "asof":"2026-10-20","refit":"2026-10-13",
 "regime":{"coeff":0.45,"csi_all_above_MA120":false,"hs300_above_MA60":false},
 "capital":300000,"n_holdings":36,"invested_value":135012,"invested_pct":0.45,"cash_pct":0.55,
 "cash_sleeve":{"bond_weight":0.8,"gold_weight":0.2,"cash_value":164987.8,
                "bond_value":131990.2,"gold_value":32997.6,
                "instruments":["国债ETF(如 511010/511260)","黄金ETF(518880)"]},
 "holdings":[{"code":"002191","name":"劲嘉股份","industry":"轻工制造","board":"MAIN",
              "weight":0.0109,"price":3.284,"shares":1000,"lots":10,"amount":3283.97,
              "rank":3,"score":1.23}],
 "orders":[{"code":"002191","name":"劲嘉股份","side":"buy","shares":1000,"lots":10,
            "price_ref":3.284,"amount":3283.97,"reason":"initial_build"}],
 "candidates":[{"rank":1,"code":"002602","name":"世纪华通","industry":"传媒","board":"MAIN",
                "score":2.31,"price":12.19,"in_target":true,"excluded_reason":null}],
 "notes":["信号基于数据最后交易日收盘计算，建议下一交易日开盘执行；","..."],
 "chart_url":"/api/chart/index_line?code=csi_all&days=250"
}
```
- `holdings` 按 rank 排序；`orders` 为"从零建仓"的买入清单（若用户上传当前持仓可后续扩展）。
- `candidates` 为最新复合分全市场排序前 top 名（含目标内/目标外），用于"个股推荐"页展示。

## GET /api/recommend/export?type=target|orders[&model=wf_rs10]  → text/csv（下载目标/订单）
- 默认 model=当前默认模型（wf_h10）；指定 model 时按该模型实时生成（与页面一致）；
- 文件名：`latest_target_<model>.csv` / `latest_orders_<model>.csv`。

## GET /api/stock/{code}
```json
{
 "code":"000001","name":"平安银行","industry":"银行","board":"MAIN","asof":"2026-10-20",
 "price":11.44,"chg_1d":-0.004,"chg_5d":0.01,"chg_20d":0.03,"chg_60d":0.08,"chg_120d":0.12,
 "amount_20d_avg":2.5e9,"mcap":2.2e11,"list_days":5000,"in_universe":true,
 "in_target":false,"target_weight":null,
 "score":0.87,"score_rank":120,"score_pct":0.96,
 "factors":[{"key":"volume-stat:R716","name_cn":"10日稳定度缩放量价相关","feature":0.8,
             "weight":0.06,"contribution":0.048}],
 "fundamentals":{"ep_ttm":0.06,"bp":0.9,"roe_ann":11.2,"profit_growth":3.3,"rev_growth":1.8},
 "chart_url":"/api/chart/stock_kline?code=000001&days=250"
}
```

## GET /api/stock/search?q=平安&limit=20
```json
[{"code":"000001","name":"平安银行","industry":"银行"},{"...":"..."}]
```

## POST /api/backtest
请求体（均可缺省）：
```json
{"model":"wf_h10","n_holdings":40,"rebalance_days":10,
 "cash_bond":0.8,"cash_gold":0.2,"overlay":true,"stop_loss":true,
 "start":"2020-01-01","end":"2026-10-20","max_new":6,
 "band_rel":0.30,"band_abs":6000}
```
返回 `{"job_id":"bt_xxx"}`；用 `GET /api/jobs/{job_id}` 轮询（约 2~8 秒）。

## GET /api/jobs/{job_id}
```json
{"id":"bt_xxx","kind":"backtest","status":"running|done|error","progress":0.5,
 "message":"正在回测 2020-01-08 ~ 2026-10-20","error":null,"elapsed":1.2,
 "result":{
   "params":{ "...请求参数..." },
   "metrics":{"cagr":0.0936,"ann_vol":0.1003,"sharpe":0.7928,"sortino":1.08,
              "max_dd":-0.1,"calmar":0.9354,"win_rate_monthly":0.634,
              "turnover_ann":2.684,"cost_ratio":0.0138,"n_trades":2061,
              "avg_holdings":36.8,"years":6.49,"start":"2020-01-08","end":"2026-10-20"},
   "periods":{"full":{...同上...},"val":{...},"oos":{...}},
   "yearly":[{"year":2020,"ret":0.0873}],
   "benchmarks":{"hs300":{"cagr":0.004,"max_dd":-0.456,"sharpe":0.036},
                 "csi1000":{"..."},"all_eq":{"..."}},
   "nav":{"dates":["2020-01-08",...],"nav":[300000.0,...],"dd":[0.0,...]},
   "chart_url":"/api/chart/backtest_nav/bt_xxx",
   "trades_summary":{"n_buy":1046,"n_sell":1015,"cost_total":37412.0,
                     "by_year":[{"year":2020,"n":123,"cost":1000.0}]}
 }
}
```
- `nav` 可能下采样到 ≤800 点（前端画折线足够）。
- update 类 job（数据更新）额外返回 `"log_tail":["...","..."]`（最后 ~30 行）。

## POST /api/data/update
请求体：`{"source":"tencent"}` → `{"job_id":"upd_xxx"}`；进程内跑 `scripts/update_data.py`（约 10~15 分钟）。
完成后服务自动重新加载数据（`/api/meta` 的 `data.last_date` 会变）。

> 输出目录语义（2026-11-02 修复）：`C.DATA_RAW` 会自动选行数更多的目录，ext 建好后它
> 就是指 ext 本身。脚本据此区分两种模式——源是原始包 `was_stock_data` → 派生
> `was_stock_data_ext`；源已是 ext → **原地更新**（跳过 `shutil.copy2`/`copytree`，
> 否则源=目标会抛 `PermissionError[WinError 32]`）。原地模式下若面板领先于
> `indices.csv`，脚本会单独补齐指数缺口后退出（exit 0），不再整段重抓个股。
>
> 第 6 步（`[6/6]`）同步三条**独立数据源**，与个股面板同尾：
> `bond_000012.parquet`（上证国债指数，volume 按重叠日实测倍数还原 ×100）、
> `gold_518880.parquet`（518880 **不复权**收盘）、`build/etf/etf_panel.npz`
> （3 只宽基，走 `build_etf_data.py --update`，按重叠日复权系数接续历史）。
> 三者都做重叠日校验（bond ≤5e-5、gold/ETF ≤1e-6 相对误差），失败只报 `[失败]` 不中断更新。

## GET /api/data/status
```json
{"source":"ext","last_date":"2026-10-20","orig_last_date":"2026-10-14",
 "ext_dir_exists":true,"update_running":false,"last_update":"2026-10-21T03:00:00",
 "update_job_id":null}
```

## GET /api/factors/summary
```json
{"counts":{"total":823,"quant":800,"momentum":8,"fundamental":15},
 "stage2":200,"pool":34,
 "families":[{"family":"Trend","n":148,"dev_abs_icir":0.42,"coverage":0.99}],
 "pool_detail":[{"factor_key":"volume-stat:R716","name_cn":"...","category_en":"Price Volume",
                 "weight":0.06,"last_weight":0.06}],
 "fundamental_stats":[{"factor_key":"fundamental-extra:F004","name_cn":"年化ROE",
                       "dev_ic":0.045,"dev_icir":0.30,"coverage":1.0}],
 "composite_ic":{"mean":0.0823,"icir":0.79,"n_weeks":326,
                 "by_year":[{"year":2020,"ic":0.09,"n":48}]},
 "statistics":{"stage2_ic_median":0.02,"stage2_icir_median":0.13}}
```

## POST /api/backtest_strategy（全策略回测）
```json
{"kind":"stock|etf|sector|cb|lowturn|combo","start_date":"2020-01-02","end_date":""}
```
按 `kind` 分派的参数：

| kind | 参数 |
|---|---|
| `stock` | `model`、`n_holdings`、`rebalance_days`（**默认 40，生产口径**）、`max_new`、`cash_bond`、`cash_gold`、`overlay`、`stop_loss` |
| `etf` | `entry_n`、`exit_n`、`min_amount`、`max_pos`、`pos_style` |
| `sector` | `entry_n`、`mom_n`、`top_n`、`rebalance_days`、`min_amount` |
| `cb` | `n_holdings`(20)、`rebalance_days`(10)、`pool_mult`(3)、`min_price`(90)、`rank`(`ytm_duration`\|`ytm`) |
| `lowturn` | `n_holdings`(10)、`lookback`(20)、`min_list_days`(250)、`min_adv20`(亿，0=不限)、`rebalance_weight`(true) |
| `combo` | `method`(`fixed`\|`eq`\|`ivp`\|`erc`\|`mdp`)、`weights{main,etf,sector,cb}`、`cov_window`、`rebal`、`include_cb` |

- `cb` 走 `qf.cb.backtest()`，与每日信号 `scripts/cb_signal.py` 共用 `_elig()/_pick()` 口径；YTM 矩阵复用 `build/cb/ytm_matrix_*.npy` 缓存，首次约 1.9s、命中缓存 0.1s。成本按所选区间**先截取再补扣**（否则 factor 从 2019 年累积，区间内 drag 偏小）。
- `lowturn` 走 `qf.lowturn.backtest()`，与实盘信号 `scripts/lowturn_signal.py` 共用 `turnover_panel/select` 口径。策略逻辑：上月末按 20 日均换手率（成交额/总市值）升序取最低 10 只，剔除 ST/退市/停牌/上市不满一年，下月首个交易日开盘等权买入，**只换变动**（退出全卖、新进补足、保留的仅校准漂移，阈值 2%）。返回额外带 `n_months / keep_ratio / turnover_ann / cagr_strict / last_picks`。
- `stock` 的起点会被推到**打分相位日**（`start_row % 5 == phase`，与 `run_backtest.py` 同一步骤），所以 `start_date` 传 2020-01-02 实际从 2020-01-08 建仓，响应的 `start` 字段回显真实起点。不这样做的话初始建仓日跟着用户选的日期走，40 日网格上的持仓路径会与生产净值分叉。
- `combo` 资产池为 main/etf/sector/cb 四类。**向后兼容**：固定权重不填 `cb` 时结果与三资产一致；动态方法需显式 `include_cb:true` 才纳入第四类。
- 返回统一结构：`kind / kind_cn / start / end / metrics / excess / periods{full,val,oos} / yearly / nav / params / exposure / cost`。

### 成本口径 `cost_model`
`"gross"`（默认，界面现行口径）｜ `"net"`（统一成本口径，补扣费用）。四个策略原本内嵌成本不一致——个股回测已按 config 扣满（40bp 滑点 + 佣金 + 印花税），ETF/板块/转债只扣了 5bp 滑点，因此毛收益不可横向比较。

净额模式只补「目标成本 − 已内嵌成本」的差额，避免重复扣费：

| kind | 单边目标 | 已内嵌 | 补扣 | 实测年换手 | 年化成本 |
|---|---|---|---|---|---|
| stock | 49.3bp | 实测 `cost_bp_per_trade`≈50.7bp | 0 | 0.7x | 0.00% |
| etf | 6.0bp | 5.0bp | 1.0bp | 22.4x | 0.45% |
| sector | 11.0bp | 5.0bp | 6.0bp | 7.1x | 0.85% |
| cb | 20.0bp | 5.0bp | 15.0bp | 7.2x | 2.15% |
| lowturn | 49.3bp | 49.3bp（双账本） | — | 7.1x | 3.51% |

`lowturn` 是唯一用**双账本**的策略：毛/净是两套独立的份额账本（各自逐笔扣 49.3bp），不做补扣近似，
所以 `delta_bp` 为 `null`、`basis` 为 `"double_book"`、`embedded_bp` 为 `null`，前端据此换一套文案。
它的 `turnover_ann` 是**双边成交额口径**（每期买入额+卖出额 / 组合 × 12），买卖两边已计入，
因此 `ann_cost = 换手 × 单边费率`，**不再 ×2**（曾写成 ×2，显示 7.02% 而实际拖累 4.01pp）。
`cagr_gross / cagr_net / drag_pp` 会跟随所选 `start_date` 重算。

扣费方式：`年成本 = 换手 × 2 × 补扣bp`（`lowturn` 除外，见上），按 252 日摊成复利日因子
（不改变收益的时间结构）。响应中 `cost` 字段含 `model/target_bp/embedded_bp/delta_bp/turnover_ann/ann_cost/applied`，
净额模式下还会带 `cagr_gross/cagr_net/sharpe_gross/sharpe_net/drag_pp`；`combo` 额外带 `by_asset` 分资产明细。

## GET /api/reports
```json
{"files":[{"name":"回测报告.md","path":"/reports/回测报告.md","size":12285,
           "mtime":"2026-10-21T22:06:00","kind":"report"}]}
```

## 图表接口（PNG）
- `GET /api/chart/index_line?code=csi_all&days=250` — 指数收盘 + MA20/60/120 折线
- `GET /api/chart/stock_kline?code=000001&days=250` — 个股 K 线（红涨绿跌）+ 成交量
- `GET /api/chart/industry_bar?days=20` — 行业区间收益横向柱状图
- `GET /api/chart/backtest_nav/{job_id}` — 回测净值 + 回撤（两联图）
- 中文用 Microsoft YaHei 渲染；请求带 `&_=timestamp` 防缓存。

## 静态资源
- `/` → `web/static/index.html`
- `/static/*` → `web/static/*`
- `/reports/*`、`/signals/*` → 仓库对应目录（供报告/CSV 下载与查看）
- `/api` 前缀之外未命中 → 404 JSON。

## 前端页面要求（单页、原生 JS）

左侧导航 **13 个页签，按工作流分 5 组**（界面设计方案 v2，见 `reports/界面设计方案_2026-10-29.md`）：

| 组 | 页签 |
|---|---|
| 决策（每天必看） | 今日推荐、看盘 |
| 执行（下单与核对） | 实盘跟踪、配置对比 |
| 分析（为什么这样选） | 回测、个股、因子、研究 |
| 策略（各条腿表现） | 可转债、ETF趋势、板块轮动 |
| 参考（偶尔查阅） | 数据、报告 |

- **设计 token 集中在 `style.css` 的 `:root`**：`--bg / --surface / --surface-2 / --surface-3 / --line / --text / --text-2 / --text-3 / --accent / --up / --down / --warn`。
  旧变量名（`--panel*`、`--border*`、`--muted*`、`--accent-dim` 等）保留为别名，新增样式一律用新名。
- **一个强调色**（`--accent` 蓝）+ 一组涨跌色（`--up` 红 / `--down` 绿，A 股惯例）+ `--warn` 琥珀。
  动作标签（买入 / 卖出）刻意不用涨跌色——买入用 accent、卖出用 warn，避免「买入=红=涨」的语义混淆。
  组件里**不许写死颜色**，服务端 `web/charts.py` 的 matplotlib 配色同步用同一组色值。
- 字号：正文 13.5px、表格 12.5px、辅助 11.5px，**最小不低于 11px**；数字一律 `tabular-nums` 等宽对齐。
- 图标：全部 inline SVG（16×16、stroke 1.4），不使用字符图标（跨字体字重不一致）。
- 长文说明（免责、方法、局限）一律进 `<details class="fold card">`，默认收起，不占首屏。
- 顶栏：标题、当前页、数据日期、模型 refit、服务器加载进度、刷新按钮、加载状态。
- 今日推荐：KPI → **按策略看推荐**（四条腿卡片，点选筛选清单）→ 主栅格（左「今日操作清单」含筛选标签 / 右 316px「预算与剩余 + 空仓与现金」）→ 候选榜 → 信号说明 → 折叠的重要提示；
- 本月推荐：KPI → 本月名单 + （本期换仓 / 行业分布）→ 净值与近 12 个月 → 三种交易方式 + 分年度与基准 → 策略说明；
- 看盘：4 指数卡片（带趋势状态点）、市场宽度卡片、行业热力表、涨跌榜、指数图；
- 回测：策略类型卡片（单策略 / 组合配置）+ 参数表单 + 运行 + 轮询进度 + 指标卡 + 净值与回撤图 + 逐年表 + 分区间 + 基准对比 + 交易统计；
- 个股：搜索框（回车/按钮）→ 详情卡 + K 线图 + 因子贡献表 + 基本面；
- 数据：一键更新 / 只重算策略 + 总进度条 + 阶段列表 + 一致性自检 + 数据源状态；- 因子：统计卡、因子族表、入模池表（含权重）、基本面因子表、复合分 IC 表；
- 可转债：KPI（全期/回撤/样本外/建议仓位）、策略净值、全市场 YTM 中位（含 YTM=0 虚线 + 末端当前值）、仓位建议、目标持仓、多资产组合对比；
- 报告：报告文件列表（新窗口打开）。
- 响应式断点：≤1180px 主栅格转单列、KPI 两列；≤1023px 侧栏收为图标；≤719px 侧栏转顶部横排。
- 无障碍：`:focus-visible` 2px accent 描边、提示文字对比度 ≥4.5:1、`prefers-reduced-motion` 下关闭动效。
- 错误提示 toast；所有网络失败要有可读提示；不阻塞页面。

## 多模型支持（2026-10-21 新增，2026-11-02 改为白名单）

模型注册表由 `web/services.py` 的 `MODEL_SHOW` 白名单决定，**顺序即白名单声明顺序**。

- 历史实现是"扫描 `build/models/*` 再排除黑名单"，结果每跑一次实验
  （`exp_*`、`wf_h10_*` 变体）下拉框就多几项，且字母序下主模型 `wf_h10`
  排在第 23 位，选主模型要往下滚很久。现已改为白名单：
- 磁盘上新目录**默认不可见**。确认可上线后在 `MODEL_SHOW` 登记一行
  `name -> (label, desc, kind)`，`kind` 取 `"prod"`（生产）或 `"research"`（研究对照）；
  注册表有 5 秒 TTL，改完白名单**无需重启**即可生效；
- 主模型必须排白名单第一项（前端默认选中项取 `default`）；
- 非白名单模型带 `?model=` 请求一律 **HTTP 400**，不再静默降级成"有价格、没分数、没因子"的空页面；
- 只剩一个模型时，前端自动隐藏模型下拉框（`state.models.length > 1` 判断）。

入库判据（为什么最后只留 `wf_h10`）：

| 模型 | 结论 | 依据 |
|---|---|---|
| `wf_h40` | 全期无改善 | 5.88% vs 6.13%，2024-26 好但 2020-21 掉 2.3pp，样本外 +(-0.5)bps |
| `wf_v1` | 已被取代 | 5 日标签早期版本，`wf_h10` 是其继任者 |
| `wf_h10_fund` | 未采用 | 37 因子含 4 个基本面，路线三实验否决 |
| `wf_rs10` | 不可比 | 仅 1 次 refit（2026-10-13）、6 因子，IC 曲线为全期回填，不是 walk-forward |

### GET /api/meta → model 字段
```json
{"model": {
  "name":"wf_h10", "default":"wf_h10",
  "models":[
    {"name":"wf_h10","label":"左侧主模型","desc":"量价反转+低波，34 因子，10 日标签（默认）",
     "kind":"prod","pool":34,"refit":"2026-10-13","horizon":10}
  ],
  "refit":"...","n_pool":34,"horizon":10,"method":"blend","capital":300000}}
```

### GET /api/models
```json
{"default":"wf_h10","models":[{"name":"wf_h10","label":"...","desc":"...","kind":"prod","pool":34,"refit":"...","horizon":10}]}
```

### 带 model 参数的接口
- `GET /api/recommend?model=wf_h10&top=60` — 返回体含 `"model"` 字段；默认 `wf_h10`；
- `GET /api/stock/{code}?model=wf_h10` — 复合分/排名/因子贡献按该模型计算；
- `POST /api/backtest` body 加 `"model":"wf_h10"` — 结果 `result.params.model` 回显；
  未知模型返回 **HTTP 400** `{"error":"未知模型 ...，可用：[...]"}`（个股接口同样 400）。
- 回测内部：不同模型的 `composite_score` 行数可能不同（模型训练时的数据长度），服务会自动 pad/截断到当前行情长度。
- `GET /api/factors/summary`：`counts` 新增 `trend`（trend-extra 因子数）与 `models`（可选模型名列表）。

## 本月推荐（2026-11-02 新增）

### GET /api/monthly_recommend
低换手月度策略的**当月视图**。与 `/api/today_recommend` 的分工：
today 回答「今天四条腿分别做什么」（四侧合并成一张执行清单）；
monthly 只讲低换手这一侧的一个自然月（名单 / 换入换出 / 行业分布 / 历史）。

- 数据源：`signals/lowturn_signal.json`（单策略满额口径）、`signals/latest_combo.json`
  的 `lowturn` 节点（按组合权重切预算后的股数）、`build/runs/lowturn/{result,picks}.json`
  与 `nav.parquet`。缺 `lowturn_signal.json` 时 **503**（"低换手信号不存在"）。
- 份额口径：有组合信号时用 `weights.lowturn` 切出的 legs（`sized_by="combo"`），
  否则退回单策略满额口径（`sized_by="standalone"`，页面会标注）。

```json
{"strategy":"lowturn_monthly","label":"低换手月度",
 "asof":"2026-10-30","target_month":"2026-11","status":"month_end",
 "status_cn":"月末已算出下月名单，等次月首个交易日开盘建仓",
 "next_action":"2026-11 首个交易日开盘",
 "capital":300000,"weight":0.3,"budget":90000,"sized_by":"combo",
 "invested":82787.4,"invested_pct":0.9199,"cash":7212.6,"n_pick":10,
 "rows":[{"code":"601988","name":"中国银行","sector":"银行","price":7.067,
          "shares":1200,"amount":8480.7,"turn20_pct":0.00154,
          "mcap_yi":14895.3,"adv20_yi":22.62,"is_new":false}],
 "rotation":{"prev_month":"2026-10",
             "entered":[{"code":"601158","name":"重庆水务"}],
             "exited":[{"code":"001872","name":"招商港口"}],"kept":8},
 "sector_dist":[{"sector":"银行","n":4,"amount":34677.6,"pct":0.4189}],
 "perf":{"cagr":0.1002,"sharpe":0.754,"max_dd":-0.127,"vol":0.1157,
         "gross_cagr":0.1403,"strict_cagr":0.0054,"cost_drag_pp":4.01,
         "turnover_ann":7.12,"turnover_full_switch":24.0,"cost_bp":49.27,
         "avg_overlap":0.7068,"win_rate":0.573,"n_months":82,
         "avg_mcap_yi":3720.4,"avg_turn20_pct":0.001,"corr_main":0.36},
 "yearly":{"2024":0.423},
 "bench":[{"key":"csi_all","name":"中证全指","cagr":0.0192,"sharpe":0.12,"max_dd":-0.393},
          {"key":"all_eq","name":"全A等权","cagr":0.0811,"sharpe":0.394,"max_dd":-0.3263},
          {"key":"div_etf","name":"中证红利ETF","cagr":0.1006,"sharpe":0.61,"max_dd":-0.1723,
           "note":"对照基准（含股息再投资）"}],
 "monthly_returns":[{"month":"2026-10","ret":0.0358}],
 "nav_series":[{"date":"2024-10-10","nav":1.0}],
 "stale":{"level":"ok"},"params":{"lookback":20,"min_list_days":250}}
```

字段口径：
- `perf.cagr` = **只换变动**（净，含 49.27bp 单边成本），`gross_cagr` 同路径零成本，
  `strict_cagr` 每月全清全建；三者差就是「不折腾」省下的成本，不要混用；
- `turn20_pct` 是**小数**（0.00154 = 0.154%），原始 `turn20_bp` 是基点，前端 `fmtPctU` 再乘 100；
- `nav_series` 取 `nav.parquet` 的 `nav_real`（真实账本），最近 500 个交易日、首值归一到 1.0。

**2026-11-02 两处修正**：

1. **产物已统一口径**。`build/runs/lowturn/*` 原先由 `build/exp_lowturn.py` 自带的一份
   独立回测实现生成，与 `qf/lowturn.py` 漂移（页面 9.84%、报告 10.02%，回撤 −10.68% vs −12.70%）。
   现在 `exp_lowturn.py` 只是 `qf.lowturn.backtest()` 的薄封装，单一口径：
   **10.02% / Sharpe 0.754 / 回撤 −12.70%，毛 14.03%，严格全换 0.54%**。
   `perf_strict` 读 `result.json` 的新字段 `perf_strict`，旧产物退回 `perf_net`。
2. **`bench` 必带「中证红利ETF」**。低换手持仓常年四大行 + 中石油 + 人寿，本质是红利组合；
   只给中证全指（不含股息、不做红利筛选）会把超额虚增约 8pp。红利 ETF 用 515080 后复权价
   （含分红再投资），区间与策略净值对齐。同区间：低换手 10.02%/0.754 vs 红利 10.06%/0.610
   —— **收益打平，优势只在波动更低**。

### /api/today_recommend 新增 `strategies`
四条腿各一项，供前端「按策略看推荐」卡片与筛选使用：

```json
[{"key":"main","name":"主策略","kind_cn":"荐股","desc":"左侧低波反转 · 40 只等权 · 40 日调仓",
  "jump":"backtest","weight":0.4,"budget":120000,"n":39,"amount":123522.7,
  "cagr":0.0733,"sharpe":0.606,"max_dd":-0.1332,
  "phase_band":{"cagr_mean":0.0597,"cagr_median":0.0599,"cagr_min":0.0385,
                "cagr_max":0.0744,"sharpe_median":0.4765,"n_phases":8,
                "prod_offset":10,"n_better":1,"rebalance_days":40}}]
```

- `phase_band`（2026-11-03 新增）读 `build/eval/phase_band.json`，由 `scripts/phase_band.py`
  在管道 `phase_band` 阶段生成。**主策略的 `cagr` 是单点值**：`rebalance_offset=10`
  只是 40 日调仓下 8 个与周频打分行对齐的相位之一，同一份分数的 CAGR 跨相位从 3.85%
  铺到 7.44%。前端展示主策略收益时应同时给出区间或中位数，不要把 `cagr` 当成策略能力。
- `jump` 是该策略的详情页：`main→backtest`、`lowturn→monthly`、`etf→etf`、`sector→sector`；
- 个股侧年化/Sharpe 走 `qf.metrics.performance(nav, rf=现金收益率)`（即 7.11% / 0.61），
  低换手取 `build/runs/lowturn/result.json` 的 `perf_real`，ETF/板块取 `build/eval/*_stats.json`；
  **不要自己用日历日折年化**，会算出 6.82%/0.76 与回测页打架。
