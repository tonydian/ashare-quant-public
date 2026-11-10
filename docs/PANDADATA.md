# PandaData 数据源（PANDADATA）

结论先行：**能拉数**。2026-10-29 当晚 20 次 `call_pandadata` 全部 `ok:true`，18 个不同请求。
本文档不是猜测，是从那批真实调用结果 + 各 agent 的 `data_gaps` / `risks` 自我声明里反推出来的契约。

> 写这份文档的原因：PandaData 的 MCP 工具是**会话级加载**的。已授权、服务端在线、凭据有效，
> 但一个在授权之前打开的会话拿不到工具（`ToolSearch` 精确搜 `mcp__pandadata__auth_status` 也取不到）。
> 遇到"取不到工具"，不要怀疑连接器坏了，**新开一个会话**即可。

## 一、接入姿势

MCP 端点：`https://pandadatamcp.pandaaiquant.com/mcp`（streamable-http），OAuth scope `pandadata:read`。

调用顺序（SKILL.md 强制）：

1. `auth_status` —— 先看授权，`reauth_required` 就停下让用户走 OAuth，不要循环重试
2. `search_methods` —— 用业务词找方法，比 `list_methods` 省 token
3. `get_method_doc` —— **每次取数前必读**，参数名/单位/枚举/示例以它为准
4. `call_pandadata` —— 传 `method` + `params`

辅助：`sdk_status`（服务状态诊断）、`list_methods`（浏览全目录）。

## 二、已验证可用的方法（17 个）

行数是 2026-10-29 那次真实返回。

| 方法 | 关键参数 | 返回字段 | 实测行数 | 备注 |
|---|---|---|---:|---|
| `get_trade_cal` | `start_date`,`end_date`,`exchange=SH`,`fields` | nature_date, is_trade | 50 | 区间 30 交易日 |
| `get_last_trade_date` | `exchange=SH` | date | 1 | 20261029 |
| `get_trade_list` | `date` | symbol, date | **5216** | 全市场当日代码表 |
| `get_stock_detail` | `symbol[]`,`fields` | symbol, name | 6 | 取名称/静态信息 |
| `get_stock_daily` | `start_date`,`end_date`,`symbol[]`,`fields` | date,symbol,close,pre_close | 20（预览） | **无复权参数**，见坑 2/3 |
| `get_index_daily` | `start_date`,`end_date`,`symbol[]`,`fields` | date,symbol,close,pre_close,amount | 111 | 指数行情 |
| `get_index_indicator` | `start_date`,`end_date`,`symbol`(单个!) | date,symbol,pe_ttm,pe_lyr,pb_lf,pb_lyr | 60 | 指数估值，**滞后一日** |
| `get_margin` | `start_date`,`end_date`,`symbol[]` | date,symbol,margin_balance,buy_on_margin_value,short_balance,total_balance,margin_type | **8262/日** | 两融，**T+1** |
| `get_lhb_list` | `start_date`,`end_date` | symbol,date,type,reason,amount,deviation | 67 | 上榜汇总 |
| `get_lhb_detail` | +`side`,`fields` | symbol,date,type,side,agency,b_value,s_value | 670 | **跨类型重复，见坑 5** |
| `get_block_trade` | `start_date`,`end_date` | symbol,date,price,volume,amount | 297/10日 | **T+1**，当日无数据 |
| `get_concept_list` | 无参 | concept,start_date,end_date | **622**（620 唯一） | **概念带生效区间** |
| `get_concept_constituents` | `concept[]` 或 `concept_stock[]`，可加 `date` | concept,concept_stock,date | **9926** | 双向可查 |
| `get_option_underlying_volatility` | `symbol`（单个）,`start_date`,`end_date` | date,symbol,historical_volatility,period,close | 1737 | 9 个周期 5/10/30/60/90/120/180/250/500 |
| `get_stock_disclosure_date` | `start_date`,`end_date`（过滤**预约披露日**，非报告期）,`symbol`（可选单股） | symbol,name,end_date,actual_disclosure_date,expected_disclosure_date,first_change_date,first_publish_date,latest_predisclosure_date | **170,469**（16 片合并，44 期） | 财报真实公告日，**只能走 JSON 溢出落盘**，见坑 9/10/11 |
| `get_hsgt_hold` | `start_date`,`end_date` | — | **0** | **北向，4 次全空，已废** |
| `get_factor` | `start_date`,`end_date`,`factors[]`（**必填**）,`symbol[]`,`type=stock`,`index_component` | 见 §七 | **5152/日** | **PIT 对齐的财报+量价宽表，见 §七** |
| `get_adj_factor` | `symbol[]`,`start_date`,`end_date`,`fields` | symbol,ex_date,ex_cum_factor,ex_factor,ex_end_date,announcement_date | 2/年（单股） | 复权因子，`ex_cum_factor` 累计前复权 |

**文档里有、但网关没部署（`code: 100002, Unknown method`）**——整批 10 个转债方法，
`get_method_doc` 都能拿到完整契约，但调用一律失败，别浪费时间：
`get_bond_basic` / `get_bond_conv_price_change` / `get_bond_conv_price` / `get_bond_redeem` /
`get_bond_put` / `get_bond_rating` / `get_bond_amount` / `get_bond_daily` / `get_bond_industry` /
`get_bond_cb_list`。**「文档有 ≠ 网关可用」**，落地前先发一次空参数调用探活。

未验证但文档里出现过、值得优先试的：`get_fina_forecast`（业绩预告）、
`get_restricted_list`（解禁）、`get_stock_pledge_stat`（质押）、`get_macro_*`（宏观）。

## 三、硬约束（踩过的坑，直接决定可行性）

1. **网关 JSON 上限 8,388,608 字节（8MB）**。全市场无 `symbol` 过滤的查询会直接
   `response_too_large`，必须显式传 symbol 列表再分批。
   `get_margin` 全市场单日 8262 行约 3.4MB，是最接近上限还活着的一档。
2. **parquet 传输会截断**。`get_stock_daily` 单次只回 **20 行预览**；同一接口的
   "预览上限约 566 行"。全市场 21 个交易日 109,702 行 → 只拿到 20 行，
   `truncated=true (truncation_reasons=[row_limit])`。**拿不到全市场历史日线。**
3. **`get_stock_daily` 没有复权参数**，返回的是接口默认价，除权除息会制造收益率噪声。
   → 只要价量数据，一律走我们自己的 `panel_ohlcv.npz`（后复权，5281×2428），不要用它的。
4. **`get_hsgt_hold` 是死的**：全市场、单只、放宽到 2 个月窗口，4 次调用全部返回空
   parquet（`size 0`，空文件 sha256 `e3b0c442...b855`）。**北向维度这条线放弃。**
5. **`get_lhb_detail` 跨类型重复计数**：同一只股票在不同上榜类型（如 `G0007` 与 `T0020`）
   下的席位明细**逐行完全相同**。直接对 670 行求和会把同一笔钱算两遍
   （实测同一份数据：未去重 -20.76 亿 → 跨类型去重 -23.58 亿 → 单日榜口径 -8.64 亿）。
   引用净额必须标注口径，我们自己算还得按 `(symbol,type)` 先去重。
6. **字段投影是硬性的**：`fields` 里没写的列不会返回。想拿 `date` 列就必须显式请求，
   否则 `date_range` 只能记 `not_returned`。想拿席位必须显式请求 `agency`，否则全是 null。
7. **T+1 普遍存在**：两融、大宗交易、指数估值的最新数据都比行情晚一个交易日。
   当日 `get_block_trade` 返回 0 行不等于"没有大宗交易"，是没发布。
8. **`get_index_indicator` 一次只能传单个指数**，传数组返回 0 行。
9. **「溢出」不等于「失败」**。MCP 回传有 token 上限，超过就报
   `result (N characters) exceeds maximum allowed tokens`，**但整包 JSON 已经写到磁盘**：
   `~/.workbuddy/projects/<proj>/<session>/tool-results/mcp-pandadata-call_pandadata-*.txt`
   内容是完整 `{ok, method, params, result, gateway, request_id}`，`json.load` 可直接读。
   → 全市场取数的正确姿势是**故意让它溢出，然后去读落盘文件**，不要试图缩小到不溢出。
10. **`preview_rows` 挡不住溢出**。它只控制本地预览，溢出判定发生在 preview 之前。
    设 `preview_rows=3` 照样会因为整包太大而溢出。
11. **「超时」和「溢出」是两种失败，处理方式相反**：
    - `code: "timeout"` / `detail: "Pandadata gateway request timed out"` → 真失败，要**缩窗口**。
    - 溢出 → 成功，去读文件。
    判断顺序：**先缩窗口，再看是否还超时**。`fields` 裁剪基本没用——`get_stock_disclosure_date`
    只请求 3 个字段仍返回 4 个（该方法忽略 `fields`），2023 全年片减到 3 字段仍超时。
    瓶颈是「行数 × 年数」，1 年片是最稳的粒度。
12. **`start_date`/`end_date` 过滤的可能不是你以为的字段**。
    `get_stock_disclosure_date` 传 `end_date=20260430`，返回里却有 `actual_disclosure_date=20260515` 的行
    → 过滤的是**预约披露日**，不是实际披露日。所以一期财报的披露可能跨出窗口边界，
    分片时要留余量（一期报告至少覆盖「法定截止日 + 30 天」）。
13. **`first_publish_date` / `latest_predisclosure_date` 在全市场批量查询下 100% 为 None，
    但单股查询（带 `symbol`）有值**。同一字段的可用性取决于查询通道，不能一概判为坏字段。

## 四、对我们这个系统：什么有用、什么没用

对照 `docs/DATA_NOTES.md` 的现有数据。

### 没用（我们已有更好的）

- **个股日线**：它截断到 20 行且无复权；我们有 5281×2428 后复权面板 + 不复权价 + 时点市值。不要换。
- **指数行情**：`indices.csv` 已有 hs300/csi500/csi1000/csi_all。
- **交易日历**：`panel_ohlcv.dates` 就是。
- **北向持股**：它是空的。

### 有用（我们真缺）

| 数据 | 补什么 | 优先级 |
|---|---|---:|
| `get_margin` 全市场两融余额/融资买入/融券余额 | **杠杆与资金面维度**。威科夫审计明确指出我们的因子池"无换手、无市值外的资金面"，这是最直接的补口 | ★★★ |
| `get_concept_list` + `get_concept_constituents`（带 `date`） | **历史时点概念归属**。现有 `industry.npy` 只有 45 类且新旧分类混杂；概念 620 个、成分关系 9926 条，且每个概念有 `start_date/end_date` → 天然可做 PIT，无前视 | ★★★ |
| `get_fina_forecast` 业绩预告（待验证） | **PEAD 的燃料**。我们 `fund/` 只有实际财报、无披露日期、无预告；遗留待办里的"PEAD 负面剔除"卡的就是这个 | ★★★ |
| `get_restricted_list` 解禁 / `get_stock_pledge_stat` 质押（待验证） | 个股风险剔除，与现有 ST 静态过滤互补 | ★★ |
| `get_lhb_detail` / `get_block_trade` | 事件类因子（龙虎榜净买、大宗折溢价）。注意坑 5 的去重 | ★★ |
| `get_option_underlying_volatility` 9 周期 HV | 市场情绪/择时。但只有 510050 系列，覆盖窄 | ★ |
| `get_index_indicator` PE/PB | 只覆盖少数指数且滞后，做全市场估值分位不够 | ★ |

**一句话判断**：这个源的价值不在"再来一份日线"，而在**资金面（两融）+ 题材归属（概念 PIT）+ 事件（业绩预告/解禁）**这三类我们完全没有的东西。

## 五、落地顺序与执行结果

### 5.1 已执行（2026-10-30）

| 步骤 | 结果 | 产物 |
|---|---|---|
| `search_methods` 搜 5 组关键词 | 业绩预告→`get_fina_forecast`（存在但未取数）；解禁→`get_restricted_list`；质押→`get_stock_pledge_stat`；融资融券→`get_margin`（可用）；宏观→55 个 `get_macro_*`（可用但不做） | — |
| `get_margin` 3 个交易日全市场 | **8262 行/日**，正常落盘 | `build/pandadata/margin_sample.parquet` |
| `get_concept_list` 全量 | 622 行 / 620 唯一概念，**每个概念带 `start_date`/`end_date`** | `build/pandadata/concept_list.json` |
| `get_concept_constituents` 抽 20 概念 × 2 个 date | **成分随 date 变化 → PIT 成立** | 见 `reports/数据缺口盘点_2026-10-30.md` §3 B2 |
| `get_stock_disclosure_date` 全量 | **170,469 行 / 44 期（2015Q4–2026Q3）连续无断档** | `build/pandadata/disclosure_date.parquet` |
| 调用日志 | 每次 method/params/row_count/字节量/状态 | `build/pandadata/call_log.json` |

### 5.2 还没做

1. `get_fina_forecast`（业绩预告）——PEAD 的燃料，未取数。
2. `get_restricted_list`（解禁）/ `get_stock_pledge_stat`（质押）——未取数。
3. 转债 4 张表——**网关未部署，做不了**，见第二节。
4. 概念 PIT 上因子——成分语义已验证成立，但**无移除日期**，需先解决「退概念」的时点问题。

**原则不变：在样本外 IC 出来之前，不碰生产模型。**

## 六、失败复盘（别再踩）

- 全市场不带 `symbol` 直接查 → `response_too_large`。先传白名单。
- 只传 `symbol` 数组给 `get_index_indicator` → 0 行。它只吃单个。
- `get_option_underlying_volatility` 带 `period=20` 且窗口只有 1 个月 → 0 行。
  去掉 `period`、放宽到年内窗口才行。
- 龙虎榜把 `get_lhb_list`(67 条汇总) 和 `get_lhb_detail`(670 条明细) 混着求和 → 口径打架。
  两个接口层级不同，不可相加。
- 空 parquet 不等于"没数据"，但连续多次、放宽窗口仍空，就是真没数据（北向即如此）。
  区分清楚，别把环境缺失写成"资金没有流出"。
- 看到 `exceeds maximum allowed tokens` 就以为调用失败、重试 → 白等。**那其实是成功**，
  数据已经在 `tool-results/*.txt` 里，去读文件（坑 9）。
- 整批 10 个 `get_bond_*` 方法：`get_method_doc` 全部返回完整契约，调用全部 `Unknown method`。
  **文档齐不等于网关通**，先发一次探活再规划工作量。
- 想靠「缩 `fields`」救 `get_stock_disclosure_date` 的 2023 全年片，两次尝试都超时。
  这个方法的 `fields` 根本不生效，只能缩时间窗口。

## 七、量化因子库（2026-10-30 探测）

`category="量化因子数据"` 共 7 个方法，**只有 2 个活着**：

| 方法 | 状态 | 说明 |
|---|---|---|
| `get_factor` | ✅ 可用 | PIT 对齐的财报+量价宽表，**本节的结论都基于它** |
| `get_adj_factor` | ✅ 可用 | 复权因子，`ex_cum_factor` = 累计前复权因子 |
| `get_factor_exposure_style` | ❌ `Unknown method` | Barra 25 个风格暴露 |
| `get_factor_style_return` | ❌ `Unknown method` | 55 个因子日度收益 |
| `get_factor_style_residual` | ❌ `Unknown method` | 残差风险/收益 |
| `get_factor_style_covariance` | ❌ `Unknown method` | 风格因子协方差 |
| `get_rf_rate` | ❌ `Unknown method` | 无风险利率 |

**Barra 那一整套（exposure / return / residual / covariance / rf_rate）全部未部署**，
文档写得最详细、字段最漂亮（25 个风格因子），但一个都调不通。别按文档规划工作量。

### 7.1 `get_factor` 的字段：没有因子清单，只能探

文档只列了 8 个基础量价字段，其余写「详见下载文件」——**那个文件拿不到**。
实测服务端对不认识的因子名**静默丢弃**（不报错、不返回该列），所以探测办法是：
塞一大批候选进去，看回来哪几列。已知**存在**的字段（约 41 个）：

| 类别 | 字段 |
|---|---|
| 量价 | `open` `close` `high` `low` `volume` `amount` `market_cap` `turnover` |
| 利润表 | `revenue` `operating_revenue` `net_profit` `gross_profit` `selling_expense` `investment_income` `fair_value_change_income` `non_operating_expense` `other_income` |
| 资产负债 | `total_assets` `total_liabilities` `total_equity` `inventory` `capital_reserve` `goodwill` `intangible_assets` `current_assets` `current_liabilities` `non_current_assets` `non_current_liabilities` `advance_from_customers` `bond_payable` `contract_liabilities` `deferred_income` `long_term_loans` `short_term_loans` `minority_interest` `paid_in_capital` `surplus_reserve` `undistributed_profit` `notes_payable` `construction_in_progress` `other_current_assets` `other_non_current_assets` `oil_and_gas_assets` |
| 估值 | `dividend_yield` |

**确认不存在**（传了不返回）：所有技术指标（`ma*` `rsi` `macd` `kdj` `boll` `atr` `bias` `roc`
`volatility` `std` `skew` `volume_ratio`）、所有估值比率（`pe` `pb` `ps` `ep` `bp` `pe_ttm`
`pb_mrq`）、所有财务比率（`roe` `roa` `roic` `gross_margin` `net_margin` `debt_ratio`
`current_ratio` `asset_turnover`）、所有每股指标（`eps` `bps` `basic_eps` `diluted_eps`）、
所有现金流（`net_operating_cash_flow` `free_cash_flow` `ocf` `cfo` `cfi` `cff`）、
股本类（`total_share` `float_share` `free_float` `shares`）。

→ 结论：**这不是因子库，是「日频 PIT 对齐的三大报表宽表」**。真正新增的是资产负债表，
我们本地 `fund/*.parquet` 完全没有这些科目；`dividend_yield` 也是新的。
**比率型因子一个都没有**，全得自己算。

### 7.2 最重要的一点：财报字段自带严格 PIT

平安银行 `000001.SZ` 2025 年 `net_profit` 的跳变点：

| 出现日 | net_profit | 对应报告期 |
|---|---|---|
| 20250102 | 397.29 亿 | 2024Q3 累计 |
| 20250314 | 445.08 亿 | 2024 年报 |
| 20250418 | 140.96 亿 | 2025Q1 |
| 20250822 | 248.70 亿 | 2025 中报 |
| 20251024 | 383.39 亿 | 2025Q3 |

**20 只跨周一至周五披露的样本股，跳变日与本地 `actual_disclosure_date` 全部精确相等
（20/20，0 天偏差）**。所以日期的语义是「**公告登记日当天生效**」，不做 T+1 延迟。

⚠️ **使用时必须自己 `shift(1)`**：A 股多数公告是盘后发布，`get_factor` 在公告日当天
就给了新值，直接用等于前视 1 天。平安银行那 4 期「周五跳变、周六公告」正是这个问题的
显影——公告日登记为周六、实际周五晚间发布。

财报字段**从 2017 年起可用**（2016-06-30 只有量价+市值，`net_profit` 等被静默丢弃）。
正好卡在项目 `DEV_START=2017-01-01` 上，不构成损失。

### 7.3 它能补上上一轮 20% 的披露日缺口

`get_stock_disclosure_date` 对 Q1/Q4 系统性缺行（覆盖率 Q1 80%、Q4 82%），上一轮判定
「无法修复」。但 `get_factor` 的财报值**有 Q1**：

- 五粮液 `000858` 在 **20240429** 首次出现 2024Q1 净利 146.35 亿——而披露日接口
  对 `20240331` 期**根本没有五粮液**
- `000021` 在 **20260428** 由 9.89 亿跳到 2.88 亿——正是披露日接口缺失的 `20260331` 期

→ **「财报值首次变化的日期」= 真实披露日**，可以作为兜底数据源回填那 20%。
一个限制：**同日/相邻披露多期时，宽表只保留最新一期**（五粮液 2023 年报的 302 亿在日序列里
从未出现，被 4/29 的 Q1 值直接顶替）。所以反推只能拿到「最后可见」的那期，年报可能被跳过。

### 7.4 硬约束（比别的接口更严）

1. **`preview_rows` 上限 2000**（别的接口没这个限制）。单日全市场 5152 行 → 必须拆 3 次。
2. **没有 `offset`/`limit` 分页参数**，只能用 `symbol` 列表切批。
3. 返回还受 **`byte_limit`** 约束：20 股 × 25 天 × 2 字段标称 500 行，实际只回 381 行
   就 `truncated (byte_limit)`。所以**行数上限和字节上限哪个先撞不一定**。
4. `source: "parquet"` + `server_path` + `sha256`，但**网关侧文件拿不到本地**，只有预览行回传。
5. `factors` 全部无效时返回 **HTTP 400 无 detail**（不是空表），别把 400 当参数错误去猜。

→ **全量重建不可行**：日频 5000 股 × 2000 天 = 1000 万行，按 2000 行/次要 5000+ 次调用。
可行的用法只有两种：**按需精准拉**（指定 symbol + 短窗口），或**按期取截面**
（披露截止日后取一次，4 期/年 × 9 年 = 36 个截面 × 3 批 = 108 次调用）。

### 7.5 通道吞吐实测：MCP 建不了库，必须走 SDK

上一条的「108 次调用」估算是按 `preview_rows=2000` 算的，**实测不成立**。
真正的瓶颈是 **byte_limit**，服务端单次只回约 3.5 万字节：

| 请求 | 字段数 | total_rows | 实际回传 | 结论 |
|---|---:|---:|---:|---|
| 单日全市场 | 4 | 5146 | **268 行** | 268 × 4 字段 ≈ 3.5 万字节 |
| 单日全市场 | 10 | 5146 | **133 行** | 字段翻倍，行数减半 |
| 单日 + `index_component=000300` | 6 | 300 | **109 行** | 见下 |

- **`preview_rows=2000` 拿不到 2000 行**，被 byte_limit 先截断，且**不会溢出落盘**
  （坑 9 那套「故意溢出再读文件」在这儿失效，因为截断发生在网关侧读 parquet 时）
- **`index_component` 参数会作为一列返回**，是个 100+ 字符的位掩码字符串，把字节预算吃掉一半
  （300 行只回 109 行）。想用指数池限流，反而更亏
- 单日全市场资产负债表（4 字段）→ **20 批/截面**；30 个截面 = **600 次调用**。不现实

**结论：MCP 通道适合探活、拿小样本、验证语义，不适合建数据仓库。**

正解是本地 SDK：`panda_data` 已在 **PyPI 上（1.0.5）**，`pip install panda_data` 即可。
SDK 直连能一次拿完整 parquet，没有 3.5 万字节限制——上一轮记录「本地无 SDK」的前提
已经不成立，是当时没查 PyPI。

⚠️ SDK 需要**独立凭据**（用户名/密码或 token）。MCP 的 `sdk_status` 返回
`data_mode: "gateway"`、`auth_enabled: true`，说明 MCP 用的是 OAuth 授权给网关的通道，
**不能直接复用给 SDK**。装完包还得有账号凭据才能真正取数。

### 7.6 SDK 已装好，能力面比 MCP 大一个数量级

`panda_data` 1.0.5 已装进 `D:/Python/Python313`。**直接 `pip install` 会失败**
（`metadata-generation-failed`，它把 numpy 拖去从源码编译，本机没有 MSVC），
正确姿势是绕开依赖解析：

```bash
python -m pip download panda_data --no-deps --no-binary :none: -d D:/tmp/pdwhl
python -m pip install --no-deps --no-index --find-links D:/tmp/pdwhl panda_data
```

它本身是纯 Python wheel（184 KB），本机 numpy 2.4.6 / pandas 3.0.3 已够用。

**装完暴露 325 个 `get_*` 方法**，完整清单见 `docs/PANDADATA_SDK_METHODS.md`。
和 MCP 的 17 个对比，差的不是数量，是**整块能力**：

| 我们之前判「拿不到」的 | MCP | SDK |
|---|---|---|
| Barra 风格因子（exposure/return/residual/covariance/rf） | 全部 `Unknown method` | **8 个方法都在** |
| 可转债（conv_basic / call_exec / call_progress / price_change / term / daily） | 10 个全 `Unknown method`，判定「这条路是死的」 | **9 个都在** |
| 财务三大报表 | 无 | `get_fina_statement` / `get_fina_reports` / `get_fina_ex` |
| 一致预期 | 无 | `get_stock_ncycl_consensus` / `get_stock_recommendation_estimate` |
| 质押 / 解禁 | 文档有、未验 | `get_stock_pledge` / `get_stock_pledge_stat` / `get_restricted_list` |
| 北向 | 4 次全空，判「已废」 | `get_hsgt_hold` 在，待重验 |
| 宏观 | 未验 | **56 个 `get_macro_*`** |
| 股票指标（pv / mktfin / operating）+ 行业中位数 | 无 | 8 个，行业横截面标准化可直接用 |

→ **上一轮那些「网关没部署、这条路是死的」结论，全部要作废并重估。**
MCP 只是 SDK 的一个受限子集，它的 `Unknown method` 只说明**网关注册表里没有**，
不说明数据不存在。以后探活不能只用 MCP。

认证：`init_token(username, password, base_url='http://pandadata.pandaaiquant.com')`
会缓存 token 到 `user.json`（`ttl=14400` 秒），之后 `init()` 即可。
`scripts/probe_pandadata_sdk.py` 可查当前认证状态。



