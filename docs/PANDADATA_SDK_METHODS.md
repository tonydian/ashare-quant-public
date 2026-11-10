# PandaData SDK 方法清单

由 `scripts/list_pandadata_sdk_methods.py` 反射生成，共 **325** 个 `get_*` 方法。
包：`panda_data` 1.0.5（PyPI，纯 Python wheel）。

> MCP 网关只暴露了 17 个可用方法，而 SDK 有这一整张表。
> **MCP 报 `Unknown method` 不等于数据不存在，只是网关没部署那个接口。**

## 分类汇总

| 主题 | 方法数 | 说明 |
|---|---:|---|
| **Barra 风格因子** | 8 | 风格暴露 / 日度收益 / 残差 / 协方差 / 无风险利率。MCP 全部 Unknown，SDK 有 |
| **因子与复权** | 4 | get_factor 是 PIT 财报+量价宽表；get_adj_factor 是复权因子 |
| **财务报表** | 6 | fina_statement / fina_reports / fina_ex / fina_forecast / fina_performance |
| **可转债** | 9 | MCP 整批 Unknown，SDK 有——这是当前唯一能改变「可转债不可回测」的东西 |
| **股票行情** | 14 | 日/周/月/分钟/实时行情 |
| **股票指标** | 8 | pv / mktfin / operating 三套指标，以及行业中位数（可直接做横截面标准化） |
| **一致预期与分析师** | 4 | ncycl_consensus / recommendation_estimate / recommendation_consensus |
| **股东与股权** | 9 | 股东持股 / 股东户数 / 股本结构 |
| **质押与解禁** | 4 | 质押、解禁、累计担保 |
| **分红与股本变动** | 16 | 分红、配股、拆股、回购、定增、超配 |
| **公司事件** | 12 | 诉讼、重大合同、股东大会、投资者关系、证监会审核、状态变更 |
| **IPO** | 6 | 首发上市、发行、询价、网下配售 |
| **内部人与高管** | 2 | 内部人交易 |
| **董监高与投资者** | 16 | 投资者活动、集中度、榜单 |
| **资金与龙虎榜** | 6 | 龙虎榜、资金流、两融、北向、大宗。MCP 里北向是死的，SDK 待验 |
| **经纪商** | 12 | 席位盈亏、评级、净持仓 |
| **指数** | 9 | 行情、成分、权重、估值 |
| **行业与概念** | 4 | 行业成分、概念成分 |
| **基金与 ETF** | 23 | 净值、持仓、风格、ETF 申赎 |
| **期货** | 29 | 行情、持仓、基差、库存、期限结构 |
| **期权** | 15 | 行情、隐含波动率、风险指标 |
| **宏观** | 56 | 约 70 个宏观序列 + 宏观日历 |
| **港股/美股** | 29 | 港股、美股、海外 |
| **交易工具** | 6 | 交易日历、代码表、停牌、ST |
| **认证与客户端** | 2 | 会话管理 |
| 其他 | 16 | — |

## Barra 风格因子（8）

风格暴露 / 日度收益 / 残差 / 协方差 / 无风险利率。MCP 全部 Unknown，SDK 有

```
get_factor_exposure_style
get_factor_style_covariance
get_factor_style_residual
get_factor_style_return
get_fund_bond_style
get_fund_investment_style
get_fund_stock_style
get_rf_rate
```

## 因子与复权（4）

get_factor 是 PIT 财报+量价宽表；get_adj_factor 是复权因子

```
get_adj_factor
get_adj_factor_hk
get_factor
get_factor_hk
```

## 财务报表（6）

fina_statement / fina_reports / fina_ex / fina_forecast / fina_performance

```
get_audit_opinion
get_fina_ex
get_fina_forecast
get_fina_performance
get_fina_reports
get_fina_statement
```

## 可转债（9）

MCP 整批 Unknown，SDK 有——这是当前唯一能改变「可转债不可回测」的东西

```
get_bond_cash_flow
get_bond_conv_basic
get_bond_conv_call_exec
get_bond_conv_call_progress
get_bond_conv_price_change
get_bond_conv_term
get_bond_coupon
get_bond_daily
get_bond_option
```

## 股票行情（14）

日/周/月/分钟/实时行情

```
get_market_data
get_market_min_data
get_stock_daily
get_stock_daily_post
get_stock_daily_pre
get_stock_detail
get_stock_detail_all
get_stock_min
get_stock_min_post
get_stock_min_pre
get_stock_monthly
get_stock_rt_daily
get_stock_rt_min
get_stock_weekly
```

## 股票指标（8）

pv / mktfin / operating 三套指标，以及行业中位数（可直接做横截面标准化）

```
get_stock_industry_median
get_stock_mktfin_indicator
get_stock_mktfin_metric
get_stock_operating_indicator
get_stock_operating_metric
get_stock_pv_indicator
get_stock_pv_metric
get_stock_sector_median
```

## 一致预期与分析师（4）

ncycl_consensus / recommendation_estimate / recommendation_consensus

```
get_stock_ncycl_consensus
get_stock_ncycl_estimate
get_stock_recommendation_consensus
get_stock_recommendation_estimate
```

## 股东与股权（9）

股东持股 / 股东户数 / 股本结构

```
get_holder_count
get_share_float
get_stock_equity_illegal
get_stock_equity_nature
get_stock_equity_placard
get_stock_shareholder_change
get_stock_shareholder_holding
get_stock_shareholder_report
get_top_holders
```

## 质押与解禁（4）

质押、解禁、累计担保

```
get_cumu_guarantee
get_restricted_list
get_stock_pledge
get_stock_pledge_stat
```

## 分红与股本变动（16）

分红、配股、拆股、回购、定增、超配

```
get_fund_etf_dividend
get_fund_etf_split
get_hk_etf_dividend_event
get_repurchase
get_stock_allotment
get_stock_cash_dividend
get_stock_dividend
get_stock_dividend_activity
get_stock_dividend_amount
get_stock_dividend_event
get_stock_over_allotment
get_stock_preferred_dividend
get_stock_private_placement
get_stock_split
get_stock_status_over_allotment
get_us_etf_dividend_activity
```

## 公司事件（12）

诉讼、重大合同、股东大会、投资者关系、证监会审核、状态变更

```
get_stock_csrc_approval
get_stock_financial_activity
get_stock_financial_event
get_stock_ir_activity
get_stock_ir_event
get_stock_litigation_arbitration
get_stock_market_activity
get_stock_market_event
get_stock_material_contract
get_stock_meeting_activity
get_stock_meeting_event
get_stock_status_change
```

## IPO（6）

首发上市、发行、询价、网下配售

```
get_stock_ipo_first_day
get_stock_ipo_issuance
get_stock_ipo_listing
get_stock_ipo_offline_bid
get_stock_ipo_offline_placing
get_stock_ipo_offline_stat
```

## 内部人与高管（2）

内部人交易

```
get_stock_insider_trade
get_stock_insider_transaction
```

## 董监高与投资者（16）

投资者活动、集中度、榜单

```
get_hk_etf_investor_concentration
get_hk_etf_top20_concentration
get_investor_activity
get_investor_brief_detail
get_investor_brief_qa
get_stock_investor_centralization
get_stock_investor_concentration
get_stock_investor_leaderboard
get_stock_top20_centralization
get_stock_top20_concentration
get_us_etf_investor_centralization
get_us_etf_investor_leaderboard
get_us_etf_top20_centralization
get_us_warrant_investor_centralization
get_us_warrant_investor_leaderboard
get_us_warrant_top20_centralization
```

## 资金与龙虎榜（6）

龙虎榜、资金流、两融、北向、大宗。MCP 里北向是死的，SDK 待验

```
get_block_trade
get_hsgt_hold
get_lhb_detail
get_lhb_list
get_margin
get_stock_flow
```

## 经纪商（12）

席位盈亏、评级、净持仓

```
get_broker_build_process
get_broker_flow_daily
get_broker_grade
get_broker_loss_rank
get_broker_ls_ratio
get_broker_netmarg
get_broker_netmarg_change
get_broker_oi_value
get_broker_profit
get_broker_profit_rank
get_broker_totlmarg
get_broker_variety_profit
```

## 指数（9）

行情、成分、权重、估值

```
get_index_component
get_index_constituent
get_index_daily
get_index_detail
get_index_detail_all
get_index_indicator
get_index_min
get_index_weights
get_index_weights_day
```

## 行业与概念（4）

行业成分、概念成分

```
get_concept_constituents
get_concept_list
get_industry_constituents
get_industry_detail
```

## 基金与 ETF（23）

净值、持仓、风格、ETF 申赎

```
get_fund_asset_allocation
get_fund_asset_allocation_detail
get_fund_bond_distribution
get_fund_category_changes
get_fund_daily
get_fund_daily_post
get_fund_daily_pre
get_fund_detail
get_fund_etf_constituents
get_fund_etf_cr
get_fund_etf_cr_limits
get_fund_etf_cr_net
get_fund_etf_listing
get_fund_etf_suspend
get_fund_holding_detail
get_fund_index_category
get_fund_industry_allocation
get_fund_min
get_fund_nav
get_fund_nav_deviation
get_fund_performance_metrics
get_fund_stock_portfolio_change
get_fund_stock_trade
```

## 期货（29）

行情、持仓、基差、库存、期限结构

```
get_future_basis
get_future_calendar_arbitrage
get_future_contract_indicators
get_future_contract_pool
get_future_contract_rank
get_future_daily
get_future_daily_post
get_future_detail
get_future_detail_all
get_future_dominant
get_future_dominant_corr
get_future_free_ratio
get_future_free_spread
get_future_inventory
get_future_ls_ratio
get_future_market_post
get_future_min
get_future_net_flow
get_future_netcap_change
get_future_netposi_rank
get_future_nonbroker_net
get_future_spot_profit
get_future_symbol_posi
get_future_term_structure
get_future_trader_quote
get_future_variety_mcap
get_future_variety_posi
get_future_virtual_ratio
get_future_warehouse_receipt
```

## 期权（15）

行情、隐含波动率、风险指标

```
get_option_commodity_rank
get_option_daily
get_option_detail
get_option_exercise
get_option_implied_volatility
get_option_margin
get_option_risk_indicators
get_option_spot_market
get_option_static
get_option_trade_rank
get_option_trade_stats
get_option_underlying_detail
get_option_underlying_exercise
get_option_underlying_volatility
get_option_variety_rank
```

## 宏观（56）

约 70 个宏观序列 + 宏观日历

```
get_macro_ad
get_macro_ag
get_macro_ar
get_macro_au
get_macro_bm
get_macro_cal
get_macro_cal_config
get_macro_cal_info
get_macro_ce
get_macro_ch
get_macro_ci
get_macro_cm
get_macro_detail
get_macro_dt
get_macro_ec
get_macro_ed
get_macro_ee
get_macro_eh
get_macro_en
get_macro_ep
get_macro_ew
get_macro_fa
get_macro_fb
get_macro_fe
get_macro_fi
get_macro_fs
get_macro_gb
get_macro_ha
get_macro_in
get_macro_ir
get_macro_is
get_macro_li
get_macro_mb
get_macro_md
get_macro_me
get_macro_na
get_macro_nf
get_macro_of
get_macro_or
get_macro_ph
get_macro_pi
get_macro_pm
get_macro_pp
get_macro_pr
get_macro_rb
get_macro_rc
get_macro_re
get_macro_se
get_macro_sm
get_macro_st
get_macro_te
get_macro_th
get_macro_tm
get_macro_tr
get_macro_ut
get_macro_wr
```

## 港股/美股（29）

港股、美股、海外

```
get_hk_daily
get_hk_daily_post
get_hk_daily_pre
get_hk_detail
get_hk_detail_all
get_hk_etf_daily
get_hk_etf_detail
get_hk_etf_detail_all
get_hk_etf_investor_ranking
get_hk_index_constituents
get_hk_index_daily
get_hk_index_detail
get_oversea_stock_list
get_us_daily
get_us_daily_post
get_us_daily_pre
get_us_detail
get_us_detail_all
get_us_etf_daily
get_us_etf_detail
get_us_etf_detail_all
get_us_index_constituents
get_us_index_daily
get_us_index_detail
get_us_min
get_us_short_interest
get_us_warrant_daily
get_us_warrant_detail
get_us_warrant_detail_all
```

## 交易工具（6）

交易日历、代码表、停牌、ST

```
get_last_trade_date
get_prev_trade_date
get_st_list
get_suspended_list
get_trade_cal
get_trade_list
```

## 认证与客户端（2）

会话管理

```
get_client
get_token_ttl
```

## 其他（16）

```
get_factory
get_stock_competitor
get_stock_competitor_information
get_stock_disclosure_date
get_stock_industry
get_stock_intermediary
get_stock_intermediary_information
get_stock_investor_ranking
get_stock_issuer_credit_rating
get_stock_preferred_detail
get_stock_preferred_placement
get_stock_preferred_rating
get_stock_preferred_shares
get_stock_preferred_trading
get_stock_rela_party_trans
get_stock_related_party
```
