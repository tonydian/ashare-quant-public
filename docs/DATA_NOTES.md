# 数据说明（DATA_NOTES）

数据包：`D:\wkf\getrich\data_extract\was_stock_data`（只读！不要修改）。

自检：`D:\wkf\getrich\.venv\Scripts\python.exe verify_quick.py`（在数据目录下运行）。

## 面板文件

| 文件 | 形状/字段 | 说明 |
|---|---|---|
| `panel_ohlcv.npz` | `dates(2428)`, `codes(5281)`, `open/high/low/close/volume/amount(2428,5281) float32` | `close` 是**后复权**价（不是成交价）；`volume` 股，`amount` 元 |
| `raw_close.npz` | `rawclose(2428,5281)` | 新浪**不复权**收盘价（实际价代理），空格子 25% |
| `market_cap.npz` | `mcap(2428,5281)` | 时点市值 = 不复权收盘 × 当日总股本（PIT），未插补 |
| `industry.npy` | `(2428,5281) int16` | 行业索引，-1 未知；`industry_cats.txt` 为名称表（45 类，含新旧两套分类混杂） |
| `industry_history_raw.csv` | 原始行业变更表 | 可选 |
| `meta.csv` | 5481 行：code/name/list_date/board/shares/float_shares/delist_date/is_delisted | 比面板多 200 个指数代码，用前与 codes 交集；面板代码顺序为原始 npz 顺序（未排序，需 map 对齐） |
| `delisted_universe.csv` | 255 只退市股 | code/name/list_date/delist_date/bars/first/last |
| `fund/YYYYMMDD.parquet` | 42 期：code,name,eps,revenue,rev_yoy,net_profit,profit_yoy,profit_qoq,bps,roe,ocfps,gpm,em_industry,period | 原始财报，**本身无披露日期**。PIT 对齐已升级为**每股真实公告日**（`build/pandadata/disclosure_date.parquet`，来源 PandaData `get_stock_disclosure_date`）；缺失时回退「法定披露截止日 + 15 天缓冲」（Q1 +45 / 半年报 +77 / Q3 +46 / 年报 +135）。见 `qf/fundamentals.py` |
| `indices.csv` | hs300/csi500/csi1000/csi_all 日线，1995?~2026-10-16 | 比面板多 2 天（10-15/10-16） |
| `bond_000012.parquet` | 上证国债指数日线 | 2003 起，可用于现金替代/债券配置研究 |
| `gold_518880.parquet` | 黄金 ETF 518880 收盘 | 2013 起 |

## 已知数据缺陷（来自 README，务必落实）

1. 复权因子最大 14557（600601 等 2 只 >5000 的股票在"整手可执行性"上会失真；
   本系统按实际价过滤 MAX_PRICE=200，天然剔除）。
2. 市值按格子覆盖 75%，按股票 95.1%；缺失 258 只 = 255 退市 + 3 只在市。
   → 市值缺失的股票不进入选股域（`mcap` 非有限即剔除），不做插补。
3. 行业覆盖：有行情格子未知 3.6%。→ 行业中性化时未知行业单独一组。
4. **无历史指数成分股** → 禁止做指数增强；基准只能用于事后对比。
5. 财报原始文件无披露日期 → 已接 PandaData 真实公告日做 PIT 对齐（`build/pandadata/disclosure_date.parquet`，44 期 2015Q4–2026Q3，170,469 行）。
   **上游对一季报/年报有系统性缺行**（覆盖率 Q1 80.0% / Q4 81.8%，Q2 98.6% / Q3 98.1%），缺失格子回退固定滞后。
   实测固定滞后比真实公告日**系统性偏晚 15~20 天**；改用真实公告日后 DEV 因子 IC 提升 13/15（均值 +0.0018）。
   主模型 `wf_h10` 用 `--no-fundamental` 训练，**不消费基本面因子**，故本次改造不影响主口径 7.16%。
6. **无 ST/*ST 历史状态** → 系统用 `meta.name` 当前名称做静态 ST 过滤（存
   在轻微前视），必须做"加/不加"敏感性对比，报告里披露。
7. 停牌日：`close` 为 NaN、`volume` 为 NaN。数据层 `qf/data.py` 已做：
   价格前向填充、`alive` 界定上市/退市区间、`is_suspended` 标记。
8. 涨跌停：无官方涨跌停价 → 用不复权前收盘 × 板块规则（主板 10%、双创 20%、
   创业板 2020-08-24 前 10%）取整到分计算，近似识别"开盘涨停不能买/跌停不能卖"。
   ST 5% 无 PIT 数据，忽略（ST 已由名称过滤剔除）。

## 数据覆盖（2026-10-14 截面）

- 有效股票约 4989 只；选股域（qf 默认过滤后）约 53% 格子（约 2800 只/日）；
- 30 万本金、单票最大 8% → 单票 2.4 万，100 股一手 → 股价上限约 240 元，
  配置 `MAX_PRICE=200` 留余量；
- 流动性下限 20 日均成交额 3000 万，30 万资金冲击可忽略。
