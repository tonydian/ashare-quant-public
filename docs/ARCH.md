# A 股量化系统 — 架构与分工

> 本文件是团队协作的单一事实来源（single source of truth）。所有成员开工前先读本文件 +
> `docs/INTERFACES.md` + `docs/DATA_NOTES.md`。

## 0. 任务目标（用户）

- 用 `PandaFactor` 算子体系 + QuantSkills 因子库（volume-stat-alpha 216 个、
  directional-alpha 296 个、risk-pattern-alpha 288 个，共 800 个）搭建 A 股量化系统；
- 本机数据包：`D:\wkf\getrich\data_extract\was_stock_data`（2016-10-14 ~ 2026-10-14，
  2428 交易日 × 5281 只含 255 只退市股，后复权 OHLCV + 不复权收盘 + 逐日市值 + 行业 +
  42 期财报 + 基准指数）；
- 本金 30 万人民币，有双创权限；目标：高年化复合收益 + 足够低的回撤；
- 用 factor-review 技能对因子库做三层复盘；
- 交付：可运行代码 + 回测报告 + 因子库复盘 + 实盘信号（下一交易日目标持仓/订单）。

## 1. 系统架构（数据流）

```
was_stock_data (npz/parquet/csv)
        │
        ▼
qf/data.py     MarketData：后复权面板 / 不复权价 / 市值 / 行业 / alive / 涨跌停 /
        │                  buyable / sellable / eligible / 前向收益 / 基准
        ▼
qf/ops.py      向量化算子（PandaFactor 语义的 numpy 面板版：DELAY/STDDEV/
        │      CORRELATION/SUM/RANK/TS_RANK/EMA/...）
        ▼
qf/factors.py  800 因子 = 33 个 base × 8 个 transform × 4 个 window（来自三个因子库
        │      的 factor_index.json；与 reference factor.py 逐点对齐，见
        │      scripts/check_factor_equiv.py）
        ▼
scripts/build_factors.py + qf/evaluate.py
        │      → build/factors/catalog.parquet        （因子目录）
        │      → build/eval/factor_metrics.parquet    （单因子 IC/ICIR/分组/换手）
        │      → build/factors/weekly_top.npz         （候选因子周频面板）
        │      → journal/factor_cards/<key>.json      （因子卡）
        │      → build/eval/factor_corr.npz           （候选因子相关性）
        ▼
qf/model.py    因子选择（ICIR + 相关性剪枝 + 因子族配额）→ 复合打分
        │      （滚动岭回归 / ICIR 加权；只用 t 时点之前的数据，walk-forward）
        ▼
qf/portfolio.py + qf/execution.py + qf/backtest.py
        │      选股域过滤 → Top N → 行业/板块/单票约束 → 整手取整 →
        │      T+1 开盘成交（涨跌停/停牌不可成交）→ 佣金/印花税/滑点 →
        │      风控 overlay（趋势/宽度/回撤/波动目标/个股止损）
        ▼
build/runs/<run_id>/  净值、权重、成交、持仓、指标、图
        ▼
reports/   回测报告 / 因子库复盘 / 风险说明 / 实盘信号
signals/   最新目标持仓 + 订单（30 万整手）
```

## 2. 目录与写入边界（重要：多人共享文件系统）

| 路径 | Owner | 说明 |
|---|---|---|
| `qf/config.py`, `qf/data.py`, `qf/ops.py`, `qf/factors.py` | Lead | 已基本完成，改动须经 Lead |
| `qf/evaluate.py`, `scripts/build_factors.py`, `scripts/evaluate_factors.py`, `build/factors/`, `build/eval/`, `journal/` | factor-lab | 因子计算 + 评估 + 因子卡 |
| `qf/portfolio.py`, `qf/execution.py`, `qf/backtest.py`, `qf/metrics.py`, `tests/test_backtest*.py` | backtest-engine | 组合构建 + 执行模拟 + 回测引擎 |
| `qf/model.py`, `scripts/run_research.py`, `scripts/run_backtest.py`, `build/models/`, `build/runs/` | Lead | 因子选择/权重/走查驱动 |
| `qf/report.py`, `scripts/make_report.py`, `scripts/daily_signal.py`, `reports/`, `signals/` | Lead（后续可交 verifier 复核） | 报告 + 实盘信号 |
| `tests/` | 各自 | 只加自己的测试文件，不改别人的 |
| `docs/` | Lead | 架构/接口/数据说明 |

**原则**：不要改别人 owner 的文件；需要接口变动先发消息。每个任务在共享任务板登记
write scope，完成后 complete。

## 3. 方法论铁律（防止过拟合/前视）

1. **三段切分**：研究期 `DEV_START~DEV_END`（2017-2019）用于因子族判断与超参数选择；
   验证期 2020-2021；最终样本外 2022-2026-10。最终报告以 2022 起为主，同时给出
   2020 起的走查结果与逐年表现。
2. **Walk-forward**：每次再估计（每 3 个月）只使用当时之前的数据；因子选择也用滚动窗口
   IC（至少 2 年），不允许用全样本 IC 挑因子后回测同一段。
3. **执行假设**：t 日收盘生成信号 → t+1 开盘成交；停牌不可交易；开盘涨停不能买、
   开盘跌停不能卖；买入后 T+1 才能卖；整手 100 股；本金 30 万。
4. **成本**：佣金 2.5bp（最低 5 元/笔）、印花税（2023-08-28 前 10bp / 之后 5bp，卖出）、
   过户费 0.1bp、滑点 15bp/边；另做 25bp 滑点压力测试。
5. **禁止生存偏差**：退市股保留（alive 由首末有效 bar 界定）；ST 用名称口径过滤是
   非严格 PIT 的妥协，必须做「加/不加 ST 过滤」敏感性对比。
6. **禁止未来函数**：因子只用 t 及之前数据；标签用 t+1 开盘起算（fwd_open_ret）；
   财报已按「法定披露截止日 +15 天」滞后实现 PIT 基本面因子（v2 实验，见 reports/回测报告.md §4.7）。
7. **报告必须诚实**：不挑最优参数反复回测；给出参数敏感性、压力期表现、失效风险。
   闲置现金替代资产（国债/黄金）属于 portfolio 层配置，需单独披露（§4.6）。

## 4. 交付清单

- 代码：`ashare_quant/`（可复现：`scripts/run_pipeline.py` 全流程）
- `reports/回测报告.md`（含图表、逐年/逐月、压力测试、敏感性、与基准对比）
- `reports/因子库复盘报告.md`（factor-review 三层结构：量化盘点/结构分析/研究建议）
- `reports/系统说明与风险声明.md`
- `signals/latest_target.csv` + `signals/latest_orders.csv`（30 万整手订单）
- `reports/回测指标.xlsx`（指标、因子卡、持仓、信号的汇总工作簿）

## 5. 时间预算与优先级

P0（必须先跑通）：数据层 → 因子引擎（已验证等价）→ 800 因子计算评估 → 复合打分 →
回测引擎 → walk-forward 回测 → 指标/图 → 报告 → 实盘信号。
P1（已完成/有时间就做）：财报补充因子族（15 个，已完成 v2 实验并保留为研究分支）、
闲置现金替代资产（80% 国债 + 20% 黄金，已进入主回测）、
参数敏感性网格、因子衰减监控脚本、压力测试情景。
P2：实盘接入说明（券商/QMT）、日更脚本（update_data.py 已提供行情增量更新）。

## 6. 协作约定

- 每完成一个阶段，把关键产物路径与数字写到共享任务板对应任务里（一句话结论 + 文件路径）。
- 数字口径必须与 `docs/INTERFACES.md` 一致；不确定时先读代码再问。
- 不要长时间占用 4 核机器：重计算（因子批量计算、回测网格）串行跑，跑之前确认没有
  其他人在跑重任务（在任务板留言）。
- Python 环境：`D:\wkf\getrich\.venv\Scripts\python.exe`（已装 numpy/pandas/pyarrow/
  matplotlib/scipy/sklearn/lightgbm/statsmodels/openpyxl/docx/pptx）。
- 数据只读，不要修改 `data_extract/` 与 `vendor/`。
