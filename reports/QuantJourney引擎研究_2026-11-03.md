# QuantJourney Backtester 研究

> 日期：2026-11-03　｜　对象：`QuantJourneyOrg/quantjourney-bt`（Apache-2.0，PyPI `quantjourney-bt`）
> 方法：clone 源码逐模块读 + 把**我们自己的 A 股 ETF 数据灌进它的引擎实跑**，不看官网宣传数字。
> 落盘：`vendor/qj_bt/`（源码 + 隔离 venv）、`vendor/qj_bt/qj_out/run_cn.py`（对照脚本）、
> `build/eval/qj_cn_etf_cost_matrix.csv`（实测结果）

---

## 0. 一句话结论

**它的价值不在策略库，而在"把回测当成一份可审计的报告"这套工程与统计纪律；策略模板搬到 A 股基本不赚钱，引擎也不该替换（缺 A 股交易制度），但有 5 个模块级设计值得直接搬。**

三个最值得记住的点：

1. **再平衡被拆成 5 层可选择触发条件**（日历 / 漂移带 / 跟踪误差 / 信号变化 / 熔断 / 换手预算），而不是我们现在的"固定 N 日调一次仓"。这是它最原创的部分。
2. **它自己承认了两个坑**：默认的 walk-forward 只是"同一次全样本回测的切片"（不逐折重拟合）；旧的 PBO 实现会把教科书级过拟合判成"不过拟合"，因此已废弃。这种自我否定在开源量化项目里很少见，也说明它的统计模块是被认真审过的。
3. **成本默认值乐观得离谱**：权重模式默认 `total_bps = 1.0`（万分之一个边），且**完全没有印花税、T+1、涨跌停、整手约束**。直接拿它的结果当 A 股实盘预期会系统性高估。

---

## 1. 它是什么

| 项 | 内容 |
|---|---|
| 仓库 | `QuantJourneyOrg/quantjourney-bt`，本地 clone 版本 **0.13.0**（PyPI 0.12.5） |
| 许可 | Apache-2.0，26 MB，纯 Python |
| 定位 | "Python-native backtesting engine for reproducible portfolio research" |
| 商业结构 | 引擎开源，**数据/托管报告/PDF 事实表在云端付费**（`docs/public_scope.md` 划得很清楚） |
| 策略数 | 50 个可跑示例：**25 个权重制 + 20 个订单制 + 5 个 walk-forward/优化** |
| 依赖 | pandas / numpy / scipy / matplotlib / seaborn / pydantic / rich + 自家 `quantjourney_ti` |

代码量分布（读源码时最该看的地方）：

```
backtester/core.py                  主干：数据准备 → 信号 → 权重/订单 → 账本 → NAV → 报告
backtester/portfolio/rebalance.py   1500 行，五层再平衡政策（本项目最该抄的部分）
backtester/execution/fill_engine.py  956 行，订单生命周期与成交
backtester/execution/slippage.py     141 行，4 种滑点模型
backtester/walkforward/statistics/   DSR / PBO(已废弃) / overfit / 排名稳定性
backtester/risk/                     vol_target / risk_parity / inverse_vol / position_limit / pre_trade
```

---

## 2. 回测机制（读源码，不是读文档）

### 2.1 两套执行模式

- **weights 模式**（组合视角）：`_compute_signals()` → `_compute_weights()` → 目标权重 → 再平衡政策决定今天是否真的动 → **未调仓日权重随价格自然漂移** → 成本按成交额 × bps → NAV 递推。
- **orders 模式**（执行视角）：下显式订单（市价/限价/止损/止损限价/移动止损/括号/OCO），走 fill engine，落 trade blotter。

关键一行在 `core.py:2163`：

```python
target_weights = target_weights.shift(1).fillna(0.0)   # Shift weights forward by 1 day
```

**信号当天出、次日才生效**，这是最基本的前视防护，它做对了。

### 2.2 五层再平衡政策（最值得搬）

`RebalancePolicy` 把"什么时候调仓"拆成可叠加的多层，任一层触发就动：

| 层 | 触发 | 参数 | 我们现在的对应物 |
|---|---|---|---|
| L1 | 日历 | `D / W / BME / BQE / BYE / 21D / 自定义日期表` | 只有"每 40 个交易日" |
| L2a | **漂移带** | `drift_threshold`（如 5% 偏离才动） | **没有** |
| L2b | 跟踪误差 | `tracking_error_threshold` + 滚动 21 日 | 没有 |
| L3 | 信号变化 | `rebalance_on_signal_change` | 等价于固定日历 |
| L4 | 风险熔断 | 回撤触发清仓 + 冷却期 | 没有 |
| L5 | **换手/成本预算** | 252 日滚动预算 | 有成本目标，但不是"预算式"约束 |

执行时点支持 `OPEN / CLOSE`；`VWAP_WINDOW` 被**显式拒绝**（没有日内 VWAP 执行器就不让填），这种"宁可不给也不假装"的处理值得学。

### 2.3 成本与滑点

| 项 | 实现 |
|---|---|
| 滑点模型 | `NoSlippage` / `FixedBpsSlippage(bps)` / `VolatilitySlippage(vol_factor)`（按 bar 振幅）/ `MarketImpactSlippage`（**Almgren √(Q/ADV)**） |
| 佣金 | `ZeroCommission` / `PerShareCommission(每股)` / `FixedBpsCommission(bps)` / `TieredCommission` |
| 权重模式成本 | `FixedBpsWeightCostModel(total_bps=...)`，**默认 1.0bp** |
| 订单模式成本 | 示例里普遍 5bp 滑点 + 每股 $0.005（IB 风格）或 1bp 佣金 |
| 现金缓冲 | **默认持有 5% 现金**（`cash_buffer`） |

### 2.4 Walk-forward 与统计

- 折型：rolling / expanding / anchored，**`purge_days=5`** + `extra_pre_oos_purge_pct`（旧名 `embargo_pct`）在检验窗之前挖隔离带。
- **默认 `slice_diagnostics` 模式**：日志原文 "the OOS series is an in-sample slice"——它是把**同一次全样本回测**切成训练/检验两段分别打分，**不逐折重拟合**。要真重拟合必须传 backtester factory 走 `per_fold_refit`。这条极易被误读成"walk-forward 已通过"。
- 统计：`deflated_sharpe(..., effective_n_trials=...)`（支持有效独立试验数，处理参数高度相关）、`overfit_ratio`、`efficiency`（OOS/IS CAGR）、滚动 top-K 排名稳定性。
- **旧 PBO 实现已废弃**，注释写得很直白：它"没有 trials、没有选择、没有排名"，会把 IS 3.0 → OOS 0.05 这种教科书级过拟合判成"no overfit"。现在返回 `nan` 并警告。

---

## 3. 策略目录（50 个）

| 类别 | 数量 | 内容 |
|---|---|---|
| 权重制 | 25 | 趋势（SMA/MACD/Bollinger）、动量（时序/截面）、反转、配对（协整/比价 z）、风险平价、波动率目标、防御轮动、熔断、TE+成本闸门、漂移带、FX 与连续合约代理 |
| 订单制 | 20 | 市价/限价/突破止损/止损限价/移动止损/括号/OCO、日内 1m~1h、月度轮动、整手下单、期货 ATR 整合约 |
| WF/优化 | 5 | 滚动、扩张、锚定+purge、网格搜索、Optuna TPE |

诚实度加分项（README 自己写的边界）：
- W13–W16 的多空**不建模融券成本/借券费**；
- W23–W25 的 FX/期货是**价格收益代理**，不算手数、不加保证金、不做换月；
- O19–O20 用了合约乘数，但**不校验保证金与购买力**。

---

## 4. 实测：把我们的 A 股 ETF 灌进去

**数据**：我们自己的 `build/etf/etf_panel.npz` → 三只宽基 ETF `510300 / 510500 / 512100`，
取三只都有数据的区间 **2016-11-04 ~ 2026-11-02（2425 个交易日）**，初始 100 万，`adj_close = close`。
脚本：`vendor/qj_bt/qj_out/run_cn.py`（复刻它的 W01/W02/W04/W17 规则，只换标的与区间）。

### 4.1 结果（CAGR / Sharpe / MaxDD）

| 策略 | QJ 默认 1bp | 现实 ETF 12.5bp/边 | 保守 25bp/边 |
|---|---|---|---|
| SMA 趋势 · **日频**再平衡 | 1.86% / 0.21 / −26.8% | 1.67% | 1.47% |
| SMA 趋势 · 月末 + 5% 漂移带 | 1.83% / 0.20 / −26.7% | 1.69% | 1.53% |
| 双动量 · 季度（12M 前 2 且为正） | **−2.25%** / −0.05 / −37.4% | −2.45% | −2.66% |
| 波动率目标趋势 · 月度（15% 目标） | 2.24% / 0.22 / −32.1% | 1.96% | 1.65% |

### 4.2 三个结论

1. **引擎没在骗人**：喂真实数据，它给出的是"难看但可信"的数字（1.5~2.2% 年化、回撤 −27%~−33%），没有偷偷放大。这点比很多开源回测框架干净。
2. **它的慢趋势/动量模板在 A 股宽基上不赚钱**：双动量季度甚至 −2.25%。这与本项目已有结论方向一致（A 股个股是反转、板块才动量、宽基慢趋势长期是震荡），说明**问题在规则不在引擎**。我们自己 ETF 腿用的是"突破 55 日新高 + 最多持 2 只 + 低暴露"，规则完全不同，**数字不可直接对齐**，这里只比摩擦处理，不比策略优劣。
3. **成本敏感度取决于换手，不取决于频率**：1bp→25bp，日频掉 0.39pp、月度掉 0.30pp——几乎一样。因为慢信号本身换手就低，"降频"不等于"省钱"，**降换手才省钱**。这条正好印证我们给低换手侧设 49.27bp 成本目标、给组合设换手预算的做法。

---

## 5. 对我们致命的缺口（不能靠它做 A 股结论）

| 缺口 | 影响 |
|---|---|
| **无 T+1** | 当日买入可当日卖出，日内策略与"止损当天生效"全部失真 |
| **无涨跌停** | 涨停买不进、跌停卖不掉，突破类策略（我们 ETF 腿的核心）会被系统性高估 |
| **无印花税** | 个股卖出 10bp 显式成本缺失（ETF 免印花税，所以 ETF 腿影响小、**个股腿影响大**） |
| **无整手（100 股/份）** | 小资金 + 高价股时权重无法精确落地，成本与偏离都会被低估 |
| **无 ST/停牌/退市/上市未满 N 日** | 我们的选池判据有一整套这类过滤，它没有 |
| **复权口径单一** | 只接受"全部 OHLC 都在复权价基准上"，我们的 `raw_close`（真实成交价）口径用不上，涨跌停判定会错 |
| **计价货币 USD** | `base_currency` 默认 USD，且本地输入只认 USD 日线股票；CNY 与汇率、跨境 ETF 溢价都不涉及 |
| **数据源** | 开源侧只有 Yahoo/示例/自备 CSV，美股为主；A 股要自己灌 |

结论：**不能用它给 A 股出结论**，只能当"方法论参考 + 交叉验证工具"。

---

## 6. 建议：搬这 5 件，别搬这 3 件

### 值得搬（按性价比排序）

1. **L2a 漂移带 + L5 换手预算**：把"每 40 日调仓"改成"每 40 日检查、偏离 >X% 才动、且受 252 日换手预算约束"。改动小、直接降成本，且不改变信号本身。
2. **DSR + effective_n_trials**：我们已经在做 8 相位带和 bootstrap CI，但**没有对"试了多少个方案"做多重检验折扣**。今天这份 `价值维度结项` 里 4 个方案选 1 个、`族预算/族上限` 又试了若干组——这些都该记进试验数，用 DSR 打折后再宣布"上线"。
3. **purge/embargo 显式化**：我们的 train/val/OOS 是硬切分，但没有"切点前挖 5 日隔离带"的概念。因子是 5 日/10 日标签，标签窗口跨切点就是轻度泄漏，值得补一行代码。
4. **订单级执行模拟**（哪怕只做简版）：现在我们的 ETF 腿按收盘价成交。至少加"次日开盘 + 滑点 + 成交额参与率上限（如 ≤5% ADV）"，这比继续调因子更能改善真实性。
5. **run metadata 审计包**：每次回测把"数据版本 / 参数 / 成本假设 / 代码 commit"写进产物。我们报告里的口径漂移（比如这次发现的两个红测）本质就是缺这个。

### 不要搬

1. **策略模板**：慢趋势、双动量、配对、风险平价这些在 A 股宽基上实测亏钱，且没有涨跌停/印花税处理，搬过来只会得到漂亮的假数字。
2. **默认成本参数**（1bp 佣金 + 5bp 滑点、无税）：比我们现在的 49.27bp 目标乐观一个数量级。
3. **`slice_diagnostics` 当 walk-forward 用**：它不重拟合，用它宣布"样本外通过"等于自欺。

---

## 7. 复现路径

```bash
# 1) 源码（已 clone 到 vendor/qj_bt，26MB，未装进项目 venv）
cd vendor/qj_bt
py -3.13 -m venv .venv
PYTHONPATH= .venv/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
  "pandas>=2.0,<3" "numpy<3" httpx requests quantjourney_ti matplotlib seaborn scipy \
  psutil python-dateutil "pydantic>=2" questionary rich

# 2) 我们自己的三只 ETF 已转成它的本地格式
#    vendor/qj_bt/_local/data/cn_etf_3.csv（7275 行 = 2425 日 × 3 只）
PYTHONPATH=. .venv/Scripts/python.exe strategy.py example_weights_01_sma_daily \
  --data-file _local/data/cn_etf_3.csv --csv-config --quiet --output qj_out/w01

# 3) 成本敏感性矩阵（12 组 = 4 策略 × 3 档成本）
PYTHONPATH=. .venv/Scripts/python.exe qj_out/run_cn.py
```

踩坑记录：
- 只装了 pandas 3.x 会炸，**必须锁 `pandas<3`**。
- pip 直连 PyPI 在本机几乎不动，换清华镜像后 22 MB/s。
- `--sample-data` 是**合成数据**（正弦 + 第 1150 天人为造的一次暴跌），只用于验证安装，**不能拿它的数字当市场证据**。
- 除 `example_weights_01_sma_daily` 外，其它示例**没有 opt-in `csv_config`**，用 `--csv-config` 会报 `This strategy does not opt in to CSV universe/period replacement`，必须自己写构造参数（`run_cn.py` 就是这么做的）。
- 报告中文在 matplotlib 里会丢字形（Segoe UI 缺 CJK），不影响数字。
