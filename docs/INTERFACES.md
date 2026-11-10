# 接口契约（INTERFACES）— 所有成员必须遵守

Python 环境：`D:\wkf\getrich\.venv\Scripts\python.exe`（numpy 2.5 / pandas 3.0 / pyarrow /
matplotlib / scipy / sklearn / lightgbm / statsmodels）。项目根目录 `D:\wkf\getrich\ashare_quant`。

## 1. 已完成的核心模块（Lead owner，只读使用）

### `qf/config.py`
全部常量：路径 `C.DATA_RAW / C.BUILD / C.REPORTS / C.SIGNALS / C.JOURNAL`，
交易参数 `C.CAPITAL=300000`、`C.COMMISSION_RATE`、`C.STAMP_TAX_SELL`、`C.TRANSFER_FEE`、
`C.SLIPPAGE_BP`、`C.LOT=100`、`C.MIN_COMMISSION=5`；
组合参数 `C.N_HOLDINGS=20`、`C.MAX_WEIGHT=0.08`、`C.MIN_WEIGHT=0.02`、
`C.INDUSTRY_MAX=0.25`、`C.BOARD_MAX_GROWTH=0.50`、`C.REBALANCE_DAYS=5`、
`C.TURNOVER_BUFFER=2.0`、风控 `C.STOP_LOSS=-0.15`、`C.DD_THROTTLE_1/2`、`C.VOL_TARGET`；
研究切分 `C.DEV_START/END`、`C.VALID_START/END`、`C.TEST_START`。

### `qf/data.py::MarketData`（`qf.data.load_market(verbose=True)`）
单例数据对象，字段全部是 numpy 数组 `(T=2428, N=5281)`，日期 `md.dates: DatetimeIndex`，
代码 `md.codes: np.ndarray[str]`（**未排序**，用 `md.code_index(code)` 查位置）。

| 字段 | 类型 | 含义 |
|---|---|---|
| `md.close/open/high/low` | float32 (T,N) | 后复权价（停牌 ffill，上市前/退市后 NaN）。**因子计算用** |
| `md.price / price_open / price_high / price_low` | float32 (T,N) | 常数缩放后复权价，最后有效日≈实际价。**回测/整手/止损用** |
| `md.raw_close / raw_open` | float32 (T,N) | 不复权价（涨跌停判断、参考） |
| `md.volume / amount` | float32 (T,N) | 停牌为 0；`amount` 单位元 |
| `md.mcap / logmcap` | float32 (T,N) | 逐日总市值（元）/对数 |
| `md.industry` | int16 (T,N) | 行业索引，-1 未知 |
| `md.industry_cats` | list[str] | 行业名（45） |
| `md.alive` | bool | 上市且存续（首末有效 bar 之间） |
| `md.is_suspended` | bool | alive 但当日无 bar |
| `md.list_days` | int32 | 已上市交易日数 |
| `md.limit_up / limit_down` | float32 | 涨停价/跌停价（不复权口径） |
| `md.limit_up_at_open / limit_dn_at_open` | bool | 开盘即涨停/跌停 |
| `md.eligible` | bool | 选股域（t 日收盘可参与打分的口径） |
| `md.buyable` | bool | t 日开盘可买（选股域 + 未涨停 + 有开盘价） |
| `md.sellable` | bool | t 日开盘可卖（未跌停 + 未停牌） |
| `md.bench` | DataFrame(index=dates) | 基准日收益：hs300/csi500/csi1000/csi_all/all_eq |
| `md.ret_daily` | float32 (T,N) | 日收益（=后复权总收益） |
| `md.fwd_ret(h)` / `md.fwd_open_ret(h)` | float32 | t 收盘→t+h 收盘 / t+1 开盘→t+1+h 开盘 收益 |
| `md.first_valid / last_valid` | int (N,) | 首末有效 bar 行号 |
| `md.board` | object (N,) | `MAIN` / `STAR`（688/689）/ `CHINEXT`（300/301） |
| `md.names / md.meta / md.is_delisted / md.list_date / md.delist_date` | | 元信息 |

### `qf/ops.py` — 向量化算子（numpy 面板版 PandaFactor）
`shift, pct_change, delay, roll_sum, roll_mean, roll_std, roll_min, roll_max,
roll_skew, roll_kurt, roll_cov, roll_corr, roll_slope, roll_rank_last, ema, ema_alpha,
cs_rank, cs_zscore, tanh_clip`。输入输出 `(T,N) float32`，NaN 语义与 pandas 对齐。

### `qf/factors.py`
- `load_catalog() -> pd.DataFrame`：800 行因子目录，列：
  `lib, factor_id, slug, name_cn, name_en, category_cn, category_en, base, transform,
  window, norm, params(dict), path, factor_key`。**factor_key 形如 `volume-stat:R020`**，
  全系统唯一主键。
- `FactorInputs(md)`：因子计算输入（close/open/high/low 用后复权，volume/amount 原始口径）。
- `compute_factor(inp, entry) -> np.ndarray (T,N)`：按 catalog 一行算因子。
- `_base_signal` / `_transform_signal`：33 个 base、8 个 transform 的实现。
- 等价性自检：`scripts/check_factor_equiv.py`（与 vendor 中 reference factor.py 对比）。

## 2. factor-lab 交付物（Task: 因子库构建与评估）

### 2.1 `qf/evaluate.py`（factor-lab 写）
建议函数（可调整，但必须覆盖语义）：
```python
def rank_ic(factor, label, mask, rows=None) -> np.ndarray  # (n_rows,) 逐期 Spearman IC
def quantile_returns(factor, label, mask, rows, n_q=5) -> np.ndarray  # (n_rows,) 或各组均值
def factor_metrics(md, factor, label, rows, name) -> dict
def early_screen(md, cat, inp, rows_screen, label) -> pd.DataFrame
```
**口径（必须严格遵守）**：
- 样本行 `rows`：交易日的行号（int），只在这些日计算横截面统计；`mask` 为当日选股域
  （用 `md.eligible[t]`，且 factor/label 有限）。
- 标签：`label = md.fwd_open_ret(C.FWD_HORIZON)`（t+1 开盘买、t+6 开盘卖，5 日收益）；
  禁止用 `fwd_ret` 当日收盘起算做回测标签（研究 IC 可额外报告 1/10/20 日）。
- Rank IC：每日横截面 Spearman（rank 后 Pearson），只在有效股票数 ≥ 100 的日子计算。
- ICIR = mean(IC)/std(IC)，t 值 = mean/std×sqrt(n)；正 IC 占比。
- 分组：按因子值当日横截面排序分 5 组（等权），组收益均值（5 日）；多空 Q5-Q1。
- 换手：Top20% 组相邻调仓日（每周）成员重叠度的变化 `1 - |S_t ∩ S_{t-1}| / |S_{t-1}|` 均值。
- 覆盖率：`finite(factor) & md.eligible` / `md.eligible` 的均值。
- 稳健性：分年度 IC（2017…2026）均值；`icon` 一致性（正 IC 年份占比）。
- 全部函数禁止未来数据：只用 `<=t` 的因子值（因子本身已是 t 时点），标签用未来收益
  仅用于评估，不进入因子计算。

### 2.2 `scripts/build_factors.py`（factor-lab 写）
流程：
1. `md = load_market()`；`inp = FactorInputs(md)`；`cat = load_catalog()`。
2. 选股域 `mask_all = md.eligible`（世界期 2017-01-03 起；`md.dates >= DEV_START`）。
3. 逐 (base, window) 组计算（`factors.iter_groups`），组内 8 个 transform 一次算完
   （利用缓存避免重复算 base）。
4. **两阶段**：
   - Stage-1 初筛：`rows_screen` = 2017-01 起每隔 `C.REBALANCE_DAYS` 的交易日
     （约 490 个）；对每个因子算 IC/ICIR/覆盖率/多空/换手，写入
     `build/eval/factor_metrics_screen.parquet`（800 行）。
   - Stage-2 细分：取初筛按 `|ICIR|` 排名前 200 的因子（并保证每个 lib/每类 category
     至少若干），在**全部每周行**上复核并分年度统计，写
     `build/eval/factor_metrics.parquet`；同时对这 200 个因子保存周频面板
     `build/factors/weekly_top.npz`（键 = factor_key，值 = float32 (n_weeks, N)，
     NaN 允许）+ `build/factors/weekly_meta.npz`（`dates`=每周行号对应的日期、
     `label` = 同一周的 `fwd_open_ret(5)`、`eligible` 周频掩码）。
5. 每个因子写因子卡 `journal/factor_cards/<factor_key 把 : 换成 _>.json`：
   `{factor_key, lib, factor_id, name_cn, category, base, transform, window,
     ic_5d, icir_5d, tstat, ic_pos_ratio, ic_by_year{dict}, turnover_top20,
     coverage, ls_mean_5d, monotonicity, corr_max_cluster, created_at}`。
6. `journal/INDEX.md`：按 lib/category 汇总的因子库索引（表格）。
7. `journal/runs.jsonl` 追加一条实验记录（见 §4 的 journal 规范）。
8. 打印计时与进度；`build/factors/catalog.parquet` 保存 `load_catalog()` 副本。

估计耗时：因子计算 ~10-20 分钟；Stage-1 评估 ~10 分钟；Stage-2 ~10 分钟。
**必须用单进程串行**（机器 4 核，其他人也在跑）。中间结果落盘，支持断点续跑。

### 2.3 `scripts/evaluate_factors.py`（factor-lab 写）
对 Stage-2 的 200 个候选因子生成：
- `build/eval/factor_corr.npz`：候选因子在周频行的 Spearman 相关矩阵（或按族聚合的
  相关矩阵，n×n 最多 200×200，float32）。
- `build/eval/family_stats.parquet`：按 lib / category_en / base / transform 聚合的
  ICIR 均值/中位数、覆盖率、正 IC 占比、最优因子。
- `build/eval/review_summary.json`：factor-review Layer1/2 需要的量化输入：
  `n_factors, n_pass(|ICIR|>阈值且覆盖率>90%), accept_rate, ic_distribution(分位),
   turnover_distribution, family_coverage, top20, bottom20, cluster_representatives`。
- `reports/因子库复盘_数据附录.md`（可选，供 Lead 引用）。

## 3. backtest-engine 交付物（Task: 组合构建 + 执行模拟 + 回测引擎）

### 3.1 `qf/execution.py`
```python
@dataclass
class Position:
    code: int          # md.codes 的下标 j
    shares: int        # 股数（缩放后复权价空间，整手）
    entry_price: float # 买入成交价（md.price_open 口径，含滑点）
    entry_date: int    # 交易日行号
    buy_date: int

@dataclass
class Trade:
    date: int; code: int; side: str  # 'buy'/'sell'
    shares: int; price: float; amount: float
    commission: float; stamp: float; transfer: float; slip: float
    reason: str  # 'rebalance'/'stop_loss'/'liquidate'

class ExecutionEngine:
    def __init__(self, md, cfg=C):
        self.cash = cfg.CAPITAL; self.positions = {}; ...
    def nav(self, t: int) -> float:  # cash + Σ shares × md.price[t,j]
    def execute(self, t: int, target_weights: dict[int,float], reasons=None) -> list[Trade]:
        """在 t 日**开盘**执行。先卖出超配/清仓（受 sellable 约束），再买入低配
        （受 buyable 约束、现金约束）。整手取整；成本：佣金 max(5, 2.5bp×额)、
        卖出印花税 5bp、过户费 0.1bp、滑点 15bp（买价上浮/卖价下调）。
        无法成交的买单（涨停/停牌/现金不足/整手）顺延到下一交易日（由 backtest 循环重试）。
        返回 trades 列表并更新 self.cash / self.positions。"""
    def stop_loss_sells(self, t: int, cfg) -> None:
        """收盘后检查：md.price[t,j]/entry_price-1 <= STOP_LOSS 的持仓标记，
        下一交易日开盘卖出（由 backtest 处理）。"""
```
关键约束：`md.price_open` 可能为 NaN（停牌）→ 不可交易；`md.buyable/sellable` 已含
涨跌停/停牌/选股域；买入需要 `md.eligible[t]` 之外是否允许旧持仓（是，允许继续持有）。
现金不足时按权重比例缩减买量（或按目标顺序），保底留 1% 现金。

### 3.2 `qf/portfolio.py`
```python
def build_target(md, t: int, scores: np.ndarray, prev_weights: dict[int,float],
                 cfg=C, n_holdings=None) -> dict[int,float]:
    """t 日收盘打分 → 目标权重字典 {j: w}，w 之和 ≤ 1（余下为现金）。
    规则：
    1. 候选池 = md.eligible[t] & isfinite(scores)。
    2. 排名（降序）；缓冲区 TURNOVER_BUFFER：prev_weights 中的股票若排名
       < n_holdings*C.TURNOVER_BUFFER 则优先保留。
    3. 行业约束：单行业累计权重 ≤ C.INDUSTRY_MAX（未知行业单独一组）；
       双创累计 ≤ C.BOARD_MAX_GROWTH。
    4. 单票 ≤ C.MAX_WEIGHT；每只 ≥ C.MIN_WEIGHT（达不到则跳过该票）。
    5. 权重可按分数斜率/等权两种模式；缺省等权（更稳健）。
    6. 返回 dict；不做整手取整（在 execution 做）。"""
```
### 3.3 `qf/backtest.py`
```python
@dataclass
class BacktestResult:
    nav: pd.Series            # 日净值（index=md.dates[start:end]）
    daily_ret: pd.Series
    weights: pd.DataFrame     # 调仓日目标权重（列=code）
    trades: pd.DataFrame      # 所有成交
    holdings: pd.DataFrame    # 调仓后持仓快照
    exposures: pd.DataFrame   # 行业/板块/市值/现金暴露
    metrics: dict
    config: dict

def run_backtest(md, scores: np.ndarray, start_row: int, end_row: int,
                 cfg=C, risk_overlay: bool = True, rebalance_days=None,
                 stop_loss=True, verbose=True,
                 cash_asset_returns: np.ndarray | None = None) -> BacktestResult:
    """scores: (T,N) 复合因子分值（t 日收盘可用，NaN 表示无信号）。
    流程：
    - 每 C.REBALANCE_DAYS 个交易日为一个调仓日（从 start_row 起固定节奏）。
    - 调仓日 t 收盘后：过滤 → 目标权重 → 生成订单；t+1 开盘执行（重试机制：
      未成交订单挂到下一个交易日开盘，最多 N 次；仍不成交则放弃）。
    - 每日：先执行昨日订单（开盘），再用 md.price 收盘价更新净值
      （净值 = cash + Σ shares×price[t]）。
    - 风控 overlay（risk_overlay=True）：
      a) 趋势：csi_all 收盘 > MA120 且 hs300 > MA60 为满仓；否则根据
         (csi_all>MA120?1:0.6)*(hs300>MA60?1:0.75) 缩放目标权重（记为 regime 系数）；
      b) 回撤节流：策略自身从历史峰值回撤 > DD_THROTTLE_1 → 目标暴露 ≤ DD_LEVEL_1；
         > DD_THROTTLE_2 → ≤ DD_LEVEL_2；恢复需回撤收回一半；
      c) 个股止损：收盘价相对 entry_price ≤ STOP_LOSS → 下一开盘卖出（stop_loss）。
    - 现金计息：cash_asset_returns 为 None 时按 C.CASH_YIELD 日计息（1.5%/365）；
      否则按传入的日收益数组计息（v2 起用 `load_cash_asset_returns(md, 0.8, 0.2)`
      = 80% 国债指数 + 20% 黄金 ETF 的混合日收益，见 reports/回测报告.md §4.6）。
    - 基准净值同时输出（md.bench）。
    """

def load_cash_asset_returns(md, bond_w=0.8, gold_w=0.2, cfg=C) -> np.ndarray:
    """闲置资金替代资产的日收益 (T,)：国债指数/黄金 ETF 按权重混合，对齐 md.dates（ffill）。
    读取 cfg.BOND_FILE / cfg.GOLD_FILE；缺失日回退 CASH_YIELD/365。"""
```
性能：循环 ~2400 个交易日，每日只处理持仓（≤几十只），必须快（<60s/次）。

### 3.4 `qf/metrics.py`
```python
def performance(nav, bench_returns=None, rf=0.0) -> dict
   # 年化收益(CAGR)、年化波动、Sharpe、Sortino、最大回撤、最大回撤持续天数、
   # Calmar、月胜率、日胜率、偏度、峰度、beta/alpha(vs 各基准)、跟踪误差、信息比率
def yearly_returns(nav) -> pd.Series
def monthly_returns_table(nav) -> pd.DataFrame  # index=year, columns=month + 'YTD'
def drawdown_series(nav) -> pd.Series
def rolling_metrics(nav, window=252) -> pd.DataFrame
def stress_periods(nav) -> pd.DataFrame  # 预设区间：2018 全年、2020-01~03、2022 全年、
   # 2024-01~02（小微盘崩）、2024-09-24 起 10 日、2025-04（关税）、最新 6 个月
```

### 3.5 测试（backtest-engine 写）
- `tests/test_execution.py`：用构造的小面板验证：整手取整、最低佣金、印花税卖出、
  涨停不能买/跌停不能卖、停牌不能交易、T+1（当天买不能当天卖）、现金不为负。
- `tests/test_backtest.py`：用真实 md + 合成分数（如 `-md.fwd... ` 禁止！用
  `ops.cs_rank(-md.ret_daily 累积20日)` 这种只依赖过去的简单信号）跑 2017-2026，
  检查：净值有限、无未来函数（信号 shift 1 后重跑结果应不同/滞后）、交易成本为正、
  调仓后权重满足约束（单票/行业/双创/持仓数）。
- 用 `python -m pytest tests -q` 通过。

## 4. journal 规范（factor-lab 与 Lead 共用）
`journal/runs.jsonl`：每行一个 JSON：
```json
{"run_id":"2026-10-21T01:30:00_lab_build_factors","stage":"factor_build",
 "status":"ok|failed","n_factors":800,"n_candidates":200,"elapsed_sec":900,
 "artifacts":["build/eval/factor_metrics.parquet"], "note":"..."}
```
`journal/best.json`：`{"best_composite_run": "...", "metrics": {...}, "updated_at": "..."}`。

## 5. 命名/口径约定
- 股票主键统一用 `md.codes[j]`（6 位字符串）和下标 `j`；落盘 CSV 用 code 字符串。
- 所有"未来收益"以 `md.fwd_open_ret(5)` 为准（t+1 开盘到 t+6 开盘）。
- 周频行：`rows_weekly = np.arange(start, T, C.REBALANCE_DAYS)`（从 2017-01-03 起）。
- 所有随机过程固定 `C.RANDOM_SEED`。
- 报告中的年化 = (1+累计)^(252/交易日数)-1；最大回撤用日净值；Calmar = CAGR/|MaxDD|。
