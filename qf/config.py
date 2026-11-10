# -*- coding: utf-8 -*-
"""Global configuration for the A-share quant system.

数据源：was_stock_data_2026-10-20.zip（2016-10-14 ~ 2026-10-14，2428 交易日 × 5281 只，
含 255 只退市股；后复权行情 + 不复权收盘 + 逐日总市值 + 行业 + 42 期财报 + 基准指数）。

所有路径均可通过环境变量 QF_* 覆盖，便于迁移。
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
PKG = Path(__file__).resolve().parent.parent          # ashare_quant/

#: 作者原始环境（历史默认值，保持本机行为逐位不变）
_ORIG_ROOT = Path(r"D:\wkf\getrich")


def _resolve_root() -> Path:
    """决定 QF_ROOT：环境变量 > 原始环境（若存在）> 代码包的上一级目录。

    迁移场景（2026-11-10 为打包分发新增）：把 ``ashare_quant/`` 和
    ``data_extract/``、``vendor/`` 放在同一个文件夹下，无需设置任何环境
    变量即可运行。原来固定写死 ``D:\\wkf\\getrich``，换台电脑就找不到数据。
    """
    env = os.environ.get("QF_ROOT")
    if env:
        return Path(env)
    try:
        if (_ORIG_ROOT / "data_extract").exists():
            return _ORIG_ROOT
    except OSError:
        pass
    return PKG.parent


ROOT = _resolve_root()
def _pick_data_raw() -> Path:
    """选择行情数据目录：环境变量 > 自动选最新的（ext 若存在且更新则用 ext）。

    背景：update_data.py 把增量数据写到 was_stock_data_ext（不改原始包）。
    为避免"用了旧数据却不自知"，这里自动比较两个目录的 panel 行数，取较新者。
    """
    env = os.environ.get("QF_DATA_RAW")
    if env:
        return Path(env)
    base = ROOT / "data_extract" / "was_stock_data"
    ext = ROOT / "data_extract" / "was_stock_data_ext"
    if not ext.exists():
        return base
    if not base.exists():
        return ext
    try:
        import numpy as _np
        n_base = int(_np.load(base / "panel_ohlcv.npz")["close"].shape[0])
        n_ext = int(_np.load(ext / "panel_ohlcv.npz")["close"].shape[0])
        return ext if n_ext >= n_base else base
    except Exception:
        return ext


DATA_RAW = _pick_data_raw()
DATA_SOURCE = "ext" if "ext" in str(DATA_RAW) else "base"


def _pick_vendor() -> Path:
    """选择因子库目录：环境变量 > ``ROOT/vendor``（完整源码）> ``PKG/vendor``（仓库自带索引）。

    分发的仓库里只有三个 ``factor_index.json``（约 1 MB），没有因子源码 ——
    ``load_catalog()`` 只读索引，因子的实际实现是 ``qf/ops.py`` 里的向量化重写，
    不依赖源码文件。所以只带索引就够跑。本机有完整 vendor 时优先用完整的。
    """
    env = os.environ.get("QF_VENDOR")
    if env:
        return Path(env)
    full = ROOT / "vendor"
    if (full / "skill-volume-stat-alpha").exists():
        return full
    return PKG / "vendor"


VENDOR = _pick_vendor()
BUILD = PKG / "build"
REPORTS = PKG / "reports"
SIGNALS = PKG / "signals"
JOURNAL = PKG / "journal"
DOCS = PKG / "docs"

for _p in (BUILD, REPORTS, SIGNALS, JOURNAL, DOCS):
    _p.mkdir(parents=True, exist_ok=True)

BUILD_FACTORS = BUILD / "factors"
BUILD_EVAL = BUILD / "eval"
BUILD_MODELS = BUILD / "models"
BUILD_RUNS = BUILD / "runs"
for _p in (BUILD_FACTORS, BUILD_EVAL, BUILD_MODELS, BUILD_RUNS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 数据参数
PANEL_FILE = DATA_RAW / "panel_ohlcv.npz"
RAWCLOSE_FILE = DATA_RAW / "raw_close.npz"
MCAP_FILE = DATA_RAW / "market_cap.npz"
INDUSTRY_FILE = DATA_RAW / "industry.npy"
INDUSTRY_CATS = DATA_RAW / "industry_cats.txt"
META_FILE = DATA_RAW / "meta.csv"
INDICES_FILE = DATA_RAW / "indices.csv"
BOND_FILE = DATA_RAW / "bond_000012.parquet"
GOLD_FILE = DATA_RAW / "gold_518880.parquet"
FUND_DIR = DATA_RAW / "fund"          # 42 期原始财报（按报告期）

# ---------------------------------------------------------------- 回测/交易参数
CAPITAL = 300_000.0        # 本金（元）
LOT = 100                  # A 股整手
MIN_COMMISSION = 5.0       # 单笔最低佣金（元）—— 30万/40只=7500元/笔，5元=6.67bp，最低收费主导
COMMISSION_RATE = 0.0001   # 佣金 万0.1 = 1bp（用户真实费率，2026-10-28 更新）
STAMP_TAX_SELL = 0.0005    # 印花税 5bp（仅卖出，2023-08-28 起）
TRANSFER_FEE = 0.00001     # 过户费 0.1bp（双边）
SLIPPAGE_BP = 40.0         # 单边滑点（bp）—— 按持仓市值估算（69%在100亿以下），2026-10-28 更新
SLIPPAGE_BP_STRESS = 25.0  # 压力测试滑点

# 组合参数
# 注：config 默认值统一为生产口径 40（此前 20 与生产脚本 --n-holdings 40 不一致，
#     直接调用 run_backtest 会静默按 20 只跑，得到 6.60% 而非 6.13%）。
N_HOLDINGS = 40            # 目标持仓数量（生产口径；上限受 MIN_WEIGHT 约束 ≤50）
MAX_WEIGHT = 0.08          # 单票最大权重
MIN_WEIGHT = 0.02          # 单票最小权重
INDUSTRY_MAX = 0.25        # 单行业最大权重
BOARD_MAX_GROWTH = 0.50    # 双创（创业板+科创板）合计最大权重
REBALANCE_DAYS = 5         # 调仓周期（交易日），5 = 周频
TURNOVER_BUFFER = 2.0      # 缓冲区：旧持仓排名在 top N*buffer 内则保留

# 选股域
MIN_LIST_DAYS = 120        # 上市满 120 交易日
MIN_PRICE = 2.0            # 低价股剔除（面值退市风险代理）
MAX_PRICE = 200.0          # 单手成本上限（300k 资金可执行性）
MIN_AMOUNT_20D = 3e7       # 20 日均成交额下限（元）
MIN_MCAP = 1.5e9           # 总市值下限（元）
MAX_MCAP = 8e11            # 超大盘剔除（弹性不足）
EXCLUDE_ST = True          # 名称含 ST/*ST/退 的剔除（注意：名称非 PIT，仅作稳健性过滤）

# 风险控制
# 个股硬止损（相对入场价，收盘触发次日开盘卖）。
# 2026-10-29 由 -0.15 放宽至 -0.30：实证见 reports/新模型探索_2026-10-29.md
#   单调改善 -15/-20/-25/-30% → CAGR 6.13/6.36/6.69/7.23%，Sharpe 0.55→0.62，DD +1.29pp；
#   连续净值切段三段全改善、8 相位中 6 个更好、5 档滑点全正。
#   逻辑：组合每 40 日已按分数重排，价格止损对同一件事判两次且更粗糙；-30% 等价"极端兜底"。
STOP_LOSS = -0.30          # 个股硬止损
DD_THROTTLE_1 = -0.10      # 组合回撤阈值 1：降杠杆至 60%
DD_LEVEL_1 = 0.60
DD_THROTTLE_2 = -0.15      # 组合回撤阈值 2：降杠杆至 40%
DD_LEVEL_2 = 0.40
TREND_MULT_LO = 0.60       # 趋势 regime：csi_all 破 MA120 时的仓位系数
TREND_MULT_HI = 0.75       # 趋势 regime：hs300 破 MA60 时的仓位系数
TREND_INDEX = "csi_all"    # 长期趋势指数（csi_all / csi1000 / csi500 / hs300）
# 波动率目标仓位开关（二阶矩择时，与趋势 regime / 回撤节流相乘叠加）：
#   目标年化波动 / 组合近期已实现波动，截断在 [VOL_MIN_EXP, VOL_MAX_EXP]。
#   0 = 关闭（默认）。原 VOL_TARGET=0.22 从未被任何代码引用，已收编为本开关。
VOL_TARGET = 0.0
VOL_LOOKBACK = 20          # 已实现波动回看窗口（交易日）
VOL_MIN_EXP = 0.50         # 缩放下限：高波动时最少保留的股票仓位
VOL_MAX_EXP = 1.00         # 缩放上限：不加杠杆
CASH_YIELD = 0.015         # 纯现金年化收益（货币基金/逆回购代理）
# 闲置资金替代资产配置（v2 起）：国债指数/黄金 ETF 组合，按"闲置现金"内部权重分配。
#   0.0 表示不用替代资产（纯现金计息，v1 口径）；默认 100% 国债。
CASH_BOND_WEIGHT = 1.00
CASH_GOLD_WEIGHT = 0.00

# 因子/模型
FWD_HORIZON = 5            # 标签：未来 5 日收益（与周频调仓匹配）
IC_MIN_ABS = 0.005         # 初筛 |RankIC| 下限
FACTOR_CORR_MAX = 0.75     # 因子相关性剪枝阈值
MAX_FACTORS = 40           # 组合使用的最大因子数
TRAIN_YEARS = 3            # 训练窗口（年）
REFIT_MONTHS = 3           # 权重再估计频率（月）
RIDGE_ALPHA = 100.0        # 岭回归正则系数（按样本量标准化）

# 研究切分（用于诚实的样本外评估）
DEV_START = "2017-01-01"
DEV_END = "2019-12-31"          # 因子族/超参数研究期
VALID_START = "2020-01-01"
VALID_END = "2021-12-31"        # 走查验证期
TEST_START = "2022-01-01"       # 最终样本外（走查从 2020 起，2022 起为完全样本外报告）
LIVE_ASOF = "2026-10-14"        # 数据最后日期（信号基于该日收盘）

# 基准
BENCHMARKS = ["hs300", "csi500", "csi1000", "csi_all"]

# 行业/板块映射
STAR_PREFIX = ("688", "689")            # 科创板
CHINEXT_PREFIX = ("300", "301")         # 创业板
CHINEXT_20PCT_DATE = "2020-08-24"       # 创业板涨跌幅 20% 起始日

RANDOM_SEED = 20261021
