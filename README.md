# A 股多因子量化系统

> 面向 30 万人民币个人账户（含创业板/科创板权限）的 A 股多因子量化**研究**与信号生成系统。
> 生产模型出信号不需要联网、不需要 GPU，在一台普通笔记本上约 **100 秒**跑完。

**本项目仅供量化研究与教学，历史回测不代表未来收益，不构成投资建议。**

---

## 这是什么

一套完整的 A 股横截面选股系统，从原始行情到可执行的整手委托单，全链路都可复现：

| 环节 | 做法 |
|---|---|
| 因子库 | 800 个 OHLCV 因子（QuantSkills 三库）+ 自建中期反转族 + 换手族 + 价值因子 |
| 选池 | DEV 期（2017-2019）ICIR 初筛 → 每类 ≤3、池内两两 \|ρ\|≤0.70 → **21 个因子** |
| 权重 | 每季度滚动 3 年重估（ICIR 与岭回归各 50%） |
| 组合 | **40 只等权**、40 日调仓、行业 ≤25%、单票 ≤8%、每期最多换 6 只 |
| 风控 | 趋势 regime（csi_all MA120 + hs300 MA60）+ 回撤节流（10%/15%）+ 个股 −30% 止损 |
| 执行 | T+1 开盘、整手 100 股、佣金 2.5bp（最低 5 元）+ 印花税 + 过户费 + 滑点 40bp |

组合层是**四腿结构**：主策略 40%（个股）+ 低换手月度 30%（个股）+ 宽基 ETF 趋势 20% + 可转债 10%。
四条腿的权重经 ERC / 逆波动 / 最小方差 / 最大 Sharpe 四种优化全部验证过，
**都不如固定权重**（根因是四条腿的 Sharpe 排名前后秩相关 −0.2，等于随机），所以保持固定。

### 回测表现（2020-01-08 ~ 2026-11-09，主策略腿）

| 指标 | 全期 | 验证期 2020-2021 | 样本外 2022-2026 |
|---|---|---|---|
| 年化复合收益 | **8.78%** | 16.19% | 5.18% |
| 年化波动 | 10.44% | 9.85% | 10.58% |
| Sharpe (rf=1.5%) | **0.715** | 1.42 | 0.39 |
| 最大回撤 | −14.59% | −7.80% | −14.61% |
| 年单边换手 | 0.76x | — | — |

> ⚠️ **必读的口径说明**：上表是**生产相位**（`rebalance_offset=10`）的单点结果。40 日调仓与周频打分行
> 对齐后共 8 个相位，同一份分数跨相位 CAGR 从 **5.46% 铺到 9.00%**（中位 6.60%），生产相位排第 3/8。
> **对外表述应给区间或中位数，不要只报一个点。** 明细见 `build/eval/phase_band.json`（需自行生成）。

---

## 快速开始

### 前置条件

- Python **3.11 ~ 3.13**（开发验证于 3.13.9 / Windows 11；macOS、Linux 同样可用）
- 约 **2 GB** 磁盘（代码 5 MB + 数据 263 MB + 运行产物若干）
- 能访问公网（仅首次拉数据时需要，之后出信号可完全离线）

### 六步跑通

```bash
# 1. 取代码
git clone https://github.com/tonydian/ashare-quant-public ashare_quant
cd ashare_quant

# 2. 建虚拟环境
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\activate             # Windows

# 3. 装依赖（5 个包，无 web 框架、无深度学习框架）
pip install -r requirements.txt

# 4. 准备数据（368 MB，在 Releases 里，见下节）
#    在 ashare_quant 的【上一级目录】解压：会生成 data_extract/was_stock_data
#    和 ashare_quant/build/altdata（可转债数据）

# 5. 自检
python tools/check_env.py          # 28 项检查，全 [OK] 才能继续

# 6. 出信号（先转债腿，再四腿合成）
python scripts/cb_signal.py        # 约 40 秒
python scripts/daily_signal.py --four   # 约 70 秒
```

`daily_signal.py --four` 会输出四腿完整方案到 `signals/latest_combo.json` 和
`reports/实盘组合_<日期>.md`（含每只股票的代码、参考价、股数、金额）。

> **仓库里没有数据，也没有预生成的因子面板和回测产物。**
> `build/` 下 4.9 GB 的因子/模型/回测产物、263 MB 行情包都在 `.gitignore` 里。
> 仓库只保留出信号必需的 3 个模型元数据文件（`build/models/wf_h10/`，约 22 KB）。
> **数据走 Releases 下载，不放仓库**（Git 仓库不适合存二进制大文件，每次改都会存一份副本）。

---

## 数据从哪来（最容易卡住的一步）

完整说明见 **[`数据获取指南.md`](数据获取指南.md)**，这里只给结论。

### 最快的一条路：从 Releases 下整套数据

打开 **[Releases 页面](https://github.com/tonydian/ashare-quant-public/releases/tag/v1.0)**，下其中一个：

| 附件 | 大小 | 说明 |
|---|---:|---|
| `was_data_full.zip` | 368 MB | **推荐**。全量，含行情面板与可转债数据，解压即用，不用联网 |
| `was_data_light.zip` | 72 MB | 网络慢就用这个：只含免费接口拿不到的部分，行情自己拉（约 20 分钟） |

在 `ashare_quant` 的**上一级目录**解压（zip 内部已带 `data_extract/` 与 `ashare_quant/build/` 前缀），
然后回到第 5 步自检。校验值（`md5sum` / `certutil -hashfile x.zip MD5`）：

```
was_data_full.zip   9faa0dbc6b910b09cf3a917437f64484
was_data_light.zip  3387863601b5ae678417ca5bba6a8c97
```

### 为什么不能全靠自己拉

系统最少需要这几样，缺一个都跑不起来：

| 文件 | 体积 | 能否自己拉 | 来源 |
|---|---:|---|---|
| `panel_ohlcv.npz` | 178 MB | **能** | 腾讯公开行情接口（免 token） |
| `raw_close.npz` | 22 MB | **能** | 同上 |
| `indices.csv` | 193 KB | **能** | 指数官方日线 |
| `market_cap.npz` | 34 MB | **不能** | 逐日 PIT 总股本，免费接口拿不到历史股本 |
| `industry.npy` | 25 MB | **不能** | 行业成分的历史变更记录 |
| `fund/*.parquet` | 27 MB | **不能** | 42 期财报（基本面因子用） |
| `meta.csv` | 322 KB | **不能** | 股票基本信息（含退市股） |

能自建的行情部分用 `tools/fetch_data_full.py` 拉，实测与原包偏差 **1.19e-07**（float32 精度极限）：

```bash
python tools/fetch_data_full.py --workers 16     # 约 20 分钟，支持断点续传
```

**剩下的 62 MB（市值 + 财报 + 行业 + 退市名单）免费接口拿不到，必须用 Release 里的包。**
特别注意：用「当前股本 × 历史价格」倒推市值是**不可用**的 —— 实测 2020 年中位偏差 42%、p90 318%，
会直接毁掉市值中性化。原因与实测数据见指南原文。

目录结构要求：

```
<你的工作目录>/
├── ashare_quant/                  ← 本仓库
│   └── build/altdata/             ← 可转债面板（数据包附带，2 个 parquet）
└── data_extract/
    └── was_stock_data/            ← 行情与参考数据放这里
        ├── panel_ohlcv.npz
        ├── raw_close.npz
        ├── market_cap.npz
        ├── industry.npy
        ├── meta.csv
        ├── indices.csv
        └── fund/
```

如果你想放到别处，设环境变量 `QF_DATA_RAW` 指向那个目录即可（`qf/config.py` 优先读它）。

---

## 常用命令

```bash
# 可转债腿信号（每天先跑这个，约 40 秒；不跑第四腿会是空的）
python scripts/cb_signal.py

# 出信号（四腿完整方案，约 70 秒）
python scripts/daily_signal.py --four

# 只出主策略（40 只）
python scripts/daily_signal.py

# 主策略回测（生产参数）
python scripts/run_backtest.py --start 2020-01-08

# 本地网页看板 → http://127.0.0.1:8765/
python scripts/web_server.py

# 每日增量更新行情（几十秒，只补最后几天）
python scripts/update_data.py --workers 16

# 全流程重跑（约 26 分钟，不含数据下载）
python scripts/run_pipeline.py --skip-data

# 测试
python -m pytest tests -q
```

`run_pipeline.py` 会依次执行：因子面板 → 模型重训 → 回测 → 四腿组合 → 信号 → 一致性校验。
**任何一次数据变更后都必须整条链重跑**，否则产物与数据不一致（`verify` 阶段会报错）。这是设计如此，
不是 bug —— 回测产物是数据的函数，数据变了产物就该重算。

---

## 目录结构

```
ashare_quant/
├── qf/                    核心库：数据加载 / 800 因子算子 / 打分模型 / 组合 / 回测 / 指标 / 图表
├── scripts/               可执行脚本（daily_signal / run_backtest / run_pipeline / update_data / web_server ...）
├── tools/                 check_env.py（环境自检）/ fetch_data_full.py（全量拉行情）/ 修复工具
├── tests/                 单元测试 + 独立审计（未来函数扰动、账务复算、指标复算）
├── web/                   本地看板（纯标准库 HTTP + 原生前端，无第三方 web 依赖）
├── docs/                  架构 / 数据说明 / 接口契约 / Web API
├── reports/               研究报告与回测报告（含方法论复盘、证伪记录）
├── signals/               一份现成的信号样例，可对照输出格式
├── build/                 ⚠️ 不入库。因子面板 / 模型 / 回测产物（4.9 GB，需自行生成）
│   └── models/wf_h10/     ✅ 入库（出信号必需的 3 个元数据文件）
├── data/                  ⚠️ 不入库
└── journal/               ⚠️ 不入库（运行日志）
```

---

## 文档索引

| 文档 | 什么时候看 |
|---|---|
| **[`开始阅读.md`](开始阅读.md)** | 刚拿到代码，想用 1 分钟弄清该先干什么 |
| **[`环境搭建指南.md`](环境搭建指南.md)** | 配置环境、遇到报错要查对照表 |
| **[`数据获取指南.md`](数据获取指南.md)** | 准备数据、搞不清哪些能自己拉 |
| `reports/回测报告.md` | 想看完整绩效、逐年逐月、压力期、局限性 |
| `reports/系统说明与风险声明.md` | 想知道系统边界和风险 |
| `reports/独立验证报告.md` | 想验证有没有未来函数 |
| `docs/ARCH.md` | 想改代码，先看模块分层 |
| `docs/DATA_NOTES.md` | 数据处理细节（复权、停牌、涨跌停、退市股） |

`reports/` 里有不少**证伪记录**（哪些方案试过、为什么否掉），看 `*证伪*` / `*可行性*` 结尾的文件。

---

## 已知限制

1. **数据是硬依赖**，没有完整数据跑不起来，且 4 类数据无法从免费接口获取。
2. **单点回测数字不可外推**：8 个相位铺开 3.5pp，报告区间比报告点更有意义。
3. **alpha 集中在空头端**：A 股 OHLCV 因子的 IC 高但纯多头收益有限，2025 年风格不利时尤其明显。
4. **小资金有结构性成本冲突**：30 万预算 + regime 0.45 → 12 万只投 5.4 万 → 40 只每笔约 1,300 元 →
   5 元最低佣金 = 37bp。这是「多持仓 + 择时半仓」在小资金上的必然结果，不是 bug。
5. **数据更新会微调历史回测**：`md.price` 的缩放基准取「最后有效交易日」的 `raw_close/close`，
   任何数据修复都会让历史绝对价整体平移，进而微调整手取整与止损门槛。**任何一次数据更新后，
   历史回测数字都会小幅变化**，这是设计特性。

---

## 风险声明

本项目为量化研究与教学用途。历史回测不代表未来收益，不构成任何投资建议或收益承诺。
A 股存在涨跌停、停牌、ST、退市、流动性枯竭等风险，实盘交易请自行做好资金与风险管理。
参考的 QuantSkills 因子库与 PandaFactor 算子语义版权归原项目所有。
