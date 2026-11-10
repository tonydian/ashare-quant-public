# 因子扩展方案（新数据源 / 开源因子库 / 接入路线）

> **执行进展（2026-10-21 更新）**
> - ✅ P0 基本面因子族已落地：`qf/fundamentals.py` + `scripts/build_fundamental_extra.py`，
>   15 个 PIT 因子（法定披露截止日+15 天缓冲，已通过未来数据扰动测试），DEV ICIR 0.18~0.30。
> - ❌ v2 增强模型实验结论：基本面因子加入复合分后**多头组合表现下降**（全期 5.29%/-13.21% vs v1 7.35%/-12.57%），
>   **主模型维持 v1**，基本面族保留为质量筛选/研究分支（详见回测报告 §4.7、因子库复盘报告 §1.5）。
> - ✅ 真正的收益/回撤改善来自**闲置现金替代资产**：80% 国债指数 + 20% 黄金 ETF，
>   全期 7.35%→9.36%、回撤 -12.57%→-10.00%（回测报告 §4.6）。
> - 🔬 **右侧因子补强（2026-10-22）**：按用户要求接入了 `seuzxh/quant-factor-skill` 知识库中的
>   Alpha158/360 右侧因子 16 个（`qf/alpha_rs.py`：ROC、MA 偏离、IMAX/IMIN、QTLU/QTLD、BETA、RSQR、RESI、CORR、SUMP/VSUMP），
>   并与 directional-alpha 296 因子库联合重建了右侧模型 wf_rs10 v2（趋势质量 60% + 回调买点 40%，
>   全期 +6.34%/-11.98%，2022+ +5.13%，旧版 +1.24%/-18.23%/2022+ -1.48%）；结论仍是研究分支、不替换主策略。
> - 🔬 **右侧趋势模型（2026-10-21 新增）**：`qf/trend.py` 16 个右侧趋势/动量因子（52 周新高、MA200 偏离、
>   ADX/DI、Donchian 突破、12-1 动量、相对强度、趋势 R² 等），`scripts/run_trend_model.py` 用
>   长多约束 ICIR walk-forward 产出 `build/models/wf_rs10`。结论：单独回测 CAGR +1.30% / MaxDD -18.23%
>   （同参数主模型 +9.22% / -10.00%），OOS IC 0.017；与主模型选股排名相关 -0.04；5% 集成权重仅微弱改善
>   且样本外无提升 → 保留为研究分支/网页可选模型，不替换主策略（详见 reports/右侧模型研究.md）。
> - ⏭️ 下一步优先级调整为：P1 资金流/预期类新数据源（北向、融资、龙虎榜、股东户数、分析师预期）
>   > P3 多资产（更多宏观/商品）> P2 公式因子移植（同质化高）> P4 自动挖掘。

> 背景：当前系统主模型用 34 个因子（来自 QuantSkills 三库 800 个 OHLCV 因子 + 自建 momentum-extra 8 个），
> 问题不是「因子不够多」，而是**信息源单一**：全部来自日频 OHLCV，80% 的族集中在量价/波动/反转，
> 多头端 alpha 有限（见 `reports/回测报告.md` §7）。扩展方向应是**新信息源**，而非继续堆同质公式。
>
> 本文档给出三层扩展路线、免费开源资源清单（含 2026 年仍活跃的项目）、以及接入本系统的具体步骤。

## 一、三层扩展路线（按投入产出排序）

### P0：用现有数据包就能做的「基本面因子族」（推荐先做，1-2 天）

数据包已含 `fund/YYYYMMDD.parquet` 42 期财报（2016Q1~2026Q2），字段：
`eps, revenue, rev_yoy, net_profit, profit_yoy, profit_qoq, bps, roe, ocfps, gpm, period`。
做法：按「A 股法定披露截止日 + 15 天缓冲」做 **PIT 对齐**（Q1 +45 天、半年报 +77 天、Q3 +46 天、年报 +135 天），
构造 **12~15 个 PIT 基本面因子**（实际落地 15 个，见 qf/fundamentals.py；已通过未来数据扰动测试）：

| 类别 | 因子 | 定义（示意） |
|---|---|---|
| 价值 | EP、BP、SP | eps/price、bps/price、revenue/市值（用当日市值） |
| 质量 | ROE、毛利率、盈利质量 | roe、gpm、ocfps/eps（经营现金流含量） |
| 成长 | 营收增速、利润增速、环比加速 | rev_yoy、profit_yoy、profit_yoy 的 4 季差分 |
| 稳健 | 低杠杆代理、规模 | 用市值/股本数据 |
| 预期变化 | 单季利润同比的时序变化 | profit_yoy - 上季 profit_yoy |

**接入成本极低**：新增 `qf/fundamentals.py` 生成周频面板，注册 `catalog`（lib=`fundamental-extra`），
直接进现有 `qf/model.py` 的 DEV 选池 + walk-forward，无需改模型/回测代码。
**预期**：IC 提升 0.003~0.01，2022/2024 类熊市回撤降低 2~4pct（低频锚，与量价因子互补）。

### P1：资金流 / 情绪 / 股东类因子（akshare 免费接口，2-3 天 + 每日更新）

| 因子族 | akshare 接口（示例） | 因子设计 |
|---|---|---|
| 北向资金 | `stock_hsgt_individual_em` / 持股明细 | 北向持股占比、20 日净买入/流通市值 |
| 融资融券 | `stock_margin_detail_sse` / `stock_margin_detail_szse` | 融资余额/流通市值、融资买入额占比、环比变化 |
| 龙虎榜 | `stock_lhb_detail_em` | 20 日龙虎榜净买入、机构席位占比 |
| 大宗交易 | `stock_dzjy_mrmx` | 折价率、成交额/流通市值 |
| 股东户数 | `stock_zh_a_gdhs` | 户数环比变化（筹码集中度） |
| 分红/解禁 | `stock_fhps_em` / `stock_restricted_release_queue_em` | 股息率、未来 60 日解禁市值占比 |
| 分析师预期 | `stock_profit_forecast_em` / `stock_analyst_detail_em` | 一致预期净利润上调幅度、目标价空间 |
| 情绪/微结构 | `stock_zt_pool_em`、`stock_market_activity_legu` | 涨停封板率、连板高度、市场活跃度（偏择时/regime） |

注意：这些数据历史长度、字段口径、更新稳定性都不如行情；建议先做「北向 + 融资 + 龙虎榜 + 股东户数」四个，
并做缺失率/覆盖年限检查，只保留 2017 年以来覆盖 >80% 的因子。

### P2：开源公式因子库移植（同质化高，需去重）

见下节清单。移植时要过三道关：① 与现有 800 因子的相关性剪枝（|ρ|>0.7 不重复入池）；
② DEV 期增量 IC（相对现有 34 因子池的残差 IC）；③ 无未来函数扰动测试（复用 `tests/test_audit_factors.py`）。

### P3：多资产 / 宏观（用于 regime 与资产配置，不直接选股）

数据包已有 `bond_000012.parquet`（国债指数）、`gold_518880.parquet`（黄金 ETF）。
可加：国债收益率曲线（`bond_zh_us_rate`）、期限利差、信用利差、商品指数（`futures_main_sina`）、
人民币汇率、沪深 300 波动率/股指期货基差（`futures_main_sina` 的 IF/IC/IM）、期权隐含波动率。
用途：改进趋势/回撤 overlay、现金替代（债券 ETF）、股债性价比择时——这是提高**风险调整收益**最直接的路。

### P4：自动因子挖掘（算力+过拟合风险，最后做）

Microsoft **RD-Agent**（与 Qlib 配套的自动研发 Agent）：自动提出假设、写因子代码、回测、迭代。
可在我们现有 walk-forward 框架下接入，但必须保留 DEV/验证/样本外三段切分与多重检验校正，
否则极容易过拟合。

## 二、免费开源因子库 / 数据集清单（2026 年仍活跃）

| 项目 | 内容 | 数量 | 许可 | 接入建议 |
|---|---|---|---|---|
| [microsoft/qlib](https://github.com/microsoft/qlib)（47k★） | **Alpha158 / Alpha360** 因子集 + 数据/模型框架；`qlib.contrib.data.loader` 有完整表达式 | 158 + 360 | MIT | 表达式可直接改写为本系统 `qf/ops.py` 面板算子；与本系统同源度最高 |
| [seuzxh/quant-factor-skill](https://github.com/seuzxh/quant-factor-skill) | Agent Skill：整合 Qlib Alpha158/360 + WorldQuant Alpha101 的**因子卡 + 表达式 + Python 模板** | 619 | MIT | 直接读取 references 的因子表达式清单，批量移植；含语义分类 |
| [quantskills/skill-factor-calendar-formulas](https://github.com/quantskills/skill-factor-calendar-formulas) | 复现 [Factors Directory](https://factors.directory/zh) 的公式因子库（量价+流动性+情绪+基本面），自带计算脚本与 L3 验证 | **241** | GPLv3 | 输入长表即可算出因子面板；**含基本面字段**，可与 P0 合并推进 |
| [WorldQuant Alpha101](https://arxiv.org/abs/1601.00991) | 101 Formulaic Alphas（论文+多份开源实现，如 [yli188/WorldQuant_alpha101_code](https://github.com/yli188/WorldQuant_alpha101_code)） | 101（约 82 可算） | 论文公开 | 公式移植到面板算子；注意多数是短周期量价，与现有因子相关性高 |
| [DolphinDB gtja191Alpha](https://github.com/dolphindb/DolphinDBModules) | 国泰君安 **Alpha191** 短周期价量因子实现（DolphinDB 模块，含中文公式说明） | 191 | 开源模块 | 按公式移植；历史较长、A 股实证多，但同样偏量价 |
| [quantskills/skill-futures-cta-alpha](https://github.com/quantskills/skill-futures-cta-alpha) | 商品期货 CTA 因子库（date×品种×因子面板，22 因子） | 22 | GPLv3 | 多资产/CTA 扩展 |
| [PandaAI panda_factor](https://github.com/PandaAI-Tech/panda_factor) + PandaAI 因子社区 | 我们已在用的算子库；官方因子大赛/因子广场有大量社区因子 | 社区 | GPLv3 | 用其公式风格扩展表达式；因子广场需在线获取 |
| [qusong0627/LightGBM_Alpha300（QuantDB）](https://github.com/qusong0627/LightGBM_Alpha300) | **A 股数据集**（2016 至今，~5500 只）：三套复权行情、三大报表、融资融券、技术指标、估值/情绪、**L1 110 维 + L2 211 维因子宽表（329 列）**；ModelScope 托管、每周末更新 | 329 列 | Apache-2.0 | ① 作为新因子来源（L1/L2 定义可参考/直接下载）；② 作为数据更新替代源（56GB 全量，可按分区下）；注意与现有数据包口径不同 |
| [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent) | 自动因子挖掘 Agent（与 Qlib 配套） | — | MIT | P4 阶段使用，需防过拟合 |
| [ta-lib](https://github.com/TA-Lib/ta-lib-python) / [pandas-ta](https://github.com/twopirllc/pandas-ta) | 技术指标算子库（RSI/MACD/ATR/BBands 等） | 150+ | BSD/MIT | 只作算子参考，指标本身与现有 risk-pattern 库高度重复 |
| [Alphalens-reloaded](https://github.com/stefan-jansen/alphalens-reloaded) | 因子评价（IC/分组/换手，Quantopian Alphalens 维护版） | — | Apache-2.0 | 可选：我们已有 `qf/evaluate.py`，不必替换，可交叉校验 |

> 说明：以上均为公开仓库/数据集，许可证以各仓库为准（GPL 项目注意衍生代码的传染性）。
> 因子公式本身（Alpha101/Alpha191）是公开数学表达式，可自行实现；使用数据时注意交易所/数据商版权（仅研究用途）。

## 三、接入本系统的具体步骤（以基本面因子族为例）

1. **数据层**：`qf/fundamentals.py`（已实现）
   - 读 42 期 `fund/*.parquet`，按 `period` 计算披露可得日（法定截止日 +15 天），得到每个交易日每股最新可得财报值；
   - 生成 `(T,N)` 面板 + `available_mask`（未披露期 NaN），与 `md.dates/codes` 严格对齐；
   - 派生因子（EP/BP/ROE/gpm/rev_yoy/profit_yoy/盈利质量…），做横截面去极值。
2. **因子目录**：注册 `catalog` 行（`lib="fundamental-extra"`, `category_en="Fundamental"`, `base`, `transform`），
   与现有 800 因子同样的字段结构，`qf/model.py` 无需改动。
3. **评估**：`scripts/build_factors.py` 同一套流程（DEV 期 ICIR + 覆盖率 + 相关性剪枝）；
   额外检查：与现有 34 因子的最大相关性、相对现有复合分的增量 IC（残差 IC）。
4. **选池与回测**：进 DEV 选池（每族配额）+ walk-forward；对比基线（当前 34 因子）在 2022+ 样本外的
   CAGR/MaxDD/Calmar 与 2022/2024 回撤。
5. **验收清单**：
   - [ ] PIT 对齐：用财报披露日而非报告期；未来数据扰动测试通过；
   - [ ] 覆盖率：入选股域内 >90%，2017 年以来连续；
   - [ ] 增量性：与现有因子 |ρ|<0.7；相对基线复合分有正增量 IC；
   - [ ] 样本外：不降低 2022-2026 的 Calmar，最大回撤不恶化；
   - [ ] 记录：因子卡 + `journal/runs.jsonl`，可复现。

## 四、现实预期与红线

- **不要指望「多就是好」**：现有 800 因子中 80% 集中在量价/波动/反转，新增 600 个 OHLCV 公式因子边际信息有限；
  基本面/资金流/宏观才是新信息维度。
- **每个新因子族都要过同一条 walk-forward 与增量检验**，禁止用全样本 IC 挑因子；
- **数据更新成本**：资金流/分析师数据需要日更，建议单独建更新脚本（复用 `scripts/update_data.py` 的模式），
  并记录数据源、抓取时间、字段口径；
- **容量**：基本面/资金流因子换手更低，对 30 万账户更友好；L2 高频因子容量小，30 万可用但放大资金后受限。
