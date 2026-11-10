# 安慰剂对照（placebo）—— 加过滤类实验的强制步骤

> 工具：`build/placebo.py`　测试：`tests/test_placebo.py`（15 例）
> 起因：2026-11-06 的 HVRP 实验，一个「什么都不看、纯随机扔掉 20%」的对照组
> 拿到了实验组 83% 的收益改善。

## 1. 为什么必须做

在本项目的回测结构里，**任何「让候选池变小」的改动都会自动降杠杆**：

```
剔除部分股票  →  这些股票的分数变 NaN
             →  build_target 认为它们不该持有 → 强制卖出
             →  max_new_entries=6 限制每日补仓速度 → 补不回来
             →  平均持仓 34.5 → 20.4，平均仓位 56.4% → 34.9%
             →  在低涨幅窗口（2020-2026 沪深300 六年零涨幅）里，
                Sharpe 与 MaxDD 自动改善
```

这份改善**与因子信息含量完全无关**，但会让实验结果好看得多。

HVRP 那一轮的实测：

| | Sharpe | MaxDD | 8 相位胜率 | 平均持仓 | 平均仓位 |
|---|---:|---:|---:|---:|---:|
| BASE | 0.590 | −14.84% | — | 34.5 | 56.4% |
| F2（剔除 HVRP>80%） | 0.796 | −8.80% | Sharpe 7/8、MaxDD 8/8 | 20.4 | 34.9% |
| R1~R5（随机剔除 20%） | 0.666~0.765 | −8.8%~−9.4% | — | ~20.1 | ~34.5% |

F2 只比随机组高 0.085，而它**只动了 BASE 前 35 名里的 1.9%**（中位 0%），
被剔除的股票在模型分数里平均处于 28.8 分位 —— 本来就不买。

**结论：8 相位不足以防住这类错误。**必须配随机对照组。

## 2. 三条硬规则

1. **比例必须一致**：随机组的 `drop_frac` 要与实验组的实际剔除比例相同
   （用 `actual_drop_frac` 核对，别用名义值）。
2. **作用域必须一致**：随机抽样只能在「当日可交易 **且** 分数有限」的候选里做。
   从全市场抽会让对照组混入本来就不在候选池的股票。
3. **至少 5 个种子**，并报告区间而不是均值。单一种子自己就能差 0.1 个 Sharpe。

## 3. 用法（四步）

```python
import sys
sys.path.insert(0, "."); sys.path.insert(0, "build")
import numpy as np
import placebo

# ① 实验组 + 对照组一起进 variants
variants = {"BASE": S, "F2": np.where(mask, np.nan, S)}
variants.update(placebo.build_set(S, md.eligible, drop_frac=0.20,
                                  seeds=range(1, 6), rows=score_rows))

# ② 照常跑八相位，得到长表（columns: config / phase / cagr / sharpe / maxdd ...）

# ③ 对照
cmp = placebo.compare(df, treat="F2")
print(placebo.format_table(cmp))
print(placebo.format_verdict(*placebo.judge(cmp)))

# ④ 机械解释检查
print(placebo.leverage_check(df, treat="F2"))
```

`compare` 的输出：

```
指标             实验组      随机均值                  随机区间        差值      优于种子     配对t
Sharpe      +0.796    +0.711      [+0.666, +0.765]    +0.085       5/5    1.83
CAGR        +6.49%    +6.21%      [+5.88%, +6.60%]    +0.27%       3/5    0.80
MaxDD       -8.80%    -9.08%      [-9.37%, -8.77%]    +0.28%       4/5    0.59
```

## 4. 判据（`judge` 的默认值）

| 条件 | 阈值 | 理由 |
|---|---|---|
| 关键指标（默认 Sharpe）优于随机种子的比例 | ≥ 80%（4/5） | 单个种子噪声可到 0.1 Sharpe |
| 关键指标**逐相位配对 t** | ≥ 2.0 | 8 个相位就是 8 个样本，别用未配对 t |
| 其余指标（CAGR / MaxDD）差值 | 不得为负 | 不许用「回撤换收益」当 alpha |

HVRP 的 F2 在这套判据下的结果：Sharpe 5/5 种子都赢，但配对 t=1.83 < 2.0，
且 CAGR 只优于 3/5 → **未通过**。

请注意这个判据比项目原有的「8 相位胜率」严格得多：F2 按旧判据是 7/8 通过。

## 5. 什么时候不适用

- **不是「过滤」类的改动**（改权重、改调仓频率、改风控阈值）：它们不改变候选池，
  不会白拿降杠杆红利，用 8 相位 + 分段一致性即可。
- **已确认是「降杠杆」就是目的**的改动（例如故意加仓位上限）：此时要报的是
  「同等目标仓位下的收益」，不是「原始仓位下的 Sharpe」。

## 6. 相关文件

- `build/placebo.py` —— 工具本体（`random_mask` / `build_set` / `compare` / `judge` / `leverage_check`）
- `tests/test_placebo.py` —— 15 个不变量测试，含「F2 式假信号必须被判不通过」
- `build/exp_hvrp_random.py` —— 参考实现，照抄这个骨架即可
- `reports/HVRP高成交量溢价因子评估_2026-11-06.md` —— 事件全过程
