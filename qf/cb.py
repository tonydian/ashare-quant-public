# -*- coding: utf-8 -*-
"""可转债策略生产模块：数据加载 + 到期收益率（YTM）+ 选券。

数据来源（build/fetch_cb.py 抓取，本地落盘）：
    build/altdata/cb_meta.parquet    每只转债一行（条款、规模、上市/退市日）
    build/altdata/cb_daily.parquet   长表 date/code/o/h/l/c/volume

为什么用 YTM 而不是「双低」：
    东财没有历史转股价报表，用「初始转股价 + 正股复权因子」反推的转股价值
    与快照校正后偏差 p10/p90 = -70/+98pp（下修 + 占位值混杂），双低/低溢价
    结论不可信。YTM 只依赖「票息序列 + 到期赎回价 + 到期日」三个发行时静态
    条款，无估算环节。

YTM 口径（精确 IRR）：
    剩余 yl 年、共 m=ceil(yl) 笔现金流，第 i 笔在 yl-(m-1-i) 年后
    （最后一笔恰在到期日）。现金流 = 前 m-1 笔为当期票息，第 m 笔为到期
    赎回价（已含最后一期利息，故不重复计入）。用二分求解 pv(irr)=价格。
    简单算术年化对短久期高估 1~2pp（实测美锦转债 +2.26pp），会系统性
    偏向短久期，故不用。

前视控制：
    * 信号基于 t 日收盘，t+1 开盘成交；
    * 不用评级（RATING 是当前快照，非 PIT）；
    * 不退市前剔除（数据含 748 只已退市转债），避免幸存偏差；
    * 信用风险用「价格下限 + 发行规模下限 + 上市满 N 日 + 20日均成交额」代理。

已知局限：
    * 无法从当前数据判断「已公告强赎」状态（东财列表无该字段），实盘需人工
      核对公告；强赎公告后转债通常在 1 个月内按 ~100 元被赎回，价格会向 100 收敛；
    * 违约债确实会被选中（实测搜特转债被选中 53 次），这是策略的结构性暴露。
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

_CN = "一二三四五六七八九十"
_DEFAULT_REDEEM = 108.0


# ------------------------------------------------------------------ 条款解析
def parse_coupons(txt) -> list[float] | None:
    """『第一年0.40%、第二年0.60%…』→ [0.004, 0.006, ...]。"""
    if not isinstance(txt, str):
        return None
    got = re.findall(rf"第([{_CN}]+)年\s*([\d.]+)\s*%", txt)
    cs = [float(v) / 100 for _, v in got] if got else []
    if not cs:
        cs = [float(x) / 100 for x in re.findall(r"([\d.]+)\s*%", txt)[:8]]
    cs = [c for c in cs if 0.0 <= c <= 0.08]   # 票息 0~8%，滤掉"面值100%"
    return cs if cs else None


def parse_redeem(txt) -> float:
    """到期赎回价（面值百分比，含最后一期利息）。默认 108。"""
    if not isinstance(txt, str):
        return _DEFAULT_REDEEM
    for pat in (r"面值的?\s*([\d.]+)\s*%", r"面值的?\s*([\d.]+)\s*元"):
        for m in re.finditer(pat, txt):
            v = float(m.group(1))
            if 100.0 <= v <= 125.0:
                return v
    return _DEFAULT_REDEEM


def solve_ytm(price: float, coupons: list[float], redeem: float, years_left: float,
              tol: float = 1e-7) -> float:
    """单只转债的精确到期收益率（年化 IRR）。price 为全价（元/百元面值）。"""
    if not coupons or not np.isfinite(price) or price <= 0 or not np.isfinite(years_left):
        return np.nan
    m = int(min(max(np.ceil(years_left), 1), len(coupons)))
    cfs = [coupons[len(coupons) - m + i] for i in range(m - 1)] + [redeem / 100.0]
    tmat = [years_left - (m - 1 - i) for i in range(m)]
    px = price / 100.0
    lo, hi = -0.95, 1.50
    for _ in range(60):
        mid = (lo + hi) / 2
        pv = 0.0
        for cf, tt in zip(cfs, tmat):
            pv += cf / (1.0 + mid) ** max(tt, 1e-6)
        if pv > px:          # 折现值高于价格 → 收益率太低 → 抬高下界
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2


# ------------------------------------------------------------------ 数据
class CBData:
    """可转债面板（按股票交易日历对齐）。"""

    def __init__(self, md=None, alt: Path | None = None, verbose: bool = False):
        self.alt = Path(alt) if alt else C.BUILD / "altdata"
        meta_p = self.alt / "cb_meta.parquet"
        daily_p = self.alt / "cb_daily.parquet"
        if not meta_p.exists() or not daily_p.exists():
            raise RuntimeError("可转债数据不存在，请先运行 build/fetch_cb.py")

        meta = pd.read_parquet(meta_p)
        daily = pd.read_parquet(daily_p)
        daily["date"] = pd.to_datetime(daily["date"])

        # 有正股面板的才纳入（YTM 不依赖正股，但退市/规模字段靠 meta；
        # 保留无正股对应者也可以用，这里不强制）
        meta = meta.drop_duplicates("code").reset_index(drop=True)

        if md is not None:
            cal = md.dates
        else:
            cal = pd.DatetimeIndex(sorted(daily["date"].unique()))
        daily = daily[daily["date"].isin(cal)]

        self.meta = meta
        self.cal = pd.DatetimeIndex(cal)
        self.n_days = len(self.cal)
        self.codes = meta["code"].tolist()
        self.code_ix = {c: i for i, c in enumerate(self.codes)}
        n = len(self.codes)
        self._n = n

        dpos = {d: i for i, d in enumerate(self.cal)}
        ti = daily["date"].map(dpos)
        ci = daily["code"].map(self.code_ix)
        ok = (ti.notna() & ci.notna()).to_numpy()
        ti = ti[ok].to_numpy().astype(int)
        ci = ci[ok].to_numpy().astype(int)
        vals = {c: daily[c].to_numpy()[ok] for c in ("open", "high", "low", "close", "volume")}

        def mat(col):
            m = np.full((self.n_days, n), np.nan, dtype=np.float32)
            m[ti, ci] = vals[col]
            return m

        self.close = mat("close")
        self.open = mat("open")
        self.high = mat("high")
        self.low = mat("low")
        self.vol = mat("volume")
        self.amount = self.close * self.vol          # 近似成交额（元）

        # ---- 条款
        coupons = [parse_coupons(meta["coupon_text"].iloc[j]) for j in range(n)]
        redeem = np.array([parse_redeem(meta["redeem_text"].iloc[j]) for j in range(n)])
        span = ((meta["maturity"] - meta["list_date"]).dt.days / 365.25).to_numpy()
        years = np.array([
            len(coupons[j]) if coupons[j]
            else (int(round(span[j])) if np.isfinite(span[j]) and 1 <= span[j] <= 30 else 6)
            for j in range(n)], dtype=int)
        self.coupons, self.redeem, self.years = coupons, redeem, years

        # ---- 生命周期位置
        self.list_pos = np.array([self._date_pos(d, 0) for d in meta["list_date"]])
        self.delist_pos = np.array([self._date_pos(d, self.n_days) for d in meta["delist_date"]])
        self.issue_size = meta["issue_size"].to_numpy(dtype=np.float64)
        self.maturity = meta["maturity"].to_numpy()

        # ---- 剩余年限（按日历广播）
        self.years_left = np.full((self.n_days, n), np.nan, dtype=np.float32)
        calv = self.cal.values
        for j in range(n):
            if pd.isna(self.maturity[j]):
                continue
            self.years_left[:, j] = ((self.maturity[j] - calv) /
                                     np.timedelta64(1, "D") / 365.25)
        self.years_left = np.clip(self.years_left, 0.05, 20.0)

        # ---- 20 日均成交额
        self.amount_20d = (pd.DataFrame(self.amount)
                           .rolling(20, min_periods=10).mean().to_numpy())

        if verbose:
            print(f"[cb] {n} 只转债（含已退市 {meta['delist_date'].notna().sum()}）"
                  f"  {self.n_days} 交易日  {self.cal[0].date()} ~ {self.cal[-1].date()}")
            print(f"[cb] 有价格格 {np.isfinite(self.close).mean():.1%}；"
                  f"票息解析成功 {sum(c is not None for c in coupons)}/{n}")

    # ---------------- 单日视图
    def ytm_on(self, t: int) -> np.ndarray:
        """信号日 t 的全市场 YTM（向量化：按剩余年限 m 分组二分）。"""
        n = self._n
        out = np.full(n, np.nan, dtype=np.float64)
        px = self.close[t].astype(np.float64)
        yl = self.years_left[t].astype(np.float64)
        ok = np.isfinite(px) & (px > 0) & np.isfinite(yl)
        cols = np.flatnonzero(ok)
        if not len(cols):
            return out
        for j in cols:
            out[j] = solve_ytm(px[j], self.coupons[j], self.redeem[j], yl[j])
        return out

    def ytm_matrix(self, t0: int = 0) -> np.ndarray:
        """[t0:] 全历史 YTM 矩阵（回测用，较慢：逐日逐只）。"""
        T = self.n_days - t0
        out = np.full((T, self._n), np.nan, dtype=np.float32)
        for t in range(t0, self.n_days):
            out[t - t0] = self.ytm_on(t)
        return out

    # ---------------- 内部：筛选 + 选券（select 与 backtest 共用，保证口径一致）
    def _elig(self, t: int, ytm: np.ndarray, min_price: float = 90.0,
              max_price: float = 1e9, min_issue: float = 2.0,
              min_list_days: int = 10, min_amount: float = 5e6,
              min_ytm: float = -9.0, min_years_left: float = 0.30) -> np.ndarray:
        """信号日 t 的合格券掩码。"""
        px = self.close[t].astype(np.float64)
        amt = self.amount_20d[t]
        yl = self.years_left[t].astype(np.float64)
        return (np.isfinite(px) & (px > 0)
                & (px >= min_price) & (px <= max_price)
                & (t >= self.list_pos + min_list_days)
                & (t < self.delist_pos)
                & (self.issue_size >= min_issue)
                & np.isfinite(ytm) & (ytm >= min_ytm)
                & (amt >= min_amount)
                & (yl >= min_years_left))

    def _pick(self, t: int, ytm: np.ndarray, elig: np.ndarray, n_hold: int,
              pool_mult: float = 3.0, rank: str = "ytm_duration") -> np.ndarray:
        """在合格券中挑 n_hold 只，返回列索引。"""
        idx = np.flatnonzero(elig)
        if not len(idx):
            return idx[:0]
        if rank == "ytm_duration" and pool_mult > 1.0:
            pool = idx[np.argsort(-ytm[idx])[:max(int(n_hold * pool_mult), n_hold)]]
            kv = self.years_left[t][pool].astype(np.float64)
            kv = np.where(np.isfinite(kv), kv, np.nanmedian(kv) if len(kv) else 1.0)
            return pool[np.argsort(kv)[:n_hold]]
        return idx[np.argsort(-ytm[idx])[:n_hold]]

    # ---------------- 选券
    def select(self, t: int, n_hold: int = 20,
               min_price: float = 90.0, max_price: float = 1e9,
               min_issue: float = 2.0, min_list_days: int = 10,
               min_amount: float = 5e6, min_ytm: float = -9.0,
               min_years_left: float = 0.30,
               pool_mult: float = 3.0, rank: str = "ytm_duration",
               score_fn=None) -> pd.DataFrame:
        """信号日 t 的选券结果。

        rank 口径：
          "ytm"           纯 YTM 降序取前 n_hold（旧口径）
          "ytm_duration"  先用 YTM 取前 n_hold*pool_mult 建池，池内按**剩余年限升序**
                          取 n_hold（默认，2026-10-29 起的生产口径）

        min_years_left 默认 0.30 年（约 110 天）而非 0：
            回测显示短久期改善的 80% 来自剩余期限 <0.2 年的债，但那段实盘几乎
            不可交易（转债到期前约 5 个交易日即停止交易，兑付资金再冻结 3~7 天）。
            加 ≥0.30 年约束后（更贴近可执行）：
                纯 YTM        16.62% / Sharpe 1.371 / DD -20.64%
                短久期 3x/20  18.11% / Sharpe 1.531 / DD -13.58%
            12 组「池子倍数 × 持仓数」中 11 组 CAGR 与 Sharpe 优于基线，
            **回撤 12/12 全部优于基线**；相位极差 1.32%。
        注意：这是「YTM 主导 + 久期微调」，不是按 YTM÷年限 排序（后者会把 YTM
        压到次要位置，实测反而更差）。
        """
        px = self.close[t].astype(np.float64)
        ytm = self.ytm_on(t)
        amt = self.amount_20d[t]
        yl = self.years_left[t].astype(np.float64)

        elig = self._elig(t, ytm, min_price=min_price, max_price=max_price,
                          min_issue=min_issue, min_list_days=min_list_days,
                          min_amount=min_amount, min_ytm=min_ytm,
                          min_years_left=min_years_left)
        idx = np.flatnonzero(elig)
        if not len(idx):
            return pd.DataFrame(columns=["code", "name", "price", "ytm", "years_left",
                                         "issue_size", "amount_20d", "score", "weight"])
        if score_fn is not None:
            score = score_fn(t, idx)
            order = np.argsort(score)
            pick = idx[order[:min(n_hold, len(idx))]]
        else:
            pick = self._pick(t, ytm, elig, n_hold, pool_mult=pool_mult, rank=rank)

        df = pd.DataFrame({
            "code": [self.codes[j] for j in pick],
            "name": self.meta["name"].iloc[pick].to_numpy(),
            "price": px[pick],
            "ytm": ytm[pick],
            "years_left": yl[pick],
            "issue_size": self.issue_size[pick],
            "amount_20d": amt[pick],
            "rating": self.meta["rating"].iloc[pick].to_numpy(),
            "maturity": pd.to_datetime(self.maturity[pick]),
            "score": ytm[pick],
            "weight": 1.0 / max(len(pick), 1),
        })
        return df.sort_values("ytm", ascending=False).reset_index(drop=True)

    def universe_stats(self, t: int) -> dict:
        """全市场环境指标（用于判断当前是否值得配转债）。"""
        px = self.close[t].astype(np.float64)
        ytm = self.ytm_on(t)
        amt = self.amount_20d[t]
        live = (np.isfinite(px) & (px > 0) & (t >= self.list_pos + 10)
                & (t < self.delist_pos) & np.isfinite(ytm))
        if not live.any():
            return {}
        y = ytm[live]
        return {
            "n_live": int(live.sum()),
            "ytm_median": float(np.median(y)),
            "ytm_p25": float(np.percentile(y, 25)),
            "ytm_p75": float(np.percentile(y, 75)),
            "ytm_p90": float(np.percentile(y, 90)),
            "ytm_pct": float((y <= np.median(y)).mean()),   # 占位，见下
            "price_median": float(np.median(px[live])),
            "price_below_100": float((px[live] < 100).mean()),
            "amount_median": float(np.nanmedian(amt[live])),
        }

    def _date_pos(self, d, default=0):
        if pd.isna(d):
            return default
        i = int(np.searchsorted(self.cal.values, np.datetime64(d)))
        return min(i, self.n_days)

    # ---------------- 便捷
    def date_index(self, asof: str) -> int:
        i = int(np.searchsorted(self.cal.values, np.datetime64(pd.Timestamp(asof)),
                                side="right")) - 1
        if i < 0 or self.cal[i] < pd.Timestamp(asof) - pd.Timedelta(days=1):
            raise ValueError(f"转债数据没有 {asof} 附近的数据（最后 {self.cal[-1].date()}）")
        return i

    @property
    def last_date(self) -> pd.Timestamp:
        return self.cal[-1]

    # ---------------- YTM 矩阵缓存
    def ytm_matrix_cached(self, t0: int = 0, cache_dir=None) -> np.ndarray:
        """缓存版 YTM 矩阵：命中 build/cb/ytm_matrix_<起始日>.npy 时只补算尾部。

        逐日逐只求解约 1s/日，全历史要 40 分钟，看板不可能现算。缓存文件
        按「起始日」命名，覆盖行数不足以对齐当前日历，故用行数差补算尾部；
        数据回补导致历史变化时需手动删缓存（fetch_cb.py 会打印提示）。
        """
        cache_dir = Path(cache_dir) if cache_dir else C.BUILD / "cb"
        key = f"ytm_matrix_{self.cal[t0].date()}.npy"
        p = cache_dir / key
        want = (self.n_days - t0, self._n)
        M = None
        if p.exists():
            try:
                M = np.load(p)
                if M.shape[1] == want[1] and 0 < M.shape[0] <= want[0]:
                    pass            # 可用（可能缺尾部）
                else:
                    M = None
            except Exception:
                M = None
        if M is None:
            M = np.full(want, np.nan, dtype=np.float32)
            have = 0
        else:
            have = M.shape[0]
        if have < want[0]:
            new = np.full(want, np.nan, dtype=np.float32)
            if have:                       # 缓存缺失时 M 是空壳，不能整块赋给 new[:0]
                new[:have] = M[:have]
            for k in range(have, want[0]):
                new[k] = self.ytm_on(t0 + k)
            M = new
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.save(p, M)
            except Exception:
                pass
        return M


def backtest(cb: CBData, start: str = "2019-01-02", end: str | None = None,
             n_hold: int = 20, rebal: int = 10, pool_mult: float = 3.0,
             rank: str = "ytm_duration", min_price: float = 90.0,
             min_issue: float = 2.0, min_list_days: int = 10,
             min_amount: float = 5e6, min_ytm: float = -9.0,
             min_years_left: float = 0.30, cost_bp: float = 5.0,
             cash_yield: float = 0.015, ytm: np.ndarray | None = None) -> dict:
    """可转债策略回测（与 scripts/cb_signal.py 同口径）。

    规则与生产一致：
        * t 日收盘用 `_elig` + `_pick` 选券，**t+1 收盘**结算收益（收盘→收盘）；
        * 每 rebal 个交易日调仓一次，换手成本 `turn × cost_bp × 2`（双边）；
        * 未投资部分按 cash_yield 计息；
        * 退市日次日权重清零、资金转现金（不假设按面值赎回，用最后成交价结算）；
        * 债券停牌/无价格当日收益记 0（权重不动，不虚增收益）。

    返回 dict：nav / exposure / n_held / turnover_per_reb / n_reb / first_selection。
    """
    cal = cb.cal
    t0 = int(np.searchsorted(cal.values, np.datetime64(pd.Timestamp(start))))
    t1 = (cb.n_days if not end
          else min(int(np.searchsorted(cal.values, np.datetime64(pd.Timestamp(end)),
                                       side="right")), cb.n_days))
    if t1 - t0 < 30:
        raise RuntimeError(f"转债回测区间过短（{t1 - t0} 个交易日）")

    if ytm is None:
        ytm = cb.ytm_matrix_cached(t0)
    close = cb.close
    n = cb._n
    target = np.zeros(n)
    nav = 1.0
    navs, expos, held = [], [], []
    turn_tot, n_reb = 0.0, 0
    hold_until = -1
    first_sel = None

    for t in range(t0, t1):
        px = close[t]
        if t >= hold_until:
            row = ytm[t - t0]
            elig = cb._elig(t, row, min_price=min_price, min_issue=min_issue,
                            min_list_days=min_list_days, min_amount=min_amount,
                            min_ytm=min_ytm, min_years_left=min_years_left)
            pick = cb._pick(t, row, elig, n_hold, pool_mult=pool_mult, rank=rank)
            w_new = np.zeros(n)
            if len(pick):
                w_new[pick] = 1.0 / len(pick)
            if first_sel is None and len(pick):
                first_sel = {"date": str(cal[t].date()),
                             "codes": [cb.codes[j] for j in pick]}
            if t > t0:
                turn = float(np.abs(w_new - target).sum() / 2.0)
                nav *= (1.0 - turn * cost_bp / 1e4 * 2.0)
                turn_tot += turn
                n_reb += 1
            target = w_new
            hold_until = t + rebal

        if t + 1 < t1:
            nxt = close[t + 1]
            with np.errstate(invalid="ignore"):
                ret = np.where(np.isfinite(px) & np.isfinite(nxt) & (px > 0),
                               nxt / px - 1.0, 0.0)
            gone = (target > 1e-9) & (cb.delist_pos <= t + 1)
            if gone.any():
                target = target.copy()
                target[gone] = 0.0
            port = float(np.nansum(target * ret)) + max(0.0, 1.0 - target.sum()) * cash_yield / 252.0
            nav *= (1.0 + port)
            target = target * (1.0 + ret)

        navs.append(nav)
        expos.append(float(target.sum()))
        held.append(int((target > 1e-9).sum()))

    idx = cal[t0:t0 + len(navs)]
    return {"nav": pd.Series(navs, index=idx),
            "exposure": np.array(expos), "n_held": np.array(held),
            "turnover_per_reb": turn_tot / max(n_reb, 1), "n_reb": n_reb,
            "first_selection": first_sel}
