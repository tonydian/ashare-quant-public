# -*- coding: utf-8 -*-
"""基本面因子族（fundamental-extra）：用数据包 42 期财报构造 PIT 基本面因子。

数据包 `fund/YYYYMMDD.parquet` 是**按报告期**组织的累计（YTD）财务数据，不含公告日期。
本模块因此需要外部披露日数据做 PIT 对齐，两级口径：

1. **真实公告日**（首选）：读取 `build/pandadata/disclosure_date.parquet`
   （PandaData `get_stock_disclosure_date` 落盘，44 期 / 2015Q4–2026Q3），
   按「股票 × 报告期」取 `actual_disclosure_date` 的**次日**作为可得日
   （A 股财报多在盘后披露，当日不可用）。
2. **固定滞后**（兜底）：法定披露截止日 + 15 天缓冲（Q1 +45、半年报 +77、Q3 +46、年报 +135）。
   仅在个别 (股票, 报告期) 缺真实公告日时使用。

实测覆盖率（对本地 fund 且已上市口径）：半年报 98.6%、三季报 98.1%、一季报 80.0%、年报 81.8%。
Q1/Q4 的系统性缺行来自上游数据源（全市场批量查与单股查均缺，如 000858 五粮液 2024Q1 无记录），
无法通过取数方式修复，故保留固定滞后兜底。

PIT 对齐是**每股独立**的：`avail_mat[p, j]` 为第 j 只股票第 p 期的可得日，
`p_idx[t, j]` 为第 j 只股票在第 t 日已知的最新报告期下标。
报告过期超过 400 天则视为缺失（避免用陈旧数据打分）。

因子（均为 t 时点可得的原始值，进入模型后再做横截面 rank/中性化）：
    F001 ep_ttm         TTM 归母净利润 / 当日总市值（盈利收益率）
    F002 bp             每股净资产 / 当日价格（账面市值比）
    F003 sp_ttm         TTM 营业收入 / 当日总市值（营收市值比）
    F004 roe_ann        年化 ROE（YTD ROE × 4/季度数）
    F005 gross_margin   毛利率（YTD）
    F006 earnings_quality  经营现金流/净利润（YTD，eps>0 时）
    F007 rev_growth     营收同比增速（YTD）
    F008 profit_growth  净利润同比增速（YTD）
    F009 profit_accel   净利润增速加速度（同比增速 - 4 个季度前同比增速）
    F010 rev_accel      营收增速加速度
    F011 roe_change     年化 ROE 变化（- 4 个季度前）
    F012 gm_change      毛利率变化（- 4 个季度前）
    F013 accruals       (EPS - 每股经营现金流)/价格（应计项目，越低越“干净”）
    F014 margin_expansion   利润增速 - 营收增速（利润率扩张）
    F015 profit_qoq     单季净利润环比增速（数据包直接给出）
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

from . import config as C

LIB = "fundamental-extra"

# (factor_id, key, name_cn)
DEFS = [
    ("F001", "ep_ttm", "TTM盈利收益率"),
    ("F002", "bp", "账面市值比"),
    ("F003", "sp_ttm", "TTM营收市值比"),
    ("F004", "roe_ann", "年化ROE"),
    ("F005", "gross_margin", "毛利率"),
    ("F006", "earnings_quality", "盈利现金含量"),
    ("F007", "rev_growth", "营收同比增速"),
    ("F008", "profit_growth", "净利润同比增速"),
    ("F009", "profit_accel", "净利润增速加速度"),
    ("F010", "rev_accel", "营收增速加速度"),
    ("F011", "roe_change", "ROE变化"),
    ("F012", "gm_change", "毛利率变化"),
    ("F013", "accruals", "应计项目/价格"),
    ("F014", "margin_expansion", "利润率扩张"),
    ("F015", "profit_qoq", "单季利润环比"),
]

# 报告期 -> 披露滞后（日历天）：**兜底口径**，按 A 股法定披露截止日 + 15 天缓冲
#   Q1 截止 4/30（+30 天）；半年报截止 8/31（+62 天）；Q3 截止 10/31（+31 天）；年报截止次年 4/30（+120 天）
# 仅在缺真实公告日时使用；实测真实披露中位滞后为 28/57/28/108 天，故兜底口径系统性偏晚 17~27 天。
LAG_DAYS = {1: 45, 2: 77, 3: 46, 4: 135}
MAX_STALE_DAYS = 400

# 真实公告日数据（PandaData get_stock_disclosure_date 落盘）
DISC_FILE = C.PKG / "build" / "pandadata" / "disclosure_date.parquet"
# 公告日 -> 可得日的缓冲（盘后披露，次日才可用于交易决策）
DISC_BUFFER_DAYS = 1


def _load_raw() -> pd.DataFrame:
    files = sorted(C.FUND_DIR.glob("*.parquet"))
    if not files:
        # 增量数据目录通常只含行情/指数/债券等，财报仍来自原始数据包
        base = pathlib.Path(C.DATA_RAW)
        alt = base.parent / base.name.replace("_ext", "") / "fund"
        files = sorted(alt.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"找不到财报数据：{C.FUND_DIR}（且回退目录为空）")
    dfs = []
    for f in files:
        d = pd.read_parquet(f)
        d["period"] = f.stem
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.drop_duplicates(["code", "period"], keep="last")
    return df


class FundamentalData:
    def __init__(self, md, use_disclosure: bool = True):
        """use_disclosure=False 时退回「固定滞后」口径（用于新旧口径 bound test 对照）。"""
        self.md = md
        raw = _load_raw()
        periods = sorted(raw["period"].unique())
        self.periods = periods
        self.period_end = pd.DatetimeIndex([pd.Timestamp(p) for p in periods])
        self.quarter = np.array([(pd.Timestamp(p).month - 1) // 3 + 1 for p in periods], dtype=np.int64)
        P, N = len(periods), md.N
        # 报告期 × 股票 矩阵
        code_idx = {c: j for j, c in enumerate(md.codes)}
        raw = raw[raw["code"].isin(code_idx)]
        self.M: dict[str, np.ndarray] = {}
        for col in ["eps", "revenue", "net_profit", "bps", "roe", "ocfps", "gpm",
                    "rev_yoy", "profit_yoy", "profit_qoq"]:
            piv = raw.pivot(index="period", columns="code", values=col).reindex(
                index=periods).reindex(columns=list(md.codes)).astype(np.float64)
            self.M[col] = piv.to_numpy()
        # 单季度净利润/营收（YTD 差分）
        self.sq_net = self._single_quarter(self.M["net_profit"])
        self.sq_rev = self._single_quarter(self.M["revenue"])
        # TTM
        self.ttm_net = self._ttm(self.sq_net)
        self.ttm_rev = self._ttm(self.sq_rev)
        # 每期 × 每股 可得日矩阵（真实公告日优先，缺失回退固定滞后）
        self.avail_mat, self.avail, self.disc_cov = self._build_avail(md, use_disclosure)
        # 每日 × 每股 对齐索引：p_idx[t, j] = 第 j 只股票在第 t 日已知的最新报告期下标（-1 表示尚无）
        self.p_idx = self._align(md.dates, P, N)
        # 数据陈旧度（天）
        dv = np.asarray(md.dates).astype("datetime64[ns]")
        valid = self.p_idx >= 0
        p_safe = np.where(valid, self.p_idx, 0)
        src_end = self.period_end.values[p_safe].astype("datetime64[ns]")
        src_end[~valid] = np.datetime64("1970-01-01", "ns")   # 占位，随后被 where 屏蔽
        with np.errstate(invalid="ignore"):
            stale = (dv[:, None] - src_end) / np.timedelta64(1, "D")
        self.stale_days = np.where(valid, stale, np.nan).astype(np.float32)
        self.stale_mask = (~valid) | (self.stale_days > MAX_STALE_DAYS)

    # ------------------------------------------------------------ 可得日矩阵
    def _build_avail(self, md, use_disclosure: bool):
        """构造 avail_mat[P, N]：真实公告日 + 缓冲，缺则回退 period_end + LAG_DAYS。"""
        P, N = len(self.periods), md.N
        lag = np.array([LAG_DAYS[int(q)] for q in self.quarter], dtype="timedelta64[D]")
        base = self.period_end.values + lag                          # (P,)
        avail_mat = np.broadcast_to(base[:, None], (P, N)).copy()    # (P, N) datetime64[ns]
        cov = np.zeros((P, N), dtype=bool)

        if use_disclosure and DISC_FILE.exists():
            d = pd.read_parquet(DISC_FILE)
            d["end_date"] = pd.to_datetime(d["end_date"], errors="coerce")
            d["actual_disclosure_date"] = pd.to_datetime(d["actual_disclosure_date"], errors="coerce")
            d["code"] = d["code"].astype(str).str.zfill(6)
            d = d.dropna(subset=["end_date", "actual_disclosure_date"])
            per2p = {ts: i for i, ts in enumerate(self.period_end)}
            code2j = {c: j for j, c in enumerate(md.codes)}
            p_arr = d["end_date"].map(per2p)
            j_arr = d["code"].map(code2j)
            m = p_arr.notna() & j_arr.notna()
            if m.any():
                pi = p_arr[m].to_numpy(np.int64)
                ji = j_arr[m].to_numpy(np.int64)
                dt = d.loc[m, "actual_disclosure_date"].to_numpy("datetime64[ns]") \
                    + np.timedelta64(DISC_BUFFER_DAYS, "D")
                avail_mat[pi, ji] = dt
                cov[pi, ji] = True

        # 保护：可得日不得早于报告期结束次日（防止上游脏数据造成「未出报表即可用」）
        floor = self.period_end.values[:, None] + np.timedelta64(1, "D")
        avail_mat = np.maximum(avail_mat, floor)

        # 注意：不做单调化。年报(Q4)与一季报(Q1)均在 4 月底披露，两者可得日会「撞车」，
        # 强制单调会把 Q1 的可得日压到不早于年报，凭空损失约 16 天提前量。
        # p_idx 的语义是「期号最大的已可得报告期」，由 _align 逐期累加实现，天然处理非单调。

        # 兼容旧接口：avail[P] 取该期全市场最晚可得日（保守上界，仅供展示/测试）
        avail = pd.DatetimeIndex(avail_mat.max(axis=1))
        # 真实公告日覆盖率（分母 = 该期本地实际有财报的股票数，不含未上市/无数据）
        has_fund = np.isfinite(self.M["revenue"]) | np.isfinite(self.M["net_profit"])
        den = has_fund.sum(axis=1)
        num = (cov & has_fund).sum(axis=1)
        disc_cov = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        return avail_mat, avail, disc_cov

    def _align(self, dates: np.ndarray, P: int, N: int) -> np.ndarray:
        """p_idx[t, j] = 满足 avail_mat[p, j] <= dates[t] 的**最大期号**；无则 -1。

        逐期累加而非 searchsorted：可得日不保证按期号单调（年报与一季报 4 月底撞车），
        后写的期号覆盖先写的，正好给出「最新的已可得报告期」。
        """
        T = len(dates)
        out = np.full((T, N), -1, dtype=np.int16)
        dv = np.asarray(dates).astype("datetime64[ns]")[:, None]
        step = np.empty((T, N), dtype=bool)
        for p in range(P):
            np.less_equal(self.avail_mat[p][None, :], dv, out=step)
            out[step] = p
        return out

    def _single_quarter(self, ytd: np.ndarray) -> np.ndarray:
        """YTD 累计 -> 单季度（同一年内差分；Q1 即单季）。"""
        P = len(self.periods)
        out = np.full_like(ytd, np.nan)
        for p in range(P):
            q = self.quarter[p]
            if q == 1:
                out[p] = ytd[p]
            else:
                prev = p - 1
                if prev >= 0 and self.quarter[prev] == q - 1 and \
                        self.period_end[p].year == self.period_end[prev].year:
                    out[p] = ytd[p] - ytd[prev]
        return out

    def _ttm(self, sq: np.ndarray) -> np.ndarray:
        P = len(self.periods)
        out = np.full_like(sq, np.nan)
        for p in range(3, P):
            win = sq[p - 3:p + 1]
            ok = np.isfinite(win).all(axis=0)
            out[p, ok] = win[:, ok].sum(axis=0)
        return out

    # ---------------------------------------------------------------- 每日对齐
    def _to_daily(self, period_mat: np.ndarray, ffill: bool = True) -> np.ndarray:
        """报告期矩阵 -> 每日矩阵（按**每股**披露可得日；ffill 处理个别报告缺失）。"""
        M = period_mat.copy()
        if ffill:
            for p in range(1, M.shape[0]):
                miss = ~np.isfinite(M[p])
                M[p, miss] = M[p - 1, miss]
        # p_idx[t, j] 指向第 j 只股票在第 t 日已知的最新报告期
        idx = np.where(self.p_idx >= 0, self.p_idx, 0).astype(np.int64)
        cols = np.arange(self.md.N, dtype=np.int64)[None, :]
        out = M[idx, cols]
        out[self.p_idx < 0] = np.nan
        out[self.stale_mask] = np.nan
        return out

    def _shift_periods(self, period_mat: np.ndarray, k: int = 4) -> np.ndarray:
        out = np.full_like(period_mat, np.nan)
        if k < len(period_mat):
            out[k:] = period_mat[:-k]
        return out

    # ---------------------------------------------------------------- 因子
    def build_factors(self) -> dict[str, np.ndarray]:
        md = self.md
        mcap = md.mcap.astype(np.float64)
        price = md.price.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            ep_ttm = self._to_daily(self.ttm_net) / np.where(mcap > 0, mcap, np.nan)
            sp_ttm = self._to_daily(self.ttm_rev) / np.where(mcap > 0, mcap, np.nan)
            bp = self._to_daily(self.M["bps"]) / np.where(price > 0, price, np.nan)
        roe_ytd = self._to_daily(self.M["roe"])
        # 当期季度数（每股独立）：p_idx + 1 作查表偏移，-1 命中哨兵位 → nan
        q_lut = np.concatenate([[np.nan], self.quarter.astype(np.float64)]).astype(np.float32)
        q_frac = q_lut[self.p_idx + 1]          # (T, N)
        with np.errstate(invalid="ignore", divide="ignore"):
            roe_ann = roe_ytd * 4.0 / q_frac
        gpm = self._to_daily(self.M["gpm"])
        eps = self._to_daily(self.M["eps"])
        ocfps = self._to_daily(self.M["ocfps"])
        with np.errstate(invalid="ignore", divide="ignore"):
            eq = np.where(eps > 0, ocfps / eps, np.nan)
            accruals = (eps - ocfps) / np.where(price > 0, price, np.nan)
        rev_yoy = self._to_daily(self.M["rev_yoy"])
        profit_yoy = self._to_daily(self.M["profit_yoy"])
        rev_yoy_4 = self._to_daily(self._shift_periods(self.M["rev_yoy"], 4))
        profit_yoy_4 = self._to_daily(self._shift_periods(self.M["profit_yoy"], 4))
        roe_4 = self._to_daily(self._shift_periods(self.M["roe"], 4)) * 4.0 / q_frac
        gpm_4 = self._to_daily(self._shift_periods(self.M["gpm"], 4))
        profit_qoq = self._to_daily(self.M["profit_qoq"])

        fac = {
            "ep_ttm": ep_ttm,
            "bp": bp,
            "sp_ttm": sp_ttm,
            "roe_ann": roe_ann,
            "gross_margin": gpm,
            "earnings_quality": eq,
            "rev_growth": rev_yoy,
            "profit_growth": profit_yoy,
            "profit_accel": profit_yoy - profit_yoy_4,
            "rev_accel": rev_yoy - rev_yoy_4,
            "roe_change": roe_ann - roe_4,
            "gm_change": gpm - gpm_4,
            "accruals": accruals,
            "margin_expansion": profit_yoy - rev_yoy,
            "profit_qoq": profit_qoq,
        }
        # 极端值截断（-1e8~1e8 之外视为数据错误）
        for k, v in fac.items():
            fac[k] = np.where(np.isfinite(v) & (np.abs(v) < 1e8), v, np.nan).astype(np.float32)
        return fac

    # ---------------------------------------------------------------- 对外
    def panels_at_rows(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        fac = self.build_factors()
        out = {}
        for fid, key, _ in DEFS:
            out[f"{LIB}:{fid}"] = fac[key][rows]
        return out

    def build_live(self, md, key: str) -> np.ndarray:
        fid = key.split(":")[1]
        for f_id, k, _ in DEFS:
            if f_id == fid:
                return self.build_factors()[k]
        raise KeyError(key)


def catalog() -> pd.DataFrame:
    rows = []
    for fid, key, name in DEFS:
        rows.append({
            "lib": LIB, "factor_id": fid, "slug": key, "name_cn": name,
            "name_en": key, "category_cn": "基本面", "category_en": "Fundamental",
            "base": key, "transform": "raw", "window": 20, "norm": 20,
            "params": {"window": 20, "norm": 20}, "path": "",
            "factor_key": f"{LIB}:{fid}",
        })
    return pd.DataFrame(rows)
