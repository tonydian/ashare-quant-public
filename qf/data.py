# -*- coding: utf-8 -*-
"""A 股面板数据层。

读取 WAS 数据包（后复权行情、不复权收盘、逐日市值、行业、基准指数），
构造研究/回测所需的干净面板与可交易掩码：

    md.close / open / high / low    后复权价（停牌日前向填充，上市前/退市后 NaN）
    md.volume / amount              成交量/额（停牌补 0）
    md.raw_close / raw_open         不复权价（用于实际价格、整手成本、涨跌停判断）
    md.mcap / logmcap               逐日总市值及其对数（PIT）
    md.industry                     行业索引（-1 未知，前向填充）
    md.alive                        上市且在存续期内
    md.buyable / sellable           执行口径可买/可卖（含涨跌停、停牌、上市天数约束）
    md.eligible                     选股域（t 日收盘可参与打分）
    md.fwd_ret(h) / fwd_open_ret(h) 标签收益（后复权总收益 / 次日开盘执行收益）
    md.bench                        基准指数日收益（hs300, csi500, csi1000, csi_all）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _read_meta() -> pd.DataFrame:
    meta = pd.read_csv(C.META_FILE, dtype={"code": str})
    meta["code"] = meta["code"].str.zfill(6)
    return meta


class MarketData:
    def __init__(self, verbose: bool = True):
        self._load(verbose)

    # ------------------------------------------------------------ 加载
    def _load(self, verbose: bool) -> None:
        z = np.load(C.PANEL_FILE, allow_pickle=False)
        self.dates = pd.DatetimeIndex(z["dates"])
        self.codes = np.array([str(c) for c in z["codes"]], dtype=object)
        self.T, self.N = z["close"].shape
        self.close_orig = z["close"].astype(np.float32)
        self.open_orig = z["open"].astype(np.float32)
        self.high_orig = z["high"].astype(np.float32)
        self.low_orig = z["low"].astype(np.float32)
        self.volume_orig = z["volume"].astype(np.float32)
        self.amount_orig = z["amount"].astype(np.float32)

        raw_raw = np.load(C.RAWCLOSE_FILE)["rawclose"].astype(np.float32)
        mcap = np.load(C.MCAP_FILE)["mcap"].astype(np.float32)
        self.raw_close_orig = raw_raw
        self.mcap_orig = mcap
        # 清洗：非正价格视为缺失（原始后复权价有 33 个 ≤0 格子，见 verifier P2#1）
        for arr in (self.close_orig, self.open_orig, self.high_orig, self.low_orig):
            arr[arr <= 0] = np.nan
        self.raw_close_orig[self.raw_close_orig <= 0] = np.nan
        self._build_alive()
        self._build_filled()

        self.industry_orig = np.load(C.INDUSTRY_FILE).astype(np.int16)
        self.industry_cats = [s for s in open(C.INDUSTRY_CATS, encoding="utf-8").read().splitlines() if s]
        self._build_industry()
        self._build_meta()
        self._build_limits()
        self._build_universe()
        self._build_bench()
        if verbose:
            print(f"[data] {self.T} 交易日 × {self.N} 只  {self.dates[0].date()} ~ {self.dates[-1].date()}")
            print(f"[data] alive 覆盖 {self.alive.mean():.1%}；可买 {self.buyable.mean():.1%}；"
                  f"选股域 {self.eligible.mean():.1%}")

    # ------------------------------------------------------------ 基础掩码
    def _build_alive(self) -> None:
        fin = np.isfinite(self.close_orig) & (self.close_orig > 0)
        self.first_valid = np.where(fin.any(axis=0), fin.argmax(axis=0), self.T)
        # last valid: 反向 argmax
        rev = fin[::-1]
        has = rev.any(axis=0)
        last = np.where(has, self.T - 1 - rev.argmax(axis=0), -1)
        self.last_valid = last
        idx = np.arange(self.T)[:, None]
        self.alive = (idx >= self.first_valid[None, :]) & (idx <= self.last_valid[None, :])
        # 正常交易（非停牌）
        self.is_suspended = self.alive & ~fin
        T, N = self.T, self.N
        self.list_days = np.zeros((T, N), dtype=np.int32)
        self.list_days[self.alive] = np.minimum(
            idx - self.first_valid[None, :] + 1, 10000
        )[self.alive].astype(np.int32)
        self.fin_orig = fin

    @staticmethod
    def _ffill(a: np.ndarray, alive: np.ndarray) -> np.ndarray:
        """沿时间前向填充，但 alive=False 的位置保持 NaN。"""
        T, N = a.shape
        x = np.where(np.isfinite(a), a, np.nan).astype(np.float32)
        out = np.where(alive, x, np.nan).copy()
        # 用 pandas 前向填充（按列独立，12.8M 单元格，一次性开销可接受）
        df = pd.DataFrame(out)
        df = df.ffill()
        # 上市前保持 NaN
        out = df.to_numpy(dtype=np.float32)
        return np.where(alive, out, np.nan)

    def _build_filled(self) -> None:
        self.close = self._ffill(self.close_orig, self.alive)
        self.open = self._ffill(self.open_orig, self.alive)
        self.high = self._ffill(self.high_orig, self.alive)
        self.low = self._ffill(self.low_orig, self.alive)
        self.raw_close = self._ffill(self.raw_close_orig, self.alive)
        # 不复权开盘价 = 复权开盘价 × (raw_close/close)；停牌/缺失处前向填充
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(np.isfinite(self.raw_close_orig) & np.isfinite(self.close_orig) & (self.close_orig > 0),
                             self.raw_close_orig / self.close_orig, np.nan)
        self.adj_factor = self._ffill(ratio, self.alive)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.raw_open = self.open * self.adj_factor
        self.raw_open = self._ffill(self.raw_open, self.alive)
        # ---- 价格口径（backtest-guard 高危修复）：
        #  * md.price 等 = 常数缩放后复权价（≈最后一个有效交易日的实际价），
        #    收益率与旧口径完全一致，继续用于估值/整手/名义额/止损 —— 会计口径不变，
        #    这样 NAV 复现、审计测试、成本口径都不受影响。
        #  * md.price_exec / price_exec_open = **point-in-time 实际价**
        #    （close × 当日 raw_close/close，停牌 ffill），**只用于价格带 2~200 元过滤**。
        #    旧口径把"最后有效日"的复权比例用于全历史价格带 → 用了未来分红/送转信息
        #    （前视）；改后价格带只用 t-1 日已知的实际价。
        scale = np.ones(self.N, dtype=np.float64)
        for j in range(self.N):
            lv = self.last_valid[j]
            if lv >= 0:
                a = self.close[lv, j]
                r = self.raw_close[lv, j]
                if np.isfinite(a) and np.isfinite(r) and a > 0 and r > 0:
                    scale[j] = r / a
        self.price_scale = scale.astype(np.float32)
        self.price = (self.close.astype(np.float64) * scale[None, :]).astype(np.float32)
        self.price_open = (self.open.astype(np.float64) * scale[None, :]).astype(np.float32)
        self.price_high = (self.high.astype(np.float64) * scale[None, :]).astype(np.float32)
        self.price_low = (self.low.astype(np.float64) * scale[None, :]).astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio_pit = np.where(
                np.isfinite(self.raw_close_orig) & np.isfinite(self.close_orig) & (self.close_orig > 0),
                self.raw_close_orig / self.close_orig, np.nan)
        self.adj_ratio_pit = self._ffill(ratio_pit, self.alive).astype(np.float32)
        self.price_exec = (self.close.astype(np.float64) * self.adj_ratio_pit).astype(np.float32)
        self.price_exec_open = (self.open.astype(np.float64) * self.adj_ratio_pit).astype(np.float32)
        self.price_scaled_legacy = self.price  # 兼容别名（对照用）
        # 日收益（= 后复权总收益，含分红再投资）——必须用 close，不能用 PIT 缩放价：
        # PIT 比例在分红/送转日会跳变，直接相除会把"复权因子变化"误算成涨跌。
        self.ret_daily = np.full((self.T, self.N), np.nan, dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.ret_daily[1:] = (self.close[1:] / self.close[:-1] - 1.0).astype(np.float32)

        vol = np.where(self.alive, np.where(np.isfinite(self.volume_orig), self.volume_orig, 0.0), np.nan)
        amt = np.where(self.alive, np.where(np.isfinite(self.amount_orig), self.amount_orig, 0.0), np.nan)
        self.volume = vol.astype(np.float32)
        self.amount = amt.astype(np.float32)
        self.mcap = self._ffill(self.mcap_orig, self.alive)

    def _build_industry(self) -> None:
        ind = self.industry_orig.copy()
        # 前向填充行业（同一只股票的行业归属），未知保持 -1
        df = pd.DataFrame(ind).replace(-1, np.nan).ffill()
        ind2 = df.to_numpy(dtype=np.float32)
        ind2 = np.where(np.isfinite(ind2), ind2, -1).astype(np.int16)
        self.industry = np.where(self.alive, ind2, -1).astype(np.int16)

    def _build_meta(self) -> None:
        meta = _read_meta()
        idx = {c: i for i, c in enumerate(meta["code"])}
        order = np.array([idx.get(c, -1) for c in self.codes])
        m = meta.iloc[[i for i in order if i >= 0]].copy() if (order >= 0).any() else meta.iloc[:0].copy()
        # 对齐到面板顺序（缺失的代码补空行）
        m = meta.reindex(order).reset_index(drop=True)
        m.index = self.codes
        self.meta = m
        self.names = m["name"].fillna("").astype(str).to_numpy()
        self.delist_date = pd.to_datetime(m["delist_date"], errors="coerce").to_numpy()
        self.list_date = pd.to_datetime(m["list_date"], errors="coerce").to_numpy()
        self.is_delisted = m["is_delisted"].fillna(False).astype(bool).to_numpy()
        # 板块
        board = np.full(self.N, "MAIN", dtype=object)
        for j, c in enumerate(self.codes):
            if c.startswith(C.STAR_PREFIX):
                board[j] = "STAR"
            elif c.startswith(C.CHINEXT_PREFIX):
                board[j] = "CHINEXT"
        self.board = board
        # ST / 退市整理（名称口径，非严格 PIT，仅作稳健性过滤，见 README 说明）
        nm = self.names
        self.name_st = np.array([("ST" in s) or ("退" in s) for s in nm])

        # 用 meta 的真实上市日期修正 list_days（_build_alive 里从面板起点计，会把
        # 面板开始前已上市的老股误判为"次新"；这里修正，使 2017-01 起老股即可入选）。
        T = self.T
        idx = np.arange(T)[:, None]
        ld = pd.to_datetime(pd.Series(self.list_date))
        first_row = np.searchsorted(self.dates.values, ld.to_numpy())
        first_row = np.where(np.isfinite(first_row), first_row, self.first_valid)
        first_row = np.maximum(first_row.astype(np.int64), self.first_valid)
        pre_listed = first_row <= 1
        base = np.where(pre_listed, 500, 0)
        days = idx - first_row[None, :] + 1 + base[None, :]
        self.list_days = np.where(self.alive, np.clip(days, 0, 100000), 0).astype(np.int32)

    def _build_limits(self) -> None:
        """涨跌停价：基于前一日不复权收盘价与板块涨跌幅规则。"""
        T, N = self.T, self.N
        rc = np.where(self.alive, self.raw_close, np.nan)
        prev = np.full_like(rc, np.nan)
        prev[1:] = rc[:-1]
        # 停牌导致的前值为 NaN：用最近有效值填充（此处用 pandas ffill 后 shift 已足够）
        prev = pd.DataFrame(prev).ffill().to_numpy(dtype=np.float32)
        pct = np.full(N, 0.10, dtype=np.float32)
        pct[self.board == "STAR"] = 0.20
        pct[self.board == "CHINEXT"] = 0.20
        # 创业板 2020-08-24 前为 10%
        chi = np.array([b == "CHINEXT" for b in self.board])
        pre = self.dates < pd.Timestamp(C.CHINEXT_20PCT_DATE)
        pct_mat = np.tile(pct[None, :], (T, 1))
        pct_mat[pre[:, None] & chi[None, :]] = 0.10
        # 2026 年 ST 无 PIT 数据：不做 5% 特殊处理（回测中 ST 由名称/价格过滤兜底）
        limit_up = np.floor(prev * (1.0 + pct_mat) * 100 + 0.5) / 100.0
        limit_dn = np.floor(prev * (1.0 - pct_mat) * 100 + 0.5) / 100.0
        self.prev_raw_close = prev.astype(np.float32)
        self.limit_pct = pct_mat.astype(np.float32)
        self.limit_up = limit_up.astype(np.float32)
        self.limit_down = limit_dn.astype(np.float32)

        op = self.raw_open
        eps = np.maximum(np.abs(prev) * 0.002, 0.005)
        self.limit_up_at_open = np.isfinite(op) & np.isfinite(limit_up) & (op >= limit_up - eps)
        self.limit_dn_at_open = np.isfinite(op) & np.isfinite(limit_dn) & (op <= limit_dn + eps)
        cl = self.raw_close
        self.limit_up_at_close = np.isfinite(cl) & np.isfinite(limit_up) & (cl >= limit_up - eps)
        self.limit_dn_at_close = np.isfinite(cl) & np.isfinite(limit_dn) & (cl <= limit_dn + eps)

    # ------------------------------------------------------------ 选股域/可交易性
    def _build_universe(self) -> None:
        T, N = self.T, self.N
        amt20 = pd.DataFrame(np.where(self.alive, self.amount, np.nan)).rolling(
            20, min_periods=10).mean().to_numpy(dtype=np.float32)
        self.amount_20d = amt20
        with np.errstate(invalid="ignore"):
            self.logmcap = np.where(self.mcap > 0, np.log(self.mcap), np.nan).astype(np.float32)
        # 价格过滤用 t-1 日缩放后实际价（避免用当日收盘决定当日是否可买）
        # 价格带用 **PIT 实际价**（backtest-guard 高危修复：旧口径用末日常数缩放价，含未来复权信息）
        # ⚠️ price_band_source='legacy' 仅供 backtest-guard 修复前后对照，
        #    **禁止用于正式回测/报告**（legacy 带存在价格带前视）；默认 'pit'。
        band_px = self.price_scaled_legacy if getattr(self, "price_band_source", "pit") == "legacy" else self.price_exec
        rc_prev = pd.DataFrame(np.where(self.alive, band_px, np.nan)).shift(1).to_numpy(dtype=np.float32)
        mcap_prev = pd.DataFrame(np.where(self.alive, self.mcap, np.nan)).shift(1).to_numpy(dtype=np.float32)
        liq_ok = np.isfinite(amt20) & (amt20 >= C.MIN_AMOUNT_20D)
        price_ok = np.isfinite(rc_prev) & (rc_prev >= C.MIN_PRICE) & (rc_prev <= C.MAX_PRICE)
        mcap_ok = np.isfinite(mcap_prev) & (mcap_prev >= C.MIN_MCAP) & (mcap_prev <= C.MAX_MCAP)
        list_ok = self.list_days >= C.MIN_LIST_DAYS
        st_ok = ~np.tile(self.name_st[None, :], (T, 1)) if C.EXCLUDE_ST else np.ones((T, N), dtype=bool)
        not_susp = ~self.is_suspended
        self.eligible = (
            self.alive & liq_ok & price_ok & mcap_ok & list_ok & st_ok & not_susp
        )
        # 实际下单口径：次日开盘可买/可卖
        op_ok = np.isfinite(self.raw_open) & (self.raw_open > 0)
        self.buyable = self.eligible & op_ok & ~self.limit_up_at_open
        self.sellable = self.alive & not_susp & op_ok & ~self.limit_dn_at_open

    # ------------------------------------------------------------ 标签/基准
    def fwd_ret(self, h: int = 5) -> np.ndarray:
        """t 日收盘 → t+h 日收盘 的后复权总收益。"""
        c = self.close.astype(np.float64)
        fwd = np.full_like(c, np.nan)
        if h < self.T:
            fwd[:-h] = c[h:] / c[:-h] - 1.0
        return fwd.astype(np.float32)

    def fwd_open_ret(self, h: int = 5) -> np.ndarray:
        """t+1 日开盘买入 → t+1+h 日开盘卖出 的总收益（与周频调仓一致）。"""
        o = self.price_open.astype(np.float64)
        fwd = np.full((self.T, self.N), np.nan, dtype=np.float32)
        if h + 1 < self.T:
            buy = o[1:self.T - h]
            sell = o[1 + h:]
            with np.errstate(invalid="ignore", divide="ignore"):
                fwd[:self.T - h - 1] = sell / buy - 1.0
        return fwd

    def _build_bench(self) -> None:
        idx = pd.read_csv(C.INDICES_FILE, index_col=0, parse_dates=True)
        idx = idx.reindex(self.dates).ffill()
        self.bench_prices = idx
        self.bench = idx.pct_change().fillna(0.0)
        # 全 A 等权基准（含退市股，生存偏差已消除）
        with np.errstate(invalid="ignore", divide="ignore"):
            r = self.price.astype(np.float64) / pd.DataFrame(self.price).shift(1).to_numpy(dtype=np.float64) - 1.0
        w = self.alive.astype(np.float64)
        w = np.where(np.isfinite(r), w, 0.0)
        num = np.nansum(np.where(w > 0, r, 0.0) * w, axis=1)
        den = w.sum(axis=1)
        eq = np.where(den > 0, num / np.maximum(den, 1), 0.0)
        self.bench["all_eq"] = eq

    # ------------------------------------------------------------ 工具
    def code_index(self, code: str) -> int:
        hits = np.flatnonzero(self.codes == code)
        return int(hits[0]) if len(hits) else -1

    def date_index(self, dt) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(dt)))

    def industry_name(self, k: int) -> str:
        if 0 <= k < len(self.industry_cats):
            return self.industry_cats[k]
        return "未知"

    def holding_days(self, asof: int) -> np.ndarray:
        return self.list_days[asof]


def load_market(verbose: bool = True) -> MarketData:
    return MarketData(verbose=verbose)
