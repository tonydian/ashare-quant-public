"""修复 `was_stock_data_ext` 增量段（2026-10-15 起）被写坏的复权价。

问题
----
`scripts/update_data.py` 用腾讯 hfq 拼接增量段，写法是 `ext = row * Cj`。
`Cj` 是常数缩放，**不改变日收益率**，所以增量段的收益率完全等于腾讯 hfq 的收益率。
而腾讯 hfq 的收益与真实成交价背离（因股而异，k 从 0.75 到 2.6）：

    000635 英力特  11-05 记 +12.10%，真实 +9.95%（深主板涨停上限 10%，12% 不可能）

修复思路
--------
不复权价的收益率在除权日会跳空，不能直接用（001285 在 10-16 做了 10 转 4.5，
不复权收益记 −28.89%，真实是 +3.09%）。改用**前复权**价的收益率——前复权已经
吸收送股/转增/分红，日收益率等于后复权收益率，是这里最省事也最准的口径。

重建公式（只动 i > T0 的行）：

    P_new[t] = P_old[base] × Qf[t] / Qf[base]

    base = 最后一个 i ≤ T0 且 P_old、Qf 同时有效的交易日（通常就是 T0）
    Qf   = 前复权收盘，沿时间前向填充（停牌日无交易，但累积基准要延续）

同时修正：
    raw_close  ← 东财 fqt=0 真实不复权收盘（原值是 `腾讯价 × Cj × adj_factor`，跟着错）
    market_cap ← 真实不复权收盘 × 每股总股本（股本沿用 T0 前的有效比值）

不改：volume / amount（已验证成交额÷成交量÷不复权价 ≈ 1.002，正常）。

用法
----
    python tools/repair_ext_panel.py --dry-run     # 只看报告，不落盘
    python tools/repair_ext_panel.py               # 真正修复（会覆盖，先备份！）
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

DATA = Path(r"D:\wkf\getrich\data_extract\was_stock_data_ext")
QF = Path(__file__).resolve().parent.parent / "build" / "eval" / "fixpanel_qfq.npz"


def _resolve_t0(given: str | None) -> np.datetime64:
    """T0 = 增量段的起点（最后一个「原始包」交易日）。

    优先用 --t0；否则读未增量更新的原始包 `was_stock_data` 的最后一天。
    原地更新模式（源=目标）下原始包不存在，必须显式传 --t0。
    """
    if given:
        return np.datetime64(given)
    base = DATA.parent / "was_stock_data" / "panel_ohlcv.npz"
    if base.exists() and base.resolve() != (DATA / "panel_ohlcv.npz").resolve():
        bd = np.load(base, allow_pickle=True)["dates"].astype("datetime64[ns]")
        return bd.max()
    raise SystemExit("无法推断 T0（找不到原始包 was_stock_data），请用 --t0 YYYY-MM-DD 指定")


def _ffill(a: np.ndarray) -> np.ndarray:
    """沿时间轴（axis 0）前向填充 NaN。"""
    out = a.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        idx = np.where(np.isfinite(col), np.arange(col.size), 0)
        np.maximum.accumulate(idx, out=idx)
        out[:, j] = col[idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
    ap.add_argument("--t0", default=None,
                    help="增量段起点（原始包最后交易日）；默认自动从 was_stock_data 推断")
    args = ap.parse_args()
    T0 = _resolve_t0(args.t0)

    z = np.load(DATA / "panel_ohlcv.npz", allow_pickle=True)
    dates = z["dates"].astype("datetime64[ns]")
    codes = np.array([str(c) for c in z["codes"]])
    P = z["close"].astype(np.float64)
    O, H, L = (z[k].astype(np.float64) for k in ("open", "high", "low"))
    V, A = z["volume"], z["amount"]

    q = np.load(QF, allow_pickle=True)
    qd = q["dates"].astype("datetime64[ns]")
    qc = np.array([str(c) for c in q["codes"]])
    assert (qc == codes).all(), "前复权面板的股票顺序与主面板不一致"

    # 把前复权窗口对齐到主面板的日期轴
    T, N = P.shape
    Qi = np.full((len(dates), N), np.nan)
    row = {d: i for i, d in enumerate(qd)}
    for t, d in enumerate(dates):
        i = row.get(d)
        if i is not None:
            Qi[t] = q["close"][i].astype(np.float64)
    Qo = np.full((T, N), np.nan)
    Qh = np.full((T, N), np.nan)
    Ql = np.full((T, N), np.nan)
    for t, d in enumerate(dates):
        i = row.get(d)
        if i is not None:
            Qo[t] = q["open"][i].astype(np.float64)
            Qh[t] = q["high"][i].astype(np.float64)
            Ql[t] = q["low"][i].astype(np.float64)

    t0 = int(np.where(dates == T0)[0][0])
    R = np.load(DATA / "panel_raw_ohlc.npz", allow_pickle=True)["close"].astype(np.float64)
    print(f"主面板 {T}×{N}，T0=行{t0} ({str(T0)})，待修行数 {T - t0 - 1}")
    print(f"前复权面板覆盖 {np.isfinite(Qi).mean() * 100:.1f}%")

    # ---- 基准日：i ≤ t0 中最后一个 P、Qf 同时有效的行
    ok = np.isfinite(P[: t0 + 1]) & np.isfinite(Qi[: t0 + 1])
    has = ok.any(axis=0)
    base = t0 - ok[::-1].argmax(axis=0)
    base = np.where(has, base, -1)
    n_base0 = int(((base == t0)).sum())
    n_baseold = int(((base >= 0) & (base < t0)).sum())
    print(f"基准日：T0 当天 {n_base0} 只，更早 {n_baseold} 只，无基准 {int((base < 0).sum())} 只（保持原样）")

    # ---- 前向填充后的前复权收盘，用于累积
    Qi_f = _ffill(Qi)
    base_idx = np.where(base >= 0, base, 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        cum = Qi_f / Qi_f[base_idx, np.arange(N)][None, :]     # 相对基准的累积倍数
    cum = np.where(cum > 0, cum, np.nan)

    # ---- 重建（只动 t > t0）
    P_new = P.copy()
    O_new, H_new, L_new = O.copy(), H.copy(), L.copy()
    sl = slice(t0 + 1, T)
    valid_base = (base >= 0)[None, :]

    anchor = np.where(valid_base, P[base_idx, np.arange(N)][None, :], np.nan)
    rebuilt = anchor * cum
    rebuilt = np.where(np.isfinite(Qi), rebuilt, np.nan)        # 当天无交易 → 置空
    P_new[sl] = rebuilt[sl]

    # open/high/low：保持当日「相对收盘」的比例，比例本身用前复权序列算
    with np.errstate(divide="ignore", invalid="ignore"):
        for src, dst in ((Qo, O_new), (Qh, H_new), (Ql, L_new)):
            ratio = src / Qi
            ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, np.nan)
            dst[sl] = rebuilt[sl] * ratio[sl]

    # ---- 校验
    def rets(x):
        r = np.full_like(x, np.nan)
        r[1:] = x[1:] / x[:-1] - 1
        return r

    is_gem = np.array([c.startswith(("300", "301", "688", "689")) for c in codes])
    lim = np.where(is_gem, 0.206, 0.106)[None, :]
    win = np.zeros(T, bool)
    win[t0 + 1:] = True

    with np.errstate(divide="ignore", invalid="ignore"):
        rp_old, rp_new, rq = rets(P), rets(P_new), rets(Qi)
    m = np.isfinite(rp_new) & np.isfinite(rq) & win[:, None]
    dev = np.abs(rp_new - rq)[m]
    print(f"\n修复后 vs 前复权收益：n={m.sum()}，最大偏差 {dev.max():.2e}"
          f"（应为 0，仅浮点误差）")

    mo = np.isfinite(rp_old) & np.isfinite(rq) & win[:, None]
    dold = np.abs(rp_old - rq)[mo]
    print(f"修复前 vs 前复权收益：中位 {np.median(dold) * 100:.3f}pp，"
          f"p99 {np.percentile(dold, 99) * 100:.3f}pp")

    v_old = (np.isfinite(rp_old) & win[:, None] & (np.abs(rp_old) > lim)).sum()
    v_new = (np.isfinite(rp_new) & win[:, None] & (np.abs(rp_new) > lim)).sum()
    print(f"超涨跌停限制：修复前 {v_old} 笔 → 修复后 {v_new} 笔")

    # ---- 抽查：普通股看最近，除权股看除权当日
    for code, want in (("000635", None), ("001285", "2026-10-16"),
                       ("688300", "2026-10-16"), ("688610", "2026-10-19"),
                       ("688809", "2026-10-23")):
        j = int(np.where(codes == code)[0][0])
        if want is None:
            s = np.where(win & np.isfinite(rp_new[:, j]))[0][-6:]
            print(f"  {code} 近 6 日：", " ".join(f"{rp_new[t, j] * 100:+.2f}%" for t in s))
        else:
            t = int(np.where(dates == np.datetime64(want))[0][0])
            print(f"  {code} {want}（除权日）：修复前 {rp_old[t, j] * 100:+.2f}% "
                  f"→ 修复后 {rp_new[t, j] * 100:+.2f}%")

    # ---- raw_close / market_cap
    rc = np.load(DATA / "raw_close.npz", allow_pickle=True)["rawclose"].astype(np.float64)
    mc = np.load(DATA / "market_cap.npz", allow_pickle=True)["mcap"].astype(np.float64)
    share = np.full(N, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = mc[base_idx, np.arange(N)] / rc[base_idx, np.arange(N)]
    share = np.where(np.isfinite(share) & (share > 0), share, np.nan)

    rc_new = rc.copy()
    rc_new[sl] = np.where(np.isfinite(R[sl]), R[sl], np.nan)
    mc_new = mc.copy()
    mc_new[sl] = rc_new[sl] * share[None, :]

    if args.dry_run:
        print("\n--dry-run：不落盘")
        return

    if not args.no_backup:
        bk = DATA.parent / "_backup_20261110"
        bk.mkdir(exist_ok=True)
        for fn in ("panel_ohlcv.npz", "raw_close.npz", "market_cap.npz"):
            if not (bk / fn).exists():
                shutil.copy2(DATA / fn, bk / fn)
        print(f"\n备份确认：{bk}")

    np.savez_compressed(DATA / "panel_ohlcv.npz",
                        dates=dates, codes=codes,
                        open=O_new.astype(np.float32), high=H_new.astype(np.float32),
                        low=L_new.astype(np.float32), close=P_new.astype(np.float32),
                        volume=V, amount=A)
    np.savez_compressed(DATA / "raw_close.npz", rawclose=rc_new.astype(np.float32))
    np.savez_compressed(DATA / "market_cap.npz", mcap=mc_new.astype(np.float32))
    print("已写入 panel_ohlcv.npz / raw_close.npz / market_cap.npz")


if __name__ == "__main__":
    main()
