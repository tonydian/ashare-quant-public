"""盘中实时行情（新浪财经为主，东方财富涨停池为辅）。

定位：**只用于「看」，绝不参与任何因子、打分、信号计算。**

本项目的策略是日频，全部计算基于收盘后的后复权面板（`panel_ohlcv.npz`）。
盘中价格与面板口径不同（后复权、T 日收盘），混用会污染回测结论。
因此本模块的输出只喂给网页展示层，`qf/` 下任何计算模块都不应 import 它。

数据源选择（2026-11-11 实测）：
- 新浪 `hq.sinajs.cn`：个股与指数实时，稳定，GBK 编码，需 Referer
- 新浪 `vip.stock.finance.sina.com.cn`：全市场排行（可排序分页）、行业板块
- 东财 `push2ex.eastmoney.com`：涨停/跌停池，稳定
- 东财 `push2.eastmoney.com`：行情，**本机实测全节点返回 0 字节**（限流），
  因此只作为可选加速，失败即退回新浪，不作为依赖

涨跌家数为什么用二分：新浪单页上限 100 条、全市场 5341 只 = 54 页，
每次拉全量要 54 次请求，盘中轮询不现实。但该接口支持按涨跌幅降序排序，
所以「上涨家数」= 第一个 `changepercent <= 0` 的绝对位置，
二分约 6 次请求即可精确定位，两次二分共约 12 次。

**口径约定**：所有涨跌幅字段（`chg_pct` / `leader_chg`）一律是**小数**
（0.0046 表示 +0.46%），与项目全局口径一致。网页侧的 `fmtPct()` 内部会 ×100，
如果这里给百分数，前端会放大一百倍。曾经踩过这个坑。

只依赖标准库，不新增任何 pip 依赖。
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
import urllib.request
from datetime import datetime

# ----------------------------------------------------------------- 常量

INDEX_LIST = [
    ("sh000001", "上证指数"),
    ("sz399001", "深证成指"),
    ("sz399006", "创业板指"),
    ("sh000300", "沪深300"),
    ("sh000905", "中证500"),
    ("sh000852", "中证1000"),
]

_HDR_SINA = {
    "Referer": "https://finance.sina.com.cn",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}
_HDR_EM = {
    "Referer": "https://quote.eastmoney.com",
    "User-Agent": _HDR_SINA["User-Agent"],
}

_SINA_Q = "http://hq.sinajs.cn/list={}"
_SINA_RANK = ("http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
              "Market_Center.getHQNodeData?page={page}&num={num}"
              "&sort=changepercent&asc=0&node=hs_a&symbol=")
_SINA_COUNT = ("http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               "Market_Center.getHQNodeStockCount?node=hs_a")
_SINA_HY = "http://vip.stock.finance.sina.com.cn/q/view/newSinaHy.php"

_ZT_UT = "7eea3edcaed734bea9cbfc24409ed989"

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[float, object]] = {}

PAGE_SIZE = 100


# ----------------------------------------------------------------- 基础

def is_trading_time(now: datetime | None = None) -> bool:
    """是否处于连续竞价时段（只判周末 + 时段，不判节假日）。

    节假日会返回上一交易日收盘数据，接口自带日期字段，展示层据此标注即可。
    """
    now = now or datetime.now()
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return (915 <= hm <= 1130) or (1300 <= hm <= 1500)


def _ttl(light: bool = True) -> float:
    """缓存时长。light=轻请求（指数/个股）；重请求（涨跌家数）单独放宽。"""
    if light:
        return 15.0 if is_trading_time() else 300.0
    return 60.0 if is_trading_time() else 900.0


def _get(url: str, headers: dict, timeout: float = 8.0) -> bytes | None:
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _get_json(url: str, headers: dict, timeout: float = 10.0):
    raw = _get(url, headers, timeout)
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None


def _cached(key: str, fn, ttl: float | None = None):
    """TTL 缓存。失败（None）不写缓存，避免一次抖动锁死一个周期。"""
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and now - hit[0] < (ttl if ttl is not None else _ttl()):
            return hit[1]
    val = fn()
    if val is not None:
        with _LOCK:
            _CACHE[key] = (now, val)
    return val


def sina_code(code: str) -> str:
    """代码 → 新浪带市场前缀的代码（股票 + 可转债）。

    沪市转债 110/111/113，深市转债 123/127/128，其余按股票规则。
    """
    c = str(code).strip().zfill(6)
    if c[0] == "1":
        return ("sz" if c[:3] in ("123", "127", "128") else "sh") + c
    if c[0] == "6":
        return "sh" + c
    if c[0] in ("0", "3"):
        return "sz" + c
    if c[0] in ("4", "8"):
        return "bj" + c
    return "sh" + c


# ----------------------------------------------------------------- 行情

_RE_LINE = re.compile(r'hq_str_([a-z]{2}\d{6})="([^"]*)"')


def _parse_sina(text: str) -> dict[str, dict]:
    """新浪批量行情。字段：名称,今开,昨收,现价,最高,最低,...,量(股),额(元),...,日期,时间"""
    out: dict[str, dict] = {}
    for sym, body in _RE_LINE.findall(text):
        parts = body.split(",")
        if len(parts) < 32:
            continue
        try:
            last, prev = float(parts[3]), float(parts[2])
            if last <= 0 or prev <= 0:
                continue  # 停牌 / 未开盘
            out[sym] = {
                "name": parts[0].strip(),
                "open": float(parts[1]),
                "prev_close": prev,
                "last": last,
                "high": float(parts[4]),
                "low": float(parts[5]),
                "volume": float(parts[8]),
                "amount": float(parts[9]),
                "chg": last - prev,
                "chg_pct": last / prev - 1.0,     # 小数口径（0.0046 = +0.46%），与项目全局一致
                "date": parts[30],
                "time": parts[31],
            }
        except (ValueError, IndexError):
            continue
    return out


def fetch_quotes(codes: list[str]) -> dict[str, dict] | None:
    """批量实时行情，返回 {6位代码: quote}。

    单批 30 只、整批失败重试 1 次。若最终拿到的数量不足请求数的一半，
    视为大数据源故障并返回 None（不写缓存，下次重新尝试）。

    为什么要有「不足一半就返回 None」这条：曾经出现过只拿到 9/59 只、
    但因为有 1 批成功就当成正常结果、并被缓存 5 分钟的情况（非交易时段 TTL），
    导致页面上 50 只票凭空消失。部分股票缺失是正常的（停牌不返回），
    但大面积缺失一定是请求层面的问题。
    """
    codes = [str(c).zfill(6) for c in codes if c]
    if not codes:
        return None

    got: dict[str, dict] = {}
    for i in range(0, len(codes), 30):
        batch = codes[i:i + 30]
        parsed: dict[str, dict] = {}
        for _ in range(2):                      # 整批失败重试一次
            raw = _get(_SINA_Q.format(",".join(sina_code(c) for c in batch)), _HDR_SINA)
            if not raw:
                continue
            parsed = _parse_sina(raw.decode("gbk", errors="replace"))
            break
        for c in batch:
            q = parsed.get(sina_code(c))
            if q:
                got[c] = q

    if not got:
        return None
    if len(got) < len(codes) * 0.5:             # 大面积缺失 → 当故障处理，不缓存
        return None
    return got


def fetch_indices() -> list[dict] | None:
    """六大指数实时。"""
    raw = _get(_SINA_Q.format(",".join(s for s, _ in INDEX_LIST)), _HDR_SINA)
    if not raw:
        return None
    parsed = _parse_sina(raw.decode("gbk", errors="replace"))
    out = []
    for sym, label in INDEX_LIST:
        q = parsed.get(sym)
        if not q:
            continue
        out.append({"code": sym[2:], "name": label,
                    "last": q["last"], "chg": q["chg"], "chg_pct": q["chg_pct"],
                    "amount": q["amount"], "time": q["time"]})
    return out or None


# ----------------------------------------------------------------- 全市场统计

def _sina_page(page: int, num: int = PAGE_SIZE) -> list[dict] | None:
    """取按涨跌幅降序的第 page 页（1 起）。"""
    d = _get_json(_SINA_RANK.format(page=page, num=num), _HDR_SINA)
    if not isinstance(d, list) or not d:
        return None
    out = []
    for r in d:
        try:
            out.append({"chg": float(r.get("changepercent") or 0.0),
                        "code": r.get("code"), "name": r.get("name"),
                        "amount": float(r.get("amount") or 0.0)})
        except (TypeError, ValueError):
            continue
    return out or None


def _sina_total() -> int | None:
    raw = _get(_SINA_COUNT, _HDR_SINA)
    if not raw:
        return None
    try:
        return int(json.loads(raw.decode("utf-8", errors="replace")))
    except Exception:
        return None


def _real_total(pages: int) -> int | None:
    """真实条目数。

    新浪 `getHQNodeStockCount` 报的值会偏小（2026-11-11 实测报 5341、实际 5350），
    所以下跌家数不能用「报告总数 - 位置」反算，必须按实际页数取。
    """
    last = _sina_page(pages)
    if last is None:
        return None
    n = (pages - 1) * PAGE_SIZE + len(last)
    if len(last) >= PAGE_SIZE:          # 末页满，说明报告值偏小，往后补查
        nxt = _sina_page(pages + 1)
        if nxt:
            n += len(nxt)
    return n


def _first_pos(pages: int, is_boundary) -> int | None:
    """降序序列里，二分找第一个满足 is_boundary 的**页**（页内只要有一条满足即算）。

    返回页码（1 起）；所有页都不满足时返回 `pages + 1`。
    `hi` 取 pages+1 而不是 pages，否则「全市场都在涨」这种边界无法与
    「分界正好在最后一页」区分开，会被错误地当成后者。
    """
    lo, hi = 1, pages + 1
    while lo < hi:
        mid = (lo + hi) // 2
        if mid > pages:                       # 理论上不会发生，防御性判断
            return pages + 1
        rows = _sina_page(mid)
        if rows is None:
            return None
        if any(is_boundary(r["chg"]) for r in rows):
            hi = mid
        else:
            lo = mid + 1
    return lo


def fetch_breadth() -> dict | None:
    """全市场涨跌家数（新浪二分）+ 涨跌停家数（东财涨停池）。

    二分口径：
      上涨 = 第一个 `chg <= 0` 的绝对位置
      跌+平的下界 = 第一个 `chg < 0` 的绝对位置
    剩余为平盘。
    """
    total = _sina_total()
    if not total:
        return None
    pages = math.ceil(total / PAGE_SIZE)

    p_zero = _first_pos(pages, lambda c: c <= 0)      # 第一个非上涨
    if p_zero is None:
        return None
    real = _real_total(pages) or total
    if p_zero > pages:                                 # 全市场都在涨
        up = real
        flat = 0
        down = 0
    else:
        rows = _sina_page(p_zero)
        if rows is None:
            return None
        idx = next((i for i, r in enumerate(rows) if r["chg"] <= 0), None)
        if idx is None:                                # 该页无越界（数据抖动）
            return None
        up = (p_zero - 1) * PAGE_SIZE + idx

        p_neg = _first_pos(pages, lambda c: c < 0)
        if p_neg is None:
            return None
        if p_neg > pages:
            down = 0
            flat = real - up
        else:
            rows2 = _sina_page(p_neg)
            if rows2 is None:
                return None
            idx2 = next((i for i, r in enumerate(rows2) if r["chg"] < 0), None)
            if idx2 is None:
                return None
            up_flat = (p_neg - 1) * PAGE_SIZE + idx2
            flat = max(up_flat - up, 0)
            down = max(real - up_flat, 0)

    out = {"up": up, "down": down, "flat": flat,
           "total": up + down + flat, "source": "sina"}
    out.update(fetch_limits() or {})
    return out


def fetch_limits() -> dict | None:
    """涨停 / 跌停家数、最高连板（东财涨停池，独立于 push2 行情接口）。"""
    day = datetime.now().strftime("%Y%m%d")
    out: dict = {}
    for kind, key in (("getTopicZTPool", "limit_up"), ("getTopicDTPool", "limit_down")):
        u = (f"http://push2ex.eastmoney.com/{kind}?ut={_ZT_UT}&dpt=wz.ztzt"
             f"&Pageindex=0&pagesize=200&sort=fbt%3Aasc&date={day}")
        z = _get_json(u, _HDR_EM)
        if not z or not z.get("data"):
            continue
        out[key] = int(z["data"].get("tc") or 0)
        if kind == "getTopicZTPool":
            pool = z["data"].get("pool") or []
            streaks = [int(x.get("lbc") or 0) for x in pool]
            if streaks:
                out["max_streak"] = max(streaks)
                top = max(pool, key=lambda x: int(x.get("lbc") or 0))
                out["top_streak"] = {"code": top.get("c"), "name": top.get("n"),
                                     "streak": int(top.get("lbc") or 0)}
    return out or None


def fetch_sectors(top: int = 8) -> list[dict] | None:
    """行业板块实时涨幅榜（新浪 newSinaHy，一次给全部行业，含领涨股）。

    字段：板块代码,名称,公司家数,均价,涨跌额,涨跌幅,成交量,成交额,
          领涨股代码,领涨股涨跌幅,领涨股现价,领涨股涨跌额,领涨股名称
    """
    raw = _get(_SINA_HY, _HDR_SINA)
    if not raw:
        return None
    text = raw.decode("gbk", errors="replace")
    m = re.search(r"=\s*(\{.*\})\s*;?\s*$", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except Exception:
        return None

    rows = []
    for _, v in d.items():
        p = str(v).split(",")
        if len(p) < 13:
            continue
        try:
            rows.append({
                "code": p[0], "name": p[1], "n_stock": int(p[2] or 0),
                "chg_pct": float(p[5] or 0) / 100.0,        # 新浪给的是百分数，转小数
                "amount": float(p[7] or 0),
                "leader_code": p[8][2:] if len(p[8]) > 2 else p[8],
                "leader_chg": float(p[9] or 0) / 100.0,
                "leader_price": float(p[10] or 0),
                "leader": p[12],
            })
        except (ValueError, IndexError):
            continue
    if not rows:
        return None
    rows.sort(key=lambda x: x["chg_pct"], reverse=True)
    return rows[:top]


# ----------------------------------------------------------------- 聚合

def overview(codes: list[str] | None = None) -> dict:
    """实时总览。任一块拿不到就是 None，绝不抛异常。"""
    data = {
        "asof": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "trading": is_trading_time(),
        "indices": _cached("indices", fetch_indices),
        "breadth": _cached("breadth", fetch_breadth, ttl=_ttl(False)),
        "sectors": _cached("sectors", fetch_sectors),
        "quotes": None,
    }
    if codes:
        key = "quotes:" + ",".join(sorted(str(c) for c in codes))[:2000]
        data["quotes"] = _cached(key, lambda: fetch_quotes(codes))
    return data


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()
