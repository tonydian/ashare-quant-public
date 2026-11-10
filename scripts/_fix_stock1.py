# -*- coding: utf-8 -*-
"""修复 + 优化：
1) 候选榜读错字段（candidates/holdings，而非 rows）→ 修复
2) 个股页 score/factors 为空：in_universe=False 时也给打分（用 live_scores）
3) 个股页整体优化：加分位条/K线/多周期涨跌更清晰/行业对比/缺失提示
"""
import pathlib

ROOT = pathlib.Path(r"D:\wkf\getrich\ashare_quant")

# ============================================================ ① 候选榜字段
p = ROOT / "web" / "static" / "app.js"
s = p.read_text(encoding="utf-8")
old = """    var rows = d.rows || d.items || d.list || [];"""
new = """    var rows = d.rows || d.items || d.list || d.candidates || d.holdings || [];"""
assert old in s
s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8")
print("① 候选榜字段已修复（candidates/holdings）")

# ============================================================ ② 后端：个股打分与因子贡献
p = ROOT / "web" / "services.py"
s = p.read_text(encoding="utf-8")
i = s.find("    def stock_detail(")
assert i > 0
# 找到 stock_detail 内 score 相关代码
seg_start = i
seg_end = s.find("    def _fundamentals_for(", i)
seg = s[seg_start:seg_end]
print(f"\nstock_detail 长度 {len(seg)} 字符")
# 打印 score 赋值处
for ln, line in enumerate(seg.splitlines(), 1):
    if "score" in line or "factors" in line or "in_universe" in line:
        print(f"  {ln:4d}: {line.strip()[:110]}")
