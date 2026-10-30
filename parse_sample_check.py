"""parse_sample_check.py — 解析抽检：3 篇 × 5 页，检查归一化块是否含关键内容。

M6 验收脚本（计划 §4.11 的 5 篇×3 页在 M1 只有 3 篇 → 按 3 篇×5 页执行）。
每页给一个 must_contain 关键词与期望块类型，命中即通过；输出合格率，失败退出 1。

用法：python parse_sample_check.py
"""
import json
import sys
from pathlib import Path

import config

NORMALIZED = Path(config.PAPERS_DIR) / "normalized"

CASES = [
    # (paper_id, page, must_contain, expected_type, note)
    ("bdfaa68d8984", 1, "Attention Is All You Need", "title", "标题"),
    ("bdfaa68d8984", 4, "Multi-Head Attention", "title", "小节标题"),
    ("bdfaa68d8984", 6, "Positional Encoding", "title", "小节标题"),
    ("bdfaa68d8984", 7, "Adam", "paragraph", "正文"),
    ("bdfaa68d8984", 6, "<table>", "table", "表格 HTML"),
    ("e9a0d3128767", 1, "Low-Rank Adaptation", "title", "标题"),
    ("e9a0d3128767", 4, "LOW-RANK-PARAMETRIZED UPDATE MATRICES", "title", "小节标题"),
    ("e9a0d3128767", 8, "GPT-3", "paragraph", "正文"),
    ("e9a0d3128767", 10, "optimal rank", "title", "小节标题"),
    ("e9a0d3128767", 11, "frac", "formula", "公式 LaTeX"),
    ("3e021fd8c6c2", 1, "摘要", "paragraph", "中文正文"),
    ("3e021fd8c6c2", 5, "文本分析", "title", "中文小节标题"),
    ("3e021fd8c6c2", 7, "247.11", "paragraph", "中文数字"),
    ("3e021fd8c6c2", 8, "315.08", "table", "中文表格"),
    ("3e021fd8c6c2", 9, "不足", "paragraph", "中文结语"),
]


def _page_text(data, page):
    for p in data.get("pages", []):
        if p.get("page") == page:
            return p.get("blocks") or []
    return []


def main():
    data_cache = {}
    passed = 0
    for pid, page, needle, want_type, note in CASES:
        if pid not in data_cache:
            path = NORMALIZED / f"{pid}.json"
            if not path.exists():
                print(f"✗ {pid} 缺少 normalized（先跑 paper_ingest.py）")
                return 1
            data_cache[pid] = json.loads(path.read_text(encoding="utf-8"))
        blocks = _page_text(data_cache[pid], page)
        hit = any(
            needle.lower() in ((b.get("text") or "") + (b.get("table_html") or "")).lower()
            and (want_type == "table" or b.get("type") == want_type)
            for b in blocks
        )
        passed += hit
        print(f"{'✓' if hit else '✗'} {pid} p{page} [{note}] must_contain={needle!r} type={want_type}")
    n = len(CASES)
    print(f"\n解析抽检：{passed}/{n} = {passed / n:.3f}（目标 1.000）")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
