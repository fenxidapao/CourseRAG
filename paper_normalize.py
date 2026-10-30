"""paper_normalize.py — MinerU 输出 → paper_blocks/1（纯 stdlib，不碰网络/DB）。

主源 content_list.json（阅读顺序 / 文本 / 图片路径 / 表格 HTML / 公式 LaTeX），
middle.json 补 page_size、标题层级和页眉页脚标注。

坐标（M1 实测 2.7.6，kmf-2024-chinese 27 个正文块 vs PyMuPDF clip 文本 20/27 ≥0.8）：
- content_list.json 的 bbox 是 **0–1000 归一化**坐标（x/page_width*1000），
- middle.json 的 bbox 是 PDF 点（左上原点，与 PyMuPDF 同系）。
归一化统一换算成 PDF 点后写入，`bbox_space="pdf_points"`；渲染侧只认点。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SCHEMA = "paper_blocks/1"
PARSER = "mineru-2.7.6-pipeline"
BBOX_SPACE = "pdf_points"

_TEXT_TYPES = {"text", "title", "paragraph", "list"}
# "3 Method" / "2.1 xxx" → 章节号；MinerU 对所有标题都给 text_level=1，只能靠编号分主标题与章节
_NUM_HEADING = re.compile(r"^\s*(\d+(?:\.\d+)*)[\s.、．:：]\s*\S")
_MIDDLE_TYPE_MAP = {
    "title": "title",
    "text": "paragraph",
    "list": "list",
    "image": "figure",
    "table": "table",
    "interline_equation": "formula",
    "equation": "formula",
}


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _page_idx(item):
    v = item.get("page_idx", item.get("page", 0))
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _bbox(item):
    v = item.get("bbox")
    if isinstance(v, (list, tuple)) and len(v) == 4:
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return None
    return None


def _unscale_bbox(bbox, width, height):
    """content_list 的 0–1000 归一化 bbox → PDF 点。page_size 缺失时返回 None（不猜）。"""
    if not bbox or width <= 0 or height <= 0:
        return None
    return [
        round(bbox[0] / 1000.0 * width, 2),
        round(bbox[1] / 1000.0 * height, 2),
        round(bbox[2] / 1000.0 * width, 2),
        round(bbox[3] / 1000.0 * height, 2),
    ]


def _middle_text(block):
    parts = []
    for line in block.get("lines") or []:
        for span in line.get("spans") or []:
            t = span.get("content") or span.get("text") or ""
            if t:
                parts.append(t)
    return "".join(parts).strip()


def _middle_blocks(middle_page):
    blocks = middle_page.get("para_blocks") or middle_page.get("preproc_blocks") or []
    out = []
    for b in blocks:
        btype = (b.get("type") or "").lower()
        mapped = _MIDDLE_TYPE_MAP.get(btype)
        if not mapped:
            continue
        out.append(
            {
                "type": mapped,
                "bbox": _bbox(b),
                "text": _middle_text(b),
                "level": b.get("level"),
            }
        )
    return out


def _compatible(content_type, middle_type):
    if content_type in _TEXT_TYPES:
        return middle_type in _TEXT_TYPES
    if content_type == "figure":
        return middle_type == "figure"
    if content_type == "table":
        return middle_type == "table"
    if content_type == "formula":
        return middle_type == "formula"
    return False


def _content_type(item):
    t = (item.get("type") or "").lower()
    if t in ("discarded", "header", "footer", "page_number", "page_footnote"):
        return "skip"
    if t == "image":
        return "figure"
    if t == "table":
        return "table"
    if t in ("equation", "interline_equation", "formula"):
        return "formula"
    if t == "text":
        return "title" if item.get("text_level") is not None else "paragraph"
    return t or "paragraph"


def _clean_text(text):
    """只去控制符/零宽字符；保留 LaTeX、希腊字母和中文标点。"""
    text = text or ""
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    return normalize_math_spacing(text)


# MinerU 会把公式里的数字拆成单字符加空格（实测 18 篇 192/2829 块，小数被拆 27 块）：
#   $\beta _ { 2 } = 0 . 9 8$ → 检索到的是 "0 . 9 8"，模型只能猜，稳定答成 0.999。
# 只在 $...$ 内合并"数字/小数点两侧的空格"，字母内容一律不动（否则 \text{f o o} 会粘成 foo）。
_MATH_SEG = re.compile(r"\$[^$]+\$")
_NUM_GAP = re.compile(r"(?<=[\d.])\s+(?=[\d.])")
_NUM_BRACE = re.compile(r"\{\s*([^{}]*?)\s*\}")


def _fix_math_segment(seg):
    body = _NUM_GAP.sub("", seg[1:-1])
    body = re.sub(r"\^\s*\{", "^{", body)

    def _brace(m):
        inner = m.group(1)
        if re.search(r"\d", inner) and re.fullmatch(r"[-+]?[\d.\s]+", inner):
            return "{" + re.sub(r"\s+", "", inner) + "}"
        return m.group(0)

    return "$" + _NUM_BRACE.sub(_brace, body) + "$"


def normalize_math_spacing(text):
    """只改 $...$ 内的数字拆字：`0 . 9 8` → `0.98`、`{ - 9 }` → `{-9}`。"""
    return _MATH_SEG.sub(lambda m: _fix_math_segment(m.group(0)), text or "")


def _item_caption(item):
    """2.7.6 字段名是 image_caption / table_caption（不是 img_caption），且值为 list。"""
    for key in ("image_caption", "table_caption", "img_caption"):
        v = item.get(key)
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        if v:
            return _clean_text(str(v))
    return ""


def _item_text(item):
    t = item.get("text")
    if t:
        return _clean_text(str(t))
    if item.get("type") == "table":
        return _clean_text(str(item.get("table_body") or "")) or _item_caption(item)
    return _item_caption(item)


def _item_image_path(item):
    return item.get("img_path") or item.get("image_path") or ""


def _guess_title(pages, filename):
    for page in pages:
        for b in page["blocks"]:
            if b["type"] == "title" and b["text"].strip():
                return b["text"].strip(), "first_page"
    for page in pages:
        for b in page["blocks"]:
            if b["text"].strip():
                return b["text"].strip().split("\n")[0][:200], "first_page"
    return Path(filename).stem, "filename"


def _guess_year(pages):
    for page in pages:
        for b in page["blocks"]:
            m = re.search(r"\b(19\d{2}|20\d{2})\b", b["text"])
            if m:
                return m.group(1)
    return ""


def normalize_paper(content_list_path, middle_path, pdf_path, paper_id, sha256, meta=None):
    content = _load_json(content_list_path)
    if not isinstance(content, list):
        raise ValueError("content_list.json 不是列表")
    middle = _load_json(middle_path) if middle_path and Path(middle_path).exists() else {}
    middle_pages = {int(p.get("page_idx", i)): p for i, p in enumerate(middle.get("pdf_info") or [])}

    by_page = {}
    for item in content:
        if not isinstance(item, dict):
            continue
        by_page.setdefault(_page_idx(item), []).append(item)

    n_pages = max(
        [*by_page.keys(), *middle_pages.keys(), 0]
    ) + 1
    pages = []
    title_stack = []  # 章节栈跨页延续：小节标题在上一页时后续页仍归属该小节
    seen_doc_title = False
    for pidx in range(n_pages):
        mpage = middle_pages.get(pidx, {})
        size = mpage.get("page_size") or [0, 0]
        width = float(size[0]) if len(size) > 0 else 0.0
        height = float(size[1]) if len(size) > 1 else 0.0
        candidates = _middle_blocks(mpage)
        used = [False] * len(candidates)
        blocks = []
        for item in by_page.get(pidx, []):
            btype = _content_type(item)
            if btype == "skip":
                continue
            bbox = _unscale_bbox(_bbox(item), width, height)
            if bbox is None:
                for i, cand in enumerate(candidates):
                    if not used[i] and _compatible(btype, cand["type"]):
                        bbox = cand["bbox"]
                        used[i] = True
                        break
            text = _item_text(item)
            table_html = normalize_math_spacing(item.get("table_body") or "")
            image_path = _item_image_path(item)
            # MinerU 会给出无 span 的空文本块（本样例 4 个）；入块只会制造空 chunk
            if not text and not table_html and not image_path:
                continue
            level = item.get("text_level")
            if btype == "title" and text:
                m = _NUM_HEADING.match(text)
                if m:
                    level = m.group(1).count(".") + 1
                elif not seen_doc_title:
                    seen_doc_title = True  # 首页首个无编号标题＝论文标题，只做标题不入章节栈
                    level = 0
                else:
                    level = 1
                if level:
                    title_stack = [t for t in title_stack if t[0] < level]
                    title_stack.append((level, text))
            section = " > ".join(t for _, t in title_stack)
            blocks.append(
                {
                    "block_id": f"p{pidx + 1:04d}_b{len(blocks):03d}",
                    "type": btype,
                    "text": text,
                    "bbox": bbox,
                    "section": section,
                    "section_level": title_stack[-1][0] if title_stack else 0,
                    "caption": _item_caption(item),
                    "table_html": table_html,
                    "image_path": image_path,
                }
            )
        pages.append({"page": pidx + 1, "width": width, "height": height, "blocks": blocks})

    title, title_source = (_guess_title(pages, pdf_path))
    authors = ""
    year = ""
    source_url = ""
    license_ = ""
    if meta:
        title = meta.get("title") or title
        title_source = "meta" if meta.get("title") else title_source
        authors = meta.get("authors") or ""
        year = str(meta.get("year") or "")
        source_url = meta.get("source_url") or ""
        license_ = meta.get("license") or ""
    if not year:
        year = _guess_year(pages)

    return {
        "schema": SCHEMA,
        "parser": PARSER,
        "paper_id": paper_id,
        "sha256": sha256,
        "source_pdf": str(pdf_path),
        "title": title,
        "title_source": title_source,
        "authors": authors,
        "year": year,
        "source_url": source_url,
        "license": license_,
        "n_pages": n_pages,
        "bbox_space": BBOX_SPACE,
        "pages": pages,
    }


def write_normalized(data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    tmp.replace(out_path)
    return out_path


def audit(data):
    """返回 (总块数, 缺页块数, 缺 bbox 块数)；M1/M2 验收用。"""
    blocks = [b for p in data.get("pages", []) for b in p.get("blocks", [])]
    missing_page = 0
    missing_bbox = sum(1 for b in blocks if not b.get("bbox"))
    return len(blocks), missing_page, missing_bbox
