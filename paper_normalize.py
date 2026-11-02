"""paper_normalize.py — MinerU 输出 → paper_blocks/1（纯 stdlib；连字修复按需用 PyMuPDF）。

主源 content_list.json（阅读顺序 / 文本 / 图片路径 / 表格 HTML / 公式 LaTeX），
middle.json 补 page_size、标题层级和页眉页脚标注。

坐标（M1 实测 2.7.6，kmf-2024-chinese 27 个正文块 vs PyMuPDF clip 文本 20/27 ≥0.8）：
- content_list.json 的 bbox 是 **0–1000 归一化**坐标（x/page_width*1000），
- middle.json 的 bbox 是 PDF 点（左上原点，与 PyMuPDF 同系）。
归一化统一换算成 PDF 点后写入，`bbox_space="pdf_points"`；渲染侧只认点。
"""
from __future__ import annotations

import difflib
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


# MinerU 对 Type1 字体输出的是字体私有连字码（\x80–\x9c），含义随字体变：
# 同一 \x80 在 ColBERT 标题字体是 ff、正文字体是 fi。前端看不见这些码，
# 表现为 "eﬀicient"→"eicient"、"Quality"→"ality"，且会污染 BM25 与上下文。
# 因此不能写死码表，用同 bbox 的 PyMuPDF 文本逐处对齐还原。
_LEGACY_LO, _LEGACY_HI = 0x7F, 0x9D
_PDF_LIGATURES = {
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
    "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "ft", "\ufb06": "st",
}
_REPAIR_MIN_RATIO = 0.6  # 图块 bbox 抽到的是图内文字而非 caption，对齐会跑偏


def has_legacy_glyph(text):
    return any(_LEGACY_LO <= ord(ch) <= _LEGACY_HI for ch in text or "")


def _strip_legacy(text):
    return "".join(ch for ch in text or "" if not _LEGACY_LO <= ord(ch) <= _LEGACY_HI)


def _pdf_ligatures(text):
    for lig, rep in _PDF_LIGATURES.items():
        text = text.replace(lig, rep)
    return text


def apply_code_map(text, mapping):
    """按字体表替换：唯一展开直接用；多个展开时按词首/词中大小写消歧（t_t vs T_h）。

    剩下的一律去掉（与前端当前可见文本一致）。
    """
    out, prev = [], ""
    for ch in text:
        if _LEGACY_LO <= ord(ch) <= _LEGACY_HI:
            cands = mapping.get(ch, set())
            cands = {cands} if isinstance(cands, str) else set(cands)
            if len(cands) > 1:
                cased = {c for c in cands if c[:1].isupper() == (not prev.isalnum())}
                cands = cased if len(cased) == 1 else set()
            if len(cands) == 1:
                rep = next(iter(cands))
                out.append(rep)
                prev = rep[-1]
            continue
        out.append(ch)
        prev = ch
    return "".join(out)


def font_glyph_map(doc):
    """解析 PDF 字体的 /Encoding /Differences → {码: {展开串}}。

    Type1 字体常只给字形名、没有 ToUnicode（ColBERT 实测），PyMuPDF 兜底会把
    `/T_h` 读成 "T"；Differences 里的 /f_i /T_h 才是权威。同一码在不同字体
    可能不同（\\x80 = fi / ff），所以是集合，唯一时才可直接用。
    """
    expand = {
        "f_i": "fi", "f_f_i": "ffi", "f_f": "ff", "f_l": "fl",
        "f_f_l": "ffl", "f_t": "ft", "t_t": "tt", "Q_u": "Qu", "T_h": "Th",
    }
    merged, seen = {}, set()
    for page in doc:
        for font in page.get_fonts(full=True):
            xref = font[0]
            if xref in seen:
                continue
            seen.add(xref)
            key = doc.xref_get_key(xref, "Encoding")
            if key[0] != "xref":
                continue
            obj = doc.xref_object(int(key[1].split()[0]))
            diff = re.search(r"Differences\s*\[(.*?)\]", obj, re.S)
            if not diff:
                continue
            code = None
            for token in diff.group(1).split():
                if token.startswith("/"):
                    if code is not None:
                        rep = expand.get(token[1:])
                        if rep:
                            merged.setdefault(chr(code), set()).add(rep)
                        code += 1
                else:
                    try:
                        code = int(token)
                    except ValueError:
                        code = None
    return merged


_REP_ORDER = ("fi", "ffi", "ff", "fl", "ffl", "ft", "tt", "Qu", "Th")


def _unique_rep(cands, prev):
    """字体表里唯一确定（或用词首/词中大小写能分开）的展开。"""
    if not cands:
        return None
    if len(cands) == 1:
        return next(iter(cands))
    cased = {c for c in cands if c[:1].isupper() == (not prev.isalnum())}
    return next(iter(cased)) if len(cased) == 1 else None


def _prior_rep(cands):
    """仍然分不开时按英语词频顺序取（fi 远多于 ff；正文 tt 多于粗体 Th 已在大小写步分开）。"""
    if not cands:
        return None
    for rep in _REP_ORDER:
        if rep in cands:
            return rep
    return next(iter(cands))


def _context_pattern(text):
    return "".join(r"\s+" if ch.isspace() else re.escape(ch) for ch in text)


def repair_codes_by_context(text, reference, font_map=None):
    """逐个连字码在 reference（同块或整页 PDF 文本）里按前后文取回展开。

    三档优先级：字体表唯一（含词首/词中大小写分叉）> 上下文匹配 > 词频先验。
    上下文只用每个码附近 ±28 字，整段匹配会因为远端一处差异全盘失败；展开
    至少 2 个字母，避免 PDF 文本本身丢字母时（/T_h → "T"）采信 1 个字母。
    """
    search = re.sub(r"<[^>]+>", " ", text) if "<td" in text else text
    positions = [i for i, ch in enumerate(search) if _LEGACY_LO <= ord(ch) <= _LEGACY_HI]
    context_reps = [None] * len(positions)
    cursor = 0
    for k, pos in enumerate(positions):
        left_start = positions[k - 1] + 1 if k else 0
        right_end = positions[k + 1] if k + 1 < len(positions) else len(search)
        left = search[max(left_start, pos - 28):pos]
        right = search[pos + 1:min(right_end, pos + 1 + 28)]
        if not right and k + 1 < len(positions):
            continue  # 紧跟下一个码：贪心通配会吞掉后文，交给字体表
        group = r"([A-Za-z]{1,4}?)" if right else r"([A-Za-z]{1,4})"
        for left_part in (left, left[-12:]):
            pattern = _context_pattern(left_part) + group + _context_pattern(right)
            match = re.search(pattern, reference[cursor:], re.S) if reference else None
            if not match:
                continue
            if len(match.group(1)) >= 2:
                context_reps[k] = match.group(1)
            cursor += match.start() + len(left_part) + len(match.group(1))
            break

    out, prev, k, resolved = [], "", 0, 0
    for ch in text:
        if not _LEGACY_LO <= ord(ch) <= _LEGACY_HI:
            out.append(ch)
            prev = ch
            continue
        cands = set(font_map.get(ch, ())) if font_map else set()
        rep = _unique_rep(cands, prev)
        if rep is None and context_reps[k]:
            rep = context_reps[k]
        if rep is None:
            rep = _prior_rep(cands)
        k += 1
        if rep:
            out.append(rep)
            prev = rep[-1]
            resolved += 1
    return "".join(out), resolved


def repair_legacy_glyphs(pages, pdf_path):
    """还原 pages 里的字体私有连字码（就地修改并返回统计）。

    两遍：高相似度块用同 bbox 文本逐处对齐；图/表等低相似度块（caption 不在
    图片 bbox 内）改在整页文本里按上下文定位，并用字体 Differences 校正展开。
    仍匹配不上的码去掉并计入 dropped。
    """
    stats = {"blocks": 0, "codes": 0, "repaired": 0, "fallback": 0, "dropped": 0}
    fields = ("text", "caption", "table_html")
    for page in pages:  # 没码就早退，不打开 PDF
        for block in page.get("blocks") or []:
            targets = [f for f in fields if has_legacy_glyph(block.get(f))]
            if not targets:
                continue
            stats["blocks"] += 1
            stats["codes"] += sum(
                1 for f in targets for ch in block[f] if _LEGACY_LO <= ord(ch) <= _LEGACY_HI
            )
    if not stats["codes"] or not pdf_path or not Path(pdf_path).exists():
        return stats
    try:
        import pymupdf
    except ImportError:  # 纯 stdlib 测试环境：保持原样，交给 audit 暴露
        return stats

    pending = []
    with pymupdf.open(pdf_path) as doc:
        font_map = font_glyph_map(doc)
        for page in pages:
            idx = int(page.get("page", 0)) - 1
            if not 0 <= idx < doc.page_count:
                continue
            pdf_page = doc[idx]
            page_text = None
            for block in page.get("blocks") or []:
                targets = [f for f in fields if has_legacy_glyph(block.get(f))]
                if not targets:
                    continue
                field = targets[0]
                source = block[field]
                if "<td" not in source:  # 表格 HTML 按标签切分会改阅读顺序，走整页定位
                    ref = ""
                    if block.get("bbox"):
                        ref = _pdf_ligatures(
                            pdf_page.get_text("text", clip=pymupdf.Rect(block["bbox"]))
                        )
                    ratio = (
                        difflib.SequenceMatcher(
                            None, _strip_legacy(source), ref, autojunk=False
                        ).ratio()
                        if ref else 0.0
                    )
                    if ratio >= _REPAIR_MIN_RATIO:
                        fixed, _ = repair_codes_by_context(source, ref, font_map)
                        if not has_legacy_glyph(fixed):
                            block[field] = fixed
                            stats["repaired"] += 1
                            targets = [f for f in targets if f != field]
                if not targets:
                    continue
                page_text = page_text if page_text is not None else _pdf_ligatures(
                    pdf_page.get_text("text")
                )
                pending.append((block, targets, page_text))

    for block, fields_left, page_text in pending:
        for f in fields_left:
            before = block.get(f) or ""
            total = sum(1 for ch in before if _LEGACY_LO <= ord(ch) <= _LEGACY_HI)
            fixed, _ = repair_codes_by_context(before, page_text, font_map)
            if has_legacy_glyph(fixed):
                fixed = apply_code_map(fixed, font_map)
            left = sum(1 for ch in fixed if _LEGACY_LO <= ord(ch) <= _LEGACY_HI)
            stats["fallback"] += total - left
            stats["dropped"] += left
            block[f] = _strip_legacy(fixed)
    return stats


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


def assign_sections(pages):
    """按标题层级重算每块 section（跨页延续）。

    必须在连字修复之后调用：标题里的 `\\x86` 等码会让 section 永远停在
    "4.2 ality–Cost Tradeo"，引用卡上看得见。
    """
    stack = []  # 章节栈跨页延续：小节标题在上一页时后续页仍归属该小节
    seen_doc_title = False
    for page in pages:
        for block in page.get("blocks") or []:
            block["section"] = " > ".join(t for _, t in stack)
            block["section_level"] = stack[-1][0] if stack else 0
            if block.get("type") != "title" or not block.get("text"):
                continue
            text = block["text"]
            m = _NUM_HEADING.match(text)
            if m:
                level = m.group(1).count(".") + 1
            elif not seen_doc_title:
                seen_doc_title = True  # 首个无编号标题＝论文标题，只做标题不入章节栈
                level = 0
            else:
                level = 1
            if level:
                stack = [t for t in stack if t[0] < level]
                stack.append((level, text))


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
            blocks.append(
                {
                    "block_id": f"p{pidx + 1:04d}_b{len(blocks):03d}",
                    "type": btype,
                    "text": text,
                    "bbox": bbox,
                    "caption": _item_caption(item),
                    "table_html": table_html,
                    "image_path": image_path,
                }
            )
        pages.append({"page": pidx + 1, "width": width, "height": height, "blocks": blocks})

    glyph_report = repair_legacy_glyphs(pages, pdf_path)
    assign_sections(pages)
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
        "ligature_repair": glyph_report,
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
