"""paper_chunker.py — paper_blocks/1 → 页内 chunk（不跨页，page_start == page_end）。

不复用课程版 _split_structured / clean_garbled_chunk：后者白名单不含希腊字母，
会把 α/β/θ 公式块整块丢弃。这里只去控制符/零宽字符，保留 LaTeX。
"""
from __future__ import annotations

import hashlib
import re

import config

_ATOMIC = {"table", "figure", "formula"}
_TEXT = {"title", "paragraph", "list", "reference"}


def clean_paper_text(text):
    text = text or ""
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    kept = []
    for ch in text:
        if ch in "\n\t" or ord(ch) >= 32:
            kept.append(ch)
    return "".join(kept).strip()


def _sentences(text):
    parts = re.split(r"(?<=[。！？!?\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split(text, max_chars):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _pack_text(blocks, max_chars):
    """把一段页内文本块打包成若干正文；超长按句读二次切分。"""
    units = []
    for b in blocks:
        text = clean_paper_text(b.get("text"))
        if not text:
            continue
        if len(text) <= max_chars:
            units.append((text, b))
        else:
            for piece in _sentences(text):
                for sub in (_hard_split(piece, max_chars) if len(piece) > max_chars else [piece]):
                    units.append((sub, b))
    chunks, buf, ids = [], [], []
    size = 0
    for text, block in units:
        add = len(text) + (1 if buf else 0)
        if buf and size + add > max_chars:
            chunks.append((buf, ids))
            buf, ids, size = [], [], 0
            add = len(text)
        buf.append(text)
        ids.append(block["block_id"])
        size += add
    if buf:
        chunks.append((buf, ids))
    return chunks


def _uid(sha, page, seq, text):
    return f"{sha[:16]}:p{page:04d}:c{seq:03d}:{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}"


def _make_chunk(data, page, seq, body, blocks_meta, block_type="text"):
    body = clean_paper_text(body)
    if not body:
        return None
    section = blocks_meta[0].get("section", "") if blocks_meta else ""
    section_level = blocks_meta[0].get("section_level", 0) if blocks_meta else 0
    title = data.get("title", "")
    sha = data.get("sha256", "")
    text = body
    page_content = (section + "\n" + body).strip() if section else body
    embed_text = " | ".join(x for x in (title, section, body) if x)
    return {
        "text": text,
        "page_content": page_content,
        "embed_text": embed_text,
        "metadata": {
            "chunk_uid": _uid(sha, page, seq, text),
            "sha256": sha,
            "paper_id": data.get("paper_id", ""),
            "paper_title": title,
            "authors": data.get("authors", ""),
            "year": str(data.get("year", "")),
            "page_start": page,
            "page_end": page,
            "page": page,
            "section": section,
            "section_level": int(section_level or 0),
            "block_type": block_type,
            "parser": data.get("parser", ""),
            "block_ids": ",".join(b["block_id"] for b in blocks_meta),
            "n_pages": int(data.get("n_pages", 0)),
            "chunk_seq": seq,
            "source": data.get("source_pdf", ""),
            "start_index": page * 10000 + seq,
            "embed_text": embed_text,
        },
    }


def _first_page_text(data):
    pages = data.get("pages") or []
    if not pages:
        return ""
    return " ".join((b.get("text") or "") for b in (pages[0].get("blocks") or []))


def _venue(data):
    """发表期刊/会议：优先取首页引用格式行（中文期刊），否则 arXiv 编号。"""
    text = _first_page_text(data)
    m = re.search(r"\[J/OL\]\.\s*([^,，]+)", text)
    if m:
        return re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", m.group(1)).strip()
    m = re.search(r"arxiv\.org/abs/([\w.\-]+)", data.get("source_url") or "", re.I)
    if m:
        return f"arXiv 预印本（arXiv:{m.group(1)}）"
    return ""


def _citation_line(data):
    pages = data.get("pages") or []
    if not pages:
        return ""
    for b in pages[0].get("blocks") or []:
        t = (b.get("text") or "").strip()
        if "引用格式" in t or ("[J" in t and "引用" in t):
            return t
    return ""


def _authors_from_citation(line):
    """从「引用格式：邢淼 , 田丽 . 标题…」取全部作者（元数据常只有第一作者）。"""
    m = re.match(r"^[^：:]*[：:]\s*(.+?)\s*[.。]\s", line or "")
    if not m:
        return ""
    names = re.sub(r"\s*,\s*", "、", m.group(1)).strip()
    names = re.sub(r"^、|、$", "", names)
    return names if 0 < len(names) <= 80 else ""


def metadata_chunk(data):
    """每篇一条元信息 chunk：标题/作者/期刊/年份（不拼整条引用格式）。

    为什么单独成块：整页 chunk 对「作者/期刊」这类短问句的 rerank 只有 0.09；
    短句块实测 0.29~0.85。embed_text 必须与 page_content 一致——竖线串会让
    reranker 掉到 0.003（2026-11-02 实测）。
    """
    title = (data.get("title") or "").strip()
    if not title:
        return None
    citation = _citation_line(data)
    authors = _authors_from_citation(citation) or (data.get("authors") or "").strip()
    year = str(data.get("year") or "").strip()
    venue = _venue(data)
    parts = [f"论文信息。标题：{title}。"]
    if authors:
        parts.append(f"作者：{authors}。")
    if venue:
        parts.append(f"发表期刊：{venue}。")
    if year:
        parts.append(f"发表年份：{year} 年。")
    body = "".join(parts)

    pages = data.get("pages") or []
    blocks = (pages[0].get("blocks") or []) if pages else []
    anchor = next((b for b in blocks if b.get("type") == "title"), blocks[0] if blocks else {})
    sha = data.get("sha256", "")
    seq = 9999
    return {
        "text": body,
        "page_content": body,
        "embed_text": body,
        "metadata": {
            "chunk_uid": _uid(sha, 1, seq, body),
            "sha256": sha,
            "paper_id": data.get("paper_id", ""),
            "paper_title": title,
            "authors": authors,
            "year": year,
            "page_start": 1,
            "page_end": 1,
            "page": 1,
            "section": "论文信息",
            "section_level": 0,
            "block_type": "metadata",
            "parser": data.get("parser", ""),
            "block_ids": anchor.get("block_id", ""),
            "n_pages": int(data.get("n_pages", 0)),
            "chunk_seq": seq,
            "source": data.get("source_pdf", ""),
            "start_index": 10000 + seq,
            "embed_text": body,
        },
    }


def chunk_paper(data, max_chars=None, min_chars=None):
    """返回 [{text, page_content, embed_text, metadata}]；每页独立，0 跨页。"""
    max_chars = int(max_chars or config.PAPER_CHUNK_MAX_CHARS)
    min_chars = int(min_chars if min_chars is not None else config.PAPER_CHUNK_MIN_CHARS)
    out = []
    for page in data.get("pages", []):
        page_no = int(page["page"])
        blocks = page.get("blocks") or []
        seq = 0
        text_blocks = []
        for b in blocks:
            btype = b.get("type", "paragraph")
            if btype == "table":
                text = clean_paper_text(b.get("table_html")) or clean_paper_text(b.get("text"))
            elif btype == "figure":
                text = clean_paper_text(b.get("text")) or clean_paper_text(b.get("caption"))
            else:
                text = clean_paper_text(b.get("text"))
            if btype in _ATOMIC:
                if text:
                    chunk = _make_chunk(data, page_no, seq, text, [b], btype)
                    if chunk:
                        out.append(chunk)
                        seq += 1
            elif btype in _TEXT and text:
                text_blocks.append(b)
        for buf, ids in _pack_text(text_blocks, max_chars):
            meta_blocks = [b for b in text_blocks if b["block_id"] in ids]
            chunk = _make_chunk(data, page_no, seq, "\n".join(buf), meta_blocks or text_blocks[:1])
            if chunk:
                out.append(chunk)
                seq += 1
        # 页尾过短文本块并入前一块（同页、都是 text）
        if len(out) >= 2 and min_chars > 0:
            last, prev = out[-1], out[-2]
            if (last["metadata"]["page"] == prev["metadata"]["page"]
                    and last["metadata"]["block_type"] == "text"
                    and prev["metadata"]["block_type"] == "text"
                    and len(last["text"]) < min_chars):
                merged = prev["text"] + "\n" + last["text"]
                if len(merged) <= max_chars * 1.2:
                    out.pop()
                    merged_ids = (prev["metadata"]["block_ids"] + "," + last["metadata"]["block_ids"]).strip(",")
                    new = _make_chunk(
                        data, page_no, prev["metadata"]["chunk_seq"], merged,
                        [b for b in text_blocks if b["block_id"] in merged_ids.split(",")],
                    )
                    if new:
                        new["metadata"]["block_ids"] = merged_ids
                        out[-1] = new
    return out


def audit_chunks(chunks):
    """返回 (总数, 缺页数, 跨页数, 缺 section 数)；M2 验收用。"""
    total = len(chunks)
    missing_page = sum(1 for c in chunks if not c["metadata"].get("page_start"))
    cross = sum(1 for c in chunks if c["metadata"].get("page_start") != c["metadata"].get("page_end"))
    no_section = sum(1 for c in chunks if not c["metadata"].get("section"))
    return total, missing_page, cross, no_section
