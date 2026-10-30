"""page_render.py — PyMuPDF 页图渲染 + bbox 高亮（M1 标定 / M2 Streamlit / M3 API）。

只提供页图，不提供整本 PDF 下载。bbox 假定为 MinerU 的 PDF 点坐标（原点左上），
M1 用 `--calibrate` 对一篇论文实测确认；画错框比不画更糟。
"""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pymupdf

import config

ZOOM = 2.0


def scale_bbox(bbox, zoom=ZOOM):
    """PDF 点 → 渲染像素（原点左上）。"""
    return [float(bbox[0]) * zoom, float(bbox[1]) * zoom,
            float(bbox[2]) * zoom, float(bbox[3]) * zoom]


def bboxes_for_blocks(paper_id, block_ids, page=None):
    """从 normalized JSON 取指定 block_id 的 bbox 列表（页图高亮用）。

    page 给定时只取该页——同一 block_id 不会跨页，但缺页过滤会把别的页框画到本页图上。
    """
    path = Path(config.PAPERS_DIR) / "normalized" / f"{paper_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = {x.strip() for x in str(block_ids or "").split(",") if x.strip()}
    out = []
    for p in data.get("pages", []):
        if page is not None and p.get("page") != page:
            continue
        for b in p.get("blocks", []):
            if b.get("block_id") in ids and b.get("bbox"):
                out.append(b["bbox"])
    return out


def render_page_png(pdf_path, page_no, blocks=None, zoom=ZOOM):
    """渲染第 page_no 页（1 起）；blocks 为 [{"bbox": [...]}, ...] 或 [[x0,y0,x1,y1], ...]。"""
    doc = pymupdf.open(str(pdf_path))
    try:
        if page_no < 1 or page_no > doc.page_count:
            raise ValueError(f"页码越界: {page_no}/{doc.page_count}")
        page = doc[page_no - 1]
        if blocks:
            for item in blocks:
                bbox = item.get("bbox") if isinstance(item, dict) else item
                if not bbox or len(bbox) != 4:
                    continue
                rect = pymupdf.Rect(*[float(x) for x in bbox])
                # 2pt 外扩：MinerU 的 bbox 贴字顶/字底，紧贴会切掉降部（2026-10-29 实测）
                rect = pymupdf.Rect(
                    rect.x0 - 2, rect.y0 - 2, rect.x1 + 2, rect.y1 + 2
                ) & page.rect
                page.draw_rect(rect, color=(1, 0, 0), width=1.5)
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
        return pix.tobytes("png")
    finally:
        doc.close()


def _norm(s):
    return re.sub(r"\s+", "", s or "")


def _find_normalized(pdf_path):
    pdf_name = Path(pdf_path).name
    for p in (Path(config.PAPERS_DIR) / "normalized").glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(data.get("source_pdf", "")).name == pdf_name:
            return p, data
    return None, None


def calibrate(pdf_path, page_no, block_id=None, normalized_path=None):
    """渲染带框页图 + 用 get_text(clip=) 核对 bbox 是否落在同一块文本上。"""
    if normalized_path:
        data = json.loads(Path(normalized_path).read_text(encoding="utf-8"))
    else:
        _, data = _find_normalized(pdf_path)
    if not data:
        raise SystemExit("找不到 normalized JSON；先跑 paper_ingest.py")
    page = next((p for p in data["pages"] if p["page"] == page_no), None)
    if not page:
        raise SystemExit(f"normalized 中没有第 {page_no} 页")
    blocks = page.get("blocks") or []
    if block_id:
        block = next((b for b in blocks if b["block_id"] == block_id), None)
        if not block:
            ids = ", ".join(b["block_id"] for b in blocks[:20])
            raise SystemExit(f"找不到 block {block_id}；该页前 20 个: {ids}")
    else:
        block = next((b for b in blocks if b.get("bbox") and b.get("text")), None)
        if not block:
            raise SystemExit("该页没有带 bbox 的文本块")
    bbox = block["bbox"]
    doc = pymupdf.open(str(pdf_path))
    try:
        raw = doc[page_no - 1].get_text("text", clip=pymupdf.Rect(*bbox))
    finally:
        doc.close()
    ratio = SequenceMatcher(None, _norm(raw), _norm(block["text"])).ratio()
    out = Path(config.PAPERS_DIR) / "logs" / f"calibrate_p{page_no}_{block['block_id']}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render_page_png(pdf_path, page_no, blocks=[block]))
    print(f"page={page_no} block={block['block_id']} bbox={bbox}")
    print(f"clip_text_len={len(raw)} block_text_len={len(block['text'])} ratio={ratio:.3f}")
    print(f"overlay={out}")
    if ratio < 0.5:
        print("[warn] bbox 与该块文本重合度低：坐标系可能不是 PDF 点/原点左上，先别用高亮")
    return out, ratio


def main():
    ap = argparse.ArgumentParser(description="页图渲染 / bbox 标定")
    ap.add_argument("--calibrate", nargs=2, metavar=("PDF", "PAGE"), help="标定模式：PDF 路径与页码")
    ap.add_argument("--block", default=None)
    ap.add_argument("--normalized", default=None)
    ap.add_argument("--out", default=None, help="普通渲染输出 PNG")
    args = ap.parse_args()
    if args.calibrate:
        calibrate(args.calibrate[0], int(args.calibrate[1]), args.block, args.normalized)
        return
    raise SystemExit("用法：python page_render.py --calibrate <pdf> <page> [--block b001] [--normalized path]")


if __name__ == "__main__":
    main()
