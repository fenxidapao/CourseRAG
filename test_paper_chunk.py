"""test_paper_chunk.py — 页内切分契约（零外部依赖）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_chunker import audit_chunks, chunk_paper, clean_paper_text, metadata_chunk


def _data():
    return {
        "schema": "paper_blocks/1",
        "parser": "mineru-2.7.6-pipeline",
        "paper_id": "abc123",
        "sha256": "f" * 64,
        "source_pdf": "data/papers/raw/test.pdf",
        "title": "Test Paper",
        "authors": "A. Author",
        "year": 2026,
        "n_pages": 2,
        "pages": [
            {
                "page": 1,
                "width": 600,
                "height": 800,
                "blocks": [
                    {"block_id": "p0001_b000", "type": "title", "text": "1 Introduction",
                     "section": "1 Introduction", "section_level": 1, "bbox": [1, 2, 3, 4]},
                    {"block_id": "p0001_b001", "type": "paragraph",
                     "text": "Body α β θ with formula \\frac{a}{b}. " * 8,
                     "section": "1 Introduction", "section_level": 1, "bbox": [1, 5, 3, 6]},
                    {"block_id": "p0001_b002", "type": "table",
                     "text": "Table 1 results", "table_html": "<table><tr><td>1</td></tr></table>",
                     "section": "1 Introduction", "section_level": 1, "bbox": [1, 7, 3, 8]},
                    {"block_id": "p0001_b003", "type": "figure", "text": "",
                     "caption": "Figure 1 architecture", "section": "1 Introduction",
                     "section_level": 1, "bbox": [1, 9, 3, 10]},
                ],
            },
            {
                "page": 2,
                "width": 600,
                "height": 800,
                "blocks": [
                    {"block_id": "p0002_b000", "type": "paragraph", "text": "Second page content.",
                     "section": "1 Introduction", "section_level": 1, "bbox": [1, 2, 3, 4]},
                ],
            },
        ],
    }


def test_chunk_paper_page_contract():
    data = _data()
    chunks = chunk_paper(data, max_chars=100, min_chars=10)
    total, missing_page, cross, _ = audit_chunks(chunks)
    assert total >= 3
    assert missing_page == 0
    assert cross == 0
    pages = {c["metadata"]["page"] for c in chunks}
    assert pages == {1, 2}
    assert all(c["metadata"]["page_start"] == c["metadata"]["page_end"] for c in chunks)
    assert any(c["metadata"]["block_type"] == "table" for c in chunks)
    table = next(c for c in chunks if c["metadata"]["block_type"] == "table")
    assert "<table>" in table["text"]
    figure = next(c for c in chunks if c["metadata"]["block_type"] == "figure")
    assert "architecture" in figure["text"]
    assert all("Test Paper" in c["embed_text"] for c in chunks)
    assert any("α" in c["text"] and "θ" in c["text"] for c in chunks)


def test_chunk_uid_stable_and_clean_keeps_latex():
    a = chunk_paper(_data(), max_chars=100, min_chars=10)
    b = chunk_paper(_data(), max_chars=100, min_chars=10)
    assert [c["metadata"]["chunk_uid"] for c in a] == [c["metadata"]["chunk_uid"] for c in b]
    cleaned = clean_paper_text("x\u200b\\frac{α}{β}\x00 y")
    assert "\x00" not in cleaned and "\\frac{α}{β}" in cleaned


def test_metadata_chunk_contract():
    """元信息块：字段标签进正文、embed_text 与正文一致、锚定首页标题块。

    embed_text 必须等于 page_content：竖线串会让 reranker 从 0.85 掉到 0.003（实测）。
    """
    data = {
        "paper_id": "cn001", "sha256": "a" * 64, "title": "中文摘要对比研究",
        "authors": "邢淼", "year": 2024, "n_pages": 3,
        "source_url": "https://www.kmf.ac.cn/p/404/",
        "pages": [{"page": 1, "blocks": [
            {"block_id": "p0001_b000", "type": "title", "text": "中文摘要对比研究"},
            {"block_id": "p0001_b001", "type": "paragraph",
             "text": "引用格式：邢淼 , 田丽 . 中文摘要对比研究 [J/OL].知 识 管 理 论 坛 , 2024, 9(5): 437-447."},
        ]}],
    }
    c = metadata_chunk(data)
    assert c["metadata"]["block_type"] == "metadata"
    assert c["metadata"]["page"] == 1 and c["metadata"]["block_ids"] == "p0001_b000"
    assert "标题：中文摘要对比研究" in c["page_content"]
    assert "作者：邢淼、田丽" in c["page_content"]
    assert "发表期刊：知识管理论坛" in c["page_content"]
    assert "发表年份：2024 年" in c["page_content"]
    assert c["embed_text"] == c["page_content"]
    assert metadata_chunk({"title": ""}) is None


if __name__ == "__main__":
    test_chunk_paper_page_contract()
    test_chunk_uid_stable_and_clean_keeps_latex()
    test_metadata_chunk_contract()
    print("PASS test_paper_chunk")
