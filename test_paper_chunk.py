"""test_paper_chunk.py — 页内切分契约（零外部依赖）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_chunker import audit_chunks, chunk_paper, clean_paper_text


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


if __name__ == "__main__":
    test_chunk_paper_page_contract()
    test_chunk_uid_stable_and_clean_keeps_latex()
    print("PASS test_paper_chunk")
