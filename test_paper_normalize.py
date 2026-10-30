"""test_paper_normalize.py — paper_blocks/1 归一化契约（零外部依赖）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_normalize import audit, normalize_paper


def _write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def test_normalize_merges_bbox_and_sections(tmp_path):
    content = [
        {"type": "text", "text": "1 Introduction", "text_level": 1, "page_idx": 0},
        {"type": "text", "text": "This is body α β θ.", "page_idx": 0},
        {"type": "image", "img_path": "images/fig1.jpg", "img_caption": ["Figure 1"], "page_idx": 0},
    ]
    middle = {
        "pdf_info": [
            {
                "page_idx": 0,
                "page_size": [600, 800],
                "para_blocks": [
                    {"type": "title", "level": 1, "bbox": [10, 10, 200, 30], "lines": []},
                    {"type": "text", "bbox": [10, 40, 500, 100], "lines": []},
                    {"type": "image", "bbox": [10, 120, 300, 300], "lines": []},
                ],
            }
        ]
    }
    _write(tmp_path / "content_list.json", content)
    _write(tmp_path / "middle.json", middle)
    meta = {"title": "Test Paper", "authors": "A. Author", "year": 2026, "license": "CC BY 4.0"}
    data = normalize_paper(
        tmp_path / "content_list.json",
        tmp_path / "middle.json",
        tmp_path / "paper.pdf",
        "abc123",
        "f" * 64,
        meta=meta,
    )
    blocks = data["pages"][0]["blocks"]
    assert data["schema"] == "paper_blocks/1"
    assert data["title"] == "Test Paper" and data["title_source"] == "meta"
    assert data["n_pages"] == 1
    assert [b["type"] for b in blocks] == ["title", "paragraph", "figure"]
    assert blocks[1]["bbox"] == [10, 40, 500, 100]
    assert blocks[1]["section"] == "1 Introduction"
    assert "α β θ" in blocks[1]["text"]
    assert blocks[2]["caption"] == "Figure 1"
    total, missing_page, missing_bbox = audit(data)
    assert (total, missing_page, missing_bbox) == (3, 0, 0)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_normalize_merges_bbox_and_sections(Path(d))
    print("PASS test_paper_normalize")
