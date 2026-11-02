"""test_paper_normalize.py — paper_blocks/1 归一化契约（零外部依赖）。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from paper_normalize import (
    apply_code_map,
    assign_sections,
    audit,
    has_legacy_glyph,
    normalize_paper,
    normalize_math_spacing,
    repair_codes_by_context,
)


def test_assign_sections_uses_repaired_titles():
    # section 必须在连字修复后重算，否则引用卡上仍是 "4.2 ality–Cost Tradeo"
    pages = [{"blocks": [
        {"type": "title", "text": "ColBERT"},
        {"type": "title", "text": "1 Introduction"},
        {"type": "paragraph", "text": "body"},
        {"type": "title", "text": "4.2 Quality–Cost Tradeoff"},
        {"type": "paragraph", "text": "body2"},
    ]}]
    assign_sections(pages)
    blocks = pages[0]["blocks"]
    assert blocks[0]["section"] == "" and blocks[0]["section_level"] == 0
    assert blocks[1]["section"] == ""
    assert blocks[2]["section"] == "1 Introduction"
    assert blocks[4]["section"] == "1 Introduction > 4.2 Quality–Cost Tradeoff"


# MinerU 连字码：同一码在不同字体含义不同（\x80 = fi / ff），必须按上下文逐处还原
@pytest.mark.parametrize(
    "broken, ref, expected",
    [
        ("pre-de\x80ned", "pre-defined", "pre-defined"),
        ("E\x81cient and E\x82ective", "Efficient and Effective", "Efficient and Effective"),
        ("ColBERT: E\x80icient and E\x80ective", "ColBERT: Efficient and Effective",
         "ColBERT: Efficient and Effective"),
        ("o\x84ine", "offline", "offline"),
        ("re\x83ects", "reflects", "reflects"),
        ("Kha\x8aab", "Khattab", "Khattab"),
        ("\x87e \x80gure", "The figure", "The figure"),
        ("Mean \x86ery Latency", "Mean Query Latency", "Mean Query Latency"),
        ("4.2 \x86ality–Cost Tradeo\x82", "4.2 Quality–Cost Tradeoff", "4.2 Quality–Cost Tradeoff"),
    ],
)
def test_repair_codes_by_context(broken, ref, expected):
    fixed, n = repair_codes_by_context(broken, ref)
    assert fixed == expected
    assert n == sum(1 for ch in broken if has_legacy_glyph(ch))


def test_repair_keeps_text_when_no_reference_match():
    # 图块 bbox 抽到的是图内文字：对不上时只去掉码，不改写其余文字
    fixed, n = repair_codes_by_context("Figure 1: e\x82ectiveness", "some other text")
    assert fixed == "Figure 1: eectiveness" and n == 0


def test_repair_prefers_unique_font_map():
    # caption 字体的 ToUnicode 丢字母，上下文搜索只拿到 "T"；字体表 /T_h 才是权威
    fixed, n = repair_codes_by_context(
        "Figure 2: \x87e figure", "Figure 2: Te figure", {"\x87": {"Th"}}
    )
    assert fixed == "Figure 2: The figure" and n == 1


def test_repair_uses_context_when_font_map_ambiguous():
    fixed, n = repair_codes_by_context("E\x80icient", "Efficient", {"\x80": {"fi", "ff"}})
    assert fixed == "Efficient" and n == 1


def test_repair_keeps_text_when_unmatched():
    fixed, n = repair_codes_by_context("no such \x87ext", "completely different page")
    assert fixed == "no such ext" and n == 0  # 认不出就只去码


def test_apply_code_map_expands_and_drops_unknown():
    assert apply_code_map("e\x81cient o\x84ine x\x8b", {"\x81": "ffi", "\x84": "ffl"}) == "efficient offline x"
    assert has_legacy_glyph("a\x80b") and not has_legacy_glyph("abc")


def test_apply_code_map_disambiguates_by_word_position():
    # \x8a 在正文是 tt、在粗体标题是 Th：按词首/词中大小写分开
    mapping = {"\x8a": {"tt", "Th"}}
    assert apply_code_map("Hofsta\x8aer", mapping) == "Hofstatter"
    assert apply_code_map("\x8ae result", mapping) == "The result"


def test_math_spacing_decimal():
    assert normalize_math_spacing("$0 . 9 8$") == "$0.98$"


def test_math_spacing_multidigit_decimal():
    assert normalize_math_spacing("$1 1 . 8 \\%$") == "$11.8 \\%$"


def test_math_spacing_numeric_brace():
    assert normalize_math_spacing("$10 ^ { - 9 }$") == "$10 ^{-9}$"


def test_math_spacing_leaves_letters():
    assert normalize_math_spacing("$\\text { a b }$") == "$\\text { a b }$"


def test_math_spacing_leaves_plain_text_columns():
    assert normalize_math_spacing("34.5 36.6") == "34.5 36.6"


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
