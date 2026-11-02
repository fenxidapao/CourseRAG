"""test_paper_citation.py — 论文引用字段与 prompt 页码标签（零外部依赖）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from rag_core import Document, RagChain, build_citations


def _paper_doc():
    return Document(
        page_content="3 Method > 3.2 Loss\nWe use cross entropy.",
        metadata={
            "source": "data/papers/raw/test.pdf",
            "paper_id": "abc123",
            "paper_title": "Test Paper",
            "page": 3,
            "page_start": 3,
            "page_end": 3,
            "section": "3 Method > 3.2 Loss",
            "block_type": "table",
            "block_ids": "p0003_b001",
            "start_index": 30001,
            "chunk_uid": "f" * 16 + ":p0003:c001:deadbeef",
        },
    )


def _course_doc():
    return Document(
        page_content="ACID 包括原子性、一致性。",
        metadata={"source": "data/教材/db.pdf", "start_index": 42},
    )


def test_paper_citation_fields_and_prompt(monkeypatch):
    monkeypatch.setattr(config, "APP_PROFILE", "paper")
    monkeypatch.setattr(config, "CITATION_LABEL_MODE", "page")
    docs = [_paper_doc()]
    cits = build_citations(docs)
    c = cits[0]
    assert c["paper_id"] == "abc123"
    assert c["title"] == "Test Paper"
    assert c["page"] == 3 and c["page_start"] == 3 and c["page_end"] == 3
    assert c["section"] == "3 Method > 3.2 Loss"
    assert c["block_type"] == "table" and c["type"] == "table"
    assert c["block_ids"] == "p0003_b001"
    prompt = RagChain(history="")._compose_prompt("What loss?", docs)
    assert "《Test Paper》第 3 页 · 3 Method > 3.2 Loss" in prompt
    assert "页码只能抄自" in prompt


def test_course_citation_keeps_offset(monkeypatch):
    monkeypatch.setattr(config, "CITATION_LABEL_MODE", "offset")
    docs = [_course_doc()]
    c = build_citations(docs)[0]
    assert "paper_id" not in c
    prompt = RagChain(history="")._compose_prompt("ACID?", docs)
    assert "offset=42" in prompt


def test_citation_sentence_splitter_merges_bare_markers():
    """句末独立成行的 [n] 要并入前一句，不能当成空陈述送引用核验。"""
    from eval_paper_quality import _cited_sentences

    got = _cited_sentences("MinerU 基于 PDF-Extract-Kit。[1][7]")
    assert got == [("MinerU 基于 PDF-Extract-Kit。 [1][7]", [1, 7])]
    assert _cited_sentences("一 [1]。二 [2]。") == [("一 [1]。", [1]), ("二 [2]。", [2])]
    assert _cited_sentences("无引用。") == []


def test_citations_merge_same_page_blocks():
    """同页已召回块的 block_ids 并入引用；跨页/别的论文不动，rank 与内容不变。"""
    a = _paper_doc()
    b = _paper_doc()
    b.metadata.update({"block_ids": "p0003_b004", "start_index": 30050})
    c = _paper_doc()
    c.metadata.update({"block_ids": "p0004_b001", "page": 4, "page_start": 4, "page_end": 4})
    cites = build_citations([a, b, c])
    assert [x["rank"] for x in cites] == [1, 2, 3]
    assert cites[0]["block_ids"] == "p0003_b001,p0003_b004"
    assert cites[1]["block_ids"] == "p0003_b004,p0003_b001"
    assert cites[2]["block_ids"] == "p0004_b001"
    assert cites[0]["content"].startswith("3 Method")


if __name__ == "__main__":
    test_paper_citation_fields_and_prompt()
    test_course_citation_keeps_offset()
    test_citations_merge_same_page_blocks()
    test_citation_sentence_splitter_merges_bare_markers()
    print("PASS test_paper_citation")
