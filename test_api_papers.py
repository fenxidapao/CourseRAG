"""test_api_papers.py — /papers 端点（M3）测试，不启 MinerU / 不连 chroma / 不联网。

路径全部重定向到 tmp_path，避免污染 data/papers 与真实索引；
解析 worker 用 `_ensure_paper_workers` 替身关掉，只断言「已入队」。
"""
import json
import os
import sys

import pytest
import pymupdf
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class FakeCol:
    def __init__(self):
        self.name = "paper_rag"

    def count(self):
        return 0

    def get(self, include=None):
        return {"metadatas": []}

    def delete(self, where=None):
        return None


@pytest.fixture
def paper_client(monkeypatch, tmp_path):
    import config
    import main
    import paper_index
    import paper_ingest

    base = tmp_path / "papers"
    for name in ("raw", "parsed", "normalized", "meta", "failed", "logs", "_state"):
        (base / name).mkdir(parents=True)

    monkeypatch.setattr(config, "APP_PROFILE", "paper")
    monkeypatch.setattr(config, "PAPERS_DIR", str(base))
    monkeypatch.setattr(config, "PAPERS_RAW_DIR", str(base / "raw"))
    monkeypatch.setattr(config, "PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setattr(paper_ingest, "PAPERS", base)
    monkeypatch.setattr(paper_ingest, "RAW", base / "raw")
    monkeypatch.setattr(paper_ingest, "PARSED", base / "parsed")
    monkeypatch.setattr(paper_ingest, "NORMALIZED", base / "normalized")
    monkeypatch.setattr(paper_ingest, "META", base / "meta")
    monkeypatch.setattr(paper_ingest, "FAILED", base / "failed")
    monkeypatch.setattr(paper_ingest, "LOGS", base / "logs")
    monkeypatch.setattr(paper_ingest, "STATE", base / "_state")
    monkeypatch.setattr(paper_ingest, "MANIFEST", base / "manifest.csv")
    monkeypatch.setattr(paper_index, "NORMALIZED", base / "normalized")
    monkeypatch.setattr(paper_index, "delete_paper", lambda pid: None)
    monkeypatch.setattr(paper_index, "upsert_paper", lambda pid: 0)

    monkeypatch.setattr(main, "_ensure_paper_workers", lambda: None)
    monkeypatch.setattr(main, "get_vector_store", lambda *a, **k: FakeCol())
    monkeypatch.setattr("health.check_ollama", lambda base_url=None, timeout=4: (True, "mock"))
    monkeypatch.setattr("health.check_reranker_cache", lambda: (True, "mock"))

    with TestClient(main.app) as c:
        yield c, base, paper_ingest, main
    while not main._PAPER_QUEUE.empty():
        main._PAPER_QUEUE.get_nowait()
        main._PAPER_QUEUE.task_done()


def _upload(client, name="论文 A.pdf"):
    return client.post(
        "/papers",
        files=[("files", (name, b"%PDF-1.4 fake", "application/pdf"))],
    )


def _seed_state(paper_ingest, base, pid="abcdef123456", pdf_name="p.pdf"):
    pdf = base / "raw" / pdf_name
    pdf.write_bytes(b"%PDF-1.4 fake")
    state = paper_ingest._new_state(pdf, "abcdef123456" + "0" * 52)
    state["paper_id"] = pid
    state.update(status="normalized", pages=2, blocks=2, chunks=2, title="T")
    paper_ingest.save_state(state)
    return state, pdf


def test_upload_rejects_non_pdf(paper_client):
    client, *_ = paper_client
    r = client.post("/papers", files=[("files", ("a.txt", b"hi", "text/plain"))])
    assert r.status_code == 400
    assert "PDF" in r.json()["detail"]


def test_upload_queues_paper(paper_client):
    client, base, paper_ingest, main = paper_client
    r = _upload(client)
    assert r.status_code == 200, r.text
    item = r.json()["files"][0]
    assert item["status"] == "queued"
    assert (base / "raw" / "论文 A.pdf").exists()
    state = paper_ingest.load_state(item["paper_id"])
    assert state["status"] == "queued"
    assert main._PAPER_QUEUE.qsize() == 1


def test_list_and_get_paper(paper_client):
    client, base, paper_ingest, _ = paper_client
    _seed_state(paper_ingest, base)
    r = client.get("/papers")
    assert r.status_code == 200
    assert [p["paper_id"] for p in r.json()["papers"]] == ["abcdef123456"]
    assert client.get("/papers/abcdef123456").status_code == 200
    assert client.get("/papers/nonexistent00").status_code == 404
    assert client.get("/papers/..%2Fetc").status_code == 404


def test_blocks_and_page_png(paper_client):
    client, base, paper_ingest, _ = paper_client
    pid = "abcdef123456"
    _seed_state(paper_ingest, base, pid)
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page(width=200, height=300)
        page.insert_text((20, 40), "hello")
    doc.save(base / "raw" / "p.pdf")
    doc.close()
    data = {
        "n_pages": 2,
        "bbox_space": "pdf_points",
        "pages": [
            {"page": 1, "blocks": [{"block_id": "p0001_b000", "bbox": [10, 10, 80, 30], "text": "hi"}]},
            {"page": 2, "blocks": [{"block_id": "p0002_b000", "bbox": [10, 10, 80, 30], "text": "yo"}]},
        ],
    }
    (base / "normalized" / f"{pid}.json").write_text(json.dumps(data), encoding="utf-8")

    r = client.get(f"/papers/{pid}/blocks?page=1")
    assert r.status_code == 200
    assert r.json()["pages"][0]["blocks"][0]["block_id"] == "p0001_b000"
    assert client.get(f"/papers/{pid}/blocks?page=9").status_code == 404

    r = client.get(f"/papers/{pid}/pages/1.png?blocks=p0001_b000")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert client.get(f"/papers/{pid}/pages/3.png").status_code == 404

    # 红线：不提供整本 PDF 下载
    for path in ("pdf", "file", "download"):
        assert client.get(f"/papers/{pid}/{path}").status_code == 404


def test_delete_paper_and_raw(paper_client):
    client, base, paper_ingest, _ = paper_client
    pid = "abcdef123456"
    _, pdf = _seed_state(paper_ingest, base, pid)
    r = client.delete(f"/papers/{pid}")
    assert r.status_code == 200
    assert r.json()["raw_deleted"] is False
    assert pdf.exists()
    assert paper_ingest.load_state(pid) is None

    _seed_state(paper_ingest, base, pid)
    r = client.delete(f"/papers/{pid}?delete_raw=true")
    assert r.json()["raw_deleted"] is True
    assert not pdf.exists()


def test_reparse_requeues(paper_client):
    client, base, paper_ingest, main = paper_client
    _seed_state(paper_ingest, base)
    r = client.post("/papers/abcdef123456/reparse")
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
    assert paper_ingest.load_state("abcdef123456")["status"] == "queued"
    assert main._PAPER_QUEUE.qsize() == 1


def test_documents_proxies_to_papers(paper_client):
    client, base, paper_ingest, _ = paper_client
    _seed_state(paper_ingest, base)
    r = client.get("/documents")
    assert r.status_code == 200
    body = r.json()
    assert [f["name"] for f in body["files"]] == ["p.pdf"]
    assert body["total_chunks"] == 2


def test_require_paper_profile_rejects_course(monkeypatch):
    from fastapi import HTTPException

    import config
    import main

    monkeypatch.setattr(config, "APP_PROFILE", "course")
    with pytest.raises(HTTPException) as ei:
        main.require_paper_profile()
    assert ei.value.status_code == 400
