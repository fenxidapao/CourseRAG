"""paper_index.py — normalized/*.json → paper_chunker → rag_core.Document → Chroma。

只被 rag_core.load_and_chunk_documents（paper profile）与 CLI 调用；
重解析按 sha256 先删后插，避免旧 chunk_uid 残留。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import config
import paper_chunker
from rag_core import Document, add_documents, get_vector_store

PAPERS = Path(config.PAPERS_DIR)
NORMALIZED = PAPERS / "normalized"


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalized_files():
    if not NORMALIZED.exists():
        return []
    return sorted(NORMALIZED.glob("*.json"))


def load_paper_chunks(paper_id=None):
    paths = [NORMALIZED / f"{paper_id}.json"] if paper_id else normalized_files()
    docs = []
    for path in paths:
        if not Path(path).exists():
            continue
        data = _load(path)
        for c in paper_chunker.chunk_paper(data):
            docs.append(Document(page_content=c["page_content"], metadata=c["metadata"]))
    return docs


def upsert_paper(paper_id):
    path = NORMALIZED / f"{paper_id}.json"
    if not path.exists():
        return 0
    data = _load(path)
    docs = load_paper_chunks(paper_id)
    if not docs:
        return 0
    col = get_vector_store(auto_rebuild=False)
    col.delete(where={"sha256": data["sha256"]})
    n = add_documents(docs)
    import paper_ingest

    state = paper_ingest.load_state(paper_id) or {"paper_id": paper_id, "sha256": data["sha256"]}
    state.update(
        status="indexed",
        chunks=n,
        pages=data.get("n_pages", state.get("pages", 0)),
        blocks=sum(len(p.get("blocks") or []) for p in data.get("pages", [])),
        title=data.get("title") or state.get("title", ""),
        updated_at=datetime.now().isoformat(timespec="seconds"),
    )
    paper_ingest.save_state(state)
    paper_ingest.write_manifest()
    import semantic_cache

    semantic_cache.invalidate()
    return n


def delete_paper(paper_id):
    col = get_vector_store(auto_rebuild=False)
    col.delete(where={"paper_id": paper_id})
    import semantic_cache

    semantic_cache.invalidate()


def rebuild_all():
    docs = load_paper_chunks()
    get_vector_store(docs)
    counts = {}
    for d in docs:
        pid = d.metadata.get("paper_id")
        counts[pid] = counts.get(pid, 0) + 1
    import paper_ingest

    for pid, n in counts.items():
        state = paper_ingest.load_state(pid)
        if state:
            state.update(status="indexed", chunks=n)
            paper_ingest.save_state(state)
    paper_ingest.write_manifest()
    return len(docs)


def audit_chunks(paper_id=None):
    docs = load_paper_chunks(paper_id)
    chunks = [
        {"text": d.page_content, "metadata": d.metadata} for d in docs
    ]
    total, missing_page, cross, no_section = paper_chunker.audit_chunks(chunks)
    papers = sorted({d.metadata.get("paper_id") for d in docs})
    return {
        "chunks": total,
        "missing_page": missing_page,
        "cross_page": cross,
        "missing_section": no_section,
        "papers": papers,
    }


def main():
    ap = argparse.ArgumentParser(description="论文 normalized → 索引")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--paper", default=None)
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()
    if args.audit:
        print(audit_chunks(args.paper))
    elif args.rebuild:
        print("chunks:", rebuild_all())
    elif args.paper:
        print("chunks:", upsert_paper(args.paper))
    else:
        raise SystemExit("用法：--rebuild | --paper <id> | --audit")


if __name__ == "__main__":
    main()
