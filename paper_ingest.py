"""paper_ingest.py — 论文 PDF → MinerU → normalized → manifest（M1；M2 接索引）。

状态机：discovered → queued → parsing → parsed → normalized → indexed / failed。
真源是 data/papers/_state/<paper_id>.json；manifest.csv 由 state 派生。
MinerU 只经 mineru_adapter 子进程调用，主环境不 import mineru。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import config
import mineru_adapter
from paper_normalize import audit, normalize_paper, write_normalized

PAPERS = Path(config.PAPERS_DIR)
RAW = Path(config.PAPERS_RAW_DIR)
PARSED = PAPERS / "parsed"
NORMALIZED = PAPERS / "normalized"
META = PAPERS / "meta"
FAILED = PAPERS / "failed"
LOGS = PAPERS / "logs"
STATE = PAPERS / "_state"
MANIFEST = PAPERS / "manifest.csv"


def _ensure_dirs():
    for d in (PAPERS, RAW, PARSED, NORMALIZED, META, FAILED, LOGS, STATE):
        d.mkdir(parents=True, exist_ok=True)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def paper_id(sha):
    return sha[:12]


def _now():
    return datetime.now().isoformat(timespec="seconds")


def _state_path(pid):
    return STATE / f"{pid}.json"


def load_state(pid):
    p = _state_path(pid)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_state(state):
    _ensure_dirs()
    p = _state_path(state["paper_id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def _update(state, **kw):
    state.update(kw)
    state["updated_at"] = _now()
    save_state(state)
    return state


def _lock_path(pid):
    return STATE / f"{pid}.lock"


def _acquire_lock(pid):
    _ensure_dirs()
    lock = _lock_path(pid)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            # 死进程留下的锁：超过 MinerU 超时时间即视为 stale
            try:
                if time.time() - lock.stat().st_mtime > config.MINERU_TIMEOUT:
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            return False
    return False


def _release_lock(pid):
    try:
        _lock_path(pid).unlink()
    except FileNotFoundError:
        pass


def _detect_lang(pdf):
    """按前 2 页文本的 CJK 占比选 OCR 语言（ch/en）；MinerU 的 -l 只影响 OCR 模型。"""
    try:
        import pymupdf

        doc = pymupdf.open(str(pdf))
        text = "".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
        doc.close()
    except Exception:
        return None
    text = "".join(ch for ch in text if not ch.isspace())
    if not text:
        return None
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "ch" if cjk / len(text) > 0.05 else "en"


def _load_meta(sha):
    p = META / f"{sha}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _new_state(pdf, sha):
    return {
        "paper_id": paper_id(sha),
        "sha256": sha,
        "filename": pdf.name,
        "source_path": str(pdf),
        "title": "",
        "source_url": "",
        "license": "",
        "status": "discovered",
        "stage": "",
        "pages": 0,
        "blocks": 0,
        "chunks": 0,
        "parse_sec": 0.0,
        "pages_per_sec": 0.0,
        "error": "",
        "created_at": _now(),
        "updated_at": _now(),
    }


def ingest_one(pdf, device=None, max_pages=None, dry_run=False, force=False):
    _ensure_dirs()
    pdf = Path(pdf)
    sha = sha256_file(pdf)
    pid = paper_id(sha)
    state = load_state(pid) or _new_state(pdf, sha)
    meta = _load_meta(sha)
    if meta:
        state["source_url"] = meta.get("source_url") or state.get("source_url", "")
        state["license"] = meta.get("license") or state.get("license", "")
        if meta.get("title"):
            state["title"] = meta["title"]

    normalized_path = NORMALIZED / f"{pid}.json"
    if not force and state.get("status") in ("normalized", "indexed") and normalized_path.exists():
        return "skip"
    if state.get("status") == "parsing" and not force:
        return "busy"

    if dry_run:
        print(f"[dry-run] {pdf.name} -> {pid} status={state.get('status')}")
        return "dry-run"

    if not _acquire_lock(pid):
        return "busy"
    try:
        _update(state, status="parsing", stage="mineru", error="")
        out_dir = PARSED / pid
        existing = mineru_adapter.find_outputs(out_dir)
        if not force and state.get("status") == "parsed" and existing["content_list"]:
            pages = mineru_adapter.count_pages(existing["middle"], existing["content_list"])
            res = {
                "content_list": existing["content_list"],
                "middle": existing["middle"],
                "returncode": 0,
                "log": "",
                "elapsed": 0.0,
                "pages": pages,
            }
        else:
            res = mineru_adapter.parse(
                pdf, out_dir, device=device, max_pages=max_pages,
                log_path=LOGS / f"{pid}.mineru.log", lang=_detect_lang(pdf),
            )
        if res["returncode"] != 0 or not res["content_list"]:
            tail = (res.get("log") or "")[-2000:]
            _update(
                state, status="failed", stage="parsing",
                error=f"mineru rc={res['returncode']}: {tail}",
                parse_sec=round(res["elapsed"], 2),
            )
            return "failed"
        pages = res["pages"]
        _update(
            state, status="parsed", stage="normalize", pages=pages,
            parse_sec=round(res["elapsed"], 2),
            pages_per_sec=round(pages / res["elapsed"], 3) if res["elapsed"] else 0.0,
        )
        data = normalize_paper(
            res["content_list"], res["middle"], pdf, pid, sha, meta=meta
        )
        write_normalized(data, normalized_path)
        total, _, missing_bbox = audit(data)
        _update(
            state, status="normalized", stage="normalized",
            pages=data["n_pages"], blocks=total,
            title=data.get("title") or state.get("title", ""),
        )
        if missing_bbox:
            print(f"[warn] {pdf.name}: {missing_bbox}/{total} 块缺 bbox")
        return "normalized"
    except Exception as e:
        _update(state, status="failed", stage="exception", error=repr(e)[:2000])
        return "failed"
    finally:
        _release_lock(pid)


def list_states():
    rows = []
    for p in sorted(STATE.glob("*.json")):
        try:
            rows.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return rows


def _manifest_rows():
    rows = []
    for s in list_states():
        rows.append({
            "paper_id": s.get("paper_id", ""),
            "filename": s.get("filename", ""),
            "title": s.get("title", ""),
            "status": s.get("status", ""),
            "pages": s.get("pages", 0),
            "blocks": s.get("blocks", 0),
            "chunks": s.get("chunks", 0),
            "parse_sec": s.get("parse_sec", 0),
            "pages_per_sec": s.get("pages_per_sec", 0),
            "source_url": s.get("source_url", ""),
            "license": s.get("license", ""),
            "error": (s.get("error") or "")[:300],
            "updated_at": s.get("updated_at", ""),
        })
    return rows


def write_manifest():
    _ensure_dirs()
    rows = _manifest_rows()
    fields = list(rows[0].keys()) if rows else [
        "paper_id", "filename", "title", "status", "pages", "blocks", "chunks",
        "parse_sec", "pages_per_sec", "source_url", "license", "error", "updated_at",
    ]
    tmp = MANIFEST.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, MANIFEST)
    return len(rows)


def list_papers(raw_dir=None):
    raw_dir = Path(raw_dir or RAW)
    return sorted(p for p in raw_dir.glob("*.pdf") if p.is_file())


def ingest_dir(raw_dir=None, only_new=False, retry_failed=False, max_pages=None,
               dry_run=False, device=None, paper=None, force=False):
    pdfs = list_papers(raw_dir)
    if paper:
        needle = str(paper).lower()
        pdfs = [p for p in pdfs if needle in p.name.lower()]
    results = {}

    def work(pdf):
        sha = sha256_file(pdf)
        pid = paper_id(sha)
        state = load_state(pid) or {}
        if only_new and state.get("status") in ("normalized", "indexed") \
                and (NORMALIZED / f"{pid}.json").exists():
            return pid, "skip"
        if state.get("status") == "failed" and not retry_failed and not force:
            return pid, "failed-skip"
        return pid, ingest_one(pdf, device=device, max_pages=max_pages,
                               dry_run=dry_run, force=force)

    with ThreadPoolExecutor(max_workers=config.PARSE_CONCURRENCY) as ex:
        for pid, result in ex.map(work, pdfs):
            results[pid] = result
            print(f"{pid} {result}")
    write_manifest()
    return results


def renormalize_one(pdf):
    """用缓存的 MinerU 产物重跑归一化（不重跑解析）；归一化规则改动后用。"""
    pdf = Path(pdf)
    sha = sha256_file(pdf)
    pid = paper_id(sha)
    state = load_state(pid) or _new_state(pdf, sha)
    out = mineru_adapter.find_outputs(PARSED / pid)
    if not out["content_list"]:
        return "no-parse"
    data = normalize_paper(out["content_list"], out["middle"], pdf, pid, sha,
                           meta=_load_meta(sha))
    write_normalized(data, NORMALIZED / f"{pid}.json")
    total, _, _ = audit(data)
    _update(state, status="normalized", stage="normalized",
            pages=data["n_pages"], blocks=total, error="")
    return "renormalized"


def renormalize_all(raw_dir=None, paper=None):
    pdfs = list_papers(raw_dir)
    if paper:
        needle = str(paper).lower()
        pdfs = [p for p in pdfs if needle in p.name.lower()]
    results = {}
    for pdf in pdfs:
        sha = sha256_file(pdf)
        pid = paper_id(sha)
        results[pid] = renormalize_one(pdf)
        print(f"{pid} {results[pid]}")
    write_manifest()
    return results


def main():
    ap = argparse.ArgumentParser(description="论文 PDF → MinerU → normalized")
    ap.add_argument("--dir", default=str(RAW))
    ap.add_argument("--only-new", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--max-pages", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--paper", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--renormalize", action="store_true",
                    help="用缓存解析产物重跑归一化（不跑 MinerU），规则改动后用")
    args = ap.parse_args()
    if args.renormalize:
        renormalize_all(raw_dir=args.dir, paper=args.paper)
        return
    ingest_dir(
        raw_dir=args.dir, only_new=args.only_new, retry_failed=args.retry_failed,
        max_pages=args.max_pages, dry_run=args.dry_run, device=args.device,
        paper=args.paper, force=args.force,
    )


if __name__ == "__main__":
    main()
