"""
test_api.py — FastAPI 端点集成测试（T9，2026-08-22）

目标：不依赖 Ollama / chroma 落盘 / 外网，用 fastapi TestClient + monkeypatch
隔离全部外部依赖，验证 API 层的：
  - 路由可达（/ /health /query /retrieve /documents /index/*）
  - 参数校验（缺失 / 非法文件类型 / 路径穿越 → 4xx）
  - 响应结构（Citation / QueryResponse / IndexStats 字段齐全）
  - 错误处理（索引不可用 503、生成失败 500）

关键设计：main.py 顶层 `from rag_core import (...)`，因此 monkeypatch
`main.<函数名>` 即可替换端点内部对 rag_core 的调用，无需触碰 rag_core 本身；
config.DATA_DIR / PERSIST_DIRECTORY 也重定向到临时目录，杜绝测试污染真实语料。

运行：pytest test_api.py -v   （无需 Ollama / chroma 持久化 / 网络）
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# 隔离查询日志（import main 时 query_log 读取 QUERY_LOG_PATH）
_tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
_tmp.close()
os.environ["QUERY_LOG_PATH"] = _tmp.name

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 本文件锁定课程版语义（默认 profile 已切 paper）；必须在 import config 之前设置
os.environ["APP_PROFILE"] = "course"

import config
import main
from fastapi.testclient import TestClient


class FakeCol:
    """get_vector_store 的替身：只实现端点用到的接口。"""

    def __init__(self, count=10, metas=None):
        self._count = count
        self._metas = metas or [{"source": "/x/a.md"} for _ in range(count)]
        self.name = config.COLLECTION_NAME

    def count(self):
        return self._count

    def get(self, include=None):
        return {"metadatas": self._metas}


class FakeDoc:
    """rag_core.Document 的替身：page_content + metadata。"""

    def __init__(self, content="ACID 包括原子性、一致性、隔离性、持久性。", source="db.txt", start=0, score=None):
        self.page_content = content
        self.metadata = {"source": source, "start_index": start}
        if score is not None:
            self.metadata["hybrid_score"] = score


@pytest.fixture
def client(monkeypatch, tmp_path):
    """隔离外部依赖后提供 TestClient（生命周期内复用）。"""
    # 数据目录 / 向量库目录重定向到临时目录
    monkeypatch.setattr(config, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(config, "PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    # 默认向量库替身（lifespan 预热 + 各端点共用）
    monkeypatch.setattr(main, "get_vector_store", lambda *a, **k: FakeCol())
    # health 的外部检查 mock 掉，避免真的请求 Ollama
    monkeypatch.setattr("health.check_ollama", lambda base_url=None, timeout=4: (True, "mock-ok"))
    monkeypatch.setattr("health.check_reranker_cache", lambda: (True, "mock-ok"))
    with TestClient(main.app) as c:
        yield c


# ==================== meta / health ====================
def test_service_info(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "CourseRAG API"
    assert any("/health" in e for e in body["endpoints"])
    assert any("/query" in e for e in body["endpoints"])
    assert "models" in body


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and isinstance(body["ok"], bool)
    assert isinstance(body["checks"], list) and len(body["checks"]) >= 2
    assert body["vector_store_count"] == 10


# ==================== /query ====================
def _patch_answer(monkeypatch, answer="ACID 包括原子性、一致性、隔离性、持久性[1]。", low_confidence=False):
    """把统一问答入口 answer_question 替换为假实现（评审 P1 起 /query 走该入口，
    原先 mock 的 retrieve_and_rerank / build_rag_chain / get_confidence 已不在
    API 层直接调用）。"""
    citations = [
        {"rank": 1, "source": "db.txt", "content": "ACID 包括原子性、一致性、隔离性、持久性。",
         "score": 0.9, "start_index": 0},
        {"rank": 2, "source": "db2.txt", "content": "第二范式 2NF 消除部分函数依赖。",
         "score": 0.7, "start_index": 50},
    ]

    def _fake_answer_question(question, mode="fast", **kwargs):
        return {
            "answer": answer,
            "citations": citations,
            "docs": None,
            "cached": False,
            "cache_sim": None,
            "rejected": False,
            "reject_signal": None,
            "grounding_ran": mode == "accurate",
            "grounded": True,
            "not_grounded": False,
            "grounded_score": 1.0 if mode == "accurate" else None,
            "low_confidence": low_confidence,
        }

    monkeypatch.setattr(main, "answer_question", _fake_answer_question)
    monkeypatch.setattr(main, "log_query", lambda **kw: None)


def test_query_fast(client, monkeypatch):
    _patch_answer(monkeypatch)
    r = client.post("/query", json={"question": "什么是 ACID？", "mode": "fast"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "ACID 包括原子性、一致性、隔离性、持久性[1]。"
    assert body["mode"] == "fast"
    assert body["grounded"] is True
    assert body["low_confidence"] is False
    assert len(body["citations"]) == 2
    assert body["citations"][0]["rank"] == 1
    assert body["citations"][0]["source"] == "db.txt"
    assert isinstance(body["latency_ms"], int)


def test_query_low_confidence(client, monkeypatch):
    _patch_answer(monkeypatch, low_confidence=True)  # 低于阈值 0.69
    r = client.post("/query", json={"question": "完全无关的问题？", "mode": "fast"})
    assert r.status_code == 200
    assert r.json()["low_confidence"] is True


def test_query_missing_question(client):
    r = client.post("/query", json={})
    assert r.status_code == 422


def test_query_accurate_mode(client, monkeypatch):
    _patch_answer(monkeypatch)
    r = client.post("/query", json={"question": "什么是 2NF？", "mode": "accurate"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "accurate"
    # 评审 P1 后 accurate 在 API 侧同样过闸门：正常路径 grounded/cached 字段语义不变
    assert body["grounded"] is True
    assert body["cached"] is False


# ==================== rag_core.answer_question 统一闸门（评审 P1） ====================
def _patch_answer_pipeline(monkeypatch, docs, gen_answer="好的回答[1]。", grounded=(True, 1.0)):
    """隔离 answer_question 的外部依赖：检索 / 链 / 接地自检 / 语义缓存。
    返回 store_calls 列表，用于断言缓存写入的 reject 标志。"""
    import rag_core

    monkeypatch.setattr(rag_core, "retrieve_and_rerank", lambda q, force_no_rerank=False, **kwargs: docs)

    class _Chain:
        def invoke(self, question, docs=None):
            return gen_answer

    monkeypatch.setattr(rag_core, "build_rag_chain",
                        lambda custom_prompt=None, history="": {"chain": _Chain()})
    monkeypatch.setattr(rag_core, "grounded_generate",
                        lambda q, d, chain: (gen_answer, grounded[0], grounded[1]))
    store_calls = []
    monkeypatch.setattr(rag_core.semantic_cache, "lookup", lambda q, m: None)
    monkeypatch.setattr(rag_core.semantic_cache, "store",
                        lambda q, m, a, c, reject=False: store_calls.append(reject))
    return store_calls


def _mk_doc(vector_score):
    import rag_core

    d = rag_core.Document("ACID 包括原子性。", {"source": "/x/db.txt", "start_index": 0})
    d.metadata["vector_score"] = vector_score
    return d


def test_answer_question_fast_no_hard_reject(monkeypatch):
    """快速模式：即使 top-1 向量分很低也不硬拒（只软提示 low_confidence）。"""
    import rag_core

    _patch_answer_pipeline(monkeypatch, [_mk_doc(0.30)])
    # use_cache=True + lookup mock 未命中：验证「未命中→检索→写入」完整缓存路径
    res = rag_core.answer_question("什么是ACID", mode="fast", use_cache=True)
    assert res["rejected"] is False
    assert res["low_confidence"] is True  # 0.30 < 0.69 软提示
    assert res["answer"] == "好的回答[1]。"
    assert res["grounding_ran"] is False  # 快速模式不跑接地自检


def test_answer_question_accurate_hard_reject(monkeypatch):
    """精确模式：top-1 向量分低于 HARD_REJECT_VECTOR_THRESHOLD → 硬拒 + 不入缓存。"""
    import rag_core

    store_calls = _patch_answer_pipeline(monkeypatch, [_mk_doc(0.30)])
    res = rag_core.answer_question("怎么追女生", mode="accurate", use_cache=True)
    assert res["rejected"] is True
    assert res["answer"] == rag_core.HARD_REJECT_ANSWER
    assert res["not_grounded"] is False  # 检索拒而非接地拒
    assert store_calls == [True]  # 拒答不缓存


def test_answer_question_accurate_normal(monkeypatch):
    """精确模式正常路径：过闸门 → 接地自检通过 → 入缓存（reject=False）。"""
    import rag_core

    store_calls = _patch_answer_pipeline(monkeypatch, [_mk_doc(0.90)], grounded=(True, 0.95))
    res = rag_core.answer_question("什么是ACID", mode="accurate", use_cache=True)
    assert res["rejected"] is False
    assert res["not_grounded"] is False
    assert res["grounded_score"] == 0.95
    assert res["answer"] == "好的回答[1]。"
    assert store_calls == [False]


def test_answer_question_fast_stream(monkeypatch):
    """评审 P3 真实流式：快速模式 on_token 逐段回调，聚合回完整答案
    （缓存/日志语义不变——store 收到的仍是完整答案而非片段）。"""
    import rag_core

    pieces = ["ACID ", "包括", "原子性[1]。"]

    class _StreamChain:
        def invoke_stream(self, question, docs=None):
            yield from pieces

    monkeypatch.setattr(rag_core, "retrieve_and_rerank",
                        lambda q, force_no_rerank=False, **kwargs: [_mk_doc(0.90)])
    monkeypatch.setattr(rag_core, "build_rag_chain",
                        lambda custom_prompt=None, history="": {"chain": _StreamChain()})
    monkeypatch.setattr(rag_core.semantic_cache, "lookup", lambda q, m: None)
    store_calls = []
    monkeypatch.setattr(rag_core.semantic_cache, "store",
                        lambda q, m, a, c, reject=False: store_calls.append((a, reject)))
    got = []
    res = rag_core.answer_question("什么是ACID", mode="fast",
                                   on_token=got.append, use_cache=True)
    assert got == pieces  # 生成中逐段回调
    assert res["answer"] == "ACID 包括原子性[1]。"  # 聚合为完整答案
    assert store_calls == [("ACID 包括原子性[1]。", False)]


def test_answer_question_accurate_stream_ignored(monkeypatch):
    """评审 P3 安全语义：精确模式即使传 on_token 也不流式（接地自检需完整答案，
    未接地时整段替换为拒答文案——边生成边展示会让已展示内容无法撤回）。"""
    import rag_core

    class _Chain:
        def invoke(self, question, docs=None):
            return "缓冲路径的完整答案[1]。"

    monkeypatch.setattr(rag_core, "retrieve_and_rerank",
                        lambda q, force_no_rerank=False, **kwargs: [_mk_doc(0.90)])
    monkeypatch.setattr(rag_core, "build_rag_chain",
                        lambda custom_prompt=None, history="": {"chain": _Chain()})
    monkeypatch.setattr(rag_core, "grounded_generate",
                        lambda q, d, chain: ("缓冲路径的完整答案[1]。", True, 1.0))
    monkeypatch.setattr(rag_core.semantic_cache, "lookup", lambda q, m: None)
    monkeypatch.setattr(rag_core.semantic_cache, "store",
                        lambda q, m, a, c, reject=False: None)
    got = []
    res = rag_core.answer_question("什么是ACID", mode="accurate",
                                   on_token=got.append, use_cache=True)
    assert got == []  # 未触发任何流式回调
    assert res["answer"] == "缓冲路径的完整答案[1]。"


def test_answer_question_on_retrieved_fires_once_before_generate(monkeypatch):
    """on_retrieved（SSE 出口）：检索完成、硬拒答判定出炉后回调一次，
    携带 citations 与 rejected——前端因此能在首 token 之前渲染来源列表。"""
    import rag_core

    events = []
    _patch_answer_pipeline(monkeypatch, [_mk_doc(0.90)], grounded=(True, 0.98))
    res = rag_core.answer_question(
        "什么是ACID", mode="accurate", use_cache=False,
        on_retrieved=lambda citations, rejected: events.append((len(citations), rejected)),
    )
    assert events == [(1, False)]  # 恰好一次、1 条引用、非拒答
    assert res["answer"] == "好的回答[1]。"


def test_answer_question_on_retrieved_reports_hard_reject(monkeypatch):
    """拒答场景：rejected=True 随 metadata 一并给出，前端可立刻提示「未覆盖」，
    不必等生成（候选文档分低于硬拒阈值）。"""
    import rag_core

    events = []
    _patch_answer_pipeline(monkeypatch, [_mk_doc(0.10)])
    res = rag_core.answer_question(
        "怎么追女生", mode="accurate", use_cache=False,
        on_retrieved=lambda citations, rejected: events.append(rejected),
    )
    assert res["rejected"] is True
    assert events == [True]


def test_answer_question_on_retrieved_fires_on_cache_hit(monkeypatch):
    """缓存命中同样触发——否则走缓存的 SSE 请求会拿不到任何引用列表。"""
    import rag_core

    cached_citations = [{"rank": 1, "source": "db.txt", "content": "ACID 包括原子性。",
                         "score": 0.9, "start_index": 0}]
    monkeypatch.setattr(
        rag_core.semantic_cache, "lookup",
        lambda q, m: {"answer": "缓存的答案[1]。", "citations": cached_citations, "sim": 0.99},
    )
    events = []
    res = rag_core.answer_question(
        "什么是ACID", mode="fast", use_cache=True,
        on_retrieved=lambda citations, rejected: events.append((citations, rejected)),
    )
    assert res["cached"] is True
    assert events == [(cached_citations, False)]  # 缓存路径的回调不吞掉


# ==================== /query/stream（SSE 流式问答） ====================
def _parse_sse(body: str):
    """SSE 响应体 → [(event, data), ...]；注释帧（保活，以 : 开头）按规范丢弃。"""
    out = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        ev = data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                ev = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        out.append((ev, data))
    return out


def _patch_stream_answer(monkeypatch, pieces=("ACID ", "包括原子性", "[1]。"),
                         rejected=False, cached=False):
    """替换 answer_question 为「会触发两个回调」的假实现，用于验证 SSE 事件序。
    真实管线的回调时机已由上面 on_token / on_retrieved 的单测覆盖，
    这里只关心 API 层是否把回调正确转译成了事件。"""
    citations = [
        {"rank": 1, "source": "db.txt", "content": "ACID 包括原子性。",
         "score": 0.9, "start_index": 0},
    ]

    def _fake(question, mode="accurate", on_token=None, on_retrieved=None, **kwargs):
        if on_retrieved:
            on_retrieved(citations, rejected)
        if on_token and mode == "fast":  # 与真实语义一致：accurate 缓冲、不回调
            for p in pieces:
                on_token(p)
        return {
            "answer": "".join(pieces), "citations": citations, "docs": None,
            "cached": cached, "cache_sim": None, "rejected": rejected,
            "reject_signal": 0.1 if rejected else None,
            "grounding_ran": mode == "accurate", "grounded": not rejected,
            "not_grounded": False, "grounded_score": 1.0, "low_confidence": False,
        }

    monkeypatch.setattr(main, "answer_question", _fake)
    monkeypatch.setattr(main, "log_query", lambda **kw: None)


def test_query_stream_fast_event_sequence(client, monkeypatch):
    """fast 模式事件序：start → metadata → token* → done，done 含完整答案。"""
    _patch_stream_answer(monkeypatch)
    r = client.post("/query/stream", json={"question": "什么是ACID", "mode": "fast"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start"
    assert kinds[1] == "metadata"
    assert kinds.count("token") == 3
    assert kinds[-1] == "done"

    meta = events[1][1]
    assert meta["rejected"] is False
    assert len(meta["citations"]) == 1 and meta["citations"][0]["source"] == "db.txt"

    # token 拼接应等于 done 的答案：前端无需自行聚合也能校验一致性
    tokens = "".join(d["text"] for e, d in events if e == "token")
    assert tokens == "ACID 包括原子性[1]。"
    assert events[-1][1]["answer"] == tokens
    assert "latency_ms" in events[-1][1]


def test_query_stream_accurate_has_metadata_but_no_token(client, monkeypatch):
    """accurate 模式：引用照发，正文不逐 token（接地自检需完整答案）。
    这是既定语义而非缺陷——测试钉住它，避免日后被无声改成「假流式」。"""
    _patch_stream_answer(monkeypatch)
    r = client.post("/query/stream", json={"question": "什么是ACID", "mode": "accurate"})
    events = _parse_sse(r.text)
    kinds = [e for e, _ in events]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert "metadata" in kinds
    assert "token" not in kinds
    assert events[-1][1]["answer"] == "ACID 包括原子性[1]。"


def test_query_stream_carries_reject_in_metadata(client, monkeypatch):
    """拒答：rejected 在 metadata 里就给出，前端不必等生成完才知道未覆盖。"""
    _patch_stream_answer(monkeypatch, rejected=True)
    r = client.post("/query/stream", json={"question": "怎么追女生", "mode": "accurate"})
    events = _parse_sse(r.text)
    meta = next(d for e, d in events if e == "metadata")
    assert meta["rejected"] is True
    assert events[-1][1]["rejected"] is True


def test_query_stream_error_event_when_pipeline_raises(client, monkeypatch):
    """管线异常：HTTP 已是 200（响应头早发出），失败只能落在 error 事件里。"""
    def _boom(*a, **kw):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(main, "answer_question", _boom)
    monkeypatch.setattr(main, "log_query", lambda **kw: None)
    r = client.post("/query/stream", json={"question": "什么是ACID", "mode": "fast"})
    assert r.status_code == 200
    events = _parse_sse(r.text)
    assert events[-1][0] == "error"
    assert "ollama down" in events[-1][1]["detail"]


def test_query_stream_newlines_do_not_break_frames(client, monkeypatch):
    """答案含空行与代码块也不能破帧——这就是 data 用 JSON 单行编码的核心理由。"""
    _patch_stream_answer(monkeypatch, pieces=("第一段\n\n第二段```python\nx=1\n```",))
    r = client.post("/query/stream", json={"question": "什么是ACID", "mode": "fast"})
    events = _parse_sse(r.text)
    kinds = [e for e, _ in events]
    # 帧数必须与事件数一致：换行若破坏边界，会多出无法解析的碎片
    assert kinds == ["start", "metadata", "token", "done"]
    body = next(d for e, d in events if e == "token")
    assert body["text"] == "第一段\n\n第二段```python\nx=1\n```"
    assert events[-1][1]["answer"] == body["text"]


def test_query_stream_rejects_unknown_source_before_streaming(client, monkeypatch):
    """source 预校验：解析不到教材时与 /query 一致返回 400，
    而不是先回 200 再把错误塞进事件流（那样调用方只能从帧里才发现参数错了）。"""
    def _boom(q, sources=None):
        raise ValueError(f"没有语料源匹配 {q!r}；可用源：a.pdf、b.pdf")

    monkeypatch.setattr(main, "resolve_source_query", _boom)
    _patch_stream_answer(monkeypatch)
    r = client.post("/query/stream",
                    json={"question": "什么是ACID", "mode": "fast", "source": "不存在的教材"})
    assert r.status_code == 400
    assert "不存在的教材" in r.json()["detail"]


# ==================== /retrieve ====================
def test_retrieve(client, monkeypatch):
    monkeypatch.setattr(
        main, "retrieve_and_rerank",
        lambda q, force_no_rerank=False, **kwargs: [FakeDoc(score=0.9), FakeDoc(content="死锁四必要条件：互斥、占有并等待、不可抢占、循环等待。", source="os.txt", start=10, score=0.6)],
    )
    r = client.post("/retrieve", json={"question": "什么是死锁？", "mode": "accurate"})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "什么是死锁？"
    assert len(body["docs"]) == 2
    assert body["docs"][0]["rank"] == 1
    assert body["docs"][0]["source"] == "db.txt"
    assert "score" in body["docs"][0]


# ==================== /documents ====================
def test_documents_list(client):
    r = client.get("/documents")
    assert r.status_code == 200
    body = r.json()
    assert "files" in body and "total_chunks" in body
    assert isinstance(body["files"], list)


def test_documents_upload_ingest(client, monkeypatch):
    # 非首次建库（count>0）→ 走增量 upsert 路径
    monkeypatch.setattr(main, "chunk_file", lambda p: [FakeDoc(source=str(p))])
    monkeypatch.setattr(main, "add_documents", lambda chunks: len(chunks))
    r = client.post("/documents", files={"file": ("note.txt", b"hello world", "text/plain")})
    assert r.status_code == 200
    body = r.json()
    assert body["file"] == "note.txt"
    assert body["added_chunks"] == 1
    # 文件确实写入了临时 data 目录
    assert (Path(config.DATA_DIR) / "note.txt").exists()


def test_documents_upload_bad_extension(client):
    r = client.post("/documents", files={"file": ("evil.exe", b"x", "application/octet-stream")})
    assert r.status_code == 400
    assert "不支持的文件类型" in r.json()["detail"]


def test_documents_upload_empty(client):
    r = client.post("/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code == 400


def test_documents_upload_path_traversal(client):
    r = client.post("/documents", files={"file": ("../evil.txt", b"x", "text/plain")})
    assert r.status_code == 400
    assert "非法文件名" in r.json()["detail"]


def test_documents_delete(client, monkeypatch):
    monkeypatch.setattr(main, "delete_by_source", lambda source: None)
    r = client.delete("/documents/ghost.txt")  # 文件不存在，也应正常返回（索引移除）
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] is True
    assert body["filename"] == "ghost.txt"
    assert body["warning"] is None


def test_safe_filename_rejects_traversal():
    """路径穿越防护：_safe_filename 必须拒绝 .. / 分隔符 / 空名。

    注意：路径穿越样例用 os.path.sep / os.path.altsep 跨平台构造——
    写死 'C:\\evil.txt' 只在 Windows 上是穿越（反斜杠是分隔符），Linux 上
    反斜杠是普通字符、Path.name 等于原串导致误判（CI 曾在此挂，2026-08-22）。
    """
    bad = ["..", ".", "", ".." + os.path.sep + "evil.txt"]
    if os.path.altsep:  # Windows 的 altsep 是 '/'; POSIX 为 None
        bad.append(".." + os.path.altsep + "evil.txt")
    for name in bad:
        with pytest.raises(Exception):
            main._safe_filename(name)
    # 合法纯文件名放行
    assert main._safe_filename("note.txt") == "note.txt"


# ==================== /index ====================
def test_index_rebuild(client, monkeypatch):
    monkeypatch.setattr(main, "load_and_chunk_documents", lambda: [FakeDoc()])
    monkeypatch.setattr(main, "get_vector_store", lambda chunks=None: FakeCol(count=1))
    r = client.post("/index/rebuild")
    assert r.status_code == 200
    body = r.json()
    assert body["chunks"] == 1
    assert body["files"] == 1


def test_index_rebuild_empty_data(client, monkeypatch):
    """空知识库重建（P1 回归，2026-08-22）：data/ 无文档时 POST /index/rebuild
    不应 500——get_vector_store([]) 对 chromadb 空 upsert 曾抛
    'Non-empty lists are required'，已修复为返回空集合。"""
    monkeypatch.setattr(main, "load_and_chunk_documents", lambda: [])
    r = client.post("/index/rebuild")
    # main.rebuild_index 对空 chunks 显式返回 400（有提示），而不是 500 崩溃
    assert r.status_code == 400
    assert "没有可索引的文档" in r.json()["detail"]


def test_get_vector_store_empty_chunks(monkeypatch, tmp_path):
    """P1 回归（2026-08-22）：rag_core.get_vector_store([]) 不应崩溃。

    修复前：chromadb 对空 upsert 抛 ValueError('Non-empty lists are required')，
    空 data/ 目录点「全量重建」直接崩。修复后：跳过 upsert、返回 count()==0 的空集合。
    本用例隔离持久化目录，不依赖 Ollama（空 chunks 不触发嵌入）。
    """
    monkeypatch.setattr(config, "PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    import rag_core

    # 泄漏防护（与下方 test_get_vector_store_readonly_no_rebuild 同款）：
    # get_vector_store 内部会把 _chroma_client/_vector_store 模块级单例绑定到本测试的
    # 临时目录；不重置的话 monkeypatch 恢复 config 后单例仍指向 tmp，后续 test_retrieval
    # 全量跑会连到空库（Ollama 未启动时该组被 skip，泄漏不可见——2026-09-03 暴露修复）
    monkeypatch.setattr(rag_core, "_chroma_client", None)
    monkeypatch.setattr(rag_core, "_vector_store", None)
    col = rag_core.get_vector_store([])
    assert col.count() == 0


def test_index_stats(client, monkeypatch):
    col = FakeCol(count=7, metas=[{"source": "/x/a.md"}, {"source": "/x/b.md"}, {"source": "/x/b.md"}])
    # 评审 P2-3 后 /index/stats 走 get_vector_store(auto_rebuild=False)，mock 须收 kwargs
    monkeypatch.setattr(main, "get_vector_store", lambda *a, **k: col)
    r = client.get("/index/stats")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 7
    assert body["source_files"] == 2  # 按 source 去重
    assert body["collection"] == config.COLLECTION_NAME


def test_index_stats_unavailable(client, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(main, "get_vector_store", _boom)
    r = client.get("/index/stats")
    assert r.status_code == 503


# ==================== 鉴权 / 上传上限（评审 P3 安全小项） ====================
def test_require_auth_enforced_when_enabled(client, monkeypatch):
    """AUTH_ENABLED=true：无密钥/错密钥 401，正确密钥放行。
    （常量时间比较 secrets.compare_digest 只改实现不改行为——防时序侧信道。）"""
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    monkeypatch.setattr(config, "API_KEY", "secret-key")
    r = client.get("/health")
    assert r.status_code == 401
    r = client.get("/health", headers={"X-API-Key": "wrong-key"})
    assert r.status_code == 401
    r = client.get("/health", headers={"X-API-Key": "secret-key"})
    assert r.status_code == 200


def test_documents_upload_too_large(client, monkeypatch):
    """评审 P3 回归：超过 MAX_UPLOAD_MB → 413，且有界读取不落盘。"""
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0)  # 上限置 0：任何非空内容都超限
    r = client.post("/documents", files={"file": ("big.txt", b"x" * 10, "text/plain")})
    assert r.status_code == 413
    assert not (Path(config.DATA_DIR) / "big.txt").exists()


# ==================== 枚举题锚点扩展召回（2026-09-11） ====================
def _mk_src_doc(source, start, text="内容"):
    import rag_core

    return rag_core.Document(text, {"source": source, "start_index": start})


def test_enum_expand_merges_anchor_recall_and_dedups(monkeypatch):
    """锚点扩展：从基础候选推导锚点 source → 锚点内密集召回 → 合并去重。"""
    import rag_core

    base = [_mk_src_doc("s1.md", 0), _mk_src_doc("s1.md", 10), _mk_src_doc("s2.md", 0)]
    extra = [_mk_src_doc("s1.md", 10), _mk_src_doc("s1.md", 20)]  # start=10 与 base 重复
    seen = {}

    def _fake_retrieve(q, k=None, where=None):
        seen["k"], seen["where"] = k, where
        return extra

    monkeypatch.setattr(rag_core, "retrieve_vector", _fake_retrieve)
    out = rag_core._enum_expand("Streamlit 有哪些输入组件", base, None, 40)

    # 锚点 = 基础候选前 ENUM_ANCHOR_SOURCES 块的 source 去重
    assert set(seen["where"]["source"]["$in"]) == {"s1.md", "s2.md"}
    assert seen["k"] == config.ENUM_ANCHOR_TOP_K
    keys = [(d.metadata["source"], d.metadata["start_index"]) for d in out]
    assert keys == [("s1.md", 0), ("s1.md", 10), ("s2.md", 0), ("s1.md", 20)]
    # 只有扩展进来的块带标记，事后可追溯「它是常规召回还是锚点捞回的」
    assert [bool(d.metadata.get("enum_expanded")) for d in out] == [False, False, False, True]


def test_enum_expand_respects_existing_filter(monkeypatch):
    """已有教材过滤时锚点就是过滤集——绝不越出范围（安全性的关键）。"""
    import rag_core

    seen = {}

    def _fake_retrieve(q, k=None, where=None):
        seen["where"] = where
        return []

    monkeypatch.setattr(rag_core, "retrieve_vector", _fake_retrieve)
    rag_core._enum_expand("有哪些", [_mk_src_doc("s1.md", 0)], ["s1.md"], 40)
    assert seen["where"]["source"]["$in"] == ["s1.md"]


def test_enum_expand_noop_without_candidates(monkeypatch):
    """基础候选为空时直接返回，不发多余召回（省一次 embedding 调用）。"""
    import rag_core

    called = []
    monkeypatch.setattr(rag_core, "retrieve_vector",
                        lambda q, k=None, where=None: called.append(1) or [])
    assert rag_core._enum_expand("有哪些", [], None, 40) == []
    assert called == []


def test_retrieve_and_rerank_widens_budget_only_for_enum(monkeypatch):
    """开关开启时：枚举题只放宽「最终进 LLM 的块数」，**不放宽 RRF 候选池**。

    2026-09-11 实测修正：曾把 candidate_k 也设成 ENUM_CANDIDATE_K，结果
    _enum_expand 的合并上限 max(pool_k, ENUM_CANDIDATE_K) 恒等于 base 大小，
    锚点扩展块被整段截掉（净新增 17 块 → 标记 0，功能静默失效）。
    故这里钉死 candidate_k 必须为 None，扩展余量由合并上限留出。
    """
    import rag_core

    captured = {}
    monkeypatch.setattr(config, "ENUM_EXPAND_ENABLED", True)
    monkeypatch.setattr(rag_core, "_resolve_source_filter", lambda q, sf: None)

    def _fake_once(question, force_no_rerank, sources, candidate_k=None, top_n=None):
        captured["candidate_k"], captured["top_n"] = candidate_k, top_n
        return []

    monkeypatch.setattr(rag_core, "_retrieve_once", _fake_once)

    rag_core.retrieve_and_rerank("Streamlit 有哪些输入组件")
    assert captured == {"candidate_k": None, "top_n": config.ENUM_RERANK_TOP_N}

    captured.clear()
    rag_core.retrieve_and_rerank("什么是死锁")
    assert captured == {"candidate_k": None, "top_n": None}


def test_enum_expand_disabled_by_default(monkeypatch):
    """默认关：即便问题是枚举题也不放宽预算——按评估纪律，未跑 A/B 前不翻默认值。"""
    import rag_core

    captured = {}
    monkeypatch.setattr(config, "ENUM_EXPAND_ENABLED", False)
    monkeypatch.setattr(rag_core, "_resolve_source_filter", lambda q, sf: None)
    monkeypatch.setattr(
        rag_core, "_retrieve_once",
        lambda q, f, s, candidate_k=None, top_n=None:
            captured.update(candidate_k=candidate_k, top_n=top_n) or [])
    rag_core.retrieve_and_rerank("Streamlit 有哪些输入组件")
    assert captured == {"candidate_k": None, "top_n": None}


def test_enum_expand_actually_contributes_chunks(monkeypatch):
    """回归守卫（2026-09-11）：锚点扩展块必须真的进得了合并结果。

    背景：合并上限写成 max(pool_k, ENUM_CANDIDATE_K)，而 pool_k 又等于
    candidate_k=ENUM_CANDIDATE_K，两者同值 → base 先吃满上限 → 扩展块被
    merged[:candidate_k] 整段截掉。实测表现为「净新增 17 块、enum_expanded 标记 0」，
    功能静默失效但没有任何报错。这条用例把「合并上限 > base 池」钉死。
    """
    import rag_core

    def _mk(source, start):
        from langchain_core.documents import Document
        return Document(page_content=f"块{start}", metadata={"source": source, "start_index": start})

    base = [_mk("教材A.md", i) for i in range(config.RERANK_CANDIDATE_K)]
    extra = [_mk("教材A.md", 1000 + i) for i in range(10)]

    monkeypatch.setattr(rag_core, "retrieve_vector", lambda q, k=None, where=None: extra)

    merged = rag_core._enum_expand(
        "有哪些组件", base, None, max(config.RERANK_CANDIDATE_K, config.ENUM_CANDIDATE_K)
    )
    n_exp = sum(1 for d in merged if d.metadata.get("enum_expanded"))
    assert n_exp == len(extra), (
        f"锚点扩展块被全部截掉（净新增 {len(extra)} 块，实际进入 {n_exp} 块）——"
        f"检查 _enum_expand 的合并上限是否小于等于 base 池大小"
    )
    assert len(merged) > len(base)


# ==================== 评审 P2 回归 ====================
def test_split_structured_offset_remap():
    """评审 P2-1 回归：含代码块的文档，切分产出的 start_index 必须是「原文偏移」。
    修复前偏移落在哨兵保护后的坐标系——代码块（原文 314 字符）被换成 4 字符哨兵后，
    其后所有块的 start_index 整体前漂 310 字符，text[start:start+len(chunk)] != chunk。"""
    import rag_core

    head = "前导文字。" * 30                            # 150 字符单行
    code = "```python\n" + "x = 1\n" * 50 + "```"      # 314 字符 → 哨兵 §B0§ 仅 4 字符
    text = head + "\n" + code + "\n" + "尾注文字。" * 30  # 尾注起点原文偏移 = 466
    chunks = rag_core._split_structured(text, 200, 0, rag_core._SEPARATORS)
    assert len(chunks) >= 2
    hit_code = False
    for chunk, start, _chapter in chunks:
        assert 0 <= start and start + len(chunk) <= len(text)
        # 核心断言：报告的偏移处，原文必须与块内容逐字对齐（引用溯源可定位的前提）
        assert text[start : start + len(chunk)] == chunk, (
            f"start_index={start} 处原文与块内容不匹配（偏移漂移）"
        )
        if "```" in chunk:
            hit_code = True
    assert hit_code  # 代码块被整体保护并进入某个块


def test_bm25_cache_built_once_per_corpus(monkeypatch):
    """评审 P2-4 回归：BM25 索引按语料版本缓存（同语料两次查询只建一次），
    _invalidate_all_chunks（增删/重建索引）后重建。"""
    import rag_core

    docs = [rag_core.Document("事务的 ACID 特性。", {"source": "a.txt", "start_index": 0}),
            rag_core.Document("死锁的四个必要条件。", {"source": "a.txt", "start_index": 20})]

    class _FakeCol:
        def count(self):
            return len(docs)

        def get(self, include=None):
            return {"documents": [d.page_content for d in docs],
                    "metadatas": [d.metadata for d in docs]}

    monkeypatch.setattr(rag_core, "get_vector_store", lambda *a, **k: _FakeCol())
    monkeypatch.setattr(rag_core, "retrieve_vector",
                        lambda q, k=None, where=None: [rag_core.Document("x", {"source": "a.txt", "start_index": 0})])
    built = {"n": 0}
    _orig_bm25 = rag_core._BM25

    def _spy(d):
        built["n"] += 1
        return _orig_bm25(d)

    monkeypatch.setattr(rag_core, "_BM25", _spy)
    monkeypatch.setattr(rag_core, "_all_chunks_cache", None)
    monkeypatch.setattr(rag_core, "_all_chunks_count", -1)
    monkeypatch.setattr(rag_core, "_bm25_cache", None)

    rag_core.hybrid_retrieve("ACID", k=5)
    rag_core.hybrid_retrieve("ACID", k=5)
    assert built["n"] == 1, "同语料两次查询不应重建 BM25 索引"
    rag_core._invalidate_all_chunks()
    rag_core.hybrid_retrieve("ACID", k=5)
    assert built["n"] == 2, "语料失效（增删/重建）后应重建 BM25 索引"


def test_get_vector_store_readonly_no_rebuild(monkeypatch, tmp_path):
    """评审 P2-3 回归：auto_rebuild=False 时空库不触发全量重建（只读端点零副作用）。"""
    import rag_core

    def _no_rebuild():
        raise AssertionError("auto_rebuild=False 不应触发 load_and_chunk_documents")

    monkeypatch.setattr(config, "PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_core, "load_and_chunk_documents", _no_rebuild)
    # 绕开其他用例遗留的模块级单例缓存，确保真实走「空库 + auto_rebuild=False」分支
    monkeypatch.setattr(rag_core, "_chroma_client", None)
    monkeypatch.setattr(rag_core, "_vector_store", None)
    col = rag_core.get_vector_store(auto_rebuild=False)
    assert col.count() == 0  # 空库如实返回，没有建索引副作用


def test_build_citations_image_fields():
    """P2 多模态（2026-09-05）：图像块 citation 透出 type/image_path/page；文本块不带这些键。"""
    import rag_core

    class ImgDoc:
        page_content = "组织架构框图：项目管理者下设产品/设计/开发/测试四组。"
        metadata = {
            "source": "data/教材/微信小程序开发实战.pdf", "start_index": -29000,
            "type": "image", "image_path": "data/images/微信小程序开发实战/p029_i00.png", "page": 29,
        }

    class TxtDoc:
        page_content = "事务的 ACID 特性。"
        metadata = {"source": "data/数据库-事务与ACID.md", "start_index": 120}

    cits = rag_core.build_citations([ImgDoc(), TxtDoc()])
    assert cits[0]["type"] == "image"
    assert cits[0]["image_path"] == "data/images/微信小程序开发实战/p029_i00.png"
    assert cits[0]["page"] == 29
    assert "type" not in cits[1] and "image_path" not in cits[1]


# ==================== Query 改写：多轮指代消解（2026-09-18） ====================
def _patch_rewrite_llm(monkeypatch, output=None, calls=None, error=None):
    """打桩 rag_core.ollama_chat：改写走它（temperature=0），不碰真 Ollama。"""
    import rag_core

    def fake_chat(messages, temperature=None):
        if calls is not None:
            calls.append({"temperature": temperature, "messages": messages})
        if error:
            raise error
        return output

    monkeypatch.setattr(rag_core, "ollama_chat", fake_chat)


def test_rewrite_question_resolves_coreference(monkeypatch):
    """有 history：LLM 把「那一章讲的编辑器」改写为独立问题，剥掉包裹引号。"""
    import rag_core

    calls = []
    _patch_rewrite_llm(monkeypatch, output='"Linux 第 3 章 Vi 编辑器有哪些常用操作？"', calls=calls)
    history = "User: Linux 第三章讲了什么\nAssistant: 讲了 Vi 编辑器的基本用法"
    rewritten, ok = rag_core.rewrite_question("那一章讲的编辑器怎么用？", history)
    assert ok is True
    assert rewritten == "Linux 第 3 章 Vi 编辑器有哪些常用操作？"
    # 改写要确定性：temperature 必须压到 0
    assert calls and calls[0]["temperature"] == 0.0


def test_rewrite_question_noop_without_history(monkeypatch):
    """单轮（history 空）零改写、零 LLM 调用——单轮行为与旧版完全一致（评估口径保护）。"""
    import rag_core

    calls = []
    _patch_rewrite_llm(monkeypatch, output="不该被调用", calls=calls)
    rewritten, ok = rag_core.rewrite_question("什么是ACID", "")
    assert ok is False and rewritten == "什么是ACID"
    rewritten, ok = rag_core.rewrite_question("什么是ACID", "   ")
    assert ok is False and rewritten == "什么是ACID"
    assert calls == []


def test_rewrite_question_disabled_by_switch(monkeypatch):
    """QUERY_REWRITE_ENABLED=false：即使有 history 也不改写（kill-switch）。"""
    import rag_core

    monkeypatch.setattr(config, "QUERY_REWRITE_ENABLED", False)
    calls = []
    _patch_rewrite_llm(monkeypatch, output="不该被调用", calls=calls)
    rewritten, ok = rag_core.rewrite_question("那个定理", "User: 讲讲第三章")
    assert ok is False and rewritten == "那个定理"
    assert calls == []


def test_rewrite_question_falls_back_on_llm_failure(monkeypatch):
    """Ollama 宕机/超时：吞异常回退原问题——改写是增强项，不是新故障点。"""
    import rag_core

    _patch_rewrite_llm(monkeypatch, error=RuntimeError("connection refused"))
    rewritten, ok = rag_core.rewrite_question("那个定理", "User: 讲讲第三章")
    assert ok is False and rewritten == "那个定理"


def test_rewrite_question_rejects_garbage_output(monkeypatch):
    """模型不听指令输出口水话（超长解释）：判废回退，不污染下游检索。"""
    import rag_core

    _patch_rewrite_llm(monkeypatch, output="好的，根据对话历史，用户想了解的内容可以归纳为以下几点：第一……" * 10)
    rewritten, ok = rag_core.rewrite_question("那个定理", "User: 讲讲第三章")
    assert ok is False and rewritten == "那个定理"
    # 空输出同样判废
    _patch_rewrite_llm(monkeypatch, output="   ")
    rewritten, ok = rag_core.rewrite_question("那个定理", "User: 讲讲第三章")
    assert ok is False and rewritten == "那个定理"


def test_answer_question_rewritten_question_flows_through_pipeline(monkeypatch):
    """改写后的问题必须贯穿缓存 key / 检索 / 缓存写入，且 res.rewritten 可观测。"""
    import rag_core

    _patch_answer_pipeline(monkeypatch, [_mk_doc(0.90)])
    seen = {}

    def fake_lookup(q, m):
        seen["lookup"] = q
        return None

    def fake_retrieve(q, force_no_rerank=False, **kwargs):
        seen["retrieve"] = q
        return [_mk_doc(0.90)]

    monkeypatch.setattr(rag_core.semantic_cache, "lookup", fake_lookup)
    monkeypatch.setattr(rag_core, "retrieve_and_rerank", fake_retrieve)
    monkeypatch.setattr(rag_core, "ollama_chat",
                        lambda messages, temperature=None: "Linux 第 3 章 Vi 编辑器有哪些常用操作？")

    res = rag_core.answer_question(
        "那一章讲的编辑器怎么用？", mode="fast", use_cache=True,
        history="User: Linux 第三章讲了什么\nAssistant: 讲了 Vi 编辑器",
    )
    expected = "Linux 第 3 章 Vi 编辑器有哪些常用操作？"
    assert seen["lookup"] == expected   # 缓存 key 用改写后问题
    assert seen["retrieve"] == expected  # 检索用改写后问题
    assert res["rewritten"] == expected  # 日志/UI 可观测


def test_answer_question_use_rewrite_false_bypass(monkeypatch):
    """use_rewrite=False 强制旁路：即使有 history 也不改写——评估口径的显式逃生门。"""
    import rag_core

    _patch_answer_pipeline(monkeypatch, [_mk_doc(0.90)])
    seen = {}

    def fake_lookup(q, m):
        seen["lookup"] = q
        return None

    monkeypatch.setattr(rag_core.semantic_cache, "lookup", fake_lookup)
    monkeypatch.setattr(rag_core, "ollama_chat",
                        lambda messages, temperature=None: "不该被调用")

    res = rag_core.answer_question(
        "那一章讲的编辑器怎么用？", mode="fast", use_cache=True,
        history="User: Linux 第三章讲了什么", use_rewrite=False,
    )
    assert seen["lookup"] == "那一章讲的编辑器怎么用？"
    assert res["rewritten"] is None


def test_answer_question_single_turn_no_rewrite_no_llm_call(monkeypatch):
    """单轮默认路径：不触发改写 LLM（不加额外延迟），rewritten 为 None。"""
    import rag_core

    _patch_answer_pipeline(monkeypatch, [_mk_doc(0.90)])
    calls = []

    def fake_lookup(q, m):
        return None

    monkeypatch.setattr(rag_core.semantic_cache, "lookup", fake_lookup)
    monkeypatch.setattr(rag_core, "ollama_chat",
                        lambda messages, temperature=None: calls.append(1) or "x")

    res = rag_core.answer_question("什么是ACID", mode="fast", use_cache=True)
    assert calls == []          # 零改写调用：单轮延迟与旧版完全一致
    assert res["rewritten"] is None
