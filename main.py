"""
main.py — CourseRAG FastAPI 服务入口（部署层，2026-08-20 新增）。

把 rag_core 的「索引构建 / 检索 / 重排 / 生成」能力封装为 RESTful 端点，
与 Streamlit（app.py）共用同一套核心模块，无重复逻辑：
  - 问答        POST /query              检索 + 重排 + 生成（含引用溯源）
  - 流式问答    POST /query/stream       同上，但以 SSE 逐事件推送（start/metadata/token/done/error）
  - 检索        POST /retrieve           仅检索 + 重排，不生成（供前端 / 调试）
  - 文档        GET/POST /documents      列出 / 上传（保存 + 增量索引）
  - 文档删除    DELETE /documents/{name} 删文件 + 从索引移除
  - 索引        POST /index/rebuild      全量重建（data/ 全部）
  - 统计        GET /index/stats         向量库块数 / 源文档数
  - 原图        GET /images/{path}       按 image_path 返回图像字节（供多模态消费者取像素，B2）
  - 健康检查    GET /health              依赖 / Ollama / 索引自检

启动：uvicorn main:app --host 0.0.0.0 --port 8000（交互文档 http://localhost:8000/docs）

注意（与 app.py 相同的时序坑）：reranker 权重加载需访问 HuggingFace，本机
huggingface.co 不可达会超时。必须在 import 任何可能拉起 sentence_transformers /
huggingface_hub 的模块之前设置 HF_ENDPOINT——这里在 import rag_core 之前设置。
"""
import os

# 默认走镜像；若本机已缓存权重则强制离线，彻底不碰网络（与 app.py 一致）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_rerank_cache = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3"
)
if os.path.isdir(_rerank_cache):
    os.environ["HF_HUB_OFFLINE"] = "1"

import json
import queue
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

import config
from api_security import RateLimitMiddleware, require_auth
from api_schemas import (
    CheckItem,
    Citation,
    DeleteResponse,
    DocInfo,
    DocumentsResponse,
    HealthResponse,
    IndexStats,
    IngestResponse,
    PaperActionResponse,
    PaperBlocksResponse,
    PaperDeleteResponse,
    PaperInfo,
    PaperList,
    PaperUploadItem,
    PaperUploadResponse,
    QueryRequest,
    QueryResponse,
    RebuildResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from health import collect_status
from query_log import log_query
import semantic_cache
from rag_core import (
    add_documents,
    answer_question,
    build_citations,
    chunk_file,
    delete_by_source,
    get_vector_store,
    load_and_chunk_documents,
    resolve_source_query,
    retrieve_and_rerank,
)

# 允许的上传扩展名（与 rag_core._load_text 支持的后缀保持一致）
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".pptx"}


# ==================== 通用工具 ====================
def _retrieve(question: str, mode: str, source: str = None):
    """统一检索入口：fast=纯向量 top-N；accurate=向量+BM25（+reranker）。
    source（P1a）：教材过滤子串（None=按问题关键词自动路由）。
    仅供 /retrieve（纯检索端点）使用；生成问答统一走 rag_core.answer_question。"""
    return retrieve_and_rerank(
        question, force_no_rerank=(mode != "accurate"), source_filter=source
    )


def _safe_filename(name: str) -> str:
    """防路径穿越：只允许纯文件名，拒绝 .. / 分隔符。"""
    if not name or name != Path(name).name or name in (".", ".."):
        raise HTTPException(status_code=400, detail=f"非法文件名: {name!r}")
    return name


async def _read_upload(file: UploadFile):
    """有界读取上传体；超过 MAX_UPLOAD_MB 立即 413。"""
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    buf = bytearray()
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"文件过大（>{config.MAX_UPLOAD_MB}MB）。"
                "如需接入大体积扫描版 PDF，请用本地脚本 ingest_ocr.py，不经上传接口。",
            )
    content = bytes(buf)
    if not content.strip():
        raise HTTPException(status_code=400, detail="文件内容为空")
    return content


def _load_index_or_raise():
    """确保向量库可用；Ollama 未启动 / 库损坏时给出可操作的报错。
    评审 P2-3：/index/stats 是只读端点，空库不触发自动重建。"""
    try:
        return get_vector_store(auto_rebuild=False)
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"向量库不可用：{e}。请确认 Ollama 已启动（ollama serve）且模型已拉取。",
        )


# ==================== 应用生命周期 ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    profile, collection, persist = config.profile_summary()
    print(f"[profile] APP_PROFILE={profile} collection={collection} persist={persist}")
    # 启动预热：懒加载向量库（库为空且有资料会自动重建）。失败不阻塞启动——
    # Ollama 没启动时健康检查会明确报出来，而不是让服务起不来。
    try:
        get_vector_store()
        print("[api] vector store ready")
    except Exception as e:
        print(f"[api] warmup skipped (service will still start): {e}")
    yield


app = FastAPI(
    title="CourseRAG API",
    description="专业课 / 课本问答 RAG 服务。检索 + 重排 + 生成，来源可溯源。"
    "交互式文档见 /docs（Swagger UI）。",
    version="1.0.0",
    lifespan=lifespan,
    # Gate 2：全局鉴权依赖（AUTH_ENABLED=false 时为 no-op，零摩擦本地部署）。
    dependencies=[Depends(require_auth)],
)

# 跨域：演示 / 前后端分离放开（默认 "*"）；生产在 .env 用 CORS_ORIGINS 收紧为具体域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gate 2：按 IP 限流（RATE_LIMIT=0 时为 no-op）。多副本需换 Redis（文档路线 β/ε）。
app.add_middleware(RateLimitMiddleware, per_minute=config.RATE_LIMIT)

# ---------- /api 前缀（生产构建走同源，前端 BASE='/api'） ----------
# 开发模式由 Vite proxy 负责 rewrite；这里让 `npm run build` 后同源访问也能工作。
# 只改写 /api/ 开头的路径，/docs、/health 等不受影响。
@app.middleware("http")
async def _strip_api_prefix(request, call_next):
    if request.scope.get("path", "").startswith("/api/"):
        request.scope["path"] = request.scope["path"][4:]
        request.scope["raw_path"] = request.scope["path"].encode()
    return await call_next(request)



# ==================== 服务信息 ====================
@app.get("/api-info", tags=["meta"])
def service_info():
    """服务元信息（端点一览）。"""
    return {
        "name": "CourseRAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "GET  /health",
            "POST /query",
            "POST /query/stream",
            "POST /retrieve",
            "GET  /documents",
            "POST /documents",
            "DELETE /documents/{name}",
            "POST /index/rebuild",
            "GET  /index/stats",
        ],
        "models": {"embedding": config.EMBEDDING_MODEL, "llm": config.LLM_MODEL},
        # C3：如实上报生成 provider，避免「配置了云端却以为还走本地」的误判
        "llm_provider": config.LLM_PROVIDER,
        "ollama": config.OLLAMA_BASE_URL,
    }


# ==================== 健康检查 ====================
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """依赖 / Ollama / Reranker 权重自检（复用 health.py），附带向量库块数。

    依赖清单按 API 视角检查（核心栈即可，无需 streamlit / sentence-transformers——
    镜像内 reranker 默认关，不装 torch 是预期行为）。
    """
    checks = collect_status(deps=["ollama", "chromadb", "pypdf", "docx", "dotenv"])
    count = None
    try:
        # 评审 P2-3：只读端点零副作用——空库不触发自动重建，如实返回 count=None/0
        count = get_vector_store(auto_rebuild=False).count()
    except Exception:
        pass
    return HealthResponse(
        ok=all(ok for _, ok, _ in checks),
        checks=[CheckItem(label=l, ok=ok, msg=m) for l, ok, m in checks],
        vector_store_count=count,
    )


# ==================== 问答 / 检索 ====================
@app.post("/query", response_model=QueryResponse, tags=["rag"])
def query(req: QueryRequest):
    """核心问答：检索 + 重排 + 生成，返回带 [n] 内联引用的回答与来源列表。

    评审 P1（2026-08-30）起走 rag_core.answer_question 统一入口：accurate 模式
    在 API 侧同样生效硬拒答 + 接地自检闸门（此前仅 Streamlit 有，B 组拒答率口径
    只对 UI 成立）；缓存策略（拒答/未接地不缓存）也与 UI 完全一致。
    响应含 latency_ms 便于前端展示耗时。
    """
    t0 = time.time()
    try:
        mode_label = "精确" if req.mode == "accurate" else "快速"
        res = answer_question(
            req.question,
            mode=req.mode,
            custom_prompt=req.custom_prompt,
            history=req.history or "",
            source=req.source,
        )
        answer = res["answer"]
        citations = res["citations"]

        log_query(
            question=req.question,
            mode=mode_label,
            citations=citations,
            answer=answer,
            latency_ms=int((time.time() - t0) * 1000),  # #15 看板：整轮耗时
            rejected=(res["rejected"] if req.mode == "accurate" else None),
            rewritten=res.get("rewritten"),  # 多轮指代消解观测（改写发生时非 None）
        )
        # grounded 判定与 query_log._is_grounded 的启发式保持一致：
        # 非空 且 未声称「无法回答」→ 视为接地（硬拒/未接地的替换文案均含
        # 「无法回答」，故 accurate 闸门结果与该启发式天然一致）
        grounded = bool(answer.strip()) and "无法回答" not in answer
        return QueryResponse(
            answer=answer,
            mode=req.mode,
            citations=[Citation(**c) for c in citations],
            latency_ms=int((time.time() - t0) * 1000),
            grounded=grounded,
            low_confidence=res["low_confidence"],
            cached=res["cached"],
        )
    except HTTPException:
        raise
    except ValueError as e:  # P1a：显式教材子串解析不到源 → 400 带可用源清单
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"生成失败：{e}（provider={config.LLM_PROVIDER}，model={config.LLM_MODEL}）。本地 provider 需确认 Ollama 已启动；云端 provider 需确认 key 与网络。",
        )


# ==================== 流式问答（SSE） ====================
# 单帧之间最多等这么久没动静就发一条 SSE 注释帧保活。反代（Nginx 默认
# proxy_read_timeout 60s）与部分浏览器会把长时间无数据的连接判为死连接；
# 精排与首 token 之间的间隔在 CPU 兜底时会超过 30s，故必须保活。
_SSE_HEARTBEAT_SECONDS = 15


def _sse(event: str, data: dict) -> str:
    """构造一帧 SSE：`event:` 行 + `data:` 行 + 空行结束。

    为什么 data 用 JSON 而不是裸文本：json.dumps 的输出恒为**单行**——字符串里
    的换行会被转义成 \\n。代码块、多段回答因此不会破坏「空行 = 帧结束」的边界。
    若直接拼裸文本，答案里的空行会让客户端把一帧切成多帧。
    """
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _query_event_stream(req: QueryRequest, source):
    """把 answer_question 的同步回调转成可逐步消费的 SSE 事件流。

    为什么要开线程：answer_question 是同步阻塞函数，on_token 在它**内部**被同步
    调用。若在生成器体内直接调用它，必须等整轮跑完才能 yield 第一个事件，
    「流式」就只剩形式。这里让管线跑在后台线程、回调把数据塞进队列，生成器
    从队列取——同步回调因此被转成了可增量消费的流。

    事件序：start → metadata → token* → done（失败则 error 收尾）
      start     已受理（mode / question）
      metadata  检索+精排完成、硬拒答判定出炉，附 citations 与 rejected。
                此刻生成尚未开始——引用先到、正文后到，前端可立刻渲染来源列表；
                rejected=true 时前端能马上提示「知识库未覆盖」，不必干等生成。
      token     生成片段（仅 fast 模式；accurate 需完整答案做接地自检，故缓冲）
      done      终态：完整答案 + 耗时 + grounded/low_confidence/cached/rejected
      error     HTTP 200 与响应头此时早已发出，无法再改用状态码表达失败，
                故用事件承载——这是 SSE 的固有取舍，客户端按事件类型分支即可。
    """
    t0 = time.time()
    q: "queue.Queue[tuple]" = queue.Queue()
    box: dict = {}

    def on_retrieved(citations, rejected):
        q.put(("metadata", {"citations": citations, "rejected": bool(rejected)}))

    def on_token(piece):
        q.put(("token", {"text": piece}))

    def worker():
        try:
            box["res"] = answer_question(
                req.question,
                mode=req.mode,
                custom_prompt=req.custom_prompt,
                history=req.history or "",
                source=source,
                on_retrieved=on_retrieved,
                on_token=on_token,
            )
        except Exception as e:  # 线程内异常必须搬出到主流程，否则事件流会静默截断
            box["err"] = e
        finally:
            q.put(("__end__", None))

    threading.Thread(target=worker, daemon=True).start()
    yield _sse("start", {"mode": req.mode, "question": req.question})

    while True:
        try:
            kind, payload = q.get(timeout=_SSE_HEARTBEAT_SECONDS)
        except queue.Empty:
            yield ": keep-alive\n\n"  # SSE 注释帧，客户端按规范忽略
            continue
        if kind == "__end__":
            break
        yield _sse(kind, payload)

    if "err" in box:
        yield _sse("error", {"detail": f"生成失败：{box['err']}"})
        return

    res = box["res"]
    latency_ms = int((time.time() - t0) * 1000)
    answer = res["answer"]
    log_query(
        question=req.question,
        mode="精确" if req.mode == "accurate" else "快速",
        citations=res["citations"],
        answer=answer,
        latency_ms=latency_ms,
        rejected=(res["rejected"] if req.mode == "accurate" else None),
        rewritten=res.get("rewritten"),  # 多轮指代消解观测（改写发生时非 None）
    )
    # grounded 判定与 /query 及 query_log._is_grounded 保持同一启发式
    yield _sse("done", {
        "answer": answer,
        "mode": req.mode,
        "citations": res["citations"],
        "latency_ms": latency_ms,
        "grounded": bool(answer.strip()) and "无法回答" not in answer,
        "low_confidence": res["low_confidence"],
        "cached": res["cached"],
        "rejected": res["rejected"],
    })


@app.post("/query/stream", tags=["rag"])
def query_stream(req: QueryRequest):
    """流式问答（Server-Sent Events）：与 /query 同一套管线，改为一帧帧推送。

    与 /query 的差异仅在传输方式——缓存、检索、硬拒答、接地自检、日志口径完全一致，
    因为两者最终都调用 rag_core.answer_question（统一入口，不存在第二套逻辑）。

    为什么是 POST + SSE 而不是 EventSource：浏览器原生 EventSource 只支持 GET，
    问题文本塞进 query string 既受长度限制又会进访问日志。业界做法（OpenAI /
    Anthropic 的流式接口同理）是用 POST + SSE，客户端以 fetch + ReadableStream
    消费。curl 直接验证：
        curl -N -X POST http://127.0.0.1:8000/query/stream \\
             -H 'Content-Type: application/json' \\
             -d '{"question":"什么是ACID","mode":"fast"}'
    """
    # source 预校验：与 /query 的 400 语义对齐。若把解析留到流内，HTTP 200 与响应头
    # 早已发出、无法再退回 400，调用方只能从 error 事件里才得知参数错了。
    source = req.source
    if isinstance(source, str):
        try:
            source = resolve_source_query(source)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return StreamingResponse(
        _query_event_stream(req, source),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 关闭 Nginx 响应缓冲，否则帧会被攒批后才下发
        },
    )


@app.post("/retrieve", response_model=RetrieveResponse, tags=["rag"])
def retrieve(req: RetrieveRequest):
    """仅检索 + 重排（不调用 LLM），返回来源列表——供前端预检索 / 调试 / 评估。
    P1a：可选 source 教材过滤（子串），None=按问题关键词自动路由。"""
    try:
        docs = _retrieve(req.question, req.mode, source=req.source)
        return RetrieveResponse(
            query=req.question,
            mode=req.mode,
            docs=[Citation(**c) for c in build_citations(docs)],
        )
    except HTTPException:
        raise
    except ValueError as e:  # P1a：显式教材子串解析不到源 → 400 带可用源清单
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败：{e}")


# ==================== 文档管理 ====================
def _base_source(s: str) -> str:
    """取 source 的文件名部分并统一分隔符，用于跨 OS / 跨挂载路径比对。

    背景（2026-09-07 修复）：索引在 Windows 宿主构建，metadata.source 形如
    `data\\教材\\xx.pdf`（反斜杠、相对路径）；而容器内 DATA_DIR=/app/data，
    `str(Path(...))` 拼出 `/app/data/xx.pdf`。全路径比对必然全 0，
    因此统一按**文件名**匹配（data/ 及其子目录下文件名唯一）。
    """
    return (s or "").replace("\\", "/").rsplit("/", 1)[-1]


@app.get("/documents", response_model=DocumentsResponse, tags=["documents"])
def list_documents():
    """列出 data/ **及其子目录**下所有文档，并给出各自在向量库中的块数（0=尚未索引）。

    两处修复（2026-09-07）：
    1. 块数按文件名匹配而非全路径——否则 Windows 建索引 + Linux 容器读取时全失配；
    2. 递归 rglob 取代 iterdir——原实现只扫一级，data/教材/ 下的 PDF/PPTX 全部漏列。
    """
    if config.APP_PROFILE == "paper":
        import paper_ingest

        files, total = [], 0
        for s in paper_ingest.list_states():
            p = Path(s.get("source_path", ""))
            n = int(s.get("chunks", 0) or 0)
            files.append(
                DocInfo(
                    name=s.get("filename", ""),
                    size_bytes=p.stat().st_size if p.exists() else 0,
                    chunks=n,
                )
            )
            total += n
        return DocumentsResponse(files=files, total_chunks=total)

    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    files, total = [], 0
    try:
        # 评审 P2-3：只读端点零副作用（未建索引时各文件如实显示 chunks=0）
        col = get_vector_store(auto_rebuild=False)
        all_meta = col.get(include=["metadatas"])["metadatas"]
        from collections import Counter

        chunk_counter = Counter(_base_source(m.get("source")) for m in all_meta if m)
    except Exception:
        chunk_counter = {}

    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES:
            n = chunk_counter.get(p.name, 0)
            files.append(
                DocInfo(name=p.name, size_bytes=p.stat().st_size, chunks=n)
            )
            total += n
    return DocumentsResponse(files=files, total_chunks=total)


@app.post("/documents", response_model=IngestResponse, tags=["documents"])
async def upload_document(file: UploadFile = File(..., description="PDF / TXT / MD / DOCX 文件")):
    """上传课程资料：保存到 data/ 并增量索引（稳定 id 去重，重复上传不产生重复块）。

    首次建库（向量库为空）时自动退化为从 data/ 全量重建，避免已有资料被漏索引——
    与 app.py 侧边栏的首次建库行为保持一致。
    """
    if config.APP_PROFILE == "paper":
        item = await _save_paper_upload(file)
        return IngestResponse(
            file=item.filename,
            saved_to=str(Path(config.PAPERS_RAW_DIR) / item.filename),
            added_chunks=0,
        )

    name = _safe_filename(file.filename or "")
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型 {suffix!r}，仅支持 {sorted(ALLOWED_SUFFIXES)}",
        )

    data_dir = Path(config.DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / name

    content = await _read_upload(file)
    dest.write_bytes(content)

    try:
        # 评审 P2-3：显式判空走「首次建库」分支（auto_rebuild=False 避免
        # get_vector_store 内部先隐式重建一遍、再进 else 增量分支的双重劳动）
        if get_vector_store(auto_rebuild=False).count() == 0:
            # 首次建库：全量重建（与 app.py 逻辑一致）
            chunks = load_and_chunk_documents()
            get_vector_store(chunks)
            semantic_cache.invalidate()  # P2：重建后旧缓存作废
            added = len(chunks)
            print(f"[api] first index: rebuilt {added} chunks from data/")
        else:
            added = add_documents(chunk_file(dest))
        return IngestResponse(file=name, saved_to=str(dest), added_chunks=added)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"索引失败：{e}")


@app.delete("/documents/{name}", response_model=DeleteResponse, tags=["documents"])
def delete_document(name: str):
    """删除文档：移除 data/ 下的文件，并只从索引中移除该 source 的块（其余文档不受影响）。

    健壮性：文件删除失败（Windows 上被占用 / 权限不足）时不 500——
    索引照常移除，响应带 warning 提示，保证『文档内容已不可检索』是确定性的。
    """
    safe = _safe_filename(name)
    if config.APP_PROFILE == "paper":
        import paper_index
        import paper_ingest

        for s in paper_ingest.list_states():
            if s.get("filename") == safe:
                pid = s.get("paper_id", "")
                paper_index.delete_paper(pid)
                sp = Path(config.PAPERS_DIR) / "_state" / f"{pid}.json"
                sp.unlink(missing_ok=True)
                paper_ingest.write_manifest()
                return DeleteResponse(deleted=True, filename=safe, warning=None)
        raise HTTPException(status_code=404, detail=f"论文不存在: {safe}")

    data_dir = Path(config.DATA_DIR)
    path = data_dir / safe

    warning = None
    if path.exists():
        try:
            path.unlink()
        except OSError as e:
            warning = f"文件删除失败（可能被占用或权限不足）：{e}；已从索引移除该文档。"

    # 无论文件在不在，都尝试从索引移除（防止文件已被手动删除但索引残留）
    try:
        delete_by_source(str(path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移除索引失败：{e}")
    return DeleteResponse(deleted=True, filename=safe, warning=warning)


# ==================== 论文库（v2.1） ====================
def require_paper_profile():
    """论文端点只在 paper profile 下可用：course profile 调用会写进课程集合，属跨库污染。"""
    if config.APP_PROFILE != "paper":
        raise HTTPException(
            status_code=400,
            detail="论文接口仅在 APP_PROFILE=paper 时可用（当前为 course）",
        )


_PAPER_QUEUE = queue.Queue()
_PAPER_WORKERS = []
_PAPER_WORKERS_LOCK = threading.Lock()


def _paper_worker():
    import paper_index
    import paper_ingest

    while True:
        task = _PAPER_QUEUE.get()
        if task is None:
            return
        pdf, force = task
        try:
            status = paper_ingest.ingest_one(Path(pdf), force=force)
            if status in ("normalized", "skip"):
                pid = paper_ingest.paper_id(paper_ingest.sha256_file(Path(pdf)))
                state = paper_ingest.load_state(pid) or {}
                if state.get("status") != "indexed":
                    paper_index.upsert_paper(pid)
            print(f"[papers] {Path(pdf).name}: {status}")
        except Exception as e:
            print(f"[papers] worker failed: {e!r}")
        finally:
            _PAPER_QUEUE.task_done()


def _ensure_paper_workers():
    with _PAPER_WORKERS_LOCK:
        while len(_PAPER_WORKERS) < config.PARSE_CONCURRENCY:
            t = threading.Thread(target=_paper_worker, daemon=True)
            t.start()
            _PAPER_WORKERS.append(t)


def _paper_state_or_404(paper_id: str):
    import paper_ingest

    if not paper_id.isalnum():
        raise HTTPException(status_code=404, detail=f"非法 paper_id: {paper_id}")
    state = paper_ingest.load_state(paper_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"论文不存在: {paper_id}")
    return state


def _paper_pdf(state):
    path = Path(state.get("source_path", ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="论文 PDF 不存在")
    return path


def _paper_info(state):
    return PaperInfo(
        paper_id=state.get("paper_id", ""),
        filename=state.get("filename", ""),
        title=state.get("title", ""),
        status=state.get("status", ""),
        pages=int(state.get("pages", 0) or 0),
        blocks=int(state.get("blocks", 0) or 0),
        chunks=int(state.get("chunks", 0) or 0),
        parse_sec=float(state.get("parse_sec", 0) or 0),
        pages_per_sec=float(state.get("pages_per_sec", 0) or 0),
        error=state.get("error", "") or "",
        updated_at=state.get("updated_at", "") or "",
    )


async def _save_paper_upload(file: UploadFile):
    import paper_ingest

    name = _safe_filename(file.filename or "")
    if Path(name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="论文上传仅支持 PDF")
    raw = Path(config.PAPERS_RAW_DIR)
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / name
    dest.write_bytes(await _read_upload(file))
    sha = paper_ingest.sha256_file(dest)
    pid = paper_ingest.paper_id(sha)
    state = paper_ingest.load_state(pid) or paper_ingest._new_state(dest, sha)
    if state.get("status") in ("normalized", "indexed"):
        return PaperUploadItem(paper_id=pid, filename=name, status=state["status"])
    state.update(
        status="queued", stage="queued", error="",
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    paper_ingest.save_state(state)
    _PAPER_QUEUE.put((dest, False))
    return PaperUploadItem(paper_id=pid, filename=name, status="queued")


@app.post("/papers", response_model=PaperUploadResponse, tags=["papers"], dependencies=[Depends(require_paper_profile)])
async def upload_papers(files: list[UploadFile] = File(...)):
    """批量上传 PDF：落盘 + 入队，立即返回 queued；单 worker 逐篇解析，避免并发 MinerU。"""
    _ensure_paper_workers()
    items = [await _save_paper_upload(f) for f in files]
    import paper_ingest

    paper_ingest.write_manifest()
    return PaperUploadResponse(files=items)


@app.get("/papers", response_model=PaperList, tags=["papers"], dependencies=[Depends(require_paper_profile)])
def list_papers():
    import paper_ingest

    return PaperList(papers=[_paper_info(s) for s in paper_ingest.list_states()])


@app.get("/papers/{paper_id}", response_model=PaperInfo, tags=["papers"], dependencies=[Depends(require_paper_profile)])
def get_paper(paper_id: str):
    return _paper_info(_paper_state_or_404(paper_id))


@app.post("/papers/{paper_id}/reparse", response_model=PaperActionResponse, tags=["papers"], dependencies=[Depends(require_paper_profile)])
def reparse_paper(paper_id: str):
    state = _paper_state_or_404(paper_id)
    pdf = _paper_pdf(state)
    try:
        get_vector_store(auto_rebuild=False).delete(where={"sha256": state["sha256"]})
    except Exception:
        pass
    state.update(
        status="queued", stage="queued", error="",
        updated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    import paper_ingest

    paper_ingest.save_state(state)
    paper_ingest.write_manifest()
    _ensure_paper_workers()
    _PAPER_QUEUE.put((pdf, True))
    return PaperActionResponse(paper_id=paper_id, status="queued")


@app.delete("/papers/{paper_id}", response_model=PaperDeleteResponse, tags=["papers"], dependencies=[Depends(require_paper_profile)])
def delete_paper(paper_id: str, delete_raw: bool = False):
    import shutil

    import paper_index
    import paper_ingest

    state = _paper_state_or_404(paper_id)
    try:
        paper_index.delete_paper(paper_id)
    except Exception:
        pass
    norm = Path(config.PAPERS_DIR) / "normalized" / f"{paper_id}.json"
    if norm.exists():
        norm.unlink()
    parsed = Path(config.PAPERS_DIR) / "parsed" / paper_id
    if parsed.is_dir():
        shutil.rmtree(parsed, ignore_errors=True)
    state_path = Path(config.PAPERS_DIR) / "_state" / f"{paper_id}.json"
    if state_path.exists():
        state_path.unlink()
    raw_deleted = False
    if delete_raw:
        pdf = Path(state.get("source_path", ""))
        if pdf.exists():
            pdf.unlink()
            raw_deleted = True
    paper_ingest.write_manifest()
    return PaperDeleteResponse(deleted=True, paper_id=paper_id, raw_deleted=raw_deleted)


@app.get("/papers/{paper_id}/blocks", response_model=PaperBlocksResponse, tags=["papers"], dependencies=[Depends(require_paper_profile)])
def paper_blocks(paper_id: str, page: int | None = None):
    _paper_state_or_404(paper_id)
    path = Path(config.PAPERS_DIR) / "normalized" / f"{paper_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="该论文尚未完成归一化")
    data = json.loads(path.read_text(encoding="utf-8"))
    pages = data.get("pages") or []
    if page is not None:
        pages = [p for p in pages if p.get("page") == page]
        if not pages:
            raise HTTPException(status_code=404, detail=f"第 {page} 页不存在")
    return PaperBlocksResponse(
        paper_id=paper_id, n_pages=int(data.get("n_pages", 0)), pages=pages
    )


@app.get("/papers/{paper_id}/pages/{page:int}.png", tags=["papers"], dependencies=[Depends(require_paper_profile)])
def paper_page_png(paper_id: str, page: int, blocks: str = ""):
    """页图 + 可选 bbox 高亮；**不提供整本 PDF 下载**。"""
    state = _paper_state_or_404(paper_id)
    pdf = _paper_pdf(state)
    path = Path(config.PAPERS_DIR) / "normalized" / f"{paper_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="该论文尚未完成归一化")
    data = json.loads(path.read_text(encoding="utf-8"))
    n_pages = int(data.get("n_pages", 0))
    if page < 1 or page > n_pages:
        raise HTTPException(status_code=404, detail=f"页码越界: {page}/{n_pages}")
    import page_render

    ids = ",".join(blocks.split(",")[:100])
    boxes = page_render.bboxes_for_blocks(paper_id, ids, page=page)
    try:
        png = page_render.render_page_png(pdf, page, blocks=[{"bbox": b} for b in boxes])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"页图渲染失败: {e}")
    return Response(content=png, media_type="image/png", headers={"Cache-Control": "no-store"})


# ==================== 索引管理 ====================
@app.post("/index/rebuild", response_model=RebuildResponse, tags=["index"])
def rebuild_index():
    """全量重建索引：重新加载 data/ 全部文档并重建向量库（慢操作，适合初次 / 换语料后）。"""
    chunks = load_and_chunk_documents()
    if not chunks:
        raise HTTPException(status_code=400, detail="data/ 下没有可索引的文档（PDF / TXT / MD / DOCX）")
    get_vector_store(chunks)
    semantic_cache.invalidate()  # P2：重建后旧缓存作废
    return RebuildResponse(chunks=len(chunks), files=len({d.metadata["source"] for d in chunks}))


@app.get("/index/stats", response_model=IndexStats, tags=["index"])
def index_stats():
    """向量库统计：总块数、已索引源文档数、持久化目录。"""
    col = _load_index_or_raise()
    metas = col.get(include=["metadatas"])["metadatas"]
    source_files = len({m.get("source") for m in metas if m})
    return IndexStats(
        collection=col.name,
        count=col.count(),
        source_files=source_files,
        persist_directory=str(Path(config.PERSIST_DIRECTORY).resolve()),
    )


# ==================== 多模态：原图字节（B2 视觉作答） ====================
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


@app.get("/images/{rel_path:path}", tags=["multimodal"])
def get_image(rel_path: str):
    """按向量库里存的 `image_path` 返回原始图像字节。

    为什么需要这个端点（B2）：消费者（FeynmanTutor）从 /retrieve 拿到的
    `Citation.image_path` 是**索引构建机上的相对路径**（形如
    `data/images/微信小程序开发实战/p267_i00.png`）。消费者与服务不在同一文件系统时
    打不开它；而把宿主机绝对路径写进返回体又是错的（换机器即失效，交接文档已警告
    「不要写死宿主机绝对路径」）。所以由 CourseRAG 自己把像素读出来经 HTTP 返回，
    调用方不需要知道磁盘布局。

    路径规则：接受两种等价写法——① 相对 `data/images/` 的路径；② 索引里原样存的
    整段路径（自动剥掉开头的 `data/images/`）。

    防穿越：解析后必须仍落在 `data/images/` 之内，且后缀在白名单内。
    """
    images_root = (Path(config.DATA_DIR) / "images").resolve()
    raw = rel_path.replace("\\", "/").lstrip("/")
    prefix = "data/images/"
    if raw.startswith(prefix):
        raw = raw[len(prefix):]
    if not raw:
        raise HTTPException(status_code=400, detail="缺少图片路径")
    target = (images_root / raw).resolve()
    if images_root != target and images_root not in target.parents:
        raise HTTPException(status_code=400, detail=f"非法图片路径: {rel_path!r}")
    if target.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"不是允许的图像类型: {target.suffix!r}")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"图片不存在: {rel_path!r}")
    return FileResponse(target)

# ---------- React 前端静态托管（生产） ----------
# 开发模式走 Vite proxy（/api → 127.0.0.1:8000）；这里只在 `frontend/dist` 存在时挂载，
# 缺失时静默跳过，不影响 FastAPI / Streamlit / MCP 三个出口。
_FRONTEND_DIST = (Path(__file__).resolve().parent / "frontend" / "dist").resolve()
if _FRONTEND_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles

    # API 顶层前缀：这些路径下的未知请求必须 404，不能落到 SPA 兜底
    _API_TOP = {
        "api-info", "health", "query", "retrieve", "documents",
        "papers", "index", "images", "docs", "openapi.json", "redoc",
    }

    app.mount(
        "/assets",
        StaticFiles(directory=_FRONTEND_DIST / "assets"),
        name="frontend-assets",
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    def _frontend_spa(full_path: str):
        """SPA 兜底：静态文件命中就返回文件，否则交回 index.html 由前端路由接管。

        注册在所有 API 路由之后；API 前缀下的未知路径仍返回 404，
        不要被前端兜底吞成 200（否则 /papers/../etc 这类请求看起来像成功）。
        """
        if full_path.split("/", 1)[0] in _API_TOP:
            raise HTTPException(status_code=404, detail="Not Found")
        target = (_FRONTEND_DIST / full_path).resolve()
        if full_path and _FRONTEND_DIST in target.parents and target.is_file():
            return FileResponse(target)
        return FileResponse(_FRONTEND_DIST / "index.html")
