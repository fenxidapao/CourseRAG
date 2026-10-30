# -*- coding: utf-8 -*-
"""CourseRAG MCP stdio server（官方 mcp SDK 封装版，2026-09-18 从手写零依赖版迁移）

v0.2.0：由「手写 JSON-RPC 最小子集」迁移到官方 SDK 的 MCPServer（即 mcp 2.x 里
FastMCP 的更名，API 形态一致：函数类型签名自动生成 inputSchema，ToolError 对应
isError 工具结果）。工具契约（名称 / 入参 / 返回格式 / 描述文案）与手写版逐字
一致，对已接入的 MCP 客户端零破坏。

工具：两个，语义正交（一个回答「是什么」，一个回答「你有什么」）——
    course_retrieve  与 FastAPI /retrieve 同一检索实现（retrieve_and_rerank），
                     mode 语义一致（fast=纯向量 / accurate=混合检索+reranker 精排），默认 accurate。
    list_documents   列出**已建索引**的文档与块数，供模型在检索前判断知识边界；
                     刻意不列 data/ 下未索引的文件——未索引=检索不到，列出来会误导模型。

从手写版原样移植的工程守卫（踩坑留痕，迁移不丢）：
  1. stdout fd 级防污染：rag_core/chromadb/tqdm 可能绕过 sys.stdout 直写 fd1
     （实测 reranker「Loading weights」进度条污染过协议流），rag 工作全程必须包在
     _StdoutToFd2 里（SDK 2.x 的 stdio 层自带 fd 保护，但只护它自己写的帧，
     第三方库直写 fd1 仍会插进协议流，此守卫依旧必要）；
  2. 服务端参数自守：top_k 钳制到 [1,20] 而非报错——schema 只是给客户端的提示，
     模型偶发越界值时静默钳制比强制重试更友好（负值会让 docs[:top_k] 切片语义错乱）；
  3. 工具失败降级为 isError 结果（ToolError），不让进程退出。

与手写版的已知行为差异（有意为之）：
  - mode 传非法值：手写版服务端回 -32602；现在由 SDK 按 Literal["fast","accurate"]
    签名做入参校验，同样拒绝，但走 isError 工具结果（MCP 惯例：模型误调用是
    「工具执行失败」，不是协议层违规）；
  - 线上帧编码：手写版强制 ensure_ascii 纯 ASCII（防 Windows 跨代码页乱码）；
    SDK 按 MCP 规范走 UTF-8，本文件在启动时把 stdin/stdout reconfigure 为 UTF-8
    兜底（中文 Windows 默认代码页是 GBK，不 reconfigure 会解码失败/乱码）。

硬约束（与手写版一致）：
  - stdout 是协议通道，本文件任何日志/调试输出只能走 stderr。
"""
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import Field

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

SERVER_VERSION = "0.2.0"

# ---- 工具描述（与手写版逐字一致：描述是模型选工具的唯一依据）----
_COURSE_RETRIEVE_DESC = (
    "检索本地课程知识库（计算机/数学专业课笔记、教材、合规公开教程），"
    "返回带来源与相关性分数的文本块，作为课程知识点问答的事实依据。"
    "mode=accurate（默认）向量+BM25 混合检索并经 bge-reranker 精排，检索质量最高，"
    "专治「ACID / 2NF / 死锁」等纯向量词面匹配弱的精确术语难 query；"
    "mode=fast 纯向量召回，为降级档——GPU 不可用、显存不足或高并发时仍可用于保持服务可用。"
    "延迟因硬件而异——有 CUDA 时精排亚秒级，无 CUDA 回落 CPU 会显著变慢；"
    "本描述不写死延迟数字以免随硬件/版本过期。"
)

_PAPER_RETRIEVE_DESC = (
    "检索本地论文库（已由 MinerU 解析并索引的论文），返回带《论文名》+ 物理页码 + 章节 + "
    "相关性分数的文本块，作为论文问答的事实依据。与 course_retrieve 共用同一检索实现；"
    "论文库 profile 下优先使用本工具。mode=accurate（默认）混合检索+精排；mode=fast 为降级档。"
)

_LIST_DOCUMENTS_DESC = (
    "列出本地课程知识库中**已建索引**的文档（教材 / 课件 / 笔记）及其文本块数量，"
    "用于在检索前先判断知识边界：这个问题涉及的内容库里到底有没有、该用哪本书的关键词去检索。"
    "注意只反映已索引内容——未出现在清单里的主题，course_retrieve 一定检索不到，"
    "此时应直接告知用户知识库未覆盖，而不是硬检索后编造。"
    "可选用 name_contains 按文件名子串过滤（不区分大小写，如 'Linux' / '数据库' / '离散'）。"
)

_RAG = {"ready": False}


class _StdoutToFd2:
    """OS 级 fd1 保护：rag 工作期间把真实 fd1 指向 stderr。
    contextlib.redirect_stdout 只能拦 Python 层 print；tqdm/torch 等可能绕过
    sys.stdout 直写 fd1（实测 reranker「Loading weights」进度条污染过协议流），
    必须在文件描述符层重定向才彻底。退出时恢复。
    ⚠️ 竞态说明：fd1 被重定向的窗口内，SDK 若并发写协议帧会写到 stderr——但 SDK
    对同一请求的响应帧必然在工具返回之后才写，且 stdio server 逐请求处理，
    实际不构成竞争；守卫窗口已压缩到 rag 工作本身。"""

    def __enter__(self):
        sys.stdout.flush()
        self._saved = os.dup(1)
        os.dup2(2, 1)
        return self

    def __exit__(self, *exc):
        sys.stderr.flush()
        os.dup2(self._saved, 1)
        os.close(self._saved)


def _log(msg: str) -> None:
    print(f"[courserag-mcp] {msg}", file=sys.stderr, flush=True)


def _ensure_rag() -> None:
    """懒加载 rag_core 并预热向量库（与 main.py lifespan 同一 get_vector_store 语义，
    自动复用 chroma_db 现有索引）。stdout 全程重定向到 stderr。"""
    if _RAG["ready"]:
        return
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr), _StdoutToFd2():
        import rag_core
        rag_core.get_vector_store()
    _RAG["ready"] = True
    _log(f"rag_core ready in {time.time() - t0:.1f}s")


def _retrieve_real(query: str, top_k: int, mode: str) -> str:
    _ensure_rag()
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr), _StdoutToFd2():
        from rag_core import retrieve_and_rerank
        docs = retrieve_and_rerank(query, force_no_rerank=(mode != "accurate"))
    docs = docs[:top_k]
    _log(f"retrieve done in {time.time() - t0:.1f}s (mode={mode}, hits={len(docs)})")
    if not docs:
        return "未检索到相关内容（知识库无匹配文本块）。"
    lines = [f"检索到 {len(docs)} 个相关文本块（mode={mode}）：", ""]
    for rank, doc in enumerate(docs, 1):
        source = Path(doc.metadata.get("source", "unknown")).name
        score = doc.metadata.get("rerank_score") or doc.metadata.get("hybrid_score")
        start = doc.metadata.get("start_index")
        if doc.metadata.get("paper_id"):
            title = doc.metadata.get("paper_title") or source
            page = doc.metadata.get("page")
            section = doc.metadata.get("section")
            meta = f"来源: 《{title}》第 {page} 页"
            if section:
                meta += f" | 章节: {section}"
        else:
            meta = f"来源: {source}"
        if score is not None:
            meta += f" | 相关性分数: {float(score):.3f}"
        if start is not None:
            meta += f" | 原文偏移: {start}"
        lines.append(f"[{rank}] {meta}")
        lines.append(doc.page_content.strip())
        lines.append("")
    return "\n".join(lines)


def _base_name(source: str) -> str:
    """取 source 的文件名部分并统一分隔符。

    索引在 Windows 宿主构建时 metadata.source 形如 `data\\教材\\xx.pdf`（反斜杠），
    容器内则可能是 `/app/data/xx.pdf`。Linux 上 Path("a\\b.pdf").name 会原样返回整个串，
    故必须手工统一分隔符再切——与 main.py._base_source 同一处理。
    """
    return (source or "unknown").replace("\\", "/").rsplit("/", 1)[-1]


def _list_documents_real(name_contains=None) -> str:
    """列出已建索引的文档与块数（不扫 data/ 目录）。

    与 FastAPI /documents 的差异是**有意为之**：/documents 面向运维，列 data/ 下所有文件
    （含 chunks=0 的未索引文件）；本工具面向模型，只列**已索引**内容——模型的用途是判断
    「这个问题我能不能检索到」，把未索引的文件列出来只会误导它去检索一个不存在的源。
    """
    _ensure_rag()
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr), _StdoutToFd2():
        from rag_core import get_vector_store
        col = get_vector_store(auto_rebuild=False)
        metas = [m for m in col.get(include=["metadatas"])["metadatas"] if m]

    from collections import Counter
    chunk_counter, image_counter = Counter(), Counter()
    for m in metas:
        name = _base_name(m.get("source"))
        chunk_counter[name] += 1
        if m.get("type") == "image":
            image_counter[name] += 1

    total, images = sum(chunk_counter.values()), sum(image_counter.values())
    _log(f"list_documents done in {time.time() - t0:.1f}s "
         f"(docs={len(chunk_counter)}, chunks={total})")

    if not chunk_counter:
        return "知识库当前为空（尚未建索引）。请先在服务端构建索引或上传资料后再试。"

    key = (name_contains or "").strip().lower()
    items = [(n, c) for n, c in chunk_counter.items() if not key or key in n.lower()]
    if not items:
        return (
            f"没有文件名匹配 {name_contains!r} 的已索引文档。\n"
            f"当前已索引的文档：{'、'.join(sorted(chunk_counter))}"
        )
    items.sort(key=lambda kv: (-kv[1], kv[0]))

    head = f"已在索引中的课程文档共 {len(chunk_counter)} 个 / {total} 个文本块"
    if images:
        head += f"（其中教材插图描述 {images} 块）"
    if key:
        head += f"；按文件名子串 {name_contains!r} 过滤后剩 {len(items)} 个"
    lines = [head + "：", ""]
    for i, (name, count) in enumerate(items, 1):
        line = f"{i}. {name} — {count} 块"
        if image_counter.get(name):
            line += f"（含插图 {image_counter[name]}）"
        lines.append(line)
    lines += ["", "提示：以上即知识边界；清单之外的主题检索不到，应直接说明未覆盖而非硬答。"]
    return "\n".join(lines)


# ==================== 工具实现（独立于装饰器，供进程内单测直调） ====================
def _retrieve_impl(query: str, top_k: int, mode: str) -> str:
    """course_retrieve 的实现体。参数校验 + 钳制独立于 SDK 签名——
    SDK 校验拦得住协议层入参，直调（进程内 / 未来挂 HTTP）仍靠这里自守。"""
    if not isinstance(query, str) or not query.strip():
        raise ToolError("invalid params: 'query' (string) is required")
    try:
        top_k = int(top_k)
    except (TypeError, ValueError):
        raise ToolError("invalid params: 'top_k' must be an integer")
    # 钳制到 [1,20]：负值会让 docs[:top_k] 切片语义错乱，超上限绕过 schema max；
    # 静默钳制而非报错——模型偶发越界值时无需重试一轮
    top_k = max(1, min(20, top_k))
    if mode not in ("fast", "accurate"):
        raise ToolError("invalid params: 'mode' must be 'fast' or 'accurate'")
    try:
        return _retrieve_real(query.strip(), top_k, mode)
    except ToolError:
        raise
    except Exception as exc:  # 检索失败降级为 isError 结果，不让进程退出
        _log(f"retrieve failed: {exc!r}")
        raise ToolError(
            f"检索失败：{exc}\n"
            "请确认 ollama serve 已启动（嵌入模型 nomic-embed-text 可用）"
            "且 chroma_db 索引存在。"
        )


def _list_documents_impl(name_contains) -> str:
    """list_documents 的实现体。"""
    if name_contains is not None and not isinstance(name_contains, str):
        raise ToolError("invalid params: 'name_contains' must be a string")
    try:
        return _list_documents_real(name_contains)
    except ToolError:
        raise
    except Exception as exc:  # 与检索工具一致：失败降级为 isError，不带崩进程
        _log(f"list_documents failed: {exc!r}")
        raise ToolError(
            f"列出文档失败：{exc}\n"
            "请确认 chroma_db 索引存在且 Ollama 已启动（嵌入模型 nomic-embed-text 可用）。"
        )


# ==================== MCPServer 装配 ====================
server = MCPServer(name="courserag-mcp", version=SERVER_VERSION)


@server.tool(description=_COURSE_RETRIEVE_DESC, structured_output=False)
def course_retrieve(
    query: Annotated[str, Field(description="课程/课本知识点问题或检索词")],
    top_k: Annotated[int, Field(description="返回的文本块数量")] = 5,
    mode: Literal["fast", "accurate"] = "accurate",
) -> str:
    # top_k 刻意不在签名上声明 ge/le：Field 边界会被 SDK 直接拒绝（强制模型重试），
    # 而手写版契约是服务端静默钳制（越界值直接收编，零重试）——契约保持不变。
    return _retrieve_impl(query, top_k, mode)


@server.tool(description=_PAPER_RETRIEVE_DESC, structured_output=False)
def paper_retrieve(
    query: Annotated[str, Field(description="论文内容问题或检索词")],
    top_k: Annotated[int, Field(description="返回的文本块数量")] = 5,
    mode: Literal["fast", "accurate"] = "accurate",
) -> str:
    return _retrieve_impl(query, top_k, mode)


@server.tool(description=_LIST_DOCUMENTS_DESC, structured_output=False)
def list_documents(
    name_contains: Annotated[
        Optional[str],
        Field(description="可选：按源文件名子串过滤（不区分大小写）；不传=列出全部已索引文档"),
    ] = None,
) -> str:
    return _list_documents_impl(name_contains)


def main() -> None:
    # Windows 代码页防线（手写版「纯 ASCII 帧」的现代等价物）：SDK 的 stdio 层走
    # 文本流，编码跟随系统代码页——中文 Windows 默认 GBK，与 MCP 规范的 UTF-8 帧
    # 不符会乱码/解码失败。reconfigure 强制 UTF-8（本就 UTF-8 时是 no-op）。
    for _s in (sys.stdin, sys.stdout):
        try:
            _s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    _log(f"start (stdio; mcp SDK MCPServer v{SERVER_VERSION})")
    server.run("stdio")


if __name__ == "__main__":
    main()
