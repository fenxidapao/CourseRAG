"""
test_mcp.py — CourseRAG MCP stdio 服务端测试（2026-09-18 重写，配套官方 mcp SDK 2.x 版 mcp_server.py）

2026-09-18 迁移说明：mcp_server.py 从手写 JSON-RPC 换成官方 SDK 的 MCPServer 后，
本文件同步重写——协议层不再手搓帧，改用官方 client SDK（stdio_client + ClientSession）
子进程直测；参数钳制 / 校验 / 失败降级保留进程内单测（直调 _retrieve_impl /
_list_documents_impl，不触发 rag_core 懒加载，零 chromadb/ollama 依赖）。

三组用例：
  - 协议组（需 mcp 包，无需 Ollama）：官方 client 起子进程握手 / tools/list schema /
    未知工具与非法入参 → isError / 中文 UTF-8 帧往返（SDK 走 UTF-8，替代旧版
    「纯 ASCII 帧」断言——中文往返成功即证明 Windows 代码页防线生效）
  - 进程内组（零依赖）：top_k 钳制 / 参数校验 ToolError / 检索与清单失败降级 /
    过滤参数透传
  - 冒烟组（需 Ollama + 索引，无 Ollama 自动 skip）：两个工具的 tools/call 端到端

与官方 MCP Inspector 互操作证据见 mcp_interop_evidence.md（本文件是可回归的
自动化保护层，互操作实测是一次性证据，两者互补）。
"""
import os
import subprocess
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 本文件锁定课程版语义（默认 profile 已切 paper）；必须在 import config 之前设置
os.environ["APP_PROFILE"] = "course"

import config
import mcp_server
from mcp_server import (
    ToolError,
    _list_documents_impl,
    _retrieve_impl,
)
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

MCP_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")


# anyio pytest 插件驱动异步协议用例（mcp SDK 基于 anyio/asyncio）
@pytest.fixture
def anyio_backend():
    return "asyncio"


def _server_params() -> StdioServerParameters:
    # MCP SDK 默认只透传少量环境变量；显式带上 APP_PROFILE，避免子进程落回默认 paper
    return StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER],
        env={**os.environ, "APP_PROFILE": "course"},
    )


def _server_params_paper() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER],
        env={**os.environ, "APP_PROFILE": "paper"},
    )


def _paper_index_ready():
    from pathlib import Path

    if not _ollama_ready():
        return False
    norm = Path(config.PAPERS_DIR) / "normalized"
    return norm.is_dir() and any(norm.glob("*.json")) and Path("chroma_db_papers").exists()


# ==================== 协议组（官方 client SDK 子进程直测，无需 Ollama） ====================
@pytest.mark.anyio
async def test_handshake_and_tools_list_schema():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.server_info.name == "courserag-mcp"
            assert init.capabilities.tools is not None  # 声明了 tools 能力

            result = await session.list_tools()
            tools = {t.name: t for t in result.tools}
            # 工具集合本身作为断言：paper_retrieve 为论文库主入口，course_retrieve 保留为
            # 课程版兼容别名（不删），list_documents 通用于两种 profile
            assert set(tools) == {
                "paper_retrieve",
                "course_retrieve",
                "list_documents",
            }, tools

            schema = tools["course_retrieve"].input_schema
            assert "query" in schema["required"]
            assert "query" in schema["properties"]
            assert schema["properties"]["mode"]["enum"] == ["fast", "accurate"]
            assert schema["properties"]["top_k"]["default"] == 5

            ps = tools["paper_retrieve"].input_schema
            assert "query" in ps["required"]
            assert ps["properties"]["mode"]["enum"] == ["fast", "accurate"]
            assert ps["properties"]["top_k"]["default"] == 5

            ld = tools["list_documents"].input_schema
            assert ld.get("required", []) == []  # 无必填参数：模型可直接调用探查知识边界
            assert "name_contains" in ld["properties"]
            for t in result.tools:  # 描述是模型选工具的唯一依据，空描述等于工具不存在
                assert (t.description or "").strip(), t.name


@pytest.mark.anyio
async def test_call_unknown_tool_is_error():
    """未知工具 → isError 工具结果（MCP 惯例），而非协议层异常断连。"""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("no_such_tool", {})
            assert res.is_error


@pytest.mark.anyio
async def test_call_invalid_params_is_error():
    """schema 层非法入参（mode 越界 / query 缺失）→ SDK 校验拒绝为 isError。"""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(
                "course_retrieve", {"query": "ACID", "mode": "bogus"}
            )
            assert res.is_error
            res = await session.call_tool("course_retrieve", {"top_k": 3})
            assert res.is_error


@pytest.mark.anyio
async def test_empty_query_is_error_with_chinese_roundtrip():
    """query 传空白串：过得了 schema（是 string）但过不了服务端自守 → isError，
    且错误文案为中文——文案经 stdio 帧往返后仍可读，即 UTF-8 编码防线生效
    （中文 Windows 默认代码页 GBK，防线失效会乱码/解码异常）。"""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool("course_retrieve", {"query": "   "})
            assert res.is_error
            text = res.content[0].text
            assert "query" in text  # 服务端自守文案（非 SDK 泛化错误）活着回来了


def test_eof_exits_cleanly():
    """stdin 关闭 → 服务端正常退出（SDK run_stdio 读到 EOF 收尾，退出码 0）。"""
    proc = subprocess.Popen(
        [sys.executable, MCP_SERVER],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    proc.stdin.close()
    proc.wait(timeout=15)
    assert proc.returncode == 0


# ==================== 进程内组（钳制 / 校验 / 降级，零依赖） ====================
def _with_stub_retrieve(monkeypatch, calls):
    def fake_retrieve(query, top_k, mode):
        calls.append((query, top_k, mode))
        return "stub 检索结果"

    monkeypatch.setattr(mcp_server, "_retrieve_real", fake_retrieve)


def test_top_k_clamped_to_schema_range(monkeypatch):
    calls = []
    _with_stub_retrieve(monkeypatch, calls)
    _retrieve_impl("ACID", -5, "accurate")
    _retrieve_impl("ACID", 999, "accurate")
    assert [c[1] for c in calls] == [1, 20]  # 钳制而非报错：模型越界值直接收编


def test_decorated_tool_default_top_k(monkeypatch):
    """装饰后的工具函数可直接调用（同步），缺省 top_k=5 / mode=accurate。"""
    calls = []
    _with_stub_retrieve(monkeypatch, calls)
    out = mcp_server.course_retrieve(query="ACID")
    assert out == "stub 检索结果"
    assert calls == [("ACID", 5, "accurate")]


def test_invalid_params_raise_tool_error(monkeypatch):
    calls = []
    _with_stub_retrieve(monkeypatch, calls)
    with pytest.raises(ToolError):
        _retrieve_impl("  ", 5, "accurate")  # 空白 query
    with pytest.raises(ToolError):
        _retrieve_impl("ACID", "abc", "accurate")  # 非整数 top_k
    with pytest.raises(ToolError):
        _retrieve_impl("ACID", 5, "bogus")  # 非法 mode（协议层由 Literal 拦，直调自守）
    assert calls == []  # 非法参数不应触达检索


def test_retrieve_failure_degrades_to_tool_error(monkeypatch):
    def boom(query, top_k, mode):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(mcp_server, "_retrieve_real", boom)
    with pytest.raises(ToolError) as e:
        _retrieve_impl("ACID", 5, "accurate")
    assert "ollama" in str(e.value).lower() or "失败" in str(e.value)


def test_list_documents_passes_filter_through(monkeypatch):
    calls = []

    def fake(name_contains=None):
        calls.append(name_contains)
        return "stub 文档清单"

    monkeypatch.setattr(mcp_server, "_list_documents_real", fake)
    _list_documents_impl(None)                     # 不传过滤
    _list_documents_impl("Linux")                  # 子串过滤
    _list_documents_impl(None)                     # 显式 None 等同不传
    assert calls == [None, "Linux", None]


def test_list_documents_rejects_non_string_filter(monkeypatch):
    calls = []
    monkeypatch.setattr(
        mcp_server, "_list_documents_real",
        lambda k=None: calls.append(k) or "stub",
    )
    with pytest.raises(ToolError):
        _list_documents_impl(123)
    assert calls == []  # 非法参数不应触达索引


def test_list_documents_failure_degrades_to_tool_error(monkeypatch):
    def boom(name_contains=None):
        raise RuntimeError("chroma_db missing")

    monkeypatch.setattr(mcp_server, "_list_documents_real", boom)
    with pytest.raises(ToolError) as e:
        _list_documents_impl(None)
    assert "失败" in str(e.value)


# ==================== 冒烟组（需 Ollama + 索引） ====================
def _ollama_ready():
    try:
        base = config.OLLAMA_BASE_URL.rstrip("/")
        with urllib.request.urlopen(base + "/api/tags", timeout=4) as resp:
            return resp.status == 200
    except Exception:
        return False


@pytest.mark.anyio
@pytest.mark.skipif(not _ollama_ready(), reason="Ollama 未启动，跳过 MCP 真实检索冒烟")
async def test_tools_call_real_retrieve_end_to_end():
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(
                "course_retrieve",
                {"query": "什么是ACID", "top_k": 2, "mode": "fast"},
            )
            assert res.is_error is False
            text = res.content[0].text
            # 中文与术语经 stdio 往返仍完整（UTF-8 防线 + 检索真实命中）
            assert "来源" in text and "ACID" in text


@pytest.mark.anyio
@pytest.mark.skipif(not _ollama_ready(), reason="Ollama 未启动，跳过 MCP 文档清单冒烟")
async def test_tools_call_real_list_documents_end_to_end():
    """真实读取 chroma_db：清单须含块数汇总，且与过滤路径互不矛盾。"""
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            full = await session.call_tool("list_documents", {})
            assert full.is_error is False
            text = full.content[0].text
            assert "文本块" in text, text

            # 过滤一个几乎必然不存在的子串 → 必须给可用清单，而不是空串或报错
            miss = await session.call_tool(
                "list_documents", {"name_contains": "zzz_no_such_doc_zzz"}
            )
            assert miss.is_error is False
            assert "没有文件名匹配" in miss.content[0].text


@pytest.mark.anyio
@pytest.mark.skipif(not _paper_index_ready(), reason="论文索引未就绪（需 data/papers/normalized + chroma_db_papers + Ollama）")
async def test_paper_retrieve_end_to_end():
    """paper profile 下 paper_retrieve 必须给出《论文名》第 N 页（M6 三出口一致性）。"""
    async with stdio_client(_server_params_paper()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(
                "paper_retrieve",
                {"query": "multi-head attention", "top_k": 2, "mode": "fast"},
            )
            assert res.is_error is False
            text = res.content[0].text
            assert "来源" in text and "第" in text and "页" in text, text
