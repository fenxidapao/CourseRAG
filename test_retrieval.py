"""
test_retrieval.py — 检索链路冒烟测试（T9，2026-08-22）

覆盖「加载切分 → 建库 → 向量召回 → 混合检索(BM25 全库) → 置信度」整条检索链路，
验证 T8 结构切分 / 乱码清洗 / v2 混合检索没有破坏基础可用性。

与 test_core.py（纯算法，零依赖）的区别：
  - test_core.py  不依赖 chromadb/ollama，CI 直接跑
  - 本文件需要 Ollama（嵌入 nomic-embed-text）与真实语料 data/，本机跑；
    无 Ollama 时自动 skip（CI 无 Ollama → 整组跳过，不阻塞 CI）

运行（本机，需 Ollama 已启动）：
  OLLAMA_BASE_URL=http://127.0.0.1:11434 python -m pytest test_retrieval.py -v
"""
import os
import sys
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 本文件锁定课程版语义（默认 profile 已切 paper）；必须在 import config 之前设置
os.environ["APP_PROFILE"] = "course"

import config
import rag_core


def _ollama_ready():
    """探测 Ollama 是否可达（只验证服务，不验证模型——建库时会报具体缺哪个）。"""
    try:
        base = config.OLLAMA_BASE_URL.rstrip("/")
        with urllib.request.urlopen(base + "/api/tags", timeout=4) as resp:
            return resp.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_ready(),
    reason="Ollama 未启动（OLLAMA_BASE_URL=%s），跳过检索冒烟测试" % config.OLLAMA_BASE_URL,
)


# ==================== 加载与切分 ====================
def test_load_and_chunk_documents():
    docs = rag_core.load_and_chunk_documents()
    assert len(docs) > 100, "真实语料应产出 >100 块（当前 697）"
    # 每块必须带 source 与 start_index（引用溯源的前提）
    for d in docs[:20]:
        assert "source" in d.metadata
        assert "start_index" in d.metadata
        assert d.page_content.strip()


def test_chunks_clean_of_garbled():
    """乱码清洗必须有效：全语料中『非白名单字符』占比应接近 0（容忍 ·×÷° 等）。"""
    docs = rag_core.load_and_chunk_documents()
    total = bad = 0
    for d in docs:
        for ch in d.page_content:
            if ch.isspace():
                continue
            total += 1
            if not rag_core._is_good_char(ch):
                bad += 1
    ratio = bad / total if total else 0.0
    assert ratio < 0.005, f"乱码残留占比 {ratio:.4f} 过高"


# ==================== 建库 ====================
def test_vector_store_reachable():
    col = rag_core.get_vector_store()
    assert col.count() >= 100


# ==================== 向量召回 ====================
def test_vector_retrieve_hits_relevant():
    docs = rag_core.retrieve_vector("数据库事务的 ACID 特性是什么", k=5)
    assert len(docs) == 5
    # 顶部结果应包含 ACID/事务 相关块（真实语料有）
    top = " ".join(d.page_content for d in docs[:3])
    assert ("ACID" in top) or ("事务" in top)


def test_vector_retrieve_sets_score():
    docs = rag_core.retrieve_vector("什么是死锁", k=3)
    assert docs[0].metadata.get("vector_score") is not None
    assert 0.0 <= docs[0].metadata["vector_score"] <= 1.0


# ==================== 混合检索（v2：BM25 全库独立召回） ====================
def test_hybrid_retrieve_bm25_channel():
    """BM25 通道应能捞回『词面精确匹配』的块（如 2NF / 死锁）。"""
    docs = rag_core.hybrid_retrieve("第二范式 2NF 消除部分函数依赖", k=5)
    assert len(docs) > 0
    top = " ".join(d.page_content for d in docs[:3])
    assert ("2NF" in top) or ("函数依赖" in top)


def test_hybrid_retrieve_has_score():
    docs = rag_core.hybrid_retrieve("什么是死锁？四个必要条件", k=5)
    assert docs[0].metadata.get("hybrid_score") is not None


# ==================== 置信度（T3） ====================
def test_confidence_on_hit():
    docs = rag_core.retrieve_vector("数据库事务的 ACID", k=3)
    conf = rag_core.get_confidence(docs)
    assert conf is not None and 0.0 <= conf <= 1.0


# ==================== 真实流式生成（评审 P3，2026-08-30） ====================
def test_stream_generation():
    """真实流式冒烟：invoke_stream 逐片段产出、聚合非空，检索/拼链与 invoke 同源。"""
    chain = rag_core.build_rag_chain()["chain"]
    docs = rag_core.retrieve_vector("数据库事务的 ACID 特性是什么", k=3)
    parts = list(chain.invoke_stream("什么是ACID", docs=docs))
    assert len(parts) >= 1, "流式应至少产出一个片段"
    assert "".join(parts).strip(), "流式片段聚合后应为非空答案"


# ==================== P1a：教材过滤多路召回（2026-09-04） ====================
def test_auto_route_scopes_to_textbook():
    """自动路由：ACID 题 → 结果全部来自数据库教材（top-8 去污染）+ 生效标记。"""
    docs = rag_core.retrieve_and_rerank("什么是数据库事务的 ACID 特性？请分别解释。")
    assert docs, "路由后应有召回"
    assert all("数据库" in d.metadata.get("source", "") for d in docs), \
        f"存在教材外来源: {[d.metadata.get('source') for d in docs]}"
    assert docs[0].metadata.get("source_filter"), "路由生效时应有 source_filter 标记"


def test_no_route_keeps_full_corpus():
    """未命中关键词 → 不过滤（无 source_filter 标记，行为同旧版）。"""
    docs = rag_core.retrieve_and_rerank("今天天气怎么样")
    if not docs:
        pytest.skip("语料对该问题零召回，跳过")
    assert all("source_filter" not in d.metadata for d in docs)


def test_explicit_source_filter_and_error():
    """显式子串过滤生效；解析不到源 → ValueError（不静默降级）。"""
    docs = rag_core.retrieve_and_rerank("ls 命令的选项有哪些", source_filter="Linux")
    assert docs
    assert all("linux" in d.metadata.get("source", "").lower() for d in docs)
    with pytest.raises(ValueError):
        rag_core.retrieve_and_rerank("任意问题", source_filter="不存在的教材XYZ")


def test_empty_filter_fallback(monkeypatch):
    """空结果降级：过滤后召回为空 → 自动去掉 filter 重检索一次。"""
    real = rag_core.retrieve_vector
    calls = {"n": 0}

    def fake(question, k=None, where=None):
        calls["n"] += 1
        if where:  # 带 filter 的首次调用 → 模拟召回为空
            return []
        return real(question, k=k, where=where)

    monkeypatch.setattr(rag_core, "retrieve_vector", fake)
    docs = rag_core.retrieve_and_rerank(
        "数据库事务的 ACID 特性", force_no_rerank=True, source_filter="数据库"
    )
    monkeypatch.undo()
    assert calls["n"] == 2, f"应恰好两次检索（带过滤 + 降级重试），实际 {calls['n']}"
    assert docs, "降级后应有召回"
    assert all("source_filter" not in d.metadata for d in docs), "降级结果不应带过滤标记"


# ==================== P2：多模态图像块（2026-09-05） ====================
def test_load_image_chunks_contract(tmp_path):
    """load_image_chunks 契约（纯 CSV 解析，不依赖 Ollama/索引）：
    - caption 正常行 → Document（source 同教材路径规律 / start_index 负值 / type=image / image_path / page）
    - CAPTION_FAILED 行跳过
    - 清单文件缺失 → 返回 []（绝不阻断文本建库）"""
    import csv as _csv
    manifest = tmp_path / "_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.DictWriter(f, fieldnames=["book", "page", "xrefs", "raw_w", "raw_h",
                                           "out_w", "out_h", "png_path", "md5"])
        w.writeheader()
        w.writerow({"book": "测试教材", "page": "29", "xrefs": "1895", "raw_w": "1725",
                    "raw_h": "592", "out_w": "622", "out_h": "215",
                    "png_path": "data/images/测试教材/p029_i00.png", "md5": "a" * 32})
        w.writerow({"book": "测试教材", "page": "30", "xrefs": "1900", "raw_w": "900",
                    "raw_h": "700", "out_w": "900", "out_h": "700",
                    "png_path": "data/images/测试教材/p030_i00.png", "md5": "b" * 32})
    caps = tmp_path / "_image_captions.csv"
    with open(caps, "w", newline="", encoding="utf-8-sig") as f:
        w = _csv.writer(f)
        w.writerow(["png_path", "caption", "elapsed_s"])
        w.writerow(["data/images/测试教材/p029_i00.png", "组织架构框图：项目管理者下设四个组。", "2.1"])
        w.writerow(["data/images/测试教材/p030_i00.png", "[CAPTION_FAILED] timeout", "90.0"])

    docs = rag_core.load_image_chunks(str(tmp_path))
    assert len(docs) == 1, "CAPTION_FAILED 行应被跳过"
    d = docs[0]
    assert d.page_content.startswith("组织架构框图")
    assert d.metadata["source"].endswith("测试教材.pdf"), "source 应与文本块同教材路径规律"
    assert d.metadata["start_index"] == -(29 * 1000 + 0), "start_index 负值（页码*1000+图序）"
    assert d.metadata["type"] == "image"
    assert d.metadata["image_path"] == "data/images/测试教材/p029_i00.png"
    assert d.metadata["page"] == 29

    # 清单缺失 → 空列表（首次部署 / 无图像资产时不影响建库）
    assert rag_core.load_image_chunks(str(tmp_path / "nonexist")) == []


def test_image_chunks_in_real_index():
    """真实索引含图像块：captions CSV 存在时，库内 type=image 的块应与 caption 数一致（重建后）。"""
    import csv as _csv
    from pathlib import Path
    caps = Path(config.DATA_DIR) / "images" / "_image_captions.csv"
    if not caps.exists():
        pytest.skip("无图像资产（data/images/_image_captions.csv 不存在）")
    col = rag_core.get_vector_store(auto_rebuild=False)
    got = col.get(where={"type": "image"}, include=["metadatas"])
    # 按 CSV **记录**数（不是行数）——B3 换描述器后 caption 内含换行
    # （`…描述\n\n图中文字： \n文件A \n文件B`），一行记录会跨多行文本，
    # 用 `read_text().splitlines()` 数会得到 4008 而不是 215（CSV 允许带引号的字段内换行）。
    with open(caps, encoding="utf-8-sig", newline="") as f:
        n_caps = sum(1 for r in _csv.DictReader(f) if (r.get("caption") or "").strip())
    assert len(got["metadatas"]) == n_caps, f"库内图像块 {len(got['metadatas'])} != caption 记录数 {n_caps}"
    for m in got["metadatas"]:
        assert m.get("type") == "image" and m.get("image_path") and m.get("page", 0) > 0
        assert m.get("start_index", 0) < 0, "图像块 start_index 必须为负（与文本块不冲突）"
