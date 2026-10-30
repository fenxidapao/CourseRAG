"""
api_schemas.py — FastAPI 接口的 Pydantic 请求 / 响应模型（2026-08-20 新增，部署层）。

与 app.py 的引用溯源结构对齐（rank / source / content / score / start_index），
保证 Streamlit 与 API 返回的『来源』格式一致，前端可无缝复用。
所有字段带 description，Swagger UI（/docs）可直接当作接口文档。
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Mode = Literal["fast", "accurate"]


# ==================== 通用 ====================
class Citation(BaseModel):
    """单条检索来源（与 app.py 渲染的引用结构一致）。"""

    rank: int = Field(..., description="最终排名（1 起）")
    source: str = Field(..., description="源文件名（不含路径）")
    content: str = Field(..., description="命中的原文块（图像块为 caption 描述文本）")
    score: Optional[float] = Field(None, description="相关性分数：reranker 开启为 rerank_score，否则为 hybrid_score 或 None")
    start_index: Optional[int] = Field(None, description="该块在原文中的字符偏移，供溯源定位；图像块为负值")
    type: Optional[str] = Field(None, description="块类型：课程图像块为 'image'；论文块为 text/table/figure/formula")
    image_path: Optional[str] = Field(None, description="图像块对应 PNG 的相对路径（仅图像块有）")
    page: Optional[int] = Field(None, description="论文块/教材图像块的物理页码（1 起）")
    paper_id: Optional[str] = Field(None, description="论文库 paper_id（仅论文块有）")
    title: Optional[str] = Field(None, description="论文标题（仅论文块有）")
    page_start: Optional[int] = Field(None, description="论文块起始物理页（论文块恒等于 page）")
    page_end: Optional[int] = Field(None, description="论文块结束物理页（论文块恒等于 page）")
    section: Optional[str] = Field(None, description="论文块所属章节路径，如 '3 Method > 3.2 Loss'")
    block_type: Optional[str] = Field(None, description="论文块归一化类型 text/table/figure/formula")
    block_ids: Optional[str] = Field(None, description="论文块对应的归一化 block_id 逗号串，供页图高亮")


class CheckItem(BaseModel):
    """单条健康检查结果。"""

    label: str = Field(..., description="检查项名称")
    ok: bool = Field(..., description="是否通过")
    msg: str = Field(..., description="检查结果说明")


# ==================== 问答 / 检索 ====================
class QueryRequest(BaseModel):
    """POST /query 请求体。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题（课程 / 课本知识点）")
    mode: Mode = Field("accurate", description="accurate=向量+BM25 融合+reranker 精排（默认，质量最高，实测 context_recall +16.7%）；fast=纯向量 top-5（降级档：GPU 不可用/显存不足/高并发时使用）。延迟因硬件而异——有 CUDA 时精排亚秒级，纯 CPU 会显著变慢")
    custom_prompt: Optional[str] = Field(None, description="追加到系统提示词的用户自定义指令；留空用默认安全提示词")
    history: Optional[str] = Field(None, description="最近若干轮对话的文本（可选，做多轮记忆；格式随意，由调用方拼好）")
    source: Optional[str] = Field(
        None,
        description="教材过滤（可选）：源文件名子串，如 '数据库' / 'Linux' / '小程序'，命中多个文件时全部纳入；"
        "不传=按问题关键词自动路由到对应教材（rag_search_utils.SOURCE_RULES），未命中关键词=全库检索",
    )


class QueryResponse(BaseModel):
    """POST /query 响应体。"""

    answer: str = Field(..., description="模型回答（含 [n] 内联来源标注，n 对应 citations 的 rank）")
    mode: Mode = Field(..., description="实际使用的检索模式")
    citations: list[Citation] = Field(..., description="检索 + 重排后的来源列表（按 rank 升序）")
    latency_ms: int = Field(..., description="端到端耗时（毫秒，含检索 + 生成）")
    grounded: bool = Field(..., description="接地判定（启发式：非空且未声称『无法回答』）")
    low_confidence: bool = Field(
        False, description="T3：top-1 向量相似度低于 LOW_CONFIDENCE_THRESHOLD，资料可能不足"
    )
    cached: bool = Field(
        False, description="P2 语义缓存：是否命中缓存直接复用历史答案（未走检索/生成）"
    )


class RetrieveRequest(BaseModel):
    """POST /retrieve 请求体。"""

    question: str = Field(..., min_length=1, max_length=2000, description="检索问题")
    mode: Mode = Field("accurate", description="同 /query 的 mode 语义（默认 accurate）")
    source: Optional[str] = Field(
        None,
        description="教材过滤（可选）：源文件名子串，同 /query 的 source 语义；不传=自动路由",
    )


class RetrieveResponse(BaseModel):
    """POST /retrieve 响应体（仅检索，不调用 LLM 生成）。"""

    query: str = Field(..., description="原始问题")
    mode: Mode = Field(..., description="实际使用的检索模式")
    docs: list[Citation] = Field(..., description="检索 + 重排结果（按排名升序）")


# ==================== 文档管理 ====================
class DocInfo(BaseModel):
    """data/ 下单个文档的摘要。"""

    name: str = Field(..., description="文件名")
    size_bytes: int = Field(..., description="文件大小（字节）")
    chunks: int = Field(..., description="该文档在向量库中的块数（0=未索引）")


class DocumentsResponse(BaseModel):
    """GET /documents 响应体。"""

    files: list[DocInfo] = Field(..., description="文档列表（按文件名排序）")
    total_chunks: int = Field(..., description="向量库总块数")


class IngestResponse(BaseModel):
    """POST /documents 响应体（上传 + 增量索引）。"""

    file: str = Field(..., description="上传的文件名")
    saved_to: str = Field(..., description="保存的完整路径（data/ 下）")
    added_chunks: int = Field(..., description="实际新增 / 覆盖的块数（稳定 id 去重）")


class DeleteResponse(BaseModel):
    """DELETE /documents/{name} 响应体。"""

    deleted: bool = Field(..., description="是否已删除")
    filename: str = Field(..., description="被删除的文件名")
    warning: Optional[str] = Field(None, description="警告：文件删除失败但索引已移除（如文件被占用 / 权限不足）；正常删除时为 null")


# ==================== 索引管理 ====================
class RebuildResponse(BaseModel):
    """POST /index/rebuild 响应体。"""

    chunks: int = Field(..., description="重建后的总块数")
    files: int = Field(..., description="参与索引的文档数")


class IndexStats(BaseModel):
    """GET /index/stats 响应体。"""

    collection: str = Field(..., description="Chroma 集合名")
    count: int = Field(..., description="向量库总块数")
    source_files: int = Field(..., description="已索引的源文档数（按 source 去重）")
    persist_directory: str = Field(..., description="向量库持久化目录")


# ==================== 论文库（v2.1） ====================
class PaperInfo(BaseModel):
    paper_id: str
    filename: str = ""
    title: str = ""
    status: str = ""
    pages: int = 0
    blocks: int = 0
    chunks: int = 0
    parse_sec: float = 0
    pages_per_sec: float = 0
    error: str = ""
    updated_at: str = ""


class PaperList(BaseModel):
    papers: list[PaperInfo]


class PaperUploadItem(BaseModel):
    paper_id: str
    filename: str
    status: str


class PaperUploadResponse(BaseModel):
    files: list[PaperUploadItem]


class PaperActionResponse(BaseModel):
    paper_id: str
    status: str


class PaperBlocksResponse(BaseModel):
    paper_id: str
    n_pages: int
    pages: list[dict]


class PaperDeleteResponse(BaseModel):
    deleted: bool
    paper_id: str
    raw_deleted: bool = False


# ==================== 健康检查 ====================
class HealthResponse(BaseModel):
    """GET /health 响应体。"""

    ok: bool = Field(..., description="全部检查通过才为 true")
    checks: list[CheckItem] = Field(..., description="逐项检查结果（依赖 / Ollama / Reranker 权重）")
    vector_store_count: Optional[int] = Field(None, description="当前向量库块数（未能加载时为 null）")
