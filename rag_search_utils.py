"""
rag_search_utils.py — 零依赖的纯检索算法（P2-1 可单测核心）

把混合检索里的「纯算法」从 rag_core.py 抽出来，单独成模块：
  - 只依赖标准库（re / math / hashlib），不 import chromadb / ollama / sentence_transformers；
  - 因此可以在没有 LLM 栈的环境里（CI、沙箱、本机 venv 外）直接 import 并单测；
  - rag_core.py 通过 `from rag_search_utils import (...)` 重新导出，对外接口不变，
    内部调用（hybrid_retrieve / add_documents）照旧工作。

包含的纯函数：
  _tokenize(text)            中英文混合分词（拉丁串整词 + CJK 逐字）
  _BM25                       Okapi BM25 打分（自实现，无外部依赖）
  _bm25_rank(question,docs)   对同一批候选做 BM25 词面打分并降序返回（P1a 支持 allowed_idx 来源过滤）
  _rrf_fuse(ranked_lists,N)   倒数排名融合（RRF），写回 metadata['hybrid_score']
  _stable_ids(chunks)         稳定 id = md5(source::start_index)，用于增量 upsert 去重
  SOURCE_RULES / route_sources  P1a 教材级规则路由（query 关键词 → 允许的 source 集合）

文档对象约定：只需有 .page_content（str）与 .metadata（dict）两个属性，
因此单测里用轻量 duck-typed 对象即可，无需引入 langchain / rag_core.Document。
"""
import re
import math
import hashlib


def _tokenize(text: str):
    """中英文混合分词，零新依赖：
    - 连续拉丁/数字串（ACID、2NF、transaction）作为一个 token（转小写）
    - 每个 CJK 汉字作为独立 unigram（无 jieba，但足以覆盖课程精确术语的词面对齐）
    """
    text = (text or "").lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    tokens += re.findall(r"[一-鿿]", text)
    return tokens


class _BM25:
    """Okapi BM25（自实现，无外部依赖）。在给定候选文档集合上打分。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [_tokenize(d.page_content) for d in docs]
        self.n = len(self.doc_tokens)
        self.doc_lens = [len(t) for t in self.doc_tokens]
        self.avgdl = (sum(self.doc_lens) / self.n) if self.n else 0.0
        self.df = {}
        for toks in self.doc_tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

    def scores(self, query: str):
        q_tokens = _tokenize(query)
        if not q_tokens or self.n == 0:
            return [0.0] * self.n
        out = []
        for i, toks in enumerate(self.doc_tokens):
            dl = self.doc_lens[i]
            freq = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            score = 0.0
            for qt in set(q_tokens):
                f = freq.get(qt, 0)
                if f == 0 or qt not in self.df:
                    continue
                idf = math.log((self.n - self.df[qt] + 0.5) / (self.df[qt] + 0.5) + 1)
                score += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
            out.append(score)
        return out


def _bm25_rank(question, docs, top_n=None, index=None, allowed_idx=None):
    """对 docs（同一批向量候选）做 BM25 词面打分，返回按分数降序的 Document 列表。

    index（评审 P2-4，2026-08-30）：可选，传入已构建的 _BM25 实例则跳过重建——
    调用方（rag_core）按语料版本缓存索引，避免每次查询都对全库重新分词
    （O(语料)/query）。index 与 docs 必须来自同一次构建（下标一一对应）。

    allowed_idx（P1a，2026-09-04）：可选，允许参与排名的全库下标集合（元数据
    过滤多路召回的 BM25 路过滤）。语义是「过滤语料内的 top-N」——先按分数对
    全库降序排，再剔除不在 allowed_idx 的下标，最后截 top_n。与向量路的
    Chroma where 过滤（子集内 top-K）语义对齐：两路各自在自己允许的子集上
    独立召回，再 RRF 融合。
    """
    if not docs:
        return []
    if index is None:
        index = _BM25(docs)
    scores = index.scores(question)
    ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    if allowed_idx is not None:
        allow = set(allowed_idx)
        ranked = [i for i in ranked if i in allow]
    if top_n:
        ranked = ranked[:top_n]
    return [docs[i] for i in ranked]


def _rrf_fuse(ranked_lists, top_n, rrf_k=60):
    """倒数排名融合（Reciprocal Rank Fusion）。
    ranked_lists: 多个已按相关性降序排列的 Document 列表（排名从 0 起）。
    同一文档用 (source, start_index) 去重，融合分 = Σ 1/(rank + rrf_k)，降序取 top_n。
    融合分写回 doc.metadata['hybrid_score']（供 UI 在 reranker 关闭时展示）。
    """
    fused = {}
    for rl in ranked_lists:
        for rank, doc in enumerate(rl):
            key = (doc.metadata.get("source"), doc.metadata.get("start_index"))
            entry = fused.get(key)
            if entry is None:
                entry = {"doc": doc, "score": 0.0}
                fused[key] = entry
            entry["score"] += 1.0 / (rank + rrf_k)
    items = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    if top_n:
        items = items[:top_n]
    for it in items:
        it["doc"].metadata["hybrid_score"] = round(it["score"], 4)
    return [it["doc"] for it in items]


def _stable_ids(chunks):
    """稳定 id：优先 chunk_uid（论文块），否则 md5(source::start_index)（课程块）。"""
    ids = []
    for d in chunks:
        uid = d.metadata.get("chunk_uid")
        key = str(uid) if uid else f"{d.metadata['source']}::{d.metadata['start_index']}"
        ids.append(hashlib.md5(key.encode("utf-8")).hexdigest())
    return ids


# ====================== P1a：教材级规则路由（2026-09-04） ======================
# 元数据过滤多路召回的「规则路由」层：query 命中课程/教材关键词 → 限定检索范围
# 到对应教材（source 子串模式），未命中任何规则 → 不过滤（现行为不变）。
#
# 设计要点（对照 2026-09-04 方案 §2.2 / A 组 30 题逐题核对）：
#   - 规则值是「源文件名子串模式」，不是写死的绝对路径——换机器/加语料不用改表；
#   - 多规则命中取并集（如「银行家算法」同时命中 OS 与 算法引论），宁可放宽
#     也不误杀正确教材；
#   - 关键词刻意保守（单字不入表，如「图」「树」），漏路由 = 退回全库检索（安全
#     旧行为），误路由 = 可能过滤掉正确教材（有害），两害取其轻；
#   - B 组（语料外）安全性论证：filter 只收紧候选集 → top-1 相似度单调不升 →
#     检索硬拒答只会更易触发，拒答率数学上不可能因路由下降。
SOURCE_RULES = [
    # 数据库三件套：数据库-关系代数与SQL / 数据库-函数依赖与范式 / 数据库系统-事务与ACID
    (["数据库", "sql", "事务", "acid", "隔离级别", "范式", "2nf", "3nf", "bcnf",
      "函数依赖", "关系代数", "候选码", "主键", "外键", "自然连接"],
     ["数据库"]),
    # 数据结构三件套：树与二叉树 / 排序算法 / 图与最短路径
    (["数据结构", "二叉树", "哈夫曼", "遍历", "排序", "链表", "队列",
      "哈希", "散列", "最短路径", "dijkstra", "图的", "时间复杂度"],
     ["数据结构"]),
    # OS 两件套：OS-内存管理 / 操作系统-进程线程与调度
    (["操作系统", "进程", "线程", "调度", "死锁", "内存", "分页", "分段", "碎片",
      "虚拟内存", "银行家", "信号量", "互斥", "存储管理", "页面置换"],
     ["os-", "操作系统"]),
    # 计网两件套：计算机网络-TCP与HTTP / 计网-IP与子网路由
    (["计算机网络", "计网", "tcp", "udp", "http", "子网", "路由", "三次握手",
      "四次挥手", "dns", "mtu", "分片", "cidr", "以太网", "网络协议", "osi"],
     ["计算机网络", "计网"]),
    # 组成原理两件套：指令系统与寻址 / 存储与流水线
    (["组成原理", "计算机组成", "指令", "寻址", "流水线", "cache", "高速缓存",
      "存储器", "cpu", "中断"],
     ["组成原理", "计算机组成"]),
    # Python（廖雪峰教程核心章节）
    (["python", "列表", "字典", "元组", "切片", "迭代器", "生成器", "装饰器",
      "廖雪峰"],
     ["python"]),
    # Linux 四件套 PDF
    (["linux", "shell", "vim", "chmod", "软链接", "硬链接"],
     ["linux"]),
    # 微信小程序（wx. 前缀 API 调用如 wx.request）
    (["小程序", "wx.", "wxml", "wxss"],
     ["小程序"]),
    # Streamlit 参考手册
    (["streamlit"], ["streamlit"]),
    # 算法引论 pptx
    (["算法", "np", "p类", "np类", "贪心", "分治", "动态规划", "复杂度"],
     ["算法引论"]),
]

# 路由实际使用的规则表：rag_core 在 paper profile 下置空（课程规则不适用于论文库）。
# SOURCE_RULES 本身永远保留完整课程规则，test_core.py 的 len(SOURCE_RULES)>=10 不受影响。
ACTIVE_SOURCE_RULES = SOURCE_RULES


def route_sources(question, sources):
    """规则路由：question 命中 SOURCE_RULES 关键词 → 返回允许的 source 路径列表。

    - 未命中任何规则 / 关键词 → None（调用方按全库检索，行为与旧版一致）
    - 命中的模式匹配到全部源（过滤无意义）→ None
    - 多规则命中取并集；匹配对大小写不敏感（拉丁关键词统一 lower 比较）
    - sources：当前索引中实际存在的 source 路径列表（调用方从向量库取，表内
      模式只是名字子串，不假设路径格式）
    """
    q = (question or "").lower()
    if not q:
        return None
    patterns = []
    for keywords, src_patterns in ACTIVE_SOURCE_RULES:
        if any(kw in q for kw in keywords):
            patterns.extend(src_patterns)
    if not patterns:
        return None
    allowed = sorted(
        {s for s in sources if any(p.lower() in s.lower() for p in patterns)}
    )
    if not allowed or len(allowed) == len(set(sources)):
        return None
    return allowed


# ====================== 枚举/穷举类查询检测（2026-09-11） ======================
# 用途：触发「锚点扩展召回」（rag_core._enum_expand），治枚举题的供给不足。
#
# 为什么需要单独判定：held-out A 组 30 题里 5 道枚举/清单类题只占 16.7% 题量，
# 却吃掉 49.2% 的 context_precision 损失，其中「Streamlit 有哪些输入组件」是
# 全组唯一 recall 失分题（0.4）。这类问题需要的是**更多块**，而常规题多给块
# 只会灌噪声——所以必须区分对待，不能一刀切调大 RERANKER_TOP_N。
#
# 判定取向：宁可漏判（退回旧行为，安全）也不误判（多花延迟且可能引入噪声）。
# 故只用明确的**列举词**，不用「什么/为什么」等泛词。
ENUM_MARKERS = (
    "哪些", "列举", "罗列", "列出", "分别", "各自",
    "几种", "几类", "分类", "种类", "包括什么", "包含哪些",
)


def is_enum_query(question: str) -> bool:
    """是否为枚举/穷举类查询（命中任一列举词即视为是）。

    扩展召回的**安全性**来自「只改变候选规模、不改变召回范围与打分」：
    - 不改检索范围（锚点仍受既有教材过滤约束，不会越出 source_filter）
    - 不改 reranker 打分（bge-reranker 对 (query, doc) 独立打分，与候选池大小无关）
    因此硬拒答阈值（基于 rerank_score 绝对值）与 B 组拒答行为在数学上不受影响。
    """
    q = (question or "").lower()
    return any(m in q for m in ENUM_MARKERS)
