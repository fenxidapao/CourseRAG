"""
visual_rerank.py — B1 视觉重排：把**真实像素**引入检索排序（补「洞 R」）。

> 文中引用的 b1_* / b3_* / _probe_* 标定脚本已归档到仓库外 `03_一次性实验脚本/`，不在本仓库内。

## 为什么需要它

caption 路线让图像以「文本代理」身份进索引（`rag_core.load_image_chunks`：page_content
= VLM 生成的中文描述）。代价是**像素从未参与检索**——两张 caption 几乎相同的截图，
向量召回 / BM25 / bge-reranker 看到的文字几乎一样，分不出谁是谁。

本机实测（2026-10-14，`微信小程序开发实战` 四张项目结构截图，见 B1 caption 探针（已归档））：

| 文件 | 真实项目名 | 索引里的 caption |
|---|---|---|
| `p045_i00.png` | 个人信息 | 「微信小程序的项目结构，包含 images、pages 和 index 等文件夹…」 |
| `p082_i00.png` | 比较数字大小 | 「微信小程序的项目结构，包括 pages 文件夹下的 index 文件夹…」 |
| `p167_i00.png` | 用户登录 | 「…包括 pages 文件夹下的 index 文件夹，以及 index.js…」 |
| `p267_i00.png` | 电影列表 | 「…包括 images、pages、index 等文件夹及其内容…」 |

四条 caption 近乎同义；判别性 token（`tab1.wxml` / `avatar.png` / `个人信息` /
`比较数字大小`）在全部 caption 中出现 **0 次**。同一组图送 `qwen3-vl-plus` 做 4 选 1 +
位置置换：**15/16 命中（随机基线 25%）**，唯一未命中是 `dt=0.0s` 的空响应（瞬时网络
故障），成功调用 **15/15 全对**。→ 像素里有文本索引完全丢失的信息，且 VLM 能读出来。

## 设计：为什么不能只「重排已有候选」

第一版只把候选里已有的图像块互相换位（`mode="slot"`）。实测证伪（B1 池宽探针，已归档）：

    RRF 池(k=16) 有 4~7 张图 → reranker 截到 top-8 后只剩 0~2 张 → 视觉重排静默失效

而把池宽从 16 拉到 200（B1 召回探针，已归档）也救不回来：**6 道题里 5 道的目标图在
任何池宽下都不进池**。原因不在池宽，在 caption——文本通道压根排不出这些图。

所以本模块做两件事，缺一不可：

1. **专用图像通道**（`VISUAL_RECALL_K`）：绕开文本竞争，用 `where={"type":"image"}` 独立
   召回一批图像块，与文本链路的候选合并。实测该通道能把 `p267`（rank 6/215）与
   `p045`（rank 24/215）捞进候选——纯 RRF 池漏掉了后者。
2. **视觉裁决 + 置顶**（`mode="promote"`）：把合并后的候选图交给 VLM，问「哪张真的在讲 X」，
   命中则把该图**提到 rank 1**。文本块相对顺序不动。

## 三条设计约束

1. **VLM 可以说「没有一张相关」**（`best=0`）。这是防污染的安全阀：问题与图无关时，
   强制在候选里挑一张会凭空塞一张图进 top-1。只有 VLM 明确指认才置顶。
2. **fail-open。** 任何异常 / 超时 / HTTP 错误 / JSON 解析失败 / 图号越界，一律返回原顺序。
   增强项绝不能变成新的故障点（与 `QUERY_REWRITE_ENABLED` 同一纪律）。
3. **默认关闭**（`config.VISUAL_RERANK_ENABLED=false`）。按项目评估纪律（config.py §0.4），
   检索层改动必须先拿到 held-out A/B 数字再翻转默认值，不能「觉得应该更好」就打开。

## 落点

`rag_core.retrieve_and_rerank` 末尾（bge-reranker 之后、`build_citations` 之前）。
放这里而不是 `answer_question`，是因为 `retrieve_and_rerank` 是 UI / API / RAG 链
共用的统一检索入口——放这里三处出口自动一致，也保证 `eval_heldout.py` 能直接量到效果。
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import config

# 项目根 = 本文件所在目录（与 app.py:347 `Path(__file__).parent / image_path` 同一约定）
_PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_image_path(rel_path):
    """把 metadata['image_path']（相对项目根的 PNG 路径）解析为绝对路径；不存在返回 None。"""
    if not rel_path:
        return None
    p = Path(rel_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p if p.is_file() else None


def _collect_image_docs(docs):
    """挑出候选里的图像块 → [(在 docs 中的下标, doc, 绝对路径)]，跳过文件缺失的块。"""
    out = []
    for idx, d in enumerate(docs):
        if d.metadata.get("type") != "image":
            continue
        abspath = _resolve_image_path(d.metadata.get("image_path"))
        if abspath is None:
            continue
        out.append((idx, d, abspath))
    return out


def _image_channel(question, sources):
    """专用图像通道：只对图像块做向量召回，绕开文本块竞争。

    实测依据（B1 召回探针，已归档）：纯 RRF 池 k=16 时图像块仅 4~7 个，且 `p045`
    （图像通道 rank 24）完全没进池。独立通道是让视觉重排「有图可排」的前提。

    延迟 import rag_core：rag_core 顶部 import 本模块，模块级互相 import 会成环。
    """
    if config.VISUAL_RECALL_K <= 0:
        return []
    import rag_core  # noqa: PLC0415 — 有意延迟，避免与 rag_core 循环 import

    clauses = [{"type": "image"}]
    if sources:
        clauses.append({"source": {"$in": list(sources)}})
    where = clauses[0] if len(clauses) == 1 else {"$and": clauses}
    try:
        return rag_core.retrieve_vector(question, k=config.VISUAL_RECALL_K, where=where)
    except Exception as e:  # noqa: BLE001 — 图像通道失败不该影响主链路
        print(f"[visual_rerank] 图像通道召回失败，仅用文本链路候选：{type(e).__name__}: {e}")
        return []


_img_docs_cache = None


def _all_image_docs(sources=None):
    """全库图像块（读 caption CSV，不走向量库），进程内缓存。

    用于词面图像通道：215 条 caption 逐字扫描是微秒级，比一次向量查询便宜得多。
    """
    global _img_docs_cache
    if _img_docs_cache is None:
        try:
            import rag_core  # noqa: PLC0415 — 延迟 import，避免与 rag_core 循环依赖

            _img_docs_cache = rag_core.load_image_chunks()
        except Exception as e:  # noqa: BLE001
            print(f"[visual_rerank] 图像块加载失败：{type(e).__name__}: {e}")
            _img_docs_cache = []
    if not sources:
        return _img_docs_cache
    want = {str(s) for s in sources}
    return [d for d in _img_docs_cache if str(d.metadata.get("source")) in want]


def _lexical_image_channel(question, sources):
    """词面图像通道：caption 里**逐字出现**了问题判别词的图像块，按命中词数降序。

    为什么必须有这条路（2026-10-14 实测，B3 之后）：`p167_i00.png` 的 caption 里
    **逐字写着**「用户登录」，但纯向量通道 top-32 里没有它 —— `nomic-embed-text`
    分不开这批 400 字近同义目录树截图。同类失败还有「电影列表」「比较数字大小」。

    判别词由 `_content_tokens` 给出（词级、DF<5%、已过滤停用词与泛化名词），
    所以「项目」「结构」这类词不会把全库图片都捞进来 —— 只有真正稀有的词才触发。
    专有名词量级：`用户登录` / `比较数字大小` / `电影列表` 在 215 条 caption 里 DF≈0。

    另外把问题里**引号内的专有名词**与**带点标识符**整体也算作判别词（见 `_QUOTED_RE`）：
    连词成句的专有名词必须整词命中，不能被 jieba 拆碎。
    """
    toks = _content_tokens(question)
    toks |= {q.strip().lower() for q in _QUOTED_RE.findall(question) if q.strip()}
    toks |= {_norm_ident(i) for i in _IDENT_RE.findall(question)}
    if not toks:
        return []
    scored = []
    for d in _all_image_docs(sources):
        blob = _norm_ident(d.page_content)
        hits = sum(1 for t in toks if t and t in blob)
        if hits:
            scored.append((hits, d))
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored]


def _merge_candidates(question, docs, sources):
    """文本链路候选 ∪ 词面图像通道 ∪ 向量图像通道，按 image_path 去重。

    去重键用 image_path（不是 id()）：同一张图可能同时被几条路召回，但 Document 实例不同。
    顺序有意为之：**词面命中排在最前**（高精度，含问题里的专有名词/文件名），
    再放向量通道（高召回）。`VISUAL_RERANK_MAX_IMAGES` 的截断按这个顺序取，
    所以词面命中的图不会被向量噪声挤掉。
    """
    merged = list(docs)
    seen = {d.metadata.get("image_path") for d in merged if d.metadata.get("type") == "image"}
    for d in _lexical_image_channel(question, sources) + _image_channel(question, sources):
        ip = d.metadata.get("image_path")
        if ip and ip not in seen:
            seen.add(ip)
            merged.append(d)
    return merged


def _data_uri(path):
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


_PROMPT_VERIFY = """你是检索结果重排器。下面按顺序给你 {n} 张候选插图（图1 到 图{n}），它们来自同一批课程教材，彼此内容相似。

用户的问题：「{question}」

请**逐张看图**，只根据图里**实际画出来/写出来的内容**（不是图注怎么写的）判断：这张图里是否真的出现了能回答该问题的内容？

对每张命中的图，必须**逐字抄出**你在图上真正看到的、支撑该判断的文字（文件名、文件夹名、代码标识符等）。抄不出具体文字就不要算命中。

只输出一个 JSON，不要任何解释文字：
{{"hits": [{{"idx": 图号, "evidence": "你在图上真正看到的文字，逗号分隔"}}], "reason": "一句话说明"}}

注意：如果没有任何一张图真的命中，hits 必须是空数组 []。宁可给空数组，也不要猜。"""

_PROMPT_PICK = """你是检索结果重排器。下面按顺序给你 {n} 张候选插图（图1 到 图{n}），它们来自同一批课程教材，彼此内容相似。

用户的问题：「{question}」

请**逐张看图**，判断图里实际画了什么（而不是图注怎么写的），找出与用户问题视觉上最相关的那一张。

只输出一个 JSON，不要任何解释文字：
{{"best": 最相关图的编号, "reason": "一句话说明"}}

注意：如果**没有任何一张**图与问题相关，必须把 best 设为 0，不要勉强选一张。"""

# ---- 证据门（evidence gate）----
# 为什么需要它（B1 标定实验 2026-10-14 实测，脚本已归档）：不加门时 VLM 在「目标图根本不在候选里」的
# 5/6 道题上照样给出自信的假命中（把 p141 / p276 / p026 / p258 顶到 rank 1），净负收益。
#
# 两道子门，各自对应一次实测到的假阳性：
#
# 1. **文件名门**：问题里出现了 `avatar.png` 这类「带点的标识符」时，证据必须**整体含该
#    文件名**。实测反例：问 `avatar.png`，VLM 抄的证据是 `images, guest.png`——按分词
#    匹配会因共有 `png` 而误判命中（把 p141 顶到 rank 1）。`png` 在语料里 DF=0.5%，
#    光靠 DF 过滤不掉，必须按「完整文件名」比。
# 2. **稀有词门**：其余情况要求证据与问题的交集里至少有一个 DF < 5% 的判别性 token。
#    判据来自 caption 语料实测（215 条）：判别性 token `tab1`=0 / `电影`=0.5% / `个人`=0 /
#    `比较`=0 / `登录`=1.9%；而泛化填充 `项目`=12% / `结构`=15% / `微信`=25% / `文件`=53%。
#    假命中的证据（`pages` / `app.js` / `project.config.json` 这类小程序通用目录名）
#    与问题唯一的交集恰好都是这些泛化词 → 被这道门拒掉。
_EVIDENCE_MAX_DF = 0.05

# 中文虚词 + 英文疑问/泛化词：即使 DF 表不可用也要挡住的高频无信息词。
# 第二部分（B3 后补）：提问句式里的功能词 + 本语料里到处都有的泛化名词。
# 这些词在 caption 语料里 DF 很低（`名叫`/`怎样`/`那个` 甚至是 0），**光靠 DF 挡不住**，
# 但它们对「这张图是不是问题问的那张」毫无判别力 —— 必须显式列进停用词。
_STOP = {
    # 虚词
    "的", "了", "是", "在", "和", "与", "有", "个", "这", "那", "哪", "什", "么",
    "怎", "样", "请", "问", "图", "片", "中", "吗", "呢", "把", "被", "为", "对",
    "从", "到", "会", "能", "要", "就", "都", "也", "很", "并", "及", "或", "等",
    "what", "which", "the", "is", "are", "of", "in", "and", "or", "to", "for",
    # 提问句式
    "名叫", "叫做", "叫", "那个", "这个", "哪个", "哪些", "怎样", "怎么", "什么",
    "如何", "请问", "哪个", "一个", "下面", "上面", "里面", "里有", "名字", "名称",
    "介绍", "说明", "关于", "请问", "以及", "其中", "如下", "所示", "内容",
    # 本语料泛化名词（215 张教材插图里到处都是）
    "项目", "结构", "目录", "文件", "文件夹", "截图", "显示", "包含", "包括",
    "图片", "图中", "界面", "软件", "这张", "展示", "例如", "比如", "用于",
}

# 带点的标识符（文件名 / 扩展名形态）：avatar.png、tab1.wxml、app.js
_IDENT_RE = re.compile(r"[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")

# 引号里的专有名词：问题写「用户登录」时，希望**整词逐字**命中 caption，
# 而不是被 jieba 拆成 用户/登录 后只剩 `登录`（`用户` 在跨书语料里 DF 过高会被滤掉，
# 实测「用户登录」题的词面命中因此退化到 8 张、目标掉到 rank 6）。
_QUOTED_RE = re.compile(r"[「『“\"'《]([^」』”\"'》]{2,})[」』”\"'》]")

_df_cache = None


def _word_tokens(text):
    """切成「词」而不是「字」：jieba 中文词 + 拉丁/数字词，长度 ≥ 2。

    为什么必须是词级：`rag_search_utils._tokenize` 对中文**逐字**切分，于是
    `个人信息` → 个/人/信/息，四个字的 caption DF 分别是 0.474/0.093/0.363/0.251，
    **全部**高于 `_EVIDENCE_MAX_DF=0.05` → 任何中文专有名词都抽不出证据词，
    证据门对所有中文问题退化成「永远拒绝」。

    实测（B1 词频探针，已归档；语料 215 条）词级 DF：`个人信息`=0.005、
    `比较数字大小`=0.000、`用户登录`=0.000、`电影列表`=0.000；
    而泛化词 `项目`=0.163、`结构`=0.177 —— 词级 DF 能把两者干净分开。

    jieba 不可用时退化为字符级切分（保留旧行为，不静默失效）。
    """
    text = text or ""
    out = set()
    try:
        import jieba  # noqa: PLC0415

        for w in jieba.cut(text):
            w = w.strip().lower()
            if len(w) >= 2:
                out.add(w)
    except Exception:  # noqa: BLE001 — jieba 缺失不该影响检索
        from rag_search_utils import _tokenize  # noqa: PLC0415

        out |= {t for t in _tokenize(text) if len(t) >= 2}
    # 拉丁/数字词（jieba 会把 `avatar.png` 拆成 avatar / . / png）
    out |= {w.lower() for w in re.findall(r"[A-Za-z0-9_]{2,}", text)}
    return out


def _caption_df():
    """caption 语料里每个**词**的文档频率（DF），进程内缓存。语料缺失返回空 dict。"""
    global _df_cache
    if _df_cache is not None:
        return _df_cache
    from collections import Counter  # noqa: PLC0415

    df, total = Counter(), 0
    try:
        import csv  # noqa: PLC0415
        # B3：与 rag_core.load_image_chunks 读**同一个** caption 文件（换描述器后 DF
        # 先验必须跟着换，否则证据门会用旧语料的稀有度判断新语料的词）。
        path = _PROJECT_ROOT / "data" / "images" / config.IMAGE_CAPTIONS_CSV
        with open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                total += 1
                for t in _word_tokens(row.get("caption", "")):
                    df[t] += 1
        _df_cache = {t: c / max(total, 1) for t, c in df.items()}
    except Exception as e:  # noqa: BLE001 — 语料缺失不该影响检索
        print(f"[visual_rerank] caption 语料不可读（{type(e).__name__}: {e}），"
              f"证据门退化为停用词表模式")
        _df_cache = {}
    return _df_cache


def _content_tokens(text):
    """抽出「有实义」的词：语料 DF 低于阈值的**词**才算证据词（词级，非字符级）。

    DF 表不可用时（语料缺失）退化为「长度 ≥ 2 且非停用词」，保证 fail-open。
    """
    tokens = _word_tokens(text)
    df = _caption_df()
    if not df:
        return {t for t in tokens if len(t) >= 2 and t not in _STOP}
    return {t for t in tokens if df.get(t, 0.0) < _EVIDENCE_MAX_DF and t not in _STOP}


def _norm_ident(s):
    """标识符归一化：小写 + 去掉所有空白（容忍 VLM 写成 `tab1. wxml`）。"""
    return re.sub(r"\s+", "", s).lower()


def _unescape(ev):
    """正确还原 JSON 字符串里的转义（`\\u9879` → `项`、`\\"` → `"`）。

    历史 bug：这里曾用 `ev.encode().decode("unicode_escape")`。它对 `\\uXXXX` 有效，
    但对**本身就是真 UTF-8 中文**的 evidence 会先 encode 成 UTF-8 字节再被当成
    latin-1 解码，实测把 `项目管理者, 产品组` 变成 `é¡¹ç®¡çåè, äº§åç»` —— 于是
    证据门在**所有中文问题上**都把真命中判成了「无交集」，B1 静默失效。
    正确做法：交给 json 解析，失败则原样返回（绝不损坏原文）。
    """
    try:
        return json.loads(f'"{ev}"')
    except Exception:
        return ev


def _evidence_supports(question, evidence):
    """证据门：VLM 抄出的图上文字，是否足以支撑「这张图命中了问题」？

    问题里有文件名标识符 → 要求证据整体含该标识符（`avatar.png` 不能用 `png` 顶替）。
    否则 → 要求交集里至少有一个 DF < 5% 的判别性**词**（挡住 `项目`/`结构` 这类泛化词）。

    问题里抽不出任何判别性词时 **fail-open**（返回 True）：门只用来挡假阳性，
    挡不住时应该放行让 VLM 的判断生效，而不是把所有图都拒掉。
    """
    if not evidence:
        return False
    ev = _norm_ident(evidence)
    idents = {_norm_ident(i) for i in _IDENT_RE.findall(question)}
    if idents:
        hit = {i for i in idents if i in ev}
        if not hit:
            print(f"[visual_rerank] 文件名门：证据里没有问题指定的文件 "
                  f"{sorted(idents)}（evidence={evidence[:80]!r}）")
        return bool(hit)
    q_tokens = _content_tokens(question)
    if not q_tokens:
        print(f"[visual_rerank] 问题里没有判别性词（{question[:40]!r}），证据门放行")
        return True
    return bool(q_tokens & _content_tokens(evidence))


def _post(payload, timeout):
    """裸 urllib 发请求（与 model.py 同风格，零 SDK 依赖）。返回 (ok, data_or_err)。"""
    base = config.VISUAL_RERANK_API_BASE.rstrip("/")
    req = urllib.request.Request(
        base + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + config.VISUAL_RERANK_API_KEY,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
    except Exception as e:  # noqa: BLE001 — 网络层任何异常都不该炸掉问答
        return False, f"{type(e).__name__}: {e}"


def _ask_vlm(question, image_docs, timeout, mode):
    """送 N 张图问视觉裁决。返回命中图号列表（1-based，可能为空），失败返回 None。

    - `mode="verify"`：问「哪些图真的命中了」（hits 数组）——**校准过的判定**，可以说空。
    - 其他模式：问「哪张最相关」（单个 best）——相对选择，实测从不弃权，见下方注释。

    重试一次：本机实测偶发 `dt=0.0s` 空响应（瞬时网络故障），单次失败不代表模型不可用。
    """
    n = len(image_docs)
    prompt = _PROMPT_VERIFY if mode == "verify" else _PROMPT_PICK
    content = [{"type": "text", "text": prompt.format(n=n, question=question)}]
    for _, _, abspath in image_docs:
        content.append({"type": "image_url", "image_url": {"url": _data_uri(abspath)}})

    payload = {
        "model": config.VISUAL_RERANK_MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,
    }

    for attempt in (1, 2):
        t_call = time.time()
        ok, data = _post(payload, timeout)
        if not ok:
            print(f"[visual_rerank] 第 {attempt} 次调用失败：{data}")
            continue
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            print(f"[visual_rerank] 响应结构异常（第 {attempt} 次）")
            continue
        if isinstance(text, list):  # 容忍块数组形态（与 model.py:98 同一防御）
            text = " ".join(str(c.get("text", "")) for c in text if isinstance(c, dict))
        if os.environ.get("VISUAL_RERANK_DEBUG"):
            print(f"[visual_rerank][debug] raw#{attempt} ({time.time() - t_call:.1f}s): "
                  f"{text[:500]!r}")

        if mode == "verify":
            # 解析 [{"idx": n, "evidence": "..."}, ...]；evidence 允许缺省（视为无证据）
            pairs = re.findall(
                r'"idx"\s*:\s*(\d+)\s*,\s*"evidence"\s*:\s*"((?:[^"\\]|\\.)*)"', text
            )
            if not pairs:
                m = re.search(r'"hits"\s*:\s*\[\s*\]', text)
                if m:
                    return []  # 显式空数组 = 明确的「都不命中」，合法答案
                print(f"[visual_rerank] 未解析到 hits（第 {attempt} 次）：{text[:160]!r}")
                continue
            out, seen = [], set()
            for idx_s, ev in pairs:
                v = int(idx_s)
                # 越界图号直接丢弃（模型编号错乱时不能把无关图顶上 rank1）
                if 1 <= v <= n and v not in seen:
                    seen.add(v)
                    out.append((v, _unescape(ev)))
            return out

        m = re.search(r'"best"\s*:\s*(\d+)', text)
        if not m:
            print(f"[visual_rerank] 未解析到 best（第 {attempt} 次）：{text[:160]!r}")
            continue
        best = int(m.group(1))
        if 0 <= best <= n:
            return [] if best == 0 else [(best, "")]
        print(f"[visual_rerank] best={best} 超出 0..{n}（第 {attempt} 次）")
    return None


def visual_rerank(question, docs, top_n=None, sources=None):
    """B1：用视觉复核把「真正在讲这个问题」的图提到 rank 1。返回新的 Document 列表。

    行为：
      - `VISUAL_RERANK_ENABLED=false` → 原样返回（零开销、零行为变化）。
      - 合并候选里的图像块 < 2 个 → 原样返回（没有可歧义的对手，问了也白问）。
      - VLM 命中某张图 → `mode="verify"/"promote"` 把它移到 rank 1，其余保持原相对顺序；
        `mode="slot"` 只让图像块互相换位（保守模式，实测在 top-8 下常因图太少而失效）。
      - VLM 给出空 hits（都不命中）/ 任何失败 → 原样返回。

    top_n：最终结果条数。**本函数负责截断**——因为开启视觉重排时上游会保留更宽的候选池
    （否则文本 reranker 会先把图像块截掉，B1 永远看不到图）。
    """
    limit = top_n or config.RERANKER_TOP_N

    def _plain():
        return docs[:limit] if docs else docs

    if not config.VISUAL_RERANK_ENABLED or not docs:
        return _plain()
    if not config.VISUAL_RERANK_API_KEY:
        print("[visual_rerank] 未配置 VISUAL_RERANK_API_KEY，跳过视觉重排")
        return _plain()

    merged = _merge_candidates(question, docs, sources)
    image_docs = _collect_image_docs(merged)
    if len(image_docs) < 2:
        return _plain()

    image_docs = image_docs[: config.VISUAL_RERANK_MAX_IMAGES]
    t0 = time.time()
    hits = _ask_vlm(question, image_docs, config.VISUAL_RERANK_TIMEOUT,
                    config.VISUAL_RERANK_MODE)
    if hits is None:
        print(f"[visual_rerank] 视觉重排失败，保持原顺序（{time.time() - t0:.1f}s）")
        return _plain()
    if not hits:
        # 校准后的「全部不命中」——保持文本链路顺序。这是防污染的关键分支：
        # 实测（B1 标定实验，已归档）用「哪张最相关」问法时 VLM 从不弃权，5/6 道召回失败的题
        # 都被硬塞了一张无关图到 rank 1。
        print(f"[visual_rerank] VLM 判定 {len(image_docs)} 张候选图无一命中，保持原顺序"
              f"（{time.time() - t0:.1f}s）")
        return _plain()

    # 证据门：只采信「抄出的图上文字真的含问题实义词」的命中，按 VLM 给的名次取第一个。
    #
    # B3 之后追加一条放行条件：**该图自己的 caption 通过证据门也算**。
    # 依据（两条实测，都不是猜的）：
    #   1. B3 换描述器后 caption 已是逐字转写，B1 门控探针（已归档）的离线对照矩阵用的
    #      就是「caption 当证据」：真命中 **6/6 通过**，3 张干扰图 × 6 题 = **18/18 拒掉**。
    #      即这条路径的精确率已被测量过，不是把门拆了。
    #   2. VLM 现抄的证据会出错，且两次都是**真命中被误拒**：
    #      「个人信息」被抄成「个人消息」（OCR 错字）、「比较数字大小」干脆没抄根节点。
    #      两题的正确答案图 caption 里都逐字写着那个专有名词 —— 采信 caption 更可靠。
    accepted = None
    for idx, evidence in hits:
        caption = image_docs[idx - 1][1].page_content if 1 <= idx <= len(image_docs) else ""
        if (config.VISUAL_RERANK_MODE != "verify"
                or _evidence_supports(question, evidence)
                or _evidence_supports(question, caption)):
            accepted = (idx, evidence)
            break
        print(f"[visual_rerank] 丢弃第 {idx} 张的命中：证据与问题实义词无交集 "
              f"（evidence={evidence[:80]!r}）")
    if accepted is None:
        print(f"[visual_rerank] {len(hits)} 个命中全部未通过证据门，保持原顺序"
              f"（{time.time() - t0:.1f}s）")
        return _plain()

    best, evidence = accepted
    winner = image_docs[best - 1][1]
    winner.metadata["visual_rank"] = 1
    if evidence:
        winner.metadata["visual_evidence"] = evidence[:200]

    if config.VISUAL_RERANK_MODE == "promote" or config.VISUAL_RERANK_MODE == "verify":
        out = [winner] + [d for d in merged if d is not winner]
        pos = 1
    else:  # slot：只在图像块原有的槽位之间换位，不越过文本块
        order = list(range(len(image_docs)))
        order.remove(best - 1)
        order.insert(0, best - 1)
        reordered = [image_docs[i] for i in order]
        slots = [idx for idx, _, _ in image_docs]
        out = list(merged)
        for slot, (_, doc, _) in zip(slots, reordered):
            out[slot] = doc
        pos = slots[0] + 1

    print(f"[visual_rerank] {len(image_docs)} 张候选图 → 命中第 {best} 张，"
          f"置顶到 rank {pos}（{time.time() - t0:.1f}s）："
          f"{Path(winner.metadata.get('image_path', '?')).name}")
    return out[:limit]
