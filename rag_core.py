"""
rag_core.py — RAG 核心（W1 三件套 + W2 重排增强 + 引用溯源封装）

2026-08-18 重构：彻底移除 langchain 全家桶，改为直接依赖：
  - 嵌入 / 生成：ollama 官方 SDK（ollama.Client）
  - 向量库：chromadb 原生客户端（PersistentClient）
  - 切分：自实现 RecursiveCharacterTextSplitter（含中文标点分隔符 + start_index）
  - 文档加载：pypdf / python-docx / 原生读文件（不再经 langchain_community loaders）
  - 链：手工拼 prompt + ollama.chat，不再用 LCEL

为什么要去 langchain（见交接文档）：
  langchain 只是薄封装，却带来『全家桶版本必须一致』的脆弱性
  （is_data_content_block 等符号冲突），并间接 import huggingface_hub 制造
  HF_ENDPOINT 时序坑。去掉后依赖从十几个包瘦成 ollama + chromadb + sentence-transformers，
  可复现性直接上一个台阶，也更贴『自己实现关键组件』的叙事。

对外接口保持稳定（app.py / eval_local.py 无需改动）：
  load_and_chunk_documents / get_vector_store / retrieve_vector /
  retrieve_and_rerank（P1-1 起内部支持 hybrid 第一段）/ hybrid_retrieve（向量+BM25+RRF）/
  build_rag_chain（返回 {'chain','retrieve_and_rerank'}）/ ollama_chat（供 eval_local 当裁判复用）
  RagChain.invoke(question, docs=None)：docs 可不传（自行检索）或由调用方传入以复用同一份来源
  P1-4 增量索引：chunk_file(path) / add_documents(chunks)（稳定 id 增量 upsert 去重）/
  delete_by_source(source)（按 source 删单文档）。get_vector_store(chunks) 仍保留为全量重建入口。
"""
import os
import math
import re
import hashlib
from pathlib import Path

# 纯检索算法（零依赖）抽到 rag_search_utils，便于独立单测（见 test_core.py）；
# 这里重新导出，对外接口 / 内部调用（hybrid_retrieve、add_documents）均不变。
from rag_search_utils import (
    _tokenize,
    _BM25,
    _bm25_rank,
    _rrf_fuse,
    _stable_ids,
    SOURCE_RULES,
    route_sources,
    is_enum_query,
)
import rag_search_utils

import ollama
import chromadb

import config
import llm_provider
from reranker import rerank_documents
import grounding  # P1 接地自检（grounding 仅依赖 config/ollama，无循环依赖）
import semantic_cache  # 统一问答入口的缓存查询/写入（仅依赖 config/ollama，无循环依赖）
import visual_rerank as _visual_rerank  # B1 视觉重排（仅依赖 config，无循环依赖）

if config.APP_PROFILE == "paper":
    # 课程 SOURCE_RULES 不用于论文库；SOURCE_RULES 本身保留给 course/test_core
    rag_search_utils.ACTIVE_SOURCE_RULES = []


# ====================== 0. 轻量客户端 / Document ======================
# ollama 0.6.2 的模块级函数不接收 base_url，统一用 Client 实例
_client = ollama.Client(host=config.OLLAMA_BASE_URL)


class Document:
    """最小文档对象：只保留 page_content + metadata，替代 langchain_core.documents.Document。
    reranker / app / eval_local 都只读写这两个属性，故完全兼容。"""

    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


def ollama_chat(messages, temperature=None):
    """统一生成入口，返回模型输出的文本字符串。

    函数名保留为 ollama_chat：test_api.py 四处 monkeypatch 钉的是这个名字，
    改名会连带动测试；默认 provider=ollama 时行为与旧版一致。
    C3（2026-10-15）起按 config.LLM_PROVIDER 分发：ollama=本地，openai=云端。
    """
    return llm_provider.chat(messages, temperature=temperature)


def ollama_chat_stream(messages, temperature=None):
    """ollama_chat 的流式版：逐片段 yield 生成的文本。

    默认 provider=ollama 时仍是 ollama stream=True，片段内容与非流式一致，
    只是分块到达；provider=openai 时解析 OpenAI 兼容 SSE 的 delta.content。
    调用方负责把片段聚合回完整答案（缓存/日志用）。
    """
    yield from llm_provider.chat_stream(messages, temperature=temperature)


# ====================== 1. 加载 + 切分 ======================
_SEPARATORS = ["\n\n", "\n", "。", "；", "！", "？", "，", "、", " ", ""]


def _extract_pdf_text(path: Path):
    """PDF 文字层提取：优先 PyMuPDF，回退 pypdf。

    G2（2026-08-28）：实测《微信小程序开发实战》303 页 pypdf 提取中文全为 CID
    乱码（正文不可读、向量通道失效），而 PyMuPDF 对同一 PDF 296/303 页正常提取。
    根因是 CID→Unicode 映射（ToUnicode CMap），非扫描版——换提取器即可免费修复，
    无需 OCR。PyMuPDF 缺失/提取过低时回退 pypdf（保持原扫描版检测行为）。
    返回 (text, n_pages)；text 为空字符串表示无文字层。
    """
    try:
        import pymupdf  # 1.28+ 官方命名
    except ImportError:
        try:
            import fitz as pymupdf  # 旧命名
        except ImportError:
            pymupdf = None
    if pymupdf is not None:
        try:
            doc = pymupdf.open(str(path))
            pages = [p.get_text() or "" for p in doc]
            avg = sum(len(t.strip()) for t in pages) / max(len(pages), 1)
            if avg >= config.PDF_MIN_CHARS_PER_PAGE:
                return "\n".join(pages), len(pages)
            # PyMuPDF 也提不出 → 真扫描/伪文字层，落到 pypdf 再确认一次
        except Exception:
            pass
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    return "\n".join(pages), len(pages)


def _load_text(path: Path):
    """按后缀加载单文件为纯文本；不支持或读取失败返回 None（跳过不中断建库）。

    T6（2026-08-22）：PDF 加载后做文字层检测——扫描版 PDF（无文字层）此前会
    静默返回空内容、对检索零贡献且无提示（data/教材 实测 7/7 全中）。现在均字符
    低于 config.PDF_MIN_CHARS_PER_PAGE 时告警并跳过；另新增 .pptx 课件支持。
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            text, n_pages = _extract_pdf_text(path)
            avg = (len(text.strip()) / n_pages) if n_pages else 0
            if avg < config.PDF_MIN_CHARS_PER_PAGE:
                print(
                    f"[warn] 跳过疑似扫描版/无文字层 PDF: {path.name} "
                    f"(均 {avg:.0f} 字符/页，阈值 {config.PDF_MIN_CHARS_PER_PAGE})"
                )
                return None
            return text
        if suffix == ".pptx":
            # T6: PPTX 课件支持——zipfile 读 slide XML 提取 <a:t> 文本，按页拼接
            import zipfile
            import re as _re
            with zipfile.ZipFile(str(path)) as z:
                slides = sorted(
                    (x for x in z.namelist() if _re.match(r"ppt/slides/slide\d+\.xml$", x)),
                    key=lambda x: int(_re.findall(r"\d+", x)[0]),
                )
                parts = []
                for s in slides:
                    xml = z.read(s).decode("utf-8", "ignore")
                    texts = [t.strip() for t in _re.findall(r"<a:t>([^<]*)</a:t>", xml) if t.strip()]
                    parts.append("\n".join(texts))
                text = "\n\n".join(parts)
                if len(text.strip()) < config.PDF_MIN_CHARS_PER_PAGE:
                    print(f"[warn] 跳过无文字内容的 PPTX: {path.name}")
                    return None
                return text
        if suffix in (".txt", ".md"):
            return path.read_text(encoding="utf-8")
        if suffix == ".docx":
            import docx
            d = docx.Document(str(path))
            return "\n".join(p.text for p in d.paragraphs if p.text.strip())
    except Exception as e:
        print(f"[warn] 跳过无法读取的 {path}: {e}")
    return None


def _split_recursive(text, chunk_size, separators):
    """纯递归分割（不合并）：返回按分隔符切开的文本片列表。

    与 _split_text 的区别：这里只切不 merge。修复背景（B 专项）：
    旧实现每层递归都调 _merge_splits，递归返回的『已合并块』又被父级当 split 再次合并，
    overlap 逐层叠加 → 同一文本块在单个 chunk 内重复（Q21 软链接块重复两遍即此所致），
    污染嵌入、破坏检索。现分割与合并彻底分离：分割阶段纯切，合并只在顶层做一次。
    """
    if text == "":
        return []

    # 选最粗的可行分隔符（优先段落/换行，再中文标点，最终回退到逐字）
    separator = separators[-1]
    for s in separators:
        if s == "":
            separator = s
            break
        if s in text:
            separator = s
            break

    if separator == "":
        splits = list(text)
    else:
        raw = text.split(separator)
        # 把分隔符重新贴回除最后一段外的每段，保持原文连续性
        splits = [s + separator for s in raw[:-1]] + [raw[-1]]

    good = []
    for s in splits:
        if s == "":
            continue
        if len(s) < chunk_size:
            good.append(s)
        else:
            if len(separators) > 1:
                good.extend(_split_recursive(s, chunk_size, separators[1:]))
            else:
                good.append(s)
    return good


def _split_text(text, chunk_size, chunk_overlap, separators):
    """递归切分（模仿 RecursiveCharacterTextSplitter）：分割后只在顶层一次性合并。

    返回 [(chunk_text, start_index)]，start_index 为块在 text 中的精确字符起点。
    2026-09-03：原实现返回纯文本列表、由 _split_with_index 用 text.find(ch, cursor)
    反推起点——overlap 使第 k 块的真实起点 = 上块尾 - chunk_overlap < cursor，
    find 必失败、fallback 到 cursor，start_index 每块累积偏晚 chunk_overlap（pdf
    等无标题单节长文档实测第 k 块偏晚 (k-1)×150）。改由合并阶段字符游标精确定位。
    """
    return _merge_splits(_split_recursive(text, chunk_size, separators), chunk_size, chunk_overlap)


def _merge_splits(splits, chunk_size, chunk_overlap):
    """按 chunk_size 合并，并在块间保留 chunk_overlap 字符重叠（避免切断句子）。

    返回 [(chunk_text, start_index)]。splits 为 text 的连续切分（全部拼接 = 原 text），
    合并以字符游标精确记录每块起点；overlap 转移时下块起点 = 上块结束 - chunk_overlap。
    """
    docs = []
    current = ""
    current_start = 0
    for piece in splits:
        if current and len(current) + len(piece) > chunk_size:
            end = current_start + len(current)
            docs.append((current, current_start))
            if chunk_overlap > 0 and len(current) > chunk_overlap:
                current = current[-chunk_overlap:]
                current_start = end - chunk_overlap
            else:
                current = ""
                current_start = end
        current += piece
    if current:
        docs.append((current, current_start))
    return docs


def _split_with_index(text, chunk_size, chunk_overlap, separators):
    """切分并记录每块在原文中的偏移（start_index），供引用溯源定位。"""
    return _split_text(text, chunk_size, chunk_overlap, separators)


# ====================== 1.5 结构感知切分（T8，2026-08-22）======================
# 目标：代码块 / markdown 表格整体保护（不被 500 字符边界切断）；
#       按标题（# ~ ######）分层，每块 metadata 带 chapter（所属章节标题）。
# 实现：先把代码块/表格替换为哨兵串，切分后再还原；标题行作为节边界。
import re as _re


def _protect_blocks(text):
    """代码块(```...```)与连续表格行整体替换为哨兵。

    返回 (替换后文本, {哨兵: 原文本}, 偏移重映射表)。
    评审 P2-1（2026-08-30）：哨兵 §B0§（4 字符）远短于原代码块/表格，替换后所有
    后续偏移进入「保护后坐标系」；若不重映射，切分产出的 start_index 对含代码块/
    表格的文档整体漂移（引用溯源「原文偏移」定位不准）。segments 记录每个哨兵在
    保护后文本中的 (起点, 哨兵长, 原文长)，供 _remap_offset 换算回原文偏移。
    """
    blocks = {}
    segments = []  # [(protected_start, protected_len, original_len), ...]

    def _repl(m):
        key = f"§B{len(blocks)}§"
        blocks[key] = m.group(0)
        return key

    text = _re.sub(r"```.*?```", _repl, text, flags=_re.S)
    # 连续以 | 开头的行（markdown 表格）整体保护
    text = _re.sub(r"(?:^\|.*\|\s*$(\n|$))+", _repl, text, flags=_re.M)

    # 2026-09-03 修复：re.sub 回调里的 m.start() 是「本轮输入字符串」的坐标，不是最终
    # protected 文本坐标——前序代码块/表格被替换缩短后，后续条目的起点坐标整体偏大
    # （实测 §B1§ 记录 491、实际 407，差=前一代码块缩短量 84）。remap 拿偏大的坐标与
    # 保护后 offset 比较会提前 return、漏补后续哨兵差 → start_index 偏前。
    # 改为替换完成后在最终文本上扫描哨兵，坐标即最终坐标系；o_len 从 blocks 原文本取。
    for m in _re.finditer(r"§B\d+§", text):
        key = m.group(0)
        segments.append((m.start(), len(key), len(blocks[key])))
    segments.sort()  # 代码块/表格分两次 sub 替换，protected_start 需统一排序
    return text, blocks, segments


def _remap_offset(offset, segments):
    """把「保护后坐标系」的偏移换算回原文偏移（评审 P2-1）。

    segments 按 protected_start 升序。偏移落在哨兵区间内时映射到块首（原文中该
    代码块/表格的起点）——哨兵是原子、正常切分不会产生块中间的偏移；落在哨兵
    之后则累加前面所有哨兵的长度差（original_len - protected_len）。"""
    shift = 0
    for p_start, p_len, o_len in segments:
        if offset < p_start:
            return offset + shift
        if offset < p_start + p_len:
            return p_start + shift
        shift += o_len - p_len
    return offset + shift


def _restore_blocks(text, blocks):
    for k, v in blocks.items():
        text = text.replace(k, v)
    # 2026-09-03：chunk_overlap 是字符级截断，若窗口落在哨兵中间会把 §B378§ 截成
    # 残缺片段（§B378 / 378§），上方完整 replace 匹配不上 → 残留数字噪声进检索内容。
    # 残缺哨兵无可还原内容（原代码块/表格已由含完整哨兵的相邻块还原），直接删标记。
    text = _re.sub(r"§B\d+", "", text)   # 缺尾 §（overlap 截断最常见：块尾哨兵头保留）
    text = _re.sub(r"(?<![§B])\d+§", "", text)  # 缺头 §B（块首哨兵尾保留，如 "378§"）
    return text


def _split_structured(text, chunk_size, chunk_overlap, separators):
    """T8：结构感知切分。返回 [(chunk, start_index, chapter_or_None)]。
    - 代码块/表格整体保护（哨兵替换→切分→还原，永不切断）
    - 按标题分层：标题行作为节边界，块不跨章节；每节内走原 _split_with_index
    - 无标题文档退化为单节（chapter=None），行为与旧版一致
    - start_index 为「原文偏移」：切分在哨兵保护后的文本上进行，还原块内容时
      同步经 _remap_offset 把偏移换算回原文坐标系（评审 P2-1），保证引用溯源
      的原文定位对含代码块/表格的文档同样准确
    """
    protected, blocks, segments = _protect_blocks(text)
    lines = protected.split("\n")
    sections = []  # (chapter, section_start_offset, section_text)
    cur_title, cur_lines, cur_start = None, [], 0
    offset = 0
    for ln in lines:
        m = _re.match(r"^#{1,6}\s+(.+?)\s*$", ln.strip())
        if m:
            if cur_lines:
                sections.append((cur_title, cur_start, "\n".join(cur_lines)))
            cur_title = m.group(1).strip()
            cur_lines = [ln]
            cur_start = offset
        else:
            cur_lines.append(ln)
        offset += len(ln) + 1
    if cur_lines:
        sections.append((cur_title, cur_start, "\n".join(cur_lines)))

    out = []
    for chapter, sec_start, sec_text in sections:
        if not sec_text.strip():
            continue
        for chunk, rel_start in _split_with_index(sec_text, chunk_size, chunk_overlap, separators):
            # 评审 P2-1：sec_start/rel_start 是保护后坐标系的偏移，还原块内容的同时
            # 把 start_index 重映射回原文偏移（否则含代码块/表格的文档整体漂移）
            out.append((_restore_blocks(chunk, blocks),
                        _remap_offset(sec_start + rel_start, segments),
                        chapter))
    return out


# ====================== 1.6 乱码清洗（2026-08-22）======================
# 微信小程序教材 PDF 的部分页面（章节小结/图表/课后练习）存在 CID 字体编码乱码
# （pypdf 提取出 阿拉伯/天城文/控制字符 等无意义字符，如 "֥ؓ ஥ഺb"）。
# 诊断：303 页 372 块中 29 块 >30% 乱码、251 块 >10%（广泛混入）。
# 处理：乱码占比 > GARBLED_DROP_THRESHOLD → 整块丢弃；否则剥离乱码字符保留正文。
# 注：剥离改变文本长度，start_index 仍保留原文偏移（引用定位容忍少量偏差）。

def _is_good_char(ch):
    code = ord(ch)
    # 2026-08-22 修复：原白名单含 0x00A0-0x00FF（拉丁补充区），本意是放行西欧字母；
    # 但 pypdf 对 CID 乱码的另一种映射就是产出 æ/ç/è 这类拉丁字符（实测 698 块中 54 块残留），
    # 被当作"好字符"放行 → 污染嵌入。修复：拉丁补充区不再整段放行，
    # 仅保留中文教材真实会用到的几个符号（· × ÷ ° 等）；真实西欧词句极少出现，损失可忽略。
    return (
        (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF)  # CJK 基本+扩展A
        or (0x3000 <= code <= 0x303F) or (0xFF00 <= code <= 0xFFEF)  # CJK/全角标点
        or (0x20 <= code <= 0x7E) or (0x2010 <= code <= 0x206F)  # ASCII/通用标点
        or (0x00B7 == code or 0x00D7 == code or 0x00F7 == code or 0x00B0 == code)  # ·×÷°（中文教材常用）
        or (0x2018 <= code <= 0x201F)  # 引号
        or (0x2190 <= code <= 0x22FF)  # 箭头/数学运算符（教材公式）
    )


def clean_garbled_chunk(text):
    """乱码清洗：返回清洗后文本；返回 None 表示整块丢弃（乱码占比超阈值）。"""
    if not text:
        return text
    total = bad = 0
    kept = []
    for ch in text:
        if ch.isspace():
            kept.append(ch)
            continue
        total += 1
        if _is_good_char(ch):
            kept.append(ch)
        else:
            bad += 1
    ratio = bad / total if total else 0.0
    if ratio > config.GARBLED_DROP_THRESHOLD:
        return None
    return "".join(kept)


def load_and_chunk_documents(directory=None):
    if config.APP_PROFILE == "paper":
        try:
            from paper_index import load_paper_chunks
        except ImportError:
            print("[warn] paper profile 尚未接入 paper_index（M2 前）；返回空块")
            return []
        return load_paper_chunks()
    # 支持 PDF / DOCX / TXT / MD；TXT/MD 强制 utf-8（Windows 中文坑，直接写死避免再踩）
    directory = Path(directory or config.DATA_DIR)
    documents = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in (".pdf", ".txt", ".md", ".docx", ".pptx"):
            continue
        text = _load_text(path)
        if not text or not text.strip():
            continue
        for chunk, start, chapter in _split_structured(
            text, config.CHUNK_SIZE, config.CHUNK_OVERLAP, _SEPARATORS
        ):
            cleaned = clean_garbled_chunk(chunk)  # 乱码清洗：None=丢弃
            if cleaned is None:
                continue
            meta = {"source": str(path), "start_index": start}
            if chapter:
                meta["chapter"] = chapter  # T8: 所属章节标题（md 有标题时）
            documents.append(Document(page_content=cleaned, metadata=meta))
    print(f"Loaded and chunked {len(documents)} pieces from documents.")
    # P2 多模态（2026-09-05）：图像 caption 块并入重建链路——所有重建路径
    # （首次自动建库 / 侧边栏全量重建 / API /index/rebuild）自动带上图像块。
    # 增强项语义：图像清单缺失/损坏只 warn，绝不阻断文本建库。
    try:
        documents.extend(load_image_chunks())
    except Exception as e:
        print(f"[warn] 图像块加载失败（忽略，仅文本建库）: {e!r}")
    return documents


def load_image_chunks(images_dir=None):
    """P2 多模态：读 data/images/_manifest.csv + _image_captions.csv，图像描述转 Document。

    - page_content = caption 文本（检索/BM25/精排全链路与文本块同构——caption 路线的核心：
      图像以文本代理身份进统一索引）。caption 由哪个描述器生成取决于
      `_caption_images_v2.py` 那一版的产物，**不是本函数决定的**：
      换描述器（B3）之前是本地 qwen2.5vl:7b 的 2~3 句泛描述；
      之后是视觉模型逐字转写 + 一句概括。两者用 `config.IMAGE_CAPTIONS_CSV` 切换。
      实测提醒：逐字转写让 caption 从 ~110 字涨到 ~350 字，长文本会稀释向量相似度
      （见 B3 报告），所以「内容更全」不等于「召回更好」。
    - metadata 与文本块同规律：
      source = 教材 PDF 相对路径（与文本块完全一致 → P1a 教材过滤 / 按源删除天然覆盖图像块）
      start_index = -(页码*1000+图序)（稳定 id 永不与文本块冲突，文本块 start_index ≥ 0）
      type='image'、image_path=相对项目根的 PNG 路径（UI 引用区渲染原图用）、page=页码
    """
    import csv as _csv
    images_dir = Path(images_dir) if images_dir else Path(config.DATA_DIR) / "images"
    manifest = images_dir / "_manifest.csv"
    # B3：caption 文件名可配（IMAGE_CAPTIONS_CSV）。默认用换描述器后的生产文件；
    # 指向 _image_captions_v1_qwen25vl.csv 即可复跑换描述器之前的基线。
    captions = images_dir / config.IMAGE_CAPTIONS_CSV
    if not manifest.exists() or not captions.exists():
        return []
    cap_map = {}
    with open(captions, encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            cap = (r.get("caption") or "").strip()
            if cap and not cap.startswith("[CAPTION_FAILED]"):
                cap_map[r["png_path"]] = cap

    # B3 修复：嵌入/BM25 用的短文本另取一份 caption（判别力更强的 v1 描述）。
    # 取不到就回退到本 caption 的摘要（'图中文字：' 之前），再回退全文。
    embed_map = {}
    embed_csv = getattr(config, "IMAGE_EMBED_CAPTIONS_CSV", "") or ""
    if embed_csv and embed_csv != config.IMAGE_CAPTIONS_CSV:
        embed_path = images_dir / embed_csv
        if embed_path.exists():
            with open(embed_path, encoding="utf-8-sig") as f:
                for r in _csv.DictReader(f):
                    t = (r.get("caption") or "").strip()
                    if t and not t.startswith("[CAPTION_FAILED]"):
                        embed_map[r["png_path"]] = t
    docs = []
    with open(manifest, encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            cap = cap_map.get(r["png_path"])
            if not cap:
                continue
            page_i = int(r["page"])
            gi = int(r["png_path"].rsplit("_i", 1)[1].split(".")[0])
            # B3 修复（2026-10-16）：拆「嵌入文本」与「生成文本」。
            # 背景：逐字转写让 caption 中位 264 字 / 最长 1971 字，向量被原始文字稀释，
            # 图片命中率从 8/8 掉到 7/8（MISS = 微信小程序开发实战/p020）。
            # 做法：'图中文字：' 之前的摘要（中位 82 字）作为嵌入/BM25 的短文本；
            # 全量 caption 存 metadata["full_content"]，检索还原时再取回，供生成与引用使用。
            # 全部 215 条 caption 都含该分隔符（实测），取不到时回退原文，保证不丢内容。
            embed_text = (embed_map.get(r["png_path"])
                          or cap.split("图中文字：", 1)[0].strip() or cap)
            docs.append(
                Document(
                    # page_content 保持全文：B1 词面通道依赖逐字转写里的判别 token
                    # （tab1.wxml / p167 等），精排与生成也需要完整信息。
                    page_content=cap,
                    metadata={
                        "source": str(Path(config.DATA_DIR) / "教材" / f"{r['book']}.pdf"),
                        "start_index": -(page_i * 1000 + gi),
                        "type": "image",
                        "image_path": r["png_path"],
                        "page": page_i,
                        # 仅向量嵌入用短文本（v1 描述 / 摘要）；其余链路一律用 page_content
                        "embed_text": embed_text,
                    },
                )
            )
    print(f"Loaded {len(docs)} image-caption chunks.")
    return docs


# ====================== 2. 嵌入 + 向量库（带缓存） ======================
class _OllamaEmbeddingFunction:
    """chromadb 1.5.x 兼容的嵌入函数：封装 ollama 的 nomic-embed-text。
    新式接口要求 __call__(self, input) 接受文本列表并返回 embedding 列表。"""

    def __init__(self, client, model):
        self._client = client
        self._model = model
        self._batch = 32  # 分批嵌入，避免大批量触发 Ollama llama-server 动态端口失效

    def name(self):
        # chromadb 1.5.x 要求嵌入函数提供 name()，用于持久化配置冲突校验
        return "ollama"

    def _embed_batch(self, batch):
        """单批嵌入 + 重试（Ollama 0.32.9 的 llama-server 动态端口在长任务中可能失效，
        11434 会 307 重定向到新端口；失败重试走新端口即可恢复）。"""
        import time as _time
        last_err = None
        for attempt in range(4):
            try:
                resp = self._client.embed(model=self._model, input=batch)
                return resp["embeddings"]
            except Exception as e:
                last_err = e
                _time.sleep(1.5 * (attempt + 1))
        raise last_err

    def __call__(self, input):
        # 建库时 chromadb 用此嵌入文档（input 为文本列表）；分批防止长任务中断
        items = list(input) if isinstance(input, (list, tuple)) else [input]
        out = []
        for i in range(0, len(items), self._batch):
            out.extend(self._embed_batch(items[i : i + self._batch]))
        return out

    def embed_query(self, input):
        # 查询时 chromadb 1.5.x 单独调 embed_query（input 为查询文本列表，非单字符串）
        # 直接复用 __call__（批量 client.embed 对列表/字符串都适用）
        return self(input)


def get_embeddings():
    # 统一实例化嵌入函数，建库与查询共用（维度一致）
    return _OllamaEmbeddingFunction(_client, config.EMBEDDING_MODEL)


_COLLECTION_NAME = config.COLLECTION_NAME  # profile 决定：course_rag / paper_rag
_chroma_client = None
_vector_store = None


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)
    return _chroma_client


def _assert_safe_persist_dir(persist_dir):
    """rmtree 前守卫：禁止删仓库根/语料目录/另一 profile 的索引目录。"""
    resolved = Path(persist_dir).resolve()
    repo_root = Path(__file__).resolve().parent
    forbidden = {
        repo_root,
        (repo_root / "data").resolve(),
        (repo_root / "data" / "教材").resolve(),
        (repo_root / "data" / "papers" / "raw").resolve(),
    }
    if resolved in forbidden:
        raise RuntimeError(f"拒绝删除危险路径: {resolved}")
    course_dirs = {
        Path(os.getenv("PERSIST_DIRECTORY", "./chroma_db")).resolve(),
        (repo_root / "chroma_db").resolve(),
        (repo_root / "chroma_db_m3").resolve(),
    }
    paper_dirs = {
        Path(os.getenv("PAPER_PERSIST_DIRECTORY", "./chroma_db_papers")).resolve(),
        (repo_root / "chroma_db_papers").resolve(),
    }
    if config.APP_PROFILE == "paper" and resolved in course_dirs:
        raise RuntimeError(f"paper profile 拒绝删除课程索引目录: {resolved}")
    if config.APP_PROFILE == "course" and resolved in paper_dirs:
        raise RuntimeError(f"course profile 拒绝删除论文索引目录: {resolved}")


def get_vector_store(chunks=None, auto_rebuild=True):
    """chunks 不为 None → 重建并落盘；为 None → 懒加载已有库（问答阶段用）。

    auto_rebuild（评审 P2-3，2026-08-30）：库为空且 data/ 有资料时是否自动全量重建。
    默认 True（懒加载 / 启动预热 / 上传建库路径保持「免手动点重建」的既有行为）；
    **只读端点必须传 False**（GET /health、GET /index/stats、GET /documents）——
    只读操作不应有「顺手建索引」的副作用（此前空库 + Ollama 就绪时 GET /health
    会同步触发数分钟的全量重建，且该副作用对调用方完全不可见）。
    """
    global _vector_store
    ef = get_embeddings()
    client = _get_chroma_client()

    if chunks is not None:
        # 全量重建：先删旧 collection 再建，避免删文件后旧文档残留（原 langchain 版有此隐患）
        try:
            client.delete_collection(_COLLECTION_NAME)
        except Exception:
            pass
        col = client.get_or_create_collection(_COLLECTION_NAME, embedding_function=ef)
        # 2026-08-22 修复（系统测试 P1）：chunks 为空时 chromadb 的 upsert 会抛
        # "Non-empty lists are required" —— data/ 空目录重建索引时直接崩溃。
        # 空库也应返回有效集合句柄（count()==0），问答/健康检查照常工作。
        if chunks:
            # 稳定 id 统一由 _stable_ids 生成：论文块优先 chunk_uid，课程块 md5(source::start_index)
            ids = _stable_ids(chunks)
            # B3 修复（2026-10-16）：向量嵌入与正文分离。
            # documents 存全文（BM25/精排/生成/B1 词面通道都用它），
            # 但 embeddings 由 metadata["embed_text"]（短描述）算出——避免逐字转写稀释向量。
            emb_texts = [d.metadata.get("embed_text") or d.page_content for d in chunks]
            col.upsert(
                ids=ids,
                documents=[d.page_content for d in chunks],
                metadatas=[d.metadata for d in chunks],
                embeddings=ef(emb_texts),
            )
        _vector_store = col
        _invalidate_all_chunks()  # 全量重建后 BM25 通道缓存强制刷新
        print(f"Vector store created and saved ({len(chunks)} chunks).")
        return _vector_store

    if _vector_store is None:
        # 加载已有库；若旧 chroma 格式不兼容导致失败，则清空目录后重建（本地缓存可一键重建）
        try:
            _vector_store = client.get_or_create_collection(
                _COLLECTION_NAME, embedding_function=ef
            )
            # 触发一次轻量访问，确认集合可正常打开（旧格式异常会在 query 前暴露）
            _vector_store.count()
        except Exception as e:
            print(f"[warn] 已有向量库无法加载（{e}），将清空并重建：{config.PERSIST_DIRECTORY}")
            import shutil
            _assert_safe_persist_dir(config.PERSIST_DIRECTORY)
            shutil.rmtree(config.PERSIST_DIRECTORY, ignore_errors=True)
            _chroma_client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)
            _vector_store = _chroma_client.get_or_create_collection(
                _COLLECTION_NAME, embedding_function=ef
            )
        # 集合为空但本地有资料 → 自动重建（切换集合名 / 首次部署后免手动点「重建索引」）；
        # 只读端点（auto_rebuild=False）跳过，空库如实返回 count()==0
        if _vector_store.count() == 0 and auto_rebuild:
            chunks = load_and_chunk_documents()
            if chunks:
                return get_vector_store(chunks)
        print("Loaded existing vector store.")
    return _vector_store


# ====================== 2.1 增量索引 + 去重（P1-4） ======================
def chunk_file(path):
    """切分单个文件为 Document 列表（P1-4 增量索引用；metadata 同 load_and_chunk_documents）。"""
    if config.APP_PROFILE == "paper":
        # paper profile 的增量入口走 paper_index/paper_ingest；M2 接入
        print("[warn] paper profile 不支持 chunk_file 课程路径；忽略")
        return []
    path = Path(path)
    text = _load_text(path)
    if not text or not text.strip():
        return []
    docs = []
    for chunk, start, chapter in _split_structured(
        text, config.CHUNK_SIZE, config.CHUNK_OVERLAP, _SEPARATORS
    ):
        cleaned = clean_garbled_chunk(chunk)  # 乱码清洗：None=丢弃
        if cleaned is None:
            continue
        meta = {"source": str(path), "start_index": start}
        if chapter:
            meta["chapter"] = chapter
        docs.append(Document(page_content=cleaned, metadata=meta))
    return docs


def add_documents(new_chunks):
    """
    P1-4 增量 upsert：只插入/覆盖给定块，不删除已有库。
    稳定 id = md5(source::start_index)，因此重复上传同一文件 → 同 id 覆盖，不产生重复块。
    返回实际 upsert 的块数。
    """
    if not new_chunks:
        return 0
    ef = get_embeddings()
    client = _get_chroma_client()
    col = client.get_or_create_collection(_COLLECTION_NAME, embedding_function=ef)
    ids = _stable_ids(new_chunks)
    # 与全量重建同口径：向量用 embed_text（若有），正文仍是 page_content
    emb_texts = [d.metadata.get("embed_text") or d.page_content for d in new_chunks]
    col.upsert(
        ids=ids,
        documents=[d.page_content for d in new_chunks],
        metadatas=[d.metadata for d in new_chunks],
        embeddings=ef(emb_texts),
    )
    global _vector_store
    _vector_store = col
    print(f"Incremental upsert: {len(new_chunks)} chunks (dedup by stable id).")
    _invalidate_all_chunks()
    return len(new_chunks)


def delete_by_source(source):
    """
    P1-4 按源文件删除（删除该 source 的全部块）。
    source 用绝对路径字符串，须与切分时 metadata['source'] 完全一致。
    只移除该文档，其余文档不受影响（对比旧的『删文件即全量重建』）。
    """
    col = get_vector_store()  # 确保集合已加载
    col.delete(where={"source": str(source)})
    _invalidate_all_chunks()
    print(f"Deleted chunks where source={source}")


# ====================== 3. 检索 + 重排（两段式） ======================
def retrieve_vector(question, k=None, where=None):
    """
    仅做向量召回（双塔粗排），返回 top-k 个 Document（已按相似度降序）。
    单独暴露出来，供 app.py 分阶段显示进度（先召回、再重排），也供「快速模式」直出。
    T3（2026-08-22）：把 L2 distance 转 cosine 相似度写入 metadata['vector_score']，
    作为统一置信度信号（拒答/低置信阈值用）。
    P1a（2026-09-04）：where 为 Chroma 元数据过滤（如 {"source": {"$in": [...]}}），
    在过滤子集内召回 top-k；None = 全库检索（行为不变）。
    """
    k = k or config.RETRIEVE_K
    collection = get_vector_store()
    if where:
        res = collection.query(query_texts=[question], n_results=k, where=where)
    else:
        res = collection.query(query_texts=[question], n_results=k)
    texts = res["documents"][0]
    metas = res["metadatas"][0]
    dists = (res.get("distances") or [[]])[0] if res.get("distances") else [None] * len(texts)
    docs = []
    for t, m, d in zip(texts, metas, dists):
        meta = dict(m) if m else {}
        if d is not None:
            # 归一化向量下 L2^2 = 2(1-cos)，故 cosine ≈ 1 - d^2/2（nomic-embed-text 归一化）
            meta["vector_score"] = round(1.0 - (d * d) / 2.0, 4)
        docs.append(Document(page_content=t, metadata=meta))
    return docs


# ====================== 3.0 源清单与教材过滤解析（P1a，2026-09-04） ======================
def get_available_sources():
    """当前索引中实际存在的 source 路径列表（去重排序）。
    从 _get_all_chunks 派生（带缓存，count 不变时零成本）。"""
    return sorted(
        {d.metadata.get("source") for d in _get_all_chunks() if d.metadata.get("source")}
    )


def resolve_source_query(query_str, sources=None):
    """把「源文件名子串」（如 '离散数学' / 'Linux' / '小程序'）解析为实际 source 路径列表。

    - 匹配对大小写不敏感；一个子串可匹配多个源（全部返回，如 'Linux' → 四章 PDF）
    - 无匹配 / 空串 → raise ValueError（报错信息带可用源清单，供 API 400 与 UI 提示）
    显式指定的过滤条件解析不到源应报错而非静默降级——静默降级会让调用方以为
    过滤生效了，实际却在全库检索。
    """
    sources = sorted(sources) if sources else get_available_sources()
    q = (query_str or "").strip()
    if not q:
        raise ValueError(
            "教材过滤条件为空；可用源：" + "、".join(Path(s).name for s in sources)
        )
    matched = sorted(s for s in sources if q.lower() in s.lower())
    if not matched:
        raise ValueError(
            f"没有语料源匹配 {q!r}；可用源：" + "、".join(Path(s).name for s in sources)
        )
    return matched


def _resolve_source_filter(question, source_filter):
    """P1a：统一解析教材过滤意图。返回允许的 source 路径列表或 None（不过滤）。

    source_filter 语义（retrieve_and_rerank / answer_question 对外统一）：
      - None        → 自动路由：rag_search_utils.route_sources 按问题关键词规则表
                      决定是否限定教材（SOURCE_FILTER_ROUTING_ENABLED 关闭时不路由）
      - str         → 显式子串（API/UI 传入），resolve_source_query 解析；
                      解析失败 raise ValueError（上层转 400 / UI 报错）
      - list[str]   → 显式 source 路径列表（程序化调用直传）
    """
    if source_filter is None:
        if not config.SOURCE_FILTER_ROUTING_ENABLED:
            return None
        try:
            return route_sources(question, get_available_sources())
        except Exception as e:  # 路由是增强项，任何故障都退回全库检索而非报错
            print(f"[warn] 教材路由失败，忽略过滤: {e!r}")
            return None
    if isinstance(source_filter, str):
        return resolve_source_query(source_filter)
    allowed = [str(s) for s in source_filter]
    return allowed or None


# ====================== 3.1 混合检索（向量 + BM25 双通道，RRF 融合） ======================
_all_chunks_cache = None
_all_chunks_count = -1
_bm25_cache = None  # (corpus_count, _BM25, all_chunks)——评审 P2-4：BM25 索引按语料版本缓存


def _invalidate_all_chunks():
    """增删文档后调用：强制下次查询重新拉取全库（BM25 通道同步最新语料）。"""
    global _all_chunks_cache, _all_chunks_count, _bm25_cache
    _all_chunks_cache = None
    _all_chunks_count = -1
    _bm25_cache = None  # 评审 P2-4：BM25 索引与全库块缓存同款失效机制


def _get_all_chunks():
    """从向量库拉取全部块（含 metadata），供 BM25 全库独立召回使用。

    带缓存：collection.count() 变化时自动刷新（增删/重建索引后 count 必变），
    避免每次查询都全库拉取。返回 List[Document]（page_content + metadata）。
    """
    global _all_chunks_cache, _all_chunks_count
    col = get_vector_store()
    cnt = col.count()
    if _all_chunks_cache is None or cnt != _all_chunks_count:
        data = col.get(include=["documents", "metadatas"])
        _all_chunks_cache = [
            Document(page_content=t, metadata=m if m is not None else {})
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        _all_chunks_count = cnt
    return _all_chunks_cache


def _get_bm25_cache():
    """（评审 P2-4，2026-08-30）BM25 索引 + 全库块缓存：按语料块数失效，
    与 _all_chunks_cache 同款机制（增删/重建索引时 _invalidate_all_chunks 一并清掉）。

    此前 hybrid_retrieve 每次查询都 new 一个 _BM25、对全库重新分词建 df 表
    （O(语料)/query）；1219 块时尚可忽略（瓶颈在 CPU reranker），但属于纯算法层
    的重复劳动，语料扩大后线性恶化。返回 (bm25_index, all_chunks)——二者取自
    同一次构建，保证下标一一对应。"""
    global _bm25_cache
    cnt = get_vector_store().count()
    if _bm25_cache is None or _bm25_cache[0] != cnt:
        chunks = _get_all_chunks()
        # B3 修复（2026-10-16）：BM25 也用短描述打分（与 v1 基线一致）。
        # 实测：新 caption 的逐字转写（本机路径、重复表头等噪声）会拉低图像块的 BM25 排名，
        # 使 p020 类图像在 RRF 融合阶段被挤出精排池。返回的仍是原 chunks（保持下标/身份一致）。
        bm_chunks = [
            Document(page_content=(c.metadata.get("embed_text") or c.page_content),
                     metadata=c.metadata)
            for c in chunks
        ]
        _bm25_cache = (cnt, _BM25(bm_chunks), chunks)
    return _bm25_cache[1], _bm25_cache[2]


def _enum_expand(question, base_docs, anchor_sources, candidate_k):
    """枚举/穷举类查询的锚点扩展召回（2026-09-11）。

    做法：用元数据（source）把召回范围钉到「锚点文档」上，在该范围内再做一次
    密集向量召回，与基础候选合并去重。

    为什么有效：全库 top-K 召回在多文档竞争下对单一文档的覆盖是不均匀的——
    一份 120 块的文档可能只有 10 块挤得进全库 50 强，而枚举题的答案恰恰集中
    在这一份文档里。锚定之后再召回，等于把该文档的配额从「全库竞争后剩下的 10 块」
    变成「该文档内语义最相关的 24 块」。

    参数：
      base_docs      基础融合候选（RRF 后）——用于①推导锚点 ②与之合并
      anchor_sources 已生效的教材过滤集（None=未过滤，此时从 base_docs 推导锚点）
      candidate_k    合并后候选池上限
    返回合并去重后的候选列表。扩展进来的块打 metadata['enum_expanded']=True，
    便于评估/调试追溯「这个块是锚点扩展捞回来的还是常规召回的」。
    """
    if not base_docs:
        return base_docs
    anchors = list(anchor_sources) if anchor_sources else []
    if not anchors:
        for d in base_docs[: config.ENUM_ANCHOR_SOURCES]:
            s = d.metadata.get("source")
            if s and s not in anchors:
                anchors.append(s)
    if not anchors:
        return base_docs
    # 锚点内密集召回：where 限制保证绝不越出既有教材过滤范围——这是安全性的关键
    extra = retrieve_vector(
        question, k=config.ENUM_ANCHOR_TOP_K, where={"source": {"$in": anchors}}
    )
    if not extra:
        return base_docs
    for d in extra:
        d.metadata["enum_expanded"] = True
    merged, seen = [], set()
    for d in list(base_docs) + list(extra):
        key = (d.metadata.get("source"), d.metadata.get("start_index"))
        if key in seen:
            continue
        seen.add(key)
        merged.append(d)
    return merged[:candidate_k]


def hybrid_retrieve(question, k=config.RETRIEVE_K, source_filter=None, candidate_k=None):
    """真混合检索：两个独立召回通道，各自在允许范围内召回 top-K，RRF 融合取 top-N。

    - 通道 A：向量召回（双塔粗排，语义匹配）—— Chroma（P1a：可带 source where 过滤）
    - 通道 B：BM25 词面召回（精确术语匹配，如 ACID / 2NF / 死锁）—— 对允许子集打分
    两路结果用倒数排名融合（RRF）互补，返回 top-N（默认 RERANK_CANDIDATE_K）。

    相比 v1「向量 top-50 池内 BM25 重排」的根本区别：
    旧版 BM25 只对已召回的 50 块重排，正确块若不在向量 top-50 就永远捞不到；
    新版 BM25 对全库打分，正确块即使向量排名很低，只要词面匹配也能独立召回进池。
    评审 P2-4：BM25 索引经 _get_bm25_cache 按语料版本缓存，不再每次查询重建。

    P1a（2026-09-04）：source_filter 为允许的 source 路径列表（None=全库）。
    注意本函数只做执行、不做路由决策——自动路由/显式解析统一在
    retrieve_and_rerank 完成（直调本函数的旧调用方保持旧行为）。

    candidate_k（2026-09-11）：RRF 融合候选池上限，None=config.RERANK_CANDIDATE_K。
    枚举题由 retrieve_and_rerank 传入更大值；同时若 ENUM_EXPAND_ENABLED 开启，
    再叠一层锚点扩展召回（见 _enum_expand）。
    """
    where = {"source": {"$in": list(source_filter)}} if source_filter else None
    vector_docs = retrieve_vector(question, k=k, where=where)  # 通道 A：允许范围内 top-K
    if not vector_docs and not source_filter:
        return []
    bm25, all_chunks = _get_bm25_cache()
    allowed_idx = None
    if source_filter:
        sset = set(source_filter)
        allowed_idx = [
            i for i, d in enumerate(all_chunks) if d.metadata.get("source") in sset
        ]
    # 通道 B：允许子集内独立召回（allowed_idx=None 即全库，行为与旧版一致）
    bm25_docs = _bm25_rank(
        question, all_chunks, top_n=config.BM25_TOP_K, index=bm25, allowed_idx=allowed_idx
    )
    if not vector_docs and not bm25_docs:
        return []  # 过滤子集内两路皆空 → 交由上层「空结果降级」去 filter 重检索
    # B 专项：融合候选池扩到 RERANK_CANDIDATE_K（>RERANKER_TOP_N），
    # 让「仅 BM25 命中、向量缺席」的正确块先进池，再由 reranker 精排回 top-N（见 retrieve_and_rerank）。
    pool_k = candidate_k or config.RERANK_CANDIDATE_K
    base = _rrf_fuse([vector_docs, bm25_docs], top_n=pool_k)
    # 枚举题锚点扩展：把「点召回」补成「同类集合召回」（默认关，见 config.ENUM_*）
    if config.ENUM_EXPAND_ENABLED and is_enum_query(question):
        base = _enum_expand(question, base, source_filter,
                            max(pool_k, config.ENUM_CANDIDATE_K))
    return base


def _retrieve_once(question, force_no_rerank, sources, candidate_k=None, top_n=None):
    """P1a：按给定 source 允许集执行一次检索（不含路由/降级决策）。
    sources 非空时给命中块打 metadata['source_filter'] 标记（UI/调试可见实际生效范围）。

    2026-09-11：candidate_k / top_n 可选——枚举题扩展时由 retrieve_and_rerank 传入
    放大值；均为 None 时用 config 默认，行为与引入该参数前完全一致。
    """
    out_n = top_n or config.RERANKER_TOP_N
    if force_no_rerank:
        docs = retrieve_vector(
            question, k=out_n,
            where={"source": {"$in": sources}} if sources else None,
        )
    elif config.HYBRID_ENABLED:
        docs = hybrid_retrieve(question, source_filter=sources, candidate_k=candidate_k)
        if config.RERANKER_ENABLED:
            docs = rerank_documents(question, docs, top_n=out_n)
        else:
            # 不开 reranker：候选池截回 out_n，保证进 LLM 的条数与旧版一致
            docs = docs[:out_n]
    else:
        docs = retrieve_vector(
            question, k=config.RETRIEVE_K,
            where={"source": {"$in": sources}} if sources else None,
        )
        if config.RERANKER_ENABLED:
            docs = rerank_documents(question, docs, top_n=out_n)
        else:
            docs = docs[:out_n]
    if sources and docs:
        tag = "|".join(Path(s).name for s in sources)
        for d in docs:
            d.metadata["source_filter"] = tag
    return docs


def retrieve_and_rerank(question, force_no_rerank=False, source_filter=None):
    """
    统一检索入口（UI 与 RAG 链共用，保证『看到的来源』=『喂给模型的来源』）：
      - force_no_rerank=True：纯向量 top-N（「快速模式」/ 评估纯向量对照），不进 hybrid 也不进 reranker。
      - 否则：
          HYBRID_ENABLED=True  → 先做向量+BM25 的 RRF 融合得到 top-N 候选；
                                若 RERANKER_ENABLED=True 再对候选做 bge-reranker 精排。
          HYBRID_ENABLED=False → 原两段式（向量 top-K → reranker top-N）；reranker 关则纯向量 top-N。
    返回按相关性排序的 Document 列表；reranker 开启时 metadata 含 rerank_score，否则含 hybrid_score。

    P1a（2026-09-04）教材过滤多路召回：
      - source_filter=None（默认）→ 规则自动路由：问题命中教材/课程关键词（rag_search_utils.
        SOURCE_RULES）时把向量路与 BM25 路都限定到对应教材，未命中 = 全库检索（旧行为）。
      - source_filter=str → 显式源子串（如 '离散数学'），解析失败 raise ValueError。
      - source_filter=list[str] → 显式 source 路径列表。
      - 空结果降级：过滤后两路候选为空 → 自动去掉过滤重检索一次（防过度过滤召回为空）；
        降级结果不带 source_filter 标记，调用方可据此区分。

    2026-09-11 枚举题扩展召回（config.ENUM_EXPAND_ENABLED，默认关）：
      检测到枚举/穷举意图（rag_search_utils.is_enum_query）时，候选池放宽到
      ENUM_CANDIDATE_K、进 LLM 的块数放宽到 ENUM_RERANK_TOP_N，并在锚点文档内
      做一次密集召回（_enum_expand）。非枚举问题走原路径，行为零变化。
    """
    sources = _resolve_source_filter(question, source_filter)
    enum_on = config.ENUM_EXPAND_ENABLED and is_enum_query(question)
    # ⚠️ 不要把 RRF 池放宽到 ENUM_CANDIDATE_K（2026-09-11 实测修正）：
    # _enum_expand 的合并上限是 max(pool_k, ENUM_CANDIDATE_K)，而 pool_k 又等于
    # candidate_k。两者取同一数值时，base 会先吃满这个上限，锚点扩展块被
    # merged[:candidate_k] 整段截掉——实测净新增 17 块，标记数 0（功能静默失效）。
    # 正确做法：RRF 池保持常规规模，把 ENUM_CANDIDATE_K 留给「基础池 + 锚点块」之和。
    candidate_k = None
    top_n = config.ENUM_RERANK_TOP_N if enum_on else None
    docs = _retrieve_once(question, force_no_rerank, sources, candidate_k, top_n)
    if not docs and sources:
        # 空结果降级（方案 §2.2-4）：过滤把召回滤空时退回全库，宁可宽不可空
        print(f"[warn] 教材过滤后候选为空（{len(sources)} 个源），降级为全库检索")
        docs = _retrieve_once(question, force_no_rerank, None, candidate_k, top_n)
    # B1 视觉重排（2026-10-14）：落点在 bge-reranker 之后、build_citations 之前。
    # 放这里而非 answer_question，是因为本函数是 UI / API / RAG 链共用的统一检索入口——
    # 三处出口自动一致，eval_heldout.py 也能直接量到效果。
    # 传 sources：视觉模块要用它把专用图像通道也限定在同一教材范围内（否则会把别的书的图顶上来）。
    # 传 top_n：开启视觉重排时候选会被并宽（并入图像通道结果），截断必须由视觉模块统一做。
    # 默认关闭（config.VISUAL_RERANK_ENABLED）；fail-open，任何失败返回原顺序。
    docs = _visual_rerank.visual_rerank(
        question, docs, top_n=top_n or config.RERANKER_TOP_N, sources=sources
    )
    return docs


def get_confidence(docs):
    """T3（2026-08-22）：取检索结果 top-1 的向量相似度（cosine）作为置信度信号。
    各检索路径（快速/精确/hybrid/reranker）的 doc 都经过 retrieve_vector，vector_score 通常可得；
    hybrid 下若 top-1 是纯 BM25 召回块（无 vector_score）返回 None，调用方按「不提示」处理。"""
    if not docs:
        return None
    return docs[0].metadata.get("vector_score")


def is_hard_reject(docs):
    """P1（2026-08-24）：硬拒答检索阶段判定。

    返回 (reject: bool, signal: float)：
      - docs 为空 → 直接拒（检索无结果）。
      - top-1 向量相似度 < HARD_REJECT_VECTOR_THRESHOLD → 拒（明显语料外）。
      - 精确模式且 reranker 开启时，top-1 rerank_score < HARD_REJECT_RERANK_THRESHOLD → 拒。
    阈值经 eval_heldout.py（B 组拒答率应 ≥90%）校准；取 0.55 仅拦明显语料外，误拒率低。
    """
    if not docs:
        return True, 0.0
    top = docs[0]
    vs = top.metadata.get("vector_score")
    rs = top.metadata.get("rerank_score")
    if vs is not None and vs < config.HARD_REJECT_VECTOR_THRESHOLD:
        return True, vs
    if config.RERANKER_ENABLED and rs is not None and rs < config.HARD_REJECT_RERANK_THRESHOLD:
        return True, rs
    return False, (vs if vs is not None else (rs if rs is not None else 0.0))


# 硬拒答标准文案（语料外明确拒，而非软提示；符合系统提示「宁可少答不可错答」）。
if config.APP_PROFILE == "paper":
    HARD_REJECT_ANSWER = (
        "根据提供的论文库无法回答该问题——检索到的内容与问题不相关（论文库可能未收录，"
        "或该问题超出已索引论文范围）。建议补充相关论文后重试，或换一种更贴近论文内容的问法。"
    )
else:
    HARD_REJECT_ANSWER = (
        "根据提供的课程资料无法回答该问题——检索到的内容与问题不相关（资料可能不足，"
        "或该问题超出本课程范围）。建议补充相关教材/笔记后重试，或换一种更贴近课程知识点的问法。"
    )


def grounded_generate(question, docs, chain):
    """P1（2026-08-24）：带接地自检的生成。

    流程：先正常生成 → 若 GROUNDED_VERIFY_ENABLED，用独立裁判校验是否完全基于上下文；
    未接地则把答案替换为 HARD_REJECT_ANSWER 并返回 grounded=False。
    返回 (answer, grounded_flag, grounded_score, citations 不在此处理)。
    仅「精确模式」调用（保护快速模式延迟，见 app.py）。
    """
    answer = chain.invoke(question, docs=docs)
    grounded_flag = True
    grounded_score = 1.0
    if config.GROUNDED_VERIFY_ENABLED:
        context = "\n\n".join(d.page_content for d in docs)
        res = grounding.verify_grounded(context, answer)
        grounded_score = res["score"]
        if not res["grounded"]:
            answer = HARD_REJECT_ANSWER
            grounded_flag = False
    return answer, grounded_flag, grounded_score


# ====================== 4. 拼 RAG 链（含引用） ======================
SYSTEM_RULES = """你是一位严谨、准确的高校课程助教，专门解答专业课与课本问题。
请严格遵守以下规则：
1. 只能依据下面提供的【上下文】作答，严禁使用上下文之外的知识，也不得凭记忆编造。
   重要：只要【上下文】包含了与问题相关的信息，就必须基于上下文尽量答全、答准；即使某些细节在资料中未完整展开，也要把资料已给出的部分讲清楚，不要因"信息不完整"而拒答或只说"无法回答"。
2. 仅当【上下文】完全没有提供与问题相关的任何信息、或上下文内部相互矛盾时，才说明"根据提供的资料无法回答"。不要猜测，也不要因个别细节缺失就用通用知识编造补全。注意：即使上下文描述的是示例、虚构或假设性概念，只要其中已给出明确答案，就必须基于上下文如实作答，不得以此为由拒答。
3. 合理使用 Markdown（加粗、列表等）让回答清晰易读。
4. 回答语言必须与用户提问保持一致（用户用中文提问，就用中文回答）。
5. 回答要紧扣问题，避免堆砌与问题无关的内容。
6. 回答中每提及一条来自【上下文】的具体事实，必须紧跟来源标注 [n]（半角方括号，n 为下方『已知来源』中的编号，即上下文第 n 段）；同一句可标多个如 [1][2]。仅当某事实确实无法对应任何来源时才不标。严禁编造超出范围的序号（n 必须 ≤ 已知来源条数）。
   重要：回答正文中，凡来自【上下文】的事实句，句末/句首必须带至少一个 [n] 标注；整段回答不允许出现"无任何 [n] 标注的长段落"。若某句没有对应来源可标，说明该句不属于上下文内容，请删除或改写为"根据提供的资料无法确认"。"""


def _build_prompt(rules, history, context, question, sources=None):
    """拼装最终 prompt（与去 langchain 前 ChatPromptTemplate 的模板逐字一致）。
    sources: 可选，已编号的『已知来源』列表（P1-2 内联引用用），与 context 段落顺序一一对应。
    """
    sources_block = ""
    if sources:
        sources_block = (
            "\n\n【已知来源】（共 %d 条，回答中引用事实请用 [n] 标注对应序号；"
            "n 即下方编号，也是上方上下文第 n 段）\n%s"
            % (len(sources), "\n".join(sources))
        )
    return f"""{rules}

（以下是之前的对话，仅作上下文参考；回答仍须严格以【上下文】为准）
{history}

上下文：
{context}{sources_block}

问题：
{question}

回答："""


def _block_header(i, d):
    """上下文段落头：`[n]《论文名》第 N 页 · 章节\n`（无页信息时退回 `[n] `）。"""
    page = d.metadata.get("page")
    if not page:
        return f"[{i}] "
    title = d.metadata.get("paper_title") or Path(d.metadata.get("source", "unknown")).stem
    head = f"[{i}]《{title}》第 {int(page)} 页"
    section = d.metadata.get("section")
    if section:
        head += f" · {section}"
    return head + "\n"


class RagChain:
    """替代 LCEL 链：持有 history / custom_prompt，invoke(question) 走 向量召回→重排→生成。"""

    def __init__(self, custom_prompt=None, history=""):
        self.custom_prompt = custom_prompt
        self.history = history

    def _compose_prompt(self, question, docs):
        """检索（docs 未预计算时）+ 拼 prompt。invoke / invoke_stream 共用同一份
        拼链逻辑，保证流式与非流式的上下文、引用编号完全一致。"""
        # docs 由调用方预计算时直接复用（UI 用同一份做引用，避免重复检索/来源不一致）；
        # 不传则自行检索（eval_local 等场景），保持向后兼容。
        if docs is None:
            docs = retrieve_and_rerank(question)
        # paper 页级引用：context 段前直接标 [n]，与【已知来源】编号对齐。
        # 2026-10-30 实测：只给来源列表、正文不带编号时，模型经常把 [n] 数错
        # （答对 h=8 却标了第 4 页而不是第 5 页）。显式编号避免"数第几段"。
        # 2026-10-30 实测：只给 [n] 时模型仍会挂错段——同一事实的"擦边块"（只提到 350GB、
        # 不含 1.2TB）被标成来源，或把结论段落标成摘要段。段头补上「论文名·第 N 页·章节」，
        # 模型可按页/章节判断哪段真正包含该事实，页级引用命中率 0.733 → 见 README 基线。
        label_mode = config.CITATION_LABEL_MODE
        context = "\n\n".join(
            (_block_header(i, d) + d.page_content if label_mode == "page" else d.page_content)
            for i, d in enumerate(docs, 1)
        )

        # P1-2 内联引用：把检索到的块按 1..N 编号，供 prompt 要求 LLM 用 [n] 标注来源。
        # 编号顺序与 context 段落顺序一致，app.py 引用区与之一一对应。
        sources = []
        for i, d in enumerate(docs, 1):
            title = d.metadata.get("paper_title")
            page = d.metadata.get("page")
            section = d.metadata.get("section")
            mode = config.CITATION_LABEL_MODE
            use_page = bool(page) and (
                mode == "page" or (mode == "auto" and d.metadata.get("paper_id"))
            )
            if use_page:
                label = f"{i}. 《{title or Path(d.metadata.get('source', 'unknown')).stem}》第 {int(page)} 页"
                if section:
                    label += f" · {section}"
                sources.append(label)
            else:
                name = Path(d.metadata.get("source", "unknown")).name
                off = d.metadata.get("start_index")
                sources.append(f"{i}. {name}（offset={off}）")

        rules = SYSTEM_RULES
        if config.APP_PROFILE == "paper":
            rules += (
                "\n7. 引用页码只能抄自【已知来源】中「第 N 页」的数字，"
                "不得自行推算、改写或编造页码。"
                "\n8. 选择 [n] 时必须引用真正包含该事实的段落：上下文每段开头的"
                "《论文名》与「第 N 页 · 章节」就是判断依据；问题指名章节时"
                "（如结语、实验、摘要），优先引用该章节所在段落，"
                "不要把只沾边的段落（例如只出现同一个数字、却不含该结论的段落）当来源。"
                "\n9. 段首的 [n] 是该段编号：引用时必须在句末原样写出 [n]。"
                "即使用正文里已经写了论文名或页码，也不能用它替代 [n] 标注。"
            )
        if self.custom_prompt and self.custom_prompt.strip():
            rules = f"""{rules}

【用户补充要求】
{self.custom_prompt.strip()}"""

        return _build_prompt(rules, self.history, context, question, sources=sources)

    def invoke(self, question, docs=None):
        prompt = self._compose_prompt(question, docs)
        return ollama_chat(
            [{"role": "user", "content": prompt}], temperature=config.LLM_TEMPERATURE
        )

    def invoke_stream(self, question, docs=None):
        """invoke 的流式版（评审 P3）：检索/拼链与 invoke 完全一致，生成段逐片段
        yield。调用方负责聚合片段得到完整答案（用于缓存/日志/落盘）。"""
        prompt = self._compose_prompt(question, docs)
        yield from ollama_chat_stream(
            [{"role": "user", "content": prompt}], temperature=config.LLM_TEMPERATURE
        )


def build_rag_chain(custom_prompt=None, history=""):
    """
    返回 {'chain': RagChain 实例, 'retrieve_and_rerank': 检索+重排函数}
    retriever 不再单独暴露，统一走 retrieve_and_rerank，保证 UI 与链来源一致。
    history: 最近若干轮对话文本（由 app.py 拼好传入），让助手有上下文记忆。
    """
    return {"chain": RagChain(custom_prompt=custom_prompt, history=history),
            "retrieve_and_rerank": retrieve_and_rerank}


# ====================== 4.5 Query 改写（多轮指代消解，2026-09-18） ======================
REWRITE_SYSTEM_PROMPT = (
    "你是检索查询改写器。请根据对话历史，把用户的最新问题改写为一个独立、完整、"
    "无指代的检索问题（它将被送进一个看不到对话历史的检索系统）。规则：\n"
    "1. 只输出改写后的问题本身——不要解释、不要引号、不要任何前后缀；\n"
    "2. 与原问题保持同一语言（中文问句改写后仍是中文）；\n"
    "3. 只补全对话历史中已明确出现的指代对象（如「第三章那个定理」→ 补全为具体章节"
    "与定理名），严禁编造历史中没有的信息；\n"
    "4. 若最新问题本身已独立完整，原样输出它。"
)


def rewrite_question(question, history):
    """多轮指代消解改写：history + 当前问题 → 独立完整的检索问题。

    为什么需要：多轮里的「第三章那个定理怎么用」脱离历史无法检索——向量与 BM25
    都看不到「第三章」指什么。改写后的问题（search_question）统一用于语义缓存 key、
    教材自动路由（SOURCE_RULES）、检索与生成，让整条管线基于完整语义工作。

    触发条件（answer_question 已把守开关与 history 判空，此处只做最后一道自守）：
      - config.QUERY_REWRITE_ENABLED 开启 且 history 非空。
    返回 (rewritten, ok)：
      - ok=True   rewritten 为改写后问题（可能与原问题相同——模型判定本就独立完整）
      - ok=False  rewritten 为原问题（未触发 / Ollama 失败 / 输出异常），调用方回退
    任何异常都吞掉回退——改写是增强项，绝不能成为问答链路的新故障点。
    """
    fallback = (question, False)
    if not config.QUERY_REWRITE_ENABLED:
        return fallback
    if not history or not str(history).strip():
        return fallback
    try:
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "对话历史：\n%s\n\n最新问题：%s\n\n改写后的独立问题："
                % (history, question),
            },
        ]
        out = ollama_chat(messages, temperature=0.0)  # 改写要确定性，压温度
        rewritten = str(out or "").strip().strip('"“”‘’').strip()
        # 合理性守卫：非空；长度不超原问题 3 倍——指代补全只会小幅加长，
        # 模型不听指令输出口水话/整段解释时直接判废回退，不污染下游
        if not rewritten or len(rewritten) > max(len(question) * 3, 60):
            return fallback
        return rewritten, True
    except Exception:
        return fallback


# ====================== 5. 统一问答入口（评审 P1，2026-08-30） ======================
def build_citations(docs):
    """检索结果 → 引用结构（UI / API / answer_question 三处共用，杜绝各写一份漂移）。

    分数展示优先级：rerank_score → hybrid_score → vector_score
    （快速模式只有 vector_score；混合未精排时是 hybrid_score；精排后是 rerank_score）。
    """
    citations = []
    for rank, d in enumerate(docs, 1):
        score = (
            d.metadata.get("rerank_score")
            or d.metadata.get("hybrid_score")
            or d.metadata.get("vector_score")
        )
        c = {
            "rank": rank,
            "source": Path(d.metadata.get("source", "unknown")).name,
            "content": d.page_content.strip(),
            "score": score,
            "start_index": d.metadata.get("start_index"),
        }
        # 论文块：透出论文名/物理页/章节/块类型（三出口共用同一结构）
        if d.metadata.get("paper_id"):
            c.update(
                {
                    "paper_id": d.metadata.get("paper_id"),
                    "title": d.metadata.get("paper_title")
                    or Path(d.metadata.get("source", "unknown")).stem,
                    "page": d.metadata.get("page"),
                    "page_start": d.metadata.get("page_start"),
                    "page_end": d.metadata.get("page_end"),
                    "section": d.metadata.get("section"),
                    "block_type": d.metadata.get("block_type"),
                    "block_ids": d.metadata.get("block_ids"),
                    "type": d.metadata.get("block_type") or "text",
                }
            )
        # P2 多模态：图像块透出 type/image_path/page（UI 引用区渲染原图；文本块不加这些键）
        if d.metadata.get("type") == "image":
            c["type"] = "image"
            c["image_path"] = d.metadata.get("image_path")
            c["page"] = d.metadata.get("page")
        citations.append(c)
    return citations


def answer_question(question, mode="accurate", custom_prompt=None, history="",
                    use_cache=True, on_stage=None, on_token=None, source=None,
                    on_retrieved=None, use_rewrite=None):
    """统一问答入口：Streamlit / FastAPI / 评估三个出口共用同一套
    「缓存 → 检索 → 硬拒答 → 生成 → 接地自检 → 缓存写入」管线。

    背景（2026-08-30 外部评审 P1）：此前硬拒答与接地自检只挂在 app.py 的精确模式，
    main.py /query 明确绕过（「不做硬拒答/接地自检，非空即可缓存」）——held-out
    B 组 95% 拒答率只对 UI 成立，API 消费者拿到无闸门回答，且 API 缓存策略与 UI
    的「拒答不缓存」矛盾。现收敛为本函数，任何新增出口都必须走这里。

    2026-09-04 默认值由 fast 上调为 accurate（与 api_schemas.py / app.py / mcp_server.py
    统一）。两条依据：
      1. reranker GPU 化后 accurate 检索+精排段 P50 21.1s → 1.16s，端到端仅比 fast
         慢约 0.9s，而检索质量 context_recall +16.7% / context_precision +19.2%
         （reranker A/B 实证，见 EVALUATION.md §7）；
      2. 更关键——fast 无硬拒答/无接地自检，默认 fast 等于让 API 与 MCP 调用方
         默认拿到无闸门的回答，正是 2026-08-30 外部评审 P1 指出的「held-out B 组
         95% 拒答率只对 UI 成立」问题。默认值上调后，所有出口默认带闸门。
    ⚠️ 行为变更（对未显式传 mode 的调用方）：
      - 语料外问题现在返回 HARD_REJECT_ANSWER 而非生成式答案；
      - 流式 on_token 在 accurate 下一律缓冲（接地自检需完整答案），不再逐段回调；
      - 拒答结果不写入语义缓存。
    如需旧行为，显式传 mode="fast"。

    mode 语义（与检索模式一致）：
      fast     纯向量 top-N + 软提示（low_confidence），无硬拒答/无接地自检（降级档）
      accurate 混合检索（+可选 reranker）+ 硬拒答 + 接地自检（默认）
    source（P1a，2026-09-04）教材过滤：
      None      规则自动路由（问题命中教材关键词 → 限定对应教材，未命中=全库）
      str       显式源文件名子串（如 '离散数学'），解析失败 raise ValueError；
                显式指定时绕开语义缓存读写（缓存 key 不含 source 维度）
      list[str] 显式 source 路径列表
    use_cache   语义缓存开关（评估侧必须传 False，避免吃到线上缓存污染结果）
    use_rewrite Query 改写开关（2026-09-18）：None（默认）=按 config.QUERY_REWRITE_ENABLED
                且 history 非空自动判定；False=强制旁路（评估口径保护，与 use_cache=False
                同款显式旁路）；True=强制尝试（history 为空时仍是 no-op）
    on_stage    可选回调 fn(stage)，stage ∈ {"retrieve","rerank","generate"}，
                供 UI 展示分阶段进度；缓存命中不触发。
    on_token    可选回调 fn(piece)，真实流式（评审 P3）：生成中每到达一个文本片段
                即回调一次。仅快速模式生效——精确模式的接地自检必须拿到完整答案
                才能判定/替换（未接地的答案替换为拒答文案），边生成边展示会让
                已展示的未验证内容无法撤回，故精确模式一律缓冲。回调后仍把完整
                答案聚合进返回值，缓存/日志语义不变；缓存命中不触发。
    on_retrieved 可选回调 fn(citations, rejected)：检索（+精排）完成、且硬拒答判定已
                出结果的那一瞬回调一次，此时生成尚未开始。供 SSE 等流式出口先把
                「引用列表 + 是否拒答」推给客户端，不必等生成结束才拿到来源（引用
                先到、正文后到，是流式问答的标准事件顺序）。缓存命中时同样触发
                （回放缓存的 citations，rejected 恒为 False——拒答结果不写缓存）。
                不传则行为与引入本参数前完全一致（None 分支零副作用）。

    返回 dict：
      answer            最终答案（硬拒/未接地时已被替换为 HARD_REJECT_ANSWER）
      citations         [{rank,source,content,score,start_index}]（缓存命中来自缓存）
      docs              检索结果 List[Document]（缓存命中时 None）
      cached/cache_sim  是否命中语义缓存及相似度
      rewritten         Query 改写结果（2026-09-18）：改写成功且与原问不同时为改写后
                        问题，否则 None——供日志/UI 观测改写是否发生及质量
      rejected/reject_signal                 检索阶段硬拒答及信号分
      grounding_ran/grounded/not_grounded    接地自检：是否执行/是否通过/是否未通过
      grounded_score    接地自检得分（未执行或检索拒时为 None）
      low_confidence    top-1 向量分 < LOW_CONFIDENCE_THRESHOLD（软提示，两模式都有）
    """
    stage = on_stage or (lambda s: None)
    mode_label = "精确" if mode == "accurate" else "快速"

    # P1a：source 语义与 retrieve_and_rerank 对齐——None=规则自动路由；str=显式教材子串；
    # list=显式 source 路径列表。语义缓存 key 不含 source 维度（lookup 只看 question+mode），
    # 显式指定教材时必须绕开缓存读写，否则不同过滤范围的同问句会命中同一条历史答案；
    # 自动路由是 question 的确定函数（同问题必同路由），缓存语义不受影响。
    _explicit_source = source is not None
    if _explicit_source and isinstance(source, str):
        source = resolve_source_query(source)  # 解析失败 ValueError 上抛（API 400 / UI 报错）

    # 0) Query 改写（多轮指代消解，2026-09-18）：history 非空且开关开启时，把
    #    「历史 + 当前问题」改写为独立检索问题；此后缓存 key / SOURCE_RULES 自动路由 /
    #    检索 / 生成统一使用 search_question。失败回退原问题（rewrite_question 内兜底），
    #    改写永不阻塞问答。评估口径保护：冻结评估集均为单轮（无 history）天然不触发，
    #    确需强制旁路时传 use_rewrite=False。
    _rewrite_on = config.QUERY_REWRITE_ENABLED if use_rewrite is None else bool(use_rewrite)
    if _rewrite_on:
        rewritten, rewrite_ok = rewrite_question(question, history)
    else:
        rewritten, rewrite_ok = question, False
    search_question = rewritten if rewrite_ok else question

    # 1) 语义缓存：命中跳过检索/重排/生成；写入策略（拒答/未接地不缓存）统一在下方 store。
    #    key 用 search_question：多轮改写后同义问法收敛到同一条缓存（改写正是为此）；
    #    单轮（无 history）search_question == question，缓存行为与旧版完全一致。
    if use_cache and not _explicit_source:
        cached = semantic_cache.lookup(search_question, mode_label)
        if cached:
            if on_retrieved is not None:
                # SSE 出口：缓存命中同样要先给引用，否则前端拿不到来源列表
                on_retrieved(cached["citations"], False)
            return {
                "answer": cached["answer"], "citations": cached["citations"],
                "docs": None, "cached": True, "cache_sim": cached["sim"],
                "rewritten": (rewritten if rewrite_ok and rewritten != question else None),
                "rejected": False, "reject_signal": None,
                "grounding_ran": False, "grounded": True,
                "not_grounded": False, "grounded_score": None,
                "low_confidence": False,
            }

    # 2) 检索（accurate=混合+可选精排；fast=纯向量；P1a：source 教材过滤/自动路由；
    #    检索用 search_question——改写的全部意义就在这：让检索看到完整语义）
    stage("retrieve")
    docs = retrieve_and_rerank(
        search_question, force_no_rerank=(mode != "accurate"), source_filter=source
    )
    if mode == "accurate" and config.RERANKER_ENABLED:
        stage("rerank")
    citations = build_citations(docs)

    # 3) 硬拒答（仅精确模式；快速模式保留软提示 low_confidence，不拦）
    rejected, reject_signal = False, None
    if mode == "accurate":
        rejected, reject_signal = is_hard_reject(docs)

    # 3.5) 检索阶段回调（SSE 流式出口用）：此刻 citations 与硬拒答判定均已就绪，
    #      调用方可在生成开始前就把「引用 + 是否拒答」推给客户端——拒答时前端
    #      立刻能给出结论，不必干等生成。不传则完全无副作用。
    if on_retrieved is not None:
        on_retrieved(citations, rejected)

    # 4) 生成 + 接地自检（仅精确模式）
    grounded, grounded_score, grounding_ran, not_grounded = True, None, False, False
    if rejected:
        answer = HARD_REJECT_ANSWER
        grounded = False
    else:
        stage("generate")
        rag = build_rag_chain(custom_prompt=custom_prompt, history=history)
        # 生成同样用 search_question：检索基于它召回，生成与检索基于同一语义；
        # 原问题已包含在 history 文本里（chain 的 prompt 含 history），信息不丢
        if mode == "accurate":
            # 精确模式：缓冲生成（接地自检需要完整答案，见 on_token docstring）
            answer, grounded, grounded_score = grounded_generate(search_question, docs, rag["chain"])
            grounding_ran = config.GROUNDED_VERIFY_ENABLED
            not_grounded = grounding_ran and not grounded
        elif on_token is not None:
            # 快速模式 + on_token：真实流式——边生成边回调，聚合回完整答案
            parts = []
            for piece in rag["chain"].invoke_stream(search_question, docs=docs):
                parts.append(piece)
                on_token(piece)
            answer = "".join(parts)
        else:
            answer = rag["chain"].invoke(search_question, docs=docs)

    # 5) 缓存写入：拒答（检索拒/未接地）不入缓存，与 UI 既有口径一致；
    #    显式指定教材的请求不写缓存（key 无 source 维度，见上方说明）；
    #    key 同 lookup 用 search_question（读写同 key，口径一致）
    if use_cache and not _explicit_source:
        semantic_cache.store(search_question, mode_label, answer, citations,
                             reject=(rejected or not_grounded))

    conf = get_confidence(docs)
    return {
        "answer": answer, "citations": citations, "docs": docs,
        "cached": False, "cache_sim": None,
        "rewritten": (rewritten if rewrite_ok and rewritten != question else None),
        "rejected": rejected, "reject_signal": reject_signal,
        "grounding_ran": grounding_ran, "grounded": grounded,
        "not_grounded": not_grounded, "grounded_score": grounded_score,
        "low_confidence": bool(conf is not None and conf < config.LOW_CONFIDENCE_THRESHOLD),
    }
