"""
app.py — Streamlit 演示界面（改编自 RAG-KnowBot + W2 引用溯源增强）。
保留引用溯源（Sources 折叠面板），并升级为：带排名 / 相关性分数 / 原文偏移。

P0 改造（2026-08-18）：
  - P0-1 启动自检：侧边栏顶部状态条（依赖 / Ollama / Reranker 权重），消灭神秘转圈。
  - P0-2 对话记忆：最近 N 轮对话喂给 LLM；历史落盘 .chat_history.json，刷新不丢；可清空。
  - P0-3 检索体验：分阶段进度（向量召回 → [BM25 融合] → reranker 重排 → 生成）；「快速/精确」双模式。
          2026-09-04 定位调整（精排 GPU 化后重新权衡）：
          - 精确模式为**默认**。GPU 化后精排段 P50 21.1s → 1.16s，accurate 端到端仅比 fast 慢约 1s，
            却换来 context_recall +16.7% / context_precision +19.2%（EVALUATION.md §7 / §6 A/B）。
          - 快速模式改为**降级档**，保留理由是容错而非加速：RERANKER_DEVICE=auto 在无 CUDA 时回落
            CPU（21.1s），此时 fast 是唯一可用选项；显存不足 / 高并发 / 首题冷启动同理。
          - UI 不再硬编码延迟秒数（~19s 是过期 CPU 表述，且延迟硬件绑定、必然再次过期）；
            一律以 EVALUATION.md「§7 性能基线」为准。

注意：P1-4 起删除文件只按 source 从索引中移除该文档（不再全量重建）；上传只增量 upsert 新文件块。
全量重建入口保留在侧边栏「全量重建索引（data/ 全部）」按钮（走 get_vector_store(chunks)）。
"""
import streamlit as st
import os
import json
import re
import time

# reranker 权重（bge-reranker-v2-m3）加载需访问 HuggingFace。本机 huggingface.co 不可达会
# 超时卡死（页面一直转）。故在 import 任何可能拉起 sentence_transformers / huggingface_hub 的
# 模块之前设置——reranker.py 在首次调用时才 import sentence_transformers（间接 import huggingface_hub），
# 一旦它捕获到默认 HF_ENDPOINT=huggingface.co 就会卡死。
# 默认走镜像；若本机已缓存权重则强制离线，彻底不碰网络（权重已下载在 ~/.cache/huggingface）。
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
_rerank_cache = os.path.expanduser(
    "~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3"
)
if os.path.isdir(_rerank_cache):
    os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
from rag_core import (
    load_and_chunk_documents,
    get_vector_store,
    answer_question,
    get_confidence,
    get_available_sources,
    chunk_file,
    add_documents,
    delete_by_source,
)
import config
from health import collect_status
from query_log import log_query, set_feedback
import semantic_cache

HISTORY_FILE = Path(config.DATA_DIR).parent / ".chat_history.json"
MAX_HISTORY_TURNS = 6  # 喂给 LLM 的最近轮数


# ==================== 对话历史持久化 ====================
def load_history():
    """启动时从磁盘恢复对话（刷新/重启不丢）。"""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(messages):
    """每轮结束后落盘（仅存 role/content/citations，结构可 JSON 序列化）。"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def build_history_text(messages):
    """把历史拼成文本块喂给 LLM。"""
    turns = []
    for m in messages:
        role = "用户" if m["role"] == "user" else "助手"
        turns.append(f"{role}：{m['content']}")
    kept = turns[-MAX_HISTORY_TURNS * 2:] if turns else []
    return "\n".join(kept) if kept else "（无历史对话）"


def render_answer_html(answer, n_sources, mid):
    """P1-2 内联引用：把回答中的 [n]（含中文模型常用的全角【n】／［n］）渲染为可点击锚点
    （仅 1<=n<=n_sources 合法），非法编号原样保留。mid 为消息序号，避免多条消息锚点 id 冲突。"""
    # 半角 [n]、全角 【n】、全角 ［n］ 都映射到同一序号
    _BRACKET_RE = re.compile(r"\[(\d+)\]|【(\d+)】|［(\d+)］")

    def repl(m):
        k = int(m.group(1) or m.group(2) or m.group(3))
        if 1 <= k <= n_sources:
            return f'<sup><a href="#m{mid}-src-{k}" title="查看来源 {k}">[{k}]</a></sup>'
        return m.group(0)

    return _BRACKET_RE.sub(repl, answer)


def _paper_pdf_path(cit):
    name = cit.get("source", "")
    p = Path(config.PAPERS_RAW_DIR) / name
    if p.exists():
        return p
    for q in Path(config.PAPERS_DIR).rglob(name):
        return q
    return None


def render_citation(cit, mid=None):
    """引用卡：课程版显示 offset；论文版显示《标题》第 N 页 + 章节 + 页图高亮。"""
    score_txt = f" · 相关性 **{cit['score']:.3f}**" if cit.get("score") is not None else ""
    if mid is not None:
        st.markdown(
            f'<div id="m{mid}-src-{cit["rank"]}"></div>', unsafe_allow_html=True
        )
    if cit.get("paper_id"):
        header = (
            f"**#{cit['rank']} 《{cit.get('title') or cit.get('source')}》"
            f"第 {cit.get('page')} 页**{score_txt}"
        )
        if cit.get("section"):
            header += f" · {cit['section']}"
        if cit.get("block_type"):
            header += f"  `{cit['block_type']}`"
        st.markdown(header, unsafe_allow_html=True)
        pdf = _paper_pdf_path(cit)
        if pdf and cit.get("page"):
            try:
                import page_render

                boxes = page_render.bboxes_for_blocks(
                    cit["paper_id"], cit.get("block_ids"), page=int(cit["page"])
                )
                png = page_render.render_page_png(
                    pdf, int(cit["page"]), blocks=[{"bbox": b} for b in boxes]
                )
                st.image(
                    png,
                    caption=f"《{cit.get('title') or cit.get('source')}》第 {cit['page']} 页 · 引用块高亮",
                    width=560,
                )
            except Exception as e:
                st.warning(f"页图渲染失败: {e}")
        else:
            st.caption("页图不可用：找不到论文 PDF")
    else:
        st.markdown(
            f"**#{cit['rank']} {cit['source']}**{score_txt}  `offset={cit.get('start_index')}`",
            unsafe_allow_html=True,
        )
        # P2 多模态：图像块渲染原图——caption 只是检索代理文本，原图才是最终证据
        if cit.get("type") == "image" and cit.get("image_path"):
            _img = Path(__file__).parent / cit["image_path"]
            if _img.exists():
                st.image(str(_img), caption=f"📎 教材插图 · 第 {cit.get('page')} 页", width=420)
            else:
                st.warning(f"图像文件缺失: {cit['image_path']}")
    st.markdown(
        f"```text\n{cit['content'][:500]}{'...' if len(cit['content']) > 500 else ''}\n```",
        unsafe_allow_html=True,
    )


def _persist_feedback(msg, fb):
    """写反馈：回写 .query_log.jsonl（按 log_ts 定位）+ 存历史（刷新不丢）。"""
    log_ts = msg.get("log_ts")
    if log_ts:
        set_feedback(log_ts, fb)
    save_history(st.session_state.messages)


def render_feedback_buttons(mid):
    """最后一条助手消息的 👍/👎 反馈按钮（#17 最小闭环）。

    幂等：已反馈则只显示当前状态，不再给按钮；点击后回写日志 + 历史并 rerun。
    """
    msg = st.session_state.messages[mid]
    cur = msg.get("feedback")
    if cur:
        st.caption(f"反馈：{cur}")
        return
    c1, c2 = st.columns(2)
    if c1.button("👍 有用", key=f"fb_up_{mid}"):
        msg["feedback"] = "👍"
        _persist_feedback(msg, "👍")
        st.rerun()
    if c2.button("👎 不好", key=f"fb_down_{mid}"):
        msg["feedback"] = "👎"
        _persist_feedback(msg, "👎")
        st.rerun()


def _course_upload_panel():
    st.subheader("上传课程资料")
    uploaded_files = st.file_uploader("PDF / PPTX / TXT / MD / DOCX", type=["pdf", "pptx", "txt", "md", "docx"], accept_multiple_files=True)
    if uploaded_files:
        data_dir = Path(config.DATA_DIR)
        data_dir.mkdir(exist_ok=True)
        saved_paths = []
        for file in uploaded_files:
            dest = data_dir / file.name
            with open(dest, "wb") as f:
                f.write(file.getbuffer())
            saved_paths.append(dest)
            st.success(f"已保存: {file.name}")
        if st.button("⤴️ 增量索引上传文件", type="primary"):
            if get_vector_store(auto_rebuild=False).count() == 0:
                with st.spinner("首次建库：从 data/ 全量重建索引..."):
                    chunks = load_and_chunk_documents()
                    get_vector_store(chunks)
                    semantic_cache.invalidate()
                st.success("首次建库完成！")
            else:
                with st.spinner("正在增量索引（仅新上传文件）..."):
                    total = 0
                    for p in saved_paths:
                        total += add_documents(chunk_file(p))
                    st.success(f"增量索引完成！新增/覆盖 {total} 块")
            st.rerun()


def _paper_upload_panel():
    st.subheader("上传论文 PDF")
    st.caption("MinerU 子进程解析 → normalized → 页内切分 → paper_rag 索引。")
    files = st.file_uploader(
        "PDF", type=["pdf"], accept_multiple_files=True, key="paper_uploader"
    )
    if files:
        raw = Path(config.PAPERS_RAW_DIR)
        raw.mkdir(parents=True, exist_ok=True)
        for file in files:
            dest = raw / file.name
            with open(dest, "wb") as f:
                f.write(file.getbuffer())
            st.success(f"已保存: {file.name}")
        if st.button("⤴️ 解析并索引上传论文", type="primary", key="paper_ingest_btn"):
            import paper_index
            import paper_ingest

            with st.spinner("MinerU 解析中（首次会下载模型，可能较慢）..."):
                try:
                    res = paper_ingest.ingest_dir(
                        raw_dir=raw, only_new=True, retry_failed=True
                    )
                    for pid, status in res.items():
                        if status in ("normalized", "skip"):
                            n = paper_index.upsert_paper(pid)
                            st.success(f"{pid} 已索引 {n} 块")
                        else:
                            st.info(f"{pid}: {status}")
                except Exception as e:
                    st.error(f"解析/索引失败: {e}")
            st.rerun()


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="CourseRAG - 论文库问答助手" if config.APP_PROFILE == "paper" else "CourseRAG - 专业课问答助手",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 会话状态 ====================
if "messages" not in st.session_state:
    st.session_state.messages = load_history()
if "custom_prompt" not in st.session_state:
    st.session_state.custom_prompt = None
if "mode" not in st.session_state:
    # 2026-09-04 起默认精确模式：GPU 化后精排段 P50 1.16s，端到端仅比 fast 慢约 1s，
    # 检索质量却是 +16.7% context_recall / +19.2% context_precision（EVALUATION.md §6 A/B）。
    # 快速模式保留为降级档：RERANKER_DEVICE=auto 在无 CUDA 时回落 CPU（21.1s），
    # 以及显存不足 / 高并发 / 首题冷启动场景，由用户手动切换。
    st.session_state.mode = "accurate"


# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("📚 CourseRAG 控制台")
    _profile, _collection, _persist = config.profile_summary()
    st.caption(f"profile={_profile} · collection={_collection} · persist={_persist}")

    # ---- P0-1 启动自检状态条 ----
    st.subheader("🔧 环境状态")
    for label, ok, msg in collect_status():
        if ok:
            st.success(f"✓ {label}：{msg}")
        else:
            st.error(f"✗ {label}：{msg}")
    st.caption("出问题先看这里，别对着转圈发呆。")

    st.markdown("---")

    # ---- P0-3 检索模式（P1-1：精确模式叠加 向量+BM25 混合检索）----
    st.subheader("⚙️ 检索模式")
    # 文案跟随实际配置动态显示：reranker 未启用时如实标注"无精排"，避免误导用户（测试发现的问题）
    _rerank_label = "+ bge-reranker 精排" if config.RERANKER_ENABLED else "（无 reranker）"
    mode = st.radio(
        "选择检索模式",
        ["fast", "accurate"],
        index=0 if st.session_state.mode == "fast" else 1,
        # 标签只标定位（默认 / 降级档），不写死秒数：延迟硬件绑定，写死必然再次过期
        format_func=lambda m: "快速（纯向量 top-5 · 降级档）" if m == "fast" else f"精确（向量 + BM25 {_rerank_label} · 默认）",
        help="精确（默认）：向量召回 top-50 + 自实现 BM25 词面召回，RRF 融合"
        + ("后再经 bge-reranker 精排，" if config.RERANKER_ENABLED else "（当前未启用 reranker），")
        + "专治「ACID / 2NF / 死锁」等纯向量词面匹配弱的术语难 query。"
        "快速（降级档）：纯向量 top-5，跳过融合与精排，用于无 CUDA / 显存不足 / 高并发 / 首题冷启动场景。"
        "延迟基线以 EVALUATION.md「§7 性能基线」为准（数字硬件绑定，不写死在 UI）。",
    )
    st.session_state.mode = mode
    if mode == "accurate":
        if not config.HYBRID_ENABLED and not config.RERANKER_ENABLED:
            st.warning("HYBRID_ENABLED 与 RERANKER_ENABLED 均为 false，精确模式退化为纯向量。")
        elif not config.RERANKER_ENABLED:
            st.info("精确模式当前为「向量 + BM25 融合」（未启用 reranker）；"
                    "设 RERANKER_ENABLED=true 可叠加 bge-reranker 精排。")

    # ---- P1a（2026-09-04）教材过滤：限定检索范围 / 关键词自动路由 ----
    st.subheader("📚 论文过滤" if config.APP_PROFILE == "paper" else "📚 教材过滤")
    try:
        _src_names = sorted(Path(s).name for s in get_available_sources())
    except Exception:
        _src_names = []  # 索引未就绪（Ollama 未启动等）不阻塞页面
    _AUTO = "全部（自动路由）"
    _filter_choice = st.selectbox(
        "限定检索范围",
        [_AUTO] + _src_names,
        help="默认：按问题关键词自动路由到对应教材（如提到 ACID 只查数据库教材），"
        "未命中关键词时全库检索。也可手动指定某份教材，把检索范围限定在其内。",
        disabled=not _src_names,
    )
    # 选择值 = 源文件名（basename 子串也能匹配到全路径，与 API source 语义一致）
    st.session_state.source_filter = None if _filter_choice == _AUTO else _filter_choice
    st.session_state.source_auto = _filter_choice == _AUTO

    st.markdown("---")

    # ---- 上传（按 profile 分派）----
    if config.APP_PROFILE == "paper":
        _paper_upload_panel()
    else:
        _course_upload_panel()

    st.markdown("---")
    if config.APP_PROFILE == "paper":
        st.subheader("论文索引管理")
        st.caption("从 data/papers/normalized/ 全量重建论文索引。")
        if st.button("🔄 全量重建论文索引", type="primary", key="reindex_paper_btn"):
            import paper_index

            with st.spinner("正在重建论文索引..."):
                try:
                    n = paper_index.rebuild_all()
                    st.success(f"索引完成！共 {n} 块")
                    st.rerun()
                except Exception as e:
                    st.error(f"索引失败: {e}")
    else:
        st.subheader("索引管理")
        st.caption("基础库已建好时，上方上传=增量追加；此按钮用于从 data/ 全量重建（忽略已上传但未索引的文件差异）。")
        if st.button("🔄 全量重建索引（data/ 全部）", type="primary", key="reindex_btn"):
            with st.spinner("正在从 data/ 重建索引..."):
                try:
                    chunks = load_and_chunk_documents()
                    get_vector_store(chunks)
                    semantic_cache.invalidate()  # P2：重建后旧缓存作废
                    st.success(f"索引完成！共 {len(chunks)} 块")
                    st.rerun()
                except Exception as e:
                    st.error(f"索引失败: {e}")

    st.markdown("---")
    st.subheader("当前论文" if config.APP_PROFILE == "paper" else "当前文档")
    if config.APP_PROFILE == "paper":
        import paper_ingest

        states = paper_ingest.list_states()
        if not states:
            st.info("还没有论文，先上传 PDF 或把文件放进 data/papers/raw/。")
        for s in states:
            col1, col2 = st.columns([4, 1])
            col1.markdown(
                f"📄 {s.get('filename', '')} · {s.get('status', '')} · {s.get('chunks', 0)} 块"
            )
            if col2.button("🗑", key=f"delp_{s.get('paper_id')}", help="从索引和状态中删除该论文"):
                import paper_index

                paper_index.delete_paper(s["paper_id"])
                (Path(config.PAPERS_DIR) / "_state" / f"{s['paper_id']}.json").unlink(
                    missing_ok=True
                )
                paper_ingest.write_manifest()
                st.rerun()
    else:
        data_dir = Path(config.DATA_DIR)
        current_files = [f.name for f in data_dir.glob("*.*")] if data_dir.exists() else []
        if current_files:
            for fname in current_files:
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"📄 {fname}")
                if col2.button("🗑", key=f"del_{fname}", help="从索引中删除该文档（其余不受影响）"):
                    try:
                        (data_dir / fname).unlink()
                        with st.spinner("正在从索引移除该文档..."):
                            # P1-4：只删该 source 的块，其余文档检索不受影响（不再全量重建）
                            delete_by_source(str(data_dir / fname))
                        st.success(f"已删除并移出索引：{fname}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"错误: {e}")
        else:
            st.info("还没有文档，先上传或把资料放进 data/ 目录。")

    st.markdown("---")
    st.subheader("系统提示词")
    st.caption("使用默认安全提示词" if st.session_state.custom_prompt is None else "已启用自定义提示词")
    custom_input = st.text_area("自定义提示词（留空用默认）", value="", height=120,
                                 placeholder="在此输入额外指令...\n{context} 与 {question} 为必填占位符")
    col_a, col_b = st.columns(2)
    if col_a.button("应用自定义"):
        st.session_state.custom_prompt = custom_input.strip() or None
        st.success("已应用" if custom_input.strip() else "已恢复默认")
        st.rerun()
    if col_b.button("恢复默认"):
        st.session_state.custom_prompt = None
        st.success("已恢复默认")
        st.rerun()

    st.markdown("---")
    # ---- P0-2 清空对话 ----
    st.subheader("💬 对话")
    if st.button("🧹 清空对话", key="clear_btn"):
        st.session_state.messages = []
        # 修复（测试发现）：改为写入空数组而非 unlink——Windows 下文件被占用时
        # unlink 会静默失败（except: pass），导致下次 load_history 又把旧对话读回。
        # 写空数组是幂等操作，无论文件是否存在/是否被占用都能保证清空。
        try:
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception:
            pass
        st.rerun()

# ==================== 主区域 ====================
st.title("📚 CourseRAG")
if config.APP_PROFILE == "paper":
    st.markdown("基于本地 Ollama 的**论文库问答**助手（引用精确到《论文名》第 N 页，数据不出本机）")
else:
    st.markdown("基于本地 Ollama 的**专业课 / 课本问答**助手（数据不出本机）")
if st.session_state.mode == "accurate":
    rerank_part = " + bge-reranker 精排" if config.RERANKER_ENABLED else ""
    st.caption("检索策略：向量召回 top-%d + BM25 融合%s top-%d（精确模式）" % (
        config.RETRIEVE_K, rerank_part, config.RERANKER_TOP_N))
else:
    st.caption("检索策略：纯向量召回 top-%d（快速模式）" % config.RERANKER_TOP_N)

for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    avatar = "🧑‍💻" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        # P1-2 内联引用：助手回答里的 [n] 渲染为可点击锚点，定位到下方来源
        if role == "assistant" and message.get("citations"):
            st.markdown(
                render_answer_html(message["content"], len(message["citations"]), idx),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(message["content"])
        # 引用溯源：带排名 / 相关性分数 / 原文偏移，可展开核对
        if "citations" in message and message["citations"]:
            with st.expander("📌 来源（检索 + 重排结果）", expanded=False):
                for cit in message["citations"]:
                    render_citation(cit, mid=idx)
            # 最后一条助手消息：渲染 👍/👎 反馈按钮（含刷新/重载后状态恢复）
            if role == "assistant" and idx == len(st.session_state.messages) - 1:
                render_feedback_buttons(idx)

_placeholder = "问论文库里的任何问题..." if config.APP_PROFILE == "paper" else "问任何专业课问题..."
if prompt := st.chat_input(_placeholder):
    # 先用「已有历史」拼上下文（不含本次提问），再追加用户消息
    history_text = build_history_text(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 本条助手消息落盘后的索引 = 当前长度（用户已追加），用于锚点 id 去重
    mid = len(st.session_state.messages)
    t0 = time.time()  # #15 看板：整轮问答耗时计时起点
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        try:
            accurate = st.session_state.mode == "accurate"

            # ---- 统一问答入口（评审 P1，2026-08-30）：缓存 / 检索 / 硬拒答 / 生成 /
            # 接地自检 / 缓存写入全部收敛在 rag_core.answer_question，Streamlit /
            # FastAPI / 评估三个出口共用同一闸门；on_stage 回调保留分阶段进度展示 ----
            def _stage(s):
                if s == "retrieve":
                    if accurate:
                        status.update(label="① 向量召回 + BM25 融合 top-%d …" % config.RETRIEVE_K)
                    else:
                        status.update(label="① 向量召回 top-%d …" % config.RERANKER_TOP_N)
                elif s == "rerank":
                    status.update(label="② bge-reranker 精排 top-%d …" % config.RERANKER_TOP_N)
                elif s == "generate":
                    status.update(
                        label="③ 生成回答…" + ("（含接地自检）" if accurate else ""),
                        state="running",
                    )

            # ---- 真实流式回调（评审 P3）：生成中的片段实时渲染到消息占位符。
            # 快速模式在 answer_question 内逐段回调；精确模式因接地自检需要完整
            # 答案（未接地要整段替换为拒答文案），在 answer_question 内自动降级
            # 为缓冲——流式回调不会触发，最终一次性渲染。----
            _stream_parts = []

            def _on_token(piece):
                _stream_parts.append(piece)
                message_placeholder.markdown("".join(_stream_parts) + "▋")

            with st.status("正在处理…", expanded=True) as status:
                res = answer_question(
                    prompt,
                    mode=st.session_state.mode,
                    custom_prompt=st.session_state.custom_prompt,
                    history=history_text,
                    on_stage=_stage,
                    on_token=_on_token,
                    source=st.session_state.get("source_filter"),
                )
                if res["cached"]:
                    status.update(label="⚡ 语义缓存命中", state="complete")
                elif res["rejected"]:
                    status.update(
                        label="🚫 已拒答（检索不相关，signal=%.2f）" % (res["reject_signal"] or 0.0),
                        state="complete",
                    )
                else:
                    status.update(
                        label="✅ 回答完成" + ("" if res["grounded"] else "（未接地→已拒答）"),
                        state="complete",
                    )

            full_response = res["answer"]
            citations = res["citations"]
            docs = res["docs"]

            if res["cached"]:
                st.caption(f"⚡ 语义缓存命中（相似度 {res['cache_sim']:.2f}）")
            # 多轮指代消解（2026-09-18）：历史里有指代时展示改写后的独立检索问题，
            # 让「为什么检索到这份教材」可解释，也方便肉眼检查改写质量
            if res.get("rewritten"):
                st.caption(f"🔍 多轮问题已改写为独立检索问题：{res['rewritten']}")

            # 收尾渲染：真实流式下用完整答案（经 render_answer_html 处理 [n] 引用
            # 锚点）替换流式过程中的原始文本；非流式路径（精确/缓存命中）直接渲染
            message_placeholder.markdown(
                render_answer_html(full_response, len(citations), mid),
                unsafe_allow_html=True,
            )

            # ---- 拒答 / 低置信提示 ----
            # P1 硬拒答：精确模式已判定资料不足（检索不相关 rejected / 回答未接地
            # not_grounded）→ 明确拒答横幅（替代软提示）。
            # 快速模式或精确模式漏拒 → 软提示「资料可能不足」（保留旧 T3 行为）。
            _conf = None if docs is None else get_confidence(docs)
            if accurate and (res["rejected"] or res["not_grounded"]):
                if config.APP_PROFILE == "paper":
                    st.error(
                        "🚫 **已拒答**：根据提供的论文库无法回答该问题"
                        "（检索不相关 / 回答未接地）。建议补充相关论文后重试。"
                    )
                else:
                    st.error(
                        "🚫 **已拒答**：根据提供的课程资料无法回答该问题"
                        "（检索不相关 / 回答未接地）。建议补充相关教材或笔记后重试。"
                    )
            elif _conf is not None and _conf < config.LOW_CONFIDENCE_THRESHOLD:
                st.warning(
                    f"⚠️ **资料可能不足**（相关度 {_conf:.2f} < 阈值 {config.LOW_CONFIDENCE_THRESHOLD}）。"
                    "以下回答仅供参考，建议补充相关课程资料后重试。"
                )

            # ---- P1-3 查询 / 接地日志（每轮追加一行 .query_log.jsonl）----
            # 先落日志拿到 ts（供反馈按钮回写定位）；fast 模式 rejected 记 None
            log_ts = log_query(
                question=prompt,
                mode="精确" if accurate else "快速",
                citations=citations,
                answer=full_response,
                latency_ms=int((time.time() - t0) * 1000),
                rejected=(res["rejected"] if accurate else None),
                rewritten=res.get("rewritten"),  # 多轮指代消解观测
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citations": citations,
                "log_ts": log_ts,
            })
            # ---- P0-2 落盘 ----
            save_history(st.session_state.messages)

            # 展示引用
            with st.expander("📌 来源（检索 + 重排结果）", expanded=False):
                for cit in citations:
                    render_citation(cit, mid=mid)

            # 反馈按钮（仅本次生成的这条）
            render_feedback_buttons(len(st.session_state.messages) - 1)
        except Exception as e:
            st.error(f"出错了: {str(e)}")
            st.info(
                "请确认 Ollama 已启动（ollama serve）且模型已拉取；"
                "若启用了 reranker，需先 pip install sentence-transformers 并联网下载模型权重一次。"
            )
