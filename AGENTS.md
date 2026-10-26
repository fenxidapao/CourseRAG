# AGENTS.md · CourseRAG 新窗口入口

> 先读 `README.md`（权威项目说明）。本文件只做导航、当前状态、命令和坑，
> 防止新窗口翻到仓库外的旧口径。

## 1. 项目一句话

垂直领域 RAG：课程笔记 / 教材 PDF·PPTX 建库，问答带来源引用（文本段落 + 教材原图）。
当前 v1.1 口径（2026-10-16 起）：嵌入与精排在本机，生成与裁判默认走云端；
仍可切回全本地。核心链路自实现，不依赖 LangChain。

## 2. 当前权威状态（2026-10-26）

- 活跃索引：`chroma_db_m3/`，嵌入 `bge-m3`，**1434 块**（1219 文本 + 215 图像 caption）。
- 当前 `.env` 覆盖：`EMBEDDING_MODEL=bge-m3`、`PERSIST_DIRECTORY=./chroma_db_m3`、
  `LLM_PROVIDER=openai` / `deepseek-v4.1-flash`、`JUDGE_PROVIDER=openai` / `kimi-k2.6`。
- v1.1 云端复测（对外引用以 README「评估」为准）：
  冻结集 24 题 faith **0.981** / rel 0.931 / prec 0.943 / recall 0.916；
  held-out A v2（38 题）faith **0.996** / recall **0.975** → PASS；
  B 组拒答 **20/20**；图片命中 **8/8**（全部 rank=1）。
  换生成/裁判模型后与本地模式数字**不可直接比较**。
- 测试：6 套测试文件；CI 跑 `test_core / test_api / test_mcp / test_retrieval`，
  `test_visual_rerank / test_llm_provider` 为本地离线单测。
- 旧交接文档（HANDOVER / EVALUATION / INTERVIEW 等）已移出仓库，**不要引用其中的旧数字**。

## 3. 建议阅读顺序

1. `README.md` —— 功能、配置、API、Docker、评估与门控。
2. 本文件 —— 当前状态、命令、坑。
3. `config.py` + `rag_core.py` —— 参数与主链路。
4. 按任务看：`main.py` / `app.py`（出口）、`eval_*.py`（评估）、`test_*.py`（行为契约）。

## 4. 目录地图（只列关键）

| 类别 | 文件 |
|---|---|
| 核心 | `config.py`、`rag_core.py`、`rag_search_utils.py`、`reranker.py`、`visual_rerank.py`、`semantic_cache.py`、`grounding.py`、`llm_provider.py` |
| 出口 | `app.py`（Streamlit）、`main.py`（FastAPI）、`api_schemas.py`、`api_security.py`、`mcp_server.py` |
| 数据 / 索引 | `data/`（md + `images/` + `教材/`）、`chroma_db_m3/`（活跃索引） |
| 语料管线 | `_extract_images.py`（抽图）、`_caption_images_v2.py`（caption） |
| 评估 | `eval_local.py`、`eval_data*.py`、`eval_test.py`、`eval_heldout.py`、`run_ab.py`、`compare_hybrid.py` |
| 运维 | `backup_index.py`、`ingest_ocr.py`、`dashboard.py`、`check_env.py` |
| 测试 | `test_*.py`（6 套） |

## 5. 常用命令（PowerShell）

```powershell
# 环境：建议 venv；依赖见 requirements*.txt
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 自检 / 启动
python check_env.py
ollama serve
ollama pull bge-m3
streamlit run app.py
uvicorn main:app --host 127.0.0.1 --port 8000

# 测试
python test_core.py
python -m pytest test_api.py test_mcp.py test_visual_rerank.py test_llm_provider.py -v
python -m pytest test_retrieval.py -v        # 需 Ollama + 已建索引

# 评估
python eval_test.py                         # 冻结测试集 24 题
python eval_heldout.py --a-v2               # A 组 v2 + 图片命中率
python eval_heldout.py --bc-only            # B/C 拒答
```

索引备份 / 恢复、语料接入、看板：见 README「运维」节。

## 6. 坑（先看这些）

- **Ollama 必须运行**：嵌入 `bge-m3` 走本机；精排需要 `HF_ENDPOINT=https://hf-mirror.com`。
- **索引与嵌入模型必须匹配**：当前 `bge-m3` 配 `chroma_db_m3/`；旧 `chroma_db/`（nomic 索引）
  已移到仓库外。换模型或改语料 / caption / 切分参数后必须重建索引，不能混用。
- **Windows 绑定 `127.0.0.1`**：`localhost` 可能解析到 IPv6 导致连不上。
- **8 GB 显存**：本地生成 + 本地裁判同开可能让 CUDA 调用挂住（WDDM 不报 OOM）；
  默认云端就是为了避开它。切全本地时一次只跑一个大模型。
- **数据边界**：默认云端生成 / 裁判会把**问题与检索正文**发往 API；嵌入不出本机。
  `.env` 含 key，已 gitignore，绝不提交。
- **评估纪律**：`eval_data_test.py` 冻结集只回归、不调参；调参只能用 held-out A。
  改语料 / 检索参数后必须重验 A/B/C 门控。
- **重跑评估先移走同名 CSV**：`eval_local.py` 有断点续跑，旧 CSV 会让题目被跳过。
- **caption 两份都在用**：`IMAGE_EMBED_CAPTIONS_CSV`（v1 短描述）供嵌入/BM25，
  `IMAGE_CAPTIONS_CSV`（v2 长描述）供生成/引用，不要删其中任何一份。
- **视觉重排默认关闭**：`VISUAL_RERANK_ENABLED=false`；开启前先跑 held-out，且需要 VLM key。

## 7. 仓库外归档

历史文档、一次性实验、日志、旧索引、个人材料已移到：

```
../CourseRAG-仓库外整理-2026-10-26/
├── 01_个人材料-面试与简历/
├── 02_历史文档/          # HANDOVER / EVALUATION / INTERVIEW / 点评 / 方案
├── 03_一次性实验脚本/    # b1_* / _b3_* / _probe_* / 旧 compare / benchmark
├── 04_运行日志/
├── 05_评估产物/          # v1.1 CSV + 历史评估产物
├── 06_旧索引-可重建/     # chroma_db（nomic）/ chroma_db_v1
├── 07_旧归档/            # 原 _archive/
└── 08_旧工作记忆/        # 原 .workbuddy/ + .chat_history.json
```

需要旧证据时可以按目录去查，但**不要**把旧数字当成当前口径写回 README。
要复用某个实验脚本，先明确它对应的索引 / 模型 / 数据版本。

## 8. 修改纪律

- 改检索 / 生成 / 配置后：至少跑 `test_core.py` + 相关 pytest；涉及质量口径再跑 held-out。
- 不提交：`.env`、`data/教材/`、`chroma_db*/`、日志、评估 CSV。
- 本地 `git commit` 可自主进行；**`git push` 必须先征得用户确认**。
- README「当前官方基线」是唯一对外数字口径；改基线时同步 README，不要只改代码注释。
