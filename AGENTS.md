# AGENTS.md · CourseRAG 新窗口入口

> 先读 `README.md`（权威项目说明）。本文件只做导航、当前状态、命令和坑，
> 防止新窗口翻到仓库外的旧口径。

## 1. 项目一句话

论文库 RAG（v2.1）：MinerU 解析论文 PDF → 页/块结构 → 页内切分 → 索引，
回答与引用精确到《论文名》第 N 页，点引用可看页图 + bbox 高亮。
课程版仍保留（`APP_PROFILE=course`），核心链路自实现，不依赖 LangChain。

## 2. 当前权威状态（2026-10-30，v3 定稿）

- 默认 profile：`APP_PROFILE=paper`；集合 `paper_rag`；索引 `./chroma_db_papers`。
- 论文库：**18 篇 / 289 页 / 2834 块 → 1324 块 chunk**（1200 字切分；2026-10-29 由 3 篇扩容）。
  `--audit`：缺页 0、跨页 0、缺章节 20（1.5%）。
- 阈值（论文库专用）：`vec=0.40 / rerank=0.20 / low=0.50`；
  18 篇池下 20 域外 + 20 域内门控：**拒答 90%（18/20）、域内误拒 0%**；
  两道漏拒（BERT MLM 0.702/0.740、Mamba 0.478/0.341）与域内最低分重叠，阈值不可分，需接地判官。
  负样本已按扩容重标（RAG / Llama 2 MMLU 变域内 → 换 DoRA / KTO），旧 95% 与新 90% 非同题集，不可比。
  held-out 10+10：800 切分（3 篇）域外 100%、**域内误拒 20%**；
  1200 切分（18 篇，2026-10-30 只跑一次）域外 **100%**、域内误拒 **10%（1/10，论文作者/期刊题被精排 0.133 拦下）**
  → 仍未达 5% 目标，按纪律不回调阈值。
- 页级评测（**30 题覆盖 18 篇**，2026-10-30 扩题）：paper_hit@8 **0.967（29/30）** / page_hit@8 **0.900（27/30）** /
  page_strict@1 **0.700（21/30）**；原 15 题子集仍是 1.000 / 1.000 / 0.867——**15 题口径偏乐观，对外用 30 题**。
- 回答质量（2026-10-30，qwen2.5:7b 生成 / glm4:9b 裁判）：15 题归一化后 faith 0.603 / prec 0.801 / recall 0.633 /
  cite 0.867；**30 题引用页 0.600（18/30）**：原 15 题 14/15、新增 15 题 4/15，另有 2 题（HyDE/FiD）被硬拒。
  双裁判对照：qwen2.5:7b 当裁判 faith 0.874（明显更宽松）→ 四指标只报趋势。无 `[n]` 答案 0/15。
- **公式数字拆字已修**（v3.1 P0）：`normalize_math_spacing()` 只在 `$...$` 内合并 `0 . 9 8` → `0.98`；
  数学段内拆字 0（1790 段仅 1 处 `$` 跨 `<td>` 误配）；索引 1324→1323 块。
  但 **Q3 仍答 0.999**：7B 忽略排在最前的短块；把首块移到末尾 Q3 3/3 答对，全量 cite 却 0.867→0.733 → 不改顺序。
- 本地模型（v3 定稿）：生成 `LLM_PROVIDER=ollama` / `LLM_MODEL=qwen2.5:7b`（稳态 8.2 s/题，峰值 7737/8188 MiB）；
  裁判 `JUDGE_PROVIDER=local` / `JUDGE_MODEL=glm4:9b`，但 **paper 下 `GROUNDED_VERIFY_ENABLED` 默认 false**
  （8 GB 卡上 7B 生成 + 9B 裁判无法同驻）；关的只是"生成后复核"，检索前硬拒答阈值照常生效，
  所以门控 90% 是**纯阈值口径**。显式 `GROUNDED_VERIFY_ENABLED=true` 仍可开启。
  `OLLAMA_NUM_CTX=8192` **必须设**：默认 4096 会截断 8 块上下文（实测导致 121 个课题 / 28.4 BLEU 答错）。
  裁判选型：glm4:9b 在 30 对手写集 96.7%（FA 1/15、FR 0/15）、12 条真实回答 FA 1/6、FR 0/6；
  qwen2.5:7b 手写集 93.3% 但真实回答 FA 5/6（被高估），gemma3:4b 63.3%。qwen3 系列不能当裁判（思考链吃满 `num_predict`）。
  关闸实测代价：7B 对同一题（β2 取值）部分跑次答成 0.999（原文 0.98，也有跑次答对），
  裁判同文判定不稳定（跨轮精度约 50%），故只保留开关、不做默认硬门控。
  8GB 装不下生成+裁判，串行换入实测 4.2~5s；精排 + glm4:9b + 8192 峰值 7761 MiB。
- 引用编号：paper 模式下 context 每段前直接标 `[n]`（`rag_core._compose_prompt`），否则模型会数错段号、答对但引错页。
- MinerU 2.7.6 在 **`E:\mineru_venv`**（ASCII；中文路径下 fasttext 打不开 Unicode venv），
  `.env` 里 `MINERU_PYTHON=E:/mineru_venv/Scripts/python.exe`；模型源默认 modelscope。
- 课程版：tag `v1.1-course-final` / 分支 `archive/course-v1.1`；课程索引 `chroma_db_m3` 未动。
- 测试：`test_paper_normalize / test_paper_chunk / test_paper_citation / test_api_papers / test_config_profile` 已入 CI；
  离线全量 **84 passed**。
- 断点与恢复指引：仓库外 `CourseRAG-实现记录.md` 末尾「断点」节（先读它）。

## 2.1 未完成（恢复后从这里开始）

- v3 P0 已完成：paper 默认关接地门 + `test_config_profile.py` + 文档口径。
- v3 P1：`eval_paper_quality.py`（四指标 + `answer_cite_page_acc`，两段跑）；**未开始**。
- v3 P2：`eval_paper.py --group reject --heldout`（1200 切分）只跑一次；分批本地 commit（push 等用户点头）。
- 仍未闭：云端裁判 key 401（带接地的 UI 验收待重跑）；held-out 域内误拒 20%（800 切分）未达标；
  关闸后 7B 的数值错答会直接返回（见 §2）；最终 commit/push 未做。
- 历史（已完成，留档）：M5 的 qwen3:4b/llama3.1:8b 读数已被 v3 的 qwen2.5:7b 方案取代；
  `qwen3:8b` 未下载（用户决定不急），8B 档延迟/显存 UNVERIFIED。
  M6 的解析抽检 15/15、MCP paper e2e、CI、Docker slim 981 MB `/health` 200 均已完成。

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
| 数据 / 索引 | `data/papers/`（raw/parsed/normalized/meta/_state）、`chroma_db_papers/`（活跃索引）；课程索引 `chroma_db_m3/` 归档 |
| 论文管线 | `mineru_adapter.py`、`paper_ingest.py`、`paper_normalize.py`、`paper_chunker.py`、`paper_index.py`、`page_render.py` |
| 课程管线 | `_extract_images.py`（抽图）、`_caption_images_v2.py`（caption） |
| 评估 | 论文：`eval_data_paper.py`、`eval_paper.py`；课程：`eval_local.py`、`eval_data*.py`、`eval_test.py`、`eval_heldout.py`、`run_ab.py`、`compare_hybrid.py` |
| 运维 | `backup_index.py`、`ingest_ocr.py`、`dashboard.py`、`check_env.py` |
| 前端 | `frontend/`（Vite + React + AntD；`npm run dev`，proxy `/api` → 127.0.0.1:8000） |
| 测试 | 课程 6 套 + 论文 4 套（`test_paper_*`、`test_api_papers`） |

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

# 论文库（v2.1）
python paper_ingest.py --dir data/papers/raw --only-new   # MinerU 解析 → normalized
python paper_index.py --rebuild                            # normalized → 索引
python paper_index.py --audit                              # 100% 有页 / 0 跨页
python eval_paper.py --group page --gate                   # 页级评测 15 题
python eval_paper.py --group reject --scan                 # 阈值扫描
python eval_paper.py --group reject --gate                 # 拒答门控
python eval_paper.py --group reject --heldout              # held-out 只跑一次
python -m pytest test_paper_normalize.py test_paper_chunk.py test_paper_citation.py test_api_papers.py -v

# 前端（论文库）
cd frontend; npm run dev      # Vite proxy /api → 127.0.0.1:8000

# 课程版评估（旧口径，仍可用）
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
