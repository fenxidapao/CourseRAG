# CourseRAG · 专业课/课本问答助手

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.61-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker&logoColor=white)

一个垂直领域 RAG 问答应用，面向「计算机／数学专业课、课本知识问答」场景：
把课程笔记与教材 PDF/PPTX 建成知识库，提问时给出**带来源引用**的回答，答案可回溯到具体段落与教材原图。

**运行模式（2026-10 起）**：嵌入与精排留在本机（Ollama），**生成与裁判默认走云端 OpenAI 兼容 API**。
原因是本机只有 8 GB 显存：本地生成 `llama3.1:8b`（约 4.9 GB）+ 本地裁判 `qwen2.5:7b`（约 4.7 GB）装不下，
Windows WDDM 不报 OOM，而是把 CUDA 调用挂住（日志长时间无输出、CPU 仍上涨）。
仍可用 `LLM_PROVIDER=ollama` + `JUDGE_PROVIDER=local` 切回全本地，代价是显存与延迟。

> ⚠️ **数据边界**：切云端后，**用户问题与检索到的正文会发送给中转 API**；嵌入阶段不出本机。
> 需要严格「数据不出本机」时，请把 `LLM_PROVIDER` 与 `JUDGE_PROVIDER` 都设为本地。

RAG 核心（切分 / 加载 / 嵌入 / 向量检索 / 词面检索 / 融合 / 精排 / 接地自检）与评估体系均为自实现，
RAG 核心链路直接依赖仅 `ollama` + `chromadb` + `sentence-transformers` 三项，已彻底移除 LangChain 全家桶
（完整运行依赖见 `requirements.txt`，共 9 项直接依赖）。

## 这是什么问题

专业课本动辄几百页，而通用大模型对课本里的**具体术语、编号、图示**往往答不准，也说不出来源。
直接丢给云服务又有两个顾虑：教材有版权、笔记是私人的。

CourseRAG 的做法是：把课程资料**全部放在本机**建索引，用本地模型检索 + 生成，
再用「教材级过滤 + BM25 词面召回 + 精排 + 接地自检」把召回质量拉起来，
最后让每条回答都挂上可点击的来源编号——包括教材里的**插图**。

典型使用场景：

- 复习时问「2NF 和 3NF 的区别是什么？」，拿回带原文出处的对比说明
- 问「教材里那幅进程状态转换图怎么解释？」，命中图像块后在引用区直接看到原图
- 问知识库没覆盖的题目，系统**明确拒答**并说明未覆盖，而不是编一段看起来合理的答案

## 功能

- 📚 **多格式课程资料接入**：PDF / PPTX / TXT / MD / DOCX；扫描版 PDF 建库时告警跳过，不静默产生空索引
- ✂️ **递归语义切分**：800 字符 + 150 重叠，含中文分隔符感知，避免切断上下文
- 🔍 **真混合检索**：向量与自实现 BM25 **各自对全库独立召回** top-50 → RRF 倒数排名融合（候选池 16）→ bge-reranker 精排 top-8
- 🏷️ **教材级元数据过滤**：问题命中教材关键词时自动路由，向量路与 BM25 路各自带 `where` 过滤后再融合；过滤后候选为空自动降级全库（防过度过滤）；API 与 UI 两个出口可显式指定教材，**MCP 出口不支持教材过滤**（`course_retrieve` 只有 `query` / `top_k` / `mode` 三个入参，见「MCP 工具层」）
- 🖼️ **多模态语料（caption 路线）**：PyMuPDF bbox 渲染剪裁提取教材内嵌图像（自动合成矢量层、内容哈希去重）+ 本地 qwen2.5-VL 生成中文描述，图像描述作为 image 类型 chunk 与文本**统一索引**
- 📌 **引用溯源**：来源带相关性分数与原文偏移；回答正文内联 `[n]` 上标（半角／全角均可）可点击跳转；命中图像块时引用区展示教材原图
- 💬 **双出口流式问答**：Streamlit 打字机界面 + FastAPI `POST /query/stream`（SSE）；底层是 ollama `stream=True` 的真实逐 token 回调，**引用列表在生成开始前就送达**
- 🔄 **Query 改写（多轮指代消解）**：history 非空时先用 LLM（temperature=0）把「历史 + 当前问题」改写为独立检索问题；失败／超时／输出口水话自动回退原问题，单轮零开销
- 🔌 **MCP 工具层**：基于官方 `mcp` SDK，暴露 `course_retrieve`（是什么）与 `list_documents`（你有什么）两个语义正交工具
- ➕ **增量索引 + 去重**：稳定 id（`md5(source::start_index)`）自动去重，上传仅增量 upsert；按 source 删除单文档，其余不受影响
- ⚡ **语义缓存**：问句 embedding 余弦 ≥ 0.93 时跳过检索 + 精排 + 生成直接返回历史答案；索引重建后自动失效
- 📊 **本地自实现评估**：本机 Ollama 当裁判，四指标（faithfulness / answer_relevancy / context_precision / context_recall），零额外依赖
- 🧪 **held-out 门控**：A 组（语料内 + 图片题）／B 组（语料外应拒答）／C 组（对抗与注入）三组绝对指标 + 拒答率双门控
- 📈 **可观测看板 + 反馈闭环**：查询日志落 `.query_log.jsonl`，看板出 KPI / 延迟 P95 / 拒答清单；👎 可回流成评估集且不动冻结基线
- 🔒 **配置外置 + 多用户预留**：全部参数从 `.env` 读取；鉴权 / IP 限流 / 上传上限已挂载但默认关闭
- ✅ **测试与 CI**：四套测试 + GitHub Actions 3.11 矩阵（py_compile + 测试 + held-out 检索质量门禁）

## 目录结构

```
CourseRAG/
├── config.py          # 所有参数集中从 .env 读取（配置外置；含 HYBRID_ENABLED / BM25_TOP_K）
├── rag_core.py        # RAG 核心：加载切分 / 嵌入建库 / 混合检索(hybrid_retrieve) + 重排 / 增量索引(add_documents/delete_by_source) / 拼链
├── rag_search_utils.py# 零依赖纯算法（P2-1 抽出）：分词 / BM25 / RRF 融合 / 稳定 id，供 test_core 独立测试
├── reranker.py        # bge-reranker 跨编码器重排（混合检索的精排段；默认开——难 query 靠 RETRIEVE_K=50 + reranker 纠正召回失败）
├── mcp_server.py      # MCP stdio 工具层：官方 mcp SDK（MCPServer，2.x 中 FastMCP 更名），course_retrieve 与 /retrieve 同链路
├── query_log.py       # P1-3 查询日志：每轮落 .query_log.jsonl
├── app.py             # Streamlit 演示界面（含来源展示与内联 [n] 引用跳转）
├── main.py            # FastAPI 服务入口（部署层，与 app.py 共用同一套 rag_core）
├── api_schemas.py     # FastAPI 请求/响应 Pydantic 模型（Swagger 文档自动生成）
├── requirements-api.txt  # API 服务最小依赖（不含 streamlit / torch，供 Docker 镜像）
├── Dockerfile         # 多阶段构建（python:3.11-slim）
├── docker-compose.yml # 一键部署：Ollama + API 双服务
├── .dockerignore      # 镜像排除项（data/、chroma_db/、.env 等）
├── test_core.py       # P2-1 核心算法单测（零依赖，可在任意干净 Python 环境实跑）
├── eval_local.py      # 本地自实现评估（Ollama 当裁判，4 指标，零额外依赖；取代已废弃的 eval.py）
├── eval_data.py       # 评估小样本（question + reference 标准答案）
├── run_ab.py          # 混合检索 A/B 断点续跑 runner（中断后重跑自动续，隔离 reranker 测 BM25 纯贡献）
├── compare_hybrid.py  # 混合检索 vs 纯向量 A/B 对比（整体 + 区别类子集 + 逐题 Δ + 报告）
├── data/              # 课程资料（示例 md + 教材/ 子目录：真实教材 PDF/PPTX 课件）
├── chroma_db/         # 向量库持久化目录（自动生成，已 gitignore）
├── .query_log.jsonl   # 查询日志（自动生成，已 gitignore）
├── .env.example       # 配置模板
└── requirements.txt
```

## 快速开始

**前置**：Python 3.11+；Ollama 已安装并运行（本地 RAG 的固有前提——模型权重无法打包进仓库）。
下面的路径示例是 **Windows / PowerShell**，Linux 与 macOS 未验证。

```powershell
# 1. 环境
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. 配置（默认值即可跑通，详见「配置」章节）
cp .env.example .env

# 3. 模型（已拉取可跳过）
ollama serve
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 4. 启动
streamlit run app.py
```

**成功信号**（照此验证，不要只看进程是否起来）：

1. 浏览器打开 `http://localhost:8501`
2. 侧栏确认 `data/` 下示例 md 存在
3. 点「🔄 全量重建索引（data/ 全部）」，等待索引完成
4. 问 `什么是 ACID？` → **应返回带来源引用的回答**，且引用可展开看到原文片段

Windows 下也可直接双击 `start.bat`：它按 `.\venv\Scripts\python.exe` → `%COURSERAG_PYTHON%` → PATH 中 `python` 的顺序探测解释器，无硬编码路径、无中文注释（GBK 安全），任意 Windows 机器可跑。

### 两种检索模式

| 模式               | 链路                                                         | 用途                                         |
| ------------------ | ------------------------------------------------------------ | -------------------------------------------- |
| `accurate`（默认） | 向量 + BM25 融合 + reranker 精排 → **硬拒答 + 接地自检**     | 正常问答；难 query 靠扩池 + 精排纠正召回失败 |
| `fast`             | 纯向量召回 + 软提示（`low_confidence`），无硬拒答、无接地自检 | 降级档：GPU 不可用 / 显存不足 / 高并发       |

### 干净部署（换机器 / 给别人用 / 从头装）

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull nomic-embed-text
ollama pull llama3.1:8b
streamlit run app.py
```

依赖已精简：RAG 核心链路只剩 `ollama` + `chromadb` + `sentence-transformers` 三项（已彻底去除 langchain 全家桶）；
`requirements.txt` 共 9 项直接依赖（API 服务另用 `requirements-api.txt`，同为 9 项、不含 streamlit 与 torch），
显式锁定 `ollama==0.6.2` 与 `chromadb==1.5.9`，降低换机器安装失败的机率。
注意：别人使用时仍需**自备 Ollama 与模型权重**（无法打包进仓库），这是本地 RAG 的固有前提。

## 配置

全部参数从 `.env` 读取（`config.py` 集中定义），`.env.example` 是可直接复制使用的模板。
**下表「默认」列逐个取自 `config.py` 的 `os.getenv(变量名, 默认值)`**——即 `.env` 里缺这一项时的实际取值，
照表填不会与代码行为脱节。留空（`VAR=`）不等于取默认值：数值型变量留空会让 `int("")` / `float("")` 抛错。

**模型 / 服务**

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `EMBEDDING_MODEL` | `nomic-embed-text` | 嵌入模型（走 Ollama） |
| `LLM_PROVIDER` | `ollama` | 生成提供方：`ollama`=本地 / `openai`=OpenAI 兼容云端 |
| `LLM_MODEL` | `llama3.1:8b` | 生成模型名（随 provider 变化；云端推荐 `deepseek-v4.1-flash`） |
| `LLM_API_BASE` | `https://kuaipao.pro/v1` | 云端 OpenAI 兼容接口（仅 provider=openai） |
| `LLM_API_KEY` | 空 | 云端 key；留空时自动读 `~/.dsh/.credentials.yaml` 的 `KUAIPAO_API_KEY`，**不写入仓库** |
| `LLM_DISABLE_THINKING` | `true` | 云端关闭思维链；实测推理模型会把预算烧在 reasoning 上导致空回答 |
| `LLM_MAX_TOKENS` / `LLM_MAX_TOKENS_CAP` | `2048` / `8192` | 生成预算下限与自动扩容上限 |
| `LLM_TEMPERATURE` | `0.2` | 生成温度（低温度压制幻觉） |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像（reranker 权重下载用） |

**切分 / 路径**

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `CHUNK_SIZE` | `800` | 每块字符数（教材正文段落约 200-800 字符） |
| `CHUNK_OVERLAP` | `150` | 块间重叠，避免切断句子 / 上下文 |
| `DATA_DIR` | `./data` | 课程资料目录 |
| `PERSIST_DIRECTORY` | `./chroma_db` | 向量库持久化目录 |
| `PDF_MIN_CHARS_PER_PAGE` | `30` | 均字符/页低于此值视为无文字层（扫描版），建库时告警跳过 |
| `GARBLED_DROP_THRESHOLD` | `0.3` | 块级乱码字符占比超此值 → 整块丢弃，否则剥离乱码保留正文 |

**检索 / 重排**

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `HYBRID_ENABLED` | `true` | 向量 + BM25 混合检索开关 |
| `RETRIEVE_K` | `50` | 向量通道独立召回数（难 query 正确块常排 30+，池不够捞不到） |
| `BM25_TOP_K` | `50` | BM25 通道独立召回数（对全库打分，不复用向量候选池） |
| `RERANK_CANDIDATE_K` | `16` | RRF 融合后的候选池大小，供精排消费 |
| `RERANKER_ENABLED` | `true` | bge-reranker 精排开关（关掉也仍是混合检索，非纯向量） |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 精排模型（多语言，中文友好） |
| `RERANKER_TOP_N` | `8` | 精排后进 LLM 的块数（官方评估基线即 top-8 口径） |
| `RERANKER_DEVICE` | `auto` | 精排推理设备 |
| `SOURCE_FILTER_ROUTING_ENABLED` | `true` | 问题命中教材关键词时自动路由 |
| `ENUM_EXPAND_ENABLED` | `false` | 枚举类问题的候选池扩展（默认关） |
| `ENUM_RERANK_TOP_N` | `14` | 枚举类问题进 LLM 的块数（常规走 `RERANKER_TOP_N`） |

**拒答 / 接地自检**

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `HARD_REJECT_VECTOR_THRESHOLD` | `0.55` | 向量分低于此值判为语料外 → 硬拒答 |
| `HARD_REJECT_RERANK_THRESHOLD` | `0.30` | 精排分低于此值判为语料外 → 硬拒答 |
| `GROUNDED_VERIFY_ENABLED` | `true` | 接地自检开关 |
| `GROUNDED_SCORE_THRESHOLD` | `0.60` | 接地自检阈值，未达标记 `low_confidence` |
| `LOW_CONFIDENCE_THRESHOLD` | `0.69` | fast 模式低置信标记阈值 |

**缓存 / 改写**

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `QUERY_REWRITE_ENABLED` | `true` | 多轮指代消解（history 非空时生效，单轮零开销） |
| `SEMANTIC_CACHE_ENABLED` | `true` | 语义缓存开关 |
| `SEMANTIC_CACHE_SIM` | `0.93` | 语义缓存命中阈值（embedding 余弦） |
| `SEMANTIC_CACHE_MAX` | `128` | 语义缓存 LRU 容量（多副本部署需换 Redis） |

**评估裁判**

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `JUDGE_PROVIDER` | `local` | 裁判来源：`local` = 本机 Ollama / `openai` = 云端（兼容旧写法 `deepseek`） |
| `JUDGE_MODEL` | `kimi-k2.6` | 裁判模型。云端实测：kimi-k2.6 3/3、1.4~1.6s，优于 glm-5.2 / qwen3.8-flash |
| `JUDGE_API_BASE` | `https://kuaipao.pro/v1` | 裁判云端接口（仅 provider=openai） |
| `JUDGE_API_KEY` | 空 | 裁判 key；留空回退同一中转 key，**不写入仓库** |
| `JUDGE_MAX_TOKENS` | `4096` | 裁判 JSON 输出预算（含自动扩容上限 `JUDGE_MAX_TOKENS_CAP`） |

> 鉴权 / 限流 / 上传上限 / 跨域四类变量见下方「多用户预留扩展点（Gate 2）」表；完整清单以 `.env.example` 与 `config.py` 为准。

## API 服务（FastAPI + Docker 部署）

把 `rag_core` 的「索引构建 / 检索 / 重排 / 生成」暴露为 RESTful 端点，
与 Streamlit 界面**共用同一套核心模块**，保证「界面看到的来源 = API 返回的来源」。
交互式文档：启动后访问 `http://localhost:8000/docs`。

### 端点一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 服务元信息（模型 / 端点清单） |
| GET | `/health` | 依赖 / Ollama / Reranker 权重自检（复用 `health.py`） |
| POST | `/query` | 问答：检索 + 重排 + 生成，返回回答 + `[n]` 引用来源 + 耗时 |
| POST | `/query/stream` | 流式问答（SSE）：`start → metadata → token* → done`，同一套管线，引用在生成前送达 |
| POST | `/retrieve` | 仅检索 + 重排（不生成），供前端预检索 / 调试 / 评估 |
| GET | `/documents` | 列出 `data/` 文档与各自块数 |
| POST | `/documents` | 上传课程资料（multipart）→ 保存 + 增量索引（稳定 id 去重） |
| DELETE | `/documents/{name}` | 删除文件 + 只从索引移除该 source 的块 |
| POST | `/index/rebuild` | 全量重建索引（`data/` 全部） |
| GET | `/index/stats` | 向量库块数 / 源文档数 |

**流式问答（SSE）**：`/query/stream` 与 `/query` 共用同一套管线（缓存 / 检索 / 硬拒答 / 接地自检 / 日志口径完全一致），只把结果改为逐帧推送。

| 事件 | 时机 | data 字段 |
|---|---|---|
| `start` | 请求受理 | `mode` / `question` |
| `metadata` | 检索 + 精排完成、拒答判定出炉（**生成尚未开始**） | `citations` / `rejected` |
| `token` | 生成中（仅 fast 模式逐个文本片段） | `text` |
| `done` | 终态 | `answer` / `citations` / `latency_ms` / `grounded` / `low_confidence` / `cached` / `rejected` |
| `error` | 管线异常（HTTP 已是 200，失败只能走事件） | `detail` |

两个设计取舍值得说明：

1. **引用为什么在 `metadata` 而不是 `done`**：检索一完成来源就确定了，让前端在正文出现前先渲染来源列表，用户感知更快。拒答同理——`rejected=true` 时前端立刻能提示「知识库未覆盖」，不必干等生成。这也是 `rag_core.answer_question` 增加 `on_retrieved` 回调的原因。
2. **为什么用 POST 而不是浏览器原生 `EventSource`**：EventSource 只支持 GET，问题文本塞 query string 既受长度限制又会进访问日志。业界流式接口（OpenAI / Anthropic）同样是 POST + SSE，客户端用 `fetch` + `ReadableStream` 消费。

```powershell
curl -N -X POST http://localhost:8000/query/stream -H "Content-Type: application/json" -d "{\"question\":\"什么是ACID\",\"mode\":\"fast\"}"
```

> ⚠️ 精确模式（默认 accurate）**不逐 token 推送正文**：接地自检必须拿到完整答案才能判定，
> 否则已展示的未验证内容无法撤回。此时 `metadata`（引用 + 拒答）与 `done` 照常推送，
> 前端可据此展示「正在生成…」。

**实测时序**（本机 RTX 4060 Laptop / 1434 块语料 / n=3；复现脚本 `_sse_bench.py` 是**本地未跟踪文件，不随仓库分发**，克隆本仓库后无法直接复跑该脚本）：

| 模式 | metadata（引用送达） | 首 token | 端到端 |
|---|---|---|---|
| fast | **54ms** | **743ms**（占端到端 19%） | 3.96s |
| accurate | 0.4 ~ 1.4s（冷启动首轮 37.3s，含 reranker 权重加载） | —（缓冲，见上） | **34.9s** |

> **关于「模型 API 密钥」**：生成与裁判默认走云端 OpenAI 兼容接口，需要一个 API key。
> key 只从环境变量或 `~/.dsh/.credentials.yaml` 读取，**不写入仓库、不进日志、错误体回显会自动脱敏**。
> 相关运行时配置：`LLM_PROVIDER` / `LLM_MODEL` / `LLM_API_BASE`、`JUDGE_PROVIDER` / `JUDGE_MODEL` / `JUDGE_API_BASE`、
> `OLLAMA_BASE_URL`（嵌入与 reranker 仍用它）、RAG 调参与 `HF_ENDPOINT`
> （见 `.env`）。若未来切云端模型，只需在 `config.py` 增加对应 key 并改 `rag_core` 的调用层。

### 本地运行（与 Streamlit 同机）

```powershell
# 1. 装 API 依赖（与 requirements.txt 版本对齐；若已装过核心依赖则只需 fastapi 三项）
pip install -r requirements-api.txt

# 2. 启动（需 Ollama 已运行且模型已拉取，见「快速开始」）
uvicorn main:app --host 0.0.0.0 --port 8000
# 浏览器打开 http://localhost:8000/docs（Swagger UI）
```

### 多用户预留扩展点（Gate 2）

> 决策：先本地/局域网部署让同学可用，**不重写后端**；以下扩展点默认关闭，需要多人公网暴露时再开。
> 实现：`api_security.py`（`require_auth` 全局依赖 + `RateLimitMiddleware` IP 滑动窗口），`main.py` 已挂载（no-op）。

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `AUTH_ENABLED` | `false` | 开启后所有端点需 `X-API-Key` 头（值 = `API_KEY`） |
| `API_KEY` | 空 | 开启鉴权时设置的密钥 |
| `RATE_LIMIT` | `0` | 每 IP 每分钟请求上限（如 `60`），`0` = 不限；**必须填整数**——留空会让 `config.py` 的 `int("")` 抛 `ValueError`，服务直接起不来 |
| `CORS_ORIGINS` | `*` | 收紧跨域（生产填具体域名，如 `https://qa.example.com`） |
| `MAX_UPLOAD_MB` | `100` | `/documents` 上传大小上限（MB），超限返回 413；超大扫描版 PDF 走本地 `ingest_ocr.py` |

```bash
# 示例：开启鉴权 + 限流（docker compose 场景改环境变量或 .env）
AUTH_ENABLED=true API_KEY=your-secret RATE_LIMIT=60 python -m uvicorn main:app --port 8000
# 带密钥请求
curl -H "X-API-Key: your-secret" http://localhost:8000/health
```

### Docker 部署（双模式）

前置条件：**宿主 Ollama 已启动且模型已拉取**。compose 采用 API 直连宿主 Ollama
（容器内不再拉一套模型，省 ~5GB 重复下载，也避免 11434 端口冲突）。

```bash
# 模式一 · slim（默认）：轻量，无 reranker，精确模式 = 向量 + BM25 混合
docker compose up -d --build

# 模式二 · full：含 bge-reranker 精排（镜像 +~2GB，首次需下载权重 ~2.2GB）
docker compose --profile full up -d --build

# 验证
curl http://localhost:8000/health   # slim（8000）
curl http://localhost:8001/health   # full（8001）
```

- **双模式通过 `Dockerfile` 的 `WITH_RERANKER` build-arg 控制**（`docker-compose.yml` 已透传）：
  `api`（slim）只装 `requirements-api.txt`；`api-full` 追加 CPU 版 torch + `sentence-transformers`
  （`requirements-rerank.txt`）并置 `RERANKER_ENABLED=true`。reranker 权重缓存挂 `huggingface_cache` 卷，
  宿主机已下载过则复用。
- **非 root 运行**：镜像内以 `uid 10001(app)` 启动。Linux 宿主若挂载目录属主不匹配导致不可写，
  执行 `sudo chown -R 10001:10001 ./data ./chroma_db`；Windows / macOS Docker Desktop 无需此步。
- **Linux 宿主机差异**：`host.docker.internal` 是 Docker Desktop 特性，Linux 默认不解析。三选一：
  1. compose 里给服务加 `extra_hosts: ["host.docker.internal:host-gateway"]`（推荐）
  2. `network_mode: host`（此时 `OLLAMA_BASE_URL` 写 `http://127.0.0.1:11434`，但 `ports` 映射失效）
  3. 把 `OLLAMA_BASE_URL` 换成宿主在 docker0 网桥上的实际 IP（如 `172.17.0.1`）
- **若宿主 Ollama 只监听 127.0.0.1 导致容器连不上**：设 `OLLAMA_HOST=0.0.0.0:11434` 重启 Ollama
  （注意这会把 Ollama 暴露到局域网，仅在可信网络下使用）。
- **已知坑**：多阶段构建只拷 `site-packages`，pip 装的可执行脚本不在 `/usr/local/bin`
  → Dockerfile 的 CMD 必须用 `python -m uvicorn`，不能用裸 `uvicorn`。
- API 启动时自动预热向量库（空库 + 有资料时自动重建）；Ollama 未就绪不会阻塞服务启动，`/health` 会明确报告缺什么。

### 验证（curl 示例）

```bash
# 健康检查
curl http://localhost:8000/health

# 快速问答（纯向量 top-5，秒级）
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question":"什么是 ACID？","mode":"fast"}'

# 精确问答（向量 + BM25 融合 + reranker 精排，救难 query；本地环境 reranker 默认开）
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"question":"2NF 和 3NF 的区别是什么？","mode":"accurate"}'

# 仅检索（不生成，返回来源列表）
curl -X POST http://localhost:8000/retrieve -H "Content-Type: application/json" \
  -d '{"question":"死锁的四个必要条件","mode":"accurate"}'

# 上传课程资料（增量索引）
curl -X POST http://localhost:8000/documents -F "file=@./data/数据库系统-事务与ACID.md"

# 列出文档 / 查看索引统计
curl http://localhost:8000/documents
curl http://localhost:8000/index/stats
```

`/query` 响应结构（`answer` 中的 `[n]` 对应 `citations` 的 `rank`，可直接渲染跳转）：

```json
{
  "answer": "ACID 是事务的四个特性……[1][2]",
  "mode": "fast",
  "citations": [
    {"rank": 1, "source": "数据库系统-事务与ACID.md", "content": "……", "score": 0.31, "start_index": 1024}
  ],
  "latency_ms": 1523,
  "grounded": true
}
```

## MCP 工具层

`mcp_server.py` 是独立的 stdio 服务（与 Streamlit / API 并存，共用同一套检索实现），
基于官方 `mcp` SDK 的 `MCPServer`——即 2.x 中 FastMCP 的更名：类型签名自动生成 `inputSchema`，
`ToolError` 对应 `isError` 工具结果。

| 工具              | 回答的问题 | 入参                                                         | 返回                                      |
| ----------------- | ---------- | ------------------------------------------------------------ | ----------------------------------------- |
| `course_retrieve` | 是什么     | `query` / `top_k`（1-20，服务端强制钳制）/ `mode`（`fast`\|`accurate`） | 带来源与相关性分数的文本块                |
| `list_documents`  | 你有什么   | `name_contains`（可选，文件名子串，大小写不敏感）            | 已建索引文档清单 + 各自块数（含插图块数） |

**为什么是两个而不是一个**：Agent 在检索前需要先知道**知识边界**。清单里没有的主题，检索必然落空，
此时应当直接告知用户「知识库未覆盖」，而不是硬检索后编造一段看起来合理的答案。
把「边界探查」与「内容检索」拆成两个工具，模型才有机会做出这个判断——这是单工具方案给不了的能力。

`list_documents` 只列**已索引**内容，这与运维视角的 `GET /documents` 是**故意不同**的：
后者会连同 `chunks=0` 的未索引文件一起列出（运维需要知道「哪些还没入索引」），
但对模型而言那是误导——它以为能检索到，实际检索不到。

手动验证（官方 Inspector，需 Node）：

```powershell
npx -y @modelcontextprotocol/inspector --cli python mcp_server.py --method tools/list
npx -y @modelcontextprotocol/inspector --cli python mcp_server.py --method tools/call --tool-name course_retrieve --tool-arg query=什么是ACID --tool-arg mode=fast
npx -y @modelcontextprotocol/inspector --cli python mcp_server.py --method tools/call --tool-name list_documents
```

任意 MCP 兼容客户端（如 Claude Desktop）按 stdio server 配置 `command` / `args` 即可挂载。
互操作实测矩阵与协议细节记录在本地 `mcp_interop_evidence.md`（**不随仓库分发**）。

## 运维：索引备份 / 语料接入 / 看板

**索引备份与恢复**（`backup_index.py`，纯文件操作，不依赖 Ollama）：

```bash
python backup_index.py backup   # 快照 chroma_db → chroma_db_backup_<时间戳>_<块数>/
python backup_index.py list     # 列出所有备份与块数
python backup_index.py restore  # 先备份当前、再恢复指定备份
```

**语料接入**（`ingest_ocr.py`：有文字层用 pypdf 直提，0 文字层降级 OCR）：

```bash
# 先跑前 N 页验证质量，避免整本烧 1 小时+
python ingest_ocr.py "data/教材\<书>.pdf" --max-pages 10
# 质量可接受 → 全量（无 --max-pages 时自动探测：直提 or OCR）
python ingest_ocr.py "data/教材\<书>.pdf"
```

> OCR 依赖懒加载，未装不影响仓库其他功能：`pip install rapidocr_onnxruntime pymupdf`。
> PyMuPDF 渲染 PDF 页 → RapidOCR / PP-OCRv4 识别中文，onnx 模型内置 wheel、离线可用。

**可观测看板**（`dashboard.py`，零新依赖，读 `.query_log.jsonl`）：

```bash
python -m streamlit run dashboard.py   # http://localhost:8501
```

展示 KPI（累计 / 今日查询、平均延迟、拒答与未接地率）、快速 vs 精确延迟均值 + P95 + 时序折线、
模式分布、来源 Top、拒答/未接地清单、👍/👎 反馈流。日志字段：

| 字段                    | 含义                                       |
| ----------------------- | ------------------------------------------ |
| `ts`                    | 毫秒级时间戳（反馈回写定位用）             |
| `question` / `mode`     | 问题与模式                                 |
| `top_sources`           | 检索来源（rank / source / score）          |
| `latency_ms`            | 整轮问答耗时（语义缓存命中时也如实记录）   |
| `est_tokens`            | 启发式估算 token（CJK≈1 token/字，非精确） |
| `rejected` / `grounded` | 硬拒答 / 接地判定                          |
| `rewritten`             | Query 改写结果（未改写为空）               |
| `feedback`              | 👍/👎（点击按钮后回写）                      |

**反馈闭环**（`feedback_pool.py`，把用户 👎 回流成评估集，不动冻结基线）：

```bash
python feedback_pool.py list                                   # 沉淀 👎 候选（去重）
python feedback_pool.py promote 0 --group B                    # 语料外应拒答 → B 组
python feedback_pool.py promote 1 --group C --expect reject    # 模糊 → C 组（需 --expect）
python eval_heldout.py --bc-only                               # 重跑门控覆盖回流用例
```

回流用例写入 `eval_data_feedback.json`，`eval_data_heldout.py` 运行时合并进 B/C 组——
不直接改冻结常量，可审计可撤销。

## 架构

```mermaid
flowchart TB
    subgraph 离线构建
        A[课程资料<br/>PDF / PPTX / TXT / MD / DOCX] --> B[递归语义切分<br/>800 字符 + 150 重叠]
        B --> C[嵌入<br/>nomic-embed-text]
        IMG[教材插图<br/>PyMuPDF bbox 渲染剪裁] --> CAP[qwen2.5-VL<br/>批量中文描述]
        CAP --> C
        C --> D[(Chroma 向量库<br/>增量索引 + 稳定 id 去重)]
    end

    subgraph 在线问答
        Q[用户问题] --> RW{history 非空?}
        RW -->|是| W[Query 改写<br/>指代消解]
        RW -->|否| RT[教材关键词路由<br/>规则表]
        W --> RT
        RT --> V[向量召回<br/>全库独立 top-50]
        RT --> M[BM25 词面召回<br/>全库独立 top-50]
        V --> F[RRF 倒数排名融合<br/>候选池 16]
        M --> F
        F --> RR[bge-reranker 精排 top-8]
        RR --> GATE[硬拒答 + 接地自检]
        GATE --> P[拼 Prompt<br/>含来源编号]
        P --> L[llama3.1:8b 生成<br/>内联来源引用]
        L --> ANS[回答 + 来源溯源<br/>分数 / 偏移 / 原图]
    end

    D -.-> V
    D -.-> M
    Q -.->|fast 模式 纯向量| ANS
```

关键改动：向量通道与 BM25 通道**各自对全库独立召回**（不是「向量池内重排」）。
这样正确块即使向量排名靠后，只要词面匹配也能被 BM25 通道独立捞回——
这是混合检索相对旧实现有实质增益的原因（见「评估」）。

## 评估

纯本地评估，不碰 HuggingFace：裁判 LLM 用本机 Ollama（temperature=0 逼稳定 JSON 输出），embedding 用 `nomic-embed-text`。

> **关于 RAGAS**：曾尝试引入，但 0.1.22 / 0.2.x 在 import 时会拉取
> `langchain_community.chat_models.vertexai`，而 langchain_community 0.3.x 已移除该符号，无法共存
> （pip 还会把 0.1.22 自动升回 0.2.x）。故**放弃 RAGAS，改为自己实现四个指标**——零额外依赖，
> 也顺便去掉了最后一个 langchain 依赖。

指标含义（四个均由 `eval_local.py` 自实现，LLM-as-judge）：

- `faithfulness`：回答是否真的基于检索到的 context（把回答拆成陈述逐条验证，防幻觉，最关键）
- `answer_relevancy`：回答是否切题、是否直接回应问题
- `context_precision`：召回结果里相关内容是否排在前面
- `context_recall`：检索是否覆盖了回答问题所需的信息（需 `eval_data*.py` 里的 reference）

### 当前官方基线（对外引用口径）

| 实验                                                         | 结果                                                         | 口径说明                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **冻结测试集**（24 题，从未参与调参；语料 1434 块 = 1219 文本 + 215 图像 caption） | faithfulness **0.901 ± 0.022** / answer_relevancy **0.975** / context_precision **0.869** / context_recall **0.948**（3 轮独立回归） | **对外引用的绝对值口径**；`python eval_test.py` 可复现       |
| **held-out A 组 v2**（38 题 = 30 文本 + 8 图片题；门控 faith ≥0.85 且 recall ≥0.80） | faith **0.915** / relevancy 0.958 / precision 0.824 / recall **0.975** → **PASS** | 只准用它调参                                                 |
| **图片命中率 top-8**（8 题图片问答）                         | **8/8 = 100%，全部 rank=1 命中**（v1.1 bge-m3 复测）         | 期望块按 `image_path` 锚定（`eval_heldout.py` 用 `expected_image` 子串匹配）；题面为自然用户问法。逐题明细由 `python eval_heldout.py --a-v2` 本地生成，**产物未随仓库分发** |
| **held-out B 组**（20 题语料外；门控拒答率 ≥90%）            | **20/20 = 100% → PASS**                                      | 硬拒答 + 接地自检 + 教材路由共同生效                         |
| **混合检索 vs 纯向量**（隔离 reranker 的 A/B 差值）          | precision **+0.18** / recall **+0.16** / faith +0.09 / relevancy +0.05，四项全正向 | 100 题（`python run_ab.py` 的 A/B 两轮，产物 `eval_results_vector.csv` / `eval_results_hybrid.csv` 各 100 行；**产物未随仓库分发，需本地重跑**）× 697 块历史语料（A/B 当时的块数，见 `config.py` 乱码清洗注释与 `test_retrieval.py` 的块数断言）；**只看差值** |
| **reranker OFF vs ON**（59 题全量语料）                      | precision +0.120 / relevancy +0.133 / recall +0.096 / faith +0.032，四项全正向 | reranker 救「正确块排 30+」的难 query                        |
| **性能**（本地推理，数字硬件绑定，跨机不可比）               | fast 全链路 P50 **5.9s** / P95 13.4s；accurate 检索 + 精排段 GPU 稳态 P50 **1.16s**（CPU 基线 21.1s ≈ **18×** 提速）；语义缓存命中 6.4s → **40ms** | 明细与完整演化时间线见本地 `EVALUATION.md`（**不随仓库分发**） |

> **v1.1 云端模式复测（2026-10-16，生成 deepseek-v4.1-flash / 裁判 kimi-k2.6，bge-m3 嵌入）：**
> 冻结测试集 24 题 faith 0.981 / relevancy 0.931 / precision 0.943 / recall 0.916（0 网络失败）；
> held-out A v2 38 题 faith 0.996 / relevancy 0.962 / precision 0.922 / recall 0.975 → 门控 PASS；
> held-out B 组拒答 20/20 = 1.00 → PASS；图片命中率 8/8 全 rank=1。
> ⚠️ 注意：换生成/裁判模型后与下表**不可直接比较**，两套数字并列只为留痕。

> ⚠️ 下表数值是**本地模式基线**（生成 llama3.1:8b / 裁判 qwen2.5:7b）的历史实测，
> 用于和当前云端模式做对照，不代表云端模式的复测结果。
> 检索侧指标（precision / recall）三轮零方差，方差全部来自生成侧（temp 0.2）。

### held-out 门控

用来解决「评估集与语料同源 → 绝对值偏乐观」的问题：三组独立于调参过程的 held-out 集，
用绝对指标 + 拒答率双门控，作为「能否称落地」的硬证据。

- 样本：`eval_data_heldout.py` = **A 组 v2 共 38**（语料内 30 + 图片题 8）+ **B 组 20**（语料外应拒答）+ **C 组 10**（对抗 / 模糊 / 注入）
- 门控：A 组 faithfulness ≥ **0.85** 且 context_recall ≥ **0.80**；B 组拒答率 ≥ **90%**

```bash
python eval_heldout.py --a-v2      # A v2 口径（38 题含图片题）+ 图片命中率
python eval_heldout.py --bc-only   # B/C 拒答验证
python eval_heldout.py             # 全量 + 总门控
python eval_test.py                # 冻结测试集最终回归（从未参与调参）
```

**评估纪律**（这是这套数字可信的前提，不是形式）：

- `HELDOUT_A` v1 是已参与调参的**校准集**，对外正式宣布前必须用冻结测试集做最终回归。
- **语料变更纪律**：每次语料 / 切分 / 检索参数变更后，必须重验 A 组与 B/C 门控，不能只看冻结测试集
  ——语料扩增会改变检索重排（978→1219 块时 A 组指标下移、B 组拒答 100%→95%，均为检索重排变化的直接体现）。
- 重跑 A 组前**先移走旧 CSV**（`eval_local.py` 有断点续跑，同名 CSV 会让全部题目被跳过）。
- 改阈值重跑时**禁止为刷指标删 B 组题**。
- 测试集严格冻结、禁止调参；跨 run 的 delta 不作为证据。

## 说明

- **索引管理（P1-4）**：侧栏「🔄 全量重建索引（data/ 全部）」走 `get_vector_store(chunks)` 整体重建；上传文件走增量 upsert（稳定 id 自动去重）；删除单文档只按 source 从索引移除该文档，**其余文档不受影响**，不再全量重建。生产环境建议删除前「移入回收站 + 二次确认」。
- 混合检索默认开（`HYBRID_ENABLED=true`）：精确模式默认即为「向量 + BM25 融合」，即使 `RERANKER_ENABLED=false` 也非纯向量；reranker 已验证可启用：HF 镜像（hf-mirror.com）在本机可达，`bge-reranker-v2-m3` 权重已下载（约 2.2GB）。**默认 `RERANKER_ENABLED=true`**（难 query 正确块常排 30+，需 RETRIEVE_K=50 + reranker 纠回；33 样本整体增益≈0 但救难 query）。启动 app.py 须带 `HF_ENDPOINT=https://hf-mirror.com`（或离线），否则权重加载会去 huggingface.co 超时。
