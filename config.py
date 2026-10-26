"""
config.py — 所有可调参数集中外置到 .env，代码里不再写死模型名/路径。
这是工程规范的第一步：『配置与密钥分离』。
"""
import os
from dotenv import load_dotenv

# 读取项目根目录下的 .env（.env 不进 git，见 .gitignore）
load_dotenv()

# reranker 权重（bge-reranker-v2-m3）加载时需访问 HuggingFace；本机 huggingface.co 不可达会
# 超时卡死（表现为页面"一直转"）。故默认走 hf-mirror.com 镜像；可用 .env 的 HF_ENDPOINT
# 或 shell 环境变量覆盖（setdefault 不会覆盖已存在的同名变量）。
os.environ.setdefault("HF_ENDPOINT", os.getenv("HF_ENDPOINT", "https://hf-mirror.com"))

# ---------- 模型与运行时（来自 .env，给默认值便于本地直接跑）----------
# C1（2026-10-16）：嵌入模型由 nomic-embed-text(768维) 换为 bge-m3(1024维)。
# 为什么换：nomic 是英文中心模型，中文语料相似度基线偏高（config 下方阈值注释早有记录）；
# bge-m3 是多语言/中文友好模型，维度 768→1024。
# ⚠️ 换嵌入模型必须**全量重建索引**，不能复用旧 collection（维度不兼容）。
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
# Ollama 默认本地端口；换机器/远程只需改这里
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ---------- 裁判模型提供方（C2 去本地化，2026-10-16）----------
# 为什么裁判也必须切云：生产 /query 的 grounded 自检走 grounding.py，默认本地
# qwen2.5:7b（约 4.7 GB），与生成模型叠加会撑爆 8 GB 显存。
# 为什么必须独立配置：裁判若与生成同模型会产生自评偏差（同源打分虚高）。
# 实测（2026-10-16，同一 JSON 接地任务，3 次）：
#   kimi-k2.6      → 3/3 JSON 成功，1.4~1.6s（零 reasoning）
#   glm-5.2        → 3/3，6.7~24.1s
#   qwen3.8-flash  → 1/3（2 次 DNS 失败），7.8~29.6s
# 故默认裁判选 kimi-k2.6：最快、最稳，且与生成 deepseek-v4.1-flash 不同源。
#   JUDGE_PROVIDER=local  → Ollama（JUDGE_MODEL 为 Ollama 模型名）
#   JUDGE_PROVIDER=openai → OpenAI 兼容云端（兼容旧写法 deepseek）
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "local").strip().lower()
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "kimi-k2.6")
JUDGE_API_BASE = os.getenv("JUDGE_API_BASE", "https://kuaipao.pro/v1")
# 裁判 key：留空时回退到同一个中转 key（环境变量或 ~/.dsh/.credentials.yaml）
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "")
# 裁判输出是结构化 JSON，预算需求远小于生成；默认关闭思维链（同上 C3 实测原因）
JUDGE_DISABLE_THINKING = os.getenv("JUDGE_DISABLE_THINKING", "true").strip().lower() == "true"
JUDGE_MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "4096"))
JUDGE_MAX_TOKENS_CAP = int(os.getenv("JUDGE_MAX_TOKENS_CAP", "8192"))

# ---------- 生成模型提供方（C3 去本地化，2026-10-15）----------
# 背景：本地 llama3.1:8b（约 4.9 GB 显存）+ 裁判 qwen2.5:7b（约 4.7 GB）
# 在 8 GB 卡上装不下；Windows WDDM 不报 OOM，而是让 CUDA 调用挂住（见 04 纪律 28）。
# 生成先切云端，裁判另批处理。默认 ollama：不设该变量时行为与旧版完全一致，可随时回滚。
#   LLM_PROVIDER=ollama → 本地 Ollama（LLM_MODEL 必须是 Ollama 模型名）
#   LLM_PROVIDER=openai → OpenAI 兼容云端（LLM_MODEL 为云端模型名，走 LLM_API_BASE）
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
# 云端 OpenAI 兼容接口地址（仅 LLM_PROVIDER=openai 时使用）
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://kuaipao.pro/v1")
# 云端 key：留空时由 llm_provider.resolve_api_key() 依次尝试环境变量与
# ~/.dsh/.credentials.yaml 的 KUAIPAO_API_KEY；key 绝不写入仓库/日志/响应。
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# 单次请求超时（秒）；推理模型首响可能较慢，默认留足
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
# 初始 max_tokens 与自动扩容上限。推理模型（deepseek-v4.1-flash）实测
# max_tokens=256 时 256 个 token 全耗在 reasoning、content 为空；
# 故默认预算不低于 1024，空 content+reasoning 占满时自动翻倍到 cap。
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
# 是否要求云端模型关闭思维链（默认 true）。
# 实测（2026-10-15，真实 RAG 提示、8 块上下文、max_tokens=4096）：
#   deepseek-v4.1-flash 默认开启思考，3/3 次把预算全烧在 reasoning 上，
#   content 为空、finish=length，单次耗时 20~23s；失败是概率性的，不是必现。
#   显式传 reasoning_effort=none / thinking.type=disabled 后 0 reasoning，
#   正常产出 content（5.9~33s）。因此默认关闭思考；模型不支持该字段时
#   由如下的自动降级路径处理（见 llm_provider._cloud_post 的字段裁剪重试）。
LLM_DISABLE_THINKING = os.getenv("LLM_DISABLE_THINKING", "true").strip().lower() == "true"
LLM_MAX_TOKENS_CAP = int(os.getenv("LLM_MAX_TOKENS_CAP", "8192"))

# 云端瞬时错误重试（C3.1，2026-10-16）：中转实测约 21% 请求瞬时失败
# （getaddrinfo failed / SSL handshake），重试即成功。指数退避：1.5s、3s。
# 只重试瞬时类错误（DNS/SSL/超时/429/5xx）；401/403/400 不重试。
LLM_RETRY_ATTEMPTS = int(os.getenv("LLM_RETRY_ATTEMPTS", "3"))
LLM_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "1.5"))

# ---------- RAG 调参（这些数为什么这么定）----------
# 注意：实际评估基线为 800/150（.env 与 README 一致；2026-08-22 统一口径，
# v2 A/B 等全部数字均在 CHUNK_SIZE=800 下测得）。勿再改回 500/80 以免与证据脱节。
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))       # 每块字符数（教材正文段落约 200-800 字符）
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # 块间重叠，避免切断句子/上下文
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))  # 低温度压制幻觉

# ---------- 重排 Reranker（召回多、精排少，两段式检索）----------
# 向量召回是双塔粗排（只看语义相似度），reranker 是单塔交互精排，
# 把 (query, doc) 拼一起过模型，捕捉细粒度相关性。生产 RAG 标配。
# 工程化落地（2026-08-24）：决策④ reranker 默认开启（生产 RAG 标配两段式）。
# 注意：run_ab.py 仍强制 RERANKER_ENABLED=false 隔离「BM25 纯贡献」A/B，不受此处影响；
# 快速模式（UI 默认）走 force_no_rerank，本机不触发 reranker，延迟不变；
# 仅「精确模式」叠加 bge-reranker 精排（延迟 19~115s，已在 UI 文案明示）。
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")  # 多语言，中文友好
# 运行设备（2026-09-03 GPU 化改造）：auto=有 CUDA 用 GPU，否则回退 CPU；可强制 cuda/cpu。
# 前提：venv 里的 torch 必须是 CUDA 轮子（pip install torch==2.13.0 --index-url
# https://download.pytorch.org/whl/cu126）；CPU 轮子装了也不会用 GPU。
# 注意 8GB 卡与 Ollama（llama3.1:8b 约 5GB）共存：pipeline 中精排与生成串行，
# 显存紧张时把 OLLAMA_KEEP_ALIVE 调短让两模型错峰。
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto")
# 半精度开关（默认关）：fp16 推理再快约一倍，但 rerank_score 会轻微漂移——
# 硬拒答阈值 HARD_REJECT_RERANK_THRESHOLD=0.30 按 fp32 校准，开 fp16 前必须重验 B 组拒答行为。
RERANKER_FP16 = os.getenv("RERANKER_FP16", "false").lower() == "true"
# GPU 上批量打分的 batch（16=候选池全量一次过；CPU 上该值无感）
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "50"))       # 向量召回候选数（扩到 50：难 query 正确块常排 30+，池不够捞不到；配合 reranker 精排纠回）
# 重排后保留、进 LLM 的条数。2026-08-24 held-out A 组实测：5→8 后 faithfulness 0.746→0.911、
# context_recall 0.823→0.873（正确块常排 6~8 名，top-5 供给不足导致模型「无法完整回答」）。
# 不设更高（10+）：precision 已回落 0.811，噪声随 TOP_N 递增，0.85 门控余量已足。
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "8"))
# B 专项（2026-08-26）：RRF 融合候选池规模（>RERANKER_TOP_N）。
# 背景：Q21 软链接等难 query 的正确块只被 BM25 通道命中（向量通道缺席），
# 融合直接按 RERANKER_TOP_N=8 截断时其 RRF 分（仅 1/(rank+60)）被双通道块压制而淘汰，
# 永远进不了 rerank。扩到 16 让 BM25-only 正确块先入池，再由 reranker 精排回 top-8。
# 代价：rerank 候选翻倍 → 「精确模式」延迟上升（bge-reranker CPU 逐条打分）。
RERANK_CANDIDATE_K = int(os.getenv("RERANK_CANDIDATE_K", "16"))

# ---------- 混合检索（向量 + 自实现 BM25，双通道 RRF 融合）----------
# 课程资料大量精确术语（ACID / 2NF / 死锁 / 事务隔离级别），纯向量对词面匹配不如 BM25。
# hybrid = 向量全库召回 top-K ‖ BM25 全库独立召回 top-K（两个通道互不依赖）→ RRF 倒数排名融合。
# BM25 自实现（零新依赖，更贴「自实现核心组件」的取舍）；融合分数落 metadata['hybrid_score']。
HYBRID_ENABLED = os.getenv("HYBRID_ENABLED", "true").lower() == "true"
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "50"))  # BM25 通道独立召回数（对全库打分，不再复用向量候选池）

# ---------- P1a 元数据过滤多路召回（2026-09-04）----------
# 问题命中教材/课程关键词（rag_search_utils.SOURCE_RULES）时，向量路与 BM25 路都
# 限定到对应教材再 RRF 融合；未命中关键词 = 全库检索（行为与旧版一致）。
# 关闭后自动路由失效，但显式 source 参数（API/UI 指定教材）仍可用。
SOURCE_FILTER_ROUTING_ENABLED = os.getenv("SOURCE_FILTER_ROUTING_ENABLED", "true").lower() == "true"

# ---------- 枚举/穷举类查询的锚点扩展召回（2026-09-11）----------
# 背景（held-out A 组 30 题逐题核算）：5 道枚举/清单类题只占 16.7% 题量，却吃掉
# 49.2% 的 context_precision 损失；「Streamlit 有哪些输入组件」更是全组唯一
# recall 失分题（0.4）。根因不是排序错，是**供给不足**——枚举题的答案天然散落在
# 同一份文档的多个段落里，而全库 top-50 向量召回在多文档竞争下对单一文档覆盖不均：
# 一份 120 块的文档可能只有 10 块挤得进全库 50 强，再经 RRF 截 16、精排留 8，
# 一份十几项的清单必然被截断。
#
# 做法：检测到枚举意图时，先用元数据（source）把召回范围钉到锚点文档上，再在该
# 范围内做一次密集召回（ENUM_ANCHOR_TOP_K），与基础候选合并后进 reranker；
# 同时把进 LLM 的供给量放宽到 ENUM_RERANK_TOP_N。
#
# 默认关闭：按评估纪律（§0.4），检索层改动必须先跑 held-out A 组 A/B 拿到数字，
# 再决定是否翻转默认值——不能"觉得应该更好"就打开。
#
# 【2026-09-11 实测结论：否决，默认保持 false，勿再翻转】
#   held-out A 组 30 题 on/off 两轮：枚举题 prec 0.815→0.558（−0.257）、faith 0.901→0.840；
#   对照组 18 题 prec/recall 逐题逐位零变化（隔离干净，Δ 是真信号）。
#   且前置排查发现原实现是死代码——合并上限 max(pool_k, ENUM_CANDIDATE_K) 与 RRF 池同值，
#   扩展块被 merged[:candidate_k] 整段截掉（净新增 17 块 → 标记 0）。修好后扩展块能进来，
#   但落点是 rank 10/13/14、rerank 分仅 0.005~0.010，top-8 与基线逐块相同 → 零贡献。
#   最根本：12 道枚举题里 11 道基线 recall 已是 1.000 → 短板在 precision（排序）不在 recall，
#   扩召回用错了杠杆。
#   保留代码 + 加回归守卫（test_api.py），作"用数据否决自己方案"的留痕。
ENUM_EXPAND_ENABLED = os.getenv("ENUM_EXPAND_ENABLED", "false").lower() == "true"
ENUM_ANCHOR_SOURCES = int(os.getenv("ENUM_ANCHOR_SOURCES", "3"))   # 无教材过滤时，取基础候选前 N 块的 source 作锚点
ENUM_ANCHOR_TOP_K = int(os.getenv("ENUM_ANCHOR_TOP_K", "24"))      # 锚点文档内的密集召回数
ENUM_CANDIDATE_K = int(os.getenv("ENUM_CANDIDATE_K", "40"))        # 枚举题的候选池上限（含锚点扩展块）
ENUM_RERANK_TOP_N = int(os.getenv("ENUM_RERANK_TOP_N", "14"))      # 枚举题最终进 LLM 的块数（常规为 RERANKER_TOP_N=8）

# ---------- 路径 ----------
DATA_DIR = os.getenv("DATA_DIR", "./data")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "./chroma_db")

# ---------- 加载（T6: 扫描版 PDF 检测） ----------
# 均字符/页 低于此阈值视为无文字层（扫描版），建库时告警跳过而非静默空索引。
PDF_MIN_CHARS_PER_PAGE = int(os.getenv("PDF_MIN_CHARS_PER_PAGE", "30"))

# ---------- 评估裁判（T2: 独立裁判，2026-08-22；C2 于 2026-10-16 迁至文件前部）----------
# 裁判配置（JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_API_BASE / JUDGE_API_KEY /
# JUDGE_DISABLE_THINKING / JUDGE_MAX_TOKENS）已统一上移到「裁判模型提供方」区块。
# 保留本注释只为标记历史位置；此处不得再重复定义同名变量，否则会静默覆盖默认值
# （C2 实测踩过：旧块 JUDGE_API_BASE=api.deepseek.com 覆盖了新默认值，导致 401）。

# ---------- 低置信阈值（T3: 拒答/提示机制，2026-08-22） ----------
# top-1 向量相似度（cosine）低于此值 → 判定「资料可能不足」，UI 显式提示而非硬答。
# 由 _thr_scan.py 实测：语料内 59 题 top1 p5=0.6896、语料外 10 题 max=0.7686（分布有重叠，
# nomic-embed-text 中文基线相似度偏高）→ 阈值取语料内 p5，误拒≈5%，漏拒部分语料外问题；
# 因此实现为「低置信提示」而非硬拒答。改阈值后重跑 _thr_scan.py 校准。
# C1 重标定（2026-10-16，bge-m3）：原 0.69 按 nomic 分布取语料内 p5；
# 新模型实测 30 题语料内 p5=0.604、p10=0.694，20 题语料外 median=0.518/max=0.763。
# 取 0.62（≈p5~p10）：软提示命中约 3~5% 语料内题。
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.62"))

# ---------- 硬拒答 / 生成接地自检（P1，2026-08-24 工程化落地） ----------
# 软提示（LOW_CONFIDENCE_THRESHOLD）仍保留给「快速模式」；此处为「精确模式」的硬拒答闸门：
#   - 检索阶段：top-1 向量分 < HARD_REJECT_VECTOR_THRESHOLD 直接硬拒（不进生成）；
#     精确模式若开启 reranker，其最高 rerank_score < HARD_REJECT_RERANK_THRESHOLD 也判硬拒。
#   阈值取 0.55（低于软提示 0.69），仅拦明显语料外，误拒率更低。
#   - 生成阶段：GROUNDED_VERIFY_ENABLED 时，独立裁判判定回答是否完全基于上下文
#     （grounded），未接地则把答案替换为硬拒文案，杜绝软提示下的幻觉外溢。
# 实测校准：启用后跑 eval_heldout.py（B 组拒答率应 ≥90%），据结果回调阈值。
# C1 重标定（2026-10-16，bge-m3）：0.58 在实测表上误拒语料内 1/30=3.3%、
# 放过语料外约 30%（比 0.55 更紧一点，且误拒率不变）；语料外漏拒由 reranker 阈值 + 接地自检兜底。
HARD_REJECT_VECTOR_THRESHOLD = float(os.getenv("HARD_REJECT_VECTOR_THRESHOLD", "0.58"))
HARD_REJECT_RERANK_THRESHOLD = float(os.getenv("HARD_REJECT_RERANK_THRESHOLD", "0.30"))
GROUNDED_VERIFY_ENABLED = os.getenv("GROUNDED_VERIFY_ENABLED", "true").lower() == "true"
# grounded 自检：回答被支撑比例（faithfulness 口径）低于此值 → 判未接地，触发硬拒替换。
GROUNDED_SCORE_THRESHOLD = float(os.getenv("GROUNDED_SCORE_THRESHOLD", "0.60"))

# ---------- 多用户预留扩展点（P2 / Gate 2，2026-08-24） ----------
# 默认全部关闭：本地 / 局域网部署零摩擦。将来要并发 Web 服务，开开关即可，不必重写后端。
#   - AUTH_ENABLED=true：所有 API 端点要求请求头 `X-API-Key: <API_KEY>`。
#   - RATE_LIMIT>0：按客户端 IP 限制每分钟请求数（内存令牌桶，单进程够用；
#     多副本需换 Redis，见文档路线 β/ε）。
#   - CORS_ORIGINS：允许源，生产收紧为具体域名（默认 "*" 仅演示 / 前后端分离）。
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "0"))            # 每 IP 每分钟请求上限，0=不限
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# 上传大小上限（MB，评审 P3，2026-08-30）：/documents 上传接口此前无大小限制，
# 任意大文件会被整体读入内存再写盘。100MB 覆盖正常课程资料（几百页文字层 PDF
# 通常 <20MB）；扫描版大 PDF 走本地 ingest_ocr.py 接入，不经上传接口。
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "100"))

# ---------- 语义缓存（P2，2026-08-25） ----------
# 缓解精确模式 reranker 延迟（精确段口径见 EVALUATION.md）：命中"语义相似"的重复/改写问句时跳过检索/重排/生成。
#   - SEMANTIC_CACHE_ENABLED：总开关（单用户可关；多人在线热问题重复率高时开着最划算）
#   - SEMANTIC_CACHE_SIM：embedding 余弦阈值，同义改写（0.93 上下）命中，意图不同不误命中
#   - SEMANTIC_CACHE_MAX：内存 LRU 上限（单进程；多副本需换 Redis，见文档路线 β/ε）
# 只缓存正常答案，硬拒答/未接地不缓存；索引重建后自动失效（semantic_cache.invalidate）。
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() == "true"
SEMANTIC_CACHE_SIM = float(os.getenv("SEMANTIC_CACHE_SIM", "0.93"))
SEMANTIC_CACHE_MAX = int(os.getenv("SEMANTIC_CACHE_MAX", "128"))

# ---------- Query 改写（多轮指代消解，2026-09-18 落地） ----------
# 背景：多轮对话里「第三章那个定理怎么用」脱离历史无法检索——向量与 BM25 都看不到
# 「第三章」指什么。改写 = 用 LLM 把「历史 + 当前问题」改写为独立完整的检索问题，
# 再进 缓存 key / 教材自动路由（SOURCE_RULES）/ 检索 / 生成 管线。
# 触发条件：history 非空才改写——单轮行为零变化（冻结评估集均为单轮，评估口径不受影响）。
# 失败/超时/输出异常 → 回退原问题，改写永不阻塞问答（增强项，不是新故障点）。
# 模型复用 LLM_MODEL、temperature=0（改写要确定性）。评估侧需强制旁路时传 use_rewrite=False。
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "true").lower() == "true"

# ---------- 乱码清洗（2026-08-22：微信小程序 PDF CID 字体乱码） ----------
# 块级乱码字符占比 > 此阈值 → 整块丢弃（内容已不可读，纯噪声）；
# 否则剥离乱码字符保留正文。诊断：微信小程序 303 页 372 块中 29 块 >30%、
# 251 块 >10% 混入乱码（广泛存在）→ 阈值 0.3 只丢最脏块。
# 2026-08-22 修复：_is_good_char 不再整段放行拉丁补充区（0x00A0-0x00FF）——
# pypdf 对 CID 乱码的另一种映射是产出 æ/ç/è 等拉丁字符，原白名单误放行导致
# 698 块中 54 块残留；剔除后块数 698→697（1 块此前侥幸保留的乱码块被正确丢弃）。
GARBLED_DROP_THRESHOLD = float(os.getenv("GARBLED_DROP_THRESHOLD", "0.3"))

# ---------- B1 视觉重排（2026-10-14） ----------
# 洞 R：caption 路线让图像以「文本代理」身份进索引，**像素从未参与检索**。两张 caption
# 几乎相同的截图（实测 3-gram Jaccard 最高 0.833），文本召回 / BM25 / bge-reranker 只能
# 看到几乎一样的文字，原理上分不出谁是谁。
# B1 在检索候选出来后，把候选中的图像块**原图**送 VLM 复核，按视觉相关性重排图像块顺序。
# 检索链路本身完全不动（廉价文本路径照常召回），只加一次视觉复核。
#
# 【为什么默认关闭】按本文件 ENUM_EXPAND_ENABLED 留下的教训，检索层改动必须先跑
# held-out A/B 拿到数字，再决定是否翻转默认值——不能"觉得应该更好"就打开。
#
# 【实测读数（2026-10-14；原始报告与标定脚本已归档到仓库外，不在本仓库内）】
#   - 文本管线原理上无法区分：判别性 token（个人信息 / 比较数字大小 / avatar.png /
#     tab1.wxml）在全部 215 条 caption 中出现 **0 次**。
#   - qwen3-vl-plus 做 4 选 1 + 位置置换：**15/16 命中（随机基线 25%）**，唯一未命中是
#     dt=0.0s 的空响应（网络瞬时故障），成功调用 15/15 全对。
#
# 【模型选择实测（kuaipao.pro 中转，同一把 key）】
#   - deepseek-flash      → 403「This token has no access to model」→ 不可用
#   - deepseek-v4.1-flash → 能看图，但单次 64s 且 finish=length 被截断 → 不可靠
#   - qwen3-vl-plus       → 1.3s / 4 张图、345 image tokens、答案稳定 → **选用**
VISUAL_RERANK_ENABLED = os.getenv("VISUAL_RERANK_ENABLED", "false").lower() == "true"
VISUAL_RERANK_MODEL = os.getenv("VISUAL_RERANK_MODEL", "qwen3-vl-plus")
VISUAL_RERANK_API_BASE = os.getenv("VISUAL_RERANK_API_BASE", "https://kuaipao.pro/v1")
VISUAL_RERANK_API_KEY = os.getenv("VISUAL_RERANK_API_KEY", "")
# 一次请求最多送几张图（候选图像块多于此时只取前 N 个；每张图约 86 image token）
VISUAL_RERANK_MAX_IMAGES = int(os.getenv("VISUAL_RERANK_MAX_IMAGES", "8"))
# 单次 VLM 调用超时（秒）；超时即 fail-open 回退原顺序，绝不阻塞问答
VISUAL_RERANK_TIMEOUT = float(os.getenv("VISUAL_RERANK_TIMEOUT", "60"))
# 重排模式（决定给 VLM 的问法）：
#   "verify"（默认）— 逐张问「这张图里**真的出现了**能回答问题的内容吗」，返回 hits 数组。
#                     **校准过**：没有任何一张命中时会给空数组 → 保持文本链路原顺序。
#   "promote"       — 问「哪张最相关」，把选中的图提到 rank 1。
#   "slot"          — 只让候选里的图像块互相换位，不越过文本块。
# ⚠️ 实测（B1 标定实验，6 道像素题；脚本已归档）：
#   "promote" 在 5/6 道「目标图根本不在候选里」的题上**从不弃权**，硬把无关图
#   （p141/p276/p026）顶到 rank 1 —— 净负收益。相对选择式问法没有校准，问不出「都不对」。
#   "slot" 在现网 top-8 下基本无效：reranker 截断后池内常只剩 0~1 张图，
#   而本模块要求 ≥2 张才发问 → 静默跳过。两种旧模式均保留仅为对照。
VISUAL_RERANK_MODE = os.getenv("VISUAL_RERANK_MODE", "verify")
# 专用图像通道召回条数（0=关闭该通道，退化为「只用文本链路候选里的图」）。
# ⚠️ 为什么必须有这个通道（B1 召回探针实测，脚本已归档）：把 RRF 池宽从 16 拉到 200，
# 池内图像块从 5 涨到 26，但 6 道题里仍有 5 道的目标图**不进池**——瓶颈不是池宽。
# 而独立的 where={"type":"image"} 通道能把 p267（rank 6/215）、p045（rank 24/215）
# 捞出来；纯 RRF 池漏掉了后者。没有它，视觉重排经常「无图可排」。
VISUAL_RECALL_K = int(os.getenv("VISUAL_RECALL_K", "8"))

# ---------- B3 描述器（2026-10-14） ----------
# 洞 S：图像块在索引里的**全部可检索内容**就是 caption（rag_core.load_image_chunks：
# page_content = caption）。所以描述器写不出什么，检索就永远搜不到什么。
#
# 【旧链路实测缺陷】旧版描述器（已归档的 `_caption_images.py`）用本地 qwen2.5vl:7b + prompt「用 2~3 句中文
# 描述这张教材插图」。该 prompt 在**结构上**只输出叙述，不逐字抄录图中文字，于是
# `微信小程序开发实战` 四张项目结构截图的判别性 token（tab1.wxml / avatar.png /
# 用户登录 / 比较数字大小）在全部 215 条 caption 中出现 **0 次**——图像通道里它们排到
# 56/66/67/69 名（分母 98）。B1 的视觉重排因此拿不到候选（B1 上限探针，已归档）。
#
# 【换描述器实测（B3 prompt 探针，已归档；4 张 ground-truth 图）】
#   - 旧 prompt 原文 × qwen3-vl-plus          3/4 命中 ← **主杠杆是模型，不是 prompt**
#   - 纯抄录 prompt × qwen3-vl-plus           3/4
#   - 描述+抄录（HYBRID）× qwen3-vl-plus      4/4  ← 采用
#   旧 prompt 配本地 qwen2.5vl:7b 是 0/4。HYBRID 保留描述句，故不牺牲语义检索面。
#
# 【全量重跑读数（215 张，_caption_images_v2.py）】
#   平均长度 110.9 → 349.0 字（3.1×）；含文件名类 token 的 caption 34(15.8%) → 49(22.8%)；
#   215/215 都带「图中文字：」抄录段（旧为 3）；四张目标图的判别性 token **0/6 → 6/6**。
IMAGE_CAPTIONS_CSV = os.getenv("IMAGE_CAPTIONS_CSV", "_image_captions.csv")
# B3 修复（2026-10-16）：嵌入用 caption 与生成用 caption 分离。
# 实测：新描述器（逐字转写）的文本对「创建小程序需要填哪些设置项」这类问题判别力更差——
# 在 215 张图内 p020 排名：旧 v1 caption 第 10 / 新摘要第 34 / 新全文第 28。
# 故嵌入与 BM25 用 v1 短描述（判别力强），生成与引用用新 caption（信息全）。
# 置空字符串 = 不分离，退回单 caption 行为。
IMAGE_EMBED_CAPTIONS_CSV = os.getenv("IMAGE_EMBED_CAPTIONS_CSV", "_image_captions_v1_qwen25vl.csv")
