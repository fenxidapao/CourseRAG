# 演示脚本（5 分钟）

> 目标：从空白终端到「提问 → 引用第 N 页 → 点开页图看红框」。
> 所有数字来自 README / AGENTS 的最新口径，现场不要改参数。

---

## 0. 开场前 60 秒（冷启动预热）

```bash
# 1) Ollama 必须在跑（嵌入 bge-m3 / 生成 qwen2.5:7b / 裁判 glm4:9b）
ollama ls

# 2) 生产模式（推荐演示）：构建一次，FastAPI 直接托管
cd frontend && npm run build && cd ..
python -m uvicorn main:app --host 127.0.0.1 --port 8000

# 3) 预热：第一题会加载精排 + 7B（实测 112 s），稳态约 8 s/题
curl -s -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Transformer 用了多少个注意力头？","mode":"accurate"}' | head -c 200
```

开发模式（改前端时用）：`cd frontend && npm run dev` → http://127.0.0.1:5173（`/api` 代理到 8000）。
生产模式下打开 **http://127.0.0.1:8000** 就是 React 界面。

---

## 1. 三条演示问句（按顺序问）

| # | 问句 | 期望 | 看点 |
|---|---|---|---|
| 1 | 实验发现 LoRA 的秩 r 最小可以取到多少仍然有效？ | 答「r=1」，引《LoRA》第 10/11 页 | 正文题：检索 + 页级引用 |
| 2 | ColBERT 做重排时比 BERT 模型快多少倍、FLOPs 少多少？ | 答「约 170 倍」；FLOPs 倍数与引用页可能落在结论/摘要（已知波动） | 表格题：点引用看同页高亮 |
| 3 | 中文论文的作者与发表期刊是什么？ | 答「邢淼、田丽，《知识管理论坛》」，引第 1 页 | 元信息题：v3.4 元信息块 |

点击任一引用卡 → 右侧页图跳到对应页，红框落在证据块上（同页已召回块会一起框出）。

---

## 2. 一句话讲清链路（被问到时）

> MinerU 把 PDF 拆成「页 → 块」并保留 bbox；页内切分后每块带 `page/section/block_ids` 入 Chroma；
> 检索是 bge-m3 向量 + 自实现 BM25 的 RRF，再经 bge-reranker 精排；
> 生成前把来源编号成 `[n]`，回答里的 `[n]` 直接映射到页图高亮；解析与问答都在本机，8 GB 显存分时跑。

---

## 3. 口径与兜底（被追问时照读）

- 检索：30 题 `paper_hit@8 1.000 / page_hit@8 0.967 / page_strict@1 0.700`。
- 拒答：20 域外 + 20 域内 **拒答 90% / 域内误拒 0%**；这是纯阈值口径（接地复核默认关）。
- held-out：域外 100%、域内曾 10%（元信息题，已单题修好，未整组重跑）。
- 已知限制（不要现场调参）：
  - 优化器 β2 偶答 0.999（原文 0.98）——7B 在数值上的不稳定，改顺序会掉引用准确率；
  - 引用页 strict 约 0.60：内容对、偶尔引到摘要/结论页；块级 single ≈ 0.55–0.61、union ≈ 0.68–0.74，
    本地裁判对长表格有假阴性，只报趋势；
  - 中文提问英文论文的召回弱于同语言。
- 如果现场答错：直接说「这是已知限制，评估里怎么记的」，不要改阈值/重跑。

---

## 4. 收尾（30 秒）

- 代码：`feat/paper-mineru`，课程版归档在 `v1.1-course-final` / `archive/course-v1.1`。
- 数据边界：`data/papers/` 与 `chroma_db_papers/` 均未入库；只提供页图，不提供整本 PDF。
- 复现命令：`python paper_index.py --audit`、`python eval_paper.py --group page --gate`。
