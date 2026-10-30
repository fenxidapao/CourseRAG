"""eval_judge.py — 接地判官校准（本地裁判候选对比，2026-10-30）

复用生产同一套 `grounding.verify_grounded`（同一 system/user prompt、同一
min(明细重算, 自报分) 口径、同一 GROUNDED_SCORE_THRESHOLD），只换 JUDGE_MODEL。

30 对 = 15 条有据可答 + 10 条与证据矛盾 + 5 条"正确但与上下文无关"。
证据全部来自当前 18 篇语料的真实 chunk（按论文 + 关键词定位到页，整页 chunk 拼成上下文）。

用法：
  python eval_judge.py --models qwen3:1.7b,llama3.2:3b,qwen3:4b
输出：
  eval_results_paper_judge_<ts>.csv（gitignored）+ 每个模型的混淆矩阵与耗时。

指标口径：
  json_ok      = 裁判返回的 statements 非空（解析/传输成功的代理指标）
  accuracy     = 与人工标签一致的比例
  false_accept = 标签 0 却判 grounded（放过无据回答，最危险）
  false_reject = 标签 1 却判未接地（误拒有据回答）
"""
from __future__ import annotations

import argparse
import csv
import datetime
import statistics
import sys
import time

import config
import grounding
from paper_index import load_paper_chunks

# (paper 标题子串, 定位关键词, 类别, 标签, 回答)
PAIRS = [
    # ---------- 有据可答（1） ----------
    ("Attention Is All You Need", "parallel attention layers", "grounded", 1,
     "论文使用 h=8 个注意力头，每个头的 d_k=d_v=64。"),
    ("Attention Is All You Need", "Adam optimizer", "grounded", 1,
     "训练用 Adam 优化器，β1=0.9，β2=0.98，ε=1e-9，warmup 4000 步。"),
    ("Attention Is All You Need", "sine and cosine", "grounded", 1,
     "因为模型没有循环和卷积，需要注入位置信息；论文使用不同频率的正弦和余弦函数作为位置编码。"),
    ("LoRA: Low-Rank", "10,000 times", "grounded", 1,
     "LoRA 冻结预训练权重，在每层注入可训练的低秩分解矩阵；相比 GPT-3 175B 全量微调，"
     "可训练参数减少 10,000 倍，GPU 显存降低 3 倍。"),
    ("LoRA: Low-Rank", "r = 1", "grounded", 1,
     "实验发现 r=1 在 ΔW_v 和 ΔW_q 上表现已经很好，原因是不同秩学到的子空间高度重叠。"),
    ("QLoRA", "48GB GPU", "grounded", 1,
     "QLoRA 能在单张 48GB GPU 上微调 65B 模型，梯度通过冻结的 4-bit 量化模型反传到 LoRA 适配器。"),
    ("Retrieval-Augmented Generation for Knowledge", "RAG-Sequence", "grounded", 1,
     "RAG-Sequence 用同一篇文档预测每个目标 token，RAG-Token 可以为每个 token 使用不同文档。"),
    ("ColBERT", "late interaction architecture", "grounded", 1,
     "ColBERT 用 BERT 分别编码 query 和文档，再做一次细粒度的 late interaction，"
     "因此文档表示可以离线预计算。"),
    ("From Local to Global", "Leiden", "grounded", 1,
     "GraphRAG 利用图的模块化结构，把图划分成嵌套社区（如 Louvain、Leiden），"
     "再用 LLM 递归生成跨社区的分层摘要。"),
    ("Lost in the Middle", "U-shaped", "grounded", 1,
     "模型对上下文开头和结尾的信息利用更好，对中间位置的信息利用明显更差，性能呈 U 型曲线。"),
    ("MinerU", "filter out unprocessable", "grounded", 1,
     "MinerU 用 PyMuPDF 读取 PDF，过滤加密等不可处理文件，并提取文档元数据。"),
    ("Sentence-BERT", "Siamese", "grounded", 1,
     "Sentence-BERT 用孪生 BERT 网络生成句子嵌入，避免 BERT 必须把两个句子一起输入带来的巨大计算开销。"),
    ("Dense Passage Retrieval", "In-batch negatives", "grounded", 1,
     "DPR 训练时把 batch 内其他问题的段落当作负例（in-batch negatives），一次可训练 B² 个问答对。"),
    ("Self-RAG", "reflection token", "grounded", 1,
     "Self-RAG 让模型生成 reflection token，从而在推理阶段控制是否检索以及如何评价生成内容。"),
    ("Ragas", "inferred from the context", "grounded", 1,
     "RAGAS 的 faithfulness 指标先把回答拆成陈述，再判断每条陈述能否从上下文推出，"
     "分数是被支持陈述的占比。"),

    # ---------- 与证据矛盾（0） ----------
    ("Attention Is All You Need", "parallel attention layers", "contradicted", 0,
     "论文使用 h=16 个注意力头，每个头的维度 d_k=32。"),
    ("Attention Is All You Need", "Adam optimizer", "contradicted", 0,
     "训练使用 SGD 优化器，warmup 10000 步。"),
    ("Attention Is All You Need", "sine and cosine", "contradicted", 0,
     "位置编码使用可学习的位置嵌入，而不是正弦函数。"),
    ("LoRA: Low-Rank", "10,000 times", "contradicted", 0,
     "LoRA 把可训练参数减少 100 倍，GPU 显存降低 2 倍。"),
    ("LoRA: Low-Rank", "r = 1", "contradicted", 0,
     "实验发现秩 r 至少要取 64 才能有效。"),
    ("QLoRA", "48GB GPU", "contradicted", 0,
     "QLoRA 需要 8 张 48GB GPU 才能微调 65B 模型。"),
    ("Retrieval-Augmented Generation for Knowledge", "RAG-Sequence", "contradicted", 0,
     "RAG-Token 用同一篇文档预测每个目标 token。"),
    ("ColBERT", "late interaction architecture", "contradicted", 0,
     "ColBERT 把 query 和文档拼在一起送入交叉编码器，只输出一个相关性分数。"),
    ("From Local to Global", "Leiden", "contradicted", 0,
     "GraphRAG 不做社区划分，直接对全文生成一段摘要。"),
    ("Lost in the Middle", "U-shaped", "contradicted", 0,
     "模型对上下文中部信息的利用最好。"),

    # ---------- 正确但与上下文无关（0） ----------
    ("Attention Is All You Need", "parallel attention layers", "off_topic", 0,
     "BERT 的 masked language modeling 会随机遮盖 15% 的 token。"),
    ("LoRA: Low-Rank", "10,000 times", "off_topic", 0,
     "FlashAttention 通过分块计算减少 GPU 显存读写。"),
    ("ColBERT", "late interaction architecture", "off_topic", 0,
     "ViT 把图像切成 16×16 的 patch 后送入 Transformer。"),
    ("Lost in the Middle", "U-shaped", "off_topic", 0,
     "Mamba 用状态空间模型替代注意力，复杂度随序列长度线性增长。"),
    ("MinerU", "PyMuPDF", "off_topic", 0,
     "DoRA 把权重更新分解为幅度和方向两部分。"),
]


def _build_contexts():
    chunks = load_paper_chunks()
    by_page = {}
    for d in chunks:
        m = d.metadata
        by_page.setdefault((m.get("paper_id"), m.get("page")), []).append(d)
    for key in by_page:
        by_page[key].sort(key=lambda d: d.metadata.get("chunk_seq") or 0)

    rows = []
    for paper, kw, cat, label, answer in PAIRS:
        hit = None
        for d in chunks:
            if paper in (d.metadata.get("paper_title") or "") and kw.lower() in d.page_content.lower():
                hit = d
                break
        if hit is None:
            print(f"[warn] 找不到证据: {paper} / {kw}", file=sys.stderr)
            continue
        key = (hit.metadata.get("paper_id"), hit.metadata.get("page"))
        # 与生产接近：一次问答会喂 top-8 块；这里按整页拼，上限放到 8000 字，
        # 4000 会把长页后半段截掉（DPR 那题的 B² 证据就在后半段，曾误判为裁判失分）。
        ctx = "\n".join(d.page_content for d in by_page.get(key, [hit]))[:8000]
        rows.append({
            "paper": paper, "keyword": kw, "category": cat, "label": label,
            "answer": answer, "context": ctx,
            "page": hit.metadata.get("page"),
        })
    return rows


def _run_model(model, rows):
    config.JUDGE_PROVIDER = "local"
    config.JUDGE_MODEL = model
    out = []
    for i, r in enumerate(rows, 1):
        t0 = time.time()
        try:
            res = grounding.verify_grounded(r["context"], r["answer"])
            err = ""
        except Exception as e:  # verify_grounded 内部已兜底，这里只是保险
            res, err = {"grounded": False, "score": 0.0, "detail": []}, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        detail = res.get("detail") or []
        row = {
            "model": model, "idx": i, "category": r["category"], "label": r["label"],
            "grounded": int(bool(res.get("grounded"))), "score": res.get("score", 0.0),
            "statements": len(detail), "json_ok": int(bool(detail)),
            "latency_s": round(dt, 2), "page": r["page"], "answer": r["answer"][:50],
            "error": err,
        }
        out.append(row)
        flag = "OK " if row["grounded"] == r["label"] else "ERR"
        print(f"[{model}] {i:02d}/{len(rows)} {flag} {r['category']:12s} "
              f"label={r['label']} score={row['score']:.2f} st={row['statements']} {dt:.1f}s")
    return out


def _summary(rows):
    n = len(rows)
    acc = sum(r["grounded"] == r["label"] for r in rows) / n
    fa = sum(1 for r in rows if r["label"] == 0 and r["grounded"] == 1)
    fr = sum(1 for r in rows if r["label"] == 1 and r["grounded"] == 0)
    json_ok = sum(r["json_ok"] for r in rows) / n
    lat = [r["latency_s"] for r in rows]
    neg = sum(1 for r in rows if r["label"] == 0)
    pos = sum(1 for r in rows if r["label"] == 1)
    return {
        "n": n,
        "accuracy": round(acc, 3),
        "false_accept": f"{fa}/{neg}",
        "false_reject": f"{fr}/{pos}",
        "json_ok": round(json_ok, 3),
        "lat_mean": round(statistics.mean(lat), 2),
        "lat_p90": round(sorted(lat)[max(0, int(n * 0.9) - 1)], 2),
        "lat_max": round(max(lat), 2),
    }


def main():
    ap = argparse.ArgumentParser(description="接地判官校准")
    ap.add_argument("--models", default="qwen3:1.7b,llama3.2:3b,qwen3:4b")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 对（调试用）")
    args = ap.parse_args()

    rows = _build_contexts()
    if args.limit:
        rows = rows[: args.limit]
    print(f"pairs={len(rows)}  grounded={sum(r['label'] for r in rows)}  "
          f"unsupported={sum(1 for r in rows if r['label'] == 0)}")

    all_rows = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n===== {model} =====")
        all_rows.extend(_run_model(model, rows))

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"eval_results_paper_judge_{ts}.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\ncsv={path}")

    print("\n===== summary =====")
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        s = _summary([r for r in all_rows if r["model"] == model])
        print(f"{model:14s} acc={s['accuracy']:.3f} FA={s['false_accept']} "
              f"FR={s['false_reject']} json_ok={s['json_ok']:.2f} "
              f"lat={s['lat_mean']}s p90={s['lat_p90']}s max={s['lat_max']}s")


if __name__ == "__main__":
    main()
