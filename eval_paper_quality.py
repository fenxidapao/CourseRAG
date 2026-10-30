"""eval_paper_quality.py — 论文库回答质量四指标 + 引用页准确率（v3 P1，2026-10-30）

两段式（8 GB 显存：7B 生成与 9B 裁判不能同驻）：
  1) --stage generate：qwen2.5:7b 逐题答题，落 _gen.csv（含检索块，供裁判阶段复用）
  2) ollama stop qwen2.5:7b
  3) --stage judge：glm4:9b 打分（直接复用 eval_local 的四指标），落 _judge.csv

指标：faithfulness / answer_relevancy / context_precision / context_recall（均来自 eval_local）
     + answer_cite_page_acc：回答里的 [n] 映射到 citations[n-1].page，命中 expected_pages 记 1。
硬拒答题（rejected=1）不计入四指标均值，单独计数；引用页准确率只统计有引用的作答。

运行：
  python eval_paper_quality.py --stage generate
  ollama stop qwen2.5:7b
  python eval_paper_quality.py --stage judge
"""
import argparse
import csv
import datetime
import glob
import json
import os
import re
import subprocess
import sys
import time
from types import SimpleNamespace

import config
import rag_core
from eval_data_paper import PAGE_CASES, PAGE_CASES_ALL

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def used_pages(answer, cites):
    idxs = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    return [cites[i - 1] for i in idxs if 1 <= i <= len(cites)]


def _write_csv(path, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_generate(limit=0, cases_mode="base"):
    base = PAGE_CASES_ALL if cases_mode == "all" else PAGE_CASES
    cases = base[:limit] if limit else base
    rows = []
    for i, c in enumerate(cases, 1):
        t0 = time.time()
        res = rag_core.answer_question(
            c["q"], mode="accurate", use_cache=False, use_rewrite=False
        )
        latency = round(time.time() - t0, 1)
        cites = [x.get("page") for x in (res.get("citations") or [])]
        answer = (res.get("answer") or "").replace("\n", " ").strip()
        used = used_pages(answer, cites)
        hit = int(bool(set(used) & set(c["expected_pages"])))
        blocks = [d.page_content for d in (res.get("docs") or [])]
        rows.append({
            "idx": i,
            "paper_id": c["paper_id"],
            "question": c["q"],
            "expected_pages": "/".join(map(str, c["expected_pages"])),
            "reference": c["reference"],
            "rejected": int(bool(res.get("rejected"))),
            "used_pages": "/".join(map(str, used)),
            "cite_hit": hit,
            "latency_s": latency,
            "answer": answer[:700],
            "contexts_json": json.dumps(blocks, ensure_ascii=False),
        })
        print(f"{i:02d} cite_hit={hit} used={rows[-1]['used_pages'] or '-'} "
              f"want={rows[-1]['expected_pages']} rejected={rows[-1]['rejected']} {latency}s",
              flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"eval_results_paper_quality_{ts}_gen.csv")
    _write_csv(path, rows)
    print(f"\n[csv] {path}")
    print(f"[gen] n={len(rows)} cite_page_acc={sum(r['cite_hit'] for r in rows)/len(rows):.3f} "
          f"rejected={sum(r['rejected'] for r in rows)}")
    return path


def _warn_if_generator_loaded():
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return
    if config.LLM_MODEL.split(":")[0] in out:
        print(f"[warn] 生成模型 {config.LLM_MODEL} 仍在显存中；先 `ollama stop {config.LLM_MODEL}` "
              "再跑裁判阶段，否则 7B + 9B 会挤爆 8 GB", flush=True)


def run_judge(gen_csv="", limit=0):
    if not gen_csv:
        cands = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "eval_results_paper_quality_*_gen.csv")))
        if not cands:
            sys.exit("找不到 *_gen.csv，先跑 --stage generate")
        gen_csv = cands[-1]
    with open(gen_csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]

    _warn_if_generator_loaded()
    from eval_local import (faithfulness, answer_relevancy,
                            context_precision, context_recall)

    out_rows = []
    for r in rows:
        answer = r["answer"]
        blocks = json.loads(r["contexts_json"] or "[]")
        context = "\n\n".join(blocks)
        scores = {m: None for m in METRICS}
        if int(r["rejected"]) == 0 and answer:
            docs = [SimpleNamespace(page_content=b) for b in blocks]
            scores["faithfulness"] = faithfulness(context, answer)["score"]
            scores["answer_relevancy"] = answer_relevancy(r["question"], answer)["score"]
            scores["context_precision"] = context_precision(r["question"], docs)["score"]
            scores["context_recall"] = context_recall(
                r["question"], r["reference"], context)["score"]
        row = {**r, **scores}
        row.pop("contexts_json", None)
        out_rows.append(row)
        print(f"{r['idx']:>2} faith={scores['faithfulness']} rel={scores['answer_relevancy']} "
              f"prec={scores['context_precision']} rec={scores['context_recall']}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = gen_csv.replace("_gen.csv", f"_judge_{ts}.csv")
    _write_csv(path, out_rows)

    answered = [x for x in out_rows if x["faithfulness"] is not None]
    print(f"\n[csv] {path}")
    print(f"[judge] n={len(out_rows)} answered={len(answered)} "
          f"rejected={len(out_rows)-len(answered)}")
    for m in METRICS:
        if answered:
            print(f"  {m:18s} {sum(x[m] for x in answered)/len(answered):.3f}")
    if out_rows:
        print(f"  {'answer_cite_page_acc':18s} {sum(int(x['cite_hit']) for x in out_rows)/len(out_rows):.3f}")


def main():
    ap = argparse.ArgumentParser(description="论文库回答质量评估（两段式）")
    ap.add_argument("--stage", choices=["generate", "judge"], required=True)
    ap.add_argument("--gen-csv", default="", help="judge 阶段指定输入 CSV（默认取最新）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--cases", default="base", choices=["base", "all"],
                    help="base=原 15 题四指标；all=30 题只看引用页")
    args = ap.parse_args()
    if args.stage == "generate":
        run_generate(args.limit, args.cases)
    else:
        run_judge(args.gen_csv, args.limit)


if __name__ == "__main__":
    main()
