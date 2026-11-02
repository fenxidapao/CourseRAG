"""eval_paper_quality.py — 论文库回答质量四指标 + 引用口径（v3/v3.2，2026-10-30）

两段式（8 GB 显存：7B 生成与 9B 裁判不能同驻）：
  1) --stage generate：qwen2.5:7b 逐题答题，落 _gen.csv（含检索块，供后续阶段复用）
  2) ollama stop qwen2.5:7b
  3) --stage judge：glm4:9b 打分（直接复用 eval_local 的四指标），落 _judge.csv
  4) --stage citecheck：glm4:9b 判「带 [n] 的句子是否被第 n 条 chunk 支撑」，落 _cite.csv

指标：
  - faithfulness / answer_relevancy / context_precision / context_recall（来自 eval_local）
  - answer_cite_page_acc（cite_page_strict）：[n] → citations[n-1].page 命中 expected_pages 记 1；
    这是**严格下界**——事实若在别页重复，命中也不算。
  - cite_supported：每个带 [n] 的句子 → 对应的第 n 条 chunk 是否支持该句（v3.2 第二口径）。
硬拒答题不计入四指标均值；引用口径只统计有引用的作答。

运行：
  python eval_paper_quality.py --stage generate
  ollama stop qwen2.5:7b
  python eval_paper_quality.py --stage judge
  python eval_paper_quality.py --stage citecheck
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
import llm_provider
import rag_core
from eval_data_paper import PAGE_CASES, PAGE_CASES_ALL

METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def used_pages(answer, cites):
    idxs = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    return [cites[i - 1] for i in idxs if 1 <= i <= len(cites)]


_CITE_SYS = "你是严格的 RAG 引用核验器：只判断【证据段落】是否支持【陈述】，不要使用外部知识。只输出一个 JSON 对象。"
_SENT_SPLIT = re.compile(r"(?<=[。！？!?\n])")


def _cited_sentences(answer):
    """拆句，只保留带 [n] 的句子；返回 [(句子, [n...])]。

    纯引用标记的碎片（如单独一行的 `[1][7]`）并入前一句，否则会被当成空陈述送审。
    """
    frags = []
    for s in _SENT_SPLIT.split(answer or ""):
        s = s.strip()
        if not s:
            continue
        if not re.sub(r"\[(\d+)\]", "", s).strip(" \t\r\n。；;，,、.!！?？") and frags:
            frags[-1] = frags[-1] + " " + s
            continue
        frags.append(s)
    out = []
    for s in frags:
        idxs = sorted({int(x) for x in re.findall(r"\[(\d+)\]", s)})
        if idxs:
            out.append((s, idxs))
    return out


def _judge_support(sentence, chunk):
    """裁判判一条陈述是否被证据段落支持；异常/解析失败按不支持处理（保守）。"""
    user = f"""【证据段落】
{chunk[:3000]}

【陈述】
{sentence[:600]}

这条陈述中的事实是否能由【证据段落】直接支持？只输出 JSON：{{"supported": true/false}}"""
    try:
        out = llm_provider.judge_chat(
            [{"role": "system", "content": _CITE_SYS}, {"role": "user", "content": user}],
            temperature=0.0,
        )
    except Exception:
        return False
    m = re.search(r'"supported"\s*:\s*(true|false)', out or "", re.I)
    return bool(m) and m.group(1).lower() == "true"


def _union_text(blocks, metas, n):
    """第 n 条引用所在页的「本轮召回块」合并文本（v3.3 高亮范围口径）。"""
    if not metas or not (1 <= n <= len(metas)):
        return blocks[n - 1] if 1 <= n <= len(blocks) else ""
    tgt = metas[n - 1]
    same = [
        i for i, m in enumerate(metas, 1)
        if i <= len(blocks) and m.get("paper_id") == tgt.get("paper_id")
        and m.get("page") == tgt.get("page")
    ] or [n]
    return "\n\n".join(blocks[i - 1] for i in same)


def run_citecheck(gen_csv="", limit=0):
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

    out_rows = []
    for r in rows:
        blocks = json.loads(r.get("contexts_json") or "[]")
        metas = json.loads(r.get("cites_json") or "[]")
        total = supported = union_supported = 0
        detail = []
        for sent, idxs in _cited_sentences(r.get("answer", "")):
            total += 1
            ok = any(1 <= n <= len(blocks) and _judge_support(sent, blocks[n - 1]) for n in idxs)
            ok_union = ok or any(
                1 <= n <= len(blocks) and _judge_support(sent, _union_text(blocks, metas, n))
                for n in idxs
            )
            supported += int(ok)
            union_supported += int(ok_union)
            detail.append({"sentence": sent[:120], "cites": idxs,
                           "supported": ok, "union_supported": ok_union})
        row = {k: v for k, v in r.items() if k not in ("contexts_json", "cites_json")}
        row["cite_sentences"] = total
        row["cite_supported_frac"] = round(supported / total, 3) if total else None
        row["cite_supported_all"] = int(total > 0 and supported == total)
        row["cite_union_frac"] = round(union_supported / total, 3) if total else None
        row["cite_union_all"] = int(total > 0 and union_supported == total)
        row["cite_detail"] = json.dumps(detail, ensure_ascii=False)
        out_rows.append(row)
        print(f"{r['idx']:>2} sentences={total} supported={supported} "
              f"union={union_supported} page_hit={r['cite_hit']}", flush=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = gen_csv.replace("_gen.csv", f"_cite_{ts}.csv")
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    judged = [x for x in out_rows if x["cite_sentences"]]
    sents = sum(x["cite_sentences"] for x in judged)
    sup = sum(round(x["cite_supported_frac"] * x["cite_sentences"]) for x in judged)
    uni = sum(round(x["cite_union_frac"] * x["cite_sentences"]) for x in judged)
    all_ok = sum(x["cite_supported_all"] for x in judged)
    all_uni = sum(x["cite_union_all"] for x in judged)
    page_ok = sum(int(x["cite_hit"]) for x in out_rows)
    print(f"\n[csv] {path}")
    print(f"[cite] questions={len(out_rows)} with_citation={len(judged)} "
          f"page_strict={page_ok}/{len(out_rows)}={page_ok/len(out_rows):.3f}")
    print(f"       single_sentence_support={sup}/{sents}={sup/sents:.3f}  "
          f"single_question_all={all_ok}/{len(judged)}={all_ok/max(1,len(judged)):.3f}")
    print(f"       union_sentence_support={uni}/{sents}={uni/sents:.3f}  "
          f"union_question_all={all_uni}/{len(judged)}={all_uni/max(1,len(judged)):.3f}")


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
        cite_meta = [
            {"paper_id": x.get("paper_id"), "page": x.get("page"),
             "block_ids": x.get("block_ids") or ""}
            for x in (res.get("citations") or [])
        ]
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
            "answer": answer[:2000],
            "contexts_json": json.dumps(blocks, ensure_ascii=False),
            "cites_json": json.dumps(cite_meta, ensure_ascii=False),
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
    ap.add_argument("--stage", choices=["generate", "judge", "citecheck"], required=True)
    ap.add_argument("--gen-csv", default="", help="judge 阶段指定输入 CSV（默认取最新）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    ap.add_argument("--cases", default="base", choices=["base", "all"],
                    help="generate 用：base=原 15 题，all=30 题；judge/citecheck 按输入 CSV 行数跑")
    args = ap.parse_args()
    if args.stage == "generate":
        run_generate(args.limit, args.cases)
    elif args.stage == "citecheck":
        run_citecheck(args.gen_csv, args.limit)
    else:
        run_judge(args.gen_csv, args.limit)


if __name__ == "__main__":
    main()
