"""eval_paper.py — 论文库评估（页级命中 / 拒答门控 + 阈值扫描）。

用法：
  python eval_paper.py --group page --gate
  python eval_paper.py --group reject --scan      # 输出每题 top1 向量/精排分 CSV
  python eval_paper.py --group reject --grid      # 在扫描 CSV 上网格搜索阈值
  python eval_paper.py --group reject --gate --vec-thr 0.5 --rr-thr 0.2

只做检索 + 重排，不生成；CSV 写 eval_results_paper_<group>_<ts>.csv（gitignored）。
"""
from __future__ import annotations

import argparse
import csv
import datetime
import itertools
import sys

import config
from eval_data_paper import (HELDOUT_IN, HELDOUT_OUT, IN_DOMAIN, OUT_DOMAIN,
                            PAGE_CASES, PAGE_CASES_ALL)
from rag_core import retrieve_and_rerank


def _top(docs):
    if not docs:
        return None
    return docs[0]


def _out_csv(group):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"eval_results_paper_{group}_{ts}.csv"


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"csv={path}")


def run_page(mode="accurate", cases="all"):
    rows, paper_hit, page_hit, strict = [], 0, 0, 0
    case_list = PAGE_CASES_ALL if cases == "all" else PAGE_CASES
    for c in case_list:
        docs = retrieve_and_rerank(c["q"], force_no_rerank=(mode != "accurate"))
        papers = {d.metadata.get("paper_id") for d in docs}
        pages = {(d.metadata.get("paper_id"), d.metadata.get("page")) for d in docs}
        want = {(c["paper_id"], p) for p in c["expected_pages"]}
        ph = c["paper_id"] in papers
        gh = bool(want & pages)
        top = _top(docs)
        sh = bool(top) and (top.metadata.get("paper_id"), top.metadata.get("page")) in want
        paper_hit += ph
        page_hit += gh
        strict += sh
        rows.append({
            "question": c["q"],
            "paper_id": c["paper_id"],
            "expected_pages": "/".join(map(str, c["expected_pages"])),
            "paper_hit@8": int(ph),
            "page_hit@8": int(gh),
            "page_strict@1": int(sh),
            "top1_paper": top.metadata.get("paper_id") if top else "",
            "top1_page": top.metadata.get("page") if top else "",
            "top1_section": (top.metadata.get("section") or "")[:60] if top else "",
        })
    n = len(case_list)
    print(f"page cases={n} paper_hit@8={paper_hit}/{n}={paper_hit/n:.3f} "
          f"page_hit@8={page_hit}/{n}={page_hit/n:.3f} "
          f"page_strict@1={strict}/{n}={strict/n:.3f}")
    if cases == "all" and rows:
        b = rows[:len(PAGE_CASES)]
        print(f"  base15: paper_hit@8={sum(r['paper_hit@8'] for r in b)}/{len(b)} "
              f"page_hit@8={sum(r['page_hit@8'] for r in b)}/{len(b)} "
              f"page_strict@1={sum(r['page_strict@1'] for r in b)}/{len(b)}")
    _write_csv(_out_csv("page"), rows)


def _scan_one(q):
    docs = retrieve_and_rerank(q)
    top = _top(docs)
    if not top:
        return 0.0, 0.0, "", ""
    return (
        top.metadata.get("vector_score") or 0.0,
        top.metadata.get("rerank_score") or 0.0,
        top.metadata.get("paper_id") or "",
        top.metadata.get("page") or "",
    )


def run_reject_scan():
    rows = []
    for group, qs in (("in_domain", IN_DOMAIN), ("out_domain", OUT_DOMAIN)):
        for q in qs:
            vs, rs, pid, page = _scan_one(q)
            rows.append({
                "group": group,
                "question": q,
                "top1_vector": round(vs, 4),
                "top1_rerank": round(rs, 4),
                "top1_paper": pid,
                "top1_page": page,
            })
            print(f"{group:10s} v={vs:.3f} r={rs:.3f} {q[:40]}")
    _write_csv(_out_csv("reject_scan"), rows)


def _rates(rows, vec_thr, rr_thr):
    def rejected(r):
        return r["top1_vector"] < vec_thr or r["top1_rerank"] < rr_thr

    out = [r for r in rows if r["group"] == "out_domain"]
    ind = [r for r in rows if r["group"] == "in_domain"]
    return (
        sum(rejected(r) for r in out) / len(out),
        sum(rejected(r) for r in ind) / len(ind),
        vec_thr,
        rr_thr,
    )


def run_reject_grid(path):
    with open(path, encoding="utf-8-sig") as f:
        rows = [{**r, "top1_vector": float(r["top1_vector"]), "top1_rerank": float(r["top1_rerank"])}
                for r in csv.DictReader(f)]
    vecs = [x / 100 for x in range(35, 76)]
    rrs = [x / 100 for x in range(5, 51, 5)]
    best = []
    for v, r in itertools.product(vecs, rrs):
        rej, false, _, _ = _rates(rows, v, r)
        if rej >= 0.90 and false <= 0.05:
            best.append((false, -rej, v, r, rej))
    best.sort()
    print(f"候选（out_reject>=0.90 且 in_false<=0.05）: {len(best)}")
    for false, _, v, r, rej in best[:10]:
        print(f"  vec={v:.2f} rr={r:.2f} out_reject={rej:.3f} in_false={false:.3f}")
    if not best:
        print("无组合满足门控；放宽目标或检查题集")


def run_reject_gate(vec_thr=None, rr_thr=None):
    vec_thr = config.HARD_REJECT_VECTOR_THRESHOLD if vec_thr is None else vec_thr
    rr_thr = config.HARD_REJECT_RERANK_THRESHOLD if rr_thr is None else rr_thr
    rows = []
    for group, qs in (("in_domain", IN_DOMAIN), ("out_domain", OUT_DOMAIN)):
        for q in qs:
            vs, rs, pid, page = _scan_one(q)
            rows.append({"group": group, "question": q, "top1_vector": round(vs, 4),
                         "top1_rerank": round(rs, 4), "top1_paper": pid, "top1_page": page})
    rej, false, v, r = _rates(rows, vec_thr, rr_thr)
    print(f"thresholds vec={v:.2f} rr={r:.2f}")
    print(f"out_domain reject={rej:.3f} (目标 >=0.90)  in_domain false_reject={false:.3f} (目标 <=0.05)")
    _write_csv(_out_csv("reject_gate"), rows)
    return rej, false


def run_heldout(vec_thr=None, rr_thr=None):
    """阈值定稿后只跑一次：held-out 10 域内 + 10 域外，不再回调阈值。"""
    vec_thr = config.HARD_REJECT_VECTOR_THRESHOLD if vec_thr is None else vec_thr
    rr_thr = config.HARD_REJECT_RERANK_THRESHOLD if rr_thr is None else rr_thr
    rows = []
    for group, qs in (("heldout_in", HELDOUT_IN), ("heldout_out", HELDOUT_OUT)):
        for q in qs:
            vs, rs, pid, page = _scan_one(q)
            rows.append({"group": group, "question": q, "top1_vector": round(vs, 4),
                         "top1_rerank": round(rs, 4), "top1_paper": pid, "top1_page": page})
    v, r = vec_thr, rr_thr

    def rejected(x):
        return x["top1_vector"] < v or x["top1_rerank"] < r
    out = [x for x in rows if x["group"] == "heldout_out"]
    ind = [x for x in rows if x["group"] == "heldout_in"]
    rej = sum(rejected(x) for x in out) / len(out)
    false = sum(rejected(x) for x in ind) / len(ind)
    print(f"thresholds vec={v:.2f} rr={r:.2f}")
    print(f"heldout out reject={rej:.3f} (>=0.90)  heldout in false_reject={false:.3f} (<=0.05)")
    _write_csv(_out_csv("heldout"), rows)
    return rej, false


def main():
    ap = argparse.ArgumentParser(description="论文库评估")
    ap.add_argument("--group", choices=["page", "reject"], required=True)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--heldout", action="store_true")
    ap.add_argument("--grid", default=None, metavar="SCAN_CSV")
    ap.add_argument("--mode", default="accurate", choices=["accurate", "fast"])
    ap.add_argument("--cases", default="all", choices=["base", "all"],
                    help="页级题集：base=原 15 题，all=30 题（默认）")
    ap.add_argument("--vec-thr", type=float, default=None)
    ap.add_argument("--rr-thr", type=float, default=None)
    args = ap.parse_args()

    print(f"[profile] {config.profile_summary()}  reranker={config.RERANKER_ENABLED}")
    if args.group == "page":
        run_page(args.mode, cases=args.cases)
    elif args.grid:
        run_reject_grid(args.grid)
    elif args.heldout:
        run_heldout(args.vec_thr, args.rr_thr)
    elif args.scan:
        run_reject_scan()
    elif args.gate:
        rej, false = run_reject_gate(args.vec_thr, args.rr_thr)
        sys.exit(0 if rej >= 0.90 and false <= 0.05 else 1)
    else:
        raise SystemExit("reject 组需要 --scan / --grid / --gate 之一")


if __name__ == "__main__":
    main()
