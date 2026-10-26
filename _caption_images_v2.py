# -*- coding: utf-8 -*-
"""B3 步 2：换描述器 —— 用视觉模型重跑教材插图的 caption。

## 为什么要换（实测依据，不是感觉）

旧链路（已归档的 `_caption_images.py`）：本地 `qwen2.5vl:7b` + prompt 要「用 2~3 句中文描述这张教材插图」。
这个 prompt 在**结构上**就不会写出文件名 —— 它要的是「描述」，不是「抄录」。

实测（B1 caption / 上限探针，已归档；2026-10-14）：
`微信小程序开发实战` 四张项目结构截图的判别性 token（`tab1.wxml` / `avatar.png` /
`用户登录` / `比较数字大小`）在**全部 215 条 caption 中出现 0 次**；这四张图在
`where={"type":"image"}` 专用图像通道里的名次是 **56 / 66 / 67 / 69（分母 98）**。
B1 的视觉重排因此拿不到候选 —— 瓶颈在索引层，不在排序层。

## 两个杠杆分开试，实测结论

B3 prompt 探针（已归档）在四张 ground-truth 图上跑了三种 prompt（同一个 `qwen3-vl-plus`）：

    CURRENT(旧 prompt 原文)   3/4
    TRANSCRIBE(纯抄录)        3/4
    HYBRID(描述+抄录)         4/4   ← 采用

**关键发现：模型才是主杠杆。** 旧 prompt 配 `qwen3-vl-plus` 就有 3/4（旧链路 0/4），
说明「像素里明明有字、caption 里没有」主要不是 prompt 的锅，是 `qwen2.5vl:7b` 抄不出来。
HYBRID 在旧 prompt 基础上再补 1 张（`p167_i00.png` 的根节点「用户登录」——旧 prompt 把它
读成了「用户目录」，HYBRID 要求抄录才抄对）。

HYBRID 同时保留描述句，所以不牺牲语义检索面（纯 TRANSCRIBE 只剩一串文件名，对
「这张图讲了什么」类问题是退化的）。

## 用法

    # 先小样本看质量（不动全量）
    python _caption_images_v2.py --limit 12 --out data/images/_image_captions_v2.csv

    # 全量（支持断点续跑：已在输出里且 caption 非空的行会跳过）
    python _caption_images_v2.py --out data/images/_image_captions_v2.csv

    # 对比新旧
    python _caption_images_v2.py --compare data/images/_image_captions_v2.csv

API key 只从环境变量或 `~/.dsh/.credentials.yaml` 读，**不写进仓库**。
"""
import argparse
import base64
import csv
import json
import os
import pathlib
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "data" / "images"
MANIFEST = IMAGES_DIR / "_manifest.csv"
OLD_CSV = IMAGES_DIR / "_image_captions.csv"

# ---- 采用的 prompt（HYBRID，探针实测 4/4） ----
PROMPT = (
    "请用 1~2 句中文描述这张教材插图的内容类型和主要元素，然后另起一行，"
    "以「图中文字：」开头，**逐字抄出图中出现的所有可见文字**"
    "（包括标题、根节点名称、文件夹名、文件名、按钮文字、命令、数字）。"
    "不要遗漏任何文件名，不要改写，不要翻译，不要推测没看到的内容。"
    "如果图中确实没有可辨认文字，就在「图中文字：」后面写「无可辨认文字」。"
)

FIELDS = ["png_path", "caption", "elapsed_s", "model", "prompt_tokens",
          "completion_tokens", "total_tokens", "cost", "error"]


# ---------------------------------------------------------------- key

def _resolve_key():
    k = os.environ.get("VISUAL_RERANK_API_KEY") or os.environ.get("KUAIPAO_API_KEY")
    if k:
        return k.strip()
    cred = pathlib.Path.home() / ".dsh" / ".credentials.yaml"
    if cred.is_file():
        m = re.search(r"KUAIPAO_API_KEY:\s*(\S+)", cred.read_text(encoding="utf-8"))
        if m:
            return m.group(1).strip()
    raise SystemExit(
        "找不到 API key。设环境变量 VISUAL_RERANK_API_KEY，"
        "或把 KUAIPAO_API_KEY 写进 ~/.dsh/.credentials.yaml。"
    )


# ---------------------------------------------------------------- 调用

def caption_one(png: pathlib.Path, key: str, api_base: str, model: str,
                detail: str, max_tokens: int, retries: int = 4):
    """返回 (caption, elapsed_s, usage_dict, error_or_None)。"""
    b64 = base64.b64encode(png.read_bytes()).decode()
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}", "detail": detail}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }
    req = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    t0 = time.time()
    err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            txt = msg.get("content") or ""
            if isinstance(txt, list):
                txt = " ".join(str(c.get("text", "")) for c in txt if isinstance(c, dict))
            txt = re.sub(r"[ \t]+", " ", txt).strip()
            if not txt:
                # 推理模型可能把预算烧在 reasoning 上 -> 视为失败，重试
                err = f"空 content（finish={data['choices'][0].get('finish_reason')}）"
                time.sleep(2 * (attempt + 1))
                continue
            return txt, round(time.time() - t0, 2), data.get("usage", {}) or {}, None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
    return "", round(time.time() - t0, 2), {}, err


# ---------------------------------------------------------------- 数据

def load_manifest():
    with open(MANIFEST, encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if (r.get("png_path") or "").strip()]


def load_csv_map(path: pathlib.Path):
    """→ {png_path: row}"""
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {(r.get("png_path") or "").strip(): r for r in csv.DictReader(f) if r.get("png_path")}


def _done_ok(path: pathlib.Path):
    """已成功、可跳过的 png_path 集合。"""
    out = set()
    for k, r in load_csv_map(path).items():
        if (r.get("caption") or "").strip() and not (r.get("error") or "").strip():
            out.add(k)
    return out


# ---------------------------------------------------------------- 对比

TOKEN_RE = re.compile(r"\b[\w\-]+\.(?:js|json|wxml|wxss|png|jpg|jpeg|gif|py|md|txt|sh|c|h|cpp|java|html|css|xml|yaml|yml|toml|ini|cfg|conf|sql|ipynb)\b",
                      re.IGNORECASE)
QUOTED_RE = re.compile(r"[“\"'「『]([^”\"'」』]{2,20})[”\"'」』]")
# 四张 ground-truth 图的判别性 token（人工看图确认，不来自 caption）
GT = {
    "data/images/微信小程序开发实战/p267_i00.png": ["tab1.wxml", "tab2.wxml", "tab3.wxml"],
    "data/images/微信小程序开发实战/p045_i00.png": ["avatar.png"],
    "data/images/微信小程序开发实战/p167_i00.png": ["用户登录"],
    "data/images/微信小程序开发实战/p082_i00.png": ["比较数字大小"],
}


def _stats(caps: dict):
    n = len(caps)
    if not n:
        return {}
    lens = [len(v) for v in caps.values()]
    with_file = sum(1 for v in caps.values() if TOKEN_RE.search(v))
    with_quote = sum(1 for v in caps.values() if QUOTED_RE.search(v))
    with_block = sum(1 for v in caps.values() if "图中文字" in v)
    return {
        "n": n,
        "avg_len": round(sum(lens) / n, 1),
        "min_len": min(lens),
        "max_len": max(lens),
        "file_token": with_file,
        "file_token_pct": round(100 * with_file / n, 1),
        "quoted": with_quote,
        "quoted_pct": round(100 * with_quote / n, 1),
        "has_text_block": with_block,
    }


def compare(new_path: pathlib.Path):
    old = {k: (r.get("caption") or "") for k, r in load_csv_map(OLD_CSV).items()}
    new = {k: (r.get("caption") or "") for k, r in load_csv_map(new_path).items()}
    common = {k: v for k, v in new.items() if k in old and v}
    so, sn = _stats(old), _stats(common)
    print(f"旧 {OLD_CSV.name}: {so}")
    print(f"新 {new_path.name}（与旧交集 {len(common)} 条）: {sn}")
    print()
    print("四张 ground-truth 图的判别性 token 覆盖：")
    for p, toks in GT.items():
        o = old.get(p, "")
        n = new.get(p, "")
        of = [t for t in toks if t in o]
        nf = [t for t in toks if t in n]
        print(f"  {p.rsplit('/', 1)[-1]:16s} 旧 {len(of)}/{len(toks)} {of}  →  新 {len(nf)}/{len(toks)} {nf}")
    print()
    print("新 caption 里出现文件名类 token 的条数（旧 vs 新）："
          f"{so.get('file_token')} ({so.get('file_token_pct')}%) → {sn.get('file_token')} ({sn.get('file_token_pct')}%)")
    print("新 caption 里带引号标题的条数（旧 vs 新）："
          f"{so.get('quoted')} ({so.get('quoted_pct')}%) → {sn.get('quoted')} ({sn.get('quoted_pct')}%)")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(IMAGES_DIR / "_image_captions_v2.csv"))
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（0=全部）")
    ap.add_argument("--books", default="", help="逗号分隔，只跑这些书")
    ap.add_argument("--model", default=os.environ.get("B3_MODEL", "qwen3-vl-plus"))
    ap.add_argument("--api-base", default=os.environ.get("B3_API_BASE", "https://kuaipao.pro/v1"))
    ap.add_argument("--detail", default=os.environ.get("B3_DETAIL", "high"))
    ap.add_argument("--max-tokens", type=int, default=int(os.environ.get("B3_MAX_TOKENS", "700")))
    ap.add_argument("--threads", type=int, default=int(os.environ.get("B3_THREADS", "3")))
    ap.add_argument("--compare", default="", help="只对比，不跑图")
    ap.add_argument("--show", type=int, default=0, help="打印前 N 条结果")
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    if args.compare:
        compare(pathlib.Path(args.compare))
        return

    key = _resolve_key()
    rows = load_manifest()
    if args.books:
        want = {b.strip() for b in args.books.split(",") if b.strip()}
        rows = [r for r in rows if (r.get("book") or "").strip() in want]

    done = _done_ok(out_path)
    todo = [r for r in rows if r["png_path"].strip() not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"model={args.model}  detail={args.detail}  max_tokens={args.max_tokens}  threads={args.threads}")
    print(f"manifest {len(rows)} 张 | 已完成 {len(done)} 张 | 本轮待跑 {len(todo)} 张")
    print(f"输出 {out_path}")
    if not todo:
        print("没有待跑项。用 --compare 看新旧对比。")
        return

    # 首次运行写表头；续跑时保留已有行
    existing = load_csv_map(out_path)
    write_header = not out_path.is_file()
    t_start = time.time()
    ok = fail = 0
    tot_cost = 0.0

    f = open(out_path, "a", encoding="utf-8-sig", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if write_header:
        w.writeheader()

    old_map = {k: (v.get("caption") or "") for k, v in load_csv_map(OLD_CSV).items()}

    def work(r):
        rel = r["png_path"].strip()
        png = ROOT / rel
        if not png.is_file():
            return rel, "", 0.0, {}, f"文件不存在: {png}"
        cap, dt, usage, err = caption_one(png, key, args.api_base, args.model,
                                          args.detail, args.max_tokens)
        # 兜底：新描述器失败时回退旧 caption。
        # 原因：rag_core.load_image_chunks 会**跳过空 caption**，所以一条失败 =
        # 一张图从索引里消失（语料 215→214）。宁可保留旧描述，也不让语料缩水。
        if err and old_map.get(rel):
            return rel, old_map[rel], dt, usage, f"{err} [已回退旧 caption]"
        return rel, cap, dt, usage, err

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
            futs = {ex.submit(work, r): r for r in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                rel, cap, dt, usage, err = fut.result()
                cost = usage.get("cost") or 0.0
                try:
                    tot_cost += float(cost)
                except (TypeError, ValueError):
                    pass
                row = {
                    "png_path": rel, "caption": cap, "elapsed_s": dt, "model": args.model,
                    "prompt_tokens": usage.get("prompt_tokens", ""),
                    "completion_tokens": usage.get("completion_tokens", ""),
                    "total_tokens": usage.get("total_tokens", ""),
                    "cost": cost, "error": err or "",
                }
                w.writerow(row)
                f.flush()
                if err:
                    fail += 1
                    print(f"  [{i}/{len(todo)}] FAIL {rel.rsplit('/', 1)[-1]}  {err}")
                else:
                    ok += 1
                    print(f"  [{i}/{len(todo)}] ok   {rel.rsplit('/', 1)[-1]:18s} {dt:5.1f}s "
                          f"{len(cap):4d}字  {cap[:70].replace(chr(10), ' / ')}")
    finally:
        f.close()

    dt_all = time.time() - t_start
    print()
    print(f"完成 {ok} 张 / 失败 {fail} 张，耗时 {dt_all:.0f}s，累计 cost={tot_cost:.6f}")
    print(f"输出：{out_path}")
    if args.show:
        print()
        for k, r in list(load_csv_map(out_path).items())[:args.show]:
            print(f"--- {k}\n{r.get('caption')}\n")


if __name__ == "__main__":
    main()
