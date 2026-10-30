"""
grounding.py — 生成接地自检（grounded verifier，2026-08-24 工程化落地 P1）

为什么需要它（见点评文档任务2「幻觉控制 = 强系统提示 + verifier + 选择性拒答」）：
  仅靠 SYSTEM_RULES「只用来源」无法 100% 杜绝模型凭记忆补全。生产 RAG 标配一个
  verifier：生成后用独立裁判判断回答是否真的由检索上下文支撑，未接地则触发硬拒答。

与 eval_local.faithfulness 的关系：
  - 指标口径一致（把回答拆陈述、逐条判 supported、score=被支撑/总数）。
  - 但 eval_local 与 rag_core 互相 import，此处自包含实现，避免循环依赖；
    后续可把两者统一抽到 judge_utils 复用（TODO）。
  - 裁判走 llm_provider.judge_chat（按 config.JUDGE_PROVIDER 分流），温度 0 逼稳定 JSON。
"""
import json
import re

import config
import llm_provider  # C2：裁判按 config.JUDGE_PROVIDER 分流（local/openai）


def _extract_json(text: str):
    """抠第一个合法 JSON 对象；兼容 ```json 围栏。"""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _norm(text: str) -> str:
    """归一：去空白、标点、LaTeX 转义与大小写差异后比对。

    两处必须归一，否则证据永远对不上：
      - MinerU 数学文本是 `0 . 9 8` 这种带空格形式；
      - 裁判转录 evidence 时常加转义（`\\$beta _ { 2 } \\= 0 . 9 8`）。
    """
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", (text or "").lower())


def _bigrams(text: str) -> set:
    """字符二元组集合：中英混排都能粗粒度衡量"这段和回答是否讲同一件事"。"""
    text = re.sub(r"\s+", "", text.lower())
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _trim_context(context: str, answer: str, max_chars: int) -> str:
    """按与回答的相关度裁剪上下文，避免超出裁判的 num_ctx。

    2026-10-30 实测：8 块上下文 19,223 字符 = 10,836 token，glm4:9b（num_ctx=8192）
    直接 400 exceed_context_size，verify_grounded 吞异常按 0.0 判不接地 →
    正确答案被当幻觉拒掉（15 题里 2 例）。裁判不该被上下文长度拖垮。
    按段切分（grounded_generate 用 "\\n\\n" 拼块），优先保留与回答共享二元组最多的段，
    保持原始顺序，超预算则整段丢弃；单段超预算时截断。
    """
    if len(context) <= max_chars:
        return context
    blocks = [b for b in context.split("\n\n") if b.strip()]
    if not blocks:
        return context[:max_chars]
    ans_grams = _bigrams(answer)
    ranked = sorted(
        range(len(blocks)),
        key=lambda i: len(ans_grams & _bigrams(blocks[i])),
        reverse=True,
    )
    keep, total = set(), 0
    for i in ranked:
        size = len(blocks[i]) + 2
        if total + size > max_chars:
            continue
        keep.add(i)
        total += size
    if not keep:  # 单段就超预算 → 截断最相关那段
        return blocks[ranked[0]][:max_chars]
    return "\n\n".join(blocks[i] for i in sorted(keep))


_JUDGE_SYSTEM = (
    "你是严谨的 RAG 接地裁判。只依据给定的【上下文】判断，严禁使用外部知识。"
    "必须只输出一个 JSON 对象，不要任何额外文字或解释。"
)


def judge_messages(context: str, answer: str):
    """裁判 messages（抽出来供探针/测试复用，避免探针与生产 prompt 漂移）。"""
    user = f"""【上下文】
{context}

【回答】
{answer}

任务：
1. 把【回答】拆成若干条彼此独立的陈述（statements），一条只讲一个事实，
   数字、单位、公式要单独成句（例如"beta2 取 0.98"不能和"优化器是 Adam"合并）。
   不要拆分任务指令、格式说明等任何非回答内容。
2. 对每条陈述判断是否能被【上下文】直接支持（supported 为 true/false）。
3. supported=true 时，必须在 evidence 里**逐字复制**【上下文】中支撑它的连续片段
   （不超过 60 字，不许改写、不许拼接、不许跨段合并）。上下文中找不到这样的原文，
   就判 supported=false。
4. 陈述中的数值、单位、符号必须与 evidence 原文完全一致：原文写 0.98 而陈述写 0.999，
   一律 supported=false；陈述里自行推算出的、上下文没写过的数值也算不被支持。
5. 计算 score = 被支撑的陈述数 / 总陈述数（0~1，保留两位小数）。

只输出 JSON，格式：
{{"statements": [{{"statement": "陈述内容", "supported": true, "evidence": "上下文原文片段"}}], "score": 0.0}}"""
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def verify_grounded(context: str, answer: str) -> dict:
    """
    判定回答是否完全基于上下文。

    返回 {"grounded": bool, "score": float(0~1), "detail": [...]}：
      - score = 被支撑陈述数 / 总陈述数
      - grounded = score >= config.GROUNDED_SCORE_THRESHOLD
    任一异常（裁判不可用/解析失败）一律返回 grounded=False，宁可拒答不可错答。
    """
    if not context or not answer:
        return {"grounded": False, "score": 0.0, "detail": []}

    context = _trim_context(context, answer, config.GROUNDED_CONTEXT_MAX_CHARS)

    messages = judge_messages(context, answer)
    parsed = {}
    for _ in range(3):
        try:
            parsed = _extract_json(llm_provider.judge_chat(messages, temperature=0.0)) or {}
            if parsed:
                break
        except Exception:
            parsed = {}

    detail = parsed.get("statements", []) if isinstance(parsed.get("statements"), list) else []
    # 裁判自称"有原文支撑"不算数：evidence 必须能在上下文里逐字找到。
    # 2026-10-30 实测：9B 裁判会把"β2=0.999"判 supported（上下文原文 0.98），
    # 却只引"我们使用 Adam 优化器"——证据回验把这类放水判成不支撑。
    if config.GROUNDED_REQUIRE_EVIDENCE and detail:
        ctx_norm = _norm(context)
        for it in detail:
            if not isinstance(it, dict) or it.get("supported") is not True:
                continue
            ev = _norm(it.get("evidence", ""))
            if len(ev) < 4 or ev not in ctx_norm:
                it["supported"] = False
                it["evidence_ok"] = False
    # 评审 P2（2026-08-30）：与 eval_local 同口径——分数从明细重算，不信裁判自报。
    # 门控场景取「明细重算」与「裁判自报」的较小值（更保守）：
    #   - 裁判漏列陈述时（P2 复核实测：自报 0.6 但明细 2/2 全 true），漏列的往往是
    #     它没把握的陈述 → 自报分更接近真实，min 避免明细重算虚高放行幻觉；
    #   - 明细与自报一致时二者相等，行为不变。
    # 明细为空（裁判故障）：回退自报分，缺省 0.0（安全侧 → 拒答）。
    recalculated = None
    if detail:
        sup = sum(1 for it in detail if isinstance(it, dict) and it.get("supported") is True)
        recalculated = sup / len(detail)
    try:
        reported = float(parsed.get("score", 0.0))
    except (TypeError, ValueError):
        reported = 0.0
    score = min(recalculated, reported) if recalculated is not None else reported
    score = max(0.0, min(1.0, score))
    grounded = score >= config.GROUNDED_SCORE_THRESHOLD
    return {"grounded": grounded, "score": round(score, 3), "detail": detail}
