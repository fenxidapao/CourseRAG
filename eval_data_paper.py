"""eval_data_paper.py — 论文库评估题集（页级 15 题 + 拒答 20 域外 / 20 域内）。

页号均按 `data/papers/normalized/*.json` 人工核对（M1 归一化产物）；
expected_pages 可含多页：同一事实在摘要与正文各出现一次时两页都算命中。
"""

PAPERS = {
    "bdfaa68d8984": "Attention Is All You Need",
    "e9a0d3128767": "LoRA: Low-Rank Adaptation of Large Language Models",
    "3e021fd8c6c2": "国内外大语言模型生成中文论文摘要对比研究",
}

# 15 题 = 每篇 5 题；kind 仅用于分组统计
PAGE_CASES = [
    # Attention Is All You Need
    {"q": "Transformer (big) 在 WMT 2014 英德翻译上达到多少 BLEU？", "paper_id": "bdfaa68d8984",
     "expected_pages": [1, 8], "reference": "28.4 BLEU", "kind": "number"},
    {"q": "Transformer 用了多少个注意力头 h，每个头的 d_k 是多少？", "paper_id": "bdfaa68d8984",
     "expected_pages": [5], "reference": "h=8, d_k=d_v=64", "kind": "number"},
    {"q": "论文使用的优化器是什么，beta1 和 beta2 取多少？", "paper_id": "bdfaa68d8984",
     "expected_pages": [7], "reference": "Adam, β1=0.9, β2=0.98, ε=1e-9", "kind": "number"},
    {"q": "为什么需要 positional encoding，作者采用正弦函数还是学习式位置嵌入？", "paper_id": "bdfaa68d8984",
     "expected_pages": [6], "reference": "无循环/卷积需注入位置信息；选正弦", "kind": "method"},
    {"q": "训练使用了多少块 GPU、base 模型训练多久？", "paper_id": "bdfaa68d8984",
     "expected_pages": [7], "reference": "8×P100，100k steps/12 小时", "kind": "number"},
    # LoRA
    {"q": "LoRA 把可训练参数减少了多少倍？", "paper_id": "e9a0d3128767",
     "expected_pages": [1], "reference": "10,000×，GPU 显存 3×", "kind": "number"},
    {"q": "LoRA 在 Transformer 中应用到哪些权重矩阵效果最好？", "paper_id": "e9a0d3128767",
     "expected_pages": [10], "reference": "W_q 与 W_v", "kind": "method"},
    {"q": "实验发现 LoRA 的秩 r 最小可以取到多少仍然有效？", "paper_id": "e9a0d3128767",
     "expected_pages": [10], "reference": "r=1 对 W_q,W_v 足够", "kind": "number"},
    {"q": "在 GPT-3 175B 上 LoRA 把显存需求降低了多少？", "paper_id": "e9a0d3128767",
     "expected_pages": [1, 8], "reference": "3×（1.2TB→350GB）", "kind": "number"},
    {"q": "LoRA 在 RoBERTa 上相比全量微调的基线表现如何？", "paper_id": "e9a0d3128767",
     "expected_pages": [7], "reference": "在 GLUE 上匹配或超过基线", "kind": "method"},
    # 中文论文
    {"q": "研究选取了多少个国家社科基金课题作为生成摘要的题目？", "paper_id": "3e021fd8c6c2",
     "expected_pages": [1, 4], "reference": "121 个", "kind": "number"},
    {"q": "文心一言生成摘要的平均字符数是多少？", "paper_id": "3e021fd8c6c2",
     "expected_pages": [7, 8], "reference": "247.11", "kind": "number"},
    {"q": "GPT 与文心一言生成摘要在长度上的对比结论是什么？", "paper_id": "3e021fd8c6c2",
     "expected_pages": [7, 8], "reference": "GPT 平均长度远大于文心一言，约差 200 字符", "kind": "method"},
    {"q": "论文的文本分析包含哪些维度？", "paper_id": "3e021fd8c6c2",
     "expected_pages": [5], "reference": "高频词占比、词性分布、句子数量、摘要长度", "kind": "method"},
    {"q": "作者在结语中列出了哪些研究不足？", "paper_id": "3e021fd8c6c2",
     "expected_pages": [9], "reference": "仅图情领域、仅两个模型、训练不充分、提问方式影响结果", "kind": "method"},
]

# 域内 20 题（可答）：15 题页级问题 + 5 题综述型
IN_DOMAIN = [c["q"] for c in PAGE_CASES] + [
    "Transformer 的编码器和解码器分别由哪些层组成？",
    "self-attention 与 multi-head attention 的区别是什么？",
    "LoRA 的低秩分解公式是什么？",
    "论文如何理解低秩更新的作用（understanding the low-rank updates）？",
    "两个中文大语言模型生成摘要的高频词有什么差异？",
]

# 域外 20 题：10 题同领域但未收录 + 10 题随机域外
# 2026-10-29 语料扩容到 18 篇后重标：原「RAG 如何把检索结果融入生成」与
# 「Llama 2 在 MMLU 上的准确率」已被新收录的 RAG/Seven Failure Points/QLoRA
# 真正覆盖（实测 top-1 块直接给出答案），留在域外会低估拒答率，故替换为
# 仍未被收录的 DoRA / KTO 两题。这是语料覆盖变化导致的重标，不是按分数调题。
OUT_DOMAIN = [
    # 同领域（NLP/LLM），但三篇论文未覆盖
    "BERT 的 masked language modeling 预训练目标是怎么工作的？",
    "PPO 和 DPO 在 RLHF 中的区别是什么？",
    "FlashAttention-2 如何降低显存访问开销？",
    "Mamba 状态空间模型的架构与 Transformer 有何不同？",
    "Chinchilla 缩放定律给出的参数与数据配比是多少？",
    "ReAct 框架如何结合推理与工具调用？",
    "GPT-4V 是如何处理图像输入的？",
    "量化感知训练（QAT）与训练后量化（PTQ）的区别是什么？",
    "DoRA 相比 LoRA 在权重更新分解上做了什么改进？",
    "KTO 的损失函数与 DPO 有什么不同？",
    # 随机域外
    "明天北京会下雨吗？",
    "红烧肉的家常做法是什么？",
    "劳动合同到期不续签需要支付经济补偿吗？",
    "高血压患者日常饮食要注意什么？",
    "世界杯上一届冠军是哪支球队？",
    "如何计算房贷的等额本息月供？",
    "吉他初学者应该先练哪些和弦？",
    "冬天室内适合养哪些绿植？",
    "汽车刹车片多久需要更换？",
    "护照过期了怎么办理换发？",
]


def all_cases():
    return {"page": PAGE_CASES, "reject": {"in_domain": IN_DOMAIN, "out_domain": OUT_DOMAIN}}

# held-out 10+10：只在阈值定稿后跑一次，不参与选阈值
HELDOUT_IN = [
    "Transformer 的 encoder layer 由哪两个子层组成？",
    "训练时 warmup steps 取多少？",
    "LoRA 的 A 矩阵和 B 矩阵分别如何初始化？",
    "LoRA 在 GLUE 上用什么指标评估？",
    "论文如何测量不同秩学到的子空间相似度？",
    "中文论文选择了哪两个大语言模型作为研究对象？",
    "研究用什么方法判断生成的摘要是否符合中文写作规范？",
    "self-attention 的每层计算复杂度是多少？",
    "LoRA 相比 adapter 层在推理延迟上有什么优势？",
    "中文论文的作者与发表期刊是什么？",
]

HELDOUT_OUT = [
    "Vision Transformer 的 patch embedding 是怎么做的？",
    "AlphaFold 如何预测蛋白质结构？",
    "量子纠缠中的贝尔不等式是什么？",
    "如何用 Python 读取 Excel 文件？",
    "咖啡拉花的基本步骤是什么？",
    "股票市盈率的计算公式是什么？",
    "如何申请一项发明专利？",
    "马拉松训练计划应该怎么安排？",
    "法国大革命爆发于哪一年？",
    "Wi-Fi 6 与 Wi-Fi 5 的主要区别是什么？",
]
