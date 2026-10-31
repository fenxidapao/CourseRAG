"""test_paper_routing.py — 题名路由的缩写抽取契约（v3.2，零外部依赖）。

只测纯函数：从论文文本抽「自称缩写」的三种写法与通用词过滤。
真实路由（读 normalized/ 目录）依赖本地语料，不在 CI 覆盖范围内。
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("APP_PROFILE", "paper")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag_core import _acronyms_from_text


def test_parenthesized_definition():
    assert "hyde" in _acronyms_from_text(
        "We propose Hypothetical Document Embeddings (HyDE) for zero-shot retrieval."
    )


def test_or_definition():
    assert "lora" in _acronyms_from_text(
        "We propose Low-Rank Adaptation, or LoRA, which freezes the pretrained weights."
    )


def test_hyphenated_initials():
    assert "fid" in _acronyms_from_text("Figure 2: Architecture of the Fusion-in-Decoder method.")


def test_generic_acronyms_filtered():
    got = _acronyms_from_text("Retrieval-Augmented Generation (RAG) uses an LLM (large language model).")
    assert "rag" not in got and "llm" not in got
