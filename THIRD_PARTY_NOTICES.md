# Third-Party Notices

本项目通过子进程调用 MinerU 解析 PDF；MinerU 及其依赖的许可与本文档所列的
其他第三方组件相同，均归其各自作者所有。

## 解析与渲染

| 组件 | 版本 | 许可 | 用法 |
|---|---|---|---|
| [MinerU](https://github.com/opendatalab/MinerU) | 2.7.6（`mineru[pipeline]`） | AGPL-3.0 | **独立 venv，仅 subprocess 调用**（`mineru_adapter.py`），主应用不 import、不链接其代码 |
| [PyMuPDF](https://pymupdf.readthedocs.io/) | 1.28.2 | AGPL-3.0 | 页图渲染与 bbox 高亮（`page_render.py`）、文本层检测 |

> AGPL-3.0 组件的使用边界：MinerU 以独立进程运行、通过文件交换数据；PyMuPDF 作为
> Python 依赖被 import。若需闭源分发本服务，请先评估 AGPL 义务或替换为许可兼容的组件。

## 模型

| 模型 | 用途 | 许可 |
|---|---|---|
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) | 向量嵌入（Ollama `bge-m3`） | MIT |
| [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 精排 | MIT |
| [llama3.1:8b](https://ollama.com/library/llama3.1) | 本地生成 / 裁判（可选） | Llama 3.1 Community License |
| [qwen2.5:7b](https://ollama.com/library/qwen2.5) | 本地裁判（可选） | Apache-2.0 |
| [qwen3:4b / qwen3:8b](https://ollama.com/library/qwen3) | 本地生成 / 裁判（M5 计划） | Apache-2.0 |

## 样例论文（18 篇，仅本地解析，未提交仓库）

| 论文 | 来源 | 许可 |
|---|---|---|
| Attention Is All You Need | https://arxiv.org/abs/1706.03762 | arXiv non-exclusive license |
| Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | https://arxiv.org/abs/1908.10084 | CC BY-SA 4.0 |
| ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT | https://arxiv.org/abs/2004.12832 | arXiv non-exclusive license |
| Dense Passage Retrieval for Open-Domain Question Answering | https://arxiv.org/abs/2004.04906 | arXiv non-exclusive license |
| Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering | https://arxiv.org/abs/2007.01282 | arXiv non-exclusive license |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | https://arxiv.org/abs/2005.11401 | arXiv non-exclusive license |
| LoRA: Low-Rank Adaptation of Large Language Models | https://arxiv.org/abs/2106.09685 | arXiv non-exclusive license |
| Precise Zero-Shot Dense Retrieval without Relevance Labels | https://arxiv.org/abs/2212.10496 | arXiv non-exclusive license |
| Lost in the Middle: How Language Models Use Long Contexts | https://arxiv.org/abs/2307.03172 | arXiv non-exclusive license |
| QLoRA: Efficient Finetuning of Quantized LLMs | https://arxiv.org/abs/2305.14314 | CC BY 4.0 |
| Ragas: Automated Evaluation of Retrieval Augmented Generation | https://arxiv.org/abs/2309.15217 | CC BY 4.0 |
| Retrieval-Augmented Generation for Large Language Models: A Survey | https://arxiv.org/abs/2312.10997 | arXiv non-exclusive license |
| Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection | https://arxiv.org/abs/2310.11511 | CC BY 4.0 |
| From Local to Global: A Graph RAG Approach to Query-Focused Summarization | https://arxiv.org/abs/2404.16130 | CC BY 4.0 |
| M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation | https://arxiv.org/abs/2402.03216 | CC BY 4.0 |
| MinerU: An Open-Source Solution for Precise Document Content Extraction | https://arxiv.org/abs/2409.18839 | arXiv non-exclusive license |
| Seven Failure Points When Engineering a Retrieval Augmented Generation System | https://arxiv.org/abs/2401.05856 | CC BY 4.0 |
| 国内外大语言模型生成中文论文摘要对比研究 | https://www.kmf.ac.cn/CN/10.13266/j.issn.2095-5472.2024.032 | CC BY 4.0 |

`data/papers/` 已 gitignore；服务只提供页图，不提供整本 PDF 下载。使用者需自行确保
上传论文的许可与使用范围。
