#!/usr/bin/env python
"""
CourseRAG 环境自检（跨机器迁移必备）。

换电脑 / 重新部署前先跑：
    python check_env.py
缺什么会直接告诉你修复命令，避免跑到一半才爆“一直转 / ModuleNotFoundError”。

覆盖：Python 版本、关键依赖、Ollama 是否在跑 + 模型是否齐全、reranker 权重是否已缓存。
"""
import sys
import importlib
import os
import json
import urllib.request

print(f"Python: {sys.version.split()[0]}  平台: {sys.platform}")

# ---------- 1. 依赖 ----------
# 模块名 -> pip 包名（已去 langchain，直接依赖 ollama SDK + chromadb）
REQUIRED = {
    "ollama": "ollama",
    "chromadb": "chromadb",
    "streamlit": "streamlit",
    "dotenv": "python-dotenv",
    "pypdf": "pypdf",
    "pymupdf": "pymupdf",
    "docx": "python-docx",
    "sentence_transformers": "sentence-transformers",
}
missing = []
for mod, pkg in REQUIRED.items():
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(pkg)

if missing:
    print("✗ 缺依赖：", ", ".join(missing))
    print("  修复：pip install -r requirements.txt")
else:
    print("✓ 依赖齐全（ollama / chromadb / streamlit / pypdf / python-docx / sentence-transformers）")

# ---------- 2. Ollama 与模型 ----------
OLLAMA_URL = "http://localhost:11434/api/tags"
try:
    with urllib.request.urlopen(OLLAMA_URL, timeout=4) as resp:
        tags = json.load(resp).get("models", [])
    names = {m.get("name") for m in tags}
    # Ollama 返回的模型名常带 :latest 等标签（如 nomic-embed-text:latest），
    # 而 llama3.1:8b 的冒号本身是模型标签的一部分——所以用「精确 或 带冒号前缀」匹配，
    # 避免把已安装的模型误报成缺失。
    def _has(name):
        return name in names or any(n.startswith(name + ":") for n in names)
    need = {"nomic-embed-text", "llama3.1:8b"}
    lack = {x for x in need if not _has(x)}
    if lack:
        print("✗ Ollama 已启动，但缺模型：", ", ".join(sorted(lack)))
        print("  修复：ollama pull " + " ".join(sorted(lack)))
    else:
        print("✓ Ollama 运行中，嵌入/生成模型齐全")
except Exception:
    print("✗ Ollama 未启动或不可达（" + OLLAMA_URL + "）")
    print("  修复：ollama serve  或打开 Ollama 桌面应用")

# ---------- 3. reranker 权重缓存（可选）----------
cache = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3")
if os.path.isdir(cache):
    print("✓ reranker 权重已缓存（可离线加载，不依赖外网）")
else:
    print("! reranker 权重未缓存：首次启用 reranker 会联网下载 ~2.2GB")
    print("  建议先设镜像：set HF_ENDPOINT=https://hf-mirror.com  再跑（app.py 已默认走镜像）")

print("\n全部 ✓ 即可运行：streamlit run app.py")
