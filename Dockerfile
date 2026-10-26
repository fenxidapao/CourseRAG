# ===== Stage 1: 依赖安装 =====
# 多阶段构建：Stage 1 只装 Python 依赖，Stage 2 拷贝 site-packages，
# 最终镜像不含 pip 缓存 / 编译工具链，体积更小。
# 双模式：
#   WITH_RERANKER=false（默认）→ 仅 API 最小依赖（slim，不含 torch，reranker 关闭）
#   WITH_RERANKER=true        → 追加 sentence-transformers + CPU torch（完整版，reranker 开启）
# 构建示例：docker build --build-arg WITH_RERANKER=true -t coursrag-api:full .
#          docker compose --profile full up -d --build   （compose 见 docker-compose.yml）
FROM python:3.11-slim AS deps

ARG WITH_RERANKER=false
ENV WITH_RERANKER=$WITH_RERANKER

WORKDIR /app
# 先只拷依赖清单，充分利用 Docker 层缓存（改代码不用重装依赖）
COPY requirements-api.txt ./
COPY requirements-rerank.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt \
    && if [ "$WITH_RERANKER" = "true" ]; then \
         pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
         && pip install --no-cache-dir -r requirements-rerank.txt; \
       fi

# ===== Stage 2: 运行镜像 =====
FROM python:3.11-slim AS runtime

ARG WITH_RERANKER=false
ENV WITH_RERANKER=$WITH_RERANKER

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # 镜像内 reranker 开关跟随构建参数：slim=false / full=true
    RERANKER_ENABLED=$WITH_RERANKER \
    HF_ENDPOINT=https://hf-mirror.com

# 从 deps 阶段拷贝已装好的依赖（避免重复安装）
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# 拷贝应用代码（.dockerignore 已排除 data/ chroma_db*/ .env 等本地数据）
COPY . .

# 运行时数据目录（data/ 与 chroma_db_m3/ 由 docker-compose 卷挂载覆盖）
RUN mkdir -p data chroma_db chroma_db_m3

# 非 root 运行（安全基线）：镜像内进程以 uid 10001 的 app 用户启动，
# 避免容器逃逸时直接拿到 root。Linux 宿主若挂载目录属主不匹配导致不可写，
# 请在宿主执行：sudo chown -R 10001:10001 ./data ./chroma_db_m3
RUN useradd -m -u 10001 app && chown -R app:app /app
USER app

EXPOSE 8000

# 容器内健康检查：探活 /health，Ollama 未就绪时服务照常启动、由 /health 报告状态
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# 注意：多阶段构建只拷了 site-packages，pip 装的可执行脚本（uvicorn 命令）
# 不在 /usr/local/bin 下 → 必须用 python -m uvicorn（2026-08-22 实测踩坑）
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
