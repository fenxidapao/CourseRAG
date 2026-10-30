"""llm_provider.py — LLM 统一提供方（本地 Ollama / OpenAI 兼容云端）。

背景（C3+C2 去本地化，2026-10-15/16）
--------------------------------------
CourseRAG 原本「生成」走本地 Ollama（llama3.1:8b，实测约 4.9 GB 显存）、
「裁判」也走本地 Ollama（qwen2.5:7b，实测约 4.7 GB）。两者同机部署超过本机
8 GB 物理显存，Windows WDDM 不会报 OOM，而是把 CUDA 调用挂住（日志长时间无
写入、CPU 仍上涨），极易被误判成死锁。故生成与裁判都切到云端 OpenAI 兼容接口，
嵌入仍留在本地（bge-m3，约 1.2 GB，与 reranker 2.2 GB 可共存）。

生成与裁判是两套独立配置（不同 provider / model / base / key），
因为它们必须不同源——同一模型既生成又打分会产生自评偏差。

设计约束
--------
1. **模型名一律来自 config**，代码里不硬编码模型名。
2. **API key 只从环境变量或 ``~/.dsh/.credentials.yaml`` 读取**，绝不写入仓库、
   绝不回显到日志、异常信息或响应体（上游错误体回显 key 时会先脱敏）。
3. **默认 provider=ollama**：不设置 provider 时行为与旧版一致，可随时回滚。
4. **预算自适应**：推理模型（deepseek-v4.1-flash）实测会把 max_tokens 全烧在
   reasoning 上、content 为空且 finish=length；空 content + reasoning 占满预算时
   自动翻倍重试，直到 cap。
5. **默认关闭思维链**：实测 deepseek-v4.1-flash 在真实 RAG 提示下 3/3 空回答，
   关闭思维链后 4/4 成功、约 2s；不同模型认不同字段，故逐个变体尝试。

对外接口
--------
``resolve_api_key()`` / ``chat(...)`` / ``chat_stream(...)`` /
``judge_chat(messages, temperature=0.0)`` / ``provider_name()`` / ``judge_provider_name()``
"""
import json
import os
import pathlib
import re
import urllib.error
import urllib.request
from typing import Iterator

import config


class LLMError(RuntimeError):
    """一次性 LLM 调用失败（网络/鉴权/预算耗尽/无有效输出）。"""


class ThinkingUnsupported(LLMError):
    """该模型不认关闭思维链的字段（上游 400/422）。调用方应换一个变体重试。"""


class TransientLLMError(LLMError):
    """可重试的瞬时错误：DNS/SSL/连接超时/429/5xx。

    实测（2026-10-16）：本中转出现约 21% 的请求以 getaddrinfo failed /
    SSLV3_ALERT_HANDSHAKE_FAILURE 失败，重试即可成功。故这类错误单独成类，
    由 _call_with_retry 做指数退避重试，而不是直接判死。
    """


# --------------------------------------------------------------------------
# 凭据解析（绝不回显 key 本身）
# --------------------------------------------------------------------------
_CRED_FILE = pathlib.Path.home() / ".dsh" / ".credentials.yaml"
_CRED_PATTERN = re.compile(r"KUAIPAO_API_KEY\s*:\s*(\S+)")


def resolve_api_key() -> str:
    """按优先级取中转 API key：显式配置 > 环境变量 > 凭据文件。

    返回值只用于组装 Authorization 头，调用方不得写日志、不得入库。
    """
    for name in ("LLM_API_KEY", "KUAIPAO_API_KEY", "VISUAL_RERANK_API_KEY"):
        value = os.getenv(name)
        if value:
            return value.strip().strip('"').strip("'")
    return _read_credential_file()


def _read_credential_file() -> str:
    try:
        text = _CRED_FILE.read_text(encoding="utf-8")
    except OSError:
        return ""
    match = _CRED_PATTERN.search(text)
    return match.group(1) if match else ""


def key_available() -> bool:
    """供 health 检查使用：只报告有无，不泄漏内容。"""
    return bool(resolve_api_key())


def _redact(text: str) -> str:
    """把错误文案里可能出现的 API key 替换掉。

    为什么必须做：上游（中转/网关）在 401/403 时经常把请求头原样回显到
    错误体里。若我们原样抛出，key 就会进入日志、API 响应和测试输出。
    """
    if not text:
        return text
    for name in ("LLM_API_KEY", "JUDGE_API_KEY", "KUAIPAO_API_KEY", "VISUAL_RERANK_API_KEY"):
        value = os.getenv(name)
        if value:
            text = text.replace(value, "***")
    for value in (resolve_api_key(), _read_credential_file()):
        if value:
            text = text.replace(value, "***")
    return text


# --------------------------------------------------------------------------
# provider 选择
# --------------------------------------------------------------------------
_OLLAMA = "ollama"
_OPENAI = "openai"


def _normalize(value: str) -> str:
    value = (value or "").strip().lower()
    if value in ("openai", "cloud", "deepseek"):
        return _OPENAI
    return _OLLAMA


def provider_name() -> str:
    """生成 provider。"""
    return _normalize(config.LLM_PROVIDER)


def judge_provider_name() -> str:
    """裁判 provider；兼容旧的 JUDGE_PROVIDER=deepseek 写法。"""
    return _normalize(getattr(config, "JUDGE_PROVIDER", "local"))


# --------------------------------------------------------------------------
# 本地 Ollama
# --------------------------------------------------------------------------
_ollama_client = None


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        import ollama

        # 必须有超时：本地模型进入重复循环时不会自己停，httpx 默认无超时会永久挂住
        _ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL, timeout=config.LLM_TIMEOUT)
    return _ollama_client


def _ollama_options(temperature, max_tokens=None, repeat_penalty=None):
    opts = {"num_ctx": getattr(config, "OLLAMA_NUM_CTX", 8192)}
    if temperature is not None:
        opts["temperature"] = temperature
    if max_tokens:
        opts["num_predict"] = int(max_tokens)
    if repeat_penalty:
        opts["repeat_penalty"] = float(repeat_penalty)
    return opts


def _ollama_chat(messages, temperature, model, max_tokens=None, repeat_penalty=None):
    resp = _get_ollama_client().chat(
        model=model,
        messages=messages,
        options=_ollama_options(temperature, max_tokens, repeat_penalty),
    )
    return resp["message"]["content"]


def _ollama_chat_stream(messages, temperature, model):
    opts = _ollama_options(temperature)
    stream = _get_ollama_client().chat(
        model=model, messages=messages, options=opts, stream=True
    )
    for chunk in stream:
        piece = chunk["message"]["content"]
        if piece:
            yield piece


# --------------------------------------------------------------------------
# OpenAI 兼容云端
# --------------------------------------------------------------------------
# 关闭思维链的候选参数：不同中转/模型群认不同字段，逐个试到成功。
# 实测（2026-10-15，真实 RAG 提示）deepseek-v4.1-flash 支持
# reasoning_effort=none / thinking.type=disabled / enable_thinking=false。
_THINKING_OFF_ATTEMPTS = (
    {"reasoning_effort": "none"},
    {"thinking": {"type": "disabled"}},
    {"enable_thinking": False},
)


def _retry_attempts() -> int:
    return max(1, int(getattr(config, "LLM_RETRY_ATTEMPTS", 3)))


def _retry_backoff() -> float:
    return float(getattr(config, "LLM_RETRY_BACKOFF", 1.5))


def _call_with_retry(fn, label: str):
    """对瞬时错误（DNS/SSL/超时/429/5xx）做指数退避重试。

    实测（2026-10-16）本中转约 21% 请求瞬时失败（getaddrinfo failed /
    SSLV3_ALERT_HANDSHAKE_FAILURE），重试即可成功。
    非瞬时错误（401/403/400 等）不重试，立即上抛。
    """
    import time as _time
    attempts = _retry_attempts()
    last = None
    for i in range(attempts):
        try:
            return fn()
        except TransientLLMError as exc:
            last = exc
            if i + 1 < attempts:
                delay = _retry_backoff() * (2 ** i)
                print("[llm_provider] %s 瞬时失败，%.1fs 后重试（%d/%d）：%s"
                      % (label, delay, i + 2, attempts, str(exc)[:120]))
                _time.sleep(delay)
    raise LLMError("云端调用 %d 次仍失败（%s）：%s" % (attempts, label, str(last)[:200]))


def _cloud_headers(api_key: str) -> dict:
    if not api_key:
        raise LLMError(
            "云端调用缺少 API key：请设置环境变量，或写入 ~/.dsh/.credentials.yaml"
            "（键名 KUAIPAO_API_KEY）。"
        )
    return {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    }


def _parse_cloud_response(body: dict) -> tuple[str, str, str]:
    choices = body.get("choices") or [{}]
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    reasoning = message.get("reasoning_content") or ""
    return (content or "").strip(), (reasoning or "").strip(), choice.get("finish_reason") or ""


def _cloud_post(messages, temperature, max_tokens, *, model, base_url, api_key,
                thinking_off=False, thinking_style=0) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2 if temperature is None else temperature,
        "max_tokens": max_tokens,
    }
    if thinking_off and 0 <= thinking_style < len(_THINKING_OFF_ATTEMPTS):
        payload.update(_THINKING_OFF_ATTEMPTS[thinking_style])
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_cloud_headers(api_key),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = _redact(exc.read().decode("utf-8", "replace")[:300])
        except Exception:
            pass
        # 上游不认 reasoning_effort / thinking / enable_thinking 时通常回 400/422；
        # 交给调用方换下一个变体，而不是整轮判死。
        if thinking_off and exc.code in (400, 422):
            raise ThinkingUnsupported(
                "HTTP %s 不接受该关闭思考参数：%s" % (exc.code, detail)
            ) from exc
        if exc.code in (408, 409, 425, 429, 500, 502, 503, 504):
            raise TransientLLMError(
                "云端瞬时故障 HTTP %s（模型 %s）：%s" % (exc.code, model, detail)
            ) from exc
        raise LLMError(
            "云端返回 HTTP %s（模型 %s）：%s" % (exc.code, model, detail)
        ) from exc
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        raise TransientLLMError(
            "云端网络不可达（%s）：%s" % (base_url, reason)
        ) from exc
    except json.JSONDecodeError as exc:
        raise TransientLLMError("云端返回非 JSON 响应：%s" % exc) from exc


def _cloud_chat(messages, temperature, max_tokens=None, *, model, base_url, api_key,
                disable_thinking=True, budget=None, cap=None) -> str:
    """云端非流式对话：关闭思维链 → 换参数 → 预算翻倍，三重兜底。

    为什么这么复杂（2026-10-15 实测，真实 RAG 提示 1868 字符 / 8 块上下文）：
      - deepseek-v4.1-flash 默认开启思维链，3/3 次把 max_tokens 全烧在
        reasoning 上（7000~10000 字符），content 为空且 finish=length；
      - 传 reasoning_effort=none 或 thinking.type=disabled 后 0 reasoning、
        正常产出 content。但不同模型认的字段不同，故逐个试。
    """
    budget = budget or max_tokens or config.LLM_MAX_TOKENS
    cap = max(cap or config.LLM_MAX_TOKENS_CAP, budget)
    last_reason = ""
    styles = range(len(_THINKING_OFF_ATTEMPTS)) if disable_thinking else range(1)
    for style in styles:
        b = budget
        while True:
            try:
                body = _call_with_retry(
                    lambda: _cloud_post(
                        messages, temperature, b,
                        model=model, base_url=base_url, api_key=api_key,
                        thinking_off=disable_thinking, thinking_style=style,
                    ),
                    label="chat/%s" % model,
                )
            except ThinkingUnsupported as exc:
                last_reason = str(exc)
                break
            content, reasoning, finish = _parse_cloud_response(body)
            if content:
                return content
            last_reason = "finish=%s reasoning_chars=%d budget=%d style=%d" % (
                finish, len(reasoning), b, style,
            )
            if reasoning and finish == "length" and b < cap:
                b = min(b * 2, cap)
                continue
            break
    raise LLMError(
        "云端无有效输出（%s）。该模型可能持续把额度耗在推理上；"
        "可切换模型，或调 LLM_MAX_TOKENS / LLM_MAX_TOKENS_CAP。" % last_reason
    )


def _cloud_chat_stream(messages, temperature, max_tokens=None, *, model, base_url,
                       api_key, disable_thinking=True) -> Iterator[str]:
    budget = max_tokens or config.LLM_MAX_TOKENS
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2 if temperature is None else temperature,
        "max_tokens": budget,
        "stream": True,
    }
    if disable_thinking:
        # 与 _cloud_chat 同因：推理模型可能把预算全烧在 reasoning 上，
        # 流式下表现为长时间没有 delta.content。默认关闭思考。
        payload.update(_THINKING_OFF_ATTEMPTS[0])
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=_cloud_headers(api_key),
        method="POST",
    )
    def _open():
        return urllib.request.urlopen(req, timeout=config.LLM_TIMEOUT)

    try:
        resp = _call_with_retry(_open, label="chat_stream/%s" % model)
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choice = (obj.get("choices") or [{}])[0] or {}
                delta = choice.get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    yield piece
        finally:
            try:
                resp.close()
            except Exception:
                pass
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = _redact(exc.read().decode("utf-8", "replace")[:300])
        except Exception:
            pass
        raise LLMError(
            "云端流式返回 HTTP %s（模型 %s）：%s" % (exc.code, model, detail)
        ) from exc
    except urllib.error.URLError as exc:
        raise LLMError(
            "云端流式网络不可达（%s）：%s" % (base_url, exc.reason)
        ) from exc


# --------------------------------------------------------------------------
# 生成对外入口
# --------------------------------------------------------------------------
def chat(messages, temperature=None, max_tokens=None) -> str:
    """生成：非流式对话，返回纯文本。失败抛 LLMError。"""
    if provider_name() == _OPENAI:
        return _cloud_chat(
            messages, temperature, max_tokens,
            model=config.LLM_MODEL,
            base_url=config.LLM_API_BASE,
            api_key=resolve_api_key(),
            disable_thinking=config.LLM_DISABLE_THINKING,
        )
    return _ollama_chat(messages, temperature, config.LLM_MODEL)


def chat_stream(messages, temperature=None, max_tokens=None) -> Iterator[str]:
    """生成：流式对话，逐片段 yield 文本。失败抛 LLMError。"""
    if provider_name() == _OPENAI:
        yield from _cloud_chat_stream(
            messages, temperature, max_tokens,
            model=config.LLM_MODEL,
            base_url=config.LLM_API_BASE,
            api_key=resolve_api_key(),
            disable_thinking=config.LLM_DISABLE_THINKING,
        )
        return
    yield from _ollama_chat_stream(messages, temperature, config.LLM_MODEL)


# --------------------------------------------------------------------------
# 裁判对外入口（C2：与生成独立配置，禁止同模型自评）
# --------------------------------------------------------------------------
def judge_chat(messages, temperature=0.0) -> str:
    """裁判：非流式对话，返回纯文本。失败抛 LLMError。

    与 chat() 的关键区别：
      - 使用独立的 JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_API_BASE / JUDGE_API_KEY；
      - 默认 temperature=0.0，逼稳定 JSON；
      - 预算用 JUDGE_MAX_TOKENS（裁判输出是结构化 JSON，通常远小于生成）。
    """
    if judge_provider_name() == _OPENAI:
        return _cloud_chat(
            messages, temperature, None,
            model=config.JUDGE_MODEL,
            base_url=config.JUDGE_API_BASE,
            api_key=judge_api_key(),
            disable_thinking=getattr(config, "JUDGE_DISABLE_THINKING", True),
            budget=getattr(config, "JUDGE_MAX_TOKENS", 4096),
            cap=getattr(config, "JUDGE_MAX_TOKENS_CAP", 8192),
        )
    return _ollama_chat(
        messages, temperature, config.JUDGE_MODEL,
        max_tokens=getattr(config, "JUDGE_LOCAL_MAX_TOKENS", 1600),
        repeat_penalty=getattr(config, "JUDGE_LOCAL_REPEAT_PENALTY", 1.3),
    )


def judge_api_key() -> str:
    """裁判 key：JUDGE_API_KEY 优先，否则回退到同一个中转 key。"""
    explicit = getattr(config, "JUDGE_API_KEY", "") or os.getenv("JUDGE_API_KEY", "")
    if explicit:
        return explicit.strip()
    return resolve_api_key()