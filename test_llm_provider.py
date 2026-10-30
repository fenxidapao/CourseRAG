"""test_llm_provider.py — C3 生成去本地化的确定性测试（不联网、不依赖 Ollama）。

覆盖：
  1. provider 分发：默认 ollama 行为不变；openai 走云端 HTTP。
  2. key 解析优先级与「找不到 key」的失败文案（不泄漏 key）。
  3. 推理模型预算自适应：content 为空且 reasoning 占满预算时翻倍重试。
  4. 云端流式 SSE：只取 delta.content，跳过 reasoning_content。
  5. 云端异常（HTTP 4xx/5xx、网络不可达）转成 LLMError，不抛 urllib 原始异常。
  6. 生成链路末端：rag_core.ollama_chat / ollama_chat_stream 仍可被 monkeypatch
     （test_api.py 四处钉的就是这两个名字）。
"""
import io
import json

import pytest

import config
import llm_provider
import rag_core


# ====================== 1. provider 分发 ======================
def test_default_provider_is_ollama(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    called = {}

    def fake_ollama(messages, temperature, model):
        called["hit"] = True
        called["temperature"] = temperature
        return "本地回答"

    monkeypatch.setattr(llm_provider, "_ollama_chat", fake_ollama)
    out = llm_provider.chat([{"role": "user", "content": "hi"}], temperature=0.1)
    assert out == "本地回答"
    assert called["hit"] is True and called["temperature"] == 0.1


def test_openai_provider_uses_cloud(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    seen = {}

    def fake_cloud(messages, temperature, max_tokens, **kw):
        seen["model"] = kw.get("model")
        return "云端回答"

    monkeypatch.setattr(llm_provider, "_cloud_chat", fake_cloud)
    out = llm_provider.chat([{"role": "user", "content": "hi"}], temperature=0.0)
    assert out == "云端回答"
    assert seen["model"] == config.LLM_MODEL


def test_unknown_provider_falls_back_to_ollama(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "some-typo")
    monkeypatch.setattr(llm_provider, "_ollama_chat", lambda m, t, model: "fallback")
    assert llm_provider.chat([{"role": "user", "content": "x"}]) == "fallback"


def test_rag_core_delegates_to_provider(monkeypatch):
    """rag_core.ollama_chat 必须委托给 provider，而不是自己直连 Ollama。"""
    monkeypatch.setattr(llm_provider, "chat", lambda messages, temperature=None: "via-provider")
    assert rag_core.ollama_chat([{"role": "user", "content": "x"}]) == "via-provider"

    monkeypatch.setattr(
        llm_provider, "chat_stream",
        lambda messages, temperature=None: iter(["a", "b"]),
    )
    assert list(rag_core.ollama_chat_stream([{"role": "user", "content": "x"}])) == ["a", "b"]


def test_rag_core_functions_remain_monkeypatchable(monkeypatch):
    """test_api.py 四处钉 rag_core.ollama_chat 名字，改名会连带破坏测试。"""
    monkeypatch.setattr(rag_core, "ollama_chat", lambda messages, temperature=None: "patched")
    assert rag_core.ollama_chat([{"role": "user", "content": "x"}]) == "patched"


# ====================== 2. key 解析 ======================
def test_resolve_key_prefers_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "env-key")
    assert llm_provider.resolve_api_key() == "env-key"


def test_resolve_key_falls_back_to_credentials_file(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("KUAIPAO_API_KEY", raising=False)
    monkeypatch.delenv("VISUAL_RERANK_API_KEY", raising=False)
    cred = tmp_path / ".credentials.yaml"
    cred.write_text("other: x\nKUAIPAO_API_KEY: file-key\n", encoding="utf-8")
    monkeypatch.setattr(llm_provider, "_CRED_FILE", cred)
    assert llm_provider.resolve_api_key() == "file-key"


def test_missing_key_raises_without_leaking(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("KUAIPAO_API_KEY", raising=False)
    monkeypatch.delenv("VISUAL_RERANK_API_KEY", raising=False)
    monkeypatch.setattr(llm_provider, "_CRED_FILE", tmp_path / "absent.yaml")
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider.chat([{"role": "user", "content": "x"}])
    msg = str(exc.value)
    assert "API key" in msg
    # 失败文案里不能出现任何看起来像 key 的字符串
    assert "sk-" not in msg


# ====================== 3. 推理预算自适应 ======================
def _cloud_body(content="", reasoning="", finish="length"):
    return {
        "choices": [{
            "message": {"content": content, "reasoning_content": reasoning},
            "finish_reason": finish,
        }]
    }


def test_cloud_chat_auto_expands_on_reasoning_only(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 256)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS_CAP", 2048)
    budgets = []

    def fake_post(messages, temperature, max_tokens, **_kwargs):
        budgets.append(max_tokens)
        if max_tokens < 1024:
            return _cloud_body(content="", reasoning="想" * max_tokens, finish="length")
        return _cloud_body(content="最终答案", reasoning="想", finish="stop")

    monkeypatch.setattr(llm_provider, "_cloud_post", fake_post)
    out = llm_provider.chat([{"role": "user", "content": "x"}])
    assert out == "最终答案"
    assert budgets == [256, 512, 1024]   # 每次翻倍


def test_cloud_chat_raises_when_cap_exhausted(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_MAX_TOKENS", 256)
    monkeypatch.setattr(config, "LLM_MAX_TOKENS_CAP", 512)

    def fake_post(messages, temperature, max_tokens, **_kwargs):
        return _cloud_body(content="", reasoning="想" * max_tokens, finish="length")

    monkeypatch.setattr(llm_provider, "_cloud_post", fake_post)
    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider.chat([{"role": "user", "content": "x"}])
    assert "LLM_MAX_TOKENS" in str(exc.value)


def test_cloud_chat_disables_thinking_by_default(monkeypatch):
    """默认必须发送关闭思维链的参数：实测 reasoning 会烧光预算导致空 content。"""
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_DISABLE_THINKING", True)
    seen = []

    def fake_post(messages, temperature, max_tokens, **kw):
        seen.append((kw.get("thinking_off"), kw.get("thinking_style")))
        return _cloud_body(content="答案", reasoning="", finish="stop")

    monkeypatch.setattr(llm_provider, "_cloud_post", fake_post)
    assert llm_provider.chat([{"role": "user", "content": "x"}]) == "答案"
    assert seen and seen[0][0] is True      # thinking_off=True


def test_cloud_chat_can_keep_thinking_when_disabled(monkeypatch):
    """LLM_DISABLE_THINKING=false 时不注入该参数（保留原行为，便于对比）。"""
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(config, "LLM_DISABLE_THINKING", False)
    seen = []

    def fake_post(messages, temperature, max_tokens, **kw):
        seen.append(kw.get("thinking_off"))
        return _cloud_body(content="答案", reasoning="", finish="stop")

    monkeypatch.setattr(llm_provider, "_cloud_post", fake_post)
    llm_provider.chat([{"role": "user", "content": "x"}])
    assert seen == [False]


def test_thinking_off_payload_variants_exist():
    """至少提供一个关闭思考的字段，且每个都是 dict（供 payload.update）。"""
    assert len(llm_provider._THINKING_OFF_ATTEMPTS) >= 1
    for item in llm_provider._THINKING_OFF_ATTEMPTS:
        assert isinstance(item, dict) and item

def test_cloud_chat_returns_content_when_present(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(
        llm_provider, "_cloud_post",
        lambda m, t, mt, **kw: _cloud_body(content="答案", reasoning="想", finish="stop"),
    )
    assert llm_provider.chat([{"role": "user", "content": "x"}]) == "答案"


# ====================== 4. 云端流式 SSE ======================
class _FakeStreamResp:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        for line in self._lines:
            yield (line + "\n").encode("utf-8")


def _sse_line(content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return "data: " + json.dumps(
        {"choices": [{"delta": delta, "finish_reason": finish}]}, ensure_ascii=False
    )


def test_cloud_stream_yields_only_content(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    lines = [
        _sse_line(reasoning="思考中"),
        _sse_line(content="你"),
        _sse_line(content="好"),
        _sse_line(reasoning="继续思考"),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        llm_provider.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeStreamResp(lines),
    )
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "test-key")
    out = list(llm_provider.chat_stream([{"role": "user", "content": "x"}]))
    assert out == ["你", "好"]          # reasoning_content 不泄漏给用户


def test_cloud_stream_skips_malformed_frames(monkeypatch):
    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    lines = [
        "data: {not json}",
        _sse_line(content="ok"),
        "data: [DONE]",
    ]
    monkeypatch.setattr(
        llm_provider.urllib.request, "urlopen",
        lambda req, timeout=None: _FakeStreamResp(lines),
    )
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "test-key")
    assert list(llm_provider.chat_stream([{"role": "user", "content": "x"}])) == ["ok"]


# ====================== 5. 云端异常 ======================
def test_cloud_http_error_becomes_llm_error(monkeypatch):
    import urllib.error

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "test-key")

    def boom(req, timeout=None):
        raise urllib.error.HTTPError(
            req.full_url, 403, "Forbidden", {}, io.BytesIO(b'{"error":"no access"}')
        )

    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", boom)
    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider.chat([{"role": "user", "content": "x"}])
    assert "403" in str(exc.value)


def test_cloud_url_error_becomes_llm_error(monkeypatch):
    import urllib.error

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "test-key")

    def boom(req, timeout=None):
        raise urllib.error.URLError("dns fail")

    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", boom)
    with pytest.raises(llm_provider.LLMError):
        llm_provider.chat([{"role": "user", "content": "x"}])


def test_no_key_in_error_when_http_body_echoes_key(monkeypatch):
    """即使上游把 key 回显在错误体里，也不能原样抛出（防日志泄漏）。"""
    import urllib.error

    monkeypatch.setattr(config, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "sk-SECRET-DO-NOT-LEAK")

    def boom(req, timeout=None):
        body = b'{"error":"bad key sk-SECRET-DO-NOT-LEAK"}'
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(body))

    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", boom)
    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider.chat([{"role": "user", "content": "x"}])
    assert "sk-SECRET-DO-NOT-LEAK" not in str(exc.value)



# ====================== 7. 裁判（C2）：独立配置 + 统一入口 ======================
def test_judge_provider_parsing(monkeypatch):
    """兼容旧写法 deepseek；未知值回落 local。"""
    for raw, expected in [("openai", "openai"), ("deepseek", "openai"),
                          ("OPENAI", "openai"), ("local", "ollama"),
                          ("", "ollama"), ("typo", "ollama")]:
        monkeypatch.setattr(config, "JUDGE_PROVIDER", raw)
        assert llm_provider.judge_provider_name() == expected


def test_judge_chat_uses_independent_config(monkeypatch):
    """裁判必须用 JUDGE_MODEL / JUDGE_API_BASE，而不是生成模型的配置。"""
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "openai")
    monkeypatch.setattr(config, "JUDGE_MODEL", "judge-model-x")
    monkeypatch.setattr(config, "JUDGE_API_BASE", "https://judge.example/v1")
    monkeypatch.setattr(config, "LLM_MODEL", "gen-model-y")
    monkeypatch.setattr(config, "LLM_API_BASE", "https://gen.example/v1")
    monkeypatch.setattr(llm_provider, "judge_api_key", lambda: "judge-key")
    seen = {}

    def fake_cloud(messages, temperature, max_tokens, **kw):
        seen.update(kw)
        seen["temperature"] = temperature
        return "{}"

    monkeypatch.setattr(llm_provider, "_cloud_chat", fake_cloud)
    llm_provider.judge_chat([{"role": "user", "content": "x"}])
    assert seen["model"] == "judge-model-x"
    assert seen["base_url"] == "https://judge.example/v1"
    assert seen["api_key"] == "judge-key"
    assert seen["temperature"] == 0.0        # 裁判温度必须 0


def test_judge_chat_local_uses_jadge_model(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "local")
    monkeypatch.setattr(config, "JUDGE_MODEL", "qwen2.5:7b")
    seen = {}

    def fake_ollama(messages, temperature, model, max_tokens=None, repeat_penalty=None):
        seen["model"] = model
        seen["temperature"] = temperature
        seen["max_tokens"] = max_tokens
        seen["repeat_penalty"] = repeat_penalty
        return "本地裁判"

    monkeypatch.setattr(llm_provider, "_ollama_chat", fake_ollama)
    assert llm_provider.judge_chat([{"role": "user", "content": "x"}]) == "本地裁判"
    assert seen["model"] == "qwen2.5:7b"
    # 本地裁判必须有输出上限，防止重复循环写满上下文（2026-10-30 实测 40960 token / 545s）
    assert seen["max_tokens"] == config.JUDGE_LOCAL_MAX_TOKENS
    # 重复惩罚同理：glm4:9b 长 JSON 会循环重复 → 截断 → 解析失败被当"不接地"拒答
    assert seen["repeat_penalty"] == config.JUDGE_LOCAL_REPEAT_PENALTY
    assert seen["temperature"] == 0.0


def test_judge_api_key_prefers_explicit(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_API_KEY", "explicit-judge-key")
    assert llm_provider.judge_api_key() == "explicit-judge-key"


def test_judge_api_key_falls_back_to_relay_key(monkeypatch):
    monkeypatch.setattr(config, "JUDGE_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "relay-key")
    assert llm_provider.judge_api_key() == "relay-key"


def test_grounding_uses_provider_not_direct_ollama(monkeypatch):
    """真 bug 回归：grounding.py 曾直连 Ollama、忽略 JUDGE_PROVIDER。"""
    import grounding
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "openai")
    calls = []

    def fake_judge(messages, temperature=0.0):
        calls.append(temperature)
        return ('{"statements": [{"statement": "A", "supported": true, '
                '"evidence": "上下文 A"}], "score": 1.0}')

    monkeypatch.setattr(llm_provider, "judge_chat", fake_judge)
    out = grounding.verify_grounded("上下文 A", "回答 A")
    assert calls, "grounding 必须调用 llm_provider.judge_chat"
    assert out["grounded"] is True and out["score"] == 1.0


def test_grounding_trims_context_to_budget(monkeypatch):
    """真 bug 回归：8 块上下文 ≈1 万 token 撑爆裁判 num_ctx，400 被当成"不接地"。

    裁剪必须优先保住与回答共享词汇的段，并保持原始顺序。
    """
    import grounding
    monkeypatch.setattr(config, "GROUNDED_CONTEXT_MAX_CHARS", 400)
    monkeypatch.setattr(config, "JUDGE_PROVIDER", "openai")
    seen = {}

    def fake_judge(messages, temperature=0.0):
        seen["prompt"] = messages[-1]["content"]
        return '{"statements": [{"statement": "A", "supported": true, "evidence": "目标事实"}], "score": 1.0}'

    monkeypatch.setattr(llm_provider, "judge_chat", fake_judge)
    irrelevant = "无关段落" + "ABCD" * 100
    relevant = "目标事实：beta2 取 0.98"
    context = f"{irrelevant}\n\n{relevant}"
    grounding.verify_grounded(context, "beta2 取 0.98")

    assert relevant in seen["prompt"], "相关段必须保留"
    assert "ABCD" * 50 not in seen["prompt"], "无关长段应被裁掉"
    assert len(grounding._trim_context(context, "beta2 取 0.98", 400)) < len(context)


def test_eval_local_judge_call_delegates(monkeypatch):
    import eval_local
    monkeypatch.setattr(llm_provider, "judge_chat", lambda messages, temperature=0.0: "{}")
    assert eval_local._judge_call([{"role": "user", "content": "x"}]) == "{}"


def test_no_hardcoded_deepseek_chat_remains():
    """回归守卫：eval_local.py 不应再出现硬编码 deepseek-chat 的调用。"""
    import pathlib
    src = pathlib.Path(__file__).with_name("eval_local.py").read_text(encoding="utf-8")
    assert '"model": "deepseek-chat"' not in src
    assert "def _judge_deepseek" not in src


def test_config_defines_judge_vars_once():
    """回归守卫：config.py 曾重复定义 JUDGE_*，旧块覆盖新默认值导致 401。"""
    import pathlib
    import re
    src = pathlib.Path(__file__).with_name("config.py").read_text(encoding="utf-8")
    for name in ("JUDGE_PROVIDER", "JUDGE_MODEL", "JUDGE_API_BASE"):
        assigns = re.findall(rf"(?m)^{name}\s*=", src)
        assert len(assigns) == 1, f"{name} 在 config.py 中定义了 {len(assigns)} 次"


def test_judge_api_base_default_is_relay(monkeypatch):
    """JUDGE_API_BASE 默认必须是本项目中转，而不是 api.deepseek.com。"""
    import importlib
    monkeypatch.delenv("JUDGE_API_BASE", raising=False)
    import config as cfg
    importlib.reload(cfg)
    try:
        assert "kuaipao.pro" in cfg.JUDGE_API_BASE
    finally:
        importlib.reload(cfg)


# ====================== 8. 云端瞬时错误重试（C3.1） ======================
def test_transient_error_retries_then_succeeds(monkeypatch):
    """中转实测约 21% 请求瞬时失败；重试后成功，不应把整次调用判死。"""
    monkeypatch.setattr(config, "LLM_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(config, "LLM_RETRY_BACKOFF", 0.01)
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise llm_provider.TransientLLMError("getaddrinfo failed")
        return "ok"

    assert llm_provider._call_with_retry(flaky, label="t") == "ok"
    assert len(calls) == 3


def test_transient_error_exhausts_and_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(config, "LLM_RETRY_BACKOFF", 0.01)
    calls = []

    def always_fail():
        calls.append(1)
        raise llm_provider.TransientLLMError("SSL handshake failure")

    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider._call_with_retry(always_fail, label="t")
    assert len(calls) == 3
    assert "3 次" in str(exc.value)


def test_non_transient_error_not_retried(monkeypatch):
    """401/403 等非瞬时错误必须立即上抛，不浪费重试。"""
    monkeypatch.setattr(config, "LLM_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(config, "LLM_RETRY_BACKOFF", 0.01)
    calls = []

    def unauthorized():
        calls.append(1)
        raise llm_provider.LLMError("云端返回 HTTP 401")

    with pytest.raises(llm_provider.LLMError):
        llm_provider._call_with_retry(unauthorized, label="t")
    assert len(calls) == 1


def test_dns_error_is_transient(monkeypatch):
    """getaddrinfo failed 必须被归为可重试，否则评测会被 21% 的网络抖动污染。"""
    import urllib.error
    monkeypatch.setattr(config, "LLM_RETRY_ATTEMPTS", 2)
    monkeypatch.setattr(config, "LLM_RETRY_BACKOFF", 0.01)
    monkeypatch.setattr(llm_provider, "resolve_api_key", lambda: "k")

    def boom(req, timeout=None):
        raise urllib.error.URLError("[Errno 11001] getaddrinfo failed")

    monkeypatch.setattr(llm_provider.urllib.request, "urlopen", boom)
    with pytest.raises(llm_provider.LLMError) as exc:
        llm_provider._cloud_post([{"role": "user", "content": "x"}], 0.0, 64,
                                 model="m", base_url="https://x/v1", api_key="k")
    assert isinstance(exc.value, llm_provider.TransientLLMError)


# ====================== 6. provider 名称解析 ======================
@pytest.mark.parametrize("raw,expected", [
    ("openai", "openai"), ("OPENAI", "openai"), (" openai ", "openai"),
    ("ollama", "ollama"), ("", "ollama"), ("typo", "ollama"),
])
def test_provider_name_parsing(monkeypatch, raw, expected):
    monkeypatch.setattr(config, "LLM_PROVIDER", raw)
    assert llm_provider.provider_name() == expected