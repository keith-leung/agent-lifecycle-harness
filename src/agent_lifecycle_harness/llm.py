"""LLM client abstraction.

Two implementations:
- `RealLLMClient` using `openai.OpenAI` / `openai.AsyncOpenAI` (cross-vendor ready).
- `MockLLMClient` returning deterministic responses for mock mode.
- `MultiVendorLLMClient` that tries multiple providers in order, skipping 402.

The harness never prints raw API keys.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openai import APIStatusError, AsyncOpenAI, OpenAI

from agent_lifecycle_harness.config import is_mock_mode, provider_config
from agent_lifecycle_harness.debug_log import log_llm_call, log_balance_error

# Cap on how much input the mock echoes back verbatim. Without this, multi-turn
# mock runs grow exponentially (each reply echoes all prior history including
# prior replies), risking SQLite blob bloat in the checkpoint table.
_ECHO_MAX_CHARS = 4000


@dataclass
class LLMResponse:
    """Normalized LLM response."""
    content: str
    model: str
    usage: dict[str, int] | None = None


class LLMClient:
    """Thin wrapper so the rest of the harness does not depend on SDK choice."""

    async def ainvoke(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError

    def invoke_sync(self, messages: Sequence[dict[str, str]], **kwargs: Any) -> LLMResponse:
        """Synchronous entry point. Subclasses should override if they have a native sync client."""
        raise NotImplementedError


def _to_openai_message(msg: Any) -> dict[str, Any]:
    """Convert a LangChain message or dict to OpenAI API format.
    
    Preserves:
      - role/content (with human->user, ai->assistant normalization)
      - reasoning_content (MiMo thinking mode)
      - tool_calls / tool_call_id (multi-turn function calling)
    """
    if hasattr(msg, "to_openai_dict"):
        raw = msg.to_openai_dict()
    elif hasattr(msg, "dict"):
        raw = msg.dict()
    elif isinstance(msg, dict):
        raw = msg
    else:
        return {"role": "user", "content": str(msg)}

    # Normalize role: LangChain uses 'human'/'ai', OpenAI API uses 'user'/'assistant'.
    role = raw.get("role", "user")
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"

    result: dict[str, Any] = {"role": role}
    
    # Preserve content
    content = raw.get("content", "")
    if content:
        result["content"] = content
    
    # Preserve reasoning_content (MiMo thinking mode)
    reasoning = raw.get("reasoning_content")
    if reasoning:
        result["reasoning_content"] = reasoning
    
    # Preserve tool_calls (assistant messages with function calls)
    tool_calls = raw.get("tool_calls")
    if tool_calls:
        result["tool_calls"] = tool_calls
    
    # Preserve tool_call_id (tool result messages)
    tool_call_id = raw.get("tool_call_id")
    if tool_call_id:
        result["tool_call_id"] = tool_call_id
    
    # Preserve name (for tool messages)
    name = raw.get("name")
    if name:
        result["name"] = name
    
    return result


class RealLLMClient(LLMClient):
    """Production client using OpenAI-compatible APIs."""

    def __init__(
        self,
        provider_name: str,
        tier: str = "medium",
        *,
        config: dict[str, Any] | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.provider_name = provider_name
        self.tier = tier
        self.temperature = temperature
        cfg = provider_config(config or {}, provider_name)
        base_url = cfg.get("base_url", "")
        api_key = cfg.get("api_key", "")
        self._async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._sync_client = OpenAI(base_url=base_url, api_key=api_key)
        model_cfg = cfg.get("tiers", {}).get(tier, {})
        self.model = model_cfg.get("model", "")

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        try:
            response = await self._async_client.chat.completions.create(
                model=self.model,
                messages=[_to_openai_message(m) for m in messages],
                temperature=kwargs.get("temperature", self.temperature),
                **{k: v for k, v in kwargs.items() if k != "temperature"},
            )
        except APIStatusError as exc:
            log_llm_call(self.provider_name, self.model, False, error_code=str(exc.status_code), error_message=str(exc))
            if exc.status_code == 402:
                log_balance_error(self.provider_name, self.model, str(exc))
            raise
        # Some providers return plain text instead of a Completion object.
        if isinstance(response, str):
            log_llm_call(self.provider_name, self.model, True)
            return LLMResponse(content=response, model=self.model, usage=None)
        log_llm_call(self.provider_name, self.model, True)
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return LLMResponse(content=content, model=self.model, usage=usage)

    def invoke_sync(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        try:
            response = self._sync_client.chat.completions.create(
                model=self.model,
                messages=[_to_openai_message(m) for m in messages],
                temperature=kwargs.get("temperature", self.temperature),
                **{k: v for k, v in kwargs.items() if k != "temperature"},
            )
        except APIStatusError as exc:
            log_llm_call(self.provider_name, self.model, False, error_code=str(exc.status_code), error_message=str(exc))
            if exc.status_code == 402:
                log_balance_error(self.provider_name, self.model, str(exc))
            raise
        # Some providers return plain text instead of a Completion object.
        if isinstance(response, str):
            log_llm_call(self.provider_name, self.model, True)
            return LLMResponse(content=response, model=self.model, usage=None)
        log_llm_call(self.provider_name, self.model, True)
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        return LLMResponse(content=content, model=self.model, usage=usage)


class MultiVendorLLMClient(LLMClient):
    """Vendor-agnostic client that tries multiple providers in order.
    
    Automatically skips providers that return 402 (insufficient balance).
    Upper-layer code (harness, demos) does not need to know which vendor is used.
    """

    def __init__(self, provider_names: list[str], tier: str, config: dict[str, Any]) -> None:
        self._clients: list[tuple[str, LLMClient]] = []
        self.tier = tier
        self.provider_name = provider_names[0] if provider_names else ""
        self.model = ""
        self._config = config
        
        for name in provider_names:
            try:
                client = build_client(name, tier, config=config, temperature=0.0)
                # Probe to filter out 402 (insufficient balance) during init.
                # Transient errors (timeout, 429, 500, connection) are NOT
                # grounds to skip — they would make every provider unusable
                # during a gateway hiccup, and we'd rather keep the client
                # and let the real call retry than silently drop it.
                try:
                    client.invoke_sync([{"role": "user", "content": "ping"}], max_completion_tokens=1, timeout=30)
                except APIStatusError as exc:
                    if exc.status_code == 402:
                        log_balance_error(name, getattr(client, "model", ""), str(exc))
                        continue
                    # Non-402 APIStatusError (429/500/...) — transient, keep.
                except Exception:
                    # Timeout / connection error — transient, keep the client.
                    pass
                self._clients.append((name, client))
                if not self.model:
                    self.model = client.model
            except Exception:
                # If build fails entirely, skip this provider
                continue
        
        if not self._clients:
            raise RuntimeError(f"No usable providers from: {provider_names}")

    def invoke_sync(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        last_exc: APIStatusError | None = None
        for name, client in self._clients:
            try:
                return client.invoke_sync(messages, **kwargs)
            except APIStatusError as exc:
                if exc.status_code == 402:
                    last_exc = exc
                    log_balance_error(name, getattr(client, "model", ""), str(exc))
                    continue
                raise
        raise last_exc or RuntimeError("All providers failed")

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        last_exc: APIStatusError | None = None
        for name, client in self._clients:
            try:
                return await client.ainvoke(messages, **kwargs)
            except APIStatusError as exc:
                if exc.status_code == 402:
                    last_exc = exc
                    log_balance_error(name, getattr(client, "model", ""), str(exc))
                    continue
                raise
        raise last_exc or RuntimeError("All providers failed")


class MockLLMClient(LLMClient):
    """Deterministic, content-derived client for mock mode.

    The output is a mechanical echo transform of the input: the full
    concatenated input text is embedded verbatim (so any token present in
    the input — sentinel or otherwise — appears in the output, and any token
    absent from the input is absent from the output), combined with a short
    sha256 fingerprint that makes the output unique per distinct input.
    """

    def __init__(self, provider_name: str = "mock", tier: str = "medium", *, prefix: str = "") -> None:
        self.provider_name = provider_name
        self.tier = tier
        self.prefix = prefix
        self.model = f"{provider_name}-{tier}"

    def _derive_content(self, messages: Sequence[Any]) -> str:
        """Echo transform applied uniformly to every input (no sentinel special-casing).

        The echoed input is capped at _ECHO_MAX_CHARS so multi-turn mock runs
        don't produce exponentially-growing replies (each turn's reply echoes
        all prior history, including prior replies). The cap preserves the
        head of the input — where seeded sentinels like POISON live — so
        content-derived proof models still hold; only the tail (which is
        prior echo bloat) is dropped.
        """
        input_text = "".join(_message_content(m) for m in messages)
        if len(input_text) > _ECHO_MAX_CHARS:
            input_text = input_text[:_ECHO_MAX_CHARS] + "...[echo-truncated]"
        short_hash = hashlib.sha256(
            "".join(_message_content(m) for m in messages).encode()
        ).hexdigest()[:8]
        label = f"MOCK-{self.provider_name}-{self.prefix}" if self.prefix else f"MOCK-{self.provider_name}"
        return f"[{label}] echo: {input_text} | sha={short_hash}"

    async def ainvoke(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._derive_content(messages), model=self.model)

    def invoke_sync(self, messages: Sequence[Any], **kwargs: Any) -> LLMResponse:
        return LLMResponse(content=self._derive_content(messages), model=self.model)


def _message_content(msg: Any) -> str:
    if hasattr(msg, "content"):
        return msg.content or ""
    if isinstance(msg, dict):
        return msg.get("content", "")
    return str(msg)


def build_client(
    provider_name: str,
    tier: str = "medium",
    *,
    config: dict[str, Any] | None = None,
    temperature: float = 0.0,
) -> LLMClient:
    if is_mock_mode(config or {}):
        return MockLLMClient(provider_name, tier)
    return RealLLMClient(provider_name, tier, config=config, temperature=temperature)


# ============================================================================
# langmem-compatible chat model adapter
# ============================================================================

import tiktoken  # noqa: E402
from langchain_core.messages import BaseMessage  # noqa: E402
from langchain_openai import ChatOpenAI as _BaseChatOpenAI  # noqa: E402

# cl100k_base is the encoding used by gpt-4-class models; used as the
# fallback tokenizer when no model-specific tokenizer can be loaded.
_CL100K = tiktoken.get_encoding("cl100k_base")

# Cache of per-model tokenizers. Each entry is a callable str -> list[int]
# (encode). We try to load the model's native tokenizer (DeepSeek, etc.)
# via `transformers`; if unavailable or the model is unknown, we fall back
# to cl100k_base and surface that fact through `TOKENIZER_KIND`.
_TOKENIZER_CACHE: dict[str, Any] = {}
_TOKENIZER_KIND: dict[str, str] = {}  # model_name -> "native" | "approximate"


def _load_native_tokenizer(model_name: str) -> Any | None:
    """Try to load a model-specific tokenizer. Returns None on failure.

    For DeepSeek models we use the official tokenizer from `transformers`,
    which is the actual BPE tokenizer DeepSeek uses — not OpenAI's
    cl100k_base. We target DeepSeek-V4-Flash's repo because that is the
    model in use; its tokenizer was verified byte-identical to V3's
    (vocab 128000, same encode output on test strings).
    """
    name_lower = model_name.lower()
    repo: str | None = None
    if "deepseek-v4" in name_lower or "deepseek-v3" in name_lower:
        # Prefer the exact generation's tokenizer repo; fall back to V3
        # if the V4 repo is unreachable (their tokenizers are identical).
        repo = "deepseek-ai/DeepSeek-V4-Flash"
    elif "deepseek" in name_lower:
        repo = "deepseek-ai/DeepSeek-V3"
    if repo is not None:
        try:
            from transformers import AutoTokenizer  # type: ignore[import-untyped]
            return AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        except Exception:
            if repo != "deepseek-ai/DeepSeek-V3":
                try:
                    from transformers import AutoTokenizer  # type: ignore[import-untyped]
                    return AutoTokenizer.from_pretrained(
                        "deepseek-ai/DeepSeek-V3", trust_remote_code=True
                    )
                except Exception:
                    return None
            return None
    return None


def _get_encoder(model_name: str) -> tuple[Any, str]:
    """Return (encode_fn, kind) for a model. kind is "native" or "approximate"."""
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name], _TOKENIZER_KIND[model_name]
    native = _load_native_tokenizer(model_name)
    if native is not None:
        enc = native.encode
        _TOKENIZER_CACHE[model_name] = enc
        _TOKENIZER_KIND[model_name] = "native"
        return enc, "native"
    # Fallback: cl100k_base (OpenAI gpt-4 family). Marked approximate for
    # any non-OpenAI model so callers know the count is cross-model.
    enc = _CL100K.encode
    _TOKENIZER_CACHE[model_name] = enc
    _TOKENIZER_KIND[model_name] = "approximate"
    return enc, "approximate"


def tokenizer_kind_for(model_name: str) -> str:
    """Public accessor: 'native' if a model-specific tokenizer is in use,
    'approximate' if cl100k_base is standing in for an unknown tokenizer."""
    _get_encoder(model_name)  # populate cache
    return _TOKENIZER_KIND.get(model_name, "approximate")


def _content_to_text(message: BaseMessage | dict[str, Any]) -> str:
    """Flatten a message's content (str or multi-part list) into plain text."""
    content = (
        message.get("content") if isinstance(message, dict)
        else getattr(message, "content", "")
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


class ChatModel(_BaseChatOpenAI):
    """``ChatOpenAI`` whose ``get_num_tokens_from_messages`` works for any model.

    The upstream implementation
    (``langchain_openai/chat_models/base.py::get_num_tokens_from_messages``)
    raises ``NotImplementedError`` for any model name not starting with
    ``gpt-3.5-turbo`` / ``gpt-4`` / ``gpt-5``. For OpenAI-compatible
    providers used here (deepseek-v4-flash, MiniMax-M2.7, …) that branch
    is unreachable, so langmem's ``token_counter=model.get_num_tokens_from_messages``
    fails before it can do any work.

    This subclass picks a tokenizer based on the model name: a native
    tokenizer (e.g. DeepSeek-V3's official BPE tokenizer via transformers)
    when one is available, cl100k_base otherwise. The per-message overhead
    constants (3 tokens per message header, 1 for a name field, 3 per role
    tag, 3 priming) follow the gpt-4-class convention. The contract —
    ``Callable[[Sequence[BaseMessage]], int]`` — is unchanged, so langmem
    accepts it as ``token_counter`` unchanged. Use ``tokenizer_kind_for``
    to check whether the count is native or cross-model approximate.
    """

    def get_num_tokens_from_messages(
        self,
        messages: Sequence[BaseMessage],
        tools: Any = None,  # accepted for signature compat, unused
        *,
        allow_fetching_images: bool = True,
    ) -> int:
        encode, _ = _get_encoder(self.model_name)
        tokens_per_message = 3
        tokens_per_name = 1
        total = 0
        for m in messages:
            total += tokens_per_message
            role = getattr(m, "type", None) or getattr(m, "role", "")
            has_name = bool(getattr(m, "name", None)) or role == "system"
            if has_name:
                total += tokens_per_name
            total += len(encode(_content_to_text(m)))
            total += 3  # role tag + end-of-message marker
        return total + 3  # priming


def build_chat_model(
    provider_name: str,
    tier: str,
    *,
    config: dict[str, Any],
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> ChatModel:
    """Build a ``ChatModel`` (langmem-compatible) from the harness config.

    Used as the summarization model for A2: langmem invokes it to generate
    summaries and calls its ``get_num_tokens_from_messages`` for exact
    token counting.
    """
    cfg = provider_config(config, provider_name)
    base_url = cfg.get("base_url", "")
    api_key = cfg.get("api_key", "")
    model_cfg = cfg.get("tiers", {}).get(tier, {})
    model_name = model_cfg.get("model", "")
    kwargs: dict[str, Any] = {
        "model": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    return ChatModel(**kwargs)


class MockChatModel(ChatModel):
    """Deterministic, network-free ``ChatModel`` for mock-mode A2 compaction.

    langmem's ``summarize_messages`` requires a langchain ``LanguageModelLike``
    object that supports ``invoke``/``ainvoke`` (for summary generation)
    and ``get_num_tokens_from_messages`` (for token counting). In CI-MOCK
    mode there is no real endpoint, so this subclass provides a
    deterministic summary ("MOCK SUMMARY of <N> messages, ids: …") without
    touching the network. ``ChatOpenAI.__init__`` still runs, but no call
    ever leaves the process because both invoke paths are overridden.

    Construction accepts the same kwargs as ``ChatModel`` (so demos can
    swap them 1:1); the api_key/base_url are never used.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Bypass the network-touching OpenAI client init as much as
        # possible while keeping the type a ChatModel for langmem's
        # isinstance checks. Use a dummy key; nothing is called.
        kwargs.setdefault("model", "mock-chat-model")
        kwargs.setdefault("api_key", "mock-key-not-used")
        kwargs.setdefault("base_url", "http://mock.invalid")
        super().__init__(**kwargs)

    def _mock_summary(self, messages: Sequence[Any]) -> Any:
        # Summarize by id so the output is deterministic per input set.
        ids = [
            getattr(m, "id", None) or (m.get("id") if isinstance(m, dict) else "?")
            for m in messages
        ]
        # langmem wraps the user summary-request in its own prompt
        # template; the messages we see here include that wrapper. The
        # only thing we need is a stable string the running summary can
        # carry forward.
        joined = ",".join(str(i) for i in ids if i)
        from langchain_core.messages import AIMessage
        content = f"MOCK SUMMARY of {len(messages)} messages (ids: {joined[:80]})."
        return AIMessage(content=content)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        messages = input if isinstance(input, list) else (
            input.to_messages() if hasattr(input, "to_messages") else []
        )
        return self._mock_summary(messages)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.invoke(input, config, **kwargs)
