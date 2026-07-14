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
                # Probe to filter out 402 during init
                try:
                    client.invoke_sync([{"role": "user", "content": "ping"}], max_completion_tokens=1, timeout=10)
                except APIStatusError as exc:
                    if exc.status_code == 402:
                        log_balance_error(name, getattr(client, "model", ""), str(exc))
                        continue
                    # Non-402 errors (429/500) are transient; keep client anyway
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
        """Echo transform applied uniformly to every input (no sentinel special-casing)."""
        input_text = "".join(_message_content(m) for m in messages)
        short_hash = hashlib.sha256(input_text.encode()).hexdigest()[:8]
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
