"""Language-model access behind one small interface.

Everything in this project that talks to a model (the judge, the LLM-backed
memory system, a unit's brain, the designer's proposer) depends only on
``complete(...)`` below, so the provider can be swapped and tests can run with
``MockLLM``.

``AnthropicClient`` uses the official ``anthropic`` SDK.  ``CachedLLM`` wraps
any client in a content-addressed transcript store: current Claude models do
not accept a sampling temperature, so reproducible, auditable judge verdicts
come from recording what was said, not from pinning the sampler.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from ..util import hash_obj

__all__ = [
    "LLMResult",
    "LLMClient",
    "LLMConfigError",
    "AnthropicClient",
    "CachedLLM",
    "MockLLM",
    "DEFAULT_MODEL",
]

DEFAULT_MODEL = "claude-opus-5-5"

# USD per million tokens (input, output); used only for cost reporting.
_PRICES = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
}
# Models for which the server-side refusal fallback parameter is offered.
_FALLBACK_MODELS = {
    "claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
}
_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMConfigError(RuntimeError):
    """The client is misconfigured (bad key, unknown model); retrying cannot help."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    parsed: Any = None  # decoded JSON when a schema was requested
    status: str = "ok"  # ok | truncated | refused | error
    model: str = ""  # the model that actually served the request
    usage: Mapping = field(default_factory=dict)
    request_id: str | None = None
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "text": self.text, "parsed": self.parsed, "status": self.status,
            "model": self.model, "usage": dict(self.usage), "request_id": self.request_id,
        }


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        *,
        system: str | None,
        prompt: str,
        max_tokens: int = 16000,
        json_schema: Mapping | None = None,
        effort: str | None = None,
        purpose: str = "",
        cache_system: bool = False,
    ) -> LLMResult: ...


class AnthropicClient:
    """Claude via the official SDK.

    ``api_key=None`` lets the SDK resolve credentials from the environment
    (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or an ``ant auth login``
    profile).  Requests are streamed so long histories do not hit HTTP timeouts.

    For models that support it, the server-side refusal fallback is enabled by
    default: if a safety classifier declines a request the API re-runs it on a
    fallback model inside the same call.  ``LLMResult.model`` always reports the
    model that actually answered.  Pass ``use_fallbacks=False`` to turn it off.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        use_fallbacks: bool | None = None,
        max_retries: int = 4,
        timeout: float = 600.0,
        http_client: Any = None,
    ):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the SDK
            raise LLMConfigError("install the 'anthropic' package to use AnthropicClient") from exc
        self._anthropic = anthropic
        kwargs: dict = {"max_retries": max_retries, "timeout": timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if http_client is not None:
            kwargs["http_client"] = http_client
        self._client = anthropic.Anthropic(**kwargs)
        self.model = model
        self.effort = effort
        self.use_fallbacks = (model in _FALLBACK_MODELS) if use_fallbacks is None else use_fallbacks
        self._lock = threading.Lock()
        self.totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "usd": 0.0}

    def _record(self, usage: Mapping, served_model: str) -> None:
        price_in, price_out = _PRICES.get(served_model, _PRICES.get(self.model, (0.0, 0.0)))
        usd = (usage.get("input_tokens", 0) * price_in + usage.get("output_tokens", 0) * price_out) / 1e6
        with self._lock:
            self.totals["calls"] += 1
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens"):
                self.totals[k] += int(usage.get(k, 0) or 0)
            self.totals["usd"] += usd

    def complete(
        self,
        *,
        system: str | None,
        prompt: str,
        max_tokens: int = 16000,
        json_schema: Mapping | None = None,
        effort: str | None = None,
        purpose: str = "",
        cache_system: bool = False,
    ) -> LLMResult:
        anthropic = self._anthropic
        output_config: dict = {"effort": effort or self.effort}
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": dict(json_schema)}
        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if system:
            if cache_system:
                kwargs["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            else:
                kwargs["system"] = system
        try:
            if self.use_fallbacks:
                with self._client.beta.messages.stream(
                    betas=[_FALLBACK_BETA], fallbacks="default", **kwargs
                ) as stream:
                    message = stream.get_final_message()
            else:
                with self._client.messages.stream(**kwargs) as stream:
                    message = stream.get_final_message()
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as exc:
            raise LLMConfigError(f"{type(exc).__name__}: {exc}") from exc
        except anthropic.BadRequestError as exc:
            return LLMResult(text=f"bad_request: {exc}", status="error", model=self.model)
        except anthropic.RateLimitError as exc:
            return LLMResult(text=f"rate_limited: {exc}", status="error", model=self.model)
        except anthropic.APIStatusError as exc:
            return LLMResult(text=f"api_status_{exc.status_code}: {exc}", status="error", model=self.model)
        except anthropic.APIConnectionError as exc:
            return LLMResult(text=f"connection: {exc}", status="error", model=self.model)

        text = "".join(block.text for block in message.content if block.type == "text")
        usage_obj = message.usage
        usage = {
            k: int(getattr(usage_obj, k, 0) or 0)
            for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        }
        served = getattr(message, "model", self.model) or self.model
        self._record(usage, served)
        request_id = getattr(message, "_request_id", None)
        if message.stop_reason == "refusal":
            return LLMResult(text=text, status="refused", model=served, usage=usage, request_id=request_id)
        status = "truncated" if message.stop_reason == "max_tokens" else "ok"
        parsed = None
        if json_schema is not None and status == "ok":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return LLMResult(text=text, status="error", model=served, usage=usage, request_id=request_id)
        return LLMResult(text=text, parsed=parsed, status=status, model=served, usage=usage, request_id=request_id)


class CachedLLM:
    """Content-addressed transcript cache around any ``LLMClient``.

    The key covers the model and every request field, so an identical request
    always returns the recorded response.  Only ``ok`` results are stored.  The
    JSONL file is the audit trail for LLM-judged evaluations.
    """

    def __init__(self, inner: Any, path: str | None = None):
        self.inner = inner
        self.model = getattr(inner, "model", "unknown")
        self.path = path
        self._mem: dict[str, dict] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec["result"]

    def key(self, **request) -> str:
        return hash_obj({"model": self.model, **request})

    def complete(self, *, system, prompt, max_tokens=16000, json_schema=None, effort=None, purpose="", cache_system=False):
        key = self.key(
            system=system, prompt=prompt, max_tokens=max_tokens,
            json_schema=json_schema, effort=effort, purpose=purpose,
        )
        with self._lock:
            rec = self._mem.get(key)
        if rec is not None:
            self.hits += 1
            return LLMResult(
                text=rec["text"], parsed=rec.get("parsed"), status=rec["status"], model=rec.get("model", ""),
                usage=rec.get("usage", {}), request_id=rec.get("request_id"), cached=True,
            )
        self.misses += 1
        result = self.inner.complete(
            system=system, prompt=prompt, max_tokens=max_tokens, json_schema=json_schema,
            effort=effort, purpose=purpose, cache_system=cache_system,
        )
        if result.status == "ok":
            with self._lock:
                self._mem[key] = result.to_dict()
                if self.path:
                    with open(self.path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"key": key, "result": result.to_dict()}, ensure_ascii=False) + "\n")
        return result


class MockLLM:
    """Deterministic stand-in for tests and offline demos.

    ``responder`` receives the request as a dict and returns either a string
    (plain text), a dict/list (treated as the parsed JSON output), or an
    ``LLMResult`` for full control.
    """

    def __init__(self, responder: Callable[[dict], Any], model: str = "mock-llm"):
        self.responder = responder
        self.model = model
        self.calls: list[dict] = []

    def complete(self, *, system, prompt, max_tokens=16000, json_schema=None, effort=None, purpose="", cache_system=False):
        request = {
            "system": system, "prompt": prompt, "max_tokens": max_tokens,
            "json_schema": json_schema, "effort": effort, "purpose": purpose,
        }
        self.calls.append(request)
        out = self.responder(request)
        if isinstance(out, LLMResult):
            return out
        if isinstance(out, (dict, list)):
            return LLMResult(text=json.dumps(out), parsed=out, model=self.model)
        return LLMResult(text=str(out), model=self.model)
