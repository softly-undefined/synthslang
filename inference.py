# Shared OpenAI-compatible inference wrapper for all three stages.
#
# Set OPENAI_API_KEY and, for a non-OpenAI endpoint, OPENAI_BASE_URL. Model IDs are
# passed through unchanged. The wrapper handles retries, JSON extraction, and optional
# reasoning traces. Reasoning mode may require temperature 1 or an omitted temperature.

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import openai


def make_client() -> openai.OpenAI:
    """Build a client from OPENAI_API_KEY and optional OPENAI_BASE_URL."""
    base_url = os.getenv("OPENAI_BASE_URL")  # None -> official OpenAI API
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if base_url:
            api_key = "EMPTY"  # many self-hosted OpenAI-compatible servers ignore the key
        else:
            raise SystemExit(
                "OPENAI_API_KEY is not set. Export your key first:\n"
                "  export OPENAI_API_KEY=...\n"
                "(optionally set OPENAI_BASE_URL to point at a different "
                "OpenAI-compatible engine, e.g. a local vLLM server)"
            )
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs)


def chat_full(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    retries: int = 2,
) -> tuple[str, str | None]:
    """Return response text and an optional reasoning trace, retrying API errors."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            # reasoning_content is a non-standard field some engines add, so read
            # it off the dumped message rather than as a typed attribute.
            reasoning = msg.model_dump().get("reasoning_content") or None
            return (msg.content or ""), reasoning
        except Exception as err:  # transient API/network errors
            last_err = err
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat call failed for {model}: {last_err}") from last_err


def chat(
    client: openai.OpenAI,
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float | None = 0.0,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    retries: int = 2,
) -> str:
    """Return only the response text from `chat_full`."""
    content, _ = chat_full(
        client, model, prompt, system=system, temperature=temperature,
        max_tokens=max_tokens, reasoning_effort=reasoning_effort, retries=retries,
    )
    return content


def extract_json(text: str) -> Any:
    """Extract bare, fenced, or prose-wrapped JSON from a model response."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Fall back to the widest balanced-looking span for each bracket type.
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = text.find(opener), text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no JSON found in model response: {text[:200]!r}")
