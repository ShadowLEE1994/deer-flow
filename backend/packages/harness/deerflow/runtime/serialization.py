"""Canonical serialization for LangChain / LangGraph objects.

Provides a single source of truth for converting LangChain message
objects, Pydantic models, and LangGraph state dicts into plain
JSON-serialisable Python structures.

Consumers: ``deerflow.runtime.runs.worker`` (SSE publishing) and
``app.gateway.routers.threads`` (REST responses).
"""

from __future__ import annotations

import re
from typing import Any

# Matches <think>...</think> blocks (including multiline) that some models
# (GLM, DeepSeek, Qwen, etc.) embed inline in their response content.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking_from_msg(msg: dict) -> dict:
    """Strip thinking content from a single serialized message dict.

    Handles three cases:
    1. String ``content`` with ``<think>...</think>`` tags (GLM / DeepSeek inline)
    2. List ``content`` with blocks of ``type == "thinking"`` (Claude extended thinking)
    3. ``additional_kwargs.reasoning_content`` field (DeepSeek / MiniMax / Codex)
    """
    result = dict(msg)

    content = result.get("content")
    if isinstance(content, str):
        result["content"] = _THINK_TAG_RE.sub("", content).strip()
    elif isinstance(content, list):
        result["content"] = [block for block in content if not (isinstance(block, dict) and block.get("type") == "thinking")]

    add_kwargs = result.get("additional_kwargs")
    if isinstance(add_kwargs, dict) and "reasoning_content" in add_kwargs:
        result["additional_kwargs"] = {k: v for k, v in add_kwargs.items() if k != "reasoning_content"}

    return result


def _filter_thinking_recursive(obj: Any) -> Any:
    """Recursively strip thinking from any serialized LangGraph object."""
    if isinstance(obj, dict):
        if obj.get("type") in ("ai", "AIMessage", "AIMessageChunk"):
            obj = _strip_thinking_from_msg(obj)
        return {k: _filter_thinking_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_filter_thinking_recursive(item) for item in obj]
    return obj


_AI_MSG_TYPES = frozenset({"ai", "AIMessage", "AIMessageChunk"})


def _has_text_content(content: Any) -> bool:
    """Return True if *content* contains non-empty human-readable text."""
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(isinstance(b, dict) and b.get("type") == "text" and bool(b.get("text", "").strip()) for b in content)
    return False


def _is_intermediate_msg_dict(msg: Any) -> bool:
    """Return True if a serialized message dict is an intermediate agent step.

    Filters:
    - ``type == "tool"`` — ToolMessage carrying raw tool execution results.
    - ``type == "ai"`` with ``tool_calls`` but no text — pure tool-dispatch
      messages that contain only a tool-call decision and no visible content.
    """
    if not isinstance(msg, dict):
        return False
    msg_type = msg.get("type", "")
    if msg_type == "tool":
        return True
    if msg_type in _AI_MSG_TYPES:
        tool_calls = msg.get("tool_calls") or []
        return bool(tool_calls) and not _has_text_content(msg.get("content", ""))
    return False


def _filter_intermediate_from_values(obj: Any) -> Any:
    """Remove intermediate-step messages from a serialized ``values`` snapshot.

    Strips ToolMessage entries and AI-only-tool-call messages from the top-level
    ``messages`` list so the third-party caller only sees final AI text replies.
    """
    if not isinstance(obj, dict):
        return obj
    result = dict(obj)
    messages = result.get("messages")
    if isinstance(messages, list):
        result["messages"] = [m for m in messages if not _is_intermediate_msg_dict(m)]
    return result


def is_intermediate_messages_chunk(raw_chunk: Any) -> bool:
    """Return True if *raw_chunk* (first element of a messages-mode tuple) is
    an intermediate agent step that should be suppressed from the SSE stream.

    Called by ``worker.py`` before publishing each ``messages`` event so that
    ToolMessage chunks and tool-call-only AI chunks are never sent to clients
    when ``filter_intermediate_steps`` is enabled.
    """
    chunk_type = getattr(raw_chunk, "type", "")
    if chunk_type == "tool":
        return True
    if chunk_type in _AI_MSG_TYPES:
        has_tool_activity = bool(getattr(raw_chunk, "tool_call_chunks", None) or getattr(raw_chunk, "tool_calls", None))
        return has_tool_activity and not _has_text_content(getattr(raw_chunk, "content", ""))
    return False


def serialize_lc_object(obj: Any) -> Any:
    """Recursively serialize a LangChain object to a JSON-serialisable dict."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: serialize_lc_object(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_lc_object(item) for item in obj]
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    # Pydantic v1 / older objects
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    # Last resort
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def serialize_channel_values(channel_values: dict[str, Any]) -> dict[str, Any]:
    """Serialize channel values, stripping internal LangGraph keys.

    Internal keys like ``__pregel_*`` and ``__interrupt__`` are removed
    to match what the LangGraph Platform API returns.
    """
    result: dict[str, Any] = {}
    for key, value in channel_values.items():
        if key.startswith("__pregel_") or key == "__interrupt__":
            continue
        result[key] = serialize_lc_object(value)
    return result


def serialize_messages_tuple(obj: Any) -> Any:
    """Serialize a messages-mode tuple ``(chunk, metadata)``."""
    if isinstance(obj, tuple) and len(obj) == 2:
        chunk, metadata = obj
        return [serialize_lc_object(chunk), metadata if isinstance(metadata, dict) else {}]
    return serialize_lc_object(obj)


def serialize(obj: Any, *, mode: str = "", filter_thinking: bool = False, filter_intermediate_steps: bool = False) -> Any:
    """Serialize LangChain objects with mode-specific handling.

    * ``messages`` — obj is ``(message_chunk, metadata_dict)``
    * ``values`` — obj is the full state dict; ``__pregel_*`` keys stripped
    * everything else — recursive ``model_dump()`` / ``dict()`` fallback

    When *filter_thinking* is ``True``, thinking/reasoning content is stripped
    from AI messages before returning:
    - ``<think>...</think>`` inline tags (GLM, DeepSeek, Qwen, etc.)
    - Content blocks with ``type == "thinking"`` (Claude extended thinking)
    - ``additional_kwargs.reasoning_content`` field

    When *filter_intermediate_steps* is ``True``, intermediate agent steps are
    removed from ``values`` state snapshots: ToolMessage entries and AI messages
    that only contain tool_calls with no text content are excluded from the
    ``messages`` list.  (For ``messages`` mode, intermediate chunks are filtered
    upstream in ``worker.py`` before ``serialize`` is called.)
    """
    if mode == "messages":
        result = serialize_messages_tuple(obj)
    elif mode == "values":
        result = serialize_channel_values(obj) if isinstance(obj, dict) else serialize_lc_object(obj)
        if filter_intermediate_steps:
            result = _filter_intermediate_from_values(result)
    else:
        result = serialize_lc_object(obj)

    if filter_thinking:
        result = _filter_thinking_recursive(result)
    return result
