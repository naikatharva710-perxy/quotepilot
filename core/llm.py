"""Thin wrapper over the Gemini API.

Every call is logged to an in-memory trace so the buyer (and the reviewer) can see
exactly what the model was asked, which model answered, and how long it took.
Nothing in the app hardcodes model output: if there is no key, AI features are disabled
and the app says so.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODELS = ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"]


class LLMUnavailable(RuntimeError):
    pass


@dataclass
class TraceEntry:
    task: str
    model: str
    seconds: float
    ok: bool
    prompt_excerpt: str
    output_excerpt: str
    attachments: list[str] = field(default_factory=list)


TRACE: list[TraceEntry] = []


def _log(entry: TraceEntry) -> None:
    """Per-session trace when running inside Streamlit, global otherwise."""
    try:
        import streamlit as st
        st.session_state.setdefault("ai_trace", []).append(entry)
        return
    except Exception:
        pass
    TRACE.append(entry)


def trace() -> list[TraceEntry]:
    try:
        import streamlit as st
        return list(st.session_state.get("ai_trace", []))
    except Exception:
        return list(TRACE)


_KEY_PATTERN = re.compile(r"AIza[0-9A-Za-z_\-]{20,}")


def redact(text: str) -> str:
    """Strip anything that looks like a Google API key before it reaches logs or the screen."""
    text = _KEY_PATTERN.sub("[redacted-key]", str(text))
    k = api_key_raw()
    return text.replace(k, "[redacted-key]") if k else text


def api_key_raw() -> str | None:
    return _secret("GEMINI_API_KEY")


def _secret(name: str) -> str | None:
    try:
        import streamlit as st
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name)


def api_key() -> str | None:
    return _secret("GEMINI_API_KEY")


def models() -> list[str]:
    preferred = _secret("GEMINI_MODEL")
    out = [preferred] if preferred else []
    return out + [m for m in DEFAULT_MODELS if m not in out]


_client = None


def client():
    global _client
    key = api_key()
    if not key:
        raise LLMUnavailable("No GEMINI_API_KEY configured")
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=key)
    return _client


def _parse_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"[\[{].*[\]}]", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def generate(task: str, prompt: str, *, system: str | None = None, files: list[tuple[bytes, str, str]] | None = None,
             as_json: bool = True, temperature: float = 0.1, max_retries: int = 2) -> Any:
    """files: list of (bytes, mime_type, label). Returns parsed JSON (as_json) or text."""
    from google.genai import types, errors

    parts: list[Any] = []
    labels = []
    for data, mime, label in files or []:
        parts.append(types.Part.from_bytes(data=data, mime_type=mime))
        labels.append(label)
    parts.append(prompt)
    cfg = dict(temperature=temperature)
    if system:
        cfg["system_instruction"] = system
    if as_json:
        cfg["response_mime_type"] = "application/json"

    last_err: Exception | None = None
    for model in models():
        for attempt in range(max_retries + 1):
            t0 = time.time()
            try:
                resp = client().models.generate_content(model=model, contents=parts,
                                                        config=types.GenerateContentConfig(**cfg))
                text = resp.text or ""
                out = _parse_json(text) if as_json else text
                _log(TraceEntry(task, model, round(time.time() - t0, 1), True, prompt[:600], text[:600], labels))
                return out
            except errors.APIError as e:  # type: ignore[attr-defined]
                last_err = e
                _log(TraceEntry(task, model, round(time.time() - t0, 1), False, prompt[:300], redact(f"{e.code}: {str(e)[:300]}"), labels))
                if e.code == 429 and attempt < max_retries:
                    time.sleep(8 * (attempt + 1))
                    continue
                if e.code in (400,) and "API key" in str(e):
                    raise LLMUnavailable("The Gemini API key was rejected") from e
                break  # 404 / unsupported / repeated 429 -> try next model
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                _log(TraceEntry(task, model, round(time.time() - t0, 1), False, prompt[:300], redact(f"bad JSON: {e}"), labels))
                if attempt < max_retries:
                    continue
                break
    raise LLMUnavailable(redact(f"All models failed for '{task}': {last_err}"))
