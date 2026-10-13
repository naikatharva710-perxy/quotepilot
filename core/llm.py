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
import sys
import time
from dataclasses import dataclass, field
from typing import Any

# Preference order. The key's actual model list decides what is used (see available_models).
# 2.5 models are only served to projects that used them before, so they come last.
DEFAULT_MODELS = ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite",
                  "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
_DEAD: set[str] = set()          # models that returned not-found / no-access / zero quota this process
_AVAILABLE: list[str] | None = None


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


def available_models() -> list[str] | None:
    """Ask the API which models this key can call. Cached; None if listing fails."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            names = []
            for m in client().models.list():
                actions = getattr(m, "supported_actions", None) or []
                if not actions or "generateContent" in actions:
                    names.append(str(m.name).replace("models/", ""))
            _AVAILABLE = names
            _stderr(f"available models: {', '.join(n for n in names if 'gemini' in n)[:600]}")
        except Exception as e:
            _stderr(f"model listing failed: {redact(e)}")
            _AVAILABLE = []
    return _AVAILABLE or None


def models() -> list[str]:
    preferred = _secret("GEMINI_MODEL")
    order = ([preferred] if preferred else []) + [m for m in DEFAULT_MODELS if m != preferred]
    avail = available_models()
    if avail:
        listed = [m for m in order if m in avail]
        if listed:
            order = listed
    return [m for m in order if m not in _DEAD] or order


def _stderr(msg: str) -> None:
    print(f"[quotepilot] {msg}", file=sys.stderr, flush=True)


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
            text = ""
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
                msg = str(e)
                _stderr(redact(f"{task} on {model} failed: {e.code} {msg[:400]}"))
                zero_quota = e.code == 429 and ("limit: 0" in msg or "limit\": 0" in msg)
                if e.code in (403, 404) or zero_quota or (e.code == 400 and "not supported" in msg.lower()):
                    _DEAD.add(model)
                    break
                if e.code == 429 and attempt < max_retries:
                    time.sleep(8 * (attempt + 1))
                    continue
                if e.code in (400,) and "API key" in str(e):
                    raise LLMUnavailable("The Gemini API key was rejected") from e
                break  # 404 / unsupported / repeated 429 -> try next model
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
                _log(TraceEntry(task, model, round(time.time() - t0, 1), False, prompt[:300], redact(f"bad JSON: {e}"), labels))
                _stderr(redact(f"{task} on {model}: unparseable output ({e}); first 300 chars: {text[:300]!r}"))
                if attempt < max_retries:
                    continue
                break
            except LLMUnavailable:
                raise
            except Exception as e:  # network errors, timeouts, empty/blocked responses
                last_err = e
                _log(TraceEntry(task, model, round(time.time() - t0, 1), False, prompt[:300], redact(f"{type(e).__name__}: {e}"[:300]), labels))
                _stderr(redact(f"{task} on {model}: {type(e).__name__}: {str(e)[:400]}"))
                break
    _stderr(redact(f"{task}: all models failed, last error {last_err}"))
    raise LLMUnavailable(redact(f"All models failed for '{task}': {last_err}"))
