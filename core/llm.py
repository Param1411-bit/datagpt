# ============================ CORE · LLM ============================
# Groq client construction, live model discovery, and the single chat-call
# wrapper used by every stage. No Streamlit UI here beyond @st.cache_data.

from typing import Optional

import streamlit as st
from groq import Groq


SYSTEM_PROMPT = """You are DataGPT — a senior data analyst with 10+ years of experience.
Rules:
- Direct and precise. No filler phrases, no flattery.
- Every claim is backed by a specific number from the data.
- Prefix uncertain statements with "Assumption:".
- If data is insufficient, say so — do not fill the gap.
- Use numbered lists for findings.
- Always state what the data CANNOT tell us."""


# Groq deprecates models roughly monthly, so a hardcoded list rots fast
# (mixtral-8x7b-32768 and gemma2-9b-it were both retired during 2025). The list
# is pulled live from the API once a key exists; FALLBACK_MODELS only shows
# before a key is entered or if the call fails.
FALLBACK_MODELS = {
    "llama-3.3-70b-versatile": "LLaMA 3.3 · 70B  (versatile)",
    "llama-3.1-8b-instant":    "LLaMA 3.1 · 8B   (fastest)",
}

# Substrings marking a model as NOT a chat-completion model — filtered out so
# the dropdown only offers things call_llm can actually use.
_NON_CHAT_HINTS = (
    "whisper", "tts", "embed", "guard", "moderation",
    "vision-preview", "playai", "orpheus",
)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def fetch_chat_models(api_key: str) -> dict:
    """
    Pull the live model list from Groq and keep only chat-completion models.

    Cached for an hour per key so models.list() is not called on every rerun.
    Returns {model_id: label}. Falls back to FALLBACK_MODELS on any failure
    (no key yet, network error, auth error) so the UI always has options.
    """
    key = (api_key or "").strip()
    if not key:
        return dict(FALLBACK_MODELS)
    try:
        listed = Groq(api_key=key).models.list().data
        out = {}
        for m in listed:
            mid = (getattr(m, "id", "") or "").strip()
            if not mid or any(h in mid.lower() for h in _NON_CHAT_HINTS):
                continue
            out[mid] = mid
        return out or dict(FALLBACK_MODELS)
    except Exception:
        return dict(FALLBACK_MODELS)


def make_client(api_key: str) -> Optional[Groq]:
    """Return a Groq client if key is non-empty, else None."""
    k = (api_key or "").strip()
    return Groq(api_key=k) if k else None


def call_llm(
    client: Groq,
    model: str,
    user_msg: str,
    system: str = SYSTEM_PROMPT,
    temperature: float = 0.35,
    max_tokens: int = 1800,
    history: Optional[list] = None,
) -> str:
    """
    Call Groq chat completion with granular error handling.

    `history` is an optional list of prior {"role", "content"} turns inserted
    between the system prompt and the current user message, so multi-turn chat
    carries context instead of treating every question in isolation.

    Raises ValueError with a user-readable message on auth/rate/model errors.
    """
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_msg})
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "invalid_api_key" in msg.lower():
            raise ValueError("Invalid API key — verify at console.groq.com.") from exc
        if "429" in msg or "rate_limit" in msg.lower():
            raise ValueError("Rate limit hit — wait a moment and retry.") from exc
        if "404" in msg or "model_not_found" in msg.lower():
            raise ValueError(f"Model '{model}' not found — pick another.") from exc
        raise ValueError(f"Groq API error: {msg[:220]}") from exc
