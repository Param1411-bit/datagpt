# ============================ UI · SIDEBAR ============================
# Renders the sidebar and returns (groq_client, model, stage). The Groq key is
# resolved from st.secrets / GROQ_API_KEY env var first (deployment-friendly,
# like the AxT bot's os.environ pattern), falling back to a text box.

import os

import streamlit as st

from config import STAGES
from core.llm import fetch_chat_models, make_client


def _resolve_env_key() -> str:
    """Return GROQ_API_KEY from Streamlit secrets or environment, else ''."""
    try:
        if "GROQ_API_KEY" in st.secrets:          # raises if no secrets file
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY", "").strip()


def render_sidebar():
    """Render sidebar; return (groq_client, model, stage)."""
    with st.sidebar:
        st.markdown("""
        <div class='wm'>
          <div class='wm-title'>Data<span>GPT</span></div>
          <div class='wm-sub'>AI Data Analyst</div>
        </div>""", unsafe_allow_html=True)

        # ── Groq key ──
        st.markdown(
            "<p style='font-family:var(--mono);font-size:0.65rem;color:var(--dimmer);"
            "letter-spacing:2px;text-transform:uppercase;margin-bottom:0.35rem;'>GROQ</p>",
            unsafe_allow_html=True,
        )

        env_key = _resolve_env_key()
        if env_key:
            api_key = env_key
            st.markdown(
                "<span class='tag tb'>● using GROQ_API_KEY secret</span>",
                unsafe_allow_html=True,
            )
        else:
            api_key = st.text_input(
                "Groq API Key", type="password", placeholder="gsk_...",
                help="Free at console.groq.com — or set GROQ_API_KEY in secrets/env",
                label_visibility="collapsed",
            )

        model_map = fetch_chat_models(api_key)
        model = st.selectbox("Model", list(model_map.keys()),
                             format_func=lambda m: model_map.get(m, m))

        groq_client = make_client(api_key)
        if api_key:
            st.session_state["groq_ok"] = bool(groq_client)
            cls = "tg" if groq_client else "tr"
            lbl = "● connected" if groq_client else "● key format invalid"
            st.markdown(f"<span class='tag {cls}'>{lbl}</span>", unsafe_allow_html=True)
        else:
            st.session_state["groq_ok"] = False
            st.markdown("<span class='tag ta'>● no key entered</span>", unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Workflow stage ──
        st.markdown(
            "<p style='font-family:var(--mono);font-size:0.65rem;color:var(--dimmer);"
            "letter-spacing:2px;text-transform:uppercase;margin-bottom:0.35rem;'>WORKFLOW</p>",
            unsafe_allow_html=True,
        )
        stage = st.radio(
            "Workflow Stage", STAGES,
            index=STAGES.index(st.session_state["stage"]),
            label_visibility="collapsed",
        )
        st.session_state["stage"] = stage

        st.markdown("<hr>", unsafe_allow_html=True)

        # ── Active dataset card (missing % read from cache, not recomputed) ──
        df_ref = st.session_state.get("df_clean")
        if df_ref is not None:
            st.markdown(
                "<p style='font-family:var(--mono);font-size:0.65rem;color:var(--dimmer);"
                "letter-spacing:2px;text-transform:uppercase;margin-bottom:0.35rem;'>ACTIVE DATASET</p>",
                unsafe_allow_html=True,
            )
            fname = st.session_state.get("filename", "")
            if fname:
                st.markdown(
                    f"<span class='tag tb' style='font-size:0.6rem;'>{fname}</span>",
                    unsafe_allow_html=True,
                )
            ca, cb = st.columns(2)
            ca.metric("Rows", f"{df_ref.shape[0]:,}")
            cb.metric("Cols", df_ref.shape[1])
            mp = st.session_state.get("missing_pct_overall", 0.0)
            st.markdown(
                f"<span class='tag {'tr' if mp > 5 else 'tg'}'>{mp}% missing</span>",
                unsafe_allow_html=True,
            )

    return groq_client, model, stage
