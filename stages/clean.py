# ============================ STAGE 03 · CLEAN DATA ============================
import traceback

import streamlit as st

from config import STEP_ICONS
from core.transforms import apply_cleaning
from ui.components import insight_block, page_header, quality_badge
from ui.state import bump_data_version


def render(groq_client, model):
    page_header(
        "STAGE 03", "Clean Data",
        "Define → Code → Test. Every step is documented.",
    )

    df_orig = st.session_state.get("df_original")
    if df_orig is None:
        st.warning("Load a dataset in Stage 02 first.")
        st.stop()

    sugs = st.session_state.get("suggestions", [])

    if not sugs:
        insight_block(
            "Audit found no structural issues. No cleaning needed — proceed to Stage 04.",
            label="Nothing to Clean", color="green",
        )
    else:
        st.markdown(
            f"**{len(sugs)} transformation(s) queued.** "
            "Review each below, then apply all at once."
        )
        for i, s in enumerate(sugs, 1):
            icon, lbl = STEP_ICONS.get(s["type"], ("?", s["type"]))
            dim = s.get("dim", "")
            with st.expander(f"{icon}  [{i}]  {s['col']}  ·  {lbl}"):
                if dim:
                    quality_badge(dim, "issue")
                st.markdown(
                    f"<div class='rat'><div class='lbl'>Define — Analyst Reasoning</div>"
                    f"{s['reason']}</div>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    bc1, bc2 = st.columns([1, 5])
    with bc1:
        run_btn = st.button("▶ Apply All", type="primary")
    with bc2:
        rst_btn = st.button("↩ Reset to Original")

    if run_btn:
        if not sugs:
            st.info("Nothing to apply.")
        else:
            with st.spinner("Applying transformations…"):
                try:
                    df_c, log = apply_cleaning(df_orig, sugs)
                    st.session_state["df_clean"]     = df_c
                    st.session_state["cleaning_log"] = log
                    st.session_state["missing_pct_overall"] = round(
                        df_c.isnull().mean().mean() * 100, 1
                    )
                    bump_data_version()   # rebuild EDA charts on cleaned data
                    st.success(f"Done — {len(log)} step(s) applied.")
                except Exception as e:
                    st.error(f"Failed: {e}")
                    st.error(traceback.format_exc())

    if rst_btn:
        st.session_state["df_clean"]     = df_orig.copy()
        st.session_state["cleaning_log"] = []
        st.session_state["missing_pct_overall"] = round(
            df_orig.isnull().mean().mean() * 100, 1
        )
        bump_data_version()
        st.info("Reset to original.")

    # ── Test step ──
    log = st.session_state.get("cleaning_log", [])
    if log:
        st.markdown("#### Cleaning Log — Test (verify each step worked)")
        for i, entry in enumerate(log, 1):
            st.markdown(
                f"<div class='sr'>"
                f"<span class='sn'>#{i:02d}</span>"
                f"<span>{entry}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    df_c = st.session_state.get("df_clean")
    if df_c is not None and log:
        st.markdown("---")
        st.markdown("#### Before vs After")
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Original**")
            st.markdown(
                f"<span class='tag ta'>"
                f"{df_orig.shape[0]:,} rows · {df_orig.shape[1]} cols · "
                f"{df_orig.isnull().sum().sum():,} nulls</span>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_orig.head(6), width="stretch")
        with cb:
            st.markdown("**After Cleaning**")
            remaining = df_c.isnull().sum().sum()
            st.markdown(
                f"<span class='tag tg'>"
                f"{df_c.shape[0]:,} rows · {df_c.shape[1]} cols · "
                f"{remaining:,} nulls</span>",
                unsafe_allow_html=True,
            )
            st.dataframe(df_c.head(6), width="stretch")
