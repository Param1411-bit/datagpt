# ============================ UI · COMPONENTS ============================
# Reusable render helpers shared across stages. Pure presentation — they read
# arguments / session state and emit Streamlit markup.

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.state import current_version


def page_header(pill: str, title: str, sub: str = ""):
    """Render consistent stage header."""
    sub_html = f"<div class='ph-sub'>{sub}</div>" if sub else ""
    st.markdown(f"""
    <div class='ph'>
      <div class='ph-pill'>{pill}</div>
      <h1>{title}</h1>
      {sub_html}
    </div>""", unsafe_allow_html=True)


def insight_block(text: str, label: str = "DataGPT Analysis", color: str = "blue"):
    """Render a structured analyst insight box."""
    cmap = {"blue": "var(--blue)", "amber": "var(--amber)", "green": "var(--green)"}
    c = cmap.get(color, "var(--blue)")
    st.markdown(f"""
    <div class='ins' style='border-left-color:{c};'>
      <div class='lbl' style='color:{c};'>{label}</div>
      {text}
    </div>""", unsafe_allow_html=True)


def chart_rationale(why_this: str, alternatives: str, question: str):
    """Render analytical justification beneath every chart."""
    st.markdown(f"""
    <div class='rat'>
      <div class='lbl'>Chart Rationale — Why This Chart?</div>
      <b>Chosen because:</b> {why_this}<br>
      <b>Alternatives considered & rejected:</b> {alternatives}<br>
      <b>Analytical question answered:</b> {question}
    </div>""", unsafe_allow_html=True)


def quality_badge(dim: str, text: str):
    """Render a data-quality-dimension badge."""
    cls_map = {
        "Completeness": "qd-completeness",
        "Validity":     "qd-validity",
        "Accuracy":     "qd-accuracy",
        "Consistency":  "qd-consistency",
        "Tidiness":     "qd-tidiness",
    }
    cls = cls_map.get(dim, "tb")
    st.markdown(
        f"<span class='qdim {cls}'>{dim}</span> {text}",
        unsafe_allow_html=True,
    )


def stat_cards(items: list):
    """Render a row of metric stat cards."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        col.markdown(f"""
        <div class='sc'>
          <div class='v {item.get("cls","")}'>{item["val"]}</div>
          <div class='l'>{item["lbl"]}</div>
        </div>""", unsafe_allow_html=True)


def pf(fig: go.Figure):
    """
    Render a Plotly figure full-width.

    width='stretch' is the modern Streamlit API (>= ~1.4x) replacing the
    deprecated use_container_width=True. If you run an older Streamlit and hit
    a TypeError here, swap to use_container_width=True.
    """
    st.plotly_chart(fig, width="stretch")


def dataset_info_block(df: pd.DataFrame, filename: str):
    """
    Render the dataset summary + auto column descriptions shown on load.

    The per-column scan (nunique / notna / sample) is gated on the data version
    so it runs once per dataset, not on every rerun within Stage 02.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime64"]).columns.tolist()
    mem_mb   = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

    st.markdown("""
    <div class='dinfo'>
      <div class='lbl'>Step 1 — Dataset Summary</div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**File:** `{filename}`")
    c1.markdown(f"**Shape:** `{df.shape[0]:,} rows × {df.shape[1]} cols`")
    c1.markdown(f"**Memory:** `{mem_mb} MB`")
    c2.markdown(f"**Numeric cols ({len(num_cols)}):**")
    for c in num_cols[:8]:
        c2.markdown(f"  - `{c}` ({df[c].dtype})")
    c3.markdown(f"**Categorical cols ({len(cat_cols)}):**")
    for c in cat_cols[:8]:
        c3.markdown(f"  - `{c}` ({df[c].nunique()} unique)")
    if dt_cols:
        st.markdown(f"**Datetime cols:** `{', '.join(dt_cols)}`")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Step 2: Auto-generated column descriptions (gated on data version) ──
    st.markdown("""
    <div class='dinfo' style='margin-top:0.6rem;'>
      <div class='lbl'>Step 2 — Column Descriptions (auto-generated)</div>
    """, unsafe_allow_html=True)

    ver = current_version()
    if st.session_state.get("colsummary_version") != ver:
        col_rows = []
        for c in df.columns:
            nn   = int(df[c].notna().sum())
            uniq = int(df[c].nunique())
            dtype_str = str(df[c].dtype)
            sample = str(df[c].dropna().iloc[0]) if nn else "—"
            col_rows.append({
                "Column":       c,
                "Dtype":        dtype_str,
                "Non-Null":     nn,
                "Null %":       round((1 - nn / len(df)) * 100, 1),
                "Unique":       uniq,
                "Cardinality%": round(uniq / len(df) * 100, 1),
                "Sample Value": sample[:40],
            })
        st.session_state["colsummary_df"] = pd.DataFrame(col_rows)
        st.session_state["colsummary_version"] = ver

    st.dataframe(st.session_state["colsummary_df"], width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
