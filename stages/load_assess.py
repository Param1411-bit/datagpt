# ============================ STAGE 02 · LOAD & ASSESS ============================
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import STEP_ICONS
from core.charts import _layout
from core.llm import call_llm
from core.loader import load_file
from core.profiling import assess_data
from ui.components import (
    dataset_info_block, insight_block, page_header, pf, quality_badge, stat_cards,
)
from ui.state import bump_data_version


def render(groq_client, model):
    page_header(
        "STAGE 02", "Load & Assess",
        "Gather → Summary → Column Descriptions → Assess → Document Issues",
    )

    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
    if uploaded:
        try:
            with st.spinner("Parsing file…"):
                raw = load_file(uploaded.read(), uploaded.name)
            st.session_state.update({
                "df_original":  raw,
                "df_clean":     raw.copy(),
                "filename":     uploaded.name,
                "cleaning_log": [],
                "suggestions":  [],
                "assessment":   {},
            })
            # New data → recompute the cached sidebar metric and invalidate
            # every version-gated cache (charts, column summary).
            st.session_state["missing_pct_overall"] = round(
                raw.isnull().mean().mean() * 100, 1
            )
            bump_data_version()
            st.success(
                f"✓ Loaded **{uploaded.name}** — "
                f"{raw.shape[0]:,} rows × {raw.shape[1]} columns"
            )
        except ValueError as e:
            st.error(str(e))
            st.stop()

    df = st.session_state.get("df_original")
    if df is None:
        insight_block("Upload a file above to begin.", label="Waiting", color="amber")
        st.stop()

    # ── Step 1+2 — Summary + Column Descriptions ──
    dataset_info_block(df, st.session_state.get("filename", ""))

    # ── Step 3 — Programmatic Assessment ──
    st.markdown("---")
    st.markdown(
        "<div class='dinfo'><div class='lbl'>Step 3 — Programmatic Assessment</div></div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Running audit…"):
        a = assess_data(df)
        st.session_state["assessment"]  = a
        st.session_state["suggestions"] = a["suggestions"]

    n_null_cols  = len(a["missing"])
    n_null_cells = sum(a["missing"].values())
    null_pct     = round(n_null_cells / (df.shape[0] * df.shape[1]) * 100, 1)
    dup          = a["duplicate_count"]

    stat_cards([
        {"val": f"{df.shape[0]:,}",          "lbl": "Rows",          "cls": ""},
        {"val": str(df.shape[1]),             "lbl": "Columns",       "cls": ""},
        {"val": str(n_null_cols),             "lbl": "Cols w/ Nulls", "cls": "bad" if n_null_cols else "ok"},
        {"val": f"{null_pct}%",              "lbl": "Cells Missing", "cls": "bad" if null_pct > 5 else "ok"},
        {"val": str(dup),                    "lbl": "Duplicates",    "cls": "bad" if dup else "ok"},
        {"val": str(len(a["outliers"])),     "lbl": "Outlier Cols",  "cls": "warn" if a["outliers"] else "ok"},
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    t_head, t_null, t_stats, t_issues = st.tabs([
        "Head / Sample", "Nulls & Outliers", "Numeric Stats", "Issues Found",
    ])

    with t_head:
        st.markdown("*First 10 rows — raw, unmodified:*")
        st.dataframe(df.head(10), width="stretch")
        st.markdown("*Random sample (10 rows):*")
        st.dataframe(df.sample(min(10, len(df)), random_state=42), width="stretch")

    with t_null:
        if a["missing"]:
            md = pd.DataFrame({
                "Column": list(a["missing"].keys()),
                "Count":  list(a["missing"].values()),
                "Pct":    [a["missing_pct"][k] for k in a["missing"]],
            }).sort_values("Pct", ascending=False)
            fig_m = go.Figure(go.Bar(
                x=md["Column"], y=md["Pct"],
                marker=dict(
                    color=md["Pct"],
                    colorscale=[[0, "#3ec98a"], [0.3, "#f0a020"], [1, "#ef5e5e"]],
                    cmin=0, cmax=100,
                ),
                text=[f"{v:.1f}%" for v in md["Pct"]], textposition="auto",
            ))
            _layout(fig_m, "Null % per Column", xt="Column", yt="% Missing")
            pf(fig_m)
            st.dataframe(md, width="stretch")
        else:
            st.markdown(
                "<span class='tag tg'>✓ No missing values detected</span>",
                unsafe_allow_html=True,
            )

        if a["outliers"]:
            st.markdown("**IQR Outlier flags:**")
            for c, n in sorted(a["outliers"].items(), key=lambda x: -x[1]):
                pct = round(n / len(df) * 100, 1)
                st.markdown(
                    f"<span class='tag {'tr' if pct>5 else 'ta'}'>"
                    f"{c}: {n} rows ({pct}%)</span>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<span class='tag tg'>✓ No IQR outliers detected</span>",
                unsafe_allow_html=True,
            )

        if a["skewness"]:
            st.markdown("**Highly skewed columns (|skew| > 1):**")
            for c, sk in sorted(a["skewness"].items(), key=lambda x: -abs(x[1])):
                direction = "right" if sk > 0 else "left"
                st.markdown(
                    f"<span class='tag {'tr' if abs(sk)>2 else 'ta'}'>"
                    f"{c}: {sk:+.2f} ({direction}-skewed)</span>",
                    unsafe_allow_html=True,
                )

    with t_stats:
        nc = df.select_dtypes(include=[np.number]).columns.tolist()
        if nc:
            desc = df[nc].describe().round(3).T
            desc["skew"]     = df[nc].skew().round(3)
            desc["kurtosis"] = df[nc].kurtosis().round(3)
            st.dataframe(desc, width="stretch")
        else:
            st.info("No numeric columns in this dataset.")

    with t_issues:
        sugs = a["suggestions"]
        if sugs:
            st.markdown(f"**{len(sugs)} issue(s) found — classified by quality dimension:**")
            for i, s in enumerate(sugs, 1):
                icon, lbl = STEP_ICONS.get(s["type"], ("?", s["type"]))
                dim = s.get("dim", "")
                with st.expander(f"{icon}  [{i}]  {s['col']}  ·  {lbl}"):
                    if dim:
                        quality_badge(dim, "")
                    st.markdown(
                        f"<div class='ins'><div class='lbl'>Why this needs fixing</div>"
                        f"{s['reason']}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            insight_block(
                "No structural issues detected. Verify the correct file was uploaded "
                "before skipping Stage 03.",
                label="Audit Result", color="green",
            )

    # ── AI audit summary ──
    if st.session_state["groq_ok"]:
        if st.button("Get AI Audit Summary"):
            ns = {}
            for c in df.select_dtypes(include=[np.number]).columns[:6]:
                ns[c] = {
                    "mean": round(df[c].mean(), 3),
                    "std":  round(df[c].std(), 3),
                    "min":  round(df[c].min(), 3),
                    "max":  round(df[c].max(), 3),
                }
            prompt = (
                f"Dataset: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
                f"Columns: {list(df.columns)}\n"
                f"Dtypes: {a['dtypes']}\n"
                f"Missing %: {a['missing_pct']}\n"
                f"Duplicates: {a['duplicate_count']}\n"
                f"Outlier cols: {a['outliers']}\n"
                f"Skewed cols: {a['skewness']}\n"
                f"Numeric stats: {json.dumps(ns)}\n"
                f"Scope questions: {st.session_state['questions']}\n\n"
                "Reply:\n"
                "1. Dataset quality summary (2 sentences).\n"
                "2. Top 3 issues — cite specific column names.\n"
                "3. Can each scope question be answered with this data? "
                "If not, what exactly is missing?\n"
                "No preamble. Be direct."
            )
            with st.spinner("Analysing…"):
                try:
                    reply = call_llm(groq_client, model, prompt, max_tokens=650)
                    insight_block(reply)
                except ValueError as e:
                    st.error(str(e))
