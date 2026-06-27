# ============================ STAGE 05 · CONCLUSIONS ============================
import json

import numpy as np
import streamlit as st

from core.llm import call_llm
from ui.components import insight_block, page_header, stat_cards


def render(groq_client, model):
    page_header(
        "STAGE 05", "Conclusions",
        "Synthesise findings against the scope questions — then stress-test them.",
    )

    df = st.session_state.get("df_clean")
    if df is None:
        st.warning("Load a dataset in Stage 02 first.")
        st.stop()

    if not st.session_state["groq_ok"]:
        st.info("Add your Groq API key in the sidebar to generate conclusions.")
        st.stop()

    questions = st.session_state.get("questions", [])

    stat_cards([
        {"val": str(len(questions)),                              "lbl": "Scope Qs",       "cls": "ok" if questions else "warn"},
        {"val": f"{df.shape[0]:,}",                               "lbl": "Rows analysed",  "cls": ""},
        {"val": "yes" if st.session_state.get("conclusions") else "no", "lbl": "Conclusions",    "cls": "ok" if st.session_state.get("conclusions") else ""},
        {"val": "yes" if st.session_state.get("stress_test") else "no", "lbl": "Stress-tested", "cls": "ok" if st.session_state.get("stress_test") else ""},
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    if not questions:
        insight_block(
            "No scope questions defined. Go back to Stage 01 — conclusions without "
            "questions are just an undirected data dump.",
            label="Missing Scope", color="amber",
        )

    # ── Build the evidence pack the model reasons over ──
    def _evidence() -> str:
        a = st.session_state.get("assessment", {})
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        ns = {}
        for c in num_cols[:8]:
            s = df[c].dropna()
            if len(s):
                ns[c] = {
                    "mean":   round(float(s.mean()), 3),
                    "median": round(float(s.median()), 3),
                    "std":    round(float(s.std()), 3),
                    "min":    round(float(s.min()), 3),
                    "max":    round(float(s.max()), 3),
                }

        cs = {}
        for c in cat_cols[:5]:
            vc = df[c].value_counts().head(5)
            cs[c] = {str(k): int(v) for k, v in vc.items()}

        corr_sum = "n/a"
        if len(num_cols) >= 2:
            corr = df[num_cols].corr(numeric_only=True).abs()
            off = ~np.eye(corr.shape[0], dtype=bool)
            stacked = corr.where(off).stack().dropna().sort_values(ascending=False)
            pairs = [
                f"{i[0]}~{i[1]}: r={round(df[[i[0], i[1]]].corr().iloc[0,1], 3)}"
                for i in stacked.head(5).index
            ]
            corr_sum = "; ".join(pairs) if pairs else "n/a"

        return (
            f"Scope questions: {questions}\n"
            f"Shape: {df.shape[0]:,} rows × {df.shape[1]} cols\n"
            f"Cleaning steps applied: {st.session_state.get('cleaning_log', [])}\n"
            f"Outlier cols: {a.get('outliers', {})}\n"
            f"Skewed cols: {a.get('skewness', {})}\n"
            f"Numeric stats: {json.dumps(ns)}\n"
            f"Top categories: {json.dumps(cs)}\n"
            f"Strongest correlations: {corr_sum}\n"
        )

    if st.button("▶ Generate Conclusions", type="primary"):
        prompt = (
            _evidence()
            + "\nWrite conclusions as a senior analyst:\n"
            "1. Answer each scope question directly, citing the specific numbers above.\n"
            "2. Mark correlation-based claims as association, NOT causation.\n"
            "3. List what this data CANNOT tell us (gaps, confounders).\n"
            "4. End with the single most decision-relevant finding.\n"
            "Numbered. No preamble. No flattery."
        )
        with st.spinner("Synthesising conclusions…"):
            try:
                st.session_state["conclusions"] = call_llm(
                    groq_client, model, prompt, max_tokens=1200
                )
                st.session_state["stress_test"] = ""  # invalidate old stress test
            except ValueError as e:
                st.error(str(e))

    if st.session_state.get("conclusions"):
        insight_block(st.session_state["conclusions"], label="DataGPT · Conclusions")

        st.markdown("---")
        st.markdown(
            "**Stress-test** — make the model attack its own conclusions before you "
            "put your name on them."
        )
        if st.button("🔍 Stress-Test Conclusions"):
            prompt = (
                "These are an analyst's conclusions:\n\n"
                f"{st.session_state['conclusions']}\n\n"
                "Now act as a hostile peer reviewer. Be specific:\n"
                "1. Which claims are over-stated relative to the evidence?\n"
                "2. What confounders or selection effects could explain the findings?\n"
                "3. Where is correlation being read as causation?\n"
                "4. What additional data would change the conclusion?\n"
                "Numbered. Direct. Do not soften."
            )
            with st.spinner("Attacking the conclusions…"):
                try:
                    st.session_state["stress_test"] = call_llm(
                        groq_client, model, prompt, max_tokens=1000, temperature=0.5
                    )
                except ValueError as e:
                    st.error(str(e))

    if st.session_state.get("stress_test"):
        insight_block(
            st.session_state["stress_test"],
            label="DataGPT · Stress-Test (devil's advocate)", color="amber",
        )
