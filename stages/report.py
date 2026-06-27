# ============================ STAGE 06 · REPORT ============================
from datetime import datetime

import streamlit as st

from core.llm import call_llm
from ui.components import insight_block, page_header


def _assemble_markdown(exec_summary: str = "") -> str:
    """Build the full case report from session state (works with or without AI)."""
    df       = st.session_state.get("df_clean")
    fname    = st.session_state.get("filename", "dataset")
    qs       = st.session_state.get("questions", [])
    log      = st.session_state.get("cleaning_log", [])
    concl    = st.session_state.get("conclusions", "")
    stress   = st.session_state.get("stress_test", "")
    a        = st.session_state.get("assessment", {})
    stamp    = datetime.now().strftime("%Y-%m-%d %H:%M")

    parts = [f"# DataGPT Analysis Report", f"*Generated {stamp} · source: `{fname}`*", ""]

    if exec_summary:
        parts += ["## Executive Summary", exec_summary, ""]

    parts += ["## 1. Scope"]
    if qs:
        parts += [f"{i}. {q}" for i, q in enumerate(qs, 1)]
    else:
        parts += ["_No scope questions were defined._"]
    parts += [""]

    if df is not None:
        parts += [
            "## 2. Dataset",
            f"- Shape: {df.shape[0]:,} rows × {df.shape[1]} columns",
            f"- Duplicate rows found: {a.get('duplicate_count', 0)}",
            f"- Columns with nulls: {len(a.get('missing', {}))}",
            f"- Outlier columns (IQR): {', '.join(a.get('outliers', {}).keys()) or 'none'}",
            "",
        ]

    parts += ["## 3. Cleaning Log"]
    if log:
        parts += [f"- {e}" for e in log]
    else:
        parts += ["_No cleaning steps were applied._"]
    parts += [""]

    parts += ["## 4. Conclusions", concl or "_Not generated._", ""]
    parts += ["## 5. Stress-Test / Limitations", stress or "_Not generated._", ""]
    parts += ["---", "_Educational analysis. Not professional/financial advice._"]
    return "\n".join(parts)


def render(groq_client, model):
    page_header(
        "STAGE 06", "Report",
        "Assemble the documented analysis into a shareable report.",
    )

    df = st.session_state.get("df_clean")
    if df is None:
        st.warning("Load a dataset in Stage 02 first.")
        st.stop()

    if not st.session_state.get("conclusions"):
        insight_block(
            "No conclusions yet. Generate them in Stage 05 first — the report is a "
            "package of work already done, not a substitute for doing it.",
            label="Nothing to Report", color="amber",
        )

    if st.button("▶ Generate Report", type="primary"):
        exec_summary = ""
        # Optional AI executive summary — report still assembles without a key.
        if st.session_state["groq_ok"] and st.session_state.get("conclusions"):
            prompt = (
                "Write a 3-sentence executive summary of this analysis for a busy "
                "stakeholder. Lead with the decision-relevant finding. No preamble.\n\n"
                f"Scope: {st.session_state.get('questions', [])}\n"
                f"Conclusions: {st.session_state['conclusions']}"
            )
            with st.spinner("Writing executive summary…"):
                try:
                    exec_summary = call_llm(groq_client, model, prompt, max_tokens=300)
                except ValueError as e:
                    st.warning(f"Skipped AI summary: {e}")
        st.session_state["case_report"] = _assemble_markdown(exec_summary)
        st.success("Report assembled.")

    report = st.session_state.get("case_report", "")
    if report:
        st.markdown(f"<div class='ra'>{report}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Downloads**")
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        base  = (st.session_state.get("filename", "dataset").rsplit(".", 1)[0]) or "dataset"

        d1, d2, d3 = st.columns(3)
        with d1:
            st.download_button(
                "⬇ Cleaned data (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name=f"{base}_cleaned_{stamp}.csv",
                mime="text/csv",
            )
        with d2:
            st.download_button(
                "⬇ Report (Markdown)",
                report.encode("utf-8"),
                file_name=f"{base}_report_{stamp}.md",
                mime="text/markdown",
            )
        with d3:
            log_txt = "\n".join(st.session_state.get("cleaning_log", [])) or "No cleaning steps applied."
            st.download_button(
                "⬇ Cleaning log (TXT)",
                log_txt.encode("utf-8"),
                file_name=f"{base}_cleaning_log_{stamp}.txt",
                mime="text/plain",
            )
