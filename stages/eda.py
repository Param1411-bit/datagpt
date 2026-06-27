# ============================ STAGE 04 · EXPLORE (EDA) ============================
import numpy as np
import streamlit as st

from core.charts import build_eda_charts
from core.llm import SYSTEM_PROMPT, call_llm
from ui.components import chart_rationale, insight_block, page_header, pf
from ui.state import current_version


def render(groq_client, model):
    page_header(
        "STAGE 04", "Exploratory Data Analysis",
        "Every chart is chosen for a specific analytical reason.",
    )

    df_eda = st.session_state.get("df_clean")
    if df_eda is None:
        st.warning("Load a dataset in Stage 02 first.")
        st.stop()

    # ── Fix 2: build charts ONCE per dataset version, cache in session state ──
    # Avoids rebuilding (and re-hashing the frame) on every keystroke/click.
    ver = current_version()
    if st.session_state.get("eda_charts_version") != ver:
        with st.spinner("Building charts…"):
            st.session_state["eda_charts"] = build_eda_charts(df_eda)
        st.session_state["eda_charts_version"] = ver
    charts = st.session_state["eda_charts"]

    if not charts:
        st.error(
            "No charts could be generated — the dataset has no usable numeric or "
            "categorical columns. Check Stage 02 for type-conversion suggestions."
        )
        st.stop()

    # ── Fix 1: render ONE chart at a time (this is what kills the freeze) ──
    insight_block(
        f"{len(charts)} charts available. Charts are built once and cached — "
        "switching between them is instant. Each chart ships to the browser only "
        "when selected, so the page stays responsive.",
        label="How to read this stage", color="green",
    )

    titles = [c["title"] for c in charts]
    pick = st.selectbox("Select chart", titles, key="eda_chart_pick")
    ch = next(c for c in charts if c["title"] == pick)
    pf(ch["fig"])
    chart_rationale(ch["why_this"], ch["alternatives"], ch["question"])

    with st.expander("Descriptive Statistics Table (df.describe())"):
        nd = df_eda.select_dtypes(include=[np.number])
        if not nd.empty:
            d = nd.describe().round(3).T
            d["skew"]     = nd.skew().round(3)
            d["kurtosis"] = nd.kurtosis().round(3)
            st.dataframe(d, width="stretch")
        else:
            st.info("No numeric columns to describe.")

    # ── Ask DataGPT — multi-turn chat grounded in the dataset ──
    st.markdown("---")
    st.markdown("#### Ask DataGPT About the Data")

    if not st.session_state["groq_ok"]:
        st.info("Add your Groq API key in the sidebar to chat with the data.")
        return

    # Compact data context the model gets every turn (kept small on purpose).
    num_cols = df_eda.select_dtypes(include=[np.number]).columns.tolist()
    desc_ctx = (
        df_eda[num_cols[:8]].describe().round(3).to_dict() if num_cols else {}
    )
    data_ctx = (
        f"\n\nDATASET CONTEXT (answer only from this):\n"
        f"- Shape: {df_eda.shape[0]:,} rows × {df_eda.shape[1]} columns\n"
        f"- Dtypes: {df_eda.dtypes.astype(str).to_dict()}\n"
        f"- Numeric summary: {desc_ctx}\n"
        f"- Scope questions: {st.session_state['questions']}\n"
        "If the data cannot answer the question, say so explicitly."
    )

    # Render history
    for msg in st.session_state["chat_history"]:
        cls = "cu" if msg["role"] == "user" else "ca"
        st.markdown(f"<div class='{cls}'>{msg['content']}</div>", unsafe_allow_html=True)

    q = st.text_input(
        "Ask a question about the data",
        placeholder="e.g. Which column is most correlated with revenue, and how strongly?",
        key="eda_chat_input",
        label_visibility="collapsed",
    )
    ca, cb = st.columns([1, 5])
    with ca:
        ask = st.button("Ask")
    with cb:
        if st.session_state["chat_history"] and st.button("Clear chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    if ask and q.strip():
        st.session_state["chat_history"].append({"role": "user", "content": q.strip()})
        with st.spinner("Thinking…"):
            try:
                reply = call_llm(
                    groq_client, model, q.strip(),
                    system=SYSTEM_PROMPT + data_ctx,
                    history=st.session_state["chat_history"][:-1][-6:],
                    max_tokens=900,
                )
            except ValueError as e:
                reply = f"⚠ {e}"
        st.session_state["chat_history"].append({"role": "assistant", "content": reply})
        st.rerun()
