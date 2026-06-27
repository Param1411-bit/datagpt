# ============================ STAGE 01 · DEFINE SCOPE ============================
import streamlit as st

from core.llm import call_llm
from ui.components import insight_block, page_header


def render(groq_client, model):
    page_header(
        "STAGE 01", "Define Scope",
        "Write the scope questions before touching the data.",
    )

    insight_block(
        "A real analyst defines what they are trying to find out <b>before</b> loading the data. "
        "This prevents post-hoc fishing (p-hacking). "
        "Every chart and conclusion produced later is evaluated against these questions.",
        label="Why This Step Matters", color="amber",
    )

    st.markdown("""
    <div class='dinfo'>
      <div class='lbl'>Scoping Framework — five questions to answer first</div>
      <b>1.</b> What decision does this analysis need to inform?<br>
      <b>2.</b> Who is the audience, and what do they already believe?<br>
      <b>3.</b> What data is available?<br>
      <b>4.</b> What are the likely follow-up questions or objections?<br>
      <b>5.</b> What similar past analyses or data points do we have?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns([3, 2])

    with c1:
        st.markdown("**Add a Scope Question**")
        new_q = st.text_area(
            "Scope question input",
            placeholder=(
                "Examples:\n"
                "• What drives customer churn in this dataset?\n"
                "• Is there seasonality in monthly sales?\n"
                "• Which product category has the highest return rate?"
            ),
            height=105,
            label_visibility="collapsed",
        )
        if st.button("Add Question"):
            q = new_q.strip()
            if q and q not in st.session_state["questions"]:
                st.session_state["questions"].append(q)
                st.rerun()
            elif not q:
                st.warning("Enter a question first.")
            else:
                st.info("Already in scope list.")

    with c2:
        st.markdown("**Scope List**")
        if st.session_state["questions"]:
            for i, q in enumerate(st.session_state["questions"], 1):
                st.markdown(
                    f"<div class='qc'><span class='qn'>Q{i}.</span>{q}</div>",
                    unsafe_allow_html=True,
                )
            if st.button("Clear All"):
                st.session_state["questions"] = []
                st.rerun()
        else:
            st.markdown(
                "<p style='color:var(--dimmer);font-size:0.82rem;'>No questions yet.</p>",
                unsafe_allow_html=True,
            )

    if st.session_state["questions"] and st.session_state["groq_ok"]:
        st.markdown("---")
        if st.button("AI Scope Review"):
            qs_text = "\n".join(
                f"{i+1}. {q}" for i, q in enumerate(st.session_state["questions"])
            )
            prompt = (
                f"Analyst scope questions:\n{qs_text}\n\n"
                "For each question reply with:\n"
                "1. What columns/data types are needed to answer it?\n"
                "2. Most appropriate statistical method or chart type.\n"
                "3. Is it too vague? If yes, rewrite precisely.\n"
                "3 sentences per question max. Be direct."
            )
            with st.spinner("Reviewing scope…"):
                try:
                    reply = call_llm(groq_client, model, prompt, max_tokens=800)
                    insight_block(reply, label="DataGPT · Scope Review")
                except ValueError as e:
                    st.error(str(e))
    elif st.session_state["questions"]:
        st.info("Add your Groq API key in the sidebar to get an AI scope review.")
