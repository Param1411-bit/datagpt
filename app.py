# ============================ DataGPT · APP ENTRY ============================
# Single entry point. Keep this thin: it wires modules together and routes to
# the selected stage. All real work lives in core/ , ui/ and stages/ .
#
# Run from this directory:   streamlit run app.py
# (Streamlit adds this folder to sys.path, so `from core...`, `from ui...`,
#  `from stages...` resolve. Each subpackage has an empty __init__.py.)

import streamlit as st

# set_page_config MUST bstree the first Streamlit call — before any other st.* runs.
st.set_page_config(
    page_title="DataGPT — AI Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import STAGES                       # noqa: E402
from ui.styles import inject_css                # noqa: E402
from ui.state import init_state                 # noqa: E402
from ui.sidebar import render_sidebar           # noqa: E402
from stages import (                            # noqa: E402
    scope, load_assess, clean, eda, conclusions, report,
)

inject_css()
init_state()

groq_client, model, stage = render_sidebar()

# ── Stage router ──
if stage == STAGES[0]:
    scope.render(groq_client, model)
elif stage == STAGES[1]:
    load_assess.render(groq_client, model)
elif stage == STAGES[2]:
    clean.render(groq_client, model)
elif stage == STAGES[3]:
    eda.render(groq_client, model)
elif stage == STAGES[4]:
    conclusions.render(groq_client, model)
elif stage == STAGES[5]:
    report.render(groq_client, model)
