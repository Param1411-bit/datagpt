# ============================ UI · STATE ============================
# Session-state initialisation + a single "data version" counter that drives
# cache invalidation for the EDA charts and the column-summary table.
#
# The version is bumped whenever df_clean changes (load, clean, reset). Anything
# expensive that depends on the data stores its own "<thing>_version" alongside
# its cached result and rebuilds only when the two differ — so per-rerun work
# (typing in a box, clicking a button) does NOT re-run those scans.

import streamlit as st

from config import STAGES


def init_state():
    """Initialise all session-state keys once with safe defaults."""
    defaults = {
        "df_original":   None,   # raw frame — never modified
        "df_clean":      None,   # cleaned frame — used everywhere downstream
        "filename":      "",     # uploaded filename for display
        "questions":     [],     # scope questions
        "cleaning_log":  [],     # list[str] of applied steps
        "suggestions":   [],     # list[dict] from assess_data()
        "assessment":    {},     # full assessment dict
        "conclusions":   "",     # AI-generated conclusions text
        "stress_test":   "",     # stress-test output
        "case_report":   "",     # final markdown report
        "chat_history":  [],     # EDA chat messages
        "stage":         STAGES[0],
        "groq_ok":       False,

        # ── cache-invalidation bookkeeping ──
        "data_version":        0,      # bumped on every df_clean change
        "missing_pct_overall": 0.0,    # sidebar metric, recomputed on bump only
        "eda_charts":          [],     # cached chart list
        "eda_charts_version":  None,   # version the cached charts were built for
        "colsummary_df":       None,   # cached column-description table
        "colsummary_version":  None,   # version the cached table was built for
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def current_version() -> int:
    """The active data version. Compare against a cached '<x>_version' to decide rebuilds."""
    return st.session_state.get("data_version", 0)


def bump_data_version():
    """Call after any change to df_clean so dependent caches rebuild next render."""
    st.session_state["data_version"] = st.session_state.get("data_version", 0) + 1
