# ============================ UI · STYLES ============================
# The full dark theme. Kept as one string + an inject_css() call so app.py
# stays clean. The CSS itself is unchanged from the original single-file app.

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
  --bg:       #0e1117;  --surf:     #161b27;  --surf2:    #1c2333;
  --border:   #262d3d;  --border2:  #38445c;
  --text:     #cdd5ef;  --dim:      #68778f;  --dimmer:   #3a4355;
  --blue:     #4d8ef5;  --blue-bg:  #162040;
  --green:    #3ec98a;  --green-bg: #0c3020;
  --amber:    #f0a020;  --amber-bg: #3a2800;
  --red:      #ef5e5e;  --red-bg:   #3a1010;
  --mono: 'IBM Plex Mono', monospace;
  --sans: 'IBM Plex Sans', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: var(--sans); color: var(--text);
}
[data-testid="stSidebar"] {
  background: var(--surf) !important;
  border-right: 1px solid var(--border);
}

/* ── wordmark ── */
.wm { padding:1.2rem 0 1rem; border-bottom:1px solid var(--border); margin-bottom:1rem; }
.wm-title { font-family:var(--mono); font-size:1.3rem; font-weight:600; color:var(--text); }
.wm-title span { color:var(--blue); }
.wm-sub { font-size:0.62rem; color:var(--dimmer); text-transform:uppercase; letter-spacing:2.5px; margin-top:3px; }

/* ── page header ── */
.ph { border-bottom:1px solid var(--border); padding-bottom:0.9rem; margin-bottom:1.4rem; }
.ph-pill { font-family:var(--mono); font-size:0.62rem; font-weight:600; color:var(--blue);
           background:var(--blue-bg); border:1px solid var(--blue); border-radius:3px;
           padding:2px 9px; letter-spacing:1.5px; text-transform:uppercase;
           display:inline-block; margin-bottom:6px; }
.ph h1  { font-family:var(--mono); font-size:1.4rem; font-weight:600; color:var(--text); margin:0; }
.ph-sub { font-size:0.8rem; color:var(--dim); margin-top:3px; }

/* ── insight block ── */
.ins { background:var(--surf); border:1px solid var(--border); border-left:3px solid var(--blue);
       border-radius:0 5px 5px 0; padding:0.9rem 1.1rem; margin:0.7rem 0;
       font-size:0.84rem; line-height:1.75; color:var(--text); }
.ins .lbl { font-family:var(--mono); font-size:0.6rem; color:var(--blue);
            letter-spacing:2px; text-transform:uppercase; margin-bottom:0.4rem; }

/* ── chart rationale ── */
.rat { background:var(--surf2); border:1px solid var(--border); border-radius:5px;
       padding:0.75rem 1rem; margin-top:0.25rem; font-size:0.78rem;
       line-height:1.65; color:var(--dim); }
.rat .lbl { font-family:var(--mono); font-size:0.58rem; color:var(--amber);
            letter-spacing:2px; text-transform:uppercase; margin-bottom:0.3rem; }
.rat b { color:var(--text); font-weight:500; }

/* ── dataset info card ── */
.dinfo { background:var(--surf2); border:1px solid var(--border); border-radius:6px;
         padding:1rem 1.2rem; margin:0.6rem 0; font-size:0.82rem; line-height:1.7; }
.dinfo .lbl { font-family:var(--mono); font-size:0.58rem; color:var(--blue);
              letter-spacing:2px; text-transform:uppercase; margin-bottom:0.35rem; }

/* ── quality dimension badges ── */
.qdim { display:inline-block; font-family:var(--mono); font-size:0.68rem;
        padding:3px 9px; border-radius:3px; margin:3px 3px 3px 0; font-weight:600; }
.qd-completeness { background:#1a2a40; color:#5ba4f5; border:1px solid #2d5fa0; }
.qd-validity      { background:#2a1a10; color:#f0a020; border:1px solid #a06010; }
.qd-accuracy      { background:#2a1010; color:#ef5e5e; border:1px solid #a03030; }
.qd-consistency   { background:#0f2a1a; color:#3ec98a; border:1px solid #1a7040; }
.qd-tidiness      { background:#1e1a30; color:#a07af5; border:1px solid #604fa0; }

/* ── stat card ── */
.sc { background:var(--surf); border:1px solid var(--border); border-radius:6px; padding:0.9rem 1rem; }
.sc .v { font-family:var(--mono); font-size:1.55rem; font-weight:600; color:var(--text); line-height:1; margin-bottom:3px; }
.sc .v.ok   { color:var(--green); }  .sc .v.warn { color:var(--amber); }  .sc .v.bad { color:var(--red); }
.sc .l { font-size:0.65rem; color:var(--dimmer); text-transform:uppercase; letter-spacing:1.5px; }

/* ── inline tags ── */
.tag { display:inline-block; font-family:var(--mono); font-size:0.68rem;
       padding:2px 7px; border-radius:3px; margin:2px 3px 2px 0; }
.tg { background:var(--green-bg); color:var(--green); border:1px solid var(--green); }
.ta { background:var(--amber-bg); color:var(--amber); border:1px solid var(--amber); }
.tr { background:var(--red-bg);   color:var(--red);   border:1px solid var(--red); }
.tb { background:var(--blue-bg);  color:var(--blue);  border:1px solid var(--blue); }

/* ── cleaning step row ── */
.sr { display:flex; align-items:flex-start; gap:0.6rem; padding:0.5rem 0;
      border-bottom:1px solid var(--border); font-size:0.8rem; }
.sn { font-family:var(--mono); font-size:0.65rem; color:var(--dimmer); min-width:26px; margin-top:1px; }

/* ── scope question card ── */
.qc { background:var(--surf2); border:1px solid var(--border); border-radius:4px;
      padding:0.55rem 0.85rem; margin:0.25rem 0; font-size:0.82rem; line-height:1.5; }
.qn { font-family:var(--mono); font-size:0.65rem; color:var(--blue); margin-right:0.45rem; }

/* ── chat ── */
.cu { background:var(--surf2); border:1px solid var(--border); border-radius:5px 5px 0 5px;
      padding:0.65rem 0.9rem; margin:0.45rem 0 0.45rem 2rem; font-size:0.83rem; }
.ca { background:var(--surf); border:1px solid var(--border); border-left:3px solid var(--blue);
      border-radius:0 5px 5px 5px; padding:0.65rem 0.9rem;
      margin:0.45rem 2rem 0.45rem 0; font-size:0.83rem; line-height:1.7; }

/* ── report ── */
.ra { background:var(--surf); border:1px solid var(--border); border-radius:5px;
      padding:1.4rem 1.8rem; font-size:0.83rem; line-height:1.8;
      color:var(--text); white-space:pre-wrap; word-break:break-word; }

/* ── buttons ── */
.stButton > button {
  background:var(--surf2) !important; color:var(--text) !important;
  font-family:var(--mono) !important; font-size:0.76rem !important;
  font-weight:600 !important; border:1px solid var(--border2) !important;
  border-radius:4px !important; padding:0.45rem 1.3rem !important;
  transition:border-color 0.2s,color 0.2s !important;
}
.stButton > button:hover { border-color:var(--blue) !important; color:var(--blue) !important; }

/* ── inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background:var(--surf2) !important; color:var(--text) !important;
  border-color:var(--border) !important; font-family:var(--sans) !important;
  font-size:0.84rem !important;
}
.stSelectbox > div > div { background:var(--surf2) !important; color:var(--text) !important; }

/* ── tabs ── */
[data-testid="stTabs"] [role="tab"] {
  font-family:var(--mono) !important; font-size:0.73rem !important; color:var(--dim) !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color:var(--blue) !important; border-bottom-color:var(--blue) !important;
}

/* ── expander ── */
details > summary {
  font-family:var(--mono) !important; font-size:0.76rem !important;
  color:var(--dim) !important; background:var(--surf) !important;
  border:1px solid var(--border) !important; border-radius:4px !important;
  padding:0.45rem 0.75rem !important;
}

code { font-family:var(--mono) !important; background:var(--surf2) !important;
       color:var(--blue) !important; padding:1px 5px !important;
       border-radius:3px !important; font-size:0.82em !important; }
hr   { border-color:var(--border) !important; margin:1.1rem 0 !important; }
.stDataFrame { border-radius:5px; overflow:hidden; }
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track  { background:var(--bg); }
::-webkit-scrollbar-thumb  { background:var(--border2); border-radius:3px; }
</style>
"""


def inject_css():
    """Inject the global theme. Call once, right after st.set_page_config."""
    st.markdown(CSS, unsafe_allow_html=True)
