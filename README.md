# DataGPT — AI Data Analyst

A 6-stage analyst workflow (Scope → Load/Assess → Clean → EDA → Conclusions → Report)
built on Streamlit + Groq.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Groq key
Either paste it in the sidebar, or set it once so you never retype it:

- **Local:** `export GROQ_API_KEY="gsk_..."` before `streamlit run`
- **Streamlit Cloud:** add `GROQ_API_KEY="gsk_..."` in *App → Settings → Secrets*

## Project layout
```
datagpt/
├── app.py              # entry point: page config, CSS, sidebar, stage router
├── config.py           # all tunable constants (caps, stage list, step icons)
├── core/               # logic — no Streamlit UI lives here except caching
│   ├── llm.py          # Groq client + model list + chat call
│   ├── loader.py       # file upload → DataFrame (with size/row guards)
│   ├── profiling.py    # date parsing + full data-quality audit
│   ├── transforms.py   # apply queued cleaning steps
│   └── charts.py       # EDA chart builder (samples large frames)
├── ui/                 # presentation helpers
│   ├── styles.py       # the CSS theme
│   ├── state.py        # session-state init + data-version tracking
│   ├── components.py   # reusable render blocks (cards, headers, rationale)
│   └── sidebar.py      # sidebar + key/model/stage selection
└── stages/             # one module per workflow stage, each exposes render()
    ├── scope.py        ├── clean.py        ├── conclusions.py
    ├── load_assess.py  ├── eda.py          └── report.py
```

## Performance notes (why it no longer freezes)
- EDA renders **one chart at a time** via a selector, not all ~13 at once.
- Charts are built **once per dataset version** and cached in session state;
  changing the data (load / clean / reset) bumps the version to rebuild.
- Large frames are **sampled for plotting** (`SAMPLE_ROWS_FOR_CHARTS`) while
  statistics/correlations stay on the full frame.
- Sidebar + column-summary scans run **once per version**, not every rerun.
