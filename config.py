# ============================ CONFIG ============================
# Single source of truth for app-wide constants. Mirrors the AxT bot's CONFIG
# block — everything tunable lives here, not scattered through the logic.

# ---- Upload guards ----------------------------------------------------------
# Streamlit Community Cloud gives ~1 GB RAM and pandas needs several× the file
# size in memory, so reject oversized inputs before the full-frame passes run.
MAX_FILE_MB = 50
MAX_ROWS = 200_000               # lowered from 500k so a 1 GB box won't OOM
SAMPLE_ROWS_FOR_CHARTS = 50_000  # plot a sample above this; stats stay full-frame

# ---- Workflow stages (sidebar radio order) ---------------------------------
STAGES = [
    "01 · Define Scope",
    "02 · Load & Assess",
    "03 · Clean Data",
    "04 · Explore (EDA)",
    "05 · Conclusions",
    "06 · Report",
]

# ---- Cleaning-step icons (used in Stage 02 + Stage 03) ----------------------
STEP_ICONS = {
    "to_datetime":     ("🔄", "Convert → datetime"),
    "to_numeric":      ("🔄", "Convert → numeric"),
    "to_category":     ("🔄", "Convert → category"),
    "drop_duplicates": ("🗑",  "Remove duplicates"),
    "drop_col":        ("❌", "Drop column"),
    "fill_median":     ("🩹", "Impute with median"),
    "fill_mode":       ("🩹", "Impute with mode"),
    "range_flag":      ("⚠",  "Range violation — manual review"),
}
