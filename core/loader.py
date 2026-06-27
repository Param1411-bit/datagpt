# ============================ CORE · LOADER ============================
# Turn uploaded bytes into a DataFrame, with guards so an oversized file can't
# OOM the hosted container before any analysis runs.

import io

import pandas as pd
import streamlit as st

from config import MAX_FILE_MB, MAX_ROWS


@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse uploaded file bytes into a DataFrame.

    - Rejects files over MAX_FILE_MB before parsing.
    - CSV with encoding fallback (utf-8 → latin-1 → cp1252).
    - Excel via openpyxl.
    - Rejects frames over MAX_ROWS after parsing.

    Raises ValueError with a user-readable message on failure.
    """
    size_mb = len(file_bytes) / 1024 ** 2
    if size_mb > MAX_FILE_MB:
        raise ValueError(
            f"File is {size_mb:.0f} MB — over the {MAX_FILE_MB} MB cap for this "
            "hosted demo. Sample the data down, or run locally to raise the cap."
        )

    ext = filename.rsplit(".", 1)[-1].lower()
    if ext == "csv":
        df = None
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Cannot decode CSV — try re-saving with UTF-8 encoding.")
    elif ext in ("xlsx", "xls"):
        try:
            df = pd.read_excel(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Excel parse error: {e}") from e
    else:
        raise ValueError(f"Unsupported format '.{ext}'. Upload CSV or Excel.")

    if len(df) > MAX_ROWS:
        raise ValueError(
            f"{len(df):,} rows — over the {MAX_ROWS:,} row cap for this hosted "
            "demo. Sample the data down, or run locally to raise the cap."
        )
    return df
