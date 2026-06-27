# ============================ CORE · PROFILING ============================
# Date-convention detection / parsing, plus the full structural audit that
# powers Stage 02 (assessment) and Stage 03 (cleaning suggestions).

import warnings as _warnings
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# DATE PARSING  (column-level convention detection — no silent guessing)
# ─────────────────────────────────────────────────────────────────────────────

def _detect_date_convention(series: pd.Series) -> Optional[str]:
    """
    Decide day-first vs month-first at the COLUMN level so the whole column is
    parsed consistently — instead of letting each value match the first format
    that happens to succeed (which silently mis-parses ambiguous US dates).

    Logic:
      - A value position that ever exceeds 12 cannot be a month.
      - If the first slot exceeds 12 anywhere → first slot is the day → 'dmy'.
      - If the second slot exceeds 12 anywhere → second slot is the day → 'mdy'.
      - If both happen → contradictory → ambiguous (None).
      - If neither happens (every value <= 12/12) → undeterminable (None).

    Returns: 'dmy', 'mdy', or None.
    """
    parts = (
        series.dropna().astype(str).str.strip()
        .str.extract(r"^(\d{1,2})[-/](\d{1,2})[-/]\d{2,4}")
        .dropna()
    )
    if parts.empty:
        return None
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    first_gt12 = bool((first > 12).any())
    second_gt12 = bool((second > 12).any())
    if first_gt12 and second_gt12:
        return None
    if first_gt12:
        return "dmy"
    if second_gt12:
        return "mdy"
    return None


def _smart_date_parse(series: pd.Series) -> tuple:
    """
    Parse a date-like object column to datetime without silently guessing
    day/month order.

    1. Detect the column convention (dmy/mdy) from any value where one slot
       exceeds 12.
    2. Try the convention-appropriate slash/dash formats first, then a list of
       unambiguous formats (ISO, named months, compact).
    3. If the convention is undeterminable, fall back to dayfirst=True but
       FLAG it — the result may be wrong for genuinely US-format data.

    Returns: (parsed_series, note) where `note` describes the convention used
    or the assumption made (None if nothing noteworthy).
    """
    convention = _detect_date_convention(series)

    if convention == "dmy":
        primary = ["%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"]
        note = "day-first order detected from the data (a day value exceeded 12)"
    elif convention == "mdy":
        primary = ["%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y"]
        note = "month-first order detected from the data (a day value exceeded 12)"
    else:
        primary = []
        note = None

    # Unambiguous formats — order does not matter.
    unambiguous = [
        "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d",
        "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    ]
    formats = primary + unambiguous

    str_series = series.astype(str).where(series.notna(), other=None)
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    for fmt in formats:
        still_nat = result.isna() & series.notna()
        if not still_nat.any():
            break
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            parsed = pd.to_datetime(str_series[still_nat], format=fmt, errors="coerce")
        result.update(parsed[parsed.notna()])

    # Ambiguous slash/dash dates no unambiguous format caught.
    still_nat = result.isna() & series.notna()
    if still_nat.any():
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            fallback = pd.to_datetime(str_series[still_nat], dayfirst=True, errors="coerce")
        result.update(fallback[fallback.notna()])
        if convention is None:
            note = ("ORDER COULD NOT BE DETERMINED from the data — assumed "
                    "day-first; verify manually if this is US-format data")

    return result, note


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL AUDIT
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, ttl=3600, max_entries=8)
def assess_data(df: pd.DataFrame) -> dict:
    """
    Full structural audit across the five data-quality dimensions.

    Checks: shape, dtypes, missing (Completeness), duplicates (Accuracy),
    cardinality, IQR outliers (Accuracy), skewness (Validity), date-like and
    numeric-like strings (Validity), low-cardinality → category, semantic range
    violations (Accuracy).

    Returns dict with keys: shape, dtypes, missing, missing_pct,
    duplicate_count, cardinality, outliers, skewness, suggestions.
    """
    a = {}
    a["shape"]   = df.shape
    a["dtypes"]  = df.dtypes.astype(str).to_dict()

    # ── Missing (Completeness) ──
    miss = df.isnull().sum()
    a["missing"]     = miss[miss > 0].to_dict()
    a["missing_pct"] = {k: round(v / len(df) * 100, 2)
                        for k, v in a["missing"].items()}

    # ── Duplicates ──
    a["duplicate_count"] = int(df.duplicated().sum())

    # ── Cardinality ──
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    a["cardinality"] = {c: int(df[c].nunique()) for c in obj_cols}

    # ── Numeric analysis ──
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # IQR outliers (Accuracy)
    outliers = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        n = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if n:
            outliers[c] = n
    a["outliers"] = outliers

    # Skewness (Validity)
    skew = {}
    for c in num_cols:
        s = df[c].dropna()
        if len(s) >= 10:
            sk = float(s.skew())
            if abs(sk) > 1.0:
                skew[c] = round(sk, 3)
    a["skewness"] = skew

    # ── Build suggestions ──
    suggestions = []
    already = set()

    # 1. Date-like strings → datetime  [Validity]
    for c in obj_cols:
        sample = df[c].dropna().head(60).astype(str)
        if sample.empty:
            continue
        rate = sample.str.match(
            r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})|(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"
        ).mean()
        if rate > 0.6:
            suggestions.append({
                "type": "to_datetime", "col": c, "dim": "Validity",
                "reason": (
                    f"'{c}' is object dtype but {int(rate*100)}% of values "
                    "match date patterns. Text dtype prevents sorting, groupby, "
                    "and date-part extraction. Day/month order is detected at the "
                    "column level before parsing — if it cannot be determined, "
                    "the cleaning log will flag the assumption."
                ),
            })
            already.add(c)

    # 2. Numeric stored as string  [Validity]
    # Handles ₹/$/€/£, thousands separators, %, and accounting negatives (1,234).
    for c in obj_cols:
        if c in already:
            continue
        raw_sample = df[c].dropna().head(120).astype(str).str.strip()
        if raw_sample.empty:
            continue
        test = raw_sample.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
        test = test.str.replace(r"[₹$€£,%\s]", "", regex=True)
        rate = test.str.match(r"^-?\d+(\.\d+)?$").mean()
        if rate > 0.8:
            is_pct = bool(raw_sample.str.contains("%").mean() > 0.5)
            pct_note = (
                " Values carry '%', so they will be divided by 100 on conversion "
                "(e.g. 45% → 0.45)."
                if is_pct else ""
            )
            suggestions.append({
                "type": "to_numeric", "col": c, "dim": "Validity",
                "is_percent": is_pct,
                "reason": (
                    f"'{c}' is object dtype but {int(rate*100)}% of values are "
                    "numeric strings (after stripping ₹/$/£/% and separators, and "
                    "treating (x) as -x). Numeric dtype unlocks descriptive stats "
                    f"and correlation.{pct_note}"
                ),
            })
            already.add(c)

    # 3. Low-cardinality object → category  [Validity / memory]
    for c, uniq in a["cardinality"].items():
        if c in already:
            continue
        if uniq <= 25 and len(df) >= 50:
            suggestions.append({
                "type": "to_category", "col": c, "dim": "Validity",
                "reason": (
                    f"'{c}' has {uniq} unique values in {len(df):,} rows "
                    f"({round(uniq/len(df)*100,1)}% cardinality). "
                    "category dtype cuts memory by up to 5× and speeds up groupby."
                ),
            })

    # 4. Duplicates  [Accuracy]
    if a["duplicate_count"]:
        suggestions.append({
            "type": "drop_duplicates", "col": "_all_", "dim": "Accuracy",
            "reason": (
                f"{a['duplicate_count']:,} fully duplicate rows inflate row counts, "
                "skew frequency distributions, and bias any model trained on this data."
            ),
        })

    # 5. High-null columns (>50%) → drop, else impute  [Completeness]
    for c, pct in a["missing_pct"].items():
        if pct > 50:
            suggestions.append({
                "type": "drop_col", "col": c, "dim": "Completeness",
                "reason": (
                    f"'{c}' is {pct}% null. Imputing >50% introduces more synthetic "
                    "bias than the column removes. Dropping is the correct decision."
                ),
            })
        elif pct > 0:
            dtype = str(df[c].dtype)
            fill = "median" if ("int" in dtype or "float" in dtype) else "mode"
            extra = (
                f"Median preferred over mean because '{c}' also has "
                f"{outliers.get(c,0)} IQR outliers — median is outlier-robust."
                if fill == "median" and c in outliers else
                "Median preferred over mean — robust to skew." if fill == "median"
                else "Mode used for text/category — preserves modal class."
            )
            suggestions.append({
                "type": f"fill_{fill}", "col": c, "dim": "Completeness",
                "reason": f"'{c}' is {pct}% null. {extra}",
            })

    # 6. Semantic range violations  [Accuracy]
    NON_NEG = ["age", "price", "cost", "count", "qty", "quantity",
               "revenue", "sales", "salary", "amount", "weight",
               "height", "duration", "distance", "volume", "units"]
    for c in num_cols:
        if any(h in c.lower() for h in NON_NEG):
            n_neg = int((df[c] < 0).sum())
            if n_neg:
                suggestions.append({
                    "type": "range_flag", "col": c, "dim": "Accuracy",
                    "reason": (
                        f"'{c}' has {n_neg} negative values. "
                        "Domain definition says this column must be non-negative — "
                        "likely data entry errors or sign-convention bugs."
                    ),
                })

    a["suggestions"] = suggestions
    return a
