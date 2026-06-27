# ============================ CORE · TRANSFORMS ============================
# Apply the cleaning suggestions produced by profiling.assess_data. Each step
# is independent (its own try/except) so one failure never blocks the rest.

import pandas as pd

from core.profiling import _smart_date_parse


def apply_cleaning(df: pd.DataFrame, suggestions: list) -> tuple:
    """
    Apply all queued suggestions.

    Returns: (cleaned_df, log) — cleaned DataFrame and list[str] of steps.
    """
    df  = df.copy()
    log = []

    for sug in suggestions:
        t, col = sug["type"], sug["col"]
        try:
            if t == "drop_duplicates":
                before = len(df)
                df     = df.drop_duplicates()
                log.append(f"Removed {before - len(df):,} duplicate rows. [Accuracy]")

            elif t == "drop_col" and col in df.columns:
                df.drop(columns=[col], inplace=True)
                if sug.get("dim") == "Tidiness":
                    log.append(f"Dropped '{col}' (constant / zero-variance). [Tidiness]")
                else:
                    log.append(f"Dropped '{col}' (>50% null). [Completeness]")

            elif t == "to_datetime" and col in df.columns:
                before_nulls = int(df[col].isna().sum())
                parsed, conv_note = _smart_date_parse(df[col])
                df[col] = parsed
                after_nulls = int(df[col].isna().sum())
                new_nats = max(0, after_nulls - before_nulls)

                bits = [f"'{col}' → datetime64"]
                if conv_note:
                    bits.append(conv_note)
                if new_nats:
                    bits.append(
                        f"⚠ {new_nats} value(s) unparseable in any known format → "
                        "NaT (inspect those rows manually)"
                    )
                else:
                    bits.append("0 new NaTs created")
                log.append(" · ".join(bits) + ". [Validity]")

            elif t == "to_numeric" and col in df.columns:
                raw = df[col].astype(str).str.strip()
                # Flag the values that actually carry a '%' BEFORE stripping, so
                # only those get divided by 100. A mixed column ('45%' next to an
                # already-decimal '0.45') is no longer uniformly corrupted.
                pct_mask = raw.str.contains("%", na=False)
                s = raw.str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # accounting negatives
                s = s.str.replace(r"[₹$€£,%\s]", "", regex=True)
                out = pd.to_numeric(s, errors="coerce")
                if pct_mask.any():
                    # keep where NOT percent; divide only the percent rows by 100
                    out = out.where(~pct_mask, out / 100.0)
                    df[col] = out
                    n_pct = int(pct_mask.sum())
                    log.append(
                        f"'{col}' → numeric; {n_pct} '%' value(s) divided by 100 "
                        "(e.g. 45% → 0.45); non-% values left as-is. [Validity]"
                    )
                else:
                    df[col] = out
                    log.append(f"'{col}' → numeric (symbols/separators stripped). [Validity]")

            elif t == "to_category" and col in df.columns:
                df[col] = df[col].astype("category")
                log.append(f"'{col}' → category dtype. [Validity]")

            elif t == "fill_median" and col in df.columns:
                if df[col].isnull().any():
                    med = df[col].median()
                    df[col] = df[col].fillna(med)
                    log.append(f"'{col}' nulls → median ({med:.4g}). [Completeness]")

            elif t == "fill_mode" and col in df.columns:
                if df[col].isnull().any():
                    mode_s = df[col].mode()
                    if not mode_s.empty:
                        # categorical fillna requires the value to be an existing
                        # category — mode always is, so this is safe.
                        df[col] = df[col].fillna(mode_s.iloc[0])
                        log.append(
                            f"'{col}' nulls → mode ('{mode_s.iloc[0]}'). [Completeness]"
                        )

            elif t == "fix_range" and col in df.columns:
                lo    = sug.get("lo")
                hi    = sug.get("hi")
                sents = sug.get("sentinels", [])
                s = pd.to_numeric(df[col], errors="coerce")

                bad = pd.Series(False, index=s.index)
                if lo is not None:
                    bad |= s < lo
                if hi is not None:
                    bad |= s > hi
                if sents:
                    bad |= s.isin(sents)
                n_bad = int(bad.sum())

                s = s.mask(bad)                 # impossible values → NaN
                med = s.median()                # median of the VALID values only
                if pd.isna(med):
                    log.append(
                        f"⚠ '{col}': all values out of range — cannot impute, left as NaN. [Accuracy]"
                    )
                    df[col] = s
                else:
                    s = s.fillna(med)
                    if sug.get("to_int"):
                        s = s.round().astype("int64")
                        df[col] = s
                        log.append(
                            f"'{col}': {n_bad} impossible value(s) → NaN → median "
                            f"({med:.4g}); rounded to int. [Accuracy]"
                        )
                    else:
                        df[col] = s
                        log.append(
                            f"'{col}': {n_bad} impossible value(s) → NaN → median "
                            f"({med:.4g}). [Accuracy]"
                        )

            elif t == "to_integer" and col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                if s.notna().all():
                    df[col] = s.round().astype("int64")
                    log.append(f"'{col}' → int64 (was float; all values whole). [Validity]")
                else:
                    # keep NaNs intact with pandas' nullable integer dtype
                    df[col] = s.round().astype("Int64")
                    log.append(f"'{col}' → integer (nullable Int64, NaNs preserved). [Validity]")

            elif t == "range_flag":
                log.append(
                    f"⚠ '{col}' has out-of-range values — flagged, manual review needed. [Accuracy]"
                )

        except Exception as exc:
            log.append(f"⚠ Skipped '{col}' ({t}): {exc}")

    return df, log
