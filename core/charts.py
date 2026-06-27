# ============================ CORE · CHARTS ============================
# Build the standard EDA chart set, each with an analyst rationale.
#
# PERFORMANCE: this is intentionally NOT decorated with @st.cache_data. The EDA
# stage calls it once per dataset *version* and stores the result in session
# state, so caching here would just add redundant DataFrame hashing on rerun.
#
# Large frames are plotted on a sample (SAMPLE_ROWS_FOR_CHARTS) so the browser
# isn't asked to render hundreds of thousands of marks — but correlations and
# value counts are computed on the FULL frame for accuracy.

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import SAMPLE_ROWS_FOR_CHARTS


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY BASE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
# PBASE holds only paper/plot/font/margin so update_layout() never receives
# duplicate xaxis/yaxis kwargs when px/go already set them. GRID is merged
# per-chart where axis styling is needed.
PBASE = dict(
    paper_bgcolor="#161b27",
    plot_bgcolor="#161b27",
    font=dict(family="IBM Plex Sans", color="#cdd5ef", size=11),
    margin=dict(l=44, r=18, t=42, b=38),
)
GRID = dict(gridcolor="#262d3d", linecolor="#262d3d", zerolinecolor="#262d3d")


def _layout(fig: go.Figure, title: str, xt: str = "", yt: str = "",
            extra_x: dict = None, extra_y: dict = None):
    """Apply consistent dark theme to any Plotly figure without duplicate-kwarg errors."""
    xd = {**GRID, **(extra_x or {})}
    yd = {**GRID, **(extra_y or {})}
    if xt:
        xd["title"] = xt
    if yt:
        yd["title"] = yt
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis=xd,
        yaxis=yd,
        **PBASE,
    )


def build_eda_charts(df: pd.DataFrame) -> list:
    """
    Build the standard EDA chart set with per-chart rationale.

    Guarded against degenerate data such as constant columns that produce NaN
    correlations. Plots draw on `plot_df` (a sample for large frames); stats use
    the full `df`.
    """
    charts = []

    # ── Sampling guard for the browser-rendered marks ─────────────────────
    sampled = len(df) > SAMPLE_ROWS_FOR_CHARTS
    plot_df = df.sample(SAMPLE_ROWS_FOR_CHARTS, random_state=42) if sampled else df
    sample_note = (
        f" Plotted on a {SAMPLE_ROWS_FOR_CHARTS:,}-row random sample of "
        f"{len(df):,} rows for rendering speed (shape is statistically identical)."
        if sampled else ""
    )

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime64"]).columns.tolist()

    # ── 1. Histogram + mean/median overlays ───────────────────────────────
    for col in num_cols[:5]:
        s = plot_df[col].dropna()
        if len(s) < 5:
            continue
        skew_val = round(float(s.skew()), 2)
        n_bins   = min(40, max(10, int(np.sqrt(len(s)))))

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=s, nbinsx=n_bins,
            marker_color="#4d8ef5", opacity=0.8, showlegend=False,
        ))
        fig.add_vline(x=float(s.mean()),   line_dash="dash", line_color="#f0a020",
                      annotation_text=f"mean={s.mean():.2f}",
                      annotation_position="top right")
        fig.add_vline(x=float(s.median()), line_dash="dot",  line_color="#3ec98a",
                      annotation_text=f"median={s.median():.2f}",
                      annotation_position="top left")
        _layout(fig, f"Distribution — {col}", xt=col, yt="Frequency")

        skew_note = (
            f"Skew = {skew_val:+.2f} — right-skewed. Consider log-transform before modelling."
            if skew_val > 1 else
            f"Skew = {skew_val:+.2f} — left-skewed. Check for floor/ceiling effects."
            if skew_val < -1 else
            f"Skew = {skew_val:+.2f} — approximately symmetric."
        )
        charts.append({
            "title": f"Distribution — {col}", "fig": fig,
            "why_this": (
                f"Histogram with mean (amber dashed) and median (green dotted) overlaid. "
                f"{skew_note} Gap between mean and median = skew exists. "
                "Skew determines imputation choice (median, not mean) and model assumptions."
                + sample_note
            ),
            "alternatives": (
                "Box plot shows median/IQR/outliers but hides shape (bimodality, gaps). "
                "Line chart requires ordered time data. "
                "Bar chart needs manual binning. "
                "Histogram is the mandatory first chart for any numeric column."
            ),
            "question": (
                f"What is the shape of '{col}'? Normal, skewed, or multimodal? "
                "Any data entry anomalies visible?"
            ),
        })

    # ── 2. Z-score box plot grid — outlier audit ──────────────────────────
    if len(num_cols) >= 2:
        df_z = plot_df[num_cols].copy()
        for c in num_cols:
            s = df_z[c].dropna()
            if s.std() > 0:
                df_z[c] = (df_z[c] - s.mean()) / s.std()
        df_melt = df_z.melt(var_name="Column", value_name="Z-Score")

        fig = px.box(
            df_melt.dropna(), x="Column", y="Z-Score",
            color="Column",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            points="outliers",
        )
        _layout(fig,
                "Outlier Audit — Z-Score Box Plots (all numeric columns)",
                xt="Column", yt="Z-Score (normalised)")
        fig.update_layout(showlegend=False)

        charts.append({
            "title": "Outlier Audit — Box Plots", "fig": fig,
            "why_this": (
                "Z-score normalisation puts all columns on the same axis regardless of units. "
                "Points beyond whiskers (±1.5×IQR) are individual outliers — "
                "this is the fastest cross-column outlier QA view." + sample_note
            ),
            "alternatives": (
                "Individual histograms (above) give shape but not side-by-side outlier comparison. "
                "Violin plot adds KDE — useful but heavy with many columns. "
                "Box plot is the standard tool for comparing spread and spotting outliers."
            ),
            "question": (
                "Which columns have the most outliers? "
                "How does spread compare? Are outliers symmetric or one-sided?"
            ),
        })

    # ── 3. Pearson correlation heatmap (FULL frame — cheap and exact) ──────
    if len(num_cols) >= 3:
        corr = df[num_cols].corr(numeric_only=True).round(2)
        fig  = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            text=corr.values,
            texttemplate="%{text:.2f}",
            textfont=dict(size=9),
            hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
        ))
        _layout(fig, "Pearson Correlation Matrix")

        corr_abs = corr.abs()
        # pandas Copy-on-Write makes .values read-only, so np.fill_diagonal(...)
        # raises ValueError. Mask the diagonal instead of mutating in place.
        off_diag = ~np.eye(corr_abs.shape[0], dtype=bool)
        stacked = corr_abs.where(off_diag).stack().dropna()
        if not stacked.empty:
            top_pair = stacked.idxmax()
            top_r    = round(corr.loc[top_pair[0], top_pair[1]], 3)
            pair_txt = f"Strongest pair: {top_pair[0]} vs {top_pair[1]} (r={top_r}). "
        else:
            pair_txt = "No usable correlations (constant columns produce NaN). "

        charts.append({
            "title": "Correlation Matrix", "fig": fig,
            "why_this": (
                f"Covers all {len(num_cols)*(len(num_cols)-1)//2} column pairs in one view. "
                f"{pair_txt}"
                "Blue = positive, Red = negative. Essential for spotting multicollinearity. "
                "Computed on the full frame."
            ),
            "alternatives": (
                "Scatter matrix (pair plot) shows raw points but unreadable beyond ~5 columns. "
                "Individual scatters require checking n×(n-1)/2 combinations manually. "
                "Heatmap is the only scalable correlation summary."
            ),
            "question": (
                "Which numeric columns are linearly related? "
                "Are there multicollinearity concerns for regression?"
            ),
        })

    # ── 4. Scatter for most-correlated pair (manual OLS line, no statsmodels) ─
    if len(num_cols) >= 2:
        corr2 = df[num_cols].corr(numeric_only=True).abs()
        # same CoW fix — no in-place write to a read-only .values array
        off_diag2 = ~np.eye(corr2.shape[0], dtype=bool)
        stacked2 = corr2.where(off_diag2).stack().dropna()
        if not stacked2.empty:
            pair  = stacked2.idxmax()
            c1, c2 = pair[0], pair[1]
            r_val  = round(df[[c1, c2]].corr().iloc[0, 1], 3)
            color_col = cat_cols[0] if cat_cols else None

            sub = plot_df.dropna(subset=[c1, c2])
            fig = px.scatter(
                sub, x=c1, y=c2,
                color=color_col,
                opacity=0.55,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            # Manual OLS fit via numpy — avoids the statsmodels soft-dependency
            # that px.scatter(trendline="ols") requires.
            if len(sub) >= 2 and sub[c1].nunique() > 1:
                xv = sub[c1].astype(float).to_numpy()
                yv = sub[c2].astype(float).to_numpy()
                slope, intercept = np.polyfit(xv, yv, 1)
                xs = np.linspace(float(xv.min()), float(xv.max()), 100)
                fig.add_trace(go.Scatter(
                    x=xs, y=slope * xs + intercept,
                    mode="lines", name="OLS fit",
                    line=dict(color="#f0a020", width=2),
                ))
            _layout(fig, f"Scatter — {c1} vs {c2}  (r = {r_val})", xt=c1, yt=c2)

            charts.append({
                "title": f"Scatter — {c1} vs {c2}", "fig": fig,
                "why_this": (
                    f"Scatter for the strongest correlated pair (r={r_val}, computed full-frame). "
                    "OLS trendline (amber) confirms linearity or reveals that r is "
                    "inflated by a cluster of outliers. "
                    + (f"Coloured by '{color_col}' to check subgroup effects." if color_col else "")
                    + sample_note
                ),
                "alternatives": (
                    "Heatmap shows r but not non-linearity, heteroskedasticity, or clusters. "
                    "Line chart assumes time-ordering — wrong here. "
                    "Scatter is the only chart showing every individual observation relationship."
                ),
                "question": (
                    f"Is {c1}–{c2} truly linear? Are outliers driving the correlation? "
                    "Do subgroups behave differently?"
                ),
            })

    # ── 5. Horizontal bar — categorical frequency (FULL frame counts) ──────
    for col in cat_cols[:3]:
        vc = df[col].value_counts().head(20).reset_index()
        vc.columns = [col, "count"]
        vc["pct"]  = (vc["count"] / len(df) * 100).round(1)

        fig = px.bar(
            vc, x="count", y=col,
            orientation="h",
            color="count",
            color_continuous_scale=["#1c2333", "#4d8ef5"],
            text=[f"{p}%" for p in vc["pct"]],
        )
        fig.update_traces(textposition="outside")
        _layout(
            fig,
            f"Category Frequency — {col}",
            xt="Count",
            extra_y={"categoryorder": "total ascending"},
        )
        fig.update_layout(coloraxis_showscale=False)

        dom_pct = vc["pct"].iloc[-1] if not vc.empty else 0
        charts.append({
            "title": f"Frequency — {col}", "fig": fig,
            "why_this": (
                "Horizontal bar because category labels are text — they read left-to-right "
                "on the y-axis without truncation. Sorted ascending so the dominant "
                f"category is at the top (represents {dom_pct}% of records). "
                + ("⚠ Class imbalance this extreme will bias classifiers."
                   if dom_pct > 70 else "")
            ),
            "alternatives": (
                "Pie chart requires estimating angles — humans do this poorly for >3 slices. "
                "Pie hides absolute counts. Vertical bar is hard to read with long labels. "
                "Treemap is for hierarchical data — not needed for a flat category."
            ),
            "question": (
                f"What is the frequency distribution of '{col}'? "
                "Is there class imbalance? Are there rare values that are data errors?"
            ),
        })

    # ── 6. Time series + rolling mean (order-preserving downsample) ────────
    if dt_cols and num_cols:
        dt_col  = dt_cols[0]
        val_col = num_cols[0]
        ts = df[[dt_col, val_col]].dropna().sort_values(dt_col)
        # Random sampling would destroy temporal order, so downsample evenly by
        # taking every k-th row AFTER sorting — keeps trend/seasonality intact.
        if len(ts) > SAMPLE_ROWS_FOR_CHARTS:
            step = len(ts) // SAMPLE_ROWS_FOR_CHARTS + 1
            ts = ts.iloc[::step]
        if len(ts) >= 5:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts[dt_col], y=ts[val_col],
                mode="lines", line=dict(color="#4d8ef5", width=1.2),
                name="Raw", opacity=0.6,
            ))
            if len(ts) >= 20:
                window = max(5, len(ts) // 20)
                roll   = ts[val_col].rolling(window, min_periods=1).mean()
                fig.add_trace(go.Scatter(
                    x=ts[dt_col], y=roll,
                    mode="lines", line=dict(color="#f0a020", width=2, dash="dot"),
                    name=f"{window}-period rolling mean",
                ))
            _layout(fig, f"Time Series — {val_col} over {dt_col}",
                    xt=dt_col, yt=val_col)

            charts.append({
                "title": f"Time Series — {val_col}", "fig": fig,
                "why_this": (
                    "Line chart because time is continuous and ordered — connecting points "
                    "encodes the direction and rate of change. "
                    "Raw data (blue) shows volatility; rolling mean (amber dashed) reveals trend."
                ),
                "alternatives": (
                    "Bar chart suits discrete period counts (monthly totals) "
                    "but hides intra-period variation on dense series. "
                    "Scatter loses temporal ordering. "
                    "Area chart suits cumulative metrics — line is better for rate-of-change."
                ),
                "question": (
                    f"Is there trend, seasonality, or a structural break in '{val_col}' over time?"
                ),
            })

    # ── 7. Null positional heatmap (first 250 rows only — already light) ────
    miss_cols = [c for c in df.columns if df[c].isnull().any()]
    if miss_cols:
        sample = df[miss_cols].head(250).isnull().astype(int)
        fig = go.Figure(go.Heatmap(
            z=sample.values,
            x=sample.columns.tolist(),
            y=list(range(len(sample))),
            colorscale=[[0, "#161b27"], [1, "#ef5e5e"]],
            showscale=False,
            hovertemplate="Col: %{x}<br>Row: %{y}<br>Missing: %{z}<extra></extra>",
        ))
        _layout(fig, "Missing Value Map (first 250 rows) — red = missing",
                xt="Column", yt="Row index")

        charts.append({
            "title": "Missing Value Map", "fig": fig,
            "why_this": (
                "Positional heatmap: rows = records, columns = features, red = null. "
                "Vertical streaks → whole column is sparse (consider dropping). "
                "Horizontal streaks → certain rows have many nulls simultaneously "
                "(different source / system failure). "
                "Random scatter → Missing Completely At Random (MCAR) → safe to impute."
            ),
            "alternatives": (
                "Bar chart of null counts per column tells you HOW MUCH is missing — not WHERE. "
                "Pattern reveals MCAR vs systematic missingness, "
                "which determines the correct imputation strategy."
            ),
            "question": (
                "Are missing values random (MCAR) or do they cluster, "
                "suggesting a systematic data collection failure?"
            ),
        })

    return charts
