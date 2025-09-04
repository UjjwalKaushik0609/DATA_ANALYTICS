import io
import json
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# Optional deps
try:
    import plotly.express as px
    import plotly.graph_objects as go
except Exception:
    px = None
    go = None

# Optional profiling (ydata-profiling)
try:
    from ydata_profiling import ProfileReport  # noqa: F401
    from streamlit_ydata_profiling import st_profile_report
    HAS_PROFILING = True
except Exception:
    HAS_PROFILING = False

st.set_page_config(
    page_title="Data Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------
# Helpers
# ------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file: io.BytesIO, sep: str = ",", nrows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(file, sep=sep, nrows=nrows)


def detect_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    # Try parse potential dates
    for col in df.columns:
        if col not in datetime_cols and df[col].dtype == "object":
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                if parsed.notna().mean() > 0.9:
                    datetime_cols.append(col)
            except Exception:
                pass
    cat_cols = [c for c in df.columns if c not in numeric_cols + datetime_cols]
    return numeric_cols, cat_cols, datetime_cols


def kpi_card(label: str, value, delta=None):
    st.metric(label, value, delta)


# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.title("📦 Data Input")
    upload = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help="Upload your dataset in CSV format.",
    )
    sep = st.text_input("CSV separator", value=",", help="Change if your file uses ; or \t, etc.")
    sample_n = st.number_input("Read only first N rows (optional)", min_value=0, value=0, step=1000)
    st.divider()
    st.subheader("Report & Export")
    enable_profile = st.checkbox("Generate automated EDA report (ydata-profiling)", value=False,
                                 help="Requires ydata-profiling and streamlit-ydata-profiling to be installed.")
    allow_download = st.checkbox("Allow dataset downloads", value=True)

st.title("🏗️ Streamlit Data Analytics Studio")
st.caption(
    "Upload a CSV → explore → engineer features → visualize → build a dashboard → export a report & data."
)

if upload is None:
    st.info("👋 Upload a CSV in the sidebar to get started.")
    st.stop()

# Load data
nrows = int(sample_n) if sample_n and sample_n > 0 else None
try:
    df = load_csv(upload, sep=sep, nrows=nrows)
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.success(f"Loaded dataset with {df.shape[0]:,} rows × {df.shape[1]:,} columns.")

# Detect column types
num_cols, cat_cols, dt_cols = detect_column_types(df)

# Tabs
tab_overview, tab_clean, tab_features, tab_viz, tab_dash, tab_report = st.tabs(
    ["Overview", "Cleaning", "Feature engineering", "Visualization", "Dashboard", "Report"]
)

# ------------------------------
# Overview tab
# ------------------------------
with tab_overview:
    st.subheader("Peek at data")
    st.dataframe(df.head(200), use_container_width=True)

    st.subheader("Quick stats")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Rows", f"{df.shape[0]:,}")
    with col2:
        kpi_card("Columns", f"{df.shape[1]:,}")
    with col3:
        kpi_card("Numeric cols", len(num_cols))
    with col4:
        kpi_card("Categorical cols", len(cat_cols))

    st.subheader("Missing values")
    miss = df.isna().sum().sort_values(ascending=False)
    miss_df = miss[miss > 0].rename("missing").to_frame()
    if not miss_df.empty:
        st.dataframe(miss_df, use_container_width=True)
        if px:
            fig = px.bar(miss_df.reset_index(), x="index", y="missing")
            fig.update_layout(xaxis_title="Column", yaxis_title="# missing", height=360)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No missing values detected.")

# ------------------------------
# Cleaning tab
# ------------------------------
with tab_clean:
    st.subheader("Handle missing values")
    target = st.selectbox("(Optional) Select target column", [None] + df.columns.tolist())

    st.write("Choose how to impute missing values:")
    colA, colB = st.columns(2)
    with colA:
        num_strategy = st.selectbox("Numeric strategy", ["mean", "median", "most_frequent", "constant"])
        num_fill = st.number_input("Constant (numeric)", value=0.0)
    with colB:
        cat_strategy = st.selectbox("Categorical strategy", ["most_frequent", "constant"])
        cat_fill = st.text_input("Constant (categorical)", value="missing")

    clean_btn = st.button("Apply imputation", use_container_width=True)
    if clean_btn:
        df_clean = df.copy()
        if num_cols:
            if num_strategy == "constant":
                df_clean[num_cols] = df_clean[num_cols].fillna(num_fill)
            elif num_strategy == "median":
                df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].median())
            elif num_strategy == "mean":
                df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mean())
            else:
                # most_frequent
                df_clean[num_cols] = df_clean[num_cols].fillna(df_clean[num_cols].mode().iloc[0])
        if cat_cols:
            if cat_strategy == "constant":
                df_clean[cat_cols] = df_clean[cat_cols].fillna(cat_fill)
            else:
                df_clean[cat_cols] = df_clean[cat_cols].fillna(df_clean[cat_cols].mode().iloc[0])
        st.session_state["df_clean"] = df_clean
        st.success("Imputation complete. Switched to cleaned dataset for next steps.")

    df_work = st.session_state.get("df_clean", df)
    st.dataframe(df_work.head(100), use_container_width=True)

# ------------------------------
# Feature engineering tab
# ------------------------------
with tab_features:
    df_fe = st.session_state.get("df_fe", st.session_state.get("df_clean", df)).copy()

    st.subheader("Encode & scale")
    c1, c2 = st.columns(2)
    with c1:
        onehot_cols = st.multiselect("One-hot encode columns", cat_cols)
        scale_cols = st.multiselect("Standardize numeric columns", num_cols)
    with c2:
        bin_col = st.selectbox("Bin a numeric column", [None] + num_cols)
        bins = st.slider("# bins", 2, 20, 5)

    st.subheader("Date features")
    date_col = st.selectbox("Extract date features from", [None] + dt_cols)

    st.subheader("Text features")
    text_col = st.selectbox("Create text length feature from", [None] + [c for c in df.columns if df[c].dtype == "object"])

    go_btn = st.button("Apply feature engineering", use_container_width=True)

    if go_btn:
        # One-hot
        if onehot_cols:
            df_fe = pd.get_dummies(df_fe, columns=onehot_cols, drop_first=True, dtype=int)
        # Scaling
        if scale_cols:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df_fe[scale_cols] = scaler.fit_transform(df_fe[scale_cols])
        # Binning
        if bin_col:
            df_fe[f"{bin_col}_bin"] = pd.qcut(df_fe[bin_col], q=bins, duplicates="drop")
        # Date
        if date_col:
            try:
                d = pd.to_datetime(df_fe[date_col], errors="coerce")
                df_fe[f"{date_col}_year"] = d.dt.year
                df_fe[f"{date_col}_month"] = d.dt.month
                df_fe[f"{date_col}_day"] = d.dt.day
                df_fe[f"{date_col}_dow"] = d.dt.dayofweek
            except Exception:
                st.warning("Could not parse selected date column.")
        # Text
        if text_col:
            df_fe[f"{text_col}_len"] = df_fe[text_col].astype(str).str.len()

        st.session_state["df_fe"] = df_fe
        st.success("Features engineered. Dataset updated for next tabs.")

    st.dataframe(st.session_state.get("df_fe", df_fe).head(100), use_container_width=True)

    if allow_download and st.session_state.get("df_fe") is not None:
        csv_bytes = st.session_state["df_fe"].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download engineered dataset (CSV)",
            data=csv_bytes,
            file_name="engineered_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ------------------------------
# Visualization tab
# ------------------------------
with tab_viz:
    if px is None:
        st.warning("Plotly is not installed. Run: pip install plotly")
    else:
        st.subheader("Quick chart builder")
        df_plot = st.session_state.get("df_fe", st.session_state.get("df_clean", df))
        cols = df_plot.columns.tolist()
        x = st.selectbox("X axis", cols)
        y = st.selectbox("Y axis (optional)", [None] + cols)
        color = st.selectbox("Color (optional)", [None] + cols)
        chart_type = st.selectbox("Chart type", ["scatter", "line", "bar", "histogram", "box"])

        if st.button("Draw chart", use_container_width=True):
            if chart_type == "scatter":
                fig = px.scatter(df_plot, x=x, y=y, color=color)
            elif chart_type == "line":
                fig = px.line(df_plot, x=x, y=y, color=color)
            elif chart_type == "bar":
                fig = px.bar(df_plot, x=x, y=y, color=color)
            elif chart_type == "histogram":
                fig = px.histogram(df_plot, x=x, color=color)
            else:
                fig = px.box(df_plot, x=color, y=y or x)
            fig.update_layout(height=500, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Dashboard tab
# ------------------------------
with tab_dash:
    st.subheader("Interactive dashboard")
    df_dash = st.session_state.get("df_fe", st.session_state.get("df_clean", df))

    # Simple slicers
    filt_cols = st.multiselect("Add filters for categorical columns", [c for c in df_dash.columns if str(df_dash[c].dtype)=="object" or "category" in str(df_dash[c].dtype)])
    filt_values = {}
    for c in filt_cols:
        opts = sorted(df_dash[c].astype(str).unique().tolist())[:1000]
        filt_values[c] = st.multiselect(f"Filter {c}", opts)

    df_f = df_dash.copy()
    for c, vals in filt_values.items():
        if vals:
            df_f = df_f[df_f[c].astype(str).isin(vals)]

    # KPI area
    k1, k2, k3 = st.columns(3)
    with k1:
        kpi_card("Rows (filtered)", f"{df_f.shape[0]:,}")
    with k2:
        kpi_card("Columns", f"{df_f.shape[1]:,}")
    with k3:
        kpi_card("Missing cells", int(df_f.isna().sum().sum()))

    st.divider()
    colx, coly = st.columns(2)
    if px:
        with colx:
            num_for_hist = [c for c in df_f.columns if pd.api.types.is_numeric_dtype(df_f[c])]
            if num_for_hist:
                fig1 = px.histogram(df_f, x=num_for_hist[0])
                st.plotly_chart(fig1, use_container_width=True)
        with coly:
            if len(num_for_hist) > 1:
                fig2 = px.scatter(df_f, x=num_for_hist[0], y=num_for_hist[1])
                st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Group & aggregate")
    group_cols = st.multiselect("Group by", [c for c in df_f.columns if c not in df_f.select_dtypes("number").columns])
    agg_col = st.selectbox("Aggregate column (numeric)", df_f.select_dtypes("number").columns.tolist())
    agg_fn = st.selectbox("Aggregation", ["mean", "sum", "median", "min", "max", "count"])

    if st.button("Compute aggregation", use_container_width=True):
        if group_cols:
            grouped = getattr(df_f.groupby(group_cols)[agg_col], agg_fn)().reset_index(name=f"{agg_fn}_{agg_col}")
        else:
            grouped = pd.DataFrame({f"{agg_fn}_{agg_col}": [getattr(df_f[agg_col], agg_fn)()]})
        st.dataframe(grouped, use_container_width=True)
        if px and len(grouped) > 1 and group_cols:
            figg = px.bar(grouped, x=group_cols[0], y=f"{agg_fn}_{agg_col}")
            st.plotly_chart(figg, use_container_width=True)

# ------------------------------
# Report tab
# ------------------------------
with tab_report:
    st.subheader("Automated EDA report")
    df_rep = st.session_state.get("df_fe", st.session_state.get("df_clean", df))

    if enable_profile and HAS_PROFILING:
        pr = ProfileReport(df_rep, title="Data Profiling Report", explorative=True)
        st_profile_report(pr)
        # Offer HTML download
        html = pr.to_html()
        st.download_button(
            "⬇️ Download profiling report (HTML)",
            data=html,
            file_name="profiling_report.html",
            mime="text/html",
            use_container_width=True,
        )
    elif enable_profile and not HAS_PROFILING:
        st.warning(
            "ydata-profiling and streamlit-ydata-profiling are not installed.\n"
            "Install with: pip install ydata-profiling streamlit-ydata-profiling"
        )

    st.subheader("Snapshot JSON summary")
    # Simple JSON summary if profiling isn't used
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "shape": list(df_rep.shape),
        "columns": df_rep.columns.tolist(),
        "dtypes": df_rep.dtypes.astype(str).to_dict(),
        "missing_per_column": df_rep.isna().sum().to_dict(),
        "numeric_summary": df_rep.describe(include=[np.number]).to_dict(),
    }
    json_bytes = json.dumps(summary, indent=2).encode("utf-8")
    st.download_button(
        "⬇️ Download JSON summary",
        data=json_bytes,
        file_name="dataset_summary.json",
        mime="application/json",
        use_container_width=True,
    )

st.caption("Made with ❤️ using Streamlit. Save this file as app.py, then run: `streamlit run app.py`.")
