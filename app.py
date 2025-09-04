import io, json
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
except:
    px = None
    go = None

try:
    from ydata_profiling import ProfileReport
    from streamlit_ydata_profiling import st_profile_report
    HAS_PROFILING = True
except:
    HAS_PROFILING = False

st.set_page_config(page_title="Auto Data Analytics Studio", page_icon="🤖", layout="wide")

st.title("🤖 Automated Data Analytics Studio")
st.caption("Upload a CSV → automatic cleaning, feature engineering, visualization, dashboard, and report generation.")

# ---------------- Load Data ----------------
upload = st.file_uploader("Upload a CSV file", type=["csv"])
if upload is None:
    st.info("👋 Upload a CSV file to begin.")
    st.stop()

try:
    df = pd.read_csv(upload, sep=None, engine="python")
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

st.success(f"Loaded dataset with {df.shape[0]:,} rows × {df.shape[1]:,} columns.")

# ---------------- Detect Column Types ----------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
datetime_cols, cat_cols = [], []
for col in df.columns:
    if df[col].dtype == "object":
        try:
            parsed = pd.to_datetime(df[col], errors="raise")
            if parsed.notna().mean() > 0.9:
                datetime_cols.append(col)
            else:
                cat_cols.append(col)
        except:
            cat_cols.append(col)
    elif "datetime" in str(df[col].dtype):
        datetime_cols.append(col)

# ---------------- Auto Cleaning ----------------
df_clean = df.copy()
if numeric_cols:
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
if cat_cols:
    df_clean[cat_cols] = df_clean[cat_cols].fillna(df_clean[cat_cols].mode().iloc[0])

# ---------------- Auto Feature Engineering ----------------
df_fe = df_clean.copy()
if cat_cols:
    df_fe = pd.get_dummies(df_fe, columns=cat_cols, drop_first=True, dtype=int)
if numeric_cols:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_fe[numeric_cols] = scaler.fit_transform(df_fe[numeric_cols])
for dcol in datetime_cols:
    d = pd.to_datetime(df_fe[dcol], errors="coerce")
    df_fe[f"{dcol}_year"] = d.dt.year
    df_fe[f"{dcol}_month"] = d.dt.month
    df_fe[f"{dcol}_day"] = d.dt.day
    df_fe[f"{dcol}_dow"] = d.dt.dayofweek
for col in df.columns:
    if df[col].dtype == "object":
        df_fe[f"{col}_len"] = df[col].astype(str).str.len()

st.subheader("📊 Processed Dataset Preview")
st.dataframe(df_fe.head(100), use_container_width=True)

# ---------------- Automated Visualizations ----------------
st.header("📈 Automated Visualizations")
if px:
    if numeric_cols:
        st.subheader("Histograms of Numeric Columns")
        for col in numeric_cols[:3]:
            fig = px.histogram(df_clean, x=col)
            st.plotly_chart(fig, use_container_width=True)
    if cat_cols:
        st.subheader("Top Categories")
        top_col = cat_cols[0]
        fig = px.bar(df_clean[top_col].value_counts().reset_index(), x="index", y=top_col)
        st.plotly_chart(fig, use_container_width=True)
    if len(numeric_cols) > 1:
        st.subheader("Correlation Heatmap")
        corr = df_clean[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Dashboard ----------------
st.header("📊 Dashboard")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df_fe.shape[0]:,}")
col2.metric("Columns", f"{df_fe.shape[1]:,}")
col3.metric("Missing", int(df.isna().sum().sum()))

if px and len(numeric_cols) > 1:
    st.subheader("Scatter Plot")
    fig = px.scatter(df_clean, x=numeric_cols[0], y=numeric_cols[1])
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Report ----------------
st.header("📑 Automated Report")
if HAS_PROFILING:
    pr = ProfileReport(df_clean, title="Data Profiling Report", explorative=True)
    st_profile_report(pr)
else:
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "shape": list(df_fe.shape),
        "columns": df_fe.columns.tolist(),
        "dtypes": df_fe.dtypes.astype(str).to_dict(),
        "missing_per_column": df_fe.isna().sum().to_dict(),
        "numeric_summary": df_fe.describe(include=[np.number]).to_dict(),
    }
    st.json(summary)

st.caption("Made with ❤️ Fully Automated Streamlit App. Save as `app.py` and run: `streamlit run app.py`")
