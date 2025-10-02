"""
AI Auto Data Analytics — Steps 1-4 (Upload → Clean → Feature Engineering → EDA)
This script requires the following dependencies in requirements.txt:
streamlit, pandas, numpy, plotly, scikit-learn, reportlab, anthropic (optional)
"""

import os
import io
import warnings
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold

# Optional: Claude/Anthropic
try:
    # Note: Anthropic library structure is often different between versions.
    # We use the modern Anthropic client setup.
    import anthropic
    HAS_ANTHROPIC = True
except Exception:
    HAS_ANTHROPIC = False

# Reporting (PDF generation)
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False

warnings.filterwarnings("ignore")

# ----------------------------
# UI / THEME
# ----------------------------
st.set_page_config(page_title="AI Auto Data Analytics", layout="wide")
st.markdown(
    """
    <style>
     body { background-color: #0e1117; color: #FAFAFA; }
     .stApp { background-color: #0e1117; color: #FAFAFA; }
     /* Ensure main blocks/containers respect the dark theme */
     .css-1d391kg { background-color: #111318; color: #FAFAFA; } 
     h1, h2, h3, h4, h5, h6 { color: #4ADE80 !important; }
     .big { font-size:18px; }
     .mono { font-family: monospace; }
     .reportbox { background: #0b1114; padding:12px; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌌 AI Auto Data Analytics — Upload → Clean → FE → EDA (Dark Mode)")
st.write("Human-friendly, automatic Steps 1–4. Optional Claude summary if you provide a key.")

# ----------------------------
# Helper functions
# ----------------------------
def safe_read_csv(uploaded_file):
    """Safely reads a CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        return None, str(e)

def detect_datetime_and_numeric(df, logs):
    """Attempts to coerce object columns to datetime or numeric based on content."""
    for col in df.columns:
        if df[col].dtype == object:
            # 1. Try datetime
            try:
                # Use errors='coerce', then check success rate
                parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                # Require 80% success rate for coercion to be valid
                if parsed.notna().sum() / max(1, len(parsed)) > 0.8: 
                    df[col] = parsed
                    logs.append(f"Converted column '{col}' to datetime (80% non-NaT).")
                    continue
            except Exception:
                pass 
                
            # 2. Try numeric-ish
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                # Require 50% success rate
                if converted.notna().sum() / max(1, len(converted)) > 0.5:
                    df[col] = converted
                    logs.append(f"Converted column '{col}' to numeric (where possible).")
            except Exception:
                pass
    return df

def replace_common_tokens_with_nan(df):
    """Converts common 'missing' strings into numpy.nan."""
    tokens = ["Unknown", "unknown", "NA", "N/A", ""]
    return df.replace(tokens, np.nan)

def fill_missing(df, logs):
    """Fills missing values with median for numeric, or 'Unknown' for categorical."""
    for col in df.columns:
        missing = df[col].isna().sum()
        if missing > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Use median for robustness against outliers
                med = df[col].median()
                df[col].fillna(med, inplace=True)
                logs.append(f"Filled {missing} NaNs in numeric '{col}' with median={med:.2f}.")
            else:
                # Use 'Unknown' for categorical/object
                df[col].fillna("Unknown", inplace=True) 
                logs.append(f"Filled {missing} NaNs in categorical '{col}' with 'Unknown'.")
    return df

def drop_duplicates(df, logs):
    """Removes exact duplicate rows."""
    before = df.shape[0]
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - df.shape[0]
    if dropped > 0:
        logs.append(f"Dropped {dropped} duplicate rows.")
    return df

def expand_datetime_features(df, logs):
    """Extracts year, month, day, and weekday from datetime columns."""
    dt_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    for col in dt_cols:
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_day"] = df[col].dt.day
        df[f"{col}_weekday"] = df[col].dt.weekday
        logs.append(f"Expanded datetime column '{col}' -> year/month/day/weekday.")
    return df

def encode_categoricals(df, logs, one_hot_limit=10, dim_limit=1000):
    """Encodes categorical columns using One-Hot or Label Encoding."""
    df = df.copy()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    total_cols_if_onehot = df.shape[1]
    
    for col in cat_cols:
        # Convert to string to prevent encoding errors on mixed types
        df[col] = df[col].astype(str) 
        nunique = df[col].nunique(dropna=True)
        
        if nunique <= 1:
            logs.append(f"Dropped constant column '{col}' (nunique={nunique}).")
            df.drop(columns=[col], inplace=True)
            continue
            
        if nunique == 2:
            # Binary columns: always use Label Encoding (0/1)
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col])
            logs.append(f"Label-encoded binary column '{col}'.")
            
        elif nunique <= one_hot_limit and total_cols_if_onehot + nunique - 1 < dim_limit:
            # Low cardinality: use One-Hot Encoding
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            total_cols_if_onehot = df.shape[1]
            logs.append(f"One-hot encoded '{col}' -> added {dummies.shape[1]} columns.")
            
        else:
            # High cardinality: use Label Encoding
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col])
            logs.append(f"Label-encoded high-cardinality column '{col}' (nunique={nunique}).")
            
    return df

def scale_numeric(df, logs):
    """Scales numeric columns using StandardScaler."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        # Exclude columns that are already 0/1 (often created by OHE)
        cols_to_scale = [c for c in num_cols if not (df[c].isin([0, 1]).all() and df[c].nunique() == 2)]
        if cols_to_scale:
            scaler = StandardScaler()
            df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
            logs.append(f"Scaled numeric columns with StandardScaler ({len(cols_to_scale)} cols).")
        else:
            logs.append("No suitable numeric columns found for scaling.")
    return df

def drop_low_variance(df, logs, threshold=1e-4):
    """Removes numeric columns with near-zero variance."""
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        logs.append("No numeric columns available for low-variance filtering.")
        return df
        
    try:
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(numeric)
        
        cols_to_keep = numeric.columns[vt.get_support()].tolist()
        drop_cols = [c for c in numeric.columns if c not in cols_to_keep]
        
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            logs.append(f"Dropped low-variance numeric columns: {drop_cols}")
        else:
             logs.append("No numeric columns dropped by low-variance filter.")
    except Exception as e:
        logs.append(f"Skipped low-variance filtering due to error: {e}")
        pass
        
    return df

def correlation_heatmap(df_numeric, title="Correlation heatmap"):
    """Generates a Plotly correlation heatmap."""
    corr = df_numeric.corr()
    fig = px.imshow(
        corr, 
        text_auto=".2f", # Show text with 2 decimal places
        title=title,
        color_continuous_scale=px.colors.diverging.RdBu,
        aspect="auto"
    )
    fig.update_xaxes(side="top")
    return fig

def missingness_matrix(df):
    """Generates a Plotly matrix showing missing values (1 = missing)."""
    mask = df.isna().astype(int)
    fig = px.imshow(
        mask.T, 
        labels={'x':'Row Index','y':'Column'}, 
        title="Missingness matrix (1 = missing)",
        color_continuous_scale=[(0, 'rgb(0,0,0)'), (1, 'rgb(255,0,0)')]
    )
    return fig

def top_value_counts(df, col, top_n=20):
    """Calculates top value counts for plotting."""
    vc = df[col].astype(str).value_counts(dropna=True).head(top_n).reset_index()
    vc.columns = [col, 'count']
    return vc

def build_claude_prompt(df, cleaning_log, fe_notes):
    """Constructs the prompt for the Claude API call."""
    desc = df.describe(include='all').to_string()
    info_buf = io.StringIO()
    df.info(buf=info_buf)
    info_text = info_buf.getvalue()
    corr = ""
    try:
        corr_df = df.select_dtypes(include=[np.number]).corr()
        corr = corr_df.to_string()
    except Exception:
        corr = "(no numeric corr available)"
    prompt = (
        "You are an expert data scientist. Produce a concise, human-friendly EDA summary and "
        "feature-engineering summary with actionable next steps.\n\n"
        "DATAFRAME DESCRIBE:\n" + desc + "\n\n"
        "DATAFRAME INFO:\n" + info_text + "\n\n"
        "CORRELATION MATRIX:\n" + corr + "\n\n"
        "CLEANING LOG:\n" + "\n".join(cleaning_log) + "\n\n"
        "FEATURE-ENGINEERING NOTES:\n" + "\n".join(fe_notes) + "\n\n"
        "Please keep the summary short (5-8 bullets) and written in plain language."
    )
    return prompt

def call_claude(prompt_text, api_key, model="claude-3-haiku-20240307"):
    """Calls the Anthropic Claude API."""
    if not HAS_ANTHROPIC or not api_key:
        return None
    try:
        # Uses the modern Anthropic client
        client = anthropic.Anthropic(api_key=api_key) 
        
        resp = client.messages.create(
            model=model,
            system="You are a helpful data scientist assistant.",
            messages=[{"role": "user", "content": prompt_text}],
            max_tokens=800,
            temperature=0.0,
        )
        
        if resp.content and resp.content[0].type == 'text':
            return resp.content[0].text
        else:
             return f"(Claude response structure error or no text content: {resp})"
             
    except Exception as e:
        return f"(Claude call failed: {e})"

def make_pdf_report_bytes(cleaning_log, fe_notes, claude_text=None):
    """Creates a PDF report using reportlab."""
    if not HAS_REPORTLAB:
        # Should not happen if requirements.txt is correct, but safe fallback
        st.error("Reportlab library is missing. Cannot generate PDF report.")
        return None
        
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    elems = []
    
    elems.append(Paragraph("AI Auto Data Analytics Report", styles['Title']))
    elems.append(Spacer(1, 12))
    elems.append(Paragraph(f"Generated: {datetime.utcnow().isoformat()}Z", styles['Normal']))
    elems.append(Spacer(1, 12))
    
    elems.append(Paragraph("Cleaning Log:", styles['Heading2']))
    for l in cleaning_log:
        elems.append(Paragraph("- " + l, styles['Normal'])) 
    elems.append(Spacer(1, 12))
    
    elems.append(Paragraph("Feature Engineering Notes:", styles['Heading2']))
    for l in fe_notes:
        elems.append(Paragraph("- " + l, styles['Normal']))
    
    if claude_text:
        elems.append(Spacer(1, 12))
        elems.append(Paragraph("Claude-generated Summary:", styles['Heading2']))
        # Split text into paragraphs for cleaner PDF rendering
        claude_paragraphs = claude_text.split('\n')
        for p in claude_paragraphs:
            if p.strip():
                elems.append(Paragraph(p, styles['Normal']))
    
    doc.build(elems)
    buffer.seek(0)
    return buffer.getvalue()

# ----------------------------
# App flow
# ----------------------------
uploaded = st.file_uploader("📂 Upload CSV", type="csv")
if uploaded is None:
    st.info("Upload a CSV file to begin. Steps: Inspect → Clean → Feature-engineer → EDA. Optional Claude summary.")
    st.stop()

# Read CSV
df_raw, err = safe_read_csv(uploaded)
if df_raw is None:
    st.error(f"Could not read CSV: {err}")
    st.stop()

# Basic info
st.subheader("📊 Data Overview")
st.markdown(f"**Rows:** {df_raw.shape[0]:,} | **Columns:** {df_raw.shape[1]}") 
with st.expander("Show column names & dtypes"):
    dtypes = pd.DataFrame({"column": df_raw.columns, "dtype": df_raw.dtypes.astype(str)})
    st.dataframe(dtypes)

with st.expander("Show first 5 rows"):
    st.dataframe(df_raw.head())

with st.expander("Null counts (before cleaning)"):
    st.dataframe(df_raw.isnull().sum().rename("null_count").to_frame())

# Prepare logs
cleaning_log = []
fe_notes = []

# Step 2: Cleaning
st.subheader("🧹 Automatic Data Cleaning (Step 2)")

with st.spinner("Running type detection & initial token normalization..."):
    df_work = df_raw.copy()
    # Normalize tokens to NaN
    df_work = replace_common_tokens_with_nan(df_work)
    cleaning_log.append("Replaced common tokens ['Unknown','unknown','NA','N/A',''] with NaN.")
    # Type coercion
    df_work = detect_datetime_and_numeric(df_work, cleaning_log)

# Drop duplicates
df_work = drop_duplicates(df_work, cleaning_log)

# Fill missing
df_work = fill_missing(df_work, cleaning_log)

st.success("Cleaning complete.")
st.markdown("**Cleaning log:**")
for l in cleaning_log:
    st.write("- " + l)

st.markdown("### Missing values (after cleaning)")
st.dataframe(df_work.isnull().sum().rename("null_count").to_frame())

# Step 3: Feature Engineering controls
st.subheader("🛠 Feature Engineering (Step 3)")
st.write("You can toggle automatic FE steps below. Defaults are enabled.")

fe_expand_dates = st.checkbox("Expand datetime columns (year/month/day/weekday)", value=True)
fe_encode_cat = st.checkbox("Encode categorical columns (Label / One-hot)", value=True)
fe_scale = st.checkbox("Scale numeric columns (StandardScaler)", value=True)
fe_var_filter = st.checkbox("Drop near-zero variance numeric features", value=True)

# Perform FE
df_fe = df_work.copy()

if fe_expand_dates:
    df_fe = expand_datetime_features(df_fe, fe_notes)

if fe_encode_cat:
    # Set one-hot cutoff and global dimension safety
    col1, col2 = st.columns(2)
    with col1:
        one_hot_limit = st.number_input("One-hot unique-value limit (<=)", min_value=2, max_value=100, value=10, step=1)
    with col2:
        dim_limit = st.number_input("Maximum allowed total columns after one-hot (safety cap)", min_value=50, max_value=5000, value=1000, step=50)
    df_fe = encode_categoricals(df_fe, fe_notes, one_hot_limit=one_hot_limit, dim_limit=dim_limit)

if fe_scale:
    df_fe = scale_numeric(df_fe, fe_notes)

if fe_var_filter:
    df_fe = drop_low_variance(df_fe, fe_notes)

st.write("Feature engineering notes:")
for n in fe_notes:
    st.write("- " + n)

st.markdown(f"**Feature matrix shape after FE:** {df_fe.shape}")

# Step 4: EDA Visualizations
st.subheader("🔍 EDA Visualizations (Step 4)")

# Sampling control for very large datasets
nrows = df_raw.shape[0]
viz_df = df_fe
if nrows > 20000:
    sample_checkbox = st.checkbox(f"Sample data for plotting (recommended for >20k rows). Uncheck to plot full data.", value=True)
    if sample_checkbox:
        sample_n = min(nrows, 20000)
        viz_df = df_fe.sample(n=sample_n, random_state=42)
        st.info(f"Using a {sample_n}-row sample (from {nrows} rows) for plotting.")
    
# Exclude 'Unknown' from visualizations (treat as missing)
viz_df_plot = viz_df.replace("Unknown", np.nan)

numeric_cols = viz_df_plot.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = viz_df_plot.select_dtypes(include=['object','category']).columns.tolist()

# Numeric: histograms + boxplots
if numeric_cols:
    st.markdown("**Numeric distributions & boxplots**")
    for col in numeric_cols:
        c1, c2 = st.columns([1, 1])
        with c1:
            fig_h = px.histogram(viz_df_plot, x=col, nbins=30, title=f"{col} — histogram", marginal="box")
            st.plotly_chart(fig_h, use_container_width=True)
        with c2:
            fig_b = px.box(viz_df_plot, y=col, title=f"{col} — boxplot")
            st.plotly_chart(fig_b, use_container_width=True)

# Scatter matrix (limited)
if len(numeric_cols) > 1:
    limited = numeric_cols[:8]
    st.markdown("**Scatter matrix (first up to 8 numeric columns)**")
    try:
        fig_matrix = px.scatter_matrix(viz_df_plot, dimensions=limited, title="Scatter matrix")
        st.plotly_chart(fig_matrix, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not generate Scatter Matrix: {e}")

# Correlation heatmap
if len(numeric_cols) > 1:
    st.markdown("**Correlation heatmap**")
    st.plotly_chart(correlation_heatmap(viz_df_plot[numeric_cols]), use_container_width=True)

# Categorical: bar + pie (skip very high cardinality)
if cat_cols:
    st.markdown("**Categorical distributions**")
    for col in cat_cols:
        nunique_original = df_raw[col].nunique(dropna=True) 
        if nunique_original <= 40 and nunique_original >= 2:
            counts = top_value_counts(viz_df_plot, col, top_n=30)
            c1, c2 = st.columns([1, 1])
            with c1:
                fig_bar = px.bar(counts, x=col, y='count', title=f"{col} — bar")
                st.plotly_chart(fig_bar, use_container_width=True)
            with c2:
                fig_pie = px.pie(counts, names=col, values='count', title=f"{col} — pie")
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.write(f"Skipping plots for '{col}' (high cardinality or constant, N={nunique_original})")

# Missingness matrix (show original missingness before fills)
st.markdown("**Missingness matrix (original file)**")
st.plotly_chart(missingness_matrix(df_raw), use_container_width=True)

# Target distribution helper (user can pick a target)
st.subheader("Target / Column peek")
target_choice = st.selectbox("If you want a target preview, select target column (optional):", options=[None] + list(df_raw.columns))
if target_choice:
    t = df_raw[target_choice].replace(["Unknown","unknown","NA","N/A",""], np.nan)
    t_plot_df = pd.DataFrame({target_choice: t.dropna()}) 
    
    if t.dtype in [np.number, 'int64', 'float64'] and t.nunique() > 20:
        st.plotly_chart(px.histogram(t_plot_df, x=target_choice, nbins=30, title=f"{target_choice} distribution (regression-like)"), use_container_width=True)
    else:
        tcounts = t.value_counts(dropna=True).reset_index(name='count')
        tcounts.columns = [target_choice, 'count']
        st.plotly_chart(px.bar(tcounts, x=target_choice, y='count', title=f"{target_choice} counts"), use_container_width=True)
        st.plotly_chart(px.pie(tcounts, names=target_choice, values='count', title=f"{target_choice} pie"), use_container_width=True)

# Optional: ydata-profiling (if installed)
if st.checkbox("Generate ydata-profiling report (may be slow)"):
    try:
        from ydata_profiling import ProfileReport
        from streamlit_pandas_profiling import st_profile_report
        profile = ProfileReport(df_fe, explorative=True)
        st_profile_report(profile)
    except Exception as e:
        st.warning("ydata-profiling not available or failed. Install `ydata-profiling` and `streamlit-pandas-profiling` to enable: " + str(e))

# ----------------------------
# Optional: Claude (Anthropic) summary
# ----------------------------
st.subheader("📝 Optional: Claude (Anthropic) summary")
st.write("If you provide a CLAUDE_API_KEY and confirm, a summary of the data and steps will be sent to Claude for an expert review.")

claude_key_from_env = os.environ.get("CLAUDE_API_KEY", "")
claude_key = st.text_input("Paste your Claude/Anthropic API key (or leave blank to skip):", type="password", value=claude_key_from_env)
claude_confirm = st.checkbox("I confirm that I want to send dataset summary to Claude", value=False)

claude_text = None
if claude_key and claude_confirm:
    if not HAS_ANTHROPIC:
        st.warning("Anthropic SDK not installed. Install `anthropic` to enable Claude calls.")
    else:
        st.info("Calling Claude — please wait (this may take a few seconds).")
        prompt = build_claude_prompt(df_fe, cleaning_log, fe_notes)
        with st.spinner("Requesting Claude..."):
            claude_text = call_claude(prompt, claude_key)
        if claude_text and not claude_text.startswith("(Claude call failed:"):
            st.markdown("**Claude-generated summary:**")
            st.text(claude_text)
        else:
            st.warning(f"Claude did not return a summary or call failed: {claude_text}")

elif claude_key and not claude_confirm:
    st.info("You provided a key but did not confirm sending data to Claude. Check the confirm checkbox to proceed.")

else:
    st.info("No Claude API key provided — skipping remote summary.")

# ----------------------------
# Download cleaned dataset & PDF report
# ----------------------------
st.subheader("💾 Download cleaned & engineered dataset / PDF report")

# Build cleaned CSV bytes
csv_bytes = df_fe.to_csv(index=False).encode("utf-8")
st.download_button("📥 Download cleaned_data.csv", data=csv_bytes, file_name="cleaned_data.csv", mime="text/csv")

# Build PDF
if HAS_REPORTLAB:
    pdf_bytes = make_pdf_report_bytes(cleaning_log, fe_notes, claude_text=claude_text)
    st.download_button("📄 Download local PDF report", data=pdf_bytes, file_name="data_report.pdf", mime="application/pdf")
else:
    st.warning("Install `reportlab` to enable PDF report generation.")

st.success("All done — Steps 1–4 complete. You can now use the cleaned CSV for training models.")
