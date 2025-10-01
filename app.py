import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
import time

# --- Setup and Utility Functions ---

# Streamlit cache ensures the heavy data cleaning only runs once per file upload.
@st.cache_data(show_spinner="Running Automated Data Preparation and Cleaning...")
def automated_cleaning_and_engineering(df):
    
    # Work on a copy of the DataFrame
    df_cleaned = df.copy()
    cleaning_log = []
    
    # 1. Duplicate Removal
    duplicate_rows = df_cleaned.duplicated().sum()
    if duplicate_rows > 0:
        df_cleaned.drop_duplicates(inplace=True)
        cleaning_log.append(f"🗑 *Removed Duplicates:* Found and removed *{duplicate_rows}* duplicate rows.")
    else:
        cleaning_log.append("👍 *Duplicate Check:* No duplicate rows found.")

    # 2. Missing Value Imputation
    missing_data = df_cleaned.isnull().sum()
    missing_cols = missing_data[missing_data > 0]

    if not missing_cols.empty:
        cleaning_log.append("#### ⚠ *Missing Value Imputation:*")
        
        for col in missing_cols.index:
            missing_count = missing_cols[col]
            if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                # Impute numeric missing values with the median
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                cleaning_log.append(f"   - Imputed *{missing_count}* missing values in *{col}* (Numeric) with the *Median* ({median_val:.2f}).")
            else:
                # Impute categorical missing values with the mode
                mode_val = df_cleaned[col].mode().iloc[0] # Use iloc[0] for robustness
                df_cleaned[col].fillna(mode_val, inplace=True)
                cleaning_log.append(f"   - Imputed *{missing_count}* missing values in *{col}* (Categorical) with the *Mode* ('{mode_val}').")
    else:
        cleaning_log.append("🎉 *Missing Values:* No missing values found! Data is complete.")
        
    
    # 3. Feature Engineering: Categorical Encoding (for Visualization/Analysis)
    le = LabelEncoder()
    encoding_count = 0
    
    for col in df_cleaned.columns:
        # Check if the column is categorical/object and has few unique values (low-cardinality)
        if df_cleaned[col].dtype == 'object' and len(df_cleaned[col].unique()) <= 50:
            try:
                # Perform encoding and save to a new column
                df_cleaned[f'{col}_Encoded'] = le.fit_transform(df_cleaned[col])
                cleaning_log.append(f"✏ *Feature Engineering (Encoding):* Created new column *{col}_Encoded* by *Label Encoding* the original categorical column.")
                encoding_count += 1
            except Exception:
                pass # Skip if encoding fails
                
    if encoding_count == 0:
        cleaning_log.append("ℹ *Feature Engineering (Encoding):* No low-cardinality categorical columns found for automatic Label Encoding.")

    # 4. Feature Engineering: Date/Time Extraction
    # Identify potential date/time columns by keyword
    datetime_cols = [col for col in df_cleaned.columns if 'date' in col.lower() or 'time' in col.lower()]
    dt_count = 0
    
    for col in datetime_cols:
        try:
            # Attempt to convert to datetime
            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
            
            # If the conversion was successful and not all values became NaT
            if not df_cleaned[col].isnull().all():
                # Extract new features
                df_cleaned[f'{col}_Year'] = df_cleaned[col].dt.year
                df_cleaned[f'{col}_Month'] = df_cleaned[col].dt.month
                df_cleaned[f'{col}_Day'] = df_cleaned[col].dt.day
                cleaning_log.append(f"🆕 *Feature Engineering (Date/Time):* Extracted *Year, Month, and Day* features from column *{col}*.")
                dt_count += 1
        except Exception:
             pass # Skip if date conversion fails
             
    if dt_count == 0:
        cleaning_log.append("ℹ *Feature Engineering (Date/Time):* No valid date/time columns found for extraction.")

    
    return df_cleaned, cleaning_log

# --- Streamlit Application Layout ---

st.set_page_config(layout="wide", page_title="Data Preparation & Viz Bot")

st.title("🤖 Data Preparation & Visualization Bot")
st.sidebar.header("Data Upload")

# 1. File Upload
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the original data
    data_original = pd.read_csv(uploaded_file)
    
    st.header("1. Original Data Preview")
    st.dataframe(data_original.head())
    st.write(f"Original Shape: {data_original.shape}")
    st.markdown("---")

    # 2. Automated Steps (Cleaning & Engineering)
    processed_data, log_messages = automated_cleaning_and_engineering(data_original)
    
    st.header("2. Data Preparation Chat Log")
    st.info("Here is the step-by-step cleaning and feature engineering performed:")
    
    # Display the cleaning and engineering steps in a "chatbot" format
    with st.container(height=350, border=True):
        for message in log_messages:
            # Use specific markdown formatting for the main log messages
            st.markdown(message)
    st.markdown("---")
    
    st.header("3. Cleaned & Engineered Data")
    st.dataframe(processed_data.head())
    st.write(f"New Shape: {processed_data.shape}")
    st.markdown("---")

    # 4. Interactive Visualization Section
    st.header("4. Interactive Visualizations")

    # Column selection widgets
    all_columns = processed_data.columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        plot_col_x = st.selectbox("Select X-axis (Primary) column:", all_columns)
    with col2:
        plot_type = st.selectbox("Select Plot Type:", ["Histogram/Distribution", "Scatter Plot", "Bar Plot (Counts)"])
    
    if plot_col_x:
        if plot_type == "Histogram/Distribution":
            if pd.api.types.is_numeric_dtype(processed_data[plot_col_x]):
                fig = px.histogram(processed_data, x=plot_col_x, marginal="box", 
                                   title=f'Distribution of {plot_col_x}', 
                                   height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select a *numeric* column for a Histogram/Distribution plot.")
        
        elif plot_type == "Scatter Plot":
            # Require a second column for a scatter plot
            plot_col_y = st.selectbox("Select Y-axis column for Scatter Plot:", all_columns, index=min(1, len(all_columns)-1))
            if pd.api.types.is_numeric_dtype(processed_data[plot_col_x]) and pd.api.types.is_numeric_dtype(processed_data[plot_col_y]):
                fig = px.scatter(processed_data, x=plot_col_x, y=plot_col_y, 
                                 title=f'Scatter Plot of {plot_col_x} vs {plot_col_y}',
                                 height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Scatter plots require *two numeric* columns.")
                
        elif plot_type == "Bar Plot (Counts)":
            # Count the values and plot the top 20
            value_counts = processed_data[plot_col_x].value_counts().head(20).reset_index()
            value_counts.columns = [plot_col_x, 'Count']
            
            fig = px.bar(value_counts, x=plot_col_x, y='Count', 
                         title=f'Count of Top Categories/Values in {plot_col_x}',
                         height=400)
            st.plotly_chart(fig, use_container_width=True)

    # 5. Data Download
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv_output = convert_df(processed_data)
    
    st.sidebar.download_button(
        label="Download Cleaned & Engineered CSV",
        data=csv_output,
        file_name='cleaned_and_engineered_data.csv',
        mime='text/csv',
    )
    
else:
    st.info("Please upload a CSV file in the sidebar to start the automated data preparation process. The bot will detail every cleaning and engineering step!")
