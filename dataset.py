import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Streamlit App Title
st.title("📊 Contagious Diseases Analysis Tool")

# File Upload and Selection Section
st.header("📂 Data Input")

# Check for local dataset directory
DATA_DIR = "data/contagious-diseases"
local_files = []
# "hepatitis.csv",
#                "measles.csv",
#                "mumps.csv",
#                "pertussis.csv",
#                "polio.csv",
#                "rubella.csv",
#                "smallpox.csv"

if os.path.exists(DATA_DIR):
    local_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

# Create two columns for layout
col1, col2 = st.columns(2)

# Dataset selection options
if local_files:
    with col1:
        st.subheader("Local Dataset")
        selected_file = st.selectbox("Select local dataset:", local_files)
        file_path = os.path.join(DATA_DIR, selected_file)
else:
    with col1:
        st.info("No local datasets found in 'data/contagious-diseases' directory")

with col2:
    st.subheader("Upload Dataset")
    uploaded_file = st.file_uploader("Or upload CSV file:", type="csv")
    if uploaded_file:
        file_path = uploaded_file

# Check if file is selected/uploaded
if 'file_path' not in locals():
    st.stop()


# Load data
@st.cache_data
def load_data(file_path):
    if isinstance(file_path, str):
        return pd.read_csv(file_path)
    return pd.read_csv(file_path)


try:
    df = load_data(file_path)
except Exception as e:
    st.error(f"Error loading file: {str(e)}")
    st.stop()

# Show basic info
st.success(f"✅ Successfully loaded dataset: {len(df)} rows × {len(df.columns)} columns")

# Main Analysis Section
st.header("🔍 Data Exploration")

# Show Data Preview
with st.expander("📄 Dataset Preview"):
    st.dataframe(df.head())

# Show Basic Statistics
with st.expander("📊 Basic Statistics"):
    st.write(df.describe(include='all'))

# Visualization Section
st.header("📈 Data Visualization")

# Visualization controls
viz_col, display_col = st.columns([1, 3])

# Graph type selector
graph_types = {
    "Bar Chart": "categorical",
    "Line Chart": "numerical",
    "Histogram": "numerical",
    "Scatter Plot": "numerical",
    "Box Plot": "numerical",
    "Pie Chart": "categorical",
    "Heatmap": "correlation"
}

selected_graph = viz_col.selectbox("Choose visualization type:", list(graph_types.keys()))


# Visualization logic
def create_visualization():
    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        if selected_graph == "Bar Chart":
            col = viz_col.selectbox("Select category:", df.select_dtypes(include='object').columns)
            value_counts = df[col].value_counts()
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            plt.xticks(rotation=45)

        elif selected_graph == "Line Chart":
            col = viz_col.selectbox("Select metric:", df.select_dtypes(include='number').columns)
            sns.lineplot(data=df, x=df.index, y=col, ax=ax)

        elif selected_graph == "Histogram":
            col = viz_col.selectbox("Select metric:", df.select_dtypes(include='number').columns)
            sns.histplot(df[col], kde=True, ax=ax)

        elif selected_graph == "Scatter Plot":
            x_col = viz_col.selectbox("X-axis:", df.select_dtypes(include='number').columns)
            y_col = viz_col.selectbox("Y-axis:", df.select_dtypes(include='number').columns)
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)

        elif selected_graph == "Box Plot":
            col = viz_col.selectbox("Select metric:", df.select_dtypes(include='number').columns)
            sns.boxplot(data=df, y=col, ax=ax)

        elif selected_graph == "Pie Chart":
            col = viz_col.selectbox("Select category:", df.select_dtypes(include='object').columns)
            value_counts = df[col].value_counts()
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')

        elif selected_graph == "Heatmap":
            numeric_df = df.select_dtypes(include='number')
            if len(numeric_df.columns) < 2:
                st.warning("Need at least 2 numeric columns for heatmap")
                return
            corr = numeric_df.corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

        plt.tight_layout()
        display_col.pyplot(fig)

    except Exception as e:
        st.error(f"Visualization error: {str(e)}")


create_visualization()

# Advanced Analysis Section
st.header("🧪 Advanced Analysis")

# Column selector
analysis_col = st.selectbox("Select column for detailed analysis:", df.columns)

# Analysis based on data type
if pd.api.types.is_numeric_dtype(df[analysis_col]):
    st.subheader("Numerical Analysis")

    cols = st.columns(3)
    with cols[0]:
        st.metric("Mean", round(df[analysis_col].mean(), 2))
        st.metric("Median", round(df[analysis_col].median(), 2))

    with cols[1]:
        st.metric("Standard Deviation", round(df[analysis_col].std(), 2))
        st.metric("Variance", round(df[analysis_col].var(), 2))

    with cols[2]:
        st.metric("Skewness", round(df[analysis_col].skew(), 2))
        st.metric("Kurtosis", round(df[analysis_col].kurtosis(), 2))

    # Outlier detection
    st.subheader("Outlier Detection")
    q1 = df[analysis_col].quantile(0.25)
    q3 = df[analysis_col].quantile(0.75)
    iqr = q3 - q1
    outliers = df[(df[analysis_col] < (q1 - 1.5 * iqr)) | (df[analysis_col] > (q3 + 1.5 * iqr))]
    st.write(f"Detected {len(outliers)} potential outliers ({len(outliers) / len(df):.2%} of data)")

else:
    st.subheader("Categorical Analysis")
    value_counts = df[analysis_col].value_counts()

    cols = st.columns(2)
    with cols[0]:
        st.write("#### Frequency Table")
        st.write(value_counts)

    with cols[1]:
        st.write("#### Distribution")
        fig, ax = plt.subplots()
        ax.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%')
        st.pyplot(fig)

# Time series analysis (if datetime column exists)
date_cols = df.select_dtypes(include=['datetime']).columns
if len(date_cols) > 0:
    st.header("⏳ Time Series Analysis")
    date_col = st.selectbox("Select date column:", date_cols)
    metric_col = st.selectbox("Select metric:", df.select_dtypes(include='number').columns)

    df_date = df.set_index(date_col)[metric_col]

    resample_freq = st.selectbox("Resampling frequency:",
                                 ['D', 'W', 'M', 'Q', 'Y'])

    df_resampled = df_date.resample(resample_freq).mean()
    st.line_chart(df_resampled)