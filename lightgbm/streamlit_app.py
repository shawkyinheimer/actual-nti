import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

st.set_page_config(
    page_title="Bank Marketing EDA",
    page_icon="📊",
    layout="wide"
)

DATA_PATH = "/home/shawky/Documents/nti/actual nti/lightgbm/bank-direct-marketing-campaigns.csv"

st.title("📊 Bank Direct Marketing Campaign Analysis")
st.caption("Interactive EDA dashboard for the LightGBM internship project")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Dataset not found:\n{DATA_PATH}")
    st.info("Make sure the CSV exists at that exact path.")
    st.stop()
except Exception as e:
    st.error(f"Could not load the dataset: {e}")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Dashboard Controls")
st.sidebar.success("Dataset loaded successfully")

target_col = "y"

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = [
    c for c in df.select_dtypes(include="object").columns
    if c != target_col
]

# -----------------------------
# Overview
# -----------------------------
st.header("1. Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")
c4.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("Column information"):
    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Unique Values": [df[c].nunique() for c in df.columns],
        "Missing Values": [df[c].isna().sum() for c in df.columns],
    })
    st.dataframe(info, use_container_width=True)

# -----------------------------
# Target analysis
# -----------------------------
st.header("2. Target Variable Analysis")

if target_col in df.columns:
    target_counts = df[target_col].value_counts()
    target_pct = df[target_col].value_counts(normalize=True) * 100

    c1, c2, c3 = st.columns(3)
    majority = target_counts.idxmax()
    minority = target_counts.idxmin()
    ratio = target_counts[majority] / target_counts[minority]

    c1.metric("Majority Class", str(majority))
    c2.metric("Minority Class", str(minority))
    c3.metric("Imbalance Ratio", f"{ratio:.2f}:1")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    target_counts.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Target Distribution")
    axes[0].set_xlabel("Subscription")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    axes[1].pie(
        target_counts.values,
        labels=target_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    axes[1].set_title("Target Percentage")

    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        pd.DataFrame({
            "Count": target_counts,
            "Percentage": target_pct.round(2)
        }),
        use_container_width=True
    )

# -----------------------------
# Numeric analysis
# -----------------------------
st.header("3. Numeric Features")

selected_num = st.multiselect(
    "Select numeric features",
    numeric_cols,
    default=[c for c in [
        "age", "duration", "campaign",
        "pdays", "previous", "euribor3m", "nr.employed"
    ] if c in numeric_cols]
)

if selected_num:
    st.subheader("Summary Statistics")
    st.dataframe(df[selected_num].describe().T, use_container_width=True)

    feature = st.selectbox("Feature distribution", selected_num)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(df[feature].dropna(), kde=True, ax=axes[0])
    axes[0].set_title(f"Distribution of {feature}")

    sns.boxplot(y=df[feature].dropna(), ax=axes[1])
    axes[1].set_title(f"Boxplot of {feature}")

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# Categorical analysis
# -----------------------------
st.header("4. Categorical Features")

if categorical_cols:
    cat_feature = st.selectbox("Select categorical feature", categorical_cols)

    counts = df[cat_feature].value_counts()
    percentages = df[cat_feature].value_counts(normalize=True) * 100

    col1, col2 = st.columns([1, 2])

    with col1:
        st.dataframe(
            pd.DataFrame({
                "Count": counts,
                "Percentage": percentages.round(2)
            }),
            use_container_width=True
        )

    with col2:
        fig, ax = plt.subplots(figsize=(10, 5))
        counts.sort_values().plot(kind="barh", ax=ax)
        ax.set_title(f"Distribution of {cat_feature}")
        ax.set_xlabel("Count")
        st.pyplot(fig)
        plt.close(fig)

# -----------------------------
# Unknown / placeholder audit
# -----------------------------
st.header("5. Data Quality & Placeholder Audit")

unknown_rows = []
for col in df.select_dtypes(include="object").columns:
    count = (df[col] == "unknown").sum()
    if count:
        unknown_rows.append({
            "Column": col,
            "Unknown Count": int(count),
            "Unknown %": round(count / len(df) * 100, 2)
        })

if unknown_rows:
    st.subheader("'unknown' values")
    st.dataframe(pd.DataFrame(unknown_rows), use_container_width=True)
else:
    st.success("No 'unknown' placeholders found.")

if "pdays" in df.columns:
    count_999 = (df["pdays"] == 999).sum()
    st.subheader("pdays = 999")
    st.metric(
        "Rows using 999 placeholder",
        f"{count_999:,}",
        f"{count_999 / len(df) * 100:.2f}% of dataset"
    )
    st.caption(
        "In this dataset, 999 represents that the client was not previously contacted."
    )

# -----------------------------
# Features vs target
# -----------------------------
st.header("6. Features vs Target")

if target_col in df.columns:
    feature = st.selectbox(
        "Choose a feature to compare with subscription",
        [c for c in df.columns if c != target_col],
        key="target_feature"
    )

    if feature in numeric_cols:
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.boxplot(data=df, x=target_col, y=feature, ax=ax)
        ax.set_title(f"{feature} vs {target_col}")
        st.pyplot(fig)
        plt.close(fig)

        grouped = df.groupby(target_col)[feature].agg(
            ["count", "mean", "median", "std"]
        )
        st.dataframe(grouped, use_container_width=True)

    else:
        crosstab = pd.crosstab(
            df[feature],
            df[target_col],
            normalize="index"
        ) * 100

        st.subheader("Conversion Rate by Category")
        st.dataframe(crosstab.round(2), use_container_width=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        crosstab.plot(kind="bar", stacked=True, ax=ax)
        ax.set_ylabel("Percentage (%)")
        ax.set_title(f"{target_col} distribution by {feature}")
        ax.legend(title=target_col)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
        plt.close(fig)

# -----------------------------
# Correlation
# -----------------------------
st.header("7. Correlation Analysis")

if len(numeric_cols) >= 2:
    method = st.radio(
        "Correlation method",
        ["Pearson", "Spearman"],
        horizontal=True
    )

    corr = df[numeric_cols].corr(
        method=method.lower()
    )

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax
    )
    ax.set_title(f"{method} Correlation Matrix")
    st.pyplot(fig)
    plt.close(fig)

    threshold = st.slider(
        "High-correlation threshold",
        min_value=0.50,
        max_value=0.99,
        value=0.70,
        step=0.05
    )

    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                pairs.append({
                    "Feature 1": corr.columns[i],
                    "Feature 2": corr.columns[j],
                    "Correlation": round(value, 4)
                })

    if pairs:
        st.subheader("Highly Correlated Feature Pairs")
        st.dataframe(
            pd.DataFrame(pairs).sort_values(
                "Correlation",
                key=lambda x: x.abs(),
                ascending=False
            ),
            use_container_width=True
        )
    else:
        st.info("No feature pairs exceed the selected threshold.")

# -----------------------------
# Key insights
# -----------------------------
st.header("8. Key EDA Insights")

if target_col in df.columns:
    yes_pct = (df[target_col] == "yes").mean() * 100
    no_pct = (df[target_col] == "no").mean() * 100

    st.markdown(f"""
    ### Dataset
    - **{len(df):,} observations** and **{len(df.columns)} columns**
    - **{df.isna().sum().sum():,} explicit missing values**
    - **{df.duplicated().sum():,} duplicate rows**

    ### Target
    - `no`: **{no_pct:.1f}%**
    - `yes`: **{yes_pct:.1f}%**
    - The target is strongly imbalanced, so accuracy alone should not be the main model metric.

    ### Modeling considerations
    - Treat `unknown` values carefully rather than blindly replacing them.
    - `pdays = 999` is a semantic placeholder.
    - `duration` can be a very strong predictor but may introduce **data leakage** because call duration is only known after the call.
    - Highly correlated macroeconomic variables may contain redundant information.
    """)

st.sidebar.markdown("---")
st.sidebar.caption("LightGBM Internship • Bank Marketing Dataset")
