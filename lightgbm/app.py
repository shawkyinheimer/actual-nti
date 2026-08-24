import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Bank Marketing EDA", page_icon="🏦", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("/home/shawky/Documents/nti/actual nti/lightgbm/bank-direct-marketing-campaigns.csv")

df = load_data()

st.title("🏦 Bank Marketing EDA Dashboard")
page = st.sidebar.radio("Section", ["Overview", "Target", "Numeric", "Categorical", "Bivariate", "Correlation", "Insights"])

if page == "Overview":
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", len(df.columns))
    col3.metric("Nulls", df.isnull().sum().sum())
    st.dataframe(df.head())

elif page == "Target":
    tc = df['y'].value_counts()
    st.metric("Imbalance Ratio", f"{tc['no']/tc['yes']:.2f}:1")
    fig, ax = plt.subplots()
    ax.pie(tc, labels=tc.index, autopct='%1.1f%%', colors=['#e74c3c','#2ecc71'])
    st.pyplot(fig)

elif page == "Numeric":
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df[num_cols].describe().T)
    var = st.selectbox("Variable", num_cols)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[var], kde=True, ax=axes[0])
    sns.boxplot(y=df[var], ax=axes[1])
    st.pyplot(fig)

elif page == "Categorical":
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != 'y']
    var = st.selectbox("Variable", cat_cols)
    vc = df[var].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(vc.index, vc.values)
    ax.invert_yaxis()
    st.pyplot(fig)

elif page == "Bivariate":
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    var = st.selectbox("Variable", num_cols)
    fig, ax = plt.subplots(figsize=(10, 5))
    for t, c in zip(['no','yes'], ['#e74c3c','#2ecc71']):
        sns.kdeplot(df[df['y']==t][var], ax=ax, color=c, label=t, fill=True)
    ax.legend()
    st.pyplot(fig)

elif page == "Correlation":
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
    st.pyplot(fig)

elif page == "Insights":
    st.markdown("### Key Findings")
    st.info("- Severe class imbalance (8:1)")
    st.warning("- pdays=999 is placeholder (96% of data)")
    st.success("- Duration is high-signal feature")
    st.error("- Multicollinearity in macro variables")
