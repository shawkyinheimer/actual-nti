import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Config & Styling
st.set_page_config(
    page_title="Bank Marketing EDA Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Load Data Function with Caching & Null Safety
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('bank-additional-full.csv', sep=';')
        return df
    except Exception:
        return None

df = load_data()

# Check for valid dataframe load
if df is None:
    st.error("⚠️ File `bank-additional-full.csv` not found or could not be loaded. Please ensure the file is present in the working directory.")
    st.stop()

# Identify feature groups
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = [col for col in df.select_dtypes(include=['object']).columns if col != 'y']

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Select Analysis Section:",
    [
        "1. Overview & Health Check",
        "2. Target Variable (y)",
        "3. Univariate - Numeric",
        "4. Univariate - Categorical",
        "5. Bivariate Analysis",
        "6. Correlation & Multicollinearity",
        "7. Key Insights & Takeaways"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(f"**Dataset Shape:** {df.shape[0]:,} rows | {df.shape[1]} columns")

# ============================================================
# SECTION 1: OVERVIEW & HEALTH CHECK
# ============================================================
if page == "1. Overview & Health Check":
    st.title("📊 1. Dataset Overview & Structural Health Check")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{df.shape[0]:,}")
    col2.metric("Total Columns", f"{df.shape[1]}")
    col3.metric("Explicit Nulls", f"{df.isnull().sum().sum()}")
    dups = df.duplicated().sum()
    col4.metric("Duplicate Rows", f"{dups:,} ({(dups/len(df))*100:.2f}%)")

    st.markdown("---")
    
    st.subheader("📋 Data Types & Memory Usage")
    mem_usage = df.memory_usage(deep=True)
    cardinality = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str).values,
        'Unique Values': df.nunique().values,
        'Cardinality %': (df.nunique().values / len(df) * 100).round(2),
        'Memory (KB)': (mem_usage.reindex(df.columns).fillna(0) / 1024).round(2)
    })
    
    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(cardinality, use_container_width=True, height=400)
    with c2:
        st.metric("Total Memory Usage", f"{mem_usage.sum() / 1024**2:.2f} MB")
        st.write("**First 5 Rows Preview:**")
        st.dataframe(df.head(), use_container_width=True)

# ============================================================
# SECTION 2: TARGET VARIABLE ANALYSIS
# ============================================================
elif page == "2. Target Variable (y)":
    st.title("⚖️ 2. Target Variable Analysis (`y`)")
    
    if 'y' not in df.columns:
        st.error("The dataset must contain a target column named `y`.")
        st.stop()

    target_counts = df['y'].value_counts()
    target_pct = df['y'].value_counts(normalize=True) * 100
    
    majority_class = target_counts.idxmax()
    minority_class = target_counts.idxmin()
    imbalance_ratio = target_counts.iloc[0] / target_counts.iloc[-1] if len(target_counts) == 2 else np.nan

    col1, col2, col3 = st.columns(3)
    col1.metric("Majority Class", f"{majority_class}: {target_counts.iloc[0]:,} ({target_pct.iloc[0]:.2f}%)")
    col2.metric("Minority Class", f"{minority_class}: {target_counts.iloc[-1]:,} ({target_pct.iloc[-1]:.2f}%)")
    col3.metric("Imbalance Ratio", f"{imbalance_ratio:.2f} : 1" if len(target_counts) == 2 else "N/A")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Countplot
    ax1 = sns.countplot(data=df, x='y', ax=axes[0], hue='y', palette=['#e74c3c', '#2ecc71'], legend=False)
    axes[0].set_title('Target Distribution (Count)', fontweight='bold')
    for p in ax1.patches:
        ax1.annotate(f'{int(p.get_height()):,}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontweight='bold')

    # Pie Chart
    axes[1].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', 
               colors=['#e74c3c', '#2ecc71'], explode=[0, 0.05], startangle=90,
               textprops={'fontweight': 'bold'})
    axes[1].set_title('Target Distribution (Percentage)', fontweight='bold')
    
    st.pyplot(fig)

# ============================================================
# SECTION 3: UNIVARIATE - NUMERIC
# ============================================================
elif page == "3. Univariate - Numeric":
    st.title("📈 3. Univariate Analysis — Numeric Features")
    
    st.subheader("Summary Statistics, Skewness & Kurtosis")
    skew_kurt = pd.DataFrame({
        'Mean': df[numeric_cols].mean(),
        'Std Dev': df[numeric_cols].std(),
        'Median': df[numeric_cols].median(),
        'Skewness': df[numeric_cols].skew(),
        'Kurtosis': df[numeric_cols].kurtosis()
    })
    st.dataframe(skew_kurt.round(4), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Interactive Distribution Viewer")
    selected_num = st.selectbox("Select Numeric Feature to View:", numeric_cols, index=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Histogram + KDE
    sns.histplot(data=df, x=selected_num, kde=True, ax=axes[0], color='#3498db', bins=40)
    axes[0].axvline(df[selected_num].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df[selected_num].mean():.2f}')
    axes[0].axvline(df[selected_num].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {df[selected_num].median():.2f}')
    axes[0].set_title(f'Distribution of {selected_num}', fontweight='bold')
    axes[0].legend()
    
    # Boxplot
    sns.boxplot(data=df, y=selected_num, ax=axes[1], color='#e74c3c')
    axes[1].set_title(f'Boxplot of {selected_num}', fontweight='bold')
    
    stats_text = f"Mean: {df[selected_num].mean():.2f}\nMedian: {df[selected_num].median():.2f}\nStd: {df[selected_num].std():.2f}\nMin: {df[selected_num].min():.2f}\nMax: {df[selected_num].max():.2f}"
    axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    st.pyplot(fig)

# ============================================================
# SECTION 4: UNIVARIATE - CATEGORICAL
# ============================================================
elif page == "4. Univariate - Categorical":
    st.title("🏷️ 4. Univariate Analysis — Categorical Features")
    
    st.subheader("Hidden Placeholders & Missing Values Audit")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**`'unknown'` Frequencies:**")
        unknown_audit = []
        for col in df.columns:
            if df[col].dtype == 'object':
                u_count = (df[col] == 'unknown').sum()
                if u_count > 0:
                    unknown_audit.append({'Column': col, 'Unknown Count': u_count, 'Pct %': (u_count/len(df))*100})
        st.dataframe(pd.DataFrame(unknown_audit).round(2), use_container_width=True)
        
    with col2:
        st.write("**Special Placeholders Audit:**")
        pdays_999 = (df['pdays'] == 999).sum() if 'pdays' in df.columns else 0
        pout_non = (df['poutcome'] == 'nonexistent').sum() if 'poutcome' in df.columns else 0
        st.write(f"- **`pdays = 999` (Not previously contacted):** {pdays_999:,} ({(pdays_999/len(df))*100:.2f}%)")
        st.write(f"- **`poutcome = 'nonexistent'`: ** {pout_non:,} ({(pout_non/len(df))*100:.2f}%)")

    st.markdown("---")
    st.subheader("Interactive Categorical Feature Explorer")
    if not categorical_cols:
        st.info("No categorical features are available in this dataset.")
        st.stop()

    selected_cat = st.selectbox("Select Categorical Feature:", categorical_cols, index=0)
    
    val_counts = df[selected_cat].value_counts()
    props = df[selected_cat].value_counts(normalize=True) * 100
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(pd.DataFrame({'Count': val_counts, 'Percentage (%)': props.round(2)}), use_container_width=True)
    
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = sns.color_palette('husl', len(val_counts))
        bars = ax.barh(val_counts.index, val_counts.values, color=colors)
        ax.set_title(f'Distribution of {selected_cat}', fontweight='bold')
        ax.invert_yaxis()
        for bar, count in zip(bars, val_counts.values):
            ax.text(bar.get_width() + max(val_counts.values)*0.01, bar.get_y() + bar.get_height()/2,
                   f'{count:,}', va='center', fontsize=9)
        st.pyplot(fig)

# ============================================================
# SECTION 5: BIVARIATE ANALYSIS
# ============================================================
elif page == "5. Bivariate Analysis":
    st.title("🎯 5. Bivariate Analysis (Features vs Target `y`)")
    
    tab1, tab2 = st.tabs(["Numeric vs Target", "Categorical vs Target (Chi-Square)"])
    
    with tab1:
        st.subheader("Numeric Features Split by Subscription Target")
        sel_num_bivar = st.selectbox("Select Numeric Feature:", numeric_cols, index=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        for target_val, color in zip(['no', 'yes'], ['#e74c3c', '#2ecc71']):
            sns.kdeplot(data=df[df['y'] == target_val][sel_num_bivar], ax=axes[0], color=color, label=f'y={target_val}', fill=True, alpha=0.3)
        axes[0].set_title(f'Density Plot: {sel_num_bivar} by Target', fontweight='bold')
        axes[0].legend()
        
        sns.boxplot(data=df, x='y', y=sel_num_bivar, ax=axes[1], hue='y', palette=['#e74c3c', '#2ecc71'], legend=False)
        axes[1].set_title(f'Boxplot: {sel_num_bivar} by Target', fontweight='bold')
        st.pyplot(fig)

    with tab2:
        st.subheader("Chi-Square Tests & Category Subscriptions")
        
        chi_results = []
        for col in categorical_cols:
            crosstab = pd.crosstab(df[col], df['y'])
            chi2, p_val, dof, _ = stats.chi2_contingency(crosstab)
            n = crosstab.sum().sum()
            min_dim = min(crosstab.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0
            chi_results.append({
                'Feature': col, 'Chi2': round(chi2, 2), 'p-value': f"{p_val:.4e}", "Cramér's V": round(cramers_v, 4)
            })
        
        st.dataframe(pd.DataFrame(chi_results), use_container_width=True)
        
        sel_cat_bivar = st.selectbox("Select Categorical Feature for Stacked Plot:", categorical_cols, index=0)
        crosstab_norm = pd.crosstab(df[sel_cat_bivar], df['y'], normalize='index') * 100
        
        fig, ax = plt.subplots(figsize=(10, 4))
        crosstab_norm.plot(kind='barh', stacked=True, ax=ax, color=['#e74c3c', '#2ecc71'])
        ax.set_title(f'Proportion of Subscription Target by {sel_cat_bivar}', fontweight='bold')
        ax.set_xlabel('Percentage (%)')
        ax.invert_yaxis()
        st.pyplot(fig)

# ============================================================
# SECTION 6: CORRELATION & MULTICOLLINEARITY
# ============================================================
elif page == "6. Correlation & Multicollinearity":
    st.title("🔗 6. Multivariate & Correlation Analysis")
    
    corr_method = st.radio("Select Correlation Type:", ["Pearson", "Spearman"], horizontal=True)
    method = corr_method.lower()
    corr_matrix = df[numeric_cols].corr(method=method)
    
    st.subheader(f"{corr_method} Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True, ax=ax)
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("⚡ High Multicollinearity Detector")
    threshold = st.slider("Correlation Threshold (|r| >=):", 0.5, 0.99, 0.70, step=0.05)
    
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) >= threshold:
                pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    high_corr_df = pd.DataFrame(pairs).sort_values('Correlation', key=abs, ascending=False)
    
    if not high_corr_df.empty:
        st.warning(f"Found {len(high_corr_df)} highly correlated feature pair(s):")
        st.dataframe(high_corr_df.round(4), use_container_width=True)
    else:
        st.success(f"No feature pairs found with correlation >= {threshold}")

# ============================================================
# SECTION 7: INSIGHTS & TAKEAWAYS
# ============================================================
elif page == "7. Key Insights & Takeaways":
    st.title("💡 7. Summary Insights & Modeling Recommendations")
    
    st.markdown("""
    ### 📊 Key Observations & Modeling Strategy
    
    * **Target Class Imbalance (~88.7% 'no' vs ~11.3% 'yes'):**
      * Heavy class skew requires cost-sensitive learning (`class_weight='balanced'`), undersampling, or SMOTE during model training.
      * Avoid using standard Accuracy as your evaluation metric; prioritize PR-AUC, ROC-AUC, or Recall depending on business goals.

    * **High Risk of Data Leakage (`duration`):**
      * `duration` (call time) is strongly predictive of outcome success, but **it is not known before a call is made**. 
      * For realistic predictive modeling prior to customer contact, `duration` should be excluded from feature inputs.

    * **Multicollinearity in Economic Indicators:**
      * `euribor3m`, `nr.employed`, and `emp.var.rate` exhibit extreme positive pairwise correlation ($r > 0.90$).
      * Apply feature reduction (e.g., dropping redundant features, Ridge regularization, or PCA) to avoid instability in linear models.

    * **Semantic Categorical Encoding:**
      * Treat `pdays = 999` explicitly as an indicator flag (`never_contacted_before = 1`) rather than a linear metric.
      * Retain `'unknown'` values as distinct categorical levels rather than blindly imputing them.
    """)