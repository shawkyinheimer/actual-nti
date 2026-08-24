import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# -----------------------------------------------------------------------------
# 1. PAGE CONFIG & CUSTOM CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Perfume Store Procurement & Sales Analytics",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Executive Dark-ish Theme CSS
st.markdown("""
    <style>
    /* Global background adjustments */
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }

    /* KPI Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e222d 0%, #171a23 100%);
        border: 1px solid #2d313e;
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #4eb8dd;
        margin-top: 4px;
    }
    .metric-sub-alert {
        font-size: 0.8rem;
        color: #ff6b6b;
        margin-top: 4px;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #1a1e29;
        border-radius: 6px 6px 0px 0px;
        color: #a0a6b2;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #262c3a !important;
        color: #ffffff !important;
        border-bottom: 2px solid #00d2ff;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & CLEANING PIPELINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    # Load dataset
    df = pd.read_csv("/home/shawky/Documents/nti/perfumes/testing/2026.csv")

    # Safe date parsing
    df['Invoice Date'] = pd.to_datetime(df['Invoice Date'], errors='coerce')

    # Numeric conversions
    numeric_cols = [
        'Quantity (الكمية)', 'Unit Size (الوحدة)', 'Public Price (سعر الجمهور)',
        'Unit Price (السعر)', 'Total Amount (القيمة)', 'Invoice Total (الإجمالي)',
        'Invoice Paid (المدفوع)', 'Invoice Remaining (المتبقي)'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Derive product categories
    # Oils have Unit Size > 0 (measured in grams), Accessories/Pieces have Unit Size = 0
    df['Category'] = np.where(df['Unit Size (الوحدة)'] > 0, 'Oil/ml (Category B)', 'Piece/Accessory (Category A)')

    # Clean Item Names
    df['Clean Item Name'] = df['Item Name (اسم الصنف)'].astype(str).str.strip()

    return df

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading '2026.csv': {e}. Please ensure the dataset file is in the same directory.")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SIDEBAR FILTERS
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Analysis Controls")
st.sidebar.markdown("---")

# Date Filter
min_date = df_raw['Invoice Date'].min()
max_date = df_raw['Invoice Date'].max()

if pd.notnull(min_date) and pd.notnull(max_date):
    start_date, end_date = st.sidebar.date_input(
        "Invoice Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    df_filtered = df_raw[(df_raw['Invoice Date'].dt.date >= start_date) & 
                         (df_raw['Invoice Date'].dt.date <= end_date)]
else:
    df_filtered = df_raw.copy()

# Category Filter
categories = ["All"] + list(df_filtered['Category'].unique())
selected_cat = st.sidebar.selectbox("Filter Category", categories)

if selected_cat != "All":
    df_filtered = df_filtered[df_filtered['Category'] == selected_cat]

# Oil Code Multiselect
all_oil_codes = sorted(df_raw[df_raw['Category'] == 'Oil/ml (Category B)']['Code (الكود)'].unique().tolist())
selected_oil_codes = st.sidebar.multiselect("Filter Specific Oil Codes", options=all_oil_codes)

if selected_oil_codes:
    df_filtered = df_filtered[df_filtered['Code (الكود)'].isin(selected_oil_codes)]

st.sidebar.markdown("---")
st.sidebar.info("📌 **Data Context:** Procurement logs indicate oils are billed by weight in grams (`Quantity = 0`, `Unit Size = grams`).")

# -----------------------------------------------------------------------------
# 4. MAIN DASHBOARD LAYOUT & METRICS
# -----------------------------------------------------------------------------
st.title("📊 Retail Perfume Store Analytics")
st.markdown("Procurement, Cash Flow, and Inventory Performance Dashboard")

# Calculate Unique Invoice Metrics (to prevent double-counting invoice-level totals)
unique_invoices = df_filtered.drop_duplicates(subset=['Invoice Number'])

total_revenue = df_filtered['Total Amount (القيمة)'].sum()
total_paid = unique_invoices['Invoice Paid (المدفوع)'].sum()
total_remaining = unique_invoices['Invoice Remaining (المتبقي)'].sum()
total_invoices = unique_invoices['Invoice Number'].nunique()

total_grams = df_filtered[df_filtered['Category'] == 'Oil/ml (Category B)']['Unit Size (الوحدة)'].sum()
total_pieces = df_filtered[df_filtered['Category'] == 'Piece/Accessory (Category A)']['Quantity (الكمية)'].sum()

# Display KPI Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Gross Value</div>
            <div class="metric-value">{total_revenue:,.2f} <span style="font-size:1rem;">EGP</span></div>
            <div class="metric-sub">{len(df_filtered)} Line Items</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Amount Paid</div>
            <div class="metric-value" style="color:#00e676;">{total_paid:,.2f} <span style="font-size:1rem;">EGP</span></div>
            <div class="metric-sub">{((total_paid/total_revenue)*100 if total_revenue > 0 else 0):.1f}% Collected</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Outstanding Balance</div>
            <div class="metric-value" style="color:#ff5252;">{total_remaining:,.2f} <span style="font-size:1rem;">EGP</span></div>
            <div class="metric-sub-alert">{((total_remaining/total_revenue)*100 if total_revenue > 0 else 0):.1f}% Debt Ratio</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Oil Stock Procured</div>
            <div class="metric-value">{total_grams:,.0f} <span style="font-size:1rem;">Grams</span></div>
            <div class="metric-sub">Category B Items</div>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Invoices</div>
            <div class="metric-value">{total_invoices}</div>
            <div class="metric-sub">{total_pieces:,.0f} Piece Units</div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. DASHBOARD TABS
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Executive Overview", 
    "💳 Financial & Cash Flow", 
    "🧪 Product & Oil Analysis", 
    "📁 Raw Data Explorer"
])

# ---------------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ---------------------------------------------------------
with tab1:
    col_a, col_b = st.columns([12, 8])

    with col_a:
        st.subheader("Monthly Revenue & Procurement Trends")
        df_filtered['YearMonth'] = df_filtered['Invoice Date'].dt.to_period('M').astype(str)
        monthly_trend = df_filtered.groupby(['YearMonth', 'Category'])['Total Amount (القيمة)'].sum().reset_index()

        fig_trend = px.bar(
            monthly_trend, 
            x='YearMonth', 
            y='Total Amount (القيمة)', 
            color='Category',
            barmode='group',
            color_discrete_map={'Oil/ml (Category B)': '#00d2ff', 'Piece/Accessory (Category A)': '#7000ff'},
            labels={'YearMonth': 'Month', 'Total Amount (القيمة)': 'Spend / Revenue (EGP)'},
            template="plotly_dark"
        )
        fig_trend.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_b:
        st.subheader("Category Revenue Split")
        cat_split = df_filtered.groupby('Category')['Total Amount (القيمة)'].sum().reset_index()
        fig_pie = px.pie(
            cat_split, 
            names='Category', 
            values='Total Amount (القيمة)',
            hole=0.45,
            color_discrete_sequence=['#00d2ff', '#7000ff'],
            template="plotly_dark"
        )
        fig_pie.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)
# ---------------------------------------------------------
# TAB 2: FINANCIAL & CASH FLOW HEALTH
# ---------------------------------------------------------
with tab2:
    st.subheader("Invoice Credit Bottleneck Analysis")
    st.info("💡 **Cash Flow Finding:** Over 50% of the shop's procurement invoicing rests on supplier revolving credit lines (*آجل*).")

    col_f1, col_f2 = st.columns([12, 8])

    with col_f1:
        # Per Invoice Paid vs Remaining Breakdown
        inv_summary = unique_invoices[['Invoice Number', 'Invoice Paid (المدفوع)', 'Invoice Remaining (المتبقي)', 'Invoice Total (الإجمالي)']].copy()
        inv_summary['Invoice Code'] = "Inv #" + inv_summary['Invoice Number'].astype(str)

        fig_inv = go.Figure()
        fig_inv.add_trace(go.Bar(
            x=inv_summary['Invoice Code'],
            y=inv_summary['Invoice Paid (المدفوع)'],
            name='Paid Amount (EGP)',
            marker_color='#00e676'
        ))
        fig_inv.add_trace(go.Bar(
            x=inv_summary['Invoice Code'],
            y=inv_summary['Invoice Remaining (المتبقي)'],
            name='Remaining Debt (EGP)',
            marker_color='#ff5252'
        ))
        fig_inv.update_layout(
            barmode='stack',
            template="plotly_dark",
            height=400,
            xaxis_title="Invoice Identifier",
            yaxis_title="Amount (EGP)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    with col_f2:
        # Gauge chart for Debt Ratio
        debt_ratio = (total_remaining / total_revenue * 100) if total_revenue > 0 else 0
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=debt_ratio,
            number={'suffix': "%"},
            title={'text': "Outstanding Debt Ratio"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#ff5252"},
                'steps': [
                    {'range': [0, 30], 'color': "#1e3a29"},
                    {'range': [30, 50], 'color': "#3d3b1e"},
                    {'range': [50, 100], 'color': "#4a1e1e"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': 50.0
                }
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)
# ---------------------------------------------------------
# TAB 3: PRODUCT & OIL ANALYSIS
# ---------------------------------------------------------
with tab3:
    st.subheader("Oil Procurement Concentration & Top Performers")

    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("**Top 10 Procured Oils by Spend**")
        oils_df = df_filtered[df_filtered['Category'] == 'Oil/ml (Category B)']
        top_oils = oils_df.groupby('Clean Item Name')['Total Amount (القيمة)'].sum().nlargest(10).reset_index()

        fig_top_oils = px.bar(
            top_oils, 
            x='Total Amount (القيمة)', 
            y='Clean Item Name',
            orientation='h',
            color='Total Amount (القيمة)',
            color_continuous_scale='Viridis',
            template="plotly_dark",
            labels={'Total Amount (القيمة)': 'Spend (EGP)', 'Clean Item Name': 'Oil Name'}
        )
        fig_top_oils.update_layout(yaxis={'categoryorder': 'total ascending'}, height=400)
        st.plotly_chart(fig_top_oils, use_container_width=True)

    with col_p2:
        st.markdown("**Purchase Frequency vs. Spend Scatter**")
        oil_freq = oils_df.groupby(['Code (الكود)', 'Clean Item Name']).agg(
            total_spend=('Total Amount (القيمة)', 'sum'),
            total_grams=('Unit Size (الوحدة)', 'sum'),
            frequency=('Invoice Number', 'nunique')
        ).reset_index()

        fig_scatter = px.scatter(
            oil_freq,
            x='frequency',
            y='total_spend',
            size='total_grams',
            hover_name='Clean Item Name',
            color='total_spend',
            color_continuous_scale='Cividis',
            template="plotly_dark",
            labels={'frequency': 'Purchase Frequency (Invoices)', 'total_spend': 'Total Spend (EGP)', 'total_grams': 'Grams Procured'}
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.markdown("**Top Hardware & Accessories (Pieces)**")

    pieces_df = df_filtered[df_filtered['Category'] == 'Piece/Accessory (Category A)']
    if not pieces_df.empty:
        top_pieces = pieces_df.groupby('Clean Item Name').agg(
            units_sold=('Quantity (الكمية)', 'sum'),
            revenue=('Total Amount (القيمة)', 'sum')
        ).reset_index().sort_values(by='units_sold', ascending=False).head(8)

        fig_pieces = px.bar(
            top_pieces,
            x='Clean Item Name',
            y='units_sold',
            text='units_sold',
            color='revenue',
            template="plotly_dark",
            labels={'units_sold': 'Units Sold / Procured', 'Clean Item Name': 'Item Name', 'revenue': 'Revenue (EGP)'}
        )
        fig_pieces.update_layout(height=350)
        st.plotly_chart(fig_pieces, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: RAW DATA EXPLORER
# ---------------------------------------------------------
with tab4:
    st.subheader("Cleaned Dataset Explorer")

    # Search box
    search_term = st.text_input("🔍 Search by Item Name or Code", "")

    display_df = df_filtered.copy()
    if search_term:
        display_df = display_df[
            display_df['Clean Item Name'].str.contains(search_term, case=False, na=False) |
            display_df['Code (الكود)'].astype(str).str.contains(search_term, na=False)
        ]

    # Selected columns display
    show_cols = [
        'Invoice Number', 'Invoice Date', 'Code (الكود)', 'Clean Item Name', 
        'Category', 'Quantity (الكمية)', 'Unit Size (الوحدة)', 'Unit Price (السعر)', 
        'Total Amount (القيمة)', 'Invoice Paid (المدفوع)', 'Invoice Remaining (المتبقي)'
    ]

    st.dataframe(
        display_df[show_cols],
        use_container_width=True,
        height=450
    )

    # CSV Download Button
    csv_data = display_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name="cleaned_perfume_store_data.csv",
        mime="text/csv"
    )
