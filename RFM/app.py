import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from ucimlrepo import fetch_ucirepo

st.set_page_config(page_title="Online Retail EDA", page_icon="🛒", layout="wide")
sns.set_theme(style="whitegrid", palette="deep")

SERVICE_CODES = {
    "POST", "DOT", "DOTCOM POSTAGE", "M", "MANUAL", "BANK CHARGES",
    "C2", "CRUK", "D", "S", "B", "PADS", "AMAZONFEE", "ADJUST",
}
DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]

COL_ALIASES = {
    "invoice": "InvoiceNo",      "invoiceno": "InvoiceNo",
    "stockcode": "StockCode",    "stock_code": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate","invoice_date": "InvoiceDate",
    "date": "InvoiceDate",
    "unitprice": "UnitPrice",    "price": "UnitPrice",
    "customerid": "CustomerID",  "customer_id": "CustomerID",
    "customer id": "CustomerID",
    "country": "Country",
}
REQUIRED = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate",
            "UnitPrice", "Description"}


# ---------------------------------------------------------------------
# DATA LAYER (cached — heavy work runs only once)
# ---------------------------------------------------------------------
@st.cache_data(show_spinner="⬇️ Fetching dataset from UCI (first run ≈ 40 s, then cached)…")
def load_raw() -> pd.DataFrame:
    """ucimlrepo splits ids (InvoiceNo, StockCode) from features — stitch back."""
    ds = fetch_ucirepo(id=352)
    parts = []
    for name in ("ids", "features", "targets"):
        part = getattr(ds.data, name, None)
        if isinstance(part, pd.DataFrame) and part.shape[1] > 0:
            parts.append(part)
    if not parts:
        raise RuntimeError("ucimlrepo returned no data — try: pip install -U ucimlrepo")
    return parts[0] if len(parts) == 1 else pd.concat(parts, axis=1)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    key = lambda c: c.lower().replace("-", "_").strip()
    ren = {c: COL_ALIASES[key(c)] for c in df.columns
           if key(c) in COL_ALIASES and c != COL_ALIASES[key(c)]}
    if ren:
        df = df.rename(columns=ren)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Required columns not found: {missing}\n"
                         f"Available: {list(df.columns)}")
    return df


def parse_invoice_dates(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%d/%m/%Y %H:%M"):
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return parsed
    return pd.to_datetime(s, errors="coerce")


@st.cache_data(show_spinner="🧹 Cleaning & feature engineering…")
def preprocess(raw: pd.DataFrame):
    df = raw.copy()
    info = {"raw_rows": len(df), "dupes": int(df.duplicated().sum())}
    df = df.drop_duplicates()

    for col in ["InvoiceNo", "StockCode", "Country"]:
        df[col] = df[col].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["InvoiceDate"] = parse_invoice_dates(df["InvoiceDate"])
    n_bad_dates = int(df["InvoiceDate"].isna().sum())
    df = df.dropna(subset=["InvoiceDate"])

    df["Quantity"]   = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"]  = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["HasCustomerID"] = df["CustomerID"].notna()

    code2desc = (df[df["Description"] != "UNKNOWN"]
                 .groupby("StockCode")["Description"]
                 .agg(lambda s: s.mode().iat[0]))
    m = df["Description"] == "UNKNOWN"
    df.loc[m, "Description"] = df.loc[m, "StockCode"].map(code2desc)
    df["Description"] = df["Description"].fillna("UNKNOWN")

    df["IsCancelled"] = df["InvoiceNo"].str.upper().str.startswith("C")
    cancels = df[df["IsCancelled"] | (df["Quantity"] < 0)].copy()
    cancels["ReturnQty"]   = -cancels["Quantity"]
    cancels["RevenueLost"] = cancels["ReturnQty"] * cancels["UnitPrice"].fillna(0)

    sales = df[~df["IsCancelled"] & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    sales["IsServiceLine"] = sales["StockCode"].str.upper().isin(SERVICE_CODES)

    dt = sales["InvoiceDate"]
    sales["TotalRevenue"] = sales["Quantity"] * sales["UnitPrice"]
    sales["YearMonth"] = dt.dt.to_period("M")
    sales["DayOfWeek"] = dt.dt.dayofweek
    sales["DayName"]   = dt.dt.day_name()
    sales["Hour"]      = dt.dt.hour

    info.update({
        "bad_dates": n_bad_dates,
        "date_min": df["InvoiceDate"].min(),
        "date_max": df["InvoiceDate"].max(),
        "n_sales": len(sales),
        "n_cancels": len(cancels),
        "anon_share": sales.loc[~sales["HasCustomerID"], "TotalRevenue"].sum()
                       / sales["TotalRevenue"].sum(),
    })
    return sales, cancels, info


# ---------------------------------------------------------------------
# PLOT BUILDERS (return figs — Streamlit renders them)
# ---------------------------------------------------------------------
def show(fig):
    st.pyplot(fig, bbox_inches="tight")
    plt.close(fig)


def fig_distributions(sales: pd.DataFrame):
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    for i, col in enumerate(["Quantity", "UnitPrice", "TotalRevenue"]):
        sns.histplot(sales[col], bins=100, ax=axes[i, 0], color="steelblue")
        axes[i, 0].set_title(f"{col} — raw | skew={sales[col].skew():.1f}")
        p99 = sales[col].quantile(0.99)
        sns.histplot(sales.loc[sales[col] <= p99, col], bins=60,
                     ax=axes[i, 1], color="seagreen")
        axes[i, 1].set_title(f"{col} — clipped at P99 = {p99:,.2f}")
    fig.tight_layout()
    return fig


def fig_top_products(sales: pd.DataFrame, n: int = 10):
    prod = (sales[~sales["IsServiceLine"]]
            .groupby("StockCode")
            .agg(Description=("Description", "first"),
                 UnitsSold=("Quantity", "sum"),
                 Revenue=("TotalRevenue", "sum")))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, col, title in [(axes[0], "UnitsSold", f"Top {n} by UNITS"),
                           (axes[1], "Revenue",   f"Top {n} by REVENUE (£)")]:
        top = prod.nlargest(n, col).iloc[::-1]
        ax.barh(top["Description"].astype(str).str[:38], top[col], color="teal")
        ax.set_title(title)
        ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig


def fig_countries(sales: pd.DataFrame):
    ctry = (sales.groupby("Country")["TotalRevenue"].sum()
                 .sort_values(ascending=False))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    (ctry / 1e3).head(10).iloc[::-1].plot.barh(ax=axes[0], color="slateblue")
    axes[0].set(title="Top 10 countries by revenue (£k)")
    no_uk = ctry.drop(index="United Kingdom", errors="ignore")
    (no_uk / 1e3).head(10).iloc[::-1].plot.barh(ax=axes[1], color="coral")
    axes[1].set(title="Top 10 EXCL. UK (£k)")
    fig.tight_layout()
    return fig, ctry


def fig_monthly(sales: pd.DataFrame):
    monthly = (sales.groupby("YearMonth")
                    .agg(Revenue=("TotalRevenue", "sum"),
                         Orders=("InvoiceNo", "nunique")))
    fig, ax1 = plt.subplots(figsize=(13, 4.5))
    ax1.plot(monthly.index.astype(str), monthly["Revenue"] / 1e3,
             marker="o", color="navy")
    ax2 = ax1.twinx()
    ax2.plot(monthly.index.astype(str), monthly["Orders"], ls="--",
             color="darkorange", alpha=0.7)
    ax1.set(title="Monthly revenue (£k, navy) & orders (dashed orange)",
            ylabel="Revenue (£k)")
    ax1.tick_params(axis="x", rotation=45)
    if str(monthly.index[-1]) == "2011-12":     # partial final month
        ax1.annotate("Partial month\n(ends 09-Dec-2011)",
                     xy=(len(monthly) - 1, monthly["Revenue"].iloc[-1] / 1e3),
                     xytext=(len(monthly) - 4, monthly["Revenue"].max() / 1e3 * 0.6),
                     arrowprops=dict(arrowstyle="->", color="crimson"),
                     color="crimson", fontsize=9)
    fig.tight_layout()
    return fig, monthly


def fig_weekday_hour(sales: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    (sales.groupby("DayName")["TotalRevenue"].sum().reindex(DOW_ORDER) / 1e3) \
        .plot.bar(ax=axes[0], color="seagreen")
    axes[0].set(title="Revenue by day of week (£k)")
    (sales.groupby("Hour")["TotalRevenue"].sum() / 1e3) \
        .plot.bar(ax=axes[1], color="indianred")
    axes[1].set(title="Revenue by hour (£k)")
    fig.tight_layout()

    pivot = (sales.pivot_table(index="DayName", columns="Hour",
                               values="InvoiceNo", aggfunc="nunique")
                  .reindex(index=DOW_ORDER, columns=range(6, 21)))
    fig2, ax = plt.subplots(figsize=(13, 3.5))
    sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.3, ax=ax,
                cbar_kws={"label": "Unique invoices"})
    ax.set_title("Order density — day × hour (Saturday is empty in the raw data)")
    fig2.tight_layout()
    return fig, fig2


def fig_frequency(sales: pd.DataFrame):
    freq = (sales[sales["HasCustomerID"]]
            .groupby("CustomerID")["InvoiceNo"].nunique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.histplot(freq.clip(upper=15), discrete=True, ax=axes[0], color="teal")
    axes[0].set(title="Invoices per customer (15+ clipped)")
    repeat = (freq > 1).mean() * 100
    axes[1].pie([repeat, 100 - repeat],
                labels=[f"Repeat\n{repeat:.1f}%", f"One-time\n{100-repeat:.1f}%"],
                colors=["seagreen", "lightgray"], autopct="%1.1f%%")
    axes[1].set_title("Repeat-purchase rate")
    fig.tight_layout()
    return fig


def fig_pareto(sales: pd.DataFrame):
    rev = (sales[sales["HasCustomerID"]]
           .groupby("CustomerID")["TotalRevenue"].sum()
           .sort_values(ascending=False))
    cum = rev.cumsum() / rev.sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(1, len(rev) + 1) / len(rev) * 100, cum * 100)
    ax.axhline(80, ls="--", color="red")
    ax.set(xlabel="Cumulative % of customers (ranked by spend)",
           ylabel="Cumulative % of revenue",
           title="Customer revenue concentration (Pareto)")
    fig.tight_layout()
    n80 = int((cum < 0.80).sum()) + 1
    return fig, f"**{n80} customers ({100*n80/len(rev):.1f}%) generate 80% of revenue.**"


def build_rfm(sales: pd.DataFrame) -> pd.DataFrame:
    cust = sales[sales["HasCustomerID"]]
    snapshot = sales["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = (cust.groupby("CustomerID")
               .agg(Recency=("InvoiceDate", lambda s: (snapshot - s.max()).days),
                    Frequency=("InvoiceNo", "nunique"),
                    Monetary=("TotalRevenue", "sum")))
    rfm.index = rfm.index.astype(int)
    rfm["R"] = pd.qcut(rfm["Recency"].rank(method="first"), 5,
                       labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5,
                       labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["Monetary"].rank(method="first"), 5,
                       labels=[1, 2, 3, 4, 5]).astype(int)
    return rfm


def label_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    rfm = rfm.copy()
    conds = [
        (rfm["R"] >= 4) & (rfm["F"] >= 4),
        (rfm["R"] >= 3) & (rfm["F"] >= 4),
        (rfm["R"] >= 4) & (rfm["F"] <= 3),
        (rfm["R"] == 3),
        (rfm["R"] <= 2) & (rfm["F"] >= 3),
        (rfm["R"] <= 2) & (rfm["F"] <= 2),
    ]
    labels = ["Champions", "Loyal Customers", "New / Promising",
              "Needs Attention", "At Risk", "Hibernating"]
    rfm["Segment"] = np.select(conds, labels, default="Others")
    return rfm


def fig_segments(rfm: pd.DataFrame):
    seg = (rfm.groupby("Segment")
              .agg(Customers=("Monetary", "size"), Revenue=("Monetary", "sum"))
              .sort_values("Revenue", ascending=False))
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    seg["Customers"].plot.barh(ax=axes[0], color="slateblue")
    axes[0].set_title("Customers per segment")
    (seg["Revenue"] / 1e3).plot.barh(ax=axes[1], color="goldenrod")
    axes[1].set_title("Revenue per segment (£k)")
    for ax in axes:
        ax.invert_yaxis()
    fig.tight_layout()
    return fig, seg


def fig_returns(cancels: pd.DataFrame):
    top = (cancels.groupby("Description")["ReturnQty"].sum()
                  .sort_values(ascending=False).head(10).iloc[::-1])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(top.index.astype(str).str[:38], top.values, color="tomato")
    ax.set_title("Top 10 returned products (units)")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# APP LAYOUT
# ---------------------------------------------------------------------
st.title("🛒 Online Retail — Exploratory Data Analysis")
st.caption("UCI Dataset 352 · Dec 2010 – Dec 2011 · all figures in £")

# --- load (cached) ---
try:
    raw = standardize_columns(load_raw())
    sales, cancels, info = preprocess(raw)
except (ValueError, RuntimeError) as e:
    st.error(f"Data loading failed: {e}")
    st.stop()

with st.expander("🧹 Data cleaning log"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw rows", f"{info['raw_rows']:,}")
    c1.metric("Exact duplicates dropped", f"{info['dupes']:,}")
    c2.metric("Sales rows", f"{info['n_sales']:,}")
    c2.metric("Cancel/return rows", f"{info['n_cancels']:,}")
    c3.metric("Date range", f"{info['date_min']:%d %b %Y}",
              f"→ {info['date_max']:%d %b %Y}")
    c3.metric("Unparseable dates dropped", info["bad_dates"])
    c4.metric("Anonymous revenue share", f"{info['anon_share']:.1%}",
              "no CustomerID")

# --- sidebar filters ---
st.sidebar.header("🔎 Filters")
d_min, d_max = sales["InvoiceDate"].min().date(), sales["InvoiceDate"].max().date()
picked = st.sidebar.date_input("Date range", (d_min, d_max),
                               min_value=d_min, max_value=d_max)
d0, d1 = (picked if isinstance(picked, tuple) and len(picked) == 2
          else (picked, picked))

countries = st.sidebar.multiselect(
    "Countries", sorted(sales["Country"].unique()),
    default=sorted(sales["Country"].unique()))

if st.sidebar.button("🔄 Clear cache & refetch"):
    st.cache_data.clear()

# --- apply filters ---
mask = sales["InvoiceDate"].dt.date.between(d0, d1) & sales["Country"].isin(countries)
fsales = sales[mask]
if fsales.empty:
    st.warning("No sales match this filter combination — widen the date range or countries.")
    st.stop()

fcan_mask = cancels["InvoiceDate"].dt.date.between(d0, d1) & cancels["Country"].isin(countries)
fcancels = cancels[fcan_mask]

# --- KPI row ---
st.subheader("Key metrics (filtered)")
rev = fsales["TotalRevenue"].sum()
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total revenue", f"£{rev:,.0f}",
          f"{100 * rev / sales['TotalRevenue'].sum():.0f}% of dataset")
k2.metric("Invoices", f"{fsales['InvoiceNo'].nunique():,}")
k3.metric("Unique products", f"{fsales['StockCode'].nunique():,}")
k4.metric("Identified customers",
          f"{fsales.loc[fsales['HasCustomerID'], 'CustomerID'].nunique():,}")
k5.metric("Avg order value",
          f"£{fsales.groupby('InvoiceNo')['TotalRevenue'].sum().mean():,.2f}")

# --- tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📉 Distributions", "🏆 Products & Countries",
     "📅 Time Trends", "👥 Customers & RFM", "↩️ Cancellations"])

with tab1:
    st.caption("Numeric fields are extremely right-skewed — raw plots are unreadable, "
               "so each variable is also shown clipped at its 99th percentile.")
    show(fig_distributions(fsales))

with tab2:
    n = st.selectbox("How many top products?", [5, 10, 15, 20], index=1)
    show(fig_top_products(fsales, n))
    st.divider()
    fig_c, ctry = fig_countries(fsales)
    show(fig_c)
    st.dataframe(ctry.to_frame("Revenue (£)").round(0), use_container_width=True)
    st.download_button("⬇️ Download country summary (CSV)",
                       ctry.round(0).to_csv().encode(), "country_revenue.csv",
                       "text/csv")

with tab3:
    fig_m, monthly = fig_monthly(fsales)
    show(fig_m)
    show_tab = st.toggle("Show monthly table", value=False)
    if show_tab:
        st.dataframe(monthly.round(0), use_container_width=True)
    fig_w, fig_h = fig_weekday_hour(fsales)
    show(fig_w)
    show(fig_h)

with tab4:
    n_cust = fsales.loc[fsales["HasCustomerID"], "CustomerID"].nunique()
    if n_cust < 5:
        st.info("Fewer than 5 identified customers in this filter — RFM needs more data.")
    else:
        show(fig_frequency(fsales))
        fig_p, pareto_txt = fig_pareto(fsales)
        cA, cB = st.columns([2, 1])
        with cA:
            show(fig_p)
        with cB:
            st.markdown(pareto_txt)

        st.divider()
        st.subheader("RFM segmentation")
        st.caption(f"Snapshot = last purchase in selection + 1 day "
                   f"({fsales['InvoiceDate'].max():%d %b %Y}). "
                   "R/F/M scored 1–5 by quintile; R: 5 = most recent.")
        rfm = label_segments(build_rfm(fsales))
        fig_s, seg = fig_segments(rfm)
        show(fig_s)
        st.dataframe(seg.assign(RevSharePct=(
            100 * seg["Revenue"] / seg["Revenue"].sum()).round(1)),
            use_container_width=True)
        st.dataframe(rfm.head(500), use_container_width=True)
        st.download_button("⬇️ Download full RFM segments (CSV)",
                           rfm.to_csv().encode(), "rfm_segments.csv", "text/csv")

with tab5:
    st.caption("Cancellations (InvoiceNo starting with 'C' / negative quantities) "
               "are excluded from all sales metrics above and analysed here separately.")
    cA, cB = st.columns(2)
    cA.metric("Cancel/return rows", f"{len(fcancels):,}")
    cB.metric("Revenue lost (£)", f"{fcancels['RevenueLost'].sum():,.0f}")
    if not fcancels.empty:
        show(fig_returns(fcancels))
    else:
        st.info("No cancellations in this filter selection.")
