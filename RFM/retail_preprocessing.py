# ====================================================================
# preprocessing.py — Online Retail (UCI 352) preprocessing pipeline
# Usage (notebook):   from preprocessing import run_pipeline
#                     sales, cancels, log = run_pipeline()
# Usage (Streamlit):  wrap run_pipeline() in @st.cache_data
# =====================================================================
from __future__ import annotations

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
SERVICE_CODES = {  # non-product lines: excluded from product rankings only
    "POST", "DOT", "DOTCOM POSTAGE", "M", "MANUAL", "BANK CHARGES",
    "C2", "CRUK", "D", "S", "B", "PADS", "AMAZONFEE", "ADJUST",
}
COL_ALIASES = {  # any known lowercase variant -> canonical name
    "invoice": "InvoiceNo",      "invoiceno": "InvoiceNo",
    "stockcode": "StockCode",    "stock_code": "StockCode",
    "description": "Description",
    "quantity": "Quantity",
    "invoicedate": "InvoiceDate", "invoice_date": "InvoiceDate",
    "date": "InvoiceDate",
    "unitprice": "UnitPrice",    "price": "UnitPrice",
    "customerid": "CustomerID",  "customer_id": "CustomerID",
    "customer id": "CustomerID",
    "country": "Country",
}
REQUIRED = {"InvoiceNo", "StockCode", "Quantity", "InvoiceDate",
            "UnitPrice", "Description"}

# Ground truth for validation (dataset v1). tol = allowed relative drift.
EXPECTED = {
    "raw_rows":              (541_909, 0.00),
    "duplicates_dropped":    (5_268,   0.10),
    "sales_rows":            (524_878, 0.01),
    "cancels_rows":          (10_587,  0.01),
    "anon_revenue_share":    (0.165,   0.10),
}
DATE_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                "%m/%d/%Y %H:%M",   "%m/%d/%Y %H:%M:%S",
                "%d/%m/%Y %H:%M")


# ---------------------------------------------------------------------
# 1. LOAD — ucimlrepo splits ids from features; stitch them back
# ---------------------------------------------------------------------
def load_data(uci_id: int = 352) -> pd.DataFrame:
    """Fetch UCI 352 and re-join the ids/features/targets splits row-wise."""
    ds = fetch_ucirepo(id=uci_id)

    parts = []
    for name in ("ids", "features", "targets"):          # targets is empty here
        part = getattr(ds.data, name, None)
        if isinstance(part, pd.DataFrame) and part.shape[1] > 0:
            parts.append(part)
    if not parts:
        raise RuntimeError("ucimlrepo returned no data — try: pip install -U ucimlrepo")

    lens = {len(p) for p in parts}                       # splits must align row-wise
    if len(lens) != 1:
        raise RuntimeError(f"Row-count mismatch between ucimlrepo splits: {lens}")

    df = parts[0] if len(parts) == 1 else pd.concat(parts, axis=1)
    return df


# ---------------------------------------------------------------------
# 2. STANDARDIZE — alias-map column names; fail loudly if incomplete
# ---------------------------------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known column variants to canonical names; validate required set."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    key = lambda c: c.lower().replace("-", "_").strip()
    ren = {c: COL_ALIASES[key(c)] for c in df.columns
           if key(c) in COL_ALIASES and c != COL_ALIASES[key(c)]}
    df = df.rename(columns=ren)

    missing = REQUIRED - set(df.columns)
    if missing:                                          # never fail silently
        raise ValueError(f"Required columns not found: {missing}\n"
                         f"Available: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------
# 3. DATE PARSING — text -> datetime, explicit formats, coerced failures
# ---------------------------------------------------------------------
def parse_invoice_dates(s: pd.Series) -> pd.Series:
    """Parse InvoiceDate text (e.g. '12/1/2010 8:26'); NaT on failure."""
    if pd.api.types.is_datetime64_any_dtype(s):          # already parsed
        return s
    s = s.astype(str).str.strip()
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().mean() >= 0.9:                 # first format wins
            return parsed
    return pd.to_datetime(s, errors="coerce")            # last resort


# ---------------------------------------------------------------------
# 4. PREPROCESS — cleaning order is fixed; every count logged
# ---------------------------------------------------------------------
def preprocess(raw: pd.DataFrame, verbose: bool = True
               ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean raw transactions.

    Returns (sales, cancels, log). One row = one invoice line item.
    Cleaning order: dedupe -> strip -> parse dates -> coerce numerics
    -> flag HasCustomerID -> backfill Description -> split cancels
    -> filter sales -> flag service lines -> engineer features.
    """
    log: dict = {}
    df = raw.copy()
    log["raw_rows"] = len(df)

    # -- 4.1 exact duplicates: identical in all columns = double-logging
    before = len(df)
    df = df.drop_duplicates()
    log["duplicates_dropped"] = before - len(df)

    # -- 4.2 whitespace: 'POST ' and 'POST' must be one product
    for col in ["InvoiceNo", "StockCode", "Country"]:
        df[col] = df[col].astype(str).str.strip()
    df["Description"] = df["Description"].str.strip()    # .str keeps NaN

    # -- 4.3 dates: parse BEFORE anything downstream depends on them
    df["InvoiceDate"] = parse_invoice_dates(df["InvoiceDate"])
    log["unparseable_dates_dropped"] = int(df["InvoiceDate"].isna().sum())
    df = df.dropna(subset=["InvoiceDate"])

    # -- 4.4 numerics: coerce, drop rows that can't be classified
    df["Quantity"]  = pd.to_numeric(df["Quantity"],  errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    bad_num = df["Quantity"].isna() | df["UnitPrice"].isna()
    log["nonnumeric_qty_price_dropped"] = int(bad_num.sum())
    df = df[~bad_num]

    # -- 4.5 CustomerID: FLAG, never impute (an ID cannot be guessed)
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    df["HasCustomerID"] = df["CustomerID"].notna()
    df["CustomerID"] = df["CustomerID"].astype("Int64")  # clean nullable int
    log["rows_missing_customerid"] = int((~df["HasCustomerID"]).sum())

    # -- 4.6 Description backfill: same StockCode -> modal description
    code2desc = (df.dropna(subset=["Description"])
                   .groupby("StockCode")["Description"]
                   .agg(lambda s: s.mode().iat[0]))
    m = df["Description"].isna()
    df.loc[m, "Description"] = df.loc[m, "StockCode"].map(code2desc)
    log["descriptions_backfilled"] = int(m.sum())
    df["Description"] = df["Description"].fillna("UNKNOWN")

    # -- 4.7 split cancellations: 'C' invoice OR qty<0 -> separate asset
    df["IsCancelled"] = df["InvoiceNo"].str.upper().str.startswith("C")
    cancels = df[df["IsCancelled"] | (df["Quantity"] < 0)].copy()
    cancels["ReturnQty"]   = -cancels["Quantity"]        # positive units back
    cancels["RevenueLost"] = cancels["ReturnQty"] * cancels["UnitPrice"].fillna(0)
    log["cancels_rows"] = len(cancels)

    # -- 4.8 sales filter: exclude cancels AND qty<=0 / price<=0 lines
    sales_mask = ~df["IsCancelled"] & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)
    sales = df[sales_mask].copy()
    log["sales_rows"] = len(sales)          # ← add this
    # residual = adjustments/zero-price rows that are neither sales nor cancels
    log["dropped_adjustment_rows"] = len(df) - int(sales_mask.sum()) - len(cancels)

    # -- 4.9 service lines: flagged only (postage IS real revenue)
    sales["IsServiceLine"] = sales["StockCode"].str.upper().isin(SERVICE_CODES)

    # -- 4.10 feature engineering on sales
    dt = sales["InvoiceDate"]
    sales["TotalRevenue"] = sales["Quantity"] * sales["UnitPrice"]
    sales["Year"]      = dt.dt.year
    sales["Month"]     = dt.dt.month
    sales["YearMonth"] = dt.dt.to_period("M")            # Period[M]
    sales["DayOfWeek"] = dt.dt.dayofweek                 # 0=Mon .. 6=Sun
    sales["DayName"]   = dt.dt.day_name()
    sales["Hour"]      = dt.dt.hour
    sales["IsWeekend"] = sales["DayOfWeek"] >= 5

    # -- 4.11 anonymous revenue share (drives customer-analytics scope)
    total = sales["TotalRevenue"].sum()
    log["total_revenue"] = float(total)
    log["anon_revenue_share"] = float(
        sales.loc[~sales["HasCustomerID"], "TotalRevenue"].sum() / total)

    if verbose:
        _print_report(log, len(df), len(sales), len(cancels))
    return sales, cancels, log


# ---------------------------------------------------------------------
# 5. RECONCILIATION + VALIDATION — the auditor's view
# ---------------------------------------------------------------------
def _print_report(log: dict, n_unique: int, n_sales: int, n_cancels: int) -> None:
    """Print a row-reconciliation that MUST balance to zero."""
    accounted = n_sales + n_cancels + log["dropped_adjustment_rows"]
    residual = n_unique - accounted
    print("─" * 54)
    print(f" raw rows                    {log['raw_rows']:>12,}")
    print(f" − exact duplicates         -{log['duplicates_dropped']:>12,}")
    print(f" − unparseable dates        -{log['unparseable_dates_dropped']:>12,}")
    print(f" − non-numeric qty/price    -{log['nonnumeric_qty_price_dropped']:>12,}")
    print(f" = unique rows               {n_unique:>12,}")
    print(f"   ├── sales                 {n_sales:>12,}")
    print(f"   ├── cancels/returns       {n_cancels:>12,}")
    print(f"   └── dropped adjustments   {log['dropped_adjustment_rows']:>12,}")
    print(f" balance residual            {residual:>12,}  "
          f"{'✓ BALANCED' if residual == 0 else '✗ UNBALANCED — investigate!'}")
    print(f" anonymous revenue share     {log['anon_revenue_share']:>12.1%}")
    print("─" * 54)


def validate(log: dict, strict: bool = False) -> bool:
    """Compare counts vs known ground truth (± tolerance). strict=True raises."""
    ok = True
    print("\nValidation vs expected values:")
    for name, (exp, tol) in EXPECTED.items():
        got = log.get(name)
        if got is None:
            print(f"  {name:<24} MISSING from log"); ok = False; continue
        drift = abs(got - exp) / exp
        status = "PASS" if drift <= tol else "DRIFT"
        if status == "DRIFT":
            ok = False
        print(f"  {name:<24} got {got:>12,.3f} | expected {exp:>12,.0f}"
              f" | {status} (drift {drift:.1%})")
    if not ok and strict:
        raise AssertionError("Preprocessing validation failed — see report above.")
    return ok


# ---------------------------------------------------------------------
# 6. ORCHESTRATOR — one call, everything chained
# ---------------------------------------------------------------------
def run_pipeline(uci_id: int = 352, verbose: bool = True
                 ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """load -> standardize -> preprocess -> validate. Returns (sales, cancels, log)."""
    raw = standardize_columns(load_data(uci_id))
    sales, cancels, log = preprocess(raw, verbose=verbose)
    validate(log)
    return sales, cancels, log


if __name__ == "__main__":
    s, c, lg = run_pipeline()
    print(f"\nsales: {s.shape} | cancels: {c.shape}")
