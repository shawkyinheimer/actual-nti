# =====================================================================
# rfm_features.py — transaction tables -> scaled RFM feature matrix
# Depends on: preprocessing.py (sales, cancels tables)
# =====================================================================
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

FEATURES = ["Recency", "Frequency", "Monetary"]


# ---------------------------------------------------------------------
# 0. NAMED TRANSFORM FUNCS — must live at MODULE level.
#    A lambda inside make_transformer pickles as
#    'make_transformer.<locals>.<lambda>' -> joblib.dump fails.
# ---------------------------------------------------------------------
def clip_log1p(X: np.ndarray) -> np.ndarray:
    """Named (picklable) replacement for the lambda: clip negatives, then log1p."""
    return np.log1p(np.clip(X, 0, None))


# ---------------------------------------------------------------------
# 1. RFM BUILD — identified customers only, NET of returns (review #1, #2)
# ---------------------------------------------------------------------
def build_rfm(sales: pd.DataFrame, cancels: pd.DataFrame,
              snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Customer-level RFM net of returns.

    - Guests (HasCustomerID=False) are excluded here, NOT imputed.
    - Monetary = gross purchases − RevenueLost from cancellations (NET).
    - Frequency/Recency come from purchase invoices only (returns are a
      monetary correction, not a purchase event — flip if you disagree).
    """
    cust_sales   = sales[sales["HasCustomerID"]]
    cust_cancels = cancels[cancels["HasCustomerID"]]
    snapshot = snapshot or (sales["InvoiceDate"].max() + pd.Timedelta(days=1))

    pos = (cust_sales.groupby("CustomerID")
           .agg(last_purchase=("InvoiceDate", "max"),
                Frequency=("InvoiceNo", "nunique"),
                GrossMonetary=("TotalRevenue", "sum")))
    ret = (cust_cancels.groupby("CustomerID")
           .agg(ReturnsMonetary=("RevenueLost", "sum"),   # positive £ returned
                ReturnEvents=("InvoiceNo", "nunique")))

    rfm = pos.join(ret, how="outer")
    returns_only = rfm["Frequency"].isna()          # bought pre-window, returned in-window
    rfm = rfm.dropna(subset=["Frequency", "last_purchase"])  # unmeasurable -> drop
    rfm = rfm.fillna({"ReturnsMonetary": 0.0, "ReturnEvents": 0})

    rfm["Recency"]  = (snapshot - rfm["last_purchase"]).dt.days
    rfm["Monetary"] = rfm["GrossMonetary"] - rfm["ReturnsMonetary"]   # NET £
    rfm["HadFullReturn"] = rfm["Monetary"] <= 0        # <- the rows that break log1p
    rfm = rfm.drop(columns="last_purchase")
    rfm.index = rfm.index.astype(int)

    print(f"RFM customers: {len(rfm):,} | dropped returns-only (pre-window buyers): "
          f"{int(returns_only.sum())} | full-return customers (Monetary<=0): "
          f"{int(rfm['HadFullReturn'].sum())}  <- clipped under log1p, native under Yeo-Johnson")
    return rfm


# ---------------------------------------------------------------------
# 2. TRANSFORM + SCALE — Yeo-Johnson replaces log1p+StandardScaler
# ---------------------------------------------------------------------
def make_transformer(method: str = "yeo-johnson"):
    """yeo-johnson: fitted power transform + built-in standardization.
       log1p:      classic fallback (clip negatives — the hack YJ avoids)."""
    if method == "yeo-johnson":
        return PowerTransformer(method="yeo-johnson", standardize=True)  # both steps in one
    if method == "log1p":
        return Pipeline([
            ("log",   FunctionTransformer(clip_log1p)),  # module-level named fn → picklable
            ("scale", StandardScaler()),
        ])
    raise ValueError(f"Unknown method: {method}")


def make_feature_matrix(rfm: pd.DataFrame, method: str = "yeo-johnson",
                        save_path: str | None = None
                        ) -> tuple[np.ndarray, object]:
    """Fit transform on THIS sample and return (X_scaled, fitted_transformer).
       save_path: persist with joblib — REUSE for scoring new customers, never refit."""
    tr = make_transformer(method)
    Xt = tr.fit_transform(rfm[FEATURES].to_numpy())
    if save_path:
        joblib.dump(tr, save_path)
        print(f"Transformer saved -> {save_path} (load with joblib.load to score new data)")
    return Xt, tr


# ---------------------------------------------------------------------
# 3. EVIDENCE — compare both transforms so the choice is data-backed
# ---------------------------------------------------------------------
def compare_transforms(rfm: pd.DataFrame) -> pd.DataFrame:
    """Skewness of each feature: raw vs log1p+scale vs yeo-johnson.
       ( Closer to 0 = closer to Gaussian = friendlier to distance algorithms. )"""
    rows = {}
    for name, tr in [("log1p+scale", make_transformer("log1p")),
                     ("yeo-johnson", make_transformer("yeo-johnson"))]:
        Xt = pd.DataFrame(tr.fit_transform(rfm[FEATURES]), columns=FEATURES)
        rows[name] = Xt.skew()
    out = pd.DataFrame({"raw": rfm[FEATURES].skew(), **rows}).round(3)
    print("\nSkewness after transform (lower = better):")
    print(out)
    return out


# ---------------------------------------------------------------------
# 4. CLUSTER PROFILING — segment table back in £ / days / orders
# ---------------------------------------------------------------------
def profile_clusters(rfm: pd.DataFrame, labels: np.ndarray,
                     transformer=None) -> pd.DataFrame:
    """Segment table in RAW units (£, days, order counts) for business reads.
       Robust to HDBSCAN's -1 noise label (np.bincount would crash on it)."""
    lab = np.asarray(labels)
    prof = rfm.groupby(lab)[FEATURES].mean().round(1)
    prof["customers"]  = pd.Series(lab).value_counts().reindex(prof.index)
    prof["rev_share_%"] = (100 * rfm.groupby(lab)["Monetary"].sum()
                           / rfm["Monetary"].sum()).round(1)
    return prof.sort_values("Monetary", ascending=False)


if __name__ == "__main__":                       # smoke test
    from retail_preprocessing import run_pipeline
    sales, cancels, _ = run_pipeline(verbose=False)

    rfm = build_rfm(sales, cancels)
    compare_transforms(rfm)

    Xt, tr = make_feature_matrix(rfm, method="yeo-johnson", save_path="yj_transformer.joblib")
    print(f"\nFeature matrix: {Xt.shape}, mean≈0, std≈1 per column ✓")
