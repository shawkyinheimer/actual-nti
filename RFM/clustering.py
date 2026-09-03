# =====================================================================
# clustering.py — K-Means vs Fuzzy C-Means vs HDBSCAN, tuned by Optuna
# Depends on: retail_preprocessing.py, rfm_features.py (log1p matrix)
# Install:    pip install optuna scikit-fuzzy
# =====================================================================
from __future__ import annotations

import json
import pickle
import joblib
import numpy as np
import pandas as pd
import optuna
import skfuzzy as fuzz
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                             davies_bouldin_score)

try:
    from sklearn.cluster import HDBSCAN            # scikit-learn >= 1.3
except ImportError:
    from hdbscan import HDBSCAN

from rfm_features import FEATURES, build_rfm, make_feature_matrix
from retail_preprocessing import run_pipeline

RNG = 42
optuna.logging.set_verbosity(optuna.logging.WARNING)

FCM_KW = dict(error=0.01, maxiter=200)             # faster, same convergence


# ---------------------------------------------------------------------
# 1. SCORING — one convention for all algorithms
# ---------------------------------------------------------------------
def _safe_scores(Xt: np.ndarray, labels: np.ndarray) -> dict | None:
    """Internal metrics on non-noise points; None if labeling is degenerate."""
    mask = labels >= 0                                   # HDBSCAN noise = -1
    uniq = np.unique(labels[mask])
    if uniq.size < 2 or mask.sum() < 3:
        return None
    counts = pd.Series(labels[mask]).value_counts()
    return {
        "silhouette":        silhouette_score(Xt[mask], labels[mask]),
        "calinski_harabasz": calinski_harabasz_score(Xt[mask], labels[mask]),
        "davies_bouldin":    davies_bouldin_score(Xt[mask], labels[mask]),
        "n_clusters":        int(uniq.size),
        "noise_%":           round(100 * (~mask).mean(), 1),
        "min_cluster_%":     round(100 * counts.min() / counts.sum(), 1),
    }


def _objective_score(s: dict) -> float:
    """Silhouette penalized by unclustered share (HDBSCAN noise; 0 for others)."""
    return s["silhouette"] * (1.0 - s["noise_%"] / 100.0)


# ---------------------------------------------------------------------
# 2. OPTUNA OBJECTIVES
# ---------------------------------------------------------------------
def objective_kmeans(trial: optuna.Trial, Xt: np.ndarray) -> float:
    k    = trial.suggest_int("k", 3, 10)
    init = trial.suggest_categorical("init", ["k-means++", "random"])
    labels = KMeans(n_clusters=k, init=init, n_init=3,
                    random_state=RNG).fit_predict(Xt)
    s = _safe_scores(Xt, labels)
    if s is None:
        raise optuna.TrialPruned()
    return _objective_score(s)


def objective_fcm(trial: optuna.Trial, Xt: np.ndarray) -> float:
    c = trial.suggest_int("c", 3, 10)
    m = trial.suggest_float("m", 1.2, 3.0)               # fuzziness exponent > 1
    np.random.seed(RNG)
    u = fuzz.cluster.cmeans(Xt.T, c=c, m=m, **FCM_KW)[1]  # (feat, pts) layout
    s = _safe_scores(Xt, u.argmax(axis=0))               # harden memberships
    if s is None:
        raise optuna.TrialPruned()
    return _objective_score(s)


def objective_hdbscan(trial: optuna.Trial, Xt: np.ndarray) -> float:
    params = dict(
        min_cluster_size=trial.suggest_int("min_cluster_size", 10, 100),
        min_samples=trial.suggest_int("min_samples", 3, 50),
        cluster_selection_epsilon=trial.suggest_float("cluster_selection_epsilon", 0.0, 0.5),
        cluster_selection_method=trial.suggest_categorical(
            "cluster_selection_method", ["eom", "leaf"]),
    )
    labels = HDBSCAN(**params).fit_predict(Xt)
    s = _safe_scores(Xt, labels)
    if s is None:
        raise optuna.TrialPruned()
    return _objective_score(s)


def tune(Xt: np.ndarray, n_trials: int = 20) -> dict[str, optuna.Study]:
    """Separate TPE study per algorithm; identical trial budget = fair."""
    studies = {}
    for name, obj in [("K-Means", objective_kmeans),
                      ("Fuzzy C-Means", objective_fcm),
                      ("HDBSCAN", objective_hdbscan)]:
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RNG),
                                    study_name=name)
        study.optimize(lambda t, o=obj: o(t, Xt),
                       n_trials=n_trials, show_progress_bar=True)
        print(f"{name:<14} best score = {study.best_value:.4f} | params = {study.best_params}")
        studies[name] = study
    return studies


# ---------------------------------------------------------------------
# 3. FINAL FITS — distinct param vars per algorithm (no reuse bugs)
# ---------------------------------------------------------------------
def _final_fits(Xt: np.ndarray, studies: dict) -> tuple[dict, dict]:
    p_km = studies["K-Means"].best_params
    km = KMeans(n_clusters=p_km["k"], init=p_km["init"], n_init=10,
                random_state=RNG).fit(Xt)

    p_fcm = studies["Fuzzy C-Means"].best_params
    np.random.seed(RNG)
    cntr, u = fuzz.cluster.cmeans(Xt.T, c=p_fcm["c"], m=p_fcm["m"], **FCM_KW)[:2]
    fcm_labels, fcm_conf = u.argmax(axis=0), u.max(axis=0)

    p_hdb = studies["HDBSCAN"].best_params
    hdb = HDBSCAN(**p_hdb).fit(Xt)                       # keys == kwargs, guaranteed

    models = {"K-Means":      {"labels": km.labels_,   "model": km},
              "Fuzzy C-Means": {"labels": fcm_labels, "cntr": cntr,
                                "m": p_fcm["m"], "confidence": fcm_conf},
              "HDBSCAN":       {"labels": hdb.labels_, "model": hdb}}
    params = {n: s.best_params for n, s in studies.items()}
    return models, params


# ---------------------------------------------------------------------
# 4. LEADERBOARD + business-readable segments
# ---------------------------------------------------------------------
def build_leaderboard(Xt: np.ndarray, label_sets: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = {name: _safe_scores(Xt, labels) for name, labels in label_sets.items()}
    lb = pd.DataFrame(rows).T
    lb["score"] = lb["silhouette"] * (1.0 - lb["noise_%"] / 100.0)
    return lb.sort_values("score", ascending=False).round(3)


def relabel_by_monetary(labels: np.ndarray, monetary: pd.Series) -> np.ndarray:
    """Map cluster ids -> 'A','B','C'... by descending mean Monetary (-1 -> 'Noise')."""
    order = (pd.DataFrame({"c": labels, "m": monetary})
             .query("c >= 0").groupby("c")["m"].mean()
             .sort_values(ascending=False).index.tolist())
    remap = {old: chr(65 + i) for i, old in enumerate(order)}
    return np.array([remap.get(int(l), "Noise") for l in labels])


def profile_clusters(rfm: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Winner's segments in RAW units (£, days, orders). A = highest Monetary."""
    df = rfm.copy()
    df["cluster"] = relabel_by_monetary(labels, rfm["Monetary"])
    prof = df.groupby("cluster")[FEATURES].mean().round(1)
    prof["customers"]  = df["cluster"].value_counts().reindex(prof.index)
    prof["rev_share_%"] = (100 * df.groupby("cluster")["Monetary"].sum()
                           / df["Monetary"].sum()).round(1)
    return prof.sort_values("Monetary", ascending=False)


# ---------------------------------------------------------------------
# 5. PERSISTENCE — survives unpicklable transformers (e.g. stray lambdas)
# ---------------------------------------------------------------------
def _save_artifacts(path: str, models: dict, params: dict,
                    transformer, features) -> None:
    payload = {"models": models, "params": params,
               "transformer": transformer, "features": features}
    try:
        joblib.dump(payload, path)
        return
    except (pickle.PicklingError, AttributeError, TypeError) as err:
        warn = str(err)

    # fallback 1: cloudpickle embeds lambdas/local functions by value
    try:
        import cloudpickle
        with open(path, "wb") as f:
            cloudpickle.dump(payload, f)
        print(f"NOTE: joblib.dump failed ({warn[:90]}...) — saved via cloudpickle; "
              "load with pickle.load(open(path, 'rb')).")
        return
    except Exception:
        pass

    # fallback 2: drop the transformer, save a rebuild spec instead
    payload["transformer"] = None
    payload["transformer_spec"] = {"method": "log1p", "features": list(features)}
    joblib.dump(payload, path)
    print(f"NOTE: transformer unpicklable ({warn[:90]}...) — saved rebuild spec; "
          "recreate via make_feature_matrix(rfm, method='log1p').")


# ---------------------------------------------------------------------
# 6. ORCHESTRATION
# ---------------------------------------------------------------------
def run_all(n_trials: int = 20) -> dict:
    sales, cancels, _ = run_pipeline(verbose=False)
    rfm = build_rfm(sales, cancels)
    Xt, transformer = make_feature_matrix(rfm, method="log1p")   # locked choice
    print(f"Feature matrix: {Xt.shape}\n")

    studies = tune(Xt, n_trials)
    models, best_params = _final_fits(Xt, studies)

    lb = build_leaderboard(Xt, {n: m["labels"] for n, m in models.items()})
    winner = lb.index[0]
    print("\nLEADERBOARD (higher score = better):"); print(lb)
    print(f"\nWinner by internal metrics: {winner}")
    print("(caveat: silhouette favors convex blobs -> structural advantage for "
          "K-Means/FCM; judge HDBSCAN also on its noise list = outlier customers)")

    rfm_out = rfm.copy()
    rfm_out["KMeans_Label"]    = models["K-Means"]["labels"]
    rfm_out["FCM_Label"]       = models["Fuzzy C-Means"]["labels"]
    rfm_out["FCM_Confidence"]  = models["Fuzzy C-Means"]["confidence"].round(3)
    rfm_out["HDBSCAN_Label"]   = models["HDBSCAN"]["labels"]
    rfm_out["Winner_Segment"]  = relabel_by_monetary(
        models[winner]["labels"], rfm["Monetary"])

    winner_profile = profile_clusters(rfm, models[winner]["labels"])
    print(f"\nWinner profile in business units ({winner}):"); print(winner_profile)

    _save_artifacts("clustering_models.joblib", models, best_params,
                    transformer, FEATURES)
    lb.to_csv("clustering_leaderboard.csv")
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print("\nSaved: clustering_models.joblib | clustering_leaderboard.csv | best_params.json")

    return {"rfm": rfm_out, "Xt": Xt, "leaderboard": lb, "winner": winner,
            "studies": studies, "models": models,
            "winner_profile": winner_profile}


if __name__ == "__main__":
    run_all(n_trials=20)
