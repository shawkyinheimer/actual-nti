# ============================================================
# IMS Bearing Lab — notebook-faithful figures (no dataset needed)
# Run:  streamlit run app.py
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from scipy.stats import skew, kurtosis
from scipy.signal import welch, butter, sosfiltfilt, hilbert, stft, savgol_filter

st.set_page_config(page_title="IMS Bearing Lab", layout="wide")

# ====================================================
# Constants — exactly as in the notebook
# ====================================================
FS = SAMPLING_FREQUENCY = 20480
NYQUIST = FS / 2
WINDOW_SIZE = 1024
NPERSEG, NOVERLAP = 8192, 6144
DEFAULT_BAND = (2000.0, 9000.0)
BASELINE_RECORDINGS = 30
SEARCH_TOL, BAND_HALF_WIDTH, MAX_HARMONICS = 5.0, 3.0, 5
FREQ_LIMIT_PLAIN = 2000.0     # plain waterfall
FREQ_LIMIT_TARGET = 600.0     # targeted waterfall
MAX_ROWS = 400
SK_EARLY_N = SK_LATE_N = 20

SHAFT_SPEED_RPM = 2000
FR = SHAFT_SPEED_RPM / 60.0
N_BALLS, BALL_DIA, PITCH_DIA = 16, 8.4, 71.5
d_over_D = BALL_DIA / PITCH_DIA

DEFECT_FREQS = {
    "FTF":  (FR / 2) * (1 - d_over_D),
    "BSF":  (PITCH_DIA / (2 * BALL_DIA)) * FR * (1 - d_over_D**2),
    "BPFO": (N_BALLS / 2) * FR * (1 - d_over_D),
    "BPFI": (N_BALLS / 2) * FR * (1 + d_over_D),
}
DEFECT_FREQS = {k: round(v, 2) for k, v in DEFECT_FREQS.items()}
DEFECT_FREQS_FULL = {**DEFECT_FREQS, "FR": round(FR, 1),
                     "2xBSF": 2 * DEFECT_FREQS["BSF"],
                     "2xBPFO": 2 * DEFECT_FREQS["BPFO"]}
PLOT_MARKERS = DEFECT_FREQS_FULL

CHANNEL_MAP = {
    "1st_test": {"bearing_1": 0, "bearing_2": 2, "bearing_3": 4, "bearing_4": 6},
    "2nd_test": {"bearing_1": 0, "bearing_2": 1, "bearing_3": 2, "bearing_4": 3},
    "3rd_test": {"bearing_1": 0, "bearing_2": 1, "bearing_3": 2, "bearing_4": 3},
}
FAILED_BEARINGS = {"1st_test": ["bearing_3", "bearing_4"],
                   "2nd_test": ["bearing_1"], "3rd_test": ["bearing_3"]}

GROUND_TRUTH_FAULTS = {
    ("1st_test", "bearing_3"): ["BPFI"],
    ("1st_test", "bearing_4"): ["BSF", "2xBSF"],
    ("2nd_test", "bearing_1"): ["BPFO", "2xBPFO"],
    ("3rd_test", "bearing_3"): ["BPFO", "2xBPFO"],
}
EXPECTED_FAILURE_MODE = {k: v[0] for k, v in GROUND_TRUTH_FAULTS.items()}
BEARING_CHOICES = [("1st_test", "bearing_3"), ("1st_test", "bearing_4"),
                   ("2nd_test", "bearing_1"), ("3rd_test", "bearing_3")]

# life length (h) and degradation schedule (onset fraction of life, exponent)
LIFE_HOURS  = {"1st_test": 359.0, "2nd_test": 164.0, "3rd_test": 888.0}
DEGRADATION = {"1st_test": (0.55, 2.0), "2nd_test": (0.70, 2.0), "3rd_test": (0.90, 4.0)}

NUMERIC_FEATURE_COLS = [
    "mean", "std", "variance", "rms", "peak", "peak_to_peak",
    "skewness", "kurtosis", "crest_factor", "shape_factor",
    "impulse_factor", "energy", "mean_absolute",
    "dominant_frequency", "spectral_centroid", "spectral_bandwidth",
    "spectral_entropy", "spectral_energy", "max_psd",
]
TIME_FEATURES_EXTENDED = [
    "std", "rms", "peak", "peak_to_peak", "skewness", "kurtosis",
    "crest_factor", "impulse_factor", "energy",
]
LOG_Y_FEATURES = {"std", "rms", "peak", "peak_to_peak", "energy"}
FREQ_FEATURES_EXTENDED = [
    "dominant_frequency", "spectral_centroid", "spectral_bandwidth",
    "spectral_entropy", "spectral_energy", "max_psd",
]
LOG_Y_FREQ = {"spectral_energy", "max_psd"}
DIST_FEATURES = ["rms", "kurtosis", "crest_factor", "spectral_entropy"]
DIST_FEATURES_EXTENDED = ["rms", "kurtosis", "crest_factor",
                          "impulse_factor", "spectral_entropy", "spectral_centroid"]
STAGE_FRACTIONS = {"Early life": 0.02, "Middle life": 0.35,
                   "Late life": 0.75, "Near failure": 0.995}
STAGE_COLORS = {"Early life": "#55A868", "Middle life": "#DD8452",
                "Late life": "#C44E52", "Near failure": "#6B0000"}

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 100
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.titleweight"] = "bold"

def show():
    st.pyplot(plt.gcf(), clear_figure=True)

# ====================================================
# Feature extraction — verbatim from the notebook
# ====================================================
def create_windows(signal, window_size=1024):
    signal = np.asarray(signal)
    n_windows = len(signal) // window_size
    return signal[: n_windows * window_size].reshape(n_windows, window_size)

def extract_time_features(x):
    x = np.asarray(x); abs_x = np.abs(x)
    rms = np.sqrt(np.mean(x ** 2)); peak = np.max(abs_x); mean_abs = np.mean(abs_x)
    return {"mean": np.mean(x), "std": np.std(x), "variance": np.var(x), "rms": rms,
            "peak": peak, "peak_to_peak": np.ptp(x), "skewness": skew(x),
            "kurtosis": kurtosis(x),
            "crest_factor": peak / rms if rms != 0 else 0,
            "shape_factor": rms / mean_abs if mean_abs != 0 else 0,
            "impulse_factor": peak / mean_abs if mean_abs != 0 else 0,
            "energy": np.sum(x ** 2), "mean_absolute": mean_abs}

def extract_frequency_features(x, fs=20480):
    frequencies, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
    psd_sum = np.sum(psd) or 1e-12
    dominant_frequency = frequencies[np.argmax(psd)]
    spectral_centroid = np.sum(frequencies * psd) / psd_sum
    spectral_bandwidth = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * psd) / psd_sum)
    normalized_psd = psd / psd_sum
    spectral_entropy = -np.sum(normalized_psd * np.log2(normalized_psd + 1e-12))
    return {"dominant_frequency": dominant_frequency,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_entropy": spectral_entropy,
            "spectral_energy": np.sum(psd), "max_psd": np.max(psd)}

def extract_features(window, fs=20480):
    return {**extract_time_features(window),
            **extract_frequency_features(window, fs=fs)}

# ====================================================
# DSP helpers — verbatim from the notebook
# ====================================================
def bandpass_sos(band, order=4, fs=FS):
    lo, hi = float(band[0]), min(float(band[1]), fs / 2 - 200)
    return butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")

def envelope_spectrum(x, sos, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    if len(x) < nperseg:
        x = np.pad(x, (0, nperseg - len(x)))
    envelope = np.abs(hilbert(sosfiltfilt(sos, x)))
    return welch(envelope, fs=fs, nperseg=nperseg, noverlap=noverlap)

def spectral_kurtosis(x, fs=FS, nperseg=512, noverlap=384):
    f, _, Z = stft(np.nan_to_num(np.asarray(x, dtype=float)), fs=fs,
                   nperseg=nperseg, noverlap=noverlap, boundary=None, padded=False)
    m2 = np.mean(np.abs(Z) ** 2, axis=1)
    m4 = np.mean(np.abs(Z) ** 4, axis=1)
    return f, m4 / (m2 ** 2 + 1e-20) - 2.0

CANDIDATE_BANDS = [(500, 1000), (1000, 2000), (2000, 4000), (4000, 8000),
                   (1000, 3000), (2000, 6000), (3000, 8000),
                   (2000, 8000), (1000, 8000)]

def defect_band_power(freqs, psd_row, center):
    i_lo = np.searchsorted(freqs, center - SEARCH_TOL)
    i_hi = np.searchsorted(freqs, center + SEARCH_TOL, side="right")
    if i_hi <= i_lo:
        return 0.0
    peak = i_lo + int(np.argmax(psd_row[i_lo:i_hi]))
    j_lo = np.searchsorted(freqs, freqs[peak] - BAND_HALF_WIDTH)
    j_hi = np.searchsorted(freqs, freqs[peak] + BAND_HALF_WIDTH, side="right")
    return float(psd_row[j_lo:j_hi].sum())

# ====================================================
# IMS-style signal simulator (Rexnord ZA-2115 @ 2000 RPM)
# ====================================================
def simulate_signal(sev, fam, noise, f_res, rng):
    n = FS; t = np.arange(n) / FS
    sig = rng.normal(0.0, noise, n)                                  # noise floor
    for k, a in [(1, 0.020), (2, 0.012), (3, 0.008)]:                # faint shaft lines
        sig += a * np.sin(2 * np.pi * k * FR * t + rng.uniform(0, 2 * np.pi))
    sig += rng.normal(0.0, noise * 8 * sev ** 2, n)                  # broadband fault rise
    f0 = DEFECT_FREQS[fam]
    for k in (1, 2):                                                 # direct fault tones (1x, 2x)
        A = 0.35 * sev ** (2.0 + k) / k
        sig += A * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
    if sev > 0.05:                                                   # impact train → resonances
        f_mod = {"BPFI": FR, "BSF": DEFECT_FREQS["FTF"]}.get(fam)    # load-zone / cage modulation
        rl = int(0.006 * FS); rt = np.arange(rl) / FS
        f2 = min(f_res * 1.5, 8500.0)
        ring = (np.exp(-rt / 0.0015) * np.cos(2 * np.pi * f_res * rt)
                + 0.6 * np.exp(-rt / 0.0012) * np.cos(2 * np.pi * f2 * rt))
        base = 0.05 + 1.5 * sev ** 2
        for k in range(int(f0)):
            ti = k / f0 + rng.uniform(-2e-4, 2e-4)
            a = base * rng.uniform(0.5, 1.0)
            if f_mod is not None:
                a *= 0.15 + 0.85 * (0.5 + 0.5 * np.cos(2 * np.pi * f_mod * ti)) ** 2
            j = int(ti * FS)
            if 0 <= j < n - rl:
                sig[j:j + rl] += a * ring
    return sig

def make_rul(test_name, n_coarse):
    life = LIFE_HOURS[test_name]
    return np.unique(np.concatenate([
        np.linspace(0.0, life, n_coarse),
        np.linspace(0.0, 15.0, 40),        # dense final 15 h (like the 10-min sampling)
        np.linspace(0.0, 2.0, 40),         # dense final 2 h
    ]))[::-1]

# ====================================================
# Cached data builders (mirror PSD_HISTORY / features_df)
# ====================================================
@st.cache_data(show_spinner="Simulating a full bearing life (one-time)…")
def get_signals(test_name, bearing_name, n_rec, noise, f_res, seed):
    rul = make_rul(test_name, n_rec)
    fam = EXPECTED_FAILURE_MODE[(test_name, bearing_name)]
    u0, p = DEGRADATION[test_name]
    u = 1.0 - rul / LIFE_HOURS[test_name]
    sev = np.where(u < u0, 0.02, 0.02 + 0.98 * ((u - u0) / (1 - u0)) ** p)
    sev = sev * np.clip(rul / 0.15, 0.08, 1.0)   # terminal seizure drop (<0.15 h)
    sigs = np.empty((len(rul), FS), dtype=np.float32)
    for i, s in enumerate(sev):
        rng = np.random.default_rng(seed * 7919 + i)
        sigs[i] = simulate_signal(float(s), fam, noise, f_res, rng)
    return sigs, rul

@st.cache_data(show_spinner="Computing Welch PSD history…")
def get_psd_history(test_name, bearing_name, n_rec, noise, f_res, seed):
    sigs, rul = get_signals(test_name, bearing_name, n_rec, noise, f_res, seed)
    rows, freqs = [], None
    for x in sigs:
        f, p = welch(x.astype(float), fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
        rows.append(p); freqs = f
    return freqs, np.vstack(rows).astype(np.float32), rul

@st.cache_data(show_spinner="Extracting window features…")
def get_features(test_name, bearing_name, n_rec, noise, f_res, seed):
    sigs, rul = get_signals(test_name, bearing_name, n_rec, noise, f_res, seed)
    rows = []
    for i, (x, rh) in enumerate(zip(sigs, rul)):
        for wi, w in enumerate(create_windows(x.astype(float), WINDOW_SIZE)):
            row = extract_features(w, fs=FS)
            row.update(test_set=test_name, bearing=bearing_name,
                       recording_index=i, window_index=wi, RUL_hours=float(rh))
            rows.append(row)
    df = pd.DataFrame(rows)
    df["RUL_fraction"] = df["RUL_hours"] / df["RUL_hours"].max()
    u0, _ = DEGRADATION[test_name]
    df["life_stage"] = np.select([df["RUL_fraction"] > 1 - u0,
                                  df["RUL_fraction"] > 0.2],
                                 ["Healthy", "Degrading"], default="Near Failure")
    df["group"] = f"{test_name} / {bearing_name}"
    return df

@st.cache_data(show_spinner="Computing spectral kurtosis…")
def get_sk(test_name, bearing_name, n_rec, noise, f_res, seed):
    sigs, _ = get_signals(test_name, bearing_name, n_rec, noise, f_res, seed)
    f_sk = None; sk_e, sk_l = [], []
    for x in sigs[:SK_EARLY_N]:
        f_sk, s = spectral_kurtosis(x.astype(float)); sk_e.append(s)
    for x in sigs[-SK_LATE_N:]:
        _, s = spectral_kurtosis(x.astype(float)); sk_l.append(s)
    sk_e, sk_l = np.mean(sk_e, axis=0), np.mean(sk_l, axis=0)
    scores = []
    for band in CANDIDATE_BANDS:
        m = (f_sk >= band[0]) & (f_sk <= band[1])
        scores.append(sk_l[m].mean() if m.any() else -np.inf)
    return f_sk, sk_e, sk_l, CANDIDATE_BANDS[int(np.argmax(scores))]

# ====================================================
# Page 1 — PSD Evolution heatmaps (verbatim)
# ====================================================
def page_psd_evolution(key, P):
    freqs, psd, rul = get_psd_history(*key, *P)
    fam = EXPECTED_FAILURE_MODE[key]; f0 = DEFECT_FREQS[fam]

    plt.figure(figsize=(12, 6))
    plt.imshow(np.log10(psd.T + 1e-12), aspect="auto", origin="lower",
               extent=[rul[0], rul[-1], freqs[0], freqs[-1]], cmap="magma")
    plt.gca().invert_xaxis()
    plt.colorbar(label="Log10 Power Spectral Density")
    plt.xlabel("Remaining Useful Life (Hours)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"PSD Evolution over Time - {key[0]} {key[1]}")
    plt.ylim(0, 1000)
    show()

    mask = rul <= 15
    rz, pz = rul[mask], psd[mask]
    plt.figure(figsize=(12, 6))
    plt.imshow(np.log10(pz.T + 1e-12), aspect="auto", origin="lower",
               extent=[rz[0], rz[-1], freqs[0], freqs[-1]], cmap="magma")
    plt.axhline(y=f0, color="cyan", linestyle="--", alpha=0.7, label=f"{fam} 1x ({f0:.1f} Hz)")
    plt.axhline(y=f0 * 2, color="lime", linestyle="--", alpha=0.7, label=f"{fam} 2x ({f0*2:.1f} Hz)")
    plt.gca().invert_xaxis()
    plt.colorbar(label="Log10 Power Spectral Density")
    plt.xlabel("Remaining Useful Life (Hours)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"PSD Evolution (Final 15 Hours Breakdown) - {key[0]} {key[1]}")
    plt.ylim(0, 600)
    plt.legend(loc="upper right")
    plt.tight_layout()
    show()

    mask = rul <= 2.0
    rz, pz = rul[mask], psd[mask]
    plt.figure(figsize=(12, 6))
    plt.imshow(np.log10(pz.T + 1e-12), aspect="auto", origin="lower",
               extent=[rz[0], rz[-1], freqs[0], freqs[-1]], cmap="magma")
    plt.axhline(y=f0, color="cyan", linestyle="--", linewidth=1.5, label=f"{fam} 1x ({f0:.1f} Hz)")
    plt.axhline(y=f0 * 2, color="lime", linestyle="--", linewidth=1.5, label=f"{fam} 2x ({f0*2:.1f} Hz)")
    plt.gca().invert_xaxis()
    plt.colorbar(label="Log10 Power Spectral Density")
    plt.xlabel("Remaining Useful Life (Hours)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Terminal Failure Dynamics (Final 2 Hours) - {key[0]} {key[1]}")
    plt.ylim(100, 550)
    plt.legend(loc="upper right")
    plt.tight_layout()
    show()

# ====================================================
# Page 2 — Early vs Late PSD (fig 10, verbatim)
# ====================================================
def page_early_late(key, P):
    freqs, psd_matrix, _ = get_psd_history(*key, *P)
    psd_early = psd_matrix[0]
    late_idx = max(0, len(psd_matrix) - 5)
    psd_late = psd_matrix[late_idx]

    eps = 1e-12
    log_early = np.log10(np.maximum(psd_early, eps))
    log_late = np.log10(np.maximum(psd_late, eps))
    smooth_early_full = 10 ** savgol_filter(log_early, window_length=31, polyorder=2)
    smooth_late_full = 10 ** savgol_filter(log_late, window_length=31, polyorder=2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 3.5), sharex="col")

    ax_full = axes[0]
    ax_full.semilogy(freqs, psd_early, lw=0.3, color="#2ecc71", alpha=0.3, label="Early Life (Raw)")
    ax_full.semilogy(freqs, psd_late, lw=0.3, color="#e74c3c", alpha=0.3, label="Near Failure (Raw)")
    ax_full.semilogy(freqs, smooth_early_full, lw=1.5, color="#1e8449", label="Early Life (Trend)")
    ax_full.semilogy(freqs, smooth_late_full, lw=1.5, color="#b03a2e", label="Near Failure (Trend)")
    ax_full.set_ylabel("PSD ($g^2 / Hz$)", fontweight="bold")
    ax_full.set_title(f"{key[0]} / {key[1]} — Full Band (0–10 kHz)", fontsize=11, fontweight="bold")
    ax_full.grid(True, which="both", linestyle=":", alpha=0.5)
    ax_full.legend(fontsize=8, loc="upper right")

    ax_zoom = axes[1]
    mask = freqs <= 600
    smooth_early_zoom = 10 ** savgol_filter(log_early[mask], window_length=7, polyorder=2)
    smooth_late_zoom = 10 ** savgol_filter(log_late[mask], window_length=7, polyorder=2)
    ax_zoom.semilogy(freqs[mask], smooth_early_zoom, lw=1.2, color="#1e8449", label="Early Life")
    ax_zoom.semilogy(freqs[mask], smooth_late_zoom, lw=1.2, color="#b03a2e", label="Near Failure")
    ax_zoom.set_xlim(0, 600)
    ax_zoom.set_title(f"{key[0]} / {key[1]} — Fault Band (0–600 Hz)", fontsize=11, fontweight="bold")
    ax_zoom.grid(True, which="both", linestyle=":", alpha=0.5)
    for name, f0m in PLOT_MARKERS.items():
        if f0m <= 600:
            color = "#8e44ad" if "BPF" in name else "#7f8c8d"
            ax_zoom.axvline(f0m, color=color, ls="--", lw=1.0, alpha=0.7)
            ax_zoom.annotate(name, xy=(f0m, 0.95), xycoords=("data", "axes fraction"),
                             rotation=90, fontsize=8, color=color, fontweight="bold",
                             ha="right", va="top")

    axes[0].set_xlabel("Frequency (Hz)", fontweight="bold")
    axes[1].set_xlabel("Frequency (Hz)", fontweight="bold")
    plt.tight_layout()
    show()

# ====================================================
# Page 3 — PSD Waterfalls (fig 11, verbatim)
# ====================================================
def page_waterfalls(key, P):
    freqs, psd, rul = get_psd_history(*key, *P)

    band = freqs <= FREQ_LIMIT_PLAIN
    step = max(1, len(psd) // MAX_ROWS)
    psd_db = 10 * np.log10(np.maximum(psd[::step][:, band], 1e-12))
    rul_sub, freqs_band = rul[::step], freqs[band]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(psd_db, aspect="auto", origin="lower", cmap="magma",
                   extent=(freqs_band[0], freqs_band[-1], rul_sub[-1], rul_sub[0]),
                   vmin=np.percentile(psd_db, 5), vmax=np.percentile(psd_db, 99))
    fig.colorbar(im, ax=ax, label="PSD (dB)")
    for name, f0m in DEFECT_FREQS.items():
        if f0m <= FREQ_LIMIT_PLAIN:
            ax.axvline(f0m, color="cyan", ls=":", lw=0.9, alpha=0.8)
            ax.annotate(name, xy=(f0m, 0.96), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=8, color="cyan", fontweight="bold",
                        ha="right", va="top")
    ax.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax.set_ylabel("Remaining Useful Life (Hours)", fontweight="bold")
    ax.set_title(f"PSD Waterfall — {key[0]} / {key[1]}", fontweight="bold")
    plt.tight_layout()
    show()

    target_faults = GROUND_TRUTH_FAULTS.get(key, [])
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(psd_db, aspect="auto", origin="lower", cmap="magma",
                   extent=(freqs_band[0], freqs_band[-1], rul_sub[-1], rul_sub[0]),
                   vmin=np.percentile(psd_db, 5), vmax=np.percentile(psd_db, 99))
    fig.colorbar(im, ax=ax, label="PSD (dB)")
    for name, f0m in DEFECT_FREQS_FULL.items():
        if f0m <= FREQ_LIMIT_TARGET:
            is_target = name in target_faults
            color = "#00FFFF" if is_target else "#7f8c8d"
            lw = 1.8 if is_target else 0.8
            alpha = 0.95 if is_target else 0.4
            ax.axvline(f0m, color=color, ls="--" if is_target else ":", lw=lw, alpha=alpha)
            ax.annotate(name, xy=(f0m, 0.95), xycoords=("data", "axes fraction"),
                        rotation=90, fontsize=8 if not is_target else 9,
                        color=color, fontweight="bold" if is_target else "normal",
                        ha="right", va="top")
    if key == ("1st_test", "bearing_3"):
        ax.axvspan(480, 520, color="orange", alpha=0.15)
        ax.annotate("Modulated Resonance (500 Hz)", xy=(500, 0.70),
                    xycoords=("data", "axes fraction"), rotation=90, fontsize=8,
                    color="orange", fontweight="bold", ha="right", va="top")
    ax.set_ylim(0, min(rul_sub[0], 100.0))
    ax.set_xlabel("Frequency (Hz)", fontweight="bold")
    ax.set_ylabel("Remaining Useful Life (Hours)", fontweight="bold")
    ax.set_title(f"Targeted PSD Waterfall — {key[0]} / {key[1]} (Target: {', '.join(target_faults)})",
                 fontweight="bold")
    plt.tight_layout()
    show()

# ====================================================
# Page 4 — Defect Band Energy Growth (fig 12, verbatim)
# ====================================================
def page_band_energy(key, P):
    freqs, psd, rul = get_psd_history(*key, *P)
    nyquist, n_base = freqs[-1], min(BASELINE_RECORDINGS, len(psd))

    fig, ax = plt.subplots(figsize=(12, 5))
    total = psd.sum(axis=1)
    broadband_db = 10 * np.log10(total / (np.median(total[:n_base]) + 1e-12) + 1e-12)
    ax.plot(rul, broadband_db, color="gray", ls="--", lw=1.2, alpha=0.7,
            label="Broadband (all energy)")

    expected = EXPECTED_FAILURE_MODE.get(key, "")
    for name, f0m in DEFECT_FREQS.items():
        centers = [k * f0m for k in range(1, MAX_HARMONICS + 1)
                   if k * f0m < nyquist - SEARCH_TOL]
        energy = np.array([sum(defect_band_power(freqs, row, c) for c in centers)
                           for row in psd])
        growth_db = 10 * np.log10(energy / (np.median(energy[:n_base]) + 1e-12) + 1e-12)
        highlight = name == expected
        ax.plot(rul, growth_db, lw=2.5 if highlight else 1.0,
                alpha=1.0 if highlight else 0.45,
                label=f"{name} (Target Fault)" if highlight else name)

    ax.invert_xaxis()
    ax.set_xlabel("Hours before failure", fontweight="bold")
    ax.set_ylabel("Band energy growth vs early life (dB)", fontweight="bold")
    ax.set_title(f"Defect Band Energy Growth — {key[0]} / {key[1]} (Ground Truth: {expected})",
                 fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    show()

# ====================================================
# Page 5 — Envelope & SK (figs 13–14, verbatim)
# ====================================================
def page_envelope_sk(key, P):
    sigs, _ = get_signals(*key, *P)
    demo_signal = sigs[-1].astype(float)

    f_psd, psd = welch(demo_signal, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    env_naive = np.abs(hilbert(demo_signal))
    f_naive, psd_naive = welch(env_naive, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    f_bp, psd_bp = envelope_spectrum(demo_signal, bandpass_sos(DEFAULT_BAND))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].semilogy(f_psd, psd, lw=0.7, color="#4C72B0")
    axes[0].axvspan(*DEFAULT_BAND, color="#DD8452", alpha=0.25, label="demodulation band")
    axes[0].set_title("Raw PSD — impacts ring a structural resonance")
    axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("PSD")
    axes[0].legend(fontsize=8); axes[0].grid(False)

    axes[1].semilogy(f_naive, psd_naive, lw=0.8, color="gray", label="envelope, no filter")
    axes[1].semilogy(f_bp, psd_bp, lw=0.8, color="#C44E52", label="envelope, band-passed")
    axes[1].set_xlim(0, 600)
    axes[1].set_title("Envelope Spectrum — with vs without band-pass")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].legend(fontsize=8); axes[1].grid(False)
    for name, f0m in DEFECT_FREQS.items():
        axes[1].axvline(f0m, color="k", ls=":", lw=0.7)
        axes[1].annotate(name, xy=(f0m, 0.97), xycoords=("data", "axes fraction"),
                         rotation=90, fontsize=7, ha="right", va="top")
    plt.tight_layout()
    show()

    f_sk, sk_early, sk_late, best = get_sk(*key, *P)
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(f_sk, sk_early, lw=0.8, color="#55A868", label="early life (avg)")
    ax.plot(f_sk, sk_late, lw=0.8, color="#C44E52", label="near failure (avg)")
    ax.axvspan(*best, color="#DD8452", alpha=0.25,
               label=f"selected band {best[0]:.0f}–{best[1]:.0f} Hz")
    ax.set_xlim(0, NYQUIST)
    ax.set_title(f"Spectral Kurtosis — {key[0]} / {key[1]}")
    ax.set_ylabel("SK"); ax.set_xlabel("Frequency (Hz)")
    ax.legend(fontsize=8); ax.grid(False)
    plt.tight_layout()
    show()
    st.write(f"**Selected demodulation band:** {best[0]:.0f} – {best[1]:.0f} Hz")

# ====================================================
# Page 6 — Feature Trends (figs 24–25, verbatim)
# ====================================================
def page_feature_trends(key, P):
    test_name, bearing_name = key
    subset = get_features(*key, *P)

    trend = (subset.groupby("RUL_hours")[TIME_FEATURES_EXTENDED].mean()
             .reset_index().sort_values("RUL_hours", ascending=False))
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True)
    for ax, feature in zip(axes.flat, TIME_FEATURES_EXTENDED):
        ax.plot(trend["RUL_hours"], trend[feature], lw=0.8, color="#4C72B0")
        if feature in LOG_Y_FEATURES:
            ax.set_yscale("log")
        ax.set_ylabel(feature, fontsize=9)
    axes[0, 0].invert_xaxis()
    fig.suptitle(f"Time-Domain Degradation — {test_name} / {bearing_name}", fontweight="bold")
    axes[2, 1].set_xlabel("Hours before failure")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    show()

    trend = (subset.groupby("RUL_hours")[FREQ_FEATURES_EXTENDED].mean()
             .reset_index().sort_values("RUL_hours", ascending=False))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    for ax, feature in zip(axes.flat, FREQ_FEATURES_EXTENDED):
        ax.plot(trend["RUL_hours"], trend[feature], lw=0.8, color="#8172B3")
        if feature in LOG_Y_FREQ:
            ax.set_yscale("log")
        ax.set_ylabel(feature, fontsize=9)
    axes[0, 0].invert_xaxis()
    fig.suptitle(f"Spectral-Feature Degradation — {test_name} / {bearing_name}", fontweight="bold")
    axes[1, 1].set_xlabel("Hours before failure")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    show()

# ====================================================
# Page 7 — Waveform Evolution (fig 26, verbatim)
# ====================================================
def page_waveforms(key, P):
    sigs, _ = get_signals(*key, *P)
    n = len(sigs)
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.0))
    for col, (stage, frac) in enumerate(STAGE_FRACTIONS.items()):
        idx = min(int(frac * (n - 1)), n - 1)
        signal = sigs[idx].astype(float)
        axes[col].plot(signal[:2048], lw=0.5, color=STAGE_COLORS[stage])
        rms = np.sqrt(np.mean(signal ** 2))
        axes[col].set_title(f"{stage} — RMS {rms:.2f}", fontsize=10)
        if col == 0:
            axes[col].set_ylabel("Amplitude")
    for ax in axes:
        ax.set_xlabel("Sample (first 2048 of 20480)")
    plt.tight_layout()
    show()

# ====================================================
# Page 8 — Dataset Overview (figs 01, 07, 08, 24b, 31 + summary)
# ====================================================
def page_overview(P):
    frames = []
    for test_name, bearing_name in BEARING_CHOICES:
        frames.append(get_features(test_name, bearing_name, *P))
    features_df = pd.concat(frames, ignore_index=True)

    st.markdown("#### Windows per Bearing")
    window_counts = (features_df.groupby(["test_set", "bearing"]).size()
                     .reset_index(name="n_windows"))
    window_counts["label"] = window_counts["test_set"] + " / " + window_counts["bearing"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(window_counts["label"], window_counts["n_windows"], color="#4C72B0")
    ax.bar_label(bars, padding=3)
    ax.set_title("Number of Feature Windows per Failing Bearing")
    ax.set_xlabel("Test / Bearing"); ax.set_ylabel("Number of Windows")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout(); show()

    st.markdown("#### Feature Correlation")
    corr_matrix = features_df[NUMERIC_FEATURE_COLS].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True,
                linewidths=0.4, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation Matrix — Time & Frequency Domain Features")
    plt.tight_layout(); show()

    st.markdown("#### Feature Spread by Bearing")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    for ax, feature in zip(axes, DIST_FEATURES):
        sns.boxplot(data=features_df, x="group", y=feature, hue="group",
                    palette="deep", legend=False, showfliers=False, ax=ax)
        ax.set_title(f"{feature} by Bearing"); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout(); show()

    st.markdown("#### Distributions by Life Stage")
    palette = {"Healthy": "#55A868", "Degrading": "#DD8452", "Near Failure": "#C44E52"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, feature in zip(axes.flat, DIST_FEATURES_EXTENDED):
        sns.kdeplot(data=features_df, x=feature, hue="life_stage",
                    hue_order=["Healthy", "Degrading", "Near Failure"],
                    palette=palette, fill=True, alpha=0.3, common_norm=False, ax=ax)
        ax.set_title(feature)
    plt.tight_layout(); show()

    st.markdown("#### Feature Stability in Early Life")
    stability_rows = []
    for test_name, bearing_name in BEARING_CHOICES:
        sub = features_df[(features_df["test_set"] == test_name) &
                          (features_df["bearing"] == bearing_name)]
        rec = sub.groupby("recording_index")[["RUL_hours"] + NUMERIC_FEATURE_COLS].mean()
        early = rec.head(BASELINE_RECORDINGS)
        for feat in NUMERIC_FEATURE_COLS:
            mean, sd = early[feat].mean(), early[feat].std()
            cv = 100 * sd / abs(mean) if abs(mean) > 1e-9 else np.nan
            life_range = rec[feat].quantile(0.95) - rec[feat].quantile(0.05)
            trend_snr = life_range / (sd + 1e-12) if sd > 1e-12 else np.inf
            stability_rows.append({"feature": feat,
                                   "group": f"{test_name}/{bearing_name}",
                                   "cv_early_pct": round(cv, 1),
                                   "trend_snr": round(trend_snr, 1)})
    stab = pd.DataFrame(stability_rows)
    stab_table = stab.pivot(index="feature", columns="group", values="cv_early_pct")
    stab_table["median_trend_snr"] = stab.groupby("feature")["trend_snr"].median()
    stab_table = stab_table.sort_values("median_trend_snr")
    st.dataframe(stab_table, use_container_width=True)

    med_snr = stab.groupby("feature")["trend_snr"].median().sort_values()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(med_snr.index, med_snr.values, color="#4C72B0")
    ax.set_xscale("log")
    ax.axvline(1.0, color="#C44E52", ls="--", lw=1.2,
               label="noise floor: life range = 1 healthy σ")
    ax.set_xlabel("Trend SNR (life-long range / early-life std, log scale)")
    ax.set_title("Can Each Feature See Degradation Through Its Own Noise?")
    ax.legend()
    plt.tight_layout(); show()

    st.markdown("#### Summary Statistics")
    summary_stats = (features_df.groupby(["test_set", "bearing"])[NUMERIC_FEATURE_COLS]
                     .agg(["mean", "std", "min", "max"]))
    st.dataframe(summary_stats, use_container_width=True)

# ====================================================
# UI
# ====================================================
st.title("⚙️ IMS Bearing Lab")

PAGES = ["PSD Evolution (heatmaps)", "Early vs Late PSD", "PSD Waterfalls",
         "Defect Band Energy", "Envelope & SK", "Feature Trends",
         "Waveform Evolution", "Dataset Overview"]

page = st.sidebar.radio("Page", PAGES)
label = st.sidebar.selectbox("Bearing", [f"{t} / {b}" for t, b in BEARING_CHOICES], index=2)
key = tuple(label.split(" / "))
n_rec = st.sidebar.slider("Recordings across life", 60, 200, 100)
noise = st.sidebar.slider("Noise floor", 0.02, 0.15, 0.05, step=0.01)
f_res = st.sidebar.slider("Structural resonance (Hz)", 3000, 8000, 4500, step=250)
seed = st.sidebar.number_input("Seed", 0, 10**6, 7)
P = (int(n_rec), float(noise), float(f_res), int(seed))

if page == "PSD Evolution (heatmaps)":
    page_psd_evolution(key, P)
elif page == "Early vs Late PSD":
    page_early_late(key, P)
elif page == "PSD Waterfalls":
    page_waterfalls(key, P)
elif page == "Defect Band Energy":
    page_band_energy(key, P)
elif page == "Envelope & SK":
    page_envelope_sk(key, P)
elif page == "Feature Trends":
    page_feature_trends(key, P)
elif page == "Waveform Evolution":
    page_waveforms(key, P)
else:
    page_overview(P)
