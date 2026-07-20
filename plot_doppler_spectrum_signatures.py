"""
plot_doppler_spectrum_signatures.py
-----------------------------------
Reproduce the paper's per-class MEAN centre-row spectrum (the x̄₃' curve in
Fig. 6) so it can be held up against the paper, on the SAME axes:
    x-axis : Doppler velocity   full range (~±16.6 m/s)
    y-axis : magnitude          [dB]  (auto-scaled, not clipped)

Pipeline per segment (paper eq. 1-2, with a principled, clutter-safe β₃):
  0. SNR NORMALISATION (paper, Section II) — optional, on by default:
       - estimate each sample's current SNR at the centre range row s₃
         (the paper states SNRs always refer to the centre of s₃, i.e.
         BEFORE the FFT);
       - draw a target SNR ~ Uniform(3, 30) dB in dB scale;
       - if the sample's current SNR exceeds the target (it is ABOVE the
         threshold), add complex Gaussian noise so its SNR drops to the
         target. Samples already below the target are left untouched
         (you can only add noise, not signal).
     Repeated over the class this yields a uniform 3–30 dB SNR distribution,
     which raises a flat noise floor that buries window side-lobes — this is
     why the paper's spectra (notably the corner reflector R) look clean.
  1. crop centre 150 azimuth samples
  2. β₃ : find the BODY (bulk-Doppler) peak and shift it to 0 Hz
          - clutter is removed for the SEARCH with MTI (slow-time mean
            subtraction), NOT with a magic energy fraction.
          - the moving peak must beat the noise floor by SNR_DB (CFAR-like).
          - hovering edge case (body truly ~0) -> shift 0 (kept at centre).
  3. |FFT|² power
  4. dB-normalise:  10*log10( power / mean(power) )   (eq. 2 in dB, see note)
  5. average the CENTRE range row (row 2) across the class.

Class balancing / sample reuse (paper, Section II):
  Each class is balanced to N_PER_CLASS samples. Small classes (the corner
  reflector R has only 3280 collected samples) are grown by REUSING samples,
  each reuse getting an independent target-SNR draw and a fresh noise
  realisation — exactly as the paper describes ("duplicating samples and
  adding different noise components to simulate different SNRs"). Set
  N_PER_CLASS = None to use every unique sample once with no reuse.

  NB: this noise-based reuse is NOT the "synthetic data" class (the
  model-generated drone, N_c=10) that this project deliberately excludes.

Note on the scale: the paper's eq. (2) uses natural log; Fig. 6 is in dB.
dB = 10*log10(x) = (10/ln10)*ln(x) ≈ 4.34*ln(x).  We compute dB directly so
the y-axis lands on the paper's [-10, 20] range with no scaling mistake.

The SNR-estimation method is the one part the paper never specifies. It is
implemented as a pluggable registry (SNR_ESTIMATORS) so swapping methods is a
one-line change to SNR_METHOD.

Read-only w.r.t. your data. Saves figures; modifies nothing else.
Run:  python plot_doppler_spectrum_signatures.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import CLASSES, FIGURES_DIR, SNR_DATASET_PATH
from data_loading import load_dataset
from preprocessing import N_AZ, preprocess_segment
from snr_norm import compute_shared_floor_full_edge, set_global_sigma2_full_edge

# ── radar / axis constants ───────────────────────────────────────────────────
PRF    = 17_000
LAMBDA = 3e8 / 77e9
DV     = LAMBDA * (PRF / N_AZ) / 2                        # m/s per Doppler bin
VEL_AX = (np.arange(N_AZ) - N_AZ // 2) * DV              # velocity axis [m/s]

# ── SNR-normalisation constants (paper, Section II) ──────────────────────────
APPLY_SNR_NORM = True        # master switch for the whole noise step
SNR_MIN_DB     = 3.0         # uniform target SNR range, lower bound [dB]
SNR_MAX_DB     = 30.0        # uniform target SNR range, upper bound [dB]
SNR_METHOD     = "full_edge"  # key into snr_norm.SNR_ESTIMATORS — current default
                              # (see snr_norm.py module docstring "Method history")
SNR_FLOOR_EXCLUDE_CLASSES = ("R",)  # classes left OUT of the shared-floor pool
                                    # (R's own edges still leak signal; see
                                    # snr_full_edge_diagnostic.py)
N_PER_CLASS    = 4000        # balance every class to this many samples (reuse
                             # small classes with fresh noise); None = no reuse
SEED           = 0           # makes the random target SNRs + noise reproducible

# NB: the SNR-estimation and noise-injection logic lives in snr_norm.py so the
# figure and the training-data preparation use one identical implementation.

# ── fast mode ────────────────────────────────────────────────────────────────
# True  : load already-preprocessed center rows from segments_snr.npz
#         (no β₃, no re-running the full pipeline — quick visual check)
# False : full pipeline from raw segments (SNR norm + β₃ + FFT) — paper quality
FAST_MODE = False


# ─────────────────────────────────────────────────────────────────────────────
# Per-segment processing
# ─────────────────────────────────────────────────────────────────────────────
_DB_SCALE = 10.0 / np.log(10)   # convert natural log → dB


def centre_row_db(seg, rng=None):
    """Full per-segment processing → dB-normalised centre range row (length 150).

    Delegates all pipeline steps to preprocess_segment() in preprocessing.py:
      crop → SNR norm → β₃ → FFT → power → log-norm
    Then converts natural log → dB for plotting.
    Returns (db_row, snr_cur_db, snr_eff_db).
    """
    active_rng = rng if APPLY_SNR_NORM else None
    result, snr_cur, snr_eff = preprocess_segment(
        seg,
        rng=active_rng,
        snr_method=SNR_METHOD,
        snr_range=(SNR_MIN_DB, SNR_MAX_DB),
        return_snr=True,
    )
    return result[2] * _DB_SCALE, snr_cur, snr_eff   # centre range row in dB


def class_indices(ds, cls, n_per_class, rng):
    """Indices for one class, balanced to n_per_class.

    If the class has fewer unique samples than n_per_class, samples are reused
    (paper: small classes grown by duplicating + adding different noise). Each
    occurrence later gets its own noise realisation. n_per_class=None -> all
    unique samples once.
    """
    idxs = np.where(ds["label_9"] == cls)[0].copy()
    rng.shuffle(idxs)
    if n_per_class is None:
        return idxs
    if n_per_class <= len(idxs):
        return idxs[:n_per_class]
    reps = int(np.ceil(n_per_class / len(idxs)))
    return np.tile(idxs, reps)[:n_per_class]


def mean_centre_spectrum(ds, cls, rng):
    """Class-mean dB centre-row spectrum + the SNR diagnostics for the class."""
    idxs = class_indices(ds, cls, N_PER_CLASS, rng)
    acc  = np.zeros(N_AZ, dtype=np.float64)
    snr_cur_all, snr_eff_all = [], []
    for idx in idxs:
        row, snr_cur, snr_eff = centre_row_db(ds["X"][idx], rng=rng)
        acc += row
        snr_cur_all.append(snr_cur)
        snr_eff_all.append(snr_eff)
    return acc / len(idxs), np.array(snr_cur_all), np.array(snr_eff_all)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main_fast():
    """Quick path: load preprocessed center rows from segments_snr.npz.

    No β₃ centering (data is already in frequency domain) — spectra of moving
    targets will not be pinned to 0 Hz. Good for a fast sanity check; use
    FAST_MODE=False for paper-quality reproduction.
    """
    if not SNR_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {SNR_DATASET_PATH}\n"
            f"Run first:  python prepare_snr_dataset.py"
        )
    print(f"Fast mode: loading center rows from {SNR_DATASET_PATH.name} …")
    data    = np.load(SNR_DATASET_PATH, allow_pickle=True)
    X_proc  = data["X_proc"]    # (M, 5, 150) float32, natural-log normalised
    label_9 = data["label_9"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True)
    for ax, cls in zip(axes.flat, CLASSES):
        mask         = label_9 == cls
        center_rows  = X_proc[mask, 2, :] * _DB_SCALE   # (N_cls, 150)
        curve        = center_rows.mean(axis=0)
        ax.plot(VEL_AX, curve, lw=1.3)
        ax.axvline(0, color="grey", lw=0.6, ls="--")
        ax.set_title(cls, fontweight="bold")
        ax.set_xlabel("Doppler velocity [m/s]", fontsize=8)
        ax.set_ylabel("magnitude [dB]", fontsize=8)
        ax.grid(alpha=0.25)
        print(f"  {cls}: {mask.sum()} samples")

    fig.suptitle(
        "Per-class mean centre-row spectrum (fast mode — no β₃)  —  compare to paper Fig. 6",
        fontsize=12,
    )
    fig.tight_layout()
    out = FIGURES_DIR / "doppler_spectrum_signatures.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


def main():
    if FAST_MODE:
        main_fast()
        return

    ds  = load_dataset()
    rng = np.random.default_rng(SEED)
    print(f"v per bin = {DV:.3f} m/s,  max |v| = {DV*(N_AZ//2):.1f} m/s")
    print(f"SNR norm: {'ON' if APPLY_SNR_NORM else 'OFF'}  "
          f"(method={SNR_METHOD}, range={SNR_MIN_DB}-{SNR_MAX_DB} dB, "
          f"N_PER_CLASS={N_PER_CLASS}, seed={SEED})")

    # "full_edge" uses a SHARED floor constant, computed once from every class
    # except SNR_FLOOR_EXCLUDE_CLASSES (default: R only — see snr_norm.py
    # module docstring "Method history" for why). Activating it here means
    # every per-sample estimate below (all 9 classes, including R) reuses
    # this one constant instead of its own (for R, contaminated) edges.
    if APPLY_SNR_NORM and SNR_METHOD == "full_edge":
        shared_sigma2 = compute_shared_floor_full_edge(
            ds["X"], ds["label_9"], edge_flag=ds["edge_flag"],
            exclude_classes=SNR_FLOOR_EXCLUDE_CLASSES,
        )
        set_global_sigma2_full_edge(shared_sigma2)
        print(f"full_edge shared floor: sigma^2 = {10*np.log10(shared_sigma2):.2f} dB "
              f"(pooled from all classes except {SNR_FLOOR_EXCLUDE_CLASSES})")

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True)
    cur_pool, eff_pool = [], []
    for ax, cls in zip(axes.flat, CLASSES):
        curve, snr_cur, snr_eff = mean_centre_spectrum(ds, cls, rng)
        ax.plot(VEL_AX, curve, lw=1.3)
        ax.axvline(0, color="grey", lw=0.6, ls="--")     # body sits here after β₃
        ax.set_title(cls, fontweight="bold")
        ax.set_xlabel("Doppler velocity [m/s]", fontsize=8)
        ax.set_ylabel("magnitude [dB]", fontsize=8)
        ax.grid(alpha=0.25)
        if APPLY_SNR_NORM:
            cur_pool.append(snr_cur)
            eff_pool.append(snr_eff)
            below = np.mean(snr_cur < snr_eff + 1e-9) * 100  # left unchanged
            print(f"  {cls}: mean current SNR {np.nanmean(snr_cur):5.1f} dB, "
                  f"mean effective SNR {np.nanmean(snr_eff):5.1f} dB, "
                  f"{below:4.1f}% already ≤ target")

    fig.suptitle("Per-class mean centre-row spectrum after SNR-norm + β₃  —  "
                 "compare to paper Fig. 6", fontsize=12)
    fig.tight_layout()
    out = FIGURES_DIR / "doppler_spectrum_signatures.png"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")

    # ── SNR diagnostic figure: did we hit a uniform 3–30 dB distribution? ────
    if APPLY_SNR_NORM:
        cur = np.concatenate(cur_pool)
        eff = np.concatenate(eff_pool)
        fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        a1.hist(cur, bins=40, color="tab:gray")
        a1.set_title("estimated CURRENT SNR (before noise)")
        a1.set_xlabel("SNR [dB]"); a1.set_ylabel("count")
        a2.hist(eff, bins=40, color="tab:blue")
        a2.axvspan(SNR_MIN_DB, SNR_MAX_DB, color="tab:green", alpha=0.12)
        a2.set_title("EFFECTIVE SNR (after noise) — target 3–30 dB")
        a2.set_xlabel("SNR [dB]"); a2.set_ylabel("count")
        fig2.tight_layout()
        out2 = FIGURES_DIR / "snr_distribution.png"
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"Saved → {out2}")


if __name__ == "__main__":
    main()
