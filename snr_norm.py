"""
snr_norm.py
-----------
Single source of truth for the paper's SNR-normalisation step (Section II):

    "The data is normalized to have a uniform SNR distribution between 3 and
     30 dB after range compression. This is done by adding complex Gaussian
     noise to samples with a higher SNR than a desired target SNR, which is
     uniformly sampled in dB scale."

Per sample, on the centre range row s₃ (the paper states SNRs always refer to
the centre of s₃, i.e. BEFORE the FFT):

    1. estimate the current SNR;
    2. draw a target SNR ~ Uniform(SNR_MIN_DB, SNR_MAX_DB) in dB;
    3. if current SNR > target (the sample is ABOVE the threshold) add complex
       Gaussian noise so the SNR drops to the target. Samples already at/below
       the target are returned unchanged (you can only add noise, not signal).

The SNR-estimation method is the one detail the paper never specifies, so it is
a pluggable registry (SNR_ESTIMATORS); swap it via the `method` argument.

This module is imported by both plot_doppler_spectrum_signatures.py (figure reproduction) and the
training-data preparation (prepare_snr_dataset.py), so the noise applied to the
network is byte-for-byte the same step shown in the figure.

Estimator registry contract
---------------------------
Every entry in SNR_ESTIMATORS has the signature:

    fn(centre_row, wide_row, wide_seg=None) -> float [dB]

where:
  centre_row : (N_AZ_CROP,) complex  — cropped s₃ (main-beam columns only)
  wide_row   : (N_AZ_WIDE,) complex or None
               Full-width s₃ BEFORE crop/FFT/β₃/MTI. Required only by
               estimators that need the off-beam columns ("azimuth_edge").
               Legacy estimators ignore it.
  wide_seg   : (5, N_AZ_WIDE) complex or None
               Full raw segment (all 5 range rows) BEFORE crop/FFT/β₃/MTI.
               Required only by "full_edge" (needs all range rows, not just
               the centre one). Other estimators ignore it.

Method history (why "full_edge" replaced "azimuth_edge")
----------------------------------------------------------
"azimuth_edge" estimates the floor from the 150-window's own off-beam columns
(0:53, 203:256 of the raw 256), centre row only. Diagnostics on this project
(see slowtime_envelope_diagnostic.py / global_floor_diagnostic.py /
session_gain_diagnostic.py in project history) found this window is still too
close to the ~1° antenna beam (±1.15°) and too few samples (~106) per estimate,
so it stayed RCS-correlated across classes (~33 dB spread, tracking target RCS).

"full_edge" (current default) instead uses the far edges of the FULL 256-
sample sweep (±2° from beam centre — farther outside the beam) and pools ALL
5 range rows (400 samples instead of ~106), with a percentile-based, outlier-
robust floor estimator. This shrank the pooled cross-class spread from ~33 dB
to ~10 dB for 8 of 9 classes (see snr_full_edge_diagnostic.py). Class R (corner
reflector, RCS=100 m²) still leaks some of its own huge return into the edges;
a follow-up check (snr_percentile_sweep_diagnostic.py) showed a low percentile
(p≈10–25, instead of the median) cuts R's excess bias roughly in half without
moving the other 8 classes. A range-stratified check on R specifically (far-
range R samples vs. near-range) showed the residual R gap shrinks steadily
with range (13.5 dB near → 5.4 dB far) rather than jumping between sessions —
consistent with shrinking sidelobe leakage, not a genuinely different physical
noise floor for R. Practical takeaway: use compute_shared_floor_full_edge()
with exclude_classes=("R",) so the floor constant is derived only from the 8
classes it is defensible for, then apply that single constant to every class
(including R) via set_global_sigma2_full_edge() — this cannot make R's own
estimate worse (it no longer depends on R's contaminated samples), and a
~5 dB residual bias for R should still be noted, not assumed to be zero.
"""

from __future__ import annotations

import numpy as np

# default uniform target-SNR band (paper: 3–30 dB)
SNR_MIN_DB = 3.0
SNR_MAX_DB = 30.0

# Azimuth crop boundaries — must match preprocessing.AZ_START / AZ_END
AZ_CROP_START: int = 53
AZ_CROP_END:   int = 203

# ─────────────────────────────────────────────────────────────────────────────
# Optional global noise-floor override for "azimuth_edge".
# Call compute_global_n0() once and set_global_n0(val) to activate.
# When set, all per-sample azimuth_edge calls use this constant N0, which
# guarantees identical bias across classes by construction.
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_N0: float | None = None


def set_global_n0(val: float) -> None:
    """Override per-sample off-beam N0 with a pre-computed global constant."""
    global _GLOBAL_N0
    _GLOBAL_N0 = float(val)


def get_global_n0() -> float | None:
    return _GLOBAL_N0


def clear_global_n0() -> None:
    global _GLOBAL_N0
    _GLOBAL_N0 = None


# ─────────────────────────────────────────────────────────────────────────────
# Full-256-edge estimator constants + shared-floor override (current default).
# See module docstring "Method history" for why this replaced "azimuth_edge".
# ─────────────────────────────────────────────────────────────────────────────
FULL_EDGE_LO: tuple[int, int] = (0, 40)     # far-left edge of the full 256 sweep
FULL_EDGE_HI: tuple[int, int] = (216, 256)  # far-right edge of the full 256 sweep
FULL_EDGE_PEAK: tuple[int, int] = (125, 132)  # 7-col window around beam peak (col 128)
FULL_EDGE_CENTRE_ROW: int = 2                 # centre range row (row 3 of 5)
FULL_EDGE_PERCENTILE: float = 10.0            # default percentile (not median)

_GLOBAL_SIGMA2_FULL_EDGE: float | None = None


def set_global_sigma2_full_edge(val: float) -> None:
    """Activate a pre-computed shared floor for every "full_edge" call.

    Use compute_shared_floor_full_edge(..., exclude_classes=("R",)) to derive
    `val` from only the classes this estimator is defensible for, then apply
    it uniformly (including to R) — see module docstring.
    """
    global _GLOBAL_SIGMA2_FULL_EDGE
    _GLOBAL_SIGMA2_FULL_EDGE = float(val)


def get_global_sigma2_full_edge() -> float | None:
    return _GLOBAL_SIGMA2_FULL_EDGE


def clear_global_sigma2_full_edge() -> None:
    global _GLOBAL_SIGMA2_FULL_EDGE
    _GLOBAL_SIGMA2_FULL_EDGE = None


# ─────────────────────────────────────────────────────────────────────────────
# SNR estimators (pluggable).
# Signature: fn(centre_row, wide_row) -> float [dB]
# ─────────────────────────────────────────────────────────────────────────────
def snr_peak_floor(s, wide_row=None, wide_seg=None):
    """Spectral peak-to-noise-floor SNR [dB] of the centre range row.

    signal = strongest power bin after MTI (slow-time mean removed so the
             stationary clutter at DC does not masquerade as 'signal');
    noise  = median power across bins (robust noise-floor estimate).

    NOTE: this estimator is class-dependent for drones — micro-Doppler smears
    energy across most Doppler bins, so the median lands on signal, not noise.
    Use "full_edge" for a class-independent estimate.
    """
    s_mti = s - s.mean()
    P     = np.abs(np.fft.fft(s_mti)) ** 2
    peak  = P.max()
    floor = np.median(P)+ 1e-12
    return 10.0 * np.log10(peak / floor)


def snr_edge_bins(s, wide_row=None, wide_seg=None):
    """Placeholder alt method. Falls back to peak/floor."""
    return snr_peak_floor(s, wide_row)


def snr_azimuth_edge(
    s,
    wide_row,
    wide_seg=None,
    crop_start: int = AZ_CROP_START,
    crop_end:   int = AZ_CROP_END,
    percentile: float = 10.0,
):
    """Class-independent SNR estimator using off-beam azimuth columns.

    The scanning beam points away from the target outside columns
    [crop_start:crop_end], so those columns are dominated by thermal noise,
    regardless of the target's Doppler signature.

    N0  = low-percentile power of the off-beam columns (0:crop_start and
          crop_end:end of the raw s₃ row, BEFORE FFT/β₃/MTI).
          Low percentile (default 10th) rejects residual sidelobe leakage.
          If a global N0 has been set via set_global_n0(), that constant is
          used instead — it has the same bias for all classes by construction.

    Signal power = mean power of main-beam columns − N0.
    SNR_lin = max(signal, 0) / N0.

    Parameters
    ----------
    s        : (N_AZ_CROP,) complex  — cropped s₃ (unused, kept for registry
               compatibility; main-beam power is read from wide_row directly)
    wide_row : (N_AZ_WIDE,) complex  — full-width raw s₃, BEFORE crop/FFT/β₃/MTI
    """
    if wide_row is None:
        raise ValueError(
            '"azimuth_edge" requires wide_row (the full-width s₃ before crop). '
            "Pass seg[2] from preprocess_segment before the azimuth crop."
        )

    if _GLOBAL_N0 is not None:
        n0 = _GLOBAL_N0
    else:
        left  = np.abs(wide_row[:crop_start]) ** 2        # off-beam left
        right = np.abs(wide_row[crop_end:]) ** 2          # off-beam right
        off_beam = np.concatenate([left, right])
        n0 = float(np.percentile(off_beam, percentile)) + 1e-12

    main_power = float(np.mean(np.abs(wide_row[crop_start:crop_end]) ** 2))
    signal = max(main_power - n0, 0.0)

    if signal <= 0.0:
        return 0.0                                         # at or below noise floor

    return 10.0 * np.log10(signal / n0)


def full_edge_sigma2(
    wide_seg,
    edge_lo: tuple[int, int] = FULL_EDGE_LO,
    edge_hi: tuple[int, int] = FULL_EDGE_HI,
    percentile: float = FULL_EDGE_PERCENTILE,
) -> float:
    """Per-sample noise floor sigma^2 from the far edges of the full 256-sweep.

    Pools all 5 range rows, applies the low-percentile exponential-corrected
    estimator (see snr_full_edge docstring). Shared between snr_full_edge()
    and apply_snr_normalization() so estimation and noise sizing use the
    exact same floor.
    """
    edge = np.concatenate([
        wide_seg[:, edge_lo[0]:edge_lo[1]].ravel(),
        wide_seg[:, edge_hi[0]:edge_hi[1]].ravel(),
    ])
    p_edge = np.abs(edge) ** 2
    corr   = -np.log(1.0 - percentile / 100.0)
    return float(np.percentile(p_edge, percentile)) / corr + 1e-12


def snr_full_edge(
    s,
    wide_row=None,
    wide_seg=None,
    edge_lo: tuple[int, int] = FULL_EDGE_LO,
    edge_hi: tuple[int, int] = FULL_EDGE_HI,
    peak_cols: tuple[int, int] = FULL_EDGE_PEAK,
    centre_row_idx: int = FULL_EDGE_CENTRE_ROW,
    percentile: float = FULL_EDGE_PERCENTILE,
):
    """Class-independent SNR estimator using the edges of the FULL 256-sample
    azimuth sweep, pooled across all 5 range rows (current default method).

    Why this instead of "azimuth_edge": the 150-window's own edges are only
    ±1.15° from the beam centre and give ~106 samples from one row; still
    RCS-correlated across classes (see module docstring "Method history").
    This estimator uses the far edges of the untouched 256-sample sweep
    (±2° from beam centre — farther outside the ~1° beam) across all 5 range
    rows (400 samples total), which is both farther from the beam and far
    lower-variance per estimate.

    Noise floor (sigma^2): for circularly-symmetric complex Gaussian noise,
    |s|^2 ~ Exponential(1/sigma^2), whose p-th percentile is
    sigma^2 * (-ln(1 - p/100)). So sigma^2 = percentile(p) / (-ln(1-p/100)).
    p=50 (median) recovers sigma^2 = median/ln(2). A LOWER percentile (default
    10, not 50) is used because leakage from a strong target only ever pushes
    |s|^2 UP, never down — a low percentile sits further from any
    contaminated tail. See snr_percentile_sweep_diagnostic.py: this roughly
    halved class R's excess bias vs. the other 8 classes, at the cost of a
    (small, checked) increase in per-estimate noise from fewer effective
    samples.

    If a shared floor has been activated via set_global_sigma2_full_edge(),
    that constant is used instead of this sample's own edges — see
    compute_shared_floor_full_edge() and the module docstring.

    Signal power: mean power over a 7-column window centred on the beam peak
    (col 128 of 256), on the CENTRE range row only (row 2) — this matches the
    paper's statement that "stated SNRs always refer to the SNR at the centre
    of s₃".

    Parameters
    ----------
    s               : (N_AZ_CROP,) complex — cropped s₃ (unused; kept for
                      registry compatibility, signal power is read from
                      wide_seg directly so cropping/β₃/MTI never touch it)
    wide_row        : unused by this estimator (kept for registry compatibility)
    wide_seg        : (5, N_AZ_WIDE) complex — full raw segment, BEFORE
                      crop/FFT/β₃/MTI. Required.
    """
    if wide_seg is None:
        raise ValueError(
            '"full_edge" requires wide_seg (the full (5, 256) raw segment '
            "before crop). Pass seg from preprocess_segment before cropping."
        )

    if _GLOBAL_SIGMA2_FULL_EDGE is not None:
        sigma2 = _GLOBAL_SIGMA2_FULL_EDGE
    else:
        sigma2 = full_edge_sigma2(wide_seg, edge_lo, edge_hi, percentile)

    p_c    = float(np.mean(np.abs(wide_seg[centre_row_idx, peak_cols[0]:peak_cols[1]]) ** 2))
    signal = max(p_c - sigma2, 0.0)

    if signal <= 0.0:
        return 0.0                                          # at or below noise floor

    return 10.0 * np.log10(signal / sigma2)


SNR_ESTIMATORS = {
    "peak_floor":   snr_peak_floor,
    "edge_bins":    snr_edge_bins,
    "azimuth_edge": snr_azimuth_edge,
    "full_edge":    snr_full_edge,     # current default — see "Method history"
}


def estimate_snr_db(centre_row, method="peak_floor", wide_row=None, wide_seg=None):
    """Current SNR [dB] of the centre range row, via the named method.

    Parameters
    ----------
    centre_row : (N_AZ_CROP,) complex  — cropped s₃ (main-beam columns)
    method     : key into SNR_ESTIMATORS
    wide_row   : (N_AZ_WIDE,) complex or None
                 Full-width raw s₃, required only for "azimuth_edge".
    wide_seg   : (5, N_AZ_WIDE) complex or None
                 Full raw segment (all 5 rows), required only for "full_edge".
    """
    return SNR_ESTIMATORS[method](centre_row, wide_row, wide_seg)


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian-noise injection to a target SNR
# ─────────────────────────────────────────────────────────────────────────────
def add_noise_to_target(seg_cropped, snr_db_cur, target_db, rng, noise_floor=None):
    """Add complex Gaussian noise so the centre row's SNR drops to target_db.

    Only acts when snr_db_cur > target_db; otherwise returns the sample
    unchanged. The noise is applied to all range rows, as radar thermal noise
    is uniform across range bins.

    Noise sizing — MUST match the SNR estimator's definition
    --------------------------------------------------------
    noise_floor (sigma^2, linear) is the SAME floor the estimator used, so
    the current signal power at the beam peak is simply

        P_sig = sigma^2 * g_cur          (since g_cur = P_sig / sigma^2)

    and the total floor required for the target is P_sig / g_t, giving

        add_var = sigma^2 * (g_cur / g_t - 1)

    per complex sample. The previous implementation sized the noise from the
    centre row's MEAN power over all 150 columns. That mean includes the
    antenna-beam roll-off (2.3° window vs 1° beamwidth), so it sits ~2.6 dB
    below the beam-peak power the estimator measures — the injected noise was
    ~2.6 dB too weak and every "normalised" sample landed ~2.6 dB above its
    nominal SNR (verified empirically: mean +2.6 dB, p90 +4.6 dB, across all
    9 classes). The legacy row-mean sizing is kept only as a fallback for
    estimators that provide no floor (noise_floor=None).

    Returns (noisy_seg, effective_snr_db).
    """
    if target_db >= snr_db_cur:
        return seg_cropped, snr_db_cur

    g_cur = 10.0 ** (snr_db_cur / 10.0)            # current SNR, linear
    g_t   = 10.0 ** (target_db  / 10.0)            # target  SNR, linear

    if noise_floor is not None:
        # consistent, definition-matched sizing (current default path)
        add_var = max(noise_floor * (g_cur / g_t - 1.0), 0.0)
    else:
        # legacy row-mean sizing (biased ~+2.6 dB high — see docstring)
        P_tot = np.mean(np.abs(seg_cropped[2]) ** 2)   # centre-row total power
        P_sig = P_tot * g_cur / (1.0 + g_cur)          # split into signal + noise
        N_cur = P_tot - P_sig
        add_var = max(P_sig / g_t - N_cur, 0.0)        # extra noise power needed

    noise = np.sqrt(add_var / 2.0) * (
        rng.standard_normal(seg_cropped.shape)
        + 1j * rng.standard_normal(seg_cropped.shape)
    )
    return seg_cropped + noise, target_db


def apply_snr_normalization(
    seg_cropped, rng,
    method="peak_floor",
    snr_min_db=SNR_MIN_DB,
    snr_max_db=SNR_MAX_DB,
    wide_row=None,
    wide_seg=None,
    target_db=None,
):
    """One full per-sample SNR-normalisation step on a CROPPED segment.

    Parameters
    ----------
    seg_cropped : (R, N_AZ_CROP) complex — cropped segment (centre azimuth window)
    rng         : np.random.Generator    — draws the target SNR and the noise
    method      : key into SNR_ESTIMATORS
    snr_min_db, snr_max_db : uniform target-SNR band [dB]
    wide_row    : (N_AZ_WIDE,) complex or None
                  Full-width raw s₃ BEFORE crop/FFT/β₃/MTI.
                  Required when method="azimuth_edge"; ignored otherwise.
    wide_seg    : (5, N_AZ_WIDE) complex or None
                  Full raw segment (all 5 rows) BEFORE crop/FFT/β₃/MTI.
                  Required when method="full_edge"; ignored otherwise.
    target_db   : float or None
                  Externally-chosen target SNR [dB]. When given, no target is
                  drawn from Uniform(snr_min_db, snr_max_db) — used by the
                  strict target-first sampling in prepare_snr_dataset.py.

    Returns
    -------
    (noisy_seg, snr_cur_db, snr_eff_db)
        snr_cur_db : estimated SNR before noise
        snr_eff_db : SNR actually realised (= target if noise added, else cur)
    """
    snr_cur = estimate_snr_db(seg_cropped[2], method=method, wide_row=wide_row, wide_seg=wide_seg)
    target  = target_db if target_db is not None else rng.uniform(snr_min_db, snr_max_db)

    # Recover the SAME noise floor the estimator used, so noise sizing and
    # SNR estimation share one definition (see add_noise_to_target docstring).
    noise_floor = None
    if method == "full_edge":
        noise_floor = get_global_sigma2_full_edge()
        if noise_floor is None and wide_seg is not None:
            noise_floor = full_edge_sigma2(wide_seg)
    elif method == "azimuth_edge":
        noise_floor = get_global_n0()
        # per-sample fallback: recompute the same off-beam floor
        if noise_floor is None and wide_row is not None:
            left  = np.abs(wide_row[:AZ_CROP_START]) ** 2
            right = np.abs(wide_row[AZ_CROP_END:]) ** 2
            noise_floor = float(np.percentile(np.concatenate([left, right]), 10.0)) + 1e-12

    noisy, snr_eff = add_noise_to_target(seg_cropped, snr_cur, target, rng,
                                         noise_floor=noise_floor)
    return noisy, snr_cur, snr_eff


# ─────────────────────────────────────────────────────────────────────────────
# Shared-floor helper for "full_edge" (current default — see module docstring)
# ─────────────────────────────────────────────────────────────────────────────
def compute_shared_floor_full_edge(
    X_raw,
    label_9,
    edge_flag=None,
    exclude_classes: tuple[str, ...] = ("R",),
    edge_lo: tuple[int, int] = FULL_EDGE_LO,
    edge_hi: tuple[int, int] = FULL_EDGE_HI,
    percentile: float = FULL_EDGE_PERCENTILE,
) -> float:
    """Pool full-256-edge power (all 5 range rows) across every class EXCEPT
    `exclude_classes`, and return one shared sigma^2 [linear power units].

    Default excludes only "R" (corner reflector, RCS=100 m^2): diagnostics
    (snr_full_edge_diagnostic.py, snr_percentile_sweep_diagnostic.py) showed
    R's own edges still carry residual leakage from its own huge return —
    computing the shared floor from the other 8 classes only means R's
    contaminated samples never feed back into the constant that will also be
    applied to R (see module docstring "Method history"). A range-stratified
    check found the resulting bias applied to R shrinks with range (13.5 dB
    near-range -> 5.4 dB far-range) rather than staying fixed, consistent
    with residual leakage rather than a genuinely different physical floor —
    but that ~5 dB residual is not zero and should be noted, not assumed away.

    Parameters
    ----------
    X_raw           : (N, 5, N_AZ_WIDE) complex — raw segments (before any processing)
    label_9         : (N,) str — 9-class label per segment
    edge_flag       : (N,) int8 or None — 1 = synthetic FOV-edge padding;
                       filtered out when given (recommended: always pass it)
    exclude_classes : classes to leave OUT of the pooled floor (default: R only)
    edge_lo, edge_hi: far-edge column ranges of the full 256-sample sweep
    percentile      : percentile applied to the pooled edge powers (see
                       snr_full_edge docstring for why a low percentile, not
                       the median, is the default)

    Returns
    -------
    float : shared sigma^2 [linear power units]

    Usage
    -----
    sigma2 = compute_shared_floor_full_edge(X_raw, label_9, edge_flag)
    set_global_sigma2_full_edge(sigma2)   # all subsequent full_edge calls use this
    """
    mask = np.array([lbl not in exclude_classes for lbl in label_9])
    if edge_flag is not None:
        mask &= (edge_flag == 0)
    X_use = X_raw[mask]                                        # (M, 5, N_AZ_WIDE)

    left  = X_use[:, :, edge_lo[0]:edge_lo[1]]
    right = X_use[:, :, edge_hi[0]:edge_hi[1]]
    edge_pow = np.abs(np.concatenate([left, right], axis=2)) ** 2   # (M, 5, n_edge_cols)

    corr   = -np.log(1.0 - percentile / 100.0)
    sigma2 = float(np.percentile(edge_pow.ravel(), percentile)) / corr
    return sigma2 + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Global N0 helpers (legacy — "azimuth_edge" estimator)
# ─────────────────────────────────────────────────────────────────────────────
def compute_global_n0(
    X_raw,
    crop_start: int   = AZ_CROP_START,
    crop_end:   int   = AZ_CROP_END,
    percentile: float = 10.0,
) -> float:
    """Pool off-beam noise power across ALL raw segments and return a single N0.

    Uses the same low-percentile logic as snr_azimuth_edge but applied to the
    entire dataset, giving a far more stable estimate than any single sample.
    A single constant has the same bias for every class by construction, so
    it cannot introduce class-dependent SNR errors.

    Parameters
    ----------
    X_raw      : (N, 5, N_AZ_WIDE) complex  — raw segments (before any processing)
    crop_start : left edge of main-beam crop (default 53)
    crop_end   : right edge of main-beam crop (default 203)
    percentile : low percentile applied to the pooled off-beam powers (default 10)

    Returns
    -------
    float : global noise floor N0 [linear power units]

    Usage
    -----
    n0 = compute_global_n0(X_raw)
    set_global_n0(n0)           # all subsequent azimuth_edge calls use this
    """
    s3 = X_raw[:, 2, :]                                    # (N, N_AZ_WIDE) centre row
    left  = np.abs(s3[:, :crop_start]) ** 2               # (N, crop_start)
    right = np.abs(s3[:, crop_end:])   ** 2               # (N, N_AZ_WIDE-crop_end)
    off_beam = np.concatenate([left, right], axis=1)       # (N, n_off_cols)
    n0 = float(np.percentile(off_beam.ravel(), percentile))
    return n0 + 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: off-beam noise floor distribution per class
# ─────────────────────────────────────────────────────────────────────────────
def noise_floor_diagnostic(
    X_raw,
    labels,
    crop_start: int   = AZ_CROP_START,
    crop_end:   int   = AZ_CROP_END,
    percentile: float = 10.0,
    plot: bool        = True,
    out_path: str     = None,
):
    """Compute per-sample off-beam noise power and show its distribution per class.

    Interpretation
    --------------
    Tight, class-independent distributions → noise floor is uniform; use
    compute_global_n0() + set_global_n0() for maximum stability.

    Spread or class-correlated distributions → noise floor drifts (e.g. AGC,
    recording gain, range dependence); keep per-sample estimation (default).

    Parameters
    ----------
    X_raw      : (N, 5, N_AZ_WIDE) complex  — raw segments
    labels     : (N,) str                   — class label per segment (label_9)
    crop_start, crop_end, percentile        — same as compute_global_n0
    plot       : if True, show a box plot per class (requires matplotlib)
    out_path   : if given, save the figure here instead of displaying

    Returns
    -------
    dict mapping class_label -> per-sample noise power array (linear)
    """
    s3 = X_raw[:, 2, :]
    left  = np.abs(s3[:, :crop_start]) ** 2
    right = np.abs(s3[:, crop_end:])   ** 2
    off_beam = np.concatenate([left, right], axis=1)       # (N, n_off_cols)

    # per-sample low-percentile noise floor
    n0_per_sample = np.percentile(off_beam, percentile, axis=1)  # (N,)
    n0_db = 10.0 * np.log10(n0_per_sample + 1e-12)

    classes = sorted(set(labels))
    by_class = {cls: n0_db[labels == cls] for cls in classes}

    # print summary stats
    print(f"Off-beam noise floor (p{int(percentile)}) per class [dB]")
    print(f"  {'class':>6}  {'mean':>7}  {'std':>6}  {'min':>7}  {'max':>7}  n")
    for cls in classes:
        v = by_class[cls]
        print(f"  {cls:>6}  {v.mean():+7.2f}  {v.std():6.2f}"
              f"  {v.min():+7.2f}  {v.max():+7.2f}  {len(v)}")

    global_n0_db = 10.0 * np.log10(compute_global_n0(X_raw, crop_start, crop_end, percentile))
    print(f"\nGlobal pooled N0 (same percentile, all samples): {global_n0_db:.2f} dB")
    print("→ If per-class std values above are ≲1–2 dB and means are similar,")
    print("  call set_global_n0(compute_global_n0(X_raw)) for maximum stability.")

    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        data = [by_class[cls] for cls in classes]
        ax.boxplot(data, labels=classes, patch_artist=True)
        ax.axhline(global_n0_db, color="red", linestyle="--",
                   label=f"global N0 = {global_n0_db:.1f} dB")
        ax.set_xlabel("Class")
        ax.set_ylabel(f"Off-beam noise floor p{int(percentile)} [dB]")
        ax.set_title("Per-sample off-beam N0 distribution by class\n"
                     "(tight & flat → use global N0; spread/correlated → keep per-sample)")
        ax.legend()
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path, dpi=150)
            print(f"Saved → {out_path}")
        else:
            plt.show()

    return by_class
