"""
preprocessing.py
----------------
Converts raw complex segments (5, 256) into log-normalised
Range-Doppler power maps (5, 150) ready for CNN input.

Steps (matching paper Section II, eq. 1-2):
  1.  Crop centre 150 azimuth samples      → (5, 150) complex
  1b. SNR normalisation (optional)         → add Gaussian noise to uniform 3-30 dB
  2.  β₃ steering vector                   → shift bulk-Doppler peak to 0 Hz
  3.  FFT along azimuth axis + fftshift    → (5, 150) complex, 0 Hz centred
  4.  Power spectrum  |·|²                 → (5, 150) float
  5.  Log-normalise   log(x / mean(X))     → (5, 150) float  (natural log, eq. 2)

SNR normalisation (step 1b) is OFF by default (rng=None).
β₃ (step 2) is always applied. estimate_body_shift() returns 0 for static
targets (no clear Doppler peak), so no special-casing is needed per class.
"""

import numpy as np

from snr_norm import apply_snr_normalization, estimate_snr_db

# ── azimuth crop ──────────────────────────────────────────────────────────────
# centre 150 samples out of 256 (indices 53:203, centred at 128)
AZ_START = 53
AZ_END   = 203
N_AZ     = 150

# ── β₃ constants ──────────────────────────────────────────────────────────────
_CENTRE   = N_AZ // 2
_SNR_DB   = 6.0    # moving peak must beat noise floor by this (CFAR-like)
_ASYM_TOL = 0.15   # if spectrum is this symmetric about 0 → body is at 0


# ── β₃ helpers ────────────────────────────────────────────────────────────────
def estimate_body_shift(seg_cropped: np.ndarray) -> int:
    """
    Estimate the bulk-Doppler bin shift for one cropped segment.

    Uses the centre range row (row 2). Returns 0 (no shift) when:
    - no clear moving peak is found (peak below noise floor + _SNR_DB), or
    - the spectrum is roughly symmetric about 0 Hz (hovering / stationary).

    Algorithm:
      MTI (subtract slow-time mean) → removes stationary clutter physically.
      CFAR-like test → peak must exceed median noise floor by _SNR_DB.
      Symmetry guard → symmetric skirts around 0 means body is at 0, not shifted.
    """
    s     = seg_cropped[2]                                    # centre range row
    s_mti = s - s.mean()                                      # MTI: kill clutter
    P     = np.abs(np.fft.fftshift(np.fft.fft(s_mti))) ** 2  # power, 0 Hz centred

    floor    = np.median(P) + 1e-12
    cand     = int(np.argmax(P))
    prom_db  = 10 * np.log10(P[cand] / floor)

    if prom_db < _SNR_DB:
        return 0   # no clear mover → body is at ~0

    left, right = P[:_CENTRE].sum(), P[_CENTRE + 1:].sum()
    if abs(left - right) / (left + right + 1e-12) < _ASYM_TOL:
        return 0   # symmetric skirts → hovering, body at 0

    return cand - _CENTRE


def beta3_shift(seg_cropped: np.ndarray, shift_bins: int) -> np.ndarray:
    """Apply β₃: phase ramp in slow time → shifts spectrum by -shift_bins."""
    p    = np.arange(seg_cropped.shape[1])
    beta = np.exp(-1j * 2 * np.pi * shift_bins * p / seg_cropped.shape[1])
    return seg_cropped * beta[None, :]


# ── main preprocessing function ───────────────────────────────────────────────
def preprocess_segment(
    seg: np.ndarray,
    rng=None,
    snr_method: str = "peak_floor",
    snr_range: tuple = (3.0, 30.0),
    return_snr: bool = False,
    snr_target_db: float = None,
):
    """
    Process one raw segment into a log-normalised Range-Doppler map.

    Parameters
    ----------
    seg : (5, 256) complex64
        Raw complex segment from data_loading.
    rng : np.random.Generator or None
        If given, apply SNR normalisation (step 1b): draw target SNR ~
        Uniform(snr_range) dB and add Gaussian noise if current SNR exceeds it.
    snr_method : str
        SNR-estimation method — key into snr_norm.SNR_ESTIMATORS.
    snr_range : (float, float)
        Uniform target-SNR band [dB]. Default matches paper (3–30 dB).
    return_snr : bool
        If True, also return (snr_cur_db, snr_eff_db) for diagnostics.
        Only meaningful when rng is not None; both are NaN otherwise.
    snr_target_db : float or None
        Externally-chosen target SNR [dB] (strict target-first sampling,
        see prepare_snr_dataset.py). None (default) draws the target from
        Uniform(snr_range).

    Returns
    -------
    np.ndarray, shape (5, 150), float32
        Log-normalised Range-Doppler power map.
    If return_snr=True: (map, snr_cur_db, snr_eff_db)
    """
    # step 1 — crop centre 150 azimuth samples
    seg_cropped = seg[:, AZ_START:AZ_END]                  # (5, 150) complex

    # step 1b — (optional) SNR normalisation
    # wide_row is the full-width s₃ (centre row only) BEFORE crop/FFT/β₃/MTI,
    # required by "azimuth_edge". wide_seg is the full (5, 256) raw segment
    # (all range rows), required by "full_edge" (current default — see
    # snr_norm.py module docstring "Method history"). Both are harmless
    # (ignored) for other methods.
    snr_cur = snr_eff = np.nan
    if rng is not None:
        wide_row = seg[2]                                   # (256,) raw s₃, pre-crop
        wide_seg = seg                                       # (5, 256) raw, pre-crop
        snr_cur = estimate_snr_db(seg_cropped[2], method=snr_method,
                                  wide_row=wide_row, wide_seg=wide_seg)
        seg_cropped, _, snr_eff = apply_snr_normalization(
            seg_cropped, rng,
            method=snr_method,
            snr_min_db=snr_range[0],
            snr_max_db=snr_range[1],
            wide_row=wide_row,
            wide_seg=wide_seg,
            target_db=snr_target_db,
        )

    # step 2 — β₃ bulk-Doppler centering
    # estimate_body_shift returns 0 for static targets (no clear peak),
    # so no special-casing needed per class.
    shift       = estimate_body_shift(seg_cropped)
    seg_cropped = beta3_shift(seg_cropped, shift)

    # step 3 — 1D FFT per range row + centre 0 Hz
    spec  = np.fft.fftshift(np.fft.fft(seg_cropped, axis=1), axes=1)

    # step 4 — power spectrum
    power = np.abs(spec) ** 2                              # (5, 150) float

    # step 5 — log-normalise (eq. 2, natural log)
    X_norm = np.log(power / power.mean()).astype(np.float32)

    if return_snr:
        return X_norm, snr_cur, snr_eff
    return X_norm


def preprocess_dataset(X_raw: np.ndarray) -> np.ndarray:
    """Process all segments. Returns (N, 5, 150) float32."""
    N     = len(X_raw)
    X_out = np.empty((N, 5, N_AZ), dtype=np.float32)
    for i, seg in enumerate(X_raw):
        X_out[i] = preprocess_segment(seg)
    return X_out
