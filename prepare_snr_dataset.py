"""
prepare_snr_dataset.py
----------------------
Build the FIXED, pre-generated training/validation/test set with the paper's
SNR normalisation baked in (paper Section II):

    "A sample is in some cases used several times but with different additive
     noise components to generate 10000 training samples, 4000 validation
     samples and 4000 test samples in each class. Samples are only reused
     within their respective set."

For every split and every class we draw a fixed number of samples (with reuse
when the class has fewer originals — R has only 1280 training samples), and for
EACH drawn sample we:

    crop → add complex Gaussian noise to a uniform 3-30 dB target SNR
         → FFT → |·|² → log-normalise        (preprocessing.preprocess_segment)

i.e. exactly the per-sample step shown in plot_doppler_spectrum_signatures.py *before* the
averaging — except the result is stored to feed the network instead of being
averaged. The noise is fixed once (seeded), so training/eval are reproducible.

This is NOT the model-generated "synthetic drone" class the project excludes;
it is the noise-based reuse the paper applies to all nine real classes.

Output: data/processed/segments_snr.npz
    X_proc    (M, 5, 150) float32  — log-normalised maps, noise already added
    label_9   (M,)        str
    split     (M,)        int8     — 1/2/3 (train/val/test)
    snr_eff   (M,)        float32  — realised SNR [dB] (diagnostic)

Run (paper counts):       python prepare_snr_dataset.py
Quick smoke-test counts:  python prepare_snr_dataset.py --train 200 --val 80 --test 80 --out data/processed/segments_snr_small.npz
"""

import argparse
import sys
import types
from pathlib import Path

import numpy as np

# torch is only used by config.get_device(), which this script never calls
if "torch" not in sys.modules:
    sys.modules["torch"] = types.ModuleType("torch")

from config import CLASSES, SNR_DATASET_PATH
from data_loading import DEFAULT_OUT as RAW_NPZ          # data/processed/segments.npz
from preprocessing import preprocess_segment, N_AZ
from snr_norm import (
    FULL_EDGE_CENTRE_ROW,
    FULL_EDGE_PEAK,
    compute_shared_floor_full_edge,
    set_global_sigma2_full_edge,
)

DEFAULT_OUT = SNR_DATASET_PATH

SPLIT_MAP = {"train": 1, "val": 2, "test": 3}

# paper Section II per-class targets
DEFAULT_TRAIN = 10000
DEFAULT_VAL   = 4000
DEFAULT_TEST  = 4000

SNR_METHOD = "full_edge"   # current default — see snr_norm.py "Method history"
SNR_RANGE  = (3.0, 30.0)
SNR_FLOOR_EXCLUDE_CLASSES = ("R",)  # left OUT of the shared-floor pool (see
                                    # snr_full_edge_diagnostic.py — R's own
                                    # edges still leak signal from its huge RCS)

# ── strict target-first sampling ─────────────────────────────────────────────
# Loose policy (SNR_STRICT=False, the old behaviour): draw a sample, draw a
# target; if the sample's estimated SNR is below the target it is kept
# UNCHANGED. ~25% of samples then carry no injected noise — their label is
# only as good as the floor estimate — and the realised distribution is not
# uniform (deficit above ~21 dB), unlike the paper's stated outcome.
#
# Strict policy (default): draw the TARGET first (exactly Uniform 3-30 dB),
# then pick a sample whose estimated SNR is >= target + SNR_MARGIN_DB and
# noise it down. Every output sample carries real injected noise, and because
# the injected noise (known exactly) dominates the uncertain original floor,
# a floor-estimation error of ε [dB] perturbs the final label by at most
# ε·10^(-m/10) (m = margin): m=3 dB halves any floor error. Samples whose
# estimated SNR is below SNR_RANGE[0] + margin are EXCLUDED (no add-noise
# scheme can label them reliably) and counted per class. Sample reuse with
# fresh noise is exactly what the paper describes. If no candidate exists for
# a high target (low-SNR classes near 30 dB), the target is clipped down to
# the strongest sample's SNR - margin; fallbacks are counted and reported.
SNR_STRICT    = True
SNR_MARGIN_DB = 3.0


def draw_indices(pool, n_target, rng):
    """n_target indices from `pool`, reusing (shuffle+tile) when pool is small."""
    pool = np.asarray(pool).copy()
    rng.shuffle(pool)
    if n_target <= len(pool):
        return pool[:n_target]
    reps = int(np.ceil(n_target / len(pool)))
    return np.tile(pool, reps)[:n_target]


def pool_snr_db(X_raw, pool, sigma2):
    """Vectorised full_edge SNR estimate [dB] for every pool index.

    Same definition as snr_norm.snr_full_edge with the shared global floor:
    peak power = mean |.|^2 over the 7-column beam-peak window of the centre
    range row; SNR = (P_peak - sigma2) / sigma2.
    """
    seg = X_raw[pool]                                     # (n, 5, 256)
    p_c = np.mean(
        np.abs(seg[:, FULL_EDGE_CENTRE_ROW, FULL_EDGE_PEAK[0]:FULL_EDGE_PEAK[1]]) ** 2,
        axis=1,
    )
    sig = np.maximum(p_c - sigma2, 0.0)
    return 10.0 * np.log10(np.maximum(sig, 1e-300) / sigma2)


def strict_draw(pool, snr_db, n_target, rng, snr_range, margin_db):
    """Target-first sampling. Returns (indices, targets, n_fallback).

    Assumes `pool` has already been filtered to snr_db >= snr_range[0] +
    margin_db (the exclusion rule). For each of n_target draws:
    target ~ Uniform(snr_range); pick a random pool sample with estimated
    SNR >= target + margin_db. If none exists (target near the top of the
    band in a low-SNR class), fall back to clipping the target down to the
    strongest sample's SNR - margin_db, so the sample still receives real
    injected noise. Fallbacks slightly dent uniformity at the top of the
    band and are counted so they can be reported.
    """
    order = np.argsort(snr_db)                 # ascending
    snr_sorted  = snr_db[order]
    pool_sorted = pool[order]

    idxs    = np.empty(n_target, dtype=pool.dtype)
    targets = np.empty(n_target, dtype=np.float64)
    n_fb    = 0
    for k in range(n_target):
        t  = rng.uniform(*snr_range)
        lo = np.searchsorted(snr_sorted, t + margin_db)    # first candidate
        if lo >= len(pool_sorted):                          # no candidate
            n_fb += 1
            t  = max(snr_sorted[-1] - margin_db, snr_range[0])
            lo = min(np.searchsorted(snr_sorted, t + margin_db),
                     len(pool_sorted) - 1)
        j = rng.integers(lo, len(pool_sorted))
        idxs[k], targets[k] = pool_sorted[j], t
    return idxs, targets, n_fb


def build(raw_npz, out_path, n_train, n_val, n_test, seed):
    data      = np.load(raw_npz, allow_pickle=True)
    X_raw     = data["X"]            # (N, 5, 256) complex64
    label_9   = data["label_9"]      # (N,) str
    split_ids = data["split"]        # (N,) int8
    edge_flag = data["edge_flag"]    # (N,) int8 — 1 = near FOV edge (synthetic padding)
    rng       = np.random.default_rng(seed)

    targets = {"train": n_train, "val": n_val, "test": n_test}
    total   = sum(targets[s] for s in targets) * len(CLASSES)
    print(f"Preparing {total} samples "
          f"(train {n_train}, val {n_val}, test {n_test} per class × {len(CLASSES)} classes)")
    print(f"SNR norm: method={SNR_METHOD}, band={SNR_RANGE} dB, seed={seed}, "
          f"strict={SNR_STRICT}"
          + (f" (margin {SNR_MARGIN_DB} dB, exclude below "
             f"{SNR_RANGE[0] + SNR_MARGIN_DB:.0f} dB)" if SNR_STRICT else ""))
    print(f"Edge-of-FOV filter: dropping edge_flag==1 samples before pool construction\n")

    # "full_edge" uses a SHARED floor constant, computed once from every class
    # except SNR_FLOOR_EXCLUDE_CLASSES (default: R only). This must be
    # activated before any preprocess_segment() call below — see
    # snr_norm.py module docstring "Method history" and plot_doppler_spectrum_signatures.py,
    # which computes the identical constant the same way so the noise
    # applied to training data matches the noise shown in the figure.
    if SNR_METHOD == "full_edge":
        shared_sigma2 = compute_shared_floor_full_edge(
            X_raw, label_9, edge_flag=edge_flag,
            exclude_classes=SNR_FLOOR_EXCLUDE_CLASSES,
        )
        set_global_sigma2_full_edge(shared_sigma2)
        print(f"full_edge shared floor: sigma^2 = {10*np.log10(shared_sigma2):.2f} dB "
              f"(pooled from all classes except {SNR_FLOOR_EXCLUDE_CLASSES})\n")

    # Report edge-flag counts before filtering
    n_edge_total = int((edge_flag == 1).sum())
    print(f"  edge_flag==1 samples dropped: {n_edge_total} / {len(X_raw)} total")
    for split, code in SPLIT_MAP.items():
        n_drop = int(((edge_flag == 1) & (split_ids == code)).sum())
        n_kept = int(((edge_flag == 0) & (split_ids == code)).sum())
        print(f"    {split:5s}: {n_drop} dropped, {n_kept} kept")
    print()

    X_out   = np.empty((total, 5, N_AZ), dtype=np.float32)
    lab_out = np.empty(total, dtype=object)
    spl_out = np.empty(total, dtype=np.int8)
    snr_out = np.empty(total, dtype=np.float32)

    snr_by_class = {cls: [] for cls in CLASSES}   # for per-class SNR stats

    k = 0
    for split, code in SPLIT_MAP.items():
        n_target = targets[split]
        for cls in CLASSES:
            # exclude edge-of-FOV samples — they have synthetic padding, not real azimuth data
            pool = np.where(
                (label_9 == cls) & (split_ids == code) & (edge_flag == 0)
            )[0]
            if len(pool) == 0:
                raise ValueError(f"No {cls} samples in split '{split}' after edge filter")

            if SNR_STRICT:
                snr_db = pool_snr_db(X_raw, pool, shared_sigma2)
                usable = snr_db >= SNR_RANGE[0] + SNR_MARGIN_DB   # exclusion rule
                n_excl = int((~usable).sum())
                pool, snr_db = pool[usable], snr_db[usable]
                if len(pool) == 0:
                    raise ValueError(
                        f"No {cls} samples in split '{split}' above "
                        f"{SNR_RANGE[0] + SNR_MARGIN_DB:.0f} dB (exclusion rule)")
                idxs, tgts, n_fb = strict_draw(
                    pool, snr_db, n_target, rng, SNR_RANGE, SNR_MARGIN_DB)
            else:
                idxs = draw_indices(pool, n_target, rng)
                tgts, n_fb, n_excl = [None] * n_target, 0, 0

            reuse  = n_target / len(pool)
            n_used = len(np.unique(idxs))
            for idx, tgt in zip(idxs, tgts):
                seg = X_raw[idx]
                X_out[k], _, snr_eff = preprocess_segment(
                    seg, rng=rng,
                    snr_method=SNR_METHOD,
                    snr_range=SNR_RANGE,
                    return_snr=True,
                    snr_target_db=tgt,
                )
                lab_out[k] = cls
                spl_out[k] = code
                snr_out[k] = snr_eff
                snr_by_class[cls].append(snr_eff)
                k += 1
            print(f"  {split:5s} {cls:3s}: {len(pool):5d} usable, {n_used} used "
                  f"→ {n_target} (reuse ×{reuse:.1f}"
                  + (f", excluded {n_excl}, fallbacks {n_fb}" if SNR_STRICT else "")
                  + ")")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X_proc=X_out,
        label_9=lab_out.astype(str),
        split=spl_out,
        snr_eff=snr_out,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {k} samples → {out_path}  ({size_mb:.0f} MB)")

    # ── per-class realized SNR report ──────────────────────────────────────
    print(f"\nRealized SNR distribution per class (method={SNR_METHOD}, target=Uniform{SNR_RANGE} dB):")
    print(f"  {'class':>6}  {'n':>7}  {'mean':>7}  {'std':>5}  {'p10':>7}  {'p50':>7}  {'p90':>7}  dB")
    for cls in CLASSES:
        v = np.array(snr_by_class[cls])
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        print(f"  {cls:>6}  {len(v):>7}  {v.mean():+7.2f}  {v.std():5.2f}"
              f"  {p10:+7.2f}  {p50:+7.2f}  {p90:+7.2f}")

    # sample counts per split after edge filter
    print(f"\nSample counts per split (output):")
    for split, code in SPLIT_MAP.items():
        n = int((spl_out == code).sum())
        print(f"  {split:5s}: {n}")


def main():
    global SNR_STRICT
    p = argparse.ArgumentParser(description="Bake the paper's SNR normalisation into a fixed dataset.")
    p.add_argument("--train", type=int, default=DEFAULT_TRAIN, help="samples per class, train split")
    p.add_argument("--val",   type=int, default=DEFAULT_VAL,   help="samples per class, val split")
    p.add_argument("--test",  type=int, default=DEFAULT_TEST,  help="samples per class, test split")
    p.add_argument("--seed",  type=int, default=0)
    p.add_argument("--raw",   type=str, default=str(RAW_NPZ))
    p.add_argument("--out",   type=str, default=str(DEFAULT_OUT))
    # Cross-eval experiment: override the module-level SNR_STRICT switch (line ~90)
    # without editing it by hand. Omit this flag to keep the current default (True).
    p.add_argument("--strict", type=str, default=None, choices=["true", "false"],
                    help="override SNR_STRICT for this run only (true/false); "
                         "default keeps the module setting (currently %s)" % SNR_STRICT)
    a = p.parse_args()
    if a.strict is not None:
        SNR_STRICT = (a.strict == "true")
    build(a.raw, a.out, a.train, a.val, a.test, a.seed)


if __name__ == "__main__":
    main()
