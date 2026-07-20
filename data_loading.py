"""
data_loading.py
---------------
Loads the raw SAAB SIRS 77 GHz FMCW dataset (.npy) and produces a flat,
structured array of all 75 868 scan segments ready for preprocessing.

Raw file layout (130 x 6 object array):
  col 0 : label string (15 raw categories)
  col 1 : complex matrix (1280, N) — N segments per recording
           1280 = 5 range cells x 256 azimuth samples (flattened, C-order)
  col 2 : range to centre cell [m], shape (N,)
  col 3 : time of segment centre [s],  shape (N,)
  col 4 : original split — 1=train, 2=val, 3=test
  col 5 : edge-truncation flag — 1 if segment is near FOV edge

Output (.npz saved to data/processed/segments.npz):
  X           (75868, 5, 256) complex64  — raw segments
  label_raw   (75868,)        str        — original label
  label_9     (75868,)        str        — mapped to 9-class scheme
  range_m     (75868,)        float32    — range [m]
  time_s      (75868,)        float32    — time [s]
  split       (75868,)        int8       — 1/2/3
  edge_flag   (75868,)        int8       — 0/1
  track_id    (75868,)        int16      — index into original 130 rows
"""

from pathlib import Path
import numpy as np

from config import RAW_NPY_PATH, DATASET_PATH

# ── paths ────────────────────────────────────────────────────────────────────
DEFAULT_RAW = RAW_NPY_PATH
DEFAULT_OUT = DATASET_PATH

# ── label mapping: 15 raw → 9 classes (matching paper Table I) ───────────────
LABEL_MAP = {
    "D1": "D1", "D2": "D2", "D3": "D3",
    "D4": "D4", "D5": "D5", "D6": "D6",
    "human_walk": "H", "human_run": "H",
    "seagull": "B", "pigeon": "B", "raven": "B",
    "black-headed gull": "B",
    "seagull and black-headed gull": "B",
    "heron": "B",
    "CR": "R",
}


def load_dataset(raw_path: Path = DEFAULT_RAW) -> dict:
    """
    Parse the raw .npy file and return a flat dict of aligned arrays.
    No signal processing is performed here.
    """
    raw_path = Path(raw_path)
    print(f"Loading {raw_path} …")
    data = np.load(raw_path, allow_pickle=True)          # (130, 6) object
    assert data.shape == (130, 6), f"Unexpected shape {data.shape}"

    X_parts        = []
    label_raw_parts = []
    label_9_parts  = []
    range_parts    = []
    time_parts     = []
    split_parts    = []
    edge_parts     = []
    track_parts    = []

    for track_id, row in enumerate(data):
        label_raw = str(row[0][0])           # e.g. "D1", "seagull"
        segments  = row[1]                   # (1280, N) complex
        ranges    = row[2].ravel()           # (N,)
        times     = row[3].ravel()           # (N,)
        splits    = row[4].ravel()           # (N,)
        edges     = row[5].ravel()           # (N,)

        N = segments.shape[1]

        # Reshape all N segments at once: (1280,N).T → (N,1280) → (N,5,256)
        X_track = segments.T.reshape(N, 5, 256).astype(np.complex64)

        X_parts.append(X_track)
        label_raw_parts.append(np.full(N, label_raw))
        label_9_parts.append(np.full(N, LABEL_MAP[label_raw]))
        range_parts.append(ranges.astype(np.float32))
        time_parts.append(times.astype(np.float32))
        split_parts.append(splits.astype(np.int8))
        edge_parts.append(edges.astype(np.int8))
        track_parts.append(np.full(N, track_id, dtype=np.int16))

    dataset = dict(
        X          = np.concatenate(X_parts,         axis=0),
        label_raw  = np.concatenate(label_raw_parts, axis=0),
        label_9    = np.concatenate(label_9_parts,   axis=0),
        range_m    = np.concatenate(range_parts,     axis=0),
        time_s     = np.concatenate(time_parts,      axis=0),
        split      = np.concatenate(split_parts,     axis=0),
        edge_flag  = np.concatenate(edge_parts,      axis=0),
        track_id   = np.concatenate(track_parts,     axis=0),
    )

    n = len(dataset["X"])
    print(f"Loaded {n} segments from 130 recordings.")
    print(f"X shape : {dataset['X'].shape}  dtype: {dataset['X'].dtype}")
    print("\nSamples per 9-class label:")
    for cls in sorted(set(LABEL_MAP.values())):
        count = int((dataset["label_9"] == cls).sum())
        print(f"  {cls:4s}: {count:6d}")

    return dataset


def save_processed(dataset: dict, out_path: Path = DEFAULT_OUT) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def plot_range_doppler(dataset: dict, classes=None) -> None:
    """
    For each requested class, take the first available segment,
    run a 1D FFT along the azimuth axis (axis=1) for each of the
    5 range rows, and display the resulting power map.

    No steering vector, no normalisation — raw signal only.
    Y axis : range cell (0 = nearest, 4 = farthest)
    X axis : Doppler frequency [Hz]  (fftshift so 0 Hz is centre)
    Colour : power in dB
    """
    import matplotlib.pyplot as plt

    PRF   = 17_000          # pulse repetition frequency [Hz]
    N_az  = 256             # azimuth samples per segment
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(N_az, d=1.0 / PRF))

    if classes is None:
        classes = ["D1", "D2", "D3", "D4", "D5", "D6", "H", "B", "R"]

    ncols = 3
    nrows = -(-len(classes) // ncols)          # ceiling division
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.2))
    axes = axes.flat

    for ax, cls in zip(axes, classes):
        idx = np.where(dataset["label_9"] == cls)[0][0]   # first segment
        seg = dataset["X"][idx]                            # (5, 256) complex

        # 1D FFT along azimuth axis for each of the 5 range rows
        spectrum = np.abs(
            np.fft.fftshift(np.fft.fft(seg, axis=1), axes=1)
        ) ** 2                                             # (5, 256) power

        power_db = 10 * np.log10(spectrum + 1e-10)        # dB, avoid log(0)

        im = ax.imshow(
            power_db,
            aspect="auto",
            origin="lower",
            extent=[doppler_hz[0], doppler_hz[-1], 0.5, 5.5],
            cmap="inferno",
        )
        ax.set_title(cls, fontsize=11, fontweight="bold")
        ax.set_xlabel("Doppler [Hz]", fontsize=8)
        ax.set_ylabel("Range cell", fontsize=8)
        ax.set_yticks([1, 2, 3, 4, 5])
        fig.colorbar(im, ax=ax, label="dB", pad=0.02)

    # hide unused axes
    for ax in axes:
        ax.set_visible(False)

    fig.suptitle(
        "Range-Doppler power map — 1D FFT per range row — raw signal (no preprocessing)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ds = load_dataset()
    save_processed(ds)          # write data/processed/segments.npz
