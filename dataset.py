"""
dataset.py
----------
PyTorch Dataset for the preprocessed FMCW segments.

Loads the flat dataset (from data_loading.py), applies preprocessing
(from preprocessing.py), and returns tensors ready for the DataLoader.

Split convention (from original paper data):
  1 = train   2 = validation   3 = test

Class balancing (matching paper Section III):
  Total drone samples == total non-drone samples.
  The minority group is oversampled by repeating indices.
"""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import CLASS_TO_IDX
from preprocessing import preprocess_segment

DRONE_CLASSES    = {"D1", "D2", "D3", "D4", "D5", "D6"}
NONDRONE_CLASSES = {"H", "B", "R"}


def balance_drone_nondrone(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Oversample the minority group (drone or non-drone) so both totals are equal.

    Matches paper Section III: total drone samples == total non-drone samples.
    """
    drone_idx    = indices[np.isin(labels[indices], list(DRONE_CLASSES))]
    nondrone_idx = indices[np.isin(labels[indices], list(NONDRONE_CLASSES))]

    n_drone    = len(drone_idx)
    n_nondrone = len(nondrone_idx)

    if n_drone < n_nondrone:
        drone_idx = np.tile(drone_idx, int(np.ceil(n_nondrone / n_drone)))[:n_nondrone]
    elif n_nondrone < n_drone:
        nondrone_idx = np.tile(nondrone_idx, int(np.ceil(n_drone / n_nondrone)))[:n_drone]

    balanced = np.concatenate([drone_idx, nondrone_idx])
    np.random.shuffle(balanced)
    return balanced


class FMCWDataset(Dataset):
    """
    Parameters
    ----------
    X_raw      : (N, 5, 256) complex64 — raw segments
    labels     : (N,) str              — 9-class label strings
    split_ids  : (N,) int8             — 1/2/3
    split      : 'train' | 'val' | 'test'
    balance    : bool — oversample minority group to equalise drone/non-drone counts
                        (only applied to train split, as in the paper)
    """

    SPLIT_MAP = {"train": 1, "val": 2, "test": 3}

    def __init__(
        self,
        X_raw: np.ndarray,
        labels: np.ndarray,
        split_ids: np.ndarray,
        split: str = "train",
        balance: bool = True,
    ):
        split_code = self.SPLIT_MAP[split]
        mask = split_ids == split_code

        self.X_raw = X_raw[mask]
        self.labels = labels[mask]
        self.indices = np.arange(len(self.X_raw))

        if balance and split == "train":
            self.indices = self._balance(self.labels, self.indices)

    # ── balancing ─────────────────────────────────────────────────────────────
    @staticmethod
    def _balance(labels: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Repeat indices in the minority group (drone or non-drone)
        until both groups are equal in size — matches paper Section III.
        """
        drone_idx    = indices[np.isin(labels[indices], list(DRONE_CLASSES))]
        nondrone_idx = indices[np.isin(labels[indices], list(NONDRONE_CLASSES))]

        n_drone    = len(drone_idx)
        n_nondrone = len(nondrone_idx)

        if n_drone < n_nondrone:
            repeats = int(np.ceil(n_nondrone / n_drone))
            drone_idx = np.tile(drone_idx, repeats)[:n_nondrone]
        elif n_nondrone < n_drone:
            repeats = int(np.ceil(n_drone / n_nondrone))
            nondrone_idx = np.tile(nondrone_idx, repeats)[:n_drone]

        balanced = np.concatenate([drone_idx, nondrone_idx])
        np.random.shuffle(balanced)
        return balanced

    # ── Dataset interface ─────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = self.indices[i]
        x   = preprocess_segment(self.X_raw[idx])       # (5, 150) float32
        x   = torch.from_numpy(x).unsqueeze(0)          # (1, 5, 150)
        y   = CLASS_TO_IDX[self.labels[idx]]
        return x, y


def make_dataloaders(
    dataset_path: Path,
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict:
    """
    Load the .npz file and return train / val / test DataLoaders.
    """
    data = np.load(dataset_path, allow_pickle=True)
    X_raw     = data["X"]           # (N, 5, 256) complex64
    labels    = data["label_9"]     # (N,) str
    split_ids = data["split"]       # (N,) int8

    loaders = {}
    for split in ("train", "val", "test"):
        ds = FMCWDataset(
            X_raw, labels, split_ids,
            split=split,
            balance=(split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"  {split:5s}: {len(ds):6d} samples")

    return loaders


# ─────────────────────────────────────────────────────────────────────────────
# Pre-generated, SNR-normalised dataset (from prepare_snr_dataset.py)
# These maps already have the paper's Gaussian noise added and are already
# preprocessed (FFT + log-norm) and class-balanced, so there is NO on-the-fly
# preprocessing or balancing here — we just serve the stored tensors.
# ─────────────────────────────────────────────────────────────────────────────
class PreprocessedFMCWDataset(Dataset):
    """Serves already-preprocessed (5, 150) float maps for one split.

    For the train split, applies drone vs. non-drone balancing (paper Section III):
    each non-drone class (H, B, R) is tiled individually from 10,000 to 20,000
    samples so the total non-drone count (60,000) equals the total drone count
    (60,000), giving 120,000 training samples in total.
    Val and test splits are served unchanged.
    """

    SPLIT_MAP = {"train": 1, "val": 2, "test": 3}

    def __init__(self, X_proc, labels, split_ids, split, balance: bool = True):
        mask          = split_ids == self.SPLIT_MAP[split]
        X_split       = X_proc[mask]
        lab_split     = labels[mask]

        if balance and split == "train":
            X_split, lab_split = self._balance_per_class(X_split, lab_split)

        self.X_proc = X_split
        self.labels = lab_split

    @staticmethod
    def _balance_per_class(X: np.ndarray, labels: np.ndarray):
        """Tile each non-drone class individually to match the per-class drone target.

        Drone classes (D1-D6) each have N_d samples and are left untouched.
        Non-drone target per class = drone_total / len(NONDRONE_CLASSES).
        With 6 × 10,000 = 60,000 drones and 3 non-drone classes the target is
        20,000 per non-drone class, giving equal totals: 60,000 each side.

        Tiling is done class-by-class (not from a combined pool) so each
        non-drone class lands at exactly the target count.
        No new noise is generated — these are copies of existing stored samples.
        """
        drone_total  = int(np.sum(np.isin(labels, list(DRONE_CLASSES))))
        nd_target    = drone_total // len(NONDRONE_CLASSES)   # 60000 // 3 = 20000

        drone_idx    = np.where(np.isin(labels, list(DRONE_CLASSES)))[0]
        all_idx      = [drone_idx]

        for cls in sorted(NONDRONE_CLASSES):               # H, B, R — deterministic order
            cls_idx = np.where(labels == cls)[0]
            reps    = int(np.ceil(nd_target / len(cls_idx)))
            tiled   = np.tile(cls_idx, reps)[:nd_target]
            all_idx.append(tiled)

        combined = np.concatenate(all_idx)
        np.random.shuffle(combined)
        return X[combined], labels[combined]

    def __len__(self):
        return len(self.X_proc)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X_proc[i]).unsqueeze(0)   # (1, 5, 150)
        y = CLASS_TO_IDX[self.labels[i]]
        return x, y


def make_snr_dataloaders(
    dataset_path: Path,
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict:
    """Load the SNR-normalised .npz (segments_snr.npz) and return loaders.

    Preprocessing is already baked in by prepare_snr_dataset.py.
    Drone vs. non-drone balancing is applied to the train split only
    (PreprocessedFMCWDataset._balance_per_class): each non-drone class
    is tiled individually from 10,000 to 20,000 samples so drone total
    (60,000) == non-drone total (60,000), giving 120,000 train samples.
    Val and test are served as-is (4,000 per class, 36,000 each).
    """
    data      = np.load(dataset_path, allow_pickle=True)
    X_proc    = data["X_proc"]      # (M, 5, 150) float32
    labels    = data["label_9"]     # (M,) str
    split_ids = data["split"]       # (M,) int8

    loaders = {}
    for split in ("train", "val", "test"):
        ds = PreprocessedFMCWDataset(
            X_proc, labels, split_ids, split,
            balance=(split == "train"),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        print(f"  {split:5s}: {len(ds):6d} samples")

    return loaders
