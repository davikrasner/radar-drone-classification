"""
config.py
---------
Single source of truth for shared constants, paths, and device selection.
All pipeline scripts import from here — no duplication across files.
"""

from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parent

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_NPY_PATH     = PROJECT_ROOT / "data_SAAB_SIRS_77GHz_FMCW.npy"
DATASET_PATH     = PROJECT_ROOT / "data" / "processed" / "segments.npz"
SNR_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "segments_snr.npz"
MODEL_PATH       = PROJECT_ROOT / "models" / "best_model.pt"
MODELS_DIR       = PROJECT_ROOT / "models"
RESULTS_DIR      = PROJECT_ROOT / "results"
FIGURES_DIR      = RESULTS_DIR / "figures"

# ── classes ───────────────────────────────────────────────────────────────────
CLASSES      = ["D1", "D2", "D3", "D4", "D5", "D6", "H", "B", "R"]
N_CLASSES    = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# ── training hyperparameters ──────────────────────────────────────────────────
BATCH_SIZE = 128
N_EPOCHS   = 25
LR         = 1e-3
N_RUNS     = 10
SEED       = 42

# ── dataset switch ────────────────────────────────────────────────────────────
# True  = SNR-normalised dataset (segments_snr.npz) — matches the paper
# False = clean dataset (segments.npz) — original baseline without Gaussian noise
USE_SNR_DATASET = True


# ── device ────────────────────────────────────────────────────────────────────
def get_device():
    """Prefer Apple GPU (MPS), then NVIDIA (CUDA), else CPU."""
    if torch is None:
        raise ImportError("torch is not installed — run: pip install torch")
    if torch.backends.mps.is_available():
        print("✓ Using Apple GPU (MPS)")
        return torch.device("mps")
    if torch.cuda.is_available():
        print("✓ Using NVIDIA GPU (CUDA)")
        return torch.device("cuda")
    print("⚠ No GPU found — using CPU (slower)")
    return torch.device("cpu")
