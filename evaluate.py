"""
evaluate.py
-----------
Evaluate the trained model on the TEST set — run from the project root:

    python evaluate.py

Why the test set?
  The best model was *chosen* using the validation set, so the validation
  number is a little optimistic. The test set was never used for any choice,
  so it gives the honest, final result for the paper reproduction.

Outputs:
  - prints overall test accuracy + per-class recall
  - results/figures/confusion_matrix.png   (9x9 heatmap)
  - results/confusion_matrix.csv           (raw counts)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from config import (
    DATASET_PATH, SNR_DATASET_PATH, MODEL_PATH, RESULTS_DIR, FIGURES_DIR,
    N_CLASSES, BATCH_SIZE, USE_SNR_DATASET, get_device, CLASSES,
)
from model   import FMCWClassifier
from dataset import make_dataloaders, make_snr_dataloaders

DEVICE = get_device()


# ── helpers ───────────────────────────────────────────────────────────────────
def collect_predictions(model, loader):
    """Run the model over a loader and return (y_true, y_pred) arrays."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            logits = model(X)
            preds  = logits.argmax(1).cpu().numpy()
            y_pred.append(preds)
            y_true.append(y.numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def confusion_matrix(y_true, y_pred, n_classes):
    """Counts: rows = true class, columns = predicted class."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def save_confusion_csv(cm, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + CLASSES)
        for i, row in enumerate(cm):
            writer.writerow([CLASSES[i]] + row.tolist())
    print(f"Saved counts   → {path}")


def plot_confusion(cm, figures_dir):
    """Normalised confusion-matrix heatmap (per-row = recall)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set, row-normalised)")

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            val = cm_norm[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val > 0.5 else "black", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out = figures_dir / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved figure   → {out}")


# ── CLI overrides (cross-eval experiment) ───────────────────────────────────
# Both flags default to the existing config.py paths, so `python evaluate.py`
# with no flags behaves exactly as before.
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the trained model on the TEST set.")
    p.add_argument("--dataset", type=str, default=None,
                    help="override SNR_DATASET_PATH — only its 'test' split is used; "
                         f"default: {SNR_DATASET_PATH}")
    p.add_argument("--model", type=str, default=None,
                    help=f"override MODEL_PATH to load; default: {MODEL_PATH}")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    dataset_path = Path(args.dataset) if args.dataset else SNR_DATASET_PATH
    model_path = Path(args.model) if args.model else MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train first with:  python train.py"
        )

    print(f"Device : {DEVICE}")
    print(f"Model  : {model_path}\n")
    print("Building data loaders …")
    if USE_SNR_DATASET:
        print(f"  source: SNR-normalised dataset {dataset_path.name}")
        loaders = make_snr_dataloaders(dataset_path, batch_size=BATCH_SIZE)
    else:
        print(f"  source: clean dataset {DATASET_PATH.name}")
        loaders = make_dataloaders(DATASET_PATH, batch_size=BATCH_SIZE)

    model = FMCWClassifier(n_classes=N_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    y_true, y_pred = collect_predictions(model, loaders["test"])

    overall_acc = (y_true == y_pred).mean()
    print(f"\nTest accuracy: {overall_acc:.4f}  ({len(y_true)} samples)\n")

    cm = confusion_matrix(y_true, y_pred, N_CLASSES)

    # per-class recall
    print("Per-class recall:")
    for i, name in enumerate(CLASSES):
        support = cm[i].sum()
        recall  = cm[i, i] / support if support else 0.0
        print(f"  {name:>3s}: {recall:.3f}  ({support} samples)")

    save_confusion_csv(cm, RESULTS_DIR / "confusion_matrix.csv")
    plot_confusion(cm, FIGURES_DIR)


if __name__ == "__main__":
    main()
