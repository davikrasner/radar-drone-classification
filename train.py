"""
train.py
--------
Training script — run from the project root:

    python train.py

Matches the paper:
  - Adam  β1=0.9, β2=0.999, ε=1e-8, lr=0.001
  - Batch size 128
  - 25 epochs
  - Trains 10 independent networks, keeps the one with best validation accuracy
  - Cross-entropy loss
  - He initialisation (done inside model.py)

Outputs:
  - models/best_model.pt          best network (highest validation accuracy)
  - results/history.csv           per-epoch loss/accuracy of the best run
  - results/figures/loss_curve.png      train vs val loss
  - results/figures/accuracy_curve.png  train vs val accuracy
"""

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from config import (
    DATASET_PATH, SNR_DATASET_PATH, MODELS_DIR, RESULTS_DIR, FIGURES_DIR,
    N_CLASSES, BATCH_SIZE, N_EPOCHS, LR, N_RUNS, SEED,
    USE_SNR_DATASET, get_device,
)

CHECKPOINT_PATH = MODELS_DIR / "checkpoint.pt"
from model   import FMCWClassifier
from dataset import make_dataloaders, make_snr_dataloaders

DEVICE = get_device()


def set_seed(seed: int = SEED):
    """Fix all random sources so a full run is repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── helpers ───────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer=None):
    """One forward pass over loader. If optimizer is given → training mode."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss   = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


def train_one_network(loaders, run_id: int):
    model     = FMCWClassifier(n_classes=N_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

    best_val_acc = 0.0
    best_state   = None
    # per-epoch history for this run (used later for the curves)
    history = {"epoch": [], "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(1, N_EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, loaders["train"], criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, loaders["val"],   criterion)

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

        print(f"  run {run_id:2d}  epoch {epoch:2d}/{N_EPOCHS}"
              f"  train {tr_acc:.3f}  val {vl_acc:.3f}"
              f"  {'★' if vl_acc == best_val_acc else ' '}")

    return best_state, best_val_acc, history


# ── saving history & plotting ──────────────────────────────────────────────────
def save_history_csv(history: dict, path: Path):
    """Write the per-epoch numbers so the curves can be re-drawn later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc"]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in range(len(history["epoch"])):
            writer.writerow([history[c][i] for c in cols])
    print(f"Saved history  → {path}")


def plot_curves(history: dict, figures_dir: Path):
    """Two figures: loss (down is good) and accuracy (up is good)."""
    import matplotlib
    matplotlib.use("Agg")            # no display needed
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    epochs = history["epoch"]

    # loss curve
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"],   label="validation")
    plt.xlabel("epoch"); plt.ylabel("loss")
    plt.title("Loss per epoch (best run)")
    plt.legend(); plt.grid(True, alpha=0.3)
    loss_path = figures_dir / "loss_curve.png"
    plt.savefig(loss_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved figure   → {loss_path}")

    # accuracy curve
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"],   label="validation")
    plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.title("Accuracy per epoch (best run)")
    plt.legend(); plt.grid(True, alpha=0.3)
    acc_path = figures_dir / "accuracy_curve.png"
    plt.savefig(acc_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved figure   → {acc_path}")


# ── checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(run: int, best_acc: float, best_state: dict, best_history: dict):
    """Save progress after each completed run so training can be resumed."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "run":          run,
        "best_acc":     best_acc,
        "best_state":   best_state,
        "best_history": best_history,
    }, CHECKPOINT_PATH)
    print(f"  checkpoint saved (run {run}/{N_RUNS})")


def load_checkpoint():
    """Load checkpoint if it exists, otherwise return None."""
    if CHECKPOINT_PATH.exists():
        cp = torch.load(CHECKPOINT_PATH, map_location="cpu")
        print(f"Resuming from checkpoint — runs 1–{cp['run']} already done, "
              f"best val acc so far: {cp['best_acc']:.4f}")
        return cp
    return None


# ── CLI overrides (cross-eval experiment) ───────────────────────────────────
# Both flags default to the existing config.py paths, so `python train.py`
# with no flags behaves exactly as before.
def parse_args():
    p = argparse.ArgumentParser(description="Train the FMCW classifier.")
    p.add_argument("--dataset", type=str, default=None,
                    help="override SNR_DATASET_PATH (only used when USE_SNR_DATASET=True); "
                         f"default: {SNR_DATASET_PATH}")
    p.add_argument("--model-out", type=str, default=None,
                    help=f"override save path for the trained model; default: {MODELS_DIR / 'best_model.pt'}")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    dataset_path = Path(args.dataset) if args.dataset else SNR_DATASET_PATH
    model_save_path = Path(args.model_out) if args.model_out else MODELS_DIR / "best_model.pt"

    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Seed   : {SEED}")
    print(f"Dataset: {DATASET_PATH}\n")
    print("Building data loaders …")
    if USE_SNR_DATASET:
        print(f"  source: SNR-normalised dataset {dataset_path.name}")
        loaders = make_snr_dataloaders(dataset_path, batch_size=BATCH_SIZE)
    else:
        print(f"  source: clean dataset {DATASET_PATH.name}")
        loaders = make_dataloaders(DATASET_PATH, batch_size=BATCH_SIZE)

    # resume from checkpoint if available
    cp = load_checkpoint()
    start_run            = cp["run"] + 1      if cp else 1
    overall_best_acc     = cp["best_acc"]     if cp else 0.0
    overall_best_state   = cp["best_state"]   if cp else None
    overall_best_history = cp["best_history"] if cp else None

    print(f"\nTraining {N_RUNS} networks × {N_EPOCHS} epochs "
          f"(runs {start_run}–{N_RUNS}) …\n")
    t0 = time.time()

    for run in range(start_run, N_RUNS + 1):
        state, val_acc, history = train_one_network(loaders, run)
        print(f"  → run {run} best val acc: {val_acc:.4f}\n")

        if val_acc > overall_best_acc:
            overall_best_acc     = val_acc
            overall_best_state   = state
            overall_best_history = history

        save_checkpoint(run, overall_best_acc, overall_best_state, overall_best_history)

    elapsed = time.time() - t0
    print(f"Training done in {elapsed/60:.1f} min")
    print(f"Best validation accuracy: {overall_best_acc:.4f}")

    # save best model
    model = FMCWClassifier(n_classes=N_CLASSES)
    model.load_state_dict(overall_best_state)
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved best model → {model_save_path}")

    # save history of the best run + draw curves
    save_history_csv(overall_best_history, RESULTS_DIR / "history.csv")
    plot_curves(overall_best_history, FIGURES_DIR)

    # clean up checkpoint — training completed successfully
    CHECKPOINT_PATH.unlink(missing_ok=True)
    print(f"Checkpoint removed.")

    print("\nNext step: run  python evaluate.py  "
          "to score the model on the test set.")


if __name__ == "__main__":
    main()
