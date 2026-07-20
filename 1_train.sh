#!/usr/bin/env bash
# Step 1 — train the network (saves models/best_model.pt + training curves)
set -e
cd "$(dirname "$0")"
python3 train.py
echo
echo "Training graphs saved to:"
echo "  results/figures/loss_curve.png"
echo "  results/figures/accuracy_curve.png"
