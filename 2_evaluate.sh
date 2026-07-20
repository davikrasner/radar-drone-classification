#!/usr/bin/env bash
# Step 2 — evaluate the best model on the TEST set (accuracy + confusion matrix)
set -e
cd "$(dirname "$0")"
python3 evaluate.py
