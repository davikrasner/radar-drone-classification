#!/usr/bin/env bash
# Step 0 — prepare data:
#   raw .npy → segments.npz → segments_snr.npz (SNR norm + β₃ baked in)
set -e
cd "$(dirname "$0")"
python3 data_loading.py
python3 prepare_snr_dataset.py
