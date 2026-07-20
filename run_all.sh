#!/usr/bin/env bash
# Run the whole pipeline in order: prepare -> train -> evaluate
set -e
cd "$(dirname "$0")"
./0_prepare.sh
./1_train.sh
./2_evaluate.sh
