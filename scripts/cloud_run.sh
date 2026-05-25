#!/bin/bash
# Quick-launch training on cloud.  Assumes cloud_setup.sh has already run.
# Usage:  bash scripts/cloud_run.sh [--fresh]
set -euo pipefail

cd ~/open_spiel_junqi
source venv/bin/activate

# Pull latest code
git pull origin exp

# Start training (logs written to ./othello_train_v2/train.log)
python train/train_othello.py "$@"
