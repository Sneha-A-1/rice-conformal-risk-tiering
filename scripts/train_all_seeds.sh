#!/usr/bin/env bash
# Train the 5 edge-model seeds sequentially (CPU). ~1.5 h per seed on 2 cores.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/raw/logs
for s in 0 1 2 3 4; do
  if [ -f results/raw/models/seed$s/run.json ]; then echo "seed $s done, skipping"; continue; fi
  python3 src/models/train.py --seed $s 2>&1 | tee results/raw/logs/train_seed$s.log
done
