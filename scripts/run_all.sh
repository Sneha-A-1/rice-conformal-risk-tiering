#!/usr/bin/env bash
# Full pipeline (CPU). Single-threaded experiments run in two lanes after training (see run_after_training.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/verify_data.py
python3 src/data/audit_and_split.py
python3 src/data/build_cache.py
python3 src/models/extract_imagenet_features.py
python3 src/evaluation/leakage.py
bash scripts/train_all_seeds.sh
python3 src/evaluation/benchmark_latency.py
python3 src/explain/gradcam.py --seed 0
bash scripts/run_after_training.sh
python3 src/evaluation/coverage_variability.py
python3 src/scheduling/run_experiments.py --exp uploadall
python3 scripts/make_tables.py
python3 scripts/make_figures.py
