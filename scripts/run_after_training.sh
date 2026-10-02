#!/usr/bin/env bash
# Runs every post-training experiment in two single-threaded lanes (2 CPU cores, no oversubscription).
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=results/raw/logs
until [ -f results/raw/models/seed4/run.json ]; do sleep 60; done
echo "training complete $(date)" > $LOG/pipeline.log
(
  python3 src/evaluation/evaluate_models.py > $LOG/evaluate.log 2>&1
  echo "eval done $(date)" >> $LOG/pipeline.log
  python3 src/scheduling/run_experiments.py --exp outbreak > $LOG/sched_outbreak.log 2>&1
  echo "outbreak done $(date)" >> $LOG/pipeline.log
) &
(
  python3 src/security/poisoning.py --reps 10 > $LOG/poisoning.log 2>&1
  echo "poisoning done $(date)" >> $LOG/pipeline.log
  python3 src/scheduling/run_experiments.py --exp load > $LOG/sched_load.log 2>&1
  echo "load done $(date)" >> $LOG/pipeline.log
  python3 src/scheduling/run_experiments.py --exp sens > $LOG/sched_sens.log 2>&1
  python3 src/scheduling/run_experiments.py --exp conv > $LOG/sched_conv.log 2>&1
  echo "sens+conv done $(date)" >> $LOG/pipeline.log
) &
wait
echo "ALL DONE $(date)" >> $LOG/pipeline.log
