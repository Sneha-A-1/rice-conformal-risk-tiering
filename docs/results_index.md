# Result files

Every file below is written by the script in the second column; `scripts/make_tables.py` turns them into the
LaTeX macros and tables in `manuscript/generated/`, and `scripts/make_figures.py` into the figures in
`manuscript/figures/`. Nothing in the manuscript is typed by hand.

| File (in `results/raw/`) | Produced by | Experiment | Used in |
|---|---|---|---|
| `../../data/processed/audit_summary.json`, `paddy_index.csv`, `sethy_index.csv` | `src/data/audit_and_split.py` | near-duplicate audit, group-aware splits | Table 1, Fig. 2, Fig. S1 |
| `eval_e0_leakage.csv` | `src/evaluation/leakage.py` | E0: random vs group-aware split (frozen features) | Table 3 |
| `models/seed{0..4}/` (`best.pt`, `history.csv`, logits, `run.json`) | `src/models/train.py` | edge-model training | E1-E4, Fig. 3 |
| `imagenet_feats_paddy.npy` | `src/models/extract_imagenet_features.py` | frozen ImageNet features | E0, E5 |
| `eval_e1_accuracy_calibration.json`, `eval_per_class.csv`, `eval_confusion_ensemble.csv` | `src/evaluation/evaluate_models.py` | E1: accuracy and calibration | Table 4, Fig. 3, Tables S1-S2 |
| `eval_e2_conformal_splits.csv` | `src/evaluation/evaluate_models.py` | E2: conformal coverage and tiers (100 splits x 6 alphas x 3 methods) | Table 5, Fig. 4, Table S3 |
| `eval_e2b_split_variability.csv` | `src/evaluation/coverage_variability.py` | E2b: coverage variability, group-aware vs image-level calibration splits | Section 3.3 |
| `eval_e3_cascade_splits.csv` | `src/evaluation/evaluate_models.py` | E3: cascade gates at matched escalation | Table 6, Fig. 5 |
| `eval_e4_shift.json`, `eval_e4_recalibration.csv` | `src/evaluation/evaluate_models.py` | E4: cross-source shift and recalibration | Table 7, Fig. 5 |
| `eval_e7_variety.csv` | `src/evaluation/evaluate_models.py` | E7: per-variety accuracy and coverage | Table S4 |
| `poisoning_results.csv` | `src/security/poisoning.py` | E5: label-flipping poisoning and defenses | Table 8, Fig. 6, Table S5 |
| `xai_gradcam_eval.csv`, `xai_gradcam_examples.npz` | `src/explain/gradcam.py` | E6: Grad-CAM faithfulness and stability | Fig. 7 |
| `sched_pool_summary.json`, `sched_outbreak.csv`, `sched_outbreak_tests.json` | `src/scheduling/run_experiments.py --exp outbreak` | E8: outbreak scenario, ablations, upload-all | Tables 9-10, Fig. 8, Table S6 |
| `sched_uploadall.csv` | `... --exp uploadall` | E8: paired upload-all workload (alert latency of device-URGENT images) | Table 10, Fig. 8D |
| `sched_load.csv` | `... --exp load` | E8: load sweep | Fig. 8, Table S7 |
| `sched_sensitivity.csv` | `... --exp sens` | E8: sensitivity grid | Table S8 |
| `sched_convergence.npz` | `... --exp conv` | GA/PSO convergence | Fig. S2 |
| `latency_benchmark.json`, `latency_edge_s.npy`, `latency_ensemble_s.npy` | `src/evaluation/benchmark_latency.py` | E9: inference, Grad-CAM and HMAC cost | Section 3.8, simulation inputs |
| `logs/` | all pipeline scripts | console logs with timestamps | - |

`results/tables/sched_paired_tests.csv` holds the full paired-test output (all metrics) behind Table S6.
