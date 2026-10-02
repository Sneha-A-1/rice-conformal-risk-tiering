# Conformal risk tiering for trustworthy edge–cloud surveillance of rice diseases and pests

Code, configuration and result files for the manuscript
*"Conformal risk tiering for trustworthy edge–cloud surveillance of rice diseases and pests"*
(prepared for *Frontiers in Bioengineering and Biotechnology*, Research Topic "Trustworthy and Secure AI
Frameworks for Bioengineering: Explainability, Risk Assessment, and Biosafety").

## Research problem
Plant-disease classifiers report very high accuracies, but (i) benchmark images contain many near-duplicates,
(ii) model confidence is not calibrated and breaks under distribution shift, (iii) cloud–edge agricultural
platforms schedule data without regard to the biological risk it carries, and (iv) crowd-sourced retraining
can be poisoned to conceal disease. This repository implements and evaluates a framework that turns calibrated,
distribution-free uncertainty into data-retrieval priorities and risk-aware scheduling.

## Contributions
1. Reproducible near-duplicate audit of two public rice datasets and a group-aware evaluation protocol.
2. Conformal risk tiering (URGENT / STRATEGIC / ROUTINE) with a finite-sample bound on missed high-severity alerts.
3. Uncertainty-gated edge→cloud cascade (MobileNetV3-Large on device, 5-member deep ensemble in the cloud).
4. Cross-source shift evaluation and few-group target recalibration.
5. Label-flipping poisoning study (concealment attack) with confident-learning and kNN filters and a trusted calibration set.
6. RADS, a risk- and deadline-aware scheduler for fog/cloud placement, compared with FCFS, RR, EDF, Min-Min, Max-Min, GA and PSO.

## Architecture
```
device:  capture -> MobileNetV3-L (+T) -> conformal set C(x) -> tier rule -> signed alert / image / record
fog:     HMAC check -> RADS (risk-ordered queues, priority links, priority-aware placement, urgent reservation)
cloud:   5-member ensemble confirmation -> retraining pool + poisoning filter -> trusted calibration -> updates
```
See `manuscript/figures/fig1_architecture.pdf`.

## Repository layout
```
configs/default.yaml            every setting used in the paper
src/data/                       audit (perceptual hashing, near-duplicate groups, group-aware splits), image cache
src/models/                     edge-model training (5 seeds), frozen ImageNet features
src/trust/                      temperature scaling, calibration metrics, split/class-conditional conformal prediction
src/evaluation/                 E0 leakage, E1-E4 + E7 evaluation, latency benchmark, statistics, plot style
src/explain/                    Grad-CAM and its faithfulness/stability evaluation
src/security/                   HMAC integrity + replay protection, poisoning attack and defenses
src/scheduling/                 discrete-event simulator, workload, schedulers (RADS + 7 baselines), experiments
scripts/                        pipeline scripts, table/figure generators, data preparation, weight download
data/                           data instructions, file manifests (SHA-256), processed indices (no images)
results/raw/                    raw result files (CSV/JSON) produced by the experiments
results/tables                  copies of the generated LaTeX tables and the full paired-test output
results/key_results.md          headline numbers (auto-generated)
manuscript/                     Frontiers LaTeX source (main.tex, sections/, generated/, figures/, references.bib)
```

## Installation
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_weights.sh          # ImageNet MobileNetV3-Large weights (SHA-256 checked)
```
Data: follow `data/README.md` (the images are not redistributed). `python scripts/verify_data.py` checks your files
against the manifests.

## Reproducing the results
All commands are run from the repository root; the full pipeline is in `scripts/run_all.sh`.
```bash
python src/data/audit_and_split.py          # near-duplicate audit + group-aware splits  -> data/processed/
python src/data/build_cache.py              # decoded image caches
bash   scripts/train_all_seeds.sh           # 5 edge models (about 1.3 h each on 2 CPU cores)
python src/models/extract_imagenet_features.py
python src/evaluation/leakage.py            # E0
python src/evaluation/benchmark_latency.py  # E9 (run on an idle machine)
python src/explain/gradcam.py --seed 0      # E6
python src/evaluation/evaluate_models.py    # E1-E4, E7
python src/evaluation/coverage_variability.py  # E2b
python src/security/poisoning.py --reps 10   # E5
python src/scheduling/run_experiments.py --exp outbreak   # E8
python src/scheduling/run_experiments.py --exp uploadall  # E8, paired upload-all workload
python src/scheduling/run_experiments.py --exp load
python src/scheduling/run_experiments.py --exp sens
python src/scheduling/run_experiments.py --exp conv
python scripts/make_tables.py && python scripts/make_figures.py
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd supplementary && pdflatex supplementary && pdflatex supplementary
```
Every number in the manuscript is a LaTeX macro written by `scripts/make_tables.py` from `results/raw/`.

## Key results
A table of headline numbers, regenerated from the raw result files every time `scripts/make_tables.py` runs, is in
[`results/key_results.md`](results/key_results.md). The full set of results, with confidence intervals and statistical
tests, is in the manuscript (`manuscript/main.pdf`) and its supplementary material (`manuscript/supplementary/`).

## Citation
Please cite the article once published. Dataset citations: Petchiammal et al. (2023),
doi:10.1145/3570991.3570994; Sethy et al. (2020), doi:10.1016/j.compag.2020.105527 and doi:10.17632/fwcj7stb8r.1.

## License
Code: MIT (see `LICENSE`). The datasets keep their own licenses and terms of use.
