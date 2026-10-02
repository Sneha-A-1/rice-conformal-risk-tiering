# Design notes (source of truth for the Methods section)

## Framework layers (maps to PhD objectives)
| Layer | Function | Objective |
|---|---|---|
| L1 Acquisition | smartphone / field-camera leaf images + metadata (variety, age) | 4 |
| L2 Edge inference | MobileNetV3-Large (one seed), temperature-scaled, conformal prediction set | 4 |
| L3 Risk-aware data retrieval | tiering URGENT / STRATEGIC / ROUTINE from conformal set + severity | 2 |
| L4 Fog/cloud analytics | 5-member deep ensemble (cloud model), recalibration, retraining pool | 4 |
| L5 Trust | calibration, conformal coverage, cross-region shift test, Grad-CAM faithfulness, subgroup (variety) coverage | 4 |
| L6 Security | HMAC-SHA256 message integrity + replay window; poisoning-robust model update (label-flip concealment attack + filter) | 4 |
| L7 Scheduling | risk- and deadline-aware online placement of offloaded jobs on fog/cloud | 3 |

## Severity tiers (literature-justified, sensitivity analysis over alternatives)
- High-severity set H (default): blast [dean2012, savary2019], bacterial_leaf_blight [mansfield2012, savary2019],
  tungro [azzam2002], bacterial_panicle_blight [ham2011].
- Broad H (sensitivity): + bacterial_leaf_streak, brown_spot [barnwal2013].
- normal = no risk.

## Tier rule (edge, per image x with conformal set C(x) at level alpha)
- URGENT    if C(x) ∩ H ≠ ∅
- STRATEGIC if not URGENT and (|C(x)| ≠ 1)          (ambiguous or empty set)
- ROUTINE   otherwise (singleton low-severity or healthy)
Guarantee: with class-conditional (Mondrian) calibration, P(tier ≠ URGENT | Y=k) ≤ alpha_k for every k in H.

## Data actions per tier
- URGENT: alert (prediction set, calibrated risk, Grad-CAM) to fog immediately; image to cloud ensemble for confirmation.
- STRATEGIC: image to cloud ensemble; added to retraining/recalibration pool.
- ROUTINE: only a signed metadata record uploaded (batched); image kept on device (pull on demand).

## Experiments
E1 accuracy/calibration: edge single model (5 seeds), cloud ensemble; temperature scaling on val.
E2 conformal: LAC marginal, LAC class-conditional, APS; 100 random group-aware cal/test splits of holdout.
E3 cascade: final-decision accuracy, high-severity recall, escalation rate vs gating rules (ours, max-softmax, entropy, random at matched rate).
E4 shift: Sethy (4 shared classes), cluster bootstrap over near-duplicate groups; target recalibration with m labeled groups.
E5 poisoning: frozen ImageNet features + logistic head; concealment attack (H -> normal) rho in {0,.05,.1,.2,.3}; filter defense.
E6 XAI: Grad-CAM deletion/insertion AUC vs random maps; stability under flips.
E7 subgroup: per-variety accuracy and coverage.
E8 scheduling simulation: data-driven workload (real tiers, real image sizes, measured latencies); baselines FCFS, RR, EDF, Min-Min, Max-Min, GA, PSO; ablations; 30 seeds; Wilcoxon + Holm, Cliff's delta.
E9 overheads: measured edge latency, HMAC cost, bytes uploaded vs upload-all.
