"""Generate LaTeX macros (manuscript/generated/numbers.tex) and tables (manuscript/generated/tab_*.tex)
from the result files. Every number quoted in the manuscript is produced here; nothing is typed by hand.
Sections whose result files do not exist yet are skipped (their macros render as '??')."""
import json
import pathlib
import platform
import re

import numpy as np
import pandas as pd
import yaml
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
RAW = pathlib.Path(__import__("os").environ.get("RESULTS_RAW", ROOT / "results" / "raw"))
PROC = ROOT / "data" / "processed"
GEN = pathlib.Path(__import__("os").environ.get("GEN_DIR", ROOT / "manuscript" / "generated"))
GEN.mkdir(parents=True, exist_ok=True)
M = {}            # macro name -> string
TABLES = {}       # file name -> latex


def macro(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    M[name] = value


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


def num(x, d=0):
    if d == 0:
        return f"{int(round(x)):,}".replace(",", "{,}")
    return f"{x:,.{d}f}".replace(",", "{,}")


def sg(x, d=1):
    """Signed number for running text, with a typographic minus/plus."""
    return f"${'+' if x >= 0 else '-'}${abs(x):.{d}f}"


def pm(m, s, d=1, scale=100):
    return f"{scale * m:.{d}f} $\\pm$ {scale * s:.{d}f}"


# ------------------------------------------------------------------ audit
def audit():
    a = json.load(open(PROC / "audit_summary.json"))
    p, s = a["paddy"], a["sethy"]
    macro("PaddyImages", num(p["images"]))
    macro("PaddyExactDupFiles", num(p["exact_duplicate_files"]))
    macro("PaddyExactDupGroups", num(p["exact_duplicate_groups"]))
    macro("PaddyNearGroups", num(p["near_duplicate_groups"]))
    macro("PaddyInMultiGroups", num(p["images_in_multi_member_groups"]))
    macro("PaddyInMultiGroupsPct", pct(p["images_in_multi_member_groups"] / p["images"]))
    macro("PaddyConflictGroups", num(p["label_conflict_groups"]))
    macro("PaddyConflictImages", num(p["images_in_label_conflict_groups"]))
    macro("PaddyTrain", num(p["split_counts"]["train"]))
    macro("PaddyVal", num(p["split_counts"]["val"]))
    macro("PaddyHoldout", num(p["split_counts"]["holdout"]))
    macro("PaddyHoldoutGroups", num(p["split_groups"]["holdout"]))
    macro("PaddyTrainGroups", num(p["split_groups"]["train"]))
    macro("SethyImages", num(s["images"]))
    macro("SethyUnique", num(s["unique_files"]))
    macro("SethyExactDupFiles", num(s["exact_duplicate_files"]))
    macro("SethyExactDupPct", pct(s["exact_duplicate_files"] / s["images"]))
    macro("SethyGroups", num(s["near_duplicate_groups"]))
    macro("SethyImagesPerGroup", num(s["images"] / s["near_duplicate_groups"], 1))
    macro("PhashThr", str(a["phash_hamming_threshold"]))
    macro("CrossDatasetDups", num(a["cross_dataset_near_duplicates"]))
    # dataset table
    cls = CFG["classes"]
    rows = []
    cbs = p["class_by_split"]
    for c in cls:
        n_tr, n_va, n_ho = cbs["train"][c], cbs["val"][c], cbs["holdout"][c]
        sname = {v: k for k, v in CFG["sethy_map"].items()}.get(c)
        s_img = s["images_per_class"].get(c, 0)
        s_grp = s["groups_per_class"].get(c, 0)
        hi = "\\checkmark" if c in CFG["risk"]["high_severity"] else ""
        rows.append(f"{c.replace('_', ' ')} & {hi} & {num(n_tr)} & {num(n_va)} & {num(n_ho)} & "
                    f"{num(s_img) if s_img else '--'} & {num(s_grp) if s_grp else '--'} \\\\")
    TABLES["tab_datasets.tex"] = "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Class composition of the group-aware Paddy Doctor partitions and of the Sethy cross-source test set. "
        "$\\mathcal{H}$ marks the default high-severity classes. Sethy groups are near-duplicate groups "
        f"(dihedral pHash distance $\\le{a['phash_hamming_threshold']}$ bits).}}",
        "\\label{tab:datasets}", "\\small",
        "\\begin{tabular}{lcrrrrr}", "\\toprule",
        " & & \\multicolumn{3}{c}{Paddy Doctor (images)} & \\multicolumn{2}{c}{Sethy} \\\\",
        "\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}",
        "Class & $\\mathcal{H}$ & Train & Val. & Hold-out & Images & Groups \\\\", "\\midrule",
        *rows, "\\midrule",
        f"Total & & {num(p['split_counts']['train'])} & {num(p['split_counts']['val'])} & {num(p['split_counts']['holdout'])} & "
        f"{num(s['images'])} & {num(s['near_duplicate_groups'])} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def versions():
    import sklearn
    import torch
    import timm
    macro("PythonVersion", platform.python_version())
    macro("TorchVersion", torch.__version__.split("+")[0])
    macro("TimmVersion", timm.__version__)
    macro("SklearnVersion", sklearn.__version__)


KEY = [("Paddy Doctor images / near-duplicate groups", "PaddyImages", "PaddyNearGroups"),
       ("Sethy images / near-duplicate groups", "SethyImages", "SethyGroups"),
       ("Accuracy inflation by image-level split (pp, 1-NN / LR)", "LeakKnnGap", "LeakLrGap"),
       ("Edge accuracy / macro-F1 (%)", "EdgeTsAcc", "EdgeTsFone"),
       ("Cloud ensemble accuracy / macro-F1 (%)", "EnsTsAcc", "EnsTsFone"),
       ("URGENT recall / false alarms, alpha=0.1, class-cond. LAC (%)", "EdgeLcUrg", "EdgeLcFa"),
       ("Images kept on device (ROUTINE, %)", "EdgeLcRout", None),
       ("Coverage SD group-aware vs i.i.d. (pp)", "CovVarGroupSd", "CovVarIidSd"),
       ("Sethy: ensemble accuracy / coverage (%)", "ShiftEnsAcc", "ShiftEnsCov"),
       ("Poisoning rho=0.4: top-1 H recall / URGENT recall trusted cal. (%)", "PoisConFortyNoneHrec", "PoisConFortyNoneUrgT"),
       ("Outbreak 2.5 Mbit/s: URGENT violations RADS / EDF-LL (%)", "SchLowRadsUViol", "SchLowEdfUViol"),
       ("Outbreak 2.5 Mbit/s: URGENT violations GA / PSO (%)", "SchLowGaUViol", "SchLowPsoUViol"),
       ("Device traffic saved by tiered retrieval (2.5 Mbit/s)", "UaLowMBSave", None)]


def write_summary():
    def clean(v):
        return re.sub(r"\\[a-z]+|[{}$\\]", "", v.replace("$\\pm$", "±").replace("\\%", "%")).strip()
    lines = ["# Key results (auto-generated by scripts/make_tables.py)", "", "| Quantity | Value |", "|---|---|"]
    for lab, a, b in KEY:
        if a in M and (b is None or b in M):
            lines.append(f"| {lab} | {clean(M[a])}" + (f" / {clean(M[b])}" if b else "") + " |")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "key_results.md").write_text("\n".join(lines) + "\n")


def write():
    lines = ["% AUTO-GENERATED by scripts/make_tables.py -- do not edit"]
    for k in sorted(M):
        lines.append(f"\\newcommand{{\\{k}}}{{{M[k]}}}")
    (GEN / "numbers.tex").write_text("\n".join(lines) + "\n")
    for f, t in TABLES.items():
        (GEN / f).write_text("% AUTO-GENERATED by scripts/make_tables.py\n" + t + "\n")
    print(f"{len(M)} macros, {len(TABLES)} tables")


def leakage():
    d = pd.read_csv(RAW / "eval_e0_leakage.csv")
    g = d.groupby(["classifier", "protocol"])
    m, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
    for clf, tag in (("1-NN", "Knn"), ("logistic regression", "Lr")):
        for proto, ptag in (("random image-level", "Rand"), ("group-aware", "Group")):
            macro(f"Leak{tag}{ptag}Acc", pm(m.loc[(clf, proto), "accuracy"], sd.loc[(clf, proto), "accuracy"]))
            macro(f"Leak{tag}{ptag}Fone", pm(m.loc[(clf, proto), "macro_f1"], sd.loc[(clf, proto), "macro_f1"]))
        diff = m.loc[(clf, "random image-level"), "accuracy"] - m.loc[(clf, "group-aware"), "accuracy"]
        macro(f"Leak{tag}Gap", f"{100 * diff:.1f}")
    macro("LeakShareNearDup", pct(m.loc[("1-NN", "random image-level"), "test_with_neardup_in_train"]))
    rows = []
    for clf in ("1-NN", "logistic regression"):
        for proto in ("random image-level", "group-aware"):
            rows.append(f"{clf} & {proto} & {pm(m.loc[(clf, proto), 'accuracy'], sd.loc[(clf, proto), 'accuracy'])} & "
                        f"{pm(m.loc[(clf, proto), 'macro_f1'], sd.loc[(clf, proto), 'macro_f1'])} & "
                        f"{pct(m.loc[(clf, proto), 'test_with_neardup_in_train'])} \\\\")
    TABLES["tab_leakage.tex"] = "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Effect of near-duplicate leakage on image-level evaluation (frozen ImageNet features, 80/20 splits of "
        "the full Paddy Doctor set, mean $\\pm$ SD over 10 repetitions, \\%). The last column is the share of test images "
        "whose near-duplicate group also occurs in the training split.}",
        "\\label{tab:leakage}", "\\small",
        "\\begin{tabular}{llccc}", "\\toprule",
        "Classifier & Split protocol & Accuracy & Macro-F1 & Test images with near-dup.\\ in train \\\\", "\\midrule",
        *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def latency():
    L = json.load(open(RAW / "latency_benchmark.json"))
    macro("LatEdgeMean", f"{L['edge_forward_ms']['mean']:.1f}")
    macro("LatEdgePnf", f"{L['edge_forward_ms']['p95']:.1f}")
    macro("LatEnsMean", f"{L['ensemble_forward_ms']['mean']:.1f}")
    macro("LatEnsPnf", f"{L['ensemble_forward_ms']['p95']:.1f}")
    macro("LatCamMean", f"{L['gradcam_ms']['mean']:.1f}")
    macro("LatTierUs", f"{L['tiering_us']:.1f}")
    macro("EdgeParamsM", f"{L['edge_model_params_M']:.2f}")
    macro("EdgeFileMB", f"{L['edge_model_file_MB']:.1f}")
    macro("ImgBytesMeanKB", f"{L['original_image_bytes']['mean'] / 1000:.1f}")
    h = L["hmac"]
    big = [k for k in h if k.isdigit() and int(k) > 20000][0]
    macro("HmacSignImgUs", f"{h[big]['sign_us']:.0f}")
    macro("HmacVerifyImgUs", f"{h[big]['verify_us']:.0f}")
    macro("HmacSignRecUs", f"{h['600']['sign_us']:.1f}")
    macro("HmacVerifyRecUs", f"{h['600']['verify_us']:.1f}")
    macro("HmacImgKB", f"{int(big) / 1000:.1f}")
    macro("HmacSignAlertUs", f"{h['16000']['sign_us']:.1f}")
    macro("HmacUrgentPct", f"{100 * (h[big]['sign_us'] + h['16000']['sign_us']) / 1000 / L['edge_forward_ms']['mean']:.2f}")


def params_tables():
    sc = CFG["scheduling"]
    P = sc["params"]
    L = json.load(open(RAW / "latency_benchmark.json")) if (RAW / "latency_benchmark.json").exists() else None
    rows = [
        ("Fog nodes $K_f$ / devices per fog", f"{P['n_fog']} / {P['devices_per_fog']}"),
        ("Fog servers $c_f$, speed $\\mu_f$", f"{P['fog_servers']}, {P['fog_speed']}"),
        ("Cloud servers $c_c$, speed $\\mu_c$", f"{P['cloud_servers']}, {P['cloud_speed']}"),
        ("Access link $B_a$, $L_a$", f"{P['B_access']} Mbit\\,s$^{{-1}}$, {int(P['L_access']*1000)} ms"),
        ("Fog--fog backhaul $B_b$, $L_b$", f"{P['B_backhaul']} Mbit\\,s$^{{-1}}$, {int(P['L_backhaul']*1000)} ms"),
        ("Fog--cloud uplink $B_w$, $L_w$", f"{', '.join(str(b) for b in sc['outbreak_bw_mbps'])} Mbit\\,s$^{{-1}}$, {int(P['L_wan']*1000)} ms"),
        ("Deadlines URGENT / STRATEGIC", f"{sc['deadlines']['URGENT']:.0f} s / {sc['deadlines']['STRATEGIC']:.0f} s"),
        ("Alert / routine record size", f"{P['alert_bytes']/1000:.0f} kB / {P['record_bytes']} B"),
        ("Device radio power $P_{\\mathrm{tx}}$", f"{P['P_tx']} W"),
        ("Busy power per fog / cloud server", f"{P['P_fog']} W / {P['P_cloud']} W"),
        ("Cloud price $\\pi_c$", f"\\${P['cloud_price_per_server_hour']} per server-hour"),
        ("Urgent reservation $\\rho_{\\mathrm{res}}$", f"{P['fog_reserved_for_urgent']} server per fog"),
        ("Batch window (Min--Min, Max--Min, GA, PSO)", f"{P['batch_window']} s"),
        ("Arrival rate, outbreak", f"{sc['base_rate']} $\\to$ {sc['base_rate']*sc['surge']['factor']} s$^{{-1}}$ ({sc['surge']['start']}--{sc['surge']['start']+sc['surge']['length']} s)"),
        ("Horizon / replications", f"{sc['duration_s']} s / {sc['reps']} (outbreak), {sc['load_reps']} (load), {sc['sensitivity']['reps']} (sensitivity)"),
    ]
    if L:
        rows += [("Edge inference $e_i$ (measured, mean)", f"{L['edge_forward_ms']['mean']:.1f} ms"),
                 ("Cloud-ensemble work $w_i$ (measured, mean)", f"{L['ensemble_forward_ms']['mean']:.1f} ms")]
    body = [f"{a} & {b} \\\\" for a, b in rows]
    TABLES["tab_simparams.tex"] = "\n".join([
        "\\begin{table}[t]", "\\centering",
        "\\caption{Simulation parameters. Network, power and price values are assumptions describing a plausible rural "
        "deployment (varied in the sensitivity analysis); inference times were measured on the reference CPU.}",
        "\\label{tab:simparams}", "\\small", "\\begin{tabular}{ll}", "\\toprule", "Parameter & Value \\\\", "\\midrule",
        *body, "\\bottomrule", "\\end{tabular}", "\\end{table}"])
    t = CFG["train"]
    rows = [("Architecture", "MobileNetV3-Large (timm \\texttt{mobilenetv3\\_large\\_100.ra\\_in1k}), ImageNet initialization"),
            ("Input", f"training: random resized crop {CFG['image']['input_size']}$\\times${CFG['image']['input_size']} (scale {t['rrc_scale'][0]}--{t['rrc_scale'][1]}); inference: whole image, shorter side {CFG['image']['input_size']}"),
            ("Augmentation", f"horizontal/vertical flips, brightness and contrast $\\pm${int(t['color_jitter']*100)}\\%"),
            ("Optimizer", f"AdamW, learning rate {t['lr']}, weight decay {t['weight_decay']}, batch {t['batch_size']}"),
            ("Schedule", f"{t['epochs']} epochs, {t['warmup_epochs']} warm-up epoch, cosine decay; best validation macro-F1 retained"),
            ("Regularization", f"dropout {t['drop_rate']}"),
            ("Seeds", ", ".join(str(x) for x in t['seeds']) + " (cloud model: ensemble of all five)"),
            ("Calibration", "temperature scaling on the validation partition (per seed and for the ensemble)"),
            ("Conformal prediction", f"LAC (marginal, class-conditional) and APS; $\\alpha\\in\\{{{', '.join(str(a) for a in CFG['risk']['alphas'])}\\}}$; {CFG['conformal']['n_splits']} group-aware calibration/test splits"),
            ("Hardware", f"2 vCPU (x86-64), 7 GB RAM, no GPU; {t['threads']} PyTorch threads")]
    body = [f"{a} & {b} \\\\" for a, b in rows]
    TABLES["tab_hparams.tex"] = "\n".join([
        "\\begin{table}[t]", "\\centering", "\\caption{Model training and evaluation settings.}", "\\label{tab:hparams}",
        "\\small", "\\begin{tabular}{lp{0.68\\textwidth}}", "\\toprule", "Setting & Value \\\\", "\\midrule",
        *body, "\\bottomrule", "\\end{tabular}", "\\end{table}"])


def xai():
    from scipy.stats import wilcoxon
    ev = pd.read_csv(RAW / "xai_gradcam_eval.csv")
    macro("XaiN", num(len(ev)))
    macro("XaiAccShown", pct(ev.correct.mean()))
    for kind, K in (("deletion", "Del"), ("insertion", "Ins")):
        for m, M in (("gradcam", "Cam"), ("random", "Rand"), ("centre", "Cent")):
            x = ev[f"{kind}_auc_{m}"]
            macro(f"Xai{K}{M}Med", f"{x.median():.3f}")
            macro(f"Xai{K}{M}Iqr", f"{x.quantile(.25):.3f}--{x.quantile(.75):.3f}")
        for m, M in (("random", "Rand"), ("centre", "Cent")):
            p = wilcoxon(ev[f"{kind}_auc_gradcam"], ev[f"{kind}_auc_{m}"]).pvalue
            macro(f"Xai{K}P{M}", f"{p:.1e}".replace("e-0", "\\times10^{-").replace("e-", "\\times10^{-") + "}" if p < 1e-3 else f"{p:.3f}")
            better = (ev[f"{kind}_auc_gradcam"] < ev[f"{kind}_auc_{m}"]) if kind == "deletion" else (ev[f"{kind}_auc_gradcam"] > ev[f"{kind}_auc_{m}"])
            macro(f"Xai{K}Win{M}", pct(better.mean()))
    for t, T in (("flip", "Flip"), ("brightness", "Bright")):
        macro(f"XaiStab{T}Rho", f"{ev[f'stability_spearman_{t}'].median():.2f}")
        macro(f"XaiStab{T}Iou", f"{ev[f'stability_iou20_{t}'].median():.2f}")


def _tab(name, caption, label, colspec, header, rows, size="\\small", note=None, colsep=None):
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size]
    if colsep:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{colsep}pt}}")
    lines += [
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule"]
    lines += [r + " \\\\" if not r.startswith("\\midrule") else r for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    if note:
        lines += [f"\\par\\smallskip\\footnotesize {note}"]
    lines += ["\\end{table}"]
    TABLES[name] = "\n".join(lines)


def e1():
    e = json.load(open(RAW / "eval_e1_accuracy_calibration.json"))
    a = e["aggregate"]
    macro("NSeeds", str(len(e["seeds"])))
    for key, tag in (("edge_raw", "EdgeRaw"), ("edge_ts", "EdgeTs")):
        for m, M in (("accuracy", "Acc"), ("macro_f1", "Fone"), ("ece", "Ece"), ("nll", "Nll"), ("brier", "Brier"),
                     ("balanced_accuracy", "Bacc"), ("classwise_ece", "Cece")):
            mean, sd = a[key][m]
            d = 1 if m in ("accuracy", "macro_f1", "balanced_accuracy") else 3
            sc = 100 if d == 1 else 1
            macro(f"{tag}{M}", f"{sc * mean:.{d}f} $\\pm$ {sc * sd:.{d}f}")
            macro(f"{tag}{M}Mean", f"{sc * mean:.{d}f}")
    for key, tag in (("ensemble_raw", "EnsRaw"), ("ensemble_ts", "EnsTs")):
        for m, M in (("accuracy", "Acc"), ("macro_f1", "Fone"), ("ece", "Ece"), ("nll", "Nll"), ("brier", "Brier"),
                     ("balanced_accuracy", "Bacc"), ("classwise_ece", "Cece")):
            v = a[key][m]
            macro(f"{tag}{M}", f"{100 * v:.1f}" if m in ("accuracy", "macro_f1", "balanced_accuracy") else f"{v:.3f}")
    ci = e["cluster_ci"]["ensemble_ts"]
    macro("EnsAccCi", f"{100 * ci['accuracy'][1]:.1f}--{100 * ci['accuracy'][2]:.1f}")
    macro("EnsFoneCi", f"{100 * ci['macro_f1'][1]:.1f}--{100 * ci['macro_f1'][2]:.1f}")
    macro("TempMean", f"{np.mean(e['temperatures']):.2f}")
    macro("TempRange", f"{min(e['temperatures']):.2f}--{max(e['temperatures']):.2f}")
    # validation accuracy of each seed (argmax is unaffected by temperature)
    idx = pd.read_csv(PROC / "paddy_index.csv")
    y_all = idx.label.map({c: i for i, c in enumerate(CFG["classes"])}).to_numpy()
    va = []
    for sd_ in e["seeds"]:
        d = RAW / "models" / f"seed{sd_}"
        va.append(float((np.load(d / "logits_val.npy").argmax(1) == y_all[np.load(d / "index_val.npy")]).mean()))
    macro("ValAcc", pm(np.mean(va), np.std(va, ddof=1)))
    macro("ValAccMean", f"{100 * np.mean(va):.1f}")
    macro("NVal", num(e["n_val"]))
    spread = []
    for sd_ in e["seeds"]:
        h = pd.read_csv(RAW / "models" / f"seed{sd_}" / "history.csv")
        last = h.val_macro_f1.to_numpy()[-3:]
        spread.append(last.max() - last.min())
    macro("TrainLastSpread", f"{100 * max(spread):.1f}")
    macro("TempEns", f"{e['temperature_ensemble']:.2f}")
    rows = []
    for name, key, kind in (("Edge, raw", "edge_raw", "seed"), ("Edge, temperature-scaled", "edge_ts", "seed"),
                            ("Cloud ensemble, raw", "ensemble_raw", "ens"), ("Cloud ensemble, temperature-scaled", "ensemble_ts", "ens")):
        cells = []
        for m in ("accuracy", "macro_f1", "balanced_accuracy", "nll", "ece", "classwise_ece", "brier"):
            pctm = m in ("accuracy", "macro_f1", "balanced_accuracy")
            if kind == "seed":
                mean, sd = a[key][m]
                cells.append(f"{100 * mean:.1f} $\\pm$ {100 * sd:.1f}" if pctm else f"{mean:.3f} $\\pm$ {sd:.3f}")
            else:
                v = a[key][m]
                cells.append(f"{100 * v:.1f}" if pctm else f"{v:.3f}")
        rows.append(name + " & " + " & ".join(cells))
    _tab("tab_e1.tex", f"Classification and calibration on the Paddy Doctor hold-out partition ({e['n_holdout']:,} images in "
         f"{e['n_holdout_groups']:,} near-duplicate groups). Edge: mean $\\pm$ SD over {len(e['seeds'])} seeds; ensemble: single value "
         "(95\\% cluster-bootstrap intervals in the text). Accuracy, macro-F1 and balanced accuracy in \\%.",
         "tab:e1", "lccccccc", "Model & Acc. & Macro-F1 & Bal. acc. & NLL & ECE & cw-ECE & Brier", rows, size="\\footnotesize")


def e2():
    cp = pd.read_csv(RAW / "eval_e2_conformal_splits.csv")
    a0 = CFG["risk"]["alpha"]
    d = cp[cp.alpha == a0]
    g = d.groupby(["model", "method"])
    m, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
    macro("NSplits", str(int(cp.split.nunique())))
    rows = []
    for model in ("edge", "ensemble"):
        for meth in ("LAC-marginal", "LAC-classwise", "APS-marginal"):
            r = m.loc[(model, meth)]
            rows.append(f"{'Edge' if model == 'edge' else 'Ensemble'} & {meth} & {100 * r.coverage:.1f} & {100 * r.min_class_coverage:.1f} & "
                        f"{r.avg_set_size:.2f} & {100 * r.singleton_rate:.1f} & {100 * r.urgent_recall:.1f} & {100 * r.urgent_false_alarm:.1f} & "
                        f"{100 * r.rate_routine:.1f}")
            tag = ("Edge" if model == "edge" else "Ens") + {"LAC-marginal": "Lm", "LAC-classwise": "Lc", "APS-marginal": "Ap"}[meth]
            for col, M in (("coverage", "Cov"), ("min_class_coverage", "MinCov"), ("urgent_recall", "Urg"), ("urgent_false_alarm", "Fa"),
                           ("rate_routine", "Rout"), ("rate_urgent", "RateU"), ("rate_strategic", "RateS"), ("singleton_rate", "Single"),
                           ("high_sev_min_class_coverage", "HMinCov"), ("urgent_recall_broadH", "UrgB"), ("empty_rate", "Empty")):
                macro(f"{tag}{M}", f"{100 * r[col]:.1f}")
            macro(f"{tag}Size", f"{r.avg_set_size:.2f}")
            macro(f"{tag}CovSd", f"{100 * sd.loc[(model, meth)].coverage:.1f}")
    _tab("tab_e2.tex", f"Conformal prediction on the hold-out partition at $\\alpha={a0}$ (mean over {cp.split.nunique()} group-aware "
         "calibration/test splits). Coverage, minimum class-conditional coverage, singleton rate, URGENT recall "
         "(high-severity images flagged URGENT), false alarms (other images flagged URGENT) and ROUTINE rate in \\%.",
         "tab:e2", "llccccccc", "Model & Method & Cov. & Min cls. cov. & Size & Single. & URG recall & False alarm & ROUTINE", rows,
         size="\\footnotesize")
    # deviation of mean coverage from target over all alphas and methods (edge and ensemble)
    g2 = cp.groupby(["model", "method", "alpha"]).coverage.mean().reset_index()
    dev = (g2.coverage - (1 - g2.alpha))
    macro("CovDevMax", f"{100 * dev.abs().max():.1f}")
    macro("CovDevMin", f"{100 * dev.min():+.1f}")
    # mean per-class coverage (over splits) -> minimum over classes, by method (edge)
    for meth, T in (("LAC-marginal", "Lm"), ("LAC-classwise", "Lc"), ("APS-marginal", "Ap")):
        dd_ = d[(d.method == meth) & (d.model == "edge")]
        pcm = {c: dd_[f"cov_{c}"].mean() for c in CFG["classes"]}
        worst = min(pcm, key=pcm.get)
        macro(f"Edge{T}ClsCovMin", f"{100 * pcm[worst]:.1f}")
        macro(f"Edge{T}ClsCovMinName", worst.replace("_", " "))
        macro(f"Edge{T}ClsCovMax", f"{100 * max(pcm.values()):.1f}")
        if T == "Lc":
            idx = pd.read_csv(PROC / "paddy_index.csv")
            h = idx[(idx.split == "holdout") & (idx.label == worst)]
            macro("LcWorstImgs", num(len(h)))
            macro("LcWorstGroups", num(h.group.nunique()))
            hh = idx[idx.split == "holdout"]
            macro("HoldoutImgsPerGroup", f"{len(hh) / hh.group.nunique():.1f}")
            macro("LcWorstImgsPerGroup", f"{len(h) / h.group.nunique():.1f}")
            hcov = [dd_[f"cov_{c}"].mean() for c in CFG["risk"]["high_severity"]]
            macro("EdgeLcHClsCovMin", f"{100 * min(hcov):.1f}")
            macro("EdgeLcHClsCovMax", f"{100 * max(hcov):.1f}")
    # per-class coverage (supplementary)
    rows = []
    for c in CFG["classes"]:
        cells = []
        for meth in ("LAC-marginal", "LAC-classwise"):
            for mo in ("edge", "ensemble"):
                dd = d[(d.method == meth) & (d.model == mo)]
                cells.append(f"{100 * dd[f'cov_{c}'].mean():.1f}")
        rows.append(f"{c.replace('_', ' ')} & " + " & ".join(cells))
    _tab("tab_s_classcov.tex", f"Per-class coverage (\\%) of marginal and class-conditional LAC sets at $\\alpha={a0}$ (mean over "
         f"{cp.split.nunique()} group-aware calibration/test splits).", "tab:s_classcov", "lcccc",
         "& \\multicolumn{2}{c}{Marginal LAC} & \\multicolumn{2}{c}{Class-conditional LAC} \\\\ \\cmidrule(lr){2-3}\\cmidrule(lr){4-5} "
         "Class & Edge & Ensemble & Edge & Ensemble", rows)


def e3():
    cas = pd.read_csv(RAW / "eval_e3_cascade_splits.csv")
    g = cas.groupby("gate")
    m, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
    order = ["edge-only", "random", "max-softmax", "entropy", "risk-mass", "conformal-tier (proposed)", "cloud-only"]
    rows = []
    for o in order:
        r, s_ = m.loc[o], sd.loc[o]
        miss = 1 - r.high_sev_recall_or_flag
        rows.append(f"{o} & {100 * r.escalation_rate:.1f} & {r.compute_units_per_image:.2f} & {100 * r.accuracy:.1f} $\\pm$ {100 * s_.accuracy:.1f} & "
                    f"{100 * r.macro_f1:.1f} $\\pm$ {100 * s_.macro_f1:.1f} & {100 * r.high_sev_recall:.1f} $\\pm$ {100 * s_.high_sev_recall:.1f} & "
                    f"{100 * miss:.1f}")
        tag = {"edge-only": "Edge", "random": "Rand", "max-softmax": "Msp", "entropy": "Ent", "risk-mass": "Risk",
               "conformal-tier (proposed)": "Conf", "cloud-only": "Cloud"}[o]
        for col, M in (("escalation_rate", "Esc"), ("accuracy", "Acc"), ("macro_f1", "Fone"), ("high_sev_recall", "Hrec")):
            macro(f"Cas{tag}{M}", f"{100 * r[col]:.1f}")
        macro(f"Cas{tag}Miss", f"{100 * (1 - r.high_sev_recall_or_flag):.1f}")
        macro(f"Cas{tag}Units", f"{r.compute_units_per_image:.2f}")
    # paired comparison proposed vs others on macro-F1 and H recall
    from src.evaluation.stats import corrected_resampled_t
    pr = cas[cas.gate == "conformal-tier (proposed)"].sort_values("split")
    for o, tag in (("max-softmax", "Msp"), ("entropy", "Ent"), ("risk-mass", "Risk"), ("random", "Rand"),
                   ("edge-only", "Edge"), ("cloud-only", "Cloud")):
        ot = cas[cas.gate == o].sort_values("split")
        for col, M in (("macro_f1", "Fone"), ("high_sev_recall", "Hrec"), ("accuracy", "Acc"), ("high_sev_recall_or_flag", "Miss")):
            diff = pr[col].to_numpy() - ot[col].to_numpy()
            if M == "Miss":
                diff = -diff                       # missed share = 1 - recall_or_flag; report proposed minus other
            # calibration and test halves of the hold-out partition have equal size -> n_test / n_train = 1
            mu, tt, p = corrected_resampled_t(diff, 1.0, 1.0)
            macro(f"CasDiff{tag}{M}", sg(100 * mu))
            macro(f"CasP{tag}{M}", _p(p))
            macro(f"CasWin{tag}{M}", pct(((diff < 0) if M == "Miss" else (diff > 0)).mean(), 0))
            macro(f"CasTie{tag}{M}", pct((diff == 0).mean(), 0))
    macro("CasSavePct", pct(1 - m.loc["conformal-tier (proposed)"].compute_units_per_image / 5))
    _tab("tab_e3.tex", f"Uncertainty-gated cascade (mean $\\pm$ SD over {cas.split.nunique()} splits). Every gate except edge-only "
         "and cloud-only escalates the same fraction of images as the conformal tier rule on each split. Compute is "
         "measured in edge-model forward passes per image. High-sev.\\ recall: high-severity test images whose final "
         "prediction is a high-severity class; missed: high-severity images that are neither predicted as such nor flagged "
         "URGENT (only the tier rule raises URGENT flags). Values in \\%.",
         "tab:e3", "lcccccc", "Gate & Escalated & Compute & Accuracy & Macro-F1 & High-sev.\\ recall & Missed", rows, size="\\footnotesize")


def e3b():
    t = pd.read_csv(RAW / "eval_e3b_tier_diagnostics.csv")
    m = t.mean(numeric_only=True)
    macro("TierEscUrgShare", pct(m.share_escalated_urgent))
    macro("TierAccUrg", pct(m.edge_acc_urgent))
    macro("TierAccStr", pct(m.edge_acc_strategic))
    macro("TierAccRout", pct(m.edge_acc_routine))
    macro("TierUrgSingle", pct(m.urgent_singleton_share))


def _p(p):
    if p < 1e-4:
        e = int(np.floor(np.log10(p)))
        return f"{p / 10 ** e:.1f}\\times10^{{{e}}}"
    return f"{p:.4f}" if p < 0.01 else f"{p:.3f}"


def e4():
    e = json.load(open(RAW / "eval_e4_shift.json"))
    rc = pd.read_csv(RAW / "eval_e4_recalibration.csv")
    E = e["ensemble"]
    for k, M in (("accuracy", "Acc"), ("coverage_paddy_calibrated", "Cov"), ("urgent_recall_H", "Urg")):
        macro(f"ShiftEns{M}", f"{100 * E[k][0]:.1f}")
        macro(f"ShiftEns{M}Ci", f"{100 * E[k][1]:.1f}--{100 * E[k][2]:.1f}")
    macro("ShiftEnsSize", f"{E['avg_set_size']:.2f}")
    macro("ShiftEnsGroupAcc", f"{100 * E['group_mean_accuracy']:.1f}")
    macro("ShiftEnsHrec", f"{100 * E['high_sev_recall']:.1f}")
    edges = [v for k, v in e.items() if k.startswith("edge_seed")]
    macro("ShiftEdgeAcc", pm(np.mean([v["accuracy"][0] for v in edges]), np.std([v["accuracy"][0] for v in edges], ddof=1)))
    macro("ShiftEdgeCov", pm(np.mean([v["coverage_paddy_calibrated"][0] for v in edges]), np.std([v["coverage_paddy_calibrated"][0] for v in edges], ddof=1)))
    macro("ShiftEdgeUrg", pm(np.mean([v["urgent_recall_H"][0] for v in edges]), np.std([v["urgent_recall_H"][0] for v in edges], ddof=1)))
    for c in ("bacterial_leaf_blight", "blast", "brown_spot", "tungro"):
        tag = {"bacterial_leaf_blight": "Blb", "blast": "Blast", "brown_spot": "Bs", "tungro": "Tun"}[c]
        macro(f"ShiftRec{tag}", f"{100 * E['per_class_recall'][c]:.1f}")
        macro(f"ShiftCov{tag}", f"{100 * E['per_class_coverage'][c]:.1f}")
    pdist = sorted(E["pred_distribution"].items(), key=lambda kv: -kv[1])
    macro("ShiftTopPred", pdist[0][0].replace("_", " "))
    macro("ShiftTopPredPct", f"{100 * pdist[0][1]:.1f}")
    macro("ShiftSecondPred", pdist[1][0].replace("_", " "))
    macro("ShiftSecondPredPct", f"{100 * pdist[1][1]:.1f}")
    for t, T in (("URGENT", "U"), ("STRATEGIC", "S"), ("ROUTINE", "R")):
        macro(f"ShiftTier{T}", f"{100 * E['tier_rates'][t]:.1f}")
        er = [v["tier_rates"][t] for k, v in e.items() if k.startswith("edge_seed")]
        macro(f"ShiftEdgeTier{T}", f"{100 * np.mean(er):.1f}")
        macro(f"ShiftEdgeTier{T}Range", f"{100 * min(er):.1f}--{100 * max(er):.1f}")
    er = [1 - v["tier_rates"]["ROUTINE"] for k, v in e.items() if k.startswith("edge_seed")]
    macro("ShiftEdgeEscRange", f"{100 * min(er):.0f}--{100 * max(er):.0f}")
    g = rc.groupby(["model", "mode", "m_groups"])
    for (model, mode, mg), r in g:
        tag = ("Ens" if model == "ensemble" else "Edge") + ("C" if mode == "class" else "M") + {5: "Five", 10: "Ten", 20: "Twenty", 40: "Forty", 80: "Eighty"}[mg]
        macro(f"Rc{tag}Cov", f"{100 * r.coverage.mean():.1f}")
        macro(f"Rc{tag}CovLo", f"{100 * r.coverage.quantile(0.05):.1f}")
        macro(f"Rc{tag}Urg", f"{100 * r.urgent_recall_H.mean():.1f}")
        macro(f"Rc{tag}Size", f"{r.avg_set_size.mean():.2f}")
        macro(f"Rc{tag}Img", num(r.n_cal_images.mean()))
    rows = []
    for name, v in [("Cloud ensemble", E)] + [(f"Edge seed {k.replace('edge_seed', '')}", v) for k, v in e.items() if k.startswith("edge_seed")]:
        rows.append(f"{name} & {100 * v['accuracy'][0]:.1f} ({100 * v['accuracy'][1]:.1f}--{100 * v['accuracy'][2]:.1f}) & "
                    f"{100 * v['coverage_paddy_calibrated'][0]:.1f} ({100 * v['coverage_paddy_calibrated'][1]:.1f}--{100 * v['coverage_paddy_calibrated'][2]:.1f}) & "
                    f"{v['avg_set_size']:.2f} & {100 * v['urgent_recall_H'][0]:.1f} ({100 * v['urgent_recall_H'][1]:.1f}--{100 * v['urgent_recall_H'][2]:.1f})")
    _tab("tab_e4.tex", f"Cross-source evaluation on the Sethy dataset with thresholds calibrated on Paddy Doctor (class-conditional LAC, "
         f"$\\alpha={CFG['risk']['alpha']}$). 95\\% cluster-bootstrap intervals over near-duplicate groups in parentheses; URGENT recall "
         "is computed on the high-severity classes (bacterial leaf blight, blast, tungro). Values in \\%.",
         "tab:e4", "lcccc", "Model & Accuracy & Coverage & Set size & URGENT recall", rows, size="\\footnotesize")


def e7():
    v = pd.read_csv(RAW / "eval_e7_variety.csv").sort_values("n_holdout", ascending=False)
    rows = [f"{r.variety} & {int(r.n_holdout):,} & {int(r.n_groups):,} & {100 * r.ensemble_accuracy:.1f} & {100 * r.edge_accuracy_mean:.1f} & "
            f"{100 * r.marginal_coverage_mean:.1f} & {100 * r.marginal_coverage_p05:.1f}" for r in v.itertuples()]
    big = v[v.n_holdout >= 50]
    macro("VarCovMin", f"{100 * big.marginal_coverage_mean.min():.1f}")
    macro("VarCovMax", f"{100 * big.marginal_coverage_mean.max():.1f}")
    macro("VarCovMinName", big.loc[big.marginal_coverage_mean.idxmin(), "variety"])
    macro("VarAccMin", f"{100 * big.ensemble_accuracy.min():.1f}")
    macro("VarAccMax", f"{100 * big.ensemble_accuracy.max():.1f}")
    worst = big.loc[big.marginal_coverage_mean.idxmin(), "variety"]
    idx = pd.read_csv(PROC / "paddy_index.csv")
    w = idx[idx.variety == worst]
    ct = pd.crosstab(w.label, w.split)
    unseen = [c for c in ct.index if ct.loc[c].get("holdout", 0) > 0 and ct.loc[c].get("train", 0) == 0]
    macro("VarWorstN", num(int((w.split == "holdout").sum())))
    macro("VarWorstUnseenN", num(int(sum(ct.loc[c, "holdout"] for c in unseen))))
    macro("VarWorstUnseenCls", ", ".join(c.replace("_", " ") for c in unseen) if unseen else "none")
    macro("VarWorstConflict", num(int(w[w.split == "holdout"].label_conflict_group.sum())))
    macro("VarWorstTrain", num(int((w.split == "train").sum())))
    macro("VarWorstAcc", f"{100 * big.loc[big.marginal_coverage_mean.idxmin(), 'ensemble_accuracy']:.1f}")
    macro("VarMajor", v.iloc[0].variety)
    macro("VarMajorN", num(int(v.iloc[0].n_holdout)))
    _tab("tab_s_variety.tex", "Accuracy and marginal conformal coverage ($\\alpha=0.1$, cloud ensemble) by rice variety on the hold-out "
         "partition (coverage: mean and 5th percentile over the calibration/test splits). Values in \\%.", "tab:s_variety",
         "lrrcccc", "Variety & Images & Groups & Ens. acc. & Edge acc. & Coverage & Cov. P5", rows)


def e2b():
    v = pd.read_csv(RAW / "eval_e2b_split_variability.csv")
    a = CFG["risk"]["alpha"]
    g = v.groupby(["mode", "seed"]).coverage
    for mode, M in (("group-aware", "Group"), ("image-level", "Image")):
        sds = g.std().loc[mode]
        means = g.mean().loc[mode]
        macro(f"CovVar{M}Mean", f"{100 * means.mean():.1f}")
        macro(f"CovVar{M}Sd", f"{100 * sds.mean():.1f}")
        macro(f"CovVar{M}SdRange", f"{100 * sds.min():.1f}--{100 * sds.max():.1f}")
    # i.i.d. prediction: Beta(n+1-l, l) for the conditional coverage plus binomial noise of the finite test set
    n, nt = v.n_cal.mean(), v.n_test.mean()
    l = np.floor((n + 1) * a)
    A, B = n + 1 - l, l
    var_beta = A * B / ((A + B) ** 2 * (A + B + 1))
    sd_iid = np.sqrt(var_beta + (1 - a) * a / nt)
    macro("CovVarIidSd", f"{100 * sd_iid:.1f}")
    ratio = v[v["mode"] == "group-aware"].groupby("seed").coverage.std().mean() / sd_iid
    macro("CovVarRatio", f"{ratio:.1f}")
    macro("CovVarNeff", f"{n / ratio ** 2:,.0f}".replace(",", "{,}"))
    macro("CovVarNcal", f"{n:,.0f}".replace(",", "{,}"))
    macro("CovVarSplits", str(int(v.split.nunique())))


RHO_TAG = {0.0: "Zero", 0.1: "Ten", 0.2: "Twenty", 0.3: "Thirty", 0.4: "Forty"}
DEF_TAG = {"none": "None", "CL-filter": "Cl", "kNN-filter": "Knn"}


def e5():
    """Poisoning: macros for every attack x rho x defense; main table (concealment) + supplementary (random);
    paired differences across repetitions with t-based 95% intervals."""
    from src.evaluation.stats import corrected_resampled_ci
    a = json.load(open(PROC / "audit_summary.json"))["paddy"]["split_counts"]
    ratio = (a["holdout"] / 2) / a["train"]          # test half of the hold-out vs. training pool (T + U)

    def t_ci(x):
        return corrected_resampled_ci(x, ratio, 1.0)
    d = pd.read_csv(RAW / "poisoning_results.csv")
    macro("PoisReps", str(int(d.rep.nunique())))
    g = d.groupby(["attack", "rho", "defense"])
    m, sd = g.mean(numeric_only=True), g.std(numeric_only=True)
    cols = (("macro_f1", "Fone"), ("high_sev_recall", "Hrec"), ("concealment_rate", "Conc"), ("det_precision", "Dprec"),
            ("det_recall", "Drec"), ("clean_removed", "Crem"), ("urgent_recall_trusted_cal", "UrgT"),
            ("urgent_recall_poisoned_cal", "UrgP"), ("coverage_trusted_cal", "CovT"), ("coverage_poisoned_cal", "CovP"),
            ("set_size_trusted_cal", "SizeT"), ("set_size_poisoned_cal", "SizeP"), ("accuracy", "Acc"))
    for (att, rho, de), r in m.iterrows():
        tag = "Pois" + {"concealment": "Con", "random": "Rnd"}[att] + RHO_TAG[round(rho, 2)] + DEF_TAG[de]
        for col, M in cols:
            if col.startswith("set_size"):
                macro(tag + M, f"{r[col]:.2f}")
            else:
                macro(tag + M, f"{100 * r[col]:.1f}")
        macro(tag + "NPois", num(r.n_poisoned))
        macro(tag + "NFlag", num(r.n_flagged))
    # paired differences (filter minus none) on concealment rate, H recall and macro-F1 at every rho > 0
    for att, A in (("concealment", "Con"), ("random", "Rnd")):
        for rho in sorted(d.rho.unique()):
            if rho == 0:
                continue
            base = d[(d.attack == att) & (d.rho == rho) & (d.defense == "none")].sort_values("rep")
            for de in ("CL-filter", "kNN-filter"):
                x = d[(d.attack == att) & (d.rho == rho) & (d.defense == de)].sort_values("rep")
                for col, M in (("concealment_rate", "Conc"), ("high_sev_recall", "Hrec"), ("macro_f1", "Fone")):
                    mu, lo, hi = t_ci(x[col].to_numpy() - base[col].to_numpy())
                    macro(f"PoisD{A}{RHO_TAG[round(rho, 2)]}{DEF_TAG[de]}{M}", sg(100 * mu))
                    macro(f"PoisD{A}{RHO_TAG[round(rho, 2)]}{DEF_TAG[de]}{M}Ci", f"{sg(100 * lo)} to {sg(100 * hi)}")
            # trusted minus poisoned calibration (no filter)
            mu, lo, hi = t_ci(base.urgent_recall_trusted_cal.to_numpy() - base.urgent_recall_poisoned_cal.to_numpy())
            macro(f"PoisDCal{A}{RHO_TAG[round(rho, 2)]}", sg(100 * mu))
            macro(f"PoisDCal{A}{RHO_TAG[round(rho, 2)]}Ci", f"{sg(100 * lo)} to {sg(100 * hi)}")
    for att, fname, lab, cap in (("concealment", "tab_e5.tex", "tab:e5", "Concealment attack (high-severity uploads relabeled \\emph{normal})"),
                                 ("random", "tab_s_poison_random.tex", "tab:s_poison_random", "Random label-flipping attack")):
        rows = []
        for rho in sorted(d.rho.unique()):
            for de in ("none", "CL-filter", "kNN-filter"):
                r, s_ = m.loc[(att, rho, de)], sd.loc[(att, rho, de)]
                if de == "none":
                    det = "-- & -- & --"
                elif rho == 0:
                    det = f"-- & -- & {100 * r.clean_removed:.1f}"
                else:
                    det = f"{100 * r.det_precision:.1f} & {100 * r.det_recall:.1f} & {100 * r.clean_removed:.1f}"
                rows.append(f"{rho:.1f} & {de} & {100 * r.macro_f1:.1f} & {100 * r.high_sev_recall:.1f} & "
                            f"{100 * r.concealment_rate:.1f} $\\pm$ {100 * s_.concealment_rate:.1f} & {det} & "
                            f"{100 * r.urgent_recall_trusted_cal:.1f} & {100 * r.urgent_recall_poisoned_cal:.1f}")
            rows.append("\\midrule")
        rows = rows[:-1]
        _tab(fname, f"{cap}: effect on the retrained classifier head (mean over {d.rep.nunique()} repetitions; concealed: mean "
             "$\\pm$ SD; all values in \\%). $\\rho$: fraction of eligible crowd uploads that were poisoned. Concealed: "
             "high-severity test images predicted \\emph{normal}. Filter: precision and recall of the filter for poisoned "
             "uploads and share of clean uploads removed. URGENT recall: share of high-severity test images flagged URGENT by the "
             f"tier rule applied to marginal LAC sets of the retrained head ($\\alpha={CFG['risk']['alpha']}$), calibrated on a trusted "
             "or an equally poisoned calibration set.",
             lab, "llcccccccc", " & & & High-sev. & & \\multicolumn{3}{c}{Filter} & \\multicolumn{2}{c}{URGENT recall} \\\\ "
             "\\cmidrule(lr){6-8}\\cmidrule(lr){9-10} $\\rho$ & Defense & Macro-F1 & recall & Concealed & Prec. & Rec. & "
             "Clean rem. & trusted cal. & poisoned cal.", rows, size="\\scriptsize", colsep=4)


SCHED_ORDER = ["FCFS-Local", "Round-Robin", "Min-Min", "Max-Min", "GA", "PSO", "Cloud-only", "EDF-LL",
               "EDF-LL + priority links", "RADS"]
ABL_ORDER = ["RADS", "RADS w/o tiers", "RADS w/o priority links", "RADS w/o reservation", "RADS + energy/cost terms"]
SCHED_TAG = {"FCFS-Local": "Fcfs", "Round-Robin": "Rr", "Min-Min": "Minmin", "Max-Min": "Maxmin", "GA": "Ga", "PSO": "Pso",
             "EDF-LL": "Edf", "EDF-LL + priority links": "Edfp", "Cloud-only": "Cloud", "RADS": "Rads",
             "RADS w/o tiers": "RadsNoTier", "RADS w/o priority links": "RadsNoLink", "RADS w/o reservation": "RadsNoRes",
             "RADS + energy/cost terms": "RadsCost"}
BW_TAG = {5.0: "Five", 2.5: "Low", 10.0: "Ten"}
S_COLS = (("urgent_violation_rate", "UViol", 100, 1), ("urgent_p95_latency", "UPnf", 1, 2),
          ("urgent_mean_latency", "UMean", 1, 2), ("strategic_violation_rate", "SViol", 100, 1),
          ("deadline_violation_rate", "Viol", 100, 1), ("mean_latency", "Lat", 1, 1),
          ("risk_weighted_latency", "Rwl", 1, 1), ("energy_J_per_job", "Energy", 1, 2),
          ("cost_usd_per_1000_jobs", "Cost", 1, 3), ("objective", "Obj", 1, 3), ("cloud_share", "CloudShare", 100, 1),
          ("bytes_uploaded_MB", "MB", 1, 0), ("scheduler_wall_s", "Wall", 1, 1), ("n_jobs", "Jobs", 1, 0))


def _fmt(x, d):
    return num(x, d) if d == 0 else f"{x:.{d}f}"


def e8():
    ob = pd.read_csv(RAW / "sched_outbreak.csv")
    pool = json.load(open(RAW / "sched_pool_summary.json"))
    macro("PoolN", num(pool["n"]))
    for t, T in (("URGENT", "U"), ("STRATEGIC", "S"), ("ROUTINE", "R")):
        macro(f"Pool{T}", num(pool[t]))
        macro(f"Pool{T}Pct", pct(pool[t] / pool["n"]))
    macro("PoolMeanKB", f"{pool['mean_size'] / 1000:.1f}")
    macro("SchedReps", str(int(ob.rep.nunique())))
    for (bw, wl, sch), r in ob.groupby(["B_wan_mbps", "workload", "scheduler"]):
        tag = "Sch" + BW_TAG[float(bw)] + SCHED_TAG[sch] + ("All" if wl == "upload-all" else "")
        for col, M, sc, dd in S_COLS:
            if col in r:
                macro(tag + M, _fmt(sc * r[col].mean(), dd))
                macro(tag + M + "Sd", _fmt(sc * r[col].std(ddof=1), dd if dd else 1))
    # paired tests of RADS against every non-ablation variant, Holm-adjusted per bandwidth x metric
    from src.evaluation.stats import paired_tests
    tests_rows = []
    for bw in sorted(ob.B_wan_mbps.unique(), reverse=True):
        t = ob[(ob.B_wan_mbps == bw) & (ob.workload == "tiered")]
        prop = t[t.scheduler == "RADS"].sort_values("rep")
        bls = [n for n in SCHED_ORDER if n != "RADS" and n in set(t.scheduler)]
        for metric, M in (("urgent_violation_rate", "UViol"), ("urgent_p95_latency", "UPnf"), ("objective", "Obj"),
                          ("deadline_violation_rate", "Viol"), ("energy_J_per_job", "Energy"),
                          ("cost_usd_per_1000_jobs", "Cost")):
            res = paired_tests(prop[metric].to_numpy(), {b: t[t.scheduler == b].sort_values("rep")[metric].to_numpy() for b in bls})
            for r in res:
                tag = "Sch" + BW_TAG[float(bw)] + SCHED_TAG[r["baseline"]] + M
                macro(tag + "P", _p(r["p_holm"]))
                macro(tag + "Rbc", f"{r['rank_biserial']:+.2f}")
                macro(tag + "Delta", f"{r['cliffs_delta']:+.2f}")
                tests_rows.append({"B_wan_mbps": bw, "metric": metric, **r})
    pd.DataFrame(tests_rows).to_csv(RAW.parent / "tables" / "sched_paired_tests.csv" if (RAW.parent / "tables").exists() else
                                    GEN / "sched_paired_tests.csv", index=False)
    # main table: per bandwidth block
    rows = []
    for bw in sorted(ob.B_wan_mbps.unique(), reverse=True):
        t = ob[(ob.B_wan_mbps == bw) & (ob.workload == "tiered")]
        rows.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{Fog--cloud uplink {bw:g} Mbit\\,s$^{{-1}}$}}}}")
        for n in SCHED_ORDER:
            r = t[t.scheduler == n]
            if r.empty:
                continue
            nm = f"\\textbf{{{n}}}" if n == "RADS" else n
            rows.append(f"{nm} & {100 * r.urgent_violation_rate.mean():.1f} $\\pm$ {100 * r.urgent_violation_rate.std():.1f} & "
                        f"{r.urgent_p95_latency.mean():.1f} & {100 * r.strategic_violation_rate.mean():.1f} & "
                        f"{r.mean_latency.mean():.1f} & {r.energy_J_per_job.mean():.2f} & {r.cost_usd_per_1000_jobs.mean():.3f} & "
                        f"{r.objective.mean():.3f}")
        rows.append("\\midrule")
    rows = rows[:-1]
    _tab("tab_e8.tex", f"Outbreak scenario (base rate {CFG['scheduling']['base_rate']} images\\,s$^{{-1}}$, tripled for "
         f"{CFG['scheduling']['surge']['length']}~s, 70\\% of surge captures from high-severity classes): mean over "
         f"{ob.rep.nunique()} paired replications (URGENT violations: mean $\\pm$ SD). URG/STR viol.: share of URGENT/STRATEGIC jobs "
         "missing their deadline (\\%); p95: 95th-percentile URGENT response time (s); latency: mean response time over all "
         "jobs (s); energy per job (J) and cloud cost per 1000 jobs (US\\$) under the assumed power and price model; $J$: "
         "scalarized objective (Equation~\\ref{eq:objective}).",
         "tab:e8", "lccccccc", "Scheduler & URG viol.\\ (\\%) & URG p95 (s) & STR viol.\\ (\\%) & Latency (s) & Energy (J) & "
         "Cost (\\$) & $J$", rows, size="\\scriptsize", colsep=2.5)
    # paired upload-all workload (separate experiment; alert latency of images the device tiered URGENT)
    ua = pd.read_csv(RAW / "sched_uploadall.csv") if (RAW / "sched_uploadall.csv").exists() else None
    if ua is not None:
        macro("UaReps", str(int(ua.rep.nunique())))
        for (bw, wl, sch), r in ua.groupby(["B_wan_mbps", "workload", "scheduler"]):
            tag = "Ua" + BW_TAG[float(bw)] + SCHED_TAG[sch] + ("All" if wl == "upload-all" else "Tier")
            macro(tag + "AViol", f"{100 * r.alert_violation_rate.mean():.1f}")
            macro(tag + "APnf", f"{r.alert_p95_latency.mean():.1f}")
            macro(tag + "Viol", f"{100 * r.deadline_violation_rate.mean():.1f}")
            macro(tag + "MB", num(r.bytes_uploaded_MB.mean()))
        for bw in sorted(ua.B_wan_mbps.unique()):
            u = ua[ua.B_wan_mbps == bw]
            a_ = u[(u.scheduler == "RADS") & (u.workload == "upload-all")].bytes_uploaded_MB.mean()
            t_ = u[(u.scheduler == "RADS") & (u.workload == "tiered")].bytes_uploaded_MB.mean()
            macro("Ua" + BW_TAG[float(bw)] + "MBSave", pct(1 - t_ / a_))
            # paired test: tiered vs upload-all alert violations for RADS
            from scipy.stats import wilcoxon
            x = u[(u.scheduler == "RADS") & (u.workload == "tiered")].sort_values("rep").alert_violation_rate.to_numpy()
            y = u[(u.scheduler == "RADS") & (u.workload == "upload-all")].sort_values("rep").alert_violation_rate.to_numpy()
            p_ = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            macro("Ua" + BW_TAG[float(bw)] + "RadsP", _p(p_))
    # ablation + upload-all table
    rows = []
    for bw in sorted(ob.B_wan_mbps.unique(), reverse=True):
        t = ob[(ob.B_wan_mbps == bw)]
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{Fog--cloud uplink {bw:g} Mbit\\,s$^{{-1}}$}}}}")
        for n in ABL_ORDER:
            r = t[(t.scheduler == n) & (t.workload == "tiered")]
            if r.empty:
                continue
            rows.append(f"{n} & {100 * r.urgent_violation_rate.mean():.1f} $\\pm$ {100 * r.urgent_violation_rate.std():.1f} & "
                        f"{r.urgent_p95_latency.mean():.1f} & {100 * r.deadline_violation_rate.mean():.1f} & "
                        f"{r.objective.mean():.3f} & {r.bytes_uploaded_MB.mean():,.0f}")
        if ua is not None:
            u = ua[ua.B_wan_mbps == bw]
            for n in ("RADS", "EDF-LL"):
                r = u[(u.scheduler == n) & (u.workload == "upload-all")]
                if r.empty:
                    continue
                rows.append(f"{n}, upload-all workload & {100 * r.alert_violation_rate.mean():.1f} $\\pm$ {100 * r.alert_violation_rate.std():.1f} & "
                            f"{r.alert_p95_latency.mean():.1f} & {100 * r.deadline_violation_rate.mean():.1f} & "
                            f"-- & {r.bytes_uploaded_MB.mean():,.0f}")
        rows.append("\\midrule")
    rows = rows[:-1]
    _tab("tab_e8_ablation.tex", f"Ablation of RADS components and effect of tiered data retrieval in the outbreak scenario "
         f"(mean over {ob.rep.nunique()} replications). Upload-all: the same captures, but every image, including those "
         "the device would keep as ROUTINE, is uploaded in full as a STRATEGIC job (no tier priority; RADS still orders "
         "jobs by calibrated risk within the tier); for this workload the URGENT "
         "columns refer to the images that the device classified as URGENT, measured against the 10-s deadline, and $J$ is not "
         "reported because all jobs carry the same weight. MB: data "
         "volume sent from devices (images, alerts and records).",
         "tab:e8_ablation", "lccccr", "Variant & URG viol.\\ (\\%) & URG p95 (s) & All viol.\\ (\\%) & $J$ & MB", rows,
         size="\\scriptsize")
    # paired-test table (supplementary)
    tr = pd.DataFrame(tests_rows)
    rows = []
    for bw in sorted(tr.B_wan_mbps.unique(), reverse=True):
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{Fog--cloud uplink {bw:g} Mbit\\,s$^{{-1}}$}}}}")
        for b in [n for n in SCHED_ORDER if n != "RADS"]:
            x = tr[(tr.B_wan_mbps == bw) & (tr.baseline == b)]
            if x.empty:
                continue
            u = x[x.metric == "urgent_violation_rate"].iloc[0]
            o = x[x.metric == "objective"].iloc[0]
            rows.append(f"{b} & ${100 * u.median_diff:+.1f}$ & ${_p(u.p_holm)}$ & ${u.rank_biserial:+.2f}$ & "
                        f"${o.median_diff:+.3f}$ & ${_p(o.p_holm)}$ & ${o.rank_biserial:+.2f}$")
        rows.append("\\midrule")
    rows = rows[:-1]
    _tab("tab_s_sched_tests.tex", f"Paired comparison of RADS with each baseline in the outbreak scenario ({ob.rep.nunique()} "
         "paired replications with common random numbers). Median paired difference RADS minus baseline (negative favors "
         "RADS), two-sided Wilcoxon signed-rank $p$ with Holm adjustment over baselines, and matched-pairs rank-biserial "
         "correlation $r$.", "tab:s_sched_tests", "lcccccc",
         "Baseline & $\\Delta$ URG viol.\\ (pp) & $p_{\\mathrm{Holm}}$ & $r$ & $\\Delta J$ & $p_{\\mathrm{Holm}}$ & $r$", rows,
         size="\\scriptsize")


def e8_load():
    ld = pd.read_csv(RAW / "sched_load.csv")
    macro("LoadReps", str(int(ld.rep.nunique())))
    rates = sorted(ld.rate.unique())
    macro("LoadRateMin", str(int(rates[0])))
    macro("LoadRateMax", str(int(rates[-1])))
    names = [n for n in SCHED_ORDER if n in set(ld.scheduler)]
    for (rate, sch), r in ld.groupby(["rate", "scheduler"]):
        tag = "Load" + SCHED_TAG[sch] + num(rate).replace("{,}", "")
        tag = re.sub(r"\d+", lambda mm: {8: "Eight", 16: "Sixteen", 24: "TwentyFour", 32: "ThirtyTwo", 40: "Forty",
                                          48: "FortyEight"}.get(int(mm.group()), "X"), tag)
        macro(tag + "UViol", f"{100 * r.urgent_violation_rate.mean():.1f}")
        macro(tag + "UPnf", f"{r.urgent_p95_latency.mean():.2f}")
        macro(tag + "Viol", f"{100 * r.deadline_violation_rate.mean():.1f}")
    rows = []
    for n in names:
        cells = []
        for rate in rates:
            r = ld[(ld.scheduler == n) & (ld.rate == rate)]
            cells.append(f"{100 * r.urgent_violation_rate.mean():.1f}")
        rows.append(("\\textbf{RADS}" if n == "RADS" else n) + " & " + " & ".join(cells))
    _tab("tab_s_load.tex", f"URGENT deadline violations (\\%) under stationary load (uplink "
         f"{CFG['scheduling']['params']['B_wan']} Mbit\\,s$^{{-1}}$, no surge; mean over {ld.rep.nunique()} replications).",
         "tab:s_load", "l" + "c" * len(rates), "Scheduler & " + " & ".join(f"$\\lambda={int(x)}$" for x in rates), rows)


def e8_sens():
    se = pd.read_csv(RAW / "sched_sensitivity.csv")
    macro("SensReps", str(int(se.rep.nunique())))
    names = [n for n in ("FCFS-Local", "EDF-LL", "EDF-LL + priority links", "RADS") if n in set(se.scheduler)]
    rows = []
    for fs in sorted(se.fog_speed.unique()):
        for bw in sorted(se.B_wan_mbps.unique()):
            t = se[(se.fog_speed == fs) & (se.B_wan_mbps == bw)]
            cells = [f"{100 * t[t.scheduler == n].urgent_violation_rate.mean():.1f}" for n in names]
            cells += [f"{t[t.scheduler == n].objective.mean():.3f}" for n in names]
            rows.append(f"{fs:g} & {bw:g} & " + " & ".join(cells))
    # summary macros: worst-case RADS URGENT violations and number of settings where RADS has lowest violation
    g = se.groupby(["fog_speed", "B_wan_mbps", "scheduler"]).urgent_violation_rate.mean().unstack()
    FS = {0.125: "Slow", 0.25: "Mid", 0.5: "Fast"}
    for (fs, bw), row in g.iterrows():
        for sch in g.columns:
            macro(f"Sens{FS[float(fs)]}{BW_TAG[float(bw)]}{SCHED_TAG[sch]}UViol", f"{100 * row[sch]:.1f}")
    for (fs, bw), row in se.groupby(["fog_speed", "B_wan_mbps", "scheduler"]).objective.mean().unstack().iterrows():
        for sch in row.index:
            macro(f"Sens{FS[float(fs)]}{BW_TAG[float(bw)]}{SCHED_TAG[sch]}Obj", f"{row[sch]:.3f}")
    macro("SensRadsMaxUViol", f"{100 * g['RADS'].max():.1f}")
    best = (g["RADS"] <= g.drop(columns="RADS").min(axis=1) + 1e-12).sum()
    macro("SensRadsBestCount", str(int(best)))
    macro("SensSettings", str(int(len(g))))
    go = se.groupby(["fog_speed", "B_wan_mbps", "scheduler"]).objective.mean().unstack()
    macro("SensRadsBestObjCount", str(int((go["RADS"] <= go.drop(columns="RADS").min(axis=1) + 1e-12).sum())))
    ab = {"FCFS-Local": "FCFS", "EDF-LL": "EDF-LL", "EDF-LL + priority links": "EDF-LL+PL", "RADS": "RADS"}
    k = len(names)
    hdr = (f"& & \\multicolumn{{{k}}}{{c}}{{URGENT violations (\\%)}} & \\multicolumn{{{k}}}{{c}}{{Objective $J$}} \\\\ "
           f"\\cmidrule(lr){{3-{2 + k}}}\\cmidrule(lr){{{3 + k}-{2 + 2 * k}}} "
           "Fog speed & Uplink & " + " & ".join(ab[n] for n in names) + " & " + " & ".join(ab[n] for n in names))
    _tab("tab_s_sens.tex", f"Sensitivity of the outbreak results to fog-server speed (relative to the reference CPU) and "
         f"fog--cloud uplink (Mbit\\,s$^{{-1}}$): URGENT violations (\\%, first block) and objective $J$ (second block), mean "
         f"over {se.rep.nunique()} replications.", "tab:s_sens", "cc" + "c" * (2 * len(names)), hdr, rows, size="\\tiny")


def conv():
    z = np.load(RAW / "sched_convergence.npz")
    for k, T in (("GA", "Ga"), ("PSO", "Pso")):
        H = z[k]
        rel = H / H[:, :1]
        macro(f"Conv{T}Batches", num(len(H)))
        macro(f"Conv{T}Final", pct(1 - rel[:, -1].mean()))
        macro(f"Conv{T}Last", pct(rel[:, 10].mean() - rel[:, -1].mean()))


def confusion():
    cm = pd.read_csv(RAW / "eval_confusion_ensemble.csv", index_col=0).to_numpy()
    cls = CFG["classes"]
    short = {c: "".join(w[0].upper() for w in c.split("_")) for c in cls}
    short.update({"normal": "N", "brown_spot": "BS", "blast": "Bl", "bacterial_panicle_blight": "BPB", "dead_heart": "DH",
                  "downy_mildew": "DM", "hispa": "Hi", "tungro": "Tu"})
    rows = [f"{c.replace('_', ' ')} ({short[c]}) & " + " & ".join(str(int(v)) for v in cm[i]) for i, c in enumerate(cls)]
    cm = pd.DataFrame(cm, index=cls, columns=cls)
    off = [(cm.iloc[i, j], cls[i], cls[j]) for i in range(len(cls)) for j in range(len(cls)) if i != j]
    off.sort(reverse=True)
    for k, (n_, a_, b_) in enumerate(off[:3]):
        tag = ["One", "Two", "Three"][k]
        macro(f"Conf{tag}True", a_.replace("_", " "))
        macro(f"Conf{tag}Pred", b_.replace("_", " "))
        macro(f"Conf{tag}N", str(int(n_)))
    macro("ConfErrors", num(int(sum(o[0] for o in off))))
    _tab("tab_s_confusion.tex", "Confusion matrix of the temperature-scaled cloud ensemble on the hold-out partition "
         "(rows: true class, columns: predicted class).", "tab:s_confusion", "l" + "r" * len(cm.columns),
         "True class & " + " & ".join(short[c] for c in cm.columns), rows, size="\\scriptsize")


def per_class():
    pc = pd.read_csv(RAW / "eval_per_class.csv").set_index("class")
    cm = pd.read_csv(RAW / "eval_confusion_ensemble.csv", index_col=0).to_numpy()
    rows = []
    for i, c in enumerate(CFG["classes"]):
        r = pc.loc[c]
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        rec = cm[i, i] / max(cm[i].sum(), 1)
        hi = "\\checkmark" if c in CFG["risk"]["high_severity"] else ""
        rows.append(f"{c.replace('_', ' ')} & {hi} & {int(r.n_holdout):,} & {100 * r.edge_f1_mean:.1f} $\\pm$ {100 * r.edge_f1_sd:.1f} & "
                    f"{100 * prec:.1f} & {100 * rec:.1f} & {100 * r.ensemble_f1:.1f}")
        tag = "".join(w.capitalize() for w in c.split("_"))
        macro(f"Pc{tag}EnsFone", f"{100 * r.ensemble_f1:.1f}")
        macro(f"Pc{tag}EnsRec", f"{100 * rec:.1f}")
        macro(f"Pc{tag}EdgeFone", f"{100 * r.edge_f1_mean:.1f}")
    hf = [pc.loc[c, "edge_f1_mean"] for c in CFG["risk"]["high_severity"]]
    macro("EdgeHFoneMin", f"{100 * min(hf):.1f}")
    macro("EdgeHFoneMax", f"{100 * max(hf):.1f}")
    worst = pc.ensemble_f1.idxmin()
    macro("PcWorstClass", worst.replace("_", " "))
    macro("PcWorstFone", f"{100 * pc.ensemble_f1.min():.1f}")
    best = pc.ensemble_f1.idxmax()
    macro("PcBestClass", best.replace("_", " "))
    macro("PcBestFone", f"{100 * pc.ensemble_f1.max():.1f}")
    _tab("tab_s_perclass.tex", "Per-class results on the hold-out partition: F1 of the temperature-scaled edge models (mean $\\pm$ SD "
         "over five seeds) and precision, recall and F1 of the cloud ensemble (\\%). $\\mathcal{H}$: default high-severity classes.",
         "tab:s_perclass", "lcrcccc", "Class & $\\mathcal{H}$ & Images & Edge F1 & Ens.\\ precision & Ens.\\ recall & Ens.\\ F1", rows)


SECTIONS = [audit, versions, leakage, latency, params_tables, xai, e1, e2, e2b, e3, e3b, e4, e7, e5, e8, e8_load, e8_sens, conv, confusion, per_class]

if __name__ == "__main__":
    for fn in SECTIONS:
        try:
            fn()
        except (FileNotFoundError, KeyError) as e:
            print("skip", fn.__name__, e)
    write()
    if "GEN_DIR" not in __import__("os").environ:
        write_summary()
