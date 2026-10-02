"""Generate all manuscript figures from result files (PDF + 300-dpi PNG) into manuscript/figures/.
Figures whose inputs do not exist yet are skipped."""
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from evaluation import plotstyle as ps  # noqa: E402

CFG = yaml.safe_load(open(ROOT / "configs" / "default.yaml"))
RAW = pathlib.Path(__import__("os").environ.get("RESULTS_RAW", ROOT / "results" / "raw"))
PROC = ROOT / "data" / "processed"
FIG = pathlib.Path(__import__("os").environ.get("FIG_DIR", ROOT / "manuscript" / "figures"))
FIG.mkdir(parents=True, exist_ok=True)
CLASSES = CFG["classes"]
SHORT = {"bacterial_leaf_blight": "BLB", "bacterial_leaf_streak": "BLS", "bacterial_panicle_blight": "BPB",
         "blast": "Blast", "brown_spot": "Brown spot", "dead_heart": "Dead heart", "downy_mildew": "Downy mildew",
         "hispa": "Hispa", "normal": "Normal", "tungro": "Tungro"}
ps.apply()


def fig_audit():
    """Figure 2: near-duplicate structure of both datasets."""
    pdf = pd.read_csv(PROC / "paddy_index.csv")
    sdf = pd.read_csv(PROC / "sethy_index.csv")
    parr = np.load(PROC / "paddy_images.npy", mmap_mode="r")
    sarr = np.load(PROC / "sethy_images.npy", mmap_mode="r")
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.05], hspace=0.5, wspace=0.12)
    ax = fig.add_subplot(gs[0, :3])
    for k, (name, d, c, m) in enumerate([("Paddy Doctor", pdf, ps.CAT[0], "o"), ("Sethy", sdf, ps.CAT[1], "s")]):
        gsize = d.group.map(d.group.value_counts()).to_numpy()
        xs = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50])
        share = [(gsize >= x).mean() for x in xs]
        ax.plot(xs, share, marker=m, linestyle=ps.LINES[k], color=c,
                label=f"{name}: {len(d):,} images, {d.group.nunique():,} groups")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_ylim(0, 1.32)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Group size $s$ (images)")
    ax.set_ylabel("Share of images in groups\nof size $\\geq s$")
    ax.legend(loc="upper right", fontsize=6.5)
    ps.panel_label(ax, "A")
    ax2 = fig.add_subplot(gs[0, 4:])
    a = json.load(open(PROC / "audit_summary.json"))
    labels = ["Exact\nduplicate", "Near-dup.\ngroup > 1", "Label-\nconflict"]
    pv = [a["paddy"]["exact_duplicate_files"], a["paddy"]["images_in_multi_member_groups"], a["paddy"]["images_in_label_conflict_groups"]]
    sv = [a["sethy"]["exact_duplicate_files"], int(sdf.group.map(sdf.group.value_counts()).gt(1).sum()),
          int(sdf.group.isin(sdf.groupby("group").label.nunique().loc[lambda v: v > 1].index).sum())]
    x = np.arange(len(labels))
    b1 = ax2.bar(x - 0.19, np.array(pv) / a["paddy"]["images"], width=0.36, color=ps.CAT[0], label="Paddy Doctor")
    b2 = ax2.bar(x + 0.19, np.array(sv) / a["sethy"]["images"], width=0.36, color=ps.CAT[1], label="Sethy",
                 hatch="////", edgecolor="white", linewidth=0)
    for bars, vals in ((b1, pv), (b2, sv)):
        for b, v in zip(bars, vals):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:,}", ha="center", fontsize=6, color=ps.INK2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7)
    ax2.set_ylim(0, 1.32)
    ax2.set_ylabel("Share of images")
    ax2.legend(loc="upper left", fontsize=6.5, ncol=2)
    ax2.grid(axis="x", visible=False)
    ps.panel_label(ax2, "B")
    # examples: two Sethy groups and one Paddy group, 6 images each across the bottom row (3 + 3)
    rng = np.random.default_rng(3)
    sg = sdf.group.value_counts()
    big = sg[sg >= 6].index.to_numpy()
    g1 = rng.choice(big)
    pg = pdf.group.value_counts()
    pbig = pg[(pg >= 3) & (pg <= 12)].index.to_numpy()
    g2 = rng.choice(pbig)
    ex = [(sarr, sdf[sdf.group == g1].index[:3], "Sethy"), (parr, pdf[pdf.group == g2].index[:3], "Paddy Doctor")]
    col = 0
    for arr, idx, nm in ex:
        for j, i in enumerate(idx):
            a_ = fig.add_subplot(gs[1, col])
            a_.imshow(arr[i])
            a_.set_xticks([])
            a_.set_yticks([])
            a_.grid(False)
            for s in a_.spines.values():
                s.set_visible(True)
                s.set_color(ps.AXIS)
            if j == 0:
                a_.text(0, 1.04, f"{nm}: one near-duplicate group", transform=a_.transAxes, fontsize=7.5, va="bottom")
            if col == 0:
                ps.panel_label(a_, "C")
            col += 1
    ps.save(fig, FIG / "fig2_audit")


FIGS = [fig_audit]



# ---------------------------------------------------------------- results figures
def _ci_corr(x, ratio):
    """Half-width of the 95% interval of a mean over J overlapping resamples with the Nadeau-Bengio correction."""
    x = np.asarray(x, float)
    from scipy.stats import t
    return float(x.mean()), t.ppf(0.975, len(x) - 1) * np.sqrt((1 / len(x) + ratio) * x.var(ddof=1))


def _ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, 0.0
    from scipy.stats import t
    return m, t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def fig_calibration():
    """Figure 3: training curves, reliability diagrams, per-class F1."""
    e1 = json.load(open(RAW / "eval_e1_accuracy_calibration.json"))
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.1))
    ax = axs[0]
    for k, s in enumerate(e1["seeds"]):
        h = pd.read_csv(RAW / "models" / f"seed{s}" / "history.csv")
        ax.plot(h.epoch, h.val_macro_f1, color=ps.CAT[0], alpha=0.35 + 0.13 * k, marker=ps.MARKERS[k], markersize=3,
                linewidth=1.0, label=f"seed {s}")
    ax.set_xlabel("Epoch")
    ax.set_xticks([1, 2, 4, 6, 8, 10])
    ax.set_ylabel("Validation macro-F1")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ps.panel_label(ax, "A")
    ax = axs[1]
    ax.plot([0, 1], [0, 1], color=ps.MUTED, linewidth=0.8, linestyle=":")
    from scipy.special import softmax
    sys.path.insert(0, str(ROOT / "src"))
    from trust.calibration import reliability_curve
    idx = pd.read_csv(PROC / "paddy_index.csv")
    y_all = idx.label.map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    pr, pt, yy = [], [], []
    for k_, s_ in enumerate(e1["seeds"]):
        d_ = RAW / "models" / f"seed{s_}"
        lh = np.load(d_ / "logits_holdout.npy")
        pr.append(softmax(lh, axis=1))
        pt.append(softmax(lh / e1["temperatures"][k_], axis=1))
        yy.append(y_all[np.load(d_ / "index_holdout.npy")])
    curves = [(np.array(reliability_curve(np.concatenate(pr), np.concatenate(yy))), "Edge (5 seeds pooled), raw"),
              (np.array(reliability_curve(np.concatenate(pt), np.concatenate(yy))), "Edge (5 seeds pooled), temperature-scaled"),
              (np.array(e1["reliability"]["ensemble_ts"], dtype=float), "Cloud ensemble, temperature-scaled")]
    for k, (r, lab) in enumerate(curves):
        ax.plot(r[:, 3], r[:, 4], marker=ps.MARKERS[k], linestyle=ps.LINES[k], color=ps.CAT[k], label=lab)
    ax.set_xlabel("Mean confidence (bin)")
    ax.set_ylabel("Accuracy (bin)")
    ax.set_xlim(0.2, 1.01)
    ax.set_ylim(0.0, 1.01)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False)
    ps.panel_label(ax, "B")
    ax = axs[2]
    pc = pd.read_csv(RAW / "eval_per_class.csv")
    y = np.arange(len(pc))
    ax.errorbar(pc.edge_f1_mean, y + 0.15, xerr=pc.edge_f1_sd, fmt="o", color=ps.CAT[0], markersize=3.5, capsize=2,
                label="Edge (mean $\\pm$ SD, 5 seeds)")
    ax.plot(pc.ensemble_f1, y - 0.15, "s", color=ps.CAT[1], markersize=3.5, label="Cloud ensemble")
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[c] + (" *" if c in CFG["risk"]["high_severity"] else "") for c in pc["class"]], fontsize=6.5)
    ax.invert_yaxis()
    ax.set_xlabel("Hold-out F1")
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.4, -0.24), frameon=False)
    ax.grid(axis="y", visible=False)
    ps.panel_label(ax, "C")
    fig.tight_layout()
    ps.save(fig, FIG / "fig3_accuracy_calibration")


def fig_conformal():
    """Figure 4: coverage, set size and URGENT recall / false alarms versus alpha."""
    cp = pd.read_csv(RAW / "eval_e2_conformal_splits.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    methods = ["LAC-marginal", "LAC-classwise", "APS-marginal"]
    for k, mth in enumerate(methods):
        d = cp[(cp.model == "edge") & (cp.method == mth)].groupby("alpha")
        a = d.coverage.mean().index.to_numpy()
        axs[0].plot(1 - a, d.coverage.mean(), marker=ps.MARKERS[k], linestyle=ps.LINES[k], color=ps.CAT[k], label=mth)
        axs[1].plot(1 - a, d.avg_set_size.mean(), marker=ps.MARKERS[k], linestyle=ps.LINES[k], color=ps.CAT[k], label=mth)
    axs[0].plot([0.78, 1], [0.78, 1], color=ps.MUTED, linestyle=":", linewidth=0.8)
    axs[0].set_xlabel("Target coverage $1-\\alpha$")
    axs[0].set_ylabel("Empirical coverage (edge)")
    axs[0].legend(fontsize=6)
    ps.panel_label(axs[0], "A")
    axs[1].set_xlabel("Target coverage $1-\\alpha$")
    axs[1].set_ylabel("Average set size (edge)")
    ps.panel_label(axs[1], "B")
    ax = axs[2]
    for k, (model, mk) in enumerate([("edge", "o"), ("ensemble", "s")]):
        d = cp[(cp.model == model) & (cp.method == "LAC-classwise")].groupby("alpha")
        a = d.coverage.mean().index.to_numpy()
        ax.plot(1 - a, d.urgent_recall.mean(), marker=mk, color=ps.CAT[0], linestyle=ps.LINES[k], label=f"URGENT recall, {model}")
        ax.plot(1 - a, d.urgent_false_alarm.mean(), marker=mk, color=ps.CAT[1], linestyle=ps.LINES[k], label=f"False alarm, {model}")
    ax.plot([0.78, 1], [0.78, 1], color=ps.MUTED, linestyle=":", linewidth=0.8)
    ax.set_xlabel("Target coverage $1-\\alpha$")
    ax.set_ylabel("Rate (LAC class-conditional)")
    ax.legend(fontsize=5.8, loc="center left")
    ps.panel_label(ax, "C")
    fig.tight_layout()
    ps.save(fig, FIG / "fig4_conformal")


def fig_cascade_shift():
    """Figure 5: cascade gates (A), cross-source shift (B) and target recalibration (C)."""
    cas = pd.read_csv(RAW / "eval_e3_cascade_splits.csv")
    e4 = json.load(open(RAW / "eval_e4_shift.json"))
    rc = pd.read_csv(RAW / "eval_e4_recalibration.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axs = axs.ravel()
    ax = axs[0]
    order = ["edge-only", "random", "max-softmax", "entropy", "risk-mass", "conformal-tier (proposed)", "cloud-only"]
    y = np.arange(len(order))
    for k, (met, lab) in enumerate([("macro_f1", "Macro-F1"), ("high_sev_recall", "High-severity recall")]):
        m = [cas[cas.gate == g][met].mean() for g in order]
        e = [_ci_corr(cas[cas.gate == g][met], 1.0)[1] for g in order]
        ax.errorbar(m, y + (0.15 if k == 0 else -0.15), xerr=e, fmt=ps.MARKERS[k], color=ps.CAT[k], markersize=3.5, capsize=2, label=lab)
    ax.set_yticks(y)
    ax.set_yticklabels([o.replace(" (proposed)", "*") for o in order], fontsize=6.5)
    ax.set_xlabel("Hold-out test value (mean and corrected 95% interval)")
    ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    ax.grid(axis="y", visible=False)
    ps.panel_label(ax, "A")
    ax = axs[1]
    labels = ["Accuracy", "Coverage", "URGENT\nrecall"]
    keys = ["accuracy", "coverage_paddy_calibrated", "urgent_recall_H"]
    vals = [e4["ensemble"][k] for k in keys]
    x = np.arange(len(labels))
    ax.bar(x, [v[0] for v in vals], width=0.55, color=ps.CAT[1], hatch="////", edgecolor="white", linewidth=0, label="Sethy (cross-source)")
    ax.errorbar(x, [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], fmt="none", ecolor=ps.INK2, capsize=2)
    for xi, v in zip(x, vals):
        ax.text(xi, v[2] + 0.02, f"{v[0]:.2f}", ha="center", fontsize=6.5, color=ps.INK2)
    ax.axhline(1 - CFG["risk"]["alpha"], color=ps.MUTED, linestyle=":", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cloud ensemble on Sethy")
    ax.grid(axis="x", visible=False)
    ps.panel_label(ax, "B")
    ax = axs[2]
    for k, (model, mode) in enumerate([("ensemble", "marginal"), ("ensemble", "class"), ("edge", "class")]):
        d = rc[(rc.model == model) & (rc["mode"] == mode)].groupby("m_groups")
        ax.plot(d.coverage.mean().index, d.coverage.mean(), marker=ps.MARKERS[k], linestyle=ps.LINES[k], color=ps.CAT[k],
                label=f"{model}, {'class-cond.' if mode == 'class' else 'marginal'}")
        lo = d.coverage.quantile(0.05)
        ax.fill_between(lo.index, lo, d.coverage.quantile(0.95), color=ps.CAT[k], alpha=0.12, linewidth=0)
    ax.axhline(1 - CFG["risk"]["alpha"], color=ps.MUTED, linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 40, 80])
    ax.set_xticklabels(["5", "10", "20", "40", "80"])
    ax.set_xlabel("Labeled Sethy groups used to recalibrate")
    ax.set_ylabel("Coverage on remaining groups")
    ax.legend(fontsize=6, loc="lower right")
    ps.panel_label(ax, "C")
    ax = axs[3]
    for k, (model, mode) in enumerate([("ensemble", "marginal"), ("ensemble", "class"), ("edge", "class")]):
        d = rc[(rc.model == model) & (rc["mode"] == mode)].groupby("m_groups")
        ax.plot(d.avg_set_size.mean().index, d.avg_set_size.mean(), marker=ps.MARKERS[k], linestyle=ps.LINES[k],
                color=ps.CAT[k], label=f"{model}, {'class-cond.' if mode == 'class' else 'marginal'}")
    ax.axhline(e4["ensemble"]["avg_set_size"], color=ps.MUTED, linestyle=":", linewidth=0.8)
    ax.text(5, e4["ensemble"]["avg_set_size"] + 0.15, "source-calibrated (ensemble)", fontsize=6, color=ps.INK2)
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 40, 80])
    ax.set_xticklabels(["5", "10", "20", "40", "80"])
    ax.set_ylim(0, len(CLASSES))
    ax.set_xlabel("Labeled Sethy groups used to recalibrate")
    ax.set_ylabel("Average set size (of %d classes)" % len(CLASSES))
    ps.panel_label(ax, "D")
    fig.tight_layout()
    ps.save(fig, FIG / "fig5_cascade_shift")


def fig_poisoning():
    d = pd.read_csv(RAW / "poisoning_results.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.0))
    axs = axs.ravel()
    defs = ["none", "CL-filter", "kNN-filter"]
    c = d[d.attack == "concealment"]
    for k, df_ in enumerate(defs):
        g = c[c.defense == df_].groupby("rho")
        for ax, col in ((axs[0], "high_sev_recall"), (axs[1], "concealment_rate"), (axs[2], "macro_f1")):
            m = g[col].mean()
            a_ = json.load(open(PROC / "audit_summary.json"))["paddy"]["split_counts"]
            e = g[col].apply(lambda x: _ci_corr(x, (a_["holdout"] / 2) / a_["train"])[1])
            ax.errorbar(m.index, m, yerr=e, marker=ps.MARKERS[k], linestyle=ps.LINES[k], color=ps.CAT[k], label=df_,
                        capsize=2, markersize=4)
    for ax, lab, p in ((axs[0], "High-severity recall", "A"), (axs[1], "High-severity images\npredicted normal", "B"),
                       (axs[2], "Macro-F1", "C")):
        ax.set_xlabel("Poisoned fraction $\\rho$ (concealment)")
        ax.set_ylabel(lab)
        ps.panel_label(ax, p)
    axs[0].legend(fontsize=6.5)
    ax = axs[3]
    g = c[c.defense == "none"].groupby("rho")
    ax.plot(g.urgent_recall_trusted_cal.mean().index, g.urgent_recall_trusted_cal.mean(), marker="o", color=ps.CAT[0],
            label="URGENT recall, trusted calibration")
    ax.plot(g.urgent_recall_poisoned_cal.mean().index, g.urgent_recall_poisoned_cal.mean(), marker="s", linestyle="--",
            color=ps.CAT[1], label="URGENT recall, poisoned calibration")
    ax.plot(g.high_sev_recall.mean().index, g.high_sev_recall.mean(), marker="^", linestyle=":", color=ps.CAT[3],
            label="Top-1 high-severity recall")
    ax.axhline(1 - CFG["risk"]["alpha"], color=ps.MUTED, linestyle=":", linewidth=0.8)
    for x in (0.0, 0.4):
        sz = g.set_size_trusted_cal.mean().loc[x]
        ax.annotate(f"mean set size {sz:.2f}", (x, g.urgent_recall_trusted_cal.mean().loc[x]), textcoords="offset points",
                    xytext=(0, 7), ha="left" if x == 0 else "right", fontsize=6, color=ps.INK2)
    ax.set_ylim(0.45, 1.04)
    ax.set_xlabel("Poisoned fraction $\\rho$ (concealment, no filter)")
    ax.set_ylabel("Recall of high-severity images")
    ax.legend(fontsize=6, loc="lower left")
    ps.panel_label(ax, "D")
    fig.tight_layout()
    ps.save(fig, FIG / "fig6_poisoning")


def fig_xai():
    """Figure 7: Grad-CAM examples (one hold-out image per class shown) and faithfulness."""
    import timm
    import torch
    from explain.gradcam import gradcam
    from models.train import Normalizer, full_image_batch
    ev = pd.read_csv(RAW / "xai_gradcam_eval.csv")
    pdf = pd.read_csv(PROC / "paddy_index.csv")
    arr = np.load(PROC / "paddy_images.npy", mmap_mode="r")
    model = timm.create_model(CFG["train"]["arch"], pretrained=False, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(RAW / "models" / "seed0" / "best.pt", map_location="cpu"))
    model.eval()
    norm = Normalizer(CFG["image"]["mean"], CFG["image"]["std"])
    show = ["blast", "bacterial_leaf_blight", "tungro", "bacterial_panicle_blight", "brown_spot", "hispa"]
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.3, 1], hspace=0.5, wspace=0.12)
    for k, cname in enumerate(show):
        cand = ev[(ev.true == cname) & ev.correct]
        i = int(cand.index_.iloc[0]) if "index_" in cand else int(cand["index"].iloc[0])
        x = norm(full_image_batch(arr[i:i + 1], CFG["image"]["input_size"]))
        cam, logits = gradcam(model, x)
        from PIL import Image as _I
        img = arr[i].astype(float) / 255
        cm = np.asarray(_I.fromarray((cam[0].numpy() * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0])), dtype=float) / 255
        a = fig.add_subplot(gs[0, k])
        a.imshow(img)
        a.imshow(cm, cmap="magma", alpha=0.45)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
        a.set_title(SHORT[cname] + (" *" if cname in CFG["risk"]["high_severity"] else ""), fontsize=7)
        if k == 0:
            ps.panel_label(a, "A")
    gsb = gs[1, :].subgridspec(1, 2, wspace=0.1)
    for k, (kind, lab) in enumerate([("deletion", "Deletion AUC (lower = more faithful)"), ("insertion", "Insertion AUC (higher = more faithful)")]):
        ax = fig.add_subplot(gsb[0, k])
        data = [ev[f"{kind}_auc_gradcam"], ev[f"{kind}_auc_random"], ev[f"{kind}_auc_centre"]]
        bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, flierprops={"markersize": 2, "markeredgecolor": ps.MUTED},
                        medianprops={"color": ps.INK, "linewidth": 1.2})
        for patch, c in zip(bp["boxes"], [ps.CAT[0], ps.GRAY, ps.GRAY]):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
            patch.set_edgecolor(ps.INK2)
        ax.set_yticks([1, 2, 3])
        ax.set_yticklabels(["Grad-CAM", "random", "center prior"] if k == 0 else [])
        ax.set_xlabel(lab)
        ax.set_xlim(0, 1)
        ps.panel_label(ax, "BC"[k])
    ps.save(fig, FIG / "fig7_xai")


SCHED_ORDER = ["FCFS-Local", "Round-Robin", "Min-Min", "Max-Min", "GA", "PSO", "Cloud-only", "EDF-LL",
               "EDF-LL + priority links", "RADS"]
ABL_ORDER = ["RADS", "RADS w/o tiers", "RADS w/o priority links", "RADS w/o reservation", "RADS + energy/cost terms"]


def fig_sched():
    """Figure 8: load sweep (A, B), outbreak comparison (C) and ablation / upload-all (D)."""
    ob = pd.read_csv(RAW / "sched_outbreak.csv")
    ld = pd.read_csv(RAW / "sched_load.csv")
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.4), gridspec_kw={"height_ratios": [1, 1.35]})
    axs = axs.ravel()
    sel = [("FCFS-Local", 0), ("Round-Robin", 4), ("Min-Min", 3), ("EDF-LL", 1), ("RADS", 2)]
    for k, (name, ci) in enumerate(sel):
        g = ld[ld.scheduler == name].groupby("rate")
        axs[0].plot(g.urgent_violation_rate.mean().index, 100 * g.urgent_violation_rate.mean(), marker=ps.MARKERS[k],
                    linestyle=ps.LINES[k], color=ps.CAT[ci], label=name)
        axs[1].plot(g.urgent_p95_latency.mean().index, g.urgent_p95_latency.mean(), marker=ps.MARKERS[k],
                    linestyle=ps.LINES[k], color=ps.CAT[ci], label=name)
    bw0 = CFG["scheduling"]["params"]["B_wan"]
    axs[0].set_xlabel("Capture rate $\\lambda$ (images s$^{-1}$)")
    axs[0].set_ylabel("URGENT deadline violations (%)")
    axs[0].legend(fontsize=6, loc="upper left")
    axs[0].set_title(f"Stationary load, uplink {bw0:g} Mbit/s", fontsize=7)
    ps.panel_label(axs[0], "A")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("Capture rate $\\lambda$ (images s$^{-1}$)")
    axs[1].set_ylabel("URGENT p95 response time (s)")
    axs[1].axhline(CFG["scheduling"]["deadlines"]["URGENT"], color=ps.MUTED, linestyle=":", linewidth=0.8)
    axs[1].set_title(f"Stationary load, uplink {bw0:g} Mbit/s", fontsize=7)
    ps.panel_label(axs[1], "B")
    bws = sorted(ob.B_wan_mbps.unique(), reverse=True)

    ua = pd.read_csv(RAW / "sched_uploadall.csv") if (RAW / "sched_uploadall.csv").exists() else None

    def series(bw, name):
        if name.endswith("upload-all"):
            if ua is None:
                return pd.Series([np.nan])
            u = ua[(ua.B_wan_mbps == bw) & (ua.workload == "upload-all")]
            return 100 * u[u.scheduler == name.split(", ")[0]].alert_violation_rate
        t = ob[(ob.B_wan_mbps == bw) & (ob.workload == "tiered")]
        return 100 * t[t.scheduler == name].urgent_violation_rate

    def dots(ax, names, workload_of=None):
        y = np.arange(len(names))
        for k, bw in enumerate(bws):
            m, e = [], []
            for n in names:
                v = series(bw, n)
                m.append(v.mean())
                e.append(_ci(v)[1])
            ax.errorbar(m, y + (-0.17 if k == 0 else 0.17), xerr=e, fmt=ps.MARKERS[k], color=ps.CAT[k], markersize=3.5,
                        capsize=2, label=f"uplink {bw:g} Mbit/s")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel("URGENT deadline violations, outbreak (%)")
        ax.legend(fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        ax.grid(axis="y", visible=False)

    names = [n for n in SCHED_ORDER if n in set(ob.scheduler)]
    dots(axs[2], names)
    ps.panel_label(axs[2], "C")
    abl = [n for n in ABL_ORDER if n in set(ob.scheduler)] + ["RADS, upload-all", "EDF-LL, upload-all"]
    dots(axs[3], abl)
    ps.panel_label(axs[3], "D")
    fig.tight_layout()
    ps.save(fig, FIG / "fig8_scheduling")


def fig_convergence():
    z = np.load(RAW / "sched_convergence.npz")
    fig, ax = plt.subplots(figsize=(3.6, 2.5))
    for k, name in enumerate(["GA", "PSO"]):
        H = z[name]
        rel = H / H[:, :1]
        ax.plot(np.arange(H.shape[1]), rel.mean(0), marker=ps.MARKERS[k], linestyle=ps.LINES[k], color=ps.CAT[k], label=f"{name} ({len(H)} batches)")
        ax.fill_between(np.arange(H.shape[1]), np.percentile(rel, 10, 0), np.percentile(rel, 90, 0), color=ps.CAT[k], alpha=0.12, linewidth=0)
    ax.set_xlabel("Generation / iteration")
    ax.set_ylabel("Best fitness / initial best")
    ax.legend(fontsize=6.5)
    ps.save(fig, FIG / "figS2_convergence")


FIGS += [fig_calibration, fig_conformal, fig_cascade_shift, fig_poisoning, fig_xai, fig_sched, fig_convergence]


def fig_s1_groups():
    """Supplementary Figure S1: random near-duplicate groups (Paddy, Sethy) and label-conflict groups."""
    pdf = pd.read_csv(PROC / "paddy_index.csv")
    sdf = pd.read_csv(PROC / "sethy_index.csv")
    parr = np.load(PROC / "paddy_images.npy", mmap_mode="r")
    sarr = np.load(PROC / "sethy_images.npy", mmap_mode="r")
    rng = np.random.default_rng(11)
    rows = []
    pg = pdf.group.value_counts()
    for g in rng.choice(pg[pg >= 3].index, 3, replace=False):
        rows.append(("Paddy Doctor", parr, pdf[pdf.group == g]))
    conf = pdf[pdf.label_conflict_group].group.unique()
    for g in rng.choice(conf, 2, replace=False):
        d = pdf[pdf.group == g]
        # show at least one member of every label in the group first
        first = d.drop_duplicates("label")
        rows.append(("Paddy (conflict)", parr, pd.concat([first, d.drop(first.index)])))
    sg = sdf.group.value_counts()
    for g in rng.choice(sg[sg >= 6].index, 2, replace=False):
        rows.append(("Sethy", sarr, sdf[sdf.group == g]))
    fig, axs = plt.subplots(len(rows), 6, figsize=(7.2, 1.35 * len(rows)))
    for r, (name, arr, d) in enumerate(rows):
        for c in range(6):
            a = axs[r, c]
            a.axis("off")
            if c < len(d):
                i = d.index[c]
                a.imshow(arr[i])
                a.set_title(SHORT.get(d.label.iloc[c], d.label.iloc[c]), fontsize=6, pad=2)
        axs[r, 0].text(-0.08, 0.5, name, transform=axs[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=6.5)
    fig.subplots_adjust(wspace=0.05, hspace=0.35)
    ps.save(fig, FIG / "figS1_groups")


FIGS += [fig_s1_groups]


if __name__ == "__main__":
    only = sys.argv[1:]
    for f in FIGS:
        if only and f.__name__ not in only:
            continue
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
