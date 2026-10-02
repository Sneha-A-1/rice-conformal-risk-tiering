"""Data audit and group-aware splitting.

1. Computes dihedral-invariant perceptual hashes (pHash over the 8 rotations/flips)
   for every image of Paddy Doctor and the Sethy rice dataset.
2. Groups near-duplicates (Hamming distance <= threshold, union-find).
3. Reports exact duplicates, near-duplicate groups and label conflicts.
4. Writes group-aware stratified train/val/holdout splits for Paddy Doctor.

Outputs (in paths.processed):
  paddy_index.csv  sethy_index.csv  audit_summary.json
"""
import argparse
import collections
import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from near_dups import groups, hashes  # noqa: E402


def sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def original_sha(root, name, paths):
    """SHA-256 of the ORIGINAL (pre-resize) files, taken from the upload manifest when
    available (raw/<name>/<name>_manifest.csv); otherwise the hash of the file on disk."""
    man = root / "raw" / name / f"{name}_manifest.csv"
    if not man.exists():
        return [sha256(p) for p in paths], "file_on_disk"
    m = pd.read_csv(man)
    lut = dict(zip(m.out_path, m.orig_sha256))
    base = root / name
    return [lut[str(pathlib.Path(p).relative_to(base).as_posix())] for p in paths], "original_manifest"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = pathlib.Path(cfg["paths"]["data_root"])
    out = pathlib.Path(cfg["paths"]["processed"])
    out.mkdir(parents=True, exist_ok=True)
    thr = cfg["audit"]["phash_hamming_threshold"]

    # ---------------- Paddy Doctor ----------------
    pdf = pd.read_csv(root / "paddy" / "train.csv")
    pdf["path"] = [str(root / "paddy" / "train_images" / l / i) for i, l in zip(pdf.image_id, pdf.label)]
    pdf["sha256"], paddy_hash_src = original_sha(root, "paddy", pdf.path)
    print("hashing Paddy Doctor ...")
    H = hashes(pdf.path)
    pdf["group"] = groups(H, thr)
    lab = pdf.groupby("group").label.nunique()
    pdf["label_conflict_group"] = pdf.group.map(lab > 1)

    # group-aware stratified split (labels of conflict groups: majority label for stratification)
    major = pdf.groupby("group").label.agg(lambda s: s.value_counts().index[0])
    sgkf = StratifiedGroupKFold(n_splits=cfg["split"]["n_folds"], shuffle=True, random_state=cfg["seed"])
    fold = np.full(len(pdf), -1)
    for k, (_, te) in enumerate(sgkf.split(pdf, pdf.group.map(major), groups=pdf.group)):
        fold[te] = k
    s = cfg["split"]
    pdf["fold"] = fold
    pdf["split"] = np.where(fold < s["train_folds"], "train",
                            np.where(fold < s["train_folds"] + s["val_folds"], "val", "holdout"))

    # ---------------- Sethy ----------------
    spaths = sorted(str(p) for p in (root / "sethy").rglob("*.jpg"))
    sdf = pd.DataFrame({"path": spaths})
    sdf["folder"] = [pathlib.Path(p).parent.name for p in spaths]
    sdf["label"] = sdf.folder.map(cfg["sethy_map"])
    sdf["sha256"], sethy_hash_src = original_sha(root, "sethy", spaths)
    print("hashing Sethy ...")
    Hs = hashes(spaths)
    sdf["group"] = groups(Hs, thr)
    sdf["exact_group"] = sdf.sha256.astype("category").cat.codes

    # cross-dataset near-duplicates (should be none)
    from near_dups import popcount64
    cross = int(sum((popcount64(Hs ^ h).min() <= thr) for h in H[:, 0]))

    summary = {
        "paddy": {
            "images": int(len(pdf)),
            "exact_duplicate_files": int(pdf.sha256.duplicated(keep=False).sum()),
            "exact_duplicate_groups": int((pdf.sha256.value_counts() > 1).sum()),
            "near_duplicate_groups": int(pdf.group.nunique()),
            "images_in_multi_member_groups": int(pdf.group.map(pdf.group.value_counts()).gt(1).sum()),
            "label_conflict_groups": int((lab > 1).sum()),
            "images_in_label_conflict_groups": int(pdf.label_conflict_group.sum()),
            "split_counts": pdf.split.value_counts().to_dict(),
            "split_groups": pdf.groupby("split").group.nunique().to_dict(),
            "class_by_split": pd.crosstab(pdf.label, pdf.split).to_dict(),
        },
        "sethy": {
            "images": int(len(sdf)),
            "unique_files": int(sdf.sha256.nunique()),
            "exact_duplicate_files": int(sdf.sha256.duplicated(keep=False).sum()),
            "near_duplicate_groups": int(sdf.group.nunique()),
            "groups_per_class": sdf.groupby("label").group.nunique().to_dict(),
            "images_per_class": sdf.label.value_counts().to_dict(),
            "label_conflict_groups": int((sdf.groupby("group").label.nunique() > 1).sum()),
        },
        "cross_dataset_near_duplicates": cross,
        "phash_hamming_threshold": thr,
        "exact_hash_source": {"paddy": paddy_hash_src, "sethy": sethy_hash_src},
    }
    pdf.to_csv(out / "paddy_index.csv", index=False)
    sdf.to_csv(out / "sethy_index.csv", index=False)
    np.save(out / "paddy_phash.npy", H)
    np.save(out / "sethy_phash.npy", Hs)
    json.dump(summary, open(out / "audit_summary.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in summary.items()}, indent=1)[:3000])


if __name__ == "__main__":
    main()
