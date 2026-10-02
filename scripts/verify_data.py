"""Verify local data against the manifests used in the paper (SHA-256 of the resized files)."""
import hashlib
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "data"
bad = 0
for name in ("paddy", "sethy"):
    m = pd.read_csv(ROOT / "data" / "manifests" / f"{name}_manifest.csv")
    for p, h in zip(m.out_path, m.out_sha256):
        f = DATA / name / p
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != h:
            bad += 1
    print(name, "files:", len(m), "mismatches so far:", bad)
sys.exit(1 if bad else 0)
