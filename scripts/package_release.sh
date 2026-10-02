#!/usr/bin/env bash
# Copies generated tables into results/tables (figures stay in manuscript/figures), then writes a GitHub-ready zip that excludes everything listed in
# .gitignore (images, image caches, model weights, optimiser checkpoints, large feature arrays).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/tables
cp manuscript/generated/*.tex results/tables/
OUT=${1:-../rice-conformal-risk-tiering.zip}
rm -f "$OUT"
python3 - "$OUT" <<'PY'
import fnmatch, os, pathlib, sys, zipfile
root = pathlib.Path(".").resolve()
pats = [l.strip() for l in open(".gitignore") if l.strip() and not l.startswith("#")]
def ignored(rel):
    for p in pats:
        p = p.rstrip("/")
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p + "/*") or any(fnmatch.fnmatch(part, p) for part in rel.split("/")):
            return True
    return False
n = 0
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith(".") and not ignored(os.path.relpath(os.path.join(dp, d), root))]
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), root)
            if f.startswith(".") and f != ".gitignore":
                continue
            if ignored(rel):
                continue
            z.write(os.path.join(dp, f), "rice-conformal-risk-tiering/" + rel)
            n += 1
print(n, "files ->", sys.argv[1])
PY
