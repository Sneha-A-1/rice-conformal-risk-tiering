#!/usr/bin/env python3
"""
prepare_upload.py  -  shrink and pack a real image dataset for upload.

What it does
  * Optionally resizes every image so its SHORTER side is --resize pixels
    (aspect ratio kept, no cropping). Models use 224x224 inputs, so 256 loses
    nothing the experiments need and typically cuts the size by about 3x
    (more for high-resolution originals).
  * Packs the result into zip parts below --max-mb (default 450 MB).
  * Writes <name>_manifest.csv: for every file, the SHA-256 of the ORIGINAL
    file and of the uploaded version, plus original and new image sizes, so
    the preprocessing is fully documented and verifiable.
Your original files are only read, never changed.

Requirements: Python 3.8+. For --resize also:  pip install pillow

Examples
  Paddy Doctor (Kaggle version), labelled training data only, resized:
    python prepare_upload.py "D:/Downloads/paddy-disease-classification" paddy --resize 256 --exclude test_images sample_submission.csv

  Sethy rice leaf dataset, resized:
    python prepare_upload.py "D:/Downloads/Rice Leaf Disease Images" sethy --resize 256

Output: a folder upload_<name>/ next to where you run it. Upload every file in it.
"""

import argparse
import csv
import hashlib
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shrink(data: bytes, short_side: int, quality: int):
    """Return (jpeg_bytes, (w0, h0), (w1, h1)). Never enlarges an image."""
    from PIL import Image, ImageOps
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)           # respect camera orientation
        w0, h0 = im.size
        im = im.convert("RGB")
        scale = short_side / min(w0, h0)
        if scale < 1.0:
            w1, h1 = max(1, round(w0 * scale)), max(1, round(h0 * scale))
            im = im.resize((w1, h1), Image.LANCZOS)
        else:
            w1, h1 = w0, h0
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue(), (w0, h0), (w1, h1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="dataset folder (already unzipped)")
    ap.add_argument("name", help="short name for the output files, e.g. paddy or sethy")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="file or folder names to leave out (matched against every path component)")
    ap.add_argument("--resize", type=int, default=0,
                    help="shorter image side in pixels (e.g. 256). 0 = keep originals unchanged")
    ap.add_argument("--quality", type=int, default=92, help="JPEG quality when resizing (default 92)")
    ap.add_argument("--max-mb", type=float, default=450.0, help="max size of each zip part in MB (default 450)")
    ap.add_argument("--out", default=None, help="output folder (default: ./upload_<name>)")
    args = ap.parse_args()

    if args.resize:
        try:
            import PIL  # noqa: F401
        except ImportError:
            print("Pillow is needed for --resize. Install it with:  pip install pillow")
            return 1

    src = Path(args.source).expanduser().resolve()
    if not src.is_dir():
        print(f"ERROR: folder not found: {src}")
        return 1
    out = Path(args.out) if args.out else Path.cwd() / f"upload_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    excluded = set(args.exclude)
    max_bytes = int(args.max_mb * 1024 * 1024)

    files = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(src)
        if excluded & set(rel.parts):
            continue
        if rel.name.startswith(".") or rel.name in {"Thumbs.db", "desktop.ini"}:
            continue
        files.append((p, rel))
    if not files:
        print("ERROR: no files found (check the folder path and --exclude).")
        return 1
    print(f"Found {len(files)} files in {src}")
    print("Processing" + (f" (resizing shorter side to {args.resize}px)" if args.resize else "") + "...")

    rows, per_folder, failed = [], Counter(), []
    part, part_bytes, zf, parts = 0, 0, None, []
    orig_total = 0
    for i, (p, rel) in enumerate(files, 1):
        data = p.read_bytes()
        orig_total += len(data)
        row = {"relative_path": rel.as_posix(), "orig_bytes": len(data), "orig_sha256": sha256_bytes(data),
               "orig_width": "", "orig_height": "", "out_width": "", "out_height": ""}
        arcname = rel.as_posix()
        payload = data
        if args.resize and rel.suffix.lower() in IMAGE_EXT:
            try:
                payload, (w0, h0), (w1, h1) = shrink(data, args.resize, args.quality)
                row.update(orig_width=w0, orig_height=h0, out_width=w1, out_height=h1)
                arcname = rel.with_suffix(".jpg").as_posix()
            except Exception as exc:  # unreadable/corrupt image: keep the original bytes and report it
                failed.append(f"{rel} ({exc})")
        row.update(out_path=arcname, out_bytes=len(payload), out_sha256=sha256_bytes(payload))
        rows.append(row)

        if zf is None or part_bytes + len(payload) > max_bytes:
            if zf is not None:
                zf.close()
            part += 1
            part_path = out / f"{args.name}_part{part:02d}.zip"
            zf = zipfile.ZipFile(part_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True)
            parts.append(part_path)
            part_bytes = 0
        zf.writestr(arcname, payload)
        part_bytes += len(payload)

        per_folder[rel.parent.as_posix() if len(rel.parts) > 1 else "(top level)"] += 1
        if i % 1000 == 0:
            print(f"  {i}/{len(files)}")
    zf.close()

    manifest_path = out / f"{args.name}_manifest.csv"
    cols = ["relative_path", "out_path", "orig_bytes", "orig_sha256", "orig_width", "orig_height",
            "out_bytes", "out_sha256", "out_width", "out_height"]
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(out / f"{args.name}_settings.txt", "w", encoding="utf-8") as fh:
        fh.write(f"source_folder={src.name}\nresize_shorter_side={args.resize}\njpeg_quality={args.quality}\n"
                 f"excluded={sorted(excluded)}\nfiles={len(rows)}\n")

    out_total = sum(r["out_bytes"] for r in rows)
    print("\nFiles per folder:")
    for k in sorted(per_folder):
        print(f"  {k:45s} {per_folder[k]:6d}")
    print(f"\nTotal: {len(rows)} files | original {orig_total / 1024**2:.1f} MB -> upload {out_total / 1024**2:.1f} MB")
    if failed:
        print(f"\nWARNING: {len(failed)} image(s) could not be read and were packed unchanged:")
        for f in failed[:20]:
            print("  ", f)
    print("\nCreated (upload ALL of these):")
    for pp in parts:
        print(f"  {pp.name:30s} {pp.stat().st_size / 1024**2:8.1f} MB")
    for extra in (manifest_path, out / f"{args.name}_settings.txt"):
        print(f"  {extra.name:30s} {extra.stat().st_size / 1024:8.1f} KB")
    print(f"\nOutput folder: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
