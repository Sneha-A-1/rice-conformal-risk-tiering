# Data

The images are **not redistributed** with this repository (Paddy Doctor competition rules, §7B).
Download them yourself and place them as follows (paths are relative to the repository's parent folder,
configurable in `configs/default.yaml → paths.data_root`).

```
../data/
├── paddy/                      # Paddy Doctor (Kaggle "paddy-disease-classification")
│   ├── train.csv               # image_id, label, variety, age  (10,407 rows)
│   └── train_images/<class>/<image_id>.jpg   (10 class folders)
├── sethy/                      # Rice Leaf Disease Image Samples v1 (Mendeley, DOI 10.17632/fwcj7stb8r.1)
│   └── Rice Leaf Disease Images/{Bacterialblight,Blast,Brownspot,Tungro}/*.jpg   (5,932 images)
└── raw/
    ├── paddy/paddy_manifest.csv   # optional: SHA-256 manifest of the originals (see below)
    └── sethy/sethy_manifest.csv
```

## 1. Paddy Doctor
1. Create a Kaggle account and accept the rules of <https://www.kaggle.com/competitions/paddy-disease-classification>
   (data use: competition and academic research; CC BY 4.0).
2. Download the data (1.09 GB). Only `train.csv` and `train_images/` are used; `test_images/` has no public labels.

## 2. Rice Leaf Disease Image Samples ("Sethy")
Download version 1 from <https://data.mendeley.com/datasets/fwcj7stb8r/1> ("Download All") and unzip
(the archive may contain a nested zip).

## 3. Resizing used in the paper (exact reproduction)
The experiments used images whose shorter side was resized to 256 px (Lanczos, JPEG quality 92):

```bash
python scripts/prepare_upload.py <paddy-folder> paddy --resize 256 --exclude test_images sample_submission.csv --out ../data/raw/paddy
python scripts/prepare_upload.py <sethy-folder> sethy --resize 256 --out ../data/raw/sethy
# unzip ../data/raw/paddy/paddy_part*.zip into ../data/paddy and the Sethy parts into ../data/sethy
```

The manifests (`*_manifest.csv`) list the SHA-256 of every original and resized file. The manifests used
for the paper are in `data/manifests/`; `python scripts/verify_data.py` compares your files against them.

## 4. Derived files (created by the pipeline)
`data/processed/`: `paddy_index.csv`, `sethy_index.csv` (near-duplicate groups, splits), `audit_summary.json`,
perceptual hashes, and the decoded image caches (`*_images.npy`, not tracked).
