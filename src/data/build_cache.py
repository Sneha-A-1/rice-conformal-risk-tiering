"""Decode images once into uint8 memory-mapped arrays (fast CPU training).

Paddy Doctor: every image -> portrait H x W = 341 x 256 (the 4 landscape images are rotated).
Sethy:        shorter side 256, centre-cropped to 256 x 256.
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import yaml
from PIL import Image


def load_fixed(path, hw, portrait):
    im = Image.open(path).convert("RGB")
    if portrait and im.width > im.height:
        im = im.rotate(90, expand=True)
    H, W = hw
    if portrait:
        return np.asarray(im.resize((W, H), Image.BILINEAR))
    s = 256 / min(im.size)
    im = im.resize((max(256, round(im.width * s)), max(256, round(im.height * s))), Image.BILINEAR)
    l, t = (im.width - W) // 2, (im.height - H) // 2
    return np.asarray(im.crop((l, t, l + W, t + H)))


def build(index_csv, out_npy, hw, portrait):
    df = pd.read_csv(index_csv)
    arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.uint8, shape=(len(df), hw[0], hw[1], 3))
    for i, p in enumerate(df.path):
        arr[i] = load_fixed(p, hw, portrait)
    arr.flush()
    print(out_npy, arr.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    proc = pathlib.Path(cfg["paths"]["processed"])
    build(proc / "paddy_index.csv", proc / "paddy_images.npy", cfg["image"]["paddy_cache_hw"], True)
    build(proc / "sethy_index.csv", proc / "sethy_images.npy", cfg["image"]["sethy_cache_hw"], False)


if __name__ == "__main__":
    main()
