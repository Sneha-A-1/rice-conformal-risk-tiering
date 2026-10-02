"""Penultimate features of the ImageNet-pretrained (NOT fine-tuned) MobileNetV3-Large for every
Paddy Doctor image. Used by the poisoning experiment, where the cloud retrains a classifier head
on crowd-sourced uploads; a backbone never exposed to Paddy labels avoids label leakage."""
import argparse
import pathlib
import sys

import numpy as np
import timm
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from train import Normalizer, full_image_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = yaml.safe_load(open(ap.parse_args().config))
    torch.set_num_threads(cfg["train"]["threads"])
    proc = pathlib.Path(cfg["paths"]["processed"])
    arr = np.load(proc / "paddy_images.npy", mmap_mode="r")
    model = timm.create_model(cfg["train"]["arch"], pretrained=False, num_classes=0)
    sd = torch.load(pathlib.Path(cfg["paths"]["weights"]) / cfg["train"]["weights_file"], map_location="cpu")
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("classifier.")}, strict=True)
    model.eval()
    norm = Normalizer(cfg["image"]["mean"], cfg["image"]["std"])
    feats = []
    with torch.no_grad():
        for i in range(0, len(arr), 64):
            x = norm(full_image_batch(arr[i:i + 64], cfg["image"]["input_size"]))
            feats.append(model(x).half().numpy())
    out = pathlib.Path(cfg["paths"]["results"]) / "raw" / "imagenet_feats_paddy.npy"
    np.save(out, np.concatenate(feats))
    print(out)


if __name__ == "__main__":
    main()
