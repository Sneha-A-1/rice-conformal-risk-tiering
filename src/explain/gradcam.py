"""E6: Grad-CAM explanations for the edge model and their quantitative evaluation.

Faithfulness  deletion / insertion curves (Petsiuk et al., 2018): pixels are removed (replaced by a
              blurred copy) or inserted in decreasing order of saliency; area under the
              class-probability curve. Compared with random and centre-prior saliency maps.
Stability     agreement between the map of an image and the (back-transformed) map of a
              horizontally flipped or brightness-shifted copy (Spearman rho, top-20 % IoU).
Cost          extra latency of producing the map on the edge device.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import yaml
from scipy.special import softmax
from scipy.stats import spearmanr
from torchvision.transforms.v2 import functional as TF

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from models.train import Normalizer, full_image_batch  # noqa: E402


def gradcam(model, x, target=None):
    """x: normalised (B,3,H,W). Returns (cam (B,H,W) in [0,1], logits)."""
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    if target is None:
        target = logits.argmax(1)
    model.zero_grad()
    logits.gather(1, target[:, None]).sum().backward()
    w = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[:, 0]
    cam = cam - cam.amin(dim=(1, 2), keepdim=True)
    cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    return cam.detach(), logits.detach()


@torch.no_grad()
def curve(model, x, blurred, sal, target, steps=20, mode="deletion"):
    """x, blurred: (3,H,W); sal: (H,W). Returns class-probability curve of length steps+1."""
    H, W = sal.shape
    order = torch.argsort(sal.flatten(), descending=True)
    n = H * W
    batch = []
    for k in range(steps + 1):
        m = torch.zeros(n)
        m[order[: int(round(n * k / steps))]] = 1
        m = m.view(1, H, W)
        batch.append(x * (1 - m) + blurred * m if mode == "deletion" else blurred * (1 - m) + x * m)
    p = softmax(model(torch.stack(batch)).numpy(), axis=1)[:, target]
    return p


def auc(c):
    return float(np.trapezoid(c, dx=1.0 / (len(c) - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per_class", type=int, default=20)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.set_num_threads(cfg["train"]["threads"])
    proc = pathlib.Path(cfg["paths"]["processed"])
    res = pathlib.Path(cfg["paths"]["results"]) / "raw"
    classes = cfg["classes"]
    K = len(classes)
    df = pd.read_csv(proc / "paddy_index.csv")
    y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    arr = np.load(proc / "paddy_images.npy", mmap_mode="r")
    ho = np.load(res / "models" / f"seed{args.seed}" / "index_holdout.npy")

    model = timm.create_model(cfg["train"]["arch"], pretrained=False, num_classes=K)
    model.load_state_dict(torch.load(res / "models" / f"seed{args.seed}" / "best.pt", map_location="cpu"))
    model.eval()
    norm = Normalizer(cfg["image"]["mean"], cfg["image"]["std"])

    rng = np.random.default_rng(cfg["seed"])
    pick = np.concatenate([rng.choice(ho[y[ho] == k], min(args.per_class, (y[ho] == k).sum()), replace=False) for k in range(K)])
    rows, examples = [], []
    for i in pick:
        x = norm(full_image_batch(arr[i:i + 1], cfg["image"]["input_size"]))
        xr = x.clone().requires_grad_(False)
        t0 = time.perf_counter()
        with torch.no_grad():
            model(xr)
        t_fwd = time.perf_counter() - t0
        t0 = time.perf_counter()
        cam, logits = gradcam(model, xr)
        t_cam = time.perf_counter() - t0
        target = int(logits.argmax(1))
        blurred = TF.gaussian_blur(x[0], kernel_size=[31, 31], sigma=[10.0, 10.0])
        H, W = cam.shape[1:]
        rnd = torch.from_numpy(rng.random((H, W)).astype(np.float32))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        centre = 1 - (yy ** 2 + xx ** 2).sqrt() / np.sqrt(2)
        row = dict(index=int(i), true=classes[y[i]], pred=classes[target], correct=bool(target == y[i]),
                   latency_forward_s=t_fwd, latency_gradcam_s=t_cam)
        for name, sal in [("gradcam", cam[0]), ("random", rnd), ("centre", centre)]:
            row[f"deletion_auc_{name}"] = auc(curve(model, x[0], blurred, sal, target, mode="deletion"))
            row[f"insertion_auc_{name}"] = auc(curve(model, x[0], blurred, sal, target, mode="insertion"))
        # stability: horizontal flip and brightness +20 %
        camf, _ = gradcam(model, torch.flip(x, dims=[3]), torch.tensor([target]))
        camf = torch.flip(camf, dims=[2])
        xb = norm(TF.adjust_brightness(full_image_batch(arr[i:i + 1], cfg["image"]["input_size"]), 1.2))
        camb, _ = gradcam(model, xb, torch.tensor([target]))
        for name, c2 in [("flip", camf), ("brightness", camb)]:
            a, b = cam[0].flatten().numpy(), c2[0].flatten().numpy()
            row[f"stability_spearman_{name}"] = float(spearmanr(a, b).statistic)
            ta, tb = a >= np.quantile(a, 0.8), b >= np.quantile(b, 0.8)
            row[f"stability_iou20_{name}"] = float((ta & tb).sum() / (ta | tb).sum())
        rows.append(row)
        if len(examples) < 40:
            examples.append((int(i), cam[0].numpy().astype(np.float16)))
    out = pd.DataFrame(rows)
    out.to_csv(res / "xai_gradcam_eval.csv", index=False)
    np.savez_compressed(res / "xai_gradcam_examples.npz", idx=np.array([e[0] for e in examples]),
                        cams=np.stack([e[1] for e in examples]))
    print(out.drop(columns=["index", "true", "pred"]).mean(numeric_only=True).round(4).to_string())


if __name__ == "__main__":
    main()
