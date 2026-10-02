"""Fine-tune the edge classifier (MobileNetV3-Large, ImageNet-pretrained) on Paddy Doctor.

One run = one random seed. The 5 seeds give (i) repeated runs for statistics and
(ii) the 5-member deep ensemble used as the cloud-tier model.

Outputs in results/raw/models/seed{S}/:
  best.pt                 weights with the best validation macro-F1
  history.csv             per-epoch loss / accuracy / macro-F1 / time
  logits_{val,holdout,sethy}.npy   float32 logits (full-image inference)
  feats_{val,holdout,sethy}.npy    float16 penultimate features
  run.json                configuration, timings, environment
"""
import argparse
import json
import math
import pathlib
import platform
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score
from torchvision.transforms.v2 import functional as TF


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


class Normalizer:
    def __init__(self, mean, std):
        self.m = torch.tensor(mean).view(1, 3, 1, 1)
        self.s = torch.tensor(std).view(1, 3, 1, 1)

    def __call__(self, x_uint8):
        return (x_uint8.float() / 255.0 - self.m) / self.s


def augment_batch(imgs, size, scale, jitter, rng):
    """imgs: uint8 numpy (B,H,W,3) -> uint8 tensor (B,3,size,size)."""
    out = []
    for im in imgs:
        t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)
        H, W = t.shape[1:]
        for _ in range(10):  # RandomResizedCrop(scale, ratio=(3/4, 4/3))
            area = H * W * rng.uniform(*scale)
            logr = rng.uniform(math.log(3 / 4), math.log(4 / 3))
            w = int(round(math.sqrt(area * math.exp(logr))))
            h = int(round(math.sqrt(area / math.exp(logr))))
            if 0 < w <= W and 0 < h <= H:
                top, left = rng.integers(0, H - h + 1), rng.integers(0, W - w + 1)
                break
        else:
            h, w, top, left = H, W, 0, 0
        t = TF.resized_crop(t, int(top), int(left), h, w, [size, size], antialias=True)
        if rng.random() < 0.5:
            t = TF.horizontal_flip(t)
        if rng.random() < 0.5:
            t = TF.vertical_flip(t)
        t = TF.adjust_brightness(t, 1 + rng.uniform(-jitter, jitter))
        t = TF.adjust_contrast(t, 1 + rng.uniform(-jitter, jitter))
        out.append(t)
    return torch.stack(out)


def full_image_batch(imgs, short):
    """Resize whole image so its shorter side == short (no crop); uint8 tensor."""
    t = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2)
    H, W = t.shape[2:]
    s = short / min(H, W)
    return TF.resize(t, [round(H * s), round(W * s)], antialias=True)


@torch.no_grad()
def predict(model, arr, idx, norm, short, bs=64):
    model.eval()
    logits, feats = [], []
    for i in range(0, len(idx), bs):
        x = norm(full_image_batch(arr[idx[i:i + bs]], short))
        f = model.forward_head(model.forward_features(x), pre_logits=True)
        logits.append(model.get_classifier()(f).float().numpy())
        feats.append(f.half().numpy())
    return np.concatenate(logits), np.concatenate(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--limit", type=int, default=0, help="use only N training images (smoke tests)")
    ap.add_argument("--no-predict", action="store_true", help="skip final predictions (smoke tests)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    tc, ic = cfg["train"], cfg["image"]
    torch.set_num_threads(tc["threads"])
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    proc = pathlib.Path(cfg["paths"]["processed"])
    out = pathlib.Path(cfg["paths"]["results"]) / "raw" / "models" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(proc / "paddy_index.csv")
    classes = cfg["classes"]
    y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    arr = np.load(proc / "paddy_images.npy", mmap_mode="r")
    sarr = np.load(proc / "sethy_images.npy", mmap_mode="r")
    tr = np.nonzero(df.split.to_numpy() == "train")[0]
    va = np.nonzero(df.split.to_numpy() == "val")[0]
    ho = np.nonzero(df.split.to_numpy() == "holdout")[0]
    if args.limit:
        tr = rng.choice(tr, args.limit, replace=False)
    epochs = args.epochs or tc["epochs"]

    model = timm.create_model(tc["arch"], pretrained=False, num_classes=len(classes), drop_rate=tc["drop_rate"])
    sd = torch.load(pathlib.Path(cfg["paths"]["weights"]) / tc["weights_file"], map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {"classifier.weight", "classifier.bias"} and not unexpected, (missing, unexpected)
    norm = Normalizer(ic["mean"], ic["std"])

    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    steps_per_epoch = math.ceil(len(tr) / tc["batch_size"])
    total, warm = epochs * steps_per_epoch, tc["warmup_epochs"] * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return tc["lr"] * (step + 1) / warm
        return 0.5 * tc["lr"] * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    hist, best_f1, step, t0, start_ep, prev_seconds = [], -1.0, 0, time.time(), 0, 0.0
    ckpt = out / "last.pt"
    if ckpt.exists():  # resume after an interruption: restores weights, optimiser and every RNG state
        st = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        hist, best_f1, step, start_ep, prev_seconds = st["hist"], st["best_f1"], st["step"], st["epoch"], st["seconds"]
        rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"])
        random.setstate(st["py_rng"])
        print(f"resumed from epoch {start_ep}", flush=True)
    for ep in range(start_ep, epochs):
        model.train()
        perm = rng.permutation(tr)
        tl, tc_n, te = 0.0, 0, time.time()
        for b in range(steps_per_epoch):
            bi = np.sort(perm[b * tc["batch_size"]:(b + 1) * tc["batch_size"]])
            x = norm(augment_batch(arr[bi], ic["input_size"], tc["rrc_scale"], tc["color_jitter"], rng))
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = F.cross_entropy(model(x), torch.from_numpy(y[bi]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tl += loss.item() * len(bi)
            tc_n += len(bi)
            step += 1
        lv, _ = predict(model, arr, va, norm, ic["input_size"])
        pv = lv.argmax(1)
        f1 = f1_score(y[va], pv, average="macro")
        acc = float((pv == y[va]).mean())
        nll = float(F.cross_entropy(torch.from_numpy(lv), torch.from_numpy(y[va])).item())
        hist.append(dict(epoch=ep + 1, train_loss=tl / tc_n, val_acc=acc, val_macro_f1=f1, val_nll=nll,
                         epoch_seconds=time.time() - te))
        print(json.dumps(hist[-1]), flush=True)
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), out / "best.pt")
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "hist": hist, "best_f1": best_f1,
                    "step": step, "epoch": ep + 1, "seconds": prev_seconds + time.time() - t0,
                    "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "py_rng": random.getstate()}, out / "last.pt.tmp")
        (out / "last.pt.tmp").replace(ckpt)
    train_seconds = prev_seconds + time.time() - t0

    if args.no_predict:
        return
    model.load_state_dict(torch.load(out / "best.pt", map_location="cpu"))
    timings = {}
    for name, a, idx in [("val", arr, va), ("holdout", arr, ho), ("sethy", sarr, np.arange(len(sarr)))]:
        ts = time.time()
        lg, ft = predict(model, a, idx, norm, ic["input_size"])
        timings[name] = time.time() - ts
        np.save(out / f"logits_{name}.npy", lg)
        np.save(out / f"feats_{name}.npy", ft)
    np.save(out / "index_val.npy", va)
    np.save(out / "index_holdout.npy", ho)
    json.dump(dict(seed=args.seed, config=cfg, epochs=epochs, n_train=int(len(tr)), best_val_macro_f1=best_f1,
                   train_seconds=train_seconds, predict_seconds=timings, torch=torch.__version__,
                   timm=timm.__version__, python=platform.python_version(), machine=platform.machine(),
                   cpu_threads=torch.get_num_threads()), open(out / "run.json", "w"), indent=2)
    print("done", out)


if __name__ == "__main__":
    main()
