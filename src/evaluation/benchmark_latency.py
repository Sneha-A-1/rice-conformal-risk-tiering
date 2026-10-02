"""E9: measured on-device and cloud-model inference latency, Grad-CAM cost, tiering cost and
HMAC overhead on the reference CPU (2 threads). The latency samples are saved and re-used by
the scheduling simulation (edge time e_i and ensemble work w_i).

Run on an otherwise idle machine; the script records the load average before starting.
"""
import argparse
import json
import os
import pathlib
import platform
import sys
import time

import numpy as np
import pandas as pd
import timm
import torch
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from explain.gradcam import gradcam  # noqa: E402
from models.train import Normalizer, full_image_batch  # noqa: E402
from security.integrity import benchmark as hmac_benchmark  # noqa: E402


def load_model(cfg, path):
    m = timm.create_model(cfg["train"]["arch"], pretrained=False, num_classes=len(cfg["classes"]))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    torch.set_num_threads(cfg["train"]["threads"])
    proc = pathlib.Path(cfg["paths"]["processed"])
    res = pathlib.Path(cfg["paths"]["results"]) / "raw"
    arr = np.load(proc / "paddy_images.npy", mmap_mode="r")
    ho = np.load(res / "models" / "seed0" / "index_holdout.npy")
    norm = Normalizer(cfg["image"]["mean"], cfg["image"]["std"])
    seeds = [s for s in cfg["train"]["seeds"] if (res / "models" / f"seed{s}" / "best.pt").exists()]
    models = [load_model(cfg, res / "models" / f"seed{s}" / "best.pt") for s in seeds]
    # the cloud ensemble has 5 members of identical architecture; latency does not depend on the
    # weights, so missing members (if run before all seeds are trained) are copies of seed 0
    import copy
    while len(models) < 5:
        models.append(copy.deepcopy(models[0]))
    rng = np.random.default_rng(0)
    pick = rng.choice(ho, args.n, replace=False)
    load_before = os.getloadavg()

    def prep(i):
        return norm(full_image_batch(arr[i:i + 1], cfg["image"]["input_size"]))

    with torch.no_grad():                          # warm-up
        for i in pick[:10]:
            for m in models:
                m(prep(i))
    edge, ens, cam, prep_t = [], [], [], []
    with torch.no_grad():
        for i in pick:
            t0 = time.perf_counter()
            x = prep(i)
            t1 = time.perf_counter()
            models[0](x)
            t2 = time.perf_counter()
            prep_t.append(t1 - t0)
            edge.append(t2 - t1)
        for i in pick[: args.n // 3]:
            x = prep(i)
            t0 = time.perf_counter()
            for m in models:
                m(x)
            ens.append(time.perf_counter() - t0)
    for i in pick[: args.n // 3]:
        x = prep(i)
        t0 = time.perf_counter()
        gradcam(models[0], x)
        cam.append(time.perf_counter() - t0)
    # tiering cost (softmax + conformal set + tier) in numpy
    K = len(cfg["classes"])
    z = rng.normal(size=(10000, K)).astype(np.float32)
    q = np.full(K, 0.9)
    t0 = time.perf_counter()
    for row in z:
        p = np.exp(row - row.max())
        p /= p.sum()
        c = (1 - p) <= q
        _ = c[[0, 2, 3, 9]].any()
    tier_us = 1e6 * (time.perf_counter() - t0) / len(z)

    man = pd.read_csv(pathlib.Path(cfg["paths"]["data_root"]) / "raw" / "paddy" / "paddy_manifest.csv")
    img_bytes = man[man.orig_width.notna()].orig_bytes.to_numpy()
    out = {
        "machine": platform.machine(), "processor": platform.processor(), "cpu_threads": torch.get_num_threads(),
        "load_average_before": load_before, "torch": torch.__version__, "n_models_in_ensemble": len(models), "n_trained_seeds_available": len(seeds),
        "input_hw": list(prep(pick[0]).shape[2:]),
        "edge_forward_ms": {"mean": 1e3 * np.mean(edge), "p50": 1e3 * np.median(edge), "p95": 1e3 * np.percentile(edge, 95)},
        "preprocess_ms": {"mean": 1e3 * np.mean(prep_t)},
        "ensemble_forward_ms": {"mean": 1e3 * np.mean(ens), "p50": 1e3 * np.median(ens), "p95": 1e3 * np.percentile(ens, 95)},
        "gradcam_ms": {"mean": 1e3 * np.mean(cam), "p50": 1e3 * np.median(cam), "p95": 1e3 * np.percentile(cam, 95)},
        "tiering_us": tier_us,
        "edge_model_params_M": sum(p.numel() for p in models[0].parameters()) / 1e6,
        "edge_model_file_MB": (res / "models" / "seed0" / "best.pt").stat().st_size / 1e6,
        "original_image_bytes": {"mean": float(img_bytes.mean()), "median": float(np.median(img_bytes)),
                                 "p95": float(np.percentile(img_bytes, 95))},
        "hmac": hmac_benchmark(sizes=(600, 16000, int(np.median(img_bytes)))),
    }
    np.save(res / "latency_edge_s.npy", np.array(edge))
    np.save(res / "latency_ensemble_s.npy", np.array(ens))
    json.dump(out, open(res / "latency_benchmark.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
