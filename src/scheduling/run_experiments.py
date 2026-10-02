"""E8: scheduling experiments driven by real tiers, image sizes and measured latencies.

Experiments
  load      aggregate capture rate sweep, all schedulers
  outbreak  surge scenario (rate x3 for 600 s, 70 % high-severity images), schedulers + ablations
  sens      sensitivity of RADS vs best baselines to fog speed and WAN bandwidth (outbreak)
Each configuration is replicated R times with common random numbers across schedulers.
"""
import argparse
import copy
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.evaluation.stats import paired_tests  # noqa: E402
from src.scheduling.schedulers import (GA, PSO, RADS, EDFLeastLoaded, EDFLeastLoadedPrio,  # noqa: E402
                                       FCFSLocal, MaxMin, MinMin, RoundRobin)
from src.scheduling.sim import Params, Simulator, objective, summarize  # noqa: E402
from src.scheduling.workload import make_workload, make_workload_upload_all  # noqa: E402
from src.trust.calibration import fit_temperature  # noqa: E402
from src.trust.conformal import calibrate, lac_scores, predict_sets  # noqa: E402


class CloudOnly(FCFSLocal):
    name = "Cloud-only"

    def on_ready(self, job, t):
        self.sim.assign(job, self.sim.cloud, t)

    def queue_key(self, job):
        return job.t_capture + job.deadline


def build_pool(cfg, seed=0, alpha=None):
    """Tiers of the test half of the hold-out partition from edge seed `seed`."""
    proc = pathlib.Path(cfg["paths"]["processed"])
    res = pathlib.Path(cfg["paths"]["results"]) / "raw"
    classes = cfg["classes"]
    cidx = {c: i for i, c in enumerate(classes)}
    H = [cidx[c] for c in cfg["risk"]["high_severity"]]
    alpha = alpha or cfg["risk"]["alpha"]
    df = pd.read_csv(proc / "paddy_index.csv")
    y = df.label.map(cidx).to_numpy()
    md = res / "models" / f"seed{seed}"
    va, ho = np.load(md / "index_val.npy"), np.load(md / "index_holdout.npy")
    T = fit_temperature(np.load(md / "logits_val.npy"), y[va])
    p = softmax(np.load(md / "logits_holdout.npy") / T, axis=1)
    rng = np.random.default_rng(cfg["seed"] + 99)
    g = df.group.to_numpy()[ho]
    ug = np.unique(g)
    calg = set(rng.choice(ug, len(ug) // 2, replace=False))
    is_cal = np.array([x in calg for x in g])
    q = calibrate(lac_scores(p[is_cal]), y[ho][is_cal], alpha, "class", len(classes))
    sets = predict_sets(lac_scores(p[~is_cal]), q)
    urgent = sets[:, H].any(1)
    size = sets.sum(1)
    tier = np.where(urgent, "URGENT", np.where(size != 1, "STRATEGIC", "ROUTINE"))
    man = pd.read_csv(pathlib.Path(cfg["paths"]["data_root"]) / "raw" / "paddy" / "paddy_manifest.csv")
    lut = dict(zip(man.out_path, man.orig_bytes))
    paths = df.path.to_numpy()[ho][~is_cal]
    base = pathlib.Path(cfg["paths"]["data_root"]) / "paddy"
    sizes = np.array([lut[str(pathlib.Path(x).relative_to(base).as_posix())] for x in paths])
    labels = y[ho][~is_cal]
    return {"tier": tier, "risk": p[~is_cal][:, H].sum(1), "size": sizes, "label": labels,
            "isH": np.isin(labels, H)}


def params_from_cfg(cfg):
    sc = cfg["scheduling"]
    p = Params()
    for k, v in sc["params"].items():
        if k.startswith("B_"):
            v = float(v) * 1e6 / 8          # Mbit/s -> bytes/s
        setattr(p, k, v)
    p.deadline = dict(sc["deadlines"])
    return p


def run_one(params, sched, pool, rate, duration, seed, edge_t, work_t, surge=None, surge_w=None, offload_all=False,
            paired_upload_all=False):
    rng = np.random.default_rng(seed)
    if paired_upload_all:
        jobs, rec = make_workload_upload_all(params, pool, rate, duration, rng, edge_t, work_t, surge, surge_w,
                                             rng_extra=np.random.default_rng(seed + 700_000))
    else:
        jobs, rec = make_workload(params, pool, rate, duration, rng, edge_t, work_t, surge, surge_w, offload_all)
    t0 = time.perf_counter()
    sim = Simulator(params, sched)
    done = sim.run(jobs, rec)
    wall = time.perf_counter() - t0
    s = summarize(done, cloud_index=params.n_fog)
    s["objective"] = objective(done, params, sim.e_ref, sim.c_ref)
    s["records_uploaded"] = len(rec)
    s["bytes_uploaded_MB"] = (sum(j.size for j in done) + sum(params.alert_bytes for j in done if j.tier == "URGENT")
                              + sum(b for _, b in rec)) / 1e6
    s["scheduler_wall_s"] = wall
    return s


def schedulers(cfg):
    m = cfg["scheduling"]["metaheuristic"]
    return {
        "FCFS-Local": lambda: FCFSLocal(),
        "Round-Robin": lambda: RoundRobin(),
        "EDF-LL": lambda: EDFLeastLoaded(),
        "Min-Min": lambda: MinMin(),
        "Max-Min": lambda: MaxMin(),
        "GA": lambda: GA(pop=m["pop"], gens=m["iters"]),
        "PSO": lambda: PSO(particles=m["pop"], iters=m["iters"]),
        "RADS": lambda: RADS(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--exp", choices=["load", "outbreak", "uploadall", "sens", "conv", "all"], default="all")
    ap.add_argument("--reps", type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    sc = cfg["scheduling"]
    R = args.reps or sc["reps"]
    res = pathlib.Path(cfg["paths"]["results"]) / "raw"
    edge_t = np.load(res / "latency_edge_s.npy")
    work_t = np.load(res / "latency_ensemble_s.npy")
    pool = build_pool(cfg)
    json.dump({t: int((pool["tier"] == t).sum()) for t in ("URGENT", "STRATEGIC", "ROUTINE")} |
              {"n": int(len(pool["tier"])), "mean_size": float(pool["size"].mean())},
              open(res / "sched_pool_summary.json", "w"), indent=1)
    w_surge = np.where(pool["isH"], 0.7 / pool["isH"].sum(), 0.3 / (~pool["isH"]).sum())
    base = params_from_cfg(cfg)
    import src.scheduling.sim as _sim
    _sim.URGENT_DEADLINE = float(sc["deadlines"]["URGENT"])
    S = schedulers(cfg)
    dur = sc["duration_s"]

    if args.exp in ("load", "all"):
        rows = []
        S_load = {k: v for k, v in S.items() if k not in ("GA", "PSO")}
        for rate in sc["rates"]:
            for r in range(args.reps or sc["load_reps"]):
                for name, mk in S_load.items():
                    s = run_one(base, mk(), pool, rate, dur, 10_000 + r, edge_t, work_t)
                    rows.append({"rate": rate, "rep": r, "scheduler": name, **s})
            print("load", rate, flush=True)
            pd.DataFrame(rows).to_csv(res / "sched_load.csv", index=False)

    if args.exp in ("outbreak", "all"):
        rows, tests = [], {}
        surge = (sc["surge"]["start"], sc["surge"]["start"] + sc["surge"]["length"], sc["surge"]["factor"])
        variants = dict(S)
        variants.update({
            "EDF-LL + priority links": lambda: EDFLeastLoadedPrio(),
            "RADS w/o tiers": lambda: RADS(use_tiers=False),
            "RADS w/o priority links": lambda: RADS(use_priority_links=False),
            "RADS w/o reservation": lambda: RADS(use_reservation=False),
            "RADS + energy/cost terms": lambda: RADS(use_cost_terms=True),
            "Cloud-only": lambda: CloudOnly(),
        })
        for bw in sc["outbreak_bw_mbps"]:
            p = copy.deepcopy(base)
            p.B_wan = bw * 1e6 / 8
            for r in range(R):
                for name, mk in variants.items():
                    s_ = run_one(p, mk(), pool, sc["base_rate"], dur, 20_000 + r, edge_t, work_t, surge, w_surge)
                    rows.append({"B_wan_mbps": bw, "rep": r, "scheduler": name, "workload": "tiered", **s_})
                for name in ("RADS", "EDF-LL"):
                    s_ = run_one(p, S[name](), pool, sc["base_rate"], dur, 20_000 + r, edge_t, work_t, surge, w_surge,
                                 offload_all=True)
                    rows.append({"B_wan_mbps": bw, "rep": r, "scheduler": name, "workload": "upload-all", **s_})
                print("outbreak bw", bw, "rep", r, time.strftime("%H:%M:%S"), flush=True)
                pd.DataFrame(rows).to_csv(res / "sched_outbreak.csv", index=False)
            df = pd.DataFrame(rows)
            tiered = df[(df.workload == "tiered") & (df.B_wan_mbps == bw)]
            tests[str(bw)] = {}
            for metric in ("urgent_violation_rate", "urgent_p95_latency", "deadline_violation_rate", "mean_latency",
                           "energy_J_per_job", "cost_usd_per_1000_jobs", "objective", "risk_weighted_latency"):
                prop = tiered[tiered.scheduler == "RADS"].sort_values("rep")[metric].to_numpy()
                bl = {b: tiered[tiered.scheduler == b].sort_values("rep")[metric].to_numpy()
                      for b in S if b != "RADS"}
                tests[str(bw)][metric] = paired_tests(prop, bl)
            json.dump(tests, open(res / "sched_outbreak_tests.json", "w"), indent=1)

    if args.exp in ("uploadall", "all"):
        # paired no-prioritization workload: identical captures to the tiered outbreak replications (same seeds),
        # every image uploaded as an equal-priority job; alert latency measured for images the device tiered URGENT
        rows = []
        surge = (sc["surge"]["start"], sc["surge"]["start"] + sc["surge"]["length"], sc["surge"]["factor"])
        for bw in sc["outbreak_bw_mbps"]:
            p = copy.deepcopy(base)
            p.B_wan = bw * 1e6 / 8
            for r in range(R):
                for name in ("RADS", "EDF-LL"):
                    for paired in (False, True):
                        s_ = run_one(p, S[name](), pool, sc["base_rate"], dur, 20_000 + r, edge_t, work_t, surge, w_surge,
                                     paired_upload_all=paired)
                        rows.append({"B_wan_mbps": bw, "rep": r, "scheduler": name,
                                     "workload": "upload-all" if paired else "tiered", **s_})
                print("uploadall bw", bw, "rep", r, time.strftime("%H:%M:%S"), flush=True)
                pd.DataFrame(rows).to_csv(res / "sched_uploadall.csv", index=False)

    if args.exp in ("conv", "all"):
        convergence(cfg, base, pool, edge_t, work_t, w_surge, res)

    if args.exp in ("sens", "all"):
        rows = []
        surge = (sc["surge"]["start"], sc["surge"]["start"] + sc["surge"]["length"], sc["surge"]["factor"])
        for fs in sc["sensitivity"]["fog_speed"]:
            for bw in sc["sensitivity"]["B_wan_mbps"]:
                p = copy.deepcopy(base)
                p.fog_speed, p.B_wan = fs, bw * 1e6 / 8
                for r in range(sc["sensitivity"]["reps"]):
                    for name, mk in (("RADS", S["RADS"]), ("EDF-LL", S["EDF-LL"]),
                                     ("EDF-LL + priority links", lambda: EDFLeastLoadedPrio()),
                                     ("FCFS-Local", S["FCFS-Local"])):
                        s = run_one(p, mk(), pool, sc["base_rate"], dur, 30_000 + r, edge_t, work_t, surge, w_surge)
                        rows.append({"fog_speed": fs, "B_wan_mbps": bw, "rep": r, "scheduler": name, **s})
            print("sens fog_speed", fs, flush=True)
        pd.DataFrame(rows).to_csv(res / "sched_sensitivity.csv", index=False)


def convergence(cfg, base, pool, edge_t, work_t, w_surge, res):
    base = copy.deepcopy(base)
    base.B_wan = cfg["scheduling"]["outbreak_bw_mbps"][-1] * 1e6 / 8
    """Best-so-far fitness per generation/iteration of GA and PSO on every batch of one outbreak run."""
    sc = cfg["scheduling"]
    m = sc["metaheuristic"]
    surge = (sc["surge"]["start"], sc["surge"]["start"] + sc["surge"]["length"], sc["surge"]["factor"])
    out = {}
    for name, sch in (("GA", GA(pop=m["pop"], gens=m["iters"], record=True)),
                      ("PSO", PSO(particles=m["pop"], iters=m["iters"], record=True))):
        run_one(base, sch, pool, sc["base_rate"], sc["duration_s"], 20_000, edge_t, work_t, surge, w_surge)
        H = np.array([h for h in sch.history if len(h) == m["iters"] + 1])
        out[name] = H
    np.savez_compressed(res / "sched_convergence.npz", **out)


if __name__ == "__main__":
    main()
