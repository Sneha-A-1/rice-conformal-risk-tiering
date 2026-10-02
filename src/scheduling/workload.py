"""Data-driven workload generation.

Images are sampled (with replacement) from a pool of REAL held-out images, each carrying
its edge-computed tier, calibrated high-severity risk, original file size (bytes) and true
class. Inter-arrival times are exponential (Poisson process) per scenario.
"""
import numpy as np

from .sim import Job


def make_workload(params, pool, rate, duration, rng, edge_times, work_times,
                  surge=None, surge_class_weight=None, offload_all=False):
    """pool: dict of numpy arrays with keys tier, size, risk, label (same length).
    rate: aggregate images/s across all devices. surge: (t_start, t_end, multiplier).
    surge_class_weight: optional array of sampling weights used during the surge
    (e.g. up-weighting high-severity classes to emulate an outbreak).
    offload_all: ablation 'no data prioritisation' - every image becomes a STRATEGIC job.
    Returns (jobs, routine_records)."""
    n_dev = params.n_fog * params.devices_per_fog
    t, jobs, records = 0.0, [], []
    jid = 0
    base_w = np.ones(len(pool["tier"])) / len(pool["tier"])
    while True:
        r = rate
        in_surge = surge is not None and surge[0] <= t < surge[1]
        if in_surge:
            r = rate * surge[2]
        t += rng.exponential(1.0 / r)
        if t >= duration:
            break
        w = base_w
        if in_surge and surge_class_weight is not None:
            w = surge_class_weight
        i = rng.choice(len(w), p=w)
        dev = int(rng.integers(0, n_dev))
        fog = dev // params.devices_per_fog
        tier = "STRATEGIC" if offload_all else pool["tier"][i]
        if tier == "ROUTINE":
            records.append((t, params.record_bytes))
            continue
        jobs.append(Job(jid=jid, t_capture=t, device=dev, fog=fog, tier=tier, size=int(pool["size"][i]),
                        work=float(rng.choice(work_times)), risk=float(pool["risk"][i]),
                        t_edge=float(rng.choice(edge_times)), deadline=params.deadline[tier],
                        orig_tier=pool["tier"][i]))
        jid += 1
    return jobs, records


def make_workload_upload_all(params, pool, rate, duration, rng, edge_times, work_times, surge=None,
                             surge_class_weight=None, rng_extra=None):
    """Paired 'no data prioritization' workload: the same captures, devices and images as make_workload with the
    same rng (the random stream of the tiered workload is reproduced exactly), but every capture, including those the
    device would keep as ROUTINE, is uploaded as a full image and becomes an equal-priority STRATEGIC job. Work and
    edge times of the extra (ROUTINE) captures are drawn from `rng_extra` so that the shared stream is unchanged."""
    rng_extra = rng_extra or np.random.default_rng(0)
    n_dev = params.n_fog * params.devices_per_fog
    t, jobs = 0.0, []
    jid = 0
    base_w = np.ones(len(pool["tier"])) / len(pool["tier"])
    while True:
        r = rate
        in_surge = surge is not None and surge[0] <= t < surge[1]
        if in_surge:
            r = rate * surge[2]
        t += rng.exponential(1.0 / r)
        if t >= duration:
            break
        w = surge_class_weight if (in_surge and surge_class_weight is not None) else base_w
        i = rng.choice(len(w), p=w)
        dev = int(rng.integers(0, n_dev))
        fog = dev // params.devices_per_fog
        src = rng_extra if pool["tier"][i] == "ROUTINE" else rng
        work = float(src.choice(work_times))
        t_edge = float(src.choice(edge_times))
        jobs.append(Job(jid=jid, t_capture=t, device=dev, fog=fog, tier="STRATEGIC", size=int(pool["size"][i]),
                        work=work, risk=float(pool["risk"][i]), t_edge=t_edge,
                        deadline=params.deadline["STRATEGIC"], orig_tier=pool["tier"][i]))
        jid += 1
    return jobs, []
