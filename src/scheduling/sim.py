"""Discrete-event simulation of the edge-fog-cloud continuum.

Workload: every captured image is classified on the edge device; its tier
(URGENT / STRATEGIC / ROUTINE) and size come from REAL test images (see
make_workload). URGENT and STRATEGIC images create an *analysis job* (cloud-model
ensemble inference) that must be placed on one of the fog nodes or the cloud.
ROUTINE images create only a small metadata record (batched upload).

Resources
  access link   device -> local fog, dedicated per device (bandwidth B_a, latency L_a)
  backhaul      local fog -> other fog (shared FIFO per fog pair direction, B_b, L_b)
  WAN uplink    fog -> cloud (shared FIFO per fog, B_w, L_w)
  fog node k    c_f identical servers, speed mu_f (relative to the reference CPU)
  cloud         c_c identical servers, speed mu_c

All times in seconds, sizes in bytes, energy in joules, cost in USD.
"""
import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

TIERS = ("URGENT", "STRATEGIC", "ROUTINE")


@dataclass
class Params:
    n_fog: int = 4
    devices_per_fog: int = 50
    fog_servers: int = 4
    fog_speed: float = 1.0
    cloud_servers: int = 32
    cloud_speed: float = 2.0
    B_access: float = 4e6 / 8        # bytes/s
    L_access: float = 0.040
    B_backhaul: float = 50e6 / 8
    L_backhaul: float = 0.010
    B_wan: float = 20e6 / 8
    L_wan: float = 0.080
    deadline: dict = field(default_factory=lambda: {"URGENT": 10.0, "STRATEGIC": 120.0, "ROUTINE": 3600.0})
    alert_bytes: int = 16_000         # prediction set + risk + Grad-CAM thumbnail
    record_bytes: int = 600           # signed routine metadata record
    P_tx: float = 1.0                 # W, device radio while transmitting
    P_fog: float = 15.0               # W per busy fog server
    P_cloud: float = 25.0             # W per busy cloud server (incl. overhead)
    cloud_price_per_server_hour: float = 0.04
    wan_price_per_gb: float = 0.0     # ingress usually free; kept as a parameter
    # objective weights (normalised terms)
    w_tier: dict = field(default_factory=lambda: {"URGENT": 10.0, "STRATEGIC": 1.0, "ROUTINE": 0.1})
    lam_energy: float = 0.05
    lam_cost: float = 0.05
    batch_window: float = 1.0         # for batch heuristics (Min-Min, Max-Min, GA, PSO)
    fog_reserved_for_urgent: int = 1  # RADS: servers kept free for URGENT jobs at each fog


@dataclass
class Job:
    jid: int
    t_capture: float
    device: int
    fog: int
    tier: str
    size: int
    work: float                       # reference-CPU seconds of the analysis
    risk: float                       # calibrated probability mass on high-severity classes
    t_edge: float                     # on-device inference time
    deadline: float
    orig_tier: str = ""               # tier assigned on the device (kept when a workload overrides `tier`)
    # filled by the simulator
    node: int = -1                    # 0..n_fog-1 fog, n_fog = cloud
    t_ready: float = 0.0              # job description available at the controller (local fog)
    t_arrive_node: float = 0.0
    t_start: float = 0.0
    t_finish: float = 0.0
    t_done: float = 0.0               # result back at the device
    e_tx: float = 0.0
    e_compute: float = 0.0
    cost: float = 0.0


class Link:
    """Non-preemptive transmission link with a queue ordered by the scheduler's link key
    (first-in-first-out for the baselines, risk priority for RADS)."""
    def __init__(self, bw, lat):
        self.bw, self.lat = bw, lat
        self.busy_until = 0.0
        self.queue = []               # heap of (key, seq, job, dest)

    def backlog_bytes(self, key=None):
        return sum(j.size for k, _, j, _ in self.queue if key is None or k <= key)


class Node:
    def __init__(self, servers, speed, power, price=0.0):
        self.c, self.speed, self.power, self.price = servers, speed, power, price
        self.server_free = [0.0] * servers
        self.queue = []               # heap of (key, seq, job)
        self.busy_time = 0.0

    def est_start(self, t, extra_backlog=0.0):
        """Optimistic start estimate: earliest server + queued work spread over servers."""
        backlog = sum(j.work / self.speed for _, _, j in self.queue) + extra_backlog
        return max(t, min(self.server_free)) + backlog / self.c


class Simulator:
    def __init__(self, params: Params, scheduler):
        self.p, self.sched = params, scheduler
        P = params
        self.nodes = [Node(P.fog_servers, P.fog_speed, P.P_fog) for _ in range(P.n_fog)]
        self.nodes.append(Node(P.cloud_servers, P.cloud_speed, P.P_cloud, P.cloud_price_per_server_hour / 3600))
        self.cloud = P.n_fog
        self.wan = [Link(P.B_wan, P.L_wan) for _ in range(P.n_fog)]
        self.backhaul = {(a, b): Link(P.B_backhaul, P.L_backhaul)
                         for a in range(P.n_fog) for b in range(P.n_fog) if a != b}
        self.events, self.seq = [], itertools.count()
        self.done = []
        self.records_bytes = 0

    # ---------- transfer model ----------
    def link_for(self, job, j):
        return self.wan[job.fog] if j == self.cloud else self.backhaul[(job.fog, j)]

    def transfer_estimate(self, job, j, t):
        """Estimated arrival time of job's image at node j if handed to the link at t
        (counts the bytes that would be transmitted before it under the link's ordering)."""
        if j == job.fog:
            return t
        link = self.link_for(job, j)
        ahead = link.backlog_bytes(self.sched.link_key(job))
        return max(t, link.busy_until) + (ahead + job.size) / link.bw + link.lat

    def transfer(self, job, j, t):
        if j == job.fog:
            self.push(t, "arrive", job)
            return
        link = self.link_for(job, j)
        heapq.heappush(link.queue, (self.sched.link_key(job), next(self.seq), job, j))
        self.start_link(link, t)

    def start_link(self, link, t):
        if link.busy_until > t + 1e-12 or not link.queue:
            return
        _, _, job, j = heapq.heappop(link.queue)
        done = t + job.size / link.bw
        link.busy_until = done
        self.push(done, "link_done", link)
        self.push(done + link.lat, "arrive", job)

    def est_finish(self, job, j, t):
        arr = self.transfer_estimate(job, j, t)
        return self.nodes[j].est_start(arr) + job.work / self.nodes[j].speed

    def job_energy_cost(self, job, j):
        node = self.nodes[j]
        ex = job.work / node.speed
        cost = node.price * ex + (self.p.wan_price_per_gb * job.size / 1e9 if j == self.cloud else 0.0)
        return node.power * ex, cost

    # ---------- event loop ----------
    def push(self, t, kind, payload):
        heapq.heappush(self.events, (t, next(self.seq), kind, payload))

    def run(self, jobs, routine_records):
        P = self.p
        for job in jobs:
            # device uploads the image straight to its local fog (dedicated access link)
            job.e_tx = P.P_tx * (job.size / P.B_access)
            job.t_ready = job.t_capture + job.t_edge + job.size / P.B_access + P.L_access
            self.push(job.t_ready, "ready", job)
        for (t, nbytes) in routine_records:
            self.records_bytes += nbytes
        # normalisation of the energy and cost terms: values of an average job executed in the cloud
        mean_w = float(np.mean([j.work for j in jobs])) if jobs else 1.0
        cl = self.nodes[self.cloud]
        self.e_ref = cl.power * mean_w / cl.speed
        self.c_ref = max(cl.price * mean_w / cl.speed, 1e-15)
        self.sched.attach(self)
        while self.events:
            t, _, kind, obj = heapq.heappop(self.events)
            if kind == "ready":
                self.sched.on_ready(obj, t)
            elif kind == "tick":
                self.sched.on_tick(t)
            elif kind == "link_done":
                self.start_link(obj, t)
            elif kind == "arrive":
                job = obj
                job.t_arrive_node = t
                node = self.nodes[job.node]
                heapq.heappush(node.queue, (self.sched.queue_key(job), next(self.seq), job))
                self.dispatch(job.node, t)
            elif kind == "finish":
                job, si = obj
                node = self.nodes[job.node]
                node.server_free[si] = t
                back = 0.0 if job.node == job.fog else (P.L_wan if job.node == self.cloud else P.L_backhaul)
                job.t_done = t + back + P.L_access
                self.done.append(job)
                self.dispatch(job.node, t)
        return self.done

    def assign(self, job, j, t):
        job.node = j
        e, c = self.job_energy_cost(job, j)
        job.e_compute, job.cost = e, c
        self.transfer(job, j, t)

    def dispatch(self, j, t):
        node = self.nodes[j]
        while node.queue:
            free = [i for i, f in enumerate(node.server_free) if f <= t + 1e-12]
            if not free:
                return
            key, _, job = node.queue[0]
            if not self.sched.may_start(job, j, len(free), t):
                return
            heapq.heappop(node.queue)
            si = free[0]
            ex = job.work / node.speed
            job.t_start, job.t_finish = t, t + ex
            node.server_free[si] = t + ex
            node.busy_time += ex
            self.push(t + ex, "finish", (job, si))


# =================== metrics ===================
URGENT_DEADLINE = 10.0             # overwritten from the configuration by run_experiments.py
def summarize(jobs, cloud_index=None):
    out = {}
    if not jobs:
        return out
    lat = np.array([j.t_done - j.t_capture for j in jobs])
    late = np.array([(j.t_done - j.t_capture) > j.deadline for j in jobs])
    tiers = np.array([j.tier for j in jobs])
    out["n_jobs"] = len(jobs)
    out["deadline_violation_rate"] = float(late.mean())
    out["mean_latency"] = float(lat.mean())
    out["p95_latency"] = float(np.percentile(lat, 95))
    for tr in ("URGENT", "STRATEGIC"):
        m = tiers == tr
        if m.any():
            out[f"{tr.lower()}_violation_rate"] = float(late[m].mean())
            out[f"{tr.lower()}_mean_latency"] = float(lat[m].mean())
            out[f"{tr.lower()}_p95_latency"] = float(np.percentile(lat[m], 95))
    # alert latency of images whose device-side tier was URGENT, against the URGENT deadline, whatever priority the
    # workload gave them (identical to the URGENT metrics for tiered workloads; differs for the upload-all workload)
    ot = np.array([j.orig_tier or j.tier for j in jobs])
    mU = ot == "URGENT"
    if mU.any():
        dU = min(j.deadline for j in jobs if j.tier == "URGENT") if (tiers == "URGENT").any() else None
        dU = dU if dU is not None else URGENT_DEADLINE
        out["alert_violation_rate"] = float((lat[mU] > dU).mean())
        out["alert_p95_latency"] = float(np.percentile(lat[mU], 95))
        out["alert_mean_latency"] = float(lat[mU].mean())
    risk = np.array([j.risk for j in jobs])
    out["risk_weighted_latency"] = float((risk * lat).sum() / max(risk.sum(), 1e-12))
    out["energy_J_per_job"] = float(np.mean([j.e_tx + j.e_compute for j in jobs]))
    out["cost_usd_per_1000_jobs"] = float(1000 * np.mean([j.cost for j in jobs]))
    t0 = min(j.t_capture for j in jobs)
    out["makespan"] = float(max(j.t_done for j in jobs) - t0)
    out["throughput_jobs_per_s"] = float(len(jobs) / out["makespan"])
    if cloud_index is not None:
        out["cloud_share"] = float(np.mean([j.node == cloud_index for j in jobs]))
    return out


def objective(jobs, p: Params, e_ref=1.0, c_ref=1.0):
    """Scalarised objective F (lower is better), used by GA/PSO fitness and reported for all."""
    if not jobs:
        return 0.0
    f = 0.0
    for j in jobs:
        lat = j.t_done - j.t_capture
        f += p.w_tier[j.tier] * max(0.0, lat - j.deadline) / j.deadline
        f += p.w_tier[j.tier] * 0.01 * lat / j.deadline
        f += p.lam_energy * (j.e_tx + j.e_compute) / e_ref + p.lam_cost * j.cost / c_ref
    return f / len(jobs)
