"""Scheduling policies for analysis jobs on the fog/cloud continuum.

Online: FCFS-Local, Round-Robin, EDF + least-loaded (EDF-LL), RADS (proposed).
Batch (decisions every `batch_window` s): Min-Min, Max-Min, GA, PSO.
"""
import math

import numpy as np

TIER_RANK = {"URGENT": 0, "STRATEGIC": 1, "ROUTINE": 2}


class Base:
    name = "base"

    def attach(self, sim):
        self.sim = sim
        self.p = sim.p
        self.N = len(sim.nodes)

    def on_ready(self, job, t):
        raise NotImplementedError

    def on_tick(self, t):
        pass

    def queue_key(self, job):
        return job.t_ready

    def link_key(self, job):
        """Ordering of image transmissions on shared links: first-in-first-out by default."""
        return (job.t_ready, job.jid)

    def may_start(self, job, j, n_free, t):
        return True


class FCFSLocal(Base):
    name = "FCFS-Local"

    def on_ready(self, job, t):
        self.sim.assign(job, job.fog, t)


class RoundRobin(Base):
    name = "Round-Robin"

    def attach(self, sim):
        super().attach(sim)
        self.k = 0

    def on_ready(self, job, t):
        self.sim.assign(job, self.k, t)
        self.k = (self.k + 1) % self.N


class EDFLeastLoaded(Base):
    name = "EDF-LL"

    def on_ready(self, job, t):
        j = min(range(self.N), key=lambda n: self.sim.est_finish(job, n, t))
        self.sim.assign(job, j, t)

    def queue_key(self, job):
        return job.t_capture + job.deadline


class EDFLeastLoadedPrio(EDFLeastLoaded):
    """EDF-LL given the same risk-priority links as RADS (isolates the link mechanism)."""
    name = "EDF-LL + priority links"

    def link_key(self, job):
        return (TIER_RANK[job.tier], -job.risk, job.t_ready, job.jid)


# ---------------- batch heuristics ----------------
class BatchBase(Base):
    def attach(self, sim):
        super().attach(sim)
        self.buf = []
        self.next_tick = None

    def on_ready(self, job, t):
        self.buf.append(job)
        if self.next_tick is None:
            self.next_tick = t + self.p.batch_window
            self.sim.push(self.next_tick, "tick", None)

    def on_tick(self, t):
        batch, self.buf, self.next_tick = self.buf, [], None
        if batch:
            self._prep(batch, t)
            for job, j in zip(batch, self.decide(batch, t)):
                self.sim.assign(job, j, t)

    def _prep(self, batch, t):
        """Cache every quantity that is constant while one batch is being decided (the simulator state does not
        change inside `decide`): arrival estimates, server-free lists, queued work, energy and cost. The cached
        values are computed with exactly the same floating-point operations as before, so decisions are unchanged;
        this only removes repeated O(queue length) sums from the inner loop of GA/PSO/Min-Min."""
        sim = self.sim
        self._A = [[sim.transfer_estimate(job, j, t) for j in range(self.N)] for job in batch]
        self._qwork = [sum(q.work / node.speed for _, _, q in node.queue) for node in sim.nodes]
        self._minfree = [min(node.server_free) for node in sim.nodes]
        self._free = []
        for j, node in enumerate(sim.nodes):
            free = sorted(max(t, f) for f in node.server_free)
            backlog = self._qwork[j] / node.c
            self._free.append([f + backlog for f in free])
        self._order = sorted(range(len(batch)), key=lambda i: batch[i].t_capture + batch[i].deadline)
        self._ec = [[sim.job_energy_cost(job, j) for j in range(self.N)] for job in batch]

    def queue_key(self, job):
        return job.t_capture + job.deadline

    # helpers -----------------------------------------------------------
    def ect_matrix(self, batch, t):
        return np.array([[self.sim.est_finish(job, j, t) for j in range(self.N)] for job in batch])

    def estimate(self, batch, assign, t):
        """Fast estimate of finish times for a candidate assignment (list scheduling, EDF order).
        Uses the per-batch caches built by `_prep`."""
        sim = self.sim
        fin = np.empty(len(batch))
        free = [list(fl) for fl in self._free]
        A = self._A
        for i in self._order:                       # EDF order; per node this is the same order as before
            j = assign[i]
            fl = free[j]
            s = max(A[i][j], fl[0])
            f = s + batch[i].work / sim.nodes[j].speed
            fl[0] = f
            fl.sort()
            fin[i] = f
        return fin

    def fitness(self, batch, assign, t):
        p = self.p
        fin = self.estimate(batch, assign, t)
        f = 0.0
        for i, job in enumerate(batch):
            lat = fin[i] - job.t_capture
            w = p.w_tier[job.tier]
            f += w * max(0.0, lat - job.deadline) / job.deadline + w * 0.01 * lat / job.deadline
            e, c = self._ec[i][assign[i]]
            f += p.lam_energy * e / self.e_ref + p.lam_cost * c / self.c_ref
        return f

    def set_refs(self):
        self.e_ref, self.c_ref = self.sim.e_ref, self.sim.c_ref


class MinMin(BatchBase):
    name = "Min-Min"
    largest_first = False

    def decide(self, batch, t):
        sim = self.sim
        extra = np.zeros(self.N)
        left, out = list(range(len(batch))), [None] * len(batch)
        while left:
            best = None
            for i in left:
                ects = [max(self._A[i][j], self._minfree[j]) + (self._qwork[j] + extra[j]) / sim.nodes[j].c
                        + batch[i].work / sim.nodes[j].speed for j in range(self.N)]
                j = int(np.argmin(ects))
                cand = (ects[j], i, j)
                if best is None or (cand[0] > best[0] if self.largest_first else cand[0] < best[0]):
                    best = cand
            _, i, j = best
            out[i] = j
            extra[j] += batch[i].work / sim.nodes[j].speed
            left.remove(i)
        return out


class MaxMin(MinMin):
    name = "Max-Min"
    largest_first = True


class GA(BatchBase):
    name = "GA"

    def __init__(self, pop=30, gens=40, pc=0.9, seed=0, record=False):
        self.pop, self.gens, self.pc = pop, gens, pc
        self.rng = np.random.default_rng(seed)
        self.record, self.history = record, []

    def attach(self, sim):
        super().attach(sim)
        self.set_refs()

    def decide(self, batch, t):
        n, N, rng = len(batch), self.N, self.rng
        P = rng.integers(0, N, (self.pop, n))
        P[0] = [b.fog for b in batch]                       # seed with local placement
        fit = np.array([self.fitness(batch, ind, t) for ind in P])
        pm = 1.0 / max(n, 1)
        hist = [fit.min() / n]
        for _ in range(self.gens):
            new = [P[np.argmin(fit)].copy()]                 # elitism
            while len(new) < self.pop:
                a = P[min(rng.integers(0, self.pop, 2), key=lambda k: fit[k])]
                b = P[min(rng.integers(0, self.pop, 2), key=lambda k: fit[k])]
                child = np.where(rng.random(n) < 0.5, a, b) if rng.random() < self.pc else a.copy()
                mut = rng.random(n) < pm
                child[mut] = rng.integers(0, N, mut.sum())
                new.append(child)
            P = np.array(new)
            fit = np.array([self.fitness(batch, ind, t) for ind in P])
            hist.append(fit.min() / n)
        if self.record:
            self.history.append(hist)
        return list(P[np.argmin(fit)])


class PSO(BatchBase):
    name = "PSO"

    def __init__(self, particles=30, iters=40, w=0.7, c1=1.5, c2=1.5, seed=0, record=False):
        self.np_, self.iters, self.w, self.c1, self.c2 = particles, iters, w, c1, c2
        self.rng = np.random.default_rng(seed)
        self.record, self.history = record, []

    def attach(self, sim):
        super().attach(sim)
        self.set_refs()

    def decode(self, x):
        return np.clip(np.floor(x), 0, self.N - 1).astype(int)

    def decide(self, batch, t):
        n, N, rng = len(batch), self.N, self.rng
        X = rng.uniform(0, N, (self.np_, n))
        X[0] = np.array([b.fog for b in batch]) + 0.5
        V = rng.uniform(-1, 1, (self.np_, n))
        fit = np.array([self.fitness(batch, self.decode(x), t) for x in X])
        pbest, pfit = X.copy(), fit.copy()
        g = pbest[np.argmin(pfit)].copy()
        hist = [pfit.min() / n]
        for _ in range(self.iters):
            r1, r2 = rng.random((self.np_, n)), rng.random((self.np_, n))
            V = self.w * V + self.c1 * r1 * (pbest - X) + self.c2 * r2 * (g - X)
            X = np.clip(X + V, 0, N - 1e-9)
            fit = np.array([self.fitness(batch, self.decode(x), t) for x in X])
            better = fit < pfit
            pbest[better], pfit[better] = X[better], fit[better]
            g = pbest[np.argmin(pfit)].copy()
            hist.append(pfit.min() / n)
        if self.record:
            self.history.append(hist)
        return list(self.decode(g))


# ---------------- proposed ----------------
class RADS(Base):
    """Risk-Aware Deadline-constrained Scheduling (proposed).

    queue order : (tier rank, -risk, absolute deadline)
    placement   : argmin_j  w_tier * (lateness_j / D + 0.01 * response_j / D)
                            + lam_E * energy_j / E_ref + lam_C * cost_j / C_ref
                  where the queueing delay counts only work ahead of the job in priority order
    reservation : at fog nodes, `fog_reserved_for_urgent` servers are only used by URGENT jobs
    links       : images queued on shared backhaul / WAN links are transmitted in risk order
    """
    name = "RADS (proposed)"

    def __init__(self, use_tiers=True, use_reservation=True, use_cost_terms=False, use_priority_links=True):
        self.use_tiers, self.use_res, self.use_cost = use_tiers, use_reservation, use_cost_terms
        self.use_plink = use_priority_links

    def attach(self, sim):
        super().attach(sim)
        self.e_ref, self.c_ref = sim.e_ref, sim.c_ref
        # exponentially-weighted rate of URGENT work assigned to each node (reference-CPU s per s)
        self.u_rate = [0.0] * self.N
        self.u_last = [0.0] * self.N
        self.tau = 60.0

    def urgent_load(self, j, t):
        """rho_U,j: utilisation of node j by higher-priority (URGENT) work, EWMA over tau seconds."""
        node = self.sim.nodes[j]
        u = self.u_rate[j] * math.exp(-(t - self.u_last[j]) / self.tau)
        return min(0.95, u / (node.c * node.speed))

    def note_urgent(self, j, work, t):
        self.u_rate[j] = self.u_rate[j] * math.exp(-(t - self.u_last[j]) / self.tau) + work / self.tau
        self.u_last[j] = t

    def queue_key(self, job):
        if not self.use_tiers:
            return (0, 0.0, job.t_capture + job.deadline)
        return (TIER_RANK[job.tier], -job.risk, job.t_capture + job.deadline)

    def link_key(self, job):
        """Risk-priority transmission on shared links (URGENT images first)."""
        if not (self.use_plink and self.use_tiers):
            return (job.t_ready, job.jid)
        return (TIER_RANK[job.tier], -job.risk, job.t_ready, job.jid)

    def ahead_backlog(self, node, job):
        k = self.queue_key(job)
        return sum(q.work / node.speed for key, _, q in node.queue if key <= k)

    def on_ready(self, job, t):
        sim, p = self.sim, self.p
        best, best_j = math.inf, job.fog
        w = p.w_tier[job.tier] if self.use_tiers else 1.0
        for j in range(self.N):
            node = sim.nodes[j]
            arr = sim.transfer_estimate(job, j, t)
            servers = node.c
            if self.use_res and j != sim.cloud and job.tier != "URGENT":
                servers = max(1, node.c - p.fog_reserved_for_urgent)
            wait = self.ahead_backlog(node, job) / servers
            if self.use_tiers and job.tier != "URGENT":
                # lower-priority work is also delayed by URGENT jobs that arrive while it waits
                # (non-preemptive priority queue, Cobham 1954): inflate by 1 / (1 - rho_U)
                wait /= (1.0 - self.urgent_load(j, t))
            start = max(arr, sorted(node.server_free)[node.c - servers]) + wait
            fin = start + job.work / node.speed
            back = 0.0 if j == job.fog else (p.L_wan if j == sim.cloud else p.L_backhaul)
            resp = fin + back + p.L_access - job.t_capture
            J = w * (max(0.0, resp - job.deadline) / job.deadline + 0.01 * resp / job.deadline)
            if self.use_cost:
                e, c = sim.job_energy_cost(job, j)
                J += p.lam_energy * e / self.e_ref + p.lam_cost * c / self.c_ref
            if J < best:
                best, best_j = J, j
        if job.tier == "URGENT":
            self.note_urgent(best_j, job.work, t)
        sim.assign(job, best_j, t)

    def may_start(self, job, j, n_free, t):
        if not self.use_res or j == self.sim.cloud or job.tier == "URGENT":
            return True
        return n_free > self.p.fog_reserved_for_urgent
