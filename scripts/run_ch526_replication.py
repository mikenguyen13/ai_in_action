"""Replicate He, Hua, Wang, Zheng (ISR 2026) on SATPdb with ESM2-35M embeddings.

Usage: python scripts/run_ch526_replication.py [--runs 20] [--rounds 50]
Writes results/ch526_replication.json. Uses CPU worker processes (16 cores ~ 30 min).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

TASKS = {
    "task1": ["anticancer", "antibacterial", "antifungal"],
    "task2": ["antimicrobial", "antiviral", "antibacterial"],
    "task3": ["anticancer", "antibacterial", "antifungal", "antimicrobial", "antiviral", "antiparasitic"],
}
STRATS = ["random", "search", "active_learning", "ucb", "random_then_ucb", "al_then_ucb",
          "human_only", "algorithm_only", "hitl"]
KNOWLEDGE_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]
_G = {}


def _init():
    import torch
    torch.set_num_threads(1)
    from aiinaction import ch526_hitl_discovery as m
    seqs, y, names = m.load_satpdb()
    x = np.load("data/satpdb/esm2_35M.npy")
    _G.update(m=m, y=y, names=names, x=(x - x.mean(0)) / x.std(0))


def _job(args):
    task, strat, run, knowledge, rounds = args
    m = _G["m"]
    pool, _, lab = m.build_task(_G["y"], _G["names"], TASKS[task], seed=0)  # fixed pool, as in the paper
    xp = _G["x"][pool]

    def fp(xl, yl):
        return m.BayesianMLP(xp.shape[1], yl.shape[1], seed=run).fit(xl, yl).predict_samples

    r = m.simulate(strat, xp, lab, fp, rounds=rounds, knowledge=knowledge, seed=1000 + run)
    return dict(task=task, strategy=strat, run=run, knowledge=knowledge,
                hits=r.hits_by_round, n_targets=r.n_targets)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=50)
    a = ap.parse_args()
    jobs = [(t, s, r, 0.5, a.rounds) for t in TASKS for s in STRATS for r in range(a.runs)]
    jobs += [("task1", "hitl", r, k, a.rounds) for k in KNOWLEDGE_SWEEP for r in range(a.runs) if k != 0.5]
    jobs += [("task1", "human_only", r, k, a.rounds) for k in KNOWLEDGE_SWEEP for r in range(a.runs) if k != 0.5]
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(os.cpu_count(), initializer=_init) as ex:
        for i, res in enumerate(ex.map(_job, jobs, chunksize=1)):
            out.append(res)
            if i % 20 == 0:
                print(i, len(jobs), round(time.time() - t0), flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/ch526_replication.json").write_text(json.dumps(out))
    print("done", round(time.time() - t0))
