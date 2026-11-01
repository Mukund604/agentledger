"""Sweeps fault rate and audit recall through the simulation, and measures raw performance.

Agents are scripted, so this shows how the memory layer behaves under the assumed fault model.
"""
import json
import pathlib
import statistics
import time

import numpy as np

from agentledger import Ledger
from .charts import save
from .fault_injection import corrupted, faults_for, plan, run_ledger, run_plain

TASKS, SEEDS = 200, 40


def trial(rate, recall, seed):
    rng = np.random.default_rng(seed)
    steps, known = plan(rng, TASKS)
    faults = faults_for(rng, TASKS, rate)
    plain = corrupted(run_plain(steps, faults)[0], known)
    guarded = corrupted(run_ledger(steps, faults, recall, np.random.default_rng(seed + 1), False)[0], known)
    full = corrupted(run_ledger(steps, faults, recall, np.random.default_rng(seed + 1), True)[0], known)
    return plain, guarded, full


def performance():
    rows = []
    for size in (1_000, 10_000, 50_000):
        ledger = Ledger()
        start = time.perf_counter()
        for i in range(size):
            tx = ledger.begin("a")
            if i:
                tx.read(f"k{i - 1}")
            tx.write(f"k{i}", i)
            tx.commit()
        commits = size / (time.perf_counter() - start)
        target = size - 50
        start = time.perf_counter()
        undone = len(ledger.retract(target))
        rows.append(dict(size=size, commits_per_second=commits, retract_ms=(time.perf_counter() - start) * 1000, undone=undone))
        print(rows[-1], flush=True)
    return rows


def main():
    rates = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    by_rate = {r: np.array([trial(r, 0.7, s) for s in range(SEEDS)]) for r in rates}
    recalls = [0.0, 0.25, 0.5, 0.75, 1.0]
    by_recall = {r: np.array([trial(0.10, r, s) for s in range(SEEDS)]) for r in recalls}
    result = dict(
        seeds=SEEDS, tasks=TASKS,
        fault_rate={str(r): v.mean(axis=0).tolist() for r, v in by_rate.items()},
        audit_recall={str(r): v.mean(axis=0).tolist() for r, v in by_recall.items()},
        performance=performance(), order=["shared dict", "guards", "guards and retraction"])
    pathlib.Path("results/sweep.json").write_text(json.dumps(result, indent=2))
    names = ["Shared dict", "Ledger with guards", "Guards + audited retraction"]

    def draw_rate(fig, theme):
        ax = fig.add_subplot()
        for i, name in enumerate(names):
            mean = [by_rate[r][:, i].mean() * 100 for r in rates]
            se = [by_rate[r][:, i].std(ddof=1) / np.sqrt(SEEDS) * 196 for r in rates]
            ax.plot([r * 100 for r in rates], mean, color=theme["series"][i], lw=2, marker="o", ms=6, label=name)
            ax.fill_between([r * 100 for r in rates], np.array(mean) - se, np.array(mean) + se, color=theme["series"][i], alpha=0.14, lw=0)
        ax.set_xlabel("Share of agent steps that misbehave (%)")
        ax.set_ylabel("Final facts that are wrong (%)")
        ax.set_ylim(bottom=0)
        ax.set_title("Guards and retraction contain faults as the fault rate climbs")
        ax.legend(loc="upper left")
    save(draw_rate, "fault-rate")

    def draw_recall(fig, theme):
        ax = fig.add_subplot()
        for i, name in enumerate(names):
            ax.plot([r * 100 for r in recalls], [by_recall[r][:, i].mean() * 100 for r in recalls], color=theme["series"][i], lw=2, marker="o", ms=6, label=name)
        ax.set_xlabel("Share of plausible faults an auditor finds (%)")
        ax.set_ylabel("Final facts that are wrong (%)")
        ax.set_ylim(bottom=0)
        ax.set_title("Retraction is only as good as whatever detects the fault")
        ax.legend(loc="center right")
    save(draw_recall, "audit-recall")


if __name__ == "__main__":
    main()
