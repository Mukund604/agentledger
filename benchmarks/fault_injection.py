"""Simulates a multi agent pipeline with injected faults and compares memory strategies.

No LLM is involved. Agents are scripted so that every run is reproducible and ground truth is
known exactly. This measures the memory layer's behavior, not any model's.

Each task derives a fact from up to two earlier facts. A worker sometimes misbehaves:
  out_of_range  writes an impossible value, which a guard can catch
  overwrite     replaces an existing correct fact with an unsure wrong one, which a guard can catch
  plausible     writes a wrong but valid looking new fact, which no guard can catch

An auditor later finds each plausible fault with some probability, like a human reviewer or
a verification step would. It does not see the facts built on top of the fault.
"""
import json
import pathlib
import statistics

import numpy as np

from agentledger import ConflictGuard, Ledger, RangeGuard, Rejected

MOD = 1000


def truth(index, parents, known):
    return (index * 7 + sum(known[p] for p in parents)) % MOD


def plan(rng, tasks):
    steps = []
    for i in range(tasks):
        count = int(rng.integers(0, 3)) if i else 0
        steps.append((f"fact{i}", sorted({f"fact{int(j)}" for j in rng.integers(0, i, count)}) if i else []))
    known = {}
    for i, (key, parents) in enumerate(steps):
        known[key] = truth(i, parents, known)
    return steps, known


def faults_for(rng, tasks, rate):
    kinds = {}
    for i in range(tasks):
        if rng.random() < rate:
            kinds[i] = str(rng.choice(["out_of_range", "overwrite", "plausible"]))
    return kinds


def wrong(value):
    return (value + 137) % MOD


def run_plain(steps, faults):
    """Baseline: a shared dict where the last write wins."""
    memory = {}
    for i, (key, parents) in enumerate(steps):
        value = truth(i, parents, memory)
        kind = faults.get(i)
        if kind == "out_of_range":
            value = 10**6
        elif kind == "plausible":
            value = wrong(value)
        memory[key] = value
        if kind == "overwrite" and i:
            victim = f"fact{i // 2}"
            memory[victim] = wrong(memory[victim])
    return memory, 0, 0


def run_ledger(steps, faults, audit_recall, rng, recover):
    ledger = Ledger([RangeGuard("fact", 0, MOD - 1), ConflictGuard()])
    bad_commits, blocked = [], 0

    def do_task(i, key, parents, inject=None):
        tx = ledger.begin("worker")
        value = truth(i, parents, {p: tx.read(p) for p in parents})
        if inject == "out_of_range":
            value = 10**6
        if inject == "plausible":
            value = wrong(value)
        tx.write(key, value, confidence=0.9)
        tx.commit()
        return max(ledger.commits)

    for i, (key, parents) in enumerate(steps):
        kind = faults.get(i)
        try:
            commit = do_task(i, key, parents, kind)
        except Rejected:
            blocked += 1
            commit = do_task(i, key, parents)  # the orchestrator retries a rejected task
        if kind == "plausible":
            bad_commits.append(commit)
        if kind == "overwrite" and i:
            try:
                with ledger.begin("late_worker") as tx:
                    tx.write(f"fact{i // 2}", wrong(ledger.get(f"fact{i // 2}")), confidence=0.3)
            except Rejected:
                blocked += 1

    undone_total = 0
    if recover:
        found = [c for c in bad_commits if rng.random() < audit_recall]
        for commit in found:
            if not ledger.commits[commit].retracted:
                undone_total += len(ledger.retract(commit))
        # Rerun whatever is now missing, in order, with honest workers.
        for i, (key, parents) in enumerate(steps):
            if ledger.get(key) is None:
                do_task(i, key, parents)
    return ledger.snapshot(), blocked, undone_total


def corrupted(memory, known):
    return sum(memory.get(k) != v for k, v in known.items()) / len(known)


def checkpoint_cost(steps, faults, audit_recall, rng):
    """Commits a checkpoint only recovery throws away, compared with cascading retraction."""
    ledger = Ledger([RangeGuard("fact", 0, MOD - 1), ConflictGuard()])
    bad = []
    for i, (key, parents) in enumerate(steps):
        tx = ledger.begin("worker")
        value = truth(i, parents, {p: tx.read(p) for p in parents})
        if faults.get(i) == "plausible":
            value = wrong(value)
            bad.append(i)
        tx.write(key, value, confidence=0.9)
        tx.commit()
    found = [i for i in bad if rng.random() < audit_recall]
    if not found:
        return 0, 0
    first_bad = min(found)
    rollback = len(ledger.commits) - first_bad
    undone = set()
    for i in found:
        undone.update(ledger.dependents(i + 1))
        undone.add(i + 1)
    return rollback, len(undone)


def main():
    tasks, rate, audit_recall, seeds = 200, 0.10, 0.7, 30
    rows = {"Shared dict, last write wins": [], "Ledger with guards": [], "Ledger with guards and audited retraction": []}
    blocked, undone, rollback, cascade = [], [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        steps, known = plan(rng, tasks)
        faults = faults_for(rng, tasks, rate)
        plain, _, _ = run_plain(steps, faults)
        rows["Shared dict, last write wins"].append(corrupted(plain, known))
        guarded, b, _ = run_ledger(steps, faults, audit_recall, np.random.default_rng(seed + 1), False)
        rows["Ledger with guards"].append(corrupted(guarded, known))
        recovered, _, u = run_ledger(steps, faults, audit_recall, np.random.default_rng(seed + 1), True)
        rows["Ledger with guards and audited retraction"].append(corrupted(recovered, known))
        blocked.append(b)
        undone.append(u)
        a, c = checkpoint_cost(steps, faults, audit_recall, np.random.default_rng(seed + 1))
        rollback.append(a)
        cascade.append(c)
    result = {
        "settings": dict(tasks=tasks, fault_rate=rate, audit_recall=audit_recall, seeds=seeds),
        "corrupted_share": {k: dict(mean=statistics.mean(v), std=statistics.stdev(v)) for k, v in rows.items()},
        "guard_blocks_per_run": statistics.mean(blocked),
        "commits_undone_per_run": statistics.mean(undone),
        "recovery_cost": dict(rollback_to_first_fault=statistics.mean(rollback), cascading_retraction=statistics.mean(cascade)),
    }
    pathlib.Path("benchmarks/results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
