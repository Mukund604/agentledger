# agentledger

Transactional shared memory for multi agent systems. When several agents share a scratchpad, one wrong write quietly becomes every later agent's input. agentledger treats that memory like a database: agents propose, guards decide, commits are atomic, and a bad fact can be retracted along with everything built on it.

```python
from agentledger import ConflictGuard, Ledger, RangeGuard

ledger = Ledger([RangeGuard("score", 0, 100), ConflictGuard()], path="memory.db")

with ledger.begin("researcher") as tx:
    tx.write("revenue", 120, confidence=0.9, evidence=["10-K page 4"])
bad = max(ledger.commits)

with ledger.begin("analyst") as tx:
    tx.write("forecast", tx.read("revenue") * 1.1)   # the read is recorded

ledger.retract(bad)              # undoes revenue AND the forecast derived from it
ledger.get("forecast")           # None
```

## What it guarantees

* **Atomic commits.** A transaction applies fully or not at all, and an exception discards it.
* **Guards.** Types, ranges, required evidence, and blocking an unsure agent from overwriting a confident fact. Write your own by subclassing `Guard`.
* **No lost updates.** Reads are versioned, so a commit fails with `Conflict` if something it read has changed. A test runs 8 threads doing 800 increments with none lost.
* **Cascading retraction.** Each commit records what it read, so retracting one also undoes commits that used its value, and only those. A later commit that read a corrected value is left alone.
* **Checkpoints and rollback**, which only store a position in the log.
* **Audit log and durability.** Full history stays available, and an optional SQLite file survives restarts.

## Does it help?

[`benchmarks/fault_injection.py`](benchmarks/fault_injection.py) simulates a 200 step pipeline where each step derives a fact from earlier ones, over 30 seeds. 10% of steps misbehave: an impossible value, an unsure overwrite of a good fact, or a plausible wrong value that no guard can catch. An auditor later finds 70% of the plausible ones.

| Memory | Final facts that are wrong |
| :--- | ---: |
| Shared dict, last write wins | 28.9% ± 12.1 |
| Ledger with guards | 11.3% ± 9.9 |
| Ledger with guards and audited retraction | 4.0% ± 7.9 |

Recovering from the audited faults undid about 15 commits per run with cascading retraction, against about 156 for rolling back to a checkpoint before the first fault.

**These numbers come from scripted agents and no LLM.** The fault rate and auditor recall are assumptions I chose, so they show how the memory layer behaves and not how much a real system would improve. Recovery also assumes a rerun step succeeds. The remaining 4% comes from the faults the auditor never found: a ledger can only undo what something detects.

## Limitations

* It's a single process library with one lock, not a distributed store.
* Guards are only as good as their rules. `ConflictGuard` compares values for equality and trusts self reported confidence, so it can't detect that two differently worded facts contradict.
* Taint tracking only sees reads made through the transaction. An agent that copies a value out of band isn't tracked.
* Retraction scans history, so very long logs will need indexing.

## Run it

```
pip install -e ".[dev]"
pytest
python -m benchmarks.fault_injection
```

No dependencies beyond the Python standard library.
