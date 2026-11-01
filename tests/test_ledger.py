import threading

import pytest

from agentledger import Conflict, ConflictGuard, Ledger, ProvenanceGuard, RangeGuard, Rejected, TypeGuard


def put(ledger, key, value, agent="a", **kw):
    with ledger.begin(agent) as tx:
        tx.write(key, value, **kw)
    return max(ledger.commits)


def test_commit_is_visible_and_uncommitted_is_not():
    ledger = Ledger()
    tx = ledger.begin("a")
    tx.write("x", 1)
    assert ledger.get("x") is None
    assert tx.read("x") == 1
    tx.commit()
    assert ledger.get("x") == 1


def test_exception_inside_transaction_discards_writes():
    ledger = Ledger()
    with pytest.raises(ValueError):
        with ledger.begin("a") as tx:
            tx.write("x", 1)
            raise ValueError
    assert ledger.get("x") is None and not ledger.commits


def test_transaction_is_all_or_nothing_when_one_write_is_rejected():
    ledger = Ledger([RangeGuard("score", 0, 1)])
    tx = ledger.begin("a")
    tx.write("name", "ok")
    tx.write("score", 5)
    with pytest.raises(Rejected):
        tx.commit()
    assert ledger.get("name") is None
    assert ledger.rejections[0][1] == "score"


def test_lost_update_is_detected():
    ledger = Ledger()
    put(ledger, "count", 1)
    first, second = ledger.begin("a"), ledger.begin("b")
    first.write("count", first.read("count") + 1)
    second.write("count", second.read("count") + 1)
    first.commit()
    with pytest.raises(Conflict):
        second.commit()
    assert ledger.get("count") == 2


def test_read_of_missing_key_conflicts_if_key_appears():
    ledger = Ledger()
    tx = ledger.begin("a")
    assert tx.read("x") is None
    tx.write("y", 1)
    put(ledger, "x", 1)
    with pytest.raises(Conflict):
        tx.commit()


def test_concurrent_increments_never_lose_updates():
    ledger = Ledger()
    put(ledger, "n", 0)

    def worker():
        for _ in range(100):
            while True:
                tx = ledger.begin("w")
                tx.write("n", tx.read("n") + 1)
                try:
                    tx.commit()
                    break
                except Conflict:
                    continue

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert ledger.get("n") == 800


def test_guards():
    ledger = Ledger([TypeGuard("age", int), ProvenanceGuard()])
    with pytest.raises(Rejected):
        put(ledger, "age", "old", evidence=["doc"])
    with pytest.raises(Rejected):
        put(ledger, "age", 3)
    with pytest.raises(Rejected):
        put(ledger, "age", 3, evidence=["mem:missing"])
    put(ledger, "age", 3, evidence=["doc"])
    put(ledger, "age2", 4, evidence=["mem:age"])
    assert ledger.get("age2") == 4


def test_conflict_guard_requires_higher_confidence():
    ledger = Ledger([ConflictGuard()])
    put(ledger, "capital", "Paris", confidence=0.9)
    with pytest.raises(Rejected):
        put(ledger, "capital", "Lyon", confidence=0.5)
    put(ledger, "capital", "Paris", confidence=0.1)  # agreement is always fine
    put(ledger, "capital", "Marseille", confidence=0.95)
    assert ledger.get("capital") == "Marseille"


def test_checkpoint_rollback_restores_values_and_removes_new_keys():
    ledger = Ledger()
    put(ledger, "x", 1)
    ledger.checkpoint("cp")
    put(ledger, "x", 2)
    put(ledger, "y", 3)
    assert len(ledger.rollback("cp")) == 2
    assert ledger.snapshot() == {"x": 1}
    assert len(ledger.audit()) == 3  # history survives


def test_rollback_to_empty_checkpoint():
    ledger = Ledger()
    ledger.checkpoint("start")
    put(ledger, "x", 1)
    ledger.rollback("start")
    assert ledger.snapshot() == {}


def test_retract_cascades_through_derived_facts():
    ledger = Ledger()
    bad = put(ledger, "fact", "wrong")
    with ledger.begin("b") as tx:
        tx.write("derived", tx.read("fact") + "!")
    with ledger.begin("c") as tx:
        tx.write("derived2", tx.read("derived") + "?")
    put(ledger, "independent", 1)
    assert len(ledger.retract(bad)) == 3
    assert ledger.snapshot() == {"independent": 1}


def test_retract_does_not_taint_reads_of_overwritten_value():
    ledger = Ledger()
    bad = put(ledger, "fact", "wrong")
    put(ledger, "fact", "right")
    with ledger.begin("b") as tx:
        tx.write("derived", tx.read("fact"))  # read the correct value
    assert ledger.retract(bad) == [bad]
    assert ledger.get("derived") == "right" and ledger.get("fact") == "right"


def test_retract_older_write_keeps_newer_value():
    ledger = Ledger()
    first = put(ledger, "x", 1)
    put(ledger, "x", 2)
    ledger.retract(first, cascade=False)
    assert ledger.get("x") == 2


def test_retract_newer_write_restores_older_value():
    ledger = Ledger()
    put(ledger, "x", 1)
    second = put(ledger, "x", 2)
    ledger.retract(second)
    assert ledger.get("x") == 1


def test_stale_transaction_conflicts_after_retraction():
    ledger = Ledger()
    bad = put(ledger, "x", 1)
    tx = ledger.begin("a")
    tx.read("x")
    tx.write("y", 2)
    ledger.retract(bad)
    with pytest.raises(Conflict):
        tx.commit()


def test_sqlite_persistence_survives_reopen(tmp_path):
    path = tmp_path / "memory.db"
    ledger = Ledger(path=path)
    put(ledger, "x", 1)
    second = put(ledger, "x", 2)
    put(ledger, "y", {"nested": [1, 2]})
    ledger.retract(second, cascade=False)
    reopened = Ledger(path=path)
    assert reopened.snapshot() == {"x": 1, "y": {"nested": [1, 2]}}
    assert put(reopened, "z", 3) == 4
