"""Append only ledger with atomic transactions, checkpoints and taint aware retraction.

Agents never mutate memory directly. They open a transaction, read and propose writes, then
commit. A commit is all or nothing, passes every guard, and fails if something it read has
changed since. Because each commit records what it read, retracting a bad commit can
find and undo everything built on top of it.
"""
import bisect
import itertools
import json
import sqlite3
import threading
from dataclasses import dataclass, field


class Rejected(Exception):
    """A guard refused the transaction."""


class Conflict(Exception):
    """A key read by the transaction changed before it committed."""


@dataclass(frozen=True)
class Write:
    key: str
    value: object
    agent: str
    confidence: float = 0.5
    evidence: tuple = ()


@dataclass(frozen=True)
class Entry:
    value: object
    version: int
    commit: int
    agent: str
    confidence: float


@dataclass
class Commit:
    id: int
    agent: str
    writes: list
    reads: dict
    retracted: bool = False
    previous: dict = field(default_factory=dict)


class Transaction:
    def __init__(self, ledger, agent):
        self.ledger, self.agent = ledger, agent
        self.reads, self.writes, self.done = {}, {}, False

    def read(self, key):
        # A transaction sees its own pending writes first, then committed state.
        if key in self.writes:
            return self.writes[key].value
        entry = self.ledger.entry(key)
        self.reads[key] = entry.version if entry else 0
        return entry.value if entry else None

    def write(self, key, value, confidence=0.5, evidence=()):
        self.writes[key] = Write(key, value, self.agent, confidence, tuple(evidence))

    def commit(self):
        return self.ledger._commit(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None and not self.done:
            self.commit()
        return False


class Ledger:
    def __init__(self, guards=(), path=None):
        self.guards = list(guards)
        self.state, self.commits, self.rejections = {}, {}, []
        self.checkpoints = {}
        self._writers = {}  # key -> commit ids that wrote it, so retraction stays fast on long logs
        self._ids = itertools.count(1)
        self._versions = itertools.count(1)
        self._lock = threading.RLock()
        self._db = None
        if path:
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS commits (id INTEGER PRIMARY KEY, agent TEXT, writes TEXT, reads TEXT)")
            self._db.execute("CREATE TABLE IF NOT EXISTS retractions (id INTEGER PRIMARY KEY)")
            self._replay()

    def begin(self, agent):
        return Transaction(self, agent)

    def entry(self, key):
        return self.state.get(key)

    def get(self, key, default=None):
        entry = self.state.get(key)
        return entry.value if entry else default

    def snapshot(self):
        return {k: e.value for k, e in self.state.items()}

    def _commit(self, tx):
        with self._lock:
            tx.done = True
            for key, version in tx.reads.items():
                current = self.state.get(key)
                if (current.version if current else 0) != version:
                    raise Conflict(f"{key} changed since it was read")
            for write in tx.writes.values():
                for guard in self.guards:
                    verdict = guard.check(write, self)
                    if not verdict.ok:
                        self.rejections.append((tx.agent, write.key, verdict.reason))
                        raise Rejected(verdict.reason)
            commit = Commit(next(self._ids), tx.agent, list(tx.writes.values()), dict(tx.reads))
            self._apply(commit)
            if self._db:
                self._db.execute("INSERT INTO commits VALUES (?, ?, ?, ?)", (
                    commit.id, commit.agent,
                    json.dumps([[w.key, w.value, w.agent, w.confidence, list(w.evidence)] for w in commit.writes]),
                    json.dumps(commit.reads)))
                self._db.commit()
            return commit.id

    def _apply(self, commit):
        for write in commit.writes:
            self._writers.setdefault(write.key, []).append(commit.id)
            commit.previous[write.key] = self.state.get(write.key)
            self.state[write.key] = Entry(write.value, next(self._versions), commit.id, write.agent, write.confidence)
        self.commits[commit.id] = commit

    def checkpoint(self, name):
        """Marks the current point in history. Costs one integer, not a copy of memory."""
        with self._lock:
            self.checkpoints[name] = max(self.commits, default=0)

    def rollback(self, name):
        """Retracts every live commit made after the checkpoint, newest first."""
        with self._lock:
            mark = self.checkpoints[name]
            undone = [i for i in sorted(self.commits, reverse=True) if i > mark and not self.commits[i].retracted]
            for commit_id in undone:
                self._retract_one(commit_id)
            return undone

    def dependents(self, commit_id):
        """Commits that read a key written by this commit, directly or transitively."""
        tainted, found = {commit_id}, []
        readers = self._readers_after(commit_id)
        for cid in readers:
            commit = self.commits[cid]
            if any(self._read_came_from(commit, key, tainted) for key in commit.reads):
                tainted.add(cid)
                found.append(cid)
        return found

    def _readers_after(self, commit_id):
        return [cid for cid in sorted(self.commits) if cid > commit_id and not self.commits[cid].retracted and self.commits[cid].reads]

    def _read_came_from(self, commit, key, tainted):
        """True if the newest write to key before this commit belongs to a tainted commit."""
        writers = self._writers.get(key, [])
        for cid in reversed(writers[:bisect.bisect_left(writers, commit.id)]):
            if cid in tainted:
                return True
            if not self.commits[cid].retracted:
                return False
        return False

    def fork(self):
        """An isolated copy for trying a risky plan. Merge the result back with merge()."""
        branch = Ledger(self.guards)
        branch.state = dict(self.state)
        branch._base = max(self.commits, default=0)
        branch._versions = itertools.count(next(self._versions) + 1)
        branch._ids = itertools.count(branch._base + 1)
        return branch

    def merge(self, branch):
        """Replays a fork's commits as one transaction per commit. Guards and conflicts still apply."""
        merged = []
        for cid in sorted(branch.commits):
            commit = branch.commits[cid]
            if commit.retracted:
                continue
            tx = self.begin(commit.agent)
            for write in commit.writes:
                tx.writes[write.key] = write
            merged.append(tx.commit())
        return merged

    def retract(self, commit_id, cascade=True):
        """Undoes a commit and, by default, everything derived from it. Returns the ids undone."""
        with self._lock:
            targets = [commit_id] + (self.dependents(commit_id) if cascade else [])
            undone = sorted(targets, reverse=True)
            for cid in undone:
                self._retract_one(cid)
            return undone

    def _retract_one(self, commit_id):
        commit = self.commits[commit_id]
        commit.retracted = True
        for write in commit.writes:
            self._restore(write.key)
        if self._db:
            self._db.execute("INSERT OR IGNORE INTO retractions VALUES (?)", (commit_id,))
            self._db.commit()

    def _restore(self, key):
        """Sets key to the newest live write, so retracting out of order stays correct."""
        for cid in reversed(self._writers.get(key, [])):
            commit = self.commits[cid]
            if commit.retracted:
                continue
            for write in commit.writes:
                if write.key == key:
                    self.state[key] = Entry(write.value, next(self._versions), cid, write.agent, write.confidence)
                    return
        self.state.pop(key, None)

    def _replay(self):
        retracted = {r[0] for r in self._db.execute("SELECT id FROM retractions")}
        rows = self._db.execute("SELECT id, agent, writes, reads FROM commits ORDER BY id").fetchall()
        for cid, agent, writes, reads in rows:
            items = [Write(k, v, a, c, tuple(e)) for k, v, a, c, e in json.loads(writes)]
            self._apply(Commit(cid, agent, items, json.loads(reads)))
        for cid in retracted:
            self.commits[cid].retracted = True
        for key in {w.key for c in self.commits.values() for w in c.writes}:
            self._restore(key)
        self._ids = itertools.count(max(self.commits, default=0) + 1)

    def audit(self):
        """Full history including retracted commits, for debugging what agents did."""
        return [dict(id=c.id, agent=c.agent, retracted=c.retracted, reads=c.reads,
                     writes=[(w.key, w.value, w.confidence) for w in c.writes])
                for _, c in sorted(self.commits.items())]
