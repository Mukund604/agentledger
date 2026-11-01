"""Guards decide whether a proposed write may enter shared memory."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""


class Guard:
    def check(self, write, ledger) -> Verdict:
        raise NotImplementedError


class TypeGuard(Guard):
    """Keys matching a prefix must hold values of the given type."""

    def __init__(self, prefix, expected):
        self.prefix, self.expected = prefix, expected

    def check(self, write, ledger):
        if write.key.startswith(self.prefix) and not isinstance(write.value, self.expected):
            return Verdict(False, f"{write.key} must be {self.expected.__name__}")
        return Verdict(True)


class RangeGuard(Guard):
    def __init__(self, prefix, low, high):
        self.prefix, self.low, self.high = prefix, low, high

    def check(self, write, ledger):
        if write.key.startswith(self.prefix) and not self.low <= write.value <= self.high:
            return Verdict(False, f"{write.key}={write.value} outside [{self.low}, {self.high}]")
        return Verdict(True)


class ProvenanceGuard(Guard):
    """Every write must cite evidence, and every cited memory key must exist."""

    def check(self, write, ledger):
        if not write.evidence:
            return Verdict(False, f"{write.key} has no evidence")
        for ref in write.evidence:
            if ref.startswith("mem:") and ledger.get(ref[4:]) is None:
                return Verdict(False, f"{write.key} cites missing key {ref[4:]}")
        return Verdict(True)


class ConflictGuard(Guard):
    """Blocks overwriting a value with a different one unless confidence is strictly higher.

    Catches the common multi agent failure where a late, unsure agent silently replaces
    a well supported fact.
    """

    def check(self, write, ledger):
        current = ledger.entry(write.key)
        if current is None or current.value == write.value:
            return Verdict(True)
        if write.confidence > current.confidence:
            return Verdict(True)
        return Verdict(False, f"{write.key}: {write.value!r} contradicts {current.value!r} "
                              f"without higher confidence ({write.confidence} vs {current.confidence})")
