"""Build-time static plan certificates and launch-time signal-only checks.

A certificate is usable only with the same sealed artifact, scalar bindings and
an epoch base covered by the source translation proof. Memory leases and all
region/resource checks remain the responsibility of the complete build plan.
"""

from dataclasses import dataclass

from tilelang.analysis.distributed_checker import fail
from tilelang.distributed.abi import MAX_EPOCH, manifest_digest
from .plan import SignalRef


def _read_only(*args, **kwargs):
    raise TypeError("static artifact phase reports are read-only")


class _Phase(dict):
    """JSON-compatible cached diagnostics, independent of executable state."""

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _read_only


class _Phases(list):
    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = __imul__ = _read_only


@dataclass(frozen=True)
class StaticContract:
    identity: str
    scalars: tuple
    epoch_bounds: tuple
    publication_bounds: tuple | None
    wait_bounds: tuple | None
    signals: tuple
    phases: tuple

    def signal_plans(self, base, end):
        if isinstance(base, bool) or not isinstance(base, int) or not self.epoch_bounds[0] <= base <= self.epoch_bounds[1]:
            fail("static contract epoch base is outside its translation proof")
        published, waits = self.publication_bounds, self.wait_bounds
        if end is None:
            end = base + max(1, 0 if published is None else published[1])
        if isinstance(end, bool) or not isinstance(end, int) or not base < end <= MAX_EPOCH:
            fail("native prepared epoch interval must be positive and fit int32")
        if published is not None and (published[0] <= 0 or base + published[1] > end):
            fail("published epoch is outside the prepared native interval")
        if waits is not None and (base + waits[0] < 1 or base + waits[1] > end):
            fail("wait epoch is outside the prepared native interval")
        plans = {rank: [(kind, ref, base + offset, ()) for kind, ref, offset in actions] for rank, actions in enumerate(self.signals)}
        return plans, end


@dataclass(frozen=True)
class StaticLaunch:
    contract: StaticContract
    artifact: object
    base: int
    end: int
    signals: dict

    def metadata(self):
        return {
            str(rank): {
                "static_contract": self.contract.identity,
                "artifact_id": self.artifact.artifact_id,
                "scalars": [list(item) for item in self.contract.scalars],
                "epoch_base": self.base,
                "epoch_end": self.end,
                "phase_count": len(self.contract.phases[rank]),
                "signal_count": len(self.contract.signals[rank]),
            }
            for rank in (0, 1)
        }


def make_static_contract(plans, *, proof, scalars, descriptions):
    """Certify the projection after full native and fused protocol validation.

    The current two-rank transport only publishes to the other rank. Thus a
    signal cell has one writer, and its publication order is fixed by that
    writer's program. Removing nonblocking, signal-free steps preserves this
    finite progress model's success/failure and final signal values. It does
    not replace acquire/release, DMA-completion or memory-lease verification.
    """
    if proof is None or set(plans) != {0, 1}:
        return None
    signals, phases, published, waits, writers = [], [], [], [], {}
    for rank in (0, 1):
        actions, kinds = [], []
        for instruction in plans[rank]:
            kind = instruction[0]
            kinds.append(kind)
            if kind not in ("signal", "put_signal", "wait_signal"):
                continue
            ref, value = (instruction[4], instruction[5]) if kind == "put_signal" else instruction[1:3]
            if not isinstance(ref, SignalRef) or not isinstance(value, int) or isinstance(value, bool):
                return None
            if kind == "wait_signal":
                if ref.rank != rank:
                    return None
                waits.append(value)
            else:
                key = (ref.rank, ref.key, ref.index)
                if ref.rank != 1 - rank or writers.setdefault(key, rank) != rank:
                    return None
                published.append(value)
                kind = "signal"
            actions.append((kind, ref, value))
        signals.append(tuple(actions))
        phases.append(_Phases(_Phase(index=index, op=kind) for index, kind in enumerate(kinds)))
    identity = manifest_digest({"contract_version": 1, "scalars": list(scalars), "proof": list(proof), "plans": descriptions})
    return StaticContract(
        identity,
        tuple(scalars),
        tuple(proof),
        None if not published else (min(published), max(published)),
        None if not waits else (min(waits), max(waits)),
        tuple(signals),
        tuple(phases),
    )
