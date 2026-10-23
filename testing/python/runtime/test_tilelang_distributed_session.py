"""Host preflight and resource ownership for the CPU-coordinated P0 profile."""

from concurrent.futures import ThreadPoolExecutor
import copy
import struct
from threading import Event

import numpy as np
import pytest

import tilelang.distributed as D
import tilelang.language as T
import tilelang.language.distributed as TD
from tilelang.distributed.abi import MAX_EPOCH, descriptor_nbytes, manifest_digest


class Program:
    def __init__(self, manifest, callback=None):
        self.manifest = manifest
        self.callback = callback
        self.calls = 0

    def run(self, **kwargs):
        self.calls += 1
        if self.callback:
            return self.callback(**kwargs)
        return {"passed": True, "base": kwargs["scalars"]["epoch0"]}


def make_session(*, callback=None):
    session = D.init()
    send = D.alloc_symmetric(session, key="Send", shape=(2, 4), dtype="float32", remote_access="read")
    ready = D.alloc_signals(session, key="Ready", shape=(2,))
    ch = D.channel(session, key="comm")
    manifest = {
        "schema_version": 1,
        "profile": "ascend950_cannsim_v1",
        "groups": [{"name": "world", "size": 2}],
        "channels": [{"name": "comm", "group": "world"}],
        "objects": [
            {
                "key": "Send",
                "param_index": 0,
                "group": "world",
                "kind": "symmetric",
                "dtype": "float32",
                "shape": [2, 4],
                "strides": [4, 1],
                "remote_access": "read",
                "alignment": 64,
                "physical_nbytes": 32,
            },
            {
                "key": "Ready",
                "param_index": 1,
                "group": "world",
                "kind": "signal",
                "dtype": "int32",
                "shape": [2],
                "strides": [16],
                "remote_access": "write",
                "alignment": 64,
                "physical_nbytes": 128,
            },
        ],
        "descriptor": {
            "param_index": 2,
            "parameter": "__comm_descriptor",
            "nbytes": descriptor_nbytes(2, 2),
            "abi_version": 1,
            "encoding": "cpu-model-v1",
        },
    }
    return session, send, ready, ch, Program(manifest, callback)


def prepare(session, ch, kernel, epochs=3):
    return D.prepare_launch(session, kernel, channel=ch, epochs=epochs, protocol="exchange-v1", parameters={"slots": 2})


def test_epochs_reuse_requires_complete_and_preserves_signal_values():
    session, send, ready, ch, kernel = make_session()
    ticket = prepare(session, ch, kernel)
    data = {0: np.ones((2, 4), dtype="float32"), 1: np.full((2, 4), 2, dtype="float32")}
    D.copy_in(send, data, ticket=ticket)
    np.testing.assert_array_equal(send.arrays[0], data[0])
    run = D.launch(kernel, ticket=ticket)
    with pytest.raises(RuntimeError, match="QUIESCENT"):
        prepare(session, ch, kernel)
    with pytest.raises(RuntimeError, match="PREPARED"):
        D.copy_in(send, data, ticket=ticket)
    assert D.complete(run) == {"passed": True, "base": 0}
    assert D.complete(run) == run.result
    ready.arrays[0][0] = 3  # A prior device publication, not host initialization.
    second = prepare(session, ch, kernel, epochs=2)
    assert (second.base, second.end, second.launch_id) == (3, 5, 2)
    assert ready.arrays[0][0] == 3
    with pytest.raises(ValueError, match="stale"):
        D.copy_in(send, data, ticket=ticket)
    assert D.complete(D.launch(kernel, ticket=second))["base"] == 3
    D.close(session)
    D.close(session)
    with pytest.raises(RuntimeError, match="QUIESCENT"):
        prepare(session, ch, kernel)


def test_signal_cells_are_aligned_separate_and_initialized_once():
    session, _, ready, ch, kernel = make_session()
    for array in ready.arrays.values():
        assert array.ctypes.data % 64 == 0
        assert array.strides == (64,)
        assert not array.any()
    ready.arrays[0][1] = 11
    assert ready.arrays[0][0] == 0 and not ready.arrays[1].any()
    ticket = prepare(session, ch, kernel)
    with pytest.raises(ValueError, match="signals are initialized only"):
        D.copy_in(ready, np.zeros(2, dtype="int32"), ticket=ticket, rank=0)


@pytest.mark.parametrize(
    "settings",
    [
        {"profile": "ascend950_mapped_mte_v1"},
        {"world_size": 3},
        {"device": 0},
        {"process_group": object()},
    ],
)
def test_uncertified_runtime_is_rejected(settings):
    with pytest.raises(D.UnsupportedProfileError):
        D.init(**settings)


def test_epoch_limit_refuses_wrap_without_consuming_ticket():
    session, _, _, ch, kernel = make_session()
    session._epoch = MAX_EPOCH - 1
    with pytest.raises(OverflowError, match="close and recreate"):
        prepare(session, ch, kernel, epochs=2)
    assert session.state == "QUIESCENT" and session._epoch == MAX_EPOCH - 1
    final = prepare(session, ch, kernel, epochs=1)
    assert final.end == MAX_EPOCH
    D.complete(D.launch(kernel, ticket=final))
    with pytest.raises(OverflowError):
        prepare(session, ch, kernel, epochs=1)


@pytest.mark.parametrize(("field", "value"), [("shape", [4, 2]), ("dtype", "float16"), ("remote_access", "write"), ("strides", [1, 2])])
def test_allocation_mismatch_fails_before_launch(field, value):
    session, _, _, ch, kernel = make_session()
    kernel.manifest["objects"][0][field] = value
    with pytest.raises(ValueError, match=f"mismatched {field}"):
        prepare(session, ch, kernel)
    assert kernel.calls == 0 and session.state == "QUIESCENT" and session._epoch == 0


def test_foreign_ticket_and_kernel_changes_cannot_rebind_resources():
    session, send, _, ch, kernel = make_session()
    ticket = prepare(session, ch, kernel)
    _, foreign, _, _, _ = make_session()
    with pytest.raises(ValueError, match="different session"):
        D.copy_in(foreign, np.ones((2, 4), dtype="float32"), ticket=ticket, rank=0)
    with pytest.raises(ValueError, match="prepared allocation"):
        D.launch(kernel, ticket=ticket, buffers={"Send": foreign})
    with pytest.raises(ValueError, match="epoch0"):
        D.launch(kernel, ticket=ticket, scalars={"epoch0": 99})
    kernel.manifest["objects"][0]["remote_access"] = "write"
    with pytest.raises(ValueError, match="manifest differs"):
        D.launch(kernel, ticket=ticket, buffers={"Send": send})
    assert kernel.calls == 0


def test_descriptor_has_stable_metadata_and_zero_address_records():
    session, _, _, ch, kernel = make_session()
    fingerprint = manifest_digest(kernel.manifest)
    ticket = prepare(session, ch, kernel)
    encoded = ticket.descriptors[1]
    fields = struct.unpack_from("<8sIIIIQQQQQ", encoded, 0)
    assert fields == (b"TDCOSIM\0", 1, 1, 2, 2, session.generation, 1, 0, 3, 0)
    record = struct.unpack_from("<IIQQII", encoded, 64 + 32)
    assert record == (0, 1, 0, 32, 1, 0)
    assert not encoded.flags.writeable
    reordered = {key: kernel.manifest[key] for key in reversed(kernel.manifest)}
    assert manifest_digest(reordered) == fingerprint
    D.complete(D.launch(kernel, ticket=ticket))
    second = prepare(session, ch, kernel)
    assert manifest_digest(kernel.manifest) == fingerprint
    assert not np.array_equal(second.descriptors[1], encoded)


def test_copy_in_validates_all_rank_inputs_before_modifying_any():
    session, send, _, ch, kernel = make_session()
    ticket = prepare(session, ch, kernel)
    with pytest.raises(ValueError, match="storage dtype"):
        D.copy_in(send, {0: np.ones((2, 4), dtype="float32"), 1: np.ones((2, 4), dtype="float16")}, ticket=ticket)
    assert not send.arrays[0].any() and not send.arrays[1].any()
    with pytest.raises(ValueError, match="rank="):
        D.copy_in(send, np.ones((2, 4), dtype="float32"), ticket=ticket)


def test_failure_retains_allocations_and_blocks_reuse():
    def fail(**kwargs):
        kwargs["buffers"]["Send"].arrays[0][0, 0] = 7
        raise RuntimeError("injected protocol deadlock")

    session, send, _, ch, kernel = make_session(callback=fail)
    ticket = prepare(session, ch, kernel)
    with pytest.raises(RuntimeError, match="protocol deadlock"):
        D.launch(kernel, ticket=ticket)
    assert session.state == "FAILED" and send.arrays[0][0, 0] == 7
    with pytest.raises(RuntimeError, match="QUIESCENT"):
        prepare(session, ch, kernel)
    with pytest.raises(RuntimeError, match="preserve"):
        D.close(session)


def test_running_session_rejects_concurrent_write_or_second_launch():
    started, release = Event(), Event()

    def block(**kwargs):
        started.set()
        if not release.wait(timeout=10):
            raise TimeoutError("test worker was not released")
        return {"passed": True}

    session, send, _, ch, kernel = make_session(callback=block)
    ticket = prepare(session, ch, kernel)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(D.launch, kernel, ticket=ticket)
        try:
            assert started.wait(timeout=10)
            with pytest.raises(RuntimeError, match="PREPARED"):
                D.launch(kernel, ticket=ticket)
            with pytest.raises(RuntimeError, match="PREPARED"):
                D.copy_in(send, np.ones((2, 4), dtype="float32"), rank=0, ticket=ticket)
            with pytest.raises(RuntimeError, match="RUNNING"):
                D.close(session)
        finally:
            release.set()
        D.complete(future.result(timeout=10))
    assert kernel.calls == 1 and session.state == "LOCAL_DONE"


def test_manifest_mutation_after_prepare_is_detected():
    session, _, _, ch, kernel = make_session()
    ticket = prepare(session, ch, kernel)
    old = copy.deepcopy(kernel.manifest)
    ticket.manifest["objects"][0]["shape"] = [8]
    assert kernel.manifest == old
    with pytest.raises(ValueError, match="prepared manifest was modified"):
        D.launch(kernel, ticket=ticket)


def test_descriptor_replacement_after_prepare_is_detected():
    session, _, _, ch, kernel = make_session()
    ticket = prepare(session, ch, kernel)
    ticket.descriptors[1] = ticket.descriptors[1].copy()
    ticket.descriptors[1][32] ^= 1  # Change launch ID without a new prepare.
    with pytest.raises(ValueError, match="CPU descriptor was modified"):
        D.launch(kernel, ticket=ticket)
    assert kernel.calls == 0 and session.state == "PREPARED"


def test_none_returning_synchronous_adapter_can_complete():
    session, _, _, ch, kernel = make_session(callback=lambda **kwargs: None)
    ticket = prepare(session, ch, kernel)
    assert D.complete(D.launch(kernel, ticket=ticket)) is None
    assert session.state == "LOCAL_DONE"


def test_real_frontend_manifest_binds_without_a_second_signature():
    group = TD.GroupSpec("world", 2)
    channel = TD.ChannelSpec("comm", group)

    @T.prim_func
    def publish(
        Send: TD.SymBuffer((2, 4), "float32", group=group, remote_access="read"),
        Ready: TD.SignalBuffer((2,), group=group),
    ):
        rank = TD.rank(group)
        TD.signal(TD.remote(Ready, 1 - rank)[rank], 1, channel=channel, release_regions=[Send])
        TD.quiet(channel=channel)

    session, _, _, ch, _ = make_session()
    ticket = prepare(session, ch, publish, epochs=1)
    descriptor = publish.buffer_map[publish.params[-1]]
    assert descriptor.dtype == "uint8"
    assert int(descriptor.shape[0]) == ticket.descriptors[0].size
    assert ticket.manifest["descriptor"]["param_index"] == len(publish.params) - 1
