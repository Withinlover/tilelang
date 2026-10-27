"""Timing events obey the native Session's synchronization and failure lifetime."""

import pytest

from testing.ascend.distributed.test_native_runtime import Bridge, Control, make_ticket
from tilelang.ascend.distributed.native_runtime import NativeSession


class TimingBridge(Bridge):
    def __init__(self):
        super().__init__()
        self.events = []

    def create_event(self):
        event = len(self.events) + 1
        self.events.append(event)
        self.calls.append(("create_event", event))
        return event

    def record_event(self, event, stream):
        self.calls.append(("record_event", event, stream))

    def event_elapsed_seconds(self, start, end):
        self.calls.append(("elapsed", start, end))
        return 0.00125

    def destroy_event(self, event):
        self.calls.append(("destroy_event", event))


def make_session():
    return NativeSession(rank=0, world_size=2, device=0, bootstrap="tcp://127.0.0.1:25100", control=Control(), bridge=TimingBridge())


def test_event_pair_reused_and_destroyed_only_after_stream_completion():
    session = make_session()
    session.prepare_device_timing()
    session.prepare_device_timing()
    assert session.bridge.events == [1, 2]
    ticket, _, _ = make_ticket(session)
    session.mark_running(ticket)
    with pytest.raises(RuntimeError, match="matching start"):
        session.end_device_timing()
    for _ in range(2):
        session.begin_device_timing()
        with pytest.raises(RuntimeError, match="prepare_device_timing"):
            session.begin_device_timing()
        session.launch_kernel(lambda: None)
        session.end_device_timing()
        with pytest.raises(RuntimeError, match="before reading"):
            session.device_elapsed_seconds()
        session.synchronize()
        assert session.device_elapsed_seconds() == 0.00125
    session.mark_complete(ticket)
    session.close()
    calls = [call[0] for call in session.bridge.calls if call[0] != "activate"]
    assert calls.count("create_event") == 2
    assert calls.count("record_event") == 4
    assert calls[-4:] == ["synchronize", "destroy_event", "destroy_event", "close"]
    for index, name in enumerate(calls):
        if name == "elapsed":
            assert calls[index - 1] == "synchronize"


def test_failed_timed_stream_retains_events_and_never_reports_elapsed():
    session = make_session()
    session.prepare_device_timing()
    ticket, _, _ = make_ticket(session)
    session.mark_running(ticket)
    session.begin_device_timing()
    session.end_device_timing()
    session.bridge.fail_sync = True
    with pytest.raises(RuntimeError, match="injected timeout"):
        session.synchronize()
    with pytest.raises(RuntimeError, match="before reading"):
        session.device_elapsed_seconds()
    with pytest.raises(RuntimeError, match="cannot close"):
        session.close()
    assert not any(call[0] in ("elapsed", "destroy_event", "close") for call in session.bridge.calls)


def test_timing_requires_explicit_setup_before_running():
    session = make_session()
    ticket, _, _ = make_ticket(session)
    session.mark_running(ticket)
    with pytest.raises(RuntimeError, match="prepare_device_timing"):
        session.begin_device_timing()
    with pytest.raises(RuntimeError, match="idle native Session"):
        session.prepare_device_timing()
