"""Bounded adapter shutdown must not leave a separate CANN runner alive."""

import json
from contextlib import suppress
from pathlib import Path
import signal
import subprocess
import sys
import time

import tilelang  # noqa: F401
from tilelang.ascend.distributed.cosim import _Worker, _process_identity


def test_unresponsive_worker_and_separate_native_group_are_both_stopped(tmp_path):
    native = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"
    worker_source = """
import json, signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
native = subprocess.Popen([sys.executable, '-c', sys.argv[2]], start_new_session=True, stdout=subprocess.PIPE, text=True)
assert native.stdout.readline().strip() == 'ready'
Path(sys.argv[1]).write_text(json.dumps({'native': native.pid}))
print('ready', flush=True)
time.sleep(60)
"""
    worker = _Worker.__new__(_Worker)
    worker.log = (tmp_path / "worker.log").open("w")
    worker.process = subprocess.Popen(
        [sys.executable, "-c", worker_source, str(tmp_path / "pids.json"), native],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=worker.log,
        text=True,
        start_new_session=True,
    )
    native_pid = None
    try:
        assert worker.process.stdout.readline().strip() == "ready"
        native_pid = json.loads((tmp_path / "pids.json").read_text())["native"]
        assert _process_identity(native_pid) is not None
        worker.close()
        assert worker.process.returncode == -signal.SIGKILL
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            stat = Path(f"/proc/{native_pid}/stat")
            if not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("native runner survived adapter shutdown")
        assert worker.process.stdin.closed and worker.process.stdout.closed and worker.log.closed
    finally:
        import os

        for pid in (worker.process.pid, native_pid):
            if pid is not None:
                with suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
        worker.process.wait(timeout=3)
        worker.log.close()
