"""Hardware-free coverage of diagnostic isolation, failure reporting and cleanup."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from contextlib import suppress

import pytest

from examples.distributed import diagnose_native_runtime as diagnostic


@pytest.fixture
def probe_modules(monkeypatch):
    events = []
    state = {"actual": [2.0, 4.0], "device_error": None}

    class Tensor:
        def __add__(self, other):
            events.append("add")
            return self

        def cpu(self):
            events.append("cpu")
            return self

        def tolist(self):
            return state["actual"]

    def tensor(values, **kwargs):
        assert values == [1.0, 2.0] and kwargs["device"] == "npu:3"
        events.append("tensor")
        return Tensor()

    def set_device(device):
        assert device == 3
        events.append("set_device")
        if state["device_error"]:
            raise RuntimeError(state["device_error"])

    def synchronize():
        events.append("sync")

    class Bridge:
        def __init__(self, **kwargs):
            assert kwargs == {"shmem_root": "/private/shmem", "shmem_library": None}
            events.append("bridge_load")
            self.lib = SimpleNamespace(_name="/private/libbridge.so")
            self.fingerprint = {"shmem_revision": "test"}

        def device_identity(self, device):
            assert device == 3
            events.append("identity")
            return {"physical_device": 3, "soc": "test-only"}

        def initialize(self, **kwargs):
            pytest.fail("diagnostic must not initialize SHMEM")

    modules = {
        "torch": SimpleNamespace(tensor=tensor, npu=SimpleNamespace(set_device=set_device, synchronize=synchronize), float32="float32"),
        "torch_npu": SimpleNamespace(),
        "tilelang.ascend.distributed.native_runtime": SimpleNamespace(ShmemBridge=Bridge),
    }

    def load(name):
        events.append(f"import:{name}")
        monkeypatch.setitem(sys.modules, name, modules[name])
        return modules[name]

    monkeypatch.setattr(diagnostic.importlib, "import_module", load)
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for name in ("torch", "torch_npu", "tilelang"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return events, state


@pytest.mark.parametrize("mode", diagnostic.MODES)
def test_probe_orders_and_plain_never_imports_tilelang(tmp_path, probe_modules, mode):
    events, _ = probe_modules
    path = tmp_path / "probe.json"
    assert diagnostic.probe(mode, 3, path, shmem_root="/private/shmem") == 0
    report = json.loads(path.read_text())
    assert report["status"] == "passed"
    assert report["phase"] == "complete"
    torch_imports = ["import:torch", "import:torch_npu"]
    calculation = ["tensor", "add", "sync", "cpu"]
    bridge = ["import:tilelang.ascend.distributed.native_runtime", "bridge_load"]
    if mode == "plain":
        assert events == torch_imports + ["set_device"] + calculation
        assert "tilelang" not in report["modules"]
    elif mode == "bridge-first":
        assert events == bridge + torch_imports + ["set_device"] + calculation + ["identity"]
    else:
        assert events == torch_imports + ["set_device"] + calculation + bridge + calculation + ["identity"]


@pytest.mark.parametrize("failure", ["set_device", "wrong_result"])
def test_probe_preserves_exact_failure_phase(tmp_path, probe_modules, failure):
    events, state = probe_modules
    if failure == "set_device":
        state["device_error"] = "SetDevice error code is 507033; private details"
    else:
        state["actual"] = [0.0, 0.0]
    path = tmp_path / "probe.json"
    assert diagnostic.probe("plain", 3, path) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert report["phase"] == ("set_device" if failure == "set_device" else "compute.verify")
    assert report["error"]["type"] == "RuntimeError"
    assert "Traceback" in report["error"]["traceback"]
    assert events.count("tensor") == (0 if failure == "set_device" else 1)


def test_bridge_config_failure_does_not_import_any_runtime(tmp_path, probe_modules):
    events, _ = probe_modules
    path = tmp_path / "probe.json"
    assert diagnostic.probe("bridge-first", 3, path) == 1
    assert json.loads(path.read_text())["phase"] == "bridge_config"
    assert events == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux diagnostic subprocesses")
def test_cli_fresh_processes_continue_after_failure_and_limit_summary(tmp_path):
    # Synthetic modules test the supervisor, not NPU computation. They are
    # available only to these child processes, with no TileLang installation.
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "torch.py").write_text(
        """
import sys
assert 'tilelang' not in sys.modules
__version__ = 'test-torch'
float32 = 'float32'
class Npu:
    def set_device(self, device):
        if device == 0:
            raise RuntimeError('error code is 507033; /secret/private-machine/234567 EE1018 PID:123456 12:34:56.654321')
    def synchronize(self): pass
npu = Npu()
class Tensor:
    def __add__(self, other): return self
    def cpu(self): return self
    def tolist(self): return [2.0, 4.0]
def tensor(*args, **kwargs): return Tensor()
"""
    )
    (modules / "torch_npu.py").write_text("__version__ = 'test-torch-npu'\n")
    directory = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, str(Path(diagnostic.__file__)), "--devices", "0", "1", "--modes", "plain", "--output-dir", str(directory)],
        env={**os.environ, "PYTHONPATH": str(modules), "SHMEM_ROOT": "/secret/private-shmem"},
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    run = next(directory.iterdir())
    report = json.loads((run / "report.json").read_text())
    assert not report["passed"]
    probes = report["probes"]
    assert [item["status"] for item in probes] == ["failed", "passed"]
    assert probes[0]["phase"] == "set_device"
    assert probes[0]["pid"] != probes[1]["pid"] != os.getpid()
    assert probes[0]["error_codes"] == ["507033", "EE1018"]
    summary_text = (run / "summary.json").read_text()
    assert "/secret/" not in summary_text
    assert str(modules) not in summary_text
    assert "traceback" not in json.loads(summary_text)["probes"][0]
    assert json.loads(summary_text)["probes"][0]["versions"]["torch_npu"] == "test-torch-npu"
    assert "private-machine" in (run / "device0-plain.log").read_text()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process group cleanup")
@pytest.mark.parametrize("exit_code,write_report", [(2, True), (0, False)])
def test_supervisor_rejects_crash_after_pass_or_missing_report(tmp_path, exit_code, write_report):
    path = tmp_path / "probe.json"
    report = {"mode": "plain", "device": 0, "status": "passed", "phase": "complete"}
    code = f"from pathlib import Path; import sys; Path({str(path)!r}).write_text({json.dumps(report)!r}); sys.exit({exit_code})"
    if not write_report:
        code = "pass"
    result = diagnostic.run_probe([sys.executable, "-c", code], path, path.with_suffix(".log"), mode="plain", device=0, timeout=5)
    assert result["status"] == "failed"
    assert result["returncode"] == exit_code


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process group cleanup")
def test_timeout_overrides_pass_and_kills_term_ignoring_descendant(tmp_path):
    path = tmp_path / "probe.json"
    child_pid_path = tmp_path / "child.pid"
    child = (
        "import signal,time,os; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"Path({str(child_pid_path)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    report = {"mode": "plain", "device": 0, "status": "passed", "phase": "complete"}
    leader = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"Path({str(path)!r}).write_text({json.dumps(report)!r}); time.sleep(120)"
    )
    pid = None
    try:
        result = diagnostic.run_probe([sys.executable, "-c", leader], path, path.with_suffix(".log"), mode="plain", device=0, timeout=1)
        assert result["status"] == "timeout" and result["timed_out"]
        assert child_pid_path.exists()
        pid = int(child_pid_path.read_text())
        stat = Path(f"/proc/{pid}/stat")
        deadline = time.monotonic() + 5
        while stat.exists() and stat.read_text().split()[2] != "Z" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert not stat.exists() or stat.read_text().split()[2] == "Z"
    finally:
        if pid is not None:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
