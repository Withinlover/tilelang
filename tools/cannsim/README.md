# CANN Simulator companion for distributed P0

This add-only companion makes this branch's Ascend 950 P0 DSL co-simulation
reproducible with a separately cloned cannsim-helper. It contains the CPU
communication protocol model, a persistent JSON worker, the fixed 64 × 64
BF16 GEMM adapter, and focused tests. It is a pure-Cube compatibility package
for this branch, not a release of the complete locally modified helper.

The exact upstream base is
[`2eb1ce5a79a76cc3f3e1c0eb4b04c6e9f6f694cf`](https://gitcode.com/Withinlover/cannsim-helper/commit/2eb1ce5a79a76cc3f3e1c0eb4b04c6e9f6f694cf).
[manifest.json](manifest.json) records that revision, the patch SHA256,
per-file SHA256 values, provenance, and supported ABI. No existing upstream
file is changed. No object, simulator binary, machine-specific input, or
unrelated Flash Attention/MIX repair is shipped.

## Install in a separate helper checkout

Run these commands from the TileLang repository root. The helper repository
must be accessible to your account. The commands assume a sibling checkout;
choose another location by changing `CANNSIM_HELPER_ROOT`.

```bash
export CANNSIM_HELPER_ROOT="$(pwd)/../cannsim-helper"
git clone git@gitcode.com:Withinlover/cannsim-helper.git "$CANNSIM_HELPER_ROOT"
git -C "$CANNSIM_HELPER_ROOT" checkout --detach 2eb1ce5a79a76cc3f3e1c0eb4b04c6e9f6f694cf
python3 tools/cannsim/apply_companion.py --helper-root "$CANNSIM_HELPER_ROOT" --check
python3 tools/cannsim/apply_companion.py --helper-root "$CANNSIM_HELPER_ROOT"
python3 tools/cannsim/apply_companion.py --helper-root "$CANNSIM_HELPER_ROOT" --verify
export CANNSIM_ROOT="$CANNSIM_HELPER_ROOT"
```

The installer uses Python's standard library and Git. It checks the exact
base, patch hash, add-only file list, clean checkout, and `git apply --check`
before applying. It refuses existing target files and never resets or
replaces local work. A repeated apply is refused; use `--verify` to check an
installed copy. Verification also rejects changes to upstream tracked files
but permits unrelated untracked logs. Ignored environments such as `.venv/`
can already exist.

Prepare the helper's environment with Python 3.11–3.13 and NumPy 2.2.6. This
can differ from TileLang's Python; the JSON worker uses the helper's `.venv`.

```bash
(cd "$CANNSIM_HELPER_ROOT" && PYTHON=python3.12 ./setup_env.sh)
(cd "$CANNSIM_HELPER_ROOT" && .venv/bin/python -m unittest discover -s tests -p 'test_cosim_*.py' -v)
```

The 49 companion tests cover protocol faults, accuracy and END failures,
pure-Cube core selection, JSON output isolation, timeout/process cleanup,
and the off-only compatibility adapter. They do not require CANN or a device.
The installer has eight additional tests in this repository:

```bash
python3 -m unittest discover -s tools/cannsim -p 'test_*.py' -v
```

For actual segment recordings, install the complete CANN 9.1.0 Toolkit and
Ascend950PR_9589 simulator, CMake, and a C++17 compiler, then source your
installation's `set_env.sh`. Follow the
[P0 local and hardware validation guide](../../docs/testing/distributed_p0.md)
for the DSL example, explicit CANN compatibility profile, and artifact checks.
`cosim.worker` is used directly by TileLang; this package does not add a new
command to the helper's existing `run.py` CLI.

## Supported boundary and ABI

The P0 segment is one pure-Cube launch: BF16 `A[64,64]` and `B[64,64]`, FP32
`C[64,64]`, `C = A @ B`, one block, tiling key zero. The object reads three
pointers at argument byte offsets 0, 8, and 16. The existing helper host
passes a 32-byte argument packet; its final reserved word is not read by this
three-pointer object. This specific compatibility was checked with the newly
compiled P0 object; it is not a general ABI adaptation for arbitrary objects.

P0 explicitly requests `expected_cores=("AIC",)`. The helper's existing
mixed-core default remains unchanged for other callers. AIC END and both
Core 0 trace exports remain required; skipping AIV END is justified by the
pure-Cube object's launch kind. The adapter retains ordinary output accuracy
checks at `rtol=atol=1e-5` and fails on native errors or incomplete traces.

`cosim/runtime.py` adapts the pinned helper's original two-argument
`record_environment` function. It supports only `mix_scheduler_fix="off"`
and records `not_applied`; `auto`, `required`, and attempts to claim an applied
repair are rejected. Pure-Cube execution needs no MIX scheduler repair. This
package includes none of the separate MIX/FA changes in the development
checkout and does not certify mixed-kernel simulation.

Communication is a CPU protocol model. Compute segments run serially in
independent single-device CANN processes; protocol ticks and simulator cycles
are separate time domains. Passing this test does not establish real peer
addressing, cache visibility, interconnect behavior, or multi-device speed.

## Provenance and license

The patch contains only the newly authored dual-rank protocol/worker/segment
code, compatibility adapter, tests, and provenance/license files. Formatting
and the segment runtime imports differ from the local extension; source
hashes and adaptations are recorded in the manifest. `cosim/LICENSE` copies
the TileLang repository license verbatim for these additions.

The pinned upstream helper tree has no license file. This companion neither
grants nor infers a license for the existing helper; its access and applicable
terms remain separate. No upstream helper source is copied into the patch.

## Local validation record (2026-10-23)

The packaged patch was applied to a second clean checkout at the exact base;
all upstream tracked files remained unchanged. Its 49 companion tests passed,
and all eight installer tests passed. TileLang's co-simulation and example
tests were also run against this clean companion checkout: 34 passed.

A real worker recording then used the newly compiled P0 pure-Cube object,
with kernel SHA256
`92d04e122accfdd7a9419f178d823041b89d4c886b846ecb83c87a602859c763`.
The worker exited zero, emitted exactly one ready and one execute JSON
response, and produced a complete AIC END trace plus both trace views.
All 4,096 FP32 outputs passed the helper's unchanged `1e-5` tolerances with
zero mismatches and maximum absolute error `2.384185791015625e-7`.
An independent FP64 recomputation gave maximum absolute error
`1.4156103134155273e-7`. The output hash exactly matched the original DSL
validation segment. The simulator repair status was `not_applied`.

This is an ABI/runtime compatibility smoke test of the portable package using
an already compiled object from the same P0 compiler/toolchain. It does not
replace the full DSL compilation and dual-rank validation described in the
[P0 validation report](../../docs/design/distributed_p0_implementation_validation.md).
