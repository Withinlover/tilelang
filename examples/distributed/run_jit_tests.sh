#!/usr/bin/env bash
# Validate the distributed JIT on an already configured two-device server.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: examples/distributed/run_jit_tests.sh [example_jit.py options]

Run CPU regressions, then launch two Ascend workers for the A -> B -> A JIT
example. Run this on a Linux server with TileLang built with USE_ASCEND=ON,
pytest, torch, CANN and SHMEM configured. No dependencies are installed or built.

Environment:
  SHMEM_ROOT       Required: pinned SHMEM source checkout.
  SHMEM_LIBRARY    Optional: path to the matching libshmem.so.
  PYTHON_BIN       Python executable (default: python3).
  OUTPUT_DIR       Parent directory for a unique run directory
                   (default: <repo>/output/distributed/jit-validation).
  TIMEOUT_SECONDS  Positive timeout for each stage (default: 900).

Example options are forwarded unchanged, including:
  --sdk-profile {native,cann-9.1}
  --bootstrap-port 29610 --control-port 29611 --heap-mib 64

The script manages --output-dir; set OUTPUT_DIR instead. Output includes
git_commit.txt, git_submodules.txt, commands.txt, cpu-tests.log, torchrun.log,
device/rank{0,1}/result.json, EXIT_CODE and summary.txt. Stage failures preserve
their exit code; GNU timeout returns 124 on timeout (or 137 after forced kill).
EOF
}

for arg in "$@"; do
    case "$arg" in
        -h|--help) usage; exit 0 ;;
        --output-dir|--output-dir=*)
            printf 'Use OUTPUT_DIR for the output parent; --output-dir is managed by this script.\n' >&2
            exit 2
            ;;
    esac
done

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../.." && pwd)
output_root=${OUTPUT_DIR:-"$repo_root/output/distributed/jit-validation"}
mkdir -p -- "$output_root"
output_root=$(cd -- "$output_root" && pwd)
run_dir=$(mktemp -d "$output_root/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
stage=preflight
cpu_exit=NOT_RUN
device_exit=NOT_RUN

finish() {
    local exit_code=$?
    trap - EXIT
    local status=FAILED
    if [[ "$exit_code" == 0 ]]; then
        status=PASSED
    fi
    printf '%s\n' "$exit_code" > "$run_dir/EXIT_CODE"
    {
        printf 'STATUS=%s\nEXIT_CODE=%s\nLAST_STAGE=%s\n' "$status" "$exit_code" "$stage"
        printf 'CPU_EXIT_CODE=%s\nTORCHRUN_EXIT_CODE=%s\n' "$cpu_exit" "$device_exit"
        printf 'OUTPUT_DIR=%s\n' "$run_dir"
        printf 'Rank-local accuracy reports: device/rank0/result.json and device/rank1/result.json\n'
        printf 'These reports are independent local checks, not an aggregated global accuracy report.\n'
    } > "$run_dir/summary.txt"
    cat "$run_dir/summary.txt"
    exit "$exit_code"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Validation output: %s\n' "$run_dir"
if [[ -z "${SHMEM_ROOT:-}" ]]; then
    printf 'SHMEM_ROOT must point to the configured SHMEM source checkout.\n' >&2
    exit 2
fi
if [[ ! -d "$SHMEM_ROOT" ]]; then
    printf 'SHMEM_ROOT is not a directory: %s\n' "$SHMEM_ROOT" >&2
    exit 2
fi
# Resolve caller-relative paths before changing into the repository root.
export SHMEM_ROOT
SHMEM_ROOT=$(cd -- "$SHMEM_ROOT" && pwd)
if [[ -n "${SHMEM_LIBRARY:-}" && "$SHMEM_LIBRARY" != /* ]]; then
    export SHMEM_LIBRARY="$PWD/$SHMEM_LIBRARY"
fi
python_bin=${PYTHON_BIN:-python3}
python_executable=$(command -v -- "$python_bin") || {
    printf 'Python executable not found: %s\n' "$python_bin" >&2
    exit 2
}
if [[ "$python_executable" != /* ]]; then
    python_executable="$PWD/$python_executable"
fi
timeout_seconds=${TIMEOUT_SECONDS:-900}
if [[ ! "$timeout_seconds" =~ ^[1-9][0-9]*$ ]]; then
    printf 'TIMEOUT_SECONDS must be a positive integer.\n' >&2
    exit 2
fi
if ! timeout --version >/dev/null 2>&1; then
    printf 'GNU timeout is required (Linux coreutils).\n' >&2
    exit 2
fi

cd -- "$repo_root"
git rev-parse HEAD > "$run_dir/git_commit.txt"
git submodule status --recursive > "$run_dir/git_submodules.txt"
git status --short > "$run_dir/git_status.txt"
export PYTHONUNBUFFERED=1

run_stage() {
    local name=$1
    local logfile=$2
    shift 2
    local -a pipeline_status
    local exit_code
    stage=$name
    printf '\nRunning %s; log: %s\n' "$name" "$logfile"
    {
        printf '%s:' "$name"
        printf ' %q' timeout --kill-after=15s "${timeout_seconds}s" "$@"
        printf '\n'
    } >> "$run_dir/commands.txt"
    set +e
    timeout --kill-after=15s "${timeout_seconds}s" "$@" 2>&1 | tee "$logfile"
    pipeline_status=("${PIPESTATUS[@]}")
    set -e
    exit_code=${pipeline_status[0]}
    if [[ "$exit_code" == 0 && "${pipeline_status[1]}" != 0 ]]; then
        exit_code=${pipeline_status[1]}
    fi
    if [[ "$name" == cpu-tests ]]; then
        cpu_exit=$exit_code
    else
        device_exit=$exit_code
    fi
    return "$exit_code"
}

run_stage cpu-tests "$run_dir/cpu-tests.log" "$python_executable" -m pytest -q \
    testing/ascend/distributed/test_native_jit.py \
    testing/ascend/distributed/test_native_runtime.py \
    testing/ascend/distributed/test_native_plan.py

run_stage torchrun "$run_dir/torchrun.log" "$python_executable" -m torch.distributed.run \
    --standalone --nproc_per_node=2 --module examples.distributed.example_jit \
    "$@" --output-dir "$run_dir/device"

stage=collect-results
for rank in 0 1; do
    if [[ ! -s "$run_dir/device/rank$rank/result.json" ]]; then
        printf 'Missing rank %s result after successful launcher exit. Inspect torchrun.log.\n' "$rank" >&2
        exit 1
    fi
done
stage=completed
