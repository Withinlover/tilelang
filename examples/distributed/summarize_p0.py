"""Export bounded, shareable P0 reports without importing TileLang or using NPUs.

Only known numeric fields, enums, version strings and commit hashes are copied.
Raw commands, paths, host names, errors and per-iteration arrays stay local.
This is a summary of existing evidence, never a new acceptance decision.
"""

import argparse
import json
import math
from pathlib import Path
import re
import statistics


MODES = ("hccl", "p0a", "p0b")
STAGES = ("cpu", "compile", "device", "performance")
OUTER = ("prepare", "staging", "staging_sync", "collective_submit", "launch", "complete", "local_gemm_submit", "final_sync", "readback")
NATIVE = (
    "validate_bindings",
    "plan_validation",
    "plan_preflight",
    "build_or_cache_validation",
    "build_preflight",
    "bind_and_submit",
    "stream_sync",
    "report_finalize",
    "total",
)
CASES = frozenset(
    (
        "phased-put",
        "phased-get",
        "phased-gemm",
        "p0a-put",
        "p0a-get",
        "p0a-vector",
        "p0a-hot-slot",
        "p0a-unused-slot",
        "p0a-negative",
        "p0b-gemm",
        "p0b-hot-slot",
        "p0b-unused-slot",
        "p0b-negative",
    )
)
ERROR_TYPES = frozenset(
    (
        "AssertionError",
        "RuntimeError",
        "ValueError",
        "TypeError",
        "TimeoutError",
        "FileNotFoundError",
        "ModuleNotFoundError",
        "ImportError",
        "OSError",
        "KeyboardInterrupt",
        "CalledProcessError",
        "ChildProcessError",
        "RunFailedError",
        "MemoryError",
    )
)
PACKAGES = ("tilelang", "torch", "torch_npu", "apache-tvm-ffi", "pytest")
REPORT_KIND = "p0_shareable_summary"


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _number(value):
    return value if type(value) in (int, float) and abs(value) <= 1e100 and math.isfinite(value) else None


def _integer(value):
    return value if type(value) is int and -(2**63) <= value < 2**63 else None


def _boolean(value):
    return value if type(value) is bool else None


def _enum(value, allowed):
    return value if isinstance(value, str) and value in allowed else None


def _ms(value):
    value = _number(value)
    return value * 1000 if value is not None and value >= 0 else None


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or numerator <= 0 or denominator <= 0:
        return None
    value = _number(numerator / denominator)
    return value if value is not None and value > 0 else None


def _hash(value):
    if isinstance(value, dict):
        value = value.get("output") if value.get("exit_code") == 0 else None
    return value.lower() if isinstance(value, str) and re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", value) else None


def _version(value):
    # Arbitrary local-version labels can contain build-host/user names.
    pattern = r"\d+(?:\.\d+){0,3}(?:(?:a|b|rc)\d+)?(?:\.(?:post|dev)\d+)?(?:\+(?:cpu|npu|cu\d+))?"
    return value if isinstance(value, str) and len(value) <= 64 and re.fullmatch(pattern, value) else None


def _error_types(*values):
    found = set()
    for value in values:
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, dict):
                item = item.get("error_type", item.get("error"))
            if isinstance(item, str):
                name = item.split(":", 1)[0]
                if name in ERROR_TYPES:
                    found.add(name)
    return sorted(found)


def _timing(value):
    value = _mapping(value)
    samples = value.get("samples_seconds")
    samples = samples if isinstance(samples, list) and samples else None
    valid = samples is not None and all(_number(item) is not None and item >= 0 for item in samples)
    count = _integer(value.get("count"))
    median = _number(value.get("median_seconds"))
    p95 = _number(value.get("p95_seconds"))
    if valid:
        if count is None:
            count = len(samples)
        if median is None:
            median = statistics.median(samples)
        if p95 is None:
            ordered = sorted(samples)
            index = (len(ordered) - 1) * 0.95
            lower, upper = math.floor(index), math.ceil(index)
            p95 = ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
    return {"samples": count, "median_ms": _ms(median), "p95_ms": _ms(p95)}


def _max_ms(value):
    values = value[:2] if isinstance(value, list) else [value]
    converted = [_ms(item) for item in values]
    return max(converted) if converted and all(item is not None for item in converted) else None


def _numerics(checks):
    checks = [item for item in checks if isinstance(item, dict)]
    errors = [_number(item.get("max_abs_error")) for item in checks]
    errors = [value for value in errors if value is not None and value >= 0]
    mismatch = [_integer(item.get("mismatches")) for item in checks]
    nonfinite = [_integer(item.get("nonfinite")) for item in checks]
    return {
        "checks": len(checks) if checks else None,
        "failed_checks": sum(item.get("passed") is False for item in checks)
        if checks and all(type(item.get("passed")) is bool for item in checks)
        else None,
        "max_abs_error": max(errors) if errors else None,
        "max_mismatches": max(mismatch) if mismatch and all(value is not None for value in mismatch) else None,
        "max_nonfinite": max(nonfinite) if nonfinite and all(value is not None for value in nonfinite) else None,
    }


def _performance(item, fallback_ranks):
    summary = _mapping(item.get("summary"))
    mode = item["mode"]
    process = _mapping(item.get("process"))
    ranks = summary.get("ranks")
    ranks = ranks if isinstance(ranks, list) and ranks else fallback_ranks.get(mode, [])
    ranks = [_mapping(rank) for rank in ranks[:2]]
    checks = [check for rank in ranks for check in rank.get("checks", []) if isinstance(check, dict)]
    stages, native = _mapping(summary.get("stages")), _mapping(summary.get("native_run"))
    params = {key: _integer(summary.get(key)) for key in ("rounds", "slots", "iterations", "warmup", "tile")}
    params.update(dtype=_enum(summary.get("dtype"), {"bfloat16", "float32"}), output_dtype=_enum(summary.get("output_dtype"), {"float32"}))
    workload = _mapping(summary.get("workload"))
    # Old benchmark schema 1 has the same hard-coded batching, but does not
    # record it. Leave the field absent instead of inferring missing evidence.
    params.update(
        communication_schedule=_enum(workload.get("communication_schedule"), {"one_batched_allgather", "per_round_ready_ack"}),
        payload_batches_per_peer=_integer(workload.get("payload_batches_per_peer")),
        logical_peer_payload_bytes=_integer(workload.get("logical_peer_payload_bytes")),
    )
    return {
        "status": _enum(item.get("status"), {"passed", "failed", "skipped"}),
        "reported_passed": _boolean(summary.get("passed")),
        "source_schema_version": _integer(summary.get("schema_version")),
        "exit_code": _integer(process.get("exit_code")),
        "timed_out": _boolean(process.get("timed_out")),
        "device_executed": _boolean(summary.get("device_executed")),
        "parameters": params,
        "numerics": _numerics(checks),
        "execution": _timing(summary.get("execution")),
        "end_to_end": _timing(summary.get("end_to_end")),
        "first_rank_max_ms": {
            key: _max_ms(summary.get(key + "_seconds")) for key in ("first_build_or_cache_load", "first_execution", "first_end_to_end")
        },
        "stages_median_ms": {key: _ms(_mapping(stages.get(key)).get("median_seconds")) for key in OUTER},
        "native_median_ms": {key: _ms(_mapping(native.get(key)).get("median_seconds")) for key in NATIVE},
        "stage_availability": "available"
        if stages
        else "legacy_no_stage_timings"
        if summary.get("schema_version") == 1
        else "not_recorded",
        "error_types": _error_types(item.get("error"), item.get("errors"), summary.get("error"), *[rank.get("error") for rank in ranks]),
    }


def compact_report(report, *, provenance=None, fallback_ranks=None):
    """Whitelist existing evidence into a bounded, non-certifying summary."""
    provenance, fallback_ranks = _mapping(provenance), _mapping(fallback_ranks)
    report = _mapping(report)
    if report.get("report_kind") == REPORT_KIND:
        return _sanitize_existing(report)
    results = report.get("results", [])
    results = [item for item in results if isinstance(item, dict)] if isinstance(results, list) else []
    if report.get("mode") in MODES:
        results = [{"stage": "performance", "mode": report["mode"], "summary": report}]
    performance = {mode: None for mode in MODES}
    cases, cpu, counts = [], None, {stage: {key: 0 for key in ("total", "passed", "failed", "skipped", "unknown")} for stage in STAGES}
    seen = set()
    for item in results:
        stage = _enum(item.get("stage"), STAGES)
        if stage is None:
            continue
        status = _enum(item.get("status"), {"passed", "failed", "skipped"})
        counts[stage]["total"] += 1
        counts[stage][status or "unknown"] += 1
        if stage == "performance" and item.get("mode") in MODES:
            performance[item["mode"]] = _performance(item, fallback_ranks)
        elif stage == "cpu":
            cpu = {key: _integer(_mapping(item.get("pytest")).get(key)) for key in ("tests", "failures", "errors", "skipped")}
        elif stage in {"compile", "device"}:
            case = _mapping(item.get("case"))
            name = _enum(case.get("name"), CASES)
            if name is None or (stage, name) in seen:
                continue
            seen.add((stage, name))
            process = _mapping(item.get("process"))
            rank_codes = item.get("rank_exit_codes")
            cases.append(
                {
                    "stage": stage,
                    "case": name,
                    "status": status,
                    "parameters": {key: _integer(case.get(key)) for key in ("rounds", "slots", "launches", "delay_ms")},
                    "fault": _enum(case.get("fault"), {"none", "corrupt_send"}),
                    "exit_code": _integer(process.get("exit_code")),
                    "timed_out": _boolean(process.get("timed_out")),
                    "rank_exit_codes": [_integer(code) for code in rank_codes[:2]] if isinstance(rank_codes, list) else None,
                    "numerics": _numerics(item.get("accuracy", []) or []),
                    "error_types": _error_types(item.get("errors"), item.get("rank_errors")),
                }
            )
    baseline = performance["hccl"]
    for value in performance.values():
        if value is None:
            continue
        value["ratios"] = {}
        for boundary in ("execution", "end_to_end"):
            original = baseline[boundary]["median_ms"] if baseline else None
            candidate = value[boundary]["median_ms"]
            value["ratios"][boundary] = {
                "hccl_over_mode": _ratio(original, candidate),
                "slowdown_x": _ratio(candidate, original),
            }
    stages = report.get("requested_stages", [])
    return {
        "schema_version": 3,
        "report_kind": REPORT_KIND,
        "evidence_scope": "summary_of_existing_evidence_not_a_new_acceptance_decision",
        "new_device_execution": False,
        "source_schema_version": _integer(report.get("schema_version")),
        "commit": _hash(provenance.get("commit")),
        "shmem_revision": _hash(provenance.get("shmem_revision")),
        "versions": {key: _version(_mapping(provenance.get("packages")).get(key)) for key in PACKAGES},
        "reported_passed": _boolean(report.get("passed")),
        "reported_device_acceptance": _enum(report.get("device_acceptance"), {"not_run", "passed_requested_cases", "failed_or_incomplete"}),
        "requested_stages": [stage for stage in STAGES if stage in stages],
        "stage_counts": counts,
        "cpu": cpu,
        "cases": cases,
        "performance": performance,
        "error_types": _error_types(
            report.get("error"),
            *[item["error_types"] for item in cases],
            *[item["error_types"] for item in performance.values() if item is not None],
        ),
    }


def _sanitize_existing(report):
    """Re-render a shareable-only copy through the same strict whitelist.

    Reconstructing source-shaped *summary fields* does not reconstruct evidence:
    no rank arrays or passed checks are invented, and assess rejects schema 3.
    """

    def seconds(value):
        value = _number(value)
        return value / 1000 if value is not None and value >= 0 else None

    def numerics(value):
        value = _mapping(value)
        return {
            key: _number(value.get(key)) if key == "max_abs_error" else _integer(value.get(key))
            for key in ("checks", "failed_checks", "max_abs_error", "max_mismatches", "max_nonfinite")
        }

    raw = {
        "schema_version": report.get("source_schema_version"),
        "passed": report.get("reported_passed"),
        "device_acceptance": report.get("reported_device_acceptance"),
        "requested_stages": report.get("requested_stages", []),
        "results": [],
    }
    original_cases = {}
    for row in report.get("cases", [])[:26]:
        row = _mapping(row)
        stage, name = _enum(row.get("stage"), {"compile", "device"}), _enum(row.get("case"), CASES)
        if stage is None or name is None:
            continue
        original_cases[stage, name] = row
        raw["results"].append(
            {
                "stage": stage,
                "status": row.get("status"),
                "case": {**_mapping(row.get("parameters")), "name": name, "fault": row.get("fault")},
                "process": {"exit_code": row.get("exit_code"), "timed_out": row.get("timed_out")},
                "rank_exit_codes": row.get("rank_exit_codes"),
            }
        )
    for mode in MODES:
        item = _mapping(_mapping(report.get("performance")).get(mode))
        if not item:
            continue
        params = _mapping(item.get("parameters"))
        summary = {
            **params,
            "schema_version": item.get("source_schema_version"),
            "passed": item.get("reported_passed"),
            "device_executed": item.get("device_executed"),
            "workload": params,
        }
        for boundary in ("execution", "end_to_end"):
            metric = _mapping(item.get(boundary))
            summary[boundary] = {
                "count": metric.get("samples"),
                "median_seconds": seconds(metric.get("median_ms")),
                "p95_seconds": seconds(metric.get("p95_ms")),
            }
        for field, keys, source in (("stages", OUTER, "stages_median_ms"), ("native_run", NATIVE, "native_median_ms")):
            values = _mapping(item.get(source))
            summary[field] = {key: {"median_seconds": seconds(values.get(key))} for key in keys if seconds(values.get(key)) is not None}
        for key in ("first_build_or_cache_load", "first_execution", "first_end_to_end"):
            summary[key + "_seconds"] = seconds(_mapping(item.get("first_rank_max_ms")).get(key))
        raw["results"].append(
            {
                "stage": "performance",
                "mode": mode,
                "status": item.get("status"),
                "summary": summary,
                "process": {"exit_code": item.get("exit_code"), "timed_out": item.get("timed_out")},
            }
        )
    converted = compact_report(
        raw, provenance={"commit": report.get("commit"), "shmem_revision": report.get("shmem_revision"), "packages": report.get("versions")}
    )
    converted["cpu"] = (
        {key: _integer(_mapping(report.get("cpu")).get(key)) for key in ("tests", "failures", "errors", "skipped")}
        if report.get("cpu") is not None
        else None
    )
    converted["stage_counts"] = {
        stage: {
            key: _integer(_mapping(_mapping(report.get("stage_counts")).get(stage)).get(key))
            for key in ("total", "passed", "failed", "skipped", "unknown")
        }
        for stage in STAGES
    }
    converted["error_types"] = _error_types(report.get("error_types"))
    for row in converted["cases"]:
        original = original_cases[row["stage"], row["case"]]
        row["numerics"], row["error_types"] = numerics(original.get("numerics")), _error_types(original.get("error_types"))
    for mode, item in converted["performance"].items():
        if item is not None:
            original = _mapping(report["performance"][mode])
            item["numerics"], item["error_types"] = numerics(original.get("numerics")), _error_types(original.get("error_types"))
    return converted


def _display(value):
    if value is None:
        return "null"
    if type(value) is bool:
        return str(value).lower()
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def render_markdown(report):
    def pair(left, right):
        return _display(left) + " / " + _display(right)

    performance = report["performance"]
    lines = [
        "# P0 shareable report",
        "",
        "Existing evidence only; export does not execute devices or certify a new run.",
        f"Commit: {_display(report['commit'])}; SHMEM: {_display(report['shmem_revision'])}.",
        f"Reported passed: {_display(report['reported_passed'])}; device acceptance: {_display(report['reported_device_acceptance'])}.",
        "Versions: " + "; ".join(f"{key}={_display(value)}" for key, value in report["versions"].items()) + ".",
        "Units: ms; rank-paired steady samples; missing fields are null. Ratios alone do not prove correctness.",
        "",
        "Performance (benchmark workload only; HCCL batches rounds, P0 uses per-round READY/ACK):",
        "| Mode | Status | Exit / timeout | N | Exec med / p95 | E2E med / p95 | Max abs error | HCCL ratio exec / e2e | Slowdown x exec / e2e |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for mode in MODES:
        item = performance[mode] or {}
        execution, e2e, ratios = _mapping(item.get("execution")), _mapping(item.get("end_to_end")), _mapping(item.get("ratios"))
        lines.append(
            "| "
            + " | ".join(
                (
                    mode,
                    _display(item.get("status")),
                    pair(item.get("exit_code"), item.get("timed_out")),
                    _display(execution.get("samples")),
                    pair(execution.get("median_ms"), execution.get("p95_ms")),
                    pair(e2e.get("median_ms"), e2e.get("p95_ms")),
                    _display(_mapping(item.get("numerics")).get("max_abs_error")),
                    pair(_mapping(ratios.get("execution")).get("hccl_over_mode"), _mapping(ratios.get("end_to_end")).get("hccl_over_mode")),
                    pair(_mapping(ratios.get("execution")).get("slowdown_x"), _mapping(ratios.get("end_to_end")).get("slowdown_x")),
                )
            )
            + " |"
        )
    lines += ["", "| Mode | rounds / slots / iterations / warmup / tile | dtype / output | Stage availability |", "|---|---|---|---|"]
    for mode in MODES:
        item = performance[mode] or {}
        params = _mapping(item.get("parameters"))
        lines.append(
            f"| {mode} | "
            + " / ".join(_display(params.get(key)) for key in ("rounds", "slots", "iterations", "warmup", "tile"))
            + f" | {_display(params.get('dtype'))} / {_display(params.get('output_dtype'))} | {_display(item.get('stage_availability'))} |"
        )
    lines += [
        "",
        "| Mode | First build or cache load (rank max) | First execution (rank max) | First E2E (rank max) |",
        "|---|---|---|---|",
    ]
    for mode in MODES:
        first = _mapping((performance[mode] or {}).get("first_rank_max_ms"))
        lines.append(
            f"| {mode} | "
            + " | ".join(_display(first.get(key)) for key in ("first_build_or_cache_load", "first_execution", "first_end_to_end"))
            + " |"
        )
    for title, field, keys in (
        ("Outer stage medians", "stages_median_ms", OUTER),
        ("Native attribution medians (inside launch; total includes children)", "native_median_ms", NATIVE),
    ):
        lines += ["", title + ":", "| Stage | hccl | p0a | p0b |", "|---|---|---|---|"]
        for key in keys:
            lines.append(
                f"| {key} | " + " | ".join(_display(_mapping((performance[mode] or {}).get(field)).get(key)) for mode in MODES) + " |"
            )
    lines += [
        "",
        "Stage counts (total/pass/fail/skip/unknown): "
        + "; ".join(
            stage + "=" + "/".join(_display(count[key]) for key in ("total", "passed", "failed", "skipped", "unknown"))
            for stage, count in report["stage_counts"].items()
        )
        + ".",
        "CPU (tests/failures/errors/skipped): "
        + "/".join(_display(_mapping(report.get("cpu")).get(key)) for key in ("tests", "failures", "errors", "skipped"))
        + ".",
        "Per-stage maxima/percentiles are not additive; host validation, staging and synchronization remain included.",
        "Legacy schema 1 has no stage attribution. Full local detail/rank files retain raw samples and diagnostics.",
    ]
    failed = [row for row in report["cases"] if row["status"] not in {"passed", "skipped"}]
    lines.append(
        "Failed/unknown cases: "
        + (
            "; ".join(
                f"{row['stage']}/{row['case']} exit={_display(row['exit_code'])},timeout={_display(row['timed_out'])}" for row in failed[:8]
            )
            if failed
            else "null"
        )
        + (f"; {len(failed) - 8} more in compact.json" if len(failed) > 8 else "")
        + "."
    )
    lines.append("Error types: " + (", ".join(report["error_types"]) if report["error_types"] else "null") + ".")
    return "\n".join(lines) + "\n"


def _read(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def load_evidence(source):
    """Use only fixed sibling filenames; never follow paths stored in JSON."""
    source = Path(source)
    root = source if source.is_dir() else source.parent
    if source.is_dir():
        source = root / ("detail.json" if (root / "detail.json").is_file() else "summary.json")
    report = _read(source)
    if report.get("report_kind") == REPORT_KIND and (root / "detail.json").is_file():
        report = _read(root / "detail.json")
    run_root = root.parent.parent if root.name in MODES and root.parent.name == "performance" else root
    provenance = _read(run_root / "provenance.json") if (run_root / "provenance.json").is_file() else {}
    ranks = {}
    for mode in MODES:
        directory = root if report.get("mode") == mode else run_root / "performance" / mode
        ranks[mode] = [_read(directory / f"rank{rank}.json") for rank in (0, 1) if (directory / f"rank{rank}.json").is_file()]
    return report, provenance, ranks, root


def write_compact(directory, report, *, summary_name="compact.json"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # Filenames are fixed by callers, not taken from report content.
    (directory / summary_name).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (directory / "report.md").write_text(render_markdown(report))


def save_local_evidence(directory, report, *, provenance=None):
    """Keep full evidence locally; the default summary contains no raw arrays."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "detail.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_compact(directory, compact_report(report, provenance=provenance), summary_name="summary.json")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="existing run directory or original summary.json")
    parser.add_argument("--output-dir", type=Path, help="default: input directory; writes report.md and compact.json")
    args = parser.parse_args(argv)
    try:
        report, provenance, ranks, directory = load_evidence(args.source)
        compact = compact_report(report, provenance=provenance, fallback_ranks=ranks)
        write_compact(args.output_dir or directory, compact)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(2, f"Export failed ({type(error).__name__}); check local input/detail files. No device test was run.\n")
    print("Wrote report.md and compact.json from existing evidence; no device test was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
