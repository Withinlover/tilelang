"""Apply the P0 companion additions to an exact, clean cannsim-helper checkout."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


BUNDLE = Path(__file__).resolve().parent


def _git(root, *arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments], text=True, capture_output=True, check=False)
    if result.returncode:
        raise ValueError(result.stderr.strip() or f"git {arguments[0]} failed")
    return result.stdout.strip()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_companion(helper_root, bundle=BUNDLE, *, check=False, verify=False):
    """Check provenance and additions before writing; never reset existing work."""
    helper_root, bundle = Path(helper_root).resolve(), Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    if Path(_git(helper_root, "rev-parse", "--show-toplevel")).resolve() != helper_root:
        raise ValueError("--helper-root must be the checkout root")
    if _git(helper_root, "rev-parse", "HEAD") != manifest["base_revision"]:
        raise ValueError(f"helper HEAD must equal {manifest['base_revision']}; use a separate clean clone")
    patch = bundle / "p0_companion.patch"
    if _sha256(patch) != manifest["patch_sha256"]:
        raise ValueError("companion patch SHA256 does not match manifest")
    files = manifest["files"]
    for name in files:
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            raise ValueError(f"invalid companion path: {name}")
    if verify:
        changed = set(_git(helper_root, "diff", "HEAD", "--name-only").splitlines()) - set(files)
        if changed:
            raise ValueError(f"upstream tracked files changed: {', '.join(sorted(changed))}")
        for name, metadata in files.items():
            path = helper_root / name
            if path.is_symlink() or not path.is_file() or _sha256(path) != metadata["sha256"]:
                raise ValueError(f"installed companion does not match: {name}")
        return f"Verified {len(files)} companion files at the pinned helper revision."
    if _git(helper_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("helper checkout is dirty; use a separate clean clone (no files were changed)")
    for name in files:
        path = helper_root / name
        if path.exists() or path.is_symlink():
            raise ValueError(f"refusing to overwrite existing companion path: {name}")
    statistics = _git(helper_root, "apply", "--numstat", str(patch))
    additions = {}
    for line in statistics.splitlines():
        added, deleted, name = line.split("\t")
        if deleted != "0" or not added.isdecimal():
            raise ValueError("companion patch must contain only text additions")
        additions[name] = int(added)
    if set(additions) != set(files):
        raise ValueError("patch file list does not match manifest")
    _git(helper_root, "apply", "--check", "--whitespace=error-all", str(patch))
    if check:
        return f"Checked {len(files)} additions; no files were changed."
    _git(helper_root, "apply", "--whitespace=error-all", str(patch))
    for name, metadata in files.items():
        if _sha256(helper_root / name) != metadata["sha256"]:
            raise ValueError(f"post-apply hash mismatch: {name}")
    return f"Applied and verified {len(files)} additions; upstream tracked files were unchanged."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper-root", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="check applicability without changing files")
    modes.add_argument("--verify", action="store_true", help="check hashes of an already applied companion")
    args = parser.parse_args()
    try:
        print(apply_companion(args.helper_root, check=args.check, verify=args.verify))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"ERROR: {error}\n")


if __name__ == "__main__":
    main()
